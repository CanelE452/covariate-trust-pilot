"""freeze -> stage-a. The spec is written before any prediction exists."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from . import prereg, screen

OUT = screen.OUT
SPEC = OUT / "pre_analysis_spec.json"


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=screen.REPO,
                          capture_output=True, text=True).stdout.strip()


def cmd_freeze(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if SPEC.exists():
        print(f"pre_analysis_spec already frozen at {SPEC}; not overwriting")
        return
    spec = {
        "estimand": prereg.ESTIMAND,
        "hypotheses": prereg.HYPOTHESES,
        "subgroups": prereg.SUBGROUPS,
        "descriptor_rules": prereg.DESCRIPTOR_RULES,
        "eligibility": prereg.ELIGIBILITY,
        "leading_zero_policy": prereg.LEADING_ZERO_POLICY,
        "bootstrap": prereg.BOOTSTRAP,
        "models": prereg.MODELS,
        "splits": prereg.SPLITS,
        "stop_rules": prereg.STOP_RULES,
        "training": dict(km_prereg.TRAINING),
        "git_commit": _git_commit(),
        "frozen_before_any_prediction": True,
    }
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    print(f"froze {SPEC}")


def cmd_stage_a(args) -> None:
    if not SPEC.exists():
        raise SystemExit("run `freeze` before generating any prediction")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = prereg.MODELS["canonical_model_seed"]
    names = args.datasets or list(screen.DATASETS)

    manifest, all_series, all_results = {}, [], {}
    for name in names:
        print(f"[{name}] loading ...")
        data = screen.load_dataset(name)
        cfg = screen.config_for(name)
        if data["y"].shape[1] != cfg.length:
            raise screen.ScreenFailure(
                f"{name}: length {data['y'].shape[1]} != frozen {cfg.length}")
        stride = prereg.SPLITS[name]["test_origin_stride"]
        split = screen.build_split(data, cfg, stride)
        n_series = data["y"].shape[0]

        variant = screen.DATASETS[name]["leading_zero"]
        tables = {v: screen.descriptor_table(data, cfg, v)
                  for v in prereg.LEADING_ZERO_POLICY["sensitivity_variants"]
                  if v == variant or v in ("raw", "first_positive")}
        primary = tables[variant]

        per_model = {}
        for role, key in (("point", prereg.MODELS["point"]),
                          ("hurdle", prereg.MODELS["hurdle"])):
            print(f"[{name}] training {role} ({key}) on {device} ...")
            fit = train_on_split(key, cfg, seed, device, split)
            per_model[role] = screen.test_metrics(fit["predictions"],
                                                  split.test, n_series)
            per_model[role]["_n_parameters"] = fit["n_parameters"]
            per_model[role]["_train_seconds"] = fit["train_seconds"]

        delta = per_model["point"]["rmse_realized"] - per_model["hurdle"]["rmse_realized"]
        frame = primary.copy()
        frame["dataset"] = name
        frame["rmse_point"] = per_model["point"]["rmse_realized"]
        frame["rmse_hurdle"] = per_model["hurdle"]["rmse_realized"]
        frame["mae_point"] = per_model["point"]["mae_realized"]
        frame["mae_hurdle"] = per_model["hurdle"]["mae_realized"]
        frame["delta_rmse"] = delta
        for key in ("occurrence_brier", "mean_p_hat_on_zero",
                    "mean_p_hat_on_positive", "positive_magnitude_rmse"):
            if key in per_model["hurdle"]:
                frame[f"hurdle_{key}"] = per_model["hurdle"][key]
        all_series.append(frame)

        results = {t: screen.analyse(frame, t)
                   for t in prereg.ELIGIBILITY["sensitivity_thresholds"]}
        for other, table in tables.items():
            if other == variant:
                continue
            alt = table.copy()
            alt["delta_rmse"] = delta
            results[f"variant::{other}"] = screen.analyse(
                alt, prereg.ELIGIBILITY["primary_threshold"])
        all_results[name] = results

        manifest[name] = {
            "n_series": int(n_series),
            "descriptor_variant_primary": variant,
            "train_end": cfg.train_end, "val_end": cfg.val_end,
            "length": cfg.length, "horizon": cfg.horizon,
            "lookback": cfg.lookback, "test_origins": split.test.origins.tolist(),
            "n_parameters": {r: int(m["_n_parameters"]) for r, m in per_model.items()},
            "train_seconds": {r: float(m["_train_seconds"]) for r, m in per_model.items()},
            "overall": {
                "rmse_point": float(np.mean(per_model["point"]["rmse_realized"])),
                "rmse_hurdle": float(np.mean(per_model["hurdle"]["rmse_realized"])),
                "mae_point": float(np.mean(per_model["point"]["mae_realized"])),
                "mae_hurdle": float(np.mean(per_model["hurdle"]["mae_realized"])),
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "hurdle_win_pct": float(np.mean(delta > 0) * 100),
                "point_win_pct": float(np.mean(delta < 0) * 100),
            },
            "availability": {
                "max_available_from": int(data["available_from"].max()),
                "n_after_day_zero": int((data["available_from"] > 0).sum()),
            },
        }

    pd.concat(all_series).to_csv(OUT / "per_series_metrics.csv", index=False)
    (OUT / "stage_a_results.json").write_text(
        json.dumps({"manifest": manifest, "results": all_results,
                    "device": str(device), "torch": torch.__version__,
                    "platform": platform.platform(),
                    "git_commit": _git_commit()},
                   indent=2, default=str))
    print(f"wrote {OUT.relative_to(screen.REPO)}")


def train_on_split(key, cfg, seed, device, split):
    """km_train.train_one's loop, verbatim, on a split we supply.

    train_one calls build_splits internally, so it cannot accept the
    availability-masked split that decision 1 requires. Everything else —
    seeding order, optimizer, batch size, epoch budget, patience, checkpoint
    criterion, prediction call — is copied unchanged from train_one, and
    test_loop_matches_train_one() asserts the hyperparameters still agree.
    """
    import copy, time
    from torch.utils.data import DataLoader, TensorDataset
    from ..om_factorization_killtest import models
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = models.BUILDERS[key](cfg.lookback, cfg.horizon).to(device)
    loss_fn = models.LOSSES[key]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=km_prereg.TRAINING["learning_rate"],
                                 weight_decay=km_prereg.TRAINING["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = km_train._loader(split.train, km_prereg.TRAINING["batch_size"],
                                    True, generator)
    val_loader = km_train._loader(split.validation, 1024, False)
    variance = torch.tensor(split.positive_variance, device=device)
    best, best_state, bad = float("inf"), None, 0
    started = time.time()
    for _ in range(km_prereg.TRAINING["max_epochs"]):
        model.train()
        for history, target, occurrence, mask, scale in train_loader:
            history, target = history.to(device), target.to(device)
            occurrence, mask = occurrence.to(device), mask.to(device)
            scale = scale.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(history, scale), target, occurrence, mask, variance)
            loss.backward()
            optimizer.step()
        current = km_train._validation_mean_mse(model, val_loader, device)
        if current < best - 1e-9:
            best, bad = current, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= km_prereg.TRAINING["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"predictions": km_train.predict(model, split.test, device),
            "n_parameters": models.count_parameters(model),
            "train_seconds": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser("external-validity SCREEN")
    sub = parser.add_subparsers(required=True)
    f = sub.add_parser("freeze"); f.set_defaults(func=cmd_freeze)
    s = sub.add_parser("stage-a")
    s.add_argument("--datasets", nargs="*", default=None)
    s.set_defaults(func=cmd_stage_a)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
