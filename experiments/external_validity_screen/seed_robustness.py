"""Is the M5 H2 rule effect an accident of one initialisation?

The frozen-rule replication ran on the canonical seed alone.  This repeats it
on three, changing nothing but `model_seed`: same Stage A training population,
same frozen candidate and control ids, same config, same protocol for Point and
Hurdle.

The seed list follows the repository's own convention.
`om_factorization_killtest.prereg.DATA["model_seeds"]` is (0, 1) and
`external_validity_screen.prereg.MODELS["canonical_model_seed"]` is 0, so the
third seed is 2 rather than an invented value.  The list is frozen before any
model is trained.

Seeds are not extra series.  Averaging the three deltas per series first and
bootstrapping over series afterwards keeps the inferential unit where the
frozen spec put it; treating 3 x 5693 as an independent sample would shrink the
interval by a factor it has not earned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from . import cli, posthoc, prereg, rule_replication as rr, screen

OUT = screen.OUT / "seed_robustness"
MODELS = OUT / "models"
SPEC = OUT / "seed_spec.json"


def seed_list() -> list[int]:
    """(0, 1) from the killtest convention, extended by one, canonical first."""
    convention = list(km_prereg.DATA["model_seeds"])
    canonical = prereg.MODELS["canonical_model_seed"]
    seeds = [canonical] + [s for s in convention if s != canonical]
    while len(seeds) < 3:
        seeds.append(max(seeds) + 1)
    return seeds[:3]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_of(values) -> str:
    return hashlib.sha256("\n".join(sorted(str(v) for v in values)).encode()).hexdigest()


def cmd_freeze(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if SPEC.exists():
        raise SystemExit(f"{SPEC} already frozen; refusing to overwrite")
    parent = json.loads(rr.SPEC.read_text())
    spec = {
        "study": "M5 H2 frozen-rule replication, seed robustness",
        "frozen_at_utc": _utc(),
        "seeds": seed_list(),
        "seed_source": {
            "killtest_model_seeds": list(km_prereg.DATA["model_seeds"]),
            "canonical_model_seed": prereg.MODELS["canonical_model_seed"],
            "rule": "canonical first, then the repository convention, extended by "
                    "one only to reach three; no invented values"},
        "training_population": "Stage A M5 1200, identical for every seed",
        "evaluation_population": {
            "candidate_id_sha256": parent["candidate_id_sha256"],
            "n_candidate": parent["n_candidate"],
            "control_id_sha256": parent["control_id_sha256"],
            "n_control": parent["n_control"]},
        "frozen_h2_cutoffs": parent["frozen_h2_cutoffs"],
        "independent_series_in_training": False,
        "varies_between_runs": ["model_seed"],
        "held_constant": ["point implementation", "hurdle implementation",
                          "parameter count", "lookback", "horizon", "optimizer",
                          "learning rate", "batch size", "max epochs", "patience",
                          "normalization", "training stride", "validation split",
                          "checkpoint rule", "candidate ids", "control ids"],
        "training": dict(km_prereg.TRAINING), "split": dict(prereg.SPLITS["m5"]),
        "aggregate_rule": ("average delta_relative across seeds per series, then "
                           "bootstrap over series; seeds are not extra series"),
        "bootstrap": dict(prereg.BOOTSTRAP),
        "verdict_rule": {
            "H2_SEED_ROBUST": "every seed effect < 0 and aggregate < 0 with CI excluding 0",
            "H2_SEED_DIRECTION_STABLE": "at least 2/3 seeds < 0 and aggregate < 0 but CI covers 0",
            "H2_SEED_UNSTABLE": "aggregate >= 0 or the sign flips repeatedly"},
        "git_commit": cli._git_commit(),
    }
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    print(json.dumps({"seeds": spec["seeds"], "seed_source": spec["seed_source"]}, indent=2))


def _checkpoint(seed: int, role: str):
    """Seed 0 is the model the single-seed replication already froze."""
    if seed == prereg.MODELS["canonical_model_seed"]:
        return rr.MODELS / f"{role}.pt"
    return MODELS / f"seed{seed}_{role}.pt"


def cmd_train(_args) -> None:
    if not SPEC.exists():
        raise SystemExit("run `freeze` first")
    spec = json.loads(SPEC.read_text())
    MODELS.mkdir(parents=True, exist_ok=True)
    cfg = screen.config_for("m5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = split = None
    provenance = {}
    for seed in spec["seeds"]:
        for role, key in (("point", prereg.MODELS["point"]),
                          ("hurdle", prereg.MODELS["hurdle"])):
            path = _checkpoint(seed, role)
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload["model_seed"] != seed or payload["trained_on"] != "stage_a_m5_1200":
                    raise screen.ScreenFailure(
                        f"{path} provenance mismatch: {payload['model_seed']}, "
                        f"{payload['trained_on']}")
                provenance[f"seed{seed}_{role}"] = {
                    "path": str(path.relative_to(screen.REPO)), "reused": True,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                print(f"seed {seed} {role}: reusing {path.name}")
                continue
            if data is None:
                print("loading Stage A's 1200 series ...")
                data = screen.load_dataset("m5")
                split = screen.build_split(data, cfg,
                                           prereg.SPLITS["m5"]["test_origin_stride"])
            print(f"seed {seed} {role}: training on {device} ...")
            fit = rr.train_frozen(key, cfg, seed, device, split)
            torch.save({"state_dict": fit["model"].state_dict(), "builder_key": key,
                        "lookback": cfg.lookback, "horizon": cfg.horizon,
                        "model_seed": seed, "trained_on": "stage_a_m5_1200",
                        "training": dict(km_prereg.TRAINING)}, path)
            provenance[f"seed{seed}_{role}"] = {
                "path": str(path.relative_to(screen.REPO)), "reused": False,
                "train_seconds": fit["train_seconds"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "model_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    print(json.dumps({k: v["reused"] for k, v in provenance.items()}, indent=2))


def _load(seed: int, role: str, device):
    from ..om_factorization_killtest import models
    payload = torch.load(_checkpoint(seed, role), map_location=device, weights_only=False)
    if payload["model_seed"] != seed or payload["trained_on"] != "stage_a_m5_1200":
        raise screen.ScreenFailure("checkpoint provenance mismatch")
    model = models.BUILDERS[payload["builder_key"]](payload["lookback"],
                                                    payload["horizon"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def cmd_evaluate(_args) -> None:
    spec = json.loads(SPEC.read_text())
    population = pd.read_csv(rr.OUT / "independent_population.csv")
    population["series_id"] = population["series_id"].astype(str)
    candidate = sorted(population.loc[population["group"] == "candidate", "series_id"])
    control = sorted(population.loc[population["group"] == "control_pool", "series_id"])
    if (_sha_of(candidate) != spec["evaluation_population"]["candidate_id_sha256"]
            or _sha_of(control) != spec["evaluation_population"]["control_id_sha256"]):
        raise screen.ScreenFailure("evaluation ids do not match the frozen spec")

    wanted = candidate + control
    group = np.array(["candidate"] * len(candidate) + ["control"] * len(control))
    cfg = screen.config_for("m5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading full M5 ...")
    full = rr.conf.m5_full()
    index = pd.Index(full["series_id"]).get_indexer(wanted)
    data = {"name": "m5", "y": full["y"][index], "z": full["z"][index],
            "series_id": np.asarray(wanted),
            "available_from": full["available_from"][index],
            "first_positive": full["first_positive"][index]}
    stage_a = set(pd.read_csv(screen.OUT / "per_series_metrics.csv")
                  .query("dataset == 'm5'")["series_id"].astype(str))
    if set(wanted) & stage_a:
        raise screen.ScreenFailure("STAGE_A_OVERLAP in the scored population")
    windows = rr.test_windows_only(data, cfg, prereg.SPLITS["m5"]["test_origin_stride"])

    frame = pd.DataFrame({"series_id": wanted, "group": group})
    per_seed = {}
    for seed in spec["seeds"]:
        metrics = {}
        for role in ("point", "hurdle"):
            print(f"seed {seed}: scoring {role} on {len(wanted)} unseen series ...")
            metrics[role] = screen.test_metrics(
                km_train.predict(_load(seed, role, device), windows, device),
                windows, len(wanted))
        raw = metrics["point"]["rmse_realized"] - metrics["hurdle"]["rmse_realized"]
        rel = np.where(metrics["point"]["rmse_realized"] <= posthoc.RELATIVE_FLOOR, np.nan,
                       raw / np.where(metrics["point"]["rmse_realized"] <= posthoc.RELATIVE_FLOOR,
                                      1.0, metrics["point"]["rmse_realized"]))
        frame[f"delta_raw_seed{seed}"] = raw
        frame[f"delta_relative_seed{seed}"] = rel
        per_seed[seed] = rel

    seed_columns = [f"delta_relative_seed{s}" for s in spec["seeds"]]
    frame["delta_relative_seedmean"] = frame[seed_columns].mean(axis=1)
    frame.to_csv(OUT / "per_series_by_seed.csv", index=False)

    is_cand = (frame["group"] == "candidate").to_numpy()
    draws, bseed = prereg.BOOTSTRAP["draws"], prereg.BOOTSTRAP["seed"]

    def contrast(values):
        c, k = values[is_cand], values[~is_cand]
        c, k = c[np.isfinite(c)], k[np.isfinite(k)]
        effect = float(c.mean() - k.mean())
        rng = np.random.default_rng(bseed)
        ci = rr._boot_ci(lambda r: (c[r.integers(0, c.size, c.size)].mean()
                                    - k[r.integers(0, k.size, k.size)].mean()), rng, draws)
        rng = np.random.default_rng(bseed)
        win = rr._boot_ci(lambda r: ((c[r.integers(0, c.size, c.size)] < 0).mean()
                                     - (k[r.integers(0, k.size, k.size)] < 0).mean()),
                          rng, draws)
        return {"candidate_mean": float(c.mean()), "candidate_median": float(np.median(c)),
                "control_mean": float(k.mean()), "control_median": float(np.median(k)),
                "candidate_point_win_rate": float((c < 0).mean()),
                "control_point_win_rate": float((k < 0).mean()),
                "point_win_rate_difference": float((c < 0).mean() - (k < 0).mean()),
                "point_win_rate_difference_ci": win,
                "effect": effect, "effect_ci": ci,
                "effect_ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                "n_candidate": int(c.size), "n_control": int(k.size)}

    by_seed = {str(s): contrast(per_seed[s]) for s in spec["seeds"]}
    aggregate = contrast(frame["delta_relative_seedmean"].to_numpy(float))

    effects = [by_seed[str(s)]["effect"] for s in spec["seeds"]]
    negative = sum(e < 0 for e in effects)
    if aggregate["effect"] >= 0:
        verdict = "H2_SEED_UNSTABLE"
    elif negative == len(effects) and aggregate["effect_ci_excludes_zero"]:
        verdict = "H2_SEED_ROBUST"
    elif negative >= 2:
        verdict = "H2_SEED_DIRECTION_STABLE"
    else:
        verdict = "H2_SEED_UNSTABLE"

    report = {"verdict": verdict, "seeds": spec["seeds"], "by_seed": by_seed,
              "aggregate_3seed": aggregate,
              "seed_effect_range": [float(min(effects)), float(max(effects))],
              "seed_effect_signs": ["negative" if e < 0 else "positive" for e in effects],
              "n_seeds_negative": int(negative),
              "aggregate_rule": spec["aggregate_rule"],
              "bootstrap": dict(prereg.BOOTSTRAP),
              "evaluated_at_utc": _utc(), "device": str(device),
              "git_commit": cli._git_commit()}
    (OUT / "seed_robustness.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "effects": effects,
                      "aggregate": aggregate["effect"],
                      "aggregate_ci": aggregate["effect_ci"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("M5 H2 seed robustness")
    sub = parser.add_subparsers(required=True)
    for name, func in (("freeze", cmd_freeze), ("train", cmd_train),
                       ("evaluate", cmd_evaluate)):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
