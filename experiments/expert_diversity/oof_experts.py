"""OOF predictions for every eligible expert, on the folds Gate-v2 already used.

Gate-v2 concluded that the Point/Hurdle pair is too alike for routing to buy
much: their convex oracle ceiling is 4.11% on M5 and 2.33% on Favorita, and a
soft weight adds only 0.15% over a hard choice.  This tests that conclusion by
asking whether any other expert the repository already contains produces errors
that fail differently.

The fold geometry, the normalisation and the origins are the ones
`structure_gate.oof` fixed, so a new expert is comparable to Point and Hurdle
row for row.  Their existing OOF predictions are reused rather than regenerated.

Classical methods pick their smoothing parameter on each fold's own validation
split, exactly as the classical benchmark did, and never see the OOF block.
Neural experts train on the fold's train window with the frozen standard config;
only the seed policy and the config the repository already froze are used, and
nothing is tuned against any evaluation window.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..external_validity_screen import classical_benchmark as CB
from ..external_validity_screen import cli, prereg, rule_replication as rr, screen
from ..om_factorization_killtest import evaluate as km_evaluate
from ..om_factorization_killtest import models as km_models
from ..om_factorization_killtest import prereg as km_prereg
from ..structure_gate.oof import OUT as GATE_OUT, build_oof_split, fold_config, fold_cutoffs

OUT = screen.OUT.parent / "expert_diversity"

#: Expert id -> how it is produced. The two DLinear models Gate-v2 used are
#: listed so the reference pair keeps a stable id, but their predictions are
#: read from the Gate-v2 artifact instead of retrained.
NEURAL = {
    "dlinear_point": {"builder": prereg.MODELS["point"], "reuse": "point"},
    "dlinear_hurdle": {"builder": prereg.MODELS["hurdle"], "reuse": "hurdle"},
    "dlinear_point_plain": {"builder": "M0_point_mse", "reuse": None},
    "dlinear_hurdle_ztnb": {"builder": "M2_hurdle_ztnb", "reuse": None},
}
CLASSICAL = ("croston", "sba", "ses", "tsb", "seasonal_naive", "naive")
REFERENCE_PAIR = ("dlinear_point", "dlinear_hurdle")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def classical_predictions(method: str, split, cfg) -> tuple[np.ndarray, dict]:
    """Fit on the fold's validation split only, then predict its OOF block."""
    history = split.test.history.astype(np.float64)
    horizon = cfg.horizon
    if method in ("croston", "sba", "ses"):
        alpha, validation_mse = CB.select_alpha(split.validation, horizon, method)
        prediction = (CB.croston_forecast(history, horizon, alpha, method)
                      if method != "ses" else CB.ses_forecast(history, horizon, alpha))
        return prediction, {"alpha": alpha, "validation_mse": validation_mse,
                            "selected_on": "fold validation split"}
    if method == "tsb":
        alpha, beta, validation_mse = km_evaluate.select_tsb(split.validation)
        p, mu = km_models.tsb_forecast(history, horizon, alpha, beta)
        return p * mu, {"alpha": alpha, "beta": beta, "validation_mse": validation_mse,
                        "selected_on": "fold validation split"}
    if method == "seasonal_naive":
        return km_models.seasonal_naive(history, horizon, cfg.period), {"period": cfg.period}
    if method == "naive":
        return CB.naive_forecast(history, horizon), {}
    raise ValueError(method)


def reused_frame(name: str, folds: int) -> pd.DataFrame:
    """Point and Hurdle OOF predictions exactly as Gate-v2 stored them."""
    pieces = []
    for k in range(folds):
        block = pd.read_parquet(GATE_OUT / f"oof_predictions_{name}_fold{k}.parquet")
        block["fold"] = k
        pieces.append(block)
    frame = pd.concat(pieces, ignore_index=True)
    frame["series_id"] = frame["series_id"].astype(str)
    return frame.rename(columns={"point_mean_prediction": "dlinear_point",
                                 "hurdle_mean_prediction": "dlinear_hurdle"})


def cmd_build(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = prereg.MODELS["canonical_model_seed"]
    manifest = {"built_at_utc": _utc(), "device": str(device), "model_seed": seed,
                "fold_geometry": "reused from structure_gate.oof", "test_used": False,
                "training": dict(km_prereg.TRAINING), "datasets": {}}

    for name in args.datasets:
        cfg = screen.config_for(name)
        data = screen.load_dataset(name)
        cutoffs = fold_cutoffs(cfg)
        base = reused_frame(name, len(cutoffs))
        keys = ["fold", "series_id", "origin", "step"]
        columns = {}
        config = {}

        for k, cutoff in enumerate(cutoffs):
            fold_cfg = fold_config(cfg, cutoff)
            split = build_oof_split(data, fold_cfg,
                                    prereg.SPLITS[name]["test_origin_stride"])
            n_series = data["y"].shape[0]
            n_origins = split.test.n_origins
            horizon = split.test.target.shape[1]
            series_axis = np.repeat(np.arange(n_series), n_origins)
            index = pd.DataFrame({
                "fold": k,
                "series_id": np.repeat(data["series_id"][series_axis], horizon),
                "origin": np.repeat(np.tile(np.asarray(split.test.origins), n_series), horizon),
                "step": np.tile(np.arange(horizon), len(series_axis))})

            block = {}
            for expert, spec in NEURAL.items():
                if spec["reuse"] is not None:
                    continue
                print(f"[{name}] fold {k}: training {expert} ({spec['builder']}) ...", flush=True)
                fit = rr.train_frozen(spec["builder"], fold_cfg, seed, device, split)
                block[expert] = fit["predictions"]["mean_prediction"].reshape(-1)
                config.setdefault(expert, {})[f"fold{k}"] = {
                    "train_seconds": fit["train_seconds"],
                    "n_parameters": fit["n_parameters"]}
            for method in CLASSICAL:
                prediction, chosen = classical_predictions(method, split, fold_cfg)
                block[method] = prediction.reshape(-1)
                config.setdefault(method, {})[f"fold{k}"] = chosen
            piece = pd.concat([index, pd.DataFrame(block)], axis=1)
            columns[k] = piece

        extra = pd.concat(columns.values(), ignore_index=True)
        merged = base.merge(extra, on=keys, how="inner")
        if len(merged) != len(base):
            raise screen.ScreenFailure(
                f"{name}: expert rows {len(merged)} != reused rows {len(base)}")
        keep = (keys + ["y_observed", "occurrence", "target_mask"]
                + list(NEURAL) + list(CLASSICAL))
        merged[keep].to_parquet(OUT / f"oof_experts_{name}.parquet", index=False)
        manifest["datasets"][name] = {
            "n_rows": int(len(merged)), "folds": [int(c) for c in cutoffs],
            "experts": list(NEURAL) + list(CLASSICAL),
            "reference_pair": list(REFERENCE_PAIR),
            "reused_from_gate_v2": [e for e, s in NEURAL.items() if s["reuse"]],
            "expert_config": config}
        print(f"[{name}] {len(merged):,} rows, {len(NEURAL) + len(CLASSICAL)} experts", flush=True)

    manifest["git_commit"] = cli._git_commit()
    (OUT / "oof_experts_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({n: b["n_rows"] for n, b in manifest["datasets"].items()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("OOF predictions for every eligible expert")
    sub = parser.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    b.set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
