"""Out-of-fold Point and Hurdle predictions, built inside the train region only.

Everything a gate could ever learn from has to come from here, because the test
split must stay unread until a GateSpec is frozen.  There is no rolling-origin
utility in the repository -- `dense_origins` and `stride_origins` exist, but
nothing that expands a training window -- so the folds are built here by
repeating Stage A's own geometry at earlier cutoffs.

Each fold gets its own ExperimentConfig with an earlier train_end, so
`screen.build_split` produces train, validation and an OOF block exactly the way
it produced Stage A's, and `train_scale` normalises on that fold's train window
alone.  A fold never sees a day at or beyond its own cutoff.

    fold k    train [0, T_k)   validation [T_k, T_k+28)   OOF origins T_k+28, +56, +84

The last fold's OOF targets end exactly at the real train_end, so no fold
touches the validation or test region the experts are finally judged on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..external_validity_screen import cli, prereg, rule_replication as rr, screen
from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from ..unified_temporal_27_v3.training import train_scale

OUT = screen.OUT.parent / "structure_gate"
N_FOLDS = 3
ORIGINS_PER_FOLD = 3


def fold_cutoffs(cfg) -> list[int]:
    """Expanding cutoffs whose OOF blocks tile backwards from train_end.

    A fold needs horizon days of validation plus ORIGINS_PER_FOLD strided
    origins plus one horizon of target, so each block spans
    horizon * (1 + ORIGINS_PER_FOLD) days.  Tiling those backwards from
    train_end keeps every OOF target strictly inside the train region.
    """
    block = cfg.horizon * (1 + ORIGINS_PER_FOLD)
    cutoffs = [cfg.train_end - block * (N_FOLDS - k) for k in range(N_FOLDS)]
    if cutoffs[0] <= cfg.lookback + cfg.horizon:
        raise screen.ScreenFailure(
            f"first fold cutoff {cutoffs[0]} leaves no usable history")
    return cutoffs


def fold_config(cfg, cutoff: int):
    block = cfg.horizon * (1 + ORIGINS_PER_FOLD)
    return replace(cfg, train_end=cutoff, val_end=cutoff + cfg.horizon,
                   length=cutoff + block)


def build_oof_split(data: dict, fold_cfg, stride: int):
    """screen.build_split, with the availability mask extended to the OOF block.

    build_split refuses outright when a series becomes available inside the
    evaluation window, because Stage A's test estimand had to stay untouched and
    its window starts after every M5 item is on sale.  An OOF block sitting a
    year earlier cannot honour that: M5's latest availability is day 1715 and
    the earliest fold evaluates at 1521.

    Masking is the honest response and it is not a new policy -- it is the same
    `_mask_before_availability` build_split already applies to train and
    validation.  A series that was not yet stocked contributes no observations
    to the loss instead of contributing a run of fabricated zeros; rows left
    with nothing observed are dropped downstream rather than counted as perfect.
    """
    arrays = {"y": data["y"], "z": data["z"]}
    scale = train_scale(arrays, fold_cfg)
    train_stride = prereg.SPLITS["train_origin_stride"]
    train_o = km_train.dense_origins(0, fold_cfg.train_end, fold_cfg.horizon,
                                     fold_cfg.lookback)[::train_stride]
    val_o = km_train.dense_origins(fold_cfg.train_end, fold_cfg.val_end,
                                   fold_cfg.horizon, fold_cfg.lookback)
    oof_o = km_train.stride_origins(fold_cfg.val_end, fold_cfg.length,
                                    fold_cfg.horizon, stride)
    train_w = screen.make_windows(arrays, train_o, 0, fold_cfg.train_end, fold_cfg, scale)
    val_w = screen.make_windows(arrays, val_o, fold_cfg.train_end, fold_cfg.val_end,
                                fold_cfg, scale)
    oof_w = screen.make_windows(arrays, oof_o, fold_cfg.val_end, fold_cfg.length,
                                fold_cfg, scale)

    available = data["available_from"]
    if available.max() > 0:
        train_w = screen._mask_before_availability(train_w, available, fold_cfg)
        val_w = screen._mask_before_availability(val_w, available, fold_cfg)
        oof_w = screen._mask_before_availability(oof_w, available, fold_cfg)

    y_train = data["y"][:, :fold_cfg.train_end]
    z_train = data["z"][:, :fold_cfg.train_end]
    positives = y_train[z_train > 0]
    variance = float(positives.var(ddof=1)) if positives.size > 1 else 1.0
    return km_train.Split(train_w, val_w, oof_w, scale, max(variance, 1e-6))


def run_fold(data: dict, cfg, cutoff: int, seed: int, device) -> dict:
    """Train both experts on this fold's past and predict its OOF block."""
    fold_cfg = fold_config(cfg, cutoff)
    split = build_oof_split(data, fold_cfg,
                            prereg.SPLITS[data["name"]]["test_origin_stride"])
    n_series = data["y"].shape[0]
    out = {"cutoff": int(cutoff), "val_end": int(fold_cfg.val_end),
           "oof_end": int(fold_cfg.length),
           "origins": split.test.origins.tolist(),
           "n_origins": int(split.test.n_origins)}
    predictions = {}
    for role, key in (("point", prereg.MODELS["point"]),
                      ("hurdle", prereg.MODELS["hurdle"])):
        fit = rr.train_frozen(key, fold_cfg, seed, device, split)
        predictions[role] = fit["predictions"]
        out[f"{role}_train_seconds"] = fit["train_seconds"]
        out[f"{role}_n_parameters"] = fit["n_parameters"]
    out["_predictions"] = predictions
    out["_windows"] = split.test
    out["_scale"] = split.scale
    out["_n_series"] = n_series
    return out


def normalized_losses(predictions: dict, windows, scale: np.ndarray,
                      n_series: int) -> pd.DataFrame:
    """Per-origin horizon MSE, normalised by the fold's own train scale.

    train_scale is the per-series mean over that fold's train window and is the
    same quantity the models divide their input by, so the loss a gate learns
    from is on the scale the experts already work in.  Without it the regret
    target would be dominated by whichever series happen to be large.
    """
    mask = windows.target_mask.astype(np.float64)
    target = windows.target.astype(np.float64)
    n_origins = windows.n_origins
    series_axis = np.repeat(np.arange(n_series), n_origins)
    origin_axis = np.tile(np.asarray(windows.origins), n_series)
    denominator = np.maximum(mask.sum(axis=1), 1.0)
    scale_per_row = scale[series_axis].astype(np.float64)

    frame = pd.DataFrame({"series_index": series_axis, "origin": origin_axis,
                          "n_observed": mask.sum(axis=1)})
    for role, pred in predictions.items():
        error = ((pred["mean_prediction"].astype(np.float64) - target) / scale_per_row[:, None]) ** 2
        frame[f"loss_{role}"] = (error * mask).sum(axis=1) / denominator
    frame["regret"] = frame["loss_point"] - frame["loss_hurdle"]
    frame = pd.concat([frame, expert_state(predictions, scale_per_row)], axis=1)
    # A row with nothing observed (pre-availability) has no loss to compare.
    return frame[frame["n_observed"] > 0].reset_index(drop=True)


def expert_state(predictions: dict, scale_per_row: np.ndarray) -> pd.DataFrame:
    """Per-origin summaries of what the two experts predicted.

    Only fields km_train.predict actually returns are used, and no true target
    enters here, so these stay legal gate inputs: they describe the forecast,
    not its error.
    """
    point = predictions["point"]["mean_prediction"].astype(np.float64)
    hurdle = predictions["hurdle"]["mean_prediction"].astype(np.float64)
    gap = hurdle - point
    scale = np.maximum(scale_per_row, 1e-9)
    out = {"xs_point_mean": point.mean(axis=1) / scale,
           "xs_hurdle_mean": hurdle.mean(axis=1) / scale,
           "xs_abs_disagreement": np.abs(gap).mean(axis=1) / scale,
           "xs_max_disagreement": np.abs(gap).max(axis=1) / scale,
           "xs_signed_disagreement": gap.mean(axis=1) / scale}
    for field, label in (("p_prediction", "p"), ("mu_prediction", "mu")):
        if field in predictions["hurdle"]:
            values = predictions["hurdle"][field].astype(np.float64)
            if label == "mu":
                values = values / scale[:, None]
            out[f"xs_hurdle_{label}_mean"] = values.mean(axis=1)
            out[f"xs_hurdle_{label}_min"] = values.min(axis=1)
            out[f"xs_hurdle_{label}_max"] = values.max(axis=1)
    return pd.DataFrame(out)


def save_fold_predictions(name: str, fold: int, predictions: dict, windows,
                          data: dict) -> None:
    """Persist predict()'s output verbatim, one row per (series, origin, step)."""
    n_series = len(data["series_id"])
    n_origins = windows.n_origins
    series_axis = np.repeat(np.arange(n_series), n_origins)
    horizon = windows.target.shape[1]
    frame = pd.DataFrame({
        "series_id": np.repeat(data["series_id"][series_axis], horizon),
        "origin": np.repeat(np.tile(np.asarray(windows.origins), n_series), horizon),
        "step": np.tile(np.arange(horizon), len(series_axis)),
        "y_observed": windows.target.reshape(-1),
        "occurrence": windows.occurrence.reshape(-1),
        "target_mask": windows.target_mask.reshape(-1),
        "point_mean_prediction": predictions["point"]["mean_prediction"].reshape(-1),
        "hurdle_mean_prediction": predictions["hurdle"]["mean_prediction"].reshape(-1),
        "hurdle_p_prediction": predictions["hurdle"]["p_prediction"].reshape(-1),
        "hurdle_mu_prediction": predictions["hurdle"]["mu_prediction"].reshape(-1)})
    frame.to_parquet(OUT / f"oof_predictions_{name}_fold{fold}.parquet", index=False)


def save_fold_predictions(name: str, fold: int, predictions: dict, windows,
                          data: dict) -> None:
    """Persist predict()'s output verbatim, one row per (series, origin, step)."""
    n_series = len(data["series_id"])
    n_origins = windows.n_origins
    series_axis = np.repeat(np.arange(n_series), n_origins)
    horizon = windows.target.shape[1]
    pd.DataFrame({
        "series_id": np.repeat(data["series_id"][series_axis], horizon),
        "origin": np.repeat(np.tile(np.asarray(windows.origins), n_series), horizon),
        "step": np.tile(np.arange(horizon), len(series_axis)),
        "y_observed": windows.target.reshape(-1),
        "occurrence": windows.occurrence.reshape(-1),
        "target_mask": windows.target_mask.reshape(-1),
        "point_mean_prediction": predictions["point"]["mean_prediction"].reshape(-1),
        "hurdle_mean_prediction": predictions["hurdle"]["mean_prediction"].reshape(-1),
        "hurdle_p_prediction": predictions["hurdle"]["p_prediction"].reshape(-1),
        "hurdle_mu_prediction": predictions["hurdle"]["mu_prediction"].reshape(-1),
    }).to_parquet(OUT / f"oof_predictions_{name}_fold{fold}.parquet", index=False)


def cmd_build(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = prereg.MODELS["canonical_model_seed"]
    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_folds": N_FOLDS, "origins_per_fold": ORIGINS_PER_FOLD,
                "model_seed": seed, "device": str(device),
                "training": dict(km_prereg.TRAINING),
                "loss": ("per-origin horizon MSE divided by the fold's own "
                         "train_scale squared; masked steps excluded"),
                "test_used": False, "datasets": {}}

    for name in args.datasets:
        cfg = screen.config_for(name)
        data = screen.load_dataset(name)
        cutoffs = fold_cutoffs(cfg)
        print(f"[{name}] train_end={cfg.train_end} folds at {cutoffs}", flush=True)
        tables, folds = [], []
        for k, cutoff in enumerate(cutoffs):
            print(f"[{name}] fold {k}: cutoff {cutoff} ...", flush=True)
            fold = run_fold(data, cfg, cutoff, seed, device)
            predictions = fold.pop("_predictions")
            table = normalized_losses(predictions, fold["_windows"],
                                      fold.pop("_scale"), fold.pop("_n_series"))
            save_fold_predictions(name, k, predictions, fold["_windows"], data)
            table["fold"] = k
            table["series_id"] = data["series_id"][table["series_index"]]
            tables.append(table)
            fold.pop("_windows")
            folds.append(fold)
            print(f"   origins {fold['origins']}  rows {len(table)}", flush=True)
        regret = pd.concat(tables, ignore_index=True)
        regret.to_parquet(OUT / f"oof_regret_{name}.parquet", index=False)
        manifest["datasets"][name] = {
            "train_end": cfg.train_end, "val_end": cfg.val_end,
            "lookback": cfg.lookback, "horizon": cfg.horizon,
            "n_series": int(data["y"].shape[0]),
            "folds": folds,
            "max_oof_target_day": int(max(f["oof_end"] for f in folds)),
            "strictly_inside_train": bool(max(f["oof_end"] for f in folds) <= cfg.train_end),
            "n_oof_rows": int(len(regret)),
            "series_id_sha256": hashlib.sha256(
                "\n".join(sorted(data["series_id"])).encode()).hexdigest()}
        print(f"[{name}] OOF rows {len(regret):,}", flush=True)

    manifest["git_commit"] = cli._git_commit()
    (OUT / "oof_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    for name, block in manifest["datasets"].items():
        if not block["strictly_inside_train"]:
            raise screen.ScreenFailure(
                f"{name}: an OOF target lands at or beyond train_end")
    print(json.dumps({n: {"n_oof_rows": b["n_oof_rows"],
                          "max_oof_target_day": b["max_oof_target_day"],
                          "train_end": b["train_end"]}
                      for n, b in manifest["datasets"].items()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("OOF expert predictions for the gate")
    sub = parser.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    b.set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
