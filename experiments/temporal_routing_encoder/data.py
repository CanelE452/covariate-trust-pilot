"""Raw history windows and expert context for one fold, on P0L1's own rows.

The quadratic coefficients are built by the same call P0L1 uses, so the rows
this returns are the rows the handcrafted gate was scored on -- the sequence
tensors are simply carried alongside them.  Nothing here reads past an origin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..external_validity_screen import prereg, screen
from ..gate_v3.oof import add_targets, quadratic_from_arrays
from ..multi_benchmark import run as MB
from ..structure_gate.oof import build_oof_split


@dataclass(frozen=True)
class FoldBundle:
    block: object            # per-origin quadratic coefficients and targets
    sequence: np.ndarray     # (rows, lookback, 3)
    context: np.ndarray      # (rows, 2 * horizon)
    series_id: np.ndarray
    origin: np.ndarray
    lookback: int
    horizon: int
    scale_source_end: int


def observed_matrix(data: dict) -> np.ndarray:
    """One boolean (N, T) for both dataset families, using each one's own rule.

    FreshRetailNet and UCI carry an explicit observed mask.  M5 and Favorita
    encode the same idea as an availability date, which is what build_split
    already masks on, so a step before that date is unobserved rather than a
    zero sale.
    """
    if "observed_mask" in data:
        return np.asarray(data["observed_mask"]).astype(bool)
    steps = np.arange(data["y"].shape[1])[None, :]
    return steps >= np.asarray(data["available_from"])[:, None]


def _history_stack(matrix: np.ndarray, origins: np.ndarray, lookback: int) -> np.ndarray:
    """(N, W, L) slices of [origin - L, origin); the origin itself is excluded."""
    out = np.empty((matrix.shape[0], len(origins), lookback), dtype=matrix.dtype)
    for w, origin in enumerate(origins):
        out[:, w] = matrix[:, origin - lookback:origin]
    return out


def fold_bundle(name: str, fold: dict, a: str, b: str, device) -> FoldBundle:
    if name in ("m5", "favorita"):
        data = screen.load_dataset(name)
        cfg = screen.config_for(name)
        stride = prereg.SPLITS[name]["test_origin_stride"]
        fold_cfg = replace(cfg, train_end=fold["train_end"],
                           val_end=fold["train_end"] + cfg.horizon,
                           length=fold["validation_end"])
        split = build_oof_split(data, fold_cfg, stride)
    else:
        data = MB.load_grid(name)
        cfg = MB.config_for(data)
        fold_cfg = replace(cfg, train_end=fold["train_end"],
                           val_end=fold["train_end"] + cfg.horizon,
                           length=fold["validation_end"])
        split = MB.build_split(data, fold_cfg, MB.STRIDE)

    experts = {e: MB.expert_predictions(MB.ID_TO_KEY.get(e, e), fold_cfg, split, device, data)
               for e in (a, b)}

    n_series = data["y"].shape[0]
    origins = np.asarray(split.test.origins)
    axis = np.repeat(np.arange(n_series), split.test.n_origins)
    # split.scale is per series; split.test.scale is already one entry per row.
    s = np.maximum(split.scale[axis].astype(np.float64), 1e-9)[:, None]
    mask = split.test.target_mask.astype(np.float64)
    yn = split.test.target.astype(np.float64) / s
    pa = experts[a].astype(np.float64) / s
    pb = experts[b].astype(np.float64) / s
    table = quadratic_from_arrays(pa, pb, yn, mask, axis)
    keep = mask.sum(axis=1) > 0

    lookback, horizon = fold_cfg.lookback, fold_cfg.horizon
    observed = observed_matrix(data)
    hist_y = split.test.history.astype(np.float64)                     # raw y, (N*W, L)
    hist_obs = _history_stack(observed, origins, lookback).reshape(len(hist_y), lookback)
    scale_row = np.maximum(split.test.scale.astype(np.float64), 1e-9)[:, None]

    # Unobserved steps carry a numerical zero in the value and occurrence
    # channels, and the mask channel is what tells them apart from a real zero.
    y_norm = np.where(hist_obs, hist_y / scale_row, 0.0)
    occurrence = np.where(hist_obs & (hist_y > 0), 1.0, 0.0)
    sequence = np.stack([y_norm, occurrence, hist_obs.astype(np.float64)], axis=-1)
    context = np.concatenate([pa, pb], axis=1)

    return FoldBundle(
        block=add_targets(table),
        sequence=sequence[keep].astype(np.float32),
        context=context[keep].astype(np.float32),
        series_id=np.asarray(data["series_id"])[axis][keep],
        origin=np.tile(origins, n_series)[keep],
        lookback=lookback, horizon=horizon,
        scale_source_end=int(fold_cfg.train_end),
    )
