"""Stage A of the external-validity SCREEN.

Loads the two prepared public benchmarks, computes train-only structure
descriptors, trains the synthetic study's Point and Hurdle-Mean DLinear
unchanged, and evaluates H1/H2/H3 against the frozen specification.

No new model, no new loss, no synthetic data.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ..om_factorization_killtest import evaluate as km_evaluate
from ..om_factorization_killtest import train as km_train
from ..unified_temporal_27_v3.config import ExperimentConfig
from ..unified_temporal_27_v3.training import make_windows, train_scale
from . import prereg

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "external_validity_screen"

DATASETS = {
    "m5": {"parquet": REPO / "data" / "processed" / "series.parquet",
           "leading_zero": "availability_aware"},
    "favorita": {"parquet": REPO / "data" / "processed" / "favorita_series.parquet",
                 "leading_zero": "raw"},
}


class ScreenFailure(RuntimeError):
    pass


# ---------------------------------------------------------------- data ----


def m5_availability(meta: pd.DataFrame) -> np.ndarray:
    """First day each (store, item) has a sell_prices row — M5's only signal."""
    calendar = pd.read_csv(REPO / "data" / "calendar.csv", usecols=["d", "wm_yr_wk"])
    calendar["day_idx"] = calendar["d"].str.slice(2).astype(int) - 1
    week_start = calendar.groupby("wm_yr_wk")["day_idx"].min()
    prices = pd.read_csv(REPO / "data" / "sell_prices.csv",
                         usecols=["store_id", "item_id", "wm_yr_wk"])
    first_week = prices.groupby(["store_id", "item_id"])["wm_yr_wk"].min()
    out = []
    for _, row in meta.iterrows():
        key = (row["store_id"], row["item_id"])
        if key not in first_week.index:
            raise ScreenFailure(f"no sell_prices row for {key}; availability "
                                "cannot be defined and the run must stop")
        out.append(int(week_start[first_week[key]]))
    return np.asarray(out, dtype=int)


def load_dataset(name: str) -> dict:
    spec = DATASETS[name]
    frame = pd.read_parquet(spec["parquet"])
    wide = frame.pivot_table(index="series_id", columns="timestamp",
                             values="target").sort_index()
    y = wide.to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        raise ScreenFailure(f"{name}: non-finite target; missing and zero must "
                            "be distinguishable before this point")
    if (y < 0).any():
        raise ScreenFailure(f"{name}: negative demand present")
    z = (y > 0).astype(np.float32)
    series_ids = wide.index.to_numpy().astype(str)

    first_positive = np.array([int(np.argmax(r > 0)) if (r > 0).any() else len(r)
                               for r in y])
    if spec["leading_zero"] == "availability_aware":
        meta = frame.groupby("series_id")[["item_id", "store_id"]].first()
        available_from = m5_availability(meta.loc[wide.index])
    else:
        available_from = np.zeros(len(series_ids), dtype=int)
    return {"name": name, "y": y, "z": z, "series_id": series_ids,
            "available_from": available_from, "first_positive": first_positive}


def config_for(name: str) -> ExperimentConfig:
    s = prereg.SPLITS[name]
    return ExperimentConfig(length=s["length"], lookback=s["lookback"],
                            horizon=s["horizon"], period=s["horizon"],
                            train_end=s["train_end"], val_end=s["val_end"])


# --------------------------------------------------------- descriptors ----


def _lag1(values: np.ndarray) -> float:
    """Lag-1 autocorrelation, NaN when it is not defined. Never zero-filled."""
    if values.size < 3 or np.std(values) == 0:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def describe_series(y_row: np.ndarray, start: int, train_end: int) -> dict:
    """Every descriptor from the train segment only."""
    segment = y_row[start:train_end]
    n_train = int(segment.size)
    events = np.flatnonzero(segment > 0)
    n_positive = int(events.size)
    out = {"n_train": n_train, "n_positive_train": n_positive,
           "zero_ratio_train": float(np.mean(segment <= 0)) if n_train else np.nan,
           "ADI_train": float(n_train / n_positive) if n_positive else np.nan,
           "CV2_positive_train": np.nan, "rho_interval_train": np.nan,
           "rho_magnitude_train": np.nan, "occurrence_binary_acf1_train": np.nan,
           "previous_gap_next_size_corr_train": np.nan}
    if n_positive > 1:
        positives = segment[events]
        mean_pos = float(positives.mean())
        std_pos = float(positives.std(ddof=1))
        out["CV2_positive_train"] = (std_pos / mean_pos) ** 2 if mean_pos > 0 else np.nan
        out["rho_magnitude_train"] = _lag1(positives.astype(float))
        gaps = np.diff(events).astype(float)
        out["rho_interval_train"] = _lag1(gaps)
        if gaps.size >= 3 and np.std(gaps) > 0 and np.std(positives[1:]) > 0:
            out["previous_gap_next_size_corr_train"] = float(
                np.corrcoef(gaps, positives[1:].astype(float))[0, 1])
    out["occurrence_binary_acf1_train"] = _lag1((segment > 0).astype(float))
    out["rho_interval_abs_train"] = abs(out["rho_interval_train"])
    return out


def descriptor_table(data: dict, cfg: ExperimentConfig, variant: str) -> pd.DataFrame:
    if variant == "raw":
        starts = np.zeros(len(data["series_id"]), dtype=int)
    elif variant == "availability_aware":
        starts = data["available_from"]
    elif variant == "first_positive":
        starts = data["first_positive"]
    else:
        raise ValueError(variant)
    rows = []
    for i, sid in enumerate(data["series_id"]):
        entry = describe_series(data["y"][i], int(starts[i]), cfg.train_end)
        entry.update({"series_id": sid, "segment_start": int(starts[i]),
                      "descriptor_variant": variant})
        rows.append(entry)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ training ----


def build_split(data: dict, cfg: ExperimentConfig, stride: int):
    """km_train.build_splits, but with the SCREEN's stride and an availability
    mask applied to train and validation targets.

    The test window starts after every series is available, so test targets are
    never masked and the primary estimand is untouched by this step.
    """
    arrays = {"y": data["y"], "z": data["z"]}
    scale = train_scale(arrays, cfg)
    train_stride = prereg.SPLITS["train_origin_stride"]
    train_o = km_train.dense_origins(0, cfg.train_end, cfg.horizon,
                                     cfg.lookback)[::train_stride]
    val_o = km_train.dense_origins(cfg.train_end, cfg.val_end, cfg.horizon,
                                   cfg.lookback)
    test_o = km_train.stride_origins(cfg.val_end, cfg.length, cfg.horizon, stride)

    train_w = make_windows(arrays, train_o, 0, cfg.train_end, cfg, scale)
    val_w = make_windows(arrays, val_o, cfg.train_end, cfg.val_end, cfg, scale)
    test_w = make_windows(arrays, test_o, cfg.val_end, cfg.length, cfg, scale)

    available = data["available_from"]
    if available.max() >= cfg.val_end:
        raise ScreenFailure("a series becomes available inside the evaluation "
                            "window; the test estimand would be contaminated")
    if available.max() > 0:
        train_w = _mask_before_availability(train_w, available, cfg)
        val_w = _mask_before_availability(val_w, available, cfg)

    y_train = data["y"][:, :cfg.train_end]
    z_train = data["z"][:, :cfg.train_end]
    positives = y_train[z_train > 0]
    variance = float(positives.var(ddof=1)) if positives.size > 1 else 1.0
    return km_train.Split(train_w, val_w, test_w, scale, max(variance, 1e-6))


def _mask_before_availability(windows, available: np.ndarray, cfg):
    """Drop targets dated before the item could be sold at all."""
    n_series, n_origins = windows.n_series, windows.n_origins
    step = np.arange(cfg.horizon)[None, :]
    absolute = windows.origins[:, None] + step                      # (W, H)
    allowed = absolute[None, :, :] >= available[:, None, None]      # (N, W, H)
    mask = windows.target_mask.reshape(n_series, n_origins, cfg.horizon)
    return replace(windows,
                   target_mask=(mask & allowed).reshape(n_series * n_origins,
                                                        cfg.horizon))


def test_metrics(prediction: dict, windows, n_series: int) -> dict:
    """Realized-y per-series metrics.

    km_evaluate.series_metrics always dereferences the oracle keys, which do
    not exist outside the synthetic study, so only its per-series reducer is
    reused here. The aggregation is therefore identical to the synthetic run;
    only the reference the error is measured against differs.
    """
    index = np.repeat(np.arange(n_series), windows.n_origins)
    mask = windows.target_mask.astype(np.float64)
    target = windows.target.astype(np.float64)
    occurrence = windows.occurrence.astype(np.float64)
    counts = np.maximum(np.bincount(index, weights=mask.sum(axis=1),
                                    minlength=n_series), 1.0)
    mean_hat = prediction["mean_prediction"].astype(np.float64)
    reduce = km_evaluate._per_series
    out = {
        "rmse_realized": np.sqrt(
            reduce((mean_hat - target) ** 2, mask, index, n_series) / counts),
        "mae_realized": reduce(np.abs(mean_hat - target), mask, index,
                               n_series) / counts,
    }
    if "p_prediction" in prediction:
        p_hat = np.clip(prediction["p_prediction"].astype(np.float64),
                        1e-7, 1 - 1e-7)
        mu_hat = prediction["mu_prediction"].astype(np.float64)
        out["occurrence_brier"] = reduce((p_hat - occurrence) ** 2, mask, index,
                                         n_series) / counts
        out["mean_p_hat_on_zero"] = reduce(
            p_hat, mask * (1.0 - occurrence), index, n_series) / np.maximum(
                np.bincount(index, weights=(mask * (1 - occurrence)).sum(axis=1),
                            minlength=n_series), 1.0)
        positive = mask * occurrence
        pos_counts = np.maximum(
            np.bincount(index, weights=positive.sum(axis=1),
                        minlength=n_series), 1.0)
        out["mean_p_hat_on_positive"] = reduce(p_hat, positive, index,
                                               n_series) / pos_counts
        out["positive_magnitude_rmse"] = np.sqrt(
            reduce((mu_hat - target) ** 2, positive, index, n_series) / pos_counts)
    return out


# ------------------------------------------------------------ analysis ----


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def _bootstrap_ci(fn, n: int, rng) -> tuple:
    draws = np.array([fn(rng.integers(0, n, size=n))
                      for _ in range(prereg.BOOTSTRAP["draws"])], dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size < 100:
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))


def analyse(table: pd.DataFrame, threshold: int) -> dict:
    """H1, H2 and H3 on one dataset at one eligibility threshold."""
    rng = np.random.default_rng(prereg.BOOTSTRAP["seed"])
    pool = table[table["n_positive_train"] >= threshold].copy()
    pool = pool[np.isfinite(pool["delta_rmse"])]
    n = len(pool)
    result = {"threshold": threshold, "n_mechanism_eligible": int(n)}
    if n < prereg.STOP_RULES["mechanism_eligible_min"]:
        result["status"] = "INSUFFICIENT_ELIGIBLE_N"
        return result

    x = pool["rho_interval_abs_train"].to_numpy(float)
    d = pool["delta_rmse"].to_numpy(float)
    result["H1"] = {"spearman": _spearman(x, d), "n": int(np.isfinite(x).sum())}
    result["H1"]["ci"] = _bootstrap_ci(lambda idx: _spearman(x[idx], d[idx]), n, rng)

    adi = pool["ADI_train"].to_numpy(float)
    mag = pool["rho_magnitude_train"].to_numpy(float)
    high_adi = adi >= np.nanmedian(adi)
    low_occ = x <= np.nanquantile(x, 1 / 3)
    persistent = mag >= np.nanquantile(mag, 2 / 3)
    candidate = high_adi & low_occ & persistent
    control = high_adi & low_occ & ~persistent
    result["H2"] = {
        "n_candidate": int(candidate.sum()), "n_control": int(control.sum()),
        "mean_candidate": float(np.nanmean(d[candidate])) if candidate.any() else np.nan,
        "median_candidate": float(np.nanmedian(d[candidate])) if candidate.any() else np.nan,
        "mean_control": float(np.nanmean(d[control])) if control.any() else np.nan,
        "point_win_rate_candidate": float(np.mean(d[candidate] < 0)) if candidate.any() else np.nan,
        "point_win_rate_control": float(np.mean(d[control] < 0)) if control.any() else np.nan,
    }
    result["H2"]["difference"] = (result["H2"]["mean_candidate"]
                                  - result["H2"]["mean_control"])
    if candidate.sum() < prereg.STOP_RULES["point_candidate_min"]:
        result["H2"]["status"] = "INSUFFICIENT_GROUP_N"
    else:
        result["H2"]["status"] = "OK"
        result["H2"]["difference_ci"] = _bootstrap_ci(
            lambda idx: (np.nanmean(d[idx][candidate[idx]])
                         - np.nanmean(d[idx][control[idx]]))
            if candidate[idx].any() and control[idx].any() else np.nan, n, rng)
        result["H2"]["mean_candidate_ci"] = _bootstrap_ci(
            lambda idx: np.nanmean(d[idx][candidate[idx]])
            if candidate[idx].any() else np.nan, n, rng)

    low = ~high_adi
    result["H3"] = {
        "corr_high_ADI": _spearman(x[high_adi], d[high_adi]),
        "corr_low_ADI": _spearman(x[low], d[low]),
        "n_high": int(high_adi.sum()), "n_low": int(low.sum()),
    }
    result["H3"]["difference"] = (result["H3"]["corr_high_ADI"]
                                  - result["H3"]["corr_low_ADI"])
    result["H3"]["difference_ci"] = _bootstrap_ci(
        lambda idx: _spearman(x[idx][high_adi[idx]], d[idx][high_adi[idx]])
        - _spearman(x[idx][low[idx]], d[idx][low[idx]]), n, rng)
    result["status"] = "OK"
    return result
