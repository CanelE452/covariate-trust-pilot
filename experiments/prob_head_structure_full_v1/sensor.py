"""Teacher-disagreement failure sensor and its frozen action policy.

The decision row at ``t`` may use everything realized up to ``t+h`` and predicts what
happens on ``[t+h, t+2h)``. No observation at or after ``t+h`` enters a feature, and no
next-origin target is ever read, so the prequential contract holds by construction.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

DISAGREEMENT_COMPONENTS = (
    "D_zero",
    "D_center",
    "D_tail",
    "D_cdf",
    "D_mean",
    "D_winner_entropy",
)
DELTA_COMPONENTS = ("D_zero", "D_center", "D_tail", "D_cdf")
WINNER_ENTROPY_TEMPERATURE = 0.5
ACTION_FACTOR_GRID = (1.05, 1.1, 1.2, 1.35, 1.5)
FLAG_QUANTILE = 0.8
HEAD_TIE_ORDER = ("NB", "HSNB", "TWEEDIE_FULL")
SENSOR_FEATURE_SETS = ("C0", "C1", "C2", "C3")
BASELINE_CHANGE_FEATURES = (
    "previous_realized_residual",
    "zero_ratio_change",
    "scale_change",
    "last_event_gap_change",
    "recent_target_variance",
)
_BASELINE_MISSING_PRONE = ("last_event_gap_change", "recent_target_variance")
RECENT_VARIANCE_WINDOW = 96
SCALE_EPSILON = 1e-8


class SensorGeometryBlocked(RuntimeError):
    """A dataset cannot supply a valid sensor training pair."""


class SingleClassMetricUndefined(ValueError):
    """A sensor metric is undefined because only one class is present."""


def _median_index(quantile_grid: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.asarray(quantile_grid, dtype=np.float64) - 0.5)))


def _nearest_index(quantile_grid: np.ndarray, probability: float) -> int:
    return int(np.argmin(np.abs(np.asarray(quantile_grid, dtype=np.float64) - probability)))


def disagreement_components(
    *,
    p_zero: np.ndarray,
    quantiles: np.ndarray,
    predictive_mean: np.ndarray,
    scale: float,
    quantile_grid: Sequence[float] | None = None,
) -> dict[str, float]:
    """Reduce teacher disagreement to its declared components for one series-origin.

    ``p_zero`` and ``predictive_mean`` are ``[head, horizon]``; ``quantiles`` is
    ``[head, horizon, n_q]``. Each component takes the population variance across
    teachers at every horizon step and then averages over the horizon.
    """
    zero = np.asarray(p_zero, dtype=np.float64)
    values = np.asarray(quantiles, dtype=np.float64)
    mean = np.asarray(predictive_mean, dtype=np.float64)
    series_scale = float(scale)
    if series_scale <= 0.0 or not np.isfinite(series_scale):
        raise ValueError("the train-only scale must be finite and positive")
    if zero.ndim != 2 or mean.shape != zero.shape or values.ndim != 3:
        raise ValueError("teacher arrays must be [head,horizon] and [head,horizon,n_q]")
    if values.shape[:2] != zero.shape:
        raise ValueError("quantiles must agree with p_zero on head and horizon")
    grid = (
        np.linspace(0.0, 1.0, values.shape[2])
        if quantile_grid is None
        else np.asarray(quantile_grid, dtype=np.float64)
    )

    scaled = values / series_scale
    center = scaled[:, :, _median_index(grid)]
    upper95 = scaled[:, :, _nearest_index(grid, 0.95)]
    upper99 = scaled[:, :, _nearest_index(grid, 0.99)]

    d_zero = float(np.mean(zero.var(axis=0, ddof=0)))
    d_center = float(np.mean(center.var(axis=0, ddof=0)))
    d_tail = float(0.5 * (np.mean(upper95.var(axis=0, ddof=0)) + np.mean(upper99.var(axis=0, ddof=0))))
    d_mean = float(np.mean((mean / series_scale).var(axis=0, ddof=0)))

    pairwise, per_head = _cdf_distances(values, zero)
    d_cdf = float(pairwise)
    shifted = -np.asarray(per_head, dtype=np.float64) / WINNER_ENTROPY_TEMPERATURE
    shifted = shifted - shifted.max()
    soft = np.exp(shifted)
    soft = soft / soft.sum()
    entropy = float(-np.sum(np.where(soft > 0.0, soft * np.log(soft), 0.0)))

    return {
        "D_zero": d_zero,
        "D_center": d_center,
        "D_tail": d_tail,
        "D_cdf": d_cdf,
        "D_mean": d_mean,
        "D_winner_entropy": entropy,
    }


def _cdf_distances(quantiles: np.ndarray, p_zero: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean pairwise CDF distance on the target-free per-step support, plus per-head means."""
    heads, horizon, _ = quantiles.shape
    pair_totals: list[float] = []
    head_totals = np.zeros((heads, heads), dtype=np.float64)
    for step in range(horizon):
        support = np.unique(np.concatenate([[0.0], quantiles[:, step, :].reshape(-1)]))
        curves = np.stack(
            [_step_cdf(quantiles[head, step, :], p_zero[head, step], support) for head in range(heads)]
        )
        for first in range(heads):
            for second in range(first + 1, heads):
                distance = float(np.mean(np.abs(curves[first] - curves[second])))
                pair_totals.append(distance)
                head_totals[first, second] += distance
                head_totals[second, first] += distance
    mean_pairwise = float(np.mean(pair_totals)) if pair_totals else 0.0
    per_head = head_totals.sum(axis=1) / max(horizon * (heads - 1), 1)
    return mean_pairwise, per_head


def _step_cdf(step_quantiles: np.ndarray, zero_mass: float, support: np.ndarray) -> np.ndarray:
    """Right-continuous empirical CDF implied by one head's step quantiles plus its atom."""
    ordered = np.sort(np.asarray(step_quantiles, dtype=np.float64))
    fractions = (np.searchsorted(ordered, support, side="right")) / max(ordered.size, 1)
    return np.clip(np.maximum(fractions, float(zero_mass) * (support >= 0.0)), 0.0, 1.0)


def disagreement_deltas(
    current: Mapping[str, float], previous: Mapping[str, float] | None
) -> dict[str, float]:
    """Current minus the same series' immediately previous origin component."""
    row: dict[str, float] = {}
    for name in DELTA_COMPONENTS:
        if previous is None or name not in previous or not np.isfinite(float(previous[name])):
            value = float("nan")
        else:
            value = float(current[name]) - float(previous[name])
        row[f"Delta_{name}"] = value
        row[f"Delta_{name}__missing"] = float(np.isnan(value))
    return row


def select_inner_pair_origins(
    *, lookback: int, horizon: int, model_train_end: int
) -> tuple[int, ...]:
    """Current origins whose feature and target horizons both fit inside model_train."""
    lowest = int(lookback)
    highest = int(model_train_end) - 2 * int(horizon)
    if highest < lowest:
        raise SensorGeometryBlocked(
            "REAL_C_SENSOR_GEOMETRY_BLOCKED: no valid inner sensor pair exists"
        )
    stride = int(horizon)
    spaced = [highest - offset * stride for offset in range(8)]
    if min(spaced) >= lowest:
        return tuple(sorted(spaced))
    origins = sorted({int(round(float(value))) for value in np.linspace(lowest, highest, 8)})
    if len(origins) < 4:
        raise SensorGeometryBlocked(
            "REAL_C_SENSOR_GEOMETRY_BLOCKED: fewer than four unique inner sensor pairs"
        )
    return tuple(origins)


def outer_feature_origins(target_origins: Sequence[int], *, horizon: int) -> tuple[int, ...]:
    """Each outer decision row sits exactly one horizon before its target origin."""
    return tuple(int(origin) - int(horizon) for origin in target_origins)


def target_labels(
    *,
    next_scrps: float,
    scrps_threshold: float,
    coverage_90: float,
    coverage_95: float,
    zero_calibration: float,
    zero_threshold: float,
    best_head_changed: bool,
) -> dict[str, int]:
    """The four frozen next-origin failure labels."""
    return {
        "target_1": int(float(next_scrps) > float(scrps_threshold)),
        "target_2": int(float(coverage_90) < 0.90 or float(coverage_95) < 0.95),
        "target_3": int(float(zero_calibration) > float(zero_threshold)),
        "target_4": int(bool(best_head_changed)),
    }


def _valid_window(
    values: np.ndarray, start: int, stop: int, available_from: int
) -> np.ndarray:
    low = max(int(start), int(available_from))
    high = max(int(stop), low)
    return values[low:high]


def _last_positive_gap(values: np.ndarray, boundary: int, available_from: int) -> float:
    window = _valid_window(values, available_from, boundary, available_from)
    positions = np.flatnonzero(window > 0.0)
    if positions.size == 0:
        return float("nan")
    return float(boundary - 1 - (int(available_from) + int(positions[-1])))


def baseline_change_features(
    values: Sequence[float] | np.ndarray,
    *,
    current_origin: int,
    horizon: int,
    available_from: int,
    scale: float,
    p0_predictive_mean: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """The five C0 change features, read only from observations before ``t+h``."""
    series = np.asarray(values, dtype=np.float64)
    t = int(current_origin)
    h = int(horizon)
    boundary = t + h
    series_scale = float(scale)
    if series_scale <= 0.0 or not np.isfinite(series_scale):
        raise ValueError("the train-only scale must be finite and positive")
    if boundary > series.size:
        raise ValueError("the decision boundary lies beyond the observed panel")

    realized = _valid_window(series, t, boundary, available_from)
    previous = _valid_window(series, t - h, t, available_from)
    forecast = np.asarray(p0_predictive_mean, dtype=np.float64)
    if forecast.shape != (h,):
        raise ValueError("the P0 predictive mean must cover exactly one horizon")

    residual = (
        float(np.mean(np.abs(forecast[: realized.size] - realized)) / series_scale)
        if realized.size
        else float("nan")
    )
    zero_change = (
        float(np.mean(realized == 0.0) - np.mean(previous == 0.0))
        if realized.size and previous.size
        else float("nan")
    )
    scale_change = (
        float(
            np.sqrt(np.mean(realized**2)) / series_scale
            - np.sqrt(np.mean(previous**2)) / series_scale
        )
        if realized.size and previous.size
        else float("nan")
    )
    gap_now = _last_positive_gap(series, boundary, available_from)
    gap_then = _last_positive_gap(series, t, available_from)
    gap_change = (
        float((gap_now - gap_then) / h)
        if np.isfinite(gap_now) and np.isfinite(gap_then)
        else float("nan")
    )
    variance_window = _valid_window(
        series, max(boundary - RECENT_VARIANCE_WINDOW, available_from), boundary, available_from
    )
    recent_variance = (
        float(variance_window.var(ddof=0) / (series_scale**2)) if variance_window.size >= 2 else float("nan")
    )

    row = {
        "previous_realized_residual": residual,
        "zero_ratio_change": zero_change,
        "scale_change": scale_change,
        "last_event_gap_change": gap_change,
        "recent_target_variance": recent_variance,
    }
    for name in _BASELINE_MISSING_PRONE:
        row[f"{name}__missing"] = float(np.isnan(row[name]))
    return row


def sensor_feature_names(feature_set: str) -> tuple[str, ...]:
    """Frozen column order for each declared sensor feature set."""
    if feature_set not in SENSOR_FEATURE_SETS:
        raise ValueError(f"unknown sensor feature set {feature_set!r}")
    baseline = tuple(BASELINE_CHANGE_FEATURES) + tuple(
        f"{name}__missing" for name in _BASELINE_MISSING_PRONE
    )
    disagreement = tuple(DISAGREEMENT_COMPONENTS) + tuple(
        f"Delta_{name}" for name in DELTA_COMPONENTS
    ) + tuple(f"Delta_{name}__missing" for name in DELTA_COMPONENTS)
    if feature_set == "C0":
        return baseline
    if feature_set == "C1":
        return disagreement
    if feature_set == "C2":
        return baseline + disagreement
    return baseline + ("D_total", "D_total__missing")


def scalar_total_disagreement(components: Mapping[str, float]) -> dict[str, float]:
    """C3 collapses the components into one scalar so C2 can be tested against it."""
    values = [float(components[name]) for name in ("D_zero", "D_center", "D_tail", "D_cdf", "D_mean")]
    total = float(np.sum(values)) if all(np.isfinite(values)) else float("nan")
    return {"D_total": total, "D_total__missing": float(np.isnan(total))}


def flag_threshold(scores: Sequence[float] | np.ndarray, *, require_flagged: bool = False) -> float:
    """q80 of the validation scores with the frozen higher interpolation and no tie reranking."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise SingleClassMetricUndefined("the validation score vector is empty or nonfinite")
    threshold = float(np.quantile(values, FLAG_QUANTILE, method="higher"))
    if require_flagged and not np.any(values > threshold):
        raise SingleClassMetricUndefined(
            "no validation row is flagged, so the action metrics are undefined and C3 fails"
        )
    return threshold


def widen_quantiles(
    quantiles: Sequence[float] | np.ndarray,
    quantile_grid: Sequence[float] | np.ndarray,
    *,
    factor: float,
    p_zero: float,
) -> np.ndarray:
    """C_A2 widening that preserves the median and p0, then restores monotone consistency."""
    values = np.asarray(quantiles, dtype=np.float64)
    grid = np.asarray(quantile_grid, dtype=np.float64)
    if values.shape != grid.shape:
        raise ValueError("quantiles and the grid must share their shape")
    if float(factor) < 1.0:
        raise ValueError("the widening factor must be at least one")
    median_index = _median_index(grid)
    median = values[median_index]
    widened = np.maximum(0.0, median + float(factor) * (values - median))
    widened[median_index] = median
    widened = np.maximum.accumulate(widened)
    widened = np.where(grid <= float(p_zero), 0.0, widened)
    return np.maximum.accumulate(widened)


__all__ = [
    "ACTION_FACTOR_GRID",
    "BASELINE_CHANGE_FEATURES",
    "DELTA_COMPONENTS",
    "DISAGREEMENT_COMPONENTS",
    "HEAD_TIE_ORDER",
    "SENSOR_FEATURE_SETS",
    "SensorGeometryBlocked",
    "SingleClassMetricUndefined",
    "baseline_change_features",
    "disagreement_components",
    "disagreement_deltas",
    "flag_threshold",
    "outer_feature_origins",
    "scalar_total_disagreement",
    "select_inner_pair_origins",
    "sensor_feature_names",
    "target_labels",
    "widen_quantiles",
]
