"""Origin-bounded structure features for the B router and the C sensor.

Every scalar is computed from availability-valid observations strictly before its
declared boundary, so mutating the boundary value or anything later cannot change a
feature. Undefined scalars keep an explicit NaN plus a dedicated missing indicator;
they are never silently encoded as zero.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

SCALE_EPSILON = 1e-8
RECENT_ZERO_WINDOW = 28
RECENT_EVENT_WINDOW = 96
AUTOCORRELATION_MAX_VALUES = 20
AUTOCORRELATION_MIN_VALUES = 3
SEASONAL_PERIOD = 7
SEASONAL_DATASETS = ("m5", "online_retail")

BASELINE_FEATURE_NAMES = (
    "train_ADI",
    "train_positive_CV2",
    "train_scale",
    "train_zero_ratio",
)

_MISSING_PRONE = (
    "train_ADI",
    "train_positive_CV2",
    "recent_mean_gap",
    "recent_gap_CV",
    "recent_positive_mean",
    "recent_positive_CV",
    "interval_autocorrelation",
    "magnitude_autocorrelation",
    "time_since_last_positive",
)

_TEMPORAL_SCALARS = (
    "recent_zero_ratio",
    "train_zero_ratio",
    "time_since_last_positive",
    "recent_mean_gap",
    "recent_gap_CV",
    "train_ADI",
    "train_positive_CV2",
    "recent_positive_mean",
    "recent_positive_CV",
    "recent_train_scale_ratio",
    "interval_autocorrelation",
    "magnitude_autocorrelation",
    "seasonal_sin",
    "seasonal_cos",
)

TEMPORAL_FEATURE_NAMES = _TEMPORAL_SCALARS + tuple(
    f"{name}__missing" for name in _MISSING_PRONE
)


class FeatureBoundaryError(ValueError):
    """A feature request would read at or after its declared time boundary."""


def _valid_prefix(
    values: np.ndarray, available_from: int, boundary: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the availability-valid observations strictly before ``boundary`` and their indices."""
    start = max(int(available_from), 0)
    stop = int(boundary)
    if stop <= start:
        raise FeatureBoundaryError("the availability-valid prefix is empty before the boundary")
    index = np.arange(start, stop)
    return np.asarray(values, dtype=np.float64)[start:stop], index


def _cv(sample: np.ndarray) -> float:
    """Sample coefficient of variation with ddof=1; undefined below two values."""
    if sample.size < 2:
        return float("nan")
    mean = float(sample.mean())
    if mean == 0.0:
        return float("nan")
    return float(sample.std(ddof=1) / mean)


def _lag_one_autocorrelation(sample: np.ndarray) -> float:
    """Lag-1 Pearson on the most recent values; undefined without variance in both vectors."""
    recent = sample[-AUTOCORRELATION_MAX_VALUES:]
    if recent.size < AUTOCORRELATION_MIN_VALUES:
        return float("nan")
    first, second = recent[:-1], recent[1:]
    if first.std() == 0.0 or second.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def train_descriptors_for_series(
    values: Sequence[float] | np.ndarray,
    *,
    available_from: int,
    train_end: int,
) -> dict[str, float]:
    """Availability-valid model_train descriptors; no observation at or after train_end."""
    window, _ = _valid_prefix(np.asarray(values, dtype=np.float64), available_from, train_end)
    positive = window[window > 0.0]
    zero_ratio = float(np.mean(window == 0.0))
    scale = float(np.sqrt(np.mean(window**2) + SCALE_EPSILON))
    adi = float(window.size / positive.size) if positive.size else float("nan")
    if positive.size >= 2 and float(positive.mean()) != 0.0:
        cv2 = float((positive.std(ddof=1) / positive.mean()) ** 2)
    else:
        cv2 = float("nan")
    return {
        "train_ADI": adi,
        "train_positive_CV2": cv2,
        "train_scale": scale,
        "train_zero_ratio": zero_ratio,
    }


def temporal_features_for_series(
    values: Sequence[float] | np.ndarray,
    *,
    origin: int,
    available_from: int,
    train_end: int,
    dataset_id: str,
) -> dict[str, float]:
    """Build one origin-bounded feature row for a single series."""
    series = np.asarray(values, dtype=np.float64)
    origin = int(origin)
    if origin <= int(available_from):
        raise FeatureBoundaryError("the origin must leave at least one observed value")
    if int(train_end) > origin:
        raise FeatureBoundaryError("model_train must end at or before the feature origin")
    if origin > series.size:
        raise FeatureBoundaryError("the origin lies beyond the observed panel")

    history, index = _valid_prefix(series, available_from, origin)
    descriptors = train_descriptors_for_series(
        series, available_from=available_from, train_end=train_end
    )

    zero_window = history[-RECENT_ZERO_WINDOW:]
    recent_zero_ratio = float(np.mean(zero_window == 0.0))
    recent_rms = float(np.sqrt(np.mean(zero_window**2) + SCALE_EPSILON))
    scale_ratio = recent_rms / descriptors["train_scale"]

    positive_positions = index[history > 0.0]
    if positive_positions.size:
        time_since_last_positive = float(origin - 1 - int(positive_positions[-1]))
    else:
        time_since_last_positive = float("nan")

    event_start = max(origin - RECENT_EVENT_WINDOW, int(available_from))
    window_positions = positive_positions[positive_positions >= event_start]
    gaps = np.diff(window_positions.astype(np.float64)) if window_positions.size >= 2 else np.array([])
    recent_mean_gap = float(gaps.mean()) if gaps.size else float("nan")
    recent_gap_cv = _cv(gaps) if gaps.size else float("nan")

    window_magnitudes = series[window_positions] if window_positions.size else np.array([])
    recent_positive_mean = float(window_magnitudes.mean()) if window_magnitudes.size else float("nan")
    recent_positive_cv = _cv(window_magnitudes) if window_magnitudes.size else float("nan")

    all_gaps = np.diff(positive_positions.astype(np.float64)) if positive_positions.size >= 2 else np.array([])
    interval_autocorrelation = _lag_one_autocorrelation(all_gaps) if all_gaps.size else float("nan")
    magnitudes = series[positive_positions] if positive_positions.size else np.array([])
    magnitude_autocorrelation = _lag_one_autocorrelation(magnitudes) if magnitudes.size else float("nan")

    if str(dataset_id).lower() in SEASONAL_DATASETS:
        angle = 2.0 * np.pi * (origin % SEASONAL_PERIOD) / SEASONAL_PERIOD
        seasonal_sin, seasonal_cos = float(np.sin(angle)), float(np.cos(angle))
    else:
        seasonal_sin = seasonal_cos = 0.0

    row: dict[str, float] = {
        "recent_zero_ratio": recent_zero_ratio,
        "train_zero_ratio": descriptors["train_zero_ratio"],
        "time_since_last_positive": time_since_last_positive,
        "recent_mean_gap": recent_mean_gap,
        "recent_gap_CV": recent_gap_cv,
        "train_ADI": descriptors["train_ADI"],
        "train_positive_CV2": descriptors["train_positive_CV2"],
        "recent_positive_mean": recent_positive_mean,
        "recent_positive_CV": recent_positive_cv,
        "recent_train_scale_ratio": scale_ratio,
        "interval_autocorrelation": interval_autocorrelation,
        "magnitude_autocorrelation": magnitude_autocorrelation,
        "seasonal_sin": seasonal_sin,
        "seasonal_cos": seasonal_cos,
    }
    for name in _MISSING_PRONE:
        row[f"{name}__missing"] = float(np.isnan(row[name]))
    return row


def build_feature_matrix(
    rows: Sequence[Mapping[str, float]],
    *,
    feature_set: str | Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Stack feature rows in one frozen column order."""
    if feature_set == "baseline":
        names = list(BASELINE_FEATURE_NAMES)
    elif feature_set == "temporal":
        names = list(TEMPORAL_FEATURE_NAMES)
    else:
        names = [str(name) for name in feature_set]
    if not names:
        raise ValueError("a feature set must name at least one column")
    if not rows:
        return np.zeros((0, len(names)), dtype=np.float64), names
    matrix = np.array(
        [[float(row[name]) for name in names] for row in rows], dtype=np.float64
    )
    return matrix, names


def fold_median_imputer(
    train_matrix: np.ndarray, feature_names: Sequence[str]
) -> dict[str, Any]:
    """Fit fold-train medians and a standard scaler; never touch heldout or outer rows."""
    matrix = np.asarray(train_matrix, dtype=np.float64)
    names = [str(name) for name in feature_names]
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("the training matrix must have one column per feature name")

    median = np.zeros(matrix.shape[1], dtype=np.float64)
    mean = np.zeros(matrix.shape[1], dtype=np.float64)
    scale = np.ones(matrix.shape[1], dtype=np.float64)
    all_missing: list[str] = []

    for column in range(matrix.shape[1]):
        observed = matrix[:, column][np.isfinite(matrix[:, column])]
        if observed.size == 0:
            all_missing.append(names[column])
            continue
        median[column] = float(np.median(observed))
        filled = np.where(np.isfinite(matrix[:, column]), matrix[:, column], median[column])
        mean[column] = float(filled.mean())
        deviation = float(filled.std())
        scale[column] = deviation if deviation > 0.0 else 1.0

    def transform(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(names):
            raise ValueError("transform input must have one column per feature name")
        filled = np.where(np.isfinite(data), data, median[None, :])
        return (filled - mean[None, :]) / scale[None, :]

    return {
        "feature_names": names,
        "median": median,
        "mean": mean,
        "scale": scale,
        "all_missing_features": all_missing,
        "record": "ALL_MISSING_TRAIN_FEATURE" if all_missing else "COMPLETE_TRAIN_FEATURES",
        "transform": transform,
    }


__all__ = [
    "BASELINE_FEATURE_NAMES",
    "TEMPORAL_FEATURE_NAMES",
    "FeatureBoundaryError",
    "build_feature_matrix",
    "fold_median_imputer",
    "temporal_features_for_series",
    "train_descriptors_for_series",
]
