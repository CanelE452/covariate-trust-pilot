"""Frozen retrieval features, prediction pairing, and runtime projections."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .metrics import pair_predictions, policy_scale_squared


CONTINUOUS_FEATURES = (
    "recent_zero_ratio",
    "log1p_time_since_last_positive",
    "log1p_mean_interarrival_gap",
    "log1p_interarrival_gap_cv",
    "log1p_positive_demand_mean",
    "log1p_positive_demand_cv",
    "log1p_recent_to_canonical_train_scale_rms_ratio",
    "log1p_point_hurdle_forecast_disagreement",
)
MISSING_FEATURES = (
    "time_since_last_positive_missing",
    "mean_interarrival_gap_missing",
    "interarrival_gap_cv_missing",
    "positive_demand_mean_missing",
    "positive_demand_cv_missing",
)
RETRIEVAL_FEATURES = CONTINUOUS_FEATURES + MISSING_FEATURES


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def extract_retrieval_features(
    history: np.ndarray,
    point_forecast: np.ndarray,
    hurdle_forecast: np.ndarray,
    *,
    canonical_train_scale: float,
) -> np.ndarray:
    """Return the frozen 8 transformed features plus 5 missing indicators."""

    values = np.asarray(history, dtype=np.float64)
    point = np.asarray(point_forecast, dtype=np.float64)
    hurdle = np.asarray(hurdle_forecast, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("history must be a nonempty one-dimensional array")
    if point.ndim != 1 or point.size == 0 or point.shape != hurdle.shape:
        raise ValueError("expert forecasts must be equal nonempty vectors")
    if not bool(
        np.isfinite(values).all()
        and np.isfinite(point).all()
        and np.isfinite(hurdle).all()
    ):
        raise ValueError("history and forecasts must be finite")
    if bool((values < 0.0).any()):
        raise ValueError("history must be nonnegative")
    scale = _finite_positive("canonical_train_scale", canonical_train_scale)

    positive_indices = np.flatnonzero(values > 0.0)
    positive_values = values[positive_indices]
    gaps = np.diff(positive_indices).astype(np.float64)

    time_since = (
        float(values.size - 1 - positive_indices[-1])
        if positive_indices.size >= 1
        else np.nan
    )
    mean_gap = float(gaps.mean()) if gaps.size >= 1 else np.nan
    gap_cv = (
        float(gaps.std(ddof=1) / gaps.mean())
        if gaps.size >= 2 and gaps.mean() > 0.0
        else np.nan
    )
    positive_mean = (
        float(positive_values.mean()) if positive_values.size >= 1 else np.nan
    )
    positive_cv = (
        float(positive_values.std(ddof=1) / positive_values.mean())
        if positive_values.size >= 2 and positive_values.mean() > 0.0
        else np.nan
    )
    missing_values = np.array(
        [time_since, mean_gap, gap_cv, positive_mean, positive_cv],
        dtype=np.float64,
    )
    missing_indicators = np.isnan(missing_values).astype(np.float64)
    transformed_missing_values = np.where(
        np.isnan(missing_values), 0.0, np.log1p(missing_values)
    )
    with np.errstate(over="raise", invalid="raise"):
        recent_rms_ratio = np.sqrt(np.mean(np.square(values))) / scale
        disagreement = np.sqrt(np.mean(np.square(hurdle - point))) / scale
    continuous = np.concatenate(
        (
            np.array([np.mean(values == 0.0)], dtype=np.float64),
            transformed_missing_values,
            np.log1p(np.array([recent_rms_ratio, disagreement])),
        )
    )
    features = np.concatenate((continuous, missing_indicators))
    if features.shape != (len(RETRIEVAL_FEATURES),) or not np.isfinite(features).all():
        raise ValueError("retrieval feature construction produced an invalid value")
    return features


def fit_robust_scaler(memory_features: np.ndarray) -> dict[str, np.ndarray]:
    """Fit median/IQR on resolved memory continuous features only."""

    features = np.asarray(memory_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(RETRIEVAL_FEATURES):
        raise ValueError("memory_features must have one column per frozen feature")
    if features.shape[0] == 0 or not bool(np.isfinite(features).all()):
        raise ValueError("memory_features must be nonempty and finite")
    continuous = features[:, : len(CONTINUOUS_FEATURES)]
    center = np.median(continuous, axis=0)
    q25, q75 = np.quantile(continuous, [0.25, 0.75], axis=0)
    iqr = q75 - q25
    constant = iqr == 0.0
    safe_iqr = np.where(constant, 1.0, iqr)
    return {
        "center": center,
        "scale": safe_iqr,
        "constant_continuous": constant,
    }


def apply_robust_scaler(
    features: np.ndarray, scaler: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Scale continuous columns and preserve explicit missing indicators."""

    array = np.asarray(features, dtype=np.float64)
    if array.shape[-1] != len(RETRIEVAL_FEATURES) or array.ndim not in (1, 2):
        raise ValueError("features must end with the frozen feature dimension")
    center = np.asarray(scaler["center"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    expected = (len(CONTINUOUS_FEATURES),)
    if center.shape != expected or scale.shape != expected:
        raise ValueError("scaler center and scale have invalid shapes")
    if not bool(np.isfinite(center).all() and np.isfinite(scale).all()) or bool(
        (scale <= 0.0).any()
    ):
        raise ValueError("scaler values must be finite with positive scales")
    result = array.copy()
    result[..., : len(CONTINUOUS_FEATURES)] = (
        result[..., : len(CONTINUOUS_FEATURES)] - center
    ) / scale
    if not bool(np.isfinite(result).all()):
        raise ValueError("robust scaling produced non-finite values")
    return result


def _prediction_array(
    predictions: Mapping[str, np.ndarray],
    key: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if key not in predictions:
        raise ValueError(f"prediction output is missing {key}")
    result = np.asarray(predictions[key], dtype=np.float64)
    if result.shape != expected_shape:
        raise ValueError(f"{key} has shape {result.shape}; expected {expected_shape}")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{key} must be finite")
    return result


def build_prediction_frame(
    data: Mapping[str, object],
    windows: object,
    point_predictions: Mapping[str, np.ndarray],
    hurdle_predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Build one fully paired step-level artifact using existing head names."""

    series_ids = np.asarray(data["series_id"]).astype(str)
    n_series = int(windows.n_series)
    n_origins = int(windows.n_origins)
    horizon = int(windows.target.shape[1])
    if series_ids.shape != (n_series,) or len(set(series_ids)) != n_series:
        raise ValueError("series_id must be unique and align with windows")
    expected_shape = (n_series * n_origins, horizon)
    point = _prediction_array(point_predictions, "mean_prediction", expected_shape)
    hurdle = _prediction_array(hurdle_predictions, "mean_prediction", expected_shape)
    hurdle_p = _prediction_array(hurdle_predictions, "p_prediction", expected_shape)
    hurdle_mu = _prediction_array(hurdle_predictions, "mu_prediction", expected_shape)
    if not bool(np.allclose(hurdle, hurdle_p * hurdle_mu, rtol=1e-5, atol=1e-6)):
        raise ValueError("Hurdle mean does not equal p_prediction * mu_prediction")

    keys = pd.DataFrame(
        {
            "dataset_id": np.repeat(str(data["name"]), n_series * n_origins * horizon),
            "series_id": np.repeat(series_ids, n_origins * horizon),
            "origin": np.tile(np.repeat(windows.origins, horizon), n_series),
            "step": np.tile(np.arange(horizon, dtype=np.int32), n_series * n_origins),
        }
    )
    point_arm = keys.assign(prediction=point.reshape(-1))
    hurdle_arm = keys.assign(prediction=hurdle.reshape(-1))
    paired = pair_predictions(point_arm, hurdle_arm).rename(
        columns={
            "point_prediction": "point_mean_prediction",
            "hurdle_prediction": "hurdle_mean_prediction",
        }
    )
    labels = keys.assign(
        y_observed=np.asarray(windows.target).reshape(-1),
        occurrence=np.asarray(windows.occurrence).reshape(-1),
        target_mask=np.asarray(windows.target_mask, dtype=bool).reshape(-1),
        hurdle_p_prediction=hurdle_p.reshape(-1),
        hurdle_mu_prediction=hurdle_mu.reshape(-1),
    )
    result = paired.merge(
        labels,
        on=["dataset_id", "series_id", "origin", "step"],
        validate="one_to_one",
        how="inner",
        sort=False,
    )
    expected_rows = n_series * n_origins * horizon
    if len(result) != expected_rows:
        raise ValueError("Point/Hurdle paired coverage is incomplete")
    return result.sort_values(
        ["dataset_id", "series_id", "origin", "step"], kind="mergesort"
    ).reset_index(drop=True)


def normalized_loss_frame(
    predictions: pd.DataFrame,
    data: Mapping[str, object],
    *,
    model_train_end: int,
    horizon: int,
) -> pd.DataFrame:
    """Aggregate primary normalized loss and secondary metrics by series-origin."""

    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, Integral):
        raise TypeError("horizon must be an integer")
    expected_horizon = int(horizon)
    if expected_horizon <= 0:
        raise ValueError("horizon must be positive")
    required = (
        "dataset_id",
        "series_id",
        "origin",
        "step",
        "y_observed",
        "point_mean_prediction",
        "hurdle_mean_prediction",
        "target_mask",
    )
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"predictions are missing columns: {missing}")
    series_ids = np.asarray(data["series_id"]).astype(str)
    if len(set(series_ids)) != len(series_ids):
        raise ValueError("data series_id values must be unique")
    values = np.asarray(data["y"])
    if values.ndim != 2 or values.shape[0] != len(series_ids):
        raise ValueError("data y must align with series_id")
    scales = {
        series_id: policy_scale_squared(values[index], model_train_end)
        for index, series_id in enumerate(series_ids)
    }
    frame = predictions.loc[:, required].copy()
    key_columns = ["dataset_id", "series_id", "origin", "step"]
    if bool(frame.duplicated(key_columns, keep=False).any()):
        raise ValueError("prediction step keys must be unique")
    frame["policy_scale_squared"] = frame["series_id"].map(scales)
    if frame["policy_scale_squared"].isna().any():
        raise ValueError("prediction artifact contains an unknown series_id")
    mask = frame["target_mask"].astype(bool)
    if not bool(mask.any()):
        raise ValueError("prediction artifact has no observed targets")
    point_error = frame["y_observed"] - frame["point_mean_prediction"]
    hurdle_error = frame["y_observed"] - frame["hurdle_mean_prediction"]
    frame["point_normalized_squared_error"] = (
        np.square(point_error) / frame["policy_scale_squared"]
    )
    frame["hurdle_normalized_squared_error"] = (
        np.square(hurdle_error) / frame["policy_scale_squared"]
    )
    frame["point_squared_error"] = np.square(point_error)
    frame["hurdle_squared_error"] = np.square(hurdle_error)
    frame["point_absolute_error"] = point_error.abs()
    frame["hurdle_absolute_error"] = hurdle_error.abs()
    frame = frame.loc[mask]
    group_columns = ["dataset_id", "series_id", "origin"]
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, observed=True):
        observed_steps = np.sort(group["step"].to_numpy(dtype=np.int64))
        if len(group) != expected_horizon or not np.array_equal(
            observed_steps, np.arange(expected_horizon, dtype=np.int64)
        ):
            raise ValueError(
                "each series-origin must contain one complete forecast horizon"
            )
        scale_values = group["policy_scale_squared"].unique()
        if len(scale_values) != 1:
            raise ValueError("policy scale changed within a series-origin")
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "n_steps": int(len(group)),
                "policy_scale_squared": float(scale_values[0]),
                "point_normalized_loss": float(
                    group["point_normalized_squared_error"].mean()
                ),
                "hurdle_normalized_loss": float(
                    group["hurdle_normalized_squared_error"].mean()
                ),
                "point_rmse": float(np.sqrt(group["point_squared_error"].mean())),
                "hurdle_rmse": float(np.sqrt(group["hurdle_squared_error"].mean())),
                "point_mae": float(group["point_absolute_error"].mean()),
                "hurdle_mae": float(group["hurdle_absolute_error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def project_full_seed0_runtime(
    *,
    smoke_train_seconds: Mapping[str, float],
    smoke_inference_seconds_per_origin: Mapping[str, float],
    smoke_n_series: int,
    full_series: Mapping[str, int],
    train_origins: Mapping[str, int],
    forecast_origins: Mapping[str, int],
) -> dict[str, object]:
    """Project four fits and their inference from measured M5 smoke work."""

    if isinstance(smoke_n_series, (bool, np.bool_)) or not isinstance(
        smoke_n_series, Integral
    ):
        raise TypeError("smoke_n_series must be an integer")
    reference_n = int(smoke_n_series)
    if reference_n <= 0 or "m5" not in train_origins:
        raise ValueError("a positive smoke size and M5 train origins are required")
    reference_windows = int(train_origins["m5"])
    if reference_windows <= 0:
        raise ValueError("M5 train-origin count must be positive")
    if set(full_series) != set(train_origins) or set(full_series) != set(forecast_origins):
        raise ValueError("dataset keys must match across projection inputs")

    train_per_arm = {
        arm: _finite_positive(f"smoke_train_seconds[{arm}]", seconds)
        for arm, seconds in smoke_train_seconds.items()
    }
    inference_per_arm = {
        arm: _finite_positive(
            f"smoke_inference_seconds_per_origin[{arm}]", seconds
        )
        for arm, seconds in smoke_inference_seconds_per_origin.items()
    }
    if set(train_per_arm) != set(inference_per_arm):
        raise ValueError("training and inference arms must match")

    by_dataset: dict[str, dict[str, float]] = {}
    total_training = 0.0
    total_inference = 0.0
    for dataset in sorted(full_series):
        series_count = int(full_series[dataset])
        origin_count = int(train_origins[dataset])
        prediction_origins = int(forecast_origins[dataset])
        if min(series_count, origin_count, prediction_origins) <= 0:
            raise ValueError("projection counts must be positive")
        series_factor = series_count / reference_n
        train_factor = series_factor * origin_count / reference_windows
        training = sum(train_per_arm.values()) * train_factor
        inference = (
            sum(inference_per_arm.values())
            * series_factor
            * prediction_origins
        )
        by_dataset[dataset] = {
            "training_seconds": float(training),
            "inference_seconds": float(inference),
            "total_seconds": float(training + inference),
        }
        total_training += training
        total_inference += inference
    return {
        "training_seconds": float(total_training),
        "inference_seconds": float(total_inference),
        "total_seconds": float(total_training + total_inference),
        "gpu_hours": float((total_training + total_inference) / 3600.0),
        "by_dataset": by_dataset,
    }


def project_retrieval_runtime(
    *,
    smoke_seconds: float,
    smoke_n_series: int,
    full_series: Mapping[str, int],
    evaluation_origins: int,
    retrieval_passes_per_dataset: int,
) -> dict[str, object]:
    """Project the current exact retrieval implementation from its smoke query.

    The smoke executes one evaluation origin with one resolved origin block.
    At the six full evaluation origins the database contains one through six
    blocks, so exact all-query/all-memory search scales with
    ``N**2 * sum(1..evaluation_origins)``.  The pass count makes source tuning
    and the real/C0/C1 target evaluations explicit rather than hiding them in
    the GPU estimate.
    """

    measured = _finite_positive("smoke_seconds", smoke_seconds)
    if isinstance(smoke_n_series, (bool, np.bool_)) or not isinstance(
        smoke_n_series, Integral
    ):
        raise TypeError("smoke_n_series must be an integer")
    reference_n = int(smoke_n_series)
    if reference_n <= 0:
        raise ValueError("smoke_n_series must be positive")
    for name, value in (
        ("evaluation_origins", evaluation_origins),
        ("retrieval_passes_per_dataset", retrieval_passes_per_dataset),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    origin_count = int(evaluation_origins)
    pass_count = int(retrieval_passes_per_dataset)
    resolved_units = origin_count * (origin_count + 1) // 2
    if not isinstance(full_series, Mapping) or not full_series:
        raise ValueError("full_series must be a nonempty mapping")

    by_dataset: dict[str, dict[str, float]] = {}
    total = 0.0
    for dataset in sorted(full_series):
        count = full_series[dataset]
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
            raise TypeError(f"full_series[{dataset}] must be an integer")
        series_count = int(count)
        if series_count <= 0:
            raise ValueError("full-series counts must be positive")
        quadratic_factor = (series_count / reference_n) ** 2
        seconds = measured * quadratic_factor * resolved_units * pass_count
        by_dataset[str(dataset)] = {
            "series": series_count,
            "quadratic_series_factor": float(quadratic_factor),
            "seconds": float(seconds),
            "hours": float(seconds / 3600.0),
        }
        total += seconds
    return {
        "smoke_seconds": measured,
        "smoke_n_series": reference_n,
        "evaluation_origins": origin_count,
        "resolved_origin_units": resolved_units,
        "retrieval_passes_per_dataset": pass_count,
        "by_dataset": by_dataset,
        "total_seconds": float(total),
        "cpu_hours": float(total / 3600.0),
        "complexity_model": "measured_seconds*(N/200)^2*sum(1..6)*passes",
    }
