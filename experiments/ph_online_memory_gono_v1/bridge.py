"""Bridge actual trainer windows and predictions into policy-evaluation cases."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd

from .evaluation import build_series_origin_cases
from .metrics import policy_scale_squared


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _policy_steps(
    predictions: pd.DataFrame,
    data: Mapping[str, object],
    *,
    model_train_end: int,
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    series_ids = np.asarray(data["series_id"]).astype(str)
    values = np.asarray(data["y"])
    if series_ids.ndim != 1 or len(set(series_ids)) != len(series_ids):
        raise ValueError("data series_id must be a unique vector")
    if values.ndim != 2 or values.shape[0] != len(series_ids):
        raise ValueError("data y must align with series_id")
    scales = {
        series_id: policy_scale_squared(values[index], model_train_end)
        for index, series_id in enumerate(series_ids)
    }
    result = predictions.copy()
    result["series_id"] = result["series_id"].astype(str)
    result["policy_scale_squared"] = result["series_id"].map(scales)
    if result["policy_scale_squared"].isna().any():
        raise ValueError("predictions contain a series absent from data")
    return result


def _history_frame(
    data: Mapping[str, object],
    windows: object,
    *,
    lookback: int,
) -> pd.DataFrame:
    width = _positive_integer("lookback", lookback)
    series_ids = np.asarray(data["series_id"]).astype(str)
    values = np.asarray(data["y"], dtype=np.float64)
    origins = np.asarray(windows.origins)
    n_series = int(windows.n_series)
    n_origins = int(windows.n_origins)
    if series_ids.shape != (n_series,) or values.shape[0] != n_series:
        raise ValueError("window series count does not align with data")
    if origins.shape != (n_origins,) or not np.issubdtype(origins.dtype, np.integer):
        raise ValueError("window origins must be an integer vector")
    stored_history = np.asarray(windows.history, dtype=np.float64)
    stored_scale = np.asarray(windows.scale, dtype=np.float64)
    if stored_history.shape != (n_series * n_origins, width):
        raise ValueError("window history shape does not match series x origins")
    if stored_scale.shape != (n_series * n_origins,):
        raise ValueError("window scale shape does not match series x origins")
    history_cube = stored_history.reshape(n_series, n_origins, width)
    scale_matrix = stored_scale.reshape(n_series, n_origins)
    rows: list[dict[str, object]] = []
    for series_index, series_id in enumerate(series_ids):
        if not bool(np.equal(scale_matrix[series_index], scale_matrix[series_index, 0]).all()):
            raise ValueError("canonical train scale changes across origins")
        for origin_index, raw_origin in enumerate(origins):
            origin = int(raw_origin)
            if origin < width or origin > values.shape[1]:
                raise ValueError("origin does not contain a full pre-origin history")
            expected = values[series_index, origin - width : origin]
            observed = history_cube[series_index, origin_index]
            if not np.array_equal(observed, expected):
                raise ValueError(
                    "stored history is not the exact pre-origin slice; possible leakage"
                )
            rows.append(
                {
                    "dataset_id": str(data["name"]),
                    "series_id": series_id,
                    "origin": origin,
                    "history": observed.copy(),
                    "canonical_train_scale": float(
                        scale_matrix[series_index, origin_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_policy_cases(
    predictions: pd.DataFrame,
    data: Mapping[str, object],
    windows: object,
    *,
    model_train_end: int,
    horizon: int,
    lookback: int,
) -> pd.DataFrame:
    """Create policy cases with train-only scales and verified causal histories."""

    forecast_horizon = _positive_integer("horizon", horizon)
    history_width = _positive_integer("lookback", lookback)
    steps = _policy_steps(
        predictions, data, model_train_end=model_train_end
    )
    histories = _history_frame(data, windows, lookback=history_width)
    return build_series_origin_cases(
        steps,
        histories,
        horizon=forecast_horizon,
        lookback=history_width,
    )


__all__ = ["build_policy_cases"]
