"""Pure causal-memory and nearest-neighbor helpers."""

from numbers import Integral, Real

import numpy as np
import pandas as pd


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def make_retrieval_key(
    history: np.ndarray, query_origin: int, lookback: int
) -> np.ndarray:
    """Copy the trailing history ending strictly before ``query_origin``."""

    array = np.asarray(history)
    if array.ndim != 1:
        raise ValueError("history must be one-dimensional")
    origin = _integer("query_origin", query_origin, minimum=0)
    width = _integer("lookback", lookback, minimum=1)
    if origin > array.shape[0]:
        raise ValueError("query_origin exceeds the available history")
    if width > origin:
        raise ValueError("lookback exceeds the pre-origin history")

    key = np.asarray(array[origin - width : origin], dtype=np.float64).copy()
    if not bool(np.isfinite(key).all()):
        raise ValueError("retrieval-key history must all be finite")
    return key


def resolved_memory_cases(
    cases: pd.DataFrame, query_origin: int, horizon: int
) -> pd.DataFrame:
    """Return cases whose complete forecast horizon is known at query time."""

    if not isinstance(cases, pd.DataFrame):
        raise TypeError("cases must be a pandas DataFrame")
    if "origin" not in cases.columns:
        raise ValueError("cases must contain an origin column")
    origin = _integer("query_origin", query_origin, minimum=0)
    forecast_horizon = _integer("horizon", horizon, minimum=1)
    try:
        case_origins = pd.to_numeric(cases["origin"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("case origins must be numeric") from exc
    if not bool(np.isfinite(case_origins).all()):
        raise ValueError("case origins must all be finite")
    if bool((case_origins < 0.0).any()):
        raise ValueError("case origins must be nonnegative")
    if not bool(np.equal(case_origins, np.floor(case_origins)).all()):
        raise ValueError("case origins must be integer-valued")

    # This subtraction form makes the exact-end boundary explicit and avoids
    # overflow in case_origin + horizon.
    resolved = case_origins <= origin - forecast_horizon
    return cases.loc[resolved].copy()


def nearest_neighbor_indices(
    query_key: np.ndarray,
    case_keys: np.ndarray,
    case_series_ids: np.ndarray,
    query_series_id: object,
    k: int,
) -> np.ndarray:
    """Return nearest eligible rows, breaking distance ties by source row."""

    query = np.asarray(query_key, dtype=np.float64)
    keys = np.asarray(case_keys, dtype=np.float64)
    series_ids = np.asarray(case_series_ids, dtype=object)
    neighbors = _integer("k", k, minimum=1)
    if query.ndim != 1:
        raise ValueError("query_key must be one-dimensional")
    if keys.ndim != 2:
        raise ValueError("case_keys must be two-dimensional")
    if keys.shape[1] != query.shape[0]:
        raise ValueError("query_key and case_keys have different widths")
    if series_ids.ndim != 1 or series_ids.shape[0] != keys.shape[0]:
        raise ValueError("case_series_ids must align one-to-one with case_keys")
    if bool(pd.isna(query_series_id)):
        raise ValueError("query_series_id must not be missing")
    if bool(pd.isna(series_ids).any()):
        raise ValueError("case_series_ids must not contain missing values")
    if not bool(np.isfinite(query).all() and np.isfinite(keys).all()):
        raise ValueError("retrieval keys must all be finite")

    eligible_indices = np.flatnonzero(series_ids != query_series_id)
    if eligible_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    differences = keys[eligible_indices] - query
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.einsum("ij,ij->i", differences, differences)
    if not bool(np.isfinite(distances).all()):
        raise ValueError("nearest-neighbor distances must all be finite")

    order = np.lexsort((eligible_indices, distances))
    return eligible_indices[order[:neighbors]].astype(np.int64, copy=False)


def select_m1_hyperparameters(
    source_frame: pd.DataFrame,
    *,
    k_grid: object,
    lambda_max_grid: object,
) -> tuple[int, float]:
    """Select M1 by mean, worst origin, smaller lambda, then larger k."""

    try:
        raw_k = tuple(k_grid)
        raw_lambda = tuple(lambda_max_grid)
    except TypeError as exc:
        raise TypeError("hyperparameter grids must be iterable") from exc
    if not raw_k or not raw_lambda:
        raise ValueError("hyperparameter grids must not be empty")

    ks = tuple(_integer("k_grid", value, minimum=1) for value in raw_k)
    lambdas = tuple(
        _finite_float("lambda_max_grid", value) for value in raw_lambda
    )
    if len(set(ks)) != len(ks) or len(set(lambdas)) != len(lambdas):
        raise ValueError("hyperparameter grids must not contain duplicates")
    if any(value < 0.0 or value > 1.0 for value in lambdas):
        raise ValueError("lambda_max_grid values must lie in [0, 1]")

    required = ("k", "lambda_max", "mean_loss", "worst_origin_loss")
    if not isinstance(source_frame, pd.DataFrame):
        raise TypeError("source_frame must be a pandas DataFrame")
    if not source_frame.columns.is_unique:
        raise ValueError("source_frame has duplicate column names")
    missing = [column for column in required if column not in source_frame.columns]
    if missing:
        raise ValueError(f"source_frame is missing columns: {missing}")
    if source_frame.empty:
        raise ValueError("source_frame contains no tuning candidates")

    candidates: dict[tuple[int, float], tuple[float, float]] = {}
    for candidate_k, candidate_lambda, mean_loss, worst_loss in source_frame.loc[
        :, required
    ].itertuples(index=False, name=None):
        key = (
            _integer("k", candidate_k, minimum=1),
            _finite_float("lambda_max", candidate_lambda),
        )
        if key[1] < 0.0 or key[1] > 1.0:
            raise ValueError("candidate lambda_max values must lie in [0, 1]")
        if key in candidates:
            raise ValueError(
                f"duplicate M1 candidate for k={key[0]}, lambda_max={key[1]}"
            )
        candidate_mean = _finite_float("mean_loss", mean_loss)
        candidate_worst = _finite_float("worst_origin_loss", worst_loss)
        if candidate_mean < 0.0 or candidate_worst < 0.0:
            raise ValueError("candidate losses must be nonnegative")
        candidates[key] = (candidate_mean, candidate_worst)

    expected = {(k_value, lam) for k_value in ks for lam in lambdas}
    if set(candidates) != expected:
        raise ValueError(
            "M1 candidate rows must exactly match k_grid x lambda_max_grid"
        )
    return min(
        expected,
        key=lambda key: (
            candidates[key][0],
            candidates[key][1],
            key[1],
            -key[0],
        ),
    )
