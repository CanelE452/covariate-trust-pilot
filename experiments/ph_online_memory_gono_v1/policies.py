"""Pure baseline-policy weighting and source-only tuning helpers."""

from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.special import expit


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_grid(
    name: str,
    values: object,
    *,
    minimum: float,
    maximum: float | None = None,
) -> tuple[float, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real numbers") from exc
    if not raw_values:
        raise ValueError(f"{name} must not be empty")

    result = tuple(_finite_float(name, value) for value in raw_values)
    if any(value < minimum for value in result):
        raise ValueError(f"{name} values must be at least {minimum}")
    if maximum is not None and any(value > maximum for value in result):
        raise ValueError(f"{name} values must be at most {maximum}")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must contain integers")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} values must be positive")
    return result


def _integer_grid(name: str, values: object) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc
    if not raw_values:
        raise ValueError(f"{name} must not be empty")
    result = tuple(_positive_integer(name, value) for value in raw_values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def _candidate_table(
    source_frame: pd.DataFrame, required_columns: tuple[str, ...]
) -> pd.DataFrame:
    if not isinstance(source_frame, pd.DataFrame):
        raise TypeError("source_frame must be a pandas DataFrame")
    if not source_frame.columns.is_unique:
        raise ValueError("source_frame has duplicate column names")
    missing = [
        column for column in required_columns if column not in source_frame.columns
    ]
    if missing:
        raise ValueError(f"source_frame is missing columns: {missing}")
    if source_frame.empty:
        raise ValueError("source_frame contains no tuning candidates")
    return source_frame.loc[:, required_columns]


def _loss(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def select_b3_alpha(
    source_frame: pd.DataFrame, *, alpha_grid: object
) -> float:
    """Select B3 by source mean loss, retaining grid order on exact ties."""

    alphas = _float_grid("alpha_grid", alpha_grid, minimum=0.0, maximum=1.0)
    table = _candidate_table(
        source_frame, ("alpha", "mean_loss", "worst_origin_loss")
    )
    candidates: dict[float, float] = {}
    for alpha, mean_loss, worst_loss in table.itertuples(index=False, name=None):
        key = _finite_float("alpha", alpha)
        if key in candidates:
            raise ValueError(f"duplicate B3 candidate for alpha={key}")
        candidates[key] = _loss("mean_loss", mean_loss)
        _loss("worst_origin_loss", worst_loss)

    if set(candidates) != set(alphas):
        raise ValueError("B3 candidate rows must exactly match alpha_grid")
    best_index, best_alpha = min(
        enumerate(alphas),
        key=lambda item: (
            candidates[item[1]],
            item[0],
        ),
    )
    del best_index
    return best_alpha


def select_b4_hyperparameters(
    source_frame: pd.DataFrame,
    *,
    eta_grid: object,
    half_lives: object,
) -> tuple[float, int]:
    """Select B4 by mean, worst origin, smaller eta, then longer memory."""

    etas = _float_grid("eta_grid", eta_grid, minimum=0.0)
    lives = _integer_grid("half_lives", half_lives)
    table = _candidate_table(
        source_frame,
        ("eta", "half_life", "mean_loss", "worst_origin_loss"),
    )
    candidates: dict[tuple[float, int], tuple[float, float]] = {}
    for eta, half_life, mean_loss, worst_loss in table.itertuples(
        index=False, name=None
    ):
        key = (
            _finite_float("eta", eta),
            _positive_integer("half_life", half_life),
        )
        if key in candidates:
            raise ValueError(
                "duplicate B4 candidate for "
                f"eta={key[0]}, half_life={key[1]}"
            )
        candidates[key] = (
            _loss("mean_loss", mean_loss),
            _loss("worst_origin_loss", worst_loss),
        )

    expected = {(eta, half_life) for eta in etas for half_life in lives}
    if set(candidates) != expected:
        raise ValueError(
            "B4 candidate rows must exactly match eta_grid x half_lives"
        )
    return min(
        expected,
        key=lambda key: (
            candidates[key][0],
            candidates[key][1],
            key[0],
            -key[1],
        ),
    )


def exponential_hurdle_weight(
    point_loss: np.ndarray,
    hurdle_loss: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Return the Hurdle softmax weight from a stable log-weight difference."""

    point = np.asarray(point_loss, dtype=np.float64)
    hurdle = np.asarray(hurdle_loss, dtype=np.float64)
    if point.shape != hurdle.shape:
        raise ValueError("point_loss and hurdle_loss must have identical shapes")
    if point.size == 0:
        raise ValueError("loss arrays must not be empty")
    if not bool(np.isfinite(point).all() and np.isfinite(hurdle).all()):
        raise ValueError("losses must all be finite")
    if bool((point < 0.0).any() or (hurdle < 0.0).any()):
        raise ValueError("losses must be nonnegative")
    rate = _finite_float("eta", eta)
    if rate < 0.0:
        raise ValueError("eta must be nonnegative")

    # log(w_h / w_p) = -eta*hurdle - (-eta*point).  Subtract
    # losses before multiplying so large negative log-weights are never
    # exponentiated separately and cannot produce a 0/0 ratio.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_weight_difference = rate * np.subtract(point, hurdle)
    weights = expit(log_weight_difference)
    if not bool(np.isfinite(weights).all()):
        raise ValueError("exponential weighting produced a non-finite value")
    return weights
