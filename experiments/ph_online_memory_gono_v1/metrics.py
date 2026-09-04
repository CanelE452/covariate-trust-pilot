"""Pure metric and prediction-pairing helpers."""

from numbers import Integral

import numpy as np
import pandas as pd


PREDICTION_KEYS = ("dataset_id", "series_id", "origin", "step")
POLICY_SCALE_EPSILON = 1e-8


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _prediction_arm(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{arm} predictions must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{arm} predictions have duplicate column names")

    required = (*PREDICTION_KEYS, "prediction")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{arm} predictions are missing columns: {missing}")
    if bool(frame.loc[:, PREDICTION_KEYS].isna().any(axis=None)):
        raise ValueError(f"{arm} prediction keys must not be missing")
    if bool(frame.duplicated(list(PREDICTION_KEYS), keep=False).any()):
        raise ValueError(f"{arm} predictions are not unique on the pairing keys")

    try:
        numeric_predictions = pd.to_numeric(
            frame["prediction"], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arm} predictions must be numeric") from exc
    if not bool(np.isfinite(numeric_predictions).all()):
        raise ValueError(f"{arm} predictions must all be finite")

    return frame.loc[:, required].rename(
        columns={"prediction": f"{arm}_prediction"}
    )


def pair_predictions(
    point: pd.DataFrame, hurdle: pd.DataFrame
) -> pd.DataFrame:
    """Pair Point and Hurdle predictions with a complete one-to-one key join."""

    point_arm = _prediction_arm(point, "point")
    hurdle_arm = _prediction_arm(hurdle, "hurdle")
    try:
        paired = point_arm.merge(
            hurdle_arm,
            on=list(PREDICTION_KEYS),
            how="outer",
            validate="one_to_one",
            indicator=True,
            sort=False,
        )
    except (pd.errors.MergeError, TypeError, ValueError) as exc:
        raise ValueError("Point/Hurdle prediction keys cannot be paired") from exc

    if not bool((paired["_merge"] == "both").all()):
        missing_point = int((paired["_merge"] == "right_only").sum())
        missing_hurdle = int((paired["_merge"] == "left_only").sum())
        raise ValueError(
            "Point/Hurdle prediction keys are incomplete "
            f"(missing Point={missing_point}, missing Hurdle={missing_hurdle})"
        )

    columns = [
        *PREDICTION_KEYS,
        "point_prediction",
        "hurdle_prediction",
    ]
    return (
        paired.loc[:, columns]
        .sort_values(list(PREDICTION_KEYS), kind="mergesort")
        .reset_index(drop=True)
    )


def policy_scale_squared(
    values: np.ndarray, model_train_end: int
) -> float:
    """Return mean squared model-training value plus the frozen epsilon."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional series")
    train_end = _integer("model_train_end", model_train_end, minimum=1)
    if train_end > array.shape[0]:
        raise ValueError("model_train_end exceeds the available series length")

    # Convert only the prefix so malformed or changed future values cannot affect
    # this causal policy scale.
    train_values = np.asarray(array[:train_end], dtype=np.float64)
    if not bool(np.isfinite(train_values).all()):
        raise ValueError("model-training values must all be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        scale = float(np.mean(np.square(train_values), dtype=np.float64))
    if not np.isfinite(scale):
        raise ValueError("model-training squared scale is not finite")
    return scale + POLICY_SCALE_EPSILON
