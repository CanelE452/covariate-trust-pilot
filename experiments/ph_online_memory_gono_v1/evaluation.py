"""Causal policy evaluation for ``PH-ONLINE-MEMORY-GONO-v1``.

All tuning entry points accept a source frame only.  Target evaluation is a
separate operation with already selected scalar hyperparameters.  Online
weights are calculated before the current case's target is scored, and every
memory lookup is restricted to cases whose whole horizon has resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .pipeline import (
    apply_robust_scaler,
    extract_retrieval_features,
    fit_robust_scaler,
)
from .policies import (
    exponential_hurdle_weight,
    select_b3_alpha,
    select_b4_hyperparameters,
)
from .retrieval import select_m1_hyperparameters


STEP_KEYS = ("dataset_id", "series_id", "origin", "step")
CASE_KEYS = ("dataset_id", "series_id", "origin")
ALPHA_GRID = tuple(index / 20.0 for index in range(21))
B4_ETA_GRID = (0.5, 2.0, 8.0, 32.0)
B4_HALF_LIVES = (1, 3)
M1_K_GRID = (32, 128)
M1_LAMBDA_MAX_GRID = (0.25, 0.5)
CONTROL_SEED = 20260904
CONFIDENCE_EPSILON = 1e-8

CaseKey = tuple[str, str, int]


@dataclass(frozen=True)
class M1NeighborSelection:
    """One immutable query view materialized from an M1 neighbor plan."""

    query_key: CaseKey
    resolved_origins: tuple[int, ...]
    neighbor_case_keys: tuple[CaseKey, ...]
    constant_continuous_features: tuple[bool, ...]


@dataclass(frozen=True)
class _M1OriginBlock:
    origin: int
    resolved_origins: tuple[int, ...]
    query_row_indices_bytes: bytes
    neighbor_row_indices_bytes: bytes
    shape: tuple[int, int]
    neighbor_counts_bytes: bytes
    constant_continuous_features: tuple[bool, ...]

    def query_row_indices(self) -> np.ndarray:
        return np.frombuffer(self.query_row_indices_bytes, dtype="<i8")

    def neighbor_row_indices(self) -> np.ndarray:
        return np.frombuffer(self.neighbor_row_indices_bytes, dtype="<i8").reshape(
            self.shape
        )

    def neighbor_counts(self) -> np.ndarray:
        return np.frombuffer(self.neighbor_counts_bytes, dtype="<i8")


@dataclass(frozen=True)
class M1NeighborPlan:
    """Reusable exact-neighbor geometry encoded in immutable byte buffers."""

    dataset_id: str
    case_fingerprint: str
    warmup_origin: int
    evaluation_origins: tuple[int, ...]
    horizon: int
    lookback: int
    max_k: int
    case_keys: tuple[CaseKey, ...]
    blocks: tuple[_M1OriginBlock, ...]

    @property
    def entries(self) -> tuple[M1NeighborSelection, ...]:
        """Materialize human-readable selections (intended for audit/tests)."""

        entries: list[M1NeighborSelection] = []
        for block in self.blocks:
            queries = block.query_row_indices()
            neighbors = block.neighbor_row_indices()
            counts = block.neighbor_counts()
            for local_index, query_row_index in enumerate(queries):
                count = int(counts[local_index])
                entries.append(
                    M1NeighborSelection(
                        query_key=self.case_keys[int(query_row_index)],
                        resolved_origins=block.resolved_origins,
                        neighbor_case_keys=tuple(
                            self.case_keys[int(row_index)]
                            for row_index in neighbors[local_index, :count]
                        ),
                        constant_continuous_features=(
                            block.constant_continuous_features
                        ),
                    )
                )
        return tuple(entries)

_STEP_COLUMNS = (
    *STEP_KEYS,
    "y_observed",
    "point_mean_prediction",
    "hurdle_mean_prediction",
    "target_mask",
    "policy_scale_squared",
)
_CASE_COLUMNS = (
    *CASE_KEYS,
    "history",
    "canonical_train_scale",
    "point_forecast",
    "hurdle_forecast",
    "target",
    "target_mask",
    "policy_scale_squared",
    "point_normalized_loss",
    "hurdle_normalized_loss",
)


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
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


def _unit_interval(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _required_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: Sequence[str],
    keys: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate column names")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    if bool(frame.loc[:, keys].isna().any(axis=None)):
        raise ValueError(f"{name} keys must not be missing")
    if bool(frame.duplicated(list(keys), keep=False).any()):
        raise ValueError(f"{name} keys must be unique")
    return frame.loc[:, columns].copy()


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
    integer: bool = False,
) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{column} must be finite")
    if nonnegative and bool((values < 0.0).any()):
        raise ValueError(f"{column} must be nonnegative")
    if positive and bool((values <= 0.0).any()):
        raise ValueError(f"{column} must be positive")
    if integer and not bool(np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{column} must be integer-valued")
    return values


def _validated_steps(frame: pd.DataFrame) -> pd.DataFrame:
    result = _required_frame(
        frame,
        name="step_predictions",
        columns=_STEP_COLUMNS,
        keys=STEP_KEYS,
    )
    _numeric_column(result, "origin", nonnegative=True, integer=True)
    _numeric_column(result, "step", nonnegative=True, integer=True)
    _numeric_column(result, "y_observed")
    _numeric_column(result, "point_mean_prediction")
    _numeric_column(result, "hurdle_mean_prediction")
    _numeric_column(result, "policy_scale_squared", positive=True)
    if not pd.api.types.is_bool_dtype(result["target_mask"].dtype):
        raise ValueError("target_mask must be Boolean")
    if not bool(result["target_mask"].any()):
        raise ValueError("step_predictions contain no observed targets")

    result["dataset_id"] = result["dataset_id"].astype(str)
    result["series_id"] = result["series_id"].astype(str)
    result["origin"] = result["origin"].astype(np.int64)
    result["step"] = result["step"].astype(np.int64)
    result = result.sort_values(list(STEP_KEYS), kind="mergesort").reset_index(
        drop=True
    )
    expected_width: int | None = None
    for _, group in result.groupby(list(CASE_KEYS), sort=False, observed=True):
        steps = group["step"].to_numpy(dtype=np.int64)
        if expected_width is None:
            expected_width = len(steps)
        if len(steps) != expected_width or not np.array_equal(
            steps, np.arange(len(steps), dtype=np.int64)
        ):
            raise ValueError(
                "each series-origin must contain the same complete 0-based step grid"
            )
        if group["policy_scale_squared"].nunique(dropna=False) != 1:
            raise ValueError("policy_scale_squared changed within a series-origin")
        if not bool(group["target_mask"].any()):
            raise ValueError("each series-origin must contain an observed target")
    return result


def convex_forecast(
    point_forecast: np.ndarray,
    hurdle_forecast: np.ndarray,
    hurdle_weight: object,
) -> np.ndarray:
    """Return the preregistered convex blend of the two paired experts."""

    point = np.asarray(point_forecast, dtype=np.float64)
    hurdle = np.asarray(hurdle_forecast, dtype=np.float64)
    weight = np.asarray(hurdle_weight, dtype=np.float64)
    if point.shape != hurdle.shape or point.size == 0:
        raise ValueError("expert forecasts must be equal nonempty arrays")
    if not bool(
        np.isfinite(point).all()
        and np.isfinite(hurdle).all()
        and np.isfinite(weight).all()
    ):
        raise ValueError("forecasts and weights must be finite")
    if bool((weight < 0.0).any() or (weight > 1.0).any()):
        raise ValueError("hurdle_weight must lie in [0, 1]")
    try:
        result = point + weight * (hurdle - point)
    except ValueError as exc:
        raise ValueError("hurdle_weight cannot be broadcast to the forecasts") from exc
    if result.shape != point.shape or not bool(np.isfinite(result).all()):
        raise ValueError("convex blending produced an invalid forecast")
    return result


def baseline_policy_steps(
    step_predictions: pd.DataFrame, *, b3_alpha: float | None = None
) -> pd.DataFrame:
    """Attach B0, B1, B2 and, when fixed, B3 step-level forecasts."""

    result = _validated_steps(step_predictions)
    point = result["point_mean_prediction"].to_numpy(dtype=np.float64)
    hurdle = result["hurdle_mean_prediction"].to_numpy(dtype=np.float64)
    result["b0_prediction"] = point
    result["b1_prediction"] = hurdle
    result["b2_prediction"] = convex_forecast(point, hurdle, 0.5)
    if b3_alpha is not None:
        alpha = _unit_interval("b3_alpha", b3_alpha)
        result["b3_prediction"] = convex_forecast(point, hurdle, alpha)
    return result


def _masked_normalized_loss(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    scale_squared: float,
) -> float:
    observed = np.asarray(mask, dtype=bool)
    actual = np.asarray(target, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    scale = _finite_float("policy_scale_squared", scale_squared)
    if actual.shape != forecast.shape or actual.shape != observed.shape:
        raise ValueError("target, prediction, and target_mask shapes must match")
    if actual.size == 0 or not bool(observed.any()) or scale <= 0.0:
        raise ValueError("loss requires observed targets and a positive scale")
    if not bool(np.isfinite(actual).all() and np.isfinite(forecast).all()):
        raise ValueError("target and prediction must be finite")
    error = actual[observed] - forecast[observed]
    return float(np.mean(np.square(error), dtype=np.float64) / scale)


def _series_origin_prediction_losses(
    step_predictions: pd.DataFrame,
    prediction_columns: Mapping[str, str],
) -> pd.DataFrame:
    missing = [
        column for column in prediction_columns.values() if column not in step_predictions
    ]
    if missing:
        raise ValueError(f"step_predictions are missing policy columns: {missing}")
    extra = step_predictions.loc[
        :, [*STEP_KEYS, *dict.fromkeys(prediction_columns.values())]
    ].copy()
    extra["dataset_id"] = extra["dataset_id"].astype(str)
    extra["series_id"] = extra["series_id"].astype(str)
    extra["origin"] = pd.to_numeric(extra["origin"], errors="raise").astype(np.int64)
    extra["step"] = pd.to_numeric(extra["step"], errors="raise").astype(np.int64)
    extra = extra.sort_values(list(STEP_KEYS), kind="mergesort").reset_index(drop=True)
    for column in prediction_columns.values():
        try:
            values = pd.to_numeric(extra[column], errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"{column} must be finite")
        extra[column] = values

    frame = _validated_steps(step_predictions)
    if not frame.loc[:, STEP_KEYS].equals(extra.loc[:, STEP_KEYS]):
        raise AssertionError("policy predictions lost alignment with pairing keys")
    for column in prediction_columns.values():
        frame[column] = extra[column].to_numpy(dtype=np.float64)

    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(list(CASE_KEYS), sort=True, observed=True):
        mask = group["target_mask"].to_numpy(dtype=bool)
        target = group["y_observed"].to_numpy(dtype=np.float64)
        scale = float(group["policy_scale_squared"].iloc[0])
        row: dict[str, object] = dict(zip(CASE_KEYS, keys, strict=True))
        row["n_steps"] = int(mask.sum())
        row["policy_scale_squared"] = scale
        for policy_name, column in prediction_columns.items():
            row[f"{policy_name}_normalized_loss"] = _masked_normalized_loss(
                target,
                group[column].to_numpy(dtype=np.float64),
                mask,
                scale,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def series_origin_expert_losses(step_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the paired Point/Hurdle losses at series-origin level."""

    return _series_origin_prediction_losses(
        step_predictions,
        {
            "point": "point_mean_prediction",
            "hurdle": "hurdle_mean_prediction",
        },
    )


def _aggregate_candidate(losses: pd.DataFrame, loss_column: str) -> tuple[float, float]:
    mean_loss = float(losses[loss_column].mean())
    worst_origin = float(losses.groupby("origin", sort=True)[loss_column].mean().max())
    return mean_loss, worst_origin


def tune_b3_source(
    source_steps: pd.DataFrame,
    *,
    evaluation_origins: Iterable[int],
    alpha_grid: Iterable[float] = ALPHA_GRID,
) -> dict[str, object]:
    """Select one static alpha using source data only."""

    frame = _validated_steps(source_steps)
    if frame["dataset_id"].nunique() != 1:
        raise ValueError("source_steps must contain exactly one source dataset")
    origins = tuple(
        _integer("evaluation_origin", origin, minimum=0)
        for origin in evaluation_origins
    )
    if (
        not origins
        or len(set(origins)) != len(origins)
        or origins != tuple(sorted(origins))
    ):
        raise ValueError("evaluation_origins must be nonempty, unique, and ordered")
    for keys, group in frame.groupby(
        ["dataset_id", "series_id"], sort=True, observed=True
    ):
        available = set(group["origin"].astype(int))
        if not set(origins).issubset(available):
            raise ValueError(
                f"source series {keys!r} is missing an evaluation origin"
            )
    frame = frame.loc[frame["origin"].isin(origins)].reset_index(drop=True)
    grid = tuple(_unit_interval("alpha_grid", value) for value in alpha_grid)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("alpha_grid must be nonempty and unique")
    rows: list[dict[str, float]] = []
    for alpha in grid:
        candidate = frame.copy()
        candidate["candidate_prediction"] = convex_forecast(
            candidate["point_mean_prediction"].to_numpy(dtype=np.float64),
            candidate["hurdle_mean_prediction"].to_numpy(dtype=np.float64),
            alpha,
        )
        losses = _series_origin_prediction_losses(
            candidate, {"candidate": "candidate_prediction"}
        )
        mean_loss, worst_loss = _aggregate_candidate(
            losses, "candidate_normalized_loss"
        )
        rows.append(
            {
                "alpha": alpha,
                "mean_loss": mean_loss,
                "worst_origin_loss": worst_loss,
            }
        )
    candidates = pd.DataFrame(rows)
    selected = select_b3_alpha(candidates, alpha_grid=grid)
    return {
        "alpha": selected,
        "evaluation_origins": origins,
        "candidates": candidates,
    }


def build_series_origin_cases(
    step_predictions: pd.DataFrame,
    histories: pd.DataFrame,
    *,
    horizon: int,
    lookback: int,
) -> pd.DataFrame:
    """Join pre-origin histories to paired forecasts and resolved expert losses."""

    forecast_horizon = _integer("horizon", horizon, minimum=1)
    history_width = _integer("lookback", lookback, minimum=1)
    steps = _validated_steps(step_predictions)
    histories_frame = _required_frame(
        histories,
        name="histories",
        columns=(*CASE_KEYS, "history", "canonical_train_scale"),
        keys=CASE_KEYS,
    )
    histories_frame["dataset_id"] = histories_frame["dataset_id"].astype(str)
    histories_frame["series_id"] = histories_frame["series_id"].astype(str)
    _numeric_column(histories_frame, "origin", nonnegative=True, integer=True)
    _numeric_column(histories_frame, "canonical_train_scale", positive=True)
    histories_frame["origin"] = histories_frame["origin"].astype(np.int64)

    history_arrays: list[np.ndarray] = []
    for history in histories_frame["history"]:
        values = np.asarray(history, dtype=np.float64)
        if values.shape != (history_width,) or not bool(np.isfinite(values).all()):
            raise ValueError("each history must be a finite lookback-length vector")
        if bool((values < 0.0).any()):
            raise ValueError("history values must be nonnegative")
        history_arrays.append(values.copy())
    histories_frame["history"] = history_arrays

    case_rows: list[dict[str, object]] = []
    for keys, group in steps.groupby(list(CASE_KEYS), sort=True, observed=True):
        if len(group) != forecast_horizon:
            raise ValueError("every series-origin must contain exactly horizon steps")
        target = group["y_observed"].to_numpy(dtype=np.float64)
        point = group["point_mean_prediction"].to_numpy(dtype=np.float64)
        hurdle = group["hurdle_mean_prediction"].to_numpy(dtype=np.float64)
        mask = group["target_mask"].to_numpy(dtype=bool)
        scale = float(group["policy_scale_squared"].iloc[0])
        case_rows.append(
            {
                **dict(zip(CASE_KEYS, keys, strict=True)),
                "point_forecast": point,
                "hurdle_forecast": hurdle,
                "target": target,
                "target_mask": mask,
                "policy_scale_squared": scale,
                "point_normalized_loss": _masked_normalized_loss(
                    target, point, mask, scale
                ),
                "hurdle_normalized_loss": _masked_normalized_loss(
                    target, hurdle, mask, scale
                ),
            }
        )
    case_frame = pd.DataFrame(case_rows)
    try:
        joined = case_frame.merge(
            histories_frame,
            on=list(CASE_KEYS),
            how="outer",
            validate="one_to_one",
            indicator=True,
            sort=False,
        )
    except pd.errors.MergeError as exc:
        raise ValueError("histories cannot be joined one-to-one to forecast cases") from exc
    if not bool((joined["_merge"] == "both").all()):
        raise ValueError("histories must exactly cover every forecast case")
    joined = joined.drop(columns="_merge")
    return joined.loc[:, _CASE_COLUMNS].sort_values(
        list(CASE_KEYS), kind="mergesort"
    ).reset_index(drop=True)


def _array(name: str, value: object, width: int, *, boolean: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if boolean and not np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must remain Boolean")
    dtype = bool if boolean else np.float64
    result = np.asarray(raw, dtype=dtype)
    if result.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},)")
    if not boolean and not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _validated_cases(cases: pd.DataFrame, *, horizon: int, lookback: int | None = None) -> pd.DataFrame:
    width = _integer("horizon", horizon, minimum=1)
    result = _required_frame(
        cases,
        name="cases",
        columns=_CASE_COLUMNS,
        keys=CASE_KEYS,
    )
    if result["dataset_id"].nunique() != 1:
        raise ValueError("cases must contain exactly one dataset")
    result["dataset_id"] = result["dataset_id"].astype(str)
    result["series_id"] = result["series_id"].astype(str)
    _numeric_column(result, "origin", nonnegative=True, integer=True)
    _numeric_column(result, "canonical_train_scale", positive=True)
    _numeric_column(result, "policy_scale_squared", positive=True)
    _numeric_column(result, "point_normalized_loss", nonnegative=True)
    _numeric_column(result, "hurdle_normalized_loss", nonnegative=True)
    result["origin"] = result["origin"].astype(np.int64)

    histories: list[np.ndarray] = []
    points: list[np.ndarray] = []
    hurdles: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for row in result.itertuples(index=False):
        history = np.asarray(row.history, dtype=np.float64)
        if history.ndim != 1 or history.size == 0 or not bool(np.isfinite(history).all()):
            raise ValueError("history must be a finite nonempty vector")
        if lookback is not None and history.shape != (_integer("lookback", lookback, minimum=1),):
            raise ValueError("history does not match the configured lookback")
        if bool((history < 0.0).any()):
            raise ValueError("history must be nonnegative")
        point = _array("point_forecast", row.point_forecast, width)
        hurdle = _array("hurdle_forecast", row.hurdle_forecast, width)
        target = _array("target", row.target, width)
        mask = _array("target_mask", row.target_mask, width, boolean=True)
        if not bool(mask.any()):
            raise ValueError("each case must contain an observed target")
        point_loss = _masked_normalized_loss(
            target, point, mask, row.policy_scale_squared
        )
        hurdle_loss = _masked_normalized_loss(
            target, hurdle, mask, row.policy_scale_squared
        )
        if not np.isclose(point_loss, row.point_normalized_loss, rtol=1e-10, atol=1e-12):
            raise ValueError("point_normalized_loss does not match paired forecasts")
        if not np.isclose(hurdle_loss, row.hurdle_normalized_loss, rtol=1e-10, atol=1e-12):
            raise ValueError("hurdle_normalized_loss does not match paired forecasts")
        histories.append(history.copy())
        points.append(point)
        hurdles.append(hurdle)
        targets.append(target)
        masks.append(mask)
    result["history"] = histories
    result["point_forecast"] = points
    result["hurdle_forecast"] = hurdles
    result["target"] = targets
    result["target_mask"] = masks
    return result.sort_values(list(CASE_KEYS), kind="mergesort").reset_index(drop=True)


def _schedule(
    warmup_origin: int,
    evaluation_origins: Iterable[int],
    horizon: int,
) -> tuple[int, tuple[int, ...], int]:
    warmup = _integer("warmup_origin", warmup_origin, minimum=0)
    width = _integer("horizon", horizon, minimum=1)
    origins = tuple(
        _integer("evaluation_origin", origin, minimum=0)
        for origin in evaluation_origins
    )
    if not origins or len(set(origins)) != len(origins):
        raise ValueError("evaluation_origins must be nonempty and unique")
    expected = tuple(warmup + width * index for index in range(1, len(origins) + 1))
    if origins != expected:
        raise ValueError(
            "evaluation_origins must be ordered non-overlapping origins after warmup"
        )
    return warmup, origins, width


def _validate_schedule_coverage(
    cases: pd.DataFrame,
    warmup_origin: int,
    evaluation_origins: tuple[int, ...],
) -> None:
    expected = {warmup_origin, *evaluation_origins}
    for keys, group in cases.groupby(
        ["dataset_id", "series_id"], sort=True, observed=True
    ):
        observed = set(group["origin"].astype(int))
        if observed != expected:
            raise ValueError(
                f"series {keys!r} must have exactly warmup and evaluation cases"
            )


def _b4_weight_table(
    cases: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: tuple[int, ...],
    horizon: int,
    eta: float,
    half_life: int,
) -> pd.DataFrame:
    learning_rate = _finite_float("eta", eta)
    if learning_rate < 0.0:
        raise ValueError("eta must be nonnegative")
    memory_half_life = _integer("half_life", half_life, minimum=1)
    gamma = 0.5 ** (1.0 / memory_half_life)
    rows: list[dict[str, object]] = []
    for (dataset_id, series_id), group in cases.groupby(
        ["dataset_id", "series_id"], sort=True, observed=True
    ):
        by_origin = group.set_index("origin", verify_integrity=True)
        for query_origin in evaluation_origins:
            resolved = (warmup_origin,) + tuple(
                origin for origin in evaluation_origins if origin < query_origin
            )
            ages = np.array(
                [(query_origin - origin) / horizon for origin in resolved],
                dtype=np.float64,
            )
            if bool((ages < 1.0).any()):
                raise AssertionError("unresolved loss reached the B4 state")
            discounts = np.power(gamma, ages)
            point_losses = by_origin.loc[
                list(resolved), "point_normalized_loss"
            ].to_numpy(dtype=np.float64)
            hurdle_losses = by_origin.loc[
                list(resolved), "hurdle_normalized_loss"
            ].to_numpy(dtype=np.float64)
            point_sum = float(np.dot(discounts, point_losses))
            hurdle_sum = float(np.dot(discounts, hurdle_losses))
            weight = float(
                exponential_hurdle_weight(
                    np.array([point_sum]), np.array([hurdle_sum]), learning_rate
                )[0]
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "series_id": series_id,
                    "origin": query_origin,
                    "resolved_origins": resolved,
                    "gamma": gamma,
                    "discounted_point_loss": point_sum,
                    "discounted_hurdle_loss": hurdle_sum,
                    "b4_hurdle_weight": weight,
                }
            )
    return pd.DataFrame(rows).sort_values(list(CASE_KEYS), kind="mergesort").reset_index(drop=True)


def evaluate_b4_cases(
    cases: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: Iterable[int],
    horizon: int,
    eta: float,
    half_life: int,
) -> pd.DataFrame:
    """Evaluate B4 prequentially; current and future losses never enter its state."""

    warmup, origins, width = _schedule(
        warmup_origin, evaluation_origins, horizon
    )
    frame = _validated_cases(cases, horizon=width)
    _validate_schedule_coverage(frame, warmup, origins)
    weights = _b4_weight_table(
        frame,
        warmup_origin=warmup,
        evaluation_origins=origins,
        horizon=width,
        eta=eta,
        half_life=half_life,
    )
    queries = frame.loc[frame["origin"].isin(origins)]
    joined = weights.merge(
        queries,
        on=list(CASE_KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    forecasts: list[np.ndarray] = []
    losses: list[float] = []
    for row in joined.itertuples(index=False):
        forecast = convex_forecast(
            row.point_forecast, row.hurdle_forecast, row.b4_hurdle_weight
        )
        forecasts.append(forecast)
        losses.append(
            _masked_normalized_loss(
                row.target, forecast, row.target_mask, row.policy_scale_squared
            )
        )
    joined["b4_forecast"] = forecasts
    joined["b4_normalized_loss"] = losses
    return joined.sort_values(list(CASE_KEYS), kind="mergesort").reset_index(drop=True)


def tune_b4_source(
    source_cases: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: Iterable[int],
    horizon: int,
    eta_grid: Iterable[float] = B4_ETA_GRID,
    half_lives: Iterable[int] = B4_HALF_LIVES,
) -> dict[str, object]:
    """Select B4 eta and half-life using source cases only."""

    warmup, origins, width = _schedule(
        warmup_origin, evaluation_origins, horizon
    )
    etas = tuple(_finite_float("eta_grid", value) for value in eta_grid)
    lives = tuple(_integer("half_lives", value, minimum=1) for value in half_lives)
    if (
        not etas
        or not lives
        or len(set(etas)) != len(etas)
        or len(set(lives)) != len(lives)
        or any(value < 0.0 for value in etas)
    ):
        raise ValueError("B4 grids must be nonempty, unique, and valid")
    rows: list[dict[str, object]] = []
    for eta in etas:
        for half_life in lives:
            evaluated = evaluate_b4_cases(
                source_cases,
                warmup_origin=warmup,
                evaluation_origins=origins,
                horizon=width,
                eta=eta,
                half_life=half_life,
            )
            mean_loss, worst_loss = _aggregate_candidate(
                evaluated, "b4_normalized_loss"
            )
            rows.append(
                {
                    "eta": eta,
                    "half_life": half_life,
                    "mean_loss": mean_loss,
                    "worst_origin_loss": worst_loss,
                }
            )
    candidates = pd.DataFrame(rows)
    selected_eta, selected_half_life = select_b4_hyperparameters(
        candidates, eta_grid=etas, half_lives=lives
    )
    return {
        "eta": selected_eta,
        "half_life": selected_half_life,
        "candidates": candidates,
    }


def _memory_value_frame(memory_values: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    columns = (
        *CASE_KEYS,
        "point_normalized_loss",
        "hurdle_normalized_loss",
    )
    values = _required_frame(
        memory_values,
        name="memory_values",
        columns=columns,
        keys=CASE_KEYS,
    )
    values["dataset_id"] = values["dataset_id"].astype(str)
    values["series_id"] = values["series_id"].astype(str)
    _numeric_column(values, "origin", nonnegative=True, integer=True)
    _numeric_column(values, "point_normalized_loss", nonnegative=True)
    _numeric_column(values, "hurdle_normalized_loss", nonnegative=True)
    values["origin"] = values["origin"].astype(np.int64)
    expected_keys = pd.MultiIndex.from_frame(cases.loc[:, CASE_KEYS])
    observed_keys = pd.MultiIndex.from_frame(values.loc[:, CASE_KEYS])
    if not expected_keys.equals(observed_keys):
        expected_set = set(expected_keys.tolist())
        observed_set = set(observed_keys.tolist())
        if expected_set != observed_set:
            raise ValueError("memory_values must exactly cover the case keys")
    return values.sort_values(list(CASE_KEYS), kind="mergesort").reset_index(drop=True)


def _query_rng(seed: int, *parts: object) -> np.random.Generator:
    fixed_seed = _integer("random_seed", seed, minimum=0)
    payload = "\x00".join([str(fixed_seed), *(str(part) for part in parts)]).encode(
        "utf-8"
    )
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)


def shuffle_paired_memory_values(
    memory_values: pd.DataFrame, *, seed: int = CONTROL_SEED
) -> pd.DataFrame:
    """C0: permute intact Point/Hurdle loss pairs within each resolved origin."""

    columns = (
        *CASE_KEYS,
        "point_normalized_loss",
        "hurdle_normalized_loss",
    )
    frame = _required_frame(
        memory_values,
        name="memory_values",
        columns=columns,
        keys=CASE_KEYS,
    )
    frame["dataset_id"] = frame["dataset_id"].astype(str)
    frame["series_id"] = frame["series_id"].astype(str)
    _numeric_column(frame, "origin", nonnegative=True, integer=True)
    _numeric_column(frame, "point_normalized_loss", nonnegative=True)
    _numeric_column(frame, "hurdle_normalized_loss", nonnegative=True)
    frame["origin"] = frame["origin"].astype(np.int64)
    frame = frame.sort_values(list(CASE_KEYS), kind="mergesort").reset_index(drop=True)
    loss_columns = ["point_normalized_loss", "hurdle_normalized_loss"]
    for (dataset_id, origin), indices in frame.groupby(
        ["dataset_id", "origin"], sort=True, observed=True
    ).groups.items():
        locations = np.asarray(list(indices), dtype=np.int64)
        pairs = frame.loc[locations, loss_columns].to_numpy(copy=True)
        rng = _query_rng(seed, "C0", dataset_id, int(origin))
        permutation = rng.permutation(len(locations))
        frame.loc[locations, loss_columns] = pairs[permutation]
    return frame


def _case_feature(row: object) -> np.ndarray:
    return extract_retrieval_features(
        row.history,
        row.point_forecast,
        row.hurdle_forecast,
        canonical_train_scale=row.canonical_train_scale,
    )


def _case_keys(frame: pd.DataFrame) -> tuple[CaseKey, ...]:
    return tuple(
        (str(row.dataset_id), str(row.series_id), int(row.origin))
        for row in frame.itertuples(index=False)
    )


def _neighbor_case_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        for value in (str(row.dataset_id), str(row.series_id), str(int(row.origin))):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        for value in (
            row.history,
            np.array([row.canonical_train_scale], dtype=np.float64),
            row.point_forecast,
            row.hurdle_forecast,
        ):
            array = np.asarray(value, dtype="<f8")
            digest.update(array.ndim.to_bytes(1, "little"))
            for dimension in array.shape:
                digest.update(int(dimension).to_bytes(8, "little"))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fit_neighbor_tree(memory_features: np.ndarray) -> cKDTree:
    return cKDTree(memory_features, compact_nodes=True, balanced_tree=True)


def _exact_neighbor_rows(
    tree: cKDTree,
    *,
    scaled_memory: np.ndarray,
    scaled_queries: np.ndarray,
    memory_row_indices: np.ndarray,
    memory_series_ids: np.ndarray,
    query_series_ids: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact neighbors with canonical-row tie breaking.

    The tree obtains a tight radius in one batched query.  A radius query then
    includes every boundary tie, after which exact squared distances and the
    canonical source-row index determine the final order.
    """

    query_count = int(scaled_queries.shape[0])
    memory_count = int(scaled_memory.shape[0])
    eligible_counts = np.array(
        [np.count_nonzero(memory_series_ids != series_id) for series_id in query_series_ids],
        dtype=np.int64,
    )
    if bool((eligible_counts == 0).any()):
        raise ValueError("same-series exclusion leaves no retrieval candidates")
    take_counts = np.minimum(eligible_counts, max_k).astype(np.int64, copy=False)
    same_series_counts = memory_count - eligible_counts
    probe = min(memory_count, max_k + int(same_series_counts.max()))
    distances, indices = tree.query(
        scaled_queries,
        k=probe,
        eps=0.0,
        p=2.0,
        workers=-1,
    )
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if probe == 1:
        distances = distances.reshape(query_count, 1)
        indices = indices.reshape(query_count, 1)
    if not bool(np.isfinite(distances).all()):
        raise ValueError("nearest-neighbor distances must all be finite")

    boundary_distances = np.empty(query_count, dtype=np.float64)
    for query_index in range(query_count):
        probe_eligible = (
            memory_series_ids[indices[query_index]]
            != query_series_ids[query_index]
        )
        eligible_distances = distances[query_index, probe_eligible]
        take = int(take_counts[query_index])
        if eligible_distances.size < take:
            raise AssertionError("neighbor probe did not contain enough eligible rows")
        boundary_distances[query_index] = float(
            np.partition(eligible_distances, take - 1)[take - 1]
        )

    epsilon = np.finfo(np.float64).eps
    radii = boundary_distances + 16.0 * epsilon * np.maximum(
        1.0, np.abs(boundary_distances)
    )
    radius_candidates = tree.query_ball_point(
        scaled_queries,
        radii,
        p=2.0,
        eps=0.0,
        workers=-1,
        return_sorted=False,
    )

    selected = np.full((query_count, max_k), -1, dtype="<i8")
    for query_index, raw_candidates in enumerate(radius_candidates):
        candidates = np.asarray(raw_candidates, dtype=np.int64)
        candidates = candidates[
            memory_series_ids[candidates] != query_series_ids[query_index]
        ]
        take = int(take_counts[query_index])
        if candidates.size < take:
            candidates = np.flatnonzero(
                memory_series_ids != query_series_ids[query_index]
            )
        differences = scaled_memory[candidates] - scaled_queries[query_index]
        with np.errstate(over="ignore", invalid="ignore"):
            squared_distances = np.einsum("ij,ij->i", differences, differences)
        if not bool(np.isfinite(squared_distances).all()):
            raise ValueError("nearest-neighbor distances must all be finite")
        canonical_rows = memory_row_indices[candidates]
        order = np.lexsort((canonical_rows, squared_distances))[:take]
        selected[query_index, :take] = canonical_rows[order]
    return selected, take_counts


def build_m1_neighbor_plan(
    cases: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: Iterable[int],
    horizon: int,
    lookback: int,
    max_k: int,
) -> M1NeighborPlan:
    """Build one reusable exact-neighbor plan for every evaluation origin."""

    warmup, origins, width = _schedule(
        warmup_origin, evaluation_origins, horizon
    )
    history_width = _integer("lookback", lookback, minimum=1)
    maximum_neighbors = _integer("max_k", max_k, minimum=1)
    frame = _validated_cases(cases, horizon=width, lookback=history_width)
    _validate_schedule_coverage(frame, warmup, origins)
    features = np.vstack(
        [_case_feature(row) for row in frame.itertuples(index=False)]
    )
    frame_origins = frame["origin"].to_numpy(dtype=np.int64)
    frame_series_ids = frame["series_id"].to_numpy(dtype=object)
    blocks: list[_M1OriginBlock] = []
    for query_origin in origins:
        memory_rows = np.flatnonzero(frame_origins + width <= query_origin).astype(
            "<i8", copy=False
        )
        query_rows = np.flatnonzero(frame_origins == query_origin).astype(
            "<i8", copy=False
        )
        if memory_rows.size == 0:
            raise ValueError("retrieval memory contains no resolved cases")
        scaler = fit_robust_scaler(features[memory_rows])
        scaled_memory = apply_robust_scaler(features[memory_rows], scaler)
        scaled_queries = apply_robust_scaler(features[query_rows], scaler)
        tree = _fit_neighbor_tree(scaled_memory)
        neighbors, counts = _exact_neighbor_rows(
            tree,
            scaled_memory=scaled_memory,
            scaled_queries=scaled_queries,
            memory_row_indices=memory_rows,
            memory_series_ids=frame_series_ids[memory_rows],
            query_series_ids=frame_series_ids[query_rows],
            max_k=maximum_neighbors,
        )
        resolved_origins = tuple(
            sorted(np.unique(frame_origins[memory_rows]).astype(int).tolist())
        )
        blocks.append(
            _M1OriginBlock(
                origin=query_origin,
                resolved_origins=resolved_origins,
                query_row_indices_bytes=query_rows.tobytes(order="C"),
                neighbor_row_indices_bytes=neighbors.tobytes(order="C"),
                shape=neighbors.shape,
                neighbor_counts_bytes=counts.astype("<i8", copy=False).tobytes(
                    order="C"
                ),
                constant_continuous_features=tuple(
                    bool(value) for value in scaler["constant_continuous"]
                ),
            )
        )
    return M1NeighborPlan(
        dataset_id=str(frame["dataset_id"].iloc[0]),
        case_fingerprint=_neighbor_case_fingerprint(frame),
        warmup_origin=warmup,
        evaluation_origins=origins,
        horizon=width,
        lookback=history_width,
        max_k=maximum_neighbors,
        case_keys=_case_keys(frame),
        blocks=tuple(blocks),
    )


def _validate_neighbor_plan(
    plan: M1NeighborPlan,
    frame: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: tuple[int, ...],
    horizon: int,
    lookback: int,
    k: int,
) -> None:
    if not isinstance(plan, M1NeighborPlan):
        raise TypeError("neighbor_plan must be an M1NeighborPlan")
    expected = (
        str(frame["dataset_id"].iloc[0]),
        warmup_origin,
        evaluation_origins,
        horizon,
        lookback,
        _case_keys(frame),
        _neighbor_case_fingerprint(frame),
    )
    observed = (
        plan.dataset_id,
        plan.warmup_origin,
        plan.evaluation_origins,
        plan.horizon,
        plan.lookback,
        plan.case_keys,
        plan.case_fingerprint,
    )
    if observed != expected:
        raise ValueError("neighbor_plan does not match the cases or schedule")
    if plan.max_k < k:
        raise ValueError("neighbor_plan max_k is smaller than the requested k")


def evaluate_m1_cases(
    cases: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: Iterable[int],
    horizon: int,
    lookback: int,
    eta: float,
    half_life: int,
    k: int,
    lambda_max: float,
    memory_values: pd.DataFrame | None = None,
    neighbor_mode: str = "nearest",
    random_seed: int = CONTROL_SEED,
    neighbor_plan: M1NeighborPlan | None = None,
) -> pd.DataFrame:
    """Evaluate M1 or C1 against a target-dataset-only resolved memory pool."""

    warmup, origins, width = _schedule(
        warmup_origin, evaluation_origins, horizon
    )
    history_width = _integer("lookback", lookback, minimum=1)
    neighbors_requested = _integer("k", k, minimum=1)
    maximum_intervention = _unit_interval("lambda_max", lambda_max)
    learning_rate = _finite_float("eta", eta)
    if learning_rate < 0.0:
        raise ValueError("eta must be nonnegative")
    if neighbor_mode not in {"nearest", "random"}:
        raise ValueError("neighbor_mode must be 'nearest' or 'random'")

    frame = _validated_cases(cases, horizon=width, lookback=history_width)
    _validate_schedule_coverage(frame, warmup, origins)
    if memory_values is None:
        values = frame.loc[
            :,
            [
                *CASE_KEYS,
                "point_normalized_loss",
                "hurdle_normalized_loss",
            ],
        ].copy()
    else:
        values = _memory_value_frame(memory_values, frame)
    value_lookup = values.set_index(list(CASE_KEYS), verify_integrity=True)

    b4 = _b4_weight_table(
        frame,
        warmup_origin=warmup,
        evaluation_origins=origins,
        horizon=width,
        eta=learning_rate,
        half_life=half_life,
    ).set_index(list(CASE_KEYS), verify_integrity=True)

    if neighbor_plan is None:
        plan = build_m1_neighbor_plan(
            frame,
            warmup_origin=warmup,
            evaluation_origins=origins,
            horizon=width,
            lookback=history_width,
            max_k=neighbors_requested,
        )
    else:
        plan = neighbor_plan
    _validate_neighbor_plan(
        plan,
        frame,
        warmup_origin=warmup,
        evaluation_origins=origins,
        horizon=width,
        lookback=history_width,
        k=neighbors_requested,
    )

    result_rows: list[dict[str, object]] = []
    frame_origins = frame["origin"].to_numpy(dtype=np.int64)
    frame_series_ids = frame["series_id"].to_numpy(dtype=object)
    for block in plan.blocks:
        query_rows = block.query_row_indices()
        planned_neighbors = block.neighbor_row_indices()
        planned_counts = block.neighbor_counts()
        memory_rows = np.flatnonzero(frame_origins + width <= block.origin)
        memory_series_ids = frame_series_ids[memory_rows]
        for local_index, query_row_index in enumerate(query_rows):
            query = frame.iloc[int(query_row_index)]
            if neighbor_mode == "nearest":
                take = min(
                    neighbors_requested,
                    int(planned_counts[local_index]),
                )
                selected_rows = planned_neighbors[local_index, :take]
            else:
                eligible = np.flatnonzero(memory_series_ids != query.series_id)
                if eligible.size == 0:
                    raise ValueError(
                        "same-series exclusion leaves no retrieval candidates"
                    )
                take = min(neighbors_requested, int(eligible.size))
                rng = _query_rng(
                    random_seed,
                    "C1",
                    query.dataset_id,
                    query.series_id,
                    int(query.origin),
                )
                selected = rng.choice(
                    eligible, size=take, replace=False
                ).astype(np.int64)
                selected_rows = memory_rows[selected]

            selected_keys = [plan.case_keys[int(index)] for index in selected_rows]
            selected_values = value_lookup.loc[selected_keys]
            point_losses = selected_values[
                "point_normalized_loss"
            ].to_numpy(dtype=np.float64)
            hurdle_losses = selected_values[
                "hurdle_normalized_loss"
            ].to_numpy(dtype=np.float64)
            local_point = float(point_losses.mean())
            local_hurdle = float(hurdle_losses.mean())
            local_weight = float(
                exponential_hurdle_weight(
                    np.array([local_point]),
                    np.array([local_hurdle]),
                    learning_rate,
                )[0]
            )
            differences = hurdle_losses - point_losses
            mean_difference = float(differences.mean())
            sd_difference = float(differences.std(ddof=0))
            confidence = abs(mean_difference) / (
                abs(mean_difference) + sd_difference + CONFIDENCE_EPSILON
            )
            intervention = maximum_intervention * confidence
            b4_row = b4.loc[
                (query.dataset_id, query.series_id, query.origin)
            ]
            b4_weight = float(b4_row["b4_hurdle_weight"])
            m1_weight = (
                (1.0 - intervention) * b4_weight
                + intervention * local_weight
            )
            if not 0.0 <= m1_weight <= 1.0:
                raise AssertionError("bounded shrinkage left the convex hull")
            forecast = convex_forecast(
                query.point_forecast, query.hurdle_forecast, m1_weight
            )
            normalized_loss = _masked_normalized_loss(
                query.target,
                forecast,
                query.target_mask,
                query.policy_scale_squared,
            )
            result_rows.append(
                {
                    "dataset_id": query.dataset_id,
                    "series_id": query.series_id,
                    "origin": int(query.origin),
                    "resolved_origins": block.resolved_origins,
                    "neighbor_case_keys": tuple(selected_keys),
                    "neighbor_series_ids": tuple(
                        key[1] for key in selected_keys
                    ),
                    "neighbor_origins": tuple(key[2] for key in selected_keys),
                    "neighbor_count": int(take),
                    "constant_continuous_features": (
                        block.constant_continuous_features
                    ),
                    "local_point_loss": local_point,
                    "local_hurdle_loss": local_hurdle,
                    "local_hurdle_weight": local_weight,
                    "mean_neighbor_loss_difference": mean_difference,
                    "sd_neighbor_loss_difference": sd_difference,
                    "retrieval_confidence": confidence,
                    "retrieval_lambda": intervention,
                    "b4_hurdle_weight": b4_weight,
                    "m1_hurdle_weight": m1_weight,
                    "point_forecast": query.point_forecast.copy(),
                    "hurdle_forecast": query.hurdle_forecast.copy(),
                    "m1_forecast": forecast,
                    "m1_normalized_loss": normalized_loss,
                }
            )
    return pd.DataFrame(result_rows).sort_values(
        list(CASE_KEYS), kind="mergesort"
    ).reset_index(drop=True)


def evaluate_c0_cases(
    cases: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: Iterable[int],
    horizon: int,
    lookback: int,
    eta: float,
    half_life: int,
    k: int,
    lambda_max: float,
    random_seed: int = CONTROL_SEED,
    neighbor_plan: M1NeighborPlan | None = None,
) -> pd.DataFrame:
    """Evaluate C0 with intact loss pairs shuffled within each origin."""

    frame = _validated_cases(cases, horizon=horizon, lookback=lookback)
    values = shuffle_paired_memory_values(
        frame.loc[
            :,
            [
                *CASE_KEYS,
                "point_normalized_loss",
                "hurdle_normalized_loss",
            ],
        ],
        seed=random_seed,
    )
    return evaluate_m1_cases(
        frame,
        warmup_origin=warmup_origin,
        evaluation_origins=evaluation_origins,
        horizon=horizon,
        lookback=lookback,
        eta=eta,
        half_life=half_life,
        k=k,
        lambda_max=lambda_max,
        memory_values=values,
        neighbor_mode="nearest",
        random_seed=random_seed,
        neighbor_plan=neighbor_plan,
    )


def evaluate_c1_cases(
    cases: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: Iterable[int],
    horizon: int,
    lookback: int,
    eta: float,
    half_life: int,
    k: int,
    lambda_max: float,
    random_seed: int = CONTROL_SEED,
    neighbor_plan: M1NeighborPlan | None = None,
) -> pd.DataFrame:
    """Evaluate C1 with uniform random eligible neighbors."""

    return evaluate_m1_cases(
        cases,
        warmup_origin=warmup_origin,
        evaluation_origins=evaluation_origins,
        horizon=horizon,
        lookback=lookback,
        eta=eta,
        half_life=half_life,
        k=k,
        lambda_max=lambda_max,
        neighbor_mode="random",
        random_seed=random_seed,
        neighbor_plan=neighbor_plan,
    )


def tune_m1_source(
    source_cases: pd.DataFrame,
    *,
    warmup_origin: int,
    evaluation_origins: Iterable[int],
    horizon: int,
    lookback: int,
    eta: float,
    half_life: int,
    k_grid: Iterable[int] = M1_K_GRID,
    lambda_max_grid: Iterable[float] = M1_LAMBDA_MAX_GRID,
) -> dict[str, object]:
    """Select M1 k/lambda on source cases with B4 parameters held fixed."""

    warmup, origins, width = _schedule(
        warmup_origin, evaluation_origins, horizon
    )
    ks = tuple(_integer("k_grid", value, minimum=1) for value in k_grid)
    lambdas = tuple(
        _unit_interval("lambda_max_grid", value) for value in lambda_max_grid
    )
    if (
        not ks
        or not lambdas
        or len(set(ks)) != len(ks)
        or len(set(lambdas)) != len(lambdas)
    ):
        raise ValueError("M1 grids must be nonempty and unique")
    neighbor_plan = build_m1_neighbor_plan(
        source_cases,
        warmup_origin=warmup,
        evaluation_origins=origins,
        horizon=width,
        lookback=lookback,
        max_k=max(ks),
    )
    rows: list[dict[str, object]] = []
    for candidate_k in ks:
        for candidate_lambda in lambdas:
            evaluated = evaluate_m1_cases(
                source_cases,
                warmup_origin=warmup,
                evaluation_origins=origins,
                horizon=width,
                lookback=lookback,
                eta=eta,
                half_life=half_life,
                k=candidate_k,
                lambda_max=candidate_lambda,
                neighbor_plan=neighbor_plan,
            )
            mean_loss, worst_loss = _aggregate_candidate(
                evaluated, "m1_normalized_loss"
            )
            rows.append(
                {
                    "k": candidate_k,
                    "lambda_max": candidate_lambda,
                    "mean_loss": mean_loss,
                    "worst_origin_loss": worst_loss,
                }
            )
    candidates = pd.DataFrame(rows)
    selected_k, selected_lambda = select_m1_hyperparameters(
        candidates, k_grid=ks, lambda_max_grid=lambdas
    )
    return {
        "k": selected_k,
        "lambda_max": selected_lambda,
        "eta": _finite_float("eta", eta),
        "half_life": _integer("half_life", half_life, minimum=1),
        "candidates": candidates,
    }
