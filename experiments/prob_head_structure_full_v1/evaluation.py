"""Common-space probabilistic evaluation for the preregistered experiment.

This module intentionally contains no likelihood or model-specific code.  Every
family is compared only through forecasts on a shared quantile/CDF space.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    WindowBatch,
    WindowRequest,
    make_history_windows,
    seal_count_primary_dataset_audit,
    verify_train_only_sample_manifest,
)
from .integrity import BranchEligibility, GateStatus


KEY_COLUMNS = ("dataset_id", "series_id", "origin", "step")
CRPS_QUANTILE_GRID = (
    0.01,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
)
EVALUATION_QUANTILE_GRID = tuple(sorted({*CRPS_QUANTILE_GRID, 0.025, 0.975}))
REPORTED_QUANTILES = (0.50, 0.80, 0.90, 0.95, 0.99)
CENTRAL_INTERVALS = (0.50, 0.80, 0.90, 0.95)


class PredictionIntegrityError(RuntimeError):
    """A forecast artifact violates an evaluation integrity contract."""


def quantile_column(probability: float) -> str:
    """Return the stable frame column name for a probability in [0, 1]."""

    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    return f"q_{int(round(probability * 1000)):03d}"


def midpoint_cell_widths(quantile_grid: Sequence[float]) -> np.ndarray:
    """Widths of midpoint-bounded probability cells, including 0/1 edges."""

    q = np.asarray(quantile_grid, dtype=np.float64)
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("quantile grid must be a finite non-empty vector")
    if q[0] <= 0.0 or q[-1] >= 1.0 or np.any(np.diff(q) <= 0.0):
        raise ValueError("quantile grid must be strictly increasing inside (0, 1)")
    boundaries = np.concatenate(([0.0], (q[:-1] + q[1:]) / 2.0, [1.0]))
    return np.diff(boundaries)


def pinball_loss(target: np.ndarray, prediction: np.ndarray, probability: float) -> np.ndarray:
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    error = target_array - prediction_array
    return np.maximum(float(probability) * error, (float(probability) - 1.0) * error)


def relative_loss_improvement(baseline_loss: float, candidate_loss: float) -> float:
    """Return positive improvement for a lower candidate loss."""

    baseline = float(baseline_loss)
    candidate = float(candidate_loss)
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        raise PredictionIntegrityError("relative loss comparison contains NaN/Inf")
    if baseline <= 0.0:
        raise PredictionIntegrityError("relative loss improvement requires a positive baseline")
    return (baseline - candidate) / baseline


def approximate_crps(
    target: np.ndarray,
    quantiles: np.ndarray,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> np.ndarray:
    """Approximate CRPS using the frozen midpoint-cell weighted pinball rule."""

    y = np.asarray(target, dtype=np.float64)
    predictions = np.asarray(quantiles, dtype=np.float64)
    q = np.asarray(quantile_grid, dtype=np.float64)
    if predictions.shape != y.shape + (q.size,):
        raise ValueError("quantiles must have target.shape + (n_quantiles,)")
    losses = np.stack(
        [pinball_loss(y, predictions[..., index], probability) for index, probability in enumerate(q)],
        axis=-1,
    )
    return 2.0 * np.sum(losses * midpoint_cell_widths(q), axis=-1)


def quantile_implied_mean(
    quantiles: np.ndarray,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> np.ndarray:
    """Integrate monotone quantiles with endpoint hold at q=.01 and q=.99.

    This is the frozen predictive-mean source for the quantile-only student and
    P3 pool.  Native teachers and linear CDF pools instead supply their analytical
    (weighted, for a pool) predictive mean to the evaluator.
    """

    values = np.asarray(quantiles, dtype=np.float64)
    q = np.asarray(quantile_grid, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != q.size:
        raise ValueError("quantiles must end with the frozen quantile dimension")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise PredictionIntegrityError("mean-source quantiles are invalid")
    if np.any(np.diff(values, axis=-1) < 0.0):
        raise PredictionIntegrityError("mean-source quantiles cross")
    interior = np.sum(
        0.5 * (values[..., :-1] + values[..., 1:]) * np.diff(q), axis=-1
    )
    return q[0] * values[..., 0] + interior + (1.0 - q[-1]) * values[..., -1]


def coverage_quantiles_from_common_grid(
    quantiles: np.ndarray,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
    *,
    p_zero: np.ndarray,
) -> dict[str, Any]:
    """Interpolate q=.025/.975 while preserving the zero-mass plateau."""

    values = np.asarray(quantiles, dtype=np.float64)
    q = np.asarray(quantile_grid, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != q.size:
        raise ValueError("quantiles must end with the common quantile dimension")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise PredictionIntegrityError("coverage-source quantiles are invalid")
    if np.any(np.diff(values, axis=-1) < 0.0):
        raise PredictionIntegrityError("coverage-source quantiles cross")
    zero_mass = np.asarray(p_zero, dtype=np.float64)
    if zero_mass.shape != values.shape[:-1] or not np.all(np.isfinite(zero_mass)):
        raise PredictionIntegrityError("coverage zero mass shape is invalid")
    if np.any((zero_mass < 0.0) | (zero_mass > 1.0)):
        raise PredictionIntegrityError("coverage zero mass is outside [0, 1]")

    def interpolate(probability: float, lower: float, upper: float) -> np.ndarray:
        lower_positions = np.flatnonzero(np.isclose(q, lower, rtol=0.0, atol=1e-12))
        upper_positions = np.flatnonzero(np.isclose(q, upper, rtol=0.0, atol=1e-12))
        if lower_positions.size != 1 or upper_positions.size != 1:
            raise ValueError(f"common grid lacks interpolation anchors {lower}/{upper}")
        left = values[..., int(lower_positions[0])]
        right = values[..., int(upper_positions[0])]
        plateau_ends_inside = (zero_mass > lower) & (zero_mass < probability)
        effective_lower = np.where(plateau_ends_inside, zero_mass, lower)
        effective_left = np.where(plateau_ends_inside, 0.0, left)
        fraction = (probability - effective_lower) / (upper - effective_lower)
        interpolated = effective_left + fraction * (right - effective_left)
        return np.where(probability <= zero_mass, 0.0, interpolated)

    return {
        "q025": interpolate(0.025, 0.01, 0.05),
        "q975": interpolate(0.975, 0.95, 0.99),
        "source": "monotone_piecewise_common_grid",
        "anchors": {"q025": [0.01, 0.05], "q975": [0.95, 0.99]},
    }


def validate_exact_prediction_keys(
    frames: Mapping[str, pd.DataFrame],
    key_columns: Sequence[str] = KEY_COLUMNS,
    *,
    target_artifact: "SealedEvaluationTarget",
) -> None:
    """Require identical, unique, same-order keys for every compared method."""

    if tuple(key_columns) != KEY_COLUMNS:
        raise PredictionIntegrityError("prediction key columns are frozen")
    if not isinstance(target_artifact, SealedEvaluationTarget):
        raise PredictionIntegrityError("a sealed common evaluation target artifact is required")
    if not frames:
        raise PredictionIntegrityError("at least one prediction frame is required")
    reference_name: str | None = None
    reference_keys: pd.DataFrame | None = None
    for name, frame in frames.items():
        missing = [column for column in key_columns if column not in frame]
        if missing:
            raise PredictionIntegrityError(f"{name} is missing prediction keys: {missing}")
        if frame.duplicated(list(key_columns)).any():
            raise PredictionIntegrityError(f"{name} contains duplicate prediction keys")
        keys = frame.loc[:, list(key_columns)].reset_index(drop=True)
        if reference_keys is None:
            reference_name = name
            reference_keys = keys
            continue
        if len(keys) != len(reference_keys):
            raise PredictionIntegrityError(
                f"prediction key row count differs: {reference_name}={len(reference_keys)}, {name}={len(keys)}"
            )
        if not keys.equals(reference_keys):
            reference_set = set(map(tuple, reference_keys.itertuples(index=False, name=None)))
            current_set = set(map(tuple, keys.itertuples(index=False, name=None)))
            if reference_set == current_set:
                raise PredictionIntegrityError(f"prediction key order differs for {name}")
            missing_keys = len(reference_set - current_set)
            extra_keys = len(current_set - reference_set)
            raise PredictionIntegrityError(
                f"prediction key membership differs for {name}: missing={missing_keys}, extra={extra_keys}"
            )
    for frame in frames.values():
        target_artifact.verify_frame(frame)


def exact_prediction_join(
    frames: Mapping[str, pd.DataFrame],
    key_columns: Sequence[str] = KEY_COLUMNS,
    *,
    target_artifact: "SealedEvaluationTarget",
) -> pd.DataFrame:
    """Join already-aligned predictions without any implicit key reordering."""

    validate_exact_prediction_keys(
        frames, key_columns, target_artifact=target_artifact
    )
    items = list(frames.items())
    joined = items[0][1].loc[:, list(key_columns)].reset_index(drop=True).copy()
    for method, frame in items:
        if not isinstance(method, str) or not method:
            raise PredictionIntegrityError("prediction method names must be non-empty strings")
        for column in frame.columns:
            if column in key_columns:
                continue
            output_column = f"{column}__{method}"
            if output_column in joined:
                raise PredictionIntegrityError(f"prediction join column collision: {output_column}")
            joined[output_column] = frame[column].reset_index(drop=True)
    return joined


def practical_winner_labels(losses: np.ndarray, *, loser_gap: float = 0.01) -> np.ndarray:
    """Return a boolean [case, head] practical-winner mask.

    A head is a loser only when it is at least one percent worse than the case
    best.  Exact ties and sub-threshold differences therefore remain winners.
    """

    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("losses must be a finite [case, head] matrix")
    best = np.min(values, axis=1, keepdims=True)
    denominator = np.maximum(np.abs(best), np.finfo(np.float64).tiny)
    relative_gap = (values - best) / denominator
    return relative_gap < float(loser_gap)


def _validate_diagnostic_role(scientific_role: str) -> str:
    role = str(scientific_role)
    prefix = "DIAGNOSTIC_CONTINUATION_AFTER_"
    if not role.startswith(prefix) or len(role) == len(prefix):
        raise ValueError("diagnostic scientific_role must name its upstream reason")
    return role


def diagnostic_empirical_cdf(
    samples: np.ndarray,
    support: np.ndarray,
    *,
    scientific_role: str,
) -> dict[str, Any]:
    """Evaluate a sample CDF while retaining mandatory diagnostic lineage."""

    draws = np.asarray(samples, dtype=np.float64)
    values = np.asarray(support, dtype=np.float64)
    if draws.ndim != 2 or values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be [case, draw] and support a non-empty vector")
    if not np.all(np.isfinite(draws)) or not np.all(np.isfinite(values)):
        raise PredictionIntegrityError("empirical CDF input contains NaN/Inf")
    if np.any(draws < 0.0) or np.any(np.diff(values) < 0.0):
        raise PredictionIntegrityError("empirical CDF has negative draws or unordered support")
    return {
        "values": np.mean(draws[:, :, None] <= values[None, None, :], axis=1),
        "source": "empirical_cdf_from_samples",
        "confirmatory_eligible": False,
        "scientific_role": _validate_diagnostic_role(scientific_role),
    }


def diagnostic_empirical_quantiles(
    samples: np.ndarray,
    probabilities: Sequence[float],
    *,
    scientific_role: str,
) -> dict[str, Any]:
    """Return a left empirical inverse with mandatory diagnostic lineage."""

    draws = np.asarray(samples, dtype=np.float64)
    q = np.asarray(probabilities, dtype=np.float64)
    if draws.ndim != 2 or q.ndim != 1 or np.any((q <= 0.0) | (q >= 1.0)):
        raise ValueError("samples must be [case, draw] and probabilities inside (0, 1)")
    if not np.all(np.isfinite(draws)) or np.any(draws < 0.0):
        raise PredictionIntegrityError("empirical quantile samples are invalid")
    ordered = np.sort(draws, axis=1)
    indices = np.ceil(q * draws.shape[1]).astype(int) - 1
    return {
        "values": ordered[:, indices],
        "source": "empirical_inverse_from_samples",
        "confirmatory_eligible": False,
        "scientific_role": _validate_diagnostic_role(scientific_role),
    }


def _validated_long_losses(
    frame: pd.DataFrame,
    *,
    head_column: str,
    loss_column: str,
    unit_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    required = {*unit_columns, head_column, loss_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"loss frame is missing columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated([*unit_columns, head_column]).any():
        raise PredictionIntegrityError("duplicate head loss at an evaluation unit")
    wide = frame.pivot(index=list(unit_columns), columns=head_column, values=loss_column)
    if wide.isna().any().any():
        raise PredictionIntegrityError("heads do not cover identical evaluation units")
    if not np.all(np.isfinite(wide.to_numpy(dtype=np.float64))):
        raise PredictionIntegrityError("head loss contains NaN/Inf")
    return wide, [str(column) for column in wide.columns]


def _average_registered_model_seed_losses(
    frame: pd.DataFrame,
    *,
    head_column: str,
    loss_column: str,
    expected_model_seeds: Sequence[int] | None,
    expected_data_seeds: Sequence[int] | None,
) -> pd.DataFrame:
    """Average forecast replicates only after proving exact registered coverage."""

    has_model_seed = "model_seed" in frame
    if expected_model_seeds is None:
        if has_model_seed:
            raise PredictionIntegrityError(
                "model_seed rows require an exact expected model-seed manifest"
            )
        result = frame.copy()
    else:
        seeds = tuple(int(seed) for seed in expected_model_seeds)
        if (
            not has_model_seed
            or not seeds
            or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) for seed in expected_model_seeds)
        ):
            raise PredictionIntegrityError("model-seed manifest is invalid")
        identity = [
            column
            for column in frame.columns
            if column not in {"model_seed", loss_column}
        ]
        if frame.duplicated([*identity, "model_seed"]).any():
            raise PredictionIntegrityError("duplicate model-seed loss replicate")
        for _, block in frame.groupby(identity, sort=False, dropna=False):
            observed = tuple(sorted(int(value) for value in block["model_seed"]))
            if observed != tuple(sorted(seeds)):
                raise PredictionIntegrityError(
                    "loss rows do not have exact model-seed coverage"
                )
        result = (
            frame.groupby(identity, sort=False, dropna=False)[loss_column]
            .mean()
            .reset_index()
        )
    if expected_data_seeds is not None:
        data_seeds = tuple(int(seed) for seed in expected_data_seeds)
        if (
            "data_seed" not in result
            or not data_seeds
            or len(set(data_seeds)) != len(data_seeds)
            or tuple(sorted(pd.unique(result["data_seed"]))) != tuple(sorted(data_seeds))
        ):
            raise PredictionIntegrityError("loss rows do not match the data-seed manifest")
    if not np.all(np.isfinite(result[loss_column].to_numpy(dtype=np.float64))):
        raise PredictionIntegrityError("seed-averaged loss contains NaN/Inf")
    return result


def summarize_practical_winners(
    frame: pd.DataFrame,
    *,
    unit_columns: Sequence[str] = ("dataset_id", "series_id", "origin"),
    head_column: str = "head",
    loss_column: str = "sCRPS",
    expected_model_seeds: Sequence[int] | None = None,
    expected_data_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarize exact and <1%-from-best winner shares without forced ties."""

    reduced = _average_registered_model_seed_losses(
        frame,
        head_column=head_column,
        loss_column=loss_column,
        expected_model_seeds=expected_model_seeds,
        expected_data_seeds=expected_data_seeds,
    )
    wide, heads = _validated_long_losses(
        reduced,
        head_column=head_column,
        loss_column=loss_column,
        unit_columns=unit_columns,
    )
    losses = wide.to_numpy(dtype=np.float64)
    minima = np.min(losses, axis=1, keepdims=True)
    exact = np.isclose(losses, minima, rtol=0.0, atol=1e-12)
    practical = practical_winner_labels(losses)
    if "dataset_id" not in wide.index.names:
        raise PredictionIntegrityError("winner macro requires dataset_id in evaluation units")
    dataset_values = wide.index.get_level_values("dataset_id")
    per_dataset: dict[str, Any] = {}
    for dataset in sorted(pd.unique(dataset_values), key=lambda value: str(value)):
        mask = np.asarray(dataset_values == dataset)
        per_dataset[str(dataset)] = {
            "unit_count": int(mask.sum()),
            "exact_best_shares": {
                head: float(exact[mask, index].mean())
                for index, head in enumerate(heads)
            },
            "practical_winner_shares": {
                head: float(practical[mask, index].mean())
                for index, head in enumerate(heads)
            },
        }
    return {
        "unit_count": len(wide),
        "exact_best_counts": {
            head: int(exact[:, index].sum()) for index, head in enumerate(heads)
        },
        "exact_best_shares": {
            head: float(exact[:, index].mean()) for index, head in enumerate(heads)
        },
        "practical_winner_counts": {
            head: int(practical[:, index].sum()) for index, head in enumerate(heads)
        },
        "practical_winner_shares": {
            head: float(
                np.mean(
                    [
                        row["practical_winner_shares"][head]
                        for row in per_dataset.values()
                    ]
                )
            )
            for head in heads
        },
        "pooled_row_practical_winner_shares": {
            head: float(practical[:, index].mean())
            for index, head in enumerate(heads)
        },
        "per_dataset": per_dataset,
        "macro_aggregation": "equal_dataset_weight",
        "model_seed_aggregation": (
            "mean_before_winner_selection"
            if expected_model_seeds is not None
            else "already_aggregated"
        ),
        "tie_policy": "all exact or sub-1pct heads retain winner status",
    }


def pairwise_relative_scrps_gaps(
    frame: pd.DataFrame,
    *,
    unit_columns: Sequence[str] = ("dataset_id", "series_id", "origin"),
    head_column: str = "head",
    loss_column: str = "sCRPS",
) -> pd.DataFrame:
    """Build the three frozen ordered S3 percentage-point relative gaps."""

    synthetic_columns = {
        "data_seed",
        "model_seed",
        "base_innovation_id",
        "d",
        "rho_I",
        "rho_M",
    }
    if synthetic_columns.intersection(frame.columns) and not synthetic_columns.issubset(
        frame.columns
    ):
        raise PredictionIntegrityError(
            "synthetic S3 rows require canonical data_seed/model_seed/base-innovation/"
            "d/rho/origin identity"
        )
    resolved_units = tuple(
        dict.fromkeys(
            [
                *unit_columns,
                *(
                    column
                    for column in (
                        "data_seed",
                        "model_seed",
                        "base_innovation_id",
                        "d",
                        "rho_I",
                        "rho_M",
                    )
                    if column in frame.columns
                ),
            ]
        )
    )
    wide, _ = _validated_long_losses(
        frame,
        head_column=head_column,
        loss_column=loss_column,
        unit_columns=resolved_units,
    )
    required = {"NB", "HSNB", "TWEEDIE_FULL"}
    if set(map(str, wide.columns)) != required:
        raise PredictionIntegrityError(f"S3 requires exactly the heads {sorted(required)}")
    pairs = (
        ("NB", "HSNB", "NB_vs_HSNB"),
        ("NB", "TWEEDIE_FULL", "NB_vs_TWEEDIE_FULL"),
        ("HSNB", "TWEEDIE_FULL", "HSNB_vs_TWEEDIE_FULL"),
    )
    rows: list[pd.DataFrame] = []
    for left, right, label in pairs:
        denominator = wide[right].to_numpy(dtype=np.float64)
        if np.any(denominator <= 0.0):
            raise PredictionIntegrityError("S3 relative gap denominator must be positive")
        block = wide.index.to_frame(index=False)
        block["pair"] = label
        block["relative_gap"] = (
            100.0
            * (wide[left].to_numpy(dtype=np.float64) - denominator)
            / denominator
        )
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def _summarize_oracle_ladder_core(
    frame: pd.DataFrame,
    *,
    required_heads: Sequence[str],
    head_column: str = "head",
    loss_column: str = "sCRPS",
    unit_columns: Sequence[str] = ("dataset_id", "series_id", "origin"),
    d_column: str = "d",
    cell_columns: Sequence[str] = ("d", "rho_I", "rho_M"),
) -> dict[str, Any]:
    """Compute global, d, cell, and series-origin oracle losses.

    Selection and evaluation use only the supplied scientific loss rows.  A
    validation-selected CDF pool is reported separately by the caller because its
    weights must be frozen on a disjoint validation artifact.
    """

    required_units = tuple(dict.fromkeys([*unit_columns, d_column, *cell_columns]))
    wide, heads = _validated_long_losses(
        frame,
        head_column=head_column,
        loss_column=loss_column,
        unit_columns=required_units,
    )
    long = wide.reset_index().melt(
        id_vars=list(required_units), var_name=head_column, value_name=loss_column
    )
    fixed_head_order = [str(head) for head in required_heads]
    if tuple(dict.fromkeys(fixed_head_order)) != tuple(fixed_head_order) or set(
        map(str, heads)
    ) != set(fixed_head_order):
        raise PredictionIntegrityError(
            f"oracle ladder requires exactly the heads {fixed_head_order}"
        )
    cell_head_means = (
        long.groupby([*cell_columns, head_column], sort=False, dropna=False)[loss_column]
        .mean()
        .reset_index()
    )
    global_means = cell_head_means.groupby(head_column, sort=False)[loss_column].mean()
    best_global_head = min(
        fixed_head_order,
        key=lambda head: (float(global_means[head]), fixed_head_order.index(head)),
    )
    best_global_loss = float(global_means[best_global_head])
    if best_global_loss <= 0.0:
        raise PredictionIntegrityError("oracle gains require a positive best-global loss")

    def conditional_oracle(columns: Sequence[str]) -> tuple[float, dict[str, str]]:
        choices: dict[tuple[Any, ...], str] = {}
        for key, block in cell_head_means.groupby(list(columns), sort=False, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            means = block.groupby(head_column, sort=False)[loss_column].mean()
            choices[key_tuple] = min(
                fixed_head_order,
                key=lambda head: (float(means[head]), fixed_head_order.index(head)),
            )
        chosen_losses: list[float] = []
        for _, row in cell_head_means.iterrows():
            key_tuple = tuple(row[column] for column in columns)
            if row[head_column] == choices[key_tuple]:
                chosen_losses.append(float(row[loss_column]))
        serialized = {"|".join(map(str, key)): head for key, head in choices.items()}
        return float(np.mean(chosen_losses)), serialized

    d_loss, d_choices = conditional_oracle((d_column,))
    cell_loss, cell_choices = conditional_oracle(tuple(cell_columns))
    row_oracle = wide.min(axis=1).rename("oracle_loss").reset_index()
    series_origin_loss = float(
        row_oracle.groupby(list(cell_columns), sort=False, dropna=False)["oracle_loss"]
        .mean()
        .mean()
    )
    return {
        "semantic_label": "test_oracle_characterization_only",
        "aggregation": "equal_cell_macro",
        "best_global_head": best_global_head,
        "best_global_loss": best_global_loss,
        "d_oracle_loss": d_loss,
        "d_oracle_gain": 1.0 - d_loss / best_global_loss,
        "d_best_heads": d_choices,
        "cell_oracle_loss": cell_loss,
        "cell_oracle_gain": 1.0 - cell_loss / best_global_loss,
        "cell_best_heads": cell_choices,
        "series_origin_oracle_loss": series_origin_loss,
        "series_origin_oracle_gain": 1.0 - series_origin_loss / best_global_loss,
        "head_order": fixed_head_order,
    }


def summarize_oracle_ladder(
    frame: pd.DataFrame,
    *,
    head_column: str = "head",
    loss_column: str = "sCRPS",
    unit_columns: Sequence[str] = ("dataset_id", "series_id", "origin"),
    d_column: str = "d",
    cell_columns: Sequence[str] = ("d", "rho_I", "rho_M"),
    expected_model_seeds: Sequence[int] | None = None,
    expected_data_seeds: Sequence[int] | None = None,
    branch_eligibility: BranchEligibility,
    tweedie_valid: GateStatus,
) -> dict[str, Any]:
    """Compute the confirmatory three-family synthetic oracle ladder."""

    if (
        not isinstance(branch_eligibility, BranchEligibility)
        or not branch_eligibility.confirmatory_eligible
        or branch_eligibility.role != "CONFIRMATORY"
        or not isinstance(tweedie_valid, GateStatus)
        or tweedie_valid.gate != "TWEEDIE_VALID"
        or tweedie_valid.status != "PASS"
    ):
        raise PredictionIntegrityError(
            "confirmatory oracle ladder requires eligible lineage and verified Tweedie"
        )
    reduced = _average_registered_model_seed_losses(
        frame,
        head_column=head_column,
        loss_column=loss_column,
        expected_model_seeds=expected_model_seeds,
        expected_data_seeds=expected_data_seeds,
    )
    result = _summarize_oracle_ladder_core(
        reduced,
        required_heads=("NB", "HSNB", "TWEEDIE_FULL"),
        head_column=head_column,
        loss_column=loss_column,
        unit_columns=unit_columns,
        d_column=d_column,
        cell_columns=cell_columns,
    )
    return {
        **result,
        "upstream_required_gates": [
            *branch_eligibility.upstream_required_gates,
            "TWEEDIE_VALID",
        ],
        "upstream_gate_status": {
            **dict(branch_eligibility.upstream_gate_status),
            "TWEEDIE_VALID": "PASS",
        },
        "confirmatory_eligible": True,
        "scientific_role": "CONFIRMATORY",
    }


def summarize_two_head_diagnostic_oracle_ladder(
    frame: pd.DataFrame,
    *,
    head_column: str = "head",
    loss_column: str = "sCRPS",
    unit_columns: Sequence[str] = ("dataset_id", "series_id", "origin"),
    d_column: str = "d",
    cell_columns: Sequence[str] = ("d", "rho_I", "rho_M"),
    expected_model_seeds: Sequence[int] | None = None,
    expected_data_seeds: Sequence[int] | None = None,
    tweedie_valid: GateStatus,
) -> dict[str, Any]:
    """Compute the explicit NB/HSNB continuation when Tweedie is blocked."""

    if (
        not isinstance(tweedie_valid, GateStatus)
        or tweedie_valid.gate != "TWEEDIE_VALID"
        or tweedie_valid.status != "HARD_FAILURE"
    ):
        raise PredictionIntegrityError(
            "two-head oracle continuation requires the verified Tweedie hard block"
        )
    reduced = _average_registered_model_seed_losses(
        frame,
        head_column=head_column,
        loss_column=loss_column,
        expected_model_seeds=expected_model_seeds,
        expected_data_seeds=expected_data_seeds,
    )
    result = _summarize_oracle_ladder_core(
        reduced,
        required_heads=("NB", "HSNB"),
        head_column=head_column,
        loss_column=loss_column,
        unit_columns=unit_columns,
        d_column=d_column,
        cell_columns=cell_columns,
    )
    return {
        **result,
        "semantic_label": "two_head_test_oracle_diagnostic_only",
        "upstream_required_gates": ["TWEEDIE_VALID"],
        "upstream_gate_status": {"TWEEDIE_VALID": "FAIL"},
        "confirmatory_eligible": False,
        "scientific_role": "DIAGNOSTIC_CONTINUATION_AFTER_TWEEDIE_BRANCH_BLOCKED_HARD",
    }


def _aggregate_metrics(step: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metric_means = [
        "sCRPS",
        "zero_brier",
        "normalized_squared_error",
        "normalized_absolute_error",
        "negative_prediction",
        "nan_inf_prediction",
    ]
    metric_means.extend(f"sQL_{int(q * 100):02d}" for q in REPORTED_QUANTILES)
    metric_means.extend(f"covered_{int(level * 100):02d}" for level in CENTRAL_INTERVALS)
    metric_means.extend(f"interval_width_{int(level * 100):02d}" for level in CENTRAL_INTERVALS)
    metric_means.extend(f"raw_interval_width_{int(level * 100):02d}" for level in CENTRAL_INTERVALS)

    def summarize(block: pd.DataFrame) -> pd.Series:
        values: dict[str, float] = {column: float(block[column].mean()) for column in metric_means}
        values["NRMSE"] = float(np.sqrt(values["normalized_squared_error"]))
        values["NMAE"] = values["normalized_absolute_error"]
        for level in CENTRAL_INTERVALS:
            code = int(level * 100)
            values[f"coverage_error_{code:02d}"] = abs(values[f"covered_{code:02d}"] - level)
        return pd.Series(values)

    if groups:
        return step.groupby(groups, sort=False, dropna=False).apply(summarize, include_groups=False).reset_index()
    return summarize(step).to_frame().T


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _target_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = [*KEY_COLUMNS, "y", "scale", "target_mask"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise PredictionIntegrityError(
            f"evaluation target artifact is missing columns: {missing}"
        )
    if frame.empty or frame.duplicated(list(KEY_COLUMNS)).any():
        raise PredictionIntegrityError(
            "evaluation target artifact requires nonempty unique prediction keys"
        )
    keys = frame.loc[:, list(KEY_COLUMNS)]
    if keys[["dataset_id", "series_id"]].isna().any().any():
        raise PredictionIntegrityError("evaluation target artifact has a null identity")
    for column in ("origin", "step"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)) or not np.equal(values, np.floor(values)).all():
            raise PredictionIntegrityError(
                "evaluation target artifact origin/step keys must be integers"
            )
    y = frame["y"].to_numpy(dtype=np.float64)
    scale = frame["scale"].to_numpy(dtype=np.float64)
    raw_mask = frame["target_mask"].to_numpy()
    if raw_mask.dtype != np.bool_ and not all(
        isinstance(value, (bool, np.bool_)) for value in raw_mask.tolist()
    ):
        raise PredictionIntegrityError("evaluation target mask must be boolean")
    mask = raw_mask.astype(bool)
    if not mask.any():
        raise PredictionIntegrityError("evaluation target artifact has no valid target rows")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise PredictionIntegrityError("evaluation target violates nonnegative count support")
    if np.any(y[mask] != np.rint(y[mask])):
        raise PredictionIntegrityError("evaluation target violates exact count support")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise PredictionIntegrityError(
            "evaluation target has an invalid train-only RMS scale"
        )
    scale_frame = pd.DataFrame(
        {
            "dataset_id": frame["dataset_id"].astype(str),
            "series_id": frame["series_id"].astype(str),
            "scale": scale,
        }
    )
    if (scale_frame.groupby(["dataset_id", "series_id"], sort=False)["scale"].nunique() != 1).any():
        raise PredictionIntegrityError(
            "evaluation target scale must be one train-only value per series"
        )
    return [
        {
            "dataset_id": str(row.dataset_id),
            "series_id": str(row.series_id),
            "origin": int(row.origin),
            "step": int(row.step),
            "y": float(row.y),
            "scale": float(row.scale),
            "target_mask": bool(row.target_mask),
        }
        for row in frame.loc[:, required].itertuples(index=False)
    ]


@dataclass(frozen=True, init=False)
class SealedEvaluationTarget:
    """Content-bound common target used by every compared forecast artifact."""

    split_name: str
    split_bounds: tuple[int, int]
    row_count: int
    canonical_payload_json: str
    artifact_sha256: str

    @classmethod
    def seal(
        cls,
        *,
        window_batch: WindowBatch,
        window_request: WindowRequest,
        panel: Mapping[str, Any],
        dataset_audit: Mapping[str, Any],
        source_manifest: Mapping[str, Any],
        sample_manifest: Mapping[str, Any],
        preregistration_sha256: str,
        dataset_manifest_sha256: str,
    ) -> "SealedEvaluationTarget":
        if not isinstance(window_batch, WindowBatch) or not isinstance(
            window_request, WindowRequest
        ):
            raise PredictionIntegrityError(
                "evaluation target requires an actual WindowBatch and WindowRequest"
            )
        if not isinstance(panel, Mapping) or not isinstance(dataset_audit, Mapping):
            raise PredictionIntegrityError(
                "evaluation target requires the actual panel and sealed dataset audit"
            )
        if (
            window_request.confirmatory_eligible is not True
            or window_batch.confirmatory_eligible is not True
        ):
            raise PredictionIntegrityError(
                "confirmatory evaluation target requires a source-attested eligible window"
            )
        raw_y = np.asarray(panel.get("raw_y", panel.get("y")), dtype=np.float64)
        if raw_y.ndim != 2:
            raise PredictionIntegrityError("evaluation target panel must be two-dimensional")
        try:
            recomputed_audit = seal_count_primary_dataset_audit(
                panel, source_manifest=source_manifest
            )
            verified_sample = verify_train_only_sample_manifest(
                panel,
                dataset_audit=recomputed_audit,
                sample_manifest=sample_manifest,
            )
            recomputed_batch = make_history_windows(
                panel,
                request=window_request,
                dataset_audit=recomputed_audit,
            )
        except (TypeError, ValueError, KeyError) as error:
            raise PredictionIntegrityError(
                "evaluation target window/panel/source lineage verification failed"
            ) from error
        if deepcopy(dict(dataset_audit)) != recomputed_audit:
            raise PredictionIntegrityError(
                "evaluation target sealed dataset audit does not match the panel"
            )
        if (
            verified_sample.get("manifest_sha256")
            != window_request.sample_manifest_sha256
            or verified_sample.get("sampled_panel_binding_sha256")
            != window_request.sampled_panel_binding_sha256
        ):
            raise PredictionIntegrityError(
                "evaluation target sample manifest differs from its window request"
            )
        array_fields = (
            "history",
            "target",
            "occurrence",
            "target_mask",
            "scale",
            "origins",
            "series_id",
        )
        if any(
            not np.array_equal(
                np.asarray(getattr(window_batch, field)),
                np.asarray(getattr(recomputed_batch, field)),
            )
            for field in array_fields
        ) or not window_batch.key_frame.equals(recomputed_batch.key_frame):
            raise PredictionIntegrityError(
                "evaluation target WindowBatch differs from canonical regeneration"
            )
        if (
            window_batch.split_name != recomputed_batch.split_name
            or window_batch.dataset_audit_sha256
            != recomputed_batch.dataset_audit_sha256
            or window_batch.sample_manifest_sha256
            != recomputed_batch.sample_manifest_sha256
            or window_batch.sampled_panel_binding_sha256
            != recomputed_batch.sampled_panel_binding_sha256
            or window_batch.split_contract_sha256
            != recomputed_batch.split_contract_sha256
            or window_batch.request_sha256 != recomputed_batch.request_sha256
        ):
            raise PredictionIntegrityError(
                "evaluation target WindowBatch request binding is invalid"
            )
        name = str(window_batch.split_name)
        if name in {
            "validation",
            "teacher_validation",
            "student_validation",
            "pool_validation",
            "router_validation",
            "sensor_validation",
        }:
            bounds = tuple(int(value) for value in window_request.split.validation)
        elif name in {"evaluation", "outer_evaluation"}:
            bounds = (
                int(window_request.origins[0]),
                int(window_request.origins[-1]) + int(window_request.split.horizon),
            )
        elif name == "warmup":
            bounds = tuple(int(value) for value in window_request.split.warmup)
        else:
            bounds = tuple(int(value) for value in window_request.split.train)
        source_manifest_sha256 = source_manifest.get("aggregate_sha256")
        panel_binding_sha256 = recomputed_audit.get("panel_binding_sha256")
        hashes = {
            "panel_binding_sha256": panel_binding_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "preregistration_sha256": preregistration_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_audit_sha256": recomputed_audit.get("audit_sha256"),
            "sample_manifest_sha256": verified_sample.get("manifest_sha256"),
            "sampled_panel_binding_sha256": verified_sample.get(
                "sampled_panel_binding_sha256"
            ),
            "window_request_sha256": window_request.request_sha256,
            "split_contract_sha256": window_request.split_contract_sha256,
        }
        if not all(_valid_sha256(value) for value in hashes.values()):
            raise PredictionIntegrityError(
                "evaluation target provenance hashes must be lowercase SHA256"
            )
        horizon = int(window_request.split.horizon)
        target_frame = window_batch.key_frame.copy()
        target_frame["y"] = np.asarray(window_batch.target, dtype=np.float64).reshape(-1)
        target_frame["scale"] = np.repeat(
            np.asarray(window_batch.scale, dtype=np.float64), horizon
        )
        target_frame["target_mask"] = np.asarray(
            window_batch.target_mask, dtype=bool
        ).reshape(-1)
        rows = _target_rows(target_frame)
        payload = {
            "schema": "prob_head_structure_full_v1.evaluation_target.v1",
            "split_name": name,
            "split_bounds": list(bounds),
            "scale_source": "train_only_rms",
            "target_support": "nonnegative_integer_count",
            "key_columns": list(KEY_COLUMNS),
            "rows": rows,
            "dataset_audit": recomputed_audit,
            **hashes,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "split_name", name)
        object.__setattr__(instance, "split_bounds", (int(bounds[0]), int(bounds[1])))
        object.__setattr__(instance, "row_count", len(rows))
        object.__setattr__(instance, "canonical_payload_json", canonical)
        object.__setattr__(
            instance,
            "artifact_sha256",
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
        return instance

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        window_batch: WindowBatch,
        window_request: WindowRequest,
        panel: Mapping[str, Any],
        dataset_audit: Mapping[str, Any],
        source_manifest: Mapping[str, Any],
        sample_manifest: Mapping[str, Any],
    ) -> "SealedEvaluationTarget":
        if not isinstance(record, Mapping) or set(record) != {"payload", "artifact_sha256"}:
            raise PredictionIntegrityError("evaluation target artifact record is malformed")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise PredictionIntegrityError("evaluation target artifact payload is malformed")
        result = cls.seal(
            window_batch=window_batch,
            window_request=window_request,
            panel=panel,
            dataset_audit=dataset_audit,
            source_manifest=source_manifest,
            sample_manifest=sample_manifest,
            preregistration_sha256=str(payload.get("preregistration_sha256", "")),
            dataset_manifest_sha256=str(payload.get("dataset_manifest_sha256", "")),
        )
        if result.as_dict() != deepcopy(dict(record)):
            raise PredictionIntegrityError("evaluation target artifact digest mismatch")
        return result

    def verify_frame(self, frame: pd.DataFrame) -> None:
        payload = json.loads(self.canonical_payload_json)
        if _target_rows(frame) != payload["rows"]:
            raise PredictionIntegrityError(
                "prediction target values do not match the sealed target artifact"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "payload": json.loads(self.canonical_payload_json),
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class ScientificForecastProvenance:
    """One of the two frozen deterministic scientific forecast representations."""

    quantile_source: str
    mean_source: str

    def __post_init__(self) -> None:
        pair = (str(self.quantile_source), str(self.mean_source))
        allowed = {
            (
                "native_exact_or_numerical_inverse",
                "analytical_predictive_mean",
            ),
            (
                "monotone_piecewise_common_grid",
                "quantile_integral_endpoint_hold",
            ),
        }
        if pair not in allowed:
            raise PredictionIntegrityError(
                "scientific provenance must be a frozen deterministic quantile/mean pair"
            )
        object.__setattr__(self, "quantile_source", pair[0])
        object.__setattr__(self, "mean_source", pair[1])


_SCIENTIFIC_FORECAST_SEAL_TOKEN = object()


def _numeric_content_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.all(np.isfinite(array)):
        raise PredictionIntegrityError("scientific forecast contains NaN/Inf")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "<f8", "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    view = memoryview(array).cast("B")
    for start in range(0, len(view), 1 << 20):
        digest.update(view[start : start + (1 << 20)])
    return digest.hexdigest()


def _forecast_content_binding(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "p_zero",
        "mean",
        *(quantile_column(q) for q in EVALUATION_QUANTILE_GRID),
    ]
    missing = [column for column in [*KEY_COLUMNS, *columns] if column not in frame]
    if missing:
        raise PredictionIntegrityError(
            f"scientific forecast artifact is missing columns: {missing}"
        )
    key_rows = [
        [str(row.dataset_id), str(row.series_id), int(row.origin), int(row.step)]
        for row in frame.loc[:, list(KEY_COLUMNS)].itertuples(index=False)
    ]
    key_json = json.dumps(
        key_rows, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return {
        "case_count": len(frame),
        "case_key_sha256": hashlib.sha256(key_json).hexdigest(),
        "p_zero_sha256": _numeric_content_sha256(frame["p_zero"].to_numpy()),
        "mean_sha256": _numeric_content_sha256(frame["mean"].to_numpy()),
        "quantile_grid": list(EVALUATION_QUANTILE_GRID),
        "quantiles_sha256": _numeric_content_sha256(
            frame[
                [quantile_column(q) for q in EVALUATION_QUANTILE_GRID]
            ].to_numpy()
        ),
    }


@dataclass(frozen=True, init=False)
class ScientificForecastArtifact:
    """Compact content seal for one deterministic scientific forecast frame."""

    provenance: ScientificForecastProvenance
    canonical_payload_json: str
    artifact_sha256: str

    @classmethod
    def _seal(
        cls,
        *,
        frame: pd.DataFrame,
        provenance: ScientificForecastProvenance,
        target_artifact: SealedEvaluationTarget,
        branch_eligibility: BranchEligibility,
        source_kind: str,
        source_binding: Mapping[str, Any],
        _token: object,
    ) -> "ScientificForecastArtifact":
        if _token is not _SCIENTIFIC_FORECAST_SEAL_TOKEN:
            raise PredictionIntegrityError(
                "scientific forecast artifacts may be created only by verified forecast adapters"
            )
        if not isinstance(provenance, ScientificForecastProvenance):
            raise PredictionIntegrityError("scientific forecast provenance is invalid")
        if not isinstance(target_artifact, SealedEvaluationTarget):
            raise PredictionIntegrityError("scientific forecast target seal is invalid")
        if not isinstance(branch_eligibility, BranchEligibility):
            raise PredictionIntegrityError("scientific forecast branch lineage is invalid")
        target_artifact.verify_frame(frame)
        _evaluate_prediction_frame_core(frame)
        _validate_zero_quantile_coherence(frame)
        if source_kind not in {"NATIVE_TEACHER", "P1", "P2", "P3", "STUDENT"}:
            raise PredictionIntegrityError("scientific forecast source kind is unregistered")
        binding = deepcopy(dict(source_binding))
        if not binding or any(
            key.endswith("_sha256") and not _valid_sha256(value)
            for key, value in binding.items()
        ):
            raise PredictionIntegrityError("scientific forecast source binding is invalid")
        payload = {
            "schema": "prob_head_structure_full_v1.scientific_forecast.v1",
            "target_artifact_sha256": target_artifact.artifact_sha256,
            "provenance": {
                "quantile_source": provenance.quantile_source,
                "mean_source": provenance.mean_source,
            },
            "branch_eligibility": branch_eligibility.as_dict(),
            "source_kind": source_kind,
            "source_binding": binding,
            "content": _forecast_content_binding(frame),
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "provenance", provenance)
        object.__setattr__(instance, "canonical_payload_json", canonical)
        object.__setattr__(
            instance,
            "artifact_sha256",
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
        return instance

    def verify(
        self,
        frame: pd.DataFrame,
        *,
        target_artifact: SealedEvaluationTarget,
        branch_eligibility: BranchEligibility,
    ) -> None:
        if not isinstance(target_artifact, SealedEvaluationTarget):
            raise PredictionIntegrityError("scientific forecast target seal is invalid")
        if not isinstance(branch_eligibility, BranchEligibility):
            raise PredictionIntegrityError("scientific forecast branch lineage is invalid")
        target_artifact.verify_frame(frame)
        payload = json.loads(self.canonical_payload_json)
        if (
            hashlib.sha256(self.canonical_payload_json.encode("utf-8")).hexdigest()
            != self.artifact_sha256
            or payload.get("target_artifact_sha256") != target_artifact.artifact_sha256
            or payload.get("branch_eligibility") != branch_eligibility.as_dict()
            or payload.get("content") != _forecast_content_binding(frame)
        ):
            raise PredictionIntegrityError(
                "scientific forecast content or lineage binding mismatch"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "payload": json.loads(self.canonical_payload_json),
            "artifact_sha256": self.artifact_sha256,
        }


def _seal_scientific_forecast_artifact(
    *,
    frame: pd.DataFrame,
    provenance: ScientificForecastProvenance,
    target_artifact: SealedEvaluationTarget,
    branch_eligibility: BranchEligibility,
    source_kind: str,
    source_binding: Mapping[str, Any],
) -> ScientificForecastArtifact:
    """Internal handoff used only after a native/pool/student adapter verifies values."""

    return ScientificForecastArtifact._seal(
        frame=frame,
        provenance=provenance,
        target_artifact=target_artifact,
        branch_eligibility=branch_eligibility,
        source_kind=source_kind,
        source_binding=source_binding,
        _token=_SCIENTIFIC_FORECAST_SEAL_TOKEN,
    )


@dataclass(frozen=True)
class EvaluationResult:
    levels: Mapping[str, pd.DataFrame]
    validity: Mapping[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, frame in self.levels.items():
            clean = frame.astype(object).where(pd.notna(frame), None)
            payload[name] = clean.to_dict(orient="records")
        payload["validity"] = dict(self.validity)
        return payload


@dataclass(frozen=True)
class DiagnosticEvaluationResult:
    """Metrics that cannot enter confirmatory tables, gates, or checkpoints."""

    levels: Mapping[str, pd.DataFrame]
    validity: Mapping[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, frame in self.levels.items():
            clean = frame.astype(object).where(pd.notna(frame), None)
            payload[name] = clean.to_dict(orient="records")
        payload["validity"] = dict(self.validity)
        return payload


def _validate_zero_quantile_coherence(frame: pd.DataFrame) -> None:
    required = ["p_zero", *(quantile_column(q) for q in EVALUATION_QUANTILE_GRID)]
    missing = [column for column in required if column not in frame]
    if missing:
        raise PredictionIntegrityError(
            f"zero-mass coherence check is missing columns: {missing}"
        )
    p_zero = frame["p_zero"].to_numpy(dtype=np.float64)
    quantiles = frame[
        [quantile_column(q) for q in EVALUATION_QUANTILE_GRID]
    ].to_numpy(dtype=np.float64)
    expected_zero = np.asarray(EVALUATION_QUANTILE_GRID)[None, :] <= p_zero[:, None]
    actual_zero = np.isclose(quantiles, 0.0, rtol=0.0, atol=1e-12)
    if not np.array_equal(expected_zero, actual_zero):
        raise PredictionIntegrityError(
            "predictive p0 and quantiles violate q<=p0 iff Q(q)=0"
        )


def _evaluate_prediction_frame_core(
    frame: pd.DataFrame,
    *,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Evaluate forecast values without assigning scientific status."""

    q = np.asarray(quantile_grid, dtype=np.float64)
    if tuple(map(float, q)) != CRPS_QUANTILE_GRID:
        raise ValueError("evaluation CRPS quantile grid is frozen")
    q_columns = [quantile_column(probability) for probability in q]
    evaluation_q = np.asarray(EVALUATION_QUANTILE_GRID, dtype=np.float64)
    evaluation_q_columns = [quantile_column(probability) for probability in evaluation_q]
    required = [
        *KEY_COLUMNS,
        "y",
        "scale",
        "target_mask",
        "p_zero",
        "mean",
        *evaluation_q_columns,
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise PredictionIntegrityError(f"prediction frame is missing columns: {missing}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise PredictionIntegrityError("prediction frame contains duplicate keys")

    raw_mask = frame["target_mask"].to_numpy()
    if raw_mask.dtype != np.bool_ and not all(
        isinstance(value, (bool, np.bool_)) for value in raw_mask.tolist()
    ):
        raise PredictionIntegrityError("target_mask must be boolean")
    valid_mask = raw_mask.astype(bool)
    if not valid_mask.any():
        raise PredictionIntegrityError("prediction frame has no valid target rows")
    original_row_count = len(frame)
    frame = frame.loc[valid_mask].reset_index(drop=True)

    y = frame["y"].to_numpy(dtype=np.float64)
    scale = frame["scale"].to_numpy(dtype=np.float64)
    p_zero = frame["p_zero"].to_numpy(dtype=np.float64)
    mean = frame["mean"].to_numpy(dtype=np.float64)
    quantiles = frame[q_columns].to_numpy(dtype=np.float64)
    evaluation_quantiles = frame[evaluation_q_columns].to_numpy(dtype=np.float64)
    prediction_values = np.column_stack([p_zero, mean, evaluation_quantiles])
    nonfinite_rate = float((~np.isfinite(prediction_values)).mean())
    if nonfinite_rate > 0.0 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(scale)):
        raise PredictionIntegrityError(f"NaN/Inf prediction or evaluation input rate is {nonfinite_rate:.6g}")
    if np.any(y < 0.0):
        raise PredictionIntegrityError("negative target violates nonnegative support")
    if np.any(scale <= 0.0):
        raise PredictionIntegrityError("train-only RMS scale must be finite and positive")
    if np.any((p_zero < 0.0) | (p_zero > 1.0)):
        raise PredictionIntegrityError("invalid zero probability outside [0, 1]")
    if np.any(evaluation_quantiles[:, 1:] < evaluation_quantiles[:, :-1]):
        raise PredictionIntegrityError("quantile crossing detected")
    negative_rate = float((prediction_values[:, 1:] < 0.0).mean())
    if negative_rate > 0.0:
        raise PredictionIntegrityError(f"negative prediction rate is {negative_rate:.6g}")

    step = frame.loc[:, list(KEY_COLUMNS)].copy()
    step["sCRPS"] = approximate_crps(y, quantiles, q) / scale
    for probability in REPORTED_QUANTILES:
        column = quantile_column(probability)
        if column not in frame:
            raise PredictionIntegrityError(f"reported quantile {probability} is absent")
        step[f"sQL_{int(probability * 100):02d}"] = (
            2.0 * pinball_loss(y, frame[column].to_numpy(dtype=np.float64), probability) / scale
        )
    step["zero_brier"] = (p_zero - (y == 0.0).astype(np.float64)) ** 2
    normalized_error = (mean - y) / scale
    step["normalized_squared_error"] = normalized_error**2
    step["normalized_absolute_error"] = np.abs(normalized_error)
    for level in CENTRAL_INTERVALS:
        lower_probability = (1.0 - level) / 2.0
        upper_probability = 1.0 - lower_probability
        lower = frame[quantile_column(lower_probability)].to_numpy(dtype=np.float64)
        upper = frame[quantile_column(upper_probability)].to_numpy(dtype=np.float64)
        code = int(level * 100)
        step[f"covered_{code:02d}"] = ((y >= lower) & (y <= upper)).astype(np.float64)
        step[f"raw_interval_width_{code:02d}"] = upper - lower
        step[f"interval_width_{code:02d}"] = (upper - lower) / scale
    step["negative_prediction"] = negative_rate
    step["nan_inf_prediction"] = nonfinite_rate

    series_origin = _aggregate_metrics(
        step, ["dataset_id", "series_id", "origin"]
    )
    series = _aggregate_metrics(series_origin, ["dataset_id", "series_id"])
    dataset = _aggregate_metrics(series, ["dataset_id"])
    levels = {
        "step": step,
        "series_origin": series_origin,
        "series": series,
        "dataset": dataset,
    }
    return levels, {
        "rows": len(frame),
        "masked_rows": original_row_count - len(frame),
        "duplicate_keys": 0,
        "quantile_crossings": 0,
        "negative_prediction_rate": negative_rate,
        "nan_inf_rate": nonfinite_rate,
    }


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    forecast_artifact: ScientificForecastArtifact,
    target_artifact: SealedEvaluationTarget,
    branch_eligibility: BranchEligibility,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> EvaluationResult:
    """Evaluate a deterministic scientific forecast with frozen provenance."""

    if not isinstance(forecast_artifact, ScientificForecastArtifact):
        raise PredictionIntegrityError(
            "confirmatory evaluation requires a sealed scientific forecast artifact"
        )
    if not isinstance(target_artifact, SealedEvaluationTarget):
        raise PredictionIntegrityError("a sealed evaluation target artifact is required")
    if (
        not isinstance(branch_eligibility, BranchEligibility)
        or not branch_eligibility.confirmatory_eligible
        or branch_eligibility.role != "CONFIRMATORY"
    ):
        raise PredictionIntegrityError(
            "confirmatory evaluation requires an eligible branch lineage"
        )
    forecast_artifact.verify(
        frame,
        target_artifact=target_artifact,
        branch_eligibility=branch_eligibility,
    )
    levels, validity = _evaluate_prediction_frame_core(
        frame, quantile_grid=quantile_grid
    )
    _validate_zero_quantile_coherence(frame)
    return EvaluationResult(
        levels=levels,
        validity={
            **validity,
            "quantile_source": forecast_artifact.provenance.quantile_source,
            "mean_source": forecast_artifact.provenance.mean_source,
            "confirmatory_eligible": True,
            "scientific_role": "CONFIRMATORY",
            "branch_eligibility": branch_eligibility.as_dict(),
            "target_artifact_sha256": target_artifact.artifact_sha256,
            "scientific_forecast_artifact_sha256": forecast_artifact.artifact_sha256,
        },
    )


def evaluate_native_diagnostic_prediction_frame(
    frame: pd.DataFrame,
    *,
    forecast_artifact: ScientificForecastArtifact,
    target_artifact: SealedEvaluationTarget,
    branch_eligibility: BranchEligibility,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> DiagnosticEvaluationResult:
    """Evaluate native deterministic forecasts after an upstream scientific failure."""

    if not isinstance(forecast_artifact, ScientificForecastArtifact):
        raise PredictionIntegrityError(
            "native diagnostic evaluation requires a sealed scientific forecast artifact"
        )
    if not isinstance(target_artifact, SealedEvaluationTarget):
        raise PredictionIntegrityError("a sealed evaluation target artifact is required")
    if (
        not isinstance(branch_eligibility, BranchEligibility)
        or branch_eligibility.confirmatory_eligible
        or not branch_eligibility.role.startswith("DIAGNOSTIC_CONTINUATION_AFTER_")
    ):
        raise PredictionIntegrityError(
            "native diagnostic evaluation requires sealed diagnostic branch lineage"
        )
    forecast_artifact.verify(
        frame,
        target_artifact=target_artifact,
        branch_eligibility=branch_eligibility,
    )
    levels, validity = _evaluate_prediction_frame_core(
        frame, quantile_grid=quantile_grid
    )
    _validate_zero_quantile_coherence(frame)
    return DiagnosticEvaluationResult(
        levels=levels,
        validity={
            **validity,
            "quantile_source": forecast_artifact.provenance.quantile_source,
            "mean_source": forecast_artifact.provenance.mean_source,
            "confirmatory_eligible": False,
            "scientific_role": branch_eligibility.role,
            "branch_eligibility": branch_eligibility.as_dict(),
            "target_artifact_sha256": target_artifact.artifact_sha256,
            "scientific_forecast_artifact_sha256": forecast_artifact.artifact_sha256,
        },
    )


def evaluate_diagnostic_prediction_frame(
    frame: pd.DataFrame,
    *,
    quantile_source: str,
    mean_source: str,
    scientific_role: str,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> DiagnosticEvaluationResult:
    """Evaluate empirical/sample forecasts in an explicitly diagnostic output."""

    if (
        str(quantile_source),
        str(mean_source),
    ) != ("empirical_inverse_from_samples", "empirical_sample_mean"):
        raise ValueError("diagnostic evaluator requires the registered empirical sources")
    role = _validate_diagnostic_role(scientific_role)
    levels, validity = _evaluate_prediction_frame_core(
        frame, quantile_grid=quantile_grid
    )
    return DiagnosticEvaluationResult(
        levels=levels,
        validity={
            **validity,
            "quantile_source": str(quantile_source),
            "mean_source": str(mean_source),
            "confirmatory_eligible": False,
            "scientific_role": role,
        },
    )
