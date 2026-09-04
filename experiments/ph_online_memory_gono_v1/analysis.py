"""Pure analysis and Go/No-Go decisions for PH-ONLINE-MEMORY-GONO-v1.

The functions in this module consume already-produced paired prediction or
series-origin loss frames.  They never fit a model and never write artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .evaluation import ALPHA_GRID
from .gates import spearman_diagnostic


TRANSFER_DATASETS = ("m5", "favorita")
BOOTSTRAP_DRAWS = 2000
ANALYSIS_SEED = 20260904
ORIGINS_PER_SERIES = 6
FORECAST_HORIZON = 28
CASE_KEYS = ("dataset_id", "series_id", "origin")
STEP_KEYS = (*CASE_KEYS, "step")

FINAL_VERDICT_TOKENS = (
    "FULL_NO_GO",
    "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE",
    "TEMPORAL_RECURRENCE_NO_GO",
    "ONLINE_MEMORY_NO_GO",
    "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
    "RETRIEVAL_UNSAFE_NO_GO",
    "RETRIEVAL_SIGNAL_NOT_IDENTIFIED",
    "RETRIEVAL_MEMORY_GO",
)

_STEP_COLUMNS = (
    *STEP_KEYS,
    "y_observed",
    "point_mean_prediction",
    "hurdle_mean_prediction",
    "target_mask",
    "policy_scale_squared",
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


def _finite(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be Boolean")
    return bool(value)


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


def _base_frame(
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
    result = frame.loc[:, columns].copy()
    if bool(result.loc[:, keys].isna().any(axis=None)):
        raise ValueError(f"{name} keys must not be missing")
    if bool(result.duplicated(list(keys), keep=False).any()):
        raise ValueError(f"{name} keys must be unique")
    result["dataset_id"] = result["dataset_id"].astype(str)
    result["series_id"] = result["series_id"].astype(str)
    return result


def _validate_six_origin_clusters(frame: pd.DataFrame, *, name: str) -> None:
    for dataset_id, dataset in frame.groupby(
        "dataset_id", sort=True, observed=True
    ):
        expected_origins = tuple(sorted(dataset["origin"].unique().tolist()))
        if len(expected_origins) != ORIGINS_PER_SERIES:
            raise ValueError(
                f"{name} dataset {dataset_id!r} must contain exactly six origins"
            )
        for series_id, series in dataset.groupby(
            "series_id", sort=True, observed=True
        ):
            origins = tuple(sorted(series["origin"].unique().tolist()))
            if origins != expected_origins:
                raise ValueError(
                    f"{name} series {(dataset_id, series_id)!r} must contain "
                    "the same six origins"
                )


def _validated_case_losses(
    frame: pd.DataFrame,
    *,
    name: str,
    loss_columns: Sequence[str],
) -> pd.DataFrame:
    if len(set(loss_columns)) != len(loss_columns):
        raise ValueError("loss column names must be distinct")
    result = _base_frame(
        frame,
        name=name,
        columns=(*CASE_KEYS, *loss_columns),
        keys=CASE_KEYS,
    )
    origins = _numeric_column(
        result, "origin", nonnegative=True, integer=True
    )
    result["origin"] = origins.astype(np.int64)
    for column in loss_columns:
        result[column] = _numeric_column(result, column, nonnegative=True)
    _validate_six_origin_clusters(result, name=name)
    return result.sort_values(list(CASE_KEYS), kind="mergesort").reset_index(
        drop=True
    )


def _validated_steps(frame: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(
        frame,
        name="step_predictions",
        columns=_STEP_COLUMNS,
        keys=STEP_KEYS,
    )
    origins = _numeric_column(
        result, "origin", nonnegative=True, integer=True
    )
    steps = _numeric_column(result, "step", nonnegative=True, integer=True)
    result["origin"] = origins.astype(np.int64)
    result["step"] = steps.astype(np.int64)
    for column in (
        "y_observed",
        "point_mean_prediction",
        "hurdle_mean_prediction",
    ):
        result[column] = _numeric_column(result, column)
    result["policy_scale_squared"] = _numeric_column(
        result, "policy_scale_squared", positive=True
    )
    if not all(
        isinstance(value, (bool, np.bool_))
        for value in result["target_mask"].tolist()
    ):
        raise ValueError("target_mask must contain only Boolean values")
    result["target_mask"] = result["target_mask"].astype(bool)
    _validate_six_origin_clusters(result, name="step_predictions")

    expected_steps = tuple(range(FORECAST_HORIZON))
    for case_key, case in result.groupby(
        list(CASE_KEYS), sort=True, observed=True
    ):
        observed = tuple(sorted(case["step"].astype(int).tolist()))
        if observed != expected_steps:
            raise ValueError(
                f"step_predictions case {case_key!r} must contain exactly steps 0..27"
            )
        if not bool(case["target_mask"].all()):
            raise ValueError(
                f"step_predictions case {case_key!r} must have all 28 targets observed"
            )
        if case["policy_scale_squared"].nunique(dropna=False) != 1:
            raise ValueError("policy_scale_squared must be constant within a case")

    scale_counts = result.groupby(
        ["dataset_id", "series_id"], sort=True, observed=True
    )["policy_scale_squared"].nunique(dropna=False)
    if bool((scale_counts != 1).any()):
        raise ValueError("policy_scale_squared must be constant for each series")
    return result.sort_values(list(STEP_KEYS), kind="mergesort").reset_index(
        drop=True
    )


def relative_improvement_percent(candidate_loss: float, baseline_loss: float) -> float:
    """Return ``100 * (1 - candidate / baseline)`` with a positive baseline."""

    candidate = _finite("candidate_loss", candidate_loss, nonnegative=True)
    baseline = _finite("baseline_loss", baseline_loss, nonnegative=True)
    if baseline <= 0.0:
        raise ValueError("baseline_loss must be positive")
    return 100.0 * (1.0 - candidate / baseline)


def summarize_loss_comparison(
    policy_losses: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
) -> dict[str, object]:
    """Keep raw direction losses and their relative improvements together."""

    if not isinstance(candidate_column, str) or not candidate_column:
        raise TypeError("candidate_column must be a nonempty string")
    if not isinstance(baseline_column, str) or not baseline_column:
        raise TypeError("baseline_column must be a nonempty string")
    frame = _validated_case_losses(
        policy_losses,
        name="policy_losses",
        loss_columns=(candidate_column, baseline_column),
    )
    if set(frame["dataset_id"].unique()) != set(TRANSFER_DATASETS):
        raise ValueError("policy_losses must contain exactly m5 and favorita")
    directions: dict[str, dict[str, float]] = {}
    for dataset in TRANSFER_DATASETS:
        subset = frame.loc[frame["dataset_id"] == dataset]
        candidate = float(subset[candidate_column].mean())
        baseline = float(subset[baseline_column].mean())
        directions[dataset] = {
            "candidate_loss": candidate,
            "baseline_loss": baseline,
            "ri_percent": relative_improvement_percent(candidate, baseline),
        }
    return {
        "directions": directions,
        "macro_ri_percent": float(
            np.mean([directions[item]["ri_percent"] for item in TRANSFER_DATASETS])
        ),
    }


def _case_loss_for_alpha(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    prediction = (
        (1.0 - alpha) * frame["point_mean_prediction"].to_numpy(dtype=np.float64)
        + alpha * frame["hurdle_mean_prediction"].to_numpy(dtype=np.float64)
    )
    error = frame["y_observed"].to_numpy(dtype=np.float64) - prediction
    normalized = np.square(error) / frame["policy_scale_squared"].to_numpy(
        dtype=np.float64
    )
    mask = frame["target_mask"].to_numpy(dtype=bool)
    work = frame.loc[:, CASE_KEYS].copy()
    work["loss_sum"] = np.where(mask, normalized, 0.0)
    cases = work.groupby(
        list(CASE_KEYS), sort=True, observed=True, as_index=False
    ).agg(loss_sum=("loss_sum", "sum"))
    cases["loss"] = cases["loss_sum"] / FORECAST_HORIZON
    if not bool(np.isfinite(cases["loss"]).all()):
        raise ValueError("computed normalized case loss must be finite")
    return cases.loc[:, [*CASE_KEYS, "loss"]]


def _grid(alpha_grid: Iterable[float]) -> tuple[float, ...]:
    alphas = tuple(_finite("alpha", value) for value in alpha_grid)
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alpha_grid must be nonempty and unique")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alpha_grid values must lie in [0, 1]")
    if tuple(sorted(alphas)) != alphas:
        raise ValueError("alpha_grid must be increasing")
    return alphas


def _best_grid_value(
    grouped: pd.DataFrame,
    *,
    alpha_columns: Sequence[str],
    alphas: Sequence[float],
) -> tuple[float, float]:
    losses = grouped.loc[:, alpha_columns].mean(axis=0).to_numpy(dtype=np.float64)
    best_index = int(np.argmin(losses))
    return float(alphas[best_index]), float(losses[best_index])


def oracle_ladder(
    step_predictions: pd.DataFrame,
    *,
    alpha_grid: Iterable[float] = ALPHA_GRID,
) -> dict[str, dict[str, object]]:
    """Compute hard and convex oracle ladders on a paired six-origin panel.

    Convex candidates use the frozen 0.05 grid by default.  Hard and convex
    results live in separate nested families so a consumer cannot silently use
    a hard denominator for a convex policy.
    """

    frame = _validated_steps(step_predictions)
    alphas = _grid(alpha_grid)
    alpha_columns = [f"alpha_{index}" for index in range(len(alphas))]
    by_alpha: list[pd.DataFrame] = []
    for alpha, column in zip(alphas, alpha_columns):
        losses = _case_loss_for_alpha(frame, alpha).rename(
            columns={"loss": column}
        )
        by_alpha.append(losses)
    cases = by_alpha[0]
    for losses in by_alpha[1:]:
        cases = cases.merge(
            losses,
            on=list(CASE_KEYS),
            how="inner",
            validate="one_to_one",
            sort=False,
        )

    point_column = alpha_columns[alphas.index(0.0)] if 0.0 in alphas else None
    hurdle_column = alpha_columns[alphas.index(1.0)] if 1.0 in alphas else None
    half_column = alpha_columns[alphas.index(0.5)] if 0.5 in alphas else None
    if point_column is None or hurdle_column is None or half_column is None:
        raise ValueError("alpha_grid must include 0.0, 0.5, and 1.0")

    result: dict[str, dict[str, object]] = {}
    for dataset_id, dataset in cases.groupby(
        "dataset_id", sort=True, observed=True
    ):
        point_loss = float(dataset[point_column].mean())
        hurdle_loss = float(dataset[hurdle_column].mean())
        half_loss = float(dataset[half_column].mean())
        if point_loss <= hurdle_loss:
            global_hard_loss = point_loss
            global_hard_expert = "point"
        else:
            global_hard_loss = hurdle_loss
            global_hard_expert = "hurdle"

        series_hard_choices: list[dict[str, object]] = []
        series_hard_losses: list[float] = []
        series_convex_choices: list[dict[str, object]] = []
        series_convex_losses: list[float] = []
        for series_id, group in dataset.groupby(
            "series_id", sort=True, observed=True
        ):
            point_group = float(group[point_column].mean())
            hurdle_group = float(group[hurdle_column].mean())
            if point_group <= hurdle_group:
                hard_expert, hard_loss = "point", point_group
            else:
                hard_expert, hard_loss = "hurdle", hurdle_group
            series_hard_choices.append(
                {
                    "series_id": str(series_id),
                    "expert": hard_expert,
                    "loss": hard_loss,
                }
            )
            series_hard_losses.append(hard_loss)
            alpha, loss = _best_grid_value(
                group, alpha_columns=alpha_columns, alphas=alphas
            )
            series_convex_choices.append(
                {"series_id": str(series_id), "alpha": alpha, "loss": loss}
            )
            series_convex_losses.append(loss)

        origin_hard_choices: list[dict[str, object]] = []
        origin_hard_losses: list[float] = []
        origin_convex_choices: list[dict[str, object]] = []
        origin_convex_losses: list[float] = []
        for case in dataset.itertuples(index=False):
            point_case = float(getattr(case, point_column))
            hurdle_case = float(getattr(case, hurdle_column))
            if point_case <= hurdle_case:
                hard_expert, hard_loss = "point", point_case
            else:
                hard_expert, hard_loss = "hurdle", hurdle_case
            origin_hard_choices.append(
                {
                    "series_id": str(case.series_id),
                    "origin": int(case.origin),
                    "expert": hard_expert,
                    "loss": hard_loss,
                }
            )
            origin_hard_losses.append(hard_loss)
            case_losses = np.asarray(
                [float(getattr(case, column)) for column in alpha_columns],
                dtype=np.float64,
            )
            best_index = int(np.argmin(case_losses))
            alpha, loss = float(alphas[best_index]), float(case_losses[best_index])
            origin_convex_choices.append(
                {
                    "series_id": str(case.series_id),
                    "origin": int(case.origin),
                    "alpha": alpha,
                    "loss": loss,
                }
            )
            origin_convex_losses.append(loss)

        global_alpha, global_convex_loss = _best_grid_value(
            dataset, alpha_columns=alpha_columns, alphas=alphas
        )
        result[str(dataset_id)] = {
            "dataset_id": str(dataset_id),
            "n_series": int(dataset["series_id"].nunique()),
            "n_origins": int(dataset["origin"].nunique()),
            "n_cases": int(len(dataset)),
            "always_point_loss": point_loss,
            "always_hurdle_loss": hurdle_loss,
            "half_half_loss": half_loss,
            "families": {
                "hard": {
                    "global_oracle_loss": global_hard_loss,
                    "global_oracle_expert": global_hard_expert,
                    "series_oracle_loss": float(np.mean(series_hard_losses)),
                    "series_choices": series_hard_choices,
                    "origin_oracle_loss": float(np.mean(origin_hard_losses)),
                    "origin_choices": origin_hard_choices,
                },
                "convex": {
                    "global_static_loss": global_convex_loss,
                    "global_static_alpha": global_alpha,
                    "series_oracle_loss": float(np.mean(series_convex_losses)),
                    "series_alphas": series_convex_choices,
                    "origin_oracle_loss": float(np.mean(origin_convex_losses)),
                    "origin_alphas": origin_convex_choices,
                    "alpha_grid": [float(alpha) for alpha in alphas],
                },
            },
        }
    return result


def _transfer_mapping(name: str, values: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(values) != set(TRANSFER_DATASETS):
        raise ValueError(f"{name} must contain exactly m5 and favorita")
    return {
        dataset: _finite(f"{name}[{dataset!r}]", values[dataset])
        for dataset in TRANSFER_DATASETS
    }


def evaluate_gate0(
    ladders: Mapping[str, Mapping[str, object]],
    *,
    heterogeneous_diagnostic: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate Gate 0 against the convex global-static denominator only."""

    if not isinstance(ladders, Mapping) or set(ladders) != set(TRANSFER_DATASETS):
        raise ValueError("ladders must contain exactly m5 and favorita")
    gains: dict[str, float] = {}
    for dataset in TRANSFER_DATASETS:
        try:
            convex = ladders[dataset]["families"]["convex"]  # type: ignore[index]
            global_loss = convex["global_static_loss"]  # type: ignore[index]
            origin_loss = convex["origin_oracle_loss"]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"ladders[{dataset!r}] lacks the convex oracle family"
            ) from exc
        gains[dataset] = relative_improvement_percent(origin_loss, global_loss)
    macro = float(np.mean(list(gains.values())))
    checks = {
        "macro_at_least_2_percent": macro >= 2.0,
        "m5_at_least_1_percent": gains["m5"] >= 1.0,
        "favorita_at_least_1_percent": gains["favorita"] >= 1.0,
    }
    passed = all(checks.values())
    result: dict[str, object] = {
        "gains_percent": gains,
        "macro_gain_percent": macro,
        "checks": checks,
        "passed": passed,
        "stage_verdict": None if passed else "POINT_HURDLE_MEMORY_NO_GO",
        "heterogeneous_diagnostic": None,
        "final_verdict": None,
    }
    if passed:
        if heterogeneous_diagnostic is not None:
            raise ValueError(
                "heterogeneous diagnostic is permitted only after Gate 0 failure"
            )
        return result

    diagnostic = (
        {"status": "NOT_AVAILABLE"}
        if heterogeneous_diagnostic is None
        else dict(heterogeneous_diagnostic)
    )
    status = diagnostic.get("status")
    if status not in {"NOT_AVAILABLE", "AVAILABLE"}:
        raise ValueError(
            "heterogeneous_diagnostic status must be NOT_AVAILABLE or AVAILABLE"
        )
    heterogeneous_pass = False
    if status == "AVAILABLE":
        if "macro_gain_percent" not in diagnostic:
            raise ValueError(
                "available heterogeneous diagnostic requires macro_gain_percent"
            )
        heterogeneous_gain = _finite(
            "heterogeneous macro_gain_percent", diagnostic["macro_gain_percent"]
        )
        heterogeneous_pass = heterogeneous_gain >= 2.0
        diagnostic["macro_gain_percent"] = heterogeneous_gain
    else:
        if set(diagnostic) != {"status"}:
            raise ValueError(
                "NOT_AVAILABLE heterogeneous diagnostic cannot contain results"
            )
        diagnostic["macro_gain_percent"] = None
    diagnostic["passed"] = heterogeneous_pass
    result["heterogeneous_diagnostic"] = diagnostic
    result["final_verdict"] = (
        "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE"
        if heterogeneous_pass
        else "FULL_NO_GO"
    )
    return result


def adjacent_delta_pairs(case_losses: pd.DataFrame) -> pd.DataFrame:
    """Build all five adjacent-origin pairs from ``Delta = L_H - L_P``."""

    frame = _validated_case_losses(
        case_losses,
        name="expert_case_losses",
        loss_columns=("point_normalized_loss", "hurdle_normalized_loss"),
    )
    frame["delta"] = (
        frame["hurdle_normalized_loss"] - frame["point_normalized_loss"]
    )
    rows: list[dict[str, object]] = []
    for (dataset_id, series_id), series in frame.groupby(
        ["dataset_id", "series_id"], sort=True, observed=True
    ):
        ordered = series.sort_values("origin", kind="mergesort")
        origins = ordered["origin"].to_numpy(dtype=np.int64)
        deltas = ordered["delta"].to_numpy(dtype=np.float64)
        for index in range(1, len(ordered)):
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "series_id": str(series_id),
                    "previous_origin": int(origins[index - 1]),
                    "origin": int(origins[index]),
                    "previous_delta": float(deltas[index - 1]),
                    "current_delta": float(deltas[index]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset_id", "previous_origin", "origin", "series_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _validated_delta_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "dataset_id",
        "series_id",
        "previous_origin",
        "origin",
        "previous_delta",
        "current_delta",
    )
    result = _base_frame(
        pairs,
        name="adjacent_delta_pairs",
        columns=columns,
        keys=("dataset_id", "series_id", "previous_origin", "origin"),
    )
    for column in ("previous_origin", "origin"):
        result[column] = _numeric_column(
            result, column, nonnegative=True, integer=True
        ).astype(np.int64)
    for column in ("previous_delta", "current_delta"):
        result[column] = _numeric_column(result, column)
    if bool((result["previous_origin"] >= result["origin"]).any()):
        raise ValueError("previous_origin must precede origin")
    counts = result.groupby(
        ["dataset_id", "series_id"], sort=True, observed=True
    ).size()
    if bool((counts != ORIGINS_PER_SERIES - 1).any()):
        raise ValueError("every series must contribute five adjacent origin pairs")
    return result.sort_values(
        ["dataset_id", "previous_origin", "origin", "series_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def shuffle_previous_delta_within_pair(
    pairs: pd.DataFrame, *, seed: int = ANALYSIS_SEED
) -> pd.DataFrame:
    """Shuffle prior Delta across series separately inside each origin pair."""

    fixed_seed = _integer("seed", seed, minimum=0)
    result = _validated_delta_pairs(pairs)
    rng = np.random.default_rng(fixed_seed)
    group_keys = ["dataset_id", "previous_origin", "origin"]
    for _, indices in result.groupby(
        group_keys, sort=True, observed=True
    ).groups.items():
        locations = np.asarray(list(indices), dtype=np.int64)
        values = result.loc[locations, "previous_delta"].to_numpy(
            dtype=np.float64, copy=True
        )
        result.loc[locations, "previous_delta"] = values[
            rng.permutation(len(values))
        ]
    return result


def _ci95(values: np.ndarray) -> list[float]:
    if values.ndim != 1 or values.size == 0 or not bool(np.isfinite(values).all()):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    interval = np.percentile(values, [2.5, 97.5], method="linear")
    return [float(interval[0]), float(interval[1])]


def evaluate_gate1a(
    diagnostics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the strict recurrence and real-minus-shuffle thresholds."""

    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(
        TRANSFER_DATASETS
    ):
        raise ValueError("diagnostics must contain exactly m5 and favorita")
    observed: dict[str, dict[str, object]] = {}
    all_checks: list[bool] = []
    for dataset in TRANSFER_DATASETS:
        diagnostic = diagnostics[dataset]
        status = diagnostic.get("real_status")
        if status not in {"OK", "DEGENERATE"}:
            raise ValueError("real_status must be OK or DEGENERATE")
        if status == "DEGENERATE":
            if diagnostic.get("real_rho") is not None:
                raise ValueError("DEGENERATE real_rho must be null")
            rho = None
            shuffled = diagnostic.get("shuffled_rho")
            if shuffled is not None:
                shuffled = _finite("shuffled_rho", shuffled)
            gap = None
            rho_pass = False
            gap_pass = False
        else:
            rho = _finite("real_rho", diagnostic.get("real_rho"))
            shuffled_raw = diagnostic.get("shuffled_rho")
            if shuffled_raw is None:
                shuffled = None
                gap = None
                gap_pass = False
            else:
                shuffled = _finite("shuffled_rho", shuffled_raw)
                gap = rho - shuffled
                gap_pass = gap > 0.05
            rho_pass = rho > 0.10
        dataset_pass = rho_pass and gap_pass
        all_checks.append(dataset_pass)
        observed[dataset] = {
            "real_status": status,
            "real_rho": rho,
            "shuffled_rho": shuffled,
            "real_minus_shuffled_rho": gap,
            "rho_above_0_10": rho_pass,
            "gap_above_0_05": gap_pass,
            "passed": dataset_pass,
        }
    return {
        "datasets": observed,
        "passed": all(all_checks),
        "failure_verdict": None
        if all(all_checks)
        else "TEMPORAL_RECURRENCE_NO_GO",
    }


def temporal_recurrence(
    case_losses: pd.DataFrame,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, object]:
    """Compute lag-one Delta Spearman, its cluster CI, and shuffled control."""

    draws = _integer("bootstrap_draws", bootstrap_draws, minimum=1)
    fixed_seed = _integer("seed", seed, minimum=0)
    pairs = adjacent_delta_pairs(case_losses)
    if set(pairs["dataset_id"].unique()) != set(TRANSFER_DATASETS):
        raise ValueError("case_losses must contain exactly m5 and favorita")
    shuffled = shuffle_previous_delta_within_pair(pairs, seed=fixed_seed)
    child_seeds = np.random.SeedSequence(fixed_seed).spawn(len(TRANSFER_DATASETS))
    details: dict[str, dict[str, object]] = {}
    gate_input: dict[str, dict[str, object]] = {}
    for dataset, child_seed in zip(TRANSFER_DATASETS, child_seeds):
        original = pairs.loc[pairs["dataset_id"] == dataset]
        control = shuffled.loc[shuffled["dataset_id"] == dataset]
        real = spearman_diagnostic(
            original["previous_delta"].to_numpy(dtype=np.float64),
            original["current_delta"].to_numpy(dtype=np.float64),
        )
        shuffled_result = spearman_diagnostic(
            control["previous_delta"].to_numpy(dtype=np.float64),
            control["current_delta"].to_numpy(dtype=np.float64),
        )
        series_ids = sorted(original["series_id"].unique().tolist())
        rows_by_series = {
            series_id: original.index[
                original["series_id"] == series_id
            ].to_numpy(dtype=np.int64)
            for series_id in series_ids
        }
        rng = np.random.default_rng(child_seed)
        bootstrap_rhos: list[float] = []
        for _ in range(draws):
            sampled = rng.choice(series_ids, size=len(series_ids), replace=True)
            locations = np.concatenate([rows_by_series[item] for item in sampled])
            resampled = pairs.loc[locations]
            diagnostic = spearman_diagnostic(
                resampled["previous_delta"].to_numpy(dtype=np.float64),
                resampled["current_delta"].to_numpy(dtype=np.float64),
            )
            if diagnostic["status"] == "OK":
                bootstrap_rhos.append(float(diagnostic["rho"]))
        ci = (
            _ci95(np.asarray(bootstrap_rhos, dtype=np.float64))
            if bootstrap_rhos
            else None
        )
        shuffled_rho = (
            float(shuffled_result["rho"])
            if shuffled_result["status"] == "OK"
            else None
        )
        gate_input[dataset] = {
            "real_status": real["status"],
            "real_rho": real["rho"],
            "shuffled_rho": shuffled_rho,
        }
        details[dataset] = {
            "real_status": real["status"],
            "real_rho": real["rho"],
            "real_pvalue": real["pvalue"],
            "rho_ci95": ci,
            "shuffled_status": shuffled_result["status"],
            "shuffled_rho": shuffled_rho,
            "real_minus_shuffled_rho": None
            if real["rho"] is None or shuffled_rho is None
            else float(real["rho"]) - shuffled_rho,
            "n_adjacent_pairs": int(len(original)),
            "bootstrap_draws": draws,
            "bootstrap_valid_draws": int(len(bootstrap_rhos)),
        }
    gate = evaluate_gate1a(gate_input)
    for dataset in TRANSFER_DATASETS:
        details[dataset].update(
            {
                "rho_above_0_10": gate["datasets"][dataset]["rho_above_0_10"],
                "gap_above_0_05": gate["datasets"][dataset]["gap_above_0_05"],
                "passed": gate["datasets"][dataset]["passed"],
            }
        )
    return {
        "datasets": details,
        "passed": gate["passed"],
        "failure_verdict": gate["failure_verdict"],
        "seed": fixed_seed,
        "uncertainty_scope": (
            "series uncertainty conditional on the six observed origins; "
            "not origin uncertainty"
        ),
    }


def paired_series_cluster_bootstrap(
    policy_losses: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, object]:
    """Bootstrap paired policy RI by independently resampling target datasets."""

    if not isinstance(candidate_column, str) or not candidate_column:
        raise TypeError("candidate_column must be a nonempty string")
    if not isinstance(baseline_column, str) or not baseline_column:
        raise TypeError("baseline_column must be a nonempty string")
    frame = _validated_case_losses(
        policy_losses,
        name="policy_losses",
        loss_columns=(candidate_column, baseline_column),
    )
    if set(frame["dataset_id"].unique()) != set(TRANSFER_DATASETS):
        raise ValueError("policy_losses must contain exactly m5 and favorita")
    n_draws = _integer("draws", draws, minimum=1)
    fixed_seed = _integer("seed", seed, minimum=0)
    child_seeds = np.random.SeedSequence(fixed_seed).spawn(len(TRANSFER_DATASETS))
    details: dict[str, dict[str, object]] = {}
    samples_by_dataset: dict[str, np.ndarray] = {}
    for dataset, child_seed in zip(TRANSFER_DATASETS, child_seeds):
        subset = frame.loc[frame["dataset_id"] == dataset]
        clusters = subset.groupby(
            "series_id", sort=True, observed=True, as_index=False
        ).agg(
            candidate_loss=(candidate_column, "mean"),
            baseline_loss=(baseline_column, "mean"),
        )
        candidate = clusters["candidate_loss"].to_numpy(dtype=np.float64)
        baseline = clusters["baseline_loss"].to_numpy(dtype=np.float64)
        point_candidate = float(candidate.mean())
        point_baseline = float(baseline.mean())
        point_ri = relative_improvement_percent(point_candidate, point_baseline)

        rng = np.random.default_rng(child_seed)
        indices = rng.integers(
            0, len(clusters), size=(n_draws, len(clusters)), endpoint=False
        )
        sampled_candidate = candidate[indices].mean(axis=1)
        sampled_baseline = baseline[indices].mean(axis=1)
        if bool((sampled_baseline <= 0.0).any()):
            raise ValueError("a bootstrap draw has zero baseline loss")
        samples = 100.0 * (1.0 - sampled_candidate / sampled_baseline)
        if not bool(np.isfinite(samples).all()):
            raise ValueError("bootstrap relative improvements must be finite")
        samples_by_dataset[dataset] = samples
        details[dataset] = {
            "candidate_loss": point_candidate,
            "baseline_loss": point_baseline,
            "ri_percent": point_ri,
            "ci95_percent": _ci95(samples),
            "bootstrap_ri_percent": [float(value) for value in samples],
            "n_series": int(len(clusters)),
        }
    macro_samples = np.mean(
        np.vstack([samples_by_dataset[item] for item in TRANSFER_DATASETS]), axis=0
    )
    macro_point = float(
        np.mean([details[item]["ri_percent"] for item in TRANSFER_DATASETS])
    )
    return {
        "directions": details,
        "macro": {
            "ri_percent": macro_point,
            "ci95_percent": _ci95(macro_samples),
            "bootstrap_ri_percent": [float(value) for value in macro_samples],
        },
        "draws": n_draws,
        "seed": fixed_seed,
        "uncertainty_scope": (
            "series uncertainty conditional on the six observed origins; "
            "each target dataset is resampled independently"
        ),
    }


def evaluate_gate1b(b4_vs_b3_percent: Mapping[str, object]) -> dict[str, object]:
    """Require B4 to improve over source-selected B3 in both directions."""

    values = _transfer_mapping("b4_vs_b3_percent", b4_vs_b3_percent)
    checks = {dataset: value >= 0.30 for dataset, value in values.items()}
    passed = all(checks.values())
    return {
        "ri_percent": values,
        "checks": checks,
        "passed": passed,
        "failure_verdict": None if passed else "ONLINE_MEMORY_NO_GO",
    }


def _validated_ci(name: str, value: object) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a two-value sequence")
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    lower = _finite(f"{name} lower", value[0])
    upper = _finite(f"{name} upper", value[1])
    if lower > upper:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return [lower, upper]


def evaluate_gate2(
    *,
    m1_vs_b4_percent: Mapping[str, object],
    m1_vs_b3_percent: Mapping[str, object],
    bootstrap: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate retrieval's incremental, absolute, and uncertainty criteria."""

    incremental = _transfer_mapping("m1_vs_b4_percent", m1_vs_b4_percent)
    absolute = _transfer_mapping("m1_vs_b3_percent", m1_vs_b3_percent)
    if not isinstance(bootstrap, Mapping):
        raise TypeError("bootstrap must be a mapping")
    try:
        direction_results = bootstrap["directions"]
        macro_result = bootstrap["macro"]
    except KeyError as exc:
        raise ValueError("bootstrap lacks directions or macro result") from exc
    if not isinstance(direction_results, Mapping) or set(direction_results) != set(
        TRANSFER_DATASETS
    ):
        raise ValueError("bootstrap directions must contain exactly m5 and favorita")
    direction_ci = {
        dataset: _validated_ci(
            f"bootstrap direction {dataset} CI",
            direction_results[dataset]["ci95_percent"],  # type: ignore[index]
        )
        for dataset in TRANSFER_DATASETS
    }
    if not isinstance(macro_result, Mapping) or "ci95_percent" not in macro_result:
        raise ValueError("bootstrap macro result lacks ci95_percent")
    macro_ci = _validated_ci("bootstrap macro CI", macro_result["ci95_percent"])
    incremental_macro = float(np.mean(list(incremental.values())))
    absolute_macro = float(np.mean(list(absolute.values())))
    checks = {
        "macro_effect": incremental_macro >= 0.20,
        "direction_safety": all(value >= -0.10 for value in incremental.values()),
        "macro_absolute_usefulness": absolute_macro >= 0.70,
        "direction_absolute_usefulness": all(
            value >= 0.30 for value in absolute.values()
        ),
        "macro_ci": macro_ci[0] > 0.0,
        "dataset_ci": any(interval[0] > 0.0 for interval in direction_ci.values()),
    }
    passed = all(checks.values())
    return {
        "m1_vs_b4_percent": incremental,
        "m1_vs_b4_macro_percent": incremental_macro,
        "m1_vs_b3_percent": absolute,
        "m1_vs_b3_macro_percent": absolute_macro,
        "direction_ci95_percent": direction_ci,
        "macro_ci95_percent": macro_ci,
        "checks": checks,
        "passed": passed,
        "failure_verdict": None
        if passed
        else "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
    }


def evaluate_gate3_safety(policy_losses: pd.DataFrame) -> dict[str, object]:
    """Evaluate worst target-origin RI and per-series Q95 tail loss."""

    columns = (
        "m1_normalized_loss",
        "b3_normalized_loss",
        "b4_normalized_loss",
    )
    frame = _validated_case_losses(
        policy_losses, name="policy_losses", loss_columns=columns
    )
    if set(frame["dataset_id"].unique()) != set(TRANSFER_DATASETS):
        raise ValueError("policy_losses must contain exactly m5 and favorita")
    origin_rows: list[dict[str, object]] = []
    for (dataset, origin), group in frame.groupby(
        ["dataset_id", "origin"], sort=True, observed=True
    ):
        m1 = float(group["m1_normalized_loss"].mean())
        b3 = float(group["b3_normalized_loss"].mean())
        b4 = float(group["b4_normalized_loss"].mean())
        origin_rows.append(
            {
                "dataset_id": str(dataset),
                "origin": int(origin),
                "m1_loss": m1,
                "b3_loss": b3,
                "b4_loss": b4,
                "ri_m1_vs_b3_percent": relative_improvement_percent(m1, b3),
                "ri_m1_vs_b4_percent": relative_improvement_percent(m1, b4),
            }
        )
    worst = min(row["ri_m1_vs_b3_percent"] for row in origin_rows)

    series = frame.groupby(
        ["dataset_id", "series_id"], sort=True, observed=True, as_index=False
    ).agg(
        m1_loss=("m1_normalized_loss", "mean"),
        b4_loss=("b4_normalized_loss", "mean"),
    )
    tail_rows: list[dict[str, object]] = []
    for dataset, group in series.groupby(
        "dataset_id", sort=True, observed=True
    ):
        q95_m1 = float(
            np.quantile(group["m1_loss"].to_numpy(dtype=np.float64), 0.95, method="linear")
        )
        q95_b4 = float(
            np.quantile(group["b4_loss"].to_numpy(dtype=np.float64), 0.95, method="linear")
        )
        if q95_b4 <= 0.0:
            raise ValueError("B4 Q95 baseline must be positive")
        tail_rows.append(
            {
                "dataset_id": str(dataset),
                "q95_m1": q95_m1,
                "q95_b4": q95_b4,
                "q95_m1_over_b4": q95_m1 / q95_b4,
            }
        )
    checks = {
        "worst_origin_at_least_minus_0_50_percent": worst >= -0.50,
        "all_q95_ratios_at_most_1_01": all(
            row["q95_m1_over_b4"] <= 1.01 for row in tail_rows
        ),
    }
    passed = all(checks.values())
    return {
        "origin_metrics": origin_rows,
        "worst_origin_ri_m1_vs_b3_percent": float(worst),
        "tail_metrics": tail_rows,
        "checks": checks,
        "passed": passed,
        "failure_verdict": None if passed else "RETRIEVAL_UNSAFE_NO_GO",
    }


def evaluate_gate3_control(
    *,
    real_percent: Mapping[str, object],
    shuffled_percent: Mapping[str, object],
    random_percent: Mapping[str, object],
) -> dict[str, object]:
    """Compare real retrieval's macro RI with shuffled and random controls."""

    real = _transfer_mapping("real_percent", real_percent)
    shuffled = _transfer_mapping("shuffled_percent", shuffled_percent)
    random = _transfer_mapping("random_percent", random_percent)
    real_macro = float(np.mean(list(real.values())))
    shuffled_macro = float(np.mean(list(shuffled.values())))
    random_macro = float(np.mean(list(random.values())))
    real_pass = real_macro > 0.0
    shuffle_pass = real_pass and shuffled_macro <= 0.25 * real_macro
    random_pass = real_pass and random_macro <= 0.50 * real_macro
    passed = real_pass and shuffle_pass and random_pass
    return {
        "direction_ri_percent": {
            "real": real,
            "shuffled": shuffled,
            "random": random,
        },
        "macro_ri_percent": {
            "real": real_macro,
            "shuffled": shuffled_macro,
            "random": random_macro,
        },
        "shuffle_over_real": shuffled_macro / real_macro if real_pass else None,
        "random_over_real": random_macro / real_macro if real_pass else None,
        "checks": {
            "real_strictly_positive": real_pass,
            "shuffle_at_most_quarter_real": shuffle_pass,
            "random_at_most_half_real": random_pass,
        },
        "passed": passed,
        "failure_verdict": None
        if passed
        else "RETRIEVAL_SIGNAL_NOT_IDENTIFIED",
    }


def evaluate_gate4_seed0(
    *,
    macro_ri_percent: float,
    ci95_percent: Sequence[float],
    direction_ri_percent: Mapping[str, object],
    safety_pass: bool,
    control_pass: bool,
) -> dict[str, object]:
    """Decide whether seed 0 is terminal, clear, or requires seed 1."""

    macro = _finite("macro_ri_percent", macro_ri_percent)
    interval = _validated_ci("ci95_percent", ci95_percent)
    directions = _transfer_mapping("direction_ri_percent", direction_ri_percent)
    safety = _boolean("safety_pass", safety_pass)
    control = _boolean("control_pass", control_pass)
    forbidden_reasons: list[str] = []
    if macro <= 0.0:
        forbidden_reasons.append("macro_nonpositive")
    if not safety:
        forbidden_reasons.append("safety_failed")
    if not control:
        forbidden_reasons.append("control_failed")
    if forbidden_reasons:
        return {
            "action": "STOP_NO_ADDITIONAL_SEED",
            "allow_seed1": False,
            "passed": False,
            "forbidden_reasons": forbidden_reasons,
            "borderline_reasons": [],
        }

    borderline_reasons: list[str] = []
    if 0.0 < macro <= 0.40:
        borderline_reasons.append("macro_in_0_to_0_40_percent")
    if interval[0] <= 0.0 <= interval[1]:
        borderline_reasons.append("ci_contains_zero")
    if interval[0] <= 0.20 <= interval[1]:
        borderline_reasons.append("ci_crosses_0_20_percent")
    if sum(value > 0.0 for value in directions.values()) == 1:
        borderline_reasons.append("exactly_one_direction_positive")
    if borderline_reasons:
        return {
            "action": "RUN_SEED1",
            "allow_seed1": True,
            "passed": None,
            "forbidden_reasons": [],
            "borderline_reasons": borderline_reasons,
        }
    return {
        "action": "ACCEPT_SEED0",
        "allow_seed1": False,
        "passed": True,
        "forbidden_reasons": [],
        "borderline_reasons": [],
    }


def _seed_average_losses(
    seed_policy_losses: pd.DataFrame,
    *,
    loss_columns: Sequence[str],
    expected_seeds: Sequence[int],
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if not isinstance(seed_policy_losses, pd.DataFrame):
        raise TypeError("seed_policy_losses must be a pandas DataFrame")
    if "model_seed" not in seed_policy_losses:
        raise ValueError("seed_policy_losses is missing model_seed")
    if not loss_columns or any(
        not isinstance(column, str) or not column for column in loss_columns
    ):
        raise TypeError("loss_columns must contain nonempty strings")
    if len(set(loss_columns)) != len(loss_columns):
        raise ValueError("loss_columns must be distinct")
    selected = _base_frame(
        seed_policy_losses,
        name="seed_policy_losses",
        columns=("model_seed", *CASE_KEYS, *loss_columns),
        keys=("model_seed", *CASE_KEYS),
    )
    seeds = _numeric_column(
        selected, "model_seed", nonnegative=True, integer=True
    ).astype(np.int64)
    selected["model_seed"] = seeds
    expected = tuple(int(value) for value in expected_seeds)
    if tuple(sorted(selected["model_seed"].unique().tolist())) != expected:
        raise ValueError(f"seed_policy_losses must contain exactly seeds {expected}")

    per_seed_frames: dict[int, pd.DataFrame] = {}
    canonical_keys: pd.MultiIndex | None = None
    validated_frames: list[pd.DataFrame] = []
    for model_seed in expected:
        frame = _validated_case_losses(
            selected.loc[selected["model_seed"] == model_seed],
            name=f"seed_{model_seed}_policy_losses",
            loss_columns=loss_columns,
        )
        keys = pd.MultiIndex.from_frame(frame.loc[:, CASE_KEYS])
        if canonical_keys is None:
            canonical_keys = keys
        elif not keys.equals(canonical_keys):
            raise ValueError("all model seeds must cover identical series-origin keys")
        per_seed_frames[model_seed] = frame.copy()
        frame["model_seed"] = model_seed
        validated_frames.append(frame)

    stacked = pd.concat(validated_frames, ignore_index=True)
    aggregations = {
        column: (column, "mean") for column in loss_columns
    }
    averaged = (
        stacked.groupby(list(CASE_KEYS), sort=True, observed=True, as_index=False)
        .agg(**aggregations)
        .sort_values(list(CASE_KEYS), kind="mergesort")
        .reset_index(drop=True)
    )
    return per_seed_frames, averaged


def _seed_average_policy_analysis(
    seed_policy_losses: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    expected_seeds: Sequence[int],
    draws: int,
    seed: int,
) -> dict[str, object]:
    if not isinstance(candidate_column, str) or not candidate_column:
        raise TypeError("candidate_column must be a nonempty string")
    if not isinstance(baseline_column, str) or not baseline_column:
        raise TypeError("baseline_column must be a nonempty string")
    per_seed_frames, averaged = _seed_average_losses(
        seed_policy_losses,
        loss_columns=(candidate_column, baseline_column),
        expected_seeds=expected_seeds,
    )
    per_seed = {
        str(model_seed): summarize_loss_comparison(
            frame,
            candidate_column=candidate_column,
            baseline_column=baseline_column,
        )
        for model_seed, frame in per_seed_frames.items()
    }
    bootstrap = paired_series_cluster_bootstrap(
        averaged,
        candidate_column=candidate_column,
        baseline_column=baseline_column,
        draws=draws,
        seed=seed,
    )
    return {
        "per_seed": per_seed,
        "seed_average_losses": averaged,
        "seed_average_bootstrap": bootstrap,
    }


def evaluate_seed_average_gate2(
    seed_policy_losses: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, object]:
    """Re-evaluate every Gate-2 criterion on seed-average policy losses.

    Only the three preregistered case-level losses are selected and averaged;
    predictions, precomputed relative improvements, and hyperparameters cannot
    enter this aggregation boundary.
    """

    per_seed, averaged = _seed_average_losses(
        seed_policy_losses,
        loss_columns=(
            "m1_normalized_loss",
            "b4_normalized_loss",
            "b3_normalized_loss",
        ),
        expected_seeds=expected_seeds,
    )
    del per_seed
    m1_vs_b4 = summarize_loss_comparison(
        averaged,
        candidate_column="m1_normalized_loss",
        baseline_column="b4_normalized_loss",
    )
    m1_vs_b3 = summarize_loss_comparison(
        averaged,
        candidate_column="m1_normalized_loss",
        baseline_column="b3_normalized_loss",
    )
    b4_vs_b3 = summarize_loss_comparison(
        averaged,
        candidate_column="b4_normalized_loss",
        baseline_column="b3_normalized_loss",
    )
    bootstrap = paired_series_cluster_bootstrap(
        averaged,
        candidate_column="m1_normalized_loss",
        baseline_column="b4_normalized_loss",
        draws=draws,
        seed=seed,
    )
    gate2 = evaluate_gate2(
        m1_vs_b4_percent={
            dataset: float(m1_vs_b4["directions"][dataset]["ri_percent"])
            for dataset in TRANSFER_DATASETS
        },
        m1_vs_b3_percent={
            dataset: float(m1_vs_b3["directions"][dataset]["ri_percent"])
            for dataset in TRANSFER_DATASETS
        },
        bootstrap=bootstrap,
    )
    return {
        "gate2": gate2,
        "seed_average_losses": averaged,
        "m1_vs_b4": m1_vs_b4,
        "m1_vs_b3": m1_vs_b3,
        "b4_vs_b3": b4_vs_b3,
        "bootstrap": bootstrap,
        "aggregation": (
            "average B3/B4/M1 policy loss by dataset/series/origin "
            "before Gate2 bootstrap"
        ),
    }


def evaluate_gate4_seed1(
    seed_policy_losses: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, object]:
    """Apply two-seed loss averaging, sign, retention, and CI rules."""

    analysis = _seed_average_policy_analysis(
        seed_policy_losses,
        candidate_column=candidate_column,
        baseline_column=baseline_column,
        expected_seeds=(0, 1),
        draws=draws,
        seed=seed,
    )
    per_seed = analysis["per_seed"]
    seed0 = float(per_seed["0"]["macro_ri_percent"])
    seed1 = float(per_seed["1"]["macro_ri_percent"])
    if seed0 <= 0.0:
        raise ValueError("seed0 effect must be positive before seed1 evaluation")
    bootstrap = analysis["seed_average_bootstrap"]
    interval = list(bootstrap["macro"]["ci95_percent"])
    same_sign = np.sign(seed1) == np.sign(seed0)
    retention = abs(seed1) / abs(seed0)
    checks = {
        "same_sign": bool(same_sign),
        "effect_retention_at_least_0_70": retention >= 0.70,
        "seed_average_ci_lower_strictly_positive": interval[0] > 0.0,
    }
    robust_effect = checks["same_sign"] and checks[
        "effect_retention_at_least_0_70"
    ]
    passed = robust_effect and checks["seed_average_ci_lower_strictly_positive"]
    still_borderline = robust_effect and interval[0] <= 0.0 < interval[1]
    if passed:
        action, pass_state, allow_seed2 = "ACCEPT_TWO_SEED", True, False
    elif still_borderline:
        action, pass_state, allow_seed2 = "RUN_SEED2", None, True
    else:
        action, pass_state, allow_seed2 = (
            "RETRIEVAL_ROBUSTNESS_NO_GO",
            False,
            False,
        )
    return {
        "seed_macro_ri_percent": [seed0, seed1],
        "checks": checks,
        "effect_retention": float(retention),
        "seed_average_ci95_percent": interval,
        "seed_average_macro_ri_percent": float(
            bootstrap["macro"]["ri_percent"]
        ),
        "passed": pass_state,
        "allow_seed2": allow_seed2,
        "action": action,
        "aggregation": "average policy loss by series-origin before bootstrap",
    }


def evaluate_gate4_seed2(
    seed_policy_losses: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = ANALYSIS_SEED,
) -> dict[str, object]:
    """Make the three-seed decision after series-origin policy-loss averaging."""

    analysis = _seed_average_policy_analysis(
        seed_policy_losses,
        candidate_column=candidate_column,
        baseline_column=baseline_column,
        expected_seeds=(0, 1, 2),
        draws=draws,
        seed=seed,
    )
    per_seed = analysis["per_seed"]
    effects = [
        float(per_seed[str(model_seed)]["macro_ri_percent"])
        for model_seed in (0, 1, 2)
    ]
    if effects[0] <= 0.0:
        raise ValueError("seed0 effect must be positive before seed2 evaluation")
    bootstrap = analysis["seed_average_bootstrap"]
    interval = list(bootstrap["macro"]["ci95_percent"])
    seed_average_effect = float(bootstrap["macro"]["ri_percent"])
    reference_sign = np.sign(effects[0])
    opposite = sum(
        value != 0.0 and np.sign(value) != reference_sign for value in effects
    )
    checks = {
        "at_most_one_opposite_sign": opposite <= 1,
        "seed_average_effect_positive": seed_average_effect > 0.0,
        "seed_average_ci_lower_strictly_positive": interval[0] > 0.0,
    }
    passed = all(checks.values())
    return {
        "seed_macro_ri_percent": effects,
        "opposite_sign_count": int(opposite),
        "seed_average_ci95_percent": interval,
        "seed_average_macro_ri_percent": seed_average_effect,
        "checks": checks,
        "passed": passed,
        "action": "ACCEPT_THREE_SEED" if passed else "RETRIEVAL_ROBUSTNESS_NO_GO",
        "aggregation": "average policy loss by series-origin before bootstrap",
    }


def _required_gate(name: str, value: object) -> bool:
    if value is None:
        raise ValueError(f"{name} is required after all prior gates pass")
    return _boolean(name, value)


def decide_final_verdict(
    *,
    gate0_pass: bool,
    heterogeneous_gate_pass: bool | None = None,
    gate1a_pass: bool | None = None,
    gate1b_pass: bool | None = None,
    gate2_pass: bool | None = None,
    gate3_safety_pass: bool | None = None,
    gate3_control_pass: bool | None = None,
    gate4_pass: bool | None = None,
) -> str:
    """Return exactly one Section 40 verdict in failure-first order."""

    if not _boolean("gate0_pass", gate0_pass):
        heterogeneous = False if heterogeneous_gate_pass is None else _boolean(
            "heterogeneous_gate_pass", heterogeneous_gate_pass
        )
        return (
            "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE"
            if heterogeneous
            else "FULL_NO_GO"
        )
    if not _required_gate("gate1a_pass", gate1a_pass):
        return "TEMPORAL_RECURRENCE_NO_GO"
    if not _required_gate("gate1b_pass", gate1b_pass):
        return "ONLINE_MEMORY_NO_GO"
    if not _required_gate("gate2_pass", gate2_pass):
        return "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO"
    if not _required_gate("gate3_safety_pass", gate3_safety_pass):
        return "RETRIEVAL_UNSAFE_NO_GO"
    if not _required_gate("gate3_control_pass", gate3_control_pass):
        return "RETRIEVAL_SIGNAL_NOT_IDENTIFIED"
    if not _required_gate("gate4_pass", gate4_pass):
        return "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO"
    return "RETRIEVAL_MEMORY_GO"
