"""No-training Gate-0 diagnostic with existing TSB and SBA experts.

The diagnostic is deliberately lazy: callers should construct the factory and
let the full-run Gate-0 orchestrator call it only after the Point/Hurdle oracle
has failed.  Classical parameters are selected on a newly built validation
split.  The warm-up plus six forecast origins are generated, but only the six
evaluation origins are scored.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from numbers import Integral, Real

import numpy as np
import pandas as pd

from experiments.external_validity_screen.classical_benchmark import (
    ALPHA_GRID as SBA_ALPHA_GRID,
    croston_forecast,
    hand_check,
    select_alpha,
)
from experiments.om_factorization_killtest.evaluate import select_tsb
from experiments.om_factorization_killtest.models import tsb_forecast
from experiments.om_factorization_killtest.prereg import TSB_GRID

from .analysis import evaluate_gate0
from .data import build_external_split
from .evaluation import ALPHA_GRID
from .metrics import policy_scale_squared


DATASET_ORDER = ("m5", "favorita")
EXPERT_ORDER = ("point", "hurdle", "tsb", "sba")
PAIR_ORDER = tuple(combinations(EXPERT_ORDER, 2))
HORIZON = 28
LOOKBACK = 96
TRAIN_ORIGIN_STRIDE = 7
CHUNK_CASES = 4096
EXPECTED_SCHEDULES = {
    "m5": {
        "length": 1941,
        "model_train_end": 1717,
        "warmup_origin": 1745,
        "evaluation_origins": (1773, 1801, 1829, 1857, 1885, 1913),
    },
    "favorita": {
        "length": 1688,
        "model_train_end": 1464,
        "warmup_origin": 1492,
        "evaluation_origins": (1520, 1548, 1576, 1604, 1632, 1660),
    },
}
CASE_KEYS = ("dataset_id", "series_id", "origin")
CASE_COLUMNS = (
    *CASE_KEYS,
    "history",
    "point_forecast",
    "hurdle_forecast",
    "target",
    "target_mask",
    "policy_scale_squared",
)


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _array(
    name: str,
    value: object,
    width: int,
    *,
    boolean: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (width,):
        raise ValueError(f"{name} must contain exactly {width} values")
    if boolean:
        if not np.issubdtype(raw.dtype, np.bool_):
            raise ValueError(f"{name} must be Boolean")
        return raw.astype(bool, copy=True)
    try:
        result = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("diagnostic report cannot contain non-finite values")
        return result
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported diagnostic report value: {type(value).__name__}")


def _validated_population(
    population: Mapping[str, object], dataset: str
) -> tuple[Mapping[str, object], object, tuple[str, ...]]:
    if not isinstance(population, Mapping):
        raise TypeError(f"population {dataset} must be a mapping")
    missing = [key for key in ("data", "cfg") if key not in population]
    if missing:
        raise ValueError(f"population {dataset} is missing {missing}")
    data = population["data"]
    cfg = population["cfg"]
    if not isinstance(data, Mapping):
        raise TypeError(f"population {dataset} data must be a mapping")
    if str(data.get("name", "")) != dataset:
        raise ValueError(f"population {dataset} has a mismatched dataset identity")
    required_data = ("series_id", "y", "z", "available_from")
    missing_data = [key for key in required_data if key not in data]
    if missing_data:
        raise ValueError(f"population {dataset} data is missing {missing_data}")
    ids = np.asarray(data["series_id"]).astype(str)
    values = np.asarray(data["y"])
    if ids.ndim != 1 or ids.size == 0 or len(set(ids.tolist())) != ids.size:
        raise ValueError(f"population {dataset} series_id must be a unique vector")
    if values.shape != (ids.size, EXPECTED_SCHEDULES[dataset]["length"]):
        raise ValueError(f"population {dataset} y has the wrong shape")
    required_cfg = {
        "n_series": ids.size,
        "length": EXPECTED_SCHEDULES[dataset]["length"],
        "train_end": EXPECTED_SCHEDULES[dataset]["model_train_end"],
        "val_end": EXPECTED_SCHEDULES[dataset]["warmup_origin"],
        "horizon": HORIZON,
        "lookback": LOOKBACK,
    }
    for field, expected in required_cfg.items():
        if not hasattr(cfg, field) or int(getattr(cfg, field)) != int(expected):
            raise ValueError(
                f"population {dataset} cfg {field} must equal {int(expected)}"
            )
    return data, cfg, tuple(ids.tolist())


def _validated_schedule(
    schedule: Mapping[str, object], dataset: str
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if not isinstance(schedule, Mapping):
        raise TypeError(f"{dataset} schedule must be a mapping")
    expected = EXPECTED_SCHEDULES[dataset]
    warmup = _integer("warmup_origin", schedule.get("warmup_origin"))
    try:
        evaluations = tuple(
            _integer("evaluation_origin", value)
            for value in schedule.get("evaluation_origins", ())
        )
        all_origins = tuple(
            _integer("forecast_origin", value)
            for value in schedule.get("all_forecast_origins", ())
        )
    except TypeError as exc:
        raise TypeError(f"{dataset} schedule origins must be sequences") from exc
    expected_evaluations = expected["evaluation_origins"]
    expected_all = (expected["warmup_origin"], *expected_evaluations)
    if warmup != expected["warmup_origin"]:
        raise ValueError(f"{dataset} warmup origin drifted")
    if evaluations != expected_evaluations or len(evaluations) != 6:
        raise ValueError(f"{dataset} evaluation-origin coverage drifted")
    if all_origins != expected_all:
        raise ValueError(f"{dataset} seven-origin schedule drifted")
    scalar_fields = {
        "horizon": HORIZON,
        "lookback": LOOKBACK,
        "train_origin_stride": TRAIN_ORIGIN_STRIDE,
        "model_train_end": expected["model_train_end"],
    }
    for field, required in scalar_fields.items():
        if _integer(field, schedule.get(field)) != required:
            raise ValueError(f"{dataset} schedule {field} drifted")
    return warmup, evaluations, all_origins


def classical_expert_predictions(
    population: Mapping[str, object], schedule: Mapping[str, object]
) -> dict[str, object]:
    """Select existing TSB/SBA parameters on validation and forecast 7 origins."""

    if not isinstance(population, Mapping) or "data" not in population:
        raise TypeError("population must be a mapping containing data")
    raw_data = population["data"]
    if not isinstance(raw_data, Mapping):
        raise TypeError("population data must be a mapping")
    dataset = str(raw_data.get("name", ""))
    if dataset not in DATASET_ORDER:
        raise ValueError(f"unknown population dataset: {dataset}")
    data, cfg, series_ids = _validated_population(population, dataset)
    _, evaluations, all_origins = _validated_schedule(schedule, dataset)

    split = build_external_split(
        dict(data),
        cfg,
        train_origin_stride=TRAIN_ORIGIN_STRIDE,
        forecast_origins=np.asarray(all_origins, dtype=np.int32),
    )
    if tuple(np.asarray(split.test.origins, dtype=np.int64).tolist()) != all_origins:
        raise ValueError("fresh classical split did not preserve forecast origins")
    if int(split.test.n_series) != len(series_ids) or int(split.test.n_origins) != 7:
        raise ValueError("fresh classical split has wrong series/origin coverage")

    audit = hand_check()
    if not isinstance(audit, Mapping) or audit.get("passed") is not True:
        raise RuntimeError("repository SBA hand-check did not PASS")

    tsb_alpha, tsb_beta, tsb_validation_mse = select_tsb(split.validation)
    sba_alpha, sba_validation_mse = select_alpha(
        split.validation, HORIZON, "sba"
    )
    history = np.asarray(split.test.history, dtype=np.float64)
    tsb_probability, tsb_magnitude = tsb_forecast(
        history, HORIZON, float(tsb_alpha), float(tsb_beta)
    )
    tsb_mean = np.asarray(tsb_probability) * np.asarray(tsb_magnitude)
    sba_mean = croston_forecast(
        history, HORIZON, float(sba_alpha), "sba"
    )
    expected_shape = (len(series_ids) * 7, HORIZON)
    for name, forecast in (("tsb", tsb_mean), ("sba", sba_mean)):
        if np.asarray(forecast).shape != expected_shape:
            raise ValueError(f"repository {name} forecast has the wrong shape")
        if not bool(np.isfinite(forecast).all()):
            raise ValueError(f"repository {name} forecast must be finite")

    case_keys = tuple(
        (dataset, series_id, origin)
        for series_id in series_ids
        for origin in all_origins
    )
    return {
        "dataset": dataset,
        "case_keys": case_keys,
        "evaluation_origins": evaluations,
        "forecasts": {
            "tsb": np.asarray(tsb_mean, dtype=np.float64),
            "sba": np.asarray(sba_mean, dtype=np.float64),
        },
        "fresh_split": {
            "history": np.asarray(split.test.history, dtype=np.float64).copy(),
            "target": np.asarray(split.test.target, dtype=np.float64).copy(),
            "target_mask": np.asarray(split.test.target_mask).copy(),
            "canonical_train_scale": np.asarray(split.test.scale, dtype=np.float64).copy(),
        },
        "selected_parameters": {
            "tsb": {
                "alpha": float(tsb_alpha),
                "beta": float(tsb_beta),
                "validation_mse": float(tsb_validation_mse),
                "grid": {key: list(value) for key, value in TSB_GRID.items()},
                "selected_on": "validation_split_only",
                "selected_on_dataset": dataset,
                "implementation": (
                    "experiments.om_factorization_killtest.evaluate.select_tsb + "
                    "experiments.om_factorization_killtest.models.tsb_forecast"
                ),
            },
            "sba": {
                "alpha": float(sba_alpha),
                "validation_mse": float(sba_validation_mse),
                "grid": list(SBA_ALPHA_GRID),
                "selected_on": "validation_split_only",
                "selected_on_dataset": dataset,
                "implementation": (
                    "experiments.external_validity_screen.classical_benchmark."
                    "select_alpha/croston_forecast"
                ),
            },
        },
        "hand_check": dict(audit),
    }


def _validated_cases(
    output: Mapping[str, object],
    population: Mapping[str, object],
    classical: Mapping[str, object],
    dataset: str,
) -> dict[str, np.ndarray]:
    if not isinstance(output, Mapping):
        raise TypeError(f"dataset output {dataset} must be a mapping")
    if str(output.get("dataset", "")) != dataset:
        raise ValueError(f"dataset output {dataset} has a mismatched dataset identity")
    if "schedule" not in output or "cases" not in output:
        raise ValueError(f"dataset output {dataset} must contain schedule and cases")
    _, evaluations, _ = _validated_schedule(output["schedule"], dataset)
    cases = output["cases"]
    if not isinstance(cases, pd.DataFrame):
        raise TypeError(f"dataset output {dataset} cases must be a DataFrame")
    if not cases.columns.is_unique:
        raise ValueError(f"dataset output {dataset} cases has duplicate columns")
    missing = [column for column in CASE_COLUMNS if column not in cases]
    if missing:
        raise ValueError(f"dataset output {dataset} cases is missing {missing}")
    frame = cases.loc[:, CASE_COLUMNS].copy()
    if frame.empty or bool(frame.loc[:, CASE_KEYS].isna().any(axis=None)):
        raise ValueError(f"dataset output {dataset} case keys must be nonempty")
    if bool(frame.duplicated(list(CASE_KEYS), keep=False).any()):
        raise ValueError(f"dataset output {dataset} case keys must be unique")
    frame["dataset_id"] = frame["dataset_id"].astype(str)
    frame["series_id"] = frame["series_id"].astype(str)
    if set(frame["dataset_id"]) != {dataset}:
        raise ValueError(f"dataset output {dataset} cases crossed dataset identity")
    origins = pd.to_numeric(frame["origin"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not bool(np.isfinite(origins).all()) or not bool(
        np.equal(origins, np.floor(origins)).all()
    ):
        raise ValueError(f"dataset output {dataset} origins must be finite integers")
    frame["origin"] = origins.astype(np.int64)

    expected_keys = tuple(classical["case_keys"])
    indexed = frame.set_index(list(CASE_KEYS), verify_integrity=True)
    observed_keys = set(indexed.index.tolist())
    if observed_keys != set(expected_keys) or len(indexed) != len(expected_keys):
        raise ValueError(
            f"dataset output {dataset} lacks one-to-one seven-origin coverage"
        )
    indexed = indexed.reindex(pd.MultiIndex.from_tuples(expected_keys, names=CASE_KEYS))

    fresh = classical["fresh_split"]
    expected_history = np.asarray(fresh["history"], dtype=np.float64)
    expected_target = np.asarray(fresh["target"], dtype=np.float64)
    expected_mask = np.asarray(fresh["target_mask"])
    expected_canonical_scale = np.asarray(
        fresh["canonical_train_scale"], dtype=np.float64
    )
    n_cases = len(expected_keys)
    arrays: dict[str, list[np.ndarray]] = {
        "history": [],
        "point": [],
        "hurdle": [],
        "target": [],
        "mask": [],
    }
    scales: list[float] = []
    data, cfg, _ = _validated_population(population, dataset)
    ids = np.asarray(data["series_id"]).astype(str)
    expected_policy_scales = {
        series_id: policy_scale_squared(
            np.asarray(data["y"])[index], int(getattr(cfg, "train_end"))
        )
        for index, series_id in enumerate(ids)
    }
    for row_index, (key, row) in enumerate(indexed.iterrows()):
        history = _array("history", row.history, LOOKBACK)
        point = _array("point_forecast", row.point_forecast, HORIZON)
        hurdle = _array("hurdle_forecast", row.hurdle_forecast, HORIZON)
        target = _array("target", row.target, HORIZON)
        mask = _array("target_mask", row.target_mask, HORIZON, boolean=True)
        if not bool(mask.all()):
            raise ValueError("target_mask must be a full observed 28-step mask")
        scale = _finite("policy_scale_squared", row.policy_scale_squared, positive=True)
        series_id = str(key[1])
        if scale != expected_policy_scales[series_id]:
            raise ValueError("policy scale does not match the model-training prefix")
        if not np.array_equal(history, expected_history[row_index]):
            raise ValueError("case history does not match the fresh causal split")
        if not np.array_equal(target, expected_target[row_index]):
            raise ValueError("case target does not match the fresh split target")
        if not np.array_equal(mask, expected_mask[row_index]):
            raise ValueError("case target mask does not match the fresh split mask")
        arrays["history"].append(history)
        arrays["point"].append(point)
        arrays["hurdle"].append(hurdle)
        arrays["target"].append(target)
        arrays["mask"].append(mask)
        scales.append(scale)

    if "canonical_train_scale" in cases:
        ordered_scale = pd.to_numeric(
            cases.set_index(list(CASE_KEYS))
            .reindex(pd.MultiIndex.from_tuples(expected_keys, names=CASE_KEYS))[
                "canonical_train_scale"
            ],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        if not np.array_equal(ordered_scale, expected_canonical_scale):
            raise ValueError("canonical train scale does not match the fresh split")

    origins_array = np.asarray([int(key[2]) for key in expected_keys], dtype=np.int64)
    evaluation_mask = np.isin(origins_array, np.asarray(evaluations, dtype=np.int64))
    if int(evaluation_mask.sum()) * 7 != n_cases * 6:
        raise ValueError("evaluation-origin coverage must be exactly six of seven origins")
    return {
        "evaluation_mask": evaluation_mask,
        "target": np.stack(arrays["target"]),
        "point": np.stack(arrays["point"]),
        "hurdle": np.stack(arrays["hurdle"]),
        "scale": np.asarray(scales, dtype=np.float64),
    }


def _candidate_label(expert_a: str, expert_b: str, alpha: float) -> str:
    return f"{expert_a}|{expert_b}|{float(alpha)}"


def _candidate(candidate_index: int) -> dict[str, object]:
    per_pair = len(ALPHA_GRID)
    pair_index, alpha_index = divmod(candidate_index, per_pair)
    expert_a, expert_b = PAIR_ORDER[pair_index]
    return {
        "expert_a": expert_a,
        "expert_b": expert_b,
        "alpha": float(ALPHA_GRID[alpha_index]),
    }


def evaluate_pairwise_family(
    targets: np.ndarray,
    scales: np.ndarray,
    forecasts: Mapping[str, np.ndarray],
    *,
    chunk_cases: int = CHUNK_CASES,
) -> dict[str, object]:
    """Evaluate the fixed union of all two-expert convex line segments."""

    actual = np.asarray(targets, dtype=np.float64)
    denominator = np.asarray(scales, dtype=np.float64)
    if actual.ndim != 2 or actual.shape[0] == 0 or actual.shape[1] != HORIZON:
        raise ValueError("targets must be a nonempty case-by-28 array")
    if denominator.shape != (actual.shape[0],):
        raise ValueError("scales must contain one value per case")
    if not bool(np.isfinite(actual).all() and np.isfinite(denominator).all()):
        raise ValueError("targets and scales must be finite")
    if bool((denominator <= 0.0).any()):
        raise ValueError("scales must be positive")
    if not isinstance(forecasts, Mapping) or tuple(forecasts) != EXPERT_ORDER:
        raise ValueError(f"forecasts must use exact expert order {EXPERT_ORDER}")
    expert_arrays: dict[str, np.ndarray] = {}
    for expert in EXPERT_ORDER:
        values = np.asarray(forecasts[expert], dtype=np.float64)
        if values.shape != actual.shape or not bool(np.isfinite(values).all()):
            raise ValueError(f"{expert} forecasts must align and be finite")
        expert_arrays[expert] = values

    width = _integer("chunk_cases", chunk_cases, minimum=1)
    candidates = tuple(
        (expert_a, expert_b, float(alpha))
        for expert_a, expert_b in PAIR_ORDER
        for alpha in ALPHA_GRID
    )
    candidate_sums = np.zeros(len(candidates), dtype=np.float64)
    choice_counts = np.zeros(len(candidates), dtype=np.int64)
    origin_best_sum = 0.0
    alphas = np.asarray(ALPHA_GRID, dtype=np.float64)
    for start in range(0, actual.shape[0], width):
        stop = min(start + width, actual.shape[0])
        block_target = actual[start:stop]
        block_scale = denominator[start:stop, None]
        block_losses: list[np.ndarray] = []
        for expert_a, expert_b in PAIR_ORDER:
            first = expert_arrays[expert_a][start:stop]
            delta = expert_arrays[expert_b][start:stop] - first
            residual = block_target - first
            residual_squared = np.mean(np.square(residual), axis=1)[:, None]
            residual_delta = np.mean(residual * delta, axis=1)[:, None]
            delta_squared = np.mean(np.square(delta), axis=1)[:, None]
            pair_losses = (
                residual_squared
                - 2.0 * residual_delta * alphas[None, :]
                + delta_squared * np.square(alphas[None, :])
            ) / block_scale
            pair_losses = np.maximum(pair_losses, 0.0)
            block_losses.append(pair_losses)
        loss_matrix = np.concatenate(block_losses, axis=1)
        if not bool(np.isfinite(loss_matrix).all()):
            raise ValueError("pairwise family produced non-finite losses")
        candidate_sums += np.sum(loss_matrix, axis=0, dtype=np.float64)
        best_indices = np.argmin(loss_matrix, axis=1)
        best_losses = loss_matrix[np.arange(stop - start), best_indices]
        origin_best_sum += float(np.sum(best_losses, dtype=np.float64))
        choice_counts += np.bincount(best_indices, minlength=len(candidates))

    means = candidate_sums / actual.shape[0]
    global_index = int(np.argmin(means))
    global_loss = float(means[global_index])
    origin_loss = origin_best_sum / actual.shape[0]
    if global_loss <= 0.0:
        raise ValueError("global static loss must be positive for relative gain")
    selection_counts = {
        _candidate_label(*candidates[index]): int(count)
        for index, count in enumerate(choice_counts)
        if count > 0
    }
    return {
        "candidate_count": len(candidates),
        "global_static": {
            "candidate": _candidate(global_index),
            "loss": global_loss,
        },
        "origin_oracle": {
            "loss": float(origin_loss),
            "selection_counts": selection_counts,
        },
        "gain_percent": float(100.0 * (1.0 - origin_loss / global_loss)),
    }


def evaluate_heterogeneous_dataset(
    population: Mapping[str, object],
    dataset_output: Mapping[str, object],
    dataset: str,
) -> dict[str, object]:
    """Evaluate one dataset so callers need not retain both raw panels in RAM."""

    if dataset not in DATASET_ORDER:
        raise ValueError(f"unsupported dataset: {dataset}")
    if not isinstance(dataset_output, Mapping) or "schedule" not in dataset_output:
        raise ValueError(f"dataset output {dataset} lacks its schedule")
    classical = classical_expert_predictions(population, dataset_output["schedule"])
    validated = _validated_cases(dataset_output, population, classical, dataset)
    keep = validated["evaluation_mask"]
    forecasts = {
        "point": validated["point"][keep],
        "hurdle": validated["hurdle"][keep],
        "tsb": np.asarray(classical["forecasts"]["tsb"])[keep],
        "sba": np.asarray(classical["forecasts"]["sba"])[keep],
    }
    family = evaluate_pairwise_family(
        validated["target"][keep], validated["scale"][keep], forecasts
    )
    gain = float(family["gain_percent"])
    n_cases = int(np.asarray(keep).sum())
    result = {
        "dataset": dataset,
        "gain_percent": gain,
        "details": {
            "n_series": n_cases // 6,
            "n_forecast_origins_generated": 7,
            "n_evaluation_origins": 6,
            "n_evaluation_cases": n_cases,
            "selected_parameters": classical["selected_parameters"],
            "global_static": family["global_static"],
            "origin_oracle": family["origin_oracle"],
            "gain_percent": gain,
            "integrity_checks": {
                "fresh_split_rebuilt": True,
                "validation_only_parameter_selection": True,
                "one_to_one_case_keys": True,
                "full_28_step_target_masks": True,
                "warmup_generated_but_not_scored": True,
                "dataset_identity_isolated": True,
                "fresh_history_target_mask_match": True,
            },
        },
        "hand_check": classical["hand_check"],
    }
    return _json_safe(result)  # type: ignore[return-value]


def assemble_heterogeneous_diagnostic(
    dataset_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Assemble the frozen two-dataset diagnostic from isolated evaluations."""

    if not isinstance(dataset_results, Mapping) or set(dataset_results) != set(
        DATASET_ORDER
    ):
        raise ValueError("dataset_results must contain exactly m5 and favorita")
    datasets: dict[str, object] = {}
    gains: dict[str, float] = {}
    shared_hand_check: object | None = None
    for dataset in DATASET_ORDER:
        item = dataset_results[dataset]
        if not isinstance(item, Mapping) or item.get("dataset") != dataset:
            raise ValueError(f"heterogeneous result identity mismatch for {dataset}")
        details = item.get("details")
        if not isinstance(details, Mapping):
            raise ValueError(f"heterogeneous result details missing for {dataset}")
        gain = _finite(f"{dataset} gain_percent", item.get("gain_percent"))
        if _finite(f"{dataset} details gain_percent", details.get("gain_percent")) != gain:
            raise ValueError(f"heterogeneous gain mismatch for {dataset}")
        hand = item.get("hand_check")
        if not isinstance(hand, Mapping) or hand.get("passed") is not True:
            raise ValueError(f"repository SBA hand-check failed for {dataset}")
        gains[dataset] = gain
        datasets[dataset] = dict(details)
        if shared_hand_check is None:
            shared_hand_check = hand
        elif shared_hand_check != hand:
            raise AssertionError("repository SBA hand-check changed between datasets")

    macro = float(np.mean([gains[dataset] for dataset in DATASET_ORDER]))
    report = {
        "status": "AVAILABLE",
        "macro_gain_percent": macro,
        "gains_percent": gains,
        "passed": macro >= 2.0,
        "family": {
            "experts": list(EXPERT_ORDER),
            "pair_order": [
                {"expert_a": first, "expert_b": second}
                for first, second in PAIR_ORDER
            ],
            "alpha_grid": [float(value) for value in ALPHA_GRID],
            "geometry": "pairwise_segment_union",
            "is_full_simplex": False,
            "candidate_count": len(PAIR_ORDER) * len(ALPHA_GRID),
            "tie_break": "first candidate in frozen pair-major then alpha-grid order",
        },
        "datasets": datasets,
        "hand_check": shared_hand_check,
        "provenance": {
            "new_models_trained": False,
            "new_classical_method_implemented": False,
            "split_builder": (
                "experiments.ph_online_memory_gono_v1.data.build_external_split"
            ),
            "selection_data": "each dataset's newly rebuilt validation split only",
            "forecast_schedule": "warmup plus six frozen evaluation origins",
            "scored_schedule": "six frozen evaluation origins only",
            "family_scope": (
                "pairwise convex extension of Point/Hurdle/TSB/SBA; not a full simplex"
            ),
        },
    }
    return _json_safe(report)  # type: ignore[return-value]


def run_heterogeneous_diagnostic(
    populations: Mapping[str, Mapping[str, object]],
    dataset_outputs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Run the secondary TSB/SBA oracle diagnostic without fitting a model."""

    if not isinstance(populations, Mapping) or set(populations) != set(DATASET_ORDER):
        raise ValueError("populations must contain exactly m5 and favorita")
    if not isinstance(dataset_outputs, Mapping) or set(dataset_outputs) != set(
        DATASET_ORDER
    ):
        raise ValueError("dataset_outputs must contain exactly m5 and favorita")
    isolated = {
        dataset: evaluate_heterogeneous_dataset(
            populations[dataset], dataset_outputs[dataset], dataset
        )
        for dataset in DATASET_ORDER
    }
    return assemble_heterogeneous_diagnostic(isolated)


def make_heterogeneous_diagnostic_factory(
    populations: Mapping[str, Mapping[str, object]],
) -> Callable[
    [Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]],
    Mapping[str, object],
]:
    """Bind raw populations and enforce lazy execution after Gate-0 failure."""

    if not isinstance(populations, Mapping) or set(populations) != set(DATASET_ORDER):
        raise ValueError("populations must contain exactly m5 and favorita")

    def diagnostic_factory(
        point_hurdle_ladders: Mapping[str, Mapping[str, object]],
        dataset_outputs: Mapping[str, Mapping[str, object]],
    ) -> Mapping[str, object]:
        initial = evaluate_gate0(point_hurdle_ladders)
        if bool(initial["passed"]):
            raise ValueError(
                "heterogeneous diagnostic requires Point/Hurdle Gate 0 to have failed"
            )
        return run_heterogeneous_diagnostic(populations, dataset_outputs)

    return diagnostic_factory


__all__ = [
    "ALPHA_GRID",
    "EXPERT_ORDER",
    "PAIR_ORDER",
    "assemble_heterogeneous_diagnostic",
    "classical_expert_predictions",
    "evaluate_heterogeneous_dataset",
    "evaluate_pairwise_family",
    "make_heterogeneous_diagnostic_factory",
    "run_heterogeneous_diagnostic",
]
