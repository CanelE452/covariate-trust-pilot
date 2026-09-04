"""Deterministic, no-write reporting for PH-ONLINE-MEMORY-GONO-v1.

The execution engine intentionally returns pandas frames alongside a JSON-safe
report.  This module is the narrow presentation boundary: it converts that
in-memory result into the preregistered Tables A--G and renders either a
scientific terminal status or the earlier runtime-approval stop.  All public
functions are pure and perform no filesystem writes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from .analysis import FINAL_VERDICT_TOKENS, decide_final_verdict


EXPERIMENT = "PH-ONLINE-MEMORY-GONO-v1"
DATASET_ORDER = ("m5", "favorita")
TABLE_KEYS = tuple(f"table_{letter}" for letter in "abcdefg")
GATE_ORDER = (
    "GATE0",
    "GATE1A",
    "GATE1B",
    "GATE2",
    "GATE3_SAFETY",
    "GATE3_CONTROL",
    "GATE4",
)

_GATE_REPORT_KEYS = {
    "GATE0": "gate0",
    "GATE1A": "gate1a",
    "GATE1B": "gate1b",
    "GATE2": "gate2",
    "GATE3_SAFETY": "gate3_safety",
    "GATE3_CONTROL": "gate3_control",
    "GATE4": "gate4",
}
_STOP_POSITIONS = {
    "GATE0": 0,
    "GATE1A": 1,
    "GATE1B": 2,
    "GATE2": 3,
    "GATE3_SAFETY": 4,
    "GATE3_CONTROL": 5,
    "GATE4_SEED0": 6,
    "GATE4_SEED1": 6,
    "GATE4_SEED2": 6,
    "GATE4": 6,
}

_TABLE_COLUMNS = {
    "table_c": (
        "dataset",
        "lag1_spearman",
        "ci95",
        "shuffled_spearman",
        "real_minus_shuffled",
        "gate_1a",
    ),
    "table_d": (
        "target",
        "source",
        "b3_loss",
        "b4_loss",
        "m1_loss",
        "ri_b4_vs_b3_percent",
        "ri_m1_vs_b3_percent",
        "ri_m1_vs_b4_percent",
        "ci95_percent",
    ),
    "table_e": (
        "target",
        "origin",
        "ri_m1_vs_b3_percent",
        "ri_m1_vs_b4_percent",
        "is_worst_origin",
    ),
    "table_f": (
        "target",
        "real_retrieval_ri_percent",
        "shuffled_value_ri_percent",
        "random_neighbor_ri_percent",
        "control_over_real_ratio",
    ),
}

_GATE_THRESHOLDS = {
    "GATE0": {
        "macro_origin_convex_oracle_gain_percent": ">= 2.0",
        "each_dataset_gain_percent": ">= 1.0",
    },
    "GATE1A": {
        "each_dataset_lag1_spearman": "> 0.10",
        "each_dataset_real_minus_shuffled": "> 0.05",
    },
    "GATE1B": {"each_transfer_b4_vs_b3_percent": ">= 0.30"},
    "GATE2": {
        "macro_m1_vs_b4_percent": ">= 0.20",
        "each_transfer_m1_vs_b4_percent": ">= -0.10",
        "macro_m1_vs_b3_percent": ">= 0.70",
        "each_transfer_m1_vs_b3_percent": ">= 0.30",
        "macro_ci95_lower_percent": "> 0",
        "at_least_one_dataset_ci95_lower_percent": "> 0",
    },
    "GATE3_SAFETY": {
        "worst_origin_m1_vs_b3_percent": ">= -0.50",
        "each_dataset_q95_m1_over_b4": "<= 1.01",
    },
    "GATE3_CONTROL": {
        "real_macro_ri_percent": "> 0",
        "shuffled_over_real": "<= 0.25",
        "random_over_real": "<= 0.50",
    },
    "GATE4": {
        "rule": (
            "frozen conditional seed-0/seed-1/seed-2 sign, retention, "
            "and seed-average series-cluster CI rules"
        )
    },
}

_GATE_INTERPRETATIONS = {
    "GATE0": (
        "Point/Hurdle convex oracle opportunity is sufficient",
        "Point/Hurdle convex oracle opportunity is insufficient",
    ),
    "GATE1A": (
        "past expert advantage recurs temporally in both datasets",
        "temporal recurrence was not identified in both datasets",
    ),
    "GATE1B": (
        "simple online combination beats source-static in both transfers",
        "simple online combination does not beat source-static in both transfers",
    ),
    "GATE2": (
        "retrieval adds the preregistered incremental and absolute value",
        "retrieval does not meet the preregistered value thresholds",
    ),
    "GATE3_SAFETY": (
        "retrieval meets origin and series-tail safety thresholds",
        "retrieval fails an origin or series-tail safety threshold",
    ),
    "GATE3_CONTROL": (
        "real retrieval is separated from shuffled and random controls",
        "the retrieval signal is not separated from controls",
    ),
    "GATE4": (
        "the required model-seed robustness rule passes",
        "the required model-seed robustness rule fails",
    ),
}

_VERDICT_INTERPRETATIONS = {
    "FULL_NO_GO": "현재 expert set에서는 routing/memory 연구 가치가 부족하다.",
    "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE": (
        "memory보다 expert set 다양성 부족이 우선 후보이다."
    ),
    "TEMPORAL_RECURRENCE_NO_GO": (
        "상보성은 있으나 과거 expert loss가 미래 advantage를 설명하지 못했다."
    ),
    "ONLINE_MEMORY_NO_GO": (
        "B4 simple online combination이 source-static을 양방향에서 이기지 못했다."
    ),
    "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO": (
        "simple online combination을 넘는 retrieval의 추가 가치는 확인되지 않았다."
    ),
    "RETRIEVAL_UNSAFE_NO_GO": "retrieval 평균 효과와 별개로 안전성 기준을 통과하지 못했다.",
    "RETRIEVAL_SIGNAL_NOT_IDENTIFIED": (
        "shuffled/random control과 구분되는 retrieval 신호가 확인되지 않았다."
    ),
    "RETRIEVAL_MEMORY_GO": (
        "모든 필수 gate를 통과해 untouched external confirmatory study 설계 가치가 있다."
    ),
}

_NEXT_ACTIONS = {
    "RETRIEVAL_MEMORY_GO": "untouched external confirmatory study를 새 계획으로 설계",
    "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO": (
        "retrieval 개발 중단, simple online combination만 유지"
    ),
    "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE": (
        "Point/Hurdle routing 중단, stronger heterogeneous expert screen 설계"
    ),
    "FULL_NO_GO": "이 연구축에서 method 개발 종료",
    "TEMPORAL_RECURRENCE_NO_GO": "temporal memory mechanism 개발 종료",
    "ONLINE_MEMORY_NO_GO": "online memory mechanism 개발 종료",
    "RETRIEVAL_UNSAFE_NO_GO": "안전성 실패로 retrieval 개발 종료",
    "RETRIEVAL_SIGNAL_NOT_IDENTIFIED": "control과 구분되지 않는 retrieval 개발 종료",
}


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _safe(value: object) -> object:
    """Return a strict JSON-compatible copy without stringifying unknown data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite("report value", value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, pd.Series):
        return _safe(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_safe(item) for item in value]
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _seed_zero(result: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    root_report = _mapping("result.report", result.get("report"))
    seed_results = result.get("seed_results")
    if seed_results is None:
        return result, root_report
    seeds = _mapping("result.seed_results", seed_results)
    seed0 = seeds.get(0, seeds.get("0"))
    seed0_result = _mapping("result.seed_results[0]", seed0)
    seed0_report = _mapping("result.seed_results[0].report", seed0_result.get("report"))
    return seed0_result, seed0_report


def _all_seed_results(result: Mapping[str, object]) -> list[tuple[int, Mapping[str, object]]]:
    raw = result.get("seed_results")
    if raw is None:
        seed0, _ = _seed_zero(result)
        return [(0, seed0)]
    seeds = _mapping("result.seed_results", raw)
    normalized: list[tuple[int, Mapping[str, object]]] = []
    for key, value in seeds.items():
        try:
            seed = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed result keys must be integer-like") from exc
        normalized.append((seed, _mapping(f"seed result {seed}", value)))
    normalized.sort(key=lambda item: item[0])
    if not normalized or normalized[0][0] != 0:
        raise ValueError("full result must contain model seed 0")
    return normalized


def _stop_marker(seed_report: Mapping[str, object]) -> str:
    stopped_after = seed_report.get("stopped_after")
    if not isinstance(stopped_after, str) or stopped_after not in _STOP_POSITIONS:
        raise ValueError("seed-0 report lacks a recognized stopped_after gate")
    return f"NOT_RUN_AFTER_{stopped_after}"


def _failure_first_position(seed_report: Mapping[str, object]) -> int:
    stopped_after = seed_report.get("stopped_after")
    if not isinstance(stopped_after, str) or stopped_after not in _STOP_POSITIONS:
        raise ValueError("seed-0 report lacks a recognized stopped_after gate")
    return _STOP_POSITIONS[stopped_after]


def _gate_result_for_position(
    root_report: Mapping[str, object],
    seed_report: Mapping[str, object],
    gate: str,
) -> object:
    if gate == "GATE2" and root_report is not seed_report:
        averaged_gate2 = root_report.get("averaged_gate2")
        if averaged_gate2 is not None:
            return averaged_gate2
    if gate == "GATE4" and root_report is not seed_report:
        robustness = root_report.get("robustness")
        if robustness is not None:
            return robustness
    return seed_report.get(_GATE_REPORT_KEYS[gate])


def _gate2_deferral_veto(seed_report: Mapping[str, object]) -> str | None:
    raw = seed_report.get("gate2")
    if not isinstance(raw, Mapping) or raw.get("decision_state") != "FINAL_FAIL":
        return None
    veto = raw.get("deferral_withdrawn_by")
    if veto is None:
        return None
    if veto not in {"GATE3_SAFETY", "GATE3_CONTROL"}:
        raise ValueError("Gate2 deferral veto is not recognized")
    if raw.get("passed") is not False or seed_report.get("stopped_after") != veto:
        raise ValueError("Gate2 deferral veto disagrees with the failure stop")
    safety = seed_report.get("gate3_safety")
    if not isinstance(safety, Mapping):
        raise ValueError("Gate2 deferral veto lacks its safety observation")
    if veto == "GATE3_SAFETY":
        if safety.get("passed") is not False:
            raise ValueError("Gate2 safety deferral veto must have failed safety")
        if seed_report.get("gate3_control") is not None or seed_report.get(
            "controls"
        ) is not None:
            raise ValueError("Gate2 safety deferral veto cannot contain controls")
    else:
        control = seed_report.get("gate3_control")
        if safety.get("passed") is not True or not isinstance(control, Mapping):
            raise ValueError("Gate2 control deferral veto requires passed safety")
        if control.get("passed") is not False:
            raise ValueError("Gate2 control deferral veto must have failed control")
    return str(veto)


def _is_gate4_eligibility_observation(
    seed_report: Mapping[str, object], gate: str
) -> bool:
    veto = _gate2_deferral_veto(seed_report)
    if veto == "GATE3_SAFETY":
        return gate == "GATE3_SAFETY"
    if veto == "GATE3_CONTROL":
        return gate in {"GATE3_SAFETY", "GATE3_CONTROL"}
    return False


def _validate_failure_first_payload(
    result: Mapping[str, object],
    root_report: Mapping[str, object],
    seed0: Mapping[str, object],
    seed_report: Mapping[str, object],
) -> None:
    """Reject downstream observations after the declared failure-first stop."""

    stop_position = _failure_first_position(seed_report)
    deferral_veto = _gate2_deferral_veto(seed_report)
    report_positions = {
        "gate1a": 1,
        "directions": 2,
        "b4_vs_b3": 2,
        "gate1b": 2,
        "m1_vs_b3": 3,
        "m1_vs_b4": 3,
        "bootstrap": 3,
        "gate2": 3,
        "gate3_safety": 4,
        "controls": 5,
        "gate3_control": 5,
        "gate4": 6,
    }
    for key, position in report_positions.items():
        if position <= stop_position or key not in seed_report:
            continue
        eligibility_keys = {"gate3_safety"}
        if deferral_veto == "GATE3_CONTROL":
            eligibility_keys.update({"controls", "gate3_control"})
        if key in eligibility_keys:
            continue
        value = seed_report[key]
        if key == "directions" and isinstance(value, Mapping) and not value:
            continue
        if value is not None:
            raise ValueError(
                f"report contains downstream {key} after failure-first stop"
            )

    tables = _mapping("seed0.tables", seed0.get("tables", {}))
    table_positions = {
        "target_b3_b4_losses": 2,
        "target_policy_losses": 3,
        "c0_losses": 5,
        "c1_losses": 5,
    }
    for key, position in table_positions.items():
        if deferral_veto == "GATE3_CONTROL" and key in {"c0_losses", "c1_losses"}:
            continue
        if position > stop_position and key in tables:
            raise ValueError(
                f"tables contain downstream {key} after failure-first stop"
            )

    all_seeds = _all_seed_results(result)
    is_multiseed = len(all_seeds) > 1
    if is_multiseed:
        if root_report.get("averaged_gate2") is None:
            raise ValueError("multiseed result lacks the final averaged Gate2")
        seed_average_analysis = _mapping(
            "seed-average Gate2 analysis",
            root_report.get("seed_average_gate2_analysis"),
        )
        if seed_average_analysis.get("gate2") != root_report.get("averaged_gate2"):
            raise ValueError("averaged Gate2 disagrees with its analysis binding")
        seed0_gate2 = _mapping("seed-0 Gate2", seed_report.get("gate2"))
        seed0_gate4 = _mapping("seed-0 Gate4", seed_report.get("gate4"))
        if seed0_gate2.get("decision_state") != "PENDING_GATE4":
            raise ValueError("additional seeds require seed-0 Gate2 PENDING_GATE4")
        if not (
            seed0_gate4.get("passed") is None
            and seed0_gate4.get("action") == "RUN_SEED1"
        ):
            raise ValueError("additional seeds require the seed-0 Gate4 request")
        root_tables = _mapping("result.tables", result.get("tables", {}))
        if "seed_average_policy_losses" not in root_tables:
            raise ValueError("multiseed result lacks seed-average policy losses")
    if stop_position < 6 and len(all_seeds) != 1:
        raise ValueError("additional model seeds exist before the Gate-4 decision")
    if (
        stop_position < 6
        and root_report is not seed_report
        and root_report.get("robustness") is not None
    ):
        raise ValueError("robustness payload exists before Gate 4")

    for position, gate in enumerate(GATE_ORDER[: stop_position + 1]):
        raw = _gate_result_for_position(root_report, seed_report, gate)
        if raw is None:
            raise ValueError(f"report is missing executed {gate}")
        gate_result = _mapping(f"{gate} result", raw)
        passed = gate_result.get("passed")
        if position < stop_position:
            pending_seed0_gate2 = (
                gate == "GATE2"
                and root_report is seed_report
                and gate_result.get("decision_state") == "PENDING_GATE4"
                and passed in {True, False}
            )
            resolved_multiseed_gate2 = (
                gate == "GATE2"
                and is_multiseed
                and root_report is not seed_report
                and passed in {True, False}
            )
            withdrawn_seed0_gate2 = (
                gate == "GATE2"
                and root_report is seed_report
                and gate_result.get("decision_state") == "FINAL_FAIL"
                and gate_result.get("deferral_withdrawn_by")
                in {"GATE3_SAFETY", "GATE3_CONTROL"}
                and passed is False
            )
            if passed is not True and not (
                pending_seed0_gate2
                or resolved_multiseed_gate2
                or withdrawn_seed0_gate2
            ):
                raise ValueError(f"{gate} must pass before the declared later stop")
        elif position < 6:
            if passed is not False:
                raise ValueError(f"declared stop {gate} must be a failed gate")
        elif root_report.get("terminal") is True:
            if not isinstance(passed, (bool, np.bool_)):
                raise ValueError("terminal Gate 4 must contain a boolean decision")
        elif not (
            passed is None and gate_result.get("action") in {"RUN_SEED1", "RUN_SEED2"}
        ):
            raise ValueError("nonterminal Gate 4 must request a conditional seed")

    gate_order = seed_report.get("gate_order")
    if gate_order is not None:
        expected = list(GATE_ORDER[: stop_position + 1])
        if expected and expected[-1] == "GATE4":
            expected[-1] = "GATE4_SEED0"
        if list(gate_order) != expected:
            raise ValueError("gate_order disagrees with the failure-first stop")


def _sentinel(table: str, seed_report: Mapping[str, object]) -> list[dict[str, object]]:
    row = {column: None for column in _TABLE_COLUMNS[table]}
    row["status"] = _stop_marker(seed_report)
    return [row]


def _frame(name: str, value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if value.empty:
        raise ValueError(f"{name} must be nonempty")
    return value.copy()


def _direction_by_target(report: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = report.get("directions", {})
    directions = _mapping("report.directions", raw)
    result: dict[str, Mapping[str, object]] = {}
    for label in sorted(directions, key=str):
        direction = _mapping(f"report.directions[{label!r}]", directions[label])
        target = direction.get("target_dataset")
        if target in DATASET_ORDER:
            if str(target) in result:
                raise ValueError(f"duplicate transfer direction for target {target}")
            result[str(target)] = direction
    return result


def _table_a(result: Mapping[str, object], seed0: Mapping[str, object]) -> list[dict[str, object]]:
    seed0_outputs = _mapping("seed0.dataset_outputs", seed0.get("dataset_outputs"))
    if set(seed0_outputs) != set(DATASET_ORDER):
        raise ValueError("seed-0 dataset outputs must contain exactly m5 and favorita")
    all_seeds = _all_seed_results(result)
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        output = _mapping(f"seed0 output {dataset}", seed0_outputs[dataset])
        schedule = _mapping(f"{dataset} schedule", output.get("schedule"))
        manifest = _mapping(f"{dataset} population manifest", output.get("population_manifest"))
        provenance = _mapping(f"{dataset} provenance", output.get("provenance"))
        point = _mapping(f"{dataset} point provenance", provenance.get("point"))
        hurdle = _mapping(f"{dataset} hurdle provenance", provenance.get("hurdle"))
        train_end = _integer("model_train_end", schedule.get("model_train_end"))
        warmup = _integer("warmup_origin", schedule.get("warmup_origin"))
        origins_raw = schedule.get("evaluation_origins")
        if not isinstance(origins_raw, Sequence) or isinstance(origins_raw, (str, bytes)):
            raise TypeError("evaluation_origins must be a sequence")
        origins = [_integer("evaluation origin", value) for value in origins_raw]
        if len(origins) != 6:
            raise ValueError("Table A requires exactly six evaluation origins")

        runtimes: dict[str, dict[str, float]] = {}
        end_to_end_runtimes: dict[str, dict[str, float]] = {}
        total_runtime = 0.0
        total_end_to_end_runtime = 0.0
        for model_seed, seed_result in all_seeds:
            outputs = _mapping(
                f"seed {model_seed} dataset_outputs", seed_result.get("dataset_outputs")
            )
            seed_output = _mapping(
                f"seed {model_seed} output {dataset}", outputs.get(dataset)
            )
            seed_provenance = _mapping(
                f"seed {model_seed} {dataset} provenance",
                seed_output.get("provenance"),
            )
            arm_times: dict[str, float] = {}
            arm_end_to_end_times: dict[str, float] = {}
            for arm in ("point", "hurdle"):
                arm_provenance = _mapping(
                    f"seed {model_seed} {dataset} {arm} provenance",
                    seed_provenance.get(arm),
                )
                if arm_provenance.get("execution_device_type") != "cuda":
                    raise ValueError(
                        f"seed {model_seed} {dataset} {arm} was not executed on CUDA"
                    )
                seconds = _finite(
                    f"seed {model_seed} {dataset} {arm} train_seconds",
                    arm_provenance.get("train_seconds"),
                )
                arm_times[arm] = seconds
                total_runtime += seconds
                end_to_end_seconds = _finite(
                    f"seed {model_seed} {dataset} {arm} end_to_end_wall_seconds",
                    arm_provenance.get("end_to_end_wall_seconds"),
                )
                arm_end_to_end_times[arm] = end_to_end_seconds
                total_end_to_end_runtime += end_to_end_seconds
            runtimes[str(model_seed)] = arm_times
            end_to_end_runtimes[str(model_seed)] = arm_end_to_end_times

        rows.append(
            {
                "dataset": dataset,
                "eligible_n": _integer(
                    f"{dataset} eligible_independent",
                    manifest.get("eligible_independent"),
                ),
                "train_end": train_end,
                "validation_interval": [train_end, warmup],
                "interval_encoding": "half_open",
                "warmup_origin": warmup,
                "evaluation_origins": origins,
                "point_params": {
                    "model_id": point.get("model_id"),
                    "n_parameters": _integer(
                        f"{dataset} point n_parameters", point.get("n_parameters")
                    ),
                },
                "hurdle_params": {
                    "model_id": hurdle.get("model_id"),
                    "n_parameters": _integer(
                        f"{dataset} hurdle n_parameters", hurdle.get("n_parameters")
                    ),
                },
                "executed_model_seeds": [seed for seed, _ in all_seeds],
                "training_runtime_seconds": total_runtime,
                "training_runtime_by_seed_seconds": runtimes,
                "actual_synchronized_end_to_end_wall_seconds": (
                    total_end_to_end_runtime
                ),
                "actual_synchronized_end_to_end_wall_by_seed_seconds": (
                    end_to_end_runtimes
                ),
                "execution_device_type": "cuda",
                "status": "OBSERVED",
            }
        )
    return rows


def _table_b(seed_report: Mapping[str, object]) -> list[dict[str, object]]:
    ladders = _mapping("report.oracle_ladder", seed_report.get("oracle_ladder"))
    if set(ladders) != set(DATASET_ORDER):
        raise ValueError("oracle ladder must contain exactly m5 and favorita")
    directions = _direction_by_target(seed_report)
    marker = _stop_marker(seed_report)
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        ladder = _mapping(f"oracle ladder {dataset}", ladders[dataset])
        families = _mapping(f"oracle ladder {dataset}.families", ladder.get("families"))
        hard = _mapping(f"oracle ladder {dataset}.families.hard", families.get("hard"))
        convex = _mapping(
            f"oracle ladder {dataset}.families.convex", families.get("convex")
        )
        direction = directions.get(dataset)
        if direction is None:
            source = None
            source_alpha = None
            source_status = marker
        else:
            source = direction.get("source_dataset")
            b3 = _mapping(f"direction to {dataset}.b3", direction.get("b3"))
            source_alpha = _finite(f"direction to {dataset} B3 alpha", b3.get("alpha"))
            source_status = "OBSERVED"
        rows.append(
            {
                "dataset": dataset,
                "always_point_loss": _finite(
                    f"{dataset} always point", ladder.get("always_point_loss")
                ),
                "always_hurdle_loss": _finite(
                    f"{dataset} always hurdle", ladder.get("always_hurdle_loss")
                ),
                "half_half_loss": _finite(
                    f"{dataset} half-half", ladder.get("half_half_loss")
                ),
                "source_dataset": source,
                "source_static_alpha": source_alpha,
                "source_static_status": source_status,
                "target_oracle_static_alpha": _finite(
                    f"{dataset} target oracle static alpha",
                    convex.get("global_static_alpha"),
                ),
                "target_oracle_static_loss": _finite(
                    f"{dataset} target oracle static loss",
                    convex.get("global_static_loss"),
                ),
                "origin_hard_oracle_loss": _finite(
                    f"{dataset} hard origin oracle", hard.get("origin_oracle_loss")
                ),
                "origin_convex_oracle_loss": _finite(
                    f"{dataset} convex origin oracle",
                    convex.get("origin_oracle_loss"),
                ),
                "status": "OBSERVED",
            }
        )
    return rows


def _table_c(seed_report: Mapping[str, object]) -> list[dict[str, object]]:
    raw = seed_report.get("gate1a")
    if raw is None:
        return _sentinel("table_c", seed_report)
    gate = _mapping("report.gate1a", raw)
    details = _mapping("report.gate1a.datasets", gate.get("datasets"))
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        item = _mapping(f"gate1a dataset {dataset}", details.get(dataset))
        ci = item.get("rho_ci95")
        rows.append(
            {
                "dataset": dataset,
                "lag1_spearman": _safe(item.get("real_rho")),
                "ci95": _safe(ci),
                "shuffled_spearman": _safe(item.get("shuffled_rho")),
                "real_minus_shuffled": _safe(
                    item.get("real_minus_shuffled_rho")
                ),
                "gate_1a": "PASS" if item.get("passed") is True else "FAIL",
                "status": "OBSERVED",
            }
        )
    return rows


def _comparison_ri(
    seed_report: Mapping[str, object], report_key: str, dataset: str
) -> float | None:
    raw = seed_report.get(report_key)
    if raw is None:
        return None
    comparison = _mapping(f"report.{report_key}", raw)
    directions = _mapping(f"report.{report_key}.directions", comparison.get("directions"))
    direction = _mapping(f"report.{report_key}.directions[{dataset}]", directions.get(dataset))
    return _finite(f"{report_key} {dataset} ri_percent", direction.get("ri_percent"))


def _table_d(
    result: Mapping[str, object],
    root_report: Mapping[str, object],
    seed0: Mapping[str, object],
    seed_report: Mapping[str, object],
) -> list[dict[str, object]]:
    seed0_tables = _mapping("seed0.tables", seed0.get("tables", {}))
    root_tables = _mapping("result.tables", result.get("tables", {}))
    seed_average = root_tables.get("seed_average_policy_losses")
    complete = (
        seed_average
        if root_report is not seed_report and seed_average is not None
        else seed0_tables.get("target_policy_losses")
    )
    partial = seed0_tables.get("target_b3_b4_losses")
    if complete is None and partial is None:
        return _sentinel("table_d", seed_report)
    frame = _frame(
        "target policy losses", complete if complete is not None else partial
    )
    required = {"dataset_id", "b3_normalized_loss", "b4_normalized_loss"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"target policy losses are missing {missing}")
    has_m1 = "m1_normalized_loss" in frame
    directions = _direction_by_target(seed_report)
    comparison_report = seed_report
    bootstrap = seed_report.get("bootstrap")
    is_seed_average = root_report is not seed_report and seed_average is not None
    if is_seed_average:
        comparison_report = _mapping(
            "seed-average Gate2 analysis",
            root_report.get("seed_average_gate2_analysis"),
        )
        bootstrap = comparison_report.get("bootstrap")
    bootstrap_directions: Mapping[str, object] = {}
    if bootstrap is not None:
        bootstrap_map = _mapping("report.bootstrap", bootstrap)
        bootstrap_directions = _mapping(
            "report.bootstrap.directions", bootstrap_map.get("directions")
        )
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        subset = frame.loc[frame["dataset_id"].astype(str) == dataset]
        if subset.empty:
            raise ValueError(f"target policy losses do not contain {dataset}")
        direction = directions.get(dataset)
        source = None if direction is None else direction.get("source_dataset")
        ci = None
        if has_m1 and dataset in bootstrap_directions:
            item = _mapping(f"bootstrap direction {dataset}", bootstrap_directions[dataset])
            ci = _safe(item.get("ci95_percent"))
        rows.append(
            {
                "target": dataset,
                "source": source,
                "b3_loss": _finite(
                    f"{dataset} B3 loss", subset["b3_normalized_loss"].mean()
                ),
                "b4_loss": _finite(
                    f"{dataset} B4 loss", subset["b4_normalized_loss"].mean()
                ),
                "m1_loss": None
                if not has_m1
                else _finite(
                    f"{dataset} M1 loss", subset["m1_normalized_loss"].mean()
                ),
                "ri_b4_vs_b3_percent": _comparison_ri(
                    comparison_report, "b4_vs_b3", dataset
                ),
                "ri_m1_vs_b3_percent": _comparison_ri(
                    comparison_report, "m1_vs_b3", dataset
                ),
                "ri_m1_vs_b4_percent": _comparison_ri(
                    comparison_report, "m1_vs_b4", dataset
                ),
                "ci95_percent": ci,
                "status": (
                    "OBSERVED_SEED_AVERAGE"
                    if is_seed_average
                    else "OBSERVED"
                    if has_m1
                    else "PARTIAL_OBSERVED_M1_NOT_RUN_AFTER_GATE1B"
                ),
            }
        )
    return rows


def _table_e(seed_report: Mapping[str, object]) -> list[dict[str, object]]:
    raw = seed_report.get("gate3_safety")
    if raw is None:
        return _sentinel("table_e", seed_report)
    gate = _mapping("report.gate3_safety", raw)
    raw_metrics = gate.get("origin_metrics")
    if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
        raise TypeError("gate3_safety.origin_metrics must be a sequence")
    metrics = [_mapping("origin safety metric", item) for item in raw_metrics]
    if not metrics:
        raise ValueError("origin safety metrics must be nonempty")
    minimum = min(
        _finite("origin safety RI vs B3", item.get("ri_m1_vs_b3_percent"))
        for item in metrics
    )
    rows: list[dict[str, object]] = []
    for item in sorted(
        metrics,
        key=lambda row: (
            DATASET_ORDER.index(str(row.get("dataset_id"))),
            _integer("safety origin", row.get("origin")),
        ),
    ):
        ri_b3 = _finite("origin safety RI vs B3", item.get("ri_m1_vs_b3_percent"))
        rows.append(
            {
                "target": str(item.get("dataset_id")),
                "origin": _integer("safety origin", item.get("origin")),
                "ri_m1_vs_b3_percent": ri_b3,
                "ri_m1_vs_b4_percent": _finite(
                    "origin safety RI vs B4", item.get("ri_m1_vs_b4_percent")
                ),
                "is_worst_origin": bool(np.isclose(ri_b3, minimum, rtol=0.0, atol=0.0)),
                "status": "OBSERVED",
            }
        )
    return rows


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _table_f(seed_report: Mapping[str, object]) -> list[dict[str, object]]:
    raw = seed_report.get("gate3_control")
    if raw is None:
        return _sentinel("table_f", seed_report)
    gate = _mapping("report.gate3_control", raw)
    directions = _mapping(
        "report.gate3_control.direction_ri_percent",
        gate.get("direction_ri_percent"),
    )
    real = _mapping("control real direction", directions.get("real"))
    shuffled = _mapping("control shuffled direction", directions.get("shuffled"))
    random = _mapping("control random direction", directions.get("random"))
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        real_value = _finite(f"{dataset} real retrieval RI", real.get(dataset))
        shuffled_value = _finite(f"{dataset} shuffled RI", shuffled.get(dataset))
        random_value = _finite(f"{dataset} random RI", random.get(dataset))
        rows.append(
            {
                "target": dataset,
                "real_retrieval_ri_percent": real_value,
                "shuffled_value_ri_percent": shuffled_value,
                "random_neighbor_ri_percent": random_value,
                "control_over_real_ratio": {
                    "shuffled": _ratio(shuffled_value, real_value),
                    "random": _ratio(random_value, real_value),
                },
                "status": "OBSERVED",
            }
        )
    return rows


def _gate_observed(gate: str, result: Mapping[str, object]) -> object:
    if gate == "GATE0":
        return {
            "gains_percent": _safe(result.get("gains_percent")),
            "macro_gain_percent": _safe(result.get("macro_gain_percent")),
            "heterogeneous_diagnostic": _safe(
                result.get("heterogeneous_diagnostic")
            ),
        }
    if gate == "GATE1A":
        return _safe(result.get("datasets"))
    if gate == "GATE1B":
        return {"ri_percent": _safe(result.get("ri_percent"))}
    if gate == "GATE2":
        return {
            key: _safe(result.get(key))
            for key in (
                "m1_vs_b4_percent",
                "m1_vs_b4_macro_percent",
                "m1_vs_b3_percent",
                "m1_vs_b3_macro_percent",
                "direction_ci95_percent",
                "macro_ci95_percent",
                "decision_state",
                "deferral_withdrawn_by",
            )
        }
    if gate == "GATE3_SAFETY":
        return {
            "worst_origin_ri_m1_vs_b3_percent": _safe(
                result.get("worst_origin_ri_m1_vs_b3_percent")
            ),
            "tail_metrics": _safe(result.get("tail_metrics")),
        }
    if gate == "GATE3_CONTROL":
        return {
            "macro_ri_percent": _safe(result.get("macro_ri_percent")),
            "shuffle_over_real": _safe(result.get("shuffle_over_real")),
            "random_over_real": _safe(result.get("random_over_real")),
        }
    return {
        key: _safe(value)
        for key, value in result.items()
        if key
        in {
            "action",
            "passed",
            "seed_macro_ri_percent",
            "seed_average_macro_ri_percent",
            "seed_average_ci95_percent",
            "effect_retention",
            "opposite_sign_count",
            "checks",
            "borderline_reasons",
            "forbidden_reasons",
        }
    }


def _table_g(
    root_report: Mapping[str, object], seed_report: Mapping[str, object]
) -> list[dict[str, object]]:
    stopped_after = seed_report.get("stopped_after")
    if not isinstance(stopped_after, str) or stopped_after not in _STOP_POSITIONS:
        raise ValueError("seed-0 report lacks a recognized stopped_after gate")
    stop_position = _STOP_POSITIONS[stopped_after]
    rows: list[dict[str, object]] = []
    for position, gate in enumerate(GATE_ORDER):
        raw = _gate_result_for_position(root_report, seed_report, gate)
        if raw is None:
            if position <= stop_position:
                raise ValueError(f"report is missing executed {gate}")
            marker = _stop_marker(seed_report)
            rows.append(
                {
                    "gate": gate,
                    "threshold": _safe(_GATE_THRESHOLDS[gate]),
                    "observed": {"status": marker},
                    "pass_fail": marker,
                    "scientific_interpretation": "not evaluated after the prior gate stop",
                    "status": marker,
                }
            )
            continue
        gate_result = _mapping(f"{gate} result", raw)
        passed = gate_result.get("passed")
        if _is_gate4_eligibility_observation(seed_report, gate):
            if not isinstance(passed, (bool, np.bool_)):
                raise ValueError("Gate4 eligibility observation must be Boolean")
            pass_fail = "PASS" if bool(passed) else "FAIL"
            interpretation = (
                "observed only to determine conditional-seed eligibility; "
                "the final failure-first decision remains Gate 2"
            )
            row_status = "ELIGIBILITY_OBSERVATION"
        elif (
            gate == "GATE2"
            and root_report is seed_report
            and gate_result.get("decision_state") == "PENDING_GATE4"
            and passed in {True, False}
        ):
            pass_fail = "PENDING_GATE4"
            interpretation = "the frozen borderline rule defers final Gate 2 to seed-average losses"
            row_status = "OBSERVED"
        elif passed is None and gate == "GATE4" and gate_result.get("action") == "RUN_SEED1":
            pass_fail = "PENDING_SEED1"
            interpretation = "the frozen borderline rule requested model seed 1"
            row_status = "OBSERVED"
        elif isinstance(passed, (bool, np.bool_)):
            pass_fail = "PASS" if bool(passed) else "FAIL"
            interpretation = _GATE_INTERPRETATIONS[gate][0 if bool(passed) else 1]
            row_status = "OBSERVED"
        else:
            raise ValueError(f"report has no recognized decision for {gate}")
        rows.append(
            {
                "gate": gate,
                "threshold": _safe(_GATE_THRESHOLDS[gate]),
                "observed": _gate_observed(gate, gate_result),
                "pass_fail": pass_fail,
                "scientific_interpretation": interpretation,
                "status": row_status,
            }
        )
    return rows


def build_tables_a_to_g(result: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    """Build the seven preregistered result tables from an in-memory full result.

    Both ``run_full_seed0`` output and the wrapper returned by
    ``run_full_protocol`` are accepted.  Tables are returned in A--G insertion
    order and contain only strict JSON values.  A failure-first stop produces
    explicit ``NOT_RUN_AFTER_<GATE>`` rows for every later unavailable table.
    """

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    seed0, seed_report = _seed_zero(result)
    root_report = _mapping("result.report", result.get("report"))
    if root_report.get("experiment") not in {None, EXPERIMENT}:
        raise ValueError("result belongs to a different experiment")
    _validate_failure_first_payload(result, root_report, seed0, seed_report)
    tables = {
        "table_a": _table_a(result, seed0),
        "table_b": _table_b(seed_report),
        "table_c": _table_c(seed_report),
        "table_d": _table_d(result, root_report, seed0, seed_report),
        "table_e": _table_e(seed_report),
        "table_f": _table_f(seed_report),
        "table_g": _table_g(root_report, seed_report),
    }
    safe = _safe(tables)
    assert isinstance(safe, dict)
    return safe  # type: ignore[return-value]


def _compact(value: object) -> str:
    return json.dumps(_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _terminal_verdict(
    root_report: Mapping[str, object], seed_report: Mapping[str, object]
) -> str:
    verdict = root_report.get("final_verdict")
    if root_report.get("terminal") is not True or verdict not in FINAL_VERDICT_TOKENS:
        raise ValueError("result does not contain a terminal scientific verdict")
    stop_position = _failure_first_position(seed_report)
    decisions: dict[str, bool] = {}
    for position, gate in enumerate(GATE_ORDER[: stop_position + 1]):
        raw = _mapping(
            f"{gate} result",
            _gate_result_for_position(root_report, seed_report, gate),
        )
        passed = raw.get("passed")
        if not isinstance(passed, (bool, np.bool_)):
            raise ValueError(f"terminal result lacks a boolean {gate} decision")
        decisions[gate] = bool(passed)
        resolved_multiseed_gate2 = (
            gate == "GATE2"
            and root_report is not seed_report
            and root_report.get("averaged_gate2") is not None
        )
        withdrawn_seed0_gate2 = (
            gate == "GATE2"
            and root_report is seed_report
            and raw.get("decision_state") == "FINAL_FAIL"
            and raw.get("deferral_withdrawn_by")
            in {"GATE3_SAFETY", "GATE3_CONTROL"}
        )
        if (
            position < stop_position
            and not decisions[gate]
            and not resolved_multiseed_gate2
            and not withdrawn_seed0_gate2
        ):
            raise ValueError(f"terminal result has a failed gate before {gate}")

    if stop_position == 0:
        gate0 = _mapping("GATE0 result", seed_report.get("gate0"))
        diagnostic = gate0.get("heterogeneous_diagnostic")
        heterogeneous_pass = bool(
            isinstance(diagnostic, Mapping) and diagnostic.get("passed") is True
        )
        expected = decide_final_verdict(
            gate0_pass=False, heterogeneous_gate_pass=heterogeneous_pass
        )
    else:
        kwargs: dict[str, bool] = {"gate0_pass": decisions["GATE0"]}
        argument_names = {
            "GATE1A": "gate1a_pass",
            "GATE1B": "gate1b_pass",
            "GATE2": "gate2_pass",
            "GATE3_SAFETY": "gate3_safety_pass",
            "GATE3_CONTROL": "gate3_control_pass",
            "GATE4": "gate4_pass",
        }
        for gate, argument in argument_names.items():
            if gate in decisions:
                kwargs[argument] = decisions[gate]
        expected = decide_final_verdict(**kwargs)
    if verdict != expected:
        raise ValueError(
            f"terminal scientific verdict is inconsistent: expected {expected}"
        )
    seed_verdict = seed_report.get("final_verdict")
    if seed_report.get("terminal") is True and seed_verdict != expected:
        raise ValueError("seed-0 terminal verdict disagrees with its gate decisions")
    return str(verdict)


def _frozen_summary(preregistration: Mapping[str, object] | None) -> object:
    if preregistration is None:
        raise ValueError("terminal status requires the frozen preregistration")
    if preregistration.get("experiment_name") != EXPERIMENT:
        raise ValueError("terminal status preregistration belongs to another experiment")
    if preregistration.get("frozen_before_any_new_model_fit") is not True:
        raise ValueError("terminal status requires a pre-fit frozen preregistration")
    digest = preregistration.get("preregistration_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("terminal status requires the preregistration SHA256")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError("terminal status preregistration SHA256 is invalid") from exc
    selected = {
        key: preregistration.get(key)
        for key in (
            "experiment_name",
            "frozen_at_utc",
            "frozen_before_any_new_model_fit",
            "preregistration_sha256",
            "repository",
            "model_seed",
            "conditional_model_seeds",
            "bootstrap",
            "policy_grids",
            "splits",
        )
        if key in preregistration
    }
    return selected


def _section(number: int, title: str, marker: str, content: object) -> str:
    text = content if isinstance(content, str) else _compact(content)
    return f"## {number}. {title}\n\n{marker} {text}"


def build_status_markdown(
    result: Mapping[str, object],
    *,
    preregistration: Mapping[str, object] | None = None,
    artifact_bindings: Mapping[str, object] | None = None,
) -> str:
    """Render the required 16-section terminal scientific status.

    This function refuses nonterminal results, including a runtime approval
    stop.  Use :func:`build_runtime_stop_status` for that distinct condition.
    """

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    root_report = _mapping("result.report", result.get("report"))
    seed0, seed_report = _seed_zero(result)
    verdict = _terminal_verdict(root_report, seed_report)
    frozen = _frozen_summary(preregistration)
    assert preregistration is not None
    if root_report.get("preregistration_sha256") != preregistration.get(
        "preregistration_sha256"
    ):
        raise ValueError("terminal report is not bound to the frozen preregistration")
    if artifact_bindings is not None and not isinstance(artifact_bindings, Mapping):
        raise TypeError("artifact_bindings must be a mapping")
    tables = build_tables_a_to_g(result)
    gate_rows = tables["table_g"]
    del seed0
    executed_seeds = root_report.get("executed_model_seeds", [0])
    worst_rows = [
        row for row in tables["table_e"] if row.get("is_worst_origin") is True
    ]
    safety_row = next(row for row in gate_rows if row["gate"] == "GATE3_SAFETY")
    robustness_row = next(row for row in gate_rows if row["gate"] == "GATE4")
    sections = [
        _section(
            1,
            "What was tested",
            "[관찰]",
            {
                "experiment": EXPERIMENT,
                "stage": root_report.get("stage", seed_report.get("stage")),
                "executed_model_seeds": executed_seeds,
                "transfers": ["m5_to_favorita", "favorita_to_m5"],
            },
        ),
        _section(
            2,
            "What was frozen before results",
            "[관찰]",
            {
                "frozen_preregistration": frozen,
                "terminal_artifact_bindings": _safe(artifact_bindings)
                if artifact_bindings is not None
                else {"status": "NOT_PROVIDED"},
            },
        ),
        _section(3, "Data and split", "[관찰]", tables["table_a"]),
        _section(
            4,
            "Runtime",
            "[관찰]",
            [
                {
                    "dataset": row["dataset"],
                    "training_runtime_seconds": row["training_runtime_seconds"],
                    "training_runtime_by_seed_seconds": row[
                        "training_runtime_by_seed_seconds"
                    ],
                    "actual_synchronized_end_to_end_wall_seconds": row[
                        "actual_synchronized_end_to_end_wall_seconds"
                    ],
                    "actual_synchronized_end_to_end_wall_by_seed_seconds": row[
                        "actual_synchronized_end_to_end_wall_by_seed_seconds"
                    ],
                }
                for row in tables["table_a"]
            ],
        ),
        _section(5, "Expert quality", "[관찰]", tables["table_b"]),
        _section(
            6,
            "Oracle opportunity",
            "[관찰]",
            next(row["observed"] for row in gate_rows if row["gate"] == "GATE0"),
        ),
        _section(7, "Temporal recurrence", "[관찰]", tables["table_c"]),
        _section(
            8,
            "B4 simple memory",
            "[관찰]",
            [
                {
                    key: row.get(key)
                    for key in (
                        "target",
                        "source",
                        "b3_loss",
                        "b4_loss",
                        "ri_b4_vs_b3_percent",
                        "status",
                    )
                }
                for row in tables["table_d"]
            ],
        ),
        _section(
            9,
            "M1 retrieval memory",
            "[관찰]",
            [
                {
                    key: row.get(key)
                    for key in (
                        "target",
                        "source",
                        "m1_loss",
                        "ri_m1_vs_b3_percent",
                        "ri_m1_vs_b4_percent",
                        "ci95_percent",
                        "status",
                    )
                }
                for row in tables["table_d"]
            ],
        ),
        _section(10, "Controls", "[관찰]", tables["table_f"]),
        _section(
            11,
            "Safety",
            "[관찰]",
            {
                "worst_origin": worst_rows
                if worst_rows
                else {"status": _stop_marker(seed_report)},
                "observed_gate_metrics": safety_row["observed"],
            },
        ),
        _section(
            12,
            "Seed robustness",
            "[관찰]",
            {
                "executed_model_seeds": executed_seeds,
                "observed_gate_metrics": robustness_row["observed"],
            },
        ),
        _section(13, "Gate table", "[판정]", gate_rows),
        _section(14, "Final interpretation", "[판정]", _VERDICT_INTERPRETATIONS[verdict]),
        _section(
            15,
            "What must NOT be claimed",
            "[판정]",
            (
                "이 development-dataset 결과를 untouched external confirmatory validation, "
                "최종 논문 성능, 인과 효과, 또는 일반적 우월성으로 주장해서는 안 된다."
            ),
        ),
        _section(
            16,
            "Exact next action",
            "[최종]",
            f"{verdict} → {_NEXT_ACTIONS[verdict]}",
        ),
    ]
    return f"FINAL VERDICT: {verdict}\n\n" + "\n\n".join(sections) + "\n"


def build_runtime_stop_status(runtime_result: Mapping[str, object]) -> str:
    """Render a runtime-approval stop without inventing a scientific verdict."""

    if not isinstance(runtime_result, Mapping):
        raise TypeError("runtime_result must be a mapping")
    payload = runtime_result
    if "runtime_gate" not in payload and isinstance(payload.get("report"), Mapping):
        payload = payload["report"]  # type: ignore[assignment]
    gate = _mapping("runtime_result.runtime_gate", payload.get("runtime_gate"))
    if gate.get("action") != "STOP_FOR_APPROVAL" or gate.get("exceeded") is not True:
        raise ValueError("runtime result is not a STOP_FOR_APPROVAL condition")
    threshold = _finite("runtime threshold_gpu_hours", gate.get("threshold_gpu_hours"))
    projected = _finite("runtime projected_gpu_hours", gate.get("projected_gpu_hours"))
    if projected <= threshold:
        raise ValueError("STOP_FOR_APPROVAL requires projected GPU hours above threshold")
    fallback = payload.get("runtime_projection_2000_per_dataset")
    if fallback is None and isinstance(runtime_result.get("report"), Mapping):
        fallback = runtime_result["report"].get(  # type: ignore[index,union-attr]
            "runtime_projection_2000_per_dataset"
        )
    if not isinstance(fallback, Mapping):
        raise ValueError(
            "STOP_FOR_APPROVAL requires a 2000-series-per-dataset runtime estimate"
        )
    observations = {
        "preregistration_sha256": _safe(payload.get("preregistration_sha256")),
        "runtime_estimate_sha256": _safe(payload.get("runtime_estimate_sha256")),
        "threshold_gpu_hours": threshold,
        "projected_full_seed0_gpu_hours": projected,
        "measured_smoke_training": _safe(payload.get("training")),
        "measured_smoke_inference": _safe(payload.get("inference")),
        "measured_cuda_peak_memory_bytes": _safe(
            payload.get("cuda_peak_memory_bytes")
        ),
        "full_seed0_projection": _safe(
            payload.get("runtime_projection_full_seed0")
        ),
        "projection_2000_series_per_dataset": _safe(fallback),
        "retrieval_projection_full_seed0": _safe(
            payload.get("retrieval_runtime_projection_full_seed0")
        ),
        "storage_projection": _safe(payload.get("serialization")),
    }
    return (
        "EXECUTION STATUS: STOP_FOR_APPROVAL\n"
        "SCIENTIFIC GATES: NOT_RUN\n\n"
        "## Runtime projection\n\n"
        f"[관찰] {_compact(observations)}\n\n"
        "## Runtime gate\n\n"
        f"[판정] 예상 full seed-0 GPU 시간이 동결된 {threshold:g}-hour gate를 초과해 "
        "failure-first 실행을 중단했다. 과학적 Gate 0--4 판정은 수행하지 않았다.\n\n"
        "## Exact next action\n\n"
        "[최종] full seed-0 실행 승인 여부를 사용자에게 요청한다.\n"
    )


__all__ = [
    "TABLE_KEYS",
    "build_runtime_stop_status",
    "build_status_markdown",
    "build_tables_a_to_g",
]
