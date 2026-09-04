"""No-write execution engine for the full online-memory seed-0 study.

This module deliberately separates expensive expert fitting from policy
analysis.  ``train_dataset_experts`` returns only in-memory objects, and
``analyze_seed0`` consumes already-computed dataset outputs.  Artifact naming,
append-only persistence, and preregistration checks belong to the protocol
layer rather than this engine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
import time

import numpy as np
import pandas as pd
import torch

from .analysis import (
    ANALYSIS_SEED,
    BOOTSTRAP_DRAWS,
    FINAL_VERDICT_TOKENS,
    decide_final_verdict,
    evaluate_gate0,
    evaluate_gate1b,
    evaluate_gate2,
    evaluate_gate3_control,
    evaluate_gate3_safety,
    evaluate_gate4_seed0,
    evaluate_gate4_seed1,
    evaluate_gate4_seed2,
    evaluate_seed_average_gate2,
    oracle_ladder,
    paired_series_cluster_bootstrap,
    summarize_loss_comparison,
    temporal_recurrence,
)
from .bridge import build_policy_cases
from .data import build_external_split, evaluation_origins
from .evaluation import (
    CONTROL_SEED,
    build_m1_neighbor_plan,
    convex_forecast,
    evaluate_b4_cases,
    evaluate_c0_cases,
    evaluate_c1_cases,
    evaluate_m1_cases,
    tune_b3_source,
    tune_b4_source,
    tune_m1_source,
)
from .pipeline import build_prediction_frame, normalized_loss_frame
from .trainer import train_one_on_split


DATASET_ORDER = ("m5", "favorita")
TRAIN_ORIGIN_STRIDE = 7
EXPECTED_MODEL_PARAMETERS = 7056
MODEL_SPECS = (
    ("point", "M0PM_point_mse_param_matched"),
    ("hurdle", "M1_factorized_mean"),
)
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


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _json_safe(value: object) -> object:
    """Convert report data without silently stringifying unsupported values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("reports cannot contain non-finite numbers")
        return result
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _dataset_schedule(dataset: str, cfg: object) -> tuple[np.ndarray, tuple[int, ...]]:
    if dataset not in EXPECTED_SCHEDULES:
        raise ValueError(f"unknown dataset: {dataset}")
    expected = EXPECTED_SCHEDULES[dataset]
    fields = {
        "length": expected["length"],
        "train_end": expected["model_train_end"],
        "val_end": expected["warmup_origin"],
        "lookback": 96,
        "horizon": 28,
    }
    for field, required in fields.items():
        if not hasattr(cfg, field) or int(getattr(cfg, field)) != int(required):
            raise ValueError(
                f"{dataset} cfg {field} must equal the frozen value {required}"
            )
    origins = evaluation_origins(int(cfg.length), int(cfg.horizon), 7)
    required_origins = np.asarray(
        (expected["warmup_origin"], *expected["evaluation_origins"]),
        dtype=np.int64,
    )
    if not np.array_equal(origins.astype(np.int64), required_origins):
        raise AssertionError(f"{dataset} full forecast schedule drifted")
    return origins.astype(np.int32), tuple(expected["evaluation_origins"])


def _cpu_state_dict(model: object) -> dict[str, torch.Tensor]:
    if not hasattr(model, "state_dict"):
        raise TypeError("canonical trainer returned a model without state_dict")
    raw = model.state_dict()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("canonical model state_dict must be a nonempty mapping")
    result: dict[str, torch.Tensor] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("checkpoint state must map strings to tensors")
        result[name] = value.detach().cpu().clone()
    return result


def _finite_provenance(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _model_provenance(
    trained: Mapping[str, object],
    *,
    model_id: str,
    model_seed: int,
    end_to_end_wall_seconds: float,
    device_type: str,
) -> dict[str, object]:
    required = (
        "best_epoch",
        "best_validation_mean_mse",
        "train_seconds",
        "n_parameters",
    )
    missing = [key for key in required if key not in trained]
    if missing:
        raise ValueError(f"canonical trainer result is missing {missing}")
    n_parameters = _integer(
        "n_parameters", trained["n_parameters"], minimum=1
    )
    if n_parameters != EXPECTED_MODEL_PARAMETERS:
        raise ValueError(
            f"{model_id} parameter count drifted: expected "
            f"{EXPECTED_MODEL_PARAMETERS}, got {n_parameters}"
        )
    elapsed = _finite_provenance(
        "end_to_end_wall_seconds", end_to_end_wall_seconds
    )
    if elapsed < 0.0:
        raise ValueError("end_to_end_wall_seconds must be nonnegative")
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("device_type must be cpu or cuda")
    return {
        "model_id": model_id,
        "model_seed": model_seed,
        "trainer": "experiments.om_factorization_killtest.train.train_one",
        "adapter": (
            "experiments.ph_online_memory_gono_v1.trainer."
            "train_one_on_split"
        ),
        "best_epoch": _integer("best_epoch", trained["best_epoch"], minimum=-1),
        "best_validation_mean_mse": _finite_provenance(
            "best_validation_mean_mse", trained["best_validation_mean_mse"]
        ),
        "train_seconds": _finite_provenance(
            "train_seconds", trained["train_seconds"]
        ),
        "n_parameters": n_parameters,
        "checkpoint_device": "cpu",
        "execution_device_type": device_type,
        "end_to_end_wall_seconds": elapsed,
        "end_to_end_wall_definition": (
            "device-synchronized wall time around canonical train_one_on_split, "
            "including its final seven-origin prediction and excluding artifact I/O"
        ),
    }


def _validated_arm_payload(
    payload: Mapping[str, object],
    *,
    dataset: str,
    arm: str,
    model_id: str,
    model_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object], dict[str, np.ndarray]]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"persisted {arm} arm must be a mapping")
    expected_identity = {
        "dataset": dataset,
        "arm": arm,
        "model_id": model_id,
        "model_seed": model_seed,
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise ValueError(f"persisted {arm} arm has mismatched {key}")
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError(f"persisted {arm} arm lacks a state_dict")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("persisted state_dict must map strings to tensors")
        state[name] = value.detach().cpu().clone()
    raw_provenance = payload.get("provenance")
    if not isinstance(raw_provenance, Mapping):
        raise ValueError(f"persisted {arm} arm lacks provenance")
    provenance = dict(raw_provenance)
    if (
        provenance.get("model_id") != model_id
        or provenance.get("model_seed") != model_seed
        or provenance.get("checkpoint_device") != "cpu"
        or provenance.get("n_parameters") != EXPECTED_MODEL_PARAMETERS
    ):
        raise ValueError(f"persisted {arm} provenance identity drifted")
    elapsed = _finite_provenance(
        f"persisted {arm} end_to_end_wall_seconds",
        provenance.get("end_to_end_wall_seconds"),
    )
    if elapsed < 0.0 or provenance.get("execution_device_type") not in {
        "cpu",
        "cuda",
    }:
        raise ValueError(f"persisted {arm} runtime provenance drifted")
    raw_predictions = payload.get("predictions")
    if not isinstance(raw_predictions, Mapping) or not raw_predictions:
        raise ValueError(f"persisted {arm} arm lacks predictions")
    predictions = {
        str(key): np.asarray(value).copy()
        for key, value in raw_predictions.items()
    }
    return state, provenance, predictions


def train_dataset_experts(
    population: Mapping[str, object],
    device: torch.device,
    *,
    model_seed: int = 0,
    persisted_arms: Mapping[str, Mapping[str, object]] | None = None,
    on_arm_complete: Callable[[str, Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Fit the paired experts, optionally resuming verified arm checkpoints."""

    if not isinstance(population, Mapping):
        raise TypeError("population must be a mapping")
    missing = [
        key for key in ("data", "cfg", "manifest") if key not in population
    ]
    if missing:
        raise ValueError(f"population is missing {missing}")
    data = population["data"]
    cfg = population["cfg"]
    manifest = population["manifest"]
    if not isinstance(data, Mapping) or "name" not in data:
        raise ValueError("population data must contain its dataset name")
    dataset = str(data["name"]).lower()
    seed = _integer("model_seed", model_seed)
    supplied_arms = {} if persisted_arms is None else dict(persisted_arms)
    valid_arm_names = {arm for arm, _model_id in MODEL_SPECS}
    unexpected_arms = sorted(set(supplied_arms) - valid_arm_names)
    if unexpected_arms:
        raise ValueError(f"unexpected persisted arms: {unexpected_arms}")
    if on_arm_complete is not None and not callable(on_arm_complete):
        raise TypeError("on_arm_complete must be callable")
    origins, evaluations = _dataset_schedule(dataset, cfg)
    torch_device = torch.device(device)

    split = build_external_split(
        data,
        cfg,
        train_origin_stride=TRAIN_ORIGIN_STRIDE,
        forecast_origins=origins.copy(),
    )
    observed_origins = np.asarray(split.test.origins, dtype=np.int64)
    if not np.array_equal(observed_origins, origins.astype(np.int64)):
        raise ValueError("external split did not preserve the frozen seven origins")
    if int(split.test.n_origins) != 7:
        raise ValueError("full split must contain warmup plus six evaluation origins")
    if int(split.test.n_series) != int(getattr(cfg, "n_series")):
        raise ValueError("full split series count does not match the population config")

    state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    provenance: dict[str, dict[str, object]] = {}
    arm_predictions: dict[str, dict[str, np.ndarray]] = {}
    for arm, model_id in MODEL_SPECS:
        if arm in supplied_arms:
            state, arm_provenance, predictions_for_arm = _validated_arm_payload(
                supplied_arms[arm],
                dataset=dataset,
                arm=arm,
                model_id=model_id,
                model_seed=seed,
            )
            state_dicts[arm] = state
            provenance[arm] = arm_provenance
            arm_predictions[arm] = predictions_for_arm
            continue
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        arm_started = time.perf_counter()
        trained = train_one_on_split(model_id, split, cfg, seed, torch_device)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        end_to_end_wall_seconds = time.perf_counter() - arm_started
        if not isinstance(trained, Mapping) or "model" not in trained:
            raise TypeError("canonical trainer must return a mapping with model")
        if "predictions" not in trained or not isinstance(
            trained["predictions"], Mapping
        ):
            raise ValueError("canonical trainer did not return test predictions")
        model = trained["model"]
        state_dicts[arm] = _cpu_state_dict(model)
        provenance[arm] = _model_provenance(
            trained,
            model_id=model_id,
            model_seed=seed,
            end_to_end_wall_seconds=end_to_end_wall_seconds,
            device_type=torch_device.type,
        )
        arm_predictions[arm] = {
            str(key): np.asarray(value).copy()
            for key, value in trained["predictions"].items()
        }
        arm_payload = {
            "schema_version": 1,
            "dataset": dataset,
            "arm": arm,
            "model_id": model_id,
            "model_seed": seed,
            "state_dict": state_dicts[arm],
            "provenance": provenance[arm],
            "predictions": arm_predictions[arm],
        }
        if on_arm_complete is not None:
            on_arm_complete(arm, arm_payload)
        del model, trained
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = build_prediction_frame(
        data,
        split.test,
        arm_predictions["point"],
        arm_predictions["hurdle"],
    )
    losses = normalized_loss_frame(
        predictions,
        data,
        model_train_end=int(cfg.train_end),
        horizon=int(cfg.horizon),
    )
    cases = build_policy_cases(
        predictions,
        data,
        split.test,
        model_train_end=int(cfg.train_end),
        horizon=int(cfg.horizon),
        lookback=int(cfg.lookback),
    )
    scale_keys = cases.loc[:, [*CASE_KEYS, "policy_scale_squared"]]
    if bool(scale_keys.duplicated(list(CASE_KEYS), keep=False).any()):
        raise ValueError("policy cases have duplicate scale keys")
    step_predictions = predictions.merge(
        scale_keys,
        on=list(CASE_KEYS),
        how="inner",
        validate="many_to_one",
        sort=False,
    ).sort_values([*CASE_KEYS, "step"], kind="mergesort").reset_index(drop=True)
    if len(step_predictions) != len(predictions):
        raise ValueError("policy scales do not cover every prediction step")

    schedule = {
        "warmup_origin": int(origins[0]),
        "evaluation_origins": [int(value) for value in evaluations],
        "all_forecast_origins": [int(value) for value in origins],
        "horizon": int(cfg.horizon),
        "lookback": int(cfg.lookback),
        "model_train_end": int(cfg.train_end),
        "train_origin_stride": TRAIN_ORIGIN_STRIDE,
    }
    return {
        "dataset": dataset,
        "model_seed": seed,
        "schedule": schedule,
        "population_manifest": _json_safe(manifest),
        "provenance": _json_safe(provenance),
        "state_dicts": state_dicts,
        "predictions": predictions,
        "step_predictions": step_predictions,
        "losses": losses,
        "cases": cases,
    }


def _validated_outputs(
    outputs: Mapping[str, Mapping[str, object]],
    *,
    expected_model_seed: int,
) -> dict[str, Mapping[str, object]]:
    if not isinstance(outputs, Mapping) or set(outputs) != set(DATASET_ORDER):
        raise ValueError("dataset_outputs must contain exactly m5 and favorita")
    result: dict[str, Mapping[str, object]] = {}
    for dataset in DATASET_ORDER:
        output = outputs[dataset]
        if not isinstance(output, Mapping):
            raise TypeError(f"dataset output {dataset} must be a mapping")
        if str(output.get("dataset", "")) != dataset:
            raise ValueError(f"dataset output {dataset} has a mismatched identity")
        if _integer("model_seed", output.get("model_seed")) != expected_model_seed:
            raise ValueError(
                f"dataset output {dataset} does not use model seed "
                f"{expected_model_seed}"
            )
        schedule = output.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError(f"dataset output {dataset} lacks a schedule")
        expected = EXPECTED_SCHEDULES[dataset]
        expected_all = (
            expected["warmup_origin"],
            *expected["evaluation_origins"],
        )
        observed_all = tuple(
            _integer("forecast origin", value)
            for value in schedule.get("all_forecast_origins", ())
        )
        observed_eval = tuple(
            _integer("evaluation origin", value)
            for value in schedule.get("evaluation_origins", ())
        )
        if (
            observed_all != expected_all
            or observed_eval != expected["evaluation_origins"]
            or schedule.get("warmup_origin") != expected["warmup_origin"]
            or schedule.get("horizon") != 28
            or schedule.get("lookback") != 96
            or schedule.get("model_train_end") != expected["model_train_end"]
        ):
            raise ValueError(f"dataset output {dataset} schedule drifted")
        for key in ("step_predictions", "losses", "cases"):
            frame = output.get(key)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError(f"dataset output {dataset} {key} must be nonempty")
            if "dataset_id" not in frame or "origin" not in frame:
                raise ValueError(f"dataset output {dataset} {key} lacks identity keys")
            if set(frame["dataset_id"].astype(str)) != {dataset}:
                raise ValueError(f"dataset output {dataset} {key} crossed datasets")
            if set(frame["origin"].astype(int)) != set(expected_all):
                raise ValueError(
                    f"dataset output {dataset} {key} must cover exactly seven origins"
                )
        result[dataset] = output
    return result


def _evaluation_frame(output: Mapping[str, object], key: str) -> pd.DataFrame:
    frame = output[key]
    schedule = output["schedule"]
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(schedule, Mapping)
    origins = list(schedule["evaluation_origins"])
    return frame.loc[frame["origin"].isin(origins)].copy().reset_index(drop=True)


def _blend_loss_frame(cases: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if not isinstance(alpha, Real) or isinstance(alpha, (bool, np.bool_)):
        raise TypeError("B3 alpha must be numeric")
    weight = float(alpha)
    if not np.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError("B3 alpha must lie in [0, 1]")
    rows: list[dict[str, object]] = []
    for row in cases.itertuples(index=False):
        forecast = convex_forecast(
            np.asarray(row.point_forecast, dtype=np.float64),
            np.asarray(row.hurdle_forecast, dtype=np.float64),
            weight,
        )
        target = np.asarray(row.target, dtype=np.float64)
        mask = np.asarray(row.target_mask)
        if (
            target.shape != (28,)
            or forecast.shape != (28,)
            or mask.shape != (28,)
        ):
            raise ValueError("B3 requires an exact 28-step forecast horizon")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError("B3 target mask must remain Boolean")
        if not bool(mask.all()):
            raise ValueError("B3 requires all 28 target steps to be observed")
        if not bool(np.isfinite(target).all()):
            raise ValueError("B3 target must be finite")
        scale = float(row.policy_scale_squared)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("B3 policy scale must be finite and positive")
        loss = float(np.square(target[mask] - forecast[mask]).mean() / scale)
        rows.append(
            {
                "dataset_id": str(row.dataset_id),
                "series_id": str(row.series_id),
                "origin": int(row.origin),
                "b3_normalized_loss": loss,
            }
        )
    return pd.DataFrame(rows).sort_values(
        list(CASE_KEYS), kind="mergesort"
    ).reset_index(drop=True)


def _loss_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    required = [*CASE_KEYS, column]
    missing = [item for item in required if item not in frame]
    if missing:
        raise ValueError(f"policy result is missing {missing}")
    result = frame.loc[:, required].copy()
    if bool(result.duplicated(list(CASE_KEYS), keep=False).any()):
        raise ValueError(f"policy result {column} has duplicate case keys")
    values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=np.float64)
    if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValueError(f"policy result {column} must be finite and nonnegative")
    result[column] = values
    return result


def _merge_loss(base: pd.DataFrame, addition: pd.DataFrame, column: str) -> pd.DataFrame:
    result = base.merge(
        _loss_column(addition, column),
        on=list(CASE_KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(result) != len(base):
        raise ValueError(f"policy result {column} does not cover the target panel")
    return result.sort_values(list(CASE_KEYS), kind="mergesort").reset_index(drop=True)


def _ri_mapping(comparison: Mapping[str, object]) -> dict[str, float]:
    directions = comparison.get("directions")
    if not isinstance(directions, Mapping) or set(directions) != set(DATASET_ORDER):
        raise ValueError("loss comparison lacks the two target datasets")
    return {
        dataset: float(directions[dataset]["ri_percent"])
        for dataset in DATASET_ORDER
    }


def _finish(
    report: dict[str, object],
    tables: dict[str, pd.DataFrame],
    *,
    stopped_after: str,
    final_verdict: str | None,
    next_action: str,
) -> dict[str, object]:
    if final_verdict is not None and final_verdict not in FINAL_VERDICT_TOKENS:
        raise AssertionError("final verdict is outside the preregistered token set")
    report.update(
        {
            "stopped_after": stopped_after,
            "final_verdict": final_verdict,
            "next_action": next_action,
            "terminal": final_verdict is not None,
        }
    )
    return {"report": _json_safe(report), "tables": tables}


_GATE2_CHECKS = {
    "macro_effect",
    "direction_safety",
    "macro_absolute_usefulness",
    "direction_absolute_usefulness",
    "macro_ci",
    "dataset_ci",
}


def _seed0_has_borderline_trigger(gate2: Mapping[str, object], macro: float) -> bool:
    raw_interval = gate2.get("macro_ci95_percent")
    if not isinstance(raw_interval, Sequence) or isinstance(raw_interval, (str, bytes)):
        raise TypeError("Gate2 macro CI must be a two-value sequence")
    if len(raw_interval) != 2:
        raise ValueError("Gate2 macro CI must contain two values")
    lower = _finite_provenance("Gate2 macro CI lower", raw_interval[0])
    upper = _finite_provenance("Gate2 macro CI upper", raw_interval[1])
    if lower > upper:
        raise ValueError("Gate2 macro CI lower exceeds upper")
    raw_directions = gate2.get("m1_vs_b4_percent")
    if not isinstance(raw_directions, Mapping) or set(raw_directions) != set(
        DATASET_ORDER
    ):
        raise ValueError("Gate2 directions must contain exactly m5 and favorita")
    directions = [
        _finite_provenance(
            f"Gate2 {dataset} m1_vs_b4_percent", raw_directions[dataset]
        )
        for dataset in DATASET_ORDER
    ]
    return bool(
        0.0 < macro <= 0.40
        or lower <= 0.0 <= upper
        or lower <= 0.20 <= upper
        or sum(value > 0.0 for value in directions) == 1
    )
def _seed0_gate2_resolution(gate2: Mapping[str, object]) -> str:
    """Classify seed-0 Gate 2 without making §39 triggers unreachable."""

    if not isinstance(gate2, Mapping):
        raise TypeError("Gate2 result must be a mapping")
    passed = gate2.get("passed")
    if not isinstance(passed, (bool, np.bool_)):
        raise ValueError("Gate2 result must contain a boolean passed decision")
    macro = _finite_provenance(
        "Gate2 m1_vs_b4_macro_percent",
        gate2.get("m1_vs_b4_macro_percent"),
    )
    checks = gate2.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != _GATE2_CHECKS:
        raise ValueError("Gate2 result must contain exactly the six frozen checks")
    if any(not isinstance(value, (bool, np.bool_)) for value in checks.values()):
        raise ValueError("Gate2 checks must be Boolean")
    failed = {name for name, value in checks.items() if not bool(value)}
    if bool(passed) != (not failed):
        raise ValueError("Gate2 passed decision disagrees with its six checks")
    if passed:
        return "PROCEED"
    if macro <= 0.0:
        return "FAIL"
    return (
        "PENDING_GATE4"
        if _seed0_has_borderline_trigger(gate2, macro)
        else "FAIL"
    )


def _analyze_seed(
    dataset_outputs: Mapping[str, Mapping[str, object]],
    *,
    model_seed: int,
    apply_seed0_gate: bool,
    heterogeneous_diagnostic_factory: Callable[
        [Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]],
        Mapping[str, object],
    ]
    | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    analysis_seed: int = ANALYSIS_SEED,
    control_seed: int = CONTROL_SEED,
) -> dict[str, object]:
    """Analyze one model seed with strict failure-first policy gates."""

    selected_model_seed = _integer("model_seed", model_seed)
    if apply_seed0_gate and selected_model_seed != 0:
        raise ValueError("the seed-0 gate can only be applied to model seed zero")
    outputs = _validated_outputs(
        dataset_outputs, expected_model_seed=selected_model_seed
    )
    draws = _integer("bootstrap_draws", bootstrap_draws, minimum=1)
    fixed_analysis_seed = _integer("analysis_seed", analysis_seed)
    fixed_control_seed = _integer("control_seed", control_seed)
    steps = {
        dataset: _evaluation_frame(outputs[dataset], "step_predictions")
        for dataset in DATASET_ORDER
    }
    expert_losses = pd.concat(
        [_evaluation_frame(outputs[dataset], "losses") for dataset in DATASET_ORDER],
        ignore_index=True,
    )
    ladders = oracle_ladder(
        pd.concat([steps[dataset] for dataset in DATASET_ORDER], ignore_index=True)
    )
    # The heterogeneous TSB/SBA fallback is secondary by preregistration.  Its
    # potentially expensive factory is therefore invoked only after the
    # Point/Hurdle Gate 0 has actually failed.
    gate0 = evaluate_gate0(ladders)
    if (
        apply_seed0_gate
        and not bool(gate0["passed"])
        and heterogeneous_diagnostic_factory is not None
    ):
        diagnostic = heterogeneous_diagnostic_factory(ladders, outputs)
        if not isinstance(diagnostic, Mapping):
            raise TypeError("heterogeneous diagnostic factory must return a mapping")
        gate0 = evaluate_gate0(ladders, heterogeneous_diagnostic=diagnostic)
    report: dict[str, object] = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": f"FULL_SEED{selected_model_seed}",
        "model_seed": selected_model_seed,
        "gate_order": ["GATE0"],
        "oracle_ladder": ladders,
        "gate0": gate0,
        "directions": {},
    }
    tables: dict[str, pd.DataFrame] = {
        "expert_evaluation_losses": expert_losses
    }
    if not bool(gate0["passed"]) and apply_seed0_gate:
        diagnostic = gate0.get("heterogeneous_diagnostic")
        heterogeneous_pass = bool(
            isinstance(diagnostic, Mapping) and diagnostic.get("passed", False)
        )
        verdict = decide_final_verdict(
            gate0_pass=False, heterogeneous_gate_pass=heterogeneous_pass
        )
        return _finish(
            report,
            tables,
            stopped_after="GATE0",
            final_verdict=verdict if apply_seed0_gate else None,
            next_action="STOP" if apply_seed0_gate else "ROBUSTNESS_FAIL",
        )

    recurrence = temporal_recurrence(
        expert_losses, bootstrap_draws=draws, seed=fixed_analysis_seed
    )
    report["gate_order"].append("GATE1A")
    report["gate1a"] = recurrence
    if not bool(recurrence["passed"]) and apply_seed0_gate:
        verdict = decide_final_verdict(gate0_pass=True, gate1a_pass=False)
        return _finish(
            report,
            tables,
            stopped_after="GATE1A",
            final_verdict=verdict if apply_seed0_gate else None,
            next_action="STOP" if apply_seed0_gate else "ROBUSTNESS_FAIL",
        )

    cases = {dataset: outputs[dataset]["cases"] for dataset in DATASET_ORDER}
    schedules = {dataset: outputs[dataset]["schedule"] for dataset in DATASET_ORDER}
    direction_specs = (("m5", "favorita"), ("favorita", "m5"))
    target_policy: dict[str, pd.DataFrame] = {}
    selected_b4: dict[str, tuple[float, int]] = {}
    direction_reports: dict[str, dict[str, object]] = {}
    for source, target in direction_specs:
        source_schedule = schedules[source]
        target_schedule = schedules[target]
        assert isinstance(source_schedule, Mapping)
        assert isinstance(target_schedule, Mapping)
        source_cases = cases[source]
        target_cases = cases[target]
        assert isinstance(source_cases, pd.DataFrame)
        assert isinstance(target_cases, pd.DataFrame)
        b3 = tune_b3_source(
            steps[source],
            evaluation_origins=source_schedule["evaluation_origins"],
        )
        b4 = tune_b4_source(
            source_cases,
            warmup_origin=int(source_schedule["warmup_origin"]),
            evaluation_origins=source_schedule["evaluation_origins"],
            horizon=int(source_schedule["horizon"]),
        )
        eta = float(b4["eta"])
        half_life = int(b4["half_life"])
        selected_b4[source] = (eta, half_life)
        target_eval_cases = target_cases.loc[
            target_cases["origin"].isin(target_schedule["evaluation_origins"])
        ].copy()
        b3_losses = _blend_loss_frame(target_eval_cases, float(b3["alpha"]))
        b4_evaluated = evaluate_b4_cases(
            target_cases,
            warmup_origin=int(target_schedule["warmup_origin"]),
            evaluation_origins=target_schedule["evaluation_origins"],
            horizon=int(target_schedule["horizon"]),
            eta=eta,
            half_life=half_life,
        )
        target_policy[target] = _merge_loss(
            b3_losses, b4_evaluated, "b4_normalized_loss"
        )
        label = f"{source}_to_{target}"
        direction_reports[label] = {
            "source_dataset": source,
            "target_dataset": target,
            "b3": b3,
            "b4": b4,
        }

    b4_policy = pd.concat(
        [target_policy[dataset] for dataset in DATASET_ORDER], ignore_index=True
    )
    b4_comparison = summarize_loss_comparison(
        b4_policy,
        candidate_column="b4_normalized_loss",
        baseline_column="b3_normalized_loss",
    )
    b4_ri = _ri_mapping(b4_comparison)
    gate1b = evaluate_gate1b(b4_ri)
    report["gate_order"].append("GATE1B")
    report["directions"] = direction_reports
    report["b4_vs_b3"] = b4_comparison
    report["gate1b"] = gate1b
    tables["target_b3_b4_losses"] = b4_policy
    if not bool(gate1b["passed"]) and apply_seed0_gate:
        verdict = decide_final_verdict(
            gate0_pass=True, gate1a_pass=True, gate1b_pass=False
        )
        return _finish(
            report,
            tables,
            stopped_after="GATE1B",
            final_verdict=verdict if apply_seed0_gate else None,
            next_action="STOP" if apply_seed0_gate else "ROBUSTNESS_FAIL",
        )

    controls: dict[str, dict[str, pd.DataFrame]] = {}
    for source, target in direction_specs:
        source_schedule = schedules[source]
        target_schedule = schedules[target]
        assert isinstance(source_schedule, Mapping)
        assert isinstance(target_schedule, Mapping)
        source_cases = cases[source]
        target_cases = cases[target]
        assert isinstance(source_cases, pd.DataFrame)
        assert isinstance(target_cases, pd.DataFrame)
        eta, half_life = selected_b4[source]
        m1_tuning = tune_m1_source(
            source_cases,
            warmup_origin=int(source_schedule["warmup_origin"]),
            evaluation_origins=source_schedule["evaluation_origins"],
            horizon=int(source_schedule["horizon"]),
            lookback=int(source_schedule["lookback"]),
            eta=eta,
            half_life=half_life,
        )
        common = {
            "warmup_origin": int(target_schedule["warmup_origin"]),
            "evaluation_origins": target_schedule["evaluation_origins"],
            "horizon": int(target_schedule["horizon"]),
            "lookback": int(target_schedule["lookback"]),
            "eta": eta,
            "half_life": half_life,
            "k": int(m1_tuning["k"]),
            "lambda_max": float(m1_tuning["lambda_max"]),
        }
        target_neighbor_plan = build_m1_neighbor_plan(
            target_cases,
            warmup_origin=common["warmup_origin"],
            evaluation_origins=common["evaluation_origins"],
            horizon=common["horizon"],
            lookback=common["lookback"],
            max_k=common["k"],
        )
        m1 = evaluate_m1_cases(
            target_cases, **common, neighbor_plan=target_neighbor_plan
        )
        c0 = evaluate_c0_cases(
            target_cases,
            **common,
            random_seed=fixed_control_seed,
            neighbor_plan=target_neighbor_plan,
        )
        c1 = evaluate_c1_cases(
            target_cases,
            **common,
            random_seed=fixed_control_seed,
            neighbor_plan=target_neighbor_plan,
        )
        target_policy[target] = _merge_loss(
            target_policy[target], m1, "m1_normalized_loss"
        )
        controls[target] = {"c0": c0, "c1": c1}
        label = f"{source}_to_{target}"
        direction_reports[label]["m1"] = m1_tuning

    policy_losses = pd.concat(
        [target_policy[dataset] for dataset in DATASET_ORDER], ignore_index=True
    )
    m1_vs_b4 = summarize_loss_comparison(
        policy_losses,
        candidate_column="m1_normalized_loss",
        baseline_column="b4_normalized_loss",
    )
    m1_vs_b3 = summarize_loss_comparison(
        policy_losses,
        candidate_column="m1_normalized_loss",
        baseline_column="b3_normalized_loss",
    )
    m1_vs_b4_ri = _ri_mapping(m1_vs_b4)
    m1_vs_b3_ri = _ri_mapping(m1_vs_b3)
    bootstrap = paired_series_cluster_bootstrap(
        policy_losses,
        candidate_column="m1_normalized_loss",
        baseline_column="b4_normalized_loss",
        draws=draws,
        seed=fixed_analysis_seed,
    )
    gate2 = evaluate_gate2(
        m1_vs_b4_percent=m1_vs_b4_ri,
        m1_vs_b3_percent=m1_vs_b3_ri,
        bootstrap=bootstrap,
    )
    report["gate_order"].append("GATE2")
    report["m1_vs_b4"] = m1_vs_b4
    report["m1_vs_b3"] = m1_vs_b3
    report["bootstrap"] = bootstrap
    gate2 = dict(gate2)
    gate2_resolution = (
        _seed0_gate2_resolution(gate2) if apply_seed0_gate else "DIAGNOSTIC_ONLY"
    )
    gate2["decision_state"] = gate2_resolution
    report["gate2"] = gate2
    tables["target_policy_losses"] = policy_losses
    if gate2_resolution == "FAIL":
        verdict = decide_final_verdict(
            gate0_pass=True,
            gate1a_pass=True,
            gate1b_pass=True,
            gate2_pass=False,
        )
        return _finish(
            report,
            tables,
            stopped_after="GATE2",
            final_verdict=verdict if apply_seed0_gate else None,
            next_action="STOP" if apply_seed0_gate else "ROBUSTNESS_FAIL",
        )

    safety = evaluate_gate3_safety(policy_losses)
    report["gate_order"].append("GATE3_SAFETY")
    report["gate3_safety"] = safety
    if not bool(safety["passed"]) and apply_seed0_gate:
        provisional_gate2_failed = not bool(gate2["passed"])
        if provisional_gate2_failed:
            gate2["decision_state"] = "FINAL_FAIL"
            gate2["deferral_withdrawn_by"] = "GATE3_SAFETY"
            verdict = decide_final_verdict(
                gate0_pass=True,
                gate1a_pass=True,
                gate1b_pass=True,
                gate2_pass=False,
            )
        else:
            verdict = decide_final_verdict(
                gate0_pass=True,
                gate1a_pass=True,
                gate1b_pass=True,
                gate2_pass=True,
                gate3_safety_pass=False,
            )
        return _finish(
            report,
            tables,
            stopped_after="GATE3_SAFETY",
            final_verdict=verdict if apply_seed0_gate else None,
            next_action="STOP" if apply_seed0_gate else "ROBUSTNESS_FAIL",
        )

    control_frames: dict[str, list[pd.DataFrame]] = {"c0": [], "c1": []}
    for dataset in DATASET_ORDER:
        for control_name in ("c0", "c1"):
            renamed = _loss_column(
                controls[dataset][control_name], "m1_normalized_loss"
            ).rename(columns={"m1_normalized_loss": f"{control_name}_normalized_loss"})
            control_frames[control_name].append(
                _merge_loss(
                    target_policy[dataset].loc[
                        :, [*CASE_KEYS, "b4_normalized_loss"]
                    ],
                    renamed,
                    f"{control_name}_normalized_loss",
                )
            )
    c0_losses = pd.concat(control_frames["c0"], ignore_index=True)
    c1_losses = pd.concat(control_frames["c1"], ignore_index=True)
    c0_comparison = summarize_loss_comparison(
        c0_losses,
        candidate_column="c0_normalized_loss",
        baseline_column="b4_normalized_loss",
    )
    c1_comparison = summarize_loss_comparison(
        c1_losses,
        candidate_column="c1_normalized_loss",
        baseline_column="b4_normalized_loss",
    )
    control_gate = evaluate_gate3_control(
        real_percent=m1_vs_b4_ri,
        shuffled_percent=_ri_mapping(c0_comparison),
        random_percent=_ri_mapping(c1_comparison),
    )
    report["gate_order"].append("GATE3_CONTROL")
    report["controls"] = {"c0_vs_b4": c0_comparison, "c1_vs_b4": c1_comparison}
    report["gate3_control"] = control_gate
    tables["c0_losses"] = c0_losses
    tables["c1_losses"] = c1_losses
    if not bool(control_gate["passed"]) and apply_seed0_gate:
        provisional_gate2_failed = not bool(gate2["passed"])
        if provisional_gate2_failed:
            gate2["decision_state"] = "FINAL_FAIL"
            gate2["deferral_withdrawn_by"] = "GATE3_CONTROL"
            verdict = decide_final_verdict(
                gate0_pass=True,
                gate1a_pass=True,
                gate1b_pass=True,
                gate2_pass=False,
            )
        else:
            verdict = decide_final_verdict(
                gate0_pass=True,
                gate1a_pass=True,
                gate1b_pass=True,
                gate2_pass=True,
                gate3_safety_pass=True,
                gate3_control_pass=False,
            )
        return _finish(
            report,
            tables,
            stopped_after="GATE3_CONTROL",
            final_verdict=verdict if apply_seed0_gate else None,
            next_action="STOP" if apply_seed0_gate else "ROBUSTNESS_FAIL",
        )

    if not apply_seed0_gate:
        return _finish(
            report,
            tables,
            stopped_after="GATE3_CONTROL",
            final_verdict=None,
            next_action="SEED_POLICY_READY",
        )

    gate4 = evaluate_gate4_seed0(
        macro_ri_percent=float(m1_vs_b4["macro_ri_percent"]),
        ci95_percent=bootstrap["macro"]["ci95_percent"],
        direction_ri_percent=m1_vs_b4_ri,
        safety_pass=True,
        control_pass=True,
    )
    report["gate_order"].append("GATE4_SEED0")
    report["gate4"] = gate4
    action = str(gate4["action"])
    if action == "RUN_SEED1":
        gate2["decision_state"] = "PENDING_GATE4"
        return _finish(
            report,
            tables,
            stopped_after="GATE4_SEED0",
            final_verdict=None,
            next_action="RUN_SEED1",
        )
    if action == "ACCEPT_SEED0":
        if not bool(gate2["passed"]):
            raise RuntimeError("Gate4 cannot accept seed 0 before Gate2 passes")
        gate2["decision_state"] = "FINAL_PASS"
        verdict = decide_final_verdict(
            gate0_pass=True,
            gate1a_pass=True,
            gate1b_pass=True,
            gate2_pass=True,
            gate3_safety_pass=True,
            gate3_control_pass=True,
            gate4_pass=True,
        )
        return _finish(
            report,
            tables,
            stopped_after="GATE4_SEED0",
            final_verdict=verdict,
            next_action="STOP",
        )
    if action == "STOP_NO_ADDITIONAL_SEED":
        verdict = decide_final_verdict(
            gate0_pass=True,
            gate1a_pass=True,
            gate1b_pass=True,
            gate2_pass=True,
            gate3_safety_pass=True,
            gate3_control_pass=True,
            gate4_pass=False,
        )
        return _finish(
            report,
            tables,
            stopped_after="GATE4_SEED0",
            final_verdict=verdict,
            next_action="STOP",
        )
    raise ValueError(f"unknown Gate4 seed0 action: {action}")


def analyze_seed0(
    dataset_outputs: Mapping[str, Mapping[str, object]],
    *,
    heterogeneous_diagnostic_factory: Callable[
        [Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]],
        Mapping[str, object],
    ]
    | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    analysis_seed: int = ANALYSIS_SEED,
    control_seed: int = CONTROL_SEED,
) -> dict[str, object]:
    """Apply the preregistered seed-0 gates in failure-first order."""

    return _analyze_seed(
        dataset_outputs,
        model_seed=0,
        apply_seed0_gate=True,
        heterogeneous_diagnostic_factory=heterogeneous_diagnostic_factory,
        bootstrap_draws=bootstrap_draws,
        analysis_seed=analysis_seed,
        control_seed=control_seed,
    )


def _require_frozen_analysis_parameters(
    bootstrap_draws: object, analysis_seed: object, control_seed: object
) -> None:
    if _integer("bootstrap_draws", bootstrap_draws, minimum=1) != BOOTSTRAP_DRAWS:
        raise ValueError(f"full runs require exactly {BOOTSTRAP_DRAWS} bootstrap draws")
    if _integer("analysis_seed", analysis_seed) != ANALYSIS_SEED:
        raise ValueError(f"full runs require analysis seed {ANALYSIS_SEED}")
    if _integer("control_seed", control_seed) != CONTROL_SEED:
        raise ValueError(f"full runs require control seed {CONTROL_SEED}")


def _population_heterogeneous_factory(
    populations: Mapping[str, Mapping[str, object]],
) -> Callable[
    [Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]],
    Mapping[str, object],
]:
    """Defer importing and running the classical diagnostic until Gate 0 fails."""

    def evaluate(
        ladders: Mapping[str, Mapping[str, object]],
        dataset_outputs: Mapping[str, Mapping[str, object]],
    ) -> Mapping[str, object]:
        # Import inside the callback so the classical path does no work on a
        # passing Point/Hurdle Gate 0.
        from .heterogeneous import make_heterogeneous_diagnostic_factory

        factory = make_heterogeneous_diagnostic_factory(populations)
        return factory(ladders, dataset_outputs)

    return evaluate


def run_full_seed0(
    populations: Mapping[str, Mapping[str, object]],
    device: torch.device,
    *,
    heterogeneous_diagnostic_factory: Callable[
        [Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]],
        Mapping[str, object],
    ]
    | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    analysis_seed: int = ANALYSIS_SEED,
    control_seed: int = CONTROL_SEED,
) -> dict[str, object]:
    """Fit both datasets sequentially, then execute the seed-0 gate ladder."""

    _require_frozen_analysis_parameters(
        bootstrap_draws, analysis_seed, control_seed
    )
    if not isinstance(populations, Mapping) or set(populations) != set(DATASET_ORDER):
        raise ValueError("populations must contain exactly m5 and favorita")
    resolved_diagnostic_factory = (
        _population_heterogeneous_factory(populations)
        if heterogeneous_diagnostic_factory is None
        else heterogeneous_diagnostic_factory
    )
    outputs: dict[str, Mapping[str, object]] = {}
    for dataset in DATASET_ORDER:
        outputs[dataset] = train_dataset_experts(
            populations[dataset], device, model_seed=0
        )
    analysis = analyze_seed0(
        outputs,
        heterogeneous_diagnostic_factory=resolved_diagnostic_factory,
        bootstrap_draws=bootstrap_draws,
        analysis_seed=analysis_seed,
        control_seed=control_seed,
    )
    return {"dataset_outputs": outputs, **analysis}


def _run_additional_seed(
    populations: Mapping[str, Mapping[str, object]],
    device: torch.device,
    *,
    model_seed: int,
    heterogeneous_diagnostic_factory: Callable[
        [Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]],
        Mapping[str, object],
    ]
    | None,
    bootstrap_draws: int,
    analysis_seed: int,
    control_seed: int,
) -> dict[str, object]:
    seed = _integer("model_seed", model_seed, minimum=1)
    if seed not in {1, 2}:
        raise ValueError("only preregistered additional model seeds 1 and 2 are allowed")
    if not isinstance(populations, Mapping) or set(populations) != set(DATASET_ORDER):
        raise ValueError("populations must contain exactly m5 and favorita")
    outputs: dict[str, Mapping[str, object]] = {}
    for dataset in DATASET_ORDER:
        outputs[dataset] = train_dataset_experts(
            populations[dataset], device, model_seed=seed
        )
    analysis = _analyze_seed(
        outputs,
        model_seed=seed,
        apply_seed0_gate=False,
        heterogeneous_diagnostic_factory=heterogeneous_diagnostic_factory,
        bootstrap_draws=bootstrap_draws,
        analysis_seed=analysis_seed,
        control_seed=control_seed,
    )
    return {"dataset_outputs": outputs, **analysis}


def _seed_policy_losses(seed_result: Mapping[str, object], model_seed: int) -> pd.DataFrame:
    tables = seed_result.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError(f"seed {model_seed} result lacks analysis tables")
    frame = tables.get("target_policy_losses")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"seed {model_seed} lacks target policy losses")
    required = [
        *CASE_KEYS,
        "m1_normalized_loss",
        "b4_normalized_loss",
        "b3_normalized_loss",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"seed {model_seed} policy losses are missing {missing}")
    result = frame.loc[:, required].copy()
    if bool(result.duplicated(list(CASE_KEYS), keep=False).any()):
        raise ValueError(f"seed {model_seed} policy losses have duplicate cases")
    result.insert(0, "model_seed", int(model_seed))
    return result


def seed_policy_losses(
    seed_result: Mapping[str, object], model_seed: int
) -> pd.DataFrame:
    """Return the frozen Gate-4 loss panel for one completed model seed."""

    return _seed_policy_losses(seed_result, model_seed)


def _robustness_verdict(passed: bool, *, averaged_gate2_pass: bool = True) -> str:
    return decide_final_verdict(
        gate0_pass=True,
        gate1a_pass=True,
        gate1b_pass=True,
        gate2_pass=bool(averaged_gate2_pass),
        gate3_safety_pass=True,
        gate3_control_pass=True,
        gate4_pass=bool(passed),
    )


def robustness_verdict(
    passed: bool, *, averaged_gate2_pass: bool = True
) -> str:
    """Map final Gate-4 and seed-average Gate-2 outcomes to one verdict."""

    return _robustness_verdict(
        passed, averaged_gate2_pass=averaged_gate2_pass
    )


def _protocol_result(
    seed_results: Mapping[int, Mapping[str, object]],
    *,
    final_verdict: str,
    robustness: Mapping[str, object] | None,
    seed_policy_losses: pd.DataFrame | None = None,
    averaged_gate2_analysis: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if final_verdict not in FINAL_VERDICT_TOKENS:
        raise AssertionError("protocol produced a non-preregistered final verdict")
    ordered_seeds = sorted(seed_results)
    report = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "FULL_PROTOCOL",
        "executed_model_seeds": ordered_seeds,
        "seed_reports": {
            str(seed): seed_results[seed]["report"] for seed in ordered_seeds
        },
        "robustness": robustness,
        "terminal": True,
        "next_action": "STOP",
        "final_verdict": final_verdict,
    }
    seed_average_losses: pd.DataFrame | None = None
    if averaged_gate2_analysis is not None:
        averaged = dict(averaged_gate2_analysis)
        raw_losses = averaged.pop("seed_average_losses", None)
        if not isinstance(raw_losses, pd.DataFrame) or raw_losses.empty:
            raise ValueError("averaged Gate2 analysis lacks seed-average losses")
        raw_gate2 = averaged.get("gate2")
        if not isinstance(raw_gate2, Mapping):
            raise ValueError("averaged Gate2 analysis lacks Gate2")
        report["averaged_gate2"] = raw_gate2
        report["seed_average_gate2_analysis"] = averaged
        seed_average_losses = raw_losses
    result: dict[str, object] = {
        "report": _json_safe(report),
        "seed_results": dict(seed_results),
        "tables": {},
    }
    if seed_policy_losses is not None:
        result["tables"] = {"seed_policy_losses": seed_policy_losses}
    if seed_average_losses is not None:
        result["tables"]["seed_average_policy_losses"] = seed_average_losses
    return result


def assemble_protocol_result(
    seed_results: Mapping[int, Mapping[str, object]],
    *,
    final_verdict: str,
    robustness: Mapping[str, object] | None,
    seed_policy_loss_frame: pd.DataFrame | None = None,
    averaged_gate2_analysis: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble the immutable-schema terminal result from persisted seeds."""

    return _protocol_result(
        seed_results,
        final_verdict=final_verdict,
        robustness=robustness,
        seed_policy_losses=seed_policy_loss_frame,
        averaged_gate2_analysis=averaged_gate2_analysis,
    )


def analyze_additional_seed(
    dataset_outputs: Mapping[str, Mapping[str, object]],
    *,
    model_seed: int,
    heterogeneous_diagnostic_factory: Callable[
        [Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]],
        Mapping[str, object],
    ]
    | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    analysis_seed: int = ANALYSIS_SEED,
    control_seed: int = CONTROL_SEED,
) -> dict[str, object]:
    """Build seed-1/2 policy losses with upstream gates diagnostic-only."""

    seed = _integer("model_seed", model_seed, minimum=1)
    if seed not in {1, 2}:
        raise ValueError("only preregistered additional model seeds 1 and 2 are allowed")
    _require_frozen_analysis_parameters(
        bootstrap_draws, analysis_seed, control_seed
    )
    return _analyze_seed(
        dataset_outputs,
        model_seed=seed,
        apply_seed0_gate=False,
        heterogeneous_diagnostic_factory=heterogeneous_diagnostic_factory,
        bootstrap_draws=bootstrap_draws,
        analysis_seed=analysis_seed,
        control_seed=control_seed,
    )


def run_full_protocol(
    populations: Mapping[str, Mapping[str, object]],
    device: torch.device,
    *,
    heterogeneous_diagnostic_factory: Callable[
        [Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]],
        Mapping[str, object],
    ]
    | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    analysis_seed: int = ANALYSIS_SEED,
    control_seed: int = CONTROL_SEED,
) -> dict[str, object]:
    """Run seeds 0, 1, and 2 only when the preceding Gate 4 requests them."""

    _require_frozen_analysis_parameters(
        bootstrap_draws, analysis_seed, control_seed
    )
    if not isinstance(populations, Mapping) or set(populations) != set(DATASET_ORDER):
        raise ValueError("populations must contain exactly m5 and favorita")
    resolved_diagnostic_factory = (
        _population_heterogeneous_factory(populations)
        if heterogeneous_diagnostic_factory is None
        else heterogeneous_diagnostic_factory
    )
    seed_results: dict[int, Mapping[str, object]] = {}
    seed0 = run_full_seed0(
        populations,
        device,
        heterogeneous_diagnostic_factory=resolved_diagnostic_factory,
        bootstrap_draws=bootstrap_draws,
        analysis_seed=analysis_seed,
        control_seed=control_seed,
    )
    seed_results[0] = seed0
    seed0_report = seed0.get("report")
    if not isinstance(seed0_report, Mapping):
        raise ValueError("seed 0 result lacks its JSON-safe report")
    seed0_verdict = seed0_report.get("final_verdict")
    if seed0_verdict is not None:
        return _protocol_result(
            seed_results,
            final_verdict=str(seed0_verdict),
            robustness=seed0_report.get("gate4")
            if isinstance(seed0_report.get("gate4"), Mapping)
            else None,
        )
    if seed0_report.get("next_action") != "RUN_SEED1":
        raise ValueError("nonterminal seed 0 result must request RUN_SEED1")

    seed1 = _run_additional_seed(
        populations,
        device,
        model_seed=1,
        heterogeneous_diagnostic_factory=resolved_diagnostic_factory,
        bootstrap_draws=bootstrap_draws,
        analysis_seed=analysis_seed,
        control_seed=control_seed,
    )
    seed_results[1] = seed1
    seed1_report = seed1.get("report")
    if not isinstance(seed1_report, Mapping):
        raise ValueError("seed 1 result lacks its JSON-safe report")
    if seed1_report.get("next_action") != "SEED_POLICY_READY":
        raise RuntimeError(
            "additional seed 1 did not produce the required policy-loss panel"
        )

    stacked = pd.concat(
        [_seed_policy_losses(seed0, 0), _seed_policy_losses(seed1, 1)],
        ignore_index=True,
    )
    gate4_seed1 = evaluate_gate4_seed1(
        stacked,
        candidate_column="m1_normalized_loss",
        baseline_column="b4_normalized_loss",
        draws=bootstrap_draws,
        seed=analysis_seed,
    )
    averaged_gate2 = evaluate_seed_average_gate2(
        stacked,
        expected_seeds=(0, 1),
        draws=bootstrap_draws,
        seed=analysis_seed,
    )
    averaged_gate2_pass = bool(averaged_gate2["gate2"]["passed"])
    seed1_action = str(gate4_seed1["action"])
    if seed1_action == "ACCEPT_TWO_SEED":
        return _protocol_result(
            seed_results,
            final_verdict=_robustness_verdict(
                True, averaged_gate2_pass=averaged_gate2_pass
            ),
            robustness=gate4_seed1,
            seed_policy_losses=stacked,
            averaged_gate2_analysis=averaged_gate2,
        )
    if seed1_action == "RETRIEVAL_ROBUSTNESS_NO_GO":
        return _protocol_result(
            seed_results,
            final_verdict=_robustness_verdict(
                False, averaged_gate2_pass=averaged_gate2_pass
            ),
            robustness=gate4_seed1,
            seed_policy_losses=stacked,
            averaged_gate2_analysis=averaged_gate2,
        )
    if seed1_action != "RUN_SEED2":
        raise ValueError(f"unknown Gate4 seed1 action: {seed1_action}")

    seed2 = _run_additional_seed(
        populations,
        device,
        model_seed=2,
        heterogeneous_diagnostic_factory=resolved_diagnostic_factory,
        bootstrap_draws=bootstrap_draws,
        analysis_seed=analysis_seed,
        control_seed=control_seed,
    )
    seed_results[2] = seed2
    seed2_report = seed2.get("report")
    if not isinstance(seed2_report, Mapping):
        raise ValueError("seed 2 result lacks its JSON-safe report")
    if seed2_report.get("next_action") != "SEED_POLICY_READY":
        raise RuntimeError(
            "additional seed 2 did not produce the required policy-loss panel"
        )
    stacked = pd.concat(
        [stacked, _seed_policy_losses(seed2, 2)], ignore_index=True
    )
    gate4_seed2 = evaluate_gate4_seed2(
        stacked,
        candidate_column="m1_normalized_loss",
        baseline_column="b4_normalized_loss",
        draws=bootstrap_draws,
        seed=analysis_seed,
    )
    averaged_gate2 = evaluate_seed_average_gate2(
        stacked,
        expected_seeds=(0, 1, 2),
        draws=bootstrap_draws,
        seed=analysis_seed,
    )
    averaged_gate2_pass = bool(averaged_gate2["gate2"]["passed"])
    seed2_action = str(gate4_seed2["action"])
    if seed2_action not in {"ACCEPT_THREE_SEED", "RETRIEVAL_ROBUSTNESS_NO_GO"}:
        raise ValueError(f"unknown Gate4 seed2 action: {seed2_action}")
    return _protocol_result(
        seed_results,
        final_verdict=_robustness_verdict(
            bool(gate4_seed2["passed"]),
            averaged_gate2_pass=averaged_gate2_pass,
        ),
        robustness=gate4_seed2,
        seed_policy_losses=stacked,
        averaged_gate2_analysis=averaged_gate2,
    )


__all__ = [
    "analyze_additional_seed",
    "analyze_seed0",
    "assemble_protocol_result",
    "robustness_verdict",
    "run_full_protocol",
    "run_full_seed0",
    "seed_policy_losses",
    "train_dataset_experts",
]
