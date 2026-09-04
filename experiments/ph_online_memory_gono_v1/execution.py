"""Append-only, restart-safe execution for the frozen full protocol.

The numerical engine in :mod:`full_run` deliberately performs no writes.  This
module supplies the operational boundary: it authorizes a full run from the
bound smoke result, checkpoints each completed arm immediately, persists each
dataset and seed analysis atomically, and resumes only artifacts whose complete
hash chain verifies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

from .analysis import (
    ANALYSIS_SEED,
    BOOTSTRAP_DRAWS,
    evaluate_gate4_seed1,
    evaluate_gate4_seed2,
    evaluate_seed_average_gate2,
)
from .artifacts import (
    exclusive_torch_save,
    exclusive_write_json,
    exclusive_write_parquet,
    file_sha256,
    payload_sha256,
    verify_preregistration,
)
from .data import load_independent_population
from .evaluation import CONTROL_SEED
from .full_run import (
    analyze_additional_seed,
    analyze_seed0,
    assemble_protocol_result,
    robustness_verdict,
    seed_policy_losses,
    train_dataset_experts,
)


EXPERIMENT = "PH-ONLINE-MEMORY-GONO-v1"
DATASET_ORDER = ("m5", "favorita")
ARM_MODEL_IDS = {
    "point": "M0PM_point_mse_param_matched",
    "hurdle": "M1_factorized_mean",
}
FRAME_NAMES = ("predictions", "step_predictions", "losses", "cases")
COMPLETION_HASH_FIELD = "completion_payload_sha256"
ANALYSIS_HASH_FIELD = "analysis_completion_payload_sha256"
FINAL_HASH_FIELD = "final_gate_report_payload_sha256"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_object(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
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
        if not math.isfinite(result):
            raise ValueError("JSON artifacts cannot contain non-finite numbers")
        return result
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported JSON artifact value: {type(value).__name__}")


def _add_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        raise ValueError(f"reserved hash field already exists: {field}")
    result[field] = payload_sha256(result)
    return result


def _verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{label} self SHA-256 is missing")
    unhashed = dict(payload)
    unhashed.pop(field, None)
    if payload_sha256(unhashed) != expected:
        raise RuntimeError(f"{label} self SHA-256 mismatch")


def _confined_path(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path must be a nonempty relative string")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute():
        raise ValueError(f"{label} path must be relative")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / candidate_relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the result root") from exc
    return resolved


def _artifact_record(path: Path, relative_to: Path) -> dict[str, object]:
    source = Path(path)
    return {
        "path": source.relative_to(relative_to).as_posix(),
        "bytes": int(source.stat().st_size),
        "sha256": file_sha256(source),
    }


def _verify_artifact_records(
    base: Path, artifacts: object, *, label: str
) -> dict[str, Path]:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError(f"{label} artifact manifest is missing")
    verified: dict[str, Path] = {}
    seen: set[Path] = set()
    for name, raw in artifacts.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError(f"{label} artifact entries must be named objects")
        path = _confined_path(base, raw.get("path"), label=f"{label} {name}")
        if path in seen:
            raise ValueError(f"{label} artifact paths must be unique")
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = raw.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{label} artifact {name} has an invalid SHA-256")
        actual = file_sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"{label} artifact {name} SHA-256 mismatch")
        expected_bytes = raw.get("bytes")
        if expected_bytes is not None and (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or path.stat().st_size != expected_bytes
        ):
            raise RuntimeError(f"{label} artifact {name} byte count mismatch")
        verified[name] = path
    return verified


def validate_runtime_authorization(
    result_root: Path,
    preregistration: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Validate the sole authorization for starting full seed-0 execution."""

    root = Path(result_root)
    current = verify_preregistration(root / "preregistered_spec.json")
    if preregistration is not None:
        supplied_sha = preregistration.get("preregistration_sha256")
        if supplied_sha != current.get("preregistration_sha256"):
            raise RuntimeError("supplied preregistration is not the current frozen spec")
    prereg = current
    if prereg.get("experiment_name") != EXPERIMENT:
        raise RuntimeError("preregistration experiment identity mismatch")
    smoke_spec = prereg.get("smoke")
    if not isinstance(smoke_spec, Mapping):
        raise RuntimeError("frozen smoke specification is missing")
    frozen_threshold = _finite_real(
        "frozen runtime threshold", smoke_spec.get("runtime_gate_gpu_hours")
    )
    if frozen_threshold != 6.0:
        raise RuntimeError("frozen full-run GPU threshold must be exactly 6.0 hours")

    runtime_path = root / "runtime_estimate.json"
    runtime = _load_json_object(runtime_path)
    if runtime.get("experiment") != EXPERIMENT:
        raise RuntimeError("runtime estimate experiment identity mismatch")
    if runtime.get("stage") != "M5_200_SERIES_SMOKE":
        raise RuntimeError("runtime estimate stage identity mismatch")
    if runtime.get("preregistration_sha256") != prereg.get(
        "preregistration_sha256"
    ):
        raise RuntimeError("runtime estimate is not bound to the current preregistration")
    smoke_device = runtime.get("device")
    if not isinstance(smoke_device, str):
        raise RuntimeError("runtime estimate lacks CUDA smoke device provenance")
    try:
        parsed_smoke_device = torch.device(smoke_device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("runtime estimate has invalid CUDA device provenance") from exc
    if parsed_smoke_device.type != "cuda":
        raise RuntimeError("runtime estimate was not measured on a CUDA device")
    peak_memory = runtime.get("cuda_peak_memory_bytes")
    if (
        isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory <= 0
    ):
        raise RuntimeError("runtime estimate lacks positive CUDA peak-memory evidence")

    attempt = runtime.get("attempt")
    if not isinstance(attempt, Mapping):
        raise RuntimeError("runtime estimate lacks its smoke attempt binding")
    attempt_path = _confined_path(root, attempt.get("path"), label="smoke attempt")
    try:
        attempt_relative = attempt_path.relative_to(root.resolve())
    except ValueError as exc:  # covered by _confined_path; keeps type narrow here.
        raise RuntimeError("smoke attempt path is invalid") from exc
    if len(attempt_relative.parts) < 2 or attempt_relative.parts[0] != "smoke":
        raise RuntimeError("runtime estimate does not reference a smoke attempt")
    if attempt.get("id") != attempt_path.name:
        raise RuntimeError("runtime smoke attempt identity does not match its directory")

    verified = _verify_artifact_records(
        root, runtime.get("artifacts"), label="smoke"
    )
    for artifact in verified.values():
        try:
            artifact.relative_to(attempt_path)
        except ValueError as exc:
            raise RuntimeError(
                "runtime estimate references an artifact outside its smoke attempt"
            ) from exc

    gate = runtime.get("runtime_gate")
    if not isinstance(gate, Mapping):
        raise RuntimeError("runtime estimate lacks runtime_gate")
    if gate.get("action") != "CONTINUE_FULL_SEED0":
        raise RuntimeError("full execution requires action CONTINUE_FULL_SEED0")
    if gate.get("exceeded") is not False:
        raise RuntimeError("full execution requires exceeded to be exactly false")
    reported_threshold = _finite_real(
        "reported runtime threshold", gate.get("threshold_gpu_hours")
    )
    projected = _finite_real(
        "projected full seed-0 GPU hours", gate.get("projected_gpu_hours")
    )
    if reported_threshold != frozen_threshold:
        raise RuntimeError("runtime threshold drifted from the frozen 6.0-hour gate")
    if projected < 0.0 or projected > frozen_threshold:
        raise RuntimeError("projected GPU runtime exceeds the frozen 6.0-hour gate")

    return {
        "preregistration": prereg,
        "runtime_estimate": runtime,
        "runtime_estimate_sha256": file_sha256(runtime_path),
        "runtime_gate": dict(gate),
        "verified_smoke_artifacts": {
            name: {
                "path": path.relative_to(root.resolve()).as_posix(),
                "sha256": file_sha256(path),
            }
            for name, path in verified.items()
        },
    }


def _verify_upstream_artifacts(
    result_root: Path, preregistration: Mapping[str, Any]
) -> None:
    """Re-check every frozen prerequisite without importing protocol eagerly."""

    # Kept as a late import so protocol.py may expose this module through its CLI
    # without a module-initialization cycle.
    from . import protocol

    prereg = dict(preregistration)
    protocol._validate_phase0_authorization()
    protocol._verify_frozen_implementation(prereg)
    protocol._read_stage0_pass(Path(result_root), prereg)
    protocol._read_pre_smoke_pass(Path(result_root), prereg)
    protocol._verify_frozen_environment(prereg)
    protocol._verify_frozen_data_sources(prereg)
    integrity = protocol.verify_forbidden_artifacts(
        protocol.REPO_ROOT, protocol.PHASE0_EVIDENCE_PATHS[0]
    )
    if not isinstance(integrity, Mapping) or integrity.get("all_unchanged") is not True:
        raise RuntimeError("a forbidden Phase0 artifact directory changed")


def _reserve_attempt(parent: Path) -> Path:
    root = Path(parent)
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = root / f"attempt_{number:04d}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            number += 1


def _write_failure(attempt_root: Path, stage: str, error: BaseException) -> None:
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "stage": stage,
        "failed_at_utc": _utc_now(),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "retry_policy": "preserve this attempt; reserve the next attempt directory",
    }
    try:
        exclusive_write_json(Path(attempt_root) / "failure.json", payload)
    except FileExistsError:
        pass


def _validate_identity(dataset: str, model_seed: int) -> None:
    if dataset not in DATASET_ORDER:
        raise ValueError(f"unknown dataset: {dataset}")
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise TypeError("model_seed must be an integer")
    if model_seed not in {0, 1, 2}:
        raise ValueError("only frozen model seeds 0, 1, and 2 are allowed")


def _validated_arm_payload(
    payload: Mapping[str, object], *, dataset: str, model_seed: int, arm: str
) -> dict[str, object]:
    _validate_identity(dataset, model_seed)
    if arm not in ARM_MODEL_IDS:
        raise ValueError(f"unknown arm: {arm}")
    if not isinstance(payload, Mapping):
        raise TypeError("arm payload must be a mapping")
    expected = {
        "dataset": dataset,
        "arm": arm,
        "model_id": ARM_MODEL_IDS[arm],
        "model_seed": model_seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{arm} payload has mismatched {key}")
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError(f"{arm} payload lacks state_dict")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("state_dict must map strings to tensors")
        if tensor.device.type != "cpu":
            raise ValueError("checkpoint tensors must already be on CPU")
        state[name] = tensor.detach().clone()
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{arm} payload lacks provenance")
    if (
        provenance.get("model_id") != ARM_MODEL_IDS[arm]
        or provenance.get("model_seed") != model_seed
        or provenance.get("checkpoint_device") != "cpu"
        or provenance.get("n_parameters") != 7056
    ):
        raise ValueError(f"{arm} provenance identity drifted")
    end_to_end = _finite_real(
        f"{arm} end_to_end_wall_seconds",
        provenance.get("end_to_end_wall_seconds"),
    )
    if end_to_end < 0.0:
        raise ValueError(f"{arm} end_to_end_wall_seconds must be nonnegative")
    train_seconds = _finite_real(
        f"{arm} train_seconds", provenance.get("train_seconds")
    )
    if train_seconds < 0.0:
        raise ValueError(f"{arm} train_seconds must be nonnegative")
    if provenance.get("execution_device_type") != "cuda":
        raise ValueError(
            f"{arm} execution_device_type must be cuda for full execution"
        )
    raw_predictions = payload.get("predictions")
    if not isinstance(raw_predictions, Mapping) or not raw_predictions:
        raise ValueError(f"{arm} payload lacks prediction heads")
    predictions: dict[str, np.ndarray] = {}
    for name, value in raw_predictions.items():
        if not isinstance(name, str):
            raise TypeError("prediction head names must be strings")
        array = np.asarray(value)
        if array.size == 0 or array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError(f"prediction head {name} must be nonempty and finite")
        predictions[name] = np.ascontiguousarray(array).copy()
    return {
        "schema_version": 1,
        **expected,
        "state_dict": state,
        "provenance": dict(provenance),
        "predictions": predictions,
    }


def _persist_arm(
    attempt_root: Path,
    payload: Mapping[str, object],
    *,
    dataset: str,
    model_seed: int,
    arm: str,
    preregistration_sha256: str,
    runtime_estimate_sha256: str,
) -> dict[str, Any]:
    validated = _validated_arm_payload(
        payload, dataset=dataset, model_seed=model_seed, arm=arm
    )
    checkpoint_path = Path(attempt_root) / f"{arm}_checkpoint.pt"
    predictions_path = Path(attempt_root) / f"{arm}_prediction_heads.pt"
    provenance_path = Path(attempt_root) / f"{arm}_provenance.json"
    completion_path = Path(attempt_root) / f"{arm}_completion.json"
    exclusive_torch_save(checkpoint_path, validated["state_dict"])
    prediction_tensors = {
        name: torch.from_numpy(array.copy())
        for name, array in validated["predictions"].items()
    }
    exclusive_torch_save(predictions_path, prediction_tensors)
    exclusive_write_json(provenance_path, _json_safe(validated["provenance"]))
    artifacts = {
        "checkpoint": _artifact_record(checkpoint_path, attempt_root),
        "prediction_heads": _artifact_record(predictions_path, attempt_root),
        "provenance": _artifact_record(provenance_path, attempt_root),
    }
    manifest = _add_self_hash(
        {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "stage": "ARM_CHECKPOINT_COMPLETE",
            "dataset": dataset,
            "model_seed": model_seed,
            "arm": arm,
            "model_id": ARM_MODEL_IDS[arm],
            "preregistration_sha256": preregistration_sha256,
            "runtime_estimate_sha256": runtime_estimate_sha256,
            "completed_at_utc": _utc_now(),
            "artifacts": artifacts,
        },
        COMPLETION_HASH_FIELD,
    )
    exclusive_write_json(completion_path, manifest)
    return manifest


def _load_arm(
    attempt_root: Path,
    *,
    dataset: str,
    model_seed: int,
    arm: str,
    preregistration_sha256: str,
    runtime_estimate_sha256: str,
) -> dict[str, object] | None:
    completion_path = Path(attempt_root) / f"{arm}_completion.json"
    if not completion_path.exists():
        return None
    manifest = _load_json_object(completion_path)
    _verify_self_hash(manifest, COMPLETION_HASH_FIELD, f"{arm} completion")
    expected = {
        "experiment": EXPERIMENT,
        "stage": "ARM_CHECKPOINT_COMPLETE",
        "dataset": dataset,
        "model_seed": model_seed,
        "arm": arm,
        "model_id": ARM_MODEL_IDS[arm],
        "preregistration_sha256": preregistration_sha256,
        "runtime_estimate_sha256": runtime_estimate_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{arm} completion binding mismatch")
    artifacts = _verify_artifact_records(
        attempt_root, manifest.get("artifacts"), label=f"{arm} checkpoint"
    )
    if set(artifacts) != {"checkpoint", "prediction_heads", "provenance"}:
        raise RuntimeError(f"{arm} completion artifact set drifted")
    # torch 2.1 emits an internal TypedStorage deprecation warning while
    # materializing otherwise ordinary tensor-only weights.  Keep the narrow
    # compatibility suppression here; all payloads still use weights_only.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
        )
        state = torch.load(
            artifacts["checkpoint"], map_location="cpu", weights_only=True
        )
        prediction_tensors = torch.load(
            artifacts["prediction_heads"], map_location="cpu", weights_only=True
        )
    provenance = _load_json_object(artifacts["provenance"])
    if not isinstance(state, Mapping) or not isinstance(prediction_tensors, Mapping):
        raise RuntimeError(f"{arm} trusted local resume payload is invalid")
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "arm": arm,
        "model_id": ARM_MODEL_IDS[arm],
        "model_seed": model_seed,
        "state_dict": dict(state),
        "provenance": provenance,
        "predictions": {
            str(name): tensor.detach().cpu().numpy().copy()
            for name, tensor in prediction_tensors.items()
            if isinstance(tensor, torch.Tensor)
        },
    }
    if len(payload["predictions"]) != len(prediction_tensors):
        raise RuntimeError(f"{arm} prediction payload contains non-tensors")
    return _validated_arm_payload(
        payload, dataset=dataset, model_seed=model_seed, arm=arm
    )


def _latest_resumable_arms(
    dataset_root: Path,
    *,
    dataset: str,
    model_seed: int,
    preregistration_sha256: str,
    runtime_estimate_sha256: str,
) -> dict[str, Mapping[str, object]]:
    attempts = sorted(
        (path for path in Path(dataset_root).glob("attempt_*") if path.is_dir()),
        reverse=True,
    )
    result: dict[str, Mapping[str, object]] = {}
    for arm in ARM_MODEL_IDS:
        for attempt in attempts:
            payload = _load_arm(
                attempt,
                dataset=dataset,
                model_seed=model_seed,
                arm=arm,
                preregistration_sha256=preregistration_sha256,
                runtime_estimate_sha256=runtime_estimate_sha256,
            )
            if payload is not None:
                result[arm] = payload
                break
    return result


def _dataset_metadata(
    output: Mapping[str, object],
    preregistration: Mapping[str, Any],
    *,
    dataset: str,
    model_seed: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    schedule = output.get("schedule")
    population_manifest = output.get("population_manifest")
    provenance = output.get("provenance")
    if not isinstance(schedule, Mapping):
        raise ValueError("dataset output lacks schedule")
    if not isinstance(population_manifest, Mapping):
        raise ValueError("dataset output lacks population_manifest")
    if not isinstance(provenance, Mapping) or set(provenance) != set(ARM_MODEL_IDS):
        raise ValueError("dataset output lacks paired model provenance")
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "stage": "FULL_DATASET_OUTPUT",
        "dataset": dataset,
        "model_seed": model_seed,
        "schedule": _json_safe(schedule),
        "split_manifest": _json_safe(schedule),
        "series_eligibility": _json_safe(population_manifest),
        "population_manifest": _json_safe(population_manifest),
        "model_provenance": _json_safe(provenance),
        "resolved_config": {
            "lookback": schedule.get("lookback"),
            "horizon": schedule.get("horizon"),
            "train_origin_stride": schedule.get("train_origin_stride"),
            "model_train_end": schedule.get("model_train_end"),
            "training": _json_safe(preregistration.get("training", {})),
            "models": _json_safe(preregistration.get("models", {})),
            "split": _json_safe(
                preregistration.get("splits", {}).get(dataset, {})
                if isinstance(preregistration.get("splits"), Mapping)
                else {}
            ),
            "eligibility": _json_safe(preregistration.get("eligibility", {})),
        },
        "environment": _json_safe(preregistration.get("environment", {})),
        "runtime": {"dataset_wall_seconds": float(elapsed_seconds)},
    }


def _persist_dataset_completion(
    attempt_root: Path,
    output: Mapping[str, object],
    preregistration: Mapping[str, Any],
    *,
    result_root: Path,
    dataset: str,
    model_seed: int,
    runtime_estimate_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if output.get("dataset") != dataset or output.get("model_seed") != model_seed:
        raise ValueError("dataset output identity mismatch")
    prereg_sha = str(preregistration["preregistration_sha256"])
    for arm in ARM_MODEL_IDS:
        if not (Path(attempt_root) / f"{arm}_completion.json").is_file():
            raise RuntimeError(f"dataset cannot complete before {arm} checkpoint")
    state_dicts = output.get("state_dicts")
    if not isinstance(state_dicts, Mapping) or set(state_dicts) != set(ARM_MODEL_IDS):
        raise ValueError("dataset output lacks paired CPU state_dicts")
    output_provenance = output.get("provenance")
    if not isinstance(output_provenance, Mapping):
        raise ValueError("dataset output lacks paired model provenance")
    for arm, state in state_dicts.items():
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"dataset output {arm} state_dict is empty")
        if any(
            not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            for value in state.values()
        ):
            raise ValueError("dataset checkpoint tensors must be on CPU")
        persisted = _load_arm(
            attempt_root,
            dataset=dataset,
            model_seed=model_seed,
            arm=str(arm),
            preregistration_sha256=prereg_sha,
            runtime_estimate_sha256=runtime_estimate_sha256,
        )
        if persisted is None:
            raise RuntimeError(f"{arm} arm checkpoint disappeared before completion")
        persisted_state = persisted["state_dict"]
        if not isinstance(persisted_state, Mapping):
            raise RuntimeError(f"{arm} arm checkpoint state is invalid")
        if set(state) != set(persisted_state) or any(
            not torch.equal(state[name], persisted_state[name]) for name in state
        ):
            raise RuntimeError(
                f"dataset {arm} state_dict does not match its arm checkpoint"
            )
        if output_provenance.get(arm) != persisted["provenance"]:
            raise RuntimeError(
                f"dataset {arm} provenance does not match its arm checkpoint"
            )

    frame_paths: dict[str, Path] = {}
    for name in FRAME_NAMES:
        frame = output.get(name)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"dataset output {name} must be a DataFrame")
        path = Path(attempt_root) / f"{name}.parquet"
        exclusive_write_parquet(path, frame)
        frame_paths[name] = path

    metadata_path = Path(attempt_root) / "metadata.json"
    metadata = _dataset_metadata(
        output,
        preregistration,
        dataset=dataset,
        model_seed=model_seed,
        elapsed_seconds=elapsed_seconds,
    )
    exclusive_write_json(metadata_path, metadata)
    resume_path = Path(attempt_root) / "resume_payload.json"
    resume_payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "dataset": dataset,
        "model_seed": model_seed,
        "trust_boundary": (
            "local append-only files only; verify completion self-hash and every "
            "artifact SHA-256 before deserialization"
        ),
        "analysis_components": {
            "schedule_and_provenance": "metadata.json",
            **{name: f"{name}.parquet" for name in FRAME_NAMES},
        },
        "arm_components": {
            arm: {
                "checkpoint": f"{arm}_checkpoint.pt",
                "prediction_heads": f"{arm}_prediction_heads.pt",
                "provenance": f"{arm}_provenance.json",
                "completion": f"{arm}_completion.json",
            }
            for arm in ARM_MODEL_IDS
        },
    }
    exclusive_write_json(resume_path, resume_payload)

    artifact_paths = {
        **frame_paths,
        "metadata": metadata_path,
        "resume_payload": resume_path,
    }
    for arm in ARM_MODEL_IDS:
        artifact_paths.update(
            {
                f"{arm}_checkpoint": Path(attempt_root) / f"{arm}_checkpoint.pt",
                f"{arm}_prediction_heads": Path(attempt_root)
                / f"{arm}_prediction_heads.pt",
                f"{arm}_provenance": Path(attempt_root) / f"{arm}_provenance.json",
                f"{arm}_completion": Path(attempt_root) / f"{arm}_completion.json",
            }
        )
    artifacts = {
        name: _artifact_record(path, attempt_root)
        for name, path in artifact_paths.items()
    }
    completion = _add_self_hash(
        {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "stage": "FULL_DATASET_COMPLETE",
            "dataset": dataset,
            "model_seed": model_seed,
            "attempt_path": Path(attempt_root)
            .relative_to(result_root)
            .as_posix(),
            "preregistration_sha256": prereg_sha,
            "runtime_estimate_sha256": runtime_estimate_sha256,
            "completed_at_utc": _utc_now(),
            "artifacts": artifacts,
        },
        COMPLETION_HASH_FIELD,
    )
    # This marker is deliberately the final write in a successful attempt.
    exclusive_write_json(Path(attempt_root) / "completion.json", completion)
    return completion


@dataclass(frozen=True)
class CompletedDataset:
    output: dict[str, object]
    manifest: dict[str, Any]
    attempt_root: Path


def _completion_attempts(parent: Path, filename: str = "completion.json") -> list[Path]:
    return sorted(
        (
            path
            for path in Path(parent).glob("attempt_*")
            if path.is_dir() and (path / filename).is_file()
        ),
        reverse=True,
    )


def load_latest_completed_dataset(
    result_root: Path,
    *,
    dataset: str,
    model_seed: int,
    preregistration_sha256: str,
    runtime_estimate_sha256: str,
    analysis_only: bool = False,
) -> CompletedDataset | None:
    """Load the newest fully verified dataset attempt, or return ``None``."""

    _validate_identity(dataset, model_seed)
    parent = Path(result_root) / f"seed{model_seed}" / "datasets" / dataset
    attempts = _completion_attempts(parent)
    if not attempts:
        return None
    attempt = attempts[0]
    manifest = _load_json_object(attempt / "completion.json")
    _verify_self_hash(manifest, COMPLETION_HASH_FIELD, "dataset completion")
    expected = {
        "experiment": EXPERIMENT,
        "stage": "FULL_DATASET_COMPLETE",
        "dataset": dataset,
        "model_seed": model_seed,
        "preregistration_sha256": preregistration_sha256,
        "runtime_estimate_sha256": runtime_estimate_sha256,
        "attempt_path": attempt.relative_to(result_root).as_posix(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("dataset completion binding mismatch")
    artifacts = _verify_artifact_records(
        attempt, manifest.get("artifacts"), label="dataset"
    )
    required = {
        *FRAME_NAMES,
        "metadata",
        "resume_payload",
        *(
            f"{arm}_{suffix}"
            for arm in ARM_MODEL_IDS
            for suffix in (
                "checkpoint",
                "prediction_heads",
                "provenance",
                "completion",
            )
        ),
    }
    if set(artifacts) != required:
        raise RuntimeError("dataset completion artifact set drifted")
    metadata = _load_json_object(artifacts["metadata"])
    resume = _load_json_object(artifacts["resume_payload"])
    if (
        metadata.get("dataset") != dataset
        or metadata.get("model_seed") != model_seed
        or resume.get("dataset") != dataset
        or resume.get("model_seed") != model_seed
    ):
        raise RuntimeError("trusted local resume metadata identity mismatch")
    frames = {
        name: pd.read_parquet(artifacts[name])
        for name in FRAME_NAMES
        if not analysis_only or name != "predictions"
    }
    output: dict[str, object] = {
        "dataset": dataset,
        "model_seed": model_seed,
        "schedule": metadata["schedule"],
        "population_manifest": metadata["population_manifest"],
        "provenance": metadata["model_provenance"],
        **frames,
    }
    if not analysis_only:
        arm_payloads = {
            arm: _load_arm(
                attempt,
                dataset=dataset,
                model_seed=model_seed,
                arm=arm,
                preregistration_sha256=preregistration_sha256,
                runtime_estimate_sha256=runtime_estimate_sha256,
            )
            for arm in ARM_MODEL_IDS
        }
        if any(payload is None for payload in arm_payloads.values()):
            raise RuntimeError("completed dataset lacks a verified arm payload")
        output["state_dicts"] = {
            arm: payload["state_dict"]
            for arm, payload in arm_payloads.items()
            if payload is not None
        }
    return CompletedDataset(output=output, manifest=manifest, attempt_root=attempt)


def _analysis_view(output: Mapping[str, object]) -> dict[str, object]:
    required = (
        "dataset",
        "model_seed",
        "schedule",
        "step_predictions",
        "losses",
        "cases",
    )
    missing = [name for name in required if name not in output]
    if missing:
        raise ValueError(f"dataset output lacks analysis fields {missing}")
    return {name: output[name] for name in required}


def _train_or_resume_dataset(
    result_root: Path,
    preregistration: Mapping[str, Any],
    runtime_estimate_sha256: str,
    *,
    dataset: str,
    model_seed: int,
    device: torch.device,
    population_loader: Callable[..., Mapping[str, object]],
    train_dataset: Callable[..., Mapping[str, object]],
) -> CompletedDataset:
    prereg_sha = str(preregistration["preregistration_sha256"])
    completed = load_latest_completed_dataset(
        result_root,
        dataset=dataset,
        model_seed=model_seed,
        preregistration_sha256=prereg_sha,
        runtime_estimate_sha256=runtime_estimate_sha256,
        analysis_only=True,
    )
    if completed is not None:
        return completed

    dataset_root = Path(result_root) / f"seed{model_seed}" / "datasets" / dataset
    resumable = _latest_resumable_arms(
        dataset_root,
        dataset=dataset,
        model_seed=model_seed,
        preregistration_sha256=prereg_sha,
        runtime_estimate_sha256=runtime_estimate_sha256,
    )
    attempt = _reserve_attempt(dataset_root)
    started = time.perf_counter()
    try:
        # Copy verified prior arm payloads into the new self-contained attempt.
        for arm, payload in resumable.items():
            _persist_arm(
                attempt,
                payload,
                dataset=dataset,
                model_seed=model_seed,
                arm=arm,
                preregistration_sha256=prereg_sha,
                runtime_estimate_sha256=runtime_estimate_sha256,
            )

        def on_arm_complete(arm: str, payload: Mapping[str, object]) -> None:
            _persist_arm(
                attempt,
                payload,
                dataset=dataset,
                model_seed=model_seed,
                arm=arm,
                preregistration_sha256=prereg_sha,
                runtime_estimate_sha256=runtime_estimate_sha256,
            )

        min_positive = preregistration.get("eligibility", {}).get(
            "primary_min_positive_train"
        )
        if min_positive != 20:
            raise RuntimeError("frozen primary eligibility must be min_positive=20")
        population = population_loader(dataset, min_positive=20)
        try:
            output = train_dataset(
                population,
                device,
                model_seed=model_seed,
                persisted_arms=resumable,
                on_arm_complete=on_arm_complete,
            )
        finally:
            del population
            gc.collect()
        elapsed = float(time.perf_counter() - started)
        _persist_dataset_completion(
            attempt,
            output,
            preregistration,
            result_root=Path(result_root),
            dataset=dataset,
            model_seed=model_seed,
            runtime_estimate_sha256=runtime_estimate_sha256,
            elapsed_seconds=elapsed,
        )
        view = _analysis_view(output)
        del output
        gc.collect()
        manifest = _load_json_object(attempt / "completion.json")
        return CompletedDataset(output=view, manifest=manifest, attempt_root=attempt)
    except BaseException as error:
        _write_failure(attempt, "FULL_DATASET_ATTEMPT", error)
        raise


@dataclass(frozen=True)
class CompletedAnalysis:
    result: dict[str, object]
    manifest: dict[str, Any]
    attempt_root: Path


def _dataset_summary(completed: CompletedDataset) -> dict[str, object]:
    """Read the small reporting view without materializing persisted frames."""

    manifest = completed.manifest
    attempt = completed.attempt_root
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("metadata"), Mapping
    ):
        raise RuntimeError("dataset completion lacks metadata artifact")
    metadata_path = _verify_artifact_records(
        attempt,
        {"metadata": artifacts["metadata"]},
        label="dataset summary",
    )["metadata"]
    metadata = _load_json_object(metadata_path)
    dataset = manifest.get("dataset")
    model_seed = manifest.get("model_seed")
    if metadata.get("dataset") != dataset or metadata.get("model_seed") != model_seed:
        raise RuntimeError("dataset summary identity mismatch")
    return {
        "dataset": dataset,
        "model_seed": model_seed,
        "schedule": metadata.get("schedule"),
        "population_manifest": metadata.get("population_manifest"),
        "provenance": metadata.get("model_provenance"),
        "resolved_config": metadata.get("resolved_config"),
        "environment": metadata.get("environment"),
        "runtime": metadata.get("runtime"),
        "artifact": {
            "path": manifest.get("attempt_path"),
            "completion_payload_sha256": manifest.get(COMPLETION_HASH_FIELD),
        },
    }


def _validate_table_name(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("analysis table names must be nonempty strings")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
        raise ValueError(f"unsafe analysis table name: {name}")
    return name


def _persist_analysis(
    result_root: Path,
    result: Mapping[str, object],
    *,
    attempt_root: Path,
    model_seed: int,
    preregistration_sha256: str,
    runtime_estimate_sha256: str,
    datasets: Mapping[str, CompletedDataset],
) -> CompletedAnalysis:
    attempt = Path(attempt_root)
    try:
        report = result.get("report")
        tables = result.get("tables")
        if not isinstance(report, Mapping):
            raise TypeError("seed analysis report must be a mapping")
        if not isinstance(tables, Mapping):
            raise TypeError("seed analysis tables must be a mapping")
        if report.get("model_seed") != model_seed:
            raise ValueError("seed analysis report identity mismatch")
        report_path = attempt / "report.json"
        exclusive_write_json(report_path, _json_safe(report))
        artifact_paths: dict[str, Path] = {"report": report_path}
        for raw_name, frame in tables.items():
            name = _validate_table_name(raw_name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"analysis table {name} must be a DataFrame")
            path = attempt / f"{name}.parquet"
            exclusive_write_parquet(path, frame)
            artifact_paths[f"table_{name}"] = path
        source_datasets = {
            dataset: {
                "path": completed.attempt_root.relative_to(result_root).as_posix(),
                "completion_file_sha256": file_sha256(
                    completed.attempt_root / "completion.json"
                ),
                "completion_payload_sha256": completed.manifest[
                    COMPLETION_HASH_FIELD
                ],
            }
            for dataset, completed in datasets.items()
        }
        manifest = _add_self_hash(
            {
                "schema_version": 1,
                "experiment": EXPERIMENT,
                "stage": "SEED_ANALYSIS_COMPLETE",
                "model_seed": model_seed,
                "attempt_path": attempt.relative_to(result_root).as_posix(),
                "preregistration_sha256": preregistration_sha256,
                "runtime_estimate_sha256": runtime_estimate_sha256,
                "completed_at_utc": _utc_now(),
                "source_datasets": source_datasets,
                "artifacts": {
                    name: _artifact_record(path, attempt)
                    for name, path in artifact_paths.items()
                },
            },
            ANALYSIS_HASH_FIELD,
        )
        # This marker is deliberately the final write in a successful analysis.
        exclusive_write_json(attempt / "completion.json", manifest)
        return CompletedAnalysis(
            result={
                "dataset_outputs": {
                    dataset: _dataset_summary(completed)
                    for dataset, completed in datasets.items()
                },
                "report": dict(report),
                "tables": dict(tables),
            },
            manifest=manifest,
            attempt_root=attempt,
        )
    except BaseException as error:
        _write_failure(attempt, "SEED_ANALYSIS_ATTEMPT", error)
        raise


def _load_latest_analysis(
    result_root: Path,
    *,
    model_seed: int,
    preregistration_sha256: str,
    runtime_estimate_sha256: str,
) -> CompletedAnalysis | None:
    analysis_root = Path(result_root) / f"seed{model_seed}" / "analysis"
    attempts = _completion_attempts(analysis_root)
    if not attempts:
        return None
    attempt = attempts[0]
    manifest = _load_json_object(attempt / "completion.json")
    _verify_self_hash(manifest, ANALYSIS_HASH_FIELD, "analysis completion")
    expected = {
        "experiment": EXPERIMENT,
        "stage": "SEED_ANALYSIS_COMPLETE",
        "model_seed": model_seed,
        "attempt_path": attempt.relative_to(result_root).as_posix(),
        "preregistration_sha256": preregistration_sha256,
        "runtime_estimate_sha256": runtime_estimate_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("analysis completion binding mismatch")
    sources = manifest.get("source_datasets")
    if not isinstance(sources, Mapping) or set(sources) != set(DATASET_ORDER):
        raise RuntimeError("analysis dataset source manifest is incomplete")
    source_datasets: dict[str, CompletedDataset] = {}
    for dataset, raw in sources.items():
        if not isinstance(raw, Mapping):
            raise RuntimeError("analysis dataset source entry is invalid")
        dataset_root = _confined_path(
            result_root, raw.get("path"), label=f"analysis source {dataset}"
        )
        completion_path = dataset_root / "completion.json"
        if not completion_path.is_file():
            raise FileNotFoundError(completion_path)
        if file_sha256(completion_path) != raw.get("completion_file_sha256"):
            raise RuntimeError("analysis source dataset completion SHA-256 mismatch")
        dataset_manifest = _load_json_object(completion_path)
        _verify_self_hash(
            dataset_manifest, COMPLETION_HASH_FIELD, "analysis source dataset"
        )
        if dataset_manifest.get(COMPLETION_HASH_FIELD) != raw.get(
            "completion_payload_sha256"
        ):
            raise RuntimeError("analysis source dataset payload SHA-256 mismatch")
        source_datasets[str(dataset)] = CompletedDataset(
            output={}, manifest=dataset_manifest, attempt_root=dataset_root
        )
    artifacts = _verify_artifact_records(
        attempt, manifest.get("artifacts"), label="seed analysis"
    )
    if "report" not in artifacts:
        raise RuntimeError("seed analysis lacks report artifact")
    report = _load_json_object(artifacts.pop("report"))
    tables: dict[str, pd.DataFrame] = {}
    for artifact_name, path in artifacts.items():
        if not artifact_name.startswith("table_"):
            raise RuntimeError("seed analysis contains an unknown artifact")
        name = _validate_table_name(artifact_name.removeprefix("table_"))
        tables[name] = pd.read_parquet(path)
    return CompletedAnalysis(
        result={
            "dataset_outputs": {
                dataset: _dataset_summary(completed)
                for dataset, completed in source_datasets.items()
            },
            "report": report,
            "tables": tables,
        },
        manifest=manifest,
        attempt_root=attempt,
    )


def _default_heterogeneous_factory_builder(
    population_loader: Callable[..., Mapping[str, object]],
) -> Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, object]]:
    """Load raw populations only if the seed analyzer reaches Gate-0 fallback."""

    def diagnostic(ladders, dataset_outputs):
        del ladders  # Gate-0 failure is established by the calling analyzer.
        from .heterogeneous import (
            assemble_heterogeneous_diagnostic,
            evaluate_heterogeneous_dataset,
        )

        isolated: dict[str, Mapping[str, object]] = {}
        for dataset in DATASET_ORDER:
            population = population_loader(dataset, min_positive=20)
            try:
                isolated[dataset] = evaluate_heterogeneous_dataset(
                    population, dataset_outputs[dataset], dataset
                )
            finally:
                del population
                gc.collect()
        return assemble_heterogeneous_diagnostic(isolated)

    return diagnostic


def _run_or_resume_seed(
    result_root: Path,
    preregistration: Mapping[str, Any],
    runtime_estimate_sha256: str,
    *,
    model_seed: int,
    device: torch.device,
    population_loader: Callable[..., Mapping[str, object]],
    train_dataset: Callable[..., Mapping[str, object]],
    seed0_analyzer: Callable[..., Mapping[str, object]],
    additional_seed_analyzer: Callable[..., Mapping[str, object]],
    heterogeneous_diagnostic_factory: Callable[..., Mapping[str, object]],
) -> CompletedAnalysis:
    prereg_sha = str(preregistration["preregistration_sha256"])
    existing = _load_latest_analysis(
        result_root,
        model_seed=model_seed,
        preregistration_sha256=prereg_sha,
        runtime_estimate_sha256=runtime_estimate_sha256,
    )
    if existing is not None:
        return existing

    datasets: dict[str, CompletedDataset] = {}
    try:
        for dataset in DATASET_ORDER:
            datasets[dataset] = _train_or_resume_dataset(
                result_root,
                preregistration,
                runtime_estimate_sha256,
                dataset=dataset,
                model_seed=model_seed,
                device=device,
                population_loader=population_loader,
                train_dataset=train_dataset,
            )
        outputs = {
            dataset: completed.output for dataset, completed in datasets.items()
        }
        analysis_attempt = _reserve_attempt(
            Path(result_root) / f"seed{model_seed}" / "analysis"
        )
        try:
            if model_seed == 0:
                result = seed0_analyzer(
                    outputs,
                    heterogeneous_diagnostic_factory=heterogeneous_diagnostic_factory,
                    bootstrap_draws=BOOTSTRAP_DRAWS,
                    analysis_seed=ANALYSIS_SEED,
                    control_seed=CONTROL_SEED,
                )
            else:
                result = additional_seed_analyzer(
                    outputs,
                    model_seed=model_seed,
                    heterogeneous_diagnostic_factory=heterogeneous_diagnostic_factory,
                    bootstrap_draws=BOOTSTRAP_DRAWS,
                    analysis_seed=ANALYSIS_SEED,
                    control_seed=CONTROL_SEED,
                )
            return _persist_analysis(
                result_root,
                result,
                attempt_root=analysis_attempt,
                model_seed=model_seed,
                preregistration_sha256=prereg_sha,
                runtime_estimate_sha256=runtime_estimate_sha256,
                datasets=datasets,
            )
        except BaseException as error:
            _write_failure(analysis_attempt, "SEED_ANALYSIS_ATTEMPT", error)
            raise
    finally:
        datasets.clear()
        gc.collect()


def _publish_final_report(
    result_root: Path,
    result: Mapping[str, object],
    *,
    preregistration_sha256: str,
    runtime_estimate_sha256: str,
    seed_analyses: Mapping[int, CompletedAnalysis],
) -> None:
    report = result.get("report")
    if not isinstance(report, Mapping):
        raise TypeError("terminal protocol result lacks report")
    if report.get("terminal") is not True or report.get("next_action") != "STOP":
        raise RuntimeError("only a terminal protocol result may be published")
    payload = dict(_json_safe(report))
    payload.update(
        {
            "preregistration_sha256": preregistration_sha256,
            "runtime_estimate_sha256": runtime_estimate_sha256,
            "seed_analysis_artifacts": {
                str(seed): {
                    "path": completed.attempt_root
                    .relative_to(result_root)
                    .as_posix(),
                    "completion_file_sha256": file_sha256(
                        completed.attempt_root / "completion.json"
                    ),
                    "completion_payload_sha256": completed.manifest[
                        ANALYSIS_HASH_FIELD
                    ],
                }
                for seed, completed in sorted(seed_analyses.items())
            },
            "completed_at_utc": _utc_now(),
        }
    )
    exclusive_write_json(
        Path(result_root) / "final_gate_report.json",
        _add_self_hash(payload, FINAL_HASH_FIELD),
    )


def _load_final_report(
    result_root: Path,
    *,
    preregistration_sha256: str,
    runtime_estimate_sha256: str,
) -> dict[str, Any] | None:
    path = Path(result_root) / "final_gate_report.json"
    if not path.exists():
        return None
    report = _load_json_object(path)
    _verify_self_hash(report, FINAL_HASH_FIELD, "final gate report")
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("terminal") is not True
        or report.get("next_action") != "STOP"
        or report.get("preregistration_sha256") != preregistration_sha256
        or report.get("runtime_estimate_sha256") != runtime_estimate_sha256
    ):
        raise RuntimeError("final gate report binding mismatch")
    sources = report.get("seed_analysis_artifacts")
    if not isinstance(sources, Mapping) or not sources:
        raise RuntimeError("final gate report lacks seed analysis sources")
    for seed, raw in sources.items():
        if not isinstance(raw, Mapping):
            raise RuntimeError("final seed analysis source is invalid")
        source_root = _confined_path(
            result_root, raw.get("path"), label=f"final seed {seed} analysis"
        )
        completion_path = source_root / "completion.json"
        if not completion_path.is_file():
            raise FileNotFoundError(completion_path)
        if file_sha256(completion_path) != raw.get("completion_file_sha256"):
            raise RuntimeError("final seed analysis completion SHA-256 mismatch")
        completion = _load_json_object(completion_path)
        _verify_self_hash(completion, ANALYSIS_HASH_FIELD, "final seed analysis")
        if completion.get(ANALYSIS_HASH_FIELD) != raw.get(
            "completion_payload_sha256"
        ):
            raise RuntimeError("final seed analysis payload SHA-256 mismatch")
    return report


def run_persisted_full_protocol(
    result_root: Path,
    device: torch.device,
    population_loader: Callable[..., Mapping[str, object]] = load_independent_population,
    *,
    heterogeneous_factory_builder: Callable[
        [Callable[..., Mapping[str, object]]], Callable[..., Mapping[str, object]]
    ]
    | None = None,
    train_dataset: Callable[..., Mapping[str, object]] = train_dataset_experts,
    seed0_analyzer: Callable[..., Mapping[str, object]] = analyze_seed0,
    additional_seed_analyzer: Callable[..., Mapping[str, object]] = analyze_additional_seed,
) -> dict[str, object]:
    """Run and persist only the seeds requested by the frozen failure-first gates."""

    # Runtime authorization is deliberately the first operation.  A bad gate
    # cannot reach an upstream checker, population loader, or trainer.
    authorization = validate_runtime_authorization(result_root)
    preregistration = authorization["preregistration"]
    if not isinstance(preregistration, Mapping):
        raise RuntimeError("runtime authorization returned an invalid preregistration")
    runtime_sha = str(authorization["runtime_estimate_sha256"])
    prereg_sha = str(preregistration["preregistration_sha256"])
    full_device = torch.device(device)
    if full_device.type != "cuda":
        raise RuntimeError("full protocol execution requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("full protocol execution requires an available CUDA device")
    _verify_upstream_artifacts(result_root, preregistration)

    existing_final = _load_final_report(
        result_root,
        preregistration_sha256=prereg_sha,
        runtime_estimate_sha256=runtime_sha,
    )
    if existing_final is not None:
        raw_sources = existing_final["seed_analysis_artifacts"]
        if not isinstance(raw_sources, Mapping):
            raise RuntimeError("final gate report seed sources are invalid")
        restored: dict[int, Mapping[str, object]] = {}
        for raw_seed, source in raw_sources.items():
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("final gate report has an invalid seed key") from exc
            completed_analysis = _load_latest_analysis(
                result_root,
                model_seed=seed,
                preregistration_sha256=prereg_sha,
                runtime_estimate_sha256=runtime_sha,
            )
            if completed_analysis is None or not isinstance(source, Mapping):
                raise RuntimeError("final seed analysis cannot be restored")
            expected_path = _confined_path(
                result_root,
                source.get("path"),
                label=f"final seed {seed} source",
            )
            if completed_analysis.attempt_root.resolve() != expected_path:
                raise RuntimeError("final seed analysis source is not the latest completion")
            restored[seed] = completed_analysis.result
        expected_seeds = existing_final.get("executed_model_seeds")
        if not isinstance(expected_seeds, list) or sorted(restored) != expected_seeds:
            raise RuntimeError("final gate report executed-seed binding mismatch")
        restored_tables: dict[str, pd.DataFrame] = {}
        if len(restored) > 1:
            stacked = pd.concat(
                [seed_policy_losses(restored[seed], seed) for seed in sorted(restored)],
                ignore_index=True,
            )
            averaged_gate2 = evaluate_seed_average_gate2(
                stacked,
                expected_seeds=tuple(sorted(restored)),
                draws=BOOTSTRAP_DRAWS,
                seed=ANALYSIS_SEED,
            )
            report_analysis = dict(averaged_gate2)
            averaged_losses = report_analysis.pop("seed_average_losses", None)
            if not isinstance(averaged_losses, pd.DataFrame):
                raise RuntimeError("restored averaged Gate2 lacks policy losses")
            if existing_final.get("averaged_gate2") != _json_safe(
                report_analysis.get("gate2")
            ):
                raise RuntimeError("final averaged Gate2 binding mismatch")
            if existing_final.get("seed_average_gate2_analysis") != _json_safe(
                report_analysis
            ):
                raise RuntimeError("final seed-average Gate2 analysis mismatch")
            robustness = existing_final.get("robustness")
            if not isinstance(robustness, Mapping) or not isinstance(
                robustness.get("passed"), bool
            ):
                raise RuntimeError("final multiseed robustness decision is invalid")
            gate2_result = report_analysis.get("gate2")
            if not isinstance(gate2_result, Mapping) or not isinstance(
                gate2_result.get("passed"), bool
            ):
                raise RuntimeError("final averaged Gate2 decision is invalid")
            expected_verdict = robustness_verdict(
                bool(robustness["passed"]),
                averaged_gate2_pass=bool(gate2_result["passed"]),
            )
            if existing_final.get("final_verdict") != expected_verdict:
                raise RuntimeError("final multiseed verdict binding mismatch")
            restored_tables["seed_policy_losses"] = stacked
            restored_tables["seed_average_policy_losses"] = averaged_losses
        elif existing_final.get("averaged_gate2") is not None:
            raise RuntimeError("single-seed final report contains averaged Gate2")
        return {
            "report": existing_final,
            "seed_results": restored,
            "tables": restored_tables,
        }

    builder = (
        _default_heterogeneous_factory_builder
        if heterogeneous_factory_builder is None
        else heterogeneous_factory_builder
    )
    heterogeneous_factory = builder(population_loader)
    if not callable(heterogeneous_factory):
        raise TypeError("heterogeneous_factory_builder must return a callable")

    completed: dict[int, CompletedAnalysis] = {}
    seed_results: dict[int, Mapping[str, object]] = {}
    seed0 = _run_or_resume_seed(
        result_root,
        preregistration,
        runtime_sha,
        model_seed=0,
        device=full_device,
        population_loader=population_loader,
        train_dataset=train_dataset,
        seed0_analyzer=seed0_analyzer,
        additional_seed_analyzer=additional_seed_analyzer,
        heterogeneous_diagnostic_factory=heterogeneous_factory,
    )
    completed[0] = seed0
    seed_results[0] = seed0.result
    seed0_report = seed0.result.get("report")
    if not isinstance(seed0_report, Mapping):
        raise ValueError("persisted seed 0 result lacks report")
    final_verdict = seed0_report.get("final_verdict")
    if final_verdict is not None:
        result = assemble_protocol_result(
            seed_results,
            final_verdict=str(final_verdict),
            robustness=(
                seed0_report.get("gate4")
                if isinstance(seed0_report.get("gate4"), Mapping)
                else None
            ),
        )
        _verify_upstream_artifacts(result_root, preregistration)
        _publish_final_report(
            result_root,
            result,
            preregistration_sha256=prereg_sha,
            runtime_estimate_sha256=runtime_sha,
            seed_analyses=completed,
        )
        return result
    if seed0_report.get("next_action") != "RUN_SEED1":
        raise ValueError("nonterminal seed 0 must request RUN_SEED1")

    seed1 = _run_or_resume_seed(
        result_root,
        preregistration,
        runtime_sha,
        model_seed=1,
        device=full_device,
        population_loader=population_loader,
        train_dataset=train_dataset,
        seed0_analyzer=seed0_analyzer,
        additional_seed_analyzer=additional_seed_analyzer,
        heterogeneous_diagnostic_factory=heterogeneous_factory,
    )
    completed[1] = seed1
    seed_results[1] = seed1.result
    seed1_report = seed1.result.get("report")
    if not isinstance(seed1_report, Mapping) or seed1_report.get(
        "next_action"
    ) != "SEED_POLICY_READY":
        raise RuntimeError("additional seed 1 did not produce a policy-loss panel")
    stacked = pd.concat(
        [seed_policy_losses(seed0.result, 0), seed_policy_losses(seed1.result, 1)],
        ignore_index=True,
    )
    gate4_seed1 = evaluate_gate4_seed1(
        stacked,
        candidate_column="m1_normalized_loss",
        baseline_column="b4_normalized_loss",
        draws=BOOTSTRAP_DRAWS,
        seed=ANALYSIS_SEED,
    )
    averaged_gate2 = evaluate_seed_average_gate2(
        stacked,
        expected_seeds=(0, 1),
        draws=BOOTSTRAP_DRAWS,
        seed=ANALYSIS_SEED,
    )
    averaged_gate2_result = averaged_gate2.get("gate2")
    if not isinstance(averaged_gate2_result, Mapping) or not isinstance(
        averaged_gate2_result.get("passed"), (bool, np.bool_)
    ):
        raise RuntimeError("seed-average Gate2 did not return a Boolean decision")
    averaged_gate2_pass = bool(averaged_gate2_result["passed"])
    action = str(gate4_seed1["action"])
    if action in {"ACCEPT_TWO_SEED", "RETRIEVAL_ROBUSTNESS_NO_GO"}:
        result = assemble_protocol_result(
            seed_results,
            final_verdict=robustness_verdict(
                action == "ACCEPT_TWO_SEED",
                averaged_gate2_pass=averaged_gate2_pass,
            ),
            robustness=gate4_seed1,
            seed_policy_loss_frame=stacked,
            averaged_gate2_analysis=averaged_gate2,
        )
        _verify_upstream_artifacts(result_root, preregistration)
        _publish_final_report(
            result_root,
            result,
            preregistration_sha256=prereg_sha,
            runtime_estimate_sha256=runtime_sha,
            seed_analyses=completed,
        )
        return result
    if action != "RUN_SEED2":
        raise ValueError(f"unknown Gate4 seed1 action: {action}")

    seed2 = _run_or_resume_seed(
        result_root,
        preregistration,
        runtime_sha,
        model_seed=2,
        device=full_device,
        population_loader=population_loader,
        train_dataset=train_dataset,
        seed0_analyzer=seed0_analyzer,
        additional_seed_analyzer=additional_seed_analyzer,
        heterogeneous_diagnostic_factory=heterogeneous_factory,
    )
    completed[2] = seed2
    seed_results[2] = seed2.result
    seed2_report = seed2.result.get("report")
    if not isinstance(seed2_report, Mapping) or seed2_report.get(
        "next_action"
    ) != "SEED_POLICY_READY":
        raise RuntimeError("additional seed 2 did not produce a policy-loss panel")
    stacked = pd.concat(
        [stacked, seed_policy_losses(seed2.result, 2)], ignore_index=True
    )
    gate4_seed2 = evaluate_gate4_seed2(
        stacked,
        candidate_column="m1_normalized_loss",
        baseline_column="b4_normalized_loss",
        draws=BOOTSTRAP_DRAWS,
        seed=ANALYSIS_SEED,
    )
    averaged_gate2 = evaluate_seed_average_gate2(
        stacked,
        expected_seeds=(0, 1, 2),
        draws=BOOTSTRAP_DRAWS,
        seed=ANALYSIS_SEED,
    )
    averaged_gate2_result = averaged_gate2.get("gate2")
    if not isinstance(averaged_gate2_result, Mapping) or not isinstance(
        averaged_gate2_result.get("passed"), (bool, np.bool_)
    ):
        raise RuntimeError("seed-average Gate2 did not return a Boolean decision")
    averaged_gate2_pass = bool(averaged_gate2_result["passed"])
    action = str(gate4_seed2["action"])
    if action not in {"ACCEPT_THREE_SEED", "RETRIEVAL_ROBUSTNESS_NO_GO"}:
        raise ValueError(f"unknown Gate4 seed2 action: {action}")
    result = assemble_protocol_result(
        seed_results,
        final_verdict=robustness_verdict(
            bool(gate4_seed2["passed"]),
            averaged_gate2_pass=averaged_gate2_pass,
        ),
        robustness=gate4_seed2,
        seed_policy_loss_frame=stacked,
        averaged_gate2_analysis=averaged_gate2,
    )
    _verify_upstream_artifacts(result_root, preregistration)
    _publish_final_report(
        result_root,
        result,
        preregistration_sha256=prereg_sha,
        runtime_estimate_sha256=runtime_sha,
        seed_analyses=completed,
    )
    return result


__all__ = [
    "CompletedAnalysis",
    "CompletedDataset",
    "load_latest_completed_dataset",
    "run_persisted_full_protocol",
    "validate_runtime_authorization",
]
