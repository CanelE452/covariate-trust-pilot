"""Failure-first orchestration for PH-ONLINE-MEMORY-GONO-v1."""

from __future__ import annotations

import argparse
import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

# Capture the caller's process environment before any third-party import can
# mutate it during module initialization.
_INHERITED_KMP_DUPLICATE_LIB_OK = os.environ.get("KMP_DUPLICATE_LIB_OK")

import numpy as np
import pandas as pd
import scipy
import torch
import pyarrow

from .artifacts import (
    exclusive_write_json,
    exclusive_write_parquet,
    exclusive_write_text,
    file_sha256,
    payload_sha256,
    verify_preregistration,
)
from .data import load_independent_population
from .execution import run_persisted_full_protocol
from .integrity import verify_forbidden_artifacts
from .prereg import build_preregistered_spec, freeze_preregistration
from .reproduction import reproduce_three_origin
from .reporting import (
    build_runtime_stop_status,
    build_status_markdown,
    build_tables_a_to_g,
)
from .smoke_run import run_m5_smoke


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "ph_online_memory_gono_v1"
PACKAGE_ROOT = REPO_ROOT / "experiments" / "ph_online_memory_gono_v1"
PHASE0_EVIDENCE_PATHS = (
    RESULTS_ROOT / "audit.json",
    RESULTS_ROOT / "audit_resolution.json",
    RESULTS_ROOT / "execution_resolution.json",
    RESULTS_ROOT / "forbidden_artifact_baseline_extension.json",
    RESULTS_ROOT / "gate2_gate4_resolution.json",
)
DEPENDENCY_PATHS = (
    REPO_ROOT / "experiments" / "__init__.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "__init__.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "cli.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "classical_benchmark.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "favorita_transfer.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "posthoc.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "rule_replication.py",
    REPO_ROOT / "experiments" / "om_factorization_killtest" / "__init__.py",
    REPO_ROOT / "experiments" / "om_factorization_killtest" / "evaluate.py",
    REPO_ROOT / "experiments" / "om_factorization_killtest" / "models.py",
    REPO_ROOT / "experiments" / "om_factorization_killtest" / "prereg.py",
    REPO_ROOT / "experiments" / "om_factorization_killtest" / "train.py",
    REPO_ROOT / "experiments" / "unified_temporal_27_v3" / "config.py",
    REPO_ROOT / "experiments" / "unified_temporal_27_v3" / "conditional_targets.py",
    REPO_ROOT / "experiments" / "unified_temporal_27_v3" / "__init__.py",
    REPO_ROOT / "experiments" / "unified_temporal_27_v3" / "model.py",
    REPO_ROOT / "experiments" / "unified_temporal_27_v3" / "scenarios.py",
    REPO_ROOT / "experiments" / "unified_temporal_27_v3" / "training.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "screen.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "prereg.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "confirmatory_h2.py",
    REPO_ROOT / "experiments" / "external_validity_screen" / "favorita_independent.py",
    REPO_ROOT / "src" / "rcoi" / "seed.py",
    REPO_ROOT / "src" / "rcoi" / "__init__.py",
    REPO_ROOT / "src" / "rcoi" / "models" / "decomposition.py",
)
DATA_SOURCE_PATHS = (
    REPO_ROOT / "data" / "sales_train_evaluation.csv",
    REPO_ROOT / "data" / "sell_prices.csv",
    REPO_ROOT / "data" / "calendar.csv",
    REPO_ROOT / "data" / "processed" / "favorita_full_pool.parquet",
    REPO_ROOT / "data" / "processed" / "series.parquet",
    REPO_ROOT / "data" / "processed" / "favorita_series.parquet",
)
RAW_PATHS = {
    "m5": (
        REPO_ROOT
        / "results"
        / "external_validity_screen"
        / "rule_replication"
        / "independent_raw_predictions.parquet"
    ),
    "favorita": (
        REPO_ROOT
        / "results"
        / "external_validity_screen"
        / "favorita_independent"
        / "independent_raw_predictions.parquet"
    ),
}

FINAL_GATE_REPORT_PATH = RESULTS_ROOT / "final_gate_report.json"
TABLES_REPORT_NAME = "tables_a_to_g.json"
RESOLVED_STATUS_NAME = "STATUS_AFTER_RESOLUTION.md"
FINALIZATION_MANIFEST_NAME = "finalization_manifest.json"

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _publish_or_verify_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a deterministic terminal artifact or verify the exact prior copy."""

    destination = Path(path)
    if destination.exists():
        if _read_json_object(destination) != payload:
            raise RuntimeError(f"existing append-only artifact differs: {destination}")
        return
    try:
        exclusive_write_json(destination, payload)
    except FileExistsError:
        if _read_json_object(destination) != payload:
            raise RuntimeError(f"raced append-only artifact differs: {destination}")


def _publish_or_verify_text(path: Path, content: str) -> None:
    destination = Path(path)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing append-only artifact differs: {destination}")
        return
    try:
        exclusive_write_text(destination, content)
    except FileExistsError:
        if destination.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"raced append-only artifact differs: {destination}")


def _validated_final_gate_report(
    root: Path,
    preregistration: dict[str, Any],
    runtime_estimate_sha256: str,
) -> dict[str, Any]:
    path = Path(root) / "final_gate_report.json"
    report = _read_json_object(path)
    expected_hash = report.get("final_gate_report_payload_sha256")
    unhashed = dict(report)
    unhashed.pop("final_gate_report_payload_sha256", None)
    if not isinstance(expected_hash, str) or payload_sha256(unhashed) != expected_hash:
        raise RuntimeError("final gate report self SHA-256 mismatch")
    expected = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "terminal": True,
        "next_action": "STOP",
        "preregistration_sha256": preregistration.get("preregistration_sha256"),
        "runtime_estimate_sha256": runtime_estimate_sha256,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError("final gate report binding mismatch")
    return report


def _reserve_attempt_directory(root: Path, stage: str) -> tuple[str, Path]:
    """Reserve a new append-only attempt directory for a restartable stage."""

    stage_root = Path(root) / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    used: list[int] = []
    for path in stage_root.glob("attempt_*"):
        suffix = path.name.removeprefix("attempt_")
        if path.is_dir() and suffix.isdigit():
            used.append(int(suffix))
    number = max(used, default=0) + 1
    attempt_id = f"attempt_{number:04d}"
    attempt_root = stage_root / attempt_id
    attempt_root.mkdir(exist_ok=False)
    return attempt_id, attempt_root


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _implementation_hashes() -> dict[str, str]:
    package_files = {
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    files = sorted(
        package_files.union(DEPENDENCY_PATHS).union(PHASE0_EVIDENCE_PATHS)
    )
    if not files:
        raise RuntimeError("no pilot implementation files were found")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return {
        path.relative_to(REPO_ROOT).as_posix(): file_sha256(path)
        for path in files
    }


def _verify_frozen_implementation(preregistration: dict[str, Any]) -> None:
    frozen = preregistration.get("implementation_sha256")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("frozen implementation hash manifest is missing")
    current = _implementation_hashes()
    if current != frozen:
        changed = sorted(
            path
            for path in set(current).union(frozen)
            if current.get(path) != frozen.get(path)
        )
        preview = ", ".join(changed[:5])
        raise RuntimeError(
            "implementation hash drift detected before execution: " + preview
        )


def _data_source_hashes() -> dict[str, str]:
    missing = [path for path in DATA_SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return {
        path.relative_to(REPO_ROOT).as_posix(): file_sha256(path)
        for path in DATA_SOURCE_PATHS
    }


def _verify_frozen_data_sources(preregistration: dict[str, Any]) -> None:
    frozen = preregistration.get("data_source_sha256")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("frozen data-source hash manifest is missing")
    current = _data_source_hashes()
    if current != frozen:
        changed = sorted(
            path
            for path in set(current).union(frozen)
            if current.get(path) != frozen.get(path)
        )
        raise RuntimeError(
            "data-source hash drift detected before smoke: "
            + ", ".join(changed[:5])
        )


def _verify_frozen_environment(preregistration: dict[str, Any]) -> None:
    frozen = preregistration.get("environment")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("frozen runtime environment manifest is missing")
    current = _environment_manifest()
    if current != frozen:
        changed = sorted(
            key
            for key in set(current).union(frozen)
            if current.get(key) != frozen.get(key)
        )
        raise RuntimeError(
            "runtime environment drift detected before smoke: "
            + ", ".join(changed)
        )


def _validate_phase0_authorization() -> None:
    (
        audit_path,
        resolution_path,
        execution_path,
        baseline_extension_path,
        gate_resolution_path,
    ) = PHASE0_EVIDENCE_PATHS
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    with resolution_path.open("r", encoding="utf-8") as handle:
        resolution = json.load(handle)
    if audit.get("experiment_name") != "PH-ONLINE-MEMORY-GONO-v1":
        raise RuntimeError("Phase 0 audit belongs to another experiment")
    if audit.get("status") != "PHASE0_AUDIT_STOP":
        raise RuntimeError("unexpected Phase 0 audit status")
    audited_repository = audit.get("repository")
    if not isinstance(audited_repository, dict):
        raise RuntimeError("Phase 0 audit lacks repository provenance")
    observed_remote = _git("remote", "get-url", "origin").rstrip("/")
    audited_remote = str(audited_repository.get("remote", "")).rstrip("/")
    if observed_remote.endswith(".git"):
        observed_remote = observed_remote[:-4]
    if audited_remote.endswith(".git"):
        audited_remote = audited_remote[:-4]
    observed_repository = {
        "remote": observed_remote,
        "branch": _git("branch", "--show-current"),
        "head_commit": _git("rev-parse", "HEAD"),
    }
    expected_repository = {
        "remote": audited_remote,
        "branch": audited_repository.get("branch"),
        "head_commit": audited_repository.get("head_commit"),
    }
    if observed_repository != expected_repository:
        raise RuntimeError("repository provenance drifted after the Phase 0 audit")
    if _git("rev-parse", "HEAD") != _git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD must remain aligned with origin/main")
    if resolution.get("status") != "PHASE0_RESOLUTION_AUTHORIZED":
        raise RuntimeError("Phase 0 resolution is not authorized")
    recorded = str(resolution.get("prior_audit", {}).get("sha256", "")).lower()
    if recorded != file_sha256(audit_path).lower():
        raise RuntimeError("Phase 0 resolution is not bound to the preserved audit")
    with execution_path.open("r", encoding="utf-8") as handle:
        execution = json.load(handle)
    if execution.get("status") != "PREFREEZE_EXECUTION_RESOLUTION":
        raise RuntimeError("pre-freeze execution resolution is invalid")
    preserved = execution.get("append_only_status_policy", {}).get(
        "preserved_phase0_status", {}
    )
    if str(preserved.get("sha256", "")).lower() != file_sha256(
        RESULTS_ROOT / "STATUS.md"
    ).lower():
        raise RuntimeError("the preserved Phase 0 STATUS artifact changed")
    plan = execution.get("failure_first_order_correction", {}).get(
        "preserved_plan", {}
    )
    if str(plan.get("sha256", "")).lower() != file_sha256(
        RESULTS_ROOT / "implementation_plan.md"
    ).lower():
        raise RuntimeError("the preserved implementation plan changed")
    with baseline_extension_path.open("r", encoding="utf-8") as handle:
        baseline_extension = json.load(handle)
    if baseline_extension.get("status") != "PREFREEZE_BASELINE_EXTENSION":
        raise RuntimeError("forbidden artifact baseline extension is invalid")
    extension_audit_sha = str(
        baseline_extension.get("prior_audit", {}).get("sha256", "")
    ).lower()
    if extension_audit_sha != file_sha256(audit_path).lower():
        raise RuntimeError("forbidden baseline extension is not bound to the audit")
    integrity = verify_forbidden_artifacts(REPO_ROOT, audit_path)
    if not integrity.get("all_unchanged"):
        raise RuntimeError("a user-forbidden result directory changed")
    with gate_resolution_path.open("r", encoding="utf-8") as handle:
        gate_resolution = json.load(handle)
    if gate_resolution.get("status") != "PREFREEZE_GATE2_GATE4_RESOLUTION":
        raise RuntimeError("Gate 2 / Gate 4 resolution is invalid")
    if str(gate_resolution.get("request", {}).get("sha256", "")).upper() != str(
        audit.get("request_attachment_sha256", "")
    ).upper():
        raise RuntimeError("Gate 2 / Gate 4 resolution request binding mismatch")
    bindings = gate_resolution.get("bindings", {})
    expected_bindings = {
        "audit_resolution": file_sha256(resolution_path).upper(),
        "execution_resolution": file_sha256(execution_path).upper(),
    }
    observed_bindings = {
        name: str(bindings.get(name, {}).get("sha256", "")).upper()
        for name in expected_bindings
    }
    if observed_bindings != expected_bindings:
        raise RuntimeError("Gate 2 / Gate 4 resolution binding mismatch")
    interpretation = gate_resolution.get("authoritative_interpretation", {})
    if interpretation.get("seed0_gate2_status", "").find("PENDING_GATE4") < 0:
        raise RuntimeError("Gate 2 borderline deferral is not authorized")
    if interpretation.get("deferrable_failed_gate2_checks") != [
        "macro_effect",
        "direction_safety",
        "macro_absolute_usefulness",
        "direction_absolute_usefulness",
        "macro_ci",
        "dataset_ci",
    ]:
        raise RuntimeError("Gate 2 deferrable-check boundary is not authorized")
    if interpretation.get("seed_average_gate2", "").find("Gate 2 A-F") < 0:
        raise RuntimeError("seed-average Gate 2 re-evaluation is not authorized")
    veto_precedence = interpretation.get("seed0_veto_precedence", "")
    if "FINAL_FAIL" not in veto_precedence or (
        "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO" not in veto_precedence
    ):
        raise RuntimeError("Gate 2 / Gate 3 veto precedence is not authorized")


def _repository_manifest() -> dict[str, Any]:
    status_lines = [
        line for line in _git("status", "--porcelain=v1").splitlines() if line
    ]
    return {
        "path": REPO_ROOT.as_posix(),
        "git_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "remote_origin": _git("remote", "get-url", "origin"),
        "dirty": bool(status_lines),
        "dirty_path_count": len(status_lines),
        "head_equals_origin_main": (
            _git("rev-parse", "HEAD") == _git("rev-parse", "origin/main")
        ),
    }


def _environment_manifest() -> dict[str, Any]:
    if _INHERITED_KMP_DUPLICATE_LIB_OK is not None:
        raise RuntimeError(
            "KMP_DUPLICATE_LIB_OK must be unset before the protocol process starts"
        )
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": importlib_metadata.version("scikit-learn"),
        "pyarrow": pyarrow.__version__,
        "kmp_duplicate_lib_ok": None,
    }


def freeze_current_preregistration(
    *, result_root: Path = RESULTS_ROOT
) -> dict[str, Any]:
    """Freeze the complete result-independent protocol before any new fit."""

    destination = Path(result_root) / "preregistered_spec.json"
    if destination.exists():
        raise FileExistsError(destination)
    _validate_phase0_authorization()
    spec = build_preregistered_spec(
        repository=_repository_manifest(),
        environment=_environment_manifest(),
        implementation_sha256=_implementation_hashes(),
        frozen_at_utc=_utc_now(),
        data_source_sha256=_data_source_hashes(),
    )
    return freeze_preregistration(destination, spec)


def run_stage0_reproduction(
    *, result_root: Path = RESULTS_ROOT
) -> dict[str, Any]:
    """Run cached three-origin validation only after preregistration is frozen."""

    root = Path(result_root)
    prereg = verify_preregistration(root / "preregistered_spec.json")
    _verify_frozen_implementation(prereg)
    _verify_frozen_environment(prereg)
    destination = root / "stage0_reproduction.json"
    if destination.exists():
        raise FileExistsError(destination)
    integrity_before = verify_forbidden_artifacts(
        REPO_ROOT, PHASE0_EVIDENCE_PATHS[0]
    )
    if not integrity_before["all_unchanged"]:
        raise RuntimeError("a forbidden result directory changed before Stage 0")
    started = _utc_now()
    report = dict(reproduce_three_origin(RAW_PATHS, REPO_ROOT / "results"))
    _verify_frozen_implementation(prereg)
    _verify_frozen_environment(prereg)
    integrity_after = verify_forbidden_artifacts(
        REPO_ROOT, PHASE0_EVIDENCE_PATHS[0]
    )
    if not integrity_after["all_unchanged"]:
        raise RuntimeError("a forbidden result directory changed during Stage 0")
    report.update(
        {
            "experiment": "PH-ONLINE-MEMORY-GONO-v1",
            "stage": "STAGE0_CACHED_THREE_ORIGIN_REPRODUCTION",
            "scientific_result": False,
            "started_at_utc": started,
            "completed_at_utc": _utc_now(),
            "preregistration_sha256": prereg["preregistration_sha256"],
            "forbidden_artifact_integrity": {
                "before_stage0": integrity_before,
                "after_stage0": integrity_after,
            },
        }
    )
    exclusive_write_json(destination, report)
    return report


def _read_stage0_pass(
    root: Path, preregistration: dict[str, Any]
) -> dict[str, Any]:
    path = root / "stage0_reproduction.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "PASS":
        raise RuntimeError("Stage 0 reproduction must PASS before smoke training")
    expected_identity = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "STAGE0_CACHED_THREE_ORIGIN_REPRODUCTION",
        "preregistration_sha256": preregistration.get(
            "preregistration_sha256"
        ),
    }
    if any(report.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("Stage 0 PASS is not bound to the current preregistration")
    frozen_hashes = preregistration.get("stage0_reproduction", {}).get(
        "frozen_sha256", {}
    )
    expected_inputs = {
        "m5": frozen_hashes.get("m5_raw"),
        "favorita": frozen_hashes.get("favorita_raw"),
    }
    expected_references = {
        "condition_discovery": frozen_hashes.get("condition_discovery_panel"),
        "recoverability": frozen_hashes.get("recoverability_panel"),
    }
    if report.get("input_sha256") != expected_inputs or report.get(
        "reference_sha256"
    ) != expected_references:
        raise RuntimeError("Stage 0 PASS input hashes are not bound to preregistration")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("Stage 0 PASS does not contain successful checks")
    return report


def run_pre_smoke_verification(
    *, result_root: Path = RESULTS_ROOT
) -> dict[str, Any]:
    """Run the frozen package tests only after cached Stage 0 has passed.

    This stage intentionally includes the real CPU trainer-equivalence fit.
    Keeping it behind Stage 0 preserves the requested failure-first order.
    """

    root = Path(result_root)
    destination = root / "pre_smoke_verification.json"
    if destination.exists():
        raise FileExistsError(destination)
    prereg = verify_preregistration(root / "preregistered_spec.json")
    _verify_frozen_implementation(prereg)
    _verify_frozen_environment(prereg)
    _verify_frozen_data_sources(prereg)
    _read_stage0_pass(root, prereg)
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        str(PACKAGE_ROOT / "tests"),
        "-v",
    ]
    environment = os.environ.copy()
    python_paths = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    started = _utc_now()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            "pre-smoke verification failed; smoke is forbidden:\n" + output[-8000:]
        )
    _verify_frozen_implementation(prereg)
    _verify_frozen_environment(prereg)
    _verify_frozen_data_sources(prereg)
    report = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "PRE_SMOKE_VERIFICATION",
        "scientific_result": False,
        "status": "PASS",
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "stage0_reproduction_sha256": file_sha256(
            root / "stage0_reproduction.json"
        ),
        "command": command,
        "test_exit_code": int(completed.returncode),
        "test_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "test_output": output,
        "required_coverage": {
            "leakage_and_contract_tests": [f"T{index}" for index in range(1, 13)],
            "real_cached_reproduction_behavior": True,
            "real_cpu_canonical_trainer_equivalence_fit": True,
            "trainer_fit_order": "after Stage 0 PASS and before CUDA smoke",
        },
    }
    exclusive_write_json(destination, report)
    return report


def _read_pre_smoke_pass(
    root: Path, preregistration: dict[str, Any]
) -> dict[str, Any]:
    path = root / "pre_smoke_verification.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "PRE_SMOKE_VERIFICATION",
        "status": "PASS",
        "preregistration_sha256": preregistration.get(
            "preregistration_sha256"
        ),
        "stage0_reproduction_sha256": file_sha256(
            root / "stage0_reproduction.json"
        ),
        "test_exit_code": 0,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError("pre-smoke verification is not bound to this run")
    return report


def _publish_runtime_stop_if_required(
    root: Path, report: dict[str, Any]
) -> None:
    gate = report.get("runtime_gate")
    if not isinstance(gate, dict) or gate.get("action") != "STOP_FOR_APPROVAL":
        return
    runtime_path = Path(root) / "runtime_estimate.json"
    bound = dict(report)
    bound["runtime_estimate_sha256"] = file_sha256(runtime_path)
    status = build_runtime_stop_status(bound)
    _publish_or_verify_text(Path(root) / RESOLVED_STATUS_NAME, status)


def _read_completed_smoke(
    root: Path, preregistration: dict[str, Any]
) -> dict[str, Any]:
    runtime_path = Path(root) / "runtime_estimate.json"
    report = _read_json_object(runtime_path)
    expected = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "M5_200_SERIES_SMOKE",
        "preregistration_sha256": preregistration.get(
            "preregistration_sha256"
        ),
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError("existing runtime estimate is not bound to this smoke")
    attempt = report.get("attempt")
    if not isinstance(attempt, dict) or not isinstance(attempt.get("path"), str):
        raise RuntimeError("existing runtime estimate lacks its smoke attempt")
    attempt_path = (Path(root) / attempt["path"]).resolve()
    resolved_root = Path(root).resolve()
    try:
        relative_attempt = attempt_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError("existing smoke attempt escapes the result root") from exc
    if (
        len(relative_attempt.parts) < 2
        or relative_attempt.parts[0] != "smoke"
        or attempt.get("id") != attempt_path.name
    ):
        raise RuntimeError("existing runtime estimate has an invalid smoke attempt")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "population_audit",
        "predictions",
        "expert_losses",
    }:
        raise RuntimeError("existing runtime estimate has an incomplete artifact set")
    for name, raw in artifacts.items():
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise RuntimeError(f"existing smoke artifact is invalid: {name}")
        artifact = (Path(root) / raw["path"]).resolve()
        try:
            artifact.relative_to(attempt_path)
        except ValueError as exc:
            raise RuntimeError(f"existing smoke artifact escapes its attempt: {name}") from exc
        if not artifact.is_file() or file_sha256(artifact) != raw.get("sha256"):
            raise RuntimeError(f"existing smoke artifact hash mismatch: {name}")
        if "bytes" in raw and artifact.stat().st_size != raw["bytes"]:
            raise RuntimeError(f"existing smoke artifact byte count mismatch: {name}")
    return report


def run_smoke_stage(*, result_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    """Audit both populations and run the preregistered M5 timing smoke."""

    root = Path(result_root)
    prereg = verify_preregistration(root / "preregistered_spec.json")
    _verify_frozen_implementation(prereg)
    _read_stage0_pass(root, prereg)
    _read_pre_smoke_pass(root, prereg)
    _verify_frozen_environment(prereg)
    _verify_frozen_data_sources(prereg)
    integrity_before = verify_forbidden_artifacts(
        REPO_ROOT, PHASE0_EVIDENCE_PATHS[0]
    )
    if not integrity_before["all_unchanged"]:
        raise RuntimeError("a forbidden result directory changed before smoke")
    runtime_path = root / "runtime_estimate.json"
    if runtime_path.exists():
        report = _read_completed_smoke(root, prereg)
        _publish_runtime_stop_if_required(root, report)
        return report
    if not torch.cuda.is_available():
        raise RuntimeError("the preregistered smoke requires an available CUDA GPU")
    attempt_id, attempt_root = _reserve_attempt_directory(root, "smoke")
    population_path = attempt_root / "population_audit.json"
    predictions_path = attempt_root / "predictions.parquet"
    losses_path = attempt_root / "expert_losses.parquet"

    # Load Favorita only long enough to audit its independent population.  This
    # keeps its full arrays out of memory before the M5 training smoke begins.
    favorita = load_independent_population("favorita", min_positive=20)
    favorita_manifest = dict(favorita["manifest"])
    del favorita
    gc.collect()
    m5 = load_independent_population("m5", min_positive=20)
    m5_manifest = dict(m5["manifest"])
    counts = {
        "m5": int(m5_manifest["eligible_independent"]),
        "favorita": int(favorita_manifest["eligible_independent"]),
    }
    population_report = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "NEW_CUTOFF_POPULATION_AUDIT",
        "scientific_result": False,
        "created_at_utc": _utc_now(),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "datasets": {"m5": m5_manifest, "favorita": favorita_manifest},
    }
    started = _utc_now()
    result = run_m5_smoke(m5, torch.device("cuda"), counts)
    exclusive_write_json(population_path, population_report)
    predictions_bytes = exclusive_write_parquet(
        predictions_path, result["predictions"]
    )
    losses_bytes = exclusive_write_parquet(losses_path, result["losses"])
    integrity_after = verify_forbidden_artifacts(
        REPO_ROOT, PHASE0_EVIDENCE_PATHS[0]
    )
    if not integrity_after["all_unchanged"]:
        raise RuntimeError("a forbidden result directory changed during smoke")
    report = dict(result["report"])
    report.update(
        {
            "attempt": {
                "id": attempt_id,
                "path": attempt_root.relative_to(root).as_posix(),
                "restart_policy": (
                    "failed attempts remain immutable; retries reserve the next directory"
                ),
            },
            "forbidden_artifact_integrity": {
                "before_smoke": integrity_before,
                "after_smoke": integrity_after,
            },
            "started_at_utc": started,
            "completed_at_utc": _utc_now(),
            "preregistration_sha256": prereg["preregistration_sha256"],
            "artifacts": {
                "population_audit": {
                    "path": population_path.relative_to(root).as_posix(),
                    "sha256": file_sha256(population_path),
                },
                "predictions": {
                    "path": predictions_path.relative_to(root).as_posix(),
                    "bytes": predictions_bytes,
                    "sha256": file_sha256(predictions_path),
                },
                "expert_losses": {
                    "path": losses_path.relative_to(root).as_posix(),
                    "bytes": losses_bytes,
                    "sha256": file_sha256(losses_path),
                },
            },
        }
    )
    exclusive_write_json(runtime_path, report)
    _publish_runtime_stop_if_required(root, report)
    return report


def run_full_stage(*, result_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    """Execute or resume the authorized full protocol and publish its reports."""

    root = Path(result_root)
    # Preserve the Phase-0 status/rulings and Git provenance before the runtime
    # authorization can reach any population loader or trainer.
    _validate_phase0_authorization()
    numerical_result = run_persisted_full_protocol(root, torch.device("cuda"))

    prereg = verify_preregistration(root / "preregistered_spec.json")
    runtime_sha = file_sha256(root / "runtime_estimate.json")
    _validate_phase0_authorization()
    _verify_frozen_implementation(prereg)
    _read_stage0_pass(root, prereg)
    _read_pre_smoke_pass(root, prereg)
    _verify_frozen_environment(prereg)
    _verify_frozen_data_sources(prereg)
    integrity_before_reporting = verify_forbidden_artifacts(
        REPO_ROOT, PHASE0_EVIDENCE_PATHS[0]
    )
    if not integrity_before_reporting["all_unchanged"]:
        raise RuntimeError("a forbidden result directory changed before reporting")

    final_path = root / "final_gate_report.json"
    final_report = _validated_final_gate_report(root, prereg, runtime_sha)
    reporting_result = dict(numerical_result)
    reporting_result["report"] = final_report
    tables = build_tables_a_to_g(reporting_result)
    tables_payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "FINAL_TABLES_A_TO_G",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "runtime_estimate_sha256": runtime_sha,
        "final_gate_report_sha256": file_sha256(final_path),
        "final_gate_report_payload_sha256": final_report[
            "final_gate_report_payload_sha256"
        ],
        "source_seed_analysis_artifacts": final_report[
            "seed_analysis_artifacts"
        ],
        "tables": tables,
    }
    tables_payload["tables_a_to_g_payload_sha256"] = payload_sha256(
        tables_payload
    )
    tables_path = root / TABLES_REPORT_NAME
    _publish_or_verify_json(tables_path, tables_payload)

    artifact_bindings = {
        "preregistered_spec": {
            "path": "preregistered_spec.json",
            "payload_sha256": prereg["preregistration_sha256"],
            "file_sha256": file_sha256(root / "preregistered_spec.json"),
        },
        "runtime_estimate": {
            "path": "runtime_estimate.json",
            "file_sha256": runtime_sha,
        },
        "final_gate_report": {
            "path": "final_gate_report.json",
            "file_sha256": file_sha256(final_path),
            "payload_sha256": final_report["final_gate_report_payload_sha256"],
        },
        "tables_a_to_g": {
            "path": TABLES_REPORT_NAME,
            "file_sha256": file_sha256(tables_path),
            "payload_sha256": tables_payload["tables_a_to_g_payload_sha256"],
        },
    }
    status = build_status_markdown(
        reporting_result,
        preregistration=prereg,
        artifact_bindings=artifact_bindings,
    )
    status_path = root / RESOLVED_STATUS_NAME
    _publish_or_verify_text(status_path, status)

    _validate_phase0_authorization()
    _verify_frozen_implementation(prereg)
    _verify_frozen_environment(prereg)
    _verify_frozen_data_sources(prereg)
    integrity_after_reporting = verify_forbidden_artifacts(
        REPO_ROOT, PHASE0_EVIDENCE_PATHS[0]
    )
    if not integrity_after_reporting["all_unchanged"]:
        raise RuntimeError("a forbidden result directory changed during reporting")

    finalization: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "FINALIZATION_COMPLETE",
        "scientific_verdict": final_report["final_verdict"],
        "artifact_bindings": {
            **artifact_bindings,
            "resolved_status": {
                "path": RESOLVED_STATUS_NAME,
                "file_sha256": file_sha256(status_path),
            },
        },
        "forbidden_artifact_integrity": integrity_after_reporting,
    }
    finalization["finalization_payload_sha256"] = payload_sha256(finalization)
    finalization_path = root / FINALIZATION_MANIFEST_NAME
    _publish_or_verify_json(finalization_path, finalization)
    return {
        "report": final_report,
        "tables": tables,
        "artifact_bindings": finalization["artifact_bindings"],
        "finalization": finalization,
    }


def main() -> None:
    parser = argparse.ArgumentParser("PH-ONLINE-MEMORY-GONO-v1 protocol")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("reproduce")
    subparsers.add_parser("verify")
    subparsers.add_parser("smoke")
    subparsers.add_parser("full")
    args = parser.parse_args()
    if args.stage == "freeze":
        report = freeze_current_preregistration()
        summary = {
            "experiment": report["experiment_name"],
            "preregistration_sha256": report["preregistration_sha256"],
        }
    elif args.stage == "reproduce":
        report = run_stage0_reproduction()
        summary = {"stage": report["stage"], "status": report["status"]}
    elif args.stage == "verify":
        report = run_pre_smoke_verification()
        summary = {"stage": report["stage"], "status": report["status"]}
    elif args.stage == "smoke":
        report = run_smoke_stage()
        summary = {
            "stage": report["stage"],
            "runtime_gate": report["runtime_gate"],
        }
    else:
        report = run_full_stage()
        summary = {
            "stage": report["report"]["stage"],
            "final_verdict": report["report"]["final_verdict"],
            "artifacts": report["artifact_bindings"],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
