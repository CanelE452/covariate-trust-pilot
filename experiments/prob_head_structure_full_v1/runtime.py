"""Immutable runtime-tier selection for PROB-HEAD-STRUCTURE-FULL-v1."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Mapping

from .preregistration import verify_preregistration


SYNTHETIC_DATA_SEEDS = (2026090501, 2026090502)
TEACHER_MODEL_SEEDS = (2026090511, 2026090512)
STUDENT_MODEL_SEEDS = (2026090511, 2026090512)

RUNTIME_TIER_CONTRACTS: dict[str, dict[str, object]] = {
    "FULL": {
        "synthetic_series_per_cell": 80,
        "synthetic_data_seeds": list(SYNTHETIC_DATA_SEEDS),
        "teacher_model_seeds": list(TEACHER_MODEL_SEEDS),
        "real_series_per_dataset": 4000,
        "student_model_seeds": list(STUDENT_MODEL_SEEDS),
        "bootstrap_draws": 2000,
        "screen_only": False,
    },
    "COMPACT": {
        "synthetic_series_per_cell": 40,
        "synthetic_data_seeds": [SYNTHETIC_DATA_SEEDS[0]],
        "teacher_model_seeds": [TEACHER_MODEL_SEEDS[0]],
        "real_series_per_dataset": 2000,
        "student_model_seeds": [STUDENT_MODEL_SEEDS[0]],
        "bootstrap_draws": 1000,
        "screen_only": False,
    },
    "MINIMAL-COMPLETE": {
        "synthetic_series_per_cell": 24,
        "synthetic_data_seeds": [SYNTHETIC_DATA_SEEDS[0]],
        "teacher_model_seeds": [TEACHER_MODEL_SEEDS[0]],
        "real_series_per_dataset": 1000,
        "student_model_seeds": [STUDENT_MODEL_SEEDS[0]],
        "bootstrap_draws": 500,
        "screen_only": True,
    },
}

_PREREGISTERED_TIER_PROJECTION = {
    "FULL": {
        "synthetic_series_per_cell": 80,
        "synthetic_DGP_seeds": 2,
        "teacher_model_seeds": 2,
        "real_series_per_dataset": 4000,
        "student_seeds": 2,
        "all_A_B_C": True,
        "bootstrap_draws": 2000,
    },
    "COMPACT": {
        "synthetic_series_per_cell": 40,
        "synthetic_DGP_seeds": 1,
        "teacher_model_seeds": 1,
        "real_series_per_dataset": 2000,
        "student_seeds": 1,
        "all_A_B_C": True,
        "bootstrap_draws": 1000,
    },
    "MINIMAL-COMPLETE": {
        "synthetic_series_per_cell": 24,
        "synthetic_DGP_seeds": 1,
        "teacher_model_seeds": 1,
        "real_series_per_dataset": 1000,
        "student_seeds": 1,
        "all_A_B_C": True,
        "bootstrap_draws": 500,
        "label": "SCREEN_ONLY",
    },
}


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_runtime_tier(projected_full_gpu_hours: float) -> str:
    """Apply the frozen threshold once to the projected complete FULL workload."""
    hours = float(projected_full_gpu_hours)
    if not math.isfinite(hours) or hours < 0.0:
        raise ValueError("projected FULL GPU hours must be finite and nonnegative")
    if hours <= 12.0:
        return "FULL"
    if hours <= 18.0:
        return "COMPACT"
    return "MINIMAL-COMPLETE"


def _validate_preregistered_runtime(frozen: Mapping[str, object]) -> None:
    payload = frozen.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("runtime decision requires a verified preregistration payload")
    identity = payload.get("identity")
    runtime = payload.get("runtime")
    if (
        not isinstance(identity, Mapping)
        or identity.get("experiment") != "PROB-HEAD-STRUCTURE-FULL-v1"
        or not isinstance(runtime, Mapping)
        or runtime.get("tiers") != _PREREGISTERED_TIER_PROJECTION
    ):
        raise ValueError("preregistration runtime-tier contract does not match the implementation")


def _decision_payload(record: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key != "runtime_decision_sha256"
    }


def seal_runtime_tier_decision(
    *,
    projected_full_gpu_seconds: float,
    smoke_projection_sha256: str,
    preregistration_path: Path,
) -> dict[str, object]:
    """Bind the one-time tier choice to the verified frozen preregistration."""
    seconds = float(projected_full_gpu_seconds)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("projected FULL GPU seconds must be finite and nonnegative")
    if re.fullmatch(r"[0-9a-f]{64}", str(smoke_projection_sha256)) is None:
        raise ValueError("smoke projection requires a lowercase SHA256")
    preregistration = Path(preregistration_path)
    frozen = verify_preregistration(preregistration)
    _validate_preregistered_runtime(frozen)
    tier = select_runtime_tier(seconds / 3600.0)
    record: dict[str, object] = {
        "record_type": "PROB_HEAD_STRUCTURE_RUNTIME_TIER_DECISION",
        "projected_full_gpu_seconds": seconds,
        "projected_full_gpu_hours": seconds / 3600.0,
        "smoke_projection_sha256": str(smoke_projection_sha256),
        "preregistration_filename": preregistration.name,
        "preregistration_payload_sha256": str(frozen["payload_sha256"]),
        "preregistration_file_sha256": _file_sha256(preregistration),
        "runtime_tier": tier,
        "tier_contract": dict(RUNTIME_TIER_CONTRACTS[tier]),
        "tier_selection_is_one_time": True,
        "tier_recomputation_after_downsizing_forbidden": True,
    }
    record["runtime_decision_sha256"] = _canonical_sha256(record)
    return record


def verify_runtime_tier_decision(
    decision: Mapping[str, object], *, preregistration_path: Path
) -> dict[str, object]:
    """Recompute the tier and verify its frozen-preregistration byte binding."""
    row = dict(decision)
    required = {
        "record_type",
        "projected_full_gpu_seconds",
        "projected_full_gpu_hours",
        "smoke_projection_sha256",
        "preregistration_filename",
        "preregistration_payload_sha256",
        "preregistration_file_sha256",
        "runtime_tier",
        "tier_contract",
        "tier_selection_is_one_time",
        "tier_recomputation_after_downsizing_forbidden",
        "runtime_decision_sha256",
    }
    if set(row) != required:
        raise ValueError("runtime-tier decision schema mismatch")
    stored_hash = row.get("runtime_decision_sha256")
    if (
        not isinstance(stored_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", stored_hash) is None
        or stored_hash != _canonical_sha256(_decision_payload(row))
    ):
        raise ValueError("runtime-tier decision SHA256 mismatch")
    preregistration = Path(preregistration_path)
    frozen = verify_preregistration(preregistration)
    _validate_preregistered_runtime(frozen)
    if (
        row["record_type"] != "PROB_HEAD_STRUCTURE_RUNTIME_TIER_DECISION"
        or row["preregistration_filename"] != preregistration.name
        or row["preregistration_payload_sha256"] != frozen["payload_sha256"]
        or row["preregistration_file_sha256"] != _file_sha256(preregistration)
        or re.fullmatch(r"[0-9a-f]{64}", str(row["smoke_projection_sha256"])) is None
        or row["tier_selection_is_one_time"] is not True
        or row["tier_recomputation_after_downsizing_forbidden"] is not True
    ):
        raise ValueError("runtime-tier decision is not bound to the frozen preregistration")
    seconds = float(row["projected_full_gpu_seconds"])
    hours = float(row["projected_full_gpu_hours"])
    if (
        not math.isfinite(seconds)
        or seconds < 0.0
        or not math.isfinite(hours)
        or hours != seconds / 3600.0
    ):
        raise ValueError("runtime-tier decision projection is invalid")
    expected_tier = select_runtime_tier(hours)
    if (
        row["runtime_tier"] != expected_tier
        or row["tier_contract"] != RUNTIME_TIER_CONTRACTS[expected_tier]
    ):
        raise ValueError("runtime-tier decision does not match the frozen thresholds")
    return row

