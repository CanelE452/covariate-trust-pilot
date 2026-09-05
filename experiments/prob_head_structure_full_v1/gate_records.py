"""Persist and restore whole ``gates.GateResult`` objects across stage attempts.

A stage computes observations and hands them to the frozen reducer in ``gates``; the
reducer alone decides PASS/FAIL. This module is the transport: it seals the complete
result into the stage payload and rebuilds it later, so the final verdict is assembled
from the reducers' own objects rather than from booleans a stage recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gates import GateResult, attach_lineage

PAYLOAD_KEY = "gate_results"


def serialize(result: GateResult) -> dict[str, Any]:
    return result.as_dict()


def deserialize(record: Mapping[str, Any]) -> GateResult:
    """Rebuild the exact reducer output that a stage sealed."""
    observations = dict(record["observations"])
    return GateResult(
        gate=str(record["gate"]),
        passed=bool(record["status"] == "PASS"),
        verdict=str(record["verdict"]),
        failure_label=record["failure_label"],
        observations=observations,
        criteria=dict(record["criteria"]),
        upstream_required_gates=tuple(str(item) for item in record["upstream_required_gates"]),
        upstream_gate_status={
            str(key): str(value) for key, value in dict(record["upstream_gate_status"]).items()
        },
        confirmatory_eligible=bool(record["confirmatory_eligible"]),
        scientific_role=str(record["scientific_role"]),
    )


def record(
    context: dict[str, Any],
    result: GateResult,
    *,
    branch: str,
    upstream: Sequence[GateResult] = (),
) -> GateResult:
    """Attach the frozen lineage, register the reducer's verdict, and keep the object.

    The ledger receives the reducer's own ``passed`` flag. No caller may pass a boolean
    it computed itself, which is what keeps section 3 of the execution contract true.
    """
    bound = attach_lineage(result, branch=branch, upstream=list(upstream))
    context["ledger"].record_gate(bound.gate, passed=bound.passed)
    store = context.setdefault("gate_result_objects", {})
    store[bound.gate] = bound
    return bound


def payload_block(results: Iterable[GateResult]) -> dict[str, Any]:
    """The two blocks every gate-producing stage adds to its sealed payload."""
    items = list(results)
    return {
        "gates": {item.gate: bool(item.passed) for item in items},
        PAYLOAD_KEY: {item.gate: serialize(item) for item in items},
    }


def _sealed_attempts(runs_root: Path) -> list[Path]:
    root = Path(runs_root)
    if not root.is_dir():
        return []
    attempts: list[Path] = []
    for stage_dir in sorted(root.iterdir()):
        if not stage_dir.is_dir():
            continue
        sealed = [
            attempt
            for attempt in sorted(stage_dir.glob("attempt_*"))
            if (attempt / "completion.json").exists()
            and (attempt / "stage_payload.json").exists()
        ]
        if sealed:
            attempts.append(sealed[-1])
    return attempts


def collect_sealed(runs_root: Path) -> dict[str, GateResult]:
    """Every gate result sealed under ``runs_root``, latest attempt per stage.

    Reading the sealed payloads rather than in-memory state is what lets the final
    verdict be reproduced from the artifacts alone.
    """
    found: dict[str, GateResult] = {}
    for attempt in _sealed_attempts(runs_root):
        payload = json.loads((attempt / "stage_payload.json").read_text(encoding="utf-8"))
        for name, record_dict in dict(payload.get(PAYLOAD_KEY, {})).items():
            found[str(name)] = deserialize(record_dict)
    return found


__all__ = [
    "PAYLOAD_KEY",
    "collect_sealed",
    "deserialize",
    "payload_block",
    "record",
    "serialize",
]
