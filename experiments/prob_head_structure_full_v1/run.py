"""Stage orchestration for the frozen thirty-step execution order.

A scientific gate failure never stops the run: it marks every downstream branch
diagnostic and the remaining stages still execute. Only a hard integrity failure or the
wall-clock cap ends the run early, and a later diagnostic success can never flip an
upstream gate back to PASS.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .integrity import publish_completion_marker, reserve_or_resume_attempt

STAGE_ORDER: tuple[str, ...] = (
    "git/repository audit",
    "existing artifact hash baseline",
    "external likelihood source audit",
    "environment creation",
    "preregistration freeze",
    "likelihood numerical unit test",
    "synthetic DGP audit",
    "200-series smoke",
    "runtime tier selection",
    "Stage S1 synthetic 18-cell teacher training",
    "Stage S2 specialization/oracle/structure analysis",
    "Stage C-SYN known-change experiment",
    "real count dataset audit/download",
    "Stage R1 real teacher training",
    "Stage R2 real complementarity",
    "CDF pool",
    "Stage A student distillation",
    "Stage B regret predictability",
    "Stage B structure-conditioned distillation",
    "Stage C failure sensor",
    "Stage C actionable policy",
    "all negative controls",
    "bootstrap",
    "final gate calculation",
    "figures",
    "STATUS",
    "artifact hash verification",
    "test suite",
    "commit",
    "optional push",
)

# Reporting derives from persisted artifacts rather than computing science, so it must
# re-render every run. Resuming it would replay an older run's report.
ALWAYS_RERUN_STAGES: frozenset[str] = frozenset(
    {
        # The frozen P2 weights stay sealed and are reused, never re-searched; the stage
        # re-runs only to attach its diagnostic outer application and the R3 gate.
        "CDF pool",
        "final gate calculation",
        "figures",
        "STATUS",
        "artifact hash verification",
        "test suite",
        "commit",
        "optional push",
    }
)

WALL_CLOCK_CAP_SECONDS = 24 * 60 * 60
RESOURCE_CAP_TOKEN = "RESOURCE_CAP_PARTIAL_COMPLETION"
HARD_STOP_STATUS = "HARD_INTEGRITY_STOP"


class HardIntegrityFailure(RuntimeError):
    """An integrity contract broke; the branch or the whole run must stop."""


class StageInputUnavailable(RuntimeError):
    """A stage could not obtain its declared input, so it must not be sealed complete."""


class StageNotImplemented(RuntimeError):
    """The stage has no implementation yet.

    It is deliberately left without a completion marker so a later run reserves a fresh
    attempt and executes it, instead of resuming a placeholder forever.
    """


class ResourceCapReached(RuntimeError):
    """The frozen wall-clock cap was reached at an atomic stage boundary."""

    def __init__(self, message: str = "") -> None:
        super().__init__(f"{RESOURCE_CAP_TOKEN}: {message}".strip(": "))


def stage_slug(stage: str) -> str:
    """A filesystem-safe, stable directory name for one stage."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(stage).lower()).strip("_")
    if not slug:
        raise ValueError("a stage name must contain at least one alphanumeric character")
    return slug


@dataclass
class StageResult:
    stage: str
    status: str
    payload: dict[str, Any]
    resumed: bool = False
    confirmatory_eligible: bool = True
    scientific_role: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "resumed": self.resumed,
            "confirmatory_eligible": self.confirmatory_eligible,
            "scientific_role": self.scientific_role,
            "payload": self.payload,
        }


@dataclass
class ExecutionLedger:
    """Immutable gate lineage: a recorded verdict can never be rewritten."""

    gates: dict[str, bool] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def record_gate(self, name: str, *, passed: bool) -> None:
        key = str(name)
        if key in self.gates and self.gates[key] != bool(passed):
            raise ValueError(
                f"gate {key} is already recorded as "
                f"{'PASS' if self.gates[key] else 'FAIL'} and cannot be rewritten"
            )
        if key not in self.gates:
            self.order.append(key)
        self.gates[key] = bool(passed)

    def status(self, name: str) -> str:
        if str(name) not in self.gates:
            return "NOT_EVALUATED"
        return "PASS" if self.gates[str(name)] else "FAIL"

    def failed_gates(self) -> list[str]:
        return [name for name in self.order if not self.gates[name]]

    def first_failure(self, required: Sequence[str]) -> str | None:
        for name in required:
            if str(name) in self.gates and not self.gates[str(name)]:
                return str(name)
        return None

    def confirmatory_eligible(self, required: Sequence[str]) -> bool:
        return self.first_failure(required) is None

    def scientific_role(self, required: Sequence[str]) -> str | None:
        failure = self.first_failure(required)
        return None if failure is None else f"DIAGNOSTIC_CONTINUATION_AFTER_{failure}"

    def branch_record(self, branch: str, required: Sequence[str]) -> dict[str, Any]:
        names = [str(name) for name in required]
        return {
            "branch": str(branch),
            "upstream_required_gates": names,
            "upstream_gate_status": {name: self.status(name) for name in names},
            "confirmatory_eligible": self.confirmatory_eligible(names),
            "scientific_role": self.scientific_role(names),
        }


def _next_attempt(root: Path, slug: str) -> Path:
    """Reserve a fresh attempt for a stage that must re-render rather than resume."""
    stage_root = root / slug
    stage_root.mkdir(parents=True, exist_ok=True)
    existing = [int(item.name[-4:]) for item in stage_root.glob("attempt_*") if item.is_dir()]
    attempt = stage_root / f"attempt_{max(existing, default=0) + 1:04d}"
    attempt.mkdir(exist_ok=False)
    return attempt.resolve()


def _write_payload(attempt: Path, payload: Mapping[str, Any]) -> Path:
    path = attempt / "stage_payload.json"
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def run_pipeline(
    runs_root: Path,
    stage_functions: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any]]],
    *,
    stages: Sequence[str] = STAGE_ORDER,
    wall_clock_cap_seconds: float = WALL_CLOCK_CAP_SECONDS,
    clock: Callable[[], float] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the frozen stage order with append-only attempts and resume."""
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    ticker = clock or time.monotonic
    started = ticker()
    ledger = ExecutionLedger()
    # The caller's dict is used directly so stage artifacts remain visible after the run.
    shared: dict[str, Any] = context if isinstance(context, dict) else dict(context or {})
    shared["ledger"] = ledger
    shared["runs_root"] = root

    results: list[dict[str, Any]] = []
    status = "COMPLETE"
    stop_reason = ""

    for stage in stages:
        if ticker() - started >= float(wall_clock_cap_seconds):
            status = RESOURCE_CAP_TOKEN
            stop_reason = f"{RESOURCE_CAP_TOKEN}: stopped before {stage}"
            break

        attempt, resumed = reserve_or_resume_attempt(root, stage_slug(stage))
        if resumed and stage in ALWAYS_RERUN_STAGES:
            attempt = _next_attempt(root, stage_slug(stage))
            resumed = False
        if resumed:
            payload = json.loads((attempt / "stage_payload.json").read_text(encoding="utf-8"))
            # A resumed stage must re-register the verdicts it sealed, otherwise the final
            # gate calculation sees an already-decided gate as never evaluated.
            for gate, verdict in dict(payload.get("gates", {})).items():
                ledger.record_gate(str(gate), passed=bool(verdict))
            results.append(
                StageResult(stage=stage, status="COMPLETE", payload=payload, resumed=True).as_dict()
            )
            continue

        function = stage_functions.get(stage)
        if function is None:
            raise KeyError(f"no stage function registered for {stage!r}")
        shared["stage"] = stage
        shared["attempt"] = attempt
        try:
            payload = dict(function(shared) or {})
        except StageInputUnavailable as error:
            # No completion marker: the stage will be retried once its input exists.
            results.append(
                StageResult(
                    stage=stage,
                    status="INPUT_UNAVAILABLE",
                    payload={"status": "STAGE_INPUT_UNAVAILABLE", "reason": str(error)},
                ).as_dict()
            )
            continue
        except StageNotImplemented as error:
            # No completion marker: the stage stays incomplete and will be retried.
            results.append(
                StageResult(
                    stage=stage,
                    status="NOT_IMPLEMENTED",
                    payload={"status": "STAGE_NOT_IMPLEMENTED", "reason": str(error)},
                ).as_dict()
            )
            continue
        except HardIntegrityFailure as error:
            status = HARD_STOP_STATUS
            stop_reason = str(error)
            results.append(
                StageResult(
                    stage=stage, status="HARD_INTEGRITY_STOP", payload={"error": str(error)}
                ).as_dict()
            )
            break

        payload_path = _write_payload(attempt, payload)
        # The marker must bind every file the stage produced, not just its payload.
        artifacts = sorted(
            (item for item in attempt.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(attempt).as_posix(),
        )
        publish_completion_marker(attempt, {"stage": stage, "status": "COMPLETE"}, artifacts)
        results.append(StageResult(stage=stage, status="COMPLETE", payload=payload).as_dict())

    return {
        "status": status,
        "stop_reason": stop_reason,
        "stages": results,
        "gates": {name: ledger.status(name) for name in ledger.order},
        "failed_scientific_gates": ledger.failed_gates(),
        "elapsed_seconds": float(ticker() - started),
        "stage_order": list(stages),
        "console_summary": list(shared.get("console_summary", [])),
    }


__all__ = [
    "HARD_STOP_STATUS",
    "RESOURCE_CAP_TOKEN",
    "ALWAYS_RERUN_STAGES",
    "STAGE_ORDER",
    "WALL_CLOCK_CAP_SECONDS",
    "ExecutionLedger",
    "HardIntegrityFailure",
    "ResourceCapReached",
    "StageInputUnavailable",
    "StageNotImplemented",
    "StageResult",
    "run_pipeline",
    "stage_slug",
]
