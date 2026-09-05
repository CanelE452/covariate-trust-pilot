"""Watch this process's memory so a long run stops itself instead of the machine.

A run that exhausts the system commit limit takes unrelated programs down with it, and
the failure surfaces far away from its cause. The guard samples the working set at each
stage boundary, records the peak into the stage payload, and raises once the process
crosses a share of physical memory that leaves the rest of the machine usable.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_LIMIT_FRACTION = 0.45
_BYTES_PER_GB = 1024.0**3


class MemoryBudgetExceeded(RuntimeError):
    """The process crossed its own memory budget at a stage boundary."""


def _psutil():
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil ships with the pinned environment
        return None
    return psutil


def total_physical_bytes() -> float:
    module = _psutil()
    if module is not None:
        return float(module.virtual_memory().total)
    return 0.0


def process_bytes() -> float:
    """Resident bytes for this process and every child it spawned."""
    module = _psutil()
    if module is None:
        return 0.0
    process = module.Process(os.getpid())
    total = float(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            total += float(child.memory_info().rss)
        except (module.NoSuchProcess, module.AccessDenied):
            continue
    return total


def system_available_bytes() -> float:
    module = _psutil()
    if module is None:
        return 0.0
    return float(module.virtual_memory().available)


class MemoryGuard:
    """Sample memory at stage boundaries and keep the peak for the report."""

    def __init__(self, *, limit_fraction: float = DEFAULT_LIMIT_FRACTION) -> None:
        self.limit_fraction = float(limit_fraction)
        self.total = total_physical_bytes()
        self.limit = self.total * self.limit_fraction if self.total else 0.0
        self.peak = 0.0
        self.peak_stage = ""

    def sample(self, stage: str) -> dict[str, Any]:
        current = process_bytes()
        if current > self.peak:
            self.peak, self.peak_stage = current, str(stage)
        return {
            "stage": str(stage),
            "process_rss_gb": round(current / _BYTES_PER_GB, 3),
            "system_available_gb": round(system_available_bytes() / _BYTES_PER_GB, 3),
            "peak_rss_gb": round(self.peak / _BYTES_PER_GB, 3),
        }

    def check(self, stage: str) -> dict[str, Any]:
        """Sample, then refuse to start the next stage once over budget."""
        reading = self.sample(stage)
        if self.limit and process_bytes() > self.limit:
            raise MemoryBudgetExceeded(
                "MEMORY_BUDGET_EXCEEDED: "
                f"{reading['process_rss_gb']} GB resident before {stage!r} exceeds the "
                f"{round(self.limit / _BYTES_PER_GB, 1)} GB budget "
                f"({self.limit_fraction:.0%} of {round(self.total / _BYTES_PER_GB, 1)} GB)"
            )
        return reading

    def summary(self) -> dict[str, Any]:
        return {
            "peak_rss_gb": round(self.peak / _BYTES_PER_GB, 3),
            "peak_stage": self.peak_stage,
            "budget_gb": round(self.limit / _BYTES_PER_GB, 1),
            "total_physical_gb": round(self.total / _BYTES_PER_GB, 1),
            "limit_fraction": self.limit_fraction,
        }


__all__ = [
    "DEFAULT_LIMIT_FRACTION",
    "MemoryBudgetExceeded",
    "MemoryGuard",
    "process_bytes",
    "system_available_bytes",
    "total_physical_bytes",
]
