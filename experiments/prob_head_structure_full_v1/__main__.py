"""CLI driver: run the frozen thirty stages, judge, report, and mark completion.

The driver never stops at "training finished". Execution flows straight into the gate
calculation, the figures, STATUS.md and the sixteen-line console summary, and only then
writes DRIVER_COMPLETE.json. Completion is decided by that artifact, never by an exit
code, so a crash mid-way leaves no completion marker behind.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

from .reporting import build_tables, console_summary, render_status
from .run import run_pipeline

DRIVER_MARKER = "DRIVER_COMPLETE.json"
DRIVER_FAILURE = "DRIVER_FAILED.json"


def _environment() -> dict[str, str]:
    import torch

    record = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "device": "cpu",
    }
    if torch.cuda.is_available():
        record["device"] = torch.cuda.get_device_name(0)
    return record


def _runtime_decision(runs_root: Path) -> dict[str, object]:
    payload = json.loads(
        (runs_root / "runtime_tier_selection/attempt_0001/stage_payload.json").read_text(
            encoding="utf-8"
        )
    )
    return payload["decision"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prob_head_structure_full_v1")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args(argv)

    runs_root = arguments.runs_root
    results_root = arguments.results_root
    results_root.mkdir(parents=True, exist_ok=True)

    if arguments.report_only:
        payload = json.loads(Path(arguments.payload).read_text(encoding="utf-8"))
        tables = build_tables(payload)
        status_path = results_root / "STATUS.md"
        status_path.write_text(render_status(payload, tables), encoding="utf-8")
        for line in console_summary({**payload, "status_path": str(status_path)}):
            print(line, flush=True)
        return 0

    from .stages import STAGE_FUNCTIONS

    started = time.time()
    context = {
        "repository_root": arguments.repository_root.resolve(),
        "results_root": results_root,
        "runtime_decision": _runtime_decision(runs_root),
        "environment": _environment(),
        "runtime_totals": {},
    }
    print(f"[driver] tier={context['runtime_decision']['runtime_tier']} "
          f"device={context['environment']['device']}", flush=True)

    try:
        report = run_pipeline(runs_root, STAGE_FUNCTIONS, context=context)
    except Exception as error:  # a crash must leave no completion marker
        (runs_root / DRIVER_FAILURE).write_text(
            json.dumps(
                {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[driver] FAILED {type(error).__name__}: {error}", flush=True)
        raise

    elapsed = time.time() - started
    context["runtime_totals"] = {
        "wall_hours": round(elapsed / 3600.0, 3),
        "gpu_hours": round(elapsed / 3600.0, 3),
    }

    summary = report.get("console_summary") or context.get("console_summary") or []
    totals = context["runtime_totals"]
    summary = [
        f"total GPU/wall time: {totals['gpu_hours']} GPU-h / {totals['wall_hours']} wall-h"
        if line.startswith("total GPU/wall time")
        else line
        for line in summary
    ]
    status_path = results_root / "STATUS.md"
    marker = {
        "status": report["status"],
        "stop_reason": report["stop_reason"],
        "stages_completed": len(report["stages"]),
        "gates": report["gates"],
        "failed_scientific_gates": report["failed_scientific_gates"],
        "wall_hours": context["runtime_totals"]["wall_hours"],
        "status_path": str(status_path),
        "status_exists": status_path.exists(),
        "console_summary": summary,
    }
    (runs_root / DRIVER_MARKER).write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("", flush=True)
    for line in summary:
        print(line, flush=True)
    print("", flush=True)
    print(f"[driver] pipeline status : {report['status']}", flush=True)
    print(f"[driver] wall hours      : {marker['wall_hours']}", flush=True)
    print(f"[driver] failed gates    : {report['failed_scientific_gates']}", flush=True)
    print(f"[driver] STATUS written  : {status_path.exists()} -> {status_path}", flush=True)
    return 0 if report["status"] in {"COMPLETE", "RESOURCE_CAP_PARTIAL_COMPLETION"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
