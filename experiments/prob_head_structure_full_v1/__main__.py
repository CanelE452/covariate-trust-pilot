"""CLI entry point for the frozen thirty-stage execution order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import build_tables, console_summary, render_status
from .run import STAGE_ORDER, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prob_head_structure_full_v1")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--payload", type=Path, help="persisted stage payload for reporting only")
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.report_only:
        payload = json.loads(Path(arguments.payload).read_text(encoding="utf-8"))
    else:
        from .stages import STAGE_FUNCTIONS

        report = run_pipeline(arguments.runs_root, STAGE_FUNCTIONS)
        payload = report.get("payload", report)

    tables = build_tables(payload)
    status_path = Path(arguments.results_root) / "STATUS.md"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(render_status(payload, tables), encoding="utf-8")
    for line in console_summary({**payload, "status_path": str(status_path)}):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
