from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.prob_head_structure_full_v1.figures import FIGURE_NAMES, render_figures
from experiments.prob_head_structure_full_v1.reporting import (
    CONSOLE_ORDER,
    STATUS_FIRST_LINE_PREFIX,
    STATUS_SECTIONS,
    TABLE_IDS,
    build_tables,
    console_summary,
    render_status,
)
from experiments.prob_head_structure_full_v1.run import (
    STAGE_ORDER,
    ExecutionLedger,
    HardIntegrityFailure,
    ResourceCapReached,
    StageResult,
    run_pipeline,
)


def _stage_functions(overrides=None):
    overrides = overrides or {}
    functions = {}
    for name in STAGE_ORDER:
        functions[name] = overrides.get(name, lambda context, name=name: {"stage": name})
    return functions


class StageOrderTests(unittest.TestCase):
    def test_the_thirty_stage_order_is_frozen_and_matches_the_protocol(self):
        self.assertEqual(len(STAGE_ORDER), 30)
        self.assertEqual(STAGE_ORDER[0], "git/repository audit")
        self.assertEqual(STAGE_ORDER[9], "Stage S1 synthetic 18-cell teacher training")
        self.assertEqual(STAGE_ORDER[23], "final gate calculation")
        self.assertEqual(STAGE_ORDER[-1], "optional push")
        self.assertEqual(len(set(STAGE_ORDER)), 30)


class LedgerLineageTests(unittest.TestCase):
    def test_a_scientific_failure_marks_downstream_diagnostic_without_stopping(self):
        ledger = ExecutionLedger()
        ledger.record_gate("R3", passed=False)
        self.assertFalse(ledger.confirmatory_eligible(["R2", "R3"]))
        self.assertEqual(
            ledger.scientific_role(["R2", "R3"]), "DIAGNOSTIC_CONTINUATION_AFTER_R3"
        )

    def test_a_later_diagnostic_success_cannot_flip_an_upstream_failure(self):
        ledger = ExecutionLedger()
        ledger.record_gate("R3", passed=False)
        ledger.record_gate("A2", passed=True)
        self.assertFalse(ledger.confirmatory_eligible(["R3", "A2"]))
        with self.assertRaises(ValueError):
            ledger.record_gate("R3", passed=True)

    def test_branch_eligibility_serializes_the_declared_fields(self):
        ledger = ExecutionLedger()
        ledger.record_gate("R2", passed=True)
        ledger.record_gate("R3", passed=False)
        record = ledger.branch_record("A_DISTILLATION", ["R2", "R3"])
        self.assertEqual(record["branch"], "A_DISTILLATION")
        self.assertEqual(record["upstream_required_gates"], ["R2", "R3"])
        self.assertEqual(record["upstream_gate_status"], {"R2": "PASS", "R3": "FAIL"})
        self.assertIs(record["confirmatory_eligible"], False)
        self.assertEqual(record["scientific_role"], "DIAGNOSTIC_CONTINUATION_AFTER_R3")


class PipelineTests(unittest.TestCase):
    def test_every_stage_runs_in_order_and_publishes_an_append_only_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_pipeline(root / "runs", _stage_functions())
            self.assertEqual([item["stage"] for item in report["stages"]], list(STAGE_ORDER))
            self.assertEqual(report["status"], "COMPLETE")
            for stage in STAGE_ORDER:
                marker = root / "runs" / _slug(stage) / "attempt_0001" / "completion.json"
                self.assertTrue(marker.exists(), stage)
                self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["status"], "COMPLETE")

    def test_a_verified_successful_attempt_is_resumed_and_never_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = {"count": 0}

            def counted(context):
                calls["count"] += 1
                return {"stage": "counted"}

            functions = _stage_functions({STAGE_ORDER[0]: counted})
            run_pipeline(root / "runs", functions)
            self.assertEqual(calls["count"], 1)
            second = run_pipeline(root / "runs", functions)
            self.assertEqual(calls["count"], 1)
            self.assertTrue(second["stages"][0]["resumed"])

    def test_a_scientific_gate_failure_does_not_stop_the_remaining_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def failing(context):
                context["ledger"].record_gate("S1", passed=False)
                return {"gate": "S1", "passed": False}

            report = run_pipeline(
                root / "runs", _stage_functions({STAGE_ORDER[10]: failing})
            )
            self.assertEqual(len(report["stages"]), 30)
            self.assertEqual(report["status"], "COMPLETE")
            self.assertIn("S1", report["failed_scientific_gates"])

    def test_a_hard_integrity_failure_stops_the_run_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def hard(context):
                raise HardIntegrityFailure("EXISTING_ARTIFACT_MUTATION_HARD_STOP")

            report = run_pipeline(root / "runs", _stage_functions({STAGE_ORDER[2]: hard}))
            self.assertEqual(report["status"], "HARD_INTEGRITY_STOP")
            self.assertEqual(len(report["stages"]), 3)
            self.assertIn("EXISTING_ARTIFACT_MUTATION_HARD_STOP", report["stop_reason"])

    def test_reaching_the_wall_clock_cap_ends_the_run_as_a_partial_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ticks = iter(range(0, 100_000, 20_000))
            report = run_pipeline(
                root / "runs", _stage_functions(), wall_clock_cap_seconds=30_000,
                clock=lambda: next(ticks),
            )
            self.assertEqual(report["status"], "RESOURCE_CAP_PARTIAL_COMPLETION")
            self.assertLess(len(report["stages"]), 30)

    def test_the_cap_error_names_the_frozen_token(self):
        self.assertIn("RESOURCE_CAP_PARTIAL_COMPLETION", str(ResourceCapReached()))


class ReportingTests(unittest.TestCase):
    def _payload(self):
        return {
            "recommendation": "ALL_NEW_METHOD_BRANCHES_NO_GO",
            "runtime_tier": "MINIMAL-COMPLETE",
            "environment": {"python": "3.10.20", "torch": "2.1.1+cu118", "gpu": "RTX 4070"},
            "gates": {"S1": "FAIL", "S2": "PASS"},
            "branches": [
                {
                    "branch": "A_DISTILLATION",
                    "upstream_required_gates": ["R2"],
                    "upstream_gate_status": {"R2": "FAIL"},
                    "confirmatory_eligible": False,
                    "scientific_role": "DIAGNOSTIC_CONTINUATION_AFTER_R2",
                }
            ],
            "observations": {"synthetic_effect": 0.0, "real_oracle_headroom": 0.0},
            "runtime": {"gpu_hours": 0.0, "wall_hours": 0.0},
            "commit": "abcdef1",
            "push": "not attempted",
        }

    def test_every_table_from_a_to_t_is_produced(self):
        tables = build_tables(self._payload())
        self.assertEqual(sorted(tables), sorted(TABLE_IDS))
        self.assertEqual(len(TABLE_IDS), 20)
        for identifier, table in tables.items():
            self.assertIn("title", table)
            self.assertIn("rows", table)

    def test_status_starts_with_the_exact_first_line_and_keeps_the_section_order(self):
        text = render_status(self._payload(), build_tables(self._payload()))
        first = text.splitlines()[0]
        self.assertTrue(first.startswith(STATUS_FIRST_LINE_PREFIX))
        self.assertEqual(first, "FINAL RECOMMENDATION: ALL_NEW_METHOD_BRANCHES_NO_GO")
        positions = [text.index(f"## {index}. {name}") for index, name in enumerate(STATUS_SECTIONS, 1)]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(STATUS_SECTIONS), 21)

    def test_the_console_summary_uses_the_frozen_sixteen_item_order(self):
        lines = console_summary(self._payload())
        self.assertEqual(len(lines), len(CONSOLE_ORDER))
        self.assertEqual(len(CONSOLE_ORDER), 16)
        self.assertTrue(lines[0].startswith("FINAL RECOMMENDATION"))

    def test_a_diagnostic_branch_is_never_reported_as_confirmatory(self):
        text = render_status(self._payload(), build_tables(self._payload()))
        self.assertIn("DIAGNOSTIC_CONTINUATION_AFTER_R2", text)
        self.assertIn("Confirmatory vs diagnostic evidence", text)


class FigureTests(unittest.TestCase):
    def test_at_most_eight_named_figures_are_rendered_from_persisted_payload(self):
        self.assertEqual(len(FIGURE_NAMES), 8)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            written = render_figures({"available": False}, out)
            self.assertLessEqual(len(written), 8)
            for path in written:
                self.assertTrue(path.exists())


def _slug(stage: str) -> str:
    from experiments.prob_head_structure_full_v1.run import stage_slug

    return stage_slug(stage)


if __name__ == "__main__":
    unittest.main()
