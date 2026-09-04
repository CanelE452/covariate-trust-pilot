from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from experiments.ph_online_memory_gono_v1 import protocol as protocol_module
from experiments.ph_online_memory_gono_v1.prereg import (
    build_preregistered_spec,
    freeze_preregistration,
)
from experiments.ph_online_memory_gono_v1.protocol import (
    _implementation_hashes,
    run_full_stage,
    run_pre_smoke_verification,
    run_smoke_stage,
    run_stage0_reproduction,
)
from experiments.ph_online_memory_gono_v1.artifacts import file_sha256, payload_sha256


def _freeze(root: Path) -> None:
    spec = build_preregistered_spec(
        repository={"git_commit": "abc", "branch": "main", "dirty": True},
        environment={"python": "test"},
        implementation_sha256=_implementation_hashes(),
        frozen_at_utc="2026-09-04T00:00:00+00:00",
        data_source_sha256={"test-source": "frozen"},
    )
    freeze_preregistration(root / "preregistered_spec.json", spec)


def _bound_stage0_pass(root: Path) -> dict:
    prereg = json.loads(
        (root / "preregistered_spec.json").read_text(encoding="utf-8")
    )
    hashes = prereg["stage0_reproduction"]["frozen_sha256"]
    return {
        "status": "PASS",
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "STAGE0_CACHED_THREE_ORIGIN_REPRODUCTION",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "input_sha256": {
            "m5": hashes["m5_raw"],
            "favorita": hashes["favorita_raw"],
        },
        "reference_sha256": {
            "condition_discovery": hashes["condition_discovery_panel"],
            "recoverability": hashes["recoverability_panel"],
        },
        "checks": {"schema": True},
    }


def _bound_verification_pass(root: Path) -> dict:
    prereg = json.loads(
        (root / "preregistered_spec.json").read_text(encoding="utf-8")
    )
    return {
        "status": "PASS",
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "PRE_SMOKE_VERIFICATION",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "stage0_reproduction_sha256": file_sha256(
            root / "stage0_reproduction.json"
        ),
        "test_exit_code": 0,
    }


class ProtocolOrderTests(unittest.TestCase):
    def test_environment_rejects_inherited_kmp_bypass_before_version_reads(self):
        with patch.object(
            protocol_module, "_INHERITED_KMP_DUPLICATE_LIB_OK", "True"
        ), patch.object(
            protocol_module.importlib_metadata,
            "version",
            side_effect=AssertionError("must fail before dependency version reads"),
        ), self.assertRaisesRegex(RuntimeError, "KMP_DUPLICATE_LIB_OK"):
            protocol_module._environment_manifest()

    def test_stage0_rejects_frozen_implementation_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _freeze(root)
            with patch(
                "experiments.ph_online_memory_gono_v1.protocol._implementation_hashes",
                return_value={"changed.py": "0" * 64},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol.reproduce_three_origin",
                return_value={"status": "PASS"},
            ) as reproduce, self.assertRaisesRegex(
                RuntimeError, "(?i)(implementation|hash|drift)"
            ):
                run_stage0_reproduction(result_root=root)
            reproduce.assert_not_called()

    def test_stage0_requires_frozen_prereg_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                run_stage0_reproduction(result_root=root)

            _freeze(root)
            with patch(
                "experiments.ph_online_memory_gono_v1.protocol.reproduce_three_origin",
                return_value={"status": "PASS", "checks": {"schema": True}},
            ) as reproduce, patch(
                "experiments.ph_online_memory_gono_v1.protocol._environment_manifest",
                return_value={"python": "test"},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol.verify_forbidden_artifacts",
                return_value={"all_unchanged": True},
            ):
                report = run_stage0_reproduction(result_root=root)
                self.assertEqual(report["status"], "PASS")
                reproduce.assert_called_once()
                stored = json.loads(
                    (root / "stage0_reproduction.json").read_text(encoding="utf-8")
                )
                self.assertEqual(stored["status"], "PASS")
                with self.assertRaises(FileExistsError):
                    run_stage0_reproduction(result_root=root)

    def test_smoke_requires_stage0_pass_and_writes_append_only_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _freeze(root)
            with self.assertRaises(FileNotFoundError):
                run_smoke_stage(result_root=root)

            (root / "stage0_reproduction.json").write_text(
                json.dumps({"status": "FAIL"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "(?i)(stage.?0|pass)"):
                run_smoke_stage(result_root=root)
            (root / "stage0_reproduction.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "(?i)(stage.?0|prereg|bound)"):
                run_smoke_stage(result_root=root)
            (root / "stage0_reproduction.json").write_text(
                json.dumps(_bound_stage0_pass(root)), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "pre_smoke_verification"
            ):
                run_smoke_stage(result_root=root)
            (root / "pre_smoke_verification.json").write_text(
                json.dumps(_bound_verification_pass(root)), encoding="utf-8"
            )

            m5 = {
                "manifest": {"dataset": "m5", "eligible_independent": 29_059},
                "data": {},
                "descriptors": pd.DataFrame(),
                "cfg": object(),
            }
            favorita = {
                "manifest": {"dataset": "favorita", "eligible_independent": 55_561},
                "data": {},
                "descriptors": pd.DataFrame(),
                "cfg": object(),
            }
            smoke = {
                "report": {
                    "experiment": "PH-ONLINE-MEMORY-GONO-v1",
                    "stage": "M5_200_SERIES_SMOKE",
                    "runtime_gate": {
                        "threshold_gpu_hours": 6.0,
                        "projected_gpu_hours": 9.0,
                        "exceeded": True,
                        "action": "STOP_FOR_APPROVAL",
                    },
                    "runtime_projection_2000_per_dataset": {
                        "gpu_hours": 0.8
                    },
                },
                "predictions": pd.DataFrame({"x": [1]}),
                "losses": pd.DataFrame({"loss": [2.0]}),
            }

            def load(name, min_positive=20):
                self.assertEqual(min_positive, 20)
                return m5 if name == "m5" else favorita

            with patch(
                "experiments.ph_online_memory_gono_v1.protocol.load_independent_population",
                side_effect=load,
            ) as population_loader, patch(
                "experiments.ph_online_memory_gono_v1.protocol.run_m5_smoke",
                return_value=smoke,
            ) as smoke_runner, patch(
                "experiments.ph_online_memory_gono_v1.protocol.torch.cuda.is_available",
                return_value=True,
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol._data_source_hashes",
                return_value={"test-source": "frozen"},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol._environment_manifest",
                return_value={"python": "test"},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol.verify_forbidden_artifacts",
                return_value={"status": "PASS", "all_unchanged": True},
            ) as integrity:
                report = run_smoke_stage(result_root=root)
                (root / "STATUS_AFTER_RESOLUTION.md").unlink()
                resumed = run_smoke_stage(result_root=root)

            self.assertEqual(population_loader.call_count, 2)
            smoke_runner.assert_called_once()
            self.assertEqual(resumed, report)
            self.assertTrue(report["runtime_gate"]["exceeded"])
            attempt = root / report["attempt"]["path"]
            self.assertTrue((attempt / "population_audit.json").exists())
            self.assertTrue((attempt / "predictions.parquet").exists())
            self.assertTrue((attempt / "expert_losses.parquet").exists())
            self.assertTrue((root / "runtime_estimate.json").exists())
            self.assertTrue((root / "STATUS_AFTER_RESOLUTION.md").exists())
            self.assertFalse((root / "final_gate_report.json").exists())
            self.assertFalse((root / "tables_a_to_g.json").exists())
            self.assertIn("artifacts", report)
            self.assertEqual(integrity.call_count, 3)
            self.assertTrue(
                report["forbidden_artifact_integrity"]["after_smoke"][
                    "all_unchanged"
                ]
            )

    def test_failed_smoke_attempt_does_not_block_a_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _freeze(root)
            (root / "stage0_reproduction.json").write_text(
                json.dumps(_bound_stage0_pass(root)), encoding="utf-8"
            )
            (root / "pre_smoke_verification.json").write_text(
                json.dumps(_bound_verification_pass(root)), encoding="utf-8"
            )
            population = {
                "manifest": {"dataset": "m5", "eligible_independent": 200},
                "data": {},
                "descriptors": pd.DataFrame(),
                "cfg": object(),
            }
            successful = {
                "report": {"runtime_gate": {"exceeded": True}},
                "predictions": pd.DataFrame({"x": [1]}),
                "losses": pd.DataFrame({"loss": [2.0]}),
            }
            common = (
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol.load_independent_population",
                    return_value=population,
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol.torch.cuda.is_available",
                    return_value=True,
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol._data_source_hashes",
                    return_value={"test-source": "frozen"},
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol._environment_manifest",
                    return_value={"python": "test"},
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol.verify_forbidden_artifacts",
                    return_value={"status": "PASS", "all_unchanged": True},
                ),
            )
            with common[0], common[1], common[2], common[3], common[4], patch(
                "experiments.ph_online_memory_gono_v1.protocol.run_m5_smoke",
                side_effect=RuntimeError("simulated failure"),
            ), self.assertRaisesRegex(RuntimeError, "simulated failure"):
                run_smoke_stage(result_root=root)

            with patch(
                "experiments.ph_online_memory_gono_v1.protocol.load_independent_population",
                return_value=population,
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol.torch.cuda.is_available",
                return_value=True,
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol._data_source_hashes",
                return_value={"test-source": "frozen"},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol._environment_manifest",
                return_value={"python": "test"},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol.verify_forbidden_artifacts",
                return_value={"status": "PASS", "all_unchanged": True},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol.run_m5_smoke",
                return_value=successful,
            ):
                report = run_smoke_stage(result_root=root)

            attempts = sorted((root / "smoke").glob("attempt_*"))
            self.assertEqual([path.name for path in attempts], ["attempt_0001", "attempt_0002"])
            self.assertEqual(report["attempt"]["id"], "attempt_0002")

    def test_pre_smoke_verification_is_bound_after_stage0_and_before_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _freeze(root)
            (root / "stage0_reproduction.json").write_text(
                json.dumps(_bound_stage0_pass(root)), encoding="utf-8"
            )
            completed = __import__("subprocess").CompletedProcess(
                args=["python", "-m", "unittest"],
                returncode=0,
                stdout="Ran 99 tests\n\nOK\n",
                stderr="",
            )
            with patch(
                "experiments.ph_online_memory_gono_v1.protocol._environment_manifest",
                return_value={"python": "test"},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol._data_source_hashes",
                return_value={"test-source": "frozen"},
            ), patch(
                "experiments.ph_online_memory_gono_v1.protocol.subprocess.run",
                return_value=completed,
            ) as execute:
                report = run_pre_smoke_verification(result_root=root)

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["test_exit_code"], 0)
            self.assertEqual(
                report["stage0_reproduction_sha256"],
                file_sha256(root / "stage0_reproduction.json"),
            )
            self.assertIn("unittest", report["command"])
            execute.assert_called_once()
            with self.assertRaises(FileExistsError):
                run_pre_smoke_verification(result_root=root)

    def test_full_stage_publishes_bound_append_only_reports_and_reuses_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _freeze(root)
            prereg = json.loads(
                (root / "preregistered_spec.json").read_text(encoding="utf-8")
            )
            (root / "runtime_estimate.json").write_text(
                json.dumps({"completed": True}), encoding="utf-8"
            )
            original_status = b"preserved phase-zero status\n"
            (root / "STATUS.md").write_bytes(original_status)

            def persisted_runner(result_root, device):
                self.assertEqual(Path(result_root), root)
                self.assertEqual(str(device), "cuda")
                final = {
                    "experiment": "PH-ONLINE-MEMORY-GONO-v1",
                    "stage": "FULL_PROTOCOL",
                    "executed_model_seeds": [0],
                    "seed_reports": {"0": {"terminal": True}},
                    "robustness": {"passed": True},
                    "terminal": True,
                    "next_action": "STOP",
                    "final_verdict": "RETRIEVAL_MEMORY_GO",
                    "preregistration_sha256": prereg[
                        "preregistration_sha256"
                    ],
                    "runtime_estimate_sha256": file_sha256(
                        root / "runtime_estimate.json"
                    ),
                    "seed_analysis_artifacts": {
                        "0": {
                            "path": "seed0/analysis/attempt_0001",
                            "completion_file_sha256": "b" * 64,
                            "completion_payload_sha256": "c" * 64,
                        }
                    },
                }
                final["final_gate_report_payload_sha256"] = payload_sha256(final)
                path = root / "final_gate_report.json"
                if not path.exists():
                    path.write_text(json.dumps(final), encoding="utf-8")
                return {"report": {}, "seed_results": {}, "tables": {}}

            integrity = {
                "status": "PASS",
                "all_unchanged": True,
                "directories": {},
            }
            patches = (
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "run_persisted_full_protocol",
                    side_effect=persisted_runner,
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "_validate_phase0_authorization"
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "_verify_frozen_implementation"
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "_verify_frozen_environment"
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "_verify_frozen_data_sources"
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol._read_stage0_pass"
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol._read_pre_smoke_pass"
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "verify_forbidden_artifacts",
                    return_value=integrity,
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "build_tables_a_to_g",
                    return_value={"table_a": [], "table_g": []},
                ),
                patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "build_status_markdown",
                    return_value="FINAL VERDICT: RETRIEVAL_MEMORY_GO\n",
                ),
            )
            with (
                patches[0] as runner,
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
                patches[8],
                patches[9],
            ):
                first = run_full_stage(result_root=root)
                table_bytes = (root / "tables_a_to_g.json").read_bytes()
                status_bytes = (root / "STATUS_AFTER_RESOLUTION.md").read_bytes()
                finalization_bytes = (root / "finalization_manifest.json").read_bytes()
                second = run_full_stage(result_root=root)

            self.assertEqual(runner.call_count, 2)
            self.assertEqual(first["report"]["final_verdict"], "RETRIEVAL_MEMORY_GO")
            self.assertEqual(second["report"], first["report"])
            self.assertEqual(
                (root / "tables_a_to_g.json").read_bytes(), table_bytes
            )
            self.assertEqual(
                (root / "STATUS_AFTER_RESOLUTION.md").read_bytes(), status_bytes
            )
            self.assertEqual(
                (root / "finalization_manifest.json").read_bytes(),
                finalization_bytes,
            )
            self.assertEqual((root / "STATUS.md").read_bytes(), original_status)
            tables = json.loads(table_bytes)
            self.assertEqual(
                tables["final_gate_report_sha256"],
                file_sha256(root / "final_gate_report.json"),
            )
            self.assertEqual(
                tables["runtime_estimate_sha256"],
                file_sha256(root / "runtime_estimate.json"),
            )


if __name__ == "__main__":
    unittest.main()
