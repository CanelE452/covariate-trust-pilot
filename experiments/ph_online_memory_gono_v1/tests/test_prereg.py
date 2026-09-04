"""Pre-result freeze and append-only artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.ph_online_memory_gono_v1.artifacts import payload_sha256
from experiments.ph_online_memory_gono_v1.prereg import (
    build_preregistered_spec,
    freeze_preregistration,
)


class PreregistrationTest(unittest.TestCase):
    def _spec(self):
        return build_preregistered_spec(
            repository={
                "git_commit": "222c82cbf91c4313e77c531742831d211c756b02",
                "branch": "main",
                "dirty": True,
            },
            environment={
                "python": "3.10.20",
                "torch": "2.1.1+cu118",
                "cuda": "11.8",
                "gpu": "NVIDIA GeForce RTX 4070",
            },
            implementation_sha256={"trainer.py": "abc123"},
            frozen_at_utc="2026-09-04T08:00:00+00:00",
            data_source_sha256={"data/source": "def456"},
        )

    def test_spec_freezes_every_result_sensitive_contract(self):
        """Catches omitting a choice that could later be tuned after seeing results."""
        spec = self._spec()

        self.assertEqual(spec["experiment_name"], "PH-ONLINE-MEMORY-GONO-v1")
        self.assertEqual(spec["lookback"], 96)
        self.assertEqual(spec["horizon"], 28)
        self.assertEqual(spec["model_seed"], 0)
        self.assertEqual(spec["bootstrap"]["draws"], 2000)
        self.assertEqual(spec["bootstrap"]["seed"], 20260904)
        self.assertEqual(
            spec["bootstrap"]["confidence_interval_percent"], [2.5, 97.5]
        )
        self.assertEqual(spec["bootstrap"]["unit"], "series cluster")
        self.assertIn("independently", spec["bootstrap"]["macro_sampling"])
        self.assertIn("same", spec["bootstrap"]["paired_policy_resampling"])
        self.assertEqual(spec["training"]["train_origin_stride"], 7)
        self.assertEqual(spec["training"]["batch_size"], 256)
        self.assertEqual(spec["training"]["max_epochs"], 30)
        self.assertEqual(spec["training"]["patience"], 5)
        self.assertEqual(
            spec["models"]["point"]["id"], "M0PM_point_mse_param_matched"
        )
        self.assertEqual(spec["models"]["hurdle"]["id"], "M1_factorized_mean")
        self.assertEqual(
            spec["trainer"]["delegates_to"],
            "experiments.om_factorization_killtest.train.train_one",
        )
        self.assertEqual(
            spec["splits"]["m5"]["evaluation_origins"],
            [1773, 1801, 1829, 1857, 1885, 1913],
        )
        self.assertEqual(
            spec["splits"]["favorita"]["evaluation_origins"],
            [1520, 1548, 1576, 1604, 1632, 1660],
        )
        self.assertEqual(spec["eligibility"]["primary_min_positive_train"], 20)
        self.assertEqual(spec["policy_grids"]["b3_alpha"], [i / 20 for i in range(21)])
        self.assertEqual(spec["policy_grids"]["b4_eta"], [0.5, 2.0, 8.0, 32.0])
        self.assertEqual(spec["policy_grids"]["b4_half_life_origins"], [1, 3])
        self.assertEqual(spec["policy_grids"]["m1_k"], [32, 128])
        self.assertEqual(spec["policy_grids"]["m1_lambda_max"], [0.25, 0.5])
        self.assertEqual(spec["controls"]["seed"], 20260904)
        self.assertEqual(spec["controls"]["rng"], "numpy.random.default_rng (PCG64)")
        self.assertIn("NUL-separated", spec["controls"]["query_seed_derivation"])
        self.assertEqual(
            spec["controls"]["C0_seed_parts"],
            ["C0", "dataset_id", "resolved_origin"],
        )
        self.assertEqual(
            spec["controls"]["C1_seed_parts"],
            ["C1", "dataset_id", "series_id", "query_origin"],
        )
        self.assertEqual(len(spec["final_verdict_tokens"]), 8)
        self.assertIn("Gate0", spec["gates"])
        self.assertIn("Gate3_control", spec["gates"])
        self.assertEqual(
            spec["gates"]["Gate0"]["on_failure"][
                "heterogeneous_origin_convex_macro_min_percent"
            ],
            2.0,
        )
        self.assertIn(
            "NOT_AVAILABLE",
            spec["gates"]["Gate0"]["on_failure"]["availability_rule"],
        )
        self.assertEqual(
            len(spec["gates"]["Gate4"]["seed1_forbidden_if_any"]), 3
        )
        self.assertEqual(
            len(spec["gates"]["Gate4"]["borderline_if_any"]), 4
        )
        self.assertIn("prediction averaging forbidden", spec["gates"]["Gate4"]["three_seed_aggregation"])
        self.assertEqual(
            spec["gates"]["Gate4"]["seed1_still_borderline_if"],
            (
                "same sign AND retention >= 0.70 AND "
                "seed-average CI lower <= 0 < upper"
            ),
        )
        self.assertEqual(
            len(spec["gates"]["Gate4"]["seed1_clear_fail_if_any"]), 3
        )
        self.assertIn(
            "diagnostic-only",
            spec["gates"]["Gate4"]["additional_seed_upstream_gates"],
        )
        self.assertIn(
            "series-origin",
            spec["gates"]["Gate0"]["origin_convex_oracle_definition"],
        )
        self.assertEqual(spec["policy_execution"]["B0"], "always Point; hurdle weight alpha=0")
        self.assertEqual(spec["policy_execution"]["B1"], "always Hurdle; hurdle weight alpha=1")
        self.assertIn("alpha=0.5", spec["policy_execution"]["B2"])
        self.assertIn("ddof=0", spec["retrieval"]["confidence"])
        self.assertIn("series_id", spec["retrieval"]["distance_tie_break"])
        self.assertIn("DO_NOT_CONSUME_NEW_CONFIRMATORY_DATASET", spec["stop_rules"])
        self.assertEqual(spec["implementation_sha256"], {"trainer.py": "abc123"})
        self.assertEqual(spec["data_source_sha256"], {"data/source": "def456"})
        self.assertIn("exact manifest equality", spec["implementation_hash_scope"])
        self.assertEqual(
            spec["smoke"]["pipeline_validation_hyperparameters"],
            {
                "b4_eta": 0.5,
                "b4_half_life_origins": 1,
                "m1_k": 32,
                "m1_lambda_max": 0.25,
                "scientific_selection": False,
            },
        )
        self.assertEqual(
            spec["smoke"]["runtime_projection"][
                "retrieval_passes_per_dataset"
            ],
            7,
        )
        heterogeneous = spec["heterogeneous_gate0_diagnostic"]
        self.assertEqual(heterogeneous["candidate_count"], 126)
        self.assertEqual(
            heterogeneous["pair_order"][0], ["point", "hurdle"]
        )
        self.assertEqual(heterogeneous["alpha_grid"], [i / 20 for i in range(21)])
        self.assertTrue(heterogeneous["run_only_after_point_hurdle_gate0_failure"])
        self.assertEqual(len(spec["execution_order"]), 6)
        self.assertIn(
            "completion manifest",
            spec["append_only_artifacts"]["full_dataset_checkpoints"],
        )
        self.assertIn(
            "final gate report",
            spec["append_only_artifacts"]["final_tables"],
        )
        self.assertEqual(
            spec["forbidden_artifact_baseline_extension"],
            "results/ph_online_memory_gono_v1/"
            "forbidden_artifact_baseline_extension.json",
        )
        self.assertEqual(
            spec["gate2_gate4_resolution"],
            "results/ph_online_memory_gono_v1/gate2_gate4_resolution.json",
        )
        self.assertEqual(
            spec["gates"]["Gate2"]["borderline_failure_status"],
            "PENDING_GATE4",
        )
        self.assertEqual(
            spec["gates"]["Gate2"]["deferrable_failed_checks"],
            [
                "macro_effect",
                "direction_safety",
                "macro_absolute_usefulness",
                "direction_absolute_usefulness",
                "macro_ci",
                "dataset_ci",
            ],
        )
        self.assertIn(
            "Gate2 A-F",
            spec["gates"]["Gate4"]["terminal_pass_rule"],
        )
        self.assertIn(
            "FINAL_FAIL",
            spec["gates"]["Gate4"]["seed0_veto_precedence"],
        )

    def test_freeze_hashes_payload_and_refuses_overwrite(self):
        """Catches a mutable preregistration or a self-referential/undefined hash."""
        spec = self._spec()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preregistered_spec.json"
            frozen = freeze_preregistration(path, spec)
            stored = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(stored, frozen)
            digest = stored.pop("preregistration_sha256")
            hash_contract = stored.pop("preregistration_hash_contract")
            self.assertEqual(
                hash_contract,
                "SHA256 of canonical UTF-8 JSON excluding preregistration_sha256 and preregistration_hash_contract",
            )
            self.assertEqual(digest, payload_sha256(stored))
            with self.assertRaises(FileExistsError):
                freeze_preregistration(path, spec)


if __name__ == "__main__":
    unittest.main()
