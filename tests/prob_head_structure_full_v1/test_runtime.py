from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from experiments.prob_head_structure_full_v1 import runtime


class RuntimeTierDecisionTests(unittest.TestCase):
    @staticmethod
    def _frozen() -> dict[str, object]:
        return {
            "payload_sha256": "a" * 64,
            "payload": {
                "identity": {"experiment": "PROB-HEAD-STRUCTURE-FULL-v1"},
                "runtime": {"tiers": deepcopy(runtime._PREREGISTERED_TIER_PROJECTION)},
            },
        }

    def test_thresholds_are_frozen_and_boundary_exact(self):
        self.assertEqual(runtime.select_runtime_tier(0.0), "FULL")
        self.assertEqual(runtime.select_runtime_tier(12.0), "FULL")
        self.assertEqual(runtime.select_runtime_tier(12.0000001), "COMPACT")
        self.assertEqual(runtime.select_runtime_tier(18.0), "COMPACT")
        self.assertEqual(runtime.select_runtime_tier(18.0000001), "MINIMAL-COMPLETE")
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                runtime.select_runtime_tier(invalid)

    def test_decision_binds_exact_tier_seeds_counts_and_preregistration_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            preregistration = Path(temporary) / "preregistered_spec_v4.json"
            preregistration.write_bytes(b"immutable-preregistration")
            with mock.patch.object(
                runtime, "verify_preregistration", return_value=self._frozen()
            ):
                decision = runtime.seal_runtime_tier_decision(
                    projected_full_gpu_seconds=13.0 * 3600.0,
                    smoke_projection_sha256="b" * 64,
                    preregistration_path=preregistration,
                )
                verified = runtime.verify_runtime_tier_decision(
                    decision, preregistration_path=preregistration
                )

                self.assertEqual(verified["runtime_tier"], "COMPACT")
                self.assertEqual(
                    verified["tier_contract"],
                    {
                        "synthetic_series_per_cell": 40,
                        "synthetic_data_seeds": [2026090501],
                        "teacher_model_seeds": [2026090511],
                        "real_series_per_dataset": 2000,
                        "student_model_seeds": [2026090511],
                        "bootstrap_draws": 1000,
                        "screen_only": False,
                    },
                )

                forged = deepcopy(decision)
                forged["runtime_tier"] = "FULL"
                forged["tier_contract"] = deepcopy(runtime.RUNTIME_TIER_CONTRACTS["FULL"])
                forged["runtime_decision_sha256"] = runtime._canonical_sha256(
                    runtime._decision_payload(forged)
                )
                with self.assertRaisesRegex(ValueError, "frozen thresholds"):
                    runtime.verify_runtime_tier_decision(
                        forged, preregistration_path=preregistration
                    )

                undersized = deepcopy(decision)
                undersized["tier_contract"]["synthetic_series_per_cell"] = 1
                undersized["runtime_decision_sha256"] = runtime._canonical_sha256(
                    runtime._decision_payload(undersized)
                )
                with self.assertRaisesRegex(ValueError, "frozen thresholds"):
                    runtime.verify_runtime_tier_decision(
                        undersized, preregistration_path=preregistration
                    )

                preregistration.write_bytes(b"changed-preregistration")
                with self.assertRaisesRegex(ValueError, "frozen preregistration"):
                    runtime.verify_runtime_tier_decision(
                        decision, preregistration_path=preregistration
                    )

    def test_mismatched_preregistered_tier_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            preregistration = Path(temporary) / "preregistered_spec_v4.json"
            preregistration.write_bytes(b"immutable-preregistration")
            frozen = self._frozen()
            frozen["payload"]["runtime"]["tiers"]["FULL"][
                "synthetic_series_per_cell"
            ] = 1
            with mock.patch.object(
                runtime, "verify_preregistration", return_value=frozen
            ):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    runtime.seal_runtime_tier_decision(
                        projected_full_gpu_seconds=1.0,
                        smoke_projection_sha256="b" * 64,
                        preregistration_path=preregistration,
                    )


class FrozenTierContractTests(unittest.TestCase):
    def test_the_implementation_tier_projection_matches_the_authoritative_preregistration(self):
        from pathlib import Path as _Path

        from experiments.prob_head_structure_full_v1.preregistration import verify_preregistration
        from experiments.prob_head_structure_full_v1.runtime import _PREREGISTERED_TIER_PROJECTION

        spec = _Path(__file__).resolve().parents[2] / (
            "results/prob_head_structure_full_v1/preregistered_spec_v4.json"
        )
        payload = verify_preregistration(spec)["payload"]
        self.assertEqual(payload["runtime"]["tiers"], _PREREGISTERED_TIER_PROJECTION)

    def test_each_tier_condition_agrees_with_the_selection_thresholds(self):
        from experiments.prob_head_structure_full_v1.runtime import (
            _PREREGISTERED_TIER_PROJECTION,
            select_runtime_tier,
        )

        self.assertEqual(_PREREGISTERED_TIER_PROJECTION["FULL"]["condition"], "projected_GPU_hours<=12")
        self.assertEqual(
            _PREREGISTERED_TIER_PROJECTION["COMPACT"]["condition"], "12<projected_GPU_hours<=18"
        )
        self.assertEqual(
            _PREREGISTERED_TIER_PROJECTION["MINIMAL-COMPLETE"]["condition"], "projected_GPU_hours>18"
        )
        self.assertEqual(select_runtime_tier(12.0), "FULL")
        self.assertEqual(select_runtime_tier(12.0001), "COMPACT")
        self.assertEqual(select_runtime_tier(18.0), "COMPACT")
        self.assertEqual(select_runtime_tier(18.0001), "MINIMAL-COMPLETE")


if __name__ == "__main__":
    unittest.main()
