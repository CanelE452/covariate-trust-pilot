from __future__ import annotations

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from experiments.prob_head_structure_full_v1 import numerical_validation
from experiments.prob_head_structure_full_v1.numerical_validation import (
    compound_poisson_gamma_logpdf,
    tweedie_validation_gate,
    validate_tweedie_against_oracles,
)


class TweedieReferenceTests(unittest.TestCase):
    def test_independent_compound_poisson_gamma_oracle_has_exact_zero_mass(self):
        """Catches an oracle that misses the Tweedie atom at zero."""
        mu, phi, p = 2.0, 0.75, 1.5
        actual = compound_poisson_gamma_logpdf(torch.tensor(0.0), mu, phi, p)
        expected = -mu ** (2 - p) / (phi * (2 - p))
        self.assertAlmostEqual(float(actual), expected, places=11)

    def test_full_crossed_grid_validates_all_rows_without_global_stop(self):
        """Catches dropped grid points, global hard stops, and malformed branch-block reports."""
        report = validate_tweedie_against_oracles()
        self.assertEqual(report["grid_points"], 100)
        self.assertEqual(report["comparison_count"], 600)
        self.assertGreaterEqual(report["finite_fraction"], 0.999)
        self.assertLessEqual(report["zero_relative_error"], 1e-5)
        self.assertLessEqual(report["median_abs_log_difference"], 1e-4)
        self.assertLessEqual(report["p99_abs_log_difference"], 1e-3)
        self.assertEqual(report["cdf_monotonicity_violations"], 0)
        self.assertEqual(report["quantile_monotonicity_violations"], 0)
        self.assertEqual(report["branch"], "PASS")
        self.assertIn("confirmatory_eligible", report)
        self.assertIsInstance(report["rows"], list)
        self.assertEqual(len(report["rows"]), 600)
        self.assertIn("float64", report["precision"])
        self.assertIn("float32", report["precision"])
        self.assertIn("reference", report["oracles"])
        self.assertIn("compound_poisson_gamma", report["oracles"])
        self.assertRegex(report["report_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(report["distribution_checks"]), 200)
        self.assertEqual(report["reference_provenance"]["package"], "tweedie")
        self.assertEqual(report["reference_provenance"]["version"], "0.0.9")
        for check in report["distribution_checks"]:
            self.assertIsNone(check["zero_identity_error"])
            self.assertLessEqual(
                check["zero_identity_max_relative_error"], 1e-5
            )
            self.assertTrue(
                np.isclose(
                    check["cdf_values"][0],
                    check["p_zero_value"],
                    rtol=1e-5,
                    atol=0.0,
                )
            )
            self.assertTrue(
                np.isclose(
                    check["p_zero_value"],
                    float(np.exp(check["log_prob_zero_value"])),
                    rtol=1e-5,
                    atol=float(np.finfo(np.float32).tiny),
                )
            )

        truncated = dict(report, rows=report["rows"][:-1])
        duplicate = dict(
            report,
            rows=report["rows"][:-1] + [dict(report["rows"][0])],
        )
        self.assertTrue(tweedie_validation_gate(truncated)["hard_block"])
        self.assertTrue(tweedie_validation_gate(duplicate)["hard_block"])

        forged_checks = deepcopy(report)
        forged_checks["distribution_checks"][0]["cdf_values"] = [
            0.8,
            0.7,
            *forged_checks["distribution_checks"][0]["cdf_values"][2:],
        ]
        forged_checks["monotonicity"] = {
            precision: {
                "cdf": 0,
                "quantile": 0,
                "cdf_failures": 0,
                "quantile_failures": 0,
            }
            for precision in ("float64", "float32")
        }
        forged_checks["failed_parameter_regions"] = []
        forged_checks["branch"] = "PASS"
        forged_checks["confirmatory_eligible"] = True
        forged_checks["gate"] = {
            "branch": "PASS",
            "confirmatory_eligible": True,
            "hard_block": False,
        }
        forged_checks["report_sha256"] = numerical_validation._validation_report_hash(
            forged_checks
        )
        self.assertTrue(tweedie_validation_gate(forged_checks)["hard_block"])

        forged_zero_identity = deepcopy(report)
        forged_zero_identity["distribution_checks"][0]["p_zero_value"] = 0.5
        forged_zero_identity["report_sha256"] = (
            numerical_validation._validation_report_hash(forged_zero_identity)
        )
        self.assertTrue(
            tweedie_validation_gate(forged_zero_identity)["hard_block"]
        )

        sensitive = next(
            row
            for row in report["rows"]
            if row["mu"] == 0.1 and row["phi"] == 0.1 and row["p"] == 1.1
            and row["y"] == 0.005000000000000001
        )
        rounded_y = float(torch.tensor(sensitive["y"], dtype=torch.float32).item())
        self.assertEqual(sensitive["y_float32_execution"], rounded_y)
        reference = numerical_validation._load_reference_factory()(
            p=sensitive["p_float32_execution"],
            mu=sensitive["mu_float32_execution"],
            phi=sensitive["phi_float32_execution"],
        )
        self.assertAlmostEqual(
            sensitive["reference_log_prob_float32_parameters"],
            float(np.asarray(reference.logpdf(rounded_y)).item()),
            places=12,
        )

    def test_cdf_probability_audit_rejects_out_of_range_values(self):
        """Catches a monotone but invalid CDF passing numerical validation."""
        with self.assertRaisesRegex(FloatingPointError, r"\[0, 1\]"):
            numerical_validation._validate_cdf_values(
                torch.tensor([0.0, 0.7, 1.01], dtype=torch.float64)
            )

    def test_branch_local_gate_is_serializable_and_never_raises_for_a_failed_validation(self):
        """Catches validation failure being turned into a process-wide exception instead of a Tweedie-only block."""
        result = tweedie_validation_gate({"branch": "TWEEDIE_BRANCH_BLOCKED_HARD", "confirmatory_eligible": False})
        self.assertEqual(result, {"branch": "TWEEDIE_BRANCH_BLOCKED_HARD", "confirmatory_eligible": False, "hard_block": True})

    def test_missing_independent_reference_becomes_json_safe_tweedie_only_block(self):
        """Catches an optional reference import failure aborting NB/HSNB continuation."""
        with patch(
            "experiments.prob_head_structure_full_v1.numerical_validation._load_reference_factory",
            side_effect=ImportError("reference unavailable"),
        ):
            report = validate_tweedie_against_oracles()

        self.assertEqual(report["branch"], "TWEEDIE_BRANCH_BLOCKED_HARD")
        self.assertFalse(report["confirmatory_eligible"])
        self.assertTrue(report["failed_parameter_regions"])
        self.assertTrue(any(row["oracle"] == "reference" for row in report["failed_parameter_regions"]))
        json.dumps(report, allow_nan=False)

    def test_nonfinite_oracle_output_is_recorded_without_inf_and_cannot_forge_pass(self):
        """Catches NaN reference values becoming Infinity or a caller-forged PASS gate."""
        with patch(
            "experiments.prob_head_structure_full_v1.numerical_validation.compound_poisson_gamma_logpdf",
            return_value=torch.tensor(float("nan"), dtype=torch.float64),
        ):
            report = validate_tweedie_against_oracles()

        self.assertEqual(report["branch"], "TWEEDIE_BRANCH_BLOCKED_HARD")
        self.assertIsNone(report["precision"]["float64"]["compound_poisson_gamma"]["max_abs_log_difference"])
        json.dumps(report, allow_nan=False)
        forged = dict(report, branch="PASS", confirmatory_eligible=True)
        self.assertEqual(
            tweedie_validation_gate(forged),
            {
                "branch": "TWEEDIE_BRANCH_BLOCKED_HARD",
                "confirmatory_eligible": False,
                "hard_block": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
