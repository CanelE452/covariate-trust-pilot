from __future__ import annotations

import unittest

import torch

from experiments.prob_head_structure_full_v1.models import build_teacher, count_parameters
from experiments.prob_head_structure_full_v1.student import (
    QUANTILE_GRID,
    build_student,
    postprocess_student_quantiles,
    quantile_integral_mean,
)


class StudentArchitectureTests(unittest.TestCase):
    def test_frozen_student_parameter_count_is_below_cap(self):
        student = build_student(lookback=96, horizon=28)
        smallest_teacher = min(
            count_parameters(build_teacher(head, lookback=96, horizon=28))
            for head in ("NB", "HSNB", "TWEEDIE_FULL")
        )
        self.assertEqual(count_parameters(student), 5_838)
        self.assertLessEqual(count_parameters(student), 1.5 * smallest_teacher)

    def test_forward_outputs_strict_p0_and_monotone_raw_scale_quantiles(self):
        torch.manual_seed(2026090511)
        student = build_student(lookback=96, horizon=28)
        history = torch.arange(3 * 96, dtype=torch.float32).reshape(3, 96) / 100.0
        scale = torch.tensor([1.0, 2.0, 4.0])
        output = student(history, scale)

        self.assertEqual(output["p0"].shape, (3, 28))
        self.assertEqual(output["quantiles"].shape, (3, 28, 21))
        self.assertTrue(bool(((output["p0"] > 0) & (output["p0"] < 1)).all()))
        self.assertTrue(bool((output["quantiles"] > 0).all()))
        self.assertTrue(
            bool((output["quantiles"][..., 1:] >= output["quantiles"][..., :-1]).all())
        )
        torch.testing.assert_close(
            output["quantiles"] / scale[:, None, None],
            output["normalized_quantiles"],
        )

    def test_postprocess_enforces_zero_mass_biconditional_at_grid_points(self):
        raw = torch.arange(1, 22, dtype=torch.float64).reshape(1, 1, 21)
        p0 = torch.tensor([[0.10]], dtype=torch.float64)
        processed = postprocess_student_quantiles(p0, raw)
        zero = processed == 0
        expected_zero = QUANTILE_GRID.to(torch.float64).reshape(1, 1, -1) <= p0[..., None]
        self.assertTrue(torch.equal(zero, expected_zero))
        self.assertEqual(float(processed[..., 2]), 0.0)  # q == p0
        self.assertGreater(float(processed[..., 3]), 0.0)

    def test_quantile_integral_mean_uses_frozen_endpoint_hold(self):
        constant = torch.full((2, 3, 21), 7.0, dtype=torch.float64)
        torch.testing.assert_close(
            quantile_integral_mean(constant), torch.full((2, 3), 7.0, dtype=torch.float64)
        )

    def test_invalid_scale_and_crossing_inputs_are_rejected(self):
        student = build_student(lookback=96, horizon=28)
        history = torch.zeros((2, 96), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "scale"):
            student(history, torch.tensor([1.0, 0.0]))
        crossing = torch.ones((1, 1, 21), dtype=torch.float32)
        crossing[..., 10] = 0.5
        with self.assertRaisesRegex(ValueError, "monotone"):
            postprocess_student_quantiles(torch.tensor([[0.1]]), crossing)


if __name__ == "__main__":
    unittest.main()
