from __future__ import annotations

import unittest

import torch

from experiments.prob_head_structure_full_v1.models import (
    HEAD_NAMES,
    build_teacher,
    count_parameters,
)


class ProbabilisticTeacherModelTests(unittest.TestCase):
    def test_parameter_matched_teachers_share_the_verified_trunk(self):
        models = {name: build_teacher(name, lookback=96, horizon=28) for name in HEAD_NAMES}
        counts = {name: count_parameters(model) for name, model in models.items()}

        self.assertLessEqual((max(counts.values()) - min(counts.values())) / min(counts.values()), 0.05)
        self.assertEqual(counts, {"NB": 7103, "HSNB": 7098, "TWEEDIE_FULL": 7115})
        for model in models.values():
            self.assertEqual(model.moving_average_kernel, 25)
            self.assertEqual(model.trend.in_features, 96)
            self.assertEqual(model.trend.out_features, 28)
            self.assertEqual(model.season.in_features, 96)
            self.assertEqual(model.season.out_features, 28)

    def test_each_teacher_emits_a_valid_distribution_on_raw_count_scale(self):
        history = torch.tensor(
            [[0.0] * 94 + [2.0, 4.0], [1.0] * 96], dtype=torch.float32
        )
        scale = torch.tensor([2.0, 4.0], dtype=torch.float32)

        for name in HEAD_NAMES:
            torch.manual_seed(2026090511)
            model = build_teacher(name, lookback=96, horizon=28)
            output = model(history, scale)
            distribution = output["distribution"]

            self.assertEqual(tuple(distribution.mean().shape), (2, 28))
            self.assertTrue(torch.isfinite(distribution.mean()).all())
            self.assertTrue((distribution.mean() > 0).all())
            self.assertTrue(torch.isfinite(distribution.p_zero()).all())
            self.assertTrue(((distribution.p_zero() >= 0) & (distribution.p_zero() <= 1)).all())
            self.assertEqual(tuple(output["normalized_mu"].shape), (2, 28))
            self.assertTrue(torch.allclose(output["mu"], output["normalized_mu"] * scale[:, None]))

    def test_tweedie_global_parameters_have_frozen_initial_values(self):
        model = build_teacher("TWEEDIE_FULL", lookback=96, horizon=28)
        output = model(torch.zeros((3, 96)), torch.ones(3))

        self.assertEqual(tuple(output["phi"].shape), (3, 28))
        self.assertEqual(tuple(output["p"].shape), (3, 28))
        self.assertTrue(torch.allclose(output["phi"], torch.ones_like(output["phi"]), atol=2e-6))
        self.assertTrue(torch.allclose(output["p"], torch.full_like(output["p"], 1.5), atol=1e-7))

    def test_forward_rejects_nonpositive_or_misaligned_train_scale(self):
        model = build_teacher("NB", lookback=96, horizon=28)
        history = torch.zeros((2, 96))

        for invalid in (torch.tensor([1.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, float("nan")])):
            with self.assertRaises(ValueError):
                model(history, invalid)


if __name__ == "__main__":
    unittest.main()
