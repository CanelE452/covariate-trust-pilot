from __future__ import annotations

import unittest

import torch

from experiments.prob_head_structure_full_v1.losses import (
    add_student_loss_sums,
    student_loss_from_sums,
    student_loss_sums,
)


class StudentLossTests(unittest.TestCase):
    def _fixture(self):
        p0 = torch.tensor([[0.2, 0.7], [0.4, 0.6]], dtype=torch.float64)
        base = torch.linspace(0.1, 4.1, 21, dtype=torch.float64)
        quantiles = base.reshape(1, 1, -1).repeat(2, 2, 1)
        y = torch.tensor([[0.0, 2.0], [1.0, 0.0]], dtype=torch.float64)
        mask = torch.tensor([[True, True], [True, False]])
        scale = torch.tensor([1.0, 2.0], dtype=torch.float64)
        teacher_p0 = torch.tensor([[0.3, 0.6], [0.5, 0.8]], dtype=torch.float64)
        teacher_q = quantiles * 1.1
        return p0, quantiles, y, mask, scale, teacher_p0, teacher_q

    def test_masked_loss_and_microbatch_sums_are_exactly_additive(self):
        fixture = self._fixture()
        whole = student_loss_sums(
            p0_student=fixture[0],
            quantiles_student=fixture[1],
            target=fixture[2],
            target_mask=fixture[3],
            scale=fixture[4],
            teacher_p0=fixture[5],
            teacher_quantiles=fixture[6],
        )
        parts = []
        for row in (slice(0, 1), slice(1, 2)):
            parts.append(
                student_loss_sums(
                    p0_student=fixture[0][row],
                    quantiles_student=fixture[1][row],
                    target=fixture[2][row],
                    target_mask=fixture[3][row],
                    scale=fixture[4][row],
                    teacher_p0=fixture[5][row],
                    teacher_quantiles=fixture[6][row],
                )
            )
        combined = add_student_loss_sums(parts)
        self.assertEqual(whole.valid_cell_count, 3)
        for name in (
            "hard_zero_numerator",
            "hard_quantile_numerator",
            "soft_zero_numerator",
            "soft_quantile_numerator",
        ):
            torch.testing.assert_close(getattr(whole, name), getattr(combined, name))
        torch.testing.assert_close(
            student_loss_from_sums(whole, lambda_soft=0.5)["loss"],
            student_loss_from_sums(combined, lambda_soft=0.5)["loss"],
        )

        poisoned_target = fixture[2].clone()
        poisoned_target[1, 1] = 1_000_000.0
        poisoned = student_loss_sums(
            p0_student=fixture[0],
            quantiles_student=fixture[1],
            target=poisoned_target,
            target_mask=fixture[3],
            scale=fixture[4],
            teacher_p0=fixture[5],
            teacher_quantiles=fixture[6],
        )
        torch.testing.assert_close(
            student_loss_from_sums(whole, lambda_soft=0.75)["loss"],
            student_loss_from_sums(poisoned, lambda_soft=0.75)["loss"],
        )

    def test_one_hot_conditioned_teacher_loss_matches_single_teacher(self):
        p0, quantiles, y, mask, scale, teacher_p0, teacher_q = self._fixture()
        multi_p0 = torch.stack((teacher_p0, 1.0 - teacher_p0), dim=-1)
        multi_q = torch.stack((teacher_q, teacher_q + 3.0), dim=-2)
        weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        single = student_loss_sums(
            p0_student=p0,
            quantiles_student=quantiles,
            target=y,
            target_mask=mask,
            scale=scale,
            teacher_p0=teacher_p0,
            teacher_quantiles=teacher_q,
        )
        routed = student_loss_sums(
            p0_student=p0,
            quantiles_student=quantiles,
            target=y,
            target_mask=mask,
            scale=scale,
            teacher_p0=multi_p0,
            teacher_quantiles=multi_q,
            teacher_weights=weights,
        )
        torch.testing.assert_close(single.soft_zero_numerator, routed.soft_zero_numerator)
        torch.testing.assert_close(
            single.soft_quantile_numerator, routed.soft_quantile_numerator
        )

    def test_all_masked_or_invalid_simplex_is_rejected(self):
        p0, quantiles, y, mask, scale, teacher_p0, teacher_q = self._fixture()
        with self.assertRaisesRegex(ValueError, "all-masked"):
            student_loss_sums(
                p0_student=p0,
                quantiles_student=quantiles,
                target=y,
                target_mask=torch.zeros_like(mask),
                scale=scale,
            )
        with self.assertRaisesRegex(ValueError, "simplex"):
            student_loss_sums(
                p0_student=p0,
                quantiles_student=quantiles,
                target=y,
                target_mask=mask,
                scale=scale,
                teacher_p0=torch.stack((teacher_p0, teacher_p0), dim=-1),
                teacher_quantiles=torch.stack((teacher_q, teacher_q), dim=-2),
                teacher_weights=torch.tensor([[0.7, 0.4], [0.7, 0.4]]),
            )


if __name__ == "__main__":
    unittest.main()
