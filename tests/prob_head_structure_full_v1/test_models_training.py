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


import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.prob_head_structure_full_v1 import training as training_module
from experiments.prob_head_structure_full_v1.training import (
    SCHEDULED_VALIDATION_EPOCHS,
    NumericalBranchBlocked,
    OutOfMemoryBranchBlocked,
    TrainingConfig,
    TrainingWindows,
    teacher_nll_objective,
    train_teacher,
    validation_scaled_crps,
)


def _windows(series: int = 8, lookback: int = 12, horizon: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    history = rng.poisson(1.2, size=(series, lookback)).astype(np.float64)
    target = rng.poisson(1.2, size=(series, horizon)).astype(np.float64)
    mask = np.ones((series, horizon), dtype=bool)
    scale = np.full(series, 1.5, dtype=np.float64)
    return TrainingWindows(history=history, target=target, target_mask=mask, scale=scale)


class TrainingContractTests(unittest.TestCase):
    def test_scheduled_validation_epochs_are_frozen_even_epochs_through_thirty(self):
        self.assertEqual(SCHEDULED_VALIDATION_EPOCHS, tuple(range(2, 31, 2)))

    def test_loss_divides_the_masked_nll_sum_by_the_valid_target_count(self):
        model = build_teacher("NB", lookback=12, horizon=4)
        windows = _windows()
        masked = TrainingWindows(
            history=windows.history,
            target=windows.target,
            target_mask=np.zeros_like(windows.target_mask),
            scale=windows.scale,
        )
        with self.assertRaises(ValueError):
            teacher_nll_objective(model, masked)

        partial = windows.target_mask.copy()
        partial[:, 2:] = False
        subset = TrainingWindows(
            history=windows.history,
            target=windows.target,
            target_mask=partial,
            scale=windows.scale,
        )
        loss = teacher_nll_objective(model, subset)
        self.assertTrue(torch.isfinite(loss))

    def test_all_masked_rows_are_removed_deterministically_before_batching(self):
        windows = _windows()
        mask = windows.target_mask.copy()
        mask[3, :] = False
        dropped = TrainingWindows(
            history=windows.history, target=windows.target, target_mask=mask, scale=windows.scale
        ).without_fully_masked_rows()
        self.assertEqual(dropped.history.shape[0], windows.history.shape[0] - 1)
        self.assertEqual(dropped.retained_rows.tolist(), [0, 1, 2, 4, 5, 6, 7])

    def test_validation_scaled_crps_uses_the_frozen_grid_and_train_only_scale(self):
        model = build_teacher("NB", lookback=12, horizon=4)
        windows = _windows()
        score = validation_scaled_crps(model, windows)
        self.assertTrue(np.isfinite(score) and score > 0)
        with self.assertRaises(ValueError):
            validation_scaled_crps(model, windows, quantile_grid=(0.1, 0.9))

    def test_training_retains_the_earliest_checkpoint_on_an_exact_tie(self):
        config = TrainingConfig(head_name="NB", lookback=12, horizon=4, seed=2026090511, maximum_epochs=4)
        result = train_teacher(config, _windows(), _windows(seed=1), score_sequence=[1.0, 1.0])
        self.assertEqual(result.best_epoch, 2)
        self.assertEqual(result.best_score, 1.0)

    def test_patience_stops_at_the_fifth_scheduled_check_without_strict_improvement(self):
        config = TrainingConfig(head_name="NB", lookback=12, horizon=4, seed=2026090511, maximum_epochs=30)
        scores = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.1]
        result = train_teacher(config, _windows(), _windows(seed=1), score_sequence=scores)
        self.assertEqual(result.best_epoch, 2)
        self.assertEqual(result.stopped_epoch, 12)
        self.assertEqual(result.stop_reason, "patience")
        self.assertEqual(result.checks_evaluated, 6)

    def test_best_checkpoint_state_is_restored_into_the_returned_model(self):
        config = TrainingConfig(head_name="NB", lookback=12, horizon=4, seed=2026090511, maximum_epochs=4)
        result = train_teacher(config, _windows(), _windows(seed=1), score_sequence=[0.5, 9.0])
        self.assertEqual(result.best_epoch, 2)
        for key, value in result.best_state.items():
            self.assertTrue(torch.equal(value, result.model.state_dict()[key]))

    def test_nan_loss_blocks_the_branch_without_changing_the_learning_rate(self):
        config = TrainingConfig(head_name="NB", lookback=12, horizon=4, seed=2026090511, maximum_epochs=2)
        with self.assertRaises(NumericalBranchBlocked) as context:
            train_teacher(config, _windows(), _windows(seed=1), _inject_nan_at_step=0)
        self.assertIn("NUMERICAL_BRANCH_BLOCKED", str(context.exception))
        self.assertEqual(config.learning_rate, 1e-3)

    def test_single_oom_retry_halves_microbatch_and_doubles_accumulation(self):
        config = TrainingConfig(head_name="NB", lookback=12, horizon=4, seed=2026090511, maximum_epochs=2)
        result = train_teacher(
            config, _windows(), _windows(seed=1), score_sequence=[1.0], _inject_oom_times=1
        )
        self.assertEqual(result.microbatch_size, config.microbatch_size // 2)
        self.assertEqual(result.gradient_accumulation, config.gradient_accumulation * 2)
        self.assertEqual(
            result.microbatch_size * result.gradient_accumulation, config.effective_batch_size
        )
        self.assertEqual(result.oom_retries, 1)

        with self.assertRaises(OutOfMemoryBranchBlocked) as context:
            train_teacher(
                config, _windows(), _windows(seed=1), score_sequence=[1.0], _inject_oom_times=2
            )
        self.assertIn("OOM_MODEL_BRANCH_BLOCKED", str(context.exception))

    def test_training_attempt_is_append_only_and_publishes_a_completion_marker(self):
        config = TrainingConfig(head_name="NB", lookback=12, horizon=4, seed=2026090511, maximum_epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            first = training_module.run_training_attempt(
                runs, "teacher_nb", config, _windows(), _windows(seed=1), score_sequence=[1.0]
            )
            self.assertFalse(first["resumed"])
            marker = Path(first["attempt"]) / "completion.json"
            self.assertTrue(marker.exists())
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "COMPLETE")

            second = training_module.run_training_attempt(
                runs, "teacher_nb", config, _windows(), _windows(seed=1), score_sequence=[1.0]
            )
            self.assertTrue(second["resumed"])
            self.assertEqual(second["attempt"], first["attempt"])
