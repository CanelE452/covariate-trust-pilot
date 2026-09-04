"""Synthetic contracts for the frozen online-policy evaluation pipeline."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.ph_online_memory_gono_v1 import evaluation as evaluation_module
from experiments.ph_online_memory_gono_v1.evaluation import (
    ALPHA_GRID,
    baseline_policy_steps,
    build_series_origin_cases,
    convex_forecast,
    evaluate_b4_cases,
    evaluate_c0_cases,
    evaluate_c1_cases,
    evaluate_m1_cases,
    series_origin_expert_losses,
    shuffle_paired_memory_values,
    tune_b3_source,
    tune_b4_source,
    tune_m1_source,
)


KEYS = ["dataset_id", "series_id", "origin"]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = (10, 12, 14, 16)
    series_ids = ("a", "b", "c", "d")
    target_by_case = {
        ("a", 10): 0.0,
        ("b", 10): 2.0,
        ("c", 10): 1.0,
        ("d", 10): 2.0,
        ("a", 12): 2.0,
        ("b", 12): 0.0,
        ("c", 12): 2.0,
        ("d", 12): 1.0,
        ("a", 14): 1.0,
        ("b", 14): 2.0,
        ("c", 14): 0.0,
        ("d", 14): 2.0,
        ("a", 16): 2.0,
        ("b", 16): 1.0,
        ("c", 16): 2.0,
        ("d", 16): 0.0,
    }
    step_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for series_index, series_id in enumerate(series_ids):
        for origin in origins:
            target = target_by_case[(series_id, origin)]
            for step in range(2):
                step_rows.append(
                    {
                        "dataset_id": "toy",
                        "series_id": series_id,
                        "origin": origin,
                        "step": step,
                        "y_observed": target,
                        "point_mean_prediction": 0.0,
                        "hurdle_mean_prediction": 2.0,
                        "target_mask": True,
                        "policy_scale_squared": 1.0,
                    }
                )
            history_rows.append(
                {
                    "dataset_id": "toy",
                    "series_id": series_id,
                    "origin": origin,
                    "history": np.array(
                        [0.0, series_index + 1.0, 0.0, float(origin - 8)]
                    ),
                    "canonical_train_scale": 1.0,
                }
            )
    return pd.DataFrame(step_rows), pd.DataFrame(history_rows)


class BaselinePolicyTests(unittest.TestCase):
    def test_b0_b1_b2_and_convex_forecast_are_exact(self):
        steps, _ = _frames()
        result = baseline_policy_steps(steps, b3_alpha=0.25)

        np.testing.assert_array_equal(
            result["b0_prediction"], result["point_mean_prediction"]
        )
        np.testing.assert_array_equal(
            result["b1_prediction"], result["hurdle_mean_prediction"]
        )
        np.testing.assert_allclose(result["b2_prediction"], 1.0)
        np.testing.assert_allclose(result["b3_prediction"], 0.5)
        np.testing.assert_allclose(
            convex_forecast(np.array([0.0, 4.0]), np.array([2.0, 8.0]), 0.25),
            np.array([0.5, 5.0]),
        )

    def test_expert_losses_are_series_origin_and_duplicate_steps_fail(self):
        steps, _ = _frames()
        losses = series_origin_expert_losses(steps)
        self.assertEqual(len(losses), 16)
        self.assertEqual(losses[KEYS].drop_duplicates().shape[0], 16)

        varied = steps.copy()
        varied["point_mean_prediction"] = np.arange(len(varied)) / 10.0
        varied["hurdle_mean_prediction"] = np.arange(len(varied))[::-1] / 10.0
        ordered_losses = series_origin_expert_losses(varied)
        shuffled_losses = series_origin_expert_losses(
            varied.sample(frac=1.0, random_state=19)
        )
        pd.testing.assert_frame_equal(ordered_losses, shuffled_losses)

        duplicated = pd.concat([steps, steps.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            series_origin_expert_losses(duplicated)

    def test_b3_tuner_is_source_only_and_uses_frozen_grid(self):
        steps, _ = _frames()
        result = tune_b3_source(steps, evaluation_origins=(12, 14, 16))
        self.assertIn(result["alpha"], ALPHA_GRID)
        self.assertEqual(len(result["candidates"]), 21)
        self.assertEqual(result["evaluation_origins"], (12, 14, 16))
        self.assertNotIn("target", inspect.signature(tune_b3_source).parameters)


class CausalMemoryTests(unittest.TestCase):
    def setUp(self):
        steps, histories = _frames()
        self.cases = build_series_origin_cases(steps, histories, horizon=2, lookback=4)

    def test_b4_first_eval_uses_only_warmup_then_only_resolved_origins(self):
        result = evaluate_b4_cases(
            self.cases,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            eta=2.0,
            half_life=1,
        )
        first = result.loc[result["origin"] == 12, "resolved_origins"]
        second = result.loc[result["origin"] == 14, "resolved_origins"]
        third = result.loc[result["origin"] == 16, "resolved_origins"]
        self.assertTrue(all(value == (10,) for value in first))
        self.assertTrue(all(value == (10, 12) for value in second))
        self.assertTrue(all(value == (10, 12, 14) for value in third))

    def test_case_target_masks_must_remain_boolean(self):
        malformed = self.cases.copy(deep=True)
        malformed.at[malformed.index[0], "target_mask"] = np.array([1, 1])
        with self.assertRaisesRegex(ValueError, "target_mask.*Boolean"):
            evaluate_b4_cases(
                malformed,
                warmup_origin=10,
                evaluation_origins=(12, 14, 16),
                horizon=2,
                eta=2.0,
                half_life=1,
            )

    def test_current_and_future_targets_cannot_change_current_b4_weight(self):
        baseline = evaluate_b4_cases(
            self.cases,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            eta=8.0,
            half_life=3,
        )
        altered = self.cases.copy(deep=True)
        for row_index in altered.index[altered["origin"] >= 14]:
            altered.at[row_index, "target"] = np.array([999.0, 999.0])
            altered.at[row_index, "point_normalized_loss"] = 998001.0
            altered.at[row_index, "hurdle_normalized_loss"] = 994009.0
        changed = evaluate_b4_cases(
            altered,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            eta=8.0,
            half_life=3,
        )
        before = baseline.loc[baseline["origin"] <= 14, "b4_hurdle_weight"]
        after = changed.loc[changed["origin"] <= 14, "b4_hurdle_weight"]
        np.testing.assert_array_equal(before.to_numpy(), after.to_numpy())

    def test_retrieval_uses_resolved_target_memory_and_excludes_same_series(self):
        result = evaluate_m1_cases(
            self.cases,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            eta=2.0,
            half_life=1,
            k=2,
            lambda_max=0.5,
        )
        for row in result.itertuples(index=False):
            self.assertNotIn(row.series_id, row.neighbor_series_ids)
            self.assertTrue(all(origin + 2 <= row.origin for origin in row.neighbor_origins))
            self.assertGreaterEqual(row.retrieval_confidence, 0.0)
            self.assertLessEqual(row.retrieval_confidence, 1.0)
            self.assertGreaterEqual(row.m1_hurdle_weight, 0.0)
            self.assertLessEqual(row.m1_hurdle_weight, 1.0)
            self.assertEqual(len(row.constant_continuous_features), 8)
            expected = convex_forecast(
                row.point_forecast, row.hurdle_forecast, row.m1_hurdle_weight
            )
            np.testing.assert_allclose(row.m1_forecast, expected)

    def test_neighbor_plan_is_immutable_and_preserves_naive_exact_results(self):
        plan = evaluation_module.build_m1_neighbor_plan(
            self.cases,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            max_k=2,
        )
        with self.assertRaises(FrozenInstanceError):
            plan.max_k = 3

        result = evaluate_m1_cases(
            self.cases,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            eta=2.0,
            half_life=1,
            k=2,
            lambda_max=0.5,
            neighbor_plan=plan,
        )
        self.assertEqual(
            result["neighbor_case_keys"].tolist(),
            [
                (("toy", "d", 10), ("toy", "c", 10)),
                (("toy", "b", 12), ("toy", "d", 10)),
                (("toy", "b", 14), ("toy", "c", 14)),
                (("toy", "d", 10), ("toy", "c", 10)),
                (("toy", "c", 12), ("toy", "d", 10)),
                (("toy", "a", 14), ("toy", "c", 14)),
                (("toy", "d", 10), ("toy", "b", 10)),
                (("toy", "d", 12), ("toy", "b", 12)),
                (("toy", "b", 14), ("toy", "d", 14)),
                (("toy", "c", 10), ("toy", "b", 10)),
                (("toy", "c", 12), ("toy", "b", 12)),
                (("toy", "c", 14), ("toy", "b", 14)),
            ],
        )
        np.testing.assert_allclose(
            result["m1_hurdle_weight"],
            [
                0.2589931043785285,
                0.8807970779778823,
                0.7310585786300049,
                0.9820137900379085,
                0.5594337848452484,
                0.7189271481914654,
                0.749832324310186,
                0.7410068956214715,
                0.5594337848452484,
                0.9820137900379085,
                0.8807970779778823,
                0.9933071490757153,
            ],
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result["m1_normalized_loss"],
            [
                2.196364877434281,
                0.5800256583859735,
                0.289317952514053,
                3.8574043352984693,
                0.7763943597431314,
                0.19171638486099143,
                0.25033546384017585,
                2.196364877434281,
                0.7763943597431314,
                0.9293491751468356,
                0.05683734647444428,
                3.946636369619701,
            ],
            rtol=1e-13,
            atol=1e-14,
        )

    def test_boundary_distance_ties_use_canonical_case_order(self):
        tied = self.cases.copy(deep=True)
        for row_index in tied.index:
            tied.at[row_index, "history"] = np.zeros(4, dtype=np.float64)
        plan = evaluation_module.build_m1_neighbor_plan(
            tied,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            max_k=2,
        )
        first_a = next(
            entry
            for entry in plan.entries
            if entry.query_key == ("toy", "a", 12)
        )
        self.assertEqual(
            first_a.neighbor_case_keys,
            (("toy", "b", 10), ("toy", "c", 10)),
        )

    def test_future_cases_do_not_enter_retrieval_features_or_scaler(self):
        baseline = evaluate_m1_cases(
            self.cases,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            eta=2.0,
            half_life=1,
            k=2,
            lambda_max=0.5,
        )
        altered = self.cases.copy(deep=True)
        for row_index in altered.index[altered["origin"] > 12]:
            altered.at[row_index, "history"] = np.array([1e9, 0.0, 1e9, 0.0])
        changed = evaluate_m1_cases(
            altered,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            eta=2.0,
            half_life=1,
            k=2,
            lambda_max=0.5,
        )
        cols = ["m1_hurdle_weight", "neighbor_series_ids", "neighbor_origins"]
        pd.testing.assert_frame_equal(
            baseline.loc[baseline["origin"] == 12, cols].reset_index(drop=True),
            changed.loc[changed["origin"] == 12, cols].reset_index(drop=True),
        )


class ControlTests(unittest.TestCase):
    def setUp(self):
        steps, histories = _frames()
        self.cases = build_series_origin_cases(steps, histories, horizon=2, lookback=4)

    def test_c0_shuffles_loss_pairs_within_origin_without_breaking_pairs(self):
        values = self.cases[
            KEYS + ["point_normalized_loss", "hurdle_normalized_loss"]
        ]
        shuffled = shuffle_paired_memory_values(values, seed=20260904)
        changed = False
        for origin in sorted(values["origin"].unique()):
            original_group = values[values["origin"] == origin]
            shuffled_group = shuffled[shuffled["origin"] == origin]
            original_pairs = sorted(
                zip(
                    original_group["point_normalized_loss"],
                    original_group["hurdle_normalized_loss"],
                )
            )
            shuffled_pairs = sorted(
                zip(
                    shuffled_group["point_normalized_loss"],
                    shuffled_group["hurdle_normalized_loss"],
                )
            )
            self.assertEqual(original_pairs, shuffled_pairs)
            changed |= not original_group.set_index("series_id")[[
                "point_normalized_loss", "hurdle_normalized_loss"
            ]].equals(shuffled_group.set_index("series_id")[[
                "point_normalized_loss", "hurdle_normalized_loss"
            ]])
        self.assertTrue(changed)

    def test_c1_random_neighbors_are_deterministic_and_exclude_same_series(self):
        kwargs = dict(
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            eta=2.0,
            half_life=1,
            k=2,
            lambda_max=0.5,
            neighbor_mode="random",
            random_seed=20260904,
        )
        kwargs.pop("neighbor_mode")
        kwargs.pop("random_seed")
        first = evaluate_c1_cases(self.cases, **kwargs)
        second = evaluate_c1_cases(
            self.cases.sample(frac=1.0, random_state=7), **kwargs
        )
        pd.testing.assert_series_equal(
            first["neighbor_case_keys"], second["neighbor_case_keys"]
        )
        for row in first.itertuples(index=False):
            self.assertNotIn(row.series_id, row.neighbor_series_ids)

    def test_c0_keeps_real_keys_and_b4_state_while_shuffling_values(self):
        kwargs = dict(
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            eta=2.0,
            half_life=1,
            k=2,
            lambda_max=0.5,
        )
        real = evaluate_m1_cases(self.cases, **kwargs)
        control = evaluate_c0_cases(self.cases, **kwargs)
        pd.testing.assert_series_equal(
            real["neighbor_case_keys"], control["neighbor_case_keys"]
        )
        np.testing.assert_array_equal(
            real["b4_hurdle_weight"], control["b4_hurdle_weight"]
        )

    def test_c0_can_reuse_an_existing_neighbor_plan_without_refitting(self):
        plan = evaluation_module.build_m1_neighbor_plan(
            self.cases,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            max_k=2,
        )
        with (
            patch.object(
                evaluation_module,
                "extract_retrieval_features",
                wraps=evaluation_module.extract_retrieval_features,
            ) as extract,
            patch.object(
                evaluation_module,
                "fit_robust_scaler",
                wraps=evaluation_module.fit_robust_scaler,
            ) as fit_scaler,
            patch.object(
                evaluation_module,
                "_fit_neighbor_tree",
                wraps=evaluation_module._fit_neighbor_tree,
            ) as fit_tree,
        ):
            control = evaluate_c0_cases(
                self.cases,
                warmup_origin=10,
                evaluation_origins=(12, 14, 16),
                horizon=2,
                lookback=4,
                eta=2.0,
                half_life=1,
                k=2,
                lambda_max=0.5,
                neighbor_plan=plan,
            )
        self.assertEqual(len(control), 12)
        self.assertEqual(extract.call_count, 0)
        self.assertEqual(fit_scaler.call_count, 0)
        self.assertEqual(fit_tree.call_count, 0)

    def test_c1_can_reuse_an_existing_neighbor_plan_without_refitting(self):
        plan = evaluation_module.build_m1_neighbor_plan(
            self.cases,
            warmup_origin=10,
            evaluation_origins=(12, 14, 16),
            horizon=2,
            lookback=4,
            max_k=2,
        )
        with patch.object(
            evaluation_module,
            "build_m1_neighbor_plan",
            side_effect=AssertionError("C1 must reuse the supplied plan"),
        ):
            first = evaluate_c1_cases(
                self.cases,
                warmup_origin=10,
                evaluation_origins=(12, 14, 16),
                horizon=2,
                lookback=4,
                eta=2.0,
                half_life=1,
                k=2,
                lambda_max=0.5,
                random_seed=20260904,
                neighbor_plan=plan,
            )
        self.assertEqual(len(first), 12)

    def test_source_tuners_never_accept_target_inputs(self):
        self.assertNotIn("target", inspect.signature(tune_b4_source).parameters)
        self.assertNotIn("target", inspect.signature(tune_m1_source).parameters)

        b4 = tune_b4_source(
            self.cases,
            warmup_origin=10,
            evaluation_origins=iter((12, 14, 16)),
            horizon=2,
            eta_grid=(0.5, 2.0),
            half_lives=(1, 3),
        )
        self.assertIn(b4["eta"], (0.5, 2.0))
        self.assertIn(b4["half_life"], (1, 3))
        self.assertEqual(len(b4["candidates"]), 4)

        m1 = tune_m1_source(
            self.cases,
            warmup_origin=10,
            evaluation_origins=iter((12, 14, 16)),
            horizon=2,
            lookback=4,
            eta=b4["eta"],
            half_life=b4["half_life"],
            k_grid=(1, 2),
            lambda_max_grid=(0.25, 0.5),
        )
        self.assertIn(m1["k"], (1, 2))
        self.assertIn(m1["lambda_max"], (0.25, 0.5))
        self.assertEqual(len(m1["candidates"]), 4)

    def test_m1_tuning_extracts_once_and_fits_once_per_origin(self):
        with (
            patch.object(
                evaluation_module,
                "extract_retrieval_features",
                wraps=evaluation_module.extract_retrieval_features,
            ) as extract,
            patch.object(
                evaluation_module,
                "fit_robust_scaler",
                wraps=evaluation_module.fit_robust_scaler,
            ) as fit_scaler,
            patch.object(
                evaluation_module,
                "_fit_neighbor_tree",
                wraps=evaluation_module._fit_neighbor_tree,
            ) as fit_tree,
        ):
            result = tune_m1_source(
                self.cases,
                warmup_origin=10,
                evaluation_origins=(12, 14, 16),
                horizon=2,
                lookback=4,
                eta=2.0,
                half_life=1,
                k_grid=(1, 2),
                lambda_max_grid=(0.25, 0.5),
            )

        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual(extract.call_count, len(self.cases))
        self.assertEqual(fit_scaler.call_count, 3)
        self.assertEqual(fit_tree.call_count, 3)


if __name__ == "__main__":
    unittest.main()
