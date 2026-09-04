"""Failure-first contracts for the pure online-memory pilot helpers."""

import inspect
import unittest

# Load NumPy/pandas before importing the experiment package.  A package import may
# eventually reach the canonical trainer, and this is the audited safe import order.
import numpy as np
import pandas as pd

from experiments.ph_online_memory_gono_v1 import (
    bootstrap,
    data,
    gates,
    metrics,
    policies,
    retrieval,
)


class PureContractTests(unittest.TestCase):
    def test_t1_retrieval_key_uses_strictly_pre_origin_history(self):
        """Catches an inclusive slice that leaks the query target into its key."""
        history = np.array([10.0, 20.0, 30.0, 999.0, -999.0])

        key = retrieval.make_retrieval_key(
            history, query_origin=3, lookback=2
        )
        changed_future_key = retrieval.make_retrieval_key(
            np.array([10.0, 20.0, 30.0, -7.0, 8.0]),
            query_origin=3,
            lookback=2,
        )

        np.testing.assert_array_equal(key, np.array([20.0, 30.0]))
        np.testing.assert_array_equal(changed_future_key, np.array([20.0, 30.0]))

    def test_t2_memory_contains_only_cases_resolved_by_query_origin(self):
        """Catches admitting an unresolved case or dropping the exact-end boundary."""
        cases = pd.DataFrame(
            {
                "case_id": ["ended_before", "ends_now", "ends_later"],
                "series_id": ["a", "b", "c"],
                "origin": [4, 5, 6],
            }
        )

        resolved = retrieval.resolved_memory_cases(
            cases, query_origin=8, horizon=3
        )

        self.assertEqual(
            resolved["case_id"].tolist(), ["ended_before", "ends_now"]
        )
        self.assertTrue(
            bool(((resolved["origin"] + 3) <= 8).all())
        )

    def test_t3_tuners_have_source_only_signatures(self):
        """Catches adding a held-out/target argument or permissive **kwargs to tuning."""
        expected_signatures = (
            (
                policies.select_b3_alpha,
                ("source_frame", "alpha_grid"),
                ("alpha_grid",),
                {"alpha_grid": (0.0, 0.5, 1.0)},
            ),
            (
                policies.select_b4_hyperparameters,
                ("source_frame", "eta_grid", "half_lives"),
                ("eta_grid", "half_lives"),
                {"eta_grid": (0.5, 2.0), "half_lives": (1, 2)},
            ),
            (
                retrieval.select_m1_hyperparameters,
                ("source_frame", "k_grid", "lambda_max_grid"),
                ("k_grid", "lambda_max_grid"),
                {"k_grid": (1, 3), "lambda_max_grid": (0.0, 1.0)},
            ),
        )
        source = pd.DataFrame({"loss": [1.0]})
        target = pd.DataFrame({"loss": [0.0]})

        for (
            tuner,
            expected_names,
            keyword_only_names,
            valid_kwargs,
        ) in expected_signatures:
            with self.subTest(tuner=tuner.__name__):
                signature = inspect.signature(tuner)
                self.assertEqual(tuple(signature.parameters), expected_names)
                self.assertEqual(
                    tuple(
                        name
                        for name, parameter in signature.parameters.items()
                        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
                    ),
                    keyword_only_names,
                )
                self.assertFalse(
                    any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in signature.parameters.values()
                    )
                )
                with self.assertRaises(TypeError):
                    signature.bind(
                        source, **valid_kwargs, target_results=target
                    )

    def test_b3_exact_mean_tie_uses_frozen_grid_order(self):
        """Catches importing B4's worst-origin tie-break into static B3."""
        source = pd.DataFrame(
            {
                "alpha": [0.0, 0.5, 1.0],
                "mean_loss": [1.0, 1.0, 2.0],
                "worst_origin_loss": [99.0, 1.0, 2.0],
            }
        )
        selected = policies.select_b3_alpha(
            source, alpha_grid=(0.0, 0.5, 1.0)
        )
        self.assertEqual(selected, 0.0)

    def test_t4_nearest_neighbors_exclude_every_same_series_case(self):
        """Catches excluding only the query row instead of the full source series."""
        case_keys = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.1, 0.0],
                [2.0, 0.0],
            ]
        )
        case_series_ids = np.array(["query", "a", "query", "b"])

        indices = retrieval.nearest_neighbor_indices(
            np.array([0.0, 0.0]),
            case_keys,
            case_series_ids,
            query_series_id="query",
            k=2,
        )

        np.testing.assert_array_equal(indices, np.array([1, 3]))

    def test_t5_point_hurdle_predictions_pair_completely_one_to_one(self):
        """Catches an inner join that hides missing rows or a many-to-many key join."""
        point = pd.DataFrame(
            {
                "dataset_id": ["m5", "m5"],
                "series_id": ["s1", "s1"],
                "origin": [100, 100],
                "step": [0, 1],
                "prediction": [10.0, 20.0],
            }
        )
        hurdle = pd.DataFrame(
            {
                "dataset_id": ["m5", "m5"],
                "series_id": ["s1", "s1"],
                "origin": [100, 100],
                "step": [1, 0],
                "prediction": [18.0, 9.0],
            }
        )

        paired = metrics.pair_predictions(point, hurdle)
        expected = pd.DataFrame(
            {
                "dataset_id": ["m5", "m5"],
                "series_id": ["s1", "s1"],
                "origin": [100, 100],
                "step": [0, 1],
                "point_prediction": [10.0, 20.0],
                "hurdle_prediction": [9.0, 18.0],
            }
        )
        pd.testing.assert_frame_equal(
            paired.reset_index(drop=True), expected
        )

        with self.assertRaises(ValueError):
            metrics.pair_predictions(point, hurdle.iloc[:1].copy())
        with self.assertRaises(ValueError):
            metrics.pair_predictions(
                pd.concat([point, point.iloc[[0]]], ignore_index=True), hurdle
            )

    def test_t6_final_six_evaluation_origins_end_at_actual_series_length(self):
        """Catches a phantom seventh origin or an off-by-one tail cutoff."""
        cases = (
            (1941, [1773, 1801, 1829, 1857, 1885, 1913]),
            (1688, [1520, 1548, 1576, 1604, 1632, 1660]),
        )
        for series_length, expected in cases:
            with self.subTest(series_length=series_length):
                origins = data.evaluation_origins(
                    series_length=series_length, horizon=28, count=6
                )
                np.testing.assert_array_equal(origins, np.array(expected))
                self.assertEqual(int(origins[-1]) + 28, series_length)

    def test_t7_policy_scale_squared_uses_model_train_prefix_and_epsilon(self):
        """Catches future-value leakage or confusing RMS model scale with MSE scale."""
        values = np.array([1.0, 3.0, 1_000.0, -1_000.0])
        changed_future = np.array([1.0, 3.0, 7.0, 8.0])

        scale = metrics.policy_scale_squared(values, model_train_end=2)
        changed_future_scale = metrics.policy_scale_squared(
            changed_future, model_train_end=2
        )

        self.assertAlmostEqual(scale, 5.00000001, places=12)
        self.assertAlmostEqual(changed_future_scale, 5.00000001, places=12)

    def test_t8_eligibility_counts_only_positive_model_train_values(self):
        """Catches counting future positives, zeros, or negative nonzero values."""
        values = np.array(
            [
                [1.0, -5.0, 2.0, 0.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            ]
        )

        eligible = data.eligible_series_mask(
            values, model_train_end=4, min_positive=3
        )

        np.testing.assert_array_equal(eligible, np.array([False, True]))

    def test_t9_bootstrap_resamples_complete_six_origin_series_clusters(self):
        """Catches row-wise bootstrap sampling that breaks within-series dependence."""
        series_ids = np.repeat(np.array(["a", "b", "c"]), 6)

        indices = bootstrap.series_cluster_resample_indices(
            series_ids, sampled_series_ids=np.array(["c", "a", "c"])
        )

        np.testing.assert_array_equal(
            indices,
            np.array(
                [
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                ]
            ),
        )

    def test_t10_exponential_hurdle_weight_is_finite_for_extreme_losses(self):
        """Catches the naive exp(-eta * loss) 0/0 underflow implementation."""
        weights = policies.exponential_hurdle_weight(
            point_loss=np.array([1_000.0, 2_000.0, 1_000_000.0]),
            hurdle_loss=np.array([2_000.0, 1_000.0, 1_000_000.0]),
            eta=1_000_000.0,
        )

        self.assertTrue(bool(np.isfinite(weights).all()))
        np.testing.assert_allclose(weights, np.array([0.0, 1.0, 0.5]))

    def test_t11_constant_spearman_predictor_is_degenerate(self):
        """Catches ranking machine-noise in a constant predictor as real signal."""
        outcome = np.array([4.0, 1.0, 3.0, 2.0])
        predictors = (
            np.ones(4),
            np.array(
                [
                    1.0,
                    np.nextafter(1.0, 2.0),
                    1.0,
                    np.nextafter(1.0, 0.0),
                ]
            ),
        )

        for predictor in predictors:
            with self.subTest(predictor=predictor):
                diagnostic = gates.spearman_diagnostic(predictor, outcome)
                self.assertEqual(diagnostic["status"], "DEGENERATE")
                self.assertIsNone(diagnostic["rho"])

    def test_t12_oracle_capture_uses_the_policy_familys_ladder(self):
        """Catches scoring a convex policy against the hard-oracle denominator."""
        captures = gates.family_matched_oracle_capture(
            policy_losses={"hard_policy": 8.0, "convex_policy": 6.0},
            policy_families={
                "hard_policy": "hard",
                "convex_policy": "convex",
            },
            oracle_ladders={
                "hard": {"baseline_loss": 10.0, "oracle_loss": 6.0},
                "convex": {"baseline_loss": 9.0, "oracle_loss": 3.0},
            },
        )

        self.assertEqual(captures, {"hard_policy": 0.5, "convex_policy": 0.5})


if __name__ == "__main__":
    unittest.main()
