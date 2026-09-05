from __future__ import annotations

import unittest

import numpy as np

from experiments.prob_head_structure_full_v1.temporal_features import (
    BASELINE_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    FeatureBoundaryError,
    build_feature_matrix,
    fold_median_imputer,
    temporal_features_for_series,
    train_descriptors_for_series,
)


class TrainDescriptorTests(unittest.TestCase):
    def test_train_descriptors_use_availability_valid_train_observations_only(self):
        values = np.array([0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 6.0], dtype=np.float64)
        descriptors = train_descriptors_for_series(values, available_from=0, train_end=8)

        self.assertAlmostEqual(descriptors["train_zero_ratio"], 5 / 8)
        self.assertAlmostEqual(descriptors["train_ADI"], 8 / 3)
        positives = np.array([2.0, 4.0, 6.0])
        expected_cv2 = (positives.std(ddof=1) / positives.mean()) ** 2
        self.assertAlmostEqual(descriptors["train_positive_CV2"], expected_cv2)
        self.assertAlmostEqual(descriptors["train_scale"], float(np.sqrt(np.mean(values**2) + 1e-8)))

    def test_leading_unavailable_observations_are_excluded(self):
        values = np.array([0.0, 0.0, 3.0, 0.0, 3.0], dtype=np.float64)
        descriptors = train_descriptors_for_series(values, available_from=2, train_end=5)
        self.assertAlmostEqual(descriptors["train_zero_ratio"], 1 / 3)
        self.assertAlmostEqual(descriptors["train_ADI"], 3 / 2)

    def test_a_train_window_without_positives_marks_the_dependent_descriptors_missing(self):
        values = np.zeros(10, dtype=np.float64)
        descriptors = train_descriptors_for_series(values, available_from=0, train_end=10)
        self.assertTrue(np.isnan(descriptors["train_ADI"]))
        self.assertTrue(np.isnan(descriptors["train_positive_CV2"]))
        self.assertAlmostEqual(descriptors["train_zero_ratio"], 1.0)


class TemporalFeatureTests(unittest.TestCase):
    def _series(self) -> np.ndarray:
        values = np.zeros(200, dtype=np.float64)
        values[np.arange(10, 200, 10)] = np.arange(1, 20, dtype=np.float64)
        return values

    def test_every_declared_feature_and_its_indicator_are_emitted(self):
        features = temporal_features_for_series(
            self._series(), origin=150, available_from=0, train_end=120, dataset_id="m5"
        )
        for name in TEMPORAL_FEATURE_NAMES:
            self.assertIn(name, features)
        for name in ("recent_gap_CV", "recent_positive_CV", "interval_autocorrelation"):
            self.assertIn(f"{name}__missing", features)

    def test_time_since_last_positive_counts_from_the_observation_before_the_origin(self):
        values = np.zeros(60, dtype=np.float64)
        values[41] = 5.0
        features = temporal_features_for_series(
            values, origin=50, available_from=0, train_end=40, dataset_id="m5"
        )
        self.assertAlmostEqual(features["time_since_last_positive"], 50 - 1 - 41)

    def test_gaps_require_both_endpoints_inside_the_trailing_window(self):
        values = np.zeros(200, dtype=np.float64)
        values[[10, 100, 140, 150]] = 1.0
        features = temporal_features_for_series(
            values, origin=160, available_from=0, train_end=90, dataset_id="m5"
        )
        # trailing 96 window is [64,160); only the 100->140->150 gaps qualify
        self.assertAlmostEqual(features["recent_mean_gap"], (40 + 10) / 2)

    def test_undefined_scalars_emit_both_a_value_and_their_indicator(self):
        values = np.zeros(200, dtype=np.float64)
        values[150] = 3.0
        features = temporal_features_for_series(
            values, origin=160, available_from=0, train_end=120, dataset_id="m5"
        )
        self.assertEqual(features["recent_gap_CV__missing"], 1.0)
        self.assertTrue(np.isnan(features["recent_gap_CV"]))
        self.assertEqual(features["recent_positive_CV__missing"], 1.0)

    def test_seasonal_phase_is_future_known_weekly_sine_and_cosine(self):
        features = temporal_features_for_series(
            self._series(), origin=147, available_from=0, train_end=120, dataset_id="m5"
        )
        angle = 2.0 * np.pi * (147 % 7) / 7.0
        self.assertAlmostEqual(features["seasonal_sin"], float(np.sin(angle)))
        self.assertAlmostEqual(features["seasonal_cos"], float(np.cos(angle)))

    def test_mutating_the_origin_and_every_later_value_leaves_features_identical(self):
        values = self._series()
        before = temporal_features_for_series(
            values, origin=150, available_from=0, train_end=120, dataset_id="m5"
        )
        mutated = values.copy()
        mutated[150:] = 987.0
        after = temporal_features_for_series(
            mutated, origin=150, available_from=0, train_end=120, dataset_id="m5"
        )
        self.assertEqual(list(before), list(after))
        for name in before:
            self.assertTrue(
                (np.isnan(before[name]) and np.isnan(after[name])) or before[name] == after[name],
                name,
            )

    def test_an_origin_at_or_before_the_history_start_is_rejected(self):
        with self.assertRaises(FeatureBoundaryError):
            temporal_features_for_series(
                self._series(), origin=0, available_from=0, train_end=120, dataset_id="m5"
            )
        with self.assertRaises(FeatureBoundaryError):
            temporal_features_for_series(
                self._series(), origin=150, available_from=0, train_end=200, dataset_id="m5"
            )


class FeatureMatrixTests(unittest.TestCase):
    def test_baseline_matrix_contains_only_the_four_declared_descriptors(self):
        rows = [
            {"train_ADI": 2.0, "train_positive_CV2": 0.5, "train_scale": 3.0, "train_zero_ratio": 0.4},
            {"train_ADI": 3.0, "train_positive_CV2": 0.7, "train_scale": 1.0, "train_zero_ratio": 0.6},
        ]
        matrix, names = build_feature_matrix(rows, feature_set="baseline")
        self.assertEqual(names, list(BASELINE_FEATURE_NAMES))
        self.assertEqual(matrix.shape, (2, 4))

    def test_all_missing_fold_feature_uses_zero_indicator_one_and_the_identity_scaler(self):
        rows = [
            {"a": np.nan, "a__missing": 1.0, "b": 2.0},
            {"a": np.nan, "a__missing": 1.0, "b": 4.0},
        ]
        matrix, names = build_feature_matrix(rows, feature_set=("a", "a__missing", "b"))
        imputer = fold_median_imputer(matrix, names)

        self.assertEqual(imputer["all_missing_features"], ["a"])
        self.assertEqual(imputer["record"], "ALL_MISSING_TRAIN_FEATURE")
        index = names.index("a")
        self.assertEqual(imputer["median"][index], 0.0)
        self.assertEqual(imputer["mean"][index], 0.0)
        self.assertEqual(imputer["scale"][index], 1.0)

        transformed = imputer["transform"](matrix)
        self.assertTrue(np.all(np.isfinite(transformed)))
        self.assertTrue(np.allclose(transformed[:, index], 0.0))

    def test_a_later_nonmissing_value_keeps_its_raw_value_under_the_frozen_scaler(self):
        train = np.array([[np.nan, 1.0], [np.nan, 3.0]])
        names = ["a", "b"]
        imputer = fold_median_imputer(train, names)
        transformed = imputer["transform"](np.array([[7.0, 3.0]]))
        self.assertAlmostEqual(transformed[0, 0], 7.0)

    def test_medians_are_never_fitted_on_rows_outside_the_declared_fold(self):
        train = np.array([[1.0, 1.0], [3.0, 3.0]])
        imputer = fold_median_imputer(train, ["a", "b"])
        self.assertAlmostEqual(imputer["median"][0], 2.0)
        held_out = np.array([[np.nan, 100.0]])
        self.assertAlmostEqual(imputer["transform"](held_out)[0, 0], (2.0 - imputer["mean"][0]) / imputer["scale"][0])


from experiments.prob_head_structure_full_v1.routing import (
    TEMPERATURE_GRID,
    InsufficientRoutingVariation,
    RoutingBranchBlocked,
    expanding_crossfit_weights,
    head_regret,
    regret_spearman,
    select_inner_origins,
    select_temperature,
    soft_routing_target,
)


class RegretAndSoftTargetTests(unittest.TestCase):
    def test_regret_is_the_gap_to_the_best_head_on_each_row(self):
        losses = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
        regret = head_regret(losses)
        self.assertTrue(np.allclose(regret, [[0.0, 1.0, 3.0], [0.0, 0.0, 0.0]]))

    def test_soft_target_is_a_stable_softmax_of_the_negative_scaled_regret(self):
        regret = np.array([[0.0, 1.0, 3.0]])
        target = soft_routing_target(regret, temperature=0.5)
        self.assertAlmostEqual(float(target.sum()), 1.0)
        self.assertGreater(target[0, 0], target[0, 1])
        huge = soft_routing_target(np.array([[0.0, 1e6]]), temperature=0.25)
        self.assertTrue(np.all(np.isfinite(huge)))
        self.assertAlmostEqual(float(huge.sum()), 1.0)

    def test_temperature_grid_is_frozen(self):
        self.assertEqual(TEMPERATURE_GRID, (0.25, 0.5, 1.0))


class InnerOriginTests(unittest.TestCase):
    def test_the_last_eight_non_overlapping_valid_origins_are_preferred(self):
        origins = select_inner_origins(lookback=96, horizon=28, model_train_end=1717)
        self.assertEqual(len(origins), 8)
        self.assertEqual(origins[-1], 1717 - 28)
        self.assertEqual(sorted(set(np.diff(origins))), [28])

    def test_the_frozen_m5_and_online_retail_inner_origins_are_reproduced(self):
        self.assertEqual(
            select_inner_origins(lookback=96, horizon=28, model_train_end=1717),
            (1493, 1521, 1549, 1577, 1605, 1633, 1661, 1689),
        )
        self.assertEqual(
            select_inner_origins(lookback=96, horizon=28, model_train_end=150),
            (96, 100, 103, 107, 111, 115, 118, 122),
        )

    def test_a_short_panel_falls_back_to_evenly_spaced_unique_origins(self):
        origins = select_inner_origins(lookback=48, horizon=28, model_train_end=150)
        self.assertEqual(len(origins), len(set(origins)))
        self.assertEqual(list(origins), sorted(origins))
        self.assertGreaterEqual(len(origins), 4)

    def test_fewer_than_four_unique_origins_blocks_the_b_branch(self):
        with self.assertRaises(RoutingBranchBlocked):
            select_inner_origins(lookback=96, horizon=28, model_train_end=100)


class TemperatureSelectionTests(unittest.TestCase):
    def test_fold_two_uses_the_fixed_tie_first_temperature(self):
        chosen = select_temperature(fold_index=2, prior_origin_scores=None)
        self.assertEqual(chosen["temperature"], 0.25)
        self.assertEqual(chosen["rule"], "k2_fixed_tie_first")

    def test_later_folds_minimize_out_of_fold_expected_regret_and_break_ties_low(self):
        scores = {0.25: 0.30, 0.5: 0.20, 1.0: 0.20}
        chosen = select_temperature(fold_index=4, prior_origin_scores=scores)
        self.assertEqual(chosen["temperature"], 0.5)
        self.assertEqual(chosen["rule"], "expanding_oof_expected_regret")


class ExpandingCrossfitTests(unittest.TestCase):
    def _panel(self, origins, rows=40, seed=3):
        rng = np.random.default_rng(seed)
        records = []
        for origin in origins:
            features = rng.normal(size=(rows, 3))
            losses = np.abs(rng.normal(size=(rows, 3))) + features[:, [0]] * 0.1
            records.append({"origin": origin, "features": features, "losses": losses})
        return records

    def test_heldout_weights_for_fold_k_never_use_fold_k_or_later_labels(self):
        origins = [10, 20, 30, 40]
        panel = self._panel(origins)
        result = expanding_crossfit_weights(panel, feature_names=["f0", "f1", "f2"])
        self.assertEqual([fold["origin"] for fold in result["folds"]], origins[1:])
        for fold in result["folds"]:
            self.assertLess(max(fold["fit_origins"]), fold["origin"])
            self.assertEqual(fold["weights"].shape, (40, 3))
            self.assertTrue(np.allclose(fold["weights"].sum(axis=1), 1.0))

    def test_earlier_heldout_weights_are_never_backfilled_after_later_folds(self):
        origins = [10, 20, 30, 40]
        panel = self._panel(origins)
        full = expanding_crossfit_weights(panel, feature_names=["f0", "f1", "f2"])
        partial = expanding_crossfit_weights(panel[:3], feature_names=["f0", "f1", "f2"])
        for early, late in zip(partial["folds"], full["folds"]):
            self.assertTrue(np.allclose(early["weights"], late["weights"]))


class RegretSpearmanTests(unittest.TestCase):
    def test_the_statistic_flattens_weights_against_negative_regret(self):
        weights = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        regret = np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
        self.assertGreater(regret_spearman(weights, regret), 0.5)

    def test_a_constant_predictor_is_reported_as_insufficient_variation(self):
        weights = np.full((4, 3), 1 / 3)
        regret = np.array([[0.0, 1.0, 2.0]] * 4)
        with self.assertRaises(InsufficientRoutingVariation):
            regret_spearman(weights, regret)


if __name__ == "__main__":
    unittest.main()
