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


from experiments.prob_head_structure_full_v1.sensor import (
    ACTION_FACTOR_GRID,
    DISAGREEMENT_COMPONENTS,
    SENSOR_FEATURE_SETS,
    SensorGeometryBlocked,
    SingleClassMetricUndefined,
    baseline_change_features,
    disagreement_components,
    disagreement_deltas,
    flag_threshold,
    outer_feature_origins,
    select_inner_pair_origins,
    sensor_feature_names,
    target_labels,
    widen_quantiles,
)


class DisagreementComponentTests(unittest.TestCase):
    def _teachers(self):
        p_zero = np.array([[0.10, 0.20], [0.30, 0.30], [0.20, 0.10]])
        quantiles = np.zeros((3, 2, 21))
        for head in range(3):
            quantiles[head] = np.linspace(0.0, 10.0 + head, 21)[None, :]
        mean = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
        return p_zero, quantiles, mean

    def test_every_declared_component_is_emitted(self):
        p_zero, quantiles, mean = self._teachers()
        components = disagreement_components(
            p_zero=p_zero, quantiles=quantiles, predictive_mean=mean, scale=1.0
        )
        for name in DISAGREEMENT_COMPONENTS:
            self.assertIn(name, components)
            self.assertTrue(np.isfinite(components[name]))

    def test_components_use_population_variance_then_average_over_horizon(self):
        p_zero, quantiles, mean = self._teachers()
        components = disagreement_components(
            p_zero=p_zero, quantiles=quantiles, predictive_mean=mean, scale=1.0
        )
        expected = float(np.mean(p_zero.var(axis=0, ddof=0)))
        self.assertAlmostEqual(components["D_zero"], expected)

    def test_identical_teachers_have_zero_disagreement_and_maximum_entropy(self):
        p_zero = np.full((3, 2), 0.25)
        quantiles = np.repeat(np.linspace(0.0, 5.0, 21)[None, None, :], 3, axis=0)
        quantiles = np.repeat(quantiles, 2, axis=1)
        mean = np.full((3, 2), 1.0)
        components = disagreement_components(
            p_zero=p_zero, quantiles=quantiles, predictive_mean=mean, scale=2.0
        )
        for name in ("D_zero", "D_center", "D_tail", "D_cdf", "D_mean"):
            self.assertAlmostEqual(components[name], 0.0)
        self.assertAlmostEqual(components["D_winner_entropy"], float(np.log(3.0)))

    def test_deltas_subtract_the_same_series_previous_origin_component(self):
        current = {"D_zero": 0.5, "D_center": 1.0, "D_tail": 2.0, "D_cdf": 0.25}
        previous = {"D_zero": 0.2, "D_center": 1.0, "D_tail": 0.5, "D_cdf": 0.25}
        deltas = disagreement_deltas(current, previous)
        self.assertAlmostEqual(deltas["Delta_D_zero"], 0.3)
        self.assertAlmostEqual(deltas["Delta_D_center"], 0.0)
        self.assertEqual(deltas["Delta_D_zero__missing"], 0.0)

        undefined = disagreement_deltas(current, None)
        self.assertEqual(undefined["Delta_D_tail__missing"], 1.0)
        self.assertTrue(np.isnan(undefined["Delta_D_tail"]))


class SensorGeometryTests(unittest.TestCase):
    def test_the_frozen_m5_inner_pair_origins_are_reproduced(self):
        origins = select_inner_pair_origins(lookback=96, horizon=28, model_train_end=1717)
        self.assertEqual(origins, (1465, 1493, 1521, 1549, 1577, 1605, 1633, 1661))

    def test_online_retail_has_no_valid_inner_pair_and_is_branch_blocked(self):
        with self.assertRaises(SensorGeometryBlocked) as context:
            select_inner_pair_origins(lookback=96, horizon=28, model_train_end=150)
        self.assertIn("REAL_C_SENSOR_GEOMETRY_BLOCKED", str(context.exception))

    def test_outer_feature_origins_precede_each_target_origin_by_one_horizon(self):
        self.assertEqual(
            outer_feature_origins((1773, 1801, 1829, 1857, 1885, 1913), horizon=28),
            (1745, 1773, 1801, 1829, 1857, 1885),
        )
        self.assertEqual(
            outer_feature_origins((206, 234, 262, 290, 318, 346), horizon=28),
            (178, 206, 234, 262, 290, 318),
        )


class SensorTargetTests(unittest.TestCase):
    def test_target_two_flags_either_undercovered_central_interval(self):
        labels = target_labels(
            next_scrps=1.0,
            scrps_threshold=2.0,
            coverage_90=0.88,
            coverage_95=0.99,
            zero_calibration=0.1,
            zero_threshold=0.5,
            best_head_changed=False,
        )
        self.assertEqual(labels["target_2"], 1)

        covered = target_labels(
            next_scrps=1.0,
            scrps_threshold=2.0,
            coverage_90=0.95,
            coverage_95=0.97,
            zero_calibration=0.1,
            zero_threshold=0.5,
            best_head_changed=False,
        )
        self.assertEqual(covered["target_2"], 0)

    def test_targets_one_and_three_are_strict_exceedances_of_their_threshold(self):
        labels = target_labels(
            next_scrps=2.0,
            scrps_threshold=2.0,
            coverage_90=1.0,
            coverage_95=1.0,
            zero_calibration=0.6,
            zero_threshold=0.5,
            best_head_changed=True,
        )
        self.assertEqual(labels["target_1"], 0)
        self.assertEqual(labels["target_3"], 1)
        self.assertEqual(labels["target_4"], 1)


class BaselineChangeFeatureTests(unittest.TestCase):
    def test_baseline_features_never_read_at_or_after_the_decision_boundary(self):
        values = np.arange(200, dtype=np.float64) % 5
        mean_forecast = np.full(28, 2.0)
        first = baseline_change_features(
            values,
            current_origin=100,
            horizon=28,
            available_from=0,
            scale=2.0,
            p0_predictive_mean=mean_forecast,
        )
        mutated = values.copy()
        mutated[128:] = 999.0
        second = baseline_change_features(
            mutated,
            current_origin=100,
            horizon=28,
            available_from=0,
            scale=2.0,
            p0_predictive_mean=mean_forecast,
        )
        for name in first:
            self.assertTrue(
                (np.isnan(first[name]) and np.isnan(second[name])) or first[name] == second[name],
                name,
            )

    def test_every_declared_c0_feature_and_indicator_is_emitted(self):
        values = np.arange(200, dtype=np.float64) % 5
        features = baseline_change_features(
            values,
            current_origin=100,
            horizon=28,
            available_from=0,
            scale=2.0,
            p0_predictive_mean=np.full(28, 2.0),
        )
        for name in (
            "previous_realized_residual",
            "zero_ratio_change",
            "scale_change",
            "last_event_gap_change",
            "recent_target_variance",
        ):
            self.assertIn(name, features)
        self.assertIn("last_event_gap_change__missing", features)


class SensorFeatureSetTests(unittest.TestCase):
    def test_the_four_feature_sets_are_frozen_and_nested_as_declared(self):
        self.assertEqual(tuple(SENSOR_FEATURE_SETS), ("C0", "C1", "C2", "C3"))
        c0 = set(sensor_feature_names("C0"))
        c1 = set(sensor_feature_names("C1"))
        c2 = set(sensor_feature_names("C2"))
        c3 = set(sensor_feature_names("C3"))
        self.assertTrue(c0.isdisjoint(c1))
        self.assertEqual(c2, c0 | c1)
        self.assertEqual(c3 - c0, {"D_total", "D_total__missing"})


class ActionPolicyTests(unittest.TestCase):
    def test_the_flag_threshold_is_the_higher_interpolation_eightieth_percentile(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(
            flag_threshold(scores), float(np.quantile(scores, 0.8, method="higher"))
        )

    def test_no_validation_row_flagged_is_an_undefined_metric_not_a_zero(self):
        with self.assertRaises(SingleClassMetricUndefined):
            flag_threshold(np.array([0.5, 0.5, 0.5]), require_flagged=True)

    def test_widening_preserves_the_median_and_stays_monotone_and_nonnegative(self):
        grid = np.linspace(0.01, 0.99, 21)
        quantiles = np.linspace(0.0, 20.0, 21)
        widened = widen_quantiles(quantiles, grid, factor=1.2, p_zero=0.05)
        median_index = int(np.argmin(np.abs(grid - 0.5)))
        self.assertAlmostEqual(widened[median_index], quantiles[median_index])
        self.assertTrue(np.all(np.diff(widened) >= 0.0))
        self.assertTrue(np.all(widened >= 0.0))

    def test_the_action_factor_grid_is_frozen(self):
        self.assertEqual(ACTION_FACTOR_GRID, (1.05, 1.1, 1.2, 1.35, 1.5))


from experiments.prob_head_structure_full_v1.controls import (
    CONTROL_SEED,
    FIFTY_PERCENT_RULE_CONTROLS,
    IdentificationFailure,
    control_rng,
    feature_row_shuffle,
    random_sensor_scores,
    recovery_ratio,
    regret_label_shuffle,
    teacher_identity_shuffle,
    teacher_name_permutation,
    teacher_quantile_shuffle,
    time_shuffle,
)


class ControlDeterminismTests(unittest.TestCase):
    def test_the_control_seed_is_frozen(self):
        self.assertEqual(CONTROL_SEED, 20260905_51 // 10 * 10 + 1 if False else 2026090551)

    def test_the_same_scope_always_yields_the_same_stream(self):
        first = control_rng("m5", "fold3", "regret").normal(size=5)
        second = control_rng("m5", "fold3", "regret").normal(size=5)
        third = control_rng("m5", "fold4", "regret").normal(size=5)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))


class ShuffleBoundaryTests(unittest.TestCase):
    def test_teacher_identity_shuffle_moves_all_three_channels_together(self):
        p_zero = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mean = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        quantiles = np.stack([p_zero, mean], axis=-1)
        shuffled = teacher_identity_shuffle(
            p_zero=p_zero, quantiles=quantiles, predictive_mean=mean, scope=("m5", "fold2")
        )
        for row in range(p_zero.shape[0]):
            self.assertEqual(sorted(shuffled["p_zero"][row]), sorted(p_zero[row]))
            order = [list(p_zero[row]).index(value) for value in shuffled["p_zero"][row]]
            self.assertEqual(
                [list(mean[row]).index(value) for value in shuffled["predictive_mean"][row]], order
            )

    def test_teacher_quantile_shuffle_permutes_whole_vectors_and_leaves_p0_alone(self):
        quantiles = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
        p_zero = np.full((4, 2), 0.25)
        shuffled = teacher_quantile_shuffle(quantiles=quantiles, p_zero=p_zero, scope=("m5",))
        self.assertTrue(np.array_equal(shuffled["p_zero"], p_zero))
        for head in range(quantiles.shape[1]):
            original = {tuple(row) for row in quantiles[:, head, :]}
            permuted = {tuple(row) for row in shuffled["quantiles"][:, head, :]}
            self.assertEqual(original, permuted)

    def test_regret_label_shuffle_permutes_the_complete_row_vector(self):
        regret = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 5.0], [1.0, 0.0, 4.0]])
        shuffled = regret_label_shuffle(regret, scope=("m5", "fold2"))
        self.assertEqual(
            sorted(tuple(row) for row in shuffled), sorted(tuple(row) for row in regret)
        )

    def test_feature_row_shuffle_keeps_every_row_intact(self):
        features = np.arange(12, dtype=np.float64).reshape(4, 3)
        shuffled = feature_row_shuffle(features, scope=("m5", "fold2"))
        self.assertEqual(
            sorted(tuple(row) for row in shuffled), sorted(tuple(row) for row in features)
        )

    def test_time_shuffle_permutes_origins_only_inside_one_series(self):
        rows = np.arange(12, dtype=np.float64).reshape(4, 3)
        series = np.array(["a", "a", "b", "b"])
        shuffled = time_shuffle(rows, series_ids=series, scope=("m5",))
        for name in ("a", "b"):
            mask = series == name
            self.assertEqual(
                sorted(tuple(row) for row in shuffled[mask]),
                sorted(tuple(row) for row in rows[mask]),
            )


class LabelSymmetryTests(unittest.TestCase):
    def test_a_global_teacher_permutation_leaves_the_components_numerically_identical(self):
        p_zero = np.array([[0.1, 0.2], [0.3, 0.3], [0.2, 0.1]])
        quantiles = np.stack(
            [np.linspace(0.0, 10.0 + head, 5)[None, :].repeat(2, axis=0) for head in range(3)]
        )
        mean = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
        report = teacher_name_permutation(
            p_zero=p_zero, quantiles=quantiles, predictive_mean=mean, scale=1.0
        )
        self.assertTrue(report["invariant"])
        self.assertLessEqual(report["max_absolute_difference"], 1e-12)
        self.assertNotEqual(report["permutation"], [0, 1, 2])


class RandomScoreTests(unittest.TestCase):
    def test_scores_are_uniform_reproducible_and_keyed_by_the_frozen_row_key(self):
        keys = [("m5", "s1", 1773), ("m5", "s2", 1773), ("m5", "s1", 1801)]
        first = random_sensor_scores(keys)
        second = random_sensor_scores(keys)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all((first >= 0.0) & (first < 1.0)))
        self.assertEqual(len(set(first.tolist())), 3)
        self.assertTrue(np.array_equal(random_sensor_scores(list(reversed(keys)))[::-1], first))


class RecoveryRuleTests(unittest.TestCase):
    def test_a_nonpositive_real_effect_is_an_identification_failure(self):
        with self.assertRaises(IdentificationFailure):
            recovery_ratio(real_effect=0.0, control_effect=0.0)
        with self.assertRaises(IdentificationFailure):
            recovery_ratio(real_effect=-0.01, control_effect=-0.02)

    def test_recovery_of_half_the_real_effect_or_more_fails_identification(self):
        passing = recovery_ratio(real_effect=0.02, control_effect=0.004)
        self.assertAlmostEqual(passing["recovery"], 0.2)
        self.assertTrue(passing["passed"])

        failing = recovery_ratio(real_effect=0.02, control_effect=0.01)
        self.assertAlmostEqual(failing["recovery"], 0.5)
        self.assertFalse(failing["passed"])

    def test_the_fifty_percent_rule_covers_exactly_the_declared_controls(self):
        self.assertEqual(len(FIFTY_PERCENT_RULE_CONTROLS), 8)
        self.assertNotIn("A single-teacher soft target", FIFTY_PERCENT_RULE_CONTROLS)
        self.assertIn("C random sensor score", FIFTY_PERCENT_RULE_CONTROLS)


if __name__ == "__main__":
    unittest.main()
