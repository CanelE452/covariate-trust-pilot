"""Feature, prediction-artifact, and runtime-projection contracts."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.ph_online_memory_gono_v1.pipeline import (
    apply_robust_scaler,
    build_prediction_frame,
    extract_retrieval_features,
    fit_robust_scaler,
    normalized_loss_frame,
    project_full_seed0_runtime,
    project_retrieval_runtime,
)
from experiments.unified_temporal_27_v3.config import ExperimentConfig
from experiments.unified_temporal_27_v3.training import make_windows, train_scale


class RetrievalFeatureTests(unittest.TestCase):
    def test_features_match_frozen_definitions_and_missing_indicators(self):
        history = np.array([0.0, 2.0, 0.0, 4.0, 0.0, 0.0])
        point = np.array([1.0, 2.0])
        hurdle = np.array([2.0, 4.0])

        features = extract_retrieval_features(
            history,
            point,
            hurdle,
            canonical_train_scale=2.0,
        )

        self.assertEqual(features.shape, (13,))
        expected_continuous = np.array(
            [
                4.0 / 6.0,
                np.log1p(2.0),
                np.log1p(2.0),
                0.0,
                np.log1p(3.0),
                np.log1p(np.sqrt(2.0) / 3.0),
                np.log1p(np.sqrt(20.0 / 6.0) / 2.0),
                np.log1p(np.sqrt(2.5) / 2.0),
            ]
        )
        np.testing.assert_allclose(features[:8], expected_continuous)
        np.testing.assert_array_equal(
            features[8:], np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        )

    def test_robust_scaler_fits_continuous_memory_only_and_keeps_indicators(self):
        memory = np.array(
            [
                [0, 1, 5, 2, 3, 4, 5, 6, 0, 0, 1, 0, 1],
                [1, 2, 5, 4, 3, 6, 7, 8, 1, 0, 0, 0, 1],
                [2, 3, 5, 6, 3, 8, 9, 10, 0, 1, 0, 1, 0],
            ],
            dtype=float,
        )
        scaler = fit_robust_scaler(memory)
        self.assertTrue(bool(scaler["constant_continuous"][2]))
        self.assertTrue(bool(scaler["constant_continuous"][4]))

        transformed = apply_robust_scaler(memory[1], scaler)
        np.testing.assert_allclose(transformed[:8], np.zeros(8))
        np.testing.assert_array_equal(transformed[8:], memory[1, 8:])


class PredictionArtifactTests(unittest.TestCase):
    def test_prediction_frame_is_fully_paired_and_losses_use_train_only_scale(self):
        cfg = ExperimentConfig(
            n_series=2,
            length=10,
            lookback=3,
            horizon=2,
            period=2,
            train_end=6,
            val_end=8,
            c_regime_switch=5,
        )
        y = np.array(
            [
                [1, 0, 1, 0, 1, 0, 2, 0, 4, 0],
                [2, 2, 0, 0, 2, 2, 0, 2, 0, 4],
            ],
            dtype=np.float32,
        )
        data = {
            "name": "toy",
            "series_id": np.array(["a", "b"]),
            "y": y,
            "z": (y > 0).astype(np.float32),
        }
        scale = train_scale(data, cfg)
        windows = make_windows(
            data, np.array([6, 8]), cfg.train_end, cfg.length, cfg, scale
        )
        point = {"mean_prediction": np.zeros((4, 2), dtype=np.float32)}
        hurdle = {
            "mean_prediction": np.ones((4, 2), dtype=np.float32),
            "p_prediction": np.full((4, 2), 0.5, dtype=np.float32),
            "mu_prediction": np.full((4, 2), 2.0, dtype=np.float32),
        }

        frame = build_prediction_frame(data, windows, point, hurdle)
        self.assertEqual(len(frame), 8)
        self.assertFalse(frame.duplicated(["dataset_id", "series_id", "origin", "step"]).any())
        np.testing.assert_allclose(
            frame["hurdle_mean_prediction"],
            frame["hurdle_p_prediction"] * frame["hurdle_mu_prediction"],
        )

        losses = normalized_loss_frame(
            frame, data, model_train_end=6, horizon=2
        )
        self.assertEqual(len(losses), 4)
        altered = {**data, "y": data["y"].copy()}
        altered["y"][:, 6:] = 9999.0
        scales_before = losses.set_index("series_id")["policy_scale_squared"].to_dict()
        scales_after = normalized_loss_frame(
            frame, altered, model_train_end=6, horizon=2
        ).set_index("series_id")["policy_scale_squared"].to_dict()
        self.assertEqual(scales_before, scales_after)

        incomplete = frame.drop(frame.index[0])
        with self.assertRaisesRegex(ValueError, "(?i)(horizon|step|complete)"):
            normalized_loss_frame(
                incomplete, data, model_train_end=6, horizon=2
            )


class RuntimeProjectionTests(unittest.TestCase):
    def test_projection_scales_by_series_and_train_window_counts(self):
        report = project_full_seed0_runtime(
            smoke_train_seconds={"point": 10.0, "hurdle": 20.0},
            smoke_inference_seconds_per_origin={"point": 1.0, "hurdle": 2.0},
            smoke_n_series=200,
            full_series={"m5": 1000, "favorita": 2000},
            train_origins={"m5": 100, "favorita": 50},
            forecast_origins={"m5": 7, "favorita": 7},
        )
        # M5 factor=5; Favorita factor=5 after accounting for half as many
        # train windows.  Both arms total 30 seconds in the smoke.
        self.assertAlmostEqual(report["training_seconds"], 300.0)
        self.assertAlmostEqual(report["inference_seconds"], 315.0)
        self.assertAlmostEqual(report["total_seconds"], 615.0)

    def test_retrieval_projection_counts_quadratic_pool_growth_and_all_passes(self):
        report = project_retrieval_runtime(
            smoke_seconds=2.0,
            smoke_n_series=200,
            full_series={"m5": 400, "favorita": 200},
            evaluation_origins=6,
            retrieval_passes_per_dataset=7,
        )
        # Six queries see 1+2+...+6 resolved origin blocks.  M5 has a
        # four-fold N^2 factor; Favorita has a one-fold factor.
        self.assertEqual(report["resolved_origin_units"], 21)
        self.assertAlmostEqual(report["by_dataset"]["m5"]["seconds"], 1176.0)
        self.assertAlmostEqual(
            report["by_dataset"]["favorita"]["seconds"], 294.0
        )
        self.assertAlmostEqual(report["total_seconds"], 1470.0)


if __name__ == "__main__":
    unittest.main()
