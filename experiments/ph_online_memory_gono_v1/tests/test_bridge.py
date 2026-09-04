from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from experiments.ph_online_memory_gono_v1.bridge import build_policy_cases
from experiments.ph_online_memory_gono_v1.pipeline import build_prediction_frame
from experiments.unified_temporal_27_v3.config import ExperimentConfig
from experiments.unified_temporal_27_v3.training import make_windows, train_scale


class ActualPredictionBridgeTests(unittest.TestCase):
    def _fixture(self):
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
                [0, 1, 2, 0, 3, 4, 0, 5, 6, 0],
                [1, 0, 2, 3, 0, 4, 5, 0, 6, 7],
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
            data, np.array([6], dtype=np.int32), 6, 8, cfg, scale
        )
        point = {"mean_prediction": np.zeros((2, 2), dtype=np.float32)}
        hurdle = {
            "mean_prediction": np.ones((2, 2), dtype=np.float32),
            "p_prediction": np.full((2, 2), 0.5, dtype=np.float32),
            "mu_prediction": np.full((2, 2), 2.0, dtype=np.float32),
        }
        predictions = build_prediction_frame(data, windows, point, hurdle)
        return cfg, data, windows, predictions

    def test_real_windows_bridge_adds_train_only_scale_and_pre_origin_history(self):
        cfg, data, windows, predictions = self._fixture()
        cases = build_policy_cases(
            predictions,
            data,
            windows,
            model_train_end=cfg.train_end,
            horizon=cfg.horizon,
            lookback=cfg.lookback,
        )
        self.assertEqual(len(cases), 2)
        for row in cases.itertuples(index=False):
            index = 0 if row.series_id == "a" else 1
            np.testing.assert_array_equal(
                row.history, data["y"][index, 3:6]
            )
            expected_scale_squared = float(
                np.mean(np.square(data["y"][index, :6], dtype=np.float64))
                + 1e-8
            )
            self.assertAlmostEqual(row.policy_scale_squared, expected_scale_squared)

    def test_bridge_rejects_window_history_that_is_not_exactly_pre_origin(self):
        cfg, data, windows, predictions = self._fixture()
        corrupted = windows.history.copy()
        corrupted[0, -1] = data["y"][0, 6]
        with self.assertRaisesRegex(ValueError, "(?i)(history|origin|leak)"):
            build_policy_cases(
                predictions,
                data,
                replace(windows, history=corrupted),
                model_train_end=cfg.train_end,
                horizon=cfg.horizon,
                lookback=cfg.lookback,
            )


if __name__ == "__main__":
    unittest.main()
