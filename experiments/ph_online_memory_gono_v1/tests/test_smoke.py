"""Synthetic smoke-construction contracts; these tests do not fit a model."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.ph_online_memory_gono_v1.data import build_external_split
from experiments.ph_online_memory_gono_v1.smoke import stratified_smoke_ids
from experiments.unified_temporal_27_v3.config import ExperimentConfig


class SmokeConstructionTest(unittest.TestCase):
    def test_stratified_sample_is_deterministic_unique_and_covers_quadrants(self):
        """Catches an unstratified, replacement, or nondeterministic smoke sample."""
        rows = []
        for zero_band in range(4):
            for scale_band in range(4):
                for repeat in range(5):
                    rows.append(
                        {
                            "series_id": f"z{zero_band}s{scale_band}r{repeat}",
                            "zero_ratio_train": zero_band + repeat / 100.0,
                            "train_scale": float(np.exp(scale_band + repeat / 100.0)),
                        }
                    )
        descriptors = pd.DataFrame(rows)

        first = stratified_smoke_ids(descriptors, n=32, seed=20260904)
        second = stratified_smoke_ids(descriptors, n=32, seed=20260904)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(len(set(first)), 32)
        selected = descriptors.set_index("series_id").loc[first]
        self.assertLess(float(selected["zero_ratio_train"].min()), 1.0)
        self.assertGreater(float(selected["zero_ratio_train"].max()), 3.0)
        self.assertLess(float(np.log(selected["train_scale"]).min()), 1.0)
        self.assertGreater(float(np.log(selected["train_scale"]).max()), 3.0)

    def test_external_split_uses_stride_and_masks_only_unavailable_targets(self):
        """Catches a wrong training stride or unmasked M5 pre-availability targets."""
        cfg = ExperimentConfig(
            n_series=2,
            length=40,
            lookback=4,
            horizon=3,
            period=3,
            train_end=20,
            val_end=23,
            c_regime_switch=10,
        )
        y = np.ones((2, cfg.length), dtype=np.float32)
        data = {
            "name": "m5",
            "series_id": np.array(["always", "late"]),
            "y": y,
            "z": (y > 0).astype(np.float32),
            "available_from": np.array([0, 8]),
        }

        split = build_external_split(
            data,
            cfg,
            train_origin_stride=2,
            forecast_origins=np.array([23, 26, 29], dtype=np.int32),
        )

        np.testing.assert_array_equal(
            split.train.origins, np.array([4, 6, 8, 10, 12, 14, 16])
        )
        np.testing.assert_array_equal(split.validation.origins, np.array([20]))
        np.testing.assert_array_equal(split.test.origins, np.array([23, 26, 29]))
        train_mask = split.train.target_mask.reshape(2, -1, cfg.horizon)
        self.assertTrue(bool(train_mask[0].all()))
        self.assertFalse(bool(train_mask[1, 0].any()))
        self.assertTrue(bool(train_mask[1, 2:].all()))
        self.assertTrue(bool(split.validation.target_mask.all()))
        self.assertTrue(bool(split.test.target_mask.all()))

    def test_external_split_rejects_fractional_and_partial_tail_origins(self):
        """Catches silent int truncation and canonical partial-horizon padding."""
        cfg = ExperimentConfig(
            n_series=1,
            length=40,
            lookback=4,
            horizon=3,
            period=3,
            train_end=20,
            val_end=23,
            c_regime_switch=10,
        )
        y = np.ones((1, cfg.length), dtype=np.float32)
        data = {
            "name": "m5",
            "series_id": np.array(["a"]),
            "y": y,
            "z": y.copy(),
            "available_from": np.array([0]),
        }
        with self.assertRaisesRegex(ValueError, "(?i)(integer|origin)"):
            build_external_split(
                data,
                cfg,
                train_origin_stride=2,
                forecast_origins=np.array([23.5, 26.0]),
            )
        with self.assertRaisesRegex(ValueError, "(?i)(horizon|length|tail|origin)"):
            build_external_split(
                data,
                cfg,
                train_origin_stride=2,
                forecast_origins=np.array([23, 39]),
            )


if __name__ == "__main__":
    unittest.main()
