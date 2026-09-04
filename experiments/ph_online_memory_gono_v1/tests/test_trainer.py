"""Contract tests for the canonical-trainer adapter."""

from __future__ import annotations

import inspect
import math
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.om_factorization_killtest import train as km_train
from experiments.ph_online_memory_gono_v1.trainer import train_one_on_split
from experiments.unified_temporal_27_v3.config import ExperimentConfig


def _supplied_split() -> tuple[km_train.Split, ExperimentConfig]:
    """Build a small real split that crosses the canonical batch boundary."""
    n_series = 52
    cfg = ExperimentConfig(
        n_series=n_series,
        length=17,
        lookback=4,
        horizon=3,
        period=3,
        train_end=9,
        val_end=13,
        c_regime_switch=8,
    )
    series = np.arange(n_series, dtype=np.int32)[:, None]
    time = np.arange(cfg.length, dtype=np.int32)[None, :]
    occurrence = ((series + 2 * time) % 5 != 0).astype(np.float32)
    positive_size = (1.0 + 0.25 * ((3 * series + time) % 7)).astype(np.float32)
    values = (occurrence * positive_size).astype(np.float32)
    data = {"y": values, "z": occurrence}
    scale = km_train.train_scale(data, cfg)

    train = km_train.make_windows(
        data,
        np.array([4, 5, 6, 7, 8], dtype=np.int32),
        0,
        cfg.train_end,
        cfg,
        scale,
    )
    validation = km_train.make_windows(
        data,
        np.array([9, 11, 12], dtype=np.int32),
        cfg.train_end,
        cfg.val_end,
        cfg,
        scale,
    )
    test = km_train.make_windows(
        data,
        np.array([13, 15, 16], dtype=np.int32),
        cfg.val_end,
        cfg.length,
        cfg,
        scale,
    )
    train_values = values[:, : cfg.train_end]
    train_occurrence = occurrence[:, : cfg.train_end]
    positives = train_values[train_occurrence > 0.0]
    positive_variance = max(float(positives.var(ddof=1)), 1e-6)
    return km_train.Split(train, validation, test, scale, positive_variance), cfg


class CanonicalTrainerAdapterTest(unittest.TestCase):
    def test_returns_the_exact_canonical_result_for_the_supplied_split(self) -> None:
        split, cfg = _supplied_split()
        self.assertGreater(split.train.history.shape[0], 256)
        self.assertTrue(np.any(~split.train.target_mask))

        prediction_keys = {
            "M0PM_point_mse_param_matched": ("mean_prediction",),
            "M1_factorized_mean": (
                "mean_prediction",
                "p_prediction",
                "mu_prediction",
            ),
        }
        device = torch.device("cpu")
        model_seed = 17

        for model_name, expected_prediction_keys in prediction_keys.items():
            with self.subTest(model_name=model_name):
                real_train_one = km_train.train_one

                with mock.patch.object(
                    km_train, "build_splits", autospec=True, return_value=split
                ):
                    expected = real_train_one(
                        model_name, {}, cfg, model_seed, device
                    )
                expected_state = {
                    key: value.detach().cpu().clone()
                    for key, value in expected["model"].state_dict().items()
                }
                expected_predictions = {
                    key: value.copy()
                    for key, value in expected["predictions"].items()
                }

                with mock.patch.object(
                    km_train, "train_one", wraps=real_train_one
                ) as canonical_call:
                    result = train_one_on_split(
                        model_name=model_name,
                        split=split,
                        cfg=cfg,
                        model_seed=model_seed,
                        device=device,
                    )

                canonical_call.assert_called_once()
                bound = inspect.signature(real_train_one).bind(
                    *canonical_call.call_args.args,
                    **canonical_call.call_args.kwargs,
                )
                bound.apply_defaults()
                self.assertIs(result["splits"], split)

                arguments = bound.arguments
                self.assertEqual(arguments["model_name"], model_name)
                self.assertIs(arguments["cfg"], cfg)
                self.assertEqual(arguments["model_seed"], model_seed)
                self.assertEqual(arguments["device"], device)

                actual_state = result["model"].state_dict()
                self.assertEqual(tuple(actual_state), tuple(expected_state))
                for key, expected_tensor in expected_state.items():
                    self.assertTrue(
                        torch.equal(actual_state[key].detach().cpu(), expected_tensor),
                        msg=f"state_dict tensor changed: {key}",
                    )

                self.assertEqual(tuple(result["predictions"]), expected_prediction_keys)
                self.assertEqual(tuple(expected_predictions), expected_prediction_keys)
                for key, expected_array in expected_predictions.items():
                    np.testing.assert_array_equal(
                        result["predictions"][key], expected_array
                    )

                self.assertEqual(result["best_epoch"], expected["best_epoch"])
                self.assertEqual(
                    result["best_validation_mean_mse"],
                    expected["best_validation_mean_mse"],
                )
                self.assertEqual(result["n_parameters"], expected["n_parameters"])
                self.assertEqual(result["n_parameters"], 54)
                self.assertFalse(result["model"].training)
                self.assertTrue(math.isfinite(result["train_seconds"]))
                self.assertGreaterEqual(result["train_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
