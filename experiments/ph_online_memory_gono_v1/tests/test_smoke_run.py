"""Mock-driven orchestration tests for the preregistered M5 smoke run."""

from __future__ import annotations

import itertools
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from experiments.om_factorization_killtest import train as km_train
from experiments.ph_online_memory_gono_v1.smoke_run import run_m5_smoke
from experiments.ph_online_memory_gono_v1.smoke import stratified_smoke_ids
from experiments.unified_temporal_27_v3.config import ExperimentConfig
from experiments.unified_temporal_27_v3.training import WindowArrays


def _test_windows(n_series: int, horizon: int) -> WindowArrays:
    origins = np.array([1745, 1773], dtype=np.int32)
    n_origins = len(origins)
    rows = n_series * n_origins
    target = np.empty((n_series, n_origins, horizon), dtype=np.float32)
    for series_index in range(n_series):
        for origin_index in range(n_origins):
            target[series_index, origin_index] = (
                1.0 + series_index % 7 + origin_index
            )
    return WindowArrays(
        history=np.ones((rows, 96), dtype=np.float32),
        target=target.reshape(rows, horizon),
        occurrence=np.ones((rows, horizon), dtype=np.float32),
        target_mask=np.ones((rows, horizon), dtype=bool),
        gap=np.zeros(rows, dtype=np.float32),
        gap_event_observed=np.zeros(rows, dtype=np.float32),
        gap_censor_lower=np.zeros(rows, dtype=np.float32),
        scale=np.ones(rows, dtype=np.float32),
        origins=origins,
        valid_lengths=np.full(n_origins, horizon, dtype=np.int32),
        n_series=n_series,
        split_start=1745,
        split_end=1801,
    )


def _population() -> tuple[dict, km_train.Split]:
    n_series = 208
    length = 1941
    series_ids = np.array([f"s{i:03d}" for i in range(n_series)])
    time = np.arange(length)[None, :]
    rows = np.arange(n_series)[:, None]
    y = (((time + rows) % 5) == 0).astype(np.float32)
    y *= (1.0 + rows % 7).astype(np.float32)
    descriptors = pd.DataFrame(
        {
            "series_id": series_ids,
            "zero_ratio_train": np.mean(y[:, :1717] == 0.0, axis=1),
            "train_scale": np.maximum(np.mean(y[:, :1717], axis=1), 1.0),
        }
    )
    cfg = ExperimentConfig(
        n_series=n_series,
        length=length,
        lookback=96,
        horizon=28,
        period=28,
        train_end=1717,
        val_end=1745,
        c_regime_switch=1000,
    )
    data = {
        "name": "m5",
        "series_id": series_ids,
        "y": y,
        "z": (y > 0.0).astype(np.float32),
        "available_from": np.zeros(n_series, dtype=np.int32),
    }
    test = _test_windows(200, cfg.horizon)
    split = km_train.Split(
        train=SimpleNamespace(n_origins=228),
        validation=SimpleNamespace(n_origins=1),
        test=test,
        scale=np.ones(200, dtype=np.float32),
        positive_variance=1.0,
    )
    return {
        "data": data,
        "descriptors": descriptors,
        "cfg": cfg,
        "manifest": {"population": "unit-test"},
    }, split


class SmokeRunnerTests(unittest.TestCase):
    def test_runs_sequential_arms_pairs_predictions_and_checks_causal_memory(self):
        population, split = _population()
        point_model = object()
        hurdle_model = object()
        train_results = [
            {
                "model": point_model,
                "train_seconds": 8.0,
                "best_epoch": 4,
                "best_validation_mean_mse": 0.8,
                "n_parameters": 7056,
            },
            {
                "model": hurdle_model,
                "train_seconds": 9.0,
                "best_epoch": 5,
                "best_validation_mean_mse": 0.7,
                "n_parameters": 7056,
            },
        ]

        def predict(model, windows, _device):
            shape = (windows.n_series * windows.n_origins, 28)
            if model is point_model:
                return {"mean_prediction": np.full(shape, 0.5, dtype=np.float32)}
            probability = np.full(shape, 0.5, dtype=np.float32)
            magnitude = np.full(shape, 2.0, dtype=np.float32)
            return {
                "mean_prediction": probability * magnitude,
                "p_prediction": probability,
                "mu_prediction": magnitude,
            }

        clock = (float(value) for value in itertools.count(step=10))
        device = torch.device("cuda:0")
        smoke_ids = stratified_smoke_ids(
            population["descriptors"], n=200, seed=20260904
        )
        policy_cases = pd.DataFrame({"placeholder": np.arange(400)})
        b4_result = pd.DataFrame(
            {
                "dataset_id": ["m5"] * 200,
                "series_id": smoke_ids,
                "origin": [1773] * 200,
                "b4_hurdle_weight": [0.5] * 200,
            }
        )
        m1_result = pd.DataFrame(
            {
                "dataset_id": ["m5"] * 200,
                "series_id": smoke_ids,
                "origin": [1773] * 200,
                "resolved_origins": [(1745,)] * 200,
                "neighbor_count": [32] * 200,
                "neighbor_series_ids": [
                    tuple(f"n{j:03d}" for j in range(32)) for _ in range(200)
                ],
                "constant_continuous_features": [(False,) * 8] * 200,
                "m1_hurdle_weight": [0.5] * 200,
                "m1_normalized_loss": [1.0] * 200,
            }
        )
        with (
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.build_external_split",
                return_value=split,
            ) as build_split,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.train_one_on_split",
                side_effect=train_results,
            ) as train,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.km_train.predict",
                side_effect=predict,
            ) as predict_call,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.time.perf_counter",
                side_effect=lambda: next(clock),
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.torch.cuda.synchronize"
            ) as synchronize,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.torch.cuda.reset_peak_memory_stats"
            ) as reset_peak,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.torch.cuda.max_memory_allocated",
                side_effect=[1_000_000, 2_000_000],
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.build_policy_cases",
                return_value=policy_cases,
            ) as build_cases,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.evaluate_b4_cases",
                return_value=b4_result,
            ) as evaluate_b4,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.smoke_run.evaluate_m1_cases",
                return_value=m1_result,
            ) as evaluate_m1,
        ):
            result = run_m5_smoke(
                population,
                device,
                full_series_counts={"m5": 29_059, "favorita": 55_561},
            )

        build_split.assert_called_once()
        split_kwargs = build_split.call_args.kwargs
        self.assertEqual(split_kwargs["train_origin_stride"], 7)
        np.testing.assert_array_equal(
            split_kwargs["forecast_origins"], np.array([1745, 1773])
        )
        self.assertEqual(
            [call.args[0] for call in train.call_args_list],
            ["M0PM_point_mse_param_matched", "M1_factorized_mean"],
        )
        self.assertTrue(all(call.args[1] is split for call in train.call_args_list))
        self.assertTrue(all(call.args[3] == 0 for call in train.call_args_list))
        self.assertEqual(predict_call.call_count, 4)
        self.assertEqual(
            [int(call.args[1].origins[0]) for call in predict_call.call_args_list],
            [1745, 1773, 1745, 1773],
        )
        self.assertEqual(reset_peak.call_count, 2)
        self.assertTrue(all(call.args == (device,) for call in reset_peak.call_args_list))
        self.assertGreaterEqual(synchronize.call_count, 12)
        build_cases.assert_called_once()
        evaluate_b4.assert_called_once()
        evaluate_m1.assert_called_once()

        predictions = result["predictions"]
        losses = result["losses"]
        report = result["report"]
        self.assertEqual(report["device"], "cuda:0")
        self.assertEqual(len(predictions), 200 * 2 * 28)
        self.assertEqual(len(losses), 200 * 2)
        self.assertFalse(
            predictions.duplicated(
                ["dataset_id", "series_id", "origin", "step"]
            ).any()
        )
        np.testing.assert_allclose(
            predictions["hurdle_mean_prediction"],
            predictions["hurdle_p_prediction"]
            * predictions["hurdle_mu_prediction"],
        )

        retrieval = report["retrieval_check"]
        self.assertEqual(retrieval["memory_origins"], [1745])
        self.assertEqual(retrieval["query_origin"], 1773)
        self.assertEqual(retrieval["query_count"], 200)
        self.assertEqual(retrieval["queries_with_same_series_neighbor"], 0)
        self.assertEqual(retrieval["min_neighbor_count"], 32)
        self.assertEqual(retrieval["max_neighbor_count"], 32)
        self.assertEqual(retrieval["b4_updated_query_count"], 200)
        self.assertEqual(retrieval["m1_updated_query_count"], 200)
        self.assertEqual(retrieval["measured_wall_seconds"], 10.0)

        self.assertEqual(report["training"]["point"]["canonical_seconds"], 8.0)
        self.assertEqual(report["training"]["hurdle"]["canonical_seconds"], 9.0)
        self.assertEqual(report["training"]["point"]["wall_seconds"], 10.0)
        self.assertEqual(report["inference"]["point"]["origin_count"], 2)
        self.assertEqual(
            report["runtime_projection_basis"]["training_seconds"],
            "canonical train_seconds",
        )
        self.assertEqual(report["cuda_peak_memory_bytes"], 2_000_000)
        self.assertIn("retrieval_runtime_projection_full_seed0", report)
        self.assertEqual(
            report["retrieval_runtime_projection_full_seed0"][
                "retrieval_passes_per_dataset"
            ],
            7,
        )
        self.assertGreater(report["serialization"]["actual_parquet_bytes"], 0)
        self.assertTrue(report["runtime_gate"]["exceeded"])
        self.assertEqual(report["runtime_gate"]["action"], "STOP_FOR_APPROVAL")
        self.assertEqual(
            set(report["runtime_projection_2000_per_dataset"]["by_dataset"]),
            {"m5", "favorita"},
        )
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
