"""Regression tests for the Gate-0 heterogeneous-expert diagnostic."""

from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from experiments.external_validity_screen.classical_benchmark import (
    croston_forecast,
    hand_check,
    select_alpha,
)
from experiments.om_factorization_killtest.evaluate import select_tsb
from experiments.om_factorization_killtest.models import tsb_forecast
from experiments.om_factorization_killtest.prereg import TSB_GRID
from experiments.ph_online_memory_gono_v1.data import (
    build_external_split,
    dataset_config,
)
from experiments.ph_online_memory_gono_v1.heterogeneous import (
    ALPHA_GRID,
    EXPERT_ORDER,
    PAIR_ORDER,
    assemble_heterogeneous_diagnostic,
    classical_expert_predictions,
    evaluate_heterogeneous_dataset,
    evaluate_pairwise_family,
    make_heterogeneous_diagnostic_factory,
    run_heterogeneous_diagnostic,
)
from experiments.ph_online_memory_gono_v1.metrics import policy_scale_squared


SCHEDULES = {
    "m5": (1745, (1773, 1801, 1829, 1857, 1885, 1913)),
    "favorita": (1492, (1520, 1548, 1576, 1604, 1632, 1660)),
}


def _population(dataset: str) -> dict[str, object]:
    cfg = dataset_config(dataset, n_series=2)
    time = np.arange(cfg.length)
    values = np.stack(
        [
            np.where(time % 7 == 0, 1.0 + time % 3, 0.0),
            np.where(time % 5 == 0, 2.0, 0.0),
        ]
    ).astype(np.float32)
    return {
        "data": {
            "name": dataset,
            "series_id": np.asarray([f"{dataset}-a", f"{dataset}-b"]),
            "y": values,
            "z": (values > 0.0).astype(np.float32),
            "available_from": np.zeros(2, dtype=np.int32),
        },
        "cfg": cfg,
        "manifest": {"dataset": dataset, "eligible_independent": 2},
    }


def _schedule(dataset: str) -> dict[str, object]:
    warmup, evaluations = SCHEDULES[dataset]
    return {
        "warmup_origin": warmup,
        "evaluation_origins": list(evaluations),
        "all_forecast_origins": [warmup, *evaluations],
        "horizon": 28,
        "lookback": 96,
        "train_origin_stride": 7,
        "model_train_end": 1717 if dataset == "m5" else 1464,
    }


def _dataset_output(population: dict[str, object]) -> dict[str, object]:
    data = population["data"]
    cfg = population["cfg"]
    dataset = str(data["name"])
    schedule = _schedule(dataset)
    split = build_external_split(
        data,
        cfg,
        train_origin_stride=7,
        forecast_origins=np.asarray(schedule["all_forecast_origins"]),
    )
    target = split.test.target.reshape(2, 7, 28)
    mask = split.test.target_mask.reshape(2, 7, 28)
    history = split.test.history.reshape(2, 7, 96)
    canonical_scale = split.test.scale.reshape(2, 7)
    rows: list[dict[str, object]] = []
    for series_index, series_id in enumerate(data["series_id"]):
        scale = policy_scale_squared(data["y"][series_index], int(cfg.train_end))
        for origin_index, origin in enumerate(schedule["all_forecast_origins"]):
            rows.append(
                {
                    "dataset_id": dataset,
                    "series_id": str(series_id),
                    "origin": int(origin),
                    "history": history[series_index, origin_index].copy(),
                    "canonical_train_scale": float(
                        canonical_scale[series_index, origin_index]
                    ),
                    "point_forecast": np.zeros(28, dtype=np.float64),
                    "hurdle_forecast": np.full(28, 2.0, dtype=np.float64),
                    "target": target[series_index, origin_index].copy(),
                    "target_mask": mask[series_index, origin_index].copy(),
                    "policy_scale_squared": float(scale),
                }
            )
    return {
        "dataset": dataset,
        "model_seed": 0,
        "schedule": schedule,
        "cases": pd.DataFrame(rows),
    }


def _failed_ladders() -> dict[str, object]:
    return {
        dataset: {
            "families": {
                "convex": {
                    "global_static_loss": 1.0,
                    "origin_oracle_loss": 1.0,
                }
            }
        }
        for dataset in SCHEDULES
    }


def _passing_ladders() -> dict[str, object]:
    return {
        dataset: {
            "families": {
                "convex": {
                    "global_static_loss": 1.0,
                    "origin_oracle_loss": 0.98,
                }
            }
        }
        for dataset in SCHEDULES
    }


class PairwiseFamilyTests(unittest.TestCase):
    def test_all_pairs_and_frozen_grid_use_first_grid_candidate_on_ties(self):
        targets = np.stack([np.full(28, 1.0), np.full(28, 5.0)])
        scales = np.ones(2)
        forecasts = {
            "point": np.full((2, 28), 0.0),
            "hurdle": np.full((2, 28), 2.0),
            "tsb": np.full((2, 28), 4.0),
            "sba": np.full((2, 28), 6.0),
        }

        result = evaluate_pairwise_family(targets, scales, forecasts)

        self.assertEqual(EXPERT_ORDER, ("point", "hurdle", "tsb", "sba"))
        self.assertEqual(len(PAIR_ORDER), 6)
        self.assertEqual(ALPHA_GRID, tuple(index / 20.0 for index in range(21)))
        self.assertEqual(result["candidate_count"], 126)
        self.assertEqual(
            result["global_static"]["candidate"],
            {"expert_a": "point", "expert_b": "tsb", "alpha": 0.75},
        )
        self.assertAlmostEqual(result["global_static"]["loss"], 4.0)
        self.assertAlmostEqual(result["origin_oracle"]["loss"], 0.0)
        self.assertEqual(
            result["origin_oracle"]["selection_counts"]["point|hurdle|0.5"],
            1,
        )
        self.assertEqual(
            result["origin_oracle"]["selection_counts"]["hurdle|sba|0.75"],
            1,
        )


class ClassicalReuseTests(unittest.TestCase):
    def test_actual_repository_selectors_and_forecasters_are_reused(self):
        population = _population("m5")
        schedule = _schedule("m5")
        result = classical_expert_predictions(population, schedule)
        split = build_external_split(
            population["data"],
            population["cfg"],
            train_origin_stride=7,
            forecast_origins=np.asarray(schedule["all_forecast_origins"]),
        )

        tsb_alpha, tsb_beta, tsb_mse = select_tsb(split.validation)
        sba_alpha, sba_mse = select_alpha(split.validation, 28, "sba")
        tsb_p, tsb_mu = tsb_forecast(
            split.test.history.astype(np.float64), 28, tsb_alpha, tsb_beta
        )
        expected_sba = croston_forecast(
            split.test.history.astype(np.float64), 28, sba_alpha, "sba"
        )

        self.assertEqual(result["selected_parameters"]["tsb"]["alpha"], tsb_alpha)
        self.assertEqual(result["selected_parameters"]["tsb"]["beta"], tsb_beta)
        self.assertEqual(
            result["selected_parameters"]["tsb"]["validation_mse"], tsb_mse
        )
        self.assertEqual(result["selected_parameters"]["sba"]["alpha"], sba_alpha)
        self.assertEqual(
            result["selected_parameters"]["sba"]["validation_mse"], sba_mse
        )
        self.assertEqual(
            result["selected_parameters"]["tsb"]["grid"],
            {key: list(value) for key, value in TSB_GRID.items()},
        )
        np.testing.assert_array_equal(result["forecasts"]["tsb"], tsb_p * tsb_mu)
        np.testing.assert_array_equal(result["forecasts"]["sba"], expected_sba)
        self.assertEqual(result["hand_check"], hand_check())
        self.assertTrue(result["hand_check"]["passed"])
        self.assertEqual(result["forecasts"]["tsb"].shape, (14, 28))


class HeterogeneousDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.populations = {dataset: _population(dataset) for dataset in SCHEDULES}
        cls.outputs = {
            dataset: _dataset_output(cls.populations[dataset])
            for dataset in SCHEDULES
        }

    def test_two_dataset_report_is_json_safe_pairwise_not_a_full_simplex(self):
        result = run_heterogeneous_diagnostic(self.populations, self.outputs)

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(set(result["gains_percent"]), set(SCHEDULES))
        self.assertAlmostEqual(
            result["macro_gain_percent"],
            float(np.mean(list(result["gains_percent"].values()))),
        )
        self.assertEqual(result["family"]["geometry"], "pairwise_segment_union")
        self.assertFalse(result["family"]["is_full_simplex"])
        self.assertEqual(result["family"]["candidate_count"], 126)
        self.assertEqual(result["provenance"]["new_models_trained"], False)
        for dataset in SCHEDULES:
            item = result["datasets"][dataset]
            self.assertEqual(item["n_evaluation_origins"], 6)
            self.assertEqual(item["n_evaluation_cases"], 12)
            self.assertEqual(
                item["selected_parameters"]["tsb"]["selected_on"],
                "validation_split_only",
            )
            self.assertEqual(
                item["selected_parameters"]["sba"]["selected_on"],
                "validation_split_only",
            )
        json.dumps(result, allow_nan=False)

    def test_isolated_dataset_evaluation_assembles_to_identical_report(self):
        isolated = {
            dataset: evaluate_heterogeneous_dataset(
                self.populations[dataset], self.outputs[dataset], dataset
            )
            for dataset in SCHEDULES
        }

        self.assertEqual(
            assemble_heterogeneous_diagnostic(isolated),
            run_heterogeneous_diagnostic(self.populations, self.outputs),
        )
        crossed = dict(isolated)
        crossed["m5"] = {**crossed["m5"], "dataset": "favorita"}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            assemble_heterogeneous_diagnostic(crossed)

    def test_factory_refuses_to_run_before_point_hurdle_gate0_failure(self):
        factory = make_heterogeneous_diagnostic_factory(self.populations)
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.heterogeneous."
            "run_heterogeneous_diagnostic"
        ) as diagnostic:
            with self.assertRaisesRegex(ValueError, "Gate 0.*failed"):
                factory(_passing_ladders(), self.outputs)
        diagnostic.assert_not_called()

        result = factory(_failed_ladders(), self.outputs)
        self.assertEqual(result["status"], "AVAILABLE")

    def test_rejects_duplicate_partial_mask_cross_dataset_and_changed_target(self):
        duplicate = {
            key: {**value, "cases": value["cases"].copy()}
            for key, value in self.outputs.items()
        }
        duplicate["m5"]["cases"] = pd.concat(
            [duplicate["m5"]["cases"], duplicate["m5"]["cases"].iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            run_heterogeneous_diagnostic(self.populations, duplicate)

        partial = {
            key: {**value, "cases": value["cases"].copy(deep=True)}
            for key, value in self.outputs.items()
        }
        mask = partial["m5"]["cases"].at[0, "target_mask"].copy()
        mask[0] = False
        partial["m5"]["cases"].at[0, "target_mask"] = mask
        with self.assertRaisesRegex(ValueError, "mask"):
            run_heterogeneous_diagnostic(self.populations, partial)

        crossed = {
            key: {**value, "cases": value["cases"].copy(deep=True)}
            for key, value in self.outputs.items()
        }
        crossed["m5"]["cases"].loc[0, "dataset_id"] = "favorita"
        with self.assertRaisesRegex(ValueError, "dataset"):
            run_heterogeneous_diagnostic(self.populations, crossed)

        changed = {
            key: {**value, "cases": value["cases"].copy(deep=True)}
            for key, value in self.outputs.items()
        }
        changed_target = changed["m5"]["cases"].at[0, "target"].copy()
        changed_target[0] += 1.0
        changed["m5"]["cases"].at[0, "target"] = changed_target
        with self.assertRaisesRegex(ValueError, "target"):
            run_heterogeneous_diagnostic(self.populations, changed)

    def test_rejects_wrong_horizon_and_missing_evaluation_origin(self):
        wrong_horizon = {
            key: {**value, "cases": value["cases"].copy(deep=True)}
            for key, value in self.outputs.items()
        }
        wrong_horizon["m5"]["cases"].at[0, "point_forecast"] = np.zeros(27)
        with self.assertRaisesRegex(ValueError, "28"):
            run_heterogeneous_diagnostic(self.populations, wrong_horizon)

        missing = {
            key: {**value, "cases": value["cases"].copy(deep=True)}
            for key, value in self.outputs.items()
        }
        origin = SCHEDULES["m5"][1][-1]
        missing["m5"]["cases"] = missing["m5"]["cases"].loc[
            missing["m5"]["cases"]["origin"] != origin
        ]
        with self.assertRaisesRegex(ValueError, "one-to-one|coverage|origin"):
            run_heterogeneous_diagnostic(self.populations, missing)


if __name__ == "__main__":
    unittest.main()
