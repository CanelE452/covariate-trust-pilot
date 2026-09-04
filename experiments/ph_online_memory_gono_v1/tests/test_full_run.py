"""Mock-driven tests for the no-write full seed execution engine."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from experiments.ph_online_memory_gono_v1 import full_run as full_run_module
from experiments.ph_online_memory_gono_v1.full_run import (
    analyze_seed0,
    run_full_protocol,
    run_full_seed0,
    train_dataset_experts,
)


SCHEDULES = {
    "m5": (1745, (1773, 1801, 1829, 1857, 1885, 1913), 1941, 1717),
    "favorita": (1492, (1520, 1548, 1576, 1604, 1632, 1660), 1688, 1464),
}


class _Model:
    def __init__(self, value: float):
        self.value = value

    def state_dict(self):
        return {"weight": torch.tensor([self.value])}


def _population(dataset: str = "m5") -> dict:
    warmup, _, length, train_end = SCHEDULES[dataset]
    del warmup
    n_series = 2
    return {
        "data": {
            "name": dataset,
            "series_id": np.array([f"{dataset}-a", f"{dataset}-b"]),
            "y": np.ones((n_series, length), dtype=np.float32),
            "z": np.ones((n_series, length), dtype=np.float32),
            "available_from": np.zeros(n_series, dtype=np.int32),
        },
        "descriptors": pd.DataFrame({"series_id": ["a", "b"]}),
        "cfg": SimpleNamespace(
            n_series=n_series,
            length=length,
            lookback=96,
            horizon=28,
            train_end=train_end,
            val_end=SCHEDULES[dataset][0],
        ),
        "manifest": {"dataset": dataset, "eligible_independent": n_series},
    }


def _dataset_output(dataset: str) -> dict:
    warmup, evaluations, _, train_end = SCHEDULES[dataset]
    all_origins = (warmup, *evaluations)
    step_rows = []
    loss_rows = []
    case_rows = []
    for origin in all_origins:
        step_rows.append(
            {
                "dataset_id": dataset,
                "series_id": f"{dataset}-s",
                "origin": origin,
                "step": 0,
                "y_observed": 0.0,
                "point_mean_prediction": np.sqrt(10.0),
                "hurdle_mean_prediction": np.sqrt(10.0),
                "target_mask": True,
                "policy_scale_squared": 1.0,
            }
        )
        loss_rows.append(
            {
                "dataset_id": dataset,
                "series_id": f"{dataset}-s",
                "origin": origin,
                "point_normalized_loss": 10.0,
                "hurdle_normalized_loss": 11.0,
            }
        )
        case_rows.append(
            {
                "dataset_id": dataset,
                "series_id": f"{dataset}-s",
                "origin": origin,
                "history": np.ones(96),
                "canonical_train_scale": 1.0,
                "point_forecast": np.full(28, np.sqrt(10.0)),
                "hurdle_forecast": np.full(28, np.sqrt(10.0)),
                "target": np.zeros(28),
                "target_mask": np.ones(28, dtype=bool),
                "policy_scale_squared": 1.0,
                "point_normalized_loss": 10.0,
                "hurdle_normalized_loss": 11.0,
            }
        )
    return {
        "dataset": dataset,
        "model_seed": 0,
        "schedule": {
            "warmup_origin": warmup,
            "evaluation_origins": list(evaluations),
            "all_forecast_origins": list(all_origins),
            "horizon": 28,
            "lookback": 96,
            "model_train_end": train_end,
        },
        "step_predictions": pd.DataFrame(step_rows),
        "losses": pd.DataFrame(loss_rows),
        "cases": pd.DataFrame(case_rows),
    }


def _policy_result(cases: pd.DataFrame, column: str, loss: float) -> pd.DataFrame:
    return cases.loc[:, ["dataset_id", "series_id", "origin"]].iloc[1:].assign(
        **{column: loss}
    )


def _completed_seed_result(seed: int, next_action: str) -> dict:
    rows = []
    for dataset in ("m5", "favorita"):
        for origin in SCHEDULES[dataset][1]:
            rows.append(
                {
                    "dataset_id": dataset,
                    "series_id": f"{dataset}-s",
                    "origin": origin,
                    "m1_normalized_loss": 8.0 + seed * 0.01,
                    "b4_normalized_loss": 9.0,
                    "b3_normalized_loss": 10.0,
                }
            )
    return {
        "dataset_outputs": {dataset: {"seed": seed} for dataset in SCHEDULES},
        "report": {
            "model_seed": seed,
            "terminal": next_action == "STOP",
            "next_action": next_action,
            "final_verdict": "RETRIEVAL_MEMORY_GO" if next_action == "STOP" else None,
        },
        "tables": {"target_policy_losses": pd.DataFrame(rows)},
    }


class DatasetExecutionTests(unittest.TestCase):
    def test_exact_seven_origins_two_sequential_canonical_arms_and_cpu_states(self):
        population = _population("m5")
        origins = np.array([1745, 1773, 1801, 1829, 1857, 1885, 1913])
        split = SimpleNamespace(
            test=SimpleNamespace(origins=origins, n_series=2, n_origins=7)
        )
        point_prediction = {"mean_prediction": np.ones((14, 28))}
        hurdle_prediction = {
            "mean_prediction": np.ones((14, 28)),
            "p_prediction": np.full((14, 28), 0.5),
            "mu_prediction": np.full((14, 28), 2.0),
        }
        trained = [
            {
                "model": _Model(1.0),
                "predictions": point_prediction,
                "best_epoch": 3,
                "best_validation_mean_mse": 1.2,
                "train_seconds": 4.0,
                "n_parameters": 7056,
            },
            {
                "model": _Model(2.0),
                "predictions": hurdle_prediction,
                "best_epoch": 4,
                "best_validation_mean_mse": 1.1,
                "train_seconds": 5.0,
                "n_parameters": 7056,
            },
        ]
        predictions = pd.DataFrame(
            {
                "dataset_id": ["m5"] * 7,
                "series_id": ["m5-a"] * 7,
                "origin": origins,
                "step": [0] * 7,
            }
        )
        losses = predictions.loc[:, ["dataset_id", "series_id", "origin"]].assign(
            point_normalized_loss=1.0, hurdle_normalized_loss=2.0
        )
        cases = losses.assign(policy_scale_squared=1.0)

        with (
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_external_split",
                return_value=split,
            ) as build_split,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.train_one_on_split",
                side_effect=trained,
            ) as train,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_prediction_frame",
                return_value=predictions,
            ) as build_predictions,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.normalized_loss_frame",
                return_value=losses,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_policy_cases",
                return_value=cases,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.torch.cuda.synchronize"
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.torch.cuda.empty_cache"
            ) as empty_cache,
        ):
            result = train_dataset_experts(
                population, torch.device("cuda:0"), model_seed=0
            )

        np.testing.assert_array_equal(
            build_split.call_args.kwargs["forecast_origins"], origins
        )
        self.assertEqual(build_split.call_args.kwargs["train_origin_stride"], 7)
        self.assertEqual(
            [call.args[0] for call in train.call_args_list],
            ["M0PM_point_mse_param_matched", "M1_factorized_mean"],
        )
        self.assertTrue(all(call.args[1] is split for call in train.call_args_list))
        self.assertEqual([call.args[3] for call in train.call_args_list], [0, 0])
        build_predictions.assert_called_once()
        prediction_args = build_predictions.call_args.args
        self.assertIs(prediction_args[0], population["data"])
        self.assertIs(prediction_args[1], split.test)
        self.assertEqual(set(prediction_args[2]), set(point_prediction))
        self.assertEqual(set(prediction_args[3]), set(hurdle_prediction))
        np.testing.assert_array_equal(
            prediction_args[2]["mean_prediction"],
            point_prediction["mean_prediction"],
        )
        np.testing.assert_array_equal(
            prediction_args[3]["mean_prediction"],
            hurdle_prediction["mean_prediction"],
        )
        self.assertEqual(empty_cache.call_count, 2)
        self.assertEqual(result["schedule"]["warmup_origin"], 1745)
        self.assertEqual(result["schedule"]["evaluation_origins"], origins[1:].tolist())
        self.assertEqual(set(result["state_dicts"]), {"point", "hurdle"})
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for state in result["state_dicts"].values()
                for tensor in state.values()
            )
        )
        json.dumps(result["provenance"], allow_nan=False)

    def test_verified_arm_payloads_skip_both_refits(self):
        population = _population("m5")
        origins = np.array([1745, 1773, 1801, 1829, 1857, 1885, 1913])
        split = SimpleNamespace(
            test=SimpleNamespace(origins=origins, n_series=2, n_origins=7)
        )
        point_prediction = {"mean_prediction": np.ones((14, 28))}
        hurdle_prediction = {
            "mean_prediction": np.ones((14, 28)),
            "p_prediction": np.full((14, 28), 0.5),
            "mu_prediction": np.full((14, 28), 2.0),
        }

        def arm_payload(arm, model_id, predictions):
            return {
                "dataset": "m5",
                "arm": arm,
                "model_id": model_id,
                "model_seed": 0,
                "state_dict": {"weight": torch.tensor([1.0])},
                "provenance": {
                    "model_id": model_id,
                    "model_seed": 0,
                    "checkpoint_device": "cpu",
                    "n_parameters": 7056,
                    "execution_device_type": "cpu",
                    "end_to_end_wall_seconds": 1.0,
                },
                "predictions": predictions,
            }

        persisted = {
            "point": arm_payload(
                "point", "M0PM_point_mse_param_matched", point_prediction
            ),
            "hurdle": arm_payload(
                "hurdle", "M1_factorized_mean", hurdle_prediction
            ),
        }
        predictions = pd.DataFrame(
            {
                "dataset_id": ["m5"] * 7,
                "series_id": ["m5-a"] * 7,
                "origin": origins,
                "step": [0] * 7,
            }
        )
        losses = predictions.loc[:, ["dataset_id", "series_id", "origin"]].assign(
            point_normalized_loss=1.0, hurdle_normalized_loss=2.0
        )
        cases = losses.assign(policy_scale_squared=1.0)
        callback = mock.Mock()
        with (
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_external_split",
                return_value=split,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.train_one_on_split",
                side_effect=AssertionError("verified arms must not be refit"),
            ) as train,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_prediction_frame",
                return_value=predictions,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.normalized_loss_frame",
                return_value=losses,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_policy_cases",
                return_value=cases,
            ),
        ):
            result = train_dataset_experts(
                population,
                torch.device("cpu"),
                model_seed=0,
                persisted_arms=persisted,
                on_arm_complete=callback,
            )
        train.assert_not_called()
        callback.assert_not_called()
        self.assertEqual(set(result["state_dicts"]), {"point", "hurdle"})

    def test_full_wrapper_fits_datasets_in_frozen_order_then_analyzes(self):
        populations = {"favorita": object(), "m5": object()}
        fitted = {"m5": {"dataset": "m5"}, "favorita": {"dataset": "favorita"}}
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.train_dataset_experts",
            side_effect=lambda population, device, model_seed: (
                fitted["m5"] if population is populations["m5"] else fitted["favorita"]
            ),
        ) as train, mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.analyze_seed0",
            return_value={"report": {"final_verdict": "FULL_NO_GO"}},
        ) as analyze:
            result = run_full_seed0(populations, torch.device("cpu"))

        self.assertEqual(
            [call.args[0] for call in train.call_args_list],
            [populations["m5"], populations["favorita"]],
        )
        analyze.assert_called_once()
        self.assertIs(analyze.call_args.args[0]["m5"], fitted["m5"])
        self.assertIs(analyze.call_args.args[0]["favorita"], fitted["favorita"])
        self.assertIn("dataset_outputs", result)


class GateOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.outputs = {
            "m5": _dataset_output("m5"),
            "favorita": _dataset_output("favorita"),
        }

    def test_gate0_failure_is_terminal_and_skips_every_later_stage(self):
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.oracle_ladder",
            return_value={"m5": {}, "favorita": {}},
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate0",
            return_value={
                "passed": False,
                "final_verdict": "FULL_NO_GO",
                "heterogeneous_diagnostic": {"status": "NOT_AVAILABLE", "passed": False},
            },
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.temporal_recurrence"
        ) as recurrence, mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.tune_b3_source"
        ) as b3:
            result = analyze_seed0(self.outputs, bootstrap_draws=3)

        recurrence.assert_not_called()
        b3.assert_not_called()
        self.assertEqual(result["report"]["final_verdict"], "FULL_NO_GO")
        self.assertEqual(result["report"]["stopped_after"], "GATE0")
        json.dumps(result["report"], allow_nan=False)

    def test_gate1a_failure_skips_policy_tuning(self):
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.oracle_ladder",
            return_value={"m5": {}, "favorita": {}},
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate0",
            return_value={"passed": True},
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.temporal_recurrence",
            return_value={"passed": False, "failure_verdict": "TEMPORAL_RECURRENCE_NO_GO"},
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.tune_b3_source"
        ) as b3:
            result = analyze_seed0(self.outputs, bootstrap_draws=3)

        b3.assert_not_called()
        self.assertEqual(
            result["report"]["final_verdict"], "TEMPORAL_RECURRENCE_NO_GO"
        )
        self.assertEqual(result["report"]["stopped_after"], "GATE1A")

    def test_gate1b_failure_skips_retrieval_tuning_and_scoring(self):
        def tune_b3(steps, **_kwargs):
            return {"alpha": 0.5, "candidates": pd.DataFrame({"alpha": [0.5]})}

        def tune_b4(cases, **_kwargs):
            return {"eta": 2.0, "half_life": 1, "candidates": pd.DataFrame({"eta": [2.0]})}

        def evaluate_b4(cases, **_kwargs):
            return _policy_result(cases, "b4_normalized_loss", 9.0)

        with (
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.oracle_ladder",
                return_value={"m5": {}, "favorita": {}},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate0",
                return_value={"passed": True},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.temporal_recurrence",
                return_value={"passed": True},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_b3_source",
                side_effect=tune_b3,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_b4_source",
                side_effect=tune_b4,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_b4_cases",
                side_effect=evaluate_b4,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate1b",
                return_value={"passed": False},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_m1_source"
            ) as tune_m1,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_m1_cases"
            ) as evaluate_m1,
        ):
            result = analyze_seed0(self.outputs, bootstrap_draws=3)

        tune_m1.assert_not_called()
        evaluate_m1.assert_not_called()
        self.assertEqual(result["report"]["stopped_after"], "GATE1B")
        self.assertEqual(result["report"]["final_verdict"], "ONLINE_MEMORY_NO_GO")

    def test_source_selected_values_are_applied_only_to_opposite_target(self):
        b3_params = {"m5": 0.2, "favorita": 0.8}
        b4_params = {"m5": (2.0, 1), "favorita": (8.0, 3)}

        def tune_b3(steps, **_kwargs):
            source = str(steps["dataset_id"].iloc[0])
            return {"alpha": b3_params[source], "candidates": pd.DataFrame({"alpha": [b3_params[source]]})}

        def tune_b4(cases, **_kwargs):
            source = str(cases["dataset_id"].iloc[0])
            eta, half_life = b4_params[source]
            return {"eta": eta, "half_life": half_life, "candidates": pd.DataFrame({"eta": [eta]})}

        def evaluate_b4(cases, **kwargs):
            target = str(cases["dataset_id"].iloc[0])
            source = "favorita" if target == "m5" else "m5"
            self.assertEqual((kwargs["eta"], kwargs["half_life"]), b4_params[source])
            return _policy_result(cases, "b4_normalized_loss", 9.0)

        def tune_m1(cases, **kwargs):
            source = str(cases["dataset_id"].iloc[0])
            self.assertEqual((kwargs["eta"], kwargs["half_life"]), b4_params[source])
            return {"k": 32, "lambda_max": 0.25, "candidates": pd.DataFrame({"k": [32]})}

        def evaluate_m1(cases, **kwargs):
            target = str(cases["dataset_id"].iloc[0])
            source = "favorita" if target == "m5" else "m5"
            self.assertEqual((kwargs["eta"], kwargs["half_life"]), b4_params[source])
            self.assertEqual((kwargs["k"], kwargs["lambda_max"]), (32, 0.25))
            self.assertEqual(kwargs["neighbor_plan"], f"plan-{target}")
            return _policy_result(cases, "m1_normalized_loss", 8.0)

        def evaluate_c0(cases, **kwargs):
            target = str(cases["dataset_id"].iloc[0])
            self.assertEqual(kwargs["neighbor_plan"], f"plan-{target}")
            return _policy_result(cases, "m1_normalized_loss", 9.5)

        def evaluate_c1(cases, **kwargs):
            target = str(cases["dataset_id"].iloc[0])
            self.assertEqual(kwargs["neighbor_plan"], f"plan-{target}")
            return _policy_result(cases, "m1_normalized_loss", 9.5)

        with (
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.oracle_ladder",
                return_value={"m5": {}, "favorita": {}},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate0",
                return_value={"passed": True},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.temporal_recurrence",
                return_value={"passed": True},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_b3_source",
                side_effect=tune_b3,
            ) as b3_tuner,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_b4_source",
                side_effect=tune_b4,
            ) as b4_tuner,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_b4_cases",
                side_effect=evaluate_b4,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_m1_source",
                side_effect=tune_m1,
            ) as m1_tuner,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_m1_neighbor_plan",
                side_effect=lambda cases, **_kwargs: (
                    "plan-" + str(cases["dataset_id"].iloc[0])
                ),
            ) as build_plan,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_m1_cases",
                side_effect=evaluate_m1,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_c0_cases",
                side_effect=evaluate_c0,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_c1_cases",
                side_effect=evaluate_c1,
            ),
        ):
            result = analyze_seed0(self.outputs, bootstrap_draws=10)

        self.assertEqual(
            [str(call.args[0]["dataset_id"].iloc[0]) for call in b3_tuner.call_args_list],
            ["m5", "favorita"],
        )
        self.assertEqual(
            [str(call.args[0]["dataset_id"].iloc[0]) for call in b4_tuner.call_args_list],
            ["m5", "favorita"],
        )
        self.assertEqual(
            [str(call.args[0]["dataset_id"].iloc[0]) for call in m1_tuner.call_args_list],
            ["m5", "favorita"],
        )
        self.assertEqual(build_plan.call_count, 2)
        report = result["report"]
        self.assertEqual(report["final_verdict"], "RETRIEVAL_MEMORY_GO")
        self.assertEqual(report["gate4"]["action"], "ACCEPT_SEED0")
        self.assertEqual(
            report["directions"]["m5_to_favorita"]["b3"]["alpha"], 0.2
        )
        self.assertEqual(
            report["directions"]["favorita_to_m5"]["b3"]["alpha"], 0.8
        )
        self.assertEqual(len(result["tables"]["target_policy_losses"]), 12)
        json.dumps(report, allow_nan=False)

    def test_additional_seed_upstream_gates_are_diagnostic_only(self):
        outputs = {
            dataset: {**output, "model_seed": 1}
            for dataset, output in self.outputs.items()
        }

        def tune_b3(_steps, **_kwargs):
            return {"alpha": 0.5, "candidates": pd.DataFrame({"alpha": [0.5]})}

        def tune_b4(_cases, **_kwargs):
            return {"eta": 2.0, "half_life": 1, "candidates": pd.DataFrame({"eta": [2.0]})}

        def evaluate_b4(cases, **_kwargs):
            return _policy_result(cases, "b4_normalized_loss", 9.0)

        def tune_m1(_cases, **_kwargs):
            return {"k": 32, "lambda_max": 0.25, "candidates": pd.DataFrame({"k": [32]})}

        def evaluated(cases, **_kwargs):
            return _policy_result(cases, "m1_normalized_loss", 8.0)

        bootstrap = {
            "directions": {
                dataset: {"ci95_percent": [1.0, 2.0]}
                for dataset in ("m5", "favorita")
            },
            "macro": {"ci95_percent": [1.0, 2.0], "ri_percent": 1.5},
        }
        heterogeneous_factory = mock.Mock(
            side_effect=AssertionError("additional seeds must not rerun Gate-0 fallback")
        )
        with (
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.oracle_ladder",
                return_value={"m5": {}, "favorita": {}},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate0",
                return_value={"passed": False, "heterogeneous_diagnostic": None},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.temporal_recurrence",
                return_value={"passed": False},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_b3_source",
                side_effect=tune_b3,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_b4_source",
                side_effect=tune_b4,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_b4_cases",
                side_effect=evaluate_b4,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate1b",
                return_value={"passed": False},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_m1_source",
                side_effect=tune_m1,
            ) as m1_tuner,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_m1_neighbor_plan",
                return_value=object(),
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_m1_cases",
                side_effect=evaluated,
            ) as m1_evaluator,
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_c0_cases",
                side_effect=evaluated,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_c1_cases",
                side_effect=evaluated,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.paired_series_cluster_bootstrap",
                return_value=bootstrap,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate2",
                return_value={"passed": False},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate3_safety",
                return_value={"passed": False},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate3_control",
                return_value={"passed": False},
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed0"
            ) as seed0_gate,
        ):
            result = full_run_module._analyze_seed(
                outputs,
                model_seed=1,
                apply_seed0_gate=False,
                heterogeneous_diagnostic_factory=heterogeneous_factory,
                bootstrap_draws=3,
                analysis_seed=20260904,
                control_seed=20260904,
            )

        self.assertEqual(m1_tuner.call_count, 2)
        self.assertEqual(m1_evaluator.call_count, 2)
        heterogeneous_factory.assert_not_called()
        seed0_gate.assert_not_called()
        self.assertEqual(result["report"]["next_action"], "SEED_POLICY_READY")
        self.assertIsNone(result["report"]["final_verdict"])
        self.assertEqual(len(result["tables"]["target_policy_losses"]), 12)

    def test_pending_gate2_is_still_terminal_when_seed0_safety_or_control_fails(self):
        def tune_b3(_steps, **_kwargs):
            return {"alpha": 0.5, "candidates": pd.DataFrame({"alpha": [0.5]})}

        def tune_b4(_cases, **_kwargs):
            return {
                "eta": 2.0,
                "half_life": 1,
                "candidates": pd.DataFrame({"eta": [2.0]}),
            }

        def policy(column, loss):
            return lambda cases, **_kwargs: _policy_result(cases, column, loss)

        pending_gate2 = {
            "passed": False,
            "m1_vs_b4_macro_percent": 0.3,
            "m1_vs_b4_percent": {"m5": 0.3, "favorita": 0.3},
            "m1_vs_b3_macro_percent": 0.2,
            "m1_vs_b3_percent": {"m5": 0.2, "favorita": 0.2},
            "macro_ci95_percent": [-0.1, 0.5],
            "direction_ci95_percent": {
                "m5": [-0.1, 0.5],
                "favorita": [-0.1, 0.5],
            },
            "checks": {
                "macro_effect": True,
                "direction_safety": True,
                "macro_absolute_usefulness": False,
                "direction_absolute_usefulness": False,
                "macro_ci": False,
                "dataset_ci": False,
            },
        }
        cases = (
            (False, True, "GATE3_SAFETY"),
            (True, False, "GATE3_CONTROL"),
        )
        for safety_pass, control_pass, veto_gate in cases:
            with self.subTest(veto_gate=veto_gate), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.oracle_ladder",
                return_value={"m5": {}, "favorita": {}},
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate0",
                return_value={"passed": True},
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.temporal_recurrence",
                return_value={"passed": True},
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_b3_source",
                side_effect=tune_b3,
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_b4_source",
                side_effect=tune_b4,
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_b4_cases",
                side_effect=policy("b4_normalized_loss", 9.0),
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.tune_m1_source",
                return_value={
                    "k": 32,
                    "lambda_max": 0.25,
                    "candidates": pd.DataFrame({"k": [32]}),
                },
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.build_m1_neighbor_plan",
                return_value=object(),
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_m1_cases",
                side_effect=policy("m1_normalized_loss", 8.0),
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_c0_cases",
                side_effect=policy("m1_normalized_loss", 9.5),
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_c1_cases",
                side_effect=policy("m1_normalized_loss", 9.5),
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate2",
                return_value=pending_gate2,
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate3_safety",
                return_value={"passed": safety_pass},
            ), mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate3_control",
                return_value={"passed": control_pass},
            ) as control_gate, mock.patch(
                "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed0"
            ) as gate4:
                result = analyze_seed0(self.outputs, bootstrap_draws=3)

            self.assertEqual(result["report"]["stopped_after"], veto_gate)
            self.assertEqual(
                result["report"]["final_verdict"],
                "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
            )
            self.assertEqual(
                result["report"]["gate2"]["decision_state"], "FINAL_FAIL"
            )
            self.assertEqual(
                result["report"]["gate2"]["deferral_withdrawn_by"], veto_gate
            )
            self.assertIn("gate3_safety", result["report"])
            if control_pass is False:
                self.assertIn("gate3_control", result["report"])
            gate4.assert_not_called()
            if not safety_pass:
                control_gate.assert_not_called()

    def test_b3_loss_requires_full_28_step_observed_horizon(self):
        cases = self.outputs["m5"]["cases"].iloc[1:2].copy()
        cases.at[cases.index[0], "target"] = np.zeros(27)
        cases.at[cases.index[0], "target_mask"] = np.ones(27, dtype=bool)
        cases.at[cases.index[0], "point_forecast"] = np.zeros(27)
        cases.at[cases.index[0], "hurdle_forecast"] = np.zeros(27)
        with self.assertRaisesRegex(ValueError, "28"):
            full_run_module._blend_loss_frame(cases, 0.5)

        cases = self.outputs["m5"]["cases"].iloc[1:2].copy()
        mask = np.ones(28, dtype=bool)
        mask[-1] = False
        cases.at[cases.index[0], "target_mask"] = mask
        with self.assertRaisesRegex(ValueError, "observed"):
            full_run_module._blend_loss_frame(cases, 0.5)

    def test_heterogeneous_diagnostic_is_lazy_and_only_runs_after_gate0_failure(self):
        factory = mock.Mock(return_value={"status": "AVAILABLE", "macro_gain_percent": 3.0})
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.oracle_ladder",
            return_value={"m5": {}, "favorita": {}},
        ) as ladder, mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate0",
            side_effect=[
                {"passed": False, "heterogeneous_diagnostic": {"status": "NOT_AVAILABLE"}},
                {
                    "passed": False,
                    "heterogeneous_diagnostic": {"status": "AVAILABLE", "passed": True},
                },
            ],
        ) as gate0:
            result = analyze_seed0(
                self.outputs,
                heterogeneous_diagnostic_factory=factory,
                bootstrap_draws=3,
            )

        factory.assert_called_once()
        self.assertIs(factory.call_args.args[0], ladder.return_value)
        self.assertIs(factory.call_args.args[1]["m5"], self.outputs["m5"])
        self.assertEqual(gate0.call_count, 2)
        self.assertEqual(
            result["report"]["final_verdict"],
            "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE",
        )


class MultiSeedProtocolTests(unittest.TestCase):
    def test_seed0_gate2_defers_any_failure_only_when_section39_is_triggered(self):
        checks = {
            "macro_effect": True,
            "direction_safety": True,
            "macro_absolute_usefulness": True,
            "direction_absolute_usefulness": True,
            "macro_ci": True,
            "dataset_ci": True,
        }
        passed = {
            "passed": True,
            "m1_vs_b4_macro_percent": 0.3,
            "m1_vs_b4_percent": {"m5": 0.3, "favorita": 0.3},
            "macro_ci95_percent": [0.1, 0.5],
            "checks": checks,
        }
        self.assertEqual(
            full_run_module._seed0_gate2_resolution(passed), "PROCEED"
        )

        for recoverable in ("macro_effect", "macro_ci"):
            borderline_checks = dict(checks)
            borderline_checks[recoverable] = False
            borderline = {
                "passed": False,
                "m1_vs_b4_macro_percent": 0.1,
                "m1_vs_b4_percent": {"m5": 0.1, "favorita": 0.1},
                "macro_ci95_percent": [-0.1, 0.3],
                "checks": borderline_checks,
            }
            self.assertEqual(
                full_run_module._seed0_gate2_resolution(borderline),
                "PENDING_GATE4",
            )

        for recoverable_under_section_39 in (
            "direction_safety",
            "macro_absolute_usefulness",
            "direction_absolute_usefulness",
            "dataset_ci",
        ):
            failed_checks = dict(checks)
            failed_checks[recoverable_under_section_39] = False
            failed = {
                "passed": False,
                "m1_vs_b4_macro_percent": 0.3,
                "m1_vs_b4_percent": {"m5": 0.3, "favorita": 0.3},
                "macro_ci95_percent": [0.1, 0.5],
                "checks": failed_checks,
            }
            self.assertEqual(
                full_run_module._seed0_gate2_resolution(failed),
                "PENDING_GATE4",
            )

        nonpositive = {
            "passed": False,
            "m1_vs_b4_macro_percent": 0.0,
            "m1_vs_b4_percent": {"m5": 0.0, "favorita": 0.0},
            "macro_ci95_percent": [-0.1, 0.1],
            "checks": {**checks, "macro_effect": False},
        }
        self.assertEqual(
            full_run_module._seed0_gate2_resolution(nonpositive), "FAIL"
        )

        no_actual_borderline_trigger = {
            "passed": False,
            "m1_vs_b4_macro_percent": 0.5,
            "m1_vs_b4_percent": {"m5": 0.5, "favorita": 0.5},
            "macro_ci95_percent": [-0.4, -0.1],
            "checks": {**checks, "macro_ci": False},
        }
        self.assertEqual(
            full_run_module._seed0_gate2_resolution(no_actual_borderline_trigger),
            "FAIL",
        )

        no_trigger_absolute_failure = {
            "passed": False,
            "m1_vs_b4_macro_percent": 0.5,
            "m1_vs_b4_percent": {"m5": 0.5, "favorita": 0.5},
            "macro_ci95_percent": [0.3, 0.7],
            "checks": {**checks, "macro_absolute_usefulness": False},
        }
        self.assertEqual(
            full_run_module._seed0_gate2_resolution(no_trigger_absolute_failure),
            "FAIL",
        )

    def test_scientific_runner_rejects_nonfrozen_bootstrap_or_control_seed(self):
        populations = {"m5": object(), "favorita": object()}
        with self.assertRaisesRegex(ValueError, "exactly 2000"):
            run_full_protocol(
                populations, torch.device("cpu"), bootstrap_draws=10
            )
        with self.assertRaisesRegex(ValueError, "control seed"):
            run_full_protocol(
                populations, torch.device("cpu"), control_seed=7
            )

    def test_seed0_terminal_never_runs_an_additional_seed(self):
        seed0 = _completed_seed_result(0, "STOP")
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.run_full_seed0",
            return_value=seed0,
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run._run_additional_seed"
        ) as additional:
            result = run_full_protocol(
                {"m5": object(), "favorita": object()}, torch.device("cpu")
            )

        additional.assert_not_called()
        self.assertEqual(result["report"]["executed_model_seeds"], [0])
        self.assertEqual(result["report"]["final_verdict"], "RETRIEVAL_MEMORY_GO")

    def test_pending_gate2_with_seed0_safety_veto_never_runs_seed1(self):
        seed0 = _completed_seed_result(0, "STOP")
        seed0["report"].update(
            {
                "final_verdict": "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
                "stopped_after": "GATE3_SAFETY",
                "gate2": {
                    "passed": False,
                    "decision_state": "FINAL_FAIL",
                    "deferral_withdrawn_by": "GATE3_SAFETY",
                },
                "gate3_safety": {"passed": False},
            }
        )
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.run_full_seed0",
            return_value=seed0,
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run._run_additional_seed"
        ) as additional:
            result = run_full_protocol(
                {"m5": object(), "favorita": object()}, torch.device("cpu")
            )

        additional.assert_not_called()
        self.assertEqual(result["report"]["executed_model_seeds"], [0])
        self.assertEqual(
            result["report"]["final_verdict"],
            "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
        )

    def test_seed1_acceptance_uses_two_seed_case_losses_and_skips_seed2(self):
        seed0 = _completed_seed_result(0, "RUN_SEED1")
        seed1 = _completed_seed_result(1, "SEED_POLICY_READY")
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.run_full_seed0",
            return_value=seed0,
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run._run_additional_seed",
            return_value=seed1,
        ) as additional, mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed1",
            return_value={"action": "ACCEPT_TWO_SEED", "passed": True},
        ) as gate4_seed1, mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed2"
        ) as gate4_seed2:
            result = run_full_protocol(
                {"m5": object(), "favorita": object()}, torch.device("cpu")
            )

        self.assertEqual([call.kwargs["model_seed"] for call in additional.call_args_list], [1])
        gate4_seed2.assert_not_called()
        stacked = gate4_seed1.call_args.args[0]
        self.assertEqual(set(stacked["model_seed"]), {0, 1})
        self.assertEqual(gate4_seed1.call_args.kwargs["candidate_column"], "m1_normalized_loss")
        self.assertEqual(gate4_seed1.call_args.kwargs["baseline_column"], "b4_normalized_loss")
        self.assertEqual(result["report"]["executed_model_seeds"], [0, 1])
        self.assertEqual(result["report"]["final_verdict"], "RETRIEVAL_MEMORY_GO")
        self.assertTrue(result["report"]["averaged_gate2"]["passed"])
        self.assertIn("seed_average_policy_losses", result["tables"])

    def test_gate4_acceptance_cannot_override_failed_averaged_gate2(self):
        seed0 = _completed_seed_result(0, "RUN_SEED1")
        seed1 = _completed_seed_result(1, "SEED_POLICY_READY")
        averaged = {
            "gate2": {"passed": False},
            "seed_average_losses": pd.DataFrame(
                {
                    "dataset_id": ["m5", "favorita"],
                    "series_id": ["m5-s", "favorita-s"],
                    "origin": [SCHEDULES["m5"][1][0], SCHEDULES["favorita"][1][0]],
                    "b3_normalized_loss": [10.0, 10.0],
                    "b4_normalized_loss": [9.0, 9.0],
                    "m1_normalized_loss": [8.0, 8.0],
                }
            ),
            "m1_vs_b4": {},
            "m1_vs_b3": {},
            "bootstrap": {},
            "aggregation": "average all policy losses by dataset/series/origin",
        }
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.run_full_seed0",
            return_value=seed0,
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run._run_additional_seed",
            return_value=seed1,
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed1",
            return_value={"action": "ACCEPT_TWO_SEED", "passed": True},
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_seed_average_gate2",
            return_value=averaged,
        ):
            result = run_full_protocol(
                {"m5": object(), "favorita": object()}, torch.device("cpu")
            )

        self.assertEqual(
            result["report"]["final_verdict"],
            "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
        )
        self.assertFalse(result["report"]["averaged_gate2"]["passed"])

    def test_seed1_clear_failure_is_terminal_and_does_not_run_seed2(self):
        seed0 = _completed_seed_result(0, "RUN_SEED1")
        seed1 = _completed_seed_result(1, "SEED_POLICY_READY")
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.run_full_seed0",
            return_value=seed0,
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run._run_additional_seed",
            return_value=seed1,
        ) as additional, mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed1",
            return_value={"action": "RETRIEVAL_ROBUSTNESS_NO_GO", "passed": False},
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed2"
        ) as gate4_seed2:
            result = run_full_protocol(
                {"m5": object(), "favorita": object()}, torch.device("cpu")
            )

        self.assertEqual([call.kwargs["model_seed"] for call in additional.call_args_list], [1])
        gate4_seed2.assert_not_called()
        self.assertEqual(result["report"]["executed_model_seeds"], [0, 1])
        self.assertEqual(
            result["report"]["final_verdict"],
            "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
        )

    def test_seed2_runs_only_for_seed1_borderline_and_returns_exact_no_go(self):
        seed0 = _completed_seed_result(0, "RUN_SEED1")
        seed1 = _completed_seed_result(1, "SEED_POLICY_READY")
        seed2 = _completed_seed_result(2, "SEED_POLICY_READY")
        with mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.run_full_seed0",
            return_value=seed0,
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run._run_additional_seed",
            side_effect=[seed1, seed2],
        ) as additional, mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed1",
            return_value={"action": "RUN_SEED2", "passed": None},
        ), mock.patch(
            "experiments.ph_online_memory_gono_v1.full_run.evaluate_gate4_seed2",
            return_value={"action": "RETRIEVAL_ROBUSTNESS_NO_GO", "passed": False},
        ) as gate4_seed2:
            result = run_full_protocol(
                {"m5": object(), "favorita": object()}, torch.device("cpu")
            )

        self.assertEqual(
            [call.kwargs["model_seed"] for call in additional.call_args_list], [1, 2]
        )
        stacked = gate4_seed2.call_args.args[0]
        self.assertEqual(set(stacked["model_seed"]), {0, 1, 2})
        self.assertEqual(result["report"]["executed_model_seeds"], [0, 1, 2])
        self.assertEqual(
            result["report"]["final_verdict"],
            "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
        )


if __name__ == "__main__":
    unittest.main()
