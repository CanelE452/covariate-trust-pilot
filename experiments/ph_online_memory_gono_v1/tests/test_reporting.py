"""Synthetic tests for the deterministic Tables A--G and status renderer."""

from __future__ import annotations

import copy
import json
import re
import unittest

import pandas as pd

from experiments.ph_online_memory_gono_v1.analysis import (
    evaluate_seed_average_gate2,
)
from experiments.ph_online_memory_gono_v1.reporting import (
    TABLE_KEYS,
    build_runtime_stop_status,
    build_status_markdown,
    build_tables_a_to_g,
)


DATASETS = ("m5", "favorita")
EXACT_VERDICTS = (
    "FULL_NO_GO",
    "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE",
    "TEMPORAL_RECURRENCE_NO_GO",
    "ONLINE_MEMORY_NO_GO",
    "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
    "RETRIEVAL_UNSAFE_NO_GO",
    "RETRIEVAL_SIGNAL_NOT_IDENTIFIED",
    "RETRIEVAL_MEMORY_GO",
)
ORIGINS = {
    "m5": [1773, 1801, 1829, 1857, 1885, 1913],
    "favorita": [1520, 1548, 1576, 1604, 1632, 1660],
}
SCHEDULES = {
    "m5": (1717, 1745),
    "favorita": (1464, 1492),
}
FROZEN_PREREGISTRATION = {
    "experiment_name": "PH-ONLINE-MEMORY-GONO-v1",
    "preregistration_sha256": "a" * 64,
    "frozen_at_utc": "2026-09-04T00:00:00Z",
    "frozen_before_any_new_model_fit": True,
}


def _oracle(dataset: str) -> dict:
    point = 10.0 if dataset == "m5" else 20.0
    return {
        "dataset_id": dataset,
        "always_point_loss": point,
        "always_hurdle_loss": point - 0.4,
        "half_half_loss": point - 0.6,
        "families": {
            "hard": {
                "global_oracle_loss": point - 0.4,
                "global_oracle_expert": "hurdle",
                "series_oracle_loss": point - 0.8,
                "origin_oracle_loss": point - 1.0,
            },
            "convex": {
                "global_static_loss": point - 0.7,
                "global_static_alpha": 0.55,
                "series_oracle_loss": point - 0.9,
                "origin_oracle_loss": point - 1.1,
            },
        },
    }


def _dataset_output(dataset: str, *, seed: int = 0) -> dict:
    train_end, warmup = SCHEDULES[dataset]
    return {
        "dataset": dataset,
        "model_seed": seed,
        "schedule": {
            "model_train_end": train_end,
            "warmup_origin": warmup,
            "evaluation_origins": ORIGINS[dataset],
            "horizon": 28,
            "lookback": 96,
        },
        "population_manifest": {
            "dataset": dataset,
            "eligible_independent": 101 if dataset == "m5" else 202,
        },
        "provenance": {
            "point": {
                "model_id": "M0PM_point_mse_param_matched",
                "n_parameters": 7056,
                "execution_device_type": "cuda",
                "train_seconds": 11.0 + seed,
                "end_to_end_wall_seconds": 12.0 + seed,
            },
            "hurdle": {
                "model_id": "M1_factorized_mean",
                "n_parameters": 7056,
                "execution_device_type": "cuda",
                "train_seconds": 13.0 + seed,
                "end_to_end_wall_seconds": 14.0 + seed,
            },
        },
    }


def _policy_losses() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for origin in ORIGINS[dataset]:
            rows.append(
                {
                    "dataset_id": dataset,
                    "series_id": f"{dataset}-s",
                    "origin": origin,
                    "b3_normalized_loss": 10.0,
                    "b4_normalized_loss": 9.0,
                    "m1_normalized_loss": 8.0,
                }
            )
    return pd.DataFrame(rows)


def _seed0_result(*, stopped_after: str = "GATE4_SEED0") -> dict:
    full = stopped_after == "GATE4_SEED0"
    report = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "FULL_SEED0",
        "model_seed": 0,
        "stopped_after": stopped_after,
        "terminal": True,
        "final_verdict": "RETRIEVAL_MEMORY_GO" if full else "FULL_NO_GO",
        "oracle_ladder": {dataset: _oracle(dataset) for dataset in DATASETS},
        "gate0": {
            "gains_percent": {"m5": 4.3, "favorita": 2.2},
            "macro_gain_percent": 3.25,
            "passed": full,
            "heterogeneous_diagnostic": None,
        },
        "directions": {},
    }
    tables: dict[str, pd.DataFrame] = {}
    if full:
        report.update(
            {
                "gate1a": {
                    "datasets": {
                        dataset: {
                            "real_rho": 0.2,
                            "rho_ci95": [0.1, 0.3],
                            "shuffled_rho": 0.01,
                            "real_minus_shuffled_rho": 0.19,
                            "passed": True,
                        }
                        for dataset in DATASETS
                    },
                    "passed": True,
                },
                "directions": {
                    "m5_to_favorita": {
                        "source_dataset": "m5",
                        "target_dataset": "favorita",
                        "b3": {"alpha": 0.35},
                    },
                    "favorita_to_m5": {
                        "source_dataset": "favorita",
                        "target_dataset": "m5",
                        "b3": {"alpha": 0.65},
                    },
                },
                "b4_vs_b3": {
                    "directions": {
                        dataset: {"ri_percent": 10.0} for dataset in DATASETS
                    },
                    "macro_ri_percent": 10.0,
                },
                "m1_vs_b3": {
                    "directions": {
                        dataset: {"ri_percent": 20.0} for dataset in DATASETS
                    },
                    "macro_ri_percent": 20.0,
                },
                "m1_vs_b4": {
                    "directions": {
                        dataset: {"ri_percent": 100.0 / 9.0}
                        for dataset in DATASETS
                    },
                    "macro_ri_percent": 100.0 / 9.0,
                },
                "bootstrap": {
                    "directions": {
                        dataset: {"ci95_percent": [1.0, 3.0]}
                        for dataset in DATASETS
                    },
                    "macro": {"ci95_percent": [1.1, 2.9]},
                },
                "gate1b": {
                    "ri_percent": {dataset: 10.0 for dataset in DATASETS},
                    "passed": True,
                },
                "gate2": {
                    "m1_vs_b4_percent": {
                        dataset: 100.0 / 9.0 for dataset in DATASETS
                    },
                    "m1_vs_b4_macro_percent": 100.0 / 9.0,
                    "m1_vs_b3_percent": {
                        dataset: 20.0 for dataset in DATASETS
                    },
                    "m1_vs_b3_macro_percent": 20.0,
                    "direction_ci95_percent": {
                        dataset: [1.0, 3.0] for dataset in DATASETS
                    },
                    "macro_ci95_percent": [1.1, 2.9],
                    "passed": True,
                },
                "gate3_safety": {
                    "origin_metrics": [
                        {
                            "dataset_id": dataset,
                            "origin": origin,
                            "ri_m1_vs_b3_percent": 20.0,
                            "ri_m1_vs_b4_percent": 100.0 / 9.0,
                        }
                        for dataset in DATASETS
                        for origin in ORIGINS[dataset]
                    ],
                    "worst_origin_ri_m1_vs_b3_percent": 20.0,
                    "tail_metrics": [
                        {"dataset_id": dataset, "q95_m1_over_b4": 0.9}
                        for dataset in DATASETS
                    ],
                    "passed": True,
                },
                "gate3_control": {
                    "direction_ri_percent": {
                        "real": {
                            dataset: 100.0 / 9.0 for dataset in DATASETS
                        },
                        "shuffled": {dataset: 1.0 for dataset in DATASETS},
                        "random": {dataset: 2.0 for dataset in DATASETS},
                    },
                    "macro_ri_percent": {
                        "real": 100.0 / 9.0,
                        "shuffled": 1.0,
                        "random": 2.0,
                    },
                    "shuffle_over_real": 0.09,
                    "random_over_real": 0.18,
                    "passed": True,
                },
                "gate4": {"action": "ACCEPT_SEED0", "passed": True},
            }
        )
        tables["target_policy_losses"] = _policy_losses()
    return {
        "dataset_outputs": {
            dataset: _dataset_output(dataset) for dataset in DATASETS
        },
        "report": report,
        "tables": tables,
    }


def _protocol_result() -> dict:
    seed0 = _seed0_result()
    return {
        "report": {
            "experiment": "PH-ONLINE-MEMORY-GONO-v1",
            "stage": "FULL_PROTOCOL",
            "executed_model_seeds": [0],
            "seed_reports": {"0": seed0["report"]},
            "robustness": seed0["report"]["gate4"],
            "terminal": True,
            "final_verdict": "RETRIEVAL_MEMORY_GO",
            "preregistration_sha256": "a" * 64,
        },
        "seed_results": {0: seed0},
        "tables": {},
    }


def _multiseed_protocol_result(*, averaged_gate2_pass: bool) -> dict:
    result = _protocol_result()
    seed0 = result["seed_results"][0]
    seed0["report"]["gate2"]["passed"] = False
    seed0["report"]["gate2"]["decision_state"] = "PENDING_GATE4"
    seed0["report"]["gate4"] = {"action": "RUN_SEED1", "passed": None}
    seed0["report"]["terminal"] = False
    seed0["report"]["final_verdict"] = None
    seed1 = copy.deepcopy(seed0)
    seed1["report"]["model_seed"] = 1
    averaged_losses = _policy_losses().copy()
    averaged_losses["m1_normalized_loss"] = 7.5
    averaged_gate2 = copy.deepcopy(seed0["report"]["gate2"])
    averaged_gate2.update(
        {
            "passed": averaged_gate2_pass,
            "decision_state": "FINAL_SEED_AVERAGE",
            "m1_vs_b4_macro_percent": 100.0 * (1.0 - 7.5 / 9.0),
        }
    )
    comparisons = {
        "m1_vs_b4": 100.0 * (1.0 - 7.5 / 9.0),
        "m1_vs_b3": 25.0,
        "b4_vs_b3": 10.0,
    }
    result["seed_results"][1] = seed1
    result["report"].update(
        {
            "executed_model_seeds": [0, 1],
            "seed_reports": {"0": seed0["report"], "1": seed1["report"]},
            "robustness": {"action": "ACCEPT_TWO_SEED", "passed": True},
            "averaged_gate2": averaged_gate2,
            "seed_average_gate2_analysis": {
                "gate2": averaged_gate2,
                **{
                    name: {
                        "directions": {
                            dataset: {"ri_percent": value}
                            for dataset in DATASETS
                        }
                    }
                    for name, value in comparisons.items()
                },
                "bootstrap": {
                    "directions": {
                        dataset: {"ci95_percent": [1.0, 3.0]}
                        for dataset in DATASETS
                    }
                },
            },
            "final_verdict": (
                "RETRIEVAL_MEMORY_GO"
                if averaged_gate2_pass
                else "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO"
            ),
        }
    )
    result["tables"] = {"seed_average_policy_losses": averaged_losses}
    return result


class ResultTableTests(unittest.TestCase):
    def test_builds_deterministic_json_safe_tables_a_through_g(self):
        result = _protocol_result()
        first = build_tables_a_to_g(result)
        second = build_tables_a_to_g(result)

        self.assertEqual(tuple(first), TABLE_KEYS)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)
        self.assertEqual([row["dataset"] for row in first["table_a"]], list(DATASETS))
        self.assertEqual(first["table_a"][0]["training_runtime_seconds"], 24.0)
        self.assertEqual(
            first["table_a"][0][
                "actual_synchronized_end_to_end_wall_seconds"
            ],
            26.0,
        )
        self.assertEqual(first["table_b"][0]["source_static_alpha"], 0.65)
        self.assertEqual(first["table_d"][0]["source"], "favorita")
        self.assertEqual(len(first["table_e"]), 12)
        self.assertTrue(all("is_worst_origin" in row for row in first["table_e"]))
        self.assertEqual(len(first["table_g"]), 7)
        self.assertTrue(all(row["pass_fail"] == "PASS" for row in first["table_g"]))

    def test_gate0_stop_emits_explicit_not_run_rows(self):
        tables = build_tables_a_to_g(_seed0_result(stopped_after="GATE0"))

        self.assertEqual(tables["table_a"][0]["eligible_n"], 101)
        self.assertEqual(tables["table_b"][0]["always_point_loss"], 10.0)
        self.assertEqual(
            tables["table_b"][0]["source_static_status"],
            "NOT_RUN_AFTER_GATE0",
        )
        for key in ("table_c", "table_d", "table_e", "table_f"):
            self.assertEqual(tables[key][0]["status"], "NOT_RUN_AFTER_GATE0")
        gate_rows = {row["gate"]: row for row in tables["table_g"]}
        self.assertEqual(gate_rows["GATE0"]["pass_fail"], "FAIL")
        self.assertEqual(
            gate_rows["GATE1A"]["pass_fail"], "NOT_RUN_AFTER_GATE0"
        )

    def test_partial_target_policy_table_marks_the_unrun_retrieval_stage(self):
        result = _seed0_result()
        result["report"]["stopped_after"] = "GATE1B"
        result["report"]["final_verdict"] = "ONLINE_MEMORY_NO_GO"
        result["report"]["gate1b"]["passed"] = False
        result["tables"] = {
            "target_b3_b4_losses": _policy_losses().drop(
                columns="m1_normalized_loss"
            )
        }
        for key in (
            "m1_vs_b3",
            "m1_vs_b4",
            "bootstrap",
            "gate2",
            "gate3_safety",
            "gate3_control",
            "gate4",
        ):
            result["report"].pop(key, None)

        rows = build_tables_a_to_g(result)["table_d"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(
                row["status"] == "PARTIAL_OBSERVED_M1_NOT_RUN_AFTER_GATE1B"
                for row in rows
            )
        )
        self.assertTrue(all(row["b4_loss"] == 9.0 for row in rows))
        self.assertTrue(all(row["m1_loss"] is None for row in rows))

    def test_rejects_downstream_gate_and_table_payload_after_an_early_stop(self):
        contaminated = _seed0_result()
        contaminated["report"]["stopped_after"] = "GATE0"
        contaminated["report"]["final_verdict"] = "FULL_NO_GO"
        contaminated["report"]["gate0"]["passed"] = False

        with self.assertRaisesRegex(ValueError, "downstream gate1a"):
            build_tables_a_to_g(contaminated)

        contaminated = _seed0_result(stopped_after="GATE0")
        contaminated["tables"]["target_policy_losses"] = _policy_losses()
        with self.assertRaisesRegex(ValueError, "downstream target_policy_losses"):
            build_tables_a_to_g(contaminated)

    def test_table_a_rejects_cpu_timing_labeled_as_actual_gpu_wall_time(self):
        result = _protocol_result()
        result["seed_results"][0]["dataset_outputs"]["m5"]["provenance"][
            "point"
        ]["execution_device_type"] = "cpu"
        with self.assertRaisesRegex(ValueError, "not executed on CUDA"):
            build_tables_a_to_g(result)

    def test_later_stops_preserve_every_already_computed_table(self):
        gate2 = _seed0_result()
        gate2["report"]["stopped_after"] = "GATE2"
        gate2["report"]["final_verdict"] = (
            "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO"
        )
        gate2["report"]["gate2"]["passed"] = False
        for key in ("gate3_safety", "gate3_control", "gate4"):
            gate2["report"].pop(key)
        gate2_tables = build_tables_a_to_g(gate2)
        self.assertEqual(gate2_tables["table_d"][0]["m1_loss"], 8.0)
        self.assertEqual(
            gate2_tables["table_e"][0]["status"], "NOT_RUN_AFTER_GATE2"
        )

        safety = _seed0_result()
        safety["report"]["stopped_after"] = "GATE3_SAFETY"
        safety["report"]["final_verdict"] = "RETRIEVAL_UNSAFE_NO_GO"
        safety["report"]["gate3_safety"]["passed"] = False
        for key in ("gate3_control", "gate4"):
            safety["report"].pop(key)
        safety_tables = build_tables_a_to_g(safety)
        self.assertEqual(safety_tables["table_e"][0]["origin"], 1773)
        self.assertEqual(
            safety_tables["table_f"][0]["status"],
            "NOT_RUN_AFTER_GATE3_SAFETY",
        )

        control = _seed0_result()
        control["report"]["stopped_after"] = "GATE3_CONTROL"
        control["report"]["final_verdict"] = "RETRIEVAL_SIGNAL_NOT_IDENTIFIED"
        control["report"]["gate3_control"]["passed"] = False
        control["report"].pop("gate4")
        control_tables = build_tables_a_to_g(control)
        self.assertEqual(
            control_tables["table_f"][0]["shuffled_value_ri_percent"], 1.0
        )
        self.assertEqual(
            control_tables["table_g"][-1]["pass_fail"],
            "NOT_RUN_AFTER_GATE3_CONTROL",
        )

    def test_nonterminal_seed0_tables_record_the_conditional_seed_request(self):
        result = _seed0_result()
        result["report"]["terminal"] = False
        result["report"]["final_verdict"] = None
        result["report"]["gate2"]["passed"] = False
        result["report"]["gate2"]["decision_state"] = "PENDING_GATE4"
        result["report"]["gate4"] = {"action": "RUN_SEED1", "passed": None}

        gate_rows = {
            row["gate"]: row for row in build_tables_a_to_g(result)["table_g"]
        }
        self.assertEqual(gate_rows["GATE2"]["pass_fail"], "PENDING_GATE4")
        gate4 = gate_rows["GATE4"]
        self.assertEqual(gate4["pass_fail"], "PENDING_SEED1")
        self.assertEqual(gate4["observed"]["action"], "RUN_SEED1")

    def test_multiseed_tables_use_final_averaged_gate2_and_policy_losses(self):
        result = _protocol_result()
        seed0 = result["seed_results"][0]
        seed0["report"]["gate2"]["passed"] = False
        seed0["report"]["gate2"]["decision_state"] = "PENDING_GATE4"
        seed0["report"]["gate4"] = {"action": "RUN_SEED1", "passed": None}
        seed1 = copy.deepcopy(seed0)
        seed1["report"]["model_seed"] = 1
        result["seed_results"][1] = seed1
        averaged_losses = _policy_losses().copy()
        averaged_losses["m1_normalized_loss"] = 7.5
        averaged_gate2 = copy.deepcopy(seed0["report"]["gate2"])
        averaged_gate2["passed"] = True
        averaged_gate2["decision_state"] = "FINAL_SEED_AVERAGE"
        averaged_gate2["m1_vs_b4_macro_percent"] = 100.0 * (1.0 - 7.5 / 9.0)
        result["report"].update(
            {
                "executed_model_seeds": [0, 1],
                "seed_reports": {
                    "0": seed0["report"],
                    "1": seed1["report"],
                },
                "robustness": {"action": "ACCEPT_TWO_SEED", "passed": True},
                "averaged_gate2": averaged_gate2,
                "seed_average_gate2_analysis": {
                    "gate2": averaged_gate2,
                    "m1_vs_b4": {
                        "directions": {
                            dataset: {"ri_percent": 100.0 * (1.0 - 7.5 / 9.0)}
                            for dataset in DATASETS
                        }
                    },
                    "m1_vs_b3": {
                        "directions": {
                            dataset: {"ri_percent": 25.0} for dataset in DATASETS
                        }
                    },
                    "b4_vs_b3": {
                        "directions": {
                            dataset: {"ri_percent": 10.0} for dataset in DATASETS
                        }
                    },
                    "bootstrap": {
                        "directions": {
                            dataset: {"ci95_percent": [1.0, 3.0]}
                            for dataset in DATASETS
                        }
                    },
                },
            }
        )
        result["tables"] = {"seed_average_policy_losses": averaged_losses}

        tables = build_tables_a_to_g(result)

        gate2_row = {row["gate"]: row for row in tables["table_g"]}["GATE2"]
        self.assertEqual(gate2_row["pass_fail"], "PASS")
        self.assertEqual(
            gate2_row["observed"]["m1_vs_b4_macro_percent"],
            averaged_gate2["m1_vs_b4_macro_percent"],
        )
        self.assertTrue(
            all(row["m1_loss"] == 7.5 for row in tables["table_d"])
        )
        self.assertTrue(
            all(row["status"] == "OBSERVED_SEED_AVERAGE" for row in tables["table_d"])
        )

    def test_multiseed_terminal_result_requires_an_averaged_gate2(self):
        result = _protocol_result()
        result["seed_results"][1] = copy.deepcopy(result["seed_results"][0])
        result["report"]["executed_model_seeds"] = [0, 1]
        result["report"]["robustness"] = {
            "action": "ACCEPT_TWO_SEED",
            "passed": True,
        }

        with self.assertRaisesRegex(ValueError, "averaged Gate2"):
            build_tables_a_to_g(result)

    def test_actual_seed_average_analysis_populates_every_table_d_ri(self):
        seed0 = _policy_losses().assign(model_seed=0)
        seed1 = _policy_losses().assign(model_seed=1)
        seed1["m1_normalized_loss"] = 7.0
        analysis = evaluate_seed_average_gate2(
            pd.concat([seed0, seed1], ignore_index=True),
            expected_seeds=(0, 1),
            draws=20,
            seed=20260904,
        )
        averaged_losses = analysis.pop("seed_average_losses")
        result = _multiseed_protocol_result(averaged_gate2_pass=True)
        result["report"]["averaged_gate2"] = analysis["gate2"]
        result["report"]["seed_average_gate2_analysis"] = analysis
        result["tables"]["seed_average_policy_losses"] = averaged_losses

        first = build_tables_a_to_g(result)["table_d"]
        second = build_tables_a_to_g(result)["table_d"]

        self.assertEqual(first, second)
        self.assertTrue(
            all(row["ri_b4_vs_b3_percent"] is not None for row in first)
        )
        for row in first:
            self.assertAlmostEqual(row["ri_b4_vs_b3_percent"], 10.0)

    def test_withdrawn_gate2_deferral_preserves_gate3_eligibility_observations(self):
        for veto in ("GATE3_SAFETY", "GATE3_CONTROL"):
            with self.subTest(veto=veto):
                result = _seed0_result()
                report = result["report"]
                report["stopped_after"] = veto
                report["final_verdict"] = (
                    "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO"
                )
                report["gate2"].update(
                    {
                        "passed": False,
                        "decision_state": "FINAL_FAIL",
                        "deferral_withdrawn_by": veto,
                    }
                )
                report["gate3_safety"]["passed"] = veto != "GATE3_SAFETY"
                if veto == "GATE3_SAFETY":
                    report.pop("gate3_control")
                    report.pop("controls", None)
                    report.pop("gate4")
                else:
                    report["gate3_control"]["passed"] = False
                    report.pop("gate4")

                gate_rows = {
                    row["gate"]: row
                    for row in build_tables_a_to_g(result)["table_g"]
                }

                self.assertEqual(gate_rows["GATE2"]["pass_fail"], "FAIL")
                self.assertEqual(
                    gate_rows["GATE3_SAFETY"]["status"],
                    "ELIGIBILITY_OBSERVATION",
                )
                expected_safety = (
                    "FAIL"
                    if veto == "GATE3_SAFETY"
                    else "PASS"
                )
                self.assertEqual(
                    gate_rows["GATE3_SAFETY"]["pass_fail"], expected_safety
                )
                if veto == "GATE3_CONTROL":
                    self.assertEqual(
                        gate_rows["GATE3_CONTROL"]["pass_fail"],
                        "FAIL",
                    )

    def test_rejects_inconsistent_gate2_deferral_veto_payload(self):
        result = _seed0_result()
        report = result["report"]
        report["stopped_after"] = "GATE3_CONTROL"
        report["final_verdict"] = (
            "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO"
        )
        report["gate2"].update(
            {
                "passed": False,
                "decision_state": "FINAL_FAIL",
                "deferral_withdrawn_by": "GATE3_SAFETY",
            }
        )
        report["gate3_control"]["passed"] = False
        report.pop("gate4")

        with self.assertRaisesRegex(ValueError, "deferral veto"):
            build_tables_a_to_g(result)


class StatusTests(unittest.TestCase):
    def test_gate4_pass_cannot_rescue_a_failed_final_averaged_gate2(self):
        status = build_status_markdown(
            _multiseed_protocol_result(averaged_gate2_pass=False),
            preregistration=FROZEN_PREREGISTRATION,
        )

        self.assertTrue(
            status.startswith(
                "FINAL VERDICT: SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO\n"
            )
        )
        self.assertIn('"gate":"GATE2"', status)
        self.assertIn('"pass_fail":"FAIL"', status)

    def test_withdrawn_gate2_deferral_has_failure_first_simple_verdict(self):
        result = _seed0_result()
        result["report"].update(
            {
                "stopped_after": "GATE3_SAFETY",
                "final_verdict": (
                    "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO"
                ),
                "preregistration_sha256": "a" * 64,
            }
        )
        result["report"]["gate2"].update(
            {
                "passed": False,
                "decision_state": "FINAL_FAIL",
                "deferral_withdrawn_by": "GATE3_SAFETY",
            }
        )
        result["report"]["gate3_safety"]["passed"] = False
        result["report"].pop("gate3_control")
        result["report"].pop("controls", None)
        result["report"].pop("gate4")

        status = build_status_markdown(
            result, preregistration=FROZEN_PREREGISTRATION
        )

        self.assertTrue(
            status.startswith(
                "FINAL VERDICT: SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO\n"
            )
        )
        self.assertIn("ELIGIBILITY_OBSERVATION", status)

    def test_terminal_status_has_exact_verdict_and_sixteen_ordered_sections(self):
        status = build_status_markdown(
            _protocol_result(),
            preregistration=FROZEN_PREREGISTRATION,
        )

        self.assertTrue(status.startswith("FINAL VERDICT: RETRIEVAL_MEMORY_GO\n"))
        headings = re.findall(r"^## (\d+)\. (.+)$", status, flags=re.MULTILINE)
        self.assertEqual([int(number) for number, _ in headings], list(range(1, 17)))
        self.assertIn("[관찰]", status)
        self.assertIn("[판정]", status)
        self.assertIn("[최종]", status)
        self.assertIn("untouched external confirmatory study", status)
        self.assertNotIn("대체로 좋아졌다", status)
        for section_number in (11, 12):
            section = status.split(f"## {section_number}.", 1)[1].split("\n\n##", 1)[0]
            self.assertNotIn("pass_fail", section)
            self.assertNotIn("scientific_interpretation", section)

    def test_runtime_stop_status_does_not_fabricate_a_scientific_verdict(self):
        status = build_runtime_stop_status(
            {
                "runtime_gate": {
                    "threshold_gpu_hours": 6.0,
                    "projected_gpu_hours": 8.5,
                    "exceeded": True,
                    "action": "STOP_FOR_APPROVAL",
                },
                "runtime_projection_2000_per_dataset": {"gpu_hours": 0.8},
            }
        )

        self.assertTrue(status.startswith("EXECUTION STATUS: STOP_FOR_APPROVAL\n"))
        self.assertNotIn("FINAL VERDICT:", status)
        self.assertIn("SCIENTIFIC GATES: NOT_RUN", status)
        self.assertIn("8.5", status)
        self.assertIn("6.0", status)

    def test_nonterminal_scientific_result_cannot_be_rendered_as_final(self):
        result = _protocol_result()
        result["report"]["terminal"] = False
        result["report"]["final_verdict"] = None
        with self.assertRaisesRegex(ValueError, "terminal scientific verdict"):
            build_status_markdown(result)

    def test_terminal_verdict_must_match_the_observed_failure_first_path(self):
        for verdict in EXACT_VERDICTS:
            result = _protocol_result()
            result["report"]["final_verdict"] = verdict
            if verdict == "RETRIEVAL_MEMORY_GO":
                status = build_status_markdown(
                    result, preregistration=FROZEN_PREREGISTRATION
                )
                self.assertTrue(status.startswith(f"FINAL VERDICT: {verdict}\n"))
            else:
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    build_status_markdown(
                        result, preregistration=FROZEN_PREREGISTRATION
                    )

        result = _protocol_result()
        result["report"]["final_verdict"] = "STOP_FOR_APPROVAL"
        with self.assertRaisesRegex(ValueError, "terminal scientific verdict"):
            build_status_markdown(
                result, preregistration=FROZEN_PREREGISTRATION
            )

    def test_terminal_status_requires_the_bound_prefit_preregistration(self):
        with self.assertRaisesRegex(ValueError, "frozen preregistration"):
            build_status_markdown(_protocol_result())

        invalid = dict(FROZEN_PREREGISTRATION)
        invalid["frozen_before_any_new_model_fit"] = False
        with self.assertRaisesRegex(ValueError, "pre-fit"):
            build_status_markdown(_protocol_result(), preregistration=invalid)

    def test_runtime_stop_rejects_an_inconsistent_or_incomplete_projection(self):
        base = {
            "runtime_gate": {
                "threshold_gpu_hours": 6.0,
                "projected_gpu_hours": 5.0,
                "exceeded": True,
                "action": "STOP_FOR_APPROVAL",
            },
            "runtime_projection_2000_per_dataset": {"gpu_hours": 0.8},
        }
        with self.assertRaisesRegex(ValueError, "above threshold"):
            build_runtime_stop_status(base)

        base["runtime_gate"]["projected_gpu_hours"] = 8.0
        base.pop("runtime_projection_2000_per_dataset")
        with self.assertRaisesRegex(ValueError, "2000-series"):
            build_runtime_stop_status(base)


if __name__ == "__main__":
    unittest.main()
