from __future__ import annotations

import copy
import math
import unittest

import numpy as np
import pandas as pd

from experiments.ph_online_memory_gono_v1.analysis import (
    FINAL_VERDICT_TOKENS,
    adjacent_delta_pairs,
    decide_final_verdict,
    evaluate_gate0,
    evaluate_gate1a,
    evaluate_gate1b,
    evaluate_gate2,
    evaluate_gate3_control,
    evaluate_gate3_safety,
    evaluate_gate4_seed0,
    evaluate_gate4_seed1,
    evaluate_gate4_seed2,
    evaluate_seed_average_gate2,
    oracle_ladder,
    paired_series_cluster_bootstrap,
    relative_improvement_percent,
    shuffle_previous_delta_within_pair,
    summarize_loss_comparison,
    temporal_recurrence,
)


DATASETS = ("m5", "favorita")
ORIGINS = (100, 128, 156, 184, 212, 240)


def _oracle_steps(dataset_id: str = "toy") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_id in ("s1", "s2"):
        for origin_index, origin in enumerate(ORIGINS):
            target = 0.0 if origin_index < 3 else 2.0
            for step in range(28):
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "series_id": series_id,
                        "origin": origin,
                        "step": step,
                        "y_observed": target,
                        "point_mean_prediction": 0.0,
                        "hurdle_mean_prediction": 2.0,
                        "target_mask": True,
                        "policy_scale_squared": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _opposing_oracle_steps() -> pd.DataFrame:
    frame = _oracle_steps()
    frame.loc[frame["series_id"] == "s1", "y_observed"] = 0.0
    frame.loc[frame["series_id"] == "s2", "y_observed"] = 2.0
    return frame


def _recurrence_losses(*, constant: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_id in DATASETS:
        for series_index in range(8):
            delta = 1.0 if constant else float(series_index)
            for origin in ORIGINS:
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "series_id": f"{dataset_id}-{series_index}",
                        "origin": origin,
                        "point_normalized_loss": 10.0,
                        "hurdle_normalized_loss": 10.0 + delta,
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap_losses() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "m5": {"s1": (0.0, 1.0), "s2": (3.0, 3.0)},
        "favorita": {"t1": (1.0, 2.0), "t2": (2.0, 2.0)},
    }
    for dataset_id, series in values.items():
        for series_id, (candidate, baseline) in series.items():
            for origin in ORIGINS:
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "series_id": series_id,
                        "origin": origin,
                        "candidate_loss": candidate,
                        "baseline_loss": baseline,
                    }
                )
    return pd.DataFrame(rows)


def _safety_losses(
    *, m1: float, b3: float, b4: float
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_id in DATASETS:
        for series_id in ("s1", "s2"):
            for origin in ORIGINS:
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "series_id": f"{dataset_id}-{series_id}",
                        "origin": origin,
                        "m1_normalized_loss": m1,
                        "b3_normalized_loss": b3,
                        "b4_normalized_loss": b4,
                    }
                )
    return pd.DataFrame(rows)


def _seed_policy_losses(seed_count: int = 3) -> pd.DataFrame:
    values = ((1.0, 2.0), (1.2, 2.0), (99.0, 100.0))
    rows: list[dict[str, object]] = []
    for model_seed, (candidate, baseline) in enumerate(values[:seed_count]):
        for dataset_id in DATASETS:
            for series_id in ("s1", "s2"):
                for origin in ORIGINS:
                    rows.append(
                        {
                            "model_seed": model_seed,
                            "dataset_id": dataset_id,
                            "series_id": f"{dataset_id}-{series_id}",
                            "origin": origin,
                            "m1_normalized_loss": candidate,
                            "b4_normalized_loss": baseline,
                            "b3_normalized_loss": 4.0,
                        }
                    )
    return pd.DataFrame(rows)


def _gate0_ladder(gain_percent: float) -> dict[str, object]:
    return {
        "families": {
            "hard": {
                "global_oracle_loss": 7.0,
                "origin_oracle_loss": 6.0,
            },
            "convex": {
                "global_static_loss": 100.0,
                "origin_oracle_loss": 100.0 * (1.0 - gain_percent / 100.0),
            },
        }
    }


class OracleLadderTests(unittest.TestCase):
    def test_grid_oracles_match_hand_derived_losses_and_keep_families_separate(self):
        # Catches reversing alpha, optimizing the wrong grouping, or mixing hard
        # and convex oracle denominators.
        result = oracle_ladder(_oracle_steps())["toy"]

        self.assertEqual(result["n_series"], 2)
        self.assertEqual(result["n_origins"], 6)
        self.assertAlmostEqual(result["always_point_loss"], 2.0)
        self.assertAlmostEqual(result["always_hurdle_loss"], 2.0)
        self.assertAlmostEqual(result["half_half_loss"], 1.0)
        self.assertAlmostEqual(result["families"]["hard"]["global_oracle_loss"], 2.0)
        self.assertAlmostEqual(result["families"]["hard"]["series_oracle_loss"], 2.0)
        self.assertAlmostEqual(result["families"]["hard"]["origin_oracle_loss"], 0.0)
        self.assertAlmostEqual(result["families"]["convex"]["global_static_alpha"], 0.5)
        self.assertAlmostEqual(result["families"]["convex"]["global_static_loss"], 1.0)
        self.assertAlmostEqual(result["families"]["convex"]["series_oracle_loss"], 1.0)
        self.assertAlmostEqual(result["families"]["convex"]["origin_oracle_loss"], 0.0)

    def test_origin_oracle_selects_per_series_origin_not_one_choice_per_timestamp(self):
        result = oracle_ladder(_opposing_oracle_steps())["toy"]
        self.assertAlmostEqual(result["families"]["hard"]["origin_oracle_loss"], 0.0)
        self.assertAlmostEqual(result["families"]["convex"]["origin_oracle_loss"], 0.0)
        self.assertEqual(len(result["families"]["hard"]["origin_choices"]), 12)
        self.assertEqual(len(result["families"]["convex"]["origin_alphas"]), 12)

    def test_oracle_rejects_unpaired_or_nonfinite_six_origin_panel(self):
        duplicate = pd.concat([_oracle_steps(), _oracle_steps().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            oracle_ladder(duplicate)

        incomplete = _oracle_steps().loc[lambda frame: frame["origin"] != ORIGINS[-1]]
        with self.assertRaisesRegex(ValueError, "six"):
            oracle_ladder(incomplete)

        nonfinite = _oracle_steps()
        nonfinite.loc[0, "point_mean_prediction"] = math.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            oracle_ladder(nonfinite)

        partial_mask = _oracle_steps()
        partial_mask.loc[0, "target_mask"] = False
        with self.assertRaisesRegex(ValueError, "(?i)(28|mask|complete)"):
            oracle_ladder(partial_mask)

        short_horizon = _oracle_steps().loc[lambda frame: frame["step"] < 27]
        with self.assertRaisesRegex(ValueError, "(?i)(28|step|horizon)"):
            oracle_ladder(short_horizon)


class GateZeroTests(unittest.TestCase):
    def test_gate0_inclusive_thresholds_use_only_convex_ladder(self):
        ladders = {"m5": _gate0_ladder(1.0), "favorita": _gate0_ladder(3.0)}
        result = evaluate_gate0(ladders)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["macro_gain_percent"], 2.0)

        changed_hard = copy.deepcopy(ladders)
        changed_hard["m5"]["families"]["hard"] = {
            "global_oracle_loss": 1e12,
            "origin_oracle_loss": 0.0,
        }
        self.assertEqual(evaluate_gate0(changed_hard)["passed"], result["passed"])

    def test_gate0_failure_records_not_available_or_heterogeneous_candidate(self):
        ladders = {dataset: _gate0_ladder(0.0) for dataset in DATASETS}
        unavailable = evaluate_gate0(
            ladders,
            heterogeneous_diagnostic={"status": "NOT_AVAILABLE"},
        )
        self.assertFalse(unavailable["passed"])
        self.assertEqual(unavailable["stage_verdict"], "POINT_HURDLE_MEMORY_NO_GO")
        self.assertEqual(unavailable["final_verdict"], "FULL_NO_GO")

        candidate = evaluate_gate0(
            ladders,
            heterogeneous_diagnostic={
                "status": "AVAILABLE",
                "macro_gain_percent": 2.0,
            },
        )
        self.assertEqual(
            candidate["final_verdict"],
            "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE",
        )


class TemporalRecurrenceTests(unittest.TestCase):
    def test_adjacent_delta_and_shuffle_are_within_pair_and_deterministic(self):
        pairs = adjacent_delta_pairs(_recurrence_losses())
        first = shuffle_previous_delta_within_pair(pairs, seed=20260904)
        second = shuffle_previous_delta_within_pair(
            pairs.sample(frac=1.0, random_state=4), seed=20260904
        )
        pd.testing.assert_frame_equal(first, second)

        group_keys = ["dataset_id", "previous_origin", "origin"]
        for key, original in pairs.groupby(group_keys, sort=True):
            shuffled = first.set_index(group_keys).loc[key]
            self.assertEqual(
                sorted(original["previous_delta"].tolist()),
                sorted(shuffled["previous_delta"].tolist()),
            )

    def test_real_rho_and_constant_degeneracy_are_explicit(self):
        result = temporal_recurrence(
            _recurrence_losses(), bootstrap_draws=100, seed=20260904
        )
        for dataset_id in DATASETS:
            self.assertEqual(result["datasets"][dataset_id]["real_status"], "OK")
            self.assertAlmostEqual(result["datasets"][dataset_id]["real_rho"], 1.0)
            self.assertEqual(result["datasets"][dataset_id]["bootstrap_draws"], 100)

        degenerate = temporal_recurrence(
            _recurrence_losses(constant=True), bootstrap_draws=20, seed=20260904
        )
        self.assertFalse(degenerate["passed"])
        for dataset_id in DATASETS:
            self.assertEqual(
                degenerate["datasets"][dataset_id]["real_status"], "DEGENERATE"
            )
            self.assertIsNone(degenerate["datasets"][dataset_id]["real_rho"])

    def test_gate1a_thresholds_are_strict(self):
        boundary = {
            dataset: {"real_status": "OK", "real_rho": 0.10, "shuffled_rho": 0.0}
            for dataset in DATASETS
        }
        self.assertFalse(evaluate_gate1a(boundary)["passed"])

        above = {
            dataset: {
                "real_status": "OK",
                "real_rho": 0.100000002,
                "shuffled_rho": 0.050000001,
            }
            for dataset in DATASETS
        }
        self.assertTrue(evaluate_gate1a(above)["passed"])


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_resamples_whole_series_and_datasets_independently(self):
        result = paired_series_cluster_bootstrap(
            _bootstrap_losses(),
            candidate_column="candidate_loss",
            baseline_column="baseline_loss",
            draws=2000,
            seed=20260904,
        )

        self.assertAlmostEqual(result["directions"]["m5"]["ri_percent"], 25.0)
        self.assertAlmostEqual(result["directions"]["favorita"]["ri_percent"], 25.0)
        self.assertAlmostEqual(result["macro"]["ri_percent"], 25.0)
        self.assertEqual(result["directions"]["m5"]["ci95_percent"], [0.0, 100.0])
        self.assertEqual(result["directions"]["favorita"]["ci95_percent"], [0.0, 50.0])

        observed = {
            round(value, 8)
            for value in result["macro"]["bootstrap_ri_percent"]
        }
        self.assertTrue(observed.issubset({0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0}))
        self.assertIn(12.5, observed)  # impossible if the two datasets share a draw
        self.assertIn("conditional on the six observed origins", result["uncertainty_scope"])

    def test_bootstrap_rejects_incomplete_or_nonfinite_clusters(self):
        incomplete = _bootstrap_losses().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "six"):
            paired_series_cluster_bootstrap(
                incomplete,
                candidate_column="candidate_loss",
                baseline_column="baseline_loss",
                draws=10,
            )

        nonfinite = _bootstrap_losses()
        nonfinite.loc[0, "candidate_loss"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_series_cluster_bootstrap(
                nonfinite,
                candidate_column="candidate_loss",
                baseline_column="baseline_loss",
                draws=10,
            )


class DownstreamGateTests(unittest.TestCase):
    def test_seed_average_gate2_averages_only_three_losses_by_exact_case_key(self):
        losses = _seed_policy_losses(seed_count=2)
        losses["prediction"] = np.arange(len(losses), dtype=np.float64)
        losses["precomputed_ri_percent"] = -999.0

        result = evaluate_seed_average_gate2(
            losses,
            expected_seeds=(0, 1),
            draws=20,
            seed=20260904,
        )

        averaged = result["seed_average_losses"]
        self.assertEqual(
            list(averaged.columns),
            [
                "dataset_id",
                "series_id",
                "origin",
                "m1_normalized_loss",
                "b4_normalized_loss",
                "b3_normalized_loss",
            ],
        )
        self.assertTrue(np.allclose(averaged["m1_normalized_loss"], 1.1))
        self.assertTrue(np.allclose(averaged["b4_normalized_loss"], 2.0))
        self.assertTrue(np.allclose(averaged["b3_normalized_loss"], 4.0))
        self.assertTrue(result["gate2"]["passed"])
        self.assertEqual(
            result["aggregation"],
            "average B3/B4/M1 policy loss by dataset/series/origin before Gate2 bootstrap",
        )

    def test_seed_average_gate2_rejects_any_cross_seed_case_key_mismatch(self):
        losses = _seed_policy_losses(seed_count=2)
        seed1_series = (losses["model_seed"] == 1) & (
            losses["series_id"] == "m5-s1"
        )
        losses.loc[seed1_series, "series_id"] = "m5-replacement"

        with self.assertRaisesRegex(ValueError, "identical series-origin keys"):
            evaluate_seed_average_gate2(
                losses,
                expected_seeds=(0, 1),
                draws=20,
                seed=20260904,
            )

    def test_loss_comparison_keeps_raw_candidate_baseline_and_ri_together(self):
        result = summarize_loss_comparison(
            _bootstrap_losses(),
            candidate_column="candidate_loss",
            baseline_column="baseline_loss",
        )
        for dataset_id in DATASETS:
            self.assertEqual(result["directions"][dataset_id]["candidate_loss"], 1.5)
            self.assertEqual(result["directions"][dataset_id]["baseline_loss"], 2.0)
            self.assertEqual(result["directions"][dataset_id]["ri_percent"], 25.0)
        self.assertEqual(result["macro_ri_percent"], 25.0)

    def test_relative_improvement_and_gate1b_use_inclusive_boundary(self):
        self.assertAlmostEqual(relative_improvement_percent(99.0, 100.0), 1.0)
        with self.assertRaisesRegex(ValueError, "baseline"):
            relative_improvement_percent(1.0, 0.0)

        result = evaluate_gate1b({dataset: 0.30 for dataset in DATASETS})
        self.assertTrue(result["passed"])
        failed = evaluate_gate1b({"m5": 0.30, "favorita": np.nextafter(0.30, -np.inf)})
        self.assertFalse(failed["passed"])

    def test_gate2_keeps_inclusive_effects_but_strict_ci_lower_bounds(self):
        bootstrap = {
            "directions": {
                "m5": {"ci95_percent": [0.0, 1.0]},
                "favorita": {"ci95_percent": [0.1, 1.0]},
            },
            "macro": {"ci95_percent": [0.0, 1.0]},
        }
        boundary = evaluate_gate2(
            m1_vs_b4_percent={"m5": -0.10, "favorita": 0.50},
            m1_vs_b3_percent={"m5": 0.30, "favorita": 1.10},
            bootstrap=bootstrap,
        )
        self.assertFalse(boundary["passed"])
        self.assertTrue(boundary["checks"]["macro_effect"])
        self.assertTrue(boundary["checks"]["direction_safety"])
        self.assertFalse(boundary["checks"]["macro_ci"])

        positive = copy.deepcopy(bootstrap)
        positive["macro"]["ci95_percent"][0] = np.nextafter(0.0, np.inf)
        positive["directions"]["m5"]["ci95_percent"][0] = np.nextafter(0.0, np.inf)
        self.assertTrue(
            evaluate_gate2(
                m1_vs_b4_percent={"m5": -0.10, "favorita": 0.50},
                m1_vs_b3_percent={"m5": 0.30, "favorita": 1.10},
                bootstrap=positive,
            )["passed"]
        )

    def test_gate3_safety_inclusive_boundaries_and_control_strict_real(self):
        q95_boundary = evaluate_gate3_safety(
            _safety_losses(m1=1.01, b3=2.0, b4=1.0)
        )
        self.assertTrue(q95_boundary["passed"])
        for row in q95_boundary["tail_metrics"]:
            self.assertAlmostEqual(row["q95_m1_over_b4"], 1.01)

        origin_boundary = evaluate_gate3_safety(
            _safety_losses(m1=1.005, b3=1.0, b4=2.0)
        )
        self.assertTrue(origin_boundary["passed"])
        self.assertAlmostEqual(origin_boundary["worst_origin_ri_m1_vs_b3_percent"], -0.5)

        just_worse = evaluate_gate3_safety(
            _safety_losses(m1=np.nextafter(1.005, np.inf), b3=1.0, b4=2.0)
        )
        self.assertFalse(just_worse["passed"])

        controls = evaluate_gate3_control(
            real_percent={dataset: 4.0 for dataset in DATASETS},
            shuffled_percent={dataset: 1.0 for dataset in DATASETS},
            random_percent={dataset: 2.0 for dataset in DATASETS},
        )
        self.assertTrue(controls["passed"])
        self.assertEqual(controls["shuffle_over_real"], 0.25)
        self.assertEqual(controls["random_over_real"], 0.5)
        self.assertFalse(
            evaluate_gate3_control(
                real_percent={dataset: 0.0 for dataset in DATASETS},
                shuffled_percent={dataset: 0.0 for dataset in DATASETS},
                random_percent={dataset: 0.0 for dataset in DATASETS},
            )["passed"]
        )

    def test_gate4_seed_execution_is_conditional_and_uses_strict_ci(self):
        stopped = evaluate_gate4_seed0(
            macro_ri_percent=0.0,
            ci95_percent=(-0.1, 0.2),
            direction_ri_percent={"m5": 0.1, "favorita": -0.1},
            safety_pass=True,
            control_pass=True,
        )
        self.assertEqual(stopped["action"], "STOP_NO_ADDITIONAL_SEED")
        self.assertFalse(stopped["allow_seed1"])

        clear = evaluate_gate4_seed0(
            macro_ri_percent=0.5,
            ci95_percent=(0.3, 0.7),
            direction_ri_percent={"m5": 0.4, "favorita": 0.6},
            safety_pass=True,
            control_pass=True,
        )
        self.assertEqual(clear["action"], "ACCEPT_SEED0")

        borderline = evaluate_gate4_seed0(
            macro_ri_percent=0.4,
            ci95_percent=(0.1, 0.5),
            direction_ri_percent={"m5": 0.3, "favorita": 0.5},
            safety_pass=True,
            control_pass=True,
        )
        self.assertEqual(borderline["action"], "RUN_SEED1")

        accepted = evaluate_gate4_seed1(
            _seed_policy_losses(seed_count=2),
            candidate_column="m1_normalized_loss",
            baseline_column="b4_normalized_loss",
        )
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["action"], "ACCEPT_TWO_SEED")

        clear_fail = _seed_policy_losses(seed_count=2)
        clear_fail.loc[clear_fail["model_seed"] == 1, "m1_normalized_loss"] = 11.0
        failed = evaluate_gate4_seed1(
            clear_fail,
            candidate_column="m1_normalized_loss",
            baseline_column="b4_normalized_loss",
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["action"], "RETRIEVAL_ROBUSTNESS_NO_GO")
        self.assertFalse(failed["allow_seed2"])

        three_seed = evaluate_gate4_seed2(
            _seed_policy_losses(),
            candidate_column="m1_normalized_loss",
            baseline_column="b4_normalized_loss",
        )
        self.assertTrue(three_seed["passed"])
        self.assertAlmostEqual(
            three_seed["seed_average_macro_ri_percent"],
            100.0 * (1.0 - (101.2 / 3.0) / (104.0 / 3.0)),
        )
        self.assertNotAlmostEqual(
            three_seed["seed_average_macro_ri_percent"],
            np.mean([50.0, 40.0, 1.0]),
        )


class FinalVerdictTests(unittest.TestCase):
    def test_every_terminal_path_returns_an_exact_section40_token(self):
        cases = [
            ({"gate0_pass": False, "heterogeneous_gate_pass": False}, "FULL_NO_GO"),
            (
                {"gate0_pass": False, "heterogeneous_gate_pass": True},
                "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE",
            ),
            (
                {"gate0_pass": True, "gate1a_pass": False},
                "TEMPORAL_RECURRENCE_NO_GO",
            ),
            (
                {"gate0_pass": True, "gate1a_pass": True, "gate1b_pass": False},
                "ONLINE_MEMORY_NO_GO",
            ),
            (
                {
                    "gate0_pass": True,
                    "gate1a_pass": True,
                    "gate1b_pass": True,
                    "gate2_pass": False,
                },
                "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
            ),
            (
                {
                    "gate0_pass": True,
                    "gate1a_pass": True,
                    "gate1b_pass": True,
                    "gate2_pass": True,
                    "gate3_safety_pass": False,
                },
                "RETRIEVAL_UNSAFE_NO_GO",
            ),
            (
                {
                    "gate0_pass": True,
                    "gate1a_pass": True,
                    "gate1b_pass": True,
                    "gate2_pass": True,
                    "gate3_safety_pass": True,
                    "gate3_control_pass": False,
                },
                "RETRIEVAL_SIGNAL_NOT_IDENTIFIED",
            ),
            (
                {
                    "gate0_pass": True,
                    "gate1a_pass": True,
                    "gate1b_pass": True,
                    "gate2_pass": True,
                    "gate3_safety_pass": True,
                    "gate3_control_pass": True,
                    "gate4_pass": True,
                },
                "RETRIEVAL_MEMORY_GO",
            ),
        ]
        observed = []
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                verdict = decide_final_verdict(**arguments)
                self.assertEqual(verdict, expected)
                observed.append(verdict)
        self.assertEqual(tuple(observed), FINAL_VERDICT_TOKENS)


if __name__ == "__main__":
    unittest.main()
