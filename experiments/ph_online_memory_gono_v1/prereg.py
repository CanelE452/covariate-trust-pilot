"""Frozen, result-independent protocol for PH-ONLINE-MEMORY-GONO-v1."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .artifacts import exclusive_write_json, payload_sha256


HASH_CONTRACT = (
    "SHA256 of canonical UTF-8 JSON excluding preregistration_sha256 and "
    "preregistration_hash_contract"
)


def build_preregistered_spec(
    *,
    repository: Mapping[str, Any],
    environment: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    frozen_at_utc: str,
    data_source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the complete protocol without its self-excluding content hash."""
    alpha_grid = [i / 20 for i in range(21)]
    verdicts = [
        "FULL_NO_GO",
        "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE",
        "TEMPORAL_RECURRENCE_NO_GO",
        "ONLINE_MEMORY_NO_GO",
        "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
        "RETRIEVAL_UNSAFE_NO_GO",
        "RETRIEVAL_SIGNAL_NOT_IDENTIFIED",
        "RETRIEVAL_MEMORY_GO",
    ]
    return {
        "schema_version": 1,
        "experiment_name": "PH-ONLINE-MEMORY-GONO-v1",
        "frozen_at_utc": frozen_at_utc,
        "frozen_before_any_new_model_fit": True,
        "repository": dict(repository),
        "environment": dict(environment),
        "development_datasets_only": True,
        "lookback": 96,
        "horizon": 28,
        "models": {
            "point": {
                "id": "M0PM_point_mse_param_matched",
                "implementation": (
                    "experiments.om_factorization_killtest.models."
                    "PointDLinearParamMatched"
                ),
                "loss": "experiments.om_factorization_killtest.models.point_loss",
                "parameters": 7056,
            },
            "hurdle": {
                "id": "M1_factorized_mean",
                "implementation": (
                    "experiments.om_factorization_killtest.models.FactorizedDLinear"
                ),
                "loss": (
                    "experiments.om_factorization_killtest.models.factorized_loss"
                ),
                "mean_semantics": (
                    "sigmoid(occurrence_logit) * positive_conditional_magnitude"
                ),
                "parameters": 7056,
            },
            "shared_target": "realized nonnegative demand",
            "shared_input": "96-step target history; occurrence is supervision only",
            "shared_backbone": "DLinear moving-average decomposition",
            "intended_differences": ["output heads", "loss"],
        },
        "trainer": {
            "identity": (
                "experiments.ph_online_memory_gono_v1.trainer.train_one_on_split"
            ),
            "delegates_to": "experiments.om_factorization_killtest.train.train_one",
            "split_injection": (
                "scoped substitution of the canonical module's build_splits lookup "
                "with the preregistered supplied Split"
            ),
            "execution": "sequential or process-isolated; never concurrent in one interpreter",
            "restart_hook": (
                "after each Point/Hurdle arm, publish CPU state_dict, provenance, "
                "and seven-origin prediction heads through an append-only completion-marked "
                "checkpoint; only hash-verified completed arms may skip a refit"
            ),
            "actual_gpu_time": (
                "sum device-synchronized per-arm wall intervals around "
                "train_one_on_split, including canonical final seven-origin prediction; "
                "exclude artifact serialization and CPU policy analysis"
            ),
            "equivalence_test": (
                "experiments/ph_online_memory_gono_v1/tests/test_trainer.py"
            ),
        },
        "model_hyperparameter_source_artifact": (
            "experiments/om_factorization_killtest/prereg.py:56-69"
        ),
        "training": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 256,
            "max_epochs": 30,
            "patience": 5,
            "validation_batch_size": 1024,
            "checkpoint_metric": "masked validation MSE of final mean prediction",
            "checkpoint_improvement_epsilon": 1e-9,
            "best_epoch_indexing": "zero_based",
            "train_origin_stride": 7,
            "arms_have_identical_budget": True,
        },
        "datasets": {
            "m5": {
                "source": "data/sales_train_evaluation.csv",
                "length": 1941,
                "preprocessing_source": (
                    "experiments.external_validity_screen.confirmatory_h2.m5_full"
                ),
                "availability_source": ["data/sell_prices.csv", "data/calendar.csv"],
                "availability_policy": (
                    "mask pre-availability train/validation targets; keep canonical "
                    "history and model train-scale zero padding"
                ),
                "stage_a_ids_source": "data/processed/series.parquet",
            },
            "favorita": {
                "source": "data/processed/favorita_full_pool.parquet",
                "raw_source": "data/train.csv",
                "length": 1688,
                "preprocessing_source": (
                    "experiments.external_validity_screen.favorita_independent.load_pool"
                ),
                "availability_policy": "raw; available_from=0 after full-pool construction",
                "stage_a_ids_source": "data/processed/favorita_series.parquet",
            },
        },
        "data_source_sha256": dict(data_source_sha256 or {}),
        "population": {
            "role": "independent natural development population",
            "stage_a_ids_excluded": True,
            "membership_recomputed_at_new_cutoff": True,
            "series_count_hard_coded": False,
        },
        "eligibility": {
            "primary_min_positive_train": 20,
            "sensitivity_min_positive_train": [15, 20, 30],
            "positive_definition": "y > 0",
            "interval": "model_train only",
            "sensitivity_execution": (
                "count membership and immutable ID hashes during population audit; "
                "do not fit models for thresholds 15 or 30"
            ),
        },
        "splits": {
            "m5": {
                "model_train": [0, 1717],
                "model_validation": [1717, 1745],
                "warmup_origin": 1745,
                "warmup_interval": [1745, 1773],
                "evaluation_origins": [1773, 1801, 1829, 1857, 1885, 1913],
                "last_interval": [1913, 1941],
            },
            "favorita": {
                "model_train": [0, 1464],
                "model_validation": [1464, 1492],
                "warmup_origin": 1492,
                "warmup_interval": [1492, 1520],
                "evaluation_origins": [1520, 1548, 1576, 1604, 1632, 1660],
                "last_interval": [1660, 1688],
            },
            "interval_encoding": "half_open",
            "warmup_excluded_from_final_metrics": True,
        },
        "model_seed": 0,
        "conditional_model_seeds": [1, 2],
        "bootstrap": {
            "draws": 2000,
            "seed": 20260904,
            "confidence_interval_percent": [2.5, 97.5],
            "unit": "series cluster",
            "cluster_contents": "all six evaluation origins for a sampled series",
            "paired_policy_resampling": (
                "all compared policy losses for a sampled series use the same "
                "cluster multiplicity within a draw"
            ),
            "direction_sampling": "resample series independently inside each target dataset",
            "macro_sampling": (
                "independently resample M5 and Favorita series clusters within each "
                "draw, compute both direction RIs, then average the two RIs"
            ),
            "uncertainty_scope": (
                "series uncertainty conditional on the six observed origins; not "
                "origin uncertainty"
            ),
            "seed_average_rule": (
                "compute policy loss within each model seed, average series-origin "
                "losses across seeds, then series-cluster bootstrap; never average predictions"
            ),
        },
        "controls": {
            "seed": 20260904,
            "C0": "shuffle paired Point/Hurdle loss values within resolved origin",
            "C1": "uniform random neighbors after same-series exclusion",
            "rng": "numpy.random.default_rng (PCG64)",
            "query_seed_derivation": (
                "SHA256 of UTF-8 NUL-separated [base_seed, *seed_parts]; "
                "interpret the first 8 digest bytes as an unsigned little-endian integer"
            ),
            "canonical_case_order": ["dataset_id", "series_id", "origin"],
            "C0_seed_parts": ["C0", "dataset_id", "resolved_origin"],
            "C0_sampling": (
                "within each dataset_id/resolved_origin group, apply one RNG "
                "permutation without replacement to intact Point/Hurdle loss pairs"
            ),
            "C1_seed_parts": ["C1", "dataset_id", "series_id", "query_origin"],
            "C1_sampling": (
                "uniform choice without replacement from canonical-order eligible "
                "memory rows after same-series exclusion"
            ),
        },
        "primary_loss": {
            "name": "train-scale normalized MSE",
            "scale_squared": "mean(y[0:model_train_end]^2) + 1e-8",
            "epsilon": 1e-8,
            "aggregation": "mean over 28 horizon steps, then series-origin mean",
        },
        "secondary_metrics": ["realized_y_RMSE", "realized_y_MAE"],
        "relative_improvement": "100 * (1 - mean_loss_A / mean_loss_B)",
        "policy_grids": {
            "b3_alpha": alpha_grid,
            "b4_eta": [0.5, 2.0, 8.0, 32.0],
            "b4_half_life_origins": [1, 3],
            "m1_k": [32, 128],
            "m1_lambda_max": [0.25, 0.5],
        },
        "policy_execution": {
            "source_target_directions": ["m5_to_favorita", "favorita_to_m5"],
            "source_tuning_origins": "six evaluation origins; warmup excluded",
            "target_hyperparameters_refit": False,
            "B0": "always Point; hurdle weight alpha=0",
            "B1": "always Hurdle; hurdle weight alpha=1",
            "B2": "50:50 convex forecast; hurdle weight alpha=0.5",
            "b3_forecast": "(1-alpha)*point + alpha*hurdle",
            "b4_discount": {
                "gamma_from_half_life": "gamma = 0.5 ** (1 / half_life_origins)",
                "age_definition": (
                    "integer count of completed forecast-origin intervals; the most "
                    "recent resolved case has age 1"
                ),
                "cumulative_loss": "sum(gamma**age * normalized_expert_loss)",
                "stable_hurdle_weight": "expit(eta * (S_point - S_hurdle))",
            },
            "first_evaluation_memory": "warmup case only",
        },
        "tie_breaking": {
            "B3": ["lower source mean loss", "grid order"],
            "B4": [
                "lower source mean loss",
                "lower source worst-origin loss",
                "smaller eta",
                "longer half-life",
            ],
            "M1": [
                "lower source mean loss",
                "lower source worst-origin loss",
                "smaller lambda_max",
                "larger k",
            ],
        },
        "retrieval": {
            "memory_value": ["point_resolved_normalized_loss", "hurdle_resolved_normalized_loss"],
            "memory_scope": "target dataset only; cases resolved before query origin",
            "same_series_excluded": True,
            "distance": "Euclidean after memory-only robust scaling",
            "distance_tie_break": (
                "deterministic resolved-memory row order after dataset, series_id, "
                "origin ordering; squared Euclidean may be used because it "
                "has the identical rank"
            ),
            "exact_search_engine": (
                "SciPy cKDTree with eps=0; batched query establishes the kth "
                "radius, query_ball_point includes every boundary tie, then exact "
                "squared Euclidean distance and canonical row index are lexsorted"
            ),
            "precomputation": (
                "extract each case feature once; fit one memory-only robust scaler "
                "and one exact tree per query origin; reuse a max-k=128 neighbor "
                "plan for k=32/128, lambda variants, M1, C0, and C1; C1 uses "
                "the shared plan only for canonical query/memory geometry and "
                "draws uniform random eligible neighbors"
            ),
            "neighbor_average": "uniform",
            "features": [
                {
                    "name": "recent_zero_ratio",
                    "formula": "mean(history == 0) over exactly 96 pre-origin steps",
                    "transform": "identity",
                    "missing": False,
                },
                {
                    "name": "time_since_last_positive",
                    "formula": "number of steps after the last positive; 0 if history[-1] > 0",
                    "transform": "log1p",
                    "missing": True,
                },
                {
                    "name": "mean_interarrival_gap",
                    "formula": "mean(diff(indices where history > 0))",
                    "transform": "log1p",
                    "missing": True,
                },
                {
                    "name": "interarrival_gap_cv",
                    "formula": "sample_std(gaps, ddof=1) / mean(gaps)",
                    "transform": "log1p",
                    "missing": True,
                },
                {
                    "name": "positive_demand_mean",
                    "formula": "mean(history[history > 0])",
                    "transform": "log1p",
                    "missing": True,
                },
                {
                    "name": "positive_demand_cv",
                    "formula": "sample_std(positive_history, ddof=1) / mean(positive_history)",
                    "transform": "log1p",
                    "missing": True,
                },
                {
                    "name": "recent_to_canonical_train_scale_rms_ratio",
                    "formula": (
                        "sqrt(mean(history**2)) / max(mean(y[0:model_train_end]), 1.0)"
                    ),
                    "transform": "log1p",
                    "missing": False,
                },
                {
                    "name": "point_hurdle_forecast_disagreement",
                    "formula": (
                        "sqrt(mean((hurdle_forecast-point_forecast)**2)) / "
                        "max(mean(y[0:model_train_end]), 1.0)"
                    ),
                    "transform": "log1p",
                    "missing": False,
                },
            ],
            "missing_policy": (
                "continuous value zero-filled after transform and paired with an explicit "
                "missing indicator; indicators are not robust-scaled"
            ),
            "feature_sufficiency": {
                "time_since_last_positive": ">=1 lookback positive",
                "mean_interarrival_gap": ">=2 lookback positives",
                "interarrival_gap_cv": ">=3 lookback positives",
                "positive_demand_mean": ">=1 lookback positive",
                "positive_demand_cv": ">=2 lookback positives and positive mean",
            },
            "scaler": {
                "fit_data": "resolved memory cases only; current query excluded",
                "center": "median",
                "scale": (
                    "75th minus 25th NumPy linear quantiles; replace exact zero IQR "
                    "with 1 and record constant flag"
                ),
            },
            "confidence": (
                "abs(mean_delta)/(abs(mean_delta)+population_sd_delta(ddof=0)+1e-8)"
            ),
            "blend": "(1-lambda_max*confidence)*B4 + lambda_max*confidence*local",
        },
        "smoke": {
            "dataset": "m5",
            "n_series": 200,
            "seed": 20260904,
            "strata": "4x4 quantile cells of availability-aware zero_ratio_train and log canonical train_scale",
            "allocation": "equal across nonempty cells; deterministic remainder by sorted cell",
            "scientific_result": False,
            "runtime_gate_gpu_hours": 6.0,
            "fallback_estimate_series_per_dataset": 2000,
            "forecast_origins": [1745, 1773],
            "pipeline_validation_hyperparameters": {
                "b4_eta": 0.5,
                "b4_half_life_origins": 1,
                "m1_k": 32,
                "m1_lambda_max": 0.25,
                "scientific_selection": False,
            },
            "runtime_projection": {
                "training": (
                    "per-arm measured M5 smoke train time multiplied by full-series "
                    "ratio and dataset train-window-count ratio"
                ),
                "inference": (
                    "per-arm measured M5 smoke inference time/origin multiplied by "
                    "full-series ratio and seven forecast origins"
                ),
                "retrieval": (
                    "measure the exact first-evaluation-origin B4+M1 call; report "
                    "CPU wall-time separately using quadratic series scaling and "
                    "resolved-origin pool growth sum(1..6)"
                ),
                "retrieval_passes_per_dataset": 7,
                "retrieval_pass_breakdown": {
                    "source_M1_grid": 4,
                    "target_real_M1": 1,
                    "target_C0": 1,
                    "target_C1": 1,
                },
                "gpu_gate_excludes_cpu_retrieval": True,
                "includes_arms": ["point", "hurdle"],
                "includes_datasets": ["m5", "favorita"],
            },
        },
        "stage0_reproduction": {
            "scientific_result": False,
            "raw_sources": {
                "m5": "results/external_validity_screen/rule_replication/independent_raw_predictions.parquet",
                "favorita": "results/external_validity_screen/favorita_independent/independent_raw_predictions.parquet",
            },
            "panel_rows": 33294,
            "condition_discovery_panel_atol": 1e-7,
            "recoverability_gain_atol": 5e-5,
            "frozen_sha256": {
                "m5_raw": "a2810787033baac622e6558a53da526f0cb9b2e80d09cf2c4dbe699d2f207f6f",
                "favorita_raw": "0df546bb479f70b8667c515c421ffe2a4f91dada9f1d50c0ee0300a776c22822",
                "condition_discovery_panel": "0917e4cc69948c72bf9c33afcccf969a8dc8bb2c74271bb00777cba1d5da532f",
                "recoverability_panel": "4c21015897e140de30177d7b8c97093c8ccadbf3ed00545ef8070e6206485e41",
            },
        },
        "heterogeneous_gate0_diagnostic": {
            "run_only_after_point_hurdle_gate0_failure": True,
            "model_seed_scope": "primary seed 0 only",
            "new_model_training": False,
            "expert_order": ["point", "hurdle", "tsb", "sba"],
            "expert_implementations": {
                "tsb_selector": "experiments.om_factorization_killtest.evaluate.select_tsb",
                "tsb_forecast": "experiments.om_factorization_killtest.models.tsb_forecast",
                "tsb_grid": "experiments.om_factorization_killtest.prereg.TSB_GRID",
                "sba_selector": "experiments.external_validity_screen.classical_benchmark.select_alpha",
                "sba_forecast": "experiments.external_validity_screen.classical_benchmark.croston_forecast",
                "sba_hand_check": "experiments.external_validity_screen.classical_benchmark.hand_check",
            },
            "classical_selection_data": "new frozen validation interval only",
            "forecast_origins": "warmup plus six evaluation origins; score six only",
            "pair_order": [
                ["point", "hurdle"],
                ["point", "tsb"],
                ["point", "sba"],
                ["hurdle", "tsb"],
                ["hurdle", "sba"],
                ["tsb", "sba"],
            ],
            "alpha_grid": alpha_grid,
            "candidate_definition": "(1-alpha)*expert_a + alpha*expert_b",
            "candidate_count": 126,
            "duplicate_pair_endpoints_preserved": True,
            "tie_break": "pair-major order, then alpha-grid order, first minimum",
            "global_static_loss": "minimum whole-panel mean over 126 candidates",
            "origin_oracle_loss": (
                "minimum candidate loss independently for each series-origin case, "
                "then average cases"
            ),
            "relative_gain_percent": "100*(1-origin_oracle/global_static)",
            "macro": "unweighted mean of M5 and Favorita relative gains",
            "pass_threshold_inclusive_percent": 2.0,
        },
        "gates": {
            "Gate0": {
                "metric": "origin convex oracle vs target-oracle best global static alpha",
                "origin_convex_oracle_definition": (
                    "for each series-origin case independently, choose the lowest-loss "
                    "alpha on the frozen grid, then average those case losses"
                ),
                "macro_min_percent": 2.0,
                "m5_min_percent": 1.0,
                "favorita_min_percent": 1.0,
                "on_failure": {
                    "diagnostic_only_experts": ["TSB", "SBA"],
                    "availability_rule": (
                        "use only an already reproducible current-repository implementation; "
                        "otherwise record NOT_AVAILABLE"
                    ),
                    "heterogeneous_origin_convex_macro_min_percent": 2.0,
                    "new_heterogeneous_method_implementation_forbidden": True,
                    "final_verdict_if_pass": (
                        "PH_ONLY_NO_GO_HETEROGENEOUS_EXPERT_CANDIDATE"
                    ),
                    "final_verdict_if_fail": "FULL_NO_GO",
                },
            },
            "Gate1A": {
                "m5_lag1_spearman_min_strict": 0.10,
                "favorita_lag1_spearman_min_strict": 0.10,
                "real_minus_shuffle_min_strict": 0.05,
            },
            "Gate1B": {
                "B4_vs_B3_each_transfer_min_percent": 0.30,
            },
            "Gate2": {
                "M1_vs_B4_macro_min_percent": 0.20,
                "M1_vs_B4_each_transfer_min_percent": -0.10,
                "M1_vs_B3_macro_min_percent": 0.70,
                "M1_vs_B3_each_transfer_min_percent": 0.30,
                "M1_vs_B4_macro_ci_lower_strict": 0.0,
                "at_least_one_dataset_ci_lower_strict": 0.0,
                "borderline_failure_status": "PENDING_GATE4",
                "deferrable_failed_checks": [
                    "macro_effect",
                    "direction_safety",
                    "macro_absolute_usefulness",
                    "direction_absolute_usefulness",
                    "macro_ci",
                    "dataset_ci",
                ],
                "borderline_deferral": (
                    "when Gate3 safety/control pass, seed0 macro is positive, and any "
                    "Gate4 borderline trigger holds, defer every seed0 Gate2 A-F "
                    "failure and re-evaluate every condition from seed-averaged losses; "
                    "a Gate2 failure is immediately terminal only when no Gate4 "
                    "borderline trigger holds"
                ),
            },
            "Gate3_safety": {
                "worst_origin_M1_vs_B3_min_percent": -0.50,
                "q95_M1_over_B4_max": 1.01,
            },
            "Gate3_control": {
                "real_retrieval_min_strict": 0.0,
                "shuffle_over_real_max": 0.25,
                "random_over_real_max": 0.50,
            },
            "Gate4": {
                "seed1_only_if_borderline": True,
                "seed1_forbidden_if_any": [
                    "seed0 M1_vs_B4_macro_percent <= 0",
                    "seed0 Gate3 safety FAIL",
                    "seed0 control condition FAIL",
                ],
                "borderline_if_any": [
                    "0 < seed0 M1_vs_B4_macro_percent <= 0.40",
                    "seed0 95% CI contains 0",
                    "seed0 95% CI crosses +0.20%",
                    "exactly one transfer direction is positive",
                ],
                "same_sign_required": True,
                "effect_retention_min": 0.70,
                "seed_average_ci_lower_strict": 0.0,
                "seed_average_construction": (
                    "average each policy's loss across seeds within identical "
                    "dataset/series/origin keys, then bootstrap the averaged losses"
                ),
                "seed1_clear_fail_if_any": [
                    "seed1 effect sign differs from positive seed0",
                    "abs(seed1 effect) / abs(seed0 effect) < 0.70",
                    "seed-average bootstrap CI upper <= 0",
                ],
                "seed1_still_borderline_if": (
                    "same sign AND retention >= 0.70 AND "
                    "seed-average CI lower <= 0 < upper"
                ),
                "seed2_only_if_still_borderline": True,
                "seed2_trigger": (
                    "run only when seed1_still_borderline_if is exactly true; "
                    "a sign/retention failure or nonpositive CI upper is terminal NO-GO"
                ),
                "three_seed_opposite_signs_max": 1,
                "three_seed_aggregation": (
                    "policy loss per seed first, then seed-average series-origin loss, "
                    "then bootstrap; prediction averaging forbidden"
                ),
                "additional_seed_upstream_gates": (
                    "Gate0 through Gate3 are diagnostic-only for seeds 1 and 2; "
                    "they must not early-stop construction of the M1 and B4 policy-loss "
                    "panels used by Gate4"
                ),
                "identical_key_set_required": True,
                "seed_average_full_gate2_recheck": (
                    "recompute every Gate2 A-F condition from seed-averaged B3, "
                    "B4, and M1 normalized policy losses"
                ),
                "terminal_pass_rule": (
                    "when Gate4 is invoked, both Gate4 robustness and the "
                    "seed-averaged Gate2 A-F evaluation must pass"
                ),
                "seed0_veto_precedence": (
                    "if raw seed0 Gate2 already passed, a Gate3 safety/control "
                    "veto keeps its own terminal verdict; if raw seed0 Gate2 failed "
                    "and was only provisionally deferred, either veto retracts the "
                    "deferral to Gate2 FINAL_FAIL and the failure-first verdict is "
                    "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO, while the Gate3 "
                    "measurement remains an eligibility observation"
                ),
                "resolution_path": (
                    "results/ph_online_memory_gono_v1/"
                    "gate2_gate4_resolution.json"
                ),
            },
        },
        "pairing": {
            "keys": ["dataset_id", "series_id", "origin", "step"],
            "expected": "one-to-one Point/Hurdle rows",
            "minimum_coverage": 0.999,
        },
        "oracle_families": {
            "hard": ["source global hard", "series hard oracle", "origin hard oracle"],
            "convex": ["source global convex", "series convex oracle", "origin convex oracle"],
            "cross_family_denominator_forbidden": True,
        },
        "final_verdict_tokens": verdicts,
        "stop_rules": {
            "HANDCRAFTED_FEATURE_GATE_STOP": "PRESERVED",
            "RAW_SEQUENCE_GATE_STOP": "PRESERVED",
            "ROUTING_MODEL_DEVELOPMENT_STOP": "PRESERVED",
            "DO_NOT_CONSUME_NEW_CONFIRMATORY_DATASET": "PRESERVED",
            "artifact_reproduction_failure": "STOP",
            "paired_coverage_below_0.999": "STOP",
            "runtime_projection_above_6_gpu_hours": "STOP_AND_REPORT_2000_SERIES_ESTIMATE",
            "gate_failure": "STOP_WITHOUT_RUNNING_LATER_STAGES",
            "design_error_after_freeze": "INVALIDATE_AND_START_NEW_VERSION",
        },
        "execution_order": [
            "repository/artifact audit and explicit resolution",
            "freeze preregistration and SHA256",
            "cached three-origin Stage 0 reproduction",
            "full frozen tests including CPU canonical-trainer equivalence fit",
            "M5 200-series CUDA smoke and runtime estimate",
            "conditional full seed execution only when GPU projection <= 6 hours",
        ],
        "append_only_artifacts": {
            "phase0_status_preserved": "results/ph_online_memory_gono_v1/STATUS.md",
            "resolved_terminal_status": (
                "results/ph_online_memory_gono_v1/STATUS_AFTER_RESOLUTION.md"
            ),
            "smoke_attempts": "smoke/attempt_NNNN; completion marker written last",
            "full_dataset_checkpoints": (
                "each Point/Hurdle arm is saved immediately under an append-only "
                "seedN/datasets/DATASET arm attempt with completion manifest last; "
                "the assembled dataset output is also completion-marked; verified completed "
                "attempts may be loaded on restart and files are never replaced"
            ),
            "final_gate_report": "exclusive create; never overwrite",
            "final_tables": (
                "tables_a_to_g.json is deterministically bound to the frozen "
                "preregistration, runtime estimate, final gate report, and seed "
                "analysis completion artifacts; identical restart reuse only"
            ),
            "finalization_manifest": (
                "finalization_manifest.json binds final report, tables, resolved "
                "status, and the final forbidden-artifact integrity check"
            ),
            "runtime_stop_status": (
                "when the >6 GPU-hour gate stops execution, write only "
                "STATUS_AFTER_RESOLUTION.md; scientific tables and final gate "
                "report remain absent"
            ),
        },
        "forbidden_primary_additions": [
            "Transformer",
            "GRU",
            "TCN",
            "foundation model",
            "Tweedie",
            "ZTNB",
            "gradient surgery",
            "joint expert training",
        ],
        "implementation_sha256": dict(implementation_sha256),
        "implementation_hash_scope": (
            "every Python file in experiments/ph_online_memory_gono_v1 plus the "
            "imported canonical model/trainer/windowing, external preprocessing, "
            "seed, and decomposition dependency files plus the preserved Phase 0 "
            "audit, authorization, pre-freeze execution resolution, and forbidden-"
            "artifact baseline extension and Gate2/Gate4 resolution enumerated "
            "by protocol.py; "
            "exact manifest equality is required before Stage 0 and smoke"
        ),
        "authorized_resolution": "results/ph_online_memory_gono_v1/audit_resolution.json",
        "execution_resolution": "results/ph_online_memory_gono_v1/execution_resolution.json",
        "forbidden_artifact_baseline_extension": (
            "results/ph_online_memory_gono_v1/"
            "forbidden_artifact_baseline_extension.json"
        ),
        "gate2_gate4_resolution": (
            "results/ph_online_memory_gono_v1/gate2_gate4_resolution.json"
        ),
    }


def freeze_preregistration(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    frozen = deepcopy(dict(spec))
    frozen.pop("preregistration_sha256", None)
    frozen.pop("preregistration_hash_contract", None)
    digest = payload_sha256(frozen)
    frozen["preregistration_hash_contract"] = HASH_CONTRACT
    frozen["preregistration_sha256"] = digest
    exclusive_write_json(Path(path), frozen)
    return frozen
