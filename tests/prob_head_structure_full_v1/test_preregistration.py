"""Behavioral contracts for the pre-fit freeze."""

from __future__ import annotations

import json
import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.prob_head_structure_full_v1.integrity import (
    ContractViolation,
    publish_completion_marker,
    reserve_or_resume_attempt,
)
from experiments.prob_head_structure_full_v1.preregistration import (
    build_preregistered_spec_v2,
    build_preregistered_spec_v3,
    build_preregistered_spec_v4,
    build_preregistration_payload,
    freeze_preregistration,
    invalidate_preregistration_before_fit,
    invalidate_preregistration_v2_before_fit,
    invalidate_preregistration_v3_before_fit,
    payload_sha256,
    promote_v4_candidate_to_authoritative,
    recover_preregistration_companion,
    verify_preregistration,
    write_preregistration_review_candidate,
)
from experiments.prob_head_structure_full_v1.evaluation import (
    CRPS_QUANTILE_GRID,
    PredictionIntegrityError,
)
from experiments.prob_head_structure_full_v1.pooling import (
    _require_p3_grid_coherence,
    invert_pooled_cdf,
)


class PreregistrationFreezeTests(unittest.TestCase):
    @staticmethod
    def _v4_candidate() -> dict:
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        v1 = root / "v1.json"
        freeze_preregistration(v1, {"version": 1})
        v2 = root / "v2.json"
        freeze_preregistration(v2, build_preregistered_spec_v2(v1))
        v3 = root / "v3.json"
        freeze_preregistration(v3, build_preregistered_spec_v3(v1, v2))
        candidate = build_preregistered_spec_v4(v1, v2, v3)
        directory.cleanup()
        return candidate

    def test_v4_candidate_freezes_adapter_and_student_architecture_without_publishing(self):
        """Catches a candidate whose unspecified activation/scale/init can change teacher or student capacity."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1 = root / "v1.json"; freeze_preregistration(v1, {"version": 1})
            v2 = root / "v2.json"; freeze_preregistration(v2, build_preregistered_spec_v2(v1))
            v3 = root / "v3.json"; freeze_preregistration(v3, build_preregistered_spec_v3(v1, v2))
            candidate = build_preregistered_spec_v4(v1, v2, v3)
        architecture = candidate["heads"]["architecture"]
        self.assertEqual(architecture["adapter"], "Linear(horizon,width)->SiLU->Linear(width,horizon*output_multiplicity)")
        self.assertEqual(architecture["mu_scale"], "multiply_train_only_RMS")
        self.assertEqual(architecture["dispersion_scale"], "unscaled")
        self.assertEqual(candidate["students"]["architecture"]["activation"], "SiLU")
        self.assertEqual(architecture["initialization"], "PyTorch_default_seeded_except_Tweedie_phi_p")

    def test_v4_freezes_exact_training_numerics_transforms_and_cross_api_ids(self):
        """Catches precision, masking, checkpoint, PMF-index, or dataset-ID drift before fitting."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["numerical_execution"],
            {
                "dtype": "float32 for training and inference",
                "forbidden": ["AMP", "autocast", "GradScaler", "TF32"],
                "torch_determinism": {
                    "use_deterministic_algorithms": {"enabled": True, "warn_only": False},
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    "silent_disable": "forbidden",
                    "unsupported_operation": "record exact operation, device, and reason; block only the affected branch",
                },
                "inference": {
                    "context": "torch.inference_mode",
                    "native_head_window_rows_per_batch": 16,
                    "native_head_q_chunk": 1,
                    "pooled_flat_case_chunk": 448,
                    "pooled_flat_case_identity": "16 window rows*28 horizon",
                    "pooled_q_chunk": 1,
                    "transfer": "immediate CPU transfer after each output chunk",
                },
                "Tweedie_series_resource_guard": {
                    "maximum_terms": 200000,
                    "clamp_or_truncate": "forbidden",
                    "terminal_action": "NUMERICAL_BRANCH_BLOCKED for the affected head",
                },
                "third_family_unavailable": {
                    "continue": "renormalized two-head diagnostic only",
                    "forbid_GO": [
                        "DISTRIBUTION_SPACE_DISTILLATION_GO",
                        "STRUCTURE_CONDITIONED_ROUTING_GO",
                        "DISAGREEMENT_SENSOR_GO",
                    ],
                },
            },
        )
        self.assertEqual(
            candidate["heads"]["parameterization_contract"],
            {
                "epsilon": 1e-6,
                "all_mu_normalized": "softplus(raw_mu)+epsilon",
                "all_mu_raw_count_scale": "mu_normalized*s_i",
                "NB_r": "softplus(raw_r)+epsilon",
                "HSNB_pi": "epsilon+(1-2*epsilon)*sigmoid(raw_pi)",
                "HSNB_r_pos": "softplus(raw_r_pos)+epsilon",
                "TWEEDIE_phi": "softplus(raw_phi)+epsilon",
                "TWEEDIE_p": "1.05+0.90*sigmoid(raw_p)",
                "TWEEDIE_raw_initialization": {"raw_phi": "inverse_softplus(1-epsilon)", "raw_p": 0.0},
            },
        )
        self.assertEqual(
            candidate["students"]["architecture"]["verified_trunk"],
            {
                "moving_average_kernel": 25,
                "maps": ["Linear(96,horizon) seasonal", "Linear(96,horizon) trend"],
                "shared_per_horizon_scalar_MLP": [1, 16, 22],
                "activation": "SiLU",
            },
        )
        self.assertEqual(
            candidate["training"]["teacher_loss_reduction"],
            {
                "numerator": "sum full negative log likelihood over the valid-target mask",
                "denominator": "valid target count",
                "all_masked_rows": "remove deterministically before batching",
            },
        )
        self.assertEqual(
            candidate["training"]["checkpoint_contract"],
            {
                "epoch_indexing": "1-based",
                "scheduled_validation_epochs": list(range(2, 31, 2)),
                "patience": "5 consecutive scheduled checks without strict improvement",
                "strict_improvement": "score<best_score with no epsilon",
                "exact_tie": "retain the earliest checkpoint",
                "terminal_check": "the stop-triggering scheduled check is final and no extra epoch or check runs",
                "last_epoch_check": "check the last actually executed or maximum epoch only when it was not already checked",
                "identity_binds": ["authoritative preregistration hash", "source manifest hash", "sample manifest hash", "head", "seed", "complete training configuration"],
            },
        )
        self.assertEqual(
            candidate["training"]["OOM_recovery"],
            {
                "automatic_retries": 1,
                "discard_partial_state": True,
                "restart": "same initial seed and identical batch order",
                "microbatch_factor": 0.5,
                "gradient_accumulation_factor": 2,
                "effective_batch_unchanged": True,
                "second_failure": "OOM_MODEL_BRANCH_BLOCKED",
            },
        )
        self.assertEqual(
            candidate["target_support"]["exact_likelihood_index_eligibility"],
            {
                "audit_tolerance_preserved": 1e-6,
                "requirement": "every count-likelihood observation must be literally integer-valued before PMF indexing",
                "near_integer_action": "COUNT_LIKELIHOOD_INDEX_AMBIGUITY_HARD_STOP",
                "round_or_cast": "forbidden",
                "enforcement": "data and integrity adapters recheck every later likelihood/evaluation target",
            },
        )
        self.assertEqual(candidate["real"]["canonical_dataset_ids"], {"M5": "m5", "Auto": "auto", "Carparts": "carparts", "RAF": "raf", "OnlineRetail": "online_retail"})
        self.assertEqual(
            candidate["real"]["FrozenPrimaryDatasetManifest"],
            {
                "dataset_id_domain": ["m5", "auto", "carparts", "raf", "online_retail"],
                "display_labels_are_report_only": True,
                "constructor_input": "the complete canonical audit payload",
                "digest": "SHA256 of canonical UTF-8 JSON audit payload",
                "arbitrary_caller_supplied_64hex": "forbidden",
                "verification": "recompute digest from bound payload and compare before every use",
            },
        )
        self.assertEqual(
            candidate["metrics"]["scientific_evaluator_contract"],
            {
                "confirmatory_quantile_source_whitelist": ["native_exact_or_numerical_inverse", "monotone_piecewise_common_grid"],
                "teacher_P1_P2_source": "native_exact_or_numerical_inverse",
                "student_P3_source": "monotone_piecewise_common_grid",
                "empirical_or_sample_helpers": "diagnostic only",
                "confirmatory_EvaluationResult_from_empirical_helpers": "forbidden",
                "required_guard_test": "diagnostic empirical helpers cannot construct a confirmatory EvaluationResult",
            },
        )

    def test_v4_freezes_methods_preflight_geometry_scalars_and_mask_reductions(self):
        """Catches post-result C geometry fallback, temporal scalar drift, or microbatch-weighted losses."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["sensors_and_actions"]["real_geometry"],
            {
                "inner_pair_rule": "integer current t>=lookback with [t,t+2*h) wholly inside model_train",
                "selection": "last eight non-overlapping valid current origins; if unavailable use eight unique evenly spaced valid integers with ascending tie resolution",
                "minimum_unique_current_origins": 4,
                "M5_inner_current_origins": [1465, 1493, 1521, 1549, 1577, 1605, 1633, 1661],
                "fit_rows": "all selected inner-pair series rows only",
                "validation_calibration_pair": "current t=validation.start; features may use realized validation [t,t+h), target is warmup [t+h,t+2*h); never fit on this pair",
                "outer_target_origins": "the six frozen evaluation origins",
                "outer_feature_current_origins": {
                    "m5": [1745, 1773, 1801, 1829, 1857, 1885],
                    "online_retail": [178, 206, 234, 262, 290, 318],
                },
                "online_retail_geometry": "zero valid inner pairs because maximum t=94 is below lookback=96",
                "online_retail_token": "REAL_C_SENSOR_GEOMETRY_BLOCKED",
                "online_retail_role": "exclude from C1 passing-dataset count; metrics undefined; never shorten frozen lookback or substitute a fit",
                "consequence": "continue M5 diagnostically; C1 deterministically FAIL because fewer than two eligible real datasets",
                "block_scope": "branch-local scientific geometry block, not global integrity failure",
                "class_eligibility": "for each target and dataset, both classes must occur in fit rows and in the fixed validation pair",
                "single_class_action": "classifier and metric undefined; target-dataset fails; no constant, threshold, or model substitution",
                "C1_primary_target": 1,
            },
        )
        self.assertEqual(
            candidate["routing"]["temporal_scalar_contract"],
            {
                "recent_window": "intersection of the trailing declared window with availability-valid observations strictly before the feature boundary",
                "time_since_last_positive": "origin-1-last_positive_index",
                "gap_window": "include a gap only when both event endpoints lie in the trailing-96 window",
                "gap_CV": "sample standard deviation ddof=1 divided by mean; minimum two gaps",
                "positive_CV": "sample standard deviation ddof=1 divided by mean; minimum two positive magnitudes",
                "autocorrelation": "lag-1 Pearson needs at least three source values and nonzero variance in both lag vectors",
                "teacher_component_variance": "population variance across teachers ddof=0 at each horizon, then mean over horizon",
                "undefined": "emit both the undefined scalar value and its dedicated missing indicator",
            },
        )
        self.assertEqual(candidate["routing"]["B0_weight_source"], "dataset-specific validation-frozen P2 global CDF weights regardless of whether P3 is the primary pool")
        self.assertEqual(
            candidate["routing"]["temperature_crossfit_contract"],
            {
                "k2": "no prior heldout origin exists, so use the fixed tie-first temperature 0.25",
                "k_ge_3_evaluation": "for each candidate temperature, expanding OOF predictions on origins 2..k-1 use only still-earlier origins",
                "k_ge_3_selection": "minimize mean OOF expected regret over origins 2..k-1; exact tie chooses the lower temperature",
                "fold_k_fit": "after selection, fit z(T) on origins 1..k-1 and produce heldout weights for origin k",
                "past_heldout_target_recalculation_or_backfill": "forbidden",
                "controls": "apply the identical strictly-prior nested selection to each controlled training fold",
            },
        )
        self.assertEqual(
            candidate["students"]["architecture"]["monotone_training_parameterization"],
            {
                "raw_base": "softplus(raw_base)",
                "increments": "20 positive softplus increments",
                "application": "used unchanged in training",
                "zero_consistency": "q<=p0 zeroing is validation/evaluation postprocessing only",
            },
        )
        self.assertEqual(
            candidate["students"]["loss_reduction"],
            {
                "target_mask": "the identical valid-target mask applies to every hard and soft loss component",
                "hard_BCE": "sum over valid target cells divided by valid target cell count",
                "hard_pinball": "sum over valid target cells and 21 quantiles divided by valid target cell count*21",
                "soft_BCE": "sum over valid target cells divided by valid target cell count",
                "soft_Huber": "sum over valid target cells and 21 quantiles divided by valid target cell count*21",
                "composition": "L_hard=hard_BCE_mean+hard_pinball_mean; L_soft=soft_BCE_mean+soft_Huber_mean; L=(1-lambda)*L_hard+lambda*L_soft",
                "microbatch_accumulation": "accumulate each component numerator and denominator across microbatches, then form means once for the effective batch",
                "masked_values": "exactly zero loss and zero gradient",
                "forbidden": "mean of microbatch means or any all-masked contribution",
            },
        )
        self.assertEqual(candidate["synthetic"]["C_SYN"]["scoring"]["threshold"], "train-half no-change scores at the 90th percentile")
        self.assertEqual(candidate["synthetic"]["C_SYN"]["scoring_operational"]["first_origin_delta"], "undefined value plus missing indicator")
        self.assertEqual(candidate["synthetic"]["C_SYN"]["scoring_operational"]["zero_train_pre_component_SD"], "component separation FAIL")

    def test_v4_freezes_nested_router_C_disagreement_and_action_boundary_details(self):
        """Catches underspecified first-fold routing, target-contaminated disagreement, or post-result action fixes."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["routing"]["temperature_crossfit_contract"],
            {
                "k2": "no prior heldout origin exists, so use the fixed tie-first temperature 0.25",
                "k_ge_3_evaluation": "for each candidate temperature, expanding OOF predictions on origins 2..k-1 use only still-earlier origins",
                "k_ge_3_selection": "minimize mean OOF expected regret over origins 2..k-1; exact tie chooses the lower temperature",
                "fold_k_fit": "after selection, fit z(T) on origins 1..k-1 and produce heldout weights for origin k",
                "past_heldout_target_recalculation_or_backfill": "forbidden",
                "controls": "apply the identical strictly-prior nested selection to each controlled training fold",
            },
        )
        self.assertEqual(
            candidate["routing"]["all_missing_fold_feature"],
            {
                "imputed_value": "fixed zero in raw feature units",
                "missing_indicator": 1,
                "scaler": {"mean": 0.0, "scale": 1.0},
                "record": "ALL_MISSING_TRAIN_FEATURE",
                "later_nonmissing_transform": "preserve its observed raw value before the frozen scaler",
                "outer_or_future_statistics": "forbidden",
            },
        )
        sensor = candidate["sensors_and_actions"]
        self.assertEqual(
            sensor["disagreement_reduction"],
            {
                "D_zero_D_center_D_tail_D_mean": "population variance across teachers ddof=0 at each horizon step, then arithmetic mean over horizon",
                "D_cdf": "at each (dataset,series,origin,step), support={0} union {Q_m,step(q) for every teacher m and common-grid q}; average absolute CDF difference across the three teacher pairs and that step support, then average over horizon",
                "delta": "current origin-level component minus the same-series immediately previous origin t-h component; missing when unavailable",
            },
        )
        csyn = candidate["synthetic"]["C_SYN"]
        self.assertEqual(csyn["features"], ["D_zero", "D_center", "D_tail", "D_cdf", "Delta D_zero", "Delta D_center", "Delta D_tail", "Delta D_cdf"])
        self.assertEqual(
            csyn["feature_preprocessing"],
            {
                "first_origin_delta": "undefined value plus missing indicator; retain all 16 origins",
                "fit": "median imputation and standardization fit on the train half only",
                "application": "apply frozen train-half medians and moments to heldout rows",
                "real_only_components_excluded": ["D_mean", "D_winner_entropy"],
            },
        )
        self.assertEqual(
            sensor["C3_scalar_contract"],
            {
                "formula": "RMS of six sensor-training-standardized disagreement levels plus absolute values of four sensor-training-standardized deltas",
                "moments": "sensor-training rows only",
                "zero_standard_deviation": "use scale 1 and record the component",
            },
        )
        self.assertEqual(
            sensor["action_policy"],
            {
                "primary_score": "C2 logistic classifier probability for target 1",
                "validation_threshold": "q80(method='higher') of validation scores; flag iff score>threshold with no tie reranking",
                "zero_flagged_validation_rows": "metrics undefined and C3 FAIL; no reselection or substitution",
                "C_A0": "keep validation-selected P0 single teacher",
                "C_A1": "switch to validation-selected primary P2/P3 pool",
                "C_A2": "for every requested nonmedian quantile including 0.025 and 0.975 use max(0,Q50+f*(Qq-Q50)); preserve q50 and p0, then zero-consistency and cumulative maximum",
                "C_A2_factor_grid": [1.05, 1.10, 1.20, 1.35, 1.50],
                "C_A2_factor_selection": "on validation flagged rows minimize mean(CE90,CE95), then sCRPS, then smaller factor",
                "C_A2_predictive_mean": "recompute from the adjusted 21-grid quantiles by endpoint-hold trapezoidal integration; record mean_source=quantile_integral_endpoint_hold",
                "C_A3_selection": "over all validation rows minimize mean(CE90,CE95), then sCRPS, then fixed NB<HSNB<TWEEDIE_FULL; freeze for outer",
                "candidate_selection": "on validation flagged rows minimize sCRPS, then mean(CE90,CE95), then fixed C_A0<C_A1<C_A2<C_A3",
                "allow_no_action": True,
                "outer_application": "apply the frozen action only to flagged outer rows without reselection",
                "selective_coverage": "recall of target-1 failures captured by top-20%-score action cases",
            },
        )

    def test_v4_freezes_control_mechanics_and_insufficient_variation_failures(self):
        """Catches non-comparable controls, all-zero metrics, or a controlled router seeing future folds."""
        candidate = self._v4_candidate()
        controls = candidate["negative_controls"]
        self.assertEqual(
            controls["B_control_training"],
            {
                "temperature": "for each control use the same strictly-prior nested expanding-crossfit selection on its controlled router training data",
                "heldout_weights": "produce origin-k weights without backfilling any prior heldout row",
                "student": "fit controlled B2 at lambda 0.25,0.50,0.75 on the identical B train rows and select lambda on validation only",
                "final_router": "may refit on all inner rows only for validation/outer use; cannot rewrite B training soft targets",
            },
        )
        self.assertEqual(
            controls["implementations"]["C no-change"],
            "on heldout no-change sequences assign calendar pseudo-label 1[origin>=288] and score with the already-fitted frozen change sensor",
        )
        self.assertEqual(controls["effect_conventions"]["C no-change"], "AUPRC_calendar_pseudo_label-pseudo_prevalence")
        self.assertEqual(candidate["routing"]["B1_gate_statistic"]["undefined"], "INSUFFICIENT_VARIATION and B1 FAIL; never encode as zero")
        self.assertEqual(candidate["sensors_and_actions"]["single_class_metric"], "undefined and the relevant target-dataset gate FAIL; never encode as zero or fit a constant fallback")
        self.assertEqual(
            candidate["sensors_and_actions"]["C0_window_and_mask_contract"],
            {
                "zero_ratio_and_scale_windows": "intersect each horizon window with availability-valid observed targets",
                "partial_availability": "reduce every C0 horizon statistic over its valid-target mask only and record its count",
                "last_event_gap": "g(b)=b-1-last_positive_index using availability-valid history strictly before b; missing if no prior positive; feature=(g(t+h)-g(t))/h",
            },
        )

    def test_v4_is_a_clean_protocol_not_a_layered_alias_patch(self):
        """Catches stale v2/v3 aliases surviving into the candidate and contradicting the user protocol."""
        candidate = self._v4_candidate()
        serialized = json.dumps(candidate, sort_keys=True)
        self.assertNotIn("A3_A4", serialized)
        self.assertNotIn("delay_hours", serialized)
        self.assertNotIn("next_origin_targets", serialized)
        self.assertNotIn("oracle pool diagnostic", serialized)
        self.assertNotIn("MINIMAL_COMPLETE", serialized)
        self.assertNotIn("Stage S1 teachers or a global teacher", serialized)
        self.assertNotIn("numerical_inverse_empirical_CDF", serialized)
        self.assertNotIn("enable where available", serialized)
        self.assertNotIn("read-only pallet-pose environment", serialized)
        self.assertEqual(candidate["version"], 4)
        self.assertEqual(candidate["authority_status"], "REVIEW_CANDIDATE_NOT_FROZEN")
        self.assertEqual(
            candidate["research_objective"],
            {
                "primary_question": "간헐 수요 시계열에서 서로 다른 predictive distribution family가 발생 간격과 발생 크기의 시간 구조에 따라 실제로 전문화되는가?",
                "subquestions": [
                    "여러 distribution teacher의 CDF/quantile 정보를 하나의 경량 student로 압축할 수 있는가?",
                    "과거 관측에서 계산 가능한 시간 구조에 따라 어떤 teacher의 distillation objective를 신뢰할지 결정할 수 있는가?",
                    "teacher disagreement를 사용해 다음 시점의 오차·miscalibration·distribution shift를 미리 감지할 수 있는가?",
                ],
                "required_judgments": [
                    "HEAD_SPECIALIZATION_GO / HEAD_SPECIALIZATION_NO_GO",
                    "DISTRIBUTION_SPACE_DISTILLATION_GO / DISTRIBUTION_SPACE_DISTILLATION_NO_GO",
                    "STRUCTURE_CONDITIONED_ROUTING_GO / STRUCTURE_CONDITIONED_ROUTING_NO_GO",
                    "DISAGREEMENT_SENSOR_GO / DISAGREEMENT_SENSOR_NO_GO",
                ],
                "terminal_decision": "select exactly one final recommended research axis from the four judgments",
            },
        )
        self.assertEqual(
            candidate["identity"],
            {
                "experiment": "PROB-HEAD-STRUCTURE-FULL-v1",
                "branch": "prob-head-structure-full-v1",
                "base_commit": "2fe2443b2d24c09ad184387b7f7287f32e0f4cd6",
                "origin_main_commit": "2fe2443b2d24c09ad184387b7f7287f32e0f4cd6",
                "source_repository": "https://github.com/CanelE452/covariate-trust-pilot",
                "research_axes_in_priority_order": [
                    "D_temporal_structure_and_distribution_head_complementarity",
                    "A_heterogeneous_distribution_teacher_distillation",
                    "B_structure_conditioned_distillation_objective_routing",
                    "C_teacher_disagreement_failure_change_sensor",
                ],
            },
        )
        self.assertEqual(
            candidate["sensors_and_actions"]["targets"],
            [
                "next-origin sCRPS가 dataset 내 상위 20%인지",
                "next-origin 90/95% interval undercoverage",
                "next-origin zero-probability calibration error가 dataset 내 상위 20%인지",
                "best teacher identity가 직전 origin에서 바뀌는지",
            ],
        )

    def test_v4_freezes_required_pre_fit_repository_reads_and_no_guess_rule(self):
        """Catches implementation starting from guessed adapters instead of the named repository evidence."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["pre_fit_repository_audit"],
            {
                "fetch_first": True,
                "record": ["actual HEAD", "actual origin/main"],
                "required_reads": [
                    "_docs/PROJECT_LOG.md",
                    "results/ph_online_memory_gono_v1/STATUS_AFTER_RESOLUTION.md",
                    "results/ph_online_memory_gono_v1/tables_a_to_g.json",
                    "results/ph_online_memory_gono_v1/preregistered_spec.json",
                    "results/paper_synthesis_verified/claim_ledger_frozen.md",
                    "results/pointhurdle_recoverability/claim_ledger_recoverability.md",
                    "results/synthetic_source_verification/",
                    "current synthetic DGP provenance and generator",
                    "existing Point/Hurdle DLinear implementation",
                    "existing M5/Favorita preprocessing and split implementation",
                ],
                "adapter_rule": "inspect actual functions and columns before writing adapters; never guess unseen names",
            },
        )

    def test_v4_freezes_strict_source_records_environment_truth_and_temporal_prefixes(self):
        """Catches source substitution, overstated environment health, or future descriptors in inner rows."""
        candidate = self._v4_candidate()
        source = candidate["source_snapshot_contract"]
        self.assertEqual(
            source["worktree_manifest"],
            {
                "path": "results/prob_head_structure_full_v1/audit/source_manifest_before.json",
                "repository_root_identity": {"label": "isolated_execution_worktree", "resolved_path": "E:/CODING/worktrees/covariate-trust-pilot-prob-head-structure-full-v1"},
                "expected_sha256_by_path": {
                    "runs/prob_head_structure_full_v1/source_snapshots/m5/sales_train_evaluation.csv": "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0",
                    "runs/prob_head_structure_full_v1/source_snapshots/m5/calendar.csv": "d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5",
                    "runs/prob_head_structure_full_v1/source_snapshots/m5/sell_prices.csv": "9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef",
                    "runs/prob_head_structure_full_v1/source_snapshots/m5/series.parquet": "aa8d96ecd6ed6eaa91274087b4b90880b5da4ec3954962d94d57e37947f13aba",
                    "runs/prob_head_structure_full_v1/source_snapshots/online_retail/online_retail_II.xlsx": "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980",
                    "runs/prob_head_structure_full_v1/source_snapshots/generator/dgp.py": "0ad1b5607dd75bfe0c825fef778e55b17bf6667d16849a30beab295a66fd3e71",
                    "experiments/prob_head_structure_full_v1/vendor/tweediegp/tweedie.py": "58790c0ad5b927fa2e3cfbc2a4c8b4b82fc8fd988ce0ce2e216b8ab162650ce7",
                    "experiments/prob_head_structure_full_v1/vendor/tweediegp/LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
                    "experiments/prob_head_structure_full_v1/vendor/tweediegp/NOTICE": "a73a18f1b1f6cc3011b71280b7309f614e956bca53f14eac2221ad5cbc2cd822",
                    "requirements-prob-head-structure-full-v1.lock": "7a8708e966140e1a5eccea104937fb7e99e6af9c845372404ca55ae50dd173f8",
                },
                "strict_verifier": "verify_source_manifest_path",
            },
        )
        self.assertEqual(
            source["original_source_record"],
            {
                "path": "results/prob_head_structure_full_v1/audit/original_source_record_before.json",
                "files": {
                    "E:/CODING/proj/covariate-trust-pilot/data/sales_train_evaluation.csv": {"bytes": 121736518, "expected_sha256": "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0"},
                    "E:/CODING/proj/covariate-trust-pilot/data/calendar.csv": {"bytes": 103469, "expected_sha256": "d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5"},
                    "E:/CODING/proj/covariate-trust-pilot/data/sell_prices.csv": {"bytes": 203395785, "expected_sha256": "9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef"},
                    "E:/CODING/proj/covariate-trust-pilot/data/processed/series.parquet": {"bytes": 2421954, "expected_sha256": "aa8d96ecd6ed6eaa91274087b4b90880b5da4ec3954962d94d57e37947f13aba"},
                    "E:/CODING/proj/covariate-trust-pilot/data/online_retail_ii/online_retail_II.xlsx": {"bytes": 45622278, "expected_sha256": "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"},
                    "E:/research_recovery/m5dataset_recovery-20260811_135203/extracted/m5dataset/experiments/temporal_dependence/dgp.py": {"bytes": 8471, "expected_sha256": "0ad1b5607dd75bfe0c825fef778e55b17bf6667d16849a30beab295a66fd3e71"},
                },
                "strict_verifier": "verify_original_source_record_path",
                "rehash_timing": ["before every resumed fit", "finalization"],
                "path_substitution": "forbidden",
                "mutation_action": "DATA_SOURCE_MUTATION hard stop",
            },
        )
        self.assertEqual(
            candidate["environment"]["isolation_truth"],
            {
                "mode": "isolated venv overlay with system-site-packages over the pallet-pose environment",
                "base_environment_policy": "procedure-treated read-only; OS ACL read-only protection is not asserted",
                "overlay_changes": "only new overlay dependencies were installed",
                "base_environment_mutated": False,
                "overlay_pip_check_warnings": ["missing chronos-forecasting", "huggingface-hub 1.27 conflicts with project requirement <1"],
                "base_pip_check_warnings": ["unsupported Rtree", "unsupported ninja"],
                "warning_scope": "pre-existing or repository-wide out-of-scope dependencies not imported by this pilot",
                "execution_gate": "enumerated required pilot imports, both torch/numpy import orders, and recorded hashes only",
                "claim_globally_clean_environment": "forbidden",
            },
        )
        self.assertEqual(
            candidate["routing"]["feature_prefix_contract"],
            {
                "inner_train_descriptors": "for origin o recompute ADI, positive CV2, zero ratio, and scale on availability-valid [available_from,o)",
                "inner_recent_features": "observations strictly before o",
                "inner_imputation": "fit only on prior expanding-crossfit rows",
                "outer_train_descriptors": "frozen availability-valid [available_from,model_train_end)",
                "outer_recent_features": "observations strictly before o",
                "sensor_prequential_features": "after [o,o+h) is realized, observations strictly before o+h",
                "fixed_teacher_exception": "fixed teacher checkpoint may have been trained on full model_train as explicitly allowed",
                "forbidden": "no actual inner or outer future target may enter router/sensor features or their normalization",
                "future_sentinel_test": "mutating values at or after the applicable boundary must leave every feature byte-identical",
            },
        )
        self.assertEqual(
            candidate["replicate_contract"],
            {
                "FULL_real_selection": "select P0 identity and P2/P3 weights once per dataset from validation metrics averaged equally over teacher model seeds",
                "FULL_real_application": "apply the selected identity/weights unchanged within each teacher-seed replicate",
                "student_teacher_pairing": {"student_replicate_0": 2026090511, "student_replicate_1": 2026090512},
                "cross_seed_CDF_averaging_or_extra_ensemble": "forbidden",
                "within_seed_scope": ["router and regret", "disagreement", "controls", "student soft targets"],
                "real_aggregation": "average seed replicate within series-origin before series and dataset",
                "real_bootstrap": "resample series with both model-seed replicates attached",
                "COMPACT_and_MINIMAL-COMPLETE": "single preregistered seed replicate",
                "synthetic": "retain every preregistered data-seed by model-seed replicate and average replicates before cell Gate summaries",
                "synthetic_bootstrap": {
                    "cluster_identity": ["data_seed", "d", "base_series_index"],
                    "attachment": "all model seeds, origins, and the nine rho_I by rho_M cells at fixed (data_seed,d)",
                    "cross_d_pairing": "forbidden; d=4 and d=8 use distinct common bases",
                    "S3": "resample d-specific clusters; each data seed still contributes all 18 cells overall",
                    "canonical_columns": ["data_seed", "d", "rho_I", "rho_M", "base_innovation_id"],
                },
            },
        )
        self.assertEqual(
            candidate["synthetic"]["generator"],
            {
                "event_state": "symmetric stationary Markov state in {-1,+1}",
                "stay_probability": "(1+rho)/2",
                "stationary_state_probability": 0.5,
                "gap_support": "{d-1,d+1}",
                "magnitude": "1+Poisson(lambda-1)",
                "magnitude_lambdas": [5, 15],
                "independent_interval_and_magnitude_streams": True,
                "common_base_innovations_across_cells": "within each fixed (data_seed,d), shared across the nine rho_I by rho_M cells",
                "cross_d_common_base": "not present; d is part of the base RNG/fingerprint identity",
                "canonical_output_columns": ["data_seed", "d", "rho_I", "rho_M", "base_innovation_id"],
                "adapter_mapping": {"rho_interval": "rho_I", "rho_magnitude": "rho_M"},
                "adapter_rule": "map the inspected generator arguments explicitly; never guess names",
            },
        )

    def test_v4_freezes_exact_A_B_C_methods_losses_features_and_controls(self):
        """Catches a method implementation being relabelled or given a result-sensitive unregistered objective."""
        candidate = self._v4_candidate()
        self.assertEqual(
            {key: candidate["students"][key] for key in ["A0", "A1", "A2", "A3", "A4"]},
            {
                "A0": "hard_only_student",
                "A1": "best_single_teacher_distillation",
                "A2": "equal_pool_teacher_distillation",
                "A3": "validation_selected_global_CDF_pool_distillation",
                "A4": "quantile_specific_teacher_pool_distillation",
            },
        )
        self.assertEqual(candidate["students"]["architecture"]["per_horizon_shared_mlp"], [1, 16, 22])
        self.assertEqual(candidate["students"]["architecture"]["predictive_mean"], {"method": "trapezoidal integral of unconditional monotone quantiles", "lower_extension": "hold Q(0.01) constant on [0,0.01]", "upper_extension": "hold Q(0.99) constant on [0.99,1]", "report": "mean_source=quantile_integral_endpoint_hold", "separate_mean_head": False})
        self.assertEqual(candidate["pools"]["P3"]["p0"], {"weights": "additional validation-selected 66-state simplex", "selection_metric": "validation Brier score for Y=0", "pool": "convex teacher p0"})
        self.assertEqual(candidate["pools"]["P3"]["postprocess"], ["cumulative maximum across quantiles", "set every grid quantile with q<=pooled p0 to zero", "cumulative maximum again"])
        self.assertEqual(candidate["pools"]["P3"]["adjustment_report"], ["pre-projection crossing rate", "post-projection crossing rate", "zero-consistency adjustment rate"])
        self.assertEqual(candidate["pools"]["P2"]["selection_order"], ["minimum validation sCRPS", "minimum validation mean(q95,q99 sQL)", "lexicographically smallest (w_NB,w_HSNB,w_TWEEDIE_FULL)"])
        self.assertEqual(candidate["pools"]["P3"]["selection_order"], ["minimum postprocessed validation sCRPS", "minimum postprocessed validation mean(q95,q99 sQL)", "smaller penalty", "lexicographically smallest full path"])
        self.assertEqual(candidate["pools"]["primary_selection"], ["minimum postprocessed validation sCRPS", "minimum postprocessed validation mean(q95,q99 sQL)", "P2 on exact tie"])
        self.assertEqual(
            candidate["students"]["loss"],
            {
                "hard": "BCE(1[Y=0],p0_student)+mean_q Pinball(y,Q_student(q)) on raw targets and raw quantiles",
                "soft_zero": "Bernoulli soft cross-entropy(p0_teacher,p0_student), KL-equivalent up to the teacher constant",
                "soft_quantiles": "mean_q Huber(Q_teacher(q)/s_i,Q_student(q)/s_i) with delta=1.0 in scaled quantile units",
                "combined": "(1-lambda)*L_hard+lambda*L_soft",
                "lambda_grid": [0.25, 0.5, 0.75],
                "lambda_selection": ["minimum validation sCRPS", "smaller lambda on exact tie"],
                "selection_data": "validation only",
            },
        )
        self.assertEqual(
            [candidate["routing"][key] for key in ["B0", "B1", "B2"]],
            [
                "global_teacher_weight_distillation",
                "baseline_feature_conditioned_distillation",
                "temporal_structure_conditioned_distillation",
            ],
        )
        self.assertEqual(
            candidate["routing"]["temporal_features"],
            [
                "recent zero ratio", "train zero ratio", "time since last positive",
                "recent mean gap", "recent gap CV", "train ADI", "train positive CV²",
                "recent positive mean", "recent positive CV", "recent/train scale ratio",
                "interval autocorrelation", "magnitude autocorrelation",
                "seasonal phase if known and future-known", "missing indicator for every missing feature",
            ],
        )
        self.assertEqual(
            candidate["routing"]["inner_origin_selection"],
            {
                "valid_origins": "every integer o with lookback<=o<=model_train_end-horizon",
                "primary": "last eight non-overlapping valid origins when all eight exist",
                "fallback": "eight unique nearest-integer origins evenly spaced over the valid range; resolve ties and duplicates in ascending order",
                "minimum_unique": 4,
                "below_minimum": "block B branch",
                "M5": [1493, 1521, 1549, 1577, 1605, 1633, 1661, 1689],
                "OnlineRetail": [96, 100, 103, 107, 111, 115, 118, 122],
            },
        )
        self.assertEqual(candidate["routing"]["temporal_feature_definitions"], {"recent zero ratio": "trailing horizon=28", "recent/train scale ratio": "trailing-28 RMS divided by availability-valid train RMS", "recent mean gap": "events within trailing lookback=96", "recent gap CV": "events within trailing lookback=96", "recent positive mean": "positives within trailing lookback=96", "recent positive CV": "positives within trailing lookback=96", "interval autocorrelation": "lag-1 Pearson on most recent up to 20 gaps in history; missing if insufficient values or zero variance", "magnitude autocorrelation": "lag-1 Pearson on most recent up to 20 positive magnitudes in history; missing if insufficient values or zero variance", "time since last positive": "all observations strictly before origin", "train features": "availability-valid model_train observations only", "seasonal phase": "future-known weekly sin/cos with period 7 for M5 and OnlineRetail"})
        self.assertEqual(candidate["routing"]["imputation_contract"], "fit fold-train medians after adding one indicator per undefined scalar; never fit medians on outer evaluation rows")
        self.assertEqual(candidate["estimators"], {
            "preprocess": "fold-train median plus missing indicators then fold-train StandardScaler",
            "router_primary": {"model": "multinomial logistic regression", "penalty": "L2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "seed": "derived deterministic model seed"},
            "router_secondary": {"model": "three independent HistGradientBoostingRegressor models, one per z_m", "learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 15, "l2_regularization": 1.0, "seed": "derived deterministic model seed", "postprocess": "clip each prediction to [0,1], renormalize each row to the simplex, and map an all-zero row to [1/3,1/3,1/3]", "role": "secondary only"},
            "sensor_primary": {"model": "one logistic regression per binary target", "penalty": "L2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "seed": "derived deterministic model seed"},
            "sensor_secondary": {"model": "one HistGradientBoostingClassifier per binary target", "learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 15, "l2_regularization": 1.0, "seed": "derived deterministic model seed", "role": "secondary only"},
            "post_result_primary_replacement": "forbidden",
            "search": "no estimator hyperparameter search beyond explicitly preregistered grids",
        })
        self.assertEqual(candidate["routing"]["temperature_selection_order"], ["lowest expanding-crossfit expected regret", "lower temperature on exact tie"])
        self.assertEqual(
            candidate["sensors_and_actions"]["components"],
            ["D_zero", "D_center", "D_tail", "D_cdf", "D_mean", "D_winner_entropy"],
        )
        self.assertEqual(candidate["sensors_and_actions"]["classifier_scope"], "fit one classifier independently for each of the four targets")
        self.assertEqual(candidate["sensors_and_actions"]["D_cdf_grid"], "for each (dataset,series,origin,step), sorted unique target-free set {0} union every teacher's step-specific Q(q) on the common grid")
        self.assertEqual(candidate["sensors_and_actions"]["D_winner_entropy_contract"], "a_m is teacher m's mean CDF distance to the other teachers on the target-free D_cdf grid; z=softmax(-a_m/0.5); D_winner_entropy=-sum_m z_m*log(z_m); no realized current target")
        self.assertEqual(candidate["sensors_and_actions"]["C3_scalar"], "root mean square of sensor-training-standardized six disagreement levels plus absolute values of four sensor-training-standardized deltas")
        self.assertEqual(candidate["sensors_and_actions"]["C0_feature_definitions"], {"previous_realized_residual": "mean-horizon absolute P0 predictive-mean residual divided by s_i", "zero_ratio_change": "current minus previous horizon zero rate", "scale_change": "current scaled RMS minus previous scaled RMS", "last_event_gap_change": "availability-valid gap difference divided by h", "recent_target_variance": "population variance over at most 96 observations ending t+h divided by s_i^2"})
        self.assertEqual(
            candidate["negative_controls"],
            {
                "A": ["teacher identity shuffle", "teacher quantile shuffle", "single-teacher soft target"],
                "B": ["regret label shuffle", "temporal feature row shuffle", "missing indicator 제거 diagnostic"],
                "C": ["time shuffle", "teacher 이름 permutation", "scale-only feature", "random sensor score", "no-change synthetic sequence"],
                "seed": 2026090551,
                "identification_failure_rule": "control이 real effect의 50% 이상 회수하면 신호 식별 실패",
                "implementations": {
                    "A teacher identity shuffle": "within each dataset/fold soft-target train row, deterministically permute the three complete teacher identities jointly across p0, every quantile, and mean while weights stay attached to original labels",
                    "A teacher quantile shuffle": "within dataset/fold/head, permute each whole monotone quantile vector across soft-target train rows with one permutation shared across q; p0 unchanged",
                    "A single-teacher soft target": "A1 diagnostic",
                    "B regret label shuffle": "within dataset/fold train rows, permute the complete three-regret vector",
                    "B temporal feature row shuffle": "within dataset/fold train rows, permute the complete extended-feature plus missing-indicator row",
                    "B missing indicator removal": "diagnostic",
                    "C time shuffle": "within dataset and series, permute the complete disagreement-level plus delta vector across eligible origins",
                    "C teacher name permutation": "one global teacher permutation; require component maxabs difference<=1e-12",
                    "C scale-only": "train RMS only",
                    "C random score": "deterministic U[0,1) generated per frozen row key",
                    "C no-change": "on heldout no-change sequences assign calendar pseudo-label 1[origin>=288] and score with the already-fitted frozen change sensor",
                },
                "boundaries": {"keys": "never shuffled", "outer_labels": "never shuffled", "fit_controls": "training rows only", "score_controls": "may replace evaluation scores without fitting"},
                "B_control_training": {
                    "temperature": "for each control use the same strictly-prior nested expanding-crossfit selection on its controlled router training data",
                    "heldout_weights": "produce origin-k weights without backfilling any prior heldout row",
                    "student": "fit controlled B2 at lambda 0.25,0.50,0.75 on the identical B train rows and select lambda on validation only",
                    "final_router": "may refit on all inner rows only for validation/outer use; cannot rewrite B training soft targets",
                },
                "effect_conventions": {
                    "A": "(L_A0-L_primary_A)/L_A0",
                    "B": "(L_B0-L_B2)/L_B0",
                    "C": "AUPRC_C2-AUPRC_C0 for target1",
                    "C_SYN": "AUPRC_change-prevalence",
                    "C no-change": "AUPRC_calendar_pseudo_label-pseudo_prevalence",
                    "macro": "equal weight per dataset",
                    "control_effect": "same formula as its matching real effect",
                },
                "recovery_rule": {"real_effect_nonpositive": "identification FAIL", "otherwise": "control_effect/real_effect", "FAIL_if_recovery_at_least": 0.5},
                "roles": {
                    "teacher 이름 permutation": "label-symmetry integrity control; require byte/numerical invariance",
                    "missing indicator 제거 diagnostic": "diagnostic only",
                },
                "fifty_percent_rule_applies_to": [
                    "A teacher identity shuffle", "A teacher quantile shuffle",
                    "B regret label shuffle", "B temporal feature row shuffle",
                    "C time shuffle", "C scale-only feature", "C random sensor score", "C no-change synthetic sequence",
                ],
                "fifty_percent_rule_excludes": ["A single-teacher soft target", "B missing indicator 제거 diagnostic", "C teacher 이름 permutation"],
            },
        )

    def test_v4_freezes_deployable_teacher_scope_and_synthetic_oracle_semantics(self):
        """Catches outer-oracle leakage into deployable teachers or per-cell post-result pool selection."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["pools"]["P0"],
            {
                "type": "validation-selected single teacher",
                "selection_data": "validation only",
                "selection_order": [
                    "minimum validation sCRPS",
                    "minimum validation mean(q95,q99 sQL)",
                    "fixed head order NB, HSNB, TWEEDIE_FULL",
                ],
            },
        )
        self.assertEqual(
            candidate["pools"]["oracle_boundary"],
            {
                "outer_best_allowed": "R1 relative-to-best comparisons and explicitly labelled S/R oracle ladders only",
                "forbidden": "outer oracle head identity may not enter pools, students, routing, or actions",
            },
        )
        self.assertEqual(candidate["students"]["A1_teacher"], "validation-selected P0 single teacher")
        self.assertEqual(candidate["real"]["model_scope"], "train teachers and students separately per dataset")
        self.assertEqual(candidate["pools"]["real_weight_scope"], "select P2/P3 separately on each dataset validation; global means constant over that dataset's series and origins, never shared across datasets")
        self.assertEqual(candidate["routing"]["B0_weight_source"], "dataset-specific validation-frozen P2 global CDF weights regardless of whether P3 is the primary pool")
        self.assertEqual(candidate["routing"]["gate_aggregation"], "macro-average datasets equally")
        self.assertEqual(
            candidate["stage_contracts"]["S2_operational"],
            {
                "outer_global_single": "minimum head loss on the cell-equal all-18-cell outer macro",
                "d_best": "minimum outer head loss separately for each d",
                "cell_best": "minimum outer head loss separately for each cell",
                "series_origin_oracle": "minimum outer head loss separately for each series-origin row",
                "oracle_labels": "all four outer-selection rungs are oracle characterization",
                "validation_CDF_pool": "one global P2 simplex selected on cell-equal synthetic validation sCRPS and applied unchanged to outer rows",
                "per_cell_or_outer_pool_selection": "forbidden",
                "gain_formula": "100*(1-L_rung/L_outer_global_single)",
            },
        )
        self.assertEqual(
            candidate["students"]["primary_distilled_student"],
            {
                "candidates": ["A3", "A4"],
                "includes_validation_selected_lambda": True,
                "selection_order": ["minimum validation sCRPS", "minimum validation mean(q95,q99 sQL)", "A3 on exact tie"],
                "sole_input_to_gates": ["A2", "A3", "A4"],
            },
        )
        self.assertEqual(candidate["stage_contracts"]["S1_exact_best_tie_order"], ["sCRPS", "tail mean(q95,q99 sQL)", "NB", "HSNB", "TWEEDIE_FULL"])
        self.assertEqual(
            candidate["pools"]["Tweedie_blocked"],
            {"NB_HSNB_weights": "renormalize over available NB and HSNB", "role": "diagnostic only", "three_family_GO": "forbidden"},
        )

    def test_v4_seals_validation_selection_provenance_and_dataset_eligibility(self):
        """Catches relabelled outer arrays selecting a pool or geometry alone admitting a real dataset."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["pools"]["selection_provenance"],
            {
                "scope": "the dataset's exact frozen validation half-open interval only",
                "required_bindings": ["authoritative preregistration hash", "source manifest hash", "content-bound primary dataset manifest", "sample manifest hash", "dataset id", "validation interval", "series/origin/step prediction-key hash", "ordered validation-group-id hash", "validation-target hash", "teacher-seed replicate ids", "teacher prediction hash"],
                "key_rule": "every selection row key must lie inside the frozen validation interval and match the frozen sample and teacher replicate; exact duplicate or missing keys fail",
                "array_without_provenance": "reject even when a caller labels it validation",
                "outer_warmup_or_test_row": "forbidden for P0, P2, P3 p0/path/penalty, primary P2-versus-P3, and student lambda selection",
                "dataset_scope": "validate and select independently per canonical dataset; FULL seed metrics are averaged equally before the single dataset-level selection",
                "verification": "recompute every bound content hash before selection and again before applying frozen weights/identity",
                "validation_grouping": {
                    "seal": "SealedValidationArtifact stores exactly one finite hashable group ID per sealed case plus validation_group_sha256",
                    "real_group_id": "series_id so every real validation selector is series-equal",
                    "synthetic_S2_P2_group_id": "canonical (data_seed,d,rho_I,rho_M) cell identity so all 18 cells are equal-weighted",
                    "aggregation": "mean loss within group, then equal arithmetic mean over frozen groups; FULL model-seed scores remain equally averaged",
                    "caller_regrouping": "forbidden; supplied group IDs must byte-for-byte equal the sealed ordered IDs",
                },
            },
        )
        self.assertEqual(
            candidate["real"]["dataset_eligibility_contract"],
            {
                "required_before_selection": ["literal expected source SHA-256 verified", "dataset construction audit passed", "model_train target support audit passed COUNT SUPPORT", "every later count-likelihood index is exact integer", "fixed split/lookback/horizon geometry passed"],
                "logic": "all requirements are conjunctive and outcome-blind before fixed-priority selection",
                "geometry_only_or_length_only": "forbidden",
                "failed_dataset": "exclude before applying the fixed non-M5 priority; never repair by rounding, truncating, or post-result substitution",
                "manifest": "the complete audit payload and its canonical digest are stored in FrozenPrimaryDatasetManifest",
            },
        )
        self.assertEqual(
            candidate["real"]["dataset_selection_manifest"],
            {
                "manifest_type": "COUNT_PRIMARY_DATASET_SELECTION_MANIFEST",
                "builder": "select_real_datasets",
                "verifier": "verify_real_dataset_selection_manifest",
                "inputs": "the complete five sealed canonical dataset audits, exactly once each in m5,auto,carparts,raf,online_retail priority order",
                "selection": "mandatory m5 plus up to the first two eligible non-m5 datasets under the frozen priority; no result-dependent replacement",
                "bindings": ["all five audit payloads", "audit_manifest_sha256", "selected full audit rows", "selected_dataset_ids", "fixed priority", "selection_manifest_sha256"],
                "reconstruction": "recompute the complete fixed-priority selection and canonical SHA256; an arbitrary caller-provided 64-hex digest is never authority",
                "consumer_coverage": "every real confirmatory target/forecast artifact carries this verified full manifest and has dataset_id in selected_dataset_ids; every real macro/aggregate gate covers exactly every selected dataset once with no extra or missing dataset",
            },
        )
        self.assertEqual(candidate["stage_contracts"]["real_dataset_selection_authority"], "after all five source/support/split audits, publish one COUNT_PRIMARY_DATASET_SELECTION_MANIFEST; every real confirmatory artifact and macro gate verifies and exhaustively covers that selection")

    def test_v4_freezes_C_operational_targets_actions_and_C_SYN_teacher_source(self):
        """Catches target leakage, a vacuous pool switch, forced harmful action, or an extra C-SYN fit."""
        candidate = self._v4_candidate()
        sensor = candidate["sensors_and_actions"]
        self.assertEqual(
            sensor["operational_targets"],
            {
                "forecast_for_targets_1_to_3": "validation-selected P0 single teacher",
                "target_1_outer": "next-origin P0 sCRPS exceeds the within-dataset outer-row 80th percentile; evaluation label only",
                "target_1_inner": "next-origin P0 sCRPS exceeds the within-dataset inner-origin 80th percentile fit only on inner labels",
                "target_2": "either empirical central-90 coverage is below 0.90 or empirical central-95 coverage is below 0.95",
                "target_3_outer": "next-origin P0 zero-probability calibration error exceeds the within-dataset outer-row 80th percentile; evaluation label only",
                "target_3_inner": "next-origin P0 zero-probability calibration error exceeds the within-dataset inner-origin 80th percentile fit only on inner labels",
                "target_4": "next-origin best-head identity under fixed tie order NB, HSNB, TWEEDIE_FULL differs from current-origin best-head identity",
            },
        )
        self.assertEqual(
            sensor["action_policy"],
            {
                "primary_score": "C2 logistic classifier probability for target 1",
                "validation_threshold": "q80(method='higher') of validation scores; flag iff score>threshold with no tie reranking",
                "zero_flagged_validation_rows": "metrics undefined and C3 FAIL; no reselection or substitution",
                "C_A0": "keep validation-selected P0 single teacher",
                "C_A1": "switch to validation-selected primary P2/P3 pool",
                "C_A2": "for every requested nonmedian quantile including 0.025 and 0.975 use max(0,Q50+f*(Qq-Q50)); preserve q50 and p0, then zero-consistency and cumulative maximum",
                "C_A2_factor_grid": [1.05, 1.10, 1.20, 1.35, 1.50],
                "C_A2_factor_selection": "on validation flagged rows minimize mean(CE90,CE95), then sCRPS, then smaller factor",
                "C_A2_predictive_mean": "recompute from the adjusted 21-grid quantiles by endpoint-hold trapezoidal integration; record mean_source=quantile_integral_endpoint_hold",
                "C_A3_selection": "over all validation rows minimize mean(CE90,CE95), then sCRPS, then fixed NB<HSNB<TWEEDIE_FULL; freeze for outer",
                "candidate_selection": "on validation flagged rows minimize sCRPS, then mean(CE90,CE95), then fixed C_A0<C_A1<C_A2<C_A3",
                "allow_no_action": True,
                "outer_application": "apply the frozen action only to flagged outer rows without reselection",
                "selective_coverage": "recall of target-1 failures captured by top-20%-score action cases",
            },
        )
        self.assertEqual(
            candidate["synthetic"]["C_SYN"]["teachers"],
            {
                "source": "already trained Stage S1 checkpoints",
                "matching_cell": {"rho_I": 0.0, "rho_M": 0.0, "match": ["d", "data_seed", "model_seed"]},
                "extra_global_fit": "forbidden",
                "FULL": "evaluate each preregistered DGP-seed by model-seed replicate separately",
                "COMPACT_and_MINIMAL-COMPLETE": "evaluate the single preregistered DGP-seed by model-seed replicate",
                "aggregation": "base-innovation cluster",
                "blocked_head": "continue available heads diagnostically and forbid DISAGREEMENT_SENSOR_GO",
            },
        )

    def test_v4_freezes_B_primary_statistic_common_loss_and_crossfit_student_rows(self):
        """Catches averaging favorable router metrics or backfilling an earlier row with future regrets."""
        routing = self._v4_candidate()["routing"]
        self.assertEqual(
            routing["B1_gate_statistic"],
            {
                "per_dataset_primary": "Spearman(flatten predicted weights over row*head,flatten negative true regrets over row*head)",
                "extended_delta": "extended-primary minus baseline-primary and must be at least 0.08",
                "shuffle_delta": "shuffled-primary minus the same baseline-primary",
                "shuffle_rule": "shuffle_delta<=0.25*real_extended_delta",
                "cross_dataset_sign": "extended-primary minus baseline-primary is strictly positive in every selected dataset",
                "secondary_metrics": ["best-head top-1 accuracy", "expected regret", "per-head regret Spearman"],
                "undefined": "INSUFFICIENT_VARIATION and B1 FAIL; never encode as zero",
            },
        )
        self.assertEqual(
            routing["D_common"],
            "teacher-specific Bernoulli soft cross-entropy for zero mass plus mean_q Huber(Q_teacher_m(q)/s_i,Q_student(q)/s_i) with delta=1.0",
        )
        self.assertEqual(
            routing["student_row_contract"],
            {
                "rows": "identical expanding-crossfit heldout inner-origin rows k=2..K for B0, B1, B2 and every B fit-requiring control",
                "B1_B2_weights": "for heldout origin k use only a router fitted on origins 1..k-1",
                "final_router_backfill": "forbidden",
                "B0_weights": "apply dataset-specific validation-frozen global weights to the identical heldout rows",
                "outer_target_in_router_fit": "forbidden",
                "parity_scope": "row parity is required within B but not between A and B",
                "required_report": "B train-row count",
            },
        )

    def test_v4_freezes_sensor_prequential_rows_primary_gate_and_action_gate_math(self):
        """Catches using data beyond t+h, pooling C1 targets, or hiding an empty false-alarm subset."""
        sensor = self._v4_candidate()["sensors_and_actions"]
        self.assertEqual(
            sensor["C1_primary_contract"],
            {
                "target": "target 1 only",
                "model": "C2 logistic classifier",
                "unit": "series-origin rows within each real dataset",
                "same_classifier_metrics": ["AUROC", "AUPRC", "C2-C0 AUPRC improvement", "C2-C3 AUPRC improvement", "Brier calibration"],
                "targets_2_to_4": "secondary diagnostic only",
            },
        )
        self.assertEqual(
            sensor["prequential_timing"],
            {
                "decision": "after [t,t+h) is realized, use the row at t to predict failure at t+h",
                "previous_realized_residual": "mean_h abs(P0_predictive_mean[t:t+h)-y[t:t+h))/s_i",
                "zero_ratio_change": "zero_rate[t,t+h)-zero_rate[t-h,t)",
                "scale_change": "RMS[t,t+h)/s_i-RMS[t-h,t)/s_i",
                "last_event_gap_change": "with g(b)=b-1-last_positive_index over availability-valid history strictly before b, use (g(t+h)-g(t))/h; missing if either g is undefined",
                "recent_target_variance": "population variance of the last min(96,available observations) ending at t+h divided by s_i^2; missing if fewer than 2",
                "data_boundary": "no observation with index >=t+h",
                "target_2": "1 iff inclusive empirical h-step central-90 coverage<0.90 OR inclusive empirical h-step central-95 coverage<0.95",
                "target_3_score": "mean_h (1[y=0]-p0)^2",
                "target_1_3_threshold": "within declared inner or outer scope use quantile(method='higher') at 0.80 and strict >",
            },
        )
        self.assertEqual(
            sensor["C3_gate_evaluation"],
            {
                "baseline": "C_A0 validation-selected P0",
                "worst_decile": "within each dataset, baseline series-origin sCRPS>=q90(method='higher')",
                "worst_decile_gain": "(L_C_A0-L_policy)/L_C_A0 then equal-weight dataset macro",
                "coverage_reduction": "compute (CE0-CEpolicy)/CE0 separately for central 90 and 95; both must be >=0.15",
                "zero_CE0": "require CEpolicy=0 and contribute reduction 0",
                "overall_deterioration": "(L_policy-L_C_A0)/L_C_A0",
                "false_alarm_subset": "flagged rows with target1=0 within each dataset",
                "false_alarm_rule": "each dataset and equal-weight macro deterioration<0.005",
                "empty_false_alarm_subset": "report undefined and fail C3",
                "selective_coverage": "recall of target1 failures captured by top-20%-score action cases",
            },
        )

    def test_v4_freezes_C_SYN_false_alarm_delay_and_component_separation(self):
        """Catches dropping undetected sequences or conflating interval and magnitude change responses."""
        scoring = self._v4_candidate()["synthetic"]["C_SYN"]["scoring_operational"]
        self.assertEqual(
            scoring,
            {
                "false_alarm_rate": "false positives among heldout changed-sequence pre-change rows divided by those rows",
                "no_change_false_positive_rate": "false positives among all heldout no-change rows divided by those rows",
                "undetected_delay": 10,
                "delay_rule": "delay in evaluated post-origin horizon units from 288; never drop an undetected changed sequence",
                "component_standardization": "absolute(post mean-pre mean) divided by train-pre component SD",
                "interval_shift_pass": "mean(D_zero,D_cdf)>mean(D_center,D_tail), strict",
                "magnitude_shift_pass": "mean(D_center,D_tail)>mean(D_zero,D_cdf), strict",
                "simultaneous_shift_in_component_test": "excluded",
                "shift_classifier_rows": "changed post-change rows only",
                "shift_classes": ["rho_I_positive", "rho_I_negative", "rho_M_positive", "rho_M_negative", "rho_I_and_rho_M"],
                "shift_metrics": ["accuracy", "macro-F1"],
                "first_origin_delta": "undefined value plus missing indicator",
                "zero_train_pre_component_SD": "component separation FAIL",
            },
        )

    def test_v4_freezes_exact_sources_splits_change_panel_and_runtime_tiers(self):
        """Catches source provenance truncation, split drift, or post-smoke sample-size discretion."""
        candidate = self._v4_candidate()
        self.assertEqual(candidate["external_references"]["generator_authority"]["file_sha256"], "0ad1b5607dd75bfe0c825fef778e55b17bf6667d16849a30beab295a66fd3e71")
        self.assertEqual(candidate["external_references"]["generator_authority"]["adaptation"], "reuse the symmetric event-index Markov generator; within each fixed (data_seed,d), key base innovations without rho before rho-specific transitions; d remains in the base fingerprint so d=4 and d=8 are never paired")
        self.assertEqual(candidate["external_references"]["tweedie"]["commit"], "f14a189d7cd80d41886041f44f40ae4db27d0067")
        self.assertEqual(
            candidate["source_snapshot_contract"]["expected_raw_sha256"],
            {
                "M5_sales_train_evaluation.csv": "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0",
                "M5_calendar.csv": "d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5",
                "M5_sell_prices.csv": "9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef",
                "M5_prior_Stage_A_series.parquet": "aa8d96ecd6ed6eaa91274087b4b90880b5da4ec3954962d94d57e37947f13aba",
                "OnlineRetail_online_retail_II.xlsx": "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980",
                "verified_generator_dgp.py": "0ad1b5607dd75bfe0c825fef778e55b17bf6667d16849a30beab295a66fd3e71",
            },
        )
        self.assertEqual(candidate["source_snapshot_contract"]["copy_method"], "byte-for-byte regular-file copy; symlinks and hardlinks forbidden")
        self.assertEqual(candidate["source_snapshot_contract"]["execution_reads"], "only hash-verified source snapshots")
        self.assertEqual(candidate["source_snapshot_contract"]["loader_expected_hash_source"], "literal preregistration constants; caller-provided overrides forbidden")
        self.assertEqual(candidate["source_snapshot_contract"]["worktree_manifest"]["expected_sha256_by_path"]["requirements-prob-head-structure-full-v1.lock"], "7a8708e966140e1a5eccea104937fb7e99e6af9c845372404ca55ae50dd173f8")
        self.assertEqual(candidate["environment"]["lock_generation"], {"tool": "uv", "version": "0.11.19", "input": "requirements-prob-head-structure-full-v1.in", "lock": "requirements-prob-head-structure-full-v1.lock", "lock_sha256": "7a8708e966140e1a5eccea104937fb7e99e6af9c845372404ca55ae50dd173f8"})
        self.assertEqual(
            candidate["integrity"]["protected_manifest_instances"],
            [
                {"repository_root_identity": "isolated_execution_worktree", "repository_root": "E:/CODING/worktrees/covariate-trust-pilot-prob-head-structure-full-v1", "before_path": "results/prob_head_structure_full_v1/audit/protected_manifest_execution_worktree_before.json"},
                {"repository_root_identity": "original_user_working_copy", "repository_root": "E:/CODING/proj/covariate-trust-pilot", "before_path": "results/prob_head_structure_full_v1/audit/protected_manifest_original_working_copy_before.json"},
            ],
        )
        self.assertEqual(candidate["integrity"]["legacy_unbound_manifest"], {"path": "results/prob_head_structure_full_v1/audit/protected_manifest_before.json", "preserve": True, "authoritative_for_final_verification": False, "reason": "schema lacked repository-root binding"})
        self.assertEqual(
            candidate["real"]["sampling"]["prior_sample_exclusion"],
            {
                "M5": {
                    "source": "runs/prob_head_structure_full_v1/source_snapshots/m5/series.parquet",
                    "source_sha256": "aa8d96ecd6ed6eaa91274087b4b90880b5da4ec3954962d94d57e37947f13aba",
                    "column": "series_id",
                    "validation": "exactly 1200 unique non-null strings",
                    "canonicalization": "lexicographic sort then UTF-8 newline join",
                    "expected_ID_list_sha256": "c73bb7506b56c76b7fbebdf67a20e0e08f48894976da7a05adfad749bba9334e",
                    "timing": "exclude exact set before descriptor calculation and stratified sampling",
                    "required_report": ["excluded count", "ID-list SHA256", "zero overlap"],
                },
                "OnlineRetail": "no exclusion because no provenance-defined prior sample exists",
            },
        )
        self.assertEqual(
            candidate["real"]["sampling"]["M5_availability_eligibility"],
            {
                "condition": "available_from < 1717",
                "expected_ineligible_series": 57,
                "support_and_descriptor_window": "[available_from,1717)",
                "train_RMS_window": "availability-observed [available_from,1717); no pre-availability zero padding enters s_i",
                "timing": "before descriptor computation and stratified sampling",
                "required_report": "availability-ineligible excluded count",
            },
        )
        self.assertEqual(candidate["real"]["selection_contract"], {"M5_required": True, "non_M5_selected_max": 2, "order": ["Auto", "Carparts", "RAF", "OnlineRetail"], "minimum_total_count_datasets": 2, "fewer_than_two_non_M5": "report shortfall without applying REAL_CROSS_DATASET_EVIDENCE_LIMITED when total datasets remain at least two", "REAL_CROSS_DATASET_EVIDENCE_LIMITED_if": "total audited count datasets < 2", "missing_M5": "count-primary selection cannot pass"})
        self.assertEqual(
            candidate["real"]["production_full_pool_attestation"],
            {
                "m5": {
                    "source_state": "canonical loader output after the frozen 1200-ID Stage-A exclusion and before train-descriptor eligibility filtering",
                    "shape": [29290, 1941],
                    "panel_binding_sha256": "3844515e707de5fff22c1df62ba918f1cab2bf901eee5ff72d33b5cc2acbb2be",
                    "ordered_series_id_sha256": "c4d8f4c4be936c875abdd6f67e1fd18f87b18caa717f6ac9cc1ec31e5535258e",
                },
                "online_retail": {
                    "source_state": "canonical frozen TweedieGP-derived loader output before train-descriptor eligibility filtering",
                    "shape": [2036, 374],
                    "panel_binding_sha256": "ff41f533c572d6f2fd603dfdac7339374a05b2cd531760ec51b9edc9ec7f4348",
                    "ordered_series_id_sha256": "96443093e11320948afd5bb57540ad48fdfb4549cb2b16a383ec83ab6ee4e2a0",
                },
                "production_input_rule": "dataset audit and sampling start from these complete canonical loader outputs; reject preselected, resealed, reordered, or caller-substituted panels",
            },
        )
        self.assertEqual(candidate["real"]["sampling"]["stratification_contract"], {"descriptor_bins": "within-dataset quartiles q=4", "strata": "joint four-dimensional descriptor bins", "allocation": "proportional largest remainder", "tie_break": "deterministic ascending series_id", "scope": "separately within each dataset", "sample_size": "min(runtime-tier requested N, eligible pool size)", "eligible_pool_smaller_than_N": "census and report", "upsampling_or_duplicate_series": "forbidden", "OnlineRetail_FULL_expected": "census because eligible pool is at most 2036 < 4000"})
        self.assertEqual(
            candidate["real"]["sampling"]["sealed_sample_manifest"],
            {
                "builder": "seal_train_only_sample_manifest",
                "runtime_tier_requested_N": {"FULL": 4000, "COMPACT": 2000, "MINIMAL-COMPLETE": 1000},
                "seed": 2026090521,
                "recompute": "availability-aware model-train descriptors and the within-dataset joint-quartile proportional-largest-remainder selection from the full sealed panel",
                "binds": ["complete sealed dataset audit and digest", "full-panel target bytes", "ordered series IDs", "availability vector", "source-manifest aggregate", "descriptor table", "eligible pool", "strata and allocations", "selected series IDs and row positions", "sampled-panel bytes"],
                "verification": "reconstruct the complete manifest and sampled subset from the full sealed panel before every use; arbitrary or outer-aware subsets and caller-provided manifests are rejected",
                "WindowRequest_and_WindowBatch": "bind manifest_sha256 and sampled_panel_binding_sha256 and reconstruct from the full panel; never accept a detached subset",
            },
        )
        self.assertEqual(candidate["target_support"]["execution_boundary"], "every target value passed to any count likelihood or evaluation must be finite, nonnegative, and literally integer-valued; the 1e-6 audit tolerance does not authorize PMF indexing")
        self.assertEqual(candidate["target_support"]["later_split_checks"], "integrity-only and cannot change sampling")
        self.assertEqual(candidate["synthetic"]["split"], {"model_train": [0, 380], "validation": [380, 408], "warmup": [408, 436], "evaluation": [436, 576]})
        self.assertEqual(
            candidate["synthetic"]["C_SYN"]["shifts"],
            [
                {"type": "rho_I_positive", "pre": {"rho_I": 0.0}, "post": {"rho_I": 0.8}},
                {"type": "rho_I_negative", "pre": {"rho_I": 0.0}, "post": {"rho_I": -0.8}},
                {"type": "rho_M_positive", "pre": {"rho_M": 0.0}, "post": {"rho_M": 0.8}},
                {"type": "rho_M_negative", "pre": {"rho_M": 0.0}, "post": {"rho_M": -0.8}},
                {"type": "rho_I_and_rho_M", "pre": {"rho_I": 0.0, "rho_M": 0.0}, "post": {"rho_I": 0.8, "rho_M": 0.8}},
                {"type": "no_change", "pre": {"rho_I": 0.0, "rho_M": 0.0}, "post": {"rho_I": 0.0, "rho_M": 0.0}},
            ],
        )
        self.assertEqual(candidate["synthetic"]["C_SYN"]["forecast_origins"], [120, 148, 176, 204, 232, 260, 288, 316, 344, 372, 400, 428, 456, 484, 512, 540])
        self.assertEqual(
            candidate["synthetic"]["C_SYN"]["scoring"],
            {
                "binary_label": "1 only for changed sequences at origins>=288; 0 for changed-sequence pre origins and every no-change row",
                "split": "deterministic 50/50 base-series-innovation split within each (d,shift), preserving paired series",
                "binary_model": "standardized L2 logistic regression on component vector plus deltas; no hyperparameter search",
                "shift_type_model": "multinomial logistic regression trained on the train half",
                "threshold": "train-half no-change scores at the 90th percentile",
                "threshold_tie_rule": "score > quantile(method='higher')",
                "evaluation": "heldout half only for AUROC, AUPRC, FAR, no-change false positive, detection delay, and shift-type classification",
                "delay": "horizon units from onset origin 288; delay 0 at origin 288 and delay 1 at origin 316",
                "component_separation": "heldout standardized post-minus-pre component changes grouped by interval versus magnitude shift family",
                "future_target_in_score": "forbidden",
            },
        )
        self.assertEqual(
            candidate["runtime"]["tiers"],
            {
                "FULL": {"condition": "projected_GPU_hours<=12", "synthetic_series_per_cell": 80, "synthetic_DGP_seeds": 2, "teacher_model_seeds": 2, "real_series_per_dataset": 4000, "student_seeds": 2, "all_A_B_C": True, "bootstrap_draws": 2000},
                "COMPACT": {"condition": "12<projected_GPU_hours<=18", "synthetic_series_per_cell": 40, "synthetic_DGP_seeds": 1, "teacher_model_seeds": 1, "real_series_per_dataset": 2000, "student_seeds": 1, "all_A_B_C": True, "bootstrap_draws": 1000},
                "MINIMAL-COMPLETE": {"condition": "projected_GPU_hours>18", "synthetic_series_per_cell": 24, "synthetic_DGP_seeds": 1, "teacher_model_seeds": 1, "real_series_per_dataset": 1000, "student_seeds": 1, "all_A_B_C": True, "bootstrap_draws": 500, "label": "SCREEN_ONLY"},
            },
        )
        self.assertEqual(
            candidate["runtime"]["tier_projection_contract"],
            {
                "measured_basis": "200-series smoke",
                "projection_target": "complete FULL workload with every frozen fit count, series-window exposure, maximum planned seed count, and every A/B/C variant",
                "overhead_fraction": 0.25,
                "selection_value": "one projected FULL-workload GPU-hour value",
                "thresholds": {"FULL": "<=12", "COMPACT": ">12 and <=18", "MINIMAL-COMPLETE": ">18"},
                "recompute_after_downsizing": "forbidden",
            },
        )
        self.assertEqual(
            candidate["environment"]["runtime_inventory_contract"],
            {
                "python_executable": ["resolved path", "SHA256"],
                "required_distributions": ["torch", "numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib", "pytest", "tweedie", "openpyxl"],
                "per_distribution": ["version", "installation location", "METADATA SHA256", "RECORD SHA256 when present"],
                "missing_METADATA_or_RECORD": "record null plus an explicit reason",
                "overlay_lock": ["path", "file SHA256"],
                "inherited_packages": "inventory even when absent from the overlay lock",
            },
        )

    def test_v4_freezes_all_execution_tests_tables_figures_status_and_console_items(self):
        """Catches an orchestrator silently omitting a required stage, audit, table, figure, or final disclosure."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["execution"]["stage_order"],
            [
                "git/repository audit", "existing artifact hash baseline", "external likelihood source audit",
                "environment creation", "preregistration freeze", "likelihood numerical unit test",
                "synthetic DGP audit", "200-series smoke", "runtime tier selection",
                "Stage S1 synthetic 18-cell teacher training", "Stage S2 specialization/oracle/structure analysis",
                "Stage C-SYN known-change experiment", "real count dataset audit/download",
                "Stage R1 real teacher training", "Stage R2 real complementarity", "CDF pool",
                "Stage A student distillation", "Stage B regret predictability",
                "Stage B structure-conditioned distillation", "Stage C failure sensor",
                "Stage C actionable policy", "all negative controls", "bootstrap",
                "final gate calculation", "figures", "STATUS", "artifact hash verification",
                "test suite", "commit", "optional push",
            ],
        )
        expected_tests = {
            "T01": "train/validation/test disjoint", "T02": "future target leakage 없음",
            "T03": "support-likelihood 일치", "T04": "NB mass/mean/variance",
            "T05": "HSNB mass/mean/zero identity", "T06": "Tweedie zero mass",
            "T07": "Tweedie independent density agreement", "T08": "CDF monotonicity",
            "T09": "quantile monotonicity", "T10": "same series/origin/step coverage",
            "T11": "train-only scaling", "T12": "CDF mixture simplex",
            "T13": "mixture quantile inversion", "T14": "student quantile non-crossing",
            "T15": "teacher test target 미사용", "T16": "router outer-test 미사용",
            "T17": "sensor next-origin leakage 없음", "T18": "series-cluster bootstrap",
            "T19": "shuffle control deterministic", "T20": "upstream fail 뒤 diagnostic label 유지",
            "T21": "diagnostic 결과가 upstream verdict를 변경하지 못함", "T22": "기존 result directories hash 불변",
            "T23": "resume idempotence", "T24": "final table 숫자가 source artifact에서 재계산 가능",
        }
        self.assertEqual(candidate["execution"]["required_tests"], expected_tests)
        self.assertEqual(
            candidate["reporting"]["tables"],
            {
                "A": "repository / environment / source audit", "B": "dataset support와 split",
                "C": "distribution numerical validation", "D": "teacher parameter count와 runtime",
                "E": "synthetic 18-cell sCRPS", "F": "synthetic winner share와 oracle gain",
                "G": "temporal structure contrasts", "H": "synthetic change-sensor 결과",
                "I": "real teacher benchmark", "J": "real winner share와 oracle ladder",
                "K": "CDF pool 결과", "L": "distillation 결과와 recovery",
                "M": "student compression/latency", "N": "structure-regret predictability",
                "O": "structure-conditioned distillation", "P": "disagreement future-failure detection",
                "Q": "actionable sensor policy", "R": "negative controls",
                "S": "모든 Gate", "T": "최종 추천 연구축",
            },
        )
        self.assertEqual(
            candidate["reporting"]["figures"],
            [
                "synthetic head winner map d=4", "synthetic head winner map d=8",
                "structure effect forest plot", "real dataset head metric comparison",
                "oracle ladder", "teacher pool vs student recovery",
                "disagreement별 future failure performance", "전체 branch decision tree",
            ],
        )
        self.assertEqual(
            candidate["reporting"]["STATUS_sections"],
            [
                "What was attempted", "What was frozen", "Runtime tier", "Environment",
                "Dataset support", "Numerical likelihood validation", "Synthetic specialization",
                "Temporal structure effect", "Real teacher quality", "Real complementarity",
                "CDF pooling", "A distillation", "B structure-conditioned routing",
                "C disagreement sensor", "Controls", "Compression/runtime",
                "Confirmatory vs diagnostic evidence", "Gate table", "Final recommendation",
                "What must not be claimed", "Exact next research action",
            ],
        )
        self.assertEqual(candidate["reporting"]["STATUS_first_line"], "FINAL RECOMMENDATION: <exact token>")
        self.assertEqual(
            candidate["reporting"]["final_console_order"],
            [
                "FINAL RECOMMENDATION", "A verdict", "B verdict", "C verdict",
                "HEAD specialization verdict", "synthetic 핵심 effect", "real oracle headroom",
                "best teacher pool gain", "student recovery", "disagreement sensor AUPRC",
                "total GPU/wall time", "scientific FAIL 목록", "diagnostic continuation 목록",
                "commit SHA", "push 상태", "STATUS 경로",
            ],
        )

    def test_v4_freezes_literal_hard_failures_gate_dependencies_and_forbidden_claims(self):
        """Catches weakening an integrity stop, changing a gate, or overstating diagnostic evidence."""
        candidate = self._v4_candidate()
        self.assertEqual(
            [item["condition"] for item in candidate["integrity"]["hard_failures"]],
            [
                "미래 target이 feature, normalization, teacher weight 선택에 들어감",
                "train/validation/test index가 겹침",
                "다른 head가 서로 다른 series/origin/step을 평가함",
                "기존 결과 파일을 덮어씀",
                "데이터 원본 hash가 실행 도중 바뀜",
                "목표변수 support와 likelihood가 맞지 않음",
                "Tweedie full likelihood 대신 deviance를 몰래 사용함",
                "확률분포가 유효하지 않음",
                "NaN/Inf 또는 probability mass 오류가 허용치를 초과함",
                "full Tweedie 구현을 독립 reference와 검증하지 못함",
                "결과를 본 뒤 threshold나 hyperparameter grid를 변경함",
            ],
        )
        self.assertEqual(candidate["branch_eligibility"]["A_DISTRIBUTION_DISTILLATION"]["upstream_required_gates"], ["R1", "R2", "R3", "A1", "A2", "A3", "A4", "CONTROL_A", "TWEEDIE_VALID"])
        self.assertEqual(candidate["branch_eligibility"]["B_STRUCTURE_CONDITIONED_ROUTING"]["upstream_required_gates"], ["R1", "R2", "B1", "B2", "CONTROL_B", "TWEEDIE_VALID"])
        self.assertEqual(candidate["branch_eligibility"]["C_DISAGREEMENT_SENSOR"]["upstream_required_gates"], ["R1", "C1", "C2", "C3", "CONTROL_C", "TWEEDIE_VALID"])
        self.assertEqual(
            candidate["reporting"]["forbidden_claims"],
            [
                "M5 등 개발 dataset 결과를 external confirmation이라 부르기",
                "diagnostic continuation을 confirmatory evidence라 부르기",
                "raw NLL로 다른 distribution family를 직접 순위화하기",
                "Tweedie deviance를 full density라고 부르기",
                "Favorita 실수 target을 반올림해 NB에 넣기",
                "teacher pool이 좋다는 이유만으로 distillation 성공이라고 부르기",
                "disagreement 상관만으로 distribution shift 원인이라고 단정하기",
                "synthetic 구조 효과를 실제 데이터 인과 효과라고 부르기",
                "하나의 seed를 일반적 효과라고 부르기",
                "downstream diagnostic 결과로 upstream FAIL을 취소하기",
            ],
        )

    def test_v4_freezes_every_gate_threshold_and_exact_failure_token(self):
        """Catches threshold drift or a scientific failure being emitted under a non-protocol label."""
        gates = self._v4_candidate()["gates"]
        self.assertEqual(
            set(gates),
            {"TWEEDIE_VALID", "DGP_BALANCE", "S1", "S2", "S3", "FINAL_HEAD", "R1", "R2", "R3", "FINAL_REAL", "A1", "A2", "A3", "A4", "CONTROL_A", "FINAL_A", "B1", "B2", "CONTROL_B", "FINAL_B", "C1", "C2", "C3", "CONTROL_C", "FINAL_C"},
        )
        self.assertEqual(
            gates["TWEEDIE_VALID"],
            {
                "PASS_all": ["full vendored Tweedie log density passes the frozen 600-row two-precision reference audit", "zero mass identity and numerical CDF monotonicity pass", "all required predictive-distribution methods are finite and valid"],
                "PASS_token": "TWEEDIE_VALID",
                "FAIL": "TWEEDIE_BRANCH_BLOCKED_HARD",
                "scope": "branch-local hard block: no three-family confirmatory stage can pass; NB/HSNB diagnostics continue with explicit ineligibility",
            },
        )
        self.assertEqual(gates["DGP_BALANCE"], {"PASS": "every required 18-cell by DGP-seed balance group passes all frozen tolerances", "FAIL": "DGP_BALANCE_FAIL", "failed_cells": "retain complete diagnostic output and remove aggregate S1 confirmatory eligibility"})
        self.assertEqual(gates["S1"]["PASS_if_any"], [{"heads_each_best_cells_min": 2, "best_cells_per_head_min": 3, "total_cells": 18}, {"heads_each_series_practical_winner_share_min": 2, "share_min": 0.15}])
        self.assertEqual(gates["S1"]["FAIL"], "HEAD_SPECIALIZATION_NO_GO")
        self.assertEqual(gates["S2"]["PASS_all"], {"cell_oracle_macro_gain_min": 0.02, "series_origin_oracle_gain_min": 0.03})
        self.assertEqual(gates["S2"]["FAIL"], "HEAD_COMPLEMENTARITY_TOO_SMALL")
        self.assertEqual(gates["S3"]["PASS"], {"temporal_contrasts_min": 1, "absolute_effect_percentage_points_min": 2.0, "cluster_bootstrap_95_CI_excludes_zero": True})
        self.assertEqual(gates["R1"]["PASS"], {"heads_min": 2, "datasets_within_5_percent_of_best_min": 2, "any_primary_dataset_deficit_over_10_percent": False, "best_in_at_least_one_of": ["zero-Brier", "tail scaled quantile loss"]})
        self.assertEqual(gates["R2"]["PASS"], {"heads_with_practical_winner_share_min": 2, "practical_winner_share_min": 0.15, "origin_oracle_macro_gain_min": 0.02, "datasets_with_oracle_gain_at_least_1_percent_min": 2, "max_pairwise_Pearson_loss_correlation_cluster_bootstrap_upper95_lt": 0.99, "dataset_level_best_head_identities_min": 2})
        self.assertEqual(gates["R3"]["PASS"], {"primary_pool_macro_sCRPS_improvement_min": 0.01, "each_dataset_sCRPS_improvement_min": -0.005, "mean_q95_q99_scaled_quantile_loss_improvement_min": 0.02, "q50_scaled_quantile_loss_deterioration_lt": 0.01, "zero_Brier_deterioration_lt": 0.01, "predictive_mean_NRMSE_deterioration_lt": 0.01})
        self.assertEqual(gates["A1"]["PASS"], {"primary_teacher_pool_sCRPS_improvement_min": 0.01, "tail_q95_q99_scaled_quantile_loss_improvement_min": 0.02})
        self.assertEqual(gates["A2"]["PASS"], {"recovery_min": 0.50, "A3_or_A4_sCRPS_improvement_over_A0_min": 0.005, "macro_bootstrap_CI_lower_gt": 0.0, "datasets_with_positive_effect_min": 2, "datasets_with_CI_lower_gt_zero_min": 1})
        self.assertEqual(gates["A3"]["PASS"]["deterioration_not_over"], 0.01)
        self.assertEqual(gates["A4"]["PASS"], {"student_parameter_multiple_of_smallest_teacher_max": 1.5, "student_latency_multiple_of_single_teacher_max": 1.3, "student_speedup_over_teacher_pool_min": 2.0, "student_peak_memory_fraction_of_teacher_pool_max": 0.5})
        self.assertEqual(gates["A4"]["FAIL"], "COMPRESSION_VALUE_NO_GO")
        self.assertEqual(gates["A4"]["parameter_count_domain"], "student and smallest-teacher parameter counts must both be positive integers; booleans and fractional values are contract errors")
        self.assertEqual(gates["B1"]["PASS"], {"datasets_with_regret_Spearman_at_least_0.20_min": 2, "extended_over_baseline_improvement_min": 0.08, "shuffle_increase_fraction_of_real_increase_max": 0.25, "cross_dataset_effect_sign_maintained": True})
        self.assertEqual(gates["B2"]["PASS"], {"B2_over_B0_macro_sCRPS_improvement_min": 0.005, "each_dataset_improvement_min": -0.0025, "macro_bootstrap_CI_lower_gt": 0.0, "datasets_with_positive_effect_min": 2, "B2_over_B1_improvement_min": 0.002, "q99_deterioration_lt": 0.01, "zero_Brier_deterioration_lt": 0.01, "worst_origin_improvement_min": -0.005})
        self.assertEqual(gates["B2"]["FAIL"], "STRUCTURE_CONDITIONED_DISTILLATION_EFFECT_NO_GO")
        self.assertEqual(gates["C1"]["PASS_on_at_least_two_real_datasets"], {"AUROC_min": 0.70, "AUPRC_min": 0.35, "C2_over_C0_AUPRC_improvement_min": 0.05, "C2_over_C3_AUPRC_improvement_min": 0.02, "Brier_calibration_worse_than_C0": False})
        self.assertEqual(gates["C2"]["PASS"]["median_detection_delay_horizons_max"], 1)
        self.assertEqual(gates["C3"]["PASS"], {"worst_decile_sCRPS_improvement_min": 0.10, "coverage_error_90_95_reduction_min": 0.15, "overall_mean_sCRPS_deterioration_lt": 0.005, "selective_coverage_min": 0.80, "false_alarm_dataset_deterioration_lt": 0.005})
        self.assertEqual(gates["C3"]["FAIL"], "DISAGREEMENT_ACTION_VALUE_NO_GO")
        self.assertEqual(gates["FINAL_HEAD"], {"required_PASS": ["DGP_BALANCE", "S1", "S2", "S3", "TWEEDIE_VALID"], "PASS": "HEAD_SPECIALIZATION_GO", "FAIL": "HEAD_SPECIALIZATION_NO_GO"})
        self.assertEqual(gates["FINAL_REAL"], {"required_PASS": ["R1", "R2", "R3", "TWEEDIE_VALID"], "PASS": "REAL_DISTRIBUTION_POOL_GO", "FAIL": "REAL_DISTRIBUTION_POOL_NO_GO"})
        self.assertEqual(gates["CONTROL_A"], {"PASS_all": {"real_effect_gt": 0.0, "each_applicable_destroying_control_recovery_lt": 0.5}, "FAIL": "SIGNAL_IDENTIFICATION_FAILURE"})
        self.assertEqual(gates["CONTROL_B"], {"PASS_all": {"real_effect_gt": 0.0, "each_applicable_destroying_control_recovery_lt": 0.5}, "FAIL": "SIGNAL_IDENTIFICATION_FAILURE"})
        self.assertEqual(gates["CONTROL_C"], {"PASS_all": {"real_effect_gt": 0.0, "each_applicable_destroying_control_recovery_lt": 0.5, "teacher_name_permutation_maxabs_le": 1e-12}, "FAIL": "SIGNAL_IDENTIFICATION_FAILURE"})
        self.assertEqual(gates["FINAL_A"], {"required_PASS": ["R1", "R2", "R3", "A1", "A2", "A3", "A4", "CONTROL_A", "TWEEDIE_VALID"], "PASS": "DISTRIBUTION_SPACE_DISTILLATION_GO", "FAIL": "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"})
        self.assertEqual(gates["FINAL_B"], {"required_PASS": ["R1", "R2", "B1", "B2", "CONTROL_B", "TWEEDIE_VALID"], "PASS": "STRUCTURE_CONDITIONED_ROUTING_GO", "FAIL": "STRUCTURE_CONDITIONED_ROUTING_NO_GO"})
        self.assertEqual(gates["FINAL_C"], {"required_PASS": ["R1", "C1", "C2", "C3", "CONTROL_C", "TWEEDIE_VALID"], "PASS": "DISAGREEMENT_SENSOR_GO", "FAIL": "DISAGREEMENT_SENSOR_NO_GO"})
        self.assertEqual(
            self._v4_candidate()["branch_eligibility"]["stage_required_gates"],
            {
                "TWEEDIE_VALID": [], "DGP_BALANCE": [],
                "S1": ["DGP_BALANCE", "TWEEDIE_VALID"], "S2": ["DGP_BALANCE", "S1", "TWEEDIE_VALID"], "S3": ["DGP_BALANCE", "S1", "S2", "TWEEDIE_VALID"],
                "R1": ["TWEEDIE_VALID"], "R2": ["R1", "TWEEDIE_VALID"], "R3": ["R1", "R2", "TWEEDIE_VALID"],
                "A1": ["R2", "R3", "TWEEDIE_VALID"], "A2": ["R2", "R3", "A1", "TWEEDIE_VALID"], "A3": ["R2", "R3", "A1", "A2", "TWEEDIE_VALID"], "A4": ["R2", "R3", "A1", "A2", "A3", "TWEEDIE_VALID"],
                "B1": ["R2", "TWEEDIE_VALID"], "B2": ["R2", "B1", "TWEEDIE_VALID"],
                "C1": ["R1", "TWEEDIE_VALID"], "C2": ["TWEEDIE_VALID"], "C3": ["R1", "C1", "C2", "TWEEDIE_VALID"],
                "CONTROL_A": ["R2", "R3", "A1", "A2", "A3", "A4", "TWEEDIE_VALID"],
                "CONTROL_B": ["R2", "B1", "B2", "TWEEDIE_VALID"], "CONTROL_C": ["R1", "C1", "C2", "C3", "TWEEDIE_VALID"],
            },
        )
        self.assertEqual(
            self._v4_candidate()["branch_eligibility"]["primary_dataset_manifest_lineage"],
            {
                "binding": "every named real-data gate input must carry the identical content-derived FrozenPrimaryDatasetManifest payload and digest",
                "required_by_final": {
                    "FINAL_REAL": ["R1", "R2", "R3"],
                    "FINAL_A": ["R1", "R2", "R3", "A1", "A2", "A3", "A4", "CONTROL_A"],
                    "FINAL_B": ["R1", "R2", "B1", "B2", "CONTROL_B"],
                    "FINAL_C": ["R1", "C1", "C3", "CONTROL_C"],
                },
                "mandatory_scope": ["aggregate gates", "latency gate A4", "negative-control gates"],
                "selection_source": "verified COUNT_PRIMARY_DATASET_SELECTION_MANIFEST reconstructed from all five sealed dataset audits",
                "exact_dataset_coverage": "every real confirmatory target/forecast dataset_id must be selected, and each real macro/aggregate gate must contain exactly all selected datasets once in manifest order",
                "exempt": ["TWEEDIE_VALID because it is global", "C2 because it is synthetic only"],
                "mismatch": "mixed or missing manifest is an integrity contract error; create no final GateResult",
            },
        )
        self.assertEqual(
            self._v4_candidate()["gate_input_validity"],
            {
                "malformed_required_input": {
                    "cases": ["missing required value", "None where not explicitly registered", "NaN or Infinity", "wrong lineage or manifest"],
                    "action": "integrity ContractViolation; create no scientific GateResult and never serialize a scientific FAIL as a substitute",
                },
                "registered_domain_undefined": {
                    "serialization": "JSON null plus an exact dataset-keyed reason token; never NaN, Infinity, zero, or an imputed score",
                    "criterion": False,
                    "B1_reasons": ["INSUFFICIENT_VARIATION"],
                    "C1_reasons": ["SINGLE_CLASS", "EMPTY_FAILURE_SET", "REAL_C_SENSOR_GEOMETRY_BLOCKED"],
                    "C3_reasons": ["EMPTY_FAILURE_SET", "ZERO_FLAGGED_VALIDATION_ROWS"],
                },
                "A2_nonpositive_recovery_denominator": {
                    "condition": "L_best_single_teacher-L_teacher_pool<=0",
                    "recovery": None,
                    "reason": "NONPOSITIVE_RECOVERY_DENOMINATOR",
                    "criterion": False,
                    "FAIL": "A_DISTILLATION_RECOVERY_NO_GO",
                },
                "R2_correlation": "a constant or otherwise undefined Pearson correlation in any observed or bootstrap replicate is recorded as degenerate and makes the clear-below-1 criterion false; never coerce it to zero",
            },
        )

    def test_v4_freezes_smoke_and_stage_measurement_contracts(self):
        """Catches silently dropping an end-to-end smoke check or a required preregistered comparison."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["stage_contracts"]["S0_smoke"],
            [
                "three teacher fits", "checkpoint save and restore", "distribution prediction",
                "CDF", "quantile", "sCRPS", "linear CDF pool", "hard-only student one epoch",
                "structure feature", "disagreement feature", "bootstrap", "report serialization",
            ],
        )
        self.assertEqual(candidate["stage_contracts"]["S2_oracle_ladder"], ["best global single head", "best head by d", "best head by cell", "series-origin oracle head", "validation-selected CDF pool"])
        self.assertEqual(candidate["stage_contracts"]["R1_report"], ["six evaluation origins or equivalent contract", "identical series/origin/step", "model seed", "parameter count", "runtime", "sCRPS", "zero-Brier", "q50/q90/q95/q99 scaled quantile loss", "NRMSE", "calibration", "winner share"])
        self.assertEqual(
            candidate["runtime"]["timing_smoke"]["measurement_boundaries"],
            {
                "training": "one fresh deterministic fit per NB/HSNB/TWEEDIE_FULL with no untimed training warmup or repeated fit; synchronize CUDA immediately before start and after finish",
                "training_includes": ["input host-to-device transfer", "forward/backward/optimizer", "every scheduled validation sCRPS check", "checkpoint serialization"],
                "training_excludes": ["source loading and hashing", "DGP or real preprocessing", "window construction", "environment audit", "final report serialization"],
                "student_one_epoch": "same boundaries as training, exactly one epoch",
                "inference_warmup_iterations": 5,
                "inference_timed_iterations": 20,
                "inference_statistic": "median GPU seconds per frozen rate unit",
                "inference_includes": ["input host-to-device transfer", "model forward", "distribution transform", "21-grid native or pooled inversion/postprocess", "immediate output transfer to CPU"],
                "inference_excludes": ["checkpoint disk loading", "input/window construction", "target metric reduction", "report serialization"],
                "P2_complete_selector_override": "time one end-to-end complete 66-state validation selection over the smoke validation cases; include every state's 21-grid pooled inversion, sealed equal_group_macro sCRPS/tail reduction, and frozen tie selection, so the generic target-metric exclusion does not apply to r_P2_selector",
                "CUDA_sync": "before and after every warmup and timed region",
                "router_sensor": "CPU wall-time report only and excluded from projected FULL GPU seconds",
            },
        )
        self.assertEqual(candidate["environment"]["packages"]["openpyxl"], "3.1.2")
        self.assertEqual(
            candidate["environment"]["required_distribution_versions"],
            {
                "torch": "2.1.1+cu118", "numpy": "1.26.4", "pandas": "2.3.3",
                "scipy": "1.12.0", "scikit-learn": "1.4.1.post1", "pyarrow": "25.0.1",
                "matplotlib": "3.10.9", "pytest": "8.4.2", "tweedie": "0.0.9", "openpyxl": "3.1.2",
            },
        )
        self.assertEqual(
            candidate["stage_contracts"]["A4_benchmark"],
            {
                "batch_sizes": [1, 256],
                "input": "fixed deterministic synthetic-shaped lookback-96 history",
                "devices": {"CPU": "torch threads=1", "CUDA": "measured separately"},
                "warmup_iterations": 20,
                "timed_iterations": 100,
                "latency_statistic": "median",
                "CUDA_timing": "synchronize before and after every timed region",
                "peak_memory": "reset and read peak allocated bytes for each measured path",
                "single_teacher": "validation-selected P0 single teacher",
                "teacher_pool": "sequential three-teacher forwards plus p0 and common-quantile pooling",
                "student": "one forward plus postprocess",
                "common_conditions": ["same dtype", "same device", "no compile", "eval mode", "no_grad"],
                "latency_gate_scope": "every CPU and CUDA combination at batch 1 and 256 must pass every latency constraint",
                "CUDA_unavailable": "A4 FAIL",
                "CUDA_peak_memory": "peak allocated bytes at each batch",
                "CPU_memory": "RSS diagnostic only",
            },
        )

    def test_v4_freezes_S3_contrast_and_all_metric_reduction_formulas(self):
        """Catches changing the temporal effect scale or dataset macro aggregation after results exist."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["stage_contracts"]["S3_analysis"],
            {
                "dependent_gap": "100*(sCRPS_head_a-sCRPS_head_b)/sCRPS_head_b",
                "ordered_pairs": ["NB_vs_HSNB", "NB_vs_TWEEDIE_FULL", "HSNB_vs_TWEEDIE_FULL"],
                "cell_order": {"d": [4, 8], "rho_I": [-0.8, 0.0, 0.8], "rho_M": [-0.8, 0.0, 0.8]},
                "orthogonal_basis": {
                    "D": {"levels": [4, 8], "weights": [-1.0, 1.0], "effect": "d=8 minus d=4"},
                    "L": {"levels": [-0.8, 0.0, 0.8], "weights": [-1.0, 0.0, 1.0], "effect": "+0.8 endpoint minus -0.8 endpoint"},
                    "Q": {"levels": [-0.8, 0.0, 0.8], "weights": [0.5, -1.0, 0.5], "effect": "mean of endpoints minus center"},
                },
                "contrast_names": [
                    "D", "I_L", "I_Q", "M_L", "M_Q",
                    "D:I_L", "D:I_Q", "D:M_L", "D:M_Q",
                    "I_L:M_L", "I_L:M_Q", "I_Q:M_L", "I_Q:M_Q",
                    "D:I_L:M_L", "D:I_L:M_Q", "D:I_Q:M_L", "D:I_Q:M_Q",
                ],
                "contrast_count": 17,
                "estimator": "sum cell means times the Kronecker product of the named basis weights; uniformly average every omitted factor",
                "scaling": "D is a two-level difference; L is an endpoint difference; Q is endpoint-mean minus center; interaction weights are exact products with no post-hoc multiplier",
                "cluster_bootstrap": "within each (data_seed,d), resample base_innovation_id clusters with all nine rho_I by rho_M cells and all model seeds attached; d strata are resampled independently",
                "confidence_interval": "percentile 2.5/97.5 base-innovation cluster-bootstrap CI with seed 2026090531",
            },
        )
        q_weights = [0.5, -1.0, 0.5]
        l_weights = [-1.0, 0.0, 1.0]
        pure_quadratic = [1.0, 0.0, 1.0]
        self.assertEqual(sum(w * y for w, y in zip(l_weights, pure_quadratic)), 0.0)
        self.assertEqual(sum(w * y for w, y in zip(q_weights, pure_quadratic)), 1.0)
        self.assertEqual(
            candidate["metrics"]["formulas"],
            {
                "sQL_q": "2*pinball(y,Q(q))/s_i",
                "NRMSE": "form step normalized squared error=((predictive_mean-y)/s_i)^2; average in the frozen step-to-series-origin-to-series-to-dataset order, then take sqrt at each reported aggregation level",
                "NMAE": "form step normalized absolute error=abs(predictive_mean-y)/s_i and average in the frozen step-to-series-origin-to-series-to-dataset order",
                "coverage_error": "abs(empirical_coverage-nominal_coverage)",
                "interval_width": "(upper_quantile-lower_quantile)/s_i",
                "tail_metric": "mean(sQL_q95,sQL_q99)",
            },
        )
        self.assertEqual(candidate["metrics"]["aggregation_order"], ["step", "series-origin", "series", "dataset"])
        self.assertEqual(candidate["metrics"]["real_macro"], "equal weight per dataset")
        self.assertEqual(candidate["metrics"]["central_intervals"], {"50%": [0.25, 0.75], "80%": [0.10, 0.90], "90%": [0.05, 0.95], "95%": [0.025, 0.975]})
        self.assertEqual(candidate["metrics"]["coverage_only_quantiles"], [0.025, 0.975])
        self.assertEqual(candidate["metrics"]["coverage_quantile_source"], {"native_teachers_and_P1_P2": "exact or numerical inverse CDF", "student": "monotone piecewise-linear interpolation between 0.01/0.05 and 0.95/0.99", "P3": "p0-aware zero plateau then monotone interpolation at .025/.975, followed by the common+derived biconditional coherence check; plain interpolation is forbidden", "report_field": "quantile_source"})
        self.assertEqual(candidate["bootstrap"]["percentile_interval"], [0.025, 0.975])
        self.assertEqual(candidate["bootstrap"]["draws_by_tier"], {"FULL": 2000, "COMPACT": 1000, "MINIMAL-COMPLETE": 500})

    def test_v4_freezes_deterministic_quantiles_pool_inversion_and_exact_P3_path(self):
        """Catches Monte-Carlo gate noise or a result-sensitive P3 search/inversion implementation."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["metrics"]["scientific_distribution_evaluation"],
            {
                "teachers": "deterministic native head quantiles",
                "P1_P2": "deterministic numerical inverse of the linear CDF pool",
                "student_and_P3": "deterministic monotone quantile representation",
                "empirical_samples_in_checkpoint_gate_or_pool_selection": "forbidden",
                "diagnostic_sample_moment_stability_draws": {"validation": 256, "evaluation": 1024},
                "draw_count_changes_between_calls": "not applicable and forbidden because scientific metrics do not draw",
            },
        )
        self.assertEqual(
            candidate["predictive_distribution_interface"]["pooled_quantile_inversion"],
            {
                "lower": 0.0,
                "initial_upper": "max(1,maximum teacher mean,maximum initial teacher q99)",
                "upper_expansion": "double until pooled CDF>=q",
                "maximum_expansions": 128,
                "bisection_iterations": 128,
                "support_tolerance": "max(1e-6,1e-6*upper)",
                "return": "upper endpoint generalized inverse",
                "probability_domain": "0<q<1",
                "zero_rule": "evaluate F(0) first and return exactly 0 whenever F(0)>=q, including q=p0",
                "callable_validation": "every evaluated CDF value must be finite, lie in [0,1], and be nondecreasing over all ordered probes and bisection brackets",
                "non_bracketing": "after 128 expansions raise CDF_INVERSION_NONBRACKETING_HARD_STOP; never return the last upper bound",
                "invalid_callable": "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP",
            },
        )
        self.assertEqual(
            candidate["pools"]["P3"]["optimization"],
            {
                "row_loss": "ell(q,w)=mean_rows pinball(y,sum_m w_m*Q_m(q))/s_i",
                "path_objective": "mean_q ell(q,w_q)+penalty*mean_adjacent ||w_q-w_(q-1)||_2^2",
                "p0_selection": ["minimum validation Brier score", "lexicographically smallest (w_NB,w_HSNB,w_TWEEDIE_FULL) on exact tie"],
                "per_penalty": "solve the full path; exact objective ties choose the lexicographically smallest full weight path",
                "postprocess": ["cumulative maximum", "set all grid quantiles q<=pooled p0 to zero", "cumulative maximum"],
                "coherence_filter": "before target-metric scoring exclude every candidate violating q<=p0 iff Q(q)=0 on the common grid plus q=.025/.975",
                "penalty_selection": ["minimum postprocessed validation sCRPS", "minimum postprocessed validation mean(q95,q99 sQL)", "smaller penalty", "lexicographically smallest full path"],
            },
        )
        self.assertEqual(
            candidate["pools"]["selection_provenance"]["validation_grouping"],
            {
                "seal": "SealedValidationArtifact stores exactly one finite hashable group ID per sealed case plus validation_group_sha256",
                "real_group_id": "series_id so every real validation selector is series-equal",
                "synthetic_S2_P2_group_id": "canonical (data_seed,d,rho_I,rho_M) cell identity so all 18 cells are equal-weighted",
                "aggregation": "mean loss within group, then equal arithmetic mean over frozen groups; FULL model-seed scores remain equally averaged",
                "caller_regrouping": "forbidden; supplied group IDs must byte-for-byte equal the sealed ordered IDs",
            },
        )
        self.assertEqual(candidate["pools"]["P2"]["validation_aggregation"], "equal_group_macro")
        self.assertEqual(
            candidate["pools"]["P3"]["distribution_contract"],
            {
                "representation": "separately pooled p0 plus a quantile-specific convex path; P3 is not a linear CDF pool and no single F_mix is claimed",
                "checked_grid": [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99],
                "derived_quantiles": {"0.025": "p0-aware monotone interpolation: return zero when .025<=p0, otherwise linearly interpolate Q(.01),Q(.05) and require positive", "0.975": "p0-aware monotone interpolation: return zero when .975<=p0, otherwise linearly interpolate Q(.95),Q(.99) and require positive"},
                "bidirectional_zero_mass_rule": "for every case and checked q: q<=p0 iff abs(Q(q))<=1e-12; therefore every q>p0 must have Q(q)>1e-12",
                "all_candidates_infeasible": "raise P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD, mark P3 unavailable, and continue P2 without changing or retuning P2",
                "selected_prediction_recheck": ["validation", "outer_evaluation", "train_rolling_origin_teacher_rows"],
                "recheck_is_prediction_only": "use sealed teacher predictions and frozen selection without targets",
                "recheck_failure": "P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD with no performance fallback, retuning, or replacement by another P3 candidate",
                "predictive_mean": "trapezoidal integral of the coherent postprocessed common 21-grid quantiles with Q(.01) held to q=0 and Q(.99) held to q=1; coverage-only q=.025/.975 are not extra integration knots",
                "mean_source": "P3_postprocessed_quantile_integral_endpoint_hold",
            },
        )
        self.assertEqual(
            candidate["teacher_generation"]["source_semantics"],
            {
                "best_single": "validation-selected P0 native distribution",
                "P1_P2": "linear CDF pool F_T(y|x)=sum_m w_m F_m(y|x), with teacher quantiles from deterministic numerical inverse",
                "P3": "coherent separately pooled p0 and postprocessed quantile-specific convex path; never describe or consume it as one linear CDF",
                "P3_mean": "P3_postprocessed_quantile_integral_endpoint_hold",
            },
        )
        self.assertEqual(candidate["pools"]["primary_selection"], ["minimum postprocessed validation sCRPS", "minimum postprocessed validation mean(q95,q99 sQL)", "P2 on exact tie"])
        self.assertEqual(
            candidate["metrics"]["improvement_contract"],
            {
                "fraction": "(L_base-L_candidate)/L_base",
                "percentage_report": "100*fraction",
                "nonpositive_denominator": "for a required baseline loss, integrity ContractViolation and no scientific GateResult",
                "S2_gate_inputs": {"cell_oracle": 0.02, "series_origin_oracle": 0.03},
                "S2_percentage_point_reports": {"cell_oracle": 2.0, "series_origin_oracle": 3.0},
            },
        )

    def test_p3_coherence_adversary_enforces_both_zero_mass_directions(self):
        """Catches zeroing q>p0 or leaving q<=p0 positive in the common/derived-grid validator."""
        q = np.asarray(CRPS_QUANTILE_GRID, dtype=np.float64)
        quantiles = np.ones((1, q.size), dtype=np.float64)
        quantiles[:, :2] = 0.0
        p_zero = np.asarray([0.03], dtype=np.float64)
        with self.assertRaisesRegex(
            PredictionIntegrityError,
            "P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD",
        ):
            _require_p3_grid_coherence(quantiles, p_zero, q)

        coherent = np.ones((1, q.size), dtype=np.float64)
        coherent[:, 0] = 0.0
        _require_p3_grid_coherence(coherent, p_zero, q)

    def test_callable_cdf_inversion_honors_zero_mass_and_rejects_invalid_callables(self):
        """Catches q=p0 drifting above zero or a non-bracketing callable returning a fake quantile."""
        zero_mass_cdf = lambda x: np.where(np.asarray(x) <= 0.0, 0.2, 1.0)
        result = invert_pooled_cdf(
            cdf_functions=(zero_mass_cdf, zero_mass_cdf, zero_mass_cdf),
            weights=np.asarray([1.0, 0.0, 0.0]),
            probabilities=[0.2],
            case_count=1,
            initial_upper=1.0,
        )
        self.assertEqual(result["quantiles"], [[0.0]])

        never_brackets = lambda x: np.full(np.asarray(x).shape, 0.2, dtype=np.float64)
        with self.assertRaisesRegex(
            PredictionIntegrityError,
            "CDF_INVERSION_NONBRACKETING_HARD_STOP",
        ):
            invert_pooled_cdf(
                cdf_functions=(never_brackets, never_brackets, never_brackets),
                weights=np.asarray([1.0, 0.0, 0.0]),
                probabilities=[0.9],
                case_count=1,
                initial_upper=1.0,
            )

        nonfinite = lambda x: np.full(np.asarray(x).shape, np.nan, dtype=np.float64)
        with self.assertRaisesRegex(
            PredictionIntegrityError,
            "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP",
        ):
            invert_pooled_cdf(
                cdf_functions=(nonfinite, nonfinite, nonfinite),
                weights=np.asarray([1.0, 0.0, 0.0]),
                probabilities=[0.5],
                case_count=1,
                initial_upper=1.0,
            )

        decreasing = lambda x: np.where(np.asarray(x) <= 0.0, 0.8, 0.7)
        with self.assertRaisesRegex(
            PredictionIntegrityError,
            "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP",
        ):
            invert_pooled_cdf(
                cdf_functions=(decreasing, decreasing, decreasing),
                weights=np.asarray([1.0, 0.0, 0.0]),
                probabilities=[0.9],
                case_count=1,
                initial_upper=1.0,
            )

    def test_v4_freezes_full_workload_runtime_exposure_formula(self):
        """Catches selecting a tier from a downsized or gate-short-circuited workload estimate."""
        runtime = self._v4_candidate()["runtime"]["FULL_projection"]
        self.assertEqual(
            runtime,
            {
                "rate_units": {
                    "teacher_train": {
                        "NB": "r_train_NB=GPU seconds/(observed smoke train-window*completed epoch)",
                        "HSNB": "r_train_HSNB=GPU seconds/(observed smoke train-window*completed epoch)",
                        "TWEEDIE_FULL": "r_train_TWEEDIE_FULL=GPU seconds/(observed smoke train-window*completed epoch)",
                    },
                    "student_train": "r_train_student=GPU seconds/(observed smoke train-window*completed epoch)",
                    "native_21q_inference": {
                        "NB": "r_native_NB_21q=GPU seconds/(flat forecast case for end-to-end native 21-grid quantile inference)",
                        "HSNB": "r_native_HSNB_21q=GPU seconds/(flat forecast case for end-to-end native 21-grid quantile inference)",
                        "TWEEDIE_FULL": "r_native_TWEEDIE_FULL_21q=GPU seconds/(flat forecast case for end-to-end native 21-grid quantile inference)",
                    },
                    "linear_CDF_pool_21q": "r_linear_pool_21q=GPU seconds/(flat forecast case for end-to-end 21-grid pooled numerical inversion)",
                    "P2_complete_66_state_validation_selector": "r_P2_selector=GPU-synchronized elapsed seconds/(flat validation forecast case) for one complete 66-state selector: every state performs all 21 pooled numerical inversions under the frozen inversion contract, equal_group_macro metric reduction, and frozen tie selection",
                    "P3_direct_21q": "r_P3_direct_21q=GPU seconds/(flat forecast case for direct postprocessed 21-grid P3 output)",
                    "student_direct_21q": "r_student_direct_21q=GPU seconds/(flat forecast case for direct postprocessed 21-grid student output)",
                },
                "smoke_rate_measurements": [
                    "r_train_NB", "r_train_HSNB", "r_train_TWEEDIE_FULL", "r_train_student",
                    "r_native_NB_21q", "r_native_HSNB_21q", "r_native_TWEEDIE_FULL_21q",
                    "r_linear_pool_21q", "r_P2_selector", "r_P3_direct_21q", "r_student_direct_21q",
                ],
                "pre_fit_inventory": {
                    "m5": {"source_geometry_series": 30490, "FULL_N_upper_bound": 4000},
                    "online_retail": {"source_geometry_series": 2036, "FULL_N_upper_bound": 2036},
                    "role": "source/geometry inventory only; formal support and eligibility audit remains at its frozen later stage",
                },
                "always_use": {"epochs": 30, "gate_short_circuit": False, "validation_checks": 15, "prediction_cache": "only the explicitly named A2/A3 pool targets are cached once"},
                "fit_counts": {
                    "S1": "18 cells*2 DGP seeds*2 model seeds*3 heads=216 teacher fits",
                    "real_teachers": "2 inventory datasets*2 model seeds*3 heads=12 teacher fits",
                    "A_per_dataset_seed": "A0 one plus A1-A4 each three lambdas=13 student fits",
                    "B_per_dataset_seed": "B0-B2 each three lambdas=9 student fits",
                    "A_destroying_controls_per_dataset_seed": "2 controls*3 lambdas=6 student fits",
                    "B_controls_per_dataset_seed": "3 controls*3 lambdas=9 student fits; each control selects its router temperature by the same prior-only crossfit contract",
                },
                "symbols": {"H": 28, "N_syn": 80, "W_syn": 257, "V": 15, "O_syn": 5, "C_SYN_types": 6, "C_SYN_d": 2, "C_SYN_origins": 16, "O_real": 6, "I_B": 8, "K_B": 7, "I_C_pairs": 8, "W_head_m5": 228, "W_head_online_retail": 4, "W_A_m5": 57, "W_A_online_retail": 1, "P2_simplex_states": 66, "P3_penalties": 4, "student_variants": 37},
                "training_work_units": {
                    "teacher_each_head": "U_teacher_head=18*2*2*80*257*30 + 2*4000*228*30 + 2*2036*4*30",
                    "student": "U_student=2*4000*30*(19*57+18*7) + 2*2036*30*(19*1+18*7)",
                },
                "native_21q_case_units_each_head": {
                    "S1_validation_and_outer": "18*2*2*80*(15+5)*28",
                    "C_SYN_all_six_panels": "6*2*2*2*80*16*28",
                    "real_validation_A_B_R": "2*4000*(15+57+8+6)*28 + 2*2036*(15+1+8+6)*28",
                    "real_C_sensor_conservative_no_cache": "2*4000*(2*8+2+2*6)*28 + 2*2036*(2+2*6)*28",
                },
                "linear_pool_21q_case_units": {
                    "synthetic_P2_outer": "18*2*2*80*5*28",
                    "real_P1_and_P2_outer": "2*(2*4000+2*2036)*6*28",
                    "A2_and_A3_train_targets_cached_once_each": "2*(2*4000*57+2*2036*1)*28",
                    "A4_pool_benchmark": "2*2*(1+256)*(20+100)*28",
                },
                "P2_complete_selector_case_units": {
                    "synthetic_validation_cases_with_seed_exposure": "18*2*2*80*1*28",
                    "real_dataset_validation_cases_with_seed_exposure": "(2*4000+2*2036)*1*28",
                    "U_P2_selection": "18*2*2*80*1*28 + (2*4000+2*2036)*1*28",
                    "state_and_quantile_work_inside_rate": "each case rate already includes all 66 simplex states and all 21 numerical inversions per state; never multiply U_P2_selection by 66 again",
                },
                "P3_direct_21q_case_units": {
                    "validation_all_p0_states_and_penalty_paths": "(66+4*66)*(2*4000+2*2036)*1*28",
                    "outer_selected_path": "(2*4000+2*2036)*6*28",
                    "A4_train_targets_cached_once": "(2*4000*57+2*2036*1)*28",
                },
                "student_direct_21q_case_units": {
                    "validation_and_outer_all_variants": "37*(2*4000+2*2036)*(15+6)*28",
                    "A4_student_benchmark": "2*2*(1+256)*(20+100)*28",
                },
                "A4_native_GPU_benchmark_case_units": "2*2*(1+256)*(20+100)*(1+3)*28 using max native-head rate; pool and student benchmark units are included in their respective maps",
                "conservative_geometry": "every inventory-selected series is assumed to expose every N*W window; this is an availability upper bound, not the later scientific dataset audit",
                "formula": "projected_FULL_GPU_sec=1.25*(sum_head[r_train_head*U_teacher_head]+r_train_student*U_student+sum_head[r_native_head_21q*U_native_head_21q]+max_head(r_native_head_21q)*U_A4_native+r_P2_selector*U_P2_selection+r_linear_pool_21q*U_linear_pool_21q+r_P3_direct_21q*U_P3_direct_21q+r_student_direct_21q*U_student_direct_21q)",
                "CPU_only": ["router fitting", "sensor fitting", "report generation"],
                "selection": "use this single projected FULL GPU-seconds value once; never recompute after tier downsizing",
            },
        )

    def test_v4_freezes_runtime_decision_as_the_only_synthetic_size_and_seed_authority(self):
        """Catches caller-selected S1/C-SYN seeds or series counts after the S0 tier decision."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["runtime"]["tier_decision_artifact"],
            {
                "record_type": "PROB_HEAD_STRUCTURE_RUNTIME_TIER_DECISION",
                "builder": "seal_runtime_tier_decision",
                "verifier": "verify_runtime_tier_decision",
                "bindings": ["official preregistration payload SHA256", "official preregistration whole-file SHA256", "S0 smoke projection SHA256"],
                "one_time_rule": "derive the tier once from projected FULL GPU seconds; recomputation after downsizing is forbidden",
                "tier_contracts": {
                    "FULL": {"synthetic_series_per_cell": 80, "synthetic_data_seeds": [2026090501, 2026090502], "teacher_model_seeds": [2026090511, 2026090512], "real_series_per_dataset": 4000, "student_model_seeds": [2026090511, 2026090512], "bootstrap_draws": 2000, "screen_only": False},
                    "COMPACT": {"synthetic_series_per_cell": 40, "synthetic_data_seeds": [2026090501], "teacher_model_seeds": [2026090511], "real_series_per_dataset": 2000, "student_model_seeds": [2026090511], "bootstrap_draws": 1000, "screen_only": False},
                    "MINIMAL-COMPLETE": {"synthetic_series_per_cell": 24, "synthetic_data_seeds": [2026090501], "teacher_model_seeds": [2026090511], "real_series_per_dataset": 1000, "student_model_seeds": [2026090511], "bootstrap_draws": 500, "screen_only": True},
                },
                "consumer_rule": "all downstream consumers verify the decision against the same official preregistration bytes before reading its tier contract",
            },
        )
        self.assertEqual(
            candidate["synthetic"]["runtime_decision_generation"],
            {
                "S1_audit_inputs": "runtime_decision and official preregistration path only; caller expected seeds or n_series are forbidden",
                "S1_complete_grid": "18 cells for every synthetic_data_seed in the sealed tier contract, each with synthetic_series_per_cell series",
                "S1_raw_regeneration_seal": "rebuild each generated block from the verified source and sealed (data_seed,d,n_series), compare generated_block_sha256, and require one common-base identity across the nine rho_I by rho_M cells at fixed (data_seed,d)",
                "C_SYN_audit_inputs": "runtime_decision and official preregistration path only; caller expected seeds or n_series are forbidden",
                "C_SYN_complete_grid": "2 d values*6 exact shift types*len(tier synthetic_data_seeds), with synthetic_series_per_cell series in every d-shift block",
                "C_SYN_block_counts_by_tier": {"FULL": 24, "COMPACT": 12, "MINIMAL-COMPLETE": 12},
                "C_SYN_raw_regeneration_seal": "rebuild every known-change block from the verified source and sealed (data_seed,d,shift_type,n_series), compare known_change_block_sha256, and require the six shifts to share one common-base identity at fixed (data_seed,d)",
            },
        )
        self.assertEqual(candidate["stage_contracts"]["runtime_tier_authority"], "S0 publishes one hash-sealed runtime-tier decision; S1 DGP audit, C-SYN audit, real sampling, seed replication, and bootstrap counts consume only its verified tier contract")
        self.assertEqual(candidate["cross_api_contracts"]["runtime_tier_decision"], "seal_runtime_tier_decision binds projected FULL GPU seconds and S0 projection SHA256 to the exact official preregistration payload/file SHA256 and exact tier mapping; downstream APIs reject detached, caller-sized, or caller-seeded execution")

    def test_v4_freezes_exhaustive_recommendation_truth_table_and_cross_api_seals(self):
        """Catches recommendation overlap or an unsealed data/prediction object crossing APIs."""
        candidate = self._v4_candidate()
        recommendation = candidate["recommendation"]
        self.assertEqual(
            recommendation["truth_table_inputs"],
            {
                "integrity_blocked": "any registered hard integrity failure",
                "A_confirmatory_GO": "FINAL_A=DISTRIBUTION_SPACE_DISTILLATION_GO and confirmatory_eligible=true",
                "C_confirmatory_GO": "FINAL_C=DISAGREEMENT_SENSOR_GO and confirmatory_eligible=true",
                "B_confirmatory_GO": "FINAL_B=STRUCTURE_CONDITIONED_ROUTING_GO and confirmatory_eligible=true",
                "characterization_clear": "S3=TEMPORAL_STRUCTURE_EFFECT_GO or R2=REAL_HEAD_COMPLEMENTARITY_GO",
            },
        )
        rows = recommendation["exhaustive_boolean_truth_table"]
        self.assertEqual(len(rows), 32)
        observed = {
            (
                row["integrity_blocked"], row["A_confirmatory_GO"],
                row["C_confirmatory_GO"], row["B_confirmatory_GO"],
                row["characterization_clear"],
            )
            for row in rows
        }
        self.assertEqual(observed, set(itertools.product([False, True], repeat=5)))
        for row in rows:
            if row["integrity_blocked"]:
                expected = "INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT"
            elif row["A_confirmatory_GO"]:
                expected = "RECOMMEND_A_DISTRIBUTION_DISTILLATION"
            elif row["C_confirmatory_GO"]:
                expected = "RECOMMEND_C_DISAGREEMENT_SENSOR"
            elif row["B_confirmatory_GO"]:
                expected = "RECOMMEND_B_STRUCTURE_CONDITIONED_ROUTING"
            elif row["characterization_clear"]:
                expected = "RECOMMEND_CHARACTERIZATION_ONLY"
            else:
                expected = "ALL_NEW_METHOD_BRANCHES_NO_GO"
            self.assertEqual(row["token"], expected)
        self.assertEqual(recommendation["precedence"], ["integrity", "A", "C", "B", "characterization", "all_no_go"])

        self.assertEqual(
            candidate["cross_api_contracts"],
            {
                "availability_aware_train_RMS": "s_i=sqrt(mean(y_i[available_from:model_train_end]^2)+1e-8); require at least one finite observed train cell and never zero-pad pre-availability cells into the mean",
                "WindowRequest": "hash-sealed canonical dataset_id, role, frozen split, panel length, origins, dataset_audit_sha256, sample_manifest_sha256, sampled_panel_binding_sha256, sampling runtime tier and seed, split-contract SHA256, and request SHA256; construction requires matching sealed dataset and train-only sample manifests and target intervals must be contained in the role-bound split",
                "count_dataset_audit": "a geometry-only audit is always ineligible; seal_count_primary_dataset_audit binds actual panel bytes, availability, series IDs, exact model-train count support, split, and verified source records before selection",
                "primary_dataset_manifest": "verify COUNT_PRIMARY_DATASET_SELECTION_MANIFEST by recomputing the complete canonical five-audit fixed-priority selection, then emit one content-bound FrozenPrimaryDatasetManifest; arbitrary 64-hex digests, mixed manifests, unselected dataset IDs, and incomplete selected-dataset coverage are rejected",
                "train_only_sample_manifest": "seal_train_only_sample_manifest uses seed 2026090521 and exact tier sizes FULL=4000, COMPACT=2000, MINIMAL-COMPLETE=1000; it recomputes train-only availability-aware descriptors, joint quartiles, proportional-largest-remainder allocation, eligible pool, selected IDs/positions, and sampled-panel binding from the sealed full panel",
                "history_window_execution": "make_history_windows requires matching sealed dataset and train-only sample manifests, reconstructs the selected subset from the full panel/raw_y plus ordered series IDs and availability binding, rejects detached subsets and caller-supplied scale, and internally recomputes availability-aware train RMS",
                "evaluation_target": "SealedEvaluationTarget reconstructs the WindowBatch from the full sealed panel and matching dataset/sample manifests, then binds split role/bounds, exact series-origin-step keys, targets, valid-target mask, internally recomputed availability-aware train RMS scale, dataset-audit/sample-manifest/sampled-panel/source/preregistration/primary-dataset-manifest hashes",
                "validation_selection": "SealedValidationArtifact binds dataset, validation interval, exact case-key hash, ordered validation-group IDs plus validation_group_sha256, validation target/scale hash, teacher-seed identities, and teacher-prediction component hashes; selectors use equal_group_macro and reject caller regrouping or outer/test arrays relabeled as validation",
                "runtime_tier_decision": "seal_runtime_tier_decision binds projected FULL GPU seconds and S0 projection SHA256 to the exact official preregistration payload/file SHA256 and exact tier mapping; downstream APIs reject detached, caller-sized, or caller-seeded execution",
                "scientific_forecast": "confirmatory EvaluationResult requires sealed target and forecast provenance with quantile_source native_exact_or_numerical_inverse for teacher/P1/P2 or monotone_piecewise_common_grid for student/P3; empirical/sample helpers are diagnostic only",
                "count_CDF_precision": "NB and HSNB exact PMF/CDF accumulation uses float64 even when model training/inference parameters originate in float32",
                "Tweedie_reference_failure": "reference failures are branch-local, JSON-safe finite/null-and-string records; NaN/Infinity never enters an artifact and affected rows cannot be hidden or clamped",
                "canonical_seed_identities": {
                    "FULL_real_teacher_model_seeds": [2026090511, 2026090512],
                    "student_pairing": {"2026090511": 2026090511, "2026090512": 2026090512},
                    "FULL_synthetic_data_seeds": [2026090501, 2026090502],
                    "FULL_synthetic_model_seeds": [2026090511, 2026090512],
                    "synthetic_row_identity": ["data_seed", "model_seed", "d", "rho_I", "rho_M", "base_innovation_id", "origin"],
                },
                "bootstrap_clusters": {
                    "real": "series_id, with all origins and both model-seed replicates attached",
                    "synthetic": "(data_seed,d,base_innovation_id), with all nine rho_I by rho_M cells, origins, and model seeds attached; never pair d=4 with d=8",
                },
            },
        )

    def test_v4_promotion_changes_only_authority_metadata_after_v3_invalidation(self):
        """Catches promotion from unauthenticated candidate bytes or a fabricated v3 invalidation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1 = root / "v1.json"; freeze_preregistration(v1, {"version": 1})
            v2 = root / "v2.json"; freeze_preregistration(v2, build_preregistered_spec_v2(v1))
            v3 = root / "v3.json"; freeze_preregistration(v3, build_preregistered_spec_v3(v1, v2))
            old_bytes = {path.name: path.read_bytes() for path in (v1, v2, v3)}
            candidate = build_preregistered_spec_v4(v1, v2, v3)
            attempt, resumed = reserve_or_resume_attempt(root / "runs", "v4_review")
            self.assertFalse(resumed)
            candidate_path = attempt / "preregistered_spec_v4_candidate.json"
            write_preregistration_review_candidate(candidate_path, candidate)
            candidate_companion = candidate_path.with_suffix(".json.sha256.json")
            completion = publish_completion_marker(
                attempt,
                {"stage": "v4_review", "official_freeze_performed": False},
                [candidate_path, candidate_companion],
            )
            completion_path = attempt / "completion.json"
            official_root = root / "results" / "prob_head_structure_full_v1"
            invalidation_path = official_root / "preregistration_invalidations_v3.json"
            invalidation = invalidate_preregistration_v3_before_fit(v1, v2, v3, invalidation_path)
            invalidation_companion = invalidation_path.with_suffix(".json.sha256.json")
            v4 = official_root / "preregistered_spec_v4.json"
            with self.assertRaisesRegex(ContractViolation, "candidate payload hash"):
                promote_v4_candidate_to_authoritative(
                    candidate_path,
                    destination=v4,
                    frozen_at_utc="2026-09-05T12:00:00Z",
                    expected_candidate_payload_sha256="0" * 64,
                    expected_candidate_file_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                    expected_candidate_companion_file_sha256=hashlib.sha256(candidate_companion.read_bytes()).hexdigest(),
                    expected_completion_file_sha256=hashlib.sha256(completion_path.read_bytes()).hexdigest(),
                    expected_completion_payload_sha256=completion["completion_payload_sha256"],
                    v1_path=v1, v2_path=v2, v3_path=v3,
                    v3_invalidation_path=invalidation_path,
                    v3_invalidation_file_sha256=hashlib.sha256(invalidation_path.read_bytes()).hexdigest(),
                    v3_invalidation_companion_file_sha256=hashlib.sha256(invalidation_companion.read_bytes()).hexdigest(),
                )
            published = promote_v4_candidate_to_authoritative(
                candidate_path,
                destination=v4,
                frozen_at_utc="2026-09-05T12:00:00Z",
                expected_candidate_payload_sha256=payload_sha256(candidate),
                expected_candidate_file_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                expected_candidate_companion_file_sha256=hashlib.sha256(candidate_companion.read_bytes()).hexdigest(),
                expected_completion_file_sha256=hashlib.sha256(completion_path.read_bytes()).hexdigest(),
                expected_completion_payload_sha256=completion["completion_payload_sha256"],
                v1_path=v1,
                v2_path=v2,
                v3_path=v3,
                v3_invalidation_path=invalidation_path,
                v3_invalidation_file_sha256=hashlib.sha256(invalidation_path.read_bytes()).hexdigest(),
                v3_invalidation_companion_file_sha256=hashlib.sha256(invalidation_companion.read_bytes()).hexdigest(),
            )
            authoritative = published["payload"]

            candidate_scientific = dict(candidate)
            candidate_scientific.pop("authority_status")
            authoritative_scientific = dict(authoritative)
            authoritative_scientific.pop("authority_status")
            freeze_metadata = authoritative_scientific.pop("freeze_metadata")
            self.assertEqual(authoritative_scientific, candidate_scientific)
            self.assertEqual(authoritative["authority_status"], "AUTHORITATIVE_FROZEN_BEFORE_ANY_MODEL_FIT")
            self.assertEqual(freeze_metadata["new_model_fits_completed"], 0)
            self.assertEqual(
                freeze_metadata["review_candidate_hashes"],
                {
                    "payload_sha256": payload_sha256(candidate),
                    "file_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                    "companion_file_sha256": hashlib.sha256(candidate_companion.read_bytes()).hexdigest(),
                    "completion_file_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
                    "completion_payload_sha256": completion["completion_payload_sha256"],
                },
            )
            self.assertEqual(invalidation["status"], "INVALID_BEFORE_ANY_MODEL_FIT")
            self.assertEqual({path.name: path.read_bytes() for path in (v1, v2, v3)}, old_bytes)
            marker_path = v4.with_suffix(".json.promotion_complete.json")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "AUTHORITATIVE_V4_PROMOTION_COMPLETE")
            self.assertEqual(
                marker["publication_order"],
                ["verified_existing_v3_invalidation_and_companion", "official_v4_envelope", "official_v4_companion", "promotion_terminal_marker"],
            )
            self.assertEqual(marker["hashes"]["official_v4_file_sha256"], hashlib.sha256(v4.read_bytes()).hexdigest())
            self.assertEqual(marker["hashes"]["official_v4_companion_file_sha256"], hashlib.sha256(v4.with_suffix(".json.sha256.json").read_bytes()).hexdigest())
            self.assertEqual(marker["hashes"]["v3_invalidation_file_sha256"], hashlib.sha256(invalidation_path.read_bytes()).hexdigest())
            self.assertEqual(verify_preregistration(v4)["payload"], authoritative)
            authoritative_for_direct_publish = json.loads(json.dumps(authoritative))
            frozen_bytes = v4.read_bytes()
            published["payload"]["authority_status"] = "MUTATED_RETURN_VALUE"
            self.assertEqual(v4.read_bytes(), frozen_bytes)
            self.assertEqual(
                verify_preregistration(v4)["payload"]["authority_status"],
                "AUTHORITATIVE_FROZEN_BEFORE_ANY_MODEL_FIT",
            )
            with self.assertRaisesRegex(ContractViolation, "promotion"):
                freeze_preregistration(
                    root / "direct-authoritative-v4.json",
                    authoritative_for_direct_publish,
                )
            with self.assertRaises(FileExistsError):
                invalidate_preregistration_v3_before_fit(v1, v2, v3, invalidation_path)

    def test_v4_freezes_crash_safe_atomic_promotion_contract(self):
        """Catches an official v4 becoming readable before every reviewed byte binding is durable."""
        contract = self._v4_candidate()["authoritative_promotion_contract"]
        self.assertEqual(
            contract,
            {
                "official_paths": {
                    "envelope": "results/prob_head_structure_full_v1/preregistered_spec_v4.json",
                    "companion": "results/prob_head_structure_full_v1/preregistered_spec_v4.json.sha256.json",
                    "terminal_marker": "results/prob_head_structure_full_v1/preregistered_spec_v4.json.promotion_complete.json",
                    "v3_invalidation": "results/prob_head_structure_full_v1/preregistration_invalidations_v3.json",
                    "v3_invalidation_companion": "results/prob_head_structure_full_v1/preregistration_invalidations_v3.json.sha256.json",
                },
                "review_preconditions": [
                    "exact expected candidate canonical-payload SHA256",
                    "exact expected candidate whole-file SHA256",
                    "exact expected candidate-companion whole-file SHA256 and companion contents",
                    "exact expected completion whole-file and completion-payload SHA256 binding exactly candidate plus companion",
                    "completion payload proves official_freeze_performed=false",
                ],
                "v3_preconditions": "the exact sibling invalidation and companion already exist, match caller-reviewed whole-file SHA256 values, and equal a fresh derivation from the immutable on-disk v1/v2/v3 chain",
                "candidate_equivalence": "rederive candidate from the proposed authoritative payload by removing only freeze_metadata and reverting authority_status; canonical bytes and reviewed payload SHA256 must match exactly",
                "publication_order": ["verified_existing_v3_invalidation_and_companion", "official_v4_envelope", "official_v4_companion", "promotion_terminal_marker"],
                "commit_point": "only exclusive creation of the terminal marker makes authoritative v4 readable; an envelope or companion without that marker is an incomplete hard stop",
                "collision_policy": "preflight and every exclusive create refuse any existing envelope, companion, or marker; never overwrite, delete, repair, or bless a partial publication",
                "failure_injection_points": ["after official_v4_envelope", "after official_v4_companion"],
                "parser": "candidate, companions, completion, invalidation, envelope, and marker reject duplicate keys and recursive nonfinite numbers including exponent overflow",
                "return_isolation": "return a deep copy only after terminal verification; caller mutation cannot change disk",
            },
        )

    def test_authoritative_promotion_failure_injection_never_commits_or_mutates_v3(self):
        """Catches a crash between publication files being mistaken for an atomic authoritative freeze."""

        def fixture(root: Path):
            v1 = root / "v1.json"; freeze_preregistration(v1, {"version": 1})
            v2 = root / "v2.json"; freeze_preregistration(v2, build_preregistered_spec_v2(v1))
            v3 = root / "v3.json"; freeze_preregistration(v3, build_preregistered_spec_v3(v1, v2))
            attempt, resumed = reserve_or_resume_attempt(root / "runs", "v4_review")
            self.assertFalse(resumed)
            candidate_path = attempt / "preregistered_spec_v4_candidate.json"
            candidate = build_preregistered_spec_v4(v1, v2, v3)
            write_preregistration_review_candidate(candidate_path, candidate)
            candidate_companion = candidate_path.with_suffix(".json.sha256.json")
            completion = publish_completion_marker(
                attempt,
                {"stage": "v4_review", "official_freeze_performed": False},
                [candidate_path, candidate_companion],
            )
            completion_path = attempt / "completion.json"
            official_root = root / "results" / "prob_head_structure_full_v1"
            invalidation_path = official_root / "preregistration_invalidations_v3.json"
            invalidate_preregistration_v3_before_fit(v1, v2, v3, invalidation_path)
            invalidation_companion = invalidation_path.with_suffix(".json.sha256.json")
            destination = official_root / "preregistered_spec_v4.json"
            kwargs = {
                "destination": destination,
                "frozen_at_utc": "2026-09-05T12:00:00Z",
                "expected_candidate_payload_sha256": payload_sha256(candidate),
                "expected_candidate_file_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                "expected_candidate_companion_file_sha256": hashlib.sha256(candidate_companion.read_bytes()).hexdigest(),
                "expected_completion_file_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
                "expected_completion_payload_sha256": completion["completion_payload_sha256"],
                "v1_path": v1, "v2_path": v2, "v3_path": v3,
                "v3_invalidation_path": invalidation_path,
                "v3_invalidation_file_sha256": hashlib.sha256(invalidation_path.read_bytes()).hexdigest(),
                "v3_invalidation_companion_file_sha256": hashlib.sha256(invalidation_companion.read_bytes()).hexdigest(),
            }
            return candidate_path, destination, invalidation_path, kwargs

        for injection, companion_exists in (("official_v4_envelope", False), ("official_v4_companion", True)):
            with self.subTest(injection=injection), tempfile.TemporaryDirectory() as directory:
                candidate_path, destination, invalidation_path, kwargs = fixture(Path(directory))
                v3_bytes = invalidation_path.read_bytes()
                v3_companion_bytes = invalidation_path.with_suffix(".json.sha256.json").read_bytes()
                with self.assertRaisesRegex(RuntimeError, "injected promotion failure"):
                    promote_v4_candidate_to_authoritative(
                        candidate_path,
                        **kwargs,
                        _failure_injection_after=injection,
                    )
                self.assertTrue(destination.exists())
                self.assertEqual(destination.with_suffix(".json.sha256.json").exists(), companion_exists)
                self.assertFalse(destination.with_suffix(".json.promotion_complete.json").exists())
                with self.assertRaises(ContractViolation):
                    verify_preregistration(destination)
                with self.assertRaises(FileExistsError):
                    promote_v4_candidate_to_authoritative(candidate_path, **kwargs)
                self.assertEqual(invalidation_path.read_bytes(), v3_bytes)
                self.assertEqual(invalidation_path.with_suffix(".json.sha256.json").read_bytes(), v3_companion_bytes)

    def test_promotion_rejects_missing_companion_duplicate_keys_and_recursive_nonfinite(self):
        """Catches self-consistent-looking review artifacts bypassing the strict promotion parser."""
        for malformed in (
            '{"version":4,"version":4,"authority_status":"REVIEW_CANDIDATE_NOT_FROZEN"}',
            '{"version":4,"authority_status":"REVIEW_CANDIDATE_NOT_FROZEN","nested":{"value":1e999}}',
        ):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                v1 = root / "v1.json"; freeze_preregistration(v1, {"version": 1})
                v2 = root / "v2.json"; freeze_preregistration(v2, build_preregistered_spec_v2(v1))
                v3 = root / "v3.json"; freeze_preregistration(v3, build_preregistered_spec_v3(v1, v2))
                attempt = root / "runs" / "v4_review" / "attempt_0001"
                attempt.mkdir(parents=True)
                candidate_path = attempt / "preregistered_spec_v4_candidate.json"
                candidate_path.write_text(malformed, encoding="utf-8")
                candidate_companion = candidate_path.with_suffix(".json.sha256.json")
                candidate_companion.write_text("{}", encoding="utf-8")
                completion = publish_completion_marker(
                    attempt,
                    {"stage": "v4_review", "official_freeze_performed": False},
                    [candidate_path, candidate_companion],
                )
                completion_path = attempt / "completion.json"
                official_root = root / "results" / "prob_head_structure_full_v1"
                invalidation_path = official_root / "preregistration_invalidations_v3.json"
                invalidate_preregistration_v3_before_fit(v1, v2, v3, invalidation_path)
                invalidation_companion = invalidation_path.with_suffix(".json.sha256.json")
                with self.assertRaisesRegex(ContractViolation, "duplicate JSON key|nonfinite JSON"):
                    promote_v4_candidate_to_authoritative(
                        candidate_path,
                        destination=official_root / "preregistered_spec_v4.json",
                        frozen_at_utc="2026-09-05T12:00:00Z",
                        expected_candidate_payload_sha256="0" * 64,
                        expected_candidate_file_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                        expected_candidate_companion_file_sha256=hashlib.sha256(candidate_companion.read_bytes()).hexdigest(),
                        expected_completion_file_sha256=hashlib.sha256(completion_path.read_bytes()).hexdigest(),
                        expected_completion_payload_sha256=completion["completion_payload_sha256"],
                        v1_path=v1, v2_path=v2, v3_path=v3,
                        v3_invalidation_path=invalidation_path,
                        v3_invalidation_file_sha256=hashlib.sha256(invalidation_path.read_bytes()).hexdigest(),
                        v3_invalidation_companion_file_sha256=hashlib.sha256(invalidation_companion.read_bytes()).hexdigest(),
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1 = root / "v1.json"; freeze_preregistration(v1, {"version": 1})
            v2 = root / "v2.json"; freeze_preregistration(v2, build_preregistered_spec_v2(v1))
            v3 = root / "v3.json"; freeze_preregistration(v3, build_preregistered_spec_v3(v1, v2))
            attempt, _ = reserve_or_resume_attempt(root / "runs", "v4_review")
            candidate_path = attempt / "preregistered_spec_v4_candidate.json"
            candidate = build_preregistered_spec_v4(v1, v2, v3)
            write_preregistration_review_candidate(candidate_path, candidate)
            companion = candidate_path.with_suffix(".json.sha256.json")
            completion = publish_completion_marker(attempt, {"stage": "v4_review", "official_freeze_performed": False}, [candidate_path, companion])
            completion_path = attempt / "completion.json"
            expected_companion_hash = hashlib.sha256(companion.read_bytes()).hexdigest()
            companion.unlink()
            official_root = root / "results" / "prob_head_structure_full_v1"
            invalidation_path = official_root / "preregistration_invalidations_v3.json"
            invalidate_preregistration_v3_before_fit(v1, v2, v3, invalidation_path)
            invalidation_companion = invalidation_path.with_suffix(".json.sha256.json")
            with self.assertRaises(ContractViolation):
                promote_v4_candidate_to_authoritative(
                    candidate_path,
                    destination=official_root / "preregistered_spec_v4.json",
                    frozen_at_utc="2026-09-05T12:00:00Z",
                    expected_candidate_payload_sha256=payload_sha256(candidate),
                    expected_candidate_file_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                    expected_candidate_companion_file_sha256=expected_companion_hash,
                    expected_completion_file_sha256=hashlib.sha256(completion_path.read_bytes()).hexdigest(),
                    expected_completion_payload_sha256=completion["completion_payload_sha256"],
                    v1_path=v1, v2_path=v2, v3_path=v3,
                    v3_invalidation_path=invalidation_path,
                    v3_invalidation_file_sha256=hashlib.sha256(invalidation_path.read_bytes()).hexdigest(),
                    v3_invalidation_companion_file_sha256=hashlib.sha256(invalidation_companion.read_bytes()).hexdigest(),
                )

    def test_review_candidate_writer_publishes_raw_payload_and_independent_hash_once(self):
        """Catches wrapping or overwriting the scratch candidate so review bytes differ from the payload."""
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "candidate.json"
            payload = {"version": 4, "authority_status": "REVIEW_CANDIDATE_NOT_FROZEN", "value": 7}
            write_preregistration_review_candidate(destination, payload)
            stored = json.loads(destination.read_text(encoding="utf-8"))
            companion = json.loads(destination.with_suffix(".json.sha256.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, payload)
            self.assertEqual(companion["payload_sha256"], payload_sha256(payload))
            self.assertEqual(companion["sha256"], hashlib.sha256(destination.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                write_preregistration_review_candidate(destination, payload)

    def test_v4_freezes_commit_and_push_scope(self):
        """Catches committing before verification or staging unrelated user work."""
        candidate = self._v4_candidate()
        self.assertEqual(
            candidate["commit_and_push"]["commit_allowed_paths"],
            [
                "experiments/prob_head_structure_full_v1/",
                "tests/prob_head_structure_full_v1/",
                "results/prob_head_structure_full_v1/ compact artifacts",
                "required requirements/lock files",
                "minimal required .gitignore/.gitattributes changes",
                "the applicable dated history record",
            ],
        )
        self.assertEqual(candidate["commit_and_push"]["commit_message"], "Run full probabilistic-head structure and A/B/C screens")
        self.assertEqual(candidate["commit_and_push"]["commit_preconditions"], ["all tests pass", "final protected-artifact hash verification passes"])
        self.assertEqual(candidate["commit_and_push"]["push_preconditions"], ["origin is the expected user repository", "authentication succeeds", "no large LFS quota problem", "no unrelated change is in the commit"])
        self.assertEqual(
            candidate["implementation_plan_v4_addendum"],
            {
                "path": "runs/prob_head_structure_full_v1/implementation_plan_v4_addendum_review.md",
                "status": "REVIEW_ONLY_NOT_AUTHORITATIVE",
                "implementation_plan_path": "results/prob_head_structure_full_v1/implementation_plan.md",
                "plan_revision": "pre-freeze review text updated to remove superseded empirical-quantile and shared router/sensor-origin instructions",
                "prior_frozen_preregistration_mutation": "forbidden",
                "conflict_rule": "only the reviewed then authoritatively frozen v4 payload governs any remaining conflict with implementation_plan.md or this review addendum",
                "covers": ["source and promotion integrity", "validation-selector provenance and dataset eligibility", "deterministic scientific metrics and P3", "exact training and numerical execution", "runtime FULL-workload projection", "closed gate and primary-dataset-manifest lineage", "controls", "routing/sensor temporal boundaries", "C-SYN geometry", "action selection", "replicate pairing"],
            },
        )
        repository = Path(__file__).resolve().parents[2]
        plan = (repository / "results/prob_head_structure_full_v1/implementation_plan.md").read_text(encoding="utf-8")
        addendum = (repository / "runs/prob_head_structure_full_v1/implementation_plan_v4_addendum_review.md").read_text(encoding="utf-8")
        self.assertNotIn("numerical inverse of the resulting empirical CDF", plan)
        self.assertIn("No scientific metric uses an empirical", plan)
        self.assertIn("sample CDF.", plan)
        self.assertIn("attempt 3", addendum)
        self.assertIn("not a freeze", addendum)

    def test_companion_recovery_never_creates_a_new_preregistration_trust_root(self):
        """Catches recovery silently blessing a missing, corrupt, or self-consistently replaced binding."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "spec.json"
            freeze_preregistration(target, {"version": 4})
            companion = target.with_suffix(target.suffix + ".sha256.json")
            companion.unlink()
            with self.assertRaisesRegex(ContractViolation, "missing companion"):
                recover_preregistration_companion(target)

            replacement_payload = {"version": 999, "self_consistent": True}
            replacement = {
                "schema_version": 1,
                "payload_hash_contract": "SHA-256 of canonical UTF-8 JSON payload",
                "payload": replacement_payload,
                "payload_sha256": payload_sha256(replacement_payload),
            }
            target.write_text(json.dumps(replacement), encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "missing companion"):
                recover_preregistration_companion(target)

            companion.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "companion mismatch"):
                recover_preregistration_companion(target)

    def test_malformed_preregistration_and_companion_errors_are_contract_violations(self):
        """Catches raw parser/type errors escaping the integrity boundary during verify or recovery."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "spec.json"
            target.write_text("not-json", encoding="utf-8")
            with self.assertRaises(ContractViolation):
                verify_preregistration(target)
            with self.assertRaises(ContractViolation):
                recover_preregistration_companion(target)

            target.unlink()
            freeze_preregistration(target, {"version": 4})
            companion = target.with_suffix(target.suffix + ".sha256.json")
            companion.write_text("[]", encoding="utf-8")
            with self.assertRaises(ContractViolation):
                verify_preregistration(target)

            target.write_text(
                '{"schema_version":1,"schema_version":1,"payload_hash_contract":"SHA-256 of canonical UTF-8 JSON payload","payload":{},"payload_sha256":"' + "0" * 64 + '"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractViolation, "duplicate JSON key"):
                verify_preregistration(target)

            target.write_text(
                '{"schema_version":1,"payload_hash_contract":"SHA-256 of canonical UTF-8 JSON payload",'
                '"payload":{"nested":[{"x":1e999}]},"payload_sha256":"' + "0" * 64 + '"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractViolation, "nonfinite"):
                verify_preregistration(target)

    def test_freeze_refuses_stale_companion_without_publishing_target(self):
        """Catches a partial freeze that creates a new payload next to an old unrelated companion."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "spec.json"
            companion = target.with_suffix(target.suffix + ".sha256.json")
            companion.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                freeze_preregistration(target, {"version": 4})
            self.assertFalse(target.exists())

    def test_v3_preserves_invalid_v2_and_corrects_method_labels_and_report_contract(self):
        """Catches authority silently moving to an incorrectly labelled A/B/C preregistration."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1 = root / "preregistration.json"
            freeze_preregistration(v1, {"version": 1})
            v2 = root / "preregistered_spec.json"
            freeze_preregistration(v2, build_preregistered_spec_v2(v1))
            invalidation = invalidate_preregistration_v2_before_fit(v1, v2, root / "preregistration_invalidations_v2.json")
            v3 = build_preregistered_spec_v3(v1, v2)
            v3_path = root / "preregistered_spec_v3.json"
            freeze_preregistration(v3_path, v3)
            with self.assertRaises(FileExistsError):
                freeze_preregistration(v3_path, v3)

        self.assertEqual(invalidation["status"], "INVALID_BEFORE_ANY_MODEL_FIT")
        self.assertEqual(v3["version"], 3)
        self.assertEqual(v3["students"]["A0"], "hard_only_student")
        self.assertEqual(v3["routing"]["B2"], "temporal_structure_conditioned_distillation")
        self.assertEqual(v3["sensors_and_actions"]["actions"]["C_A2"], "interval_widening")
        self.assertEqual(v3["likelihood_validation"]["tolerances"]["finite_fraction_min"], 0.999)
        self.assertEqual(len(v3["reporting"]["tables"]), 20)
        self.assertEqual(len(v3["reporting"]["figures"]), 8)

    def test_v2_binds_corrected_split_import_likelihood_pool_and_gate_contracts(self):
        """Catches a corrected freeze that leaves a result-sensitive v2 protocol field unspecified."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = root / "preregistration.json"
            freeze_preregistration(old_path, {"version": 1})
            payload = build_preregistered_spec_v2(old_path)

        self.assertEqual(payload["environment"]["required_import_orders"], ["torch_then_numpy", "numpy_then_torch"])
        self.assertEqual(payload["real"]["M5"]["train"], [0, 1717])
        self.assertEqual(payload["real"]["M5"]["warmup"], [1745, 1773])
        self.assertEqual(payload["likelihood_validation"]["grid"]["evaluations"], 600)
        self.assertEqual(payload["pools"]["P2"]["states"], 66)
        self.assertEqual(payload["heads"]["Tweedie_p_range"], [1.05, 1.95])
        self.assertEqual(payload["gates"]["A2"]["recovery_CI_lower_gt"], 0)
        self.assertTrue(payload["gates"]["A2"]["recovery_sign_positive"])
        self.assertEqual(payload["sensors_and_actions"]["C2_delay_unit"], "one horizon")
        self.assertEqual(payload["runtime"]["timing_smoke"]["series"], 200)
        self.assertEqual(payload["execution"]["network_retry"]["backoff_seconds"], [30, 120, 300])

    def test_invalidated_prefit_attempt_is_preserved_while_v2_is_exclusively_frozen(self):
        """Catches rewriting a bad preregistration rather than preserving an auditable before-fit correction."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = root / "preregistration.json"
            freeze_preregistration(old_path, {"version": 1})
            invalidation_path = root / "preregistration_invalidations.json"

            invalidation = invalidate_preregistration_before_fit(old_path, invalidation_path)
            v2_payload = build_preregistered_spec_v2(old_path)
            v2_path = root / "preregistered_spec.json"
            freeze_preregistration(v2_path, v2_payload)

            self.assertEqual(invalidation["status"], "INVALID_BEFORE_ANY_MODEL_FIT")
            self.assertEqual(invalidation["new_model_fits_completed"], 0)
            self.assertEqual(v2_payload["version"], 2)
            self.assertEqual(v2_payload["prior_attempt_status"], "INVALID")
            self.assertEqual(v2_payload["supersedes"]["path"], "preregistration.json")
            with self.assertRaises(FileExistsError):
                invalidate_preregistration_before_fit(old_path, invalidation_path)
            with self.assertRaises(FileExistsError):
                freeze_preregistration(v2_path, v2_payload)

    def test_payload_binds_representative_prefit_choices_that_could_otherwise_be_tuned(self):
        """Catches omitted pre-fit protocol choices, especially seeds, support, grids, and gate policy."""
        payload = build_preregistration_payload()

        self.assertEqual(payload["identity"]["experiment"], "PROB-HEAD-STRUCTURE-FULL-v1")
        self.assertEqual(payload["identity"]["branch"], "prob-head-structure-full-v1")
        self.assertEqual(payload["seeds"]["master"], 20260905)
        self.assertEqual(payload["seeds"]["bootstrap"], 2026090531)
        self.assertEqual(payload["protected_directories"][0], "results/ph_online_memory_gono_v1")
        self.assertEqual(len(payload["protected_directories"]), 14)
        self.assertEqual(payload["environment"]["python"], "3.10.20")
        self.assertTrue(payload["environment"]["kmp_duplicate_lib_ok_forbidden"])
        self.assertEqual(payload["target_support"]["integer_tolerance"], 1e-6)
        self.assertEqual(payload["synthetic"]["origins"], [436, 464, 492, 520, 548])
        self.assertEqual(payload["synthetic"]["rho_values"], [-0.8, 0.0, 0.8])
        self.assertEqual(payload["training"]["oom_retry"], {"max_retries": 1, "microbatch_multiplier": 0.5, "accumulation_multiplier": 2})
        self.assertEqual(payload["metrics"]["draws"], {"validation": 256, "evaluation": 1024})
        self.assertEqual(payload["grids"]["P3_penalty"], [0.0, 0.01, 0.1, 1.0])
        self.assertEqual(payload["retry"]["network_backoff_seconds"], [30, 120, 300])
        self.assertIn("INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT", payload["verdict_vocabulary"])

    def test_freeze_stores_complete_payload_and_companion_file_hash_without_overwrite(self):
        """Catches a mutable freeze or an invalid self-referential whole-file hash."""
        payload = {
            "experiment": "PROB-HEAD-STRUCTURE-FULL-v1",
            "seeds": {"master": 20260905},
            "split": {"horizon": 28},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "preregistration.json"

            frozen = freeze_preregistration(destination, payload)
            stored = json.loads(destination.read_text(encoding="utf-8"))
            companion = json.loads(
                destination.with_suffix(destination.suffix + ".sha256.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(stored, frozen)
            self.assertEqual(stored["payload"], payload)
            self.assertEqual(stored["payload_sha256"], payload_sha256(payload))
            self.assertEqual(verify_preregistration(destination)["payload"], payload)
            self.assertEqual(companion["relative_filename"], "preregistration.json")
            self.assertEqual(companion["algorithm"], "SHA-256")
            self.assertEqual(len(companion["sha256"]), 64)
            with self.assertRaises(FileExistsError):
                freeze_preregistration(destination, payload)

            changed = dict(payload)
            changed["split"] = {"horizon": 29}
            with self.assertRaises(FileExistsError):
                freeze_preregistration(destination, changed)


if __name__ == "__main__":
    unittest.main()
