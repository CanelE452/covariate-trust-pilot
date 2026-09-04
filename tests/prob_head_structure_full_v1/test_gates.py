from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import experiments.prob_head_structure_full_v1.data as data_module
from experiments.prob_head_structure_full_v1 import integrity
from experiments.prob_head_structure_full_v1.data import (
    REAL_SPLITS,
    RealSplit,
    audit_fixed_length_dataset,
    file_sha256,
    seal_count_primary_dataset_audit,
    select_real_datasets,
)
from experiments.prob_head_structure_full_v1.gates import (
    FrozenPrimaryDatasetManifest,
    GateResult,
    attach_lineage,
    combine_a_verdict,
    combine_b_verdict,
    combine_c_verdict,
    combine_head_verdict,
    combine_real_verdict,
    final_recommendation,
    gate_a1,
    gate_a2,
    gate_a3,
    gate_a4,
    gate_b1,
    gate_b2,
    gate_c1,
    gate_c2,
    gate_c3,
    gate_negative_control,
    gate_r1,
    gate_r2,
    gate_r3,
    gate_s1,
    gate_s2,
    gate_s3,
)


def _sealed_count_audit(dataset_id: str) -> dict[str, object]:
    if dataset_id in REAL_SPLITS:
        split = REAL_SPLITS[dataset_id]
    else:
        horizon = 12 if dataset_id == "raf" else 6
        lookback = 48
        validation_start = lookback
        warmup_start = validation_start + horizon
        first_origin = warmup_start + horizon
        split = RealSplit(
            train=(0, validation_start),
            validation=(validation_start, warmup_start),
            warmup=(warmup_start, first_origin),
            origins=tuple(first_origin + index * horizon for index in range(6)),
            horizon=horizon,
            lookback=lookback,
        )
    length = split.origins[-1] + split.horizon
    production = data_module.CANONICAL_COUNT_PRIMARY_PANEL_CONTRACTS.get(dataset_id)
    panel_shape = (
        list(production["panel_shape"])
        if production is not None
        else [2, length]
    )
    panel_binding = (
        str(production["panel_binding_sha256"])
        if production is not None
        else (dataset_id.encode("utf-8").hex() + "0" * 64)[:64]
    )
    series_binding = (
        str(production["ordered_series_id_sha256"])
        if production is not None
        else (dataset_id.encode("utf-8").hex() + "1" * 64)[:64]
    )
    record: dict[str, object] = {
        "audit_type": "COUNT_PRIMARY_DATASET_AUDIT",
        "dataset_id": dataset_id,
        "status": "PASS",
        "geometry_status": "PASS",
        "count_primary_eligible": True,
        "confirmatory_eligible": True,
        "support_audit": {
            "dataset_id": dataset_id,
            "status": "PASS",
            "target_scope": "model_train",
            "count_primary_eligible": True,
            "count_likelihood_index_exact": True,
        },
        "split_validation": data_module.validate_real_split(
            split, length=int(panel_shape[1])
        ),
        "source_records": [
            {"path": f"{dataset_id}/source", "bytes": 1, "sha256": "1" * 64}
        ],
        "source_manifest_aggregate_sha256": "2" * 64,
        "panel_shape": panel_shape,
        "panel_binding_sha256": panel_binding,
        "ordered_series_id_sha256": series_binding,
        "canonical_source_attested": True,
    }
    record["audit_sha256"] = data_module._dataset_audit_hash(record)
    return record


def _geometry_audit(dataset_id: str) -> dict[str, object]:
    return audit_fixed_length_dataset(
        dataset_id=dataset_id,
        length=1,
        lookback=48 if dataset_id != "online_retail" else 96,
        horizon=6 if dataset_id in {"auto", "carparts"} else (12 if dataset_id == "raf" else 28),
    )


PRIMARY_SELECTION = select_real_datasets(
    [
        _sealed_count_audit("m5"),
        _geometry_audit("auto"),
        _geometry_audit("carparts"),
        _geometry_audit("raf"),
        _sealed_count_audit("online_retail"),
    ]
)
PRIMARY = FrozenPrimaryDatasetManifest.from_selection_audit(PRIMARY_SELECTION)
# Implementation-plan ruling 12 freezes m5 and online_retail as the only datasets the
# data layer can attest as count-primary, so a three-dataset selection is unreachable
# through select_real_datasets. The gates policy still admits up to three datasets, so
# exercise that ordering rule at the gates layer instead of forging a data-layer audit.
PRIMARY_THREE = FrozenPrimaryDatasetManifest.from_audit_payload(
    selected_datasets=("m5", "auto", "carparts"),
    eligible_datasets_in_priority_order=("m5", "auto", "carparts"),
    audit_payload={
        "case": "gates_policy_three_dataset_ordering",
        "priority": list(data_module.REAL_DATASET_PRIORITY),
    },
)


def test_frozen_primary_dataset_manifest_policy_and_exact_mapping_order() -> None:
    assert PRIMARY.selected_datasets == ("m5", "online_retail")
    assert PRIMARY_THREE.selected_datasets == ("m5", "auto", "carparts")

    legal_three = gate_r3(
        macro_scrps_improvement=0.01,
        dataset_scrps_improvements={"m5": 0.01, "auto": 0.01, "carparts": 0.01},
        tail_sql_improvement=0.02,
        q50_deterioration=0.0,
        zero_brier_deterioration=0.0,
        nrmse_deterioration=0.0,
        primary_datasets=PRIMARY_THREE,
    )
    assert legal_three.passed
    assert legal_three.observations["primary_dataset_manifest"]["audit_sha256"] == PRIMARY_THREE.audit_sha256

    with pytest.raises(ValueError, match="exact primary dataset manifest order"):
        gate_r3(
            macro_scrps_improvement=0.01,
            dataset_scrps_improvements={"auto": 0.01, "m5": 0.01, "carparts": 0.01},
            tail_sql_improvement=0.02,
            q50_deterioration=0.0,
            zero_brier_deterioration=0.0,
            nrmse_deterioration=0.0,
            primary_datasets=PRIMARY_THREE,
        )

    invalid_specs = [
        (("online_retail", "m5"), ("m5", "online_retail")),
        (("m5", "m5"), ("m5", "online_retail")),
        (("m5", "unregistered"), ("m5", "unregistered")),
        (("m5", "carparts"), ("m5", "auto", "carparts")),
    ]
    for selected, eligible in invalid_specs:
        with pytest.raises(ValueError, match="primary dataset manifest"):
            FrozenPrimaryDatasetManifest.from_audit_payload(
                selected_datasets=selected,
                eligible_datasets_in_priority_order=eligible,
                audit_payload={"case": [*selected, *eligible]},
            )

    with pytest.raises(TypeError):
        FrozenPrimaryDatasetManifest(
            selected_datasets=("m5", "online_retail"),
            eligible_datasets_in_priority_order=("m5", "online_retail"),
            canonical_audit_payload_json="{}",
            audit_sha256="a" * 64,
        )

    invalid_geometry = deepcopy(PRIMARY_SELECTION)
    invalid_geometry["requested_non_m5"] = 999
    with pytest.raises(ValueError, match="frozen PASS artifact"):
        FrozenPrimaryDatasetManifest.from_selection_audit(invalid_geometry)
    mutated = deepcopy(PRIMARY_SELECTION)
    mutated["selected"][0]["audit_note"] = "changed after freeze"
    with pytest.raises(ValueError, match="audit payload SHA256 mismatch"):
        FrozenPrimaryDatasetManifest.from_selection_audit(
            mutated, expected_sha256=PRIMARY.audit_sha256
        )

    missing_priority_audit = deepcopy(PRIMARY_SELECTION)
    missing_priority_audit["audits"] = [
        row for row in missing_priority_audit["audits"] if row["dataset_id"] != "auto"
    ]
    with pytest.raises(ValueError, match="complete five-row sealed audit"):
        FrozenPrimaryDatasetManifest.from_selection_audit(missing_priority_audit)

    tampered_nested_audit = deepcopy(PRIMARY_SELECTION)
    tampered_nested_audit["audits"][0]["support_audit"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="sealed audit|manifest SHA256"):
        FrozenPrimaryDatasetManifest.from_selection_audit(tampered_nested_audit)

    selected_only_forgery = deepcopy(PRIMARY_SELECTION)
    selected_only_forgery["selected_dataset_ids"] = ["m5", "auto"]
    selected_only_forgery["selected"] = [
        selected_only_forgery["audits"][0],
        selected_only_forgery["audits"][1],
    ]
    with pytest.raises(ValueError, match="derived audited eligibility"):
        FrozenPrimaryDatasetManifest.from_selection_audit(selected_only_forgery)


def _complete_s3(*, passing: bool) -> list[dict[str, float | str]]:
    pairs = ("NB_vs_HSNB", "NB_vs_TWEEDIE_FULL", "HSNB_vs_TWEEDIE_FULL")
    factors = (
        "d", "rho_I_L", "rho_I_Q", "rho_M_L", "rho_M_Q",
        "d*rho_I_L", "d*rho_I_Q", "d*rho_M_L", "d*rho_M_Q",
        "rho_I_L*rho_M_L", "rho_I_L*rho_M_Q",
        "rho_I_Q*rho_M_L", "rho_I_Q*rho_M_Q",
        "d*rho_I_L*rho_M_L", "d*rho_I_L*rho_M_Q",
        "d*rho_I_Q*rho_M_L", "d*rho_I_Q*rho_M_Q",
    )
    rows = []
    for pair in pairs:
        for factor in factors:
            row = {"pair": pair, "contrast": factor, "effect": 0.0, "ci_lower": -1.0, "ci_upper": 1.0}
            rows.append(row)
    if passing:
        rows[0].update({"effect": 2.0, "ci_lower": 0.1, "ci_upper": 3.0})
    return rows


def test_synthetic_gate_boundaries_are_inclusive_and_ci_excludes_zero() -> None:
    s1 = gate_s1(
        exact_best_cell_counts={"NB": 3, "HSNB": 3, "TWEEDIE_FULL": 12},
        practical_winner_shares={"NB": 0.0, "HSNB": 0.0, "TWEEDIE_FULL": 1.0},
        total_cells=18,
    )
    assert s1.passed and s1.verdict == "HEAD_SPECIALIZATION_GO"
    assert gate_s2(cell_oracle_gain=0.02, series_origin_oracle_gain=0.03).passed

    assert gate_s3(_complete_s3(passing=True)).passed
    rows = _complete_s3(passing=False)
    rows[0].update({"effect": 3.0, "ci_lower": 0.0, "ci_upper": 4.0})
    assert not gate_s3(rows).passed
    with pytest.raises(ValueError, match="exactly"):
        gate_s3(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        gate_s3([*rows[:-1], rows[0]])


def test_real_teacher_and_complementarity_gate_literal_boundaries() -> None:
    quality = {
        "NB": {
            "relative_scrps_gap": {"d1": 0.05, "d2": 0.04},
            "zero_brier_best": True,
            "tail_sql_best": False,
        },
        "HSNB": {
            "relative_scrps_gap": {"d1": 0.02, "d2": 0.05},
            "zero_brier_best": False,
            "tail_sql_best": True,
        },
        "TWEEDIE_FULL": {
            "relative_scrps_gap": {"d1": 0.11, "d2": 0.0},
            "zero_brier_best": False,
            "tail_sql_best": False,
        },
    }
    quality["NB"]["relative_scrps_gap"] = {"m5": 0.05, "online_retail": 0.04}
    quality["HSNB"]["relative_scrps_gap"] = {"m5": 0.02, "online_retail": 0.05}
    quality["TWEEDIE_FULL"]["relative_scrps_gap"] = {"m5": 0.11, "online_retail": 0.0}
    assert gate_r1(quality, primary_datasets=PRIMARY).passed
    with pytest.raises(ValueError, match="frozen primary dataset manifest"):
        gate_r1(quality, primary_datasets=("online_retail", "m5"))

    r2 = gate_r2(
        practical_winner_shares={"NB": 0.15, "HSNB": 0.15, "TWEEDIE_FULL": 0.70},
        macro_oracle_gain=0.02,
        dataset_oracle_gains={"m5": 0.01, "online_retail": 0.01},
        pairwise_correlation_ci={"NB|HSNB": {"upper": 0.989, "degenerate_resample_present": False}, "NB|TWEEDIE_FULL": {"upper": 0.98, "degenerate_resample_present": False}, "HSNB|TWEEDIE_FULL": {"upper": 0.97, "degenerate_resample_present": False}},
        dataset_best_heads={"m5": "NB", "online_retail": "HSNB"},
        primary_datasets=PRIMARY,
    )
    assert r2.passed
    assert not gate_r2(
        practical_winner_shares={"NB": 0.15, "HSNB": 0.15, "TWEEDIE_FULL": 0.70},
        macro_oracle_gain=0.02,
        dataset_oracle_gains={"m5": 0.01, "online_retail": 0.01},
        pairwise_correlation_ci={"NB|HSNB": {"upper": 0.98, "degenerate_resample_present": True}, "NB|TWEEDIE_FULL": {"upper": 0.98, "degenerate_resample_present": False}, "HSNB|TWEEDIE_FULL": {"upper": 0.97, "degenerate_resample_present": False}},
        dataset_best_heads={"m5": "NB", "online_retail": "HSNB"},
        primary_datasets=PRIMARY,
    ).passed


def test_pool_and_a_gate_boundaries_and_strict_lower_ci() -> None:
    assert gate_r3(
        macro_scrps_improvement=0.01,
        dataset_scrps_improvements={"m5": -0.005, "online_retail": 0.03},
        tail_sql_improvement=0.02,
        q50_deterioration=0.0099,
        zero_brier_deterioration=0.0099,
        nrmse_deterioration=0.0099,
        primary_datasets=PRIMARY,
    ).passed
    assert gate_a1(
        scrps_improvement=0.01,
        tail_sql_improvement=0.02,
        primary_datasets=PRIMARY,
    ).passed
    assert gate_a2(
        best_single_loss=1.0,
        distilled_loss=0.95,
        pool_loss=0.9,
        improvement_vs_a0=0.005,
        macro_ci_lower=1e-9,
        dataset_effects={"m5": 0.01, "online_retail": 0.001},
        dataset_ci_lowers={"m5": 1e-9, "online_retail": -0.1},
        primary_datasets=PRIMARY,
    ).passed
    assert not gate_a2(
        best_single_loss=1.0,
        distilled_loss=0.95,
        pool_loss=0.9,
        improvement_vs_a0=0.005,
        macro_ci_lower=0.0,
        dataset_effects={"m5": 0.01, "online_retail": 0.001},
        dataset_ci_lowers={"m5": 1e-9, "online_retail": -0.1},
        primary_datasets=PRIMARY,
    ).passed

    assert gate_a3(
        {"zero_brier": 0.01, "q50": 0.01, "q99": 0.01, "NRMSE": 0.01, "coverage_90": 0.01, "coverage_95": 0.01},
        primary_datasets=PRIMARY,
    ).passed
    assert gate_a4(
        student_parameters=150,
        smallest_teacher_parameters=100,
        latency_by_device_batch={
            "cpu": {"1": {"student": 1.3, "single": 1.0, "pool": 2.6}, "256": {"student": 1.0, "single": 1.0, "pool": 2.0}},
            "cuda": {"1": {"student": 1.3, "single": 1.0, "pool": 2.6}, "256": {"student": 1.0, "single": 1.0, "pool": 2.0}},
        },
        cuda_peak_memory_by_batch={"1": {"student": 5.0, "pool": 10.0}, "256": {"student": 50.0, "pool": 100.0}},
        cuda_available=True,
        primary_datasets=PRIMARY,
    ).passed
    assert not gate_a4(
        student_parameters=0,
        smallest_teacher_parameters=100,
        latency_by_device_batch={
            "cpu": {"1": {"student": 1.0, "single": 1.0, "pool": 2.0}, "256": {"student": 1.0, "single": 1.0, "pool": 2.0}},
            "cuda": {"1": {"student": 1.0, "single": 1.0, "pool": 2.0}, "256": {"student": 1.0, "single": 1.0, "pool": 2.0}},
        },
        cuda_peak_memory_by_batch={"1": {"student": 5.0, "pool": 10.0}, "256": {"student": 50.0, "pool": 100.0}},
        cuda_available=True,
        primary_datasets=PRIMARY,
    ).passed

    undefined = gate_a2(
        best_single_loss=1.0,
        distilled_loss=0.9,
        pool_loss=1.0,
        improvement_vs_a0=0.5,
        macro_ci_lower=0.1,
        dataset_effects={"m5": 0.1, "online_retail": 0.1},
        dataset_ci_lowers={"m5": 0.1, "online_retail": 0.1},
        primary_datasets=PRIMARY,
    )
    assert not undefined.passed
    assert undefined.observations["recovery"] is None

    assert not gate_r3(
        macro_scrps_improvement=0.01,
        dataset_scrps_improvements={"m5": 0.01, "online_retail": 0.01},
        tail_sql_improvement=0.02,
        q50_deterioration=0.01,
        zero_brier_deterioration=0.0,
        nrmse_deterioration=0.0,
        primary_datasets=PRIMARY,
    ).passed
    with pytest.raises(ValueError, match="primary dataset manifest"):
        gate_r3(
            macro_scrps_improvement=0.1,
            dataset_scrps_improvements={"m5": 0.1},
            tail_sql_improvement=0.1,
            q50_deterioration=0.0,
            zero_brier_deterioration=0.0,
            nrmse_deterioration=0.0,
            primary_datasets=PRIMARY,
        )

    unavailable = gate_a4(
        student_parameters=100,
        smallest_teacher_parameters=100,
        latency_by_device_batch={"cpu": {"1": {"student": 1.0, "single": 1.0, "pool": 2.0}, "256": {"student": 1.0, "single": 1.0, "pool": 2.0}}},
        cuda_peak_memory_by_batch=None,
        cuda_available=False,
        primary_datasets=PRIMARY,
    )
    assert not unavailable.passed
    assert unavailable.observations["cuda_status"] == "UNAVAILABLE_EXPLICIT_GATE_FAIL"


def test_b_and_c_literal_gates() -> None:
    assert gate_b1(
        regret_spearman_by_dataset={"m5": 0.20, "online_retail": 0.21},
        undefined_reasons_by_dataset={},
        extended_minus_baseline=0.08,
        real_increment=0.08,
        shuffled_increment=0.02,
        cross_dataset_effects={"m5": 0.01, "online_retail": 0.02},
        primary_datasets=PRIMARY,
    ).passed
    assert gate_b2(
        macro_scrps_improvement=0.005,
        dataset_scrps_improvements={"m5": 0.01, "online_retail": 0.001},
        macro_ci_lower=1e-12,
        improvement_over_b1=0.002,
        q99_deterioration=0.009,
        zero_brier_deterioration=0.009,
        worst_origin_improvement=-0.005,
        primary_datasets=PRIMARY,
    ).passed
    assert not gate_b2(
        macro_scrps_improvement=0.005,
        dataset_scrps_improvements={"m5": 0.001, "online_retail": 0.001},
        macro_ci_lower=1e-12,
        improvement_over_b1=0.002,
        q99_deterioration=0.01,
        zero_brier_deterioration=0.0,
        worst_origin_improvement=-0.005,
        primary_datasets=PRIMARY,
    ).passed

    dataset_rows = {
        "m5": {"auroc": 0.70, "auprc": 0.35, "c2_minus_c0_auprc": 0.05, "c2_minus_c3_auprc": 0.02, "c2_brier": 0.10, "c0_brier": 0.10},
        "online_retail": {"auroc": 0.80, "auprc": 0.45, "c2_minus_c0_auprc": 0.06, "c2_minus_c3_auprc": 0.03, "c2_brier": 0.09, "c0_brier": 0.10},
    }
    assert gate_c1(
        dataset_rows,
        undefined_reasons_by_dataset={},
        primary_datasets=PRIMARY,
    ).passed
    assert gate_c2(auprc=0.70, false_alarm_rate=0.10, median_delay_horizons=1.0, no_change_false_positive=0.10, component_separation=True).passed
    assert gate_c3(
        worst_decile_scrps_improvement=0.10,
        coverage_error_reductions={"90": 0.15, "95": 0.15},
        mean_scrps_deterioration=0.0049,
        selective_coverage=0.80,
        false_alarm_dataset_deteriorations={"m5": 0.0049, "online_retail": 0.004},
        undefined_reasons_by_dataset={},
        primary_datasets=PRIMARY,
    ).passed
    assert not gate_c3(
        worst_decile_scrps_improvement=0.10,
        coverage_error_reductions={"90": 0.15, "95": 0.15},
        mean_scrps_deterioration=0.005,
        selective_coverage=0.80,
        false_alarm_dataset_deteriorations={"m5": 0.0, "online_retail": 0.0},
        undefined_reasons_by_dataset={},
        primary_datasets=PRIMARY,
    ).passed


def test_undefined_real_metrics_are_retained_as_explicit_gate_failures() -> None:
    b1 = gate_b1(
        regret_spearman_by_dataset={"m5": 0.20, "online_retail": None},
        undefined_reasons_by_dataset={"online_retail": "INSUFFICIENT_VARIATION"},
        extended_minus_baseline=0.08,
        real_increment=0.08,
        shuffled_increment=0.0,
        cross_dataset_effects={"m5": 0.01, "online_retail": 0.02},
        primary_datasets=PRIMARY,
    )
    assert not b1.passed
    assert b1.observations["regret_spearman_by_dataset"]["online_retail"] is None
    assert b1.observations["undefined_reasons_by_dataset"] == {
        "online_retail": "INSUFFICIENT_VARIATION"
    }

    c1_rows = {
        "m5": {"auroc": 0.70, "auprc": 0.35, "c2_minus_c0_auprc": 0.05, "c2_minus_c3_auprc": 0.02, "c2_brier": 0.10, "c0_brier": 0.10},
        "online_retail": {"auroc": None, "auprc": None, "c2_minus_c0_auprc": None, "c2_minus_c3_auprc": None, "c2_brier": None, "c0_brier": None},
    }
    c1 = gate_c1(
        c1_rows,
        undefined_reasons_by_dataset={"online_retail": "SINGLE_CLASS"},
        primary_datasets=PRIMARY,
    )
    assert not c1.passed
    assert c1.observations["datasets"]["online_retail"]["undefined_reason"] == "SINGLE_CLASS"

    c3 = gate_c3(
        worst_decile_scrps_improvement=0.10,
        coverage_error_reductions={"90": 0.15, "95": 0.15},
        mean_scrps_deterioration=0.0,
        selective_coverage=0.80,
        false_alarm_dataset_deteriorations={"m5": 0.0, "online_retail": None},
        undefined_reasons_by_dataset={"online_retail": "EMPTY_FAILURE_SET"},
        primary_datasets=PRIMARY,
    )
    assert not c3.passed
    assert c3.observations["false_alarm_dataset_deteriorations"]["online_retail"] is None


@pytest.mark.parametrize(
    ("gate_name", "reason"),
    [("B1", "WRONG"), ("C1", "INSUFFICIENT_VARIATION"), ("C3", "SINGLE_CLASS")],
)
def test_undefined_metric_reason_contract_rejects_missing_or_wrong_tokens(
    gate_name: str, reason: str
) -> None:
    if gate_name == "B1":
        call = lambda reasons: gate_b1(
            regret_spearman_by_dataset={"m5": 0.2, "online_retail": None},
            undefined_reasons_by_dataset=reasons,
            extended_minus_baseline=0.08,
            real_increment=0.08,
            shuffled_increment=0.0,
            cross_dataset_effects={"m5": 0.1, "online_retail": 0.1},
            primary_datasets=PRIMARY,
        )
    elif gate_name == "C1":
        rows = {
            dataset: {"auroc": (None if dataset == "online_retail" else 0.8), "auprc": 0.4, "c2_minus_c0_auprc": 0.1, "c2_minus_c3_auprc": 0.1, "c2_brier": 0.1, "c0_brier": 0.1}
            for dataset in PRIMARY.selected_datasets
        }
        call = lambda reasons: gate_c1(
            rows,
            undefined_reasons_by_dataset=reasons,
            primary_datasets=PRIMARY,
        )
    else:
        call = lambda reasons: gate_c3(
            worst_decile_scrps_improvement=0.1,
            coverage_error_reductions={"90": 0.15, "95": 0.15},
            mean_scrps_deterioration=0.0,
            selective_coverage=0.8,
            false_alarm_dataset_deteriorations={"m5": 0.0, "online_retail": None},
            undefined_reasons_by_dataset=reasons,
            primary_datasets=PRIMARY,
        )
    with pytest.raises(ValueError, match="undefined reason"):
        call({})
    with pytest.raises(ValueError, match="undefined reason"):
        call({"online_retail": reason})


def test_undefined_metric_contract_rejects_nan_instead_of_mapping_it_to_failure() -> None:
    with pytest.raises(ValueError, match="finite"):
        gate_b1(
            regret_spearman_by_dataset={"m5": 0.2, "online_retail": float("nan")},
            undefined_reasons_by_dataset={},
            extended_minus_baseline=0.08,
            real_increment=0.08,
            shuffled_increment=0.0,
            cross_dataset_effects={"m5": 0.1, "online_retail": 0.1},
            primary_datasets=PRIMARY,
        )


def test_control_recovery_at_half_blocks_identification() -> None:
    result = gate_negative_control(
        branch="C",
        primary_datasets=PRIMARY,
        reference_effects={"target1_auprc_gain": 0.10, "synthetic_change_signal": 0.20},
        control_effects={"time_shuffle": 0.05, "teacher_name_permutation": 0.10, "scale_only": 0.0, "random_score": 0.0, "no_change": 0.0},
        invariance_differences={"teacher_name_permutation": 0.0},
    )
    assert result.gate == "CONTROL_C"
    assert not result.passed
    assert result.failure_label == "SIGNAL_IDENTIFICATION_FAILURE"
    assert result.criteria["all_controls_below_half"] is False

    invariant_only = gate_negative_control(
        branch="C",
        primary_datasets=PRIMARY,
        reference_effects={"target1_auprc_gain": 0.10, "synthetic_change_signal": 0.50},
        control_effects={"time_shuffle": 0.0, "teacher_name_permutation": 0.10, "scale_only": 0.0, "random_score": 0.0, "no_change": 0.20},
        invariance_differences={"teacher_name_permutation": 0.0},
    )
    assert invariant_only.passed
    with pytest.raises(ValueError, match="fixed registry"):
        gate_negative_control(branch="C", primary_datasets=PRIMARY, reference_effects={"target1_auprc_gain": 0.1, "synthetic_change_signal": 0.2}, control_effects={"renamed": 0.0}, invariance_differences={})

    nonpositive_reference = gate_negative_control(
        branch="C",
        primary_datasets=PRIMARY,
        reference_effects={"target1_auprc_gain": 0.10, "synthetic_change_signal": 0.0},
        control_effects={"time_shuffle": 0.0, "teacher_name_permutation": 0.0, "scale_only": 0.0, "random_score": 0.0, "no_change": 0.0},
        invariance_differences={"teacher_name_permutation": 0.0},
    )
    assert not nonpositive_reference.passed
    assert nonpositive_reference.observations["control_references"]["no_change"]["name"] == "synthetic_change_signal"


def test_lineage_is_serializable_and_downstream_cannot_flip_upstream() -> None:
    upstream = gate_s2(cell_oracle_gain=0.0, series_origin_oracle_gain=0.0)
    downstream = gate_a1(
        scrps_improvement=0.5,
        tail_sql_improvement=0.5,
        primary_datasets=PRIMARY,
    )
    lined = attach_lineage(downstream, branch="A_DISTILLATION", upstream=[upstream])
    payload = lined.as_dict()
    assert payload["confirmatory_eligible"] is False
    assert payload["scientific_role"] == "DIAGNOSTIC_CONTINUATION_AFTER_S2"
    assert upstream.passed is False
    payload["criteria"]["scrps_improvement_at_least_1pct"] = False
    assert lined.criteria["scrps_improvement_at_least_1pct"] is True
    json.dumps(lined.as_dict(), allow_nan=False)

    later = attach_lineage(
        gate_a2(
            best_single_loss=1.0,
            distilled_loss=0.95,
            pool_loss=0.9,
            improvement_vs_a0=0.005,
            macro_ci_lower=0.01,
            dataset_effects={"m5": 0.1, "online_retail": 0.1},
            dataset_ci_lowers={"m5": 0.1, "online_retail": 0.1},
            primary_datasets=PRIMARY,
        ),
        branch="A_DISTILLATION",
        upstream=[lined],
    )
    assert later.upstream_gate_status["S2"] == "FAIL"
    assert later.scientific_role == "DIAGNOSTIC_CONTINUATION_AFTER_S2"

    reattached = attach_lineage(
        lined,
        branch="A_DISTILLATION",
        upstream=[gate_s3(_complete_s3(passing=True))],
    )
    assert reattached.upstream_gate_status["S2"] == "FAIL"
    assert reattached.upstream_gate_status["S3"] == "PASS"
    assert reattached.scientific_role == "DIAGNOSTIC_CONTINUATION_AFTER_S2"


def test_combined_method_verdicts_require_all_named_gates() -> None:
    real_manifest_gates = {
        "R1", "R2", "R3", "A1", "A2", "A3", "A4", "B1", "B2", "C1", "C3"
    }

    def passed(name: str) -> GateResult:
        observations = (
            {"primary_dataset_manifest": PRIMARY.as_dict()}
            if name in real_manifest_gates
            else {}
        )
        return GateResult.pass_result(name, f"{name}_GO", observations, {})

    def failed(name: str) -> GateResult:
        observations = (
            {"primary_dataset_manifest": PRIMARY.as_dict()}
            if name in real_manifest_gates
            else {}
        )
        return GateResult.fail_result(name, f"{name}_NO_GO", observations, {})

    def raw_passed(name: str) -> GateResult:
        return GateResult.pass_result(name, f"{name}_GO", {}, {})

    control_pass = gate_negative_control(branch="A", primary_datasets=PRIMARY, reference_effects={"branch_real_effect": 0.10}, control_effects={"teacher_identity_shuffle": 0.049, "teacher_quantile_shuffle": 0.0, "single_teacher": 0.0}, invariance_differences={})
    control_fail = gate_negative_control(branch="A", primary_datasets=PRIMARY, reference_effects={"branch_real_effect": 0.10}, control_effects={"teacher_identity_shuffle": 0.05, "teacher_quantile_shuffle": 0.0, "single_teacher": 0.0}, invariance_differences={})
    control_b = gate_negative_control(branch="B", primary_datasets=PRIMARY, reference_effects={"branch_real_effect": 0.10}, control_effects={"regret_label_shuffle": 0.0, "temporal_feature_row_shuffle": 0.0, "remove_missing_indicators": 0.0}, invariance_differences={})
    control_b_fail = gate_negative_control(branch="B", primary_datasets=PRIMARY, reference_effects={"branch_real_effect": 0.10}, control_effects={"regret_label_shuffle": 0.05, "temporal_feature_row_shuffle": 0.0, "remove_missing_indicators": 0.0}, invariance_differences={})
    control_c = gate_negative_control(branch="C", primary_datasets=PRIMARY, reference_effects={"target1_auprc_gain": 0.10, "synthetic_change_signal": 0.20}, control_effects={"time_shuffle": 0.0, "teacher_name_permutation": 0.1, "scale_only": 0.0, "random_score": 0.0, "no_change": 0.0}, invariance_differences={"teacher_name_permutation": 0.0})
    assert combine_head_verdict({name: passed(name) for name in ["DGP_BALANCE", "S1", "S2", "S3"]}) == "HEAD_SPECIALIZATION_GO"
    assert combine_head_verdict({"DGP_BALANCE": passed("DGP_BALANCE"), "S1": passed("S1"), "S2": failed("S2"), "S3": passed("S3")}) == "HEAD_SPECIALIZATION_NO_GO"
    assert combine_head_verdict({"DGP_BALANCE": failed("DGP_BALANCE"), "S1": passed("S1"), "S2": passed("S2"), "S3": passed("S3")}) == "HEAD_SPECIALIZATION_NO_GO"
    assert combine_real_verdict({name: passed(name) for name in ["R1", "R2", "R3"]}, tweedie_valid=True) == "REAL_DISTRIBUTION_POOL_GO"
    assert combine_real_verdict({"R1": failed("R1"), "R2": passed("R2"), "R3": passed("R3")}, tweedie_valid=True) == "REAL_DISTRIBUTION_POOL_NO_GO"
    ineligible_r3 = attach_lineage(passed("R3"), branch="R3", upstream=[failed("R2")])
    assert combine_real_verdict({"R1": passed("R1"), "R2": passed("R2"), "R3": ineligible_r3}, tweedie_valid=True) == "REAL_DISTRIBUTION_POOL_NO_GO"
    assert combine_real_verdict({name: passed(name) for name in ["R1", "R2", "R3"]}, tweedie_valid=False) == "REAL_DISTRIBUTION_POOL_NO_GO"
    mismatched_manifest = {
        "R1": GateResult.pass_result("R1", "R1_GO", {"primary_dataset_manifest": PRIMARY.as_dict()}, {}),
        "R2": GateResult.pass_result("R2", "R2_GO", {"primary_dataset_manifest": PRIMARY_THREE.as_dict()}, {}),
        "R3": GateResult.pass_result("R3", "R3_GO", {"primary_dataset_manifest": PRIMARY.as_dict()}, {}),
    }
    with pytest.raises(ValueError, match="identical frozen primary dataset manifest"):
        combine_real_verdict(mismatched_manifest, tweedie_valid=True)
    assert combine_a_verdict({name: passed(name) for name in ["R1", "R2", "R3", "A1", "A2", "A3", "A4"]}, tweedie_valid=True, identification_control=control_pass) == "DISTRIBUTION_SPACE_DISTILLATION_GO"
    for unbound_gate in ("A1", "A3", "A4"):
        ledger = {name: passed(name) for name in ["R1", "R2", "R3", "A1", "A2", "A3", "A4"]}
        ledger[unbound_gate] = raw_passed(unbound_gate)
        assert combine_a_verdict(
            ledger,
            tweedie_valid=True,
            identification_control=control_pass,
        ) == "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"
    assert combine_a_verdict({**{name: passed(name) for name in ["R1", "R2", "R3", "A1", "A2", "A3", "A4"]}, "R2": failed("R2")}, tweedie_valid=True, identification_control=control_pass) == "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"
    assert combine_a_verdict({**{name: passed(name) for name in ["R1", "R2", "R3", "A1", "A2", "A3", "A4"]}, "R1": failed("R1")}, tweedie_valid=True, identification_control=control_pass) == "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"
    assert combine_b_verdict({name: passed(name) for name in ["R1", "R2", "B1", "B2"]}, tweedie_valid=True, identification_control=control_b) == "STRUCTURE_CONDITIONED_ROUTING_GO"
    assert combine_b_verdict({**{name: passed(name) for name in ["R1", "R2", "B1", "B2"]}, "R1": failed("R1")}, tweedie_valid=True, identification_control=control_b) == "STRUCTURE_CONDITIONED_ROUTING_NO_GO"
    assert combine_b_verdict({name: passed(name) for name in ["R1", "R2", "B1", "B2"]}, tweedie_valid=True, identification_control=control_b_fail) == "STRUCTURE_CONDITIONED_ROUTING_NO_GO"
    assert combine_c_verdict({name: passed(name) for name in ["R1", "C1", "C2", "C3"]}, tweedie_valid=True, identification_control=control_c) == "DISAGREEMENT_SENSOR_GO"
    assert combine_c_verdict({name: passed(name) for name in ["R1", "C1", "C2", "C3"]}, tweedie_valid=False, identification_control=control_c) == "DISAGREEMENT_SENSOR_NO_GO"

    raw_control_a = GateResult.pass_result(
        "CONTROL_A", "SIGNAL_IDENTIFIED_GO", {"branch": "A"}, {}
    )
    assert combine_a_verdict(
        {name: passed(name) for name in ["R1", "R2", "R3", "A1", "A2", "A3", "A4"]},
        tweedie_valid=True,
        identification_control=raw_control_a,
    ) == "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"
    mismatched_control_a = gate_negative_control(
        branch="A",
        primary_datasets=PRIMARY_THREE,
        reference_effects={"branch_real_effect": 0.10},
        control_effects={"teacher_identity_shuffle": 0.0, "teacher_quantile_shuffle": 0.0, "single_teacher": 0.0},
        invariance_differences={},
    )
    with pytest.raises(ValueError, match="identical frozen primary dataset manifest"):
        combine_a_verdict(
            {name: passed(name) for name in ["R1", "R2", "R3", "A1", "A2", "A3", "A4"]},
            tweedie_valid=True,
            identification_control=mismatched_control_a,
        )

    assert combine_real_verdict(
        {name: raw_passed(name) for name in ["R1", "R2", "R3"]},
        tweedie_valid=True,
    ) == "REAL_DISTRIBUTION_POOL_NO_GO"
    assert combine_a_verdict(
        {name: raw_passed(name) for name in ["R1", "R2", "R3", "A1", "A2", "A3", "A4"]},
        tweedie_valid=True,
        identification_control=control_pass,
    ) == "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"
    assert combine_b_verdict(
        {name: raw_passed(name) for name in ["R1", "R2", "B1", "B2"]},
        tweedie_valid=True,
        identification_control=control_b,
    ) == "STRUCTURE_CONDITIONED_ROUTING_NO_GO"
    assert combine_c_verdict(
        {name: raw_passed(name) for name in ["R1", "C1", "C2", "C3"]},
        tweedie_valid=True,
        identification_control=control_c,
    ) == "DISAGREEMENT_SENSOR_NO_GO"

    ineligible_r2 = attach_lineage(passed("R2"), branch="R2", upstream=[failed("R1")])
    a_ledger = {name: passed(name) for name in ["R1", "R2", "R3", "A1", "A2", "A3", "A4"]}
    a_ledger["R2"] = ineligible_r2
    assert combine_a_verdict(a_ledger, tweedie_valid=True, identification_control=control_pass) == "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"integrity_blocked": True, "a_go": True, "a_confirmatory": True}, "INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT"),
        ({"a_go": True, "a_confirmatory": True, "c_go": True, "c_confirmatory": True}, "RECOMMEND_A_DISTRIBUTION_DISTILLATION"),
        ({"c_go": True, "c_confirmatory": True, "b_go": True, "b_confirmatory": True}, "RECOMMEND_C_DISAGREEMENT_SENSOR"),
        ({"b_go": True, "b_confirmatory": True}, "RECOMMEND_B_STRUCTURE_CONDITIONED_ROUTING"),
        ({"synthetic_temporal_effect": True}, "RECOMMEND_CHARACTERIZATION_ONLY"),
        ({"real_head_specialization": True}, "RECOMMEND_CHARACTERIZATION_ONLY"),
        ({}, "ALL_NEW_METHOD_BRANCHES_NO_GO"),
    ],
)
def test_final_recommendation_has_frozen_priority(kwargs: dict[str, bool], expected: str) -> None:
    assert final_recommendation(**kwargs) == expected
