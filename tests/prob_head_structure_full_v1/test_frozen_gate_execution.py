"""Regression tests for the frozen-gate re-execution contract (section 31, T25-T43).

Each test pins one way the earlier run drifted from the preregistration: a verdict
computed inside a stage, a panel that quietly included ineligible cells, a macro gate
built from a single dataset, or an unevaluated gate reported as a decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.prob_head_structure_full_v1 import data, gate_records, gates
from experiments.prob_head_structure_full_v1 import stages
from experiments.prob_head_structure_full_v1.bootstrap import factorial_temporal_contrasts

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPOSITORY_ROOT / "results/prob_head_structure_full_v1"
RUNS = REPOSITORY_ROOT / "runs/prob_head_structure_full_v1"


def _sealed(slug: str) -> dict:
    stage_root = RUNS / slug
    if not stage_root.is_dir():
        return {}
    sealed = [
        attempt
        for attempt in sorted(stage_root.glob("attempt_*"))
        if (attempt / "completion.json").exists()
    ]
    if not sealed:
        return {}
    return json.loads((sealed[-1] / "stage_payload.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------- T25 ---
def test_t25_stage_verdicts_are_the_reducer_objects() -> None:
    """Every sealed gate verdict round-trips through the frozen GateResult."""
    for slug in (
        "stage_s2_specialization_oracle_structure_analysis",
        "stage_r2_real_complementarity",
        "cdf_pool",
        "stage_a_student_distillation",
    ):
        payload = _sealed(slug)
        for name, record in dict(payload.get(gate_records.PAYLOAD_KEY, {})).items():
            restored = gate_records.deserialize(record)
            assert restored.gate == name
            assert restored.status == record["status"]
            assert bool(payload["gates"][name]) == restored.passed


def test_t25b_stages_do_not_recompute_a_threshold() -> None:
    """No stage re-derives a scientific PASS from a raw threshold comparison."""
    source = (
        REPOSITORY_ROOT / "experiments/prob_head_structure_full_v1/stages.py"
    ).read_text(encoding="utf-8")
    for banned in (
        "s1_pass =",
        "s2_pass =",
        "s3_pass =",
        "r1_pass =",
        "r2_pass =",
        "r3_pass =",
        "a1_pass =",
        "a2_pass =",
        "b1_pass =",
        "b2_pass =",
    ):
        assert banned not in source, f"stages.py still computes {banned!r} itself"


# --------------------------------------------------------------------- T26 ---
def test_t26_confirmatory_panel_excludes_unbalanced_and_incomplete_cells() -> None:
    payload = _sealed("stage_s2_specialization_oracle_structure_analysis")
    if not payload:
        pytest.skip("no sealed S2 attempt")
    panels = payload["panels"]
    unbalanced = set(payload["unbalanced_cells"])
    incomplete = {row["cell_id"] for row in payload["excluded_cells"]}
    confirmatory = set(panels["synthetic_confirmatory_panel"])
    assert not confirmatory & unbalanced
    assert not confirmatory & incomplete
    assert set(panels["synthetic_diagnostic_panel"]) >= confirmatory
    # The frozen contract keeps every cell; only eligibility is withdrawn.
    assert len(panels["synthetic_diagnostic_panel"]) == 18


# --------------------------------------------------------------------- T27 ---
def test_t27_s3_requires_the_exact_registered_contrast_set() -> None:
    rows = [
        {"pair": pair, "contrast": contrast, "effect": 0.0, "ci_lower": -1.0, "ci_upper": 1.0}
        for pair in ("NB_vs_HSNB", "NB_vs_TWEEDIE_FULL", "HSNB_vs_TWEEDIE_FULL")
        for contrast in sorted(gates._S3_FACTORS)
    ]
    assert len(rows) == 51
    assert gates.gate_s3(rows).gate == "S3"
    with pytest.raises(ValueError, match="unique 3 pairs"):
        gates.gate_s3(rows[:-1])


# --------------------------------------------------------------------- T28 ---
def test_t28_s3_cannot_pass_without_a_confidence_interval_excluding_zero() -> None:
    rows = [
        {"pair": pair, "contrast": contrast, "effect": 9.0, "ci_lower": -1.0, "ci_upper": 20.0}
        for pair in ("NB_vs_HSNB", "NB_vs_TWEEDIE_FULL", "HSNB_vs_TWEEDIE_FULL")
        for contrast in sorted(gates._S3_FACTORS)
    ]
    assert gates.gate_s3(rows).passed is False
    rows[0] = {**rows[0], "ci_lower": 1.0}
    assert gates.gate_s3(rows).passed is True


def test_t28b_this_run_could_not_form_the_s3_design() -> None:
    """The blocked NB cell removes one rho cell, so no registered design exists."""
    payload = _sealed("stage_s2_specialization_oracle_structure_analysis")
    if not payload:
        pytest.skip("no sealed S2 attempt")
    assert payload["S3_status"] == "NOT_EVALUATED"
    for record in payload["S3_refusals"].values():
        assert record["contrasts"] == 0
        assert record["refusal"]


# --------------------------------------------------------------------- T29 ---
def test_t29_single_dataset_cannot_fabricate_a_macro_manifest() -> None:
    with pytest.raises(ValueError, match="at least two datasets"):
        gates.FrozenPrimaryDatasetManifest.from_audit_payload(
            selected_datasets=["m5"],
            eligible_datasets_in_priority_order=["m5"],
            audit_payload={"any": "payload"},
        )


# --------------------------------------------------------------------- T30 ---
def test_t30_r3_is_not_decided_from_scrps_alone(frozen_manifest) -> None:
    kwargs = dict(
        macro_scrps_improvement=0.50,
        dataset_scrps_improvements={"m5": 0.5, "online_retail": 0.5},
        tail_sql_improvement=0.0,
        q50_deterioration=0.0,
        zero_brier_deterioration=0.0,
        nrmse_deterioration=0.0,
        primary_datasets=frozen_manifest,
    )
    result = gates.gate_r3(**kwargs)
    assert result.passed is False
    assert result.criteria["tail_sql_improvement_at_least_2pct"] is False


# --------------------------------------------------------------------- T31 ---
def test_t31_nonpositive_recovery_denominator_yields_null(frozen_manifest) -> None:
    result = gates.gate_a2(
        best_single_loss=1.0,
        distilled_loss=1.1,
        pool_loss=1.3,  # the pool is worse, so the denominator is negative
        improvement_vs_a0=0.0,
        macro_ci_lower=-0.1,
        dataset_effects={"m5": -0.1, "online_retail": -0.1},
        dataset_ci_lowers={"m5": -0.2, "online_retail": -0.2},
        primary_datasets=frozen_manifest,
    )
    assert result.observations["recovery"] is None
    assert result.criteria["teacher_pool_has_positive_headroom"] is False
    assert result.criteria["recovery_at_least_50pct"] is False
    assert result.passed is False


# --------------------------------------------------------------------- T32 ---
def test_t32_router_weights_join_on_explicit_keys() -> None:
    keys = pd.DataFrame(
        {
            "dataset_id": ["m5"] * 4,
            "series_id": ["a", "b", "a", "b"],
            "origin": [10, 10, 20, 20],
        }
    )
    router = keys.assign(w_NB=[0.1, 0.2, 0.3, 0.4])
    stages._assert_exact_join(router[list(stages.B_KEY_COLUMNS)], keys, label="T32")
    with pytest.raises(Exception):
        stages._assert_exact_join(
            router[list(stages.B_KEY_COLUMNS)].iloc[:3], keys, label="T32"
        )


# --------------------------------------------------------------------- T33 ---
def test_t33_row_order_does_not_change_an_exact_key_join() -> None:
    keys = pd.DataFrame(
        {
            "dataset_id": ["m5"] * 4,
            "series_id": ["a", "b", "a", "b"],
            "origin": [10, 10, 20, 20],
        }
    )
    router = keys.assign(w_NB=[0.1, 0.2, 0.3, 0.4])
    forward = keys.merge(router, on=list(stages.B_KEY_COLUMNS), how="left")
    shuffled = router.sample(frac=1.0, random_state=7).reset_index(drop=True)
    backward = keys.merge(shuffled, on=list(stages.B_KEY_COLUMNS), how="left")
    pd.testing.assert_frame_equal(forward, backward)


# --------------------------------------------------------------------- T34 ---
def test_t34_features_ignore_observations_at_and_after_the_origin() -> None:
    from experiments.prob_head_structure_full_v1.temporal_features import (
        temporal_features_for_series,
    )

    values = np.arange(400, dtype=np.float64) % 7
    origin = 200
    before = temporal_features_for_series(
        values, origin=origin, available_from=0, train_end=origin, dataset_id="m5"
    )
    tampered = values.copy()
    tampered[origin:] = 999.0
    after = temporal_features_for_series(
        tampered, origin=origin, available_from=0, train_end=origin, dataset_id="m5"
    )
    assert before == after


# --------------------------------------------------------------------- T35 ---
def test_t35_stage_b_cache_is_bound_to_its_generation() -> None:
    source = (
        REPOSITORY_ROOT / "experiments/prob_head_structure_full_v1/stages.py"
    ).read_text(encoding="utf-8")
    assert "_cache_v2" in source
    assert "stage_b_cache.v2" in source
    for binding in (
        "preregistration_payload_sha256",
        "sample_manifest_sha256",
        "teacher_checkpoint_sha256",
        "temporal_features_sha256",
    ):
        assert binding in source
    # The superseded generation must never be read by the bound loader.
    assert '"_cache" / "stage_b_inner_panel.npz"' not in source


# --------------------------------------------------------------------- T36 ---
def test_t36_csyn_produces_detection_metrics_not_only_a_status() -> None:
    payload = _sealed("stage_c_syn_known_change_experiment")
    if not payload:
        pytest.skip("no sealed C-SYN attempt")
    assert payload["rows"] > 0
    assert "detection" in payload and "auprc" in payload["detection"]
    assert "median_delay_horizons" in payload
    assert "C2" in payload[gate_records.PAYLOAD_KEY]


# --------------------------------------------------------------------- T37 ---
def test_t37_geometry_blocked_dataset_cannot_count_as_a_c1_pass() -> None:
    from experiments.prob_head_structure_full_v1.sensor import select_inner_pair_origins

    with pytest.raises(Exception):
        select_inner_pair_origins(
            lookback=96, horizon=28, model_train_end=data.REAL_SPLITS["online_retail"].train[1]
        )
    payload = _sealed("stage_c_failure_sensor")
    if payload:
        assert payload["C1_status"] == "NOT_EVALUATED"
        assert payload["confirmatory_eligible"] is False


# --------------------------------------------------------------------- T38 ---
def test_t38_unevaluated_gate_is_never_silently_a_verdict() -> None:
    payload = _sealed("final_gate_calculation")
    if not payload:
        pytest.skip("no sealed final gate attempt")
    for label, verdict in payload["verdicts"].items():
        if label in payload.get("not_evaluated", {}):
            assert verdict.startswith("NOT_EVALUATED")
            assert not verdict.endswith("_GO")


# --------------------------------------------------------------------- T39 ---
def test_t39_branch_verdicts_come_from_combine_functions() -> None:
    source = (
        REPOSITORY_ROOT / "experiments/prob_head_structure_full_v1/stages.py"
    ).read_text(encoding="utf-8")
    for name in (
        "combine_head_verdict",
        "combine_real_verdict",
        "combine_a_verdict",
        "combine_b_verdict",
        "combine_c_verdict",
    ):
        assert f"gate_module.{name}" in source


# --------------------------------------------------------------------- T40 ---
def test_t40_recommendation_comes_from_the_frozen_truth_table() -> None:
    source = (
        REPOSITORY_ROOT / "experiments/prob_head_structure_full_v1/stages.py"
    ).read_text(encoding="utf-8")
    assert "gate_module.final_recommendation(" in source
    # S1 must never promote the run to characterization.
    assert 'ledger.status("S1") == "PASS"' not in source
    assert (
        gates.final_recommendation(synthetic_temporal_effect=True)
        == "RECOMMEND_CHARACTERIZATION_ONLY"
    )
    assert (
        gates.final_recommendation(real_head_specialization=True)
        == "RECOMMEND_CHARACTERIZATION_ONLY"
    )
    assert gates.final_recommendation() == "ALL_NEW_METHOD_BRANCHES_NO_GO"


# --------------------------------------------------------------------- T41 ---
def test_t41_report_history_preserves_the_superseded_bytes() -> None:
    attempt = RESULTS / "report_history/attempt_0001"
    if not attempt.is_dir():
        pytest.skip("no preserved report history")
    supersession = json.loads((attempt / "SUPERSESSION.json").read_text(encoding="utf-8"))
    assert supersession["reason"] == "PROVISIONAL_REPORT_BEFORE_FROZEN_GATE_REEXECUTION"
    manifest = json.loads((attempt / "MANIFEST.json").read_text(encoding="utf-8"))
    import hashlib

    for row in manifest["files"]:
        digest = hashlib.sha256((attempt / row["name"]).read_bytes()).hexdigest()
        assert digest == row["sha256"]


# --------------------------------------------------------------------- T42 ---
def test_t42_every_selected_dataset_is_audited_and_bound() -> None:
    payload = _sealed("real_count_dataset_audit_download")
    if not payload:
        pytest.skip("no sealed real audit")
    assert [row["dataset_id"] for row in payload["audits"]] == list(
        data.REAL_DATASET_PRIORITY
    )
    assert len(payload["selected_dataset_ids"]) >= 2
    assert payload["selected_dataset_ids"][0] == "m5"


# --------------------------------------------------------------------- T43 ---
def test_t43_protected_manifest_v2_is_unchanged() -> None:
    import hashlib

    manifest = RESULTS / "audit/protected_manifest_before_v2.json"
    sidecar = RESULTS / "audit/protected_manifest_before_v2.json.sha256.json"
    if not manifest.exists() or not sidecar.exists():
        pytest.skip("no protected manifest v2")
    expected = json.loads(sidecar.read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest in json.dumps(expected)


@pytest.fixture
def frozen_manifest():
    """The smallest manifest the frozen policy accepts: M5 plus one audited dataset."""
    return gates.FrozenPrimaryDatasetManifest.from_audit_payload(
        selected_datasets=["m5", "online_retail"],
        eligible_datasets_in_priority_order=["m5", "online_retail"],
        audit_payload={"fixture": "two-dataset manifest"},
    )
