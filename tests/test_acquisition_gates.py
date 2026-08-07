"""Study 4 gate fixtures (items 45-50) and Study 3 non-interference (51-54)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from covariate_trust.acquisition_gates import (
    evaluate_ba0, evaluate_ba1, evaluate_ba2, evaluate_ba3, evaluate_ba4, evaluate_ba5,
    final_status,
)

THRESHOLDS = {
    "ba1_oracle_vs_no_premium": 0.02,
    "ba1_oracle_vs_random": 0.01,
    "ba2_simple_heuristic_oracle_recovery_sufficient": 0.90,
    "ba3_value_vs_random": 0.015,
    "ba3_value_oracle_recovery": 0.35,
    "ba4_q90_value_vs_wql_selector": 0.01,
    "maximum_zone_regression": 0.02,
}


def ba0_evidence(**overrides):
    checks = {
        "four_zone_portfolios": True, "fair_comparison_guard": True,
        "no_future_leakage": True, "no_current_premium_feature": True,
        "chronological_split": True, "cross_learning_false": True,
        "study3_hashes_unchanged": True,
    }
    checks.update(overrides.pop("checks", {}))
    ev = {"checks": checks,
          "portfolio_days": {"train": 239, "validation": 164, "test": 322},
          "minimum_days": {"train": 150, "validation": 100, "test": 250}}
    ev.update(overrides)
    return ev


def ba1_evidence(vs_np=0.05, vs_rand=0.03, ci=True, share=0.4, positive=0.5, fixed=0.4):
    return {"per_k": {k: {"oracle_vs_no_premium": vs_np, "oracle_vs_random": vs_rand,
                          "ci_favours_oracle": ci, "max_zone_share": share,
                          "best_fixed_zone_recovery": fixed} for k in ("1", "2")},
            "premium_positive_rate": positive}


def ba3_evidence(vs_rand=0.03, vs_np=0.02, rec=0.5, ci=True, share=0.4, zones=4):
    return {"per_k": {k: {"value_vs_random": vs_rand, "value_vs_no_premium": vs_np,
                          "oracle_recovery": rec, "ci_favours_value": ci,
                          "max_zone_share": share} for k in ("1", "2")},
            "zones_improved": zones}


def test_45_ba0_pass_and_fail():
    assert evaluate_ba0(ba0_evidence()).status == "PASS"
    bad = evaluate_ba0(ba0_evidence(checks={"no_current_premium_feature": False}))
    assert bad.status == "FAIL" and bad.decision == "INVALID_PILOT"
    short = evaluate_ba0(ba0_evidence(portfolio_days={"train": 10, "validation": 5, "test": 5}))
    assert short.status == "FAIL"
    assert final_status(short, None, None, None, None, None) == "INVALID_PILOT"


def test_46_ba1_pass_and_fail():
    good = evaluate_ba1(ba1_evidence(), THRESHOLDS)
    assert good.status == "PASS" and good.decision == "BUDGET_HEADROOM_CONFIRMED"

    flat = evaluate_ba1(ba1_evidence(vs_rand=0.002), THRESHOLDS)
    assert flat.status == "FAIL" and flat.decision == "BUDGETED_ACQUISITION_NO_GO"

    fixed = evaluate_ba1(ba1_evidence(fixed=0.99), THRESHOLDS)
    assert fixed.status == "FAIL"
    assert any("fixed single-zone" in r for r in fixed.fail_reasons)

    always = evaluate_ba1(ba1_evidence(positive=0.99), THRESHOLDS)
    assert always.status == "FAIL"

    weak = evaluate_ba1(ba1_evidence(vs_np=0.015), THRESHOLDS)
    assert weak.status == "INCONCLUSIVE"


def test_ba1_zone_concentration_guard():
    result = evaluate_ba1(ba1_evidence(share=0.9), THRESHOLDS)
    assert result.status == "INCONCLUSIVE"
    assert not result.criteria[3].passed


def test_47_ba2_simple_heuristic_sufficient():
    enough = evaluate_ba2({"best_heuristic_recovery": 0.95,
                           "heuristic_minus_value_wql_rel": 0.002}, THRESHOLDS)
    assert enough.status == "PASS"
    assert "NEW_VALUE_MODEL_NO_GO" in enough.decision

    not_enough = evaluate_ba2({"best_heuristic_recovery": 0.30,
                               "heuristic_minus_value_wql_rel": 0.05}, THRESHOLDS)
    assert not_enough.status == "FAIL"
    assert "PROCEED_TO_BA3" in not_enough.decision


def test_48_ba3_pass_fail_inconclusive():
    good = evaluate_ba3(ba3_evidence(), THRESHOLDS)
    assert good.status == "PASS" and good.decision == "FORECAST_VALUE_ROUTING_GO"

    bad = evaluate_ba3(ba3_evidence(vs_rand=-0.01, rec=0.05), THRESHOLDS)
    assert bad.status == "FAIL" and "VALUE_NOT_PREDICTABLE" in bad.decision

    middling = evaluate_ba3(ba3_evidence(vs_rand=0.005, rec=0.25), THRESHOLDS)
    assert middling.status == "INCONCLUSIVE"

    concentrated = evaluate_ba3(ba3_evidence(share=0.95), THRESHOLDS)
    assert concentrated.status == "INCONCLUSIVE"


def test_49_ba4_pass_and_fail():
    good = evaluate_ba4({"q90_selector_vs_wql_selector_on_q90": 0.03,
                         "q90_selector_vs_wql_selector_on_wql": 0.0,
                         "ci_favours_q90_selector": True, "selection_overlap": 0.6}, THRESHOLDS)
    assert good.status == "PASS"

    same = evaluate_ba4({"q90_selector_vs_wql_selector_on_q90": 0.03,
                         "q90_selector_vs_wql_selector_on_wql": 0.0,
                         "ci_favours_q90_selector": True, "selection_overlap": 0.99}, THRESHOLDS)
    assert same.status == "FAIL"

    costly = evaluate_ba4({"q90_selector_vs_wql_selector_on_q90": 0.03,
                           "q90_selector_vs_wql_selector_on_wql": -0.05,
                           "ci_favours_q90_selector": True, "selection_overlap": 0.6}, THRESHOLDS)
    assert costly.status == "FAIL"
    assert any("WQL guard" in r for r in costly.fail_reasons)


def test_50_ba5_low_count_is_not_evaluable():
    low = evaluate_ba5({"n_portfolio_days": 0, "minimum_days": 20, "value_vs_random": float("nan"),
                        "oracle_recovery": float("nan"),
                        "q90_selector_vs_random_on_q90": float("nan"), "budget_respected": True})
    assert low.status == "NOT_EVALUABLE_LOW_COUNT"
    assert "0 < 20" in low.decision

    ok = evaluate_ba5({"n_portfolio_days": 25, "minimum_days": 20, "value_vs_random": 0.02,
                       "oracle_recovery": 0.4, "q90_selector_vs_random_on_q90": 0.01,
                       "budget_respected": True})
    assert ok.status == "PASS"


def test_final_status_chain():
    ba0 = evaluate_ba0(ba0_evidence())
    ba1 = evaluate_ba1(ba1_evidence(), THRESHOLDS)
    heur = evaluate_ba2({"best_heuristic_recovery": 0.95,
                         "heuristic_minus_value_wql_rel": 0.001}, THRESHOLDS)
    assert final_status(ba0, ba1, heur, None, None, None).startswith("SIMPLE_RULE_OPERATIONAL_RESULT")

    ba2 = evaluate_ba2({"best_heuristic_recovery": 0.2,
                        "heuristic_minus_value_wql_rel": 0.05}, THRESHOLDS)
    ba3 = evaluate_ba3(ba3_evidence())if False else evaluate_ba3(ba3_evidence(), THRESHOLDS)
    ba4 = evaluate_ba4({"q90_selector_vs_wql_selector_on_q90": 0.0,
                        "q90_selector_vs_wql_selector_on_wql": 0.0,
                        "ci_favours_q90_selector": False, "selection_overlap": 0.99}, THRESHOLDS)
    status = final_status(ba0, ba1, ba2, ba3, ba4, None)
    assert status.startswith("FORECAST_VALUE_ROUTING_GO")
    assert "DECISION_SPECIFIC_VALUE_NOT_ESTABLISHED" in status


# -- Study 3 non-interference ------------------------------------------------

STUDY3_RUN = Path("runs/20260731_123122_real_vintage")


def test_52_53_study3_gates_and_d7_threshold_are_untouched():
    if not STUDY3_RUN.exists():
        pytest.skip("Study 3 run not present in this checkout")
    gate_h = json.loads((STUDY3_RUN / "tables" / "gate_h.json").read_text())
    gate_i = json.loads((STUDY3_RUN / "tables" / "gate_i.json").read_text())
    assert gate_h["status"] == "PASS"
    assert gate_i["status"] == "FAIL"

    import yaml

    cfg = yaml.safe_load(Path("configs/study3_real_vintage.yaml").read_text())
    assert cfg["proxy"]["lower_threshold"] == 0.75
    assert cfg["proxy"]["upper_threshold"] == 1.25
    assert cfg["proxy"]["d5_threshold"] == 1.00
    assert cfg["model"]["cross_learning"] is False


def test_54_study4_config_does_not_redefine_study3_periods():
    import yaml

    s3 = yaml.safe_load(Path("configs/study3_real_vintage.yaml").read_text())
    s4 = yaml.safe_load(Path("configs/study4_budgeted_acquisition.yaml").read_text())
    assert s4["periods"]["retrospective_test_start"] == s3["periods"]["heldout_test_start"]
    assert s4["periods"]["retrospective_test_end"] == s3["periods"]["heldout_test_end"]
    assert s4["experiment"]["origin_hour_utc"] == s3["experiment"]["decision_origin_hour_utc"]
    # Study 4 must not carry any D7 threshold of its own
    assert "lower_threshold" not in json.dumps(s4)
    assert "d5_threshold" not in json.dumps(s4)
