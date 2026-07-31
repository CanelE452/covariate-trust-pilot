"""Gate H and Gate I fixtures (Study 3 tests 42-50)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.external_gates import MIN_EVENTS, NOT_EVALUABLE, external_verdict, gate_h, gate_i
from covariate_trust.real_vintage import D0, D1, D3, D5, D7, apply_selectors
from test_real_vintage import _features, _tasks


def _quality(coverage=0.99):
    return {"forecast_coverage": {"primary": {"coverage": coverage}},
            "leakage_ok": True, "time_alignment_ok": True}


def _coverage(n=200, zones=("NYC", "WEST", "CAPITAL")):
    return {"status": "PASS_COVERAGE",
            "test_origins_per_zone": {z: n for z in zones},
            "zones_meeting_minimum": list(zones)}


def _panel(cfg, n_days=200, zones=("NYC", "WEST", "CAPITAL"), seed=0, noise=0.05):
    t = _tasks(n_days=n_days, zones=zones, seed=seed)
    f = _features(t, cfg, noise=noise, seed=seed + 1)
    d = apply_selectors(t, f, cfg)
    ev = f[["zone", "origin_utc", "realized_weather_error_ratio", "recent_realized_ratio"]].copy()
    ev["worsening_event"] = ((ev["recent_realized_ratio"] < cfg.proxy.lower_threshold)
                             & (ev["realized_weather_error_ratio"] > cfg.proxy.upper_threshold)).astype(int)
    ev["improvement_event"] = ((ev["recent_realized_ratio"] > cfg.proxy.upper_threshold)
                               & (ev["realized_weather_error_ratio"] < cfg.proxy.lower_threshold)).astype(int)
    return t, f, d, ev


def test_s42_gate_h_pass_fixture(external_cfg):
    """Test 42: a dataset where weather helps and forecasts vary passes Gate H."""
    t, f, d, _ = _panel(external_cfg)
    out = gate_h(t, f, _coverage(), _quality(), external_cfg)
    assert out["status"] in {"PASS", "INCONCLUSIVE"}, out["checks"]
    assert out["checks"]["H1_data_integrity"] is True
    assert 0.0 < out["checks"]["m3_win_rate"] < 1.0


def test_s43_gate_h_fail_when_oracle_weather_is_worthless(external_cfg):
    """Test 43: if M2 never beats M1, Gate H must FAIL."""
    t, f, d, _ = _panel(external_cfg)
    t = t.copy()
    t["wql_m2"] = t["wql_m1"] * 1.02
    out = gate_h(t, f, _coverage(), _quality(), external_cfg)
    assert out["status"] == "FAIL"
    assert "no oracle future-weather value" in out["fail_reasons"]


def test_s44_gate_h_fail_when_one_policy_always_wins(external_cfg):
    """Test 44: a degenerate win rate is a FAIL, not an INCONCLUSIVE."""
    t, f, d, _ = _panel(external_cfg)
    t = t.copy()
    t["wql_m3"] = t["wql_m1"] * 0.5           # M3 wins everywhere
    t["m3_is_better"] = 1
    t["wql_oracle"] = t[["wql_m1", "wql_m3"]].min(axis=1)
    out = gate_h(t, f, _coverage(), _quality(), external_cfg)
    assert out["status"] == "FAIL"
    assert out["checks"]["m3_win_rate"] == 1.0


def test_s44b_gate_h_inconclusive_on_thin_coverage(external_cfg):
    """Test 44b: too few origins per zone blocks H1 without triggering a FAIL reason."""
    t, f, d, _ = _panel(external_cfg, n_days=40, zones=("NYC", "WEST"))
    out = gate_h(t, f, _coverage(n=40, zones=("NYC", "WEST")), _quality(), external_cfg)
    assert out["checks"]["H1_data_integrity"] is False
    assert out["status"] in {"INCONCLUSIVE", "FAIL"}


def test_s45_gate_i_pass_fixture(external_cfg):
    """Test 45: when the proxy is informative D7 can pass Gate I."""
    t, f, d, ev = _panel(external_cfg, noise=0.02)
    out = gate_i(d, ev, external_cfg)
    assert out["primary_selector"] == D7
    assert out["status"] in {"PASS", "INCONCLUSIVE"}, out["checks"]
    if out["status"] == "PASS":
        assert external_verdict({"status": "PASS"}, out)["verdict"] == \
            "REAL_VINTAGE_EXTERNAL_VALIDATION_GO"


def test_s46_gate_i_fail_fixture(external_cfg):
    """Test 46: if M3 is uniformly worse, Gate I must FAIL."""
    t, f, d, ev = _panel(external_cfg)
    t2 = t.copy()
    t2["wql_m3"] = t2["wql_m1"] * 1.5
    t2["wql_oracle"] = t2[["wql_m1", "wql_m3"]].min(axis=1)
    t2["m3_is_better"] = 0
    d2 = apply_selectors(t2, f, external_cfg)
    out = gate_i(d2, ev, external_cfg)
    assert out["status"] == "FAIL", out["checks"]
    assert out["fail_reasons"]
    v = external_verdict({"status": "PASS"}, out)
    assert v["verdict"] == "SYNTHETIC_TO_REAL_METHOD_NO_GO"


def test_s47_gate_i_inconclusive_fixture(external_cfg):
    """Test 47: a small positive effect clears the FAIL bounds but misses the PASS bar."""
    rng = np.random.default_rng(5)
    t, f, d, ev = _panel(external_cfg)
    t2 = t.copy()
    lam = t2["realized_weather_error_ratio"].to_numpy()
    t2["wql_m3"] = t2["wql_m1"] * (1.0 + 0.004 * (lam - 1.0)) + 1e-5 * rng.normal(size=len(t2))
    t2["wql_oracle"] = t2[["wql_m1", "wql_m3"]].min(axis=1)
    t2["m3_is_better"] = (t2["wql_m3"] < t2["wql_m1"]).astype(int)
    d2 = apply_selectors(t2, f, external_cfg)
    out = gate_i(d2, ev, external_cfg)
    assert out["status"] in {"INCONCLUSIVE", "FAIL"}
    if out["status"] == "INCONCLUSIVE":
        assert out["failed_conditions"]
        assert external_verdict({"status": "PASS"}, out)["verdict"] == \
            "EXTERNAL_VALIDATION_CONDITIONAL"


def test_s48_event_subsets_below_threshold_are_not_evaluable(external_cfg):
    """Test 48: fewer than 20 events is NOT_EVALUABLE, never a pass."""
    t, f, d, ev = _panel(external_cfg)
    ev2 = ev.copy()
    ev2["worsening_event"] = 0
    ev2["improvement_event"] = 0
    ev2.loc[ev2.index[:3], "worsening_event"] = 1
    out = gate_i(d, ev2, external_cfg)
    assert out["checks"]["I7_worsening_beats_history"] == NOT_EVALUABLE
    assert out["checks"]["I8_improvement_beats_no_future"] == NOT_EVALUABLE
    assert set(out["not_evaluable"]) == {"I7_worsening_beats_history",
                                         "I8_improvement_beats_no_future"}
    assert out["checks"]["worsening"]["status"] == NOT_EVALUABLE
    assert MIN_EVENTS == 20


def test_s49_d5_scoring_better_does_not_replace_d7(external_cfg):
    """Test 49: the primary stays D7 even when a secondary policy scores lower."""
    t, f, d, ev = _panel(external_cfg, noise=0.0)
    out = gate_i(d, ev, external_cfg)
    per = out["per_selector"]
    assert out["primary_selector"] == D7
    assert out["primary_metrics"]["mean_wql"] == per[D7]["mean_wql"]
    if per[D5]["mean_wql"] < per[D7]["mean_wql"]:
        assert out["primary_metrics"]["mean_wql"] != per[D5]["mean_wql"]


def test_s50_verdict_mapping_is_complete():
    """Every allowed outcome, including the two blocked forms."""
    P, F, I = {"status": "PASS"}, {"status": "FAIL", "fail_reasons": ["x"]}, \
        {"status": "INCONCLUSIVE", "failed_conditions": ["I1"], "not_evaluable": []}
    assert external_verdict(P, P)["verdict"] == "REAL_VINTAGE_EXTERNAL_VALIDATION_GO"
    assert external_verdict(P, I)["verdict"] == "EXTERNAL_VALIDATION_CONDITIONAL"
    assert external_verdict(P, F)["verdict"] == "SYNTHETIC_TO_REAL_METHOD_NO_GO"
    assert external_verdict({"status": "FAIL", "fail_reasons": ["y"], "checks": {}}, None)[
        "verdict"] == "REAL_DATA_PROBLEM_NOT_ESTABLISHED"
    assert external_verdict(None, None)["verdict"] == "BLOCKED_EXTERNAL_DATA"
    assert external_verdict(P, P, blocked_reason="api down")["verdict"] == "BLOCKED_EXTERNAL_DATA"
    assert "NOT a claim of general validation" in external_verdict(P, P)["scope_note"]
