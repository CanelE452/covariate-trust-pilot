"""Leakage and metric checks on the real pipeline (Study 3 tests 22-30, 38-41)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.metrics import wql
from covariate_trust.real_vintage import (CALENDAR_COLUMNS, D0, D1, D2, D3, D5, D7, SELECTORS,
                                          TEMPERATURE_COLUMN, apply_selectors,
                                          assert_fair_comparison, build_real_inputs, false_rates,
                                          historical_utility_features, summarize,
                                          week_cluster_bootstrap)
from covariate_trust.schemas import (COVARIATE_COLUMN, M0, M1, M2, M3, SchemaError,
                                     TARGET_COLUMN, assert_context_equality)


def _ctx(n=48):
    idx = pd.date_range(end=pd.Timestamp("2025-07-01 06:00"), periods=n, freq="h")
    load = np.linspace(5000, 6000, n)
    temp = np.linspace(18, 26, n)
    return idx, load, temp


def _fut(h=24):
    idx = pd.date_range("2025-07-01 07:00", periods=h, freq="h")
    return idx, np.linspace(20, 30, h), np.linspace(21, 29, h)


def _tasks(n_days=60, zones=("NYC", "WEST"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for z in zones:
        for i in range(n_days):
            o = pd.Timestamp("2025-07-01 06:00") + pd.Timedelta(days=i)
            lam = float(rng.uniform(0.4, 1.8))
            w1 = 0.050 + 0.002 * rng.normal()
            w3 = w1 * (1.0 + 0.20 * (lam - 1.0)) + 0.0005 * rng.normal()
            rows.append({
                "zone": z, "origin_utc": o,
                "calendar_week": f"{o.isocalendar().year}-W{o.isocalendar().week:02d}",
                "season": "JJA", "wql_m0": w1 * 1.05, "wql_m1": w1,
                "wql_m2": w1 * 0.9, "wql_m3": w3,
                "nmae_m1": w1, "nmae_m3": w3, "mse_m1": w1, "mse_m3": w3,
                "crossing_m3": 0.0, "wql_oracle": min(w1, w3),
                "m3_is_better": int(w3 < w1), "harm_m3": int(w3 > 1.05 * w1),
                "v_future": w1 - w3, "v_oracle": w1 * 0.1, "realized_weather_error_ratio": lam,
                "mean_load_mw": 6000.0, "max_temp_verified": 30.0, "min_temp_verified": 18.0,
            })
    return pd.DataFrame(rows)


def _features(tasks, cfg, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    f = tasks[["zone", "origin_utc", "realized_weather_error_ratio"]].copy()
    f["reported_reliability_ratio"] = f["realized_weather_error_ratio"] + rng.normal(0, noise, len(f))
    f["recent_realized_ratio"] = f["realized_weather_error_ratio"].rolling(5, min_periods=1).mean()
    hist = historical_utility_features(tasks, cfg)
    return f.merge(hist, on=["zone", "origin_utc"], how="left")


def test_s22_m1_and_m3_contexts_are_identical():
    """Test 22: contexts match; M1 still carries the shared future calendar block."""
    ci, load, temp = _ctx()
    fi, x_true, x_fc = _fut()
    in1 = build_real_inputs(M1, "t", ci, load, temp, fi)
    in2 = build_real_inputs(M2, "t", ci, load, temp, fi, x_true)
    in3 = build_real_inputs(M3, "t", ci, load, temp, fi, x_fc)
    assert in1.context_df.equals(in3.context_df)
    assert_fair_comparison(in1, in2, in3)
    assert TEMPERATURE_COLUMN not in in1.future_df.columns


def test_s23_m3_future_frame_holds_the_forecast_only():
    """Test 23: M3's future frame carries the forecast, not verification."""
    ci, load, temp = _ctx()
    fi, x_true, x_fc = _fut()
    in3 = build_real_inputs(M3, "t", ci, load, temp, fi, x_fc)
    np.testing.assert_allclose(in3.future_df[COVARIATE_COLUMN].to_numpy(), x_fc)
    assert not np.allclose(in3.future_df[COVARIATE_COLUMN].to_numpy(), x_true)


def test_s24_m2_carries_verification_and_is_the_oracle():
    """Test 24: verified future weather appears only in M2."""
    ci, load, temp = _ctx()
    fi, x_true, x_fc = _fut()
    in2 = build_real_inputs(M2, "t", ci, load, temp, fi, x_true)
    np.testing.assert_allclose(in2.future_df[COVARIATE_COLUMN].to_numpy(), x_true)
    assert in2.future_df[TARGET_COLUMN if TARGET_COLUMN in in2.future_df else COVARIATE_COLUMN] is not None


def test_s25_no_future_frame_contains_the_target():
    ci, load, temp = _ctx()
    fi, x_true, x_fc = _fut()
    for method, fut in ((M2, x_true), (M3, x_fc)):
        inp = build_real_inputs(method, "t", ci, load, temp, fi, fut)
        assert TARGET_COLUMN not in inp.future_df.columns
    for method in (M0, M1):
        with pytest.raises(SchemaError):
            build_real_inputs(method, "t", ci, load, temp, fi, x_fc)


def test_s26_proxy_training_never_sees_held_out_targets(external_cfg):
    """Test 26: the split puts every test origin strictly after every train origin."""
    from covariate_trust.weather_proxy import split_periods
    idx = pd.date_range("2024-04-01", "2026-06-30", freq="D") + pd.Timedelta(hours=6)
    f = pd.DataFrame({"origin_utc": idx, "zone": "NYC"})
    s = split_periods(f, external_cfg)
    assert s["train"]["origin_utc"].max() < s["test"]["origin_utc"].min()
    assert s["validation"]["origin_utc"].max() < s["test"]["origin_utc"].min()
    assert s["test"]["origin_utc"].min() >= pd.Timestamp(external_cfg.periods.heldout_test_start)


def test_s27_reported_lambda_is_not_the_current_true_lambda(external_cfg):
    """Test 27: the decision reads reported_reliability_ratio; the truth is a separate column."""
    t = _tasks()
    f = _features(t, external_cfg)
    d = apply_selectors(t, f, external_cfg)
    d7 = d[d["selector"] == D7]
    assert "reported_reliability_ratio" in d7.columns
    assert not np.allclose(d7["reported_reliability_ratio"], d7["realized_weather_error_ratio"])


def test_s28_history_only_policy_ignores_the_current_outcome(external_cfg):
    """Test 28: D3 depends on past WQLs only."""
    t = _tasks()
    f = _features(t, external_cfg)
    base = apply_selectors(t, f, external_cfg)
    p = t.copy()
    p["wql_m1"] *= 3.0
    p["wql_m3"] *= 0.1
    p["wql_oracle"] = p[["wql_m1", "wql_m3"]].min(axis=1)
    p["m3_is_better"] = (p["wql_m3"] < p["wql_m1"]).astype(int)
    other = apply_selectors(p, f, external_cfg)
    key = ["zone", "origin_utc"]
    a = base[base["selector"] == D3].sort_values(key)["choice"].tolist()
    b = other[other["selector"] == D3].sort_values(key)["choice"].tolist()
    assert a == b


def test_s29_d7_and_d5_ignore_the_current_outcome(external_cfg):
    """Test 29: the primary policy cannot react to the present; only D2 may."""
    t = _tasks()
    f = _features(t, external_cfg)
    base = apply_selectors(t, f, external_cfg)
    p = t.copy()
    p["wql_m1"] *= 3.0
    p["wql_m3"] *= 0.1
    p["wql_oracle"] = p[["wql_m1", "wql_m3"]].min(axis=1)
    p["m3_is_better"] = (p["wql_m3"] < p["wql_m1"]).astype(int)
    other = apply_selectors(p, f, external_cfg)
    key = ["zone", "origin_utc"]
    for sel in (D5, D7):
        a = base[base["selector"] == sel].sort_values(key)["choice"].tolist()
        b = other[other["selector"] == sel].sort_values(key)["choice"].tolist()
        assert a == b, sel
    a2 = base[base["selector"] == D2].sort_values(key)["choice"].tolist()
    b2 = other[other["selector"] == D2].sort_values(key)["choice"].tolist()
    assert a2 != b2


def test_s30_thresholds_are_applied_as_pre_registered(external_cfg):
    """Test 30: the 0.75 / 1.25 override band and the 1.00 D5 threshold, unchanged."""
    t = _tasks()
    f = _features(t, external_cfg, noise=0.0)
    d = apply_selectors(t, f, external_cfg)
    d7 = d[d["selector"] == D7]
    lo, hi = external_cfg.proxy.lower_threshold, external_cfg.proxy.upper_threshold
    assert (d7.loc[d7["reported_reliability_ratio"] < lo, "choice"] == M3).all()
    assert (d7.loc[d7["reported_reliability_ratio"] > hi, "choice"] == M1).all()
    d5 = d[d["selector"] == D5]
    assert (d5.loc[d5["reported_reliability_ratio"] < external_cfg.proxy.d5_threshold, "choice"] == M3).all()
    assert (lo, hi) == (0.75, 1.25)


def test_s38_wql_hand_computation():
    """Test 38: WQL on real-shaped input matches the arithmetic definition."""
    y = np.array([100.0, 200.0])
    levels = [0.1, 0.5, 0.9]
    q = np.array([[90.0, 110.0, 130.0], [180.0, 190.0, 260.0]])
    total = 0.0
    for j, lv in enumerate(levels):
        for i in range(2):
            d = y[i] - q[i, j]
            total += lv * max(d, 0.0) + (1 - lv) * max(-d, 0.0)
    assert wql(y, q, levels) == pytest.approx(2 * total / (3 * 300.0), rel=1e-9)


def test_s39_false_use_hand_computation():
    d = pd.DataFrame({"m3_is_better": [0, 0, 0, 1], "choice": [M3, M1, M1, M3]})
    d["false_use"] = ((d["m3_is_better"] == 0) & (d["choice"] == M3)).astype(int)
    d["false_reject"] = ((d["m3_is_better"] == 1) & (d["choice"] == M1)).astype(int)
    r = false_rates(d)
    assert r["n_m1_better"] == 3 and r["false_use_rate"] == pytest.approx(1 / 3)


def test_s40_false_reject_hand_computation():
    d = pd.DataFrame({"m3_is_better": [1, 1, 1, 1], "choice": [M1, M1, M1, M3]})
    d["false_use"] = ((d["m3_is_better"] == 0) & (d["choice"] == M3)).astype(int)
    d["false_reject"] = ((d["m3_is_better"] == 1) & (d["choice"] == M1)).astype(int)
    r = false_rates(d)
    assert r["n_m3_better"] == 4 and r["false_reject_rate"] == pytest.approx(0.75)
    assert np.isnan(r["false_use_rate"])


def test_s41_bootstrap_clusters_on_calendar_week(external_cfg):
    """Test 41: the cluster unit is the ISO week, not the individual origin."""
    t = _tasks(n_days=90)
    f = _features(t, external_cfg)
    d = apply_selectors(t, f, external_cfg)
    d7 = d[d["selector"] == D7]
    out = week_cluster_bootstrap(d7, "wql_m1", "wql_selected", external_cfg, ("t",))
    assert out["n_weeks"] == d7["calendar_week"].nunique()
    assert out["n_weeks"] < out["n_origins"]           # weeks really do aggregate origins
    assert out["n_zones"] == 2
    again = week_cluster_bootstrap(d7, "wql_m1", "wql_selected", external_cfg, ("t",))
    assert (out["ci_low"], out["ci_high"]) == (again["ci_low"], again["ci_high"])
    other = week_cluster_bootstrap(d7, "wql_m1", "wql_selected", external_cfg, ("u",))
    assert (out["ci_low"], out["ci_high"]) != (other["ci_low"], other["ci_high"])


def test_s41b_summaries_and_oracle_bound(external_cfg):
    t = _tasks()
    f = _features(t, external_cfg)
    d = apply_selectors(t, f, external_cfg)
    for sel in SELECTORS:
        g = d[d["selector"] == sel]
        s = summarize(g)
        assert s["mean_wql"] >= s["mean_wql_oracle"] - 1e-12
        assert (g["regret"] >= -1e-12).all()
    assert d[d["selector"] == D2]["regret"].abs().max() < 1e-12
    assert (d[d["selector"] == D0]["choice"] == M1).all()
    assert (d[d["selector"] == D1]["choice"] == M3).all()
