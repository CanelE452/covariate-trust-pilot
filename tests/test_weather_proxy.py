"""Reliability proxy checks (Study 3 tests 31-37)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.weather_proxy import (EPSILON, IsotonicCalibrator, SEASONAL_LAG_HOURS,
                                           add_decision_time_features, calibration_diagnostics,
                                           coverage_status, origin_weather_errors,
                                           raw_proxy_score, rmse, split_periods)


def _features(n=120, zone="NYC", start="2024-04-01"):
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n, freq="D") + pd.Timedelta(hours=6)
    return pd.DataFrame({
        "zone": zone, "origin_utc": idx,
        "realized_weather_error_ratio": rng.uniform(0.4, 1.8, n),
        "e_baseline_rmse_168": rng.uniform(2.0, 4.0, n),
        "revision_rms": rng.uniform(0.5, 3.0, n),
    })


def test_s31_revision_and_lambda_hand_computation():
    """Test 31: the per-origin weather quantities match an arithmetic computation."""
    primary = np.array([10.0, 12.0, 14.0])
    revision = np.array([11.0, 12.0, 13.0])
    verification = np.array([10.0, 11.0, 15.0])
    naive = np.array([8.0, 9.0, 13.0])
    out = origin_weather_errors(primary, revision, verification, naive, naive)
    assert out["e_external_rmse"] == pytest.approx(np.sqrt((0 + 1 + 1) / 3))
    assert out["e_baseline_rmse_168"] == pytest.approx(np.sqrt((4 + 4 + 4) / 3))
    assert out["revision_rms"] == pytest.approx(np.sqrt((1 + 0 + 1) / 3))
    assert out["realized_weather_error_ratio"] == pytest.approx(
        out["e_external_rmse"] / (out["e_baseline_rmse_168"] + EPSILON))
    assert rmse([1.0, np.nan], [1.0, 5.0]) == pytest.approx(0.0)


def test_s32_recent_reliability_uses_only_completed_past_origins():
    """Test 32: shifting before the rolling window keeps the current origin out."""
    f = _features(40)
    out = add_decision_time_features(f, window=5)
    row = out.iloc[10]
    manual = f["realized_weather_error_ratio"].iloc[5:10].mean()
    assert row["recent_realized_ratio"] == pytest.approx(manual)
    assert out["recent_realized_ratio"].iloc[0] != out["recent_realized_ratio"].iloc[0] or True
    assert np.isnan(out["recent_realized_ratio"].iloc[0])
    # changing only the *current* row's outcome cannot change its own feature
    g = f.copy()
    g.loc[g.index[10], "realized_weather_error_ratio"] = 99.0
    out2 = add_decision_time_features(g, window=5)
    assert out2["recent_realized_ratio"].iloc[10] == pytest.approx(row["recent_realized_ratio"])


def test_s33_raw_score_weights_are_fixed(external_cfg):
    """Test 33: the raw proxy is 0.70 revision + 0.30 recent, from config."""
    assert external_cfg.proxy.revision_weight == 0.70
    assert external_cfg.proxy.recent_error_weight == 0.30
    got = raw_proxy_score(np.array([1.0, 2.0]), np.array([0.0, 1.0]), 0.70, 0.30)
    np.testing.assert_allclose(got, [0.70, 1.70])


def test_s34_isotonic_fit_uses_the_training_period_only(external_cfg):
    """Test 34: the calibrator sees train rows and nothing later."""
    f = add_decision_time_features(_features(400), window=28)
    f["raw_proxy"] = raw_proxy_score(f["revision_ratio"], f["recent_realized_ratio"], 0.7, 0.3)
    splits = split_periods(f, external_cfg)
    assert len(splits["train"]) > 0
    assert splits["train"]["origin_utc"].max() <= pd.Timestamp(
        external_cfg.periods.proxy_train_end) + pd.Timedelta(days=1)
    assert splits["test"]["origin_utc"].min() >= pd.Timestamp(
        external_cfg.periods.heldout_test_start) if len(splits["test"]) else True
    # train and test never overlap
    if len(splits["test"]):
        assert splits["train"]["origin_utc"].max() < splits["test"]["origin_utc"].min()


def test_s35_calibrator_is_frozen_after_fitting():
    """Test 35: refitting on validation or test data is impossible by construction."""
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.3, 2.0, 200)
    cal = IsotonicCalibrator().fit(raw, 0.6 * raw + 0.2)
    assert cal.fitted_ and cal.n_train_ == 200
    with pytest.raises(RuntimeError, match="frozen"):
        cal.fit(raw, 0.9 * raw)
    a = cal.predict([0.5, 1.0, 1.5])
    b = cal.predict([0.5, 1.0, 1.5])
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(cal.predict(np.linspace(0.3, 2.0, 40))) >= -1e-9)


def test_s35b_unfitted_calibrator_refuses_to_predict():
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict([1.0])


def test_s36_reported_lambda_does_not_depend_on_current_true_lambda():
    """Test 36: perturbing the current realized_weather_error_ratio leaves the decision-time proxy unchanged."""
    f = _features(60)
    base = add_decision_time_features(f, window=28)
    base["raw"] = raw_proxy_score(base["revision_ratio"], base["recent_realized_ratio"], 0.7, 0.3)
    g = f.copy()
    g.loc[g.index[45], "realized_weather_error_ratio"] = 50.0        # only the current row
    other = add_decision_time_features(g, window=28)
    other["raw"] = raw_proxy_score(other["revision_ratio"], other["recent_realized_ratio"], 0.7, 0.3)
    assert other["raw"].iloc[45] == pytest.approx(base["raw"].iloc[45])


def test_s37_proxy_reacts_to_the_current_forecast_revision():
    """Test 37: a larger run-to-run revision does raise the current raw proxy."""
    f = _features(60)
    base = add_decision_time_features(f, window=28)
    base["raw"] = raw_proxy_score(base["revision_ratio"], base["recent_realized_ratio"], 0.7, 0.3)
    g = f.copy()
    g.loc[g.index[45], "revision_rms"] = f["revision_rms"].iloc[45] * 5.0
    other = add_decision_time_features(g, window=28)
    other["raw"] = raw_proxy_score(other["revision_ratio"], other["recent_realized_ratio"], 0.7, 0.3)
    assert other["raw"].iloc[45] > base["raw"].iloc[45]


def test_s37b_calibration_diagnostics_and_coverage(external_cfg):
    rng = np.random.default_rng(2)
    true = rng.uniform(0.3, 2.0, 300)
    rep = true + rng.normal(0, 0.05, 300)
    d = calibration_diagnostics(rep, true)
    assert d["spearman"] > 0.9 and d["quartile_ratio"] > 1.2 and d["n"] == 300

    f = _features(60)
    f["origin_utc"] = pd.date_range("2025-07-01", periods=60, freq="D") + pd.Timedelta(hours=6)
    cov = coverage_status({"train": f.head(5), "validation": f.head(5), "test": f}, external_cfg)
    assert cov["status"] in {"BLOCKED_COVERAGE", "PARTIAL_COVERAGE", "PASS_COVERAGE"}
    assert cov["minimum_test_origins_per_zone"] == external_cfg.gate_h.minimum_test_origins_per_zone
