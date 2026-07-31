"""Reliability schedule and uncertainty proxy checks (follow-up tests 19-30)."""

from __future__ import annotations

import numpy as np
import pytest

from covariate_trust.admission import PSEUDO_ORIGINS
from covariate_trust.config import ConfigError, DynamicConfig
from covariate_trust.dgp import covariate_vintage, eta_path, generate_base_series
from covariate_trust.reliability_schedules import (ORACLE_MODES, P0_ORACLE, P1_CALIBRATED,
                                                   P2_OVERCONFIDENT, P3_UNDERCONFIDENT, P4_STALE,
                                                   PROXY_MODES, calibrated_proxy, lambda_at,
                                                   proxy_record, reported_lambda,
                                                   schedule_origins, schedule_table)


def test_f19_schedule_length_matches_origin_count(dynamic_cfg):
    """Test 19: every schedule declares exactly one lambda per pseudo-origin."""
    for h in dynamic_cfg.grid.horizons:
        hist, primary = schedule_origins(dynamic_cfg, h)
        assert len(hist) == dynamic_cfg.n_historical_origins == 4
        assert hist == PSEUDO_ORIGINS[h]
        assert primary == dynamic_cfg.experiment.primary_origin
        for s in dynamic_cfg.schedules:
            assert len(s.historical) == len(hist)


def test_f19b_mismatched_schedule_length_is_rejected(dynamic_cfg):
    d = dynamic_cfg.to_dict()
    d.pop("inherited_from_pilot_yaml")
    d["schedules"][0]["historical"] = [0.5, 0.5, 0.5]
    with pytest.raises(ConfigError):
        DynamicConfig.from_dict(d, dynamic_cfg.inherited)


def test_f20_current_lambda_is_independent_of_history(dynamic_cfg):
    """Test 20: the primary lambda can differ from every historical lambda."""
    for h in dynamic_cfg.grid.horizons:
        cur = lambda_at(dynamic_cfg, "S2_sudden_worsening", h, dynamic_cfg.experiment.primary_origin)
        hist = [lambda_at(dynamic_cfg, "S2_sudden_worsening", h, o)
                for o in PSEUDO_ORIGINS[h]]
        assert cur == 1.50
        assert set(hist) == {0.50}
        assert cur not in set(hist)


def test_f21_sudden_worsening_schedule(dynamic_cfg):
    """Test 21: S2 is 0.50 throughout history and 1.50 now."""
    for h in dynamic_cfg.grid.horizons:
        hist, primary = schedule_origins(dynamic_cfg, h)
        assert [lambda_at(dynamic_cfg, "S2_sudden_worsening", h, o) for o in hist] == [0.5] * 4
        assert lambda_at(dynamic_cfg, "S2_sudden_worsening", h, primary) == 1.5
        assert [lambda_at(dynamic_cfg, "S4_gradual_worsening", h, o) for o in hist] == \
            [0.50, 0.75, 1.00, 1.25]


def test_f22_sudden_improvement_schedule(dynamic_cfg):
    """Test 22: S3 is 1.50 throughout history and 0.50 now."""
    for h in dynamic_cfg.grid.horizons:
        hist, primary = schedule_origins(dynamic_cfg, h)
        assert [lambda_at(dynamic_cfg, "S3_sudden_improvement", h, o) for o in hist] == [1.5] * 4
        assert lambda_at(dynamic_cfg, "S3_sudden_improvement", h, primary) == 0.5
        assert [lambda_at(dynamic_cfg, "S5_gradual_improvement", h, o) for o in hist] == \
            [1.50, 1.25, 1.00, 0.75]


def test_f22b_schedule_table_covers_every_origin(dynamic_cfg):
    t = schedule_table(dynamic_cfg)
    expected = len(dynamic_cfg.schedules) * len(dynamic_cfg.grid.horizons) * 5
    assert len(t) == expected
    assert t.groupby(["schedule", "horizon"])["is_primary"].sum().eq(1).all()
    # historical windows must all close at or before the primary origin
    for _, r in t[~t["is_primary"]].iterrows():
        assert r["origin"] + r["horizon"] <= dynamic_cfg.experiment.primary_origin


def test_f23_eta_paths_are_independent_across_origins(small_dynamic_cfg):
    """Test 23: each origin draws its own error path."""
    pilot = small_dynamic_cfg.to_pilot_config()
    h = 24
    paths = [eta_path(pilot, 0, o, h) for o in PSEUDO_ORIGINS[h]]
    paths.append(eta_path(pilot, 0, pilot.experiment.primary_origin, h))
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            assert not np.allclose(paths[i], paths[j])
    corrs = [abs(np.corrcoef(paths[i], paths[j])[0, 1])
             for i in range(len(paths)) for j in range(i + 1, len(paths))]
    assert max(corrs) < 0.6


def test_f24_methods_share_the_eta_path_at_one_origin(small_dynamic_cfg):
    """Test 24: at a given origin every lambda (hence every method) uses the same eta."""
    pilot = small_dynamic_cfg.to_pilot_config()
    s = generate_base_series(0, pilot)
    origin, h = 800, 24
    base = eta_path(pilot, 0, origin, h)
    for lam in (0.5, 1.0, 1.5):
        v = covariate_vintage(pilot, s, origin, h, lam)
        np.testing.assert_allclose(v["eta"], base)


def test_f25_proxy_noise_is_independent_of_forecast_error(small_dynamic_cfg):
    """Test 25: the proxy stream and the eta stream are uncorrelated."""
    pilot = small_dynamic_cfg.to_pilot_config()
    ratios, eta_means = [], []
    for b_id in range(60):
        s = generate_base_series(b_id, pilot)
        v = covariate_vintage(pilot, s, pilot.experiment.primary_origin, 24, 1.0)
        rep = calibrated_proxy(small_dynamic_cfg, b_id, 0.5, 24, "S0_stable_low", 1.0)
        ratios.append(rep)
        eta_means.append(float(v["eta"].mean()))
    r = abs(np.corrcoef(ratios, eta_means)[0, 1])
    assert r < 0.35, f"proxy and eta correlated at {r:.3f}"


def test_f26_only_p0_reads_the_true_current_lambda(small_dynamic_cfg):
    """Test 26: changing the truth moves P0 exactly; P4 does not read it at all."""
    cfg = small_dynamic_cfg
    kw = dict(base_series_id=0, share=0.5, horizon=24, schedule_name="S2_sudden_worsening",
              historical_lambda_estimates=[0.5, 0.5, 0.5, 0.5])
    assert ORACLE_MODES == {P0_ORACLE}
    assert reported_lambda(cfg, P0_ORACLE, true_current_lambda=1.5, **kw) == 1.5
    assert reported_lambda(cfg, P0_ORACLE, true_current_lambda=0.5, **kw) == 0.5
    # P4 ignores the current truth entirely
    a = reported_lambda(cfg, P4_STALE, true_current_lambda=1.5, **kw)
    b = reported_lambda(cfg, P4_STALE, true_current_lambda=0.5, **kw)
    assert a == b == 0.5
    for mode in PROXY_MODES:
        rec = proxy_record(cfg, mode, 1.0, 1.5)
        assert rec["uses_true_current_lambda"] == (mode == P0_ORACLE)


def test_f27_calibrated_proxy_is_reproducible_and_median_unbiased(small_dynamic_cfg):
    """Test 27: P1 is deterministic per task and unbiased in the median."""
    cfg = small_dynamic_cfg
    a = calibrated_proxy(cfg, 3, 0.5, 24, "S0_stable_low", 1.5)
    b = calibrated_proxy(cfg, 3, 0.5, 24, "S0_stable_low", 1.5)
    c = calibrated_proxy(cfg, 4, 0.5, 24, "S0_stable_low", 1.5)
    d = calibrated_proxy(cfg, 3, 0.5, 96, "S0_stable_low", 1.5)
    assert a == b
    assert a != c and a != d

    vals = [calibrated_proxy(cfg, i, 0.5, 24, "S0_stable_low", 1.0) for i in range(500)]
    assert np.median(vals) == pytest.approx(1.0, abs=0.05)
    assert all(v > 0 for v in vals)


def test_f28_overconfident_proxy_understates_uncertainty(small_dynamic_cfg):
    """Test 28: P2 reports a smaller lambda than the calibrated proxy."""
    cfg = small_dynamic_cfg
    kw = dict(base_series_id=1, share=0.5, horizon=24, schedule_name="S1_stable_high",
              historical_lambda_estimates=[1.5] * 4, true_current_lambda=1.5)
    p1 = reported_lambda(cfg, P1_CALIBRATED, **kw)
    p2 = reported_lambda(cfg, P2_OVERCONFIDENT, **kw)
    assert p2 == pytest.approx(cfg.proxy.overconfident_multiplier * p1)
    assert p2 < p1 < 1e9
    assert proxy_record(cfg, P2_OVERCONFIDENT, p2, 1.5)["calibration_ratio"] < 1.0


def test_f29_underconfident_proxy_overstates_uncertainty(small_dynamic_cfg):
    """Test 29: P3 reports a larger lambda than the calibrated proxy."""
    cfg = small_dynamic_cfg
    kw = dict(base_series_id=1, share=0.5, horizon=24, schedule_name="S0_stable_low",
              historical_lambda_estimates=[0.5] * 4, true_current_lambda=0.5)
    p1 = reported_lambda(cfg, P1_CALIBRATED, **kw)
    p3 = reported_lambda(cfg, P3_UNDERCONFIDENT, **kw)
    assert p3 == pytest.approx(cfg.proxy.underconfident_multiplier * p1)
    assert p3 > p1
    assert proxy_record(cfg, P3_UNDERCONFIDENT, p3, 0.5)["calibration_ratio"] > 1.0


def test_f30_stale_proxy_uses_history_only(small_dynamic_cfg):
    """Test 30: P4 is exactly the mean of the historical estimates."""
    cfg = small_dynamic_cfg
    hist = [0.4, 0.6, 0.8, 1.0]
    got = reported_lambda(cfg, P4_STALE, base_series_id=0, share=0.5, horizon=24,
                          schedule_name="S4_gradual_worsening", true_current_lambda=99.0,
                          historical_lambda_estimates=hist)
    assert got == pytest.approx(np.mean(hist))
    with pytest.raises(ValueError):
        reported_lambda(cfg, P4_STALE, base_series_id=0, share=0.5, horizon=24,
                        schedule_name="S4_gradual_worsening", true_current_lambda=1.0,
                        historical_lambda_estimates=[])


def test_f30b_unknown_modes_and_schedules_raise(small_dynamic_cfg):
    with pytest.raises(KeyError):
        reported_lambda(small_dynamic_cfg, "P9", base_series_id=0, share=0.5, horizon=24,
                        schedule_name="S0_stable_low", true_current_lambda=1.0,
                        historical_lambda_estimates=[1.0])
    with pytest.raises(KeyError):
        lambda_at(small_dynamic_cfg, "nope", 24, 896)
    with pytest.raises(KeyError):
        lambda_at(small_dynamic_cfg, "S0_stable_low", 24, 999)
