"""Metric and uncertainty checks (tests 28-31 plus supporting cases)."""

from __future__ import annotations

import numpy as np
import pytest

from covariate_trust.bootstrap import BOOTSTRAP_UNIT, paired_bootstrap
from covariate_trust.metrics import (future_value, is_harm, nmae, pinball,
                                     quantile_crossing_rate, relative_delta, wql)


def test_28_wql_matches_a_hand_computation():
    """Test 28: WQL on a tiny example equals the value computed by hand."""
    y = np.array([1.0, 2.0])
    levels = [0.1, 0.5, 0.9]
    q_pred = np.array([[0.5, 1.0, 1.5],
                       [1.5, 2.5, 3.0]])
    total = 0.0
    for j, q in enumerate(levels):
        for i in range(2):
            d = y[i] - q_pred[i, j]
            total += q * max(d, 0.0) + (1 - q) * max(-d, 0.0)
    expected = 2.0 * total / (len(levels) * (np.abs(y).sum() + 1e-8))
    assert wql(y, q_pred, levels) == pytest.approx(expected)
    # explicit arithmetic, term by term:
    #   q=0.1: 0.1*0.5 + 0.1*0.5            = 0.10
    #   q=0.5: 0.0    + 0.5*0.5             = 0.25
    #   q=0.9: 0.1*0.5 + 0.1*1.0            = 0.15
    assert total == pytest.approx(0.50)
    assert wql(y, q_pred, levels) == pytest.approx(2 * 0.50 / (3 * 3.0), rel=1e-6)


def test_28b_perfect_forecast_gives_zero_wql():
    y = np.array([1.0, -2.0, 3.0])
    levels = [0.1, 0.5, 0.9]
    q_pred = np.repeat(y[:, None], 3, axis=1)
    assert wql(y, q_pred, levels) == pytest.approx(0.0, abs=1e-12)


def test_28c_pinball_is_asymmetric():
    assert pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)
    assert pinball(np.array([-1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.1)


def test_28d_derived_quantities():
    assert future_value(0.5, 0.4) == pytest.approx(0.1)
    assert relative_delta(0.5, 0.55) == pytest.approx(0.1, rel=1e-6)
    assert is_harm(0.5, 0.56, 0.05) is True
    assert is_harm(0.5, 0.52, 0.05) is False


def test_29_paired_comparison_keeps_ids_aligned():
    """Test 29: the bootstrap consumes paired observations, one per unit and cell."""
    units = np.array([0, 1, 2, 3])
    base = np.array([1.0, 1.0, 1.0, 1.0])
    treat = np.array([0.9, 0.8, 1.1, 0.7])
    res = paired_bootstrap(units, base, treat, 500, 0.95, seed_parts=("t",))
    assert res.n_units == 4
    assert res.mean_diff == pytest.approx(np.mean(base - treat))
    assert res.win_rate == pytest.approx(0.75)
    with pytest.raises(ValueError):
        paired_bootstrap(units, base, treat[:3], 100, 0.95)


def test_30_bootstrap_unit_is_the_base_series(small_cfg):
    """Test 30: resampling happens over base_series_id clusters, not over rows."""
    assert BOOTSTRAP_UNIT == "base_series_id"
    rng = np.random.default_rng(0)
    units = np.repeat(np.arange(10), 5)          # 10 series, 5 cells each
    base = rng.normal(1.0, 0.05, size=len(units))
    treat = base - 0.10                          # constant paired effect
    res = paired_bootstrap(units, base, treat, 1000, 0.95, seed_parts=("t",))
    assert res.n_units == 10
    assert res.n_observations == 50
    assert res.ci_low > 0 and res.ci_high > 0    # a real effect is detected
    assert res.mean_diff == pytest.approx(0.10, abs=1e-9)
    # a cluster bootstrap on a constant effect has (near) zero width
    assert res.ci_high - res.ci_low < 1e-6


def test_30b_null_effect_gives_a_ci_covering_zero():
    rng = np.random.default_rng(1)
    units = np.repeat(np.arange(30), 2)
    base = rng.normal(1.0, 0.1, size=len(units))
    treat = base + rng.normal(0.0, 0.1, size=len(units))
    res = paired_bootstrap(units, base, treat, 2000, 0.95, seed_parts=("t",))
    assert res.ci_low < 0 < res.ci_high
    assert not res.ci_excludes_zero


def test_30c_bootstrap_is_reproducible():
    units = np.repeat(np.arange(8), 3)
    rng = np.random.default_rng(2)
    base = rng.normal(1.0, 0.1, size=len(units))
    # the effect must vary across units, otherwise every resample gives the same
    # mean and the interval would be seed-independent by construction
    treat = base - rng.normal(0.05, 0.02, size=len(units))
    a = paired_bootstrap(units, base, treat, 500, 0.95, seed_parts=("x", 1))
    b = paired_bootstrap(units, base, treat, 500, 0.95, seed_parts=("x", 1))
    c = paired_bootstrap(units, base, treat, 500, 0.95, seed_parts=("x", 2))
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


def test_31_quantile_crossing_rate():
    """Test 31: crossing is counted on adjacent quantile pairs."""
    ok = np.array([[1.0, 2.0, 3.0], [0.0, 0.5, 0.7]])
    assert quantile_crossing_rate(ok) == 0.0
    crossed = np.array([[1.0, 0.5, 3.0], [0.0, 0.5, 0.7]])
    assert quantile_crossing_rate(crossed) == pytest.approx(1 / 4)
    assert quantile_crossing_rate(np.array([[1.0], [2.0]])) == 0.0


def test_31b_nmae_uses_the_median_quantile():
    y = np.array([1.0, 2.0])
    levels = [0.1, 0.5, 0.9]
    q_pred = np.array([[0.0, 1.5, 2.0], [0.0, 1.5, 2.0]])
    assert nmae(y, q_pred, levels) == pytest.approx((0.5 + 0.5) / 3.0, rel=1e-6)
    with pytest.raises(ValueError):
        nmae(y, q_pred[:, :2], [0.1, 0.9])
