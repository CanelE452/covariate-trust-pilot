"""Study 1B checks (follow-up tests 1-10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.boundary import (FINITE, LEFT_CENSORED, RIGHT_CENSORED, UNRESOLVED,
                                      boundary_estimates, bootstrap_crossing, curve_summary,
                                      run_boundary_study, zero_crossing)
from covariate_trust.config import ConfigError, PilotConfig
from covariate_trust.dgp import generate_base_series
from covariate_trust.followup_gates import gate_e
from covariate_trust.schemas import M1, M2, M3


def test_f01_boundary_seed_differs_from_coarse(boundary_cfg, pilot_cfg):
    """Test 1: Study 1B runs on an independent master seed."""
    assert boundary_cfg.experiment.master_seed != pilot_cfg.experiment.master_seed
    assert boundary_cfg.experiment.master_seed == 20260801
    assert pilot_cfg.experiment.master_seed == 20260730


def test_f02_boundary_series_are_not_the_coarse_series(small_boundary_cfg, pilot_cfg):
    """Test 2: no generated series is bitwise identical to a coarse-pilot series."""
    new = small_boundary_cfg.to_pilot_config()
    for b_id in small_boundary_cfg.base_series_ids:
        a = generate_base_series(b_id, new)
        for other_id in range(4):
            c = generate_base_series(other_id, pilot_cfg)
            assert not np.array_equal(a.b, c.b)
            assert not np.array_equal(a.x, c.x)


def test_f03_lambda_grid_matches_config(boundary_cfg):
    """Test 3: the lambda grid is exactly what the config declares."""
    assert list(boundary_cfg.grid.lambda_values) == [0.70, 0.85, 1.00, 1.15, 1.30]
    assert list(boundary_cfg.grid.nominal_covariate_share) == [0.25, 0.50, 0.75]
    assert list(boundary_cfg.grid.horizons) == [24, 96]
    assert boundary_cfg.grid.n_series_per_cell == 150


def test_f04_m1_and_m2_are_not_recomputed_per_lambda(small_boundary_cfg):
    """Test 4: M1/M2 are inference-cached per (series, share, horizon), not per lambda."""
    calls = []

    def fake_predict(inputs, meta):
        calls.append((meta["method"], meta["base_series_id"], meta["nominal_covariate_share"],
                      meta["horizon"], meta.get("lam")))
        return np.tile(np.linspace(-1, 1, len(small_boundary_cfg.experiment.quantile_levels)),
                       (inputs.horizon, 1))

    run_boundary_study(small_boundary_cfg, fake_predict)
    n_cells = (len(small_boundary_cfg.base_series_ids)
               * len(small_boundary_cfg.grid.nominal_covariate_share)
               * len(small_boundary_cfg.grid.horizons))
    n_lam = len(small_boundary_cfg.grid.lambda_values)
    counts = pd.Series([c[0] for c in calls]).value_counts()
    assert counts[M1] == n_cells
    assert counts[M2] == n_cells
    assert counts[M3] == n_cells * n_lam
    assert len(set(calls)) == len(calls)     # no duplicated task key


def test_f05_zero_crossing_hand_computation():
    """Test 5: interpolation matches an arithmetic hand computation."""
    lams = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    # crosses between 1.00 (+0.02) and 1.15 (-0.02): exactly halfway -> 1.075
    vals = np.array([0.10, 0.06, 0.02, -0.02, -0.08])
    point, status = zero_crossing(lams, vals)
    assert status == FINITE
    assert point == pytest.approx(1.075)

    # crosses between 0.85 (+0.03) and 1.00 (-0.09): frac = 0.03/0.12 = 0.25
    vals2 = np.array([0.10, 0.03, -0.09, -0.20, -0.30])
    p2, s2 = zero_crossing(lams, vals2)
    assert s2 == FINITE
    assert p2 == pytest.approx(0.85 + 0.25 * 0.15)

    # an exact zero at a grid point is a crossing at that point
    vals3 = np.array([0.10, 0.05, 0.0, -0.05, -0.10])
    p3, s3 = zero_crossing(lams, vals3)
    assert s3 == FINITE
    assert p3 == pytest.approx(1.00)


def test_f06_censored_and_unresolved_states():
    """Test 6: a curve that never crosses inside the grid is censored, not extrapolated."""
    lams = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    assert zero_crossing(lams, np.array([0.2, 0.2, 0.1, 0.1, 0.05]))[1] == RIGHT_CENSORED
    assert zero_crossing(lams, np.array([-0.1, -0.2, -0.3, -0.4, -0.5]))[1] == LEFT_CENSORED
    assert np.isnan(zero_crossing(lams, np.array([0.2, 0.2, 0.1, 0.1, 0.05]))[0])
    # ascending-only sign change is not the estimand
    assert zero_crossing(lams, np.array([-0.2, -0.1, 0.1, 0.2, 0.3]))[1] == UNRESOLVED


def test_f07_cluster_bootstrap_boundary_ci():
    """Test 7: the boundary CI comes from resampling series clusters."""
    lams = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    rng = np.random.default_rng(0)
    n = 60
    base = np.array([0.10, 0.06, 0.02, -0.02, -0.08])
    curves = base + rng.normal(0, 0.01, size=(n, len(lams)))
    out = bootstrap_crossing(curves, lams, 500, 0.95, seed_parts=("t",))
    assert out["ci_low"] < 1.075 < out["ci_high"]
    assert out["ci_high"] - out["ci_low"] < 0.35
    # reproducible for the same seed parts, different for different ones
    again = bootstrap_crossing(curves, lams, 500, 0.95, seed_parts=("t",))
    other = bootstrap_crossing(curves, lams, 500, 0.95, seed_parts=("u",))
    assert (out["ci_low"], out["ci_high"]) == (again["ci_low"], again["ci_high"])
    assert (out["ci_low"], out["ci_high"]) != (other["ci_low"], other["ci_high"])


def test_f08_bootstrap_valid_fraction_is_recorded():
    """Test 8: the fraction of resamples with a usable crossing is reported."""
    lams = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    rng = np.random.default_rng(1)
    good = np.array([0.10, 0.06, 0.02, -0.02, -0.08]) + rng.normal(0, 0.005, size=(40, 5))
    out = bootstrap_crossing(good, lams, 400, 0.95, seed_parts=("t",))
    assert out["valid_fraction"] == pytest.approx(1.0)

    never = np.full((40, 5), 0.2) + rng.normal(0, 0.001, size=(40, 5))
    out2 = bootstrap_crossing(never, lams, 400, 0.95, seed_parts=("t",))
    assert out2["valid_fraction"] == 0.0
    assert np.isnan(out2["ci_low"])


def test_f09_followup_uses_the_existing_dgp_unchanged(small_boundary_cfg, pilot_cfg):
    """Test 9: with the same settings the follow-up path reproduces the pilot's series."""
    derived = small_boundary_cfg.to_pilot_config()
    # the inherited dgp block must be identical to the pilot's
    assert derived.dgp == pilot_cfg.dgp
    # and with the pilot's seed the generating code returns the pilot's series bit for bit
    d = derived.to_dict()
    d["experiment"]["master_seed"] = pilot_cfg.experiment.master_seed
    same_seed = PilotConfig.from_dict(d)
    a = generate_base_series(0, same_seed)
    b = generate_base_series(0, pilot_cfg)
    np.testing.assert_array_equal(a.b, b.b)
    np.testing.assert_array_equal(a.x, b.x)


def _fake_boundary_metrics(cfg, shift: float, slope: float = 0.35) -> pd.DataFrame:
    """Synthetic tasks whose curve crosses zero at ``shift``."""
    rng = np.random.default_rng(3)
    rows = []
    for b in cfg.base_series_ids:
        for share in cfg.grid.nominal_covariate_share:
            for h in cfg.grid.horizons:
                w1 = 0.5
                for lam in cfg.grid.lambda_values:
                    v = slope * (shift - lam) + 0.002 * rng.normal()
                    rows.append({"base_series_id": b, "nominal_covariate_share": float(share),
                                 "horizon": int(h), "lam": float(lam), "origin": 896,
                                 "wql_m1": w1, "wql_m2": w1 * 0.8, "wql_m3": w1 - v,
                                 "mse_m1": w1, "mse_m3": w1 - v,
                                 "nmae_m1": w1, "nmae_m3": w1 - v, "crossing_m3": 0.0,
                                 "v_future": v, "v_future_mse": v, "v_oracle": 0.2 * w1,
                                 "relative_delta_m3": -v / w1,
                                 "harm_m3": int(-v > 0.05 * w1), "m3_wins": int(v > 0),
                                 "realized_normalized_error_rms": lam})
    return pd.DataFrame(rows)


def test_f10_gate_e_pass_fail_and_inconclusive(small_boundary_cfg):
    """Test 10: Gate E returns each verdict on the matching synthetic fixture."""
    cfg = small_boundary_cfg
    coarse = {"low_lambda_v_future": 0.05, "high_lambda_v_future": -0.05}

    tm = _fake_boundary_metrics(cfg, shift=1.0)
    bounds = boundary_estimates(tm, cfg, "v_future", "chronos_wql")
    good = gate_e(tm, bounds, cfg, coarse)
    assert good["status"] == "PASS", good["checks"]
    assert good["checks"]["n_finite_crossings"] == len(bounds)

    # never crosses: still beneficial at every lambda -> right censored everywhere
    tm_no = _fake_boundary_metrics(cfg, shift=5.0)
    bounds_no = boundary_estimates(tm_no, cfg, "v_future", "chronos_wql")
    bad = gate_e(tm_no, bounds_no, cfg, coarse)
    assert bad["status"] == "FAIL"
    assert bad["checks"]["n_finite_crossings"] == 0

    # crossings exist but the interval is far too wide to satisfy the width rule
    tm_wide = _fake_boundary_metrics(cfg, shift=1.0, slope=0.004)
    bounds_wide = boundary_estimates(tm_wide, cfg, "v_future", "chronos_wql")
    mid = gate_e(tm_wide, bounds_wide, cfg, coarse)
    assert mid["status"] in {"INCONCLUSIVE", "FAIL"}
    if mid["status"] == "INCONCLUSIVE":
        assert mid["checks"]["n_narrow_crossings"] < cfg.gate_e.min_narrow_crossings


def test_f10b_curve_summary_shape(small_boundary_cfg):
    tm = _fake_boundary_metrics(small_boundary_cfg, shift=1.0)
    cells = curve_summary(tm, small_boundary_cfg)
    expected = (len(small_boundary_cfg.grid.nominal_covariate_share)
                * len(small_boundary_cfg.grid.horizons)
                * len(small_boundary_cfg.grid.lambda_values))
    assert len(cells) == expected
    assert (cells["n_series"] == small_boundary_cfg.grid.n_series_per_cell).all()


def test_f10c_boundary_config_rejects_bad_input(boundary_cfg):
    d = boundary_cfg.to_dict()
    d.pop("inherited_from_pilot_yaml")
    d["model"]["cross_learning"] = True
    from covariate_trust.config import BoundaryConfig
    with pytest.raises(ConfigError):
        BoundaryConfig.from_dict(d, boundary_cfg.inherited)
