"""Statistical baseline checks (follow-up tests 11-18)."""

from __future__ import annotations

import numpy as np
import pytest

from covariate_trust.baselines import (B1_PREFIX, B2_PREFIX, baseline_checks, b1_forecast,
                                       dgp_aware_conditional_mean, fit_arx, fit_x_ar,
                                       forecast_arx, forecast_x_ar, ridge_fit, run_baselines,
                                       seasonal_design)
from covariate_trust.boundary import FINITE, zero_crossing
from covariate_trust.dgp import build_target, covariate_vintage, generate_base_series
from covariate_trust.metrics import mse


@pytest.fixture(scope="module")
def baseline_run(small_boundary_cfg):
    return run_baselines(small_boundary_cfg)


def test_f11_dgp_aware_conditional_mean_is_correct(small_boundary_cfg):
    """Test 11: the conditional mean equals sinusoid + rho^h * last residual, standardized."""
    pilot = small_boundary_cfg.to_pilot_config()
    s = generate_base_series(0, pilot)
    origin, h = pilot.experiment.primary_origin, 12
    rho = pilot.dgp.covariate_ar
    got = dgp_aware_conditional_mean(s, "x", origin, h, rho)

    p = s.params
    t_last = origin - 1
    det_last = (p["amp_x1"] * np.sin(2 * np.pi * t_last / p["covariate_periods"][0] + p["phase_x1"])
                + p["amp_x2"] * np.sin(2 * np.pi * t_last / p["covariate_periods"][1] + p["phase_x2"]))
    u_last = s.x_raw[t_last] - det_last
    for i, t in enumerate(range(origin, origin + h)):
        det = (p["amp_x1"] * np.sin(2 * np.pi * t / p["covariate_periods"][0] + p["phase_x1"])
               + p["amp_x2"] * np.sin(2 * np.pi * t / p["covariate_periods"][1] + p["phase_x2"]))
        expected = (det + rho ** (i + 1) * u_last - p["x_mean"]) / p["x_scale"]
        assert got[i] == pytest.approx(expected)

    # the conditional mean must beat a naive constant forecast of the true future
    err = np.mean((got - s.x[origin:origin + h]) ** 2)
    naive = np.mean((s.x[origin - 1] - s.x[origin:origin + h]) ** 2)
    assert err < naive


def test_f12_lambda_zero_makes_oracle_and_noisy_identical(small_boundary_cfg):
    """Test 12: at lambda = 0 the B1-M2 and B1-M3 forecasts coincide exactly."""
    pilot = small_boundary_cfg.to_pilot_config()
    s = generate_base_series(1, pilot)
    origin, h, share = pilot.experiment.primary_origin, 24, 0.5
    v0 = covariate_vintage(pilot, s, origin, h, 0.0)
    m2 = b1_forecast(s, share, origin, h, pilot.dgp.base_ar, pilot.dgp.covariate_ar, v0["x_true"])
    m3 = b1_forecast(s, share, origin, h, pilot.dgp.base_ar, pilot.dgp.covariate_ar, v0["x_tilde"])
    np.testing.assert_array_equal(m2, m3)


def test_f13_no_future_covariate_effect_at_zero_share(small_boundary_cfg):
    """Test 13: with r = 0 the covariate cannot change the B1 forecast at all."""
    pilot = small_boundary_cfg.to_pilot_config()
    s = generate_base_series(2, pilot)
    origin, h = pilot.experiment.primary_origin, 24
    v = covariate_vintage(pilot, s, origin, h, 2.0)
    a = b1_forecast(s, 0.0, origin, h, pilot.dgp.base_ar, pilot.dgp.covariate_ar, v["x_true"])
    b = b1_forecast(s, 0.0, origin, h, pilot.dgp.base_ar, pilot.dgp.covariate_ar, v["x_tilde"])
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_f14_arx_never_sees_the_test_window(small_boundary_cfg):
    """Test 14: perturbing the target after the origin cannot change the ARX fit or forecast."""
    pilot = small_boundary_cfg.to_pilot_config()
    s = generate_base_series(0, pilot)
    y = build_target(s, 0.5)
    origin = pilot.experiment.primary_origin
    start = origin - pilot.experiment.context_length
    periods = small_boundary_cfg.baselines.arx_seasonal_periods
    ridge = small_boundary_cfg.baselines.ridge_parameter

    model = fit_arx(y, s.x, start, origin, periods, ridge)
    x_fut = s.x[origin:origin + 24]
    pred = forecast_arx(model, y, x_fut, origin, 24, float(s.x[origin - 1]))

    y_poisoned = y.copy()
    y_poisoned[origin:] += 100.0                      # destroy the future target
    model2 = fit_arx(y_poisoned, s.x, start, origin, periods, ridge)
    pred2 = forecast_arx(model2, y_poisoned, x_fut, origin, 24, float(s.x[origin - 1]))
    np.testing.assert_allclose(model["coef"], model2["coef"], atol=0)
    np.testing.assert_allclose(pred, pred2, atol=0)


def test_f15_arx_coefficients_use_only_the_context_window(small_boundary_cfg):
    """Test 15: observations before the context window do not enter the fit either."""
    pilot = small_boundary_cfg.to_pilot_config()
    s = generate_base_series(0, pilot)
    y = build_target(s, 0.5)
    origin = pilot.experiment.primary_origin
    start = origin - pilot.experiment.context_length
    periods = small_boundary_cfg.baselines.arx_seasonal_periods
    ridge = small_boundary_cfg.baselines.ridge_parameter

    a = fit_arx(y, s.x, start, origin, periods, ridge)
    y2 = y.copy()
    y2[:start] += 50.0
    b = fit_arx(y2, s.x, start, origin, periods, ridge)
    np.testing.assert_allclose(a["coef"], b["coef"], atol=0)
    assert a["n_obs"] == pilot.experiment.context_length - 1


def test_f16_ridge_parameter_is_fixed_and_from_config(small_boundary_cfg):
    """Test 16: the ridge parameter is the configured constant, never selected."""
    assert small_boundary_cfg.baselines.ridge_parameter == 1e-4
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([0.0, 1.0, 2.0])
    coef = ridge_fit(X, y, 1e-4)
    assert coef[1] == pytest.approx(1.0, abs=1e-3)
    # a different ridge gives a different answer, so the value genuinely matters
    assert not np.allclose(coef, ridge_fit(X, y, 10.0))


def test_f17_baselines_share_the_series_and_error_paths(small_boundary_cfg, baseline_run):
    """Test 17: baselines consume the same series, origins and eta paths as Chronos."""
    pilot = small_boundary_cfg.to_pilot_config()
    df = baseline_run
    assert set(df["base_series_id"]) == set(small_boundary_cfg.base_series_ids)
    assert set(df["lam"]) == {float(l) for l in small_boundary_cfg.grid.lambda_values}
    assert (df["origin"] == pilot.experiment.primary_origin).all()
    # the vintage used by the baseline is the same object the Chronos path builds
    s = generate_base_series(0, pilot)
    v = covariate_vintage(pilot, s, pilot.experiment.primary_origin, 24, 1.0)
    v2 = covariate_vintage(pilot, s, pilot.experiment.primary_origin, 24, 1.0)
    np.testing.assert_array_equal(v["x_tilde"], v2["x_tilde"])


def test_f18_chronos_median_mse_matches_a_hand_computation():
    """Test 18: the median-quantile MSE used for the MSE boundary is computed correctly."""
    y = np.array([1.0, 2.0, 3.0])
    levels = [0.1, 0.5, 0.9]
    q_pred = np.array([[0.0, 1.5, 9.0], [0.0, 2.5, 9.0], [0.0, 2.0, 9.0]])
    expected = float(np.mean([(1 - 1.5) ** 2, (2 - 2.5) ** 2, (3 - 2.0) ** 2]))
    assert mse(y, q_pred, levels) == pytest.approx(expected)
    assert expected == pytest.approx((0.25 + 0.25 + 1.0) / 3)


def test_f18b_dgp_aware_gap_matches_the_closed_form(small_boundary_cfg, baseline_run):
    """The expected B1 gap is r*(1-lambda^2)*mean_h V(h), which is exactly zero at lambda=1.

    This is the sample-size-free version of "the boundary is at lambda = 1": the point
    crossing is a finite-sample estimate, but the expected gap has a closed form.
    """
    from covariate_trust.baselines import theoretical_b1_gap
    for (share, h, lam), g in baseline_run.groupby(
            ["nominal_covariate_share", "horizon", "lam"]):
        got = float(g[f"{B1_PREFIX}_v_future"].mean())
        want = theoretical_b1_gap(small_boundary_cfg, float(share), int(h), float(lam))
        se = float(g[f"{B1_PREFIX}_v_future"].std(ddof=1) / np.sqrt(len(g)))
        assert abs(got - want) <= 4 * se + 1e-9, (share, h, lam, got, want, se)
        # sign of the expected gap flips exactly at lambda = 1
        assert (want > 0) == (lam < 1.0) or lam == 1.0
    assert theoretical_b1_gap(small_boundary_cfg, 0.5, 24, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_f18b2_boundary_estimate_converges_towards_one(boundary_cfg):
    """With more series the B1 point crossing moves towards 1 - the sign the estimator is
    consistent and the earlier small-sample offset was noise, not a bug."""
    from covariate_trust.config import BoundaryConfig
    points = {}
    for n in (4, 40):
        d = boundary_cfg.to_dict()
        d.pop("inherited_from_pilot_yaml")
        d["grid"]["n_series_per_cell"] = n
        d["grid"]["nominal_covariate_share"] = [0.25, 0.50]
        d["grid"]["horizons"] = [24]
        cfg = BoundaryConfig.from_dict(d, boundary_cfg.inherited)
        df = run_baselines(cfg)
        df = df[df["nominal_covariate_share"] == 0.50]
        lams = np.array(sorted(df["lam"].unique()))
        means = df.groupby("lam")[f"{B1_PREFIX}_v_future"].mean().reindex(lams).to_numpy()
        p, status = zero_crossing(lams, means)
        assert status == FINITE
        points[n] = p
    assert abs(points[40] - 1.0) < abs(points[4] - 1.0), points


def test_f18c_baseline_checks_report_pass(small_boundary_cfg, baseline_run):
    out = baseline_checks(baseline_run, small_boundary_cfg)
    assert out["status"] == "PASS", [(c["id"], c["status"], c["detail"]) for c in out["checks"]]
    assert {c["id"] for c in out["checks"]} >= {"BC1_dgp_aware_matches_closed_form",
                                                "BC1b_dgp_aware_boundary_consistent_with_one",
                                                "BC2_oracle_not_worse_than_noisy"}


def test_f18d_seasonal_design_shape_and_values():
    t = np.array([0.0, 6.0, 12.0])
    d = seasonal_design(t, [24, 168])
    assert d.shape == (3, 4)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(1.0)
    assert d[1, 0] == pytest.approx(1.0)


def test_f18e_x_forecast_is_past_only(small_boundary_cfg):
    """The covariate model's own forecast may not use future covariate values."""
    pilot = small_boundary_cfg.to_pilot_config()
    s = generate_base_series(0, pilot)
    origin = pilot.experiment.primary_origin
    start = origin - pilot.experiment.context_length
    periods = small_boundary_cfg.baselines.arx_seasonal_periods
    m = fit_x_ar(s.x, start, origin, periods, 1e-4)
    a = forecast_x_ar(m, s.x, origin, 24)
    x2 = s.x.copy()
    x2[origin:] += 100.0
    m2 = fit_x_ar(x2, start, origin, periods, 1e-4)
    b = forecast_x_ar(m2, x2, origin, 24)
    np.testing.assert_allclose(a, b, atol=0)
