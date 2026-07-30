"""Covariate forecast-error checks (tests 10-14 plus supporting cases)."""

from __future__ import annotations

import numpy as np
import pytest

from covariate_trust.dgp import (conditional_variance_path, conditional_variance_raw,
                                 covariate_vintage, estimate_lambda_hat, eta_path,
                                 generate_base_series)


def test_10_conditional_variance_formula(small_cfg):
    """Test 10: V(h) follows the AR(1) h-step formula and is increasing in h."""
    rho, sigma = small_cfg.dgp.covariate_ar, small_cfg.dgp.ar_innovation_std
    for h in (1, 2, 5, 24, 96):
        expected = sigma**2 * (1 - rho ** (2 * h)) / (1 - rho**2)
        assert conditional_variance_raw(rho, sigma, h) == pytest.approx(expected)
    v = [conditional_variance_raw(rho, sigma, h) for h in range(1, 40)]
    assert all(b > a for a, b in zip(v, v[1:]))
    assert conditional_variance_raw(rho, sigma, 1) == pytest.approx(sigma**2)
    assert conditional_variance_raw(rho, sigma, 500) == pytest.approx(sigma**2 / (1 - rho**2), rel=1e-6)


def test_10b_standardized_variance_uses_the_series_scale(small_cfg):
    s = generate_base_series(0, small_cfg)
    v = conditional_variance_path(small_cfg, s, 24)
    raw = np.array([conditional_variance_raw(small_cfg.dgp.covariate_ar,
                                             small_cfg.dgp.ar_innovation_std, h)
                    for h in range(1, 25)])
    np.testing.assert_allclose(v, raw / s.scale_x**2)


def test_11_lambda_zero_leaves_the_covariate_untouched(small_cfg):
    """Test 11: at lambda = 0 the vintage equals the true future path exactly."""
    s = generate_base_series(1, small_cfg)
    v = covariate_vintage(small_cfg, s, small_cfg.experiment.primary_origin, 24, 0.0)
    np.testing.assert_allclose(v["x_tilde"], v["x_true"], rtol=0, atol=0)
    assert v["standardized_rmse"] == 0.0


def test_12_eta_path_is_shared_across_lambda(small_cfg):
    """Test 12: every lambda at the same (series, origin, horizon) reuses one eta path."""
    s = generate_base_series(2, small_cfg)
    o, h = small_cfg.experiment.primary_origin, 24
    eta = eta_path(small_cfg, s.base_series_id, o, h)
    for lam in (0.5, 1.0, 2.0):
        v = covariate_vintage(small_cfg, s, o, h, lam)
        np.testing.assert_allclose(v["eta"], eta)
        np.testing.assert_allclose(v["error"], lam * np.sqrt(v["V"]) * eta)


def test_13_error_scales_proportionally_with_lambda(small_cfg):
    """Test 13: doubling lambda doubles the realized error path."""
    s = generate_base_series(3, small_cfg)
    o = small_cfg.experiment.primary_origin
    v1 = covariate_vintage(small_cfg, s, o, 96, 1.0)
    v2 = covariate_vintage(small_cfg, s, o, 96, 2.0)
    np.testing.assert_allclose(v2["error"], 2.0 * v1["error"], rtol=1e-12)
    assert v2["standardized_rmse"] == pytest.approx(2.0 * v1["standardized_rmse"])


def test_14_normalized_error_recovers_lambda(small_cfg):
    """Test 14: the realized normalized error RMS estimates lambda."""
    o = small_cfg.experiment.primary_origin
    for lam in (0.5, 1.0, 2.0):
        rms = []
        for b_id in range(20):
            s = generate_base_series(b_id, small_cfg)
            v = covariate_vintage(small_cfg, s, o, 96, lam)
            rms.append(v["realized_normalized_error_rms"])
            assert estimate_lambda_hat(v["error"], v["V"]) == pytest.approx(
                v["realized_normalized_error_rms"])
        assert np.mean(rms) == pytest.approx(lam, rel=0.15)


def test_14b_eta_paths_differ_across_origin_and_horizon(small_cfg):
    a = eta_path(small_cfg, 0, 896, 24)
    b = eta_path(small_cfg, 0, 872, 24)
    c = eta_path(small_cfg, 0, 896, 96)
    d = eta_path(small_cfg, 1, 896, 24)
    assert not np.allclose(a, b)
    assert not np.allclose(a, c[:24])
    assert not np.allclose(a, d)


def test_14c_error_is_unbiased_on_average(small_cfg):
    o = small_cfg.experiment.primary_origin
    biases = []
    for b_id in range(30):
        s = generate_base_series(b_id, small_cfg)
        biases.append(covariate_vintage(small_cfg, s, o, 96, 1.0)["error_bias"])
    assert abs(np.mean(biases)) < 0.15
