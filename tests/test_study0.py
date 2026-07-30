"""Study 0 checks (tests 1-3 plus supporting cases)."""

from __future__ import annotations

import numpy as np
import pytest

from covariate_trust.config import ConfigError, Study0Config
from covariate_trust.study0 import (analytic_mse, evaluate_study0, posterior_weight, run_study0)


@pytest.fixture(scope="module")
def small_study0(study0_cfg):
    d = study0_cfg.to_dict()
    d["experiment"]["n_repetitions"] = 50_000
    return Study0Config.from_dict(d)


@pytest.fixture(scope="module")
def result(small_study0):
    return run_study0(small_study0)


def test_01_simulated_mse_matches_analytic(result):
    """Test 1: every simulated MSE agrees with the closed form."""
    _, summary, _ = result
    # 50k repetitions -> relative Monte-Carlo SE of roughly 0.6%; allow 2%
    assert summary["relative_error"].max() < 0.02, summary.loc[summary["relative_error"].idxmax()]


def test_02_prior_and_plugin_cross_at_lambda_one(small_study0):
    """Test 2: the prior-only and plug-in predictors cross at lambda = 1."""
    d = small_study0.dgp
    for lam, expect in ((0.5, "plugin_better"), (1.0, "equal"), (2.0, "prior_better")):
        a = analytic_mse(d.beta, d.prior_variance, d.target_noise_std, lam)
        if expect == "plugin_better":
            assert a["S1_plugin"] < a["S0_prior"]
        elif expect == "equal":
            assert a["S1_plugin"] == pytest.approx(a["S0_prior"])
        else:
            assert a["S1_plugin"] > a["S0_prior"]


def test_03_posterior_dominates_both_fixed_predictors(result):
    """Test 3: the exact posterior predictor is never worse than either fixed rule."""
    _, summary, verdict = result
    wide = summary.pivot(index="lam", columns="predictor", values="mse_simulated")
    for lam in wide.index:
        best_fixed = min(wide.loc[lam, "S0_prior"], wide.loc[lam, "S1_plugin"])
        assert wide.loc[lam, "S2_posterior"] <= best_fixed * 1.01
    assert next(c for c in verdict["checks"] if c["id"] == "C3_posterior_dominance")["status"] == "PASS"


def test_03b_posterior_equals_plugin_at_lambda_zero(result):
    _, summary, _ = result
    row = summary[(summary["lam"] == 0.0)].set_index("predictor")["mse_simulated"]
    assert row["S2_posterior"] == pytest.approx(row["S1_plugin"], abs=1e-12)


def test_03c_posterior_weight_formula():
    V = 1.0
    for lam in (0.0, 0.5, 1.0, 2.0):
        assert posterior_weight(V, lam**2 * V) == pytest.approx(1.0 / (1.0 + lam**2))


def test_03d_evaluate_detects_injected_error(result):
    """A corrupted MSE must be caught rather than silently tolerated."""
    _, summary, _ = result
    broken = summary.copy()
    broken.loc[broken.index[0], "mse_simulated"] *= 1.5
    broken["relative_error"] = (broken["mse_simulated"] - broken["mse_analytic"]).abs() / broken["mse_analytic"]
    assert evaluate_study0(broken)["status"] == "FAIL"


def test_03e_config_rejects_unknown_key(study0_cfg):
    d = study0_cfg.to_dict()
    d["experiment"]["typo"] = 1
    with pytest.raises(ConfigError):
        Study0Config.from_dict(d)


def test_03f_common_random_numbers_across_lambda(small_study0):
    """Same draws are reused for every lambda, so differences are paired."""
    raw, _, _ = run_study0(small_study0)
    prior = raw[(raw["predictor"] == "S0_prior")]
    per_lambda = prior.groupby("lam")["squared_error"].apply(lambda s: np.asarray(s)[:100])
    base = per_lambda.iloc[0]
    for arr in per_lambda:
        np.testing.assert_allclose(arr, base)
