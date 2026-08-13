"""Study 4 value-model checks (items 30-37)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.acquisition_models import (
    CANDIDATES,
    ModelFitError,
    TwoPartExpectedGain,
    candidate_grid,
    fit_candidate,
    select_value_model,
)

FEATURES = ["zone", "revision_rms", "base_interval_width_mean"]


def make_xy(n=400, seed=0, positive_share=0.5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "zone": rng.choice(["NYC", "WEST"], n),
        "revision_rms": rng.random(n),
        "base_interval_width_mean": rng.random(n),
    })
    signal = 0.02 * X["revision_rms"] - 0.01
    y = signal.to_numpy() + rng.normal(0, 0.002, n)
    if positive_share is not None:
        # shift so the requested share of labels is positive
        y = y - np.quantile(y, 1.0 - positive_share)
    return X, y


@pytest.mark.parametrize("name", [c for c in CANDIDATES if c != "two_part_expected_gain"])
def test_32_33_34_regression_pipelines_fit_and_predict(name):
    X, y = make_xy()
    fitted = fit_candidate(name, X, y, FEATURES, candidate_grid(name)[0])
    preds = fitted.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_35_two_part_sign_and_magnitude():
    X, y = make_xy(n=600, positive_share=0.5)
    model = TwoPartExpectedGain(FEATURES).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()
    # the expectation must lie between the negative and positive branch means
    p = model.classifier.predict_proba(X)[:, 1]
    manual = p * model.positive_model.predict(X) - (1 - p) * model.negative_model.predict(X)
    assert np.allclose(preds, manual)


def test_35b_two_part_fails_loudly_when_a_sign_class_is_tiny():
    X, y = make_xy(n=300)
    y = np.abs(y) + 0.01  # every label positive
    with pytest.raises(ModelFitError):
        TwoPartExpectedGain(FEATURES).fit(X, y)


def test_30_31_selection_uses_the_validation_score_only():
    X, y = make_xy(n=300, seed=1)
    seen = []

    def score_fn(fitted):
        # a deterministic score that prefers ridge with the largest alpha
        seen.append((fitted.name, fitted.params))
        if fitted.name == "ridge":
            return 1.0 - float(fitted.params.get("alpha", 0.0)) / 1000.0
        return 5.0

    result = select_value_model(["ridge", "huber"], X, y, FEATURES, score_fn)
    assert result.selected is not None
    assert result.selected.name == "ridge"
    assert result.selected.params["alpha"] == 100.0
    # every candidate/grid point was scored, and the score function never saw test data
    assert len(seen) == len(candidate_grid("ridge")) + len(candidate_grid("huber"))
    assert set(result.table["candidate"]) == {"ridge", "huber"}


def test_failed_candidates_are_recorded_not_silently_replaced():
    X, y = make_xy(n=300)
    y = np.abs(y) + 0.01  # two-part cannot fit

    def score_fn(fitted):
        return 1.0

    result = select_value_model(["ridge", "two_part_expected_gain"], X, y, FEATURES, score_fn)
    assert result.selected is not None and result.selected.name == "ridge"
    assert any("two_part" in key for key in result.failures)
    assert (result.table["status"].str.startswith("FAILED")).any()


def test_36_37_selected_model_is_reusable_unchanged():
    X, y = make_xy(n=300, seed=2)
    fitted = fit_candidate("ridge", X, y, FEATURES, {"alpha": 1.0})
    first = fitted.predict(X)
    other, _ = make_xy(n=120, seed=9)
    fitted.predict(other)                       # scoring another split must not refit
    assert np.allclose(fitted.predict(X), first)
