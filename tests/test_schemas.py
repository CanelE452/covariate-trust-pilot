"""Model-input schema and fairness checks (tests 15-20 plus supporting cases)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.dgp import build_target, covariate_vintage, generate_base_series
from covariate_trust.schemas import (COVARIATE_COLUMN, M0, M1, M2, M3, SchemaError,
                                     TARGET_COLUMN, assert_context_equality,
                                     assert_future_equality, assert_no_future_target_leak,
                                     assert_timestamp_continuity, build_inputs)


@pytest.fixture(scope="module")
def task(small_cfg):
    cfg = small_cfg
    s = generate_base_series(0, cfg)
    y = build_target(s, 0.5)
    o, h = cfg.experiment.primary_origin, 24
    c = cfg.experiment.context_length
    f = cfg.experiment.frequency
    v0 = covariate_vintage(cfg, s, o, h, 0.0)
    v2 = covariate_vintage(cfg, s, o, h, 2.0)
    return {
        "cfg": cfg,
        M0: build_inputs(M0, "t", y, s.x, o, h, c, f),
        M1: build_inputs(M1, "t", y, s.x, o, h, c, f),
        M2: build_inputs(M2, "t", y, s.x, o, h, c, f, x_future=v0["x_true"]),
        "M3_lam0": build_inputs(M3, "t", y, s.x, o, h, c, f, x_future=v0["x_tilde"]),
        "M3_lam2": build_inputs(M3, "t", y, s.x, o, h, c, f, x_future=v2["x_tilde"]),
    }


def test_15_m0_has_no_covariate_column(task):
    """Test 15: M0 never sees the covariate."""
    assert COVARIATE_COLUMN not in task[M0].context_df.columns
    assert task[M0].future_df is None


def test_16_m1_has_no_future_frame(task):
    """Test 16: M1 receives past covariates only."""
    assert task[M1].future_df is None
    assert COVARIATE_COLUMN in task[M1].context_df.columns


def test_17_m1_and_m3_contexts_are_identical(task):
    """Test 17: the only difference between M1 and M3 is the future frame."""
    assert_context_equality(task[M1], task["M3_lam2"])
    assert task[M1].context_df.equals(task["M3_lam0"].context_df)


def test_18_future_frame_never_contains_the_target(task):
    """Test 18: no future_df may carry target information."""
    for key in (M2, "M3_lam0", "M3_lam2"):
        fut = task[key].future_df
        assert TARGET_COLUMN not in fut.columns
        assert_no_future_target_leak(fut)


def test_19_m2_equals_lambda_zero_m3(task):
    """Test 19: an oracle future covariate is exactly a lambda = 0 forecast."""
    assert_future_equality(task[M2], task["M3_lam0"])
    with pytest.raises(SchemaError):
        assert_future_equality(task[M2], task["M3_lam2"])


def test_20_timestamps_are_continuous_and_aligned(task):
    """Test 20: context and future timestamps are regular and adjacent."""
    ctx = task[M2].context_df["timestamp"]
    fut = task[M2].future_df["timestamp"]
    assert_timestamp_continuity(ctx)
    assert_timestamp_continuity(fut)
    step = ctx.iloc[1] - ctx.iloc[0]
    assert fut.iloc[0] - ctx.iloc[-1] == step
    assert len(ctx) == task[M2].context_df.shape[0] == task["cfg"].experiment.context_length
    with pytest.raises(SchemaError):
        assert_timestamp_continuity(pd.Series(pd.to_datetime(["2020-01-01", "2020-01-03",
                                                             "2020-01-04"])))


def test_20b_context_window_length_and_position(task):
    cfg = task["cfg"]
    ctx = task[M1].context_df
    assert len(ctx) == cfg.experiment.context_length
    assert np.isfinite(ctx["target"]).all()


def test_20c_wrong_future_length_is_rejected(small_cfg):
    s = generate_base_series(0, small_cfg)
    y = build_target(s, 0.5)
    with pytest.raises(SchemaError):
        build_inputs(M3, "t", y, s.x, small_cfg.experiment.primary_origin, 24,
                     small_cfg.experiment.context_length, small_cfg.experiment.frequency,
                     x_future=np.zeros(23))


def test_20d_future_path_rejected_for_m0_and_m1(small_cfg):
    s = generate_base_series(0, small_cfg)
    y = build_target(s, 0.5)
    for method in (M0, M1):
        with pytest.raises(SchemaError):
            build_inputs(method, "t", y, s.x, small_cfg.experiment.primary_origin, 24,
                         small_cfg.experiment.context_length, small_cfg.experiment.frequency,
                         x_future=np.zeros(24))
