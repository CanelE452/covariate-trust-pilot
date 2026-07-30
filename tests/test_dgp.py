"""DGP checks (tests 4-9 plus supporting cases)."""

from __future__ import annotations

import numpy as np
import pytest

from covariate_trust.config import ConfigError, PilotConfig
from covariate_trust.dgp import (build_target, generate_base_series, generate_dataset,
                                 series_metadata)


def test_04_same_seed_reproduces_series(small_cfg):
    """Test 4: identical configuration and id reproduce identical series."""
    a = generate_base_series(3, small_cfg)
    b = generate_base_series(3, small_cfg)
    np.testing.assert_array_equal(a.b, b.b)
    np.testing.assert_array_equal(a.x, b.x)


def test_05_different_seed_changes_series(small_cfg):
    """Test 5: a different master seed or series id gives a different realization."""
    d = small_cfg.to_dict()
    d["experiment"]["master_seed"] = small_cfg.experiment.master_seed + 1
    other = PilotConfig.from_dict(d)
    a = generate_base_series(3, small_cfg)
    b = generate_base_series(3, other)
    c = generate_base_series(4, small_cfg)
    assert not np.allclose(a.b, b.b)
    assert not np.allclose(a.b, c.b)


def test_06_base_and_covariate_innovations_are_independent(small_cfg):
    """Test 6: the b and x innovation streams are drawn independently."""
    d = small_cfg.to_dict()
    d["grid"]["n_series_per_cell"] = 40
    cfg = PilotConfig.from_dict(d)
    corrs = []
    for b_id in cfg.base_series_ids:
        s = generate_base_series(b_id, cfg)
        corrs.append(np.corrcoef(s.b, s.x)[0, 1])
    assert abs(np.mean(corrs)) < 0.10, f"mean corr(b, x) = {np.mean(corrs):.4f}"
    # the seed namespaces themselves must differ
    s = generate_base_series(0, cfg)
    assert s.params["seed_innov_base"] != s.params["seed_innov_cov"]


def test_07_nominal_share_increases_covariate_dependence(small_cfg):
    """Test 7: corr(y, x) increases monotonically in the nominal share."""
    s = generate_base_series(1, small_cfg)
    corrs = []
    for r in small_cfg.grid.nominal_covariate_share:
        y = build_target(s, r)
        corrs.append(abs(np.corrcoef(y, s.x)[0, 1]))
    assert all(b > a for a, b in zip(corrs, corrs[1:])), corrs


def test_08_zero_share_gives_target_equal_to_base(small_cfg):
    """Test 8: at r = 0 the target is exactly the base process."""
    s = generate_base_series(2, small_cfg)
    y = build_target(s, 0.0)
    np.testing.assert_allclose(y, s.b, rtol=0, atol=1e-12)
    meta = series_metadata(s, y, 0.0, small_cfg)
    assert abs(meta["full_corr_x_y"]) < 0.30
    assert meta["realized_incremental_r2"] < 0.02


def test_09_standardization_uses_the_early_window_only(small_cfg):
    """Test 9: standardization statistics come from [0, standardization_end) only."""
    w = small_cfg.experiment.standardization_end
    s = generate_base_series(0, small_cfg)
    assert s.b[:w].mean() == pytest.approx(0.0, abs=1e-12)
    assert s.b[:w].std(ddof=0) == pytest.approx(1.0, abs=1e-12)
    assert s.x[:w].mean() == pytest.approx(0.0, abs=1e-12)
    assert s.x[:w].std(ddof=0) == pytest.approx(1.0, abs=1e-12)
    # the post-window part is generally not standardized, which is the point
    assert abs(s.b[w:].mean()) > 1e-9 or abs(s.b[w:].std() - 1.0) > 1e-9


def test_09b_variance_share_matches_nominal_share(small_cfg):
    """y = sqrt(1-r) b + sqrt(r) x, so on the early window Var(y) stays near 1."""
    s = generate_base_series(0, small_cfg)
    w = small_cfg.experiment.standardization_end
    for r in small_cfg.grid.nominal_covariate_share:
        y = build_target(s, r)
        assert y[:w].var() == pytest.approx(1.0, abs=0.25)


def test_09c_dataset_covers_every_series_and_share(small_cfg):
    series_df, meta_df, series_map = generate_dataset(small_cfg)
    n = small_cfg.grid.n_series_per_cell * len(small_cfg.grid.nominal_covariate_share)
    assert len(meta_df) == n
    assert len(series_df) == n * small_cfg.experiment.series_length
    assert set(series_map) == set(small_cfg.base_series_ids)
    assert {"realized_incremental_r2", "context_corr_x_y", "regression_residual_variance"} <= set(meta_df.columns)


def test_09d_config_rejects_leaky_standardization(small_cfg):
    d = small_cfg.to_dict()
    d["experiment"]["standardization_end"] = d["experiment"]["primary_origin"] + 1
    with pytest.raises(ConfigError):
        PilotConfig.from_dict(d)


def test_09e_config_rejects_horizon_beyond_series(small_cfg):
    d = small_cfg.to_dict()
    d["grid"]["horizons"] = [24, 512]
    with pytest.raises(ConfigError):
        PilotConfig.from_dict(d)


def test_09f_config_rejects_cross_learning(small_cfg):
    d = small_cfg.to_dict()
    d["model"]["cross_learning"] = True
    with pytest.raises(ConfigError):
        PilotConfig.from_dict(d)
