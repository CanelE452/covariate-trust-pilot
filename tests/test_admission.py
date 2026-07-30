"""Admission and gate-guard checks (tests 35-38 plus supporting cases)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.admission import (A1, A2, PSEUDO_ORIGINS, assert_no_primary_leak,
                                       build_decisions, gate_d, historical_tasks, pseudo_origins)
from covariate_trust.config import PilotConfig
from covariate_trust.dgp import covariate_vintage, generate_base_series
from covariate_trust.gates import NOT_RUN, gate_a, gate_b, gate_c, gates_abc, cell_summary
from covariate_trust.schemas import M1, M3


def _fake_task_metrics(cfg: PilotConfig, m3_better_below: float = 1.0) -> pd.DataFrame:
    """Synthetic task metrics with a controllable benefit/harm boundary."""
    rows = []
    rng = np.random.default_rng(0)
    for b in cfg.base_series_ids:
        for share in cfg.grid.nominal_covariate_share:
            for h in cfg.grid.horizons:
                w1 = 0.50 + 0.01 * rng.normal()
                w2 = w1 * (1 - 0.12 * share)
                for lam in cfg.grid.lambda_values:
                    effect = 0.12 * share * (1 - lam / m3_better_below)
                    w3 = w1 * (1 - effect) + 0.002 * rng.normal()
                    rows.append({
                        "base_series_id": b, "nominal_covariate_share": float(share),
                        "horizon": int(h), "origin": cfg.experiment.primary_origin,
                        "lam": float(lam),
                        "wql_m0": w1 * 1.05, "wql_m1": w1, "wql_m2": w2, "wql_m3": w3,
                        "nmae_m0": w1, "nmae_m1": w1, "nmae_m2": w2, "nmae_m3": w3,
                        "mse_m0": w1, "mse_m1": w1, "mse_m2": w2, "mse_m3": w3,
                        "crossing_m1": 0.0, "crossing_m3": 0.0,
                        "v_future": w1 - w3, "v_oracle": w1 - w2,
                        "relative_delta_m3": (w3 - w1) / w1,
                        "harm_m3": int(w3 > 1.05 * w1), "m3_wins": int(w3 < w1),
                        "realized_normalized_error_rms": lam, "lambda_hat": lam,
                    })
    return pd.DataFrame(rows)


def test_35_selection_never_sees_the_primary_future_target(pilot_cfg):
    """Test 35: every pseudo-origin window closes at or before the primary origin."""
    primary = pilot_cfg.experiment.primary_origin
    for h in pilot_cfg.grid.horizons:
        for o in pseudo_origins(h, pilot_cfg):
            assert o + h <= primary


def test_36_pseudo_origin_boundaries(pilot_cfg):
    """Test 36: the pseudo-origin grids are exactly the ones specified."""
    assert PSEUDO_ORIGINS[24] == [800, 824, 848, 872]
    assert PSEUDO_ORIGINS[96] == [512, 608, 704, 800]
    with pytest.raises(AssertionError):
        assert_no_primary_leak([880], 24, pilot_cfg)   # 880 + 24 = 904 > 896
    with pytest.raises(ValueError):
        pseudo_origins(48, pilot_cfg)
    tasks = historical_tasks(pilot_cfg)
    expected = (len(pilot_cfg.grid.horizons) * 4 * pilot_cfg.grid.n_series_per_cell
                * len(pilot_cfg.grid.nominal_covariate_share))
    assert len(tasks) == expected


def test_37_historical_vintages_are_reproducible_and_distinct(small_cfg):
    """Test 37: historical vintages reproduce exactly and differ from the primary one."""
    s = generate_base_series(0, small_cfg)
    a = covariate_vintage(small_cfg, s, 800, 24, 1.0)
    b = covariate_vintage(small_cfg, s, 800, 24, 1.0)
    primary = covariate_vintage(small_cfg, s, small_cfg.experiment.primary_origin, 24, 1.0)
    np.testing.assert_array_equal(a["x_tilde"], b["x_tilde"])
    assert not np.allclose(a["eta"], primary["eta"])


def test_38_gate_guard_stops_downstream_gates_on_failure(small_cfg):
    """Test 38: Gate A FAIL leaves Gates B and C NOT_RUN."""
    tm = _fake_task_metrics(small_cfg)
    tm["wql_m2"] = tm["wql_m1"]                    # oracle covariate does nothing
    cells = cell_summary(tm, small_cfg)
    out = gates_abc(tm, cells, small_cfg)
    assert out["A"]["status"] == "FAIL"
    assert out["B"]["status"] == NOT_RUN
    assert out["C"]["status"] == NOT_RUN
    assert out["all_pass"] is False


def test_38b_gate_a_detects_a_real_oracle_gain(small_cfg):
    tm = _fake_task_metrics(small_cfg)
    out = gate_a(tm, small_cfg)
    assert out["status"] in {"PASS", "INCONCLUSIVE"}
    assert out["checks"]["aggregate_relative_improvement"] > 0
    assert out["checks"]["negative_control_relative_improvement"] == pytest.approx(0.0, abs=1e-9)


def test_38c_gate_b_finds_the_constructed_boundary(small_cfg):
    tm = _fake_task_metrics(small_cfg, m3_better_below=1.0)
    cells = cell_summary(tm, small_cfg)
    out = gate_b(tm, cells, small_cfg)
    assert out["checks"]["any_low_noise_benefit"]
    assert out["checks"]["any_high_noise_harm"]
    assert out["checks"]["fraction_curves_decreasing"] >= 0.7


def test_38d_gate_c_reports_headroom(small_cfg):
    tm = _fake_task_metrics(small_cfg)
    out = gate_c(tm, small_cfg)
    assert out["checks"]["oracle_headroom"] > 0
    assert out["means"]["best_fixed"] in {"always_no_future", "always_use_future"}


def test_38e_admission_selectors_and_gate_d(small_cfg):
    """Selectors are applied per task and Gate D reports on both of them."""
    tm = _fake_task_metrics(small_cfg)
    hist_rows = []
    for _, r in tm.iterrows():
        for origin in pseudo_origins(int(r["horizon"]), small_cfg):
            hist_rows.append({
                "base_series_id": r["base_series_id"],
                "nominal_covariate_share": r["nominal_covariate_share"],
                "horizon": int(r["horizon"]), "origin": origin, "lam": r["lam"],
                "wql_m1": r["wql_m1"], "wql_m3": r["wql_m3"],
                "crossing_m3": 0.0, "lambda_hat": r["lam"],
                "realized_normalized_error_rms": r["lam"],
            })
    decisions = build_decisions(pd.DataFrame(hist_rows), tm, small_cfg)
    assert set(decisions[f"choice_{A1}"].unique()) <= {M1, M3}
    assert (decisions.loc[decisions["lam"] < 1.0, f"choice_{A2}"] == M3).all()
    assert (decisions.loc[decisions["lam"] >= 1.0, f"choice_{A2}"] == M1).all()
    np.testing.assert_allclose(
        decisions["wql_oracle"], np.minimum(decisions["wql_m1"], decisions["wql_m3"]))

    verdict = gate_d(decisions, small_cfg)
    assert verdict["status"] in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert set(verdict["selectors"]) == {A1, A2}
    for r in verdict["selectors"].values():
        assert "oracle_gap_recovery" in r and "harm_rate" in r


def test_38f_admission_requires_complete_history(small_cfg):
    tm = _fake_task_metrics(small_cfg)
    hist = pd.DataFrame([{
        "base_series_id": 0, "nominal_covariate_share": 0.0, "horizon": 24, "origin": 800,
        "lam": 0.0, "wql_m1": 0.5, "wql_m3": 0.4, "crossing_m3": 0.0, "lambda_hat": 0.0,
        "realized_normalized_error_rms": 0.0}])
    with pytest.raises(RuntimeError):
        build_decisions(hist, tm, small_cfg)
