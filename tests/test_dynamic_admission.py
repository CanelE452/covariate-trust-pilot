"""Dynamic admission selector checks (follow-up tests 31-33)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.dynamic_admission import (D0, D1, D2, D3, D4, D5, D6, D7, SELECTORS,
                                               apply_selectors, build_proxy_table,
                                               condition_summary, false_rates, proxy_summary,
                                               run_dynamic_study)
from covariate_trust.reliability_schedules import P0_ORACLE, P1_CALIBRATED, PROXY_MODES
from covariate_trust.schemas import M1, M3


def fake_tasks(cfg, n_series: int = 4) -> pd.DataFrame:
    """Deterministic task table covering every schedule and cell."""
    rng = np.random.default_rng(7)
    rows = []
    for b in range(n_series):
        for share in cfg.grid.nominal_covariate_share:
            for h in cfg.grid.horizons:
                for s in cfg.schedules:
                    hist_lam = float(np.mean(s.historical))
                    cur = float(s.current)
                    w1 = 0.50 + 0.01 * rng.normal()
                    # M3 helps when the current lambda is low and hurts when it is high
                    w3 = w1 * (1.0 + 0.25 * (cur - 1.0)) + 0.003 * rng.normal()
                    rows.append({
                        "base_series_id": b, "nominal_covariate_share": float(share),
                        "horizon": int(h), "schedule": s.name,
                        "origin": cfg.experiment.primary_origin,
                        "true_current_lambda": cur,
                        "hist_wql_m1": w1,
                        "hist_wql_m3": w1 * (1.0 + 0.25 * (hist_lam - 1.0)),
                        "hist_lambda_hat": hist_lam,
                        "hist_lambda_hat_last": float(s.historical[-1]),
                        "hist_lambda_true_mean": hist_lam,
                        "wql_m1": w1, "wql_m3": w3,
                        "nmae_m1": w1, "nmae_m3": w3, "crossing_m3": 0.0,
                        "wql_oracle": min(w1, w3),
                        "m3_is_better": int(w3 < w1),
                        "harm_m3": int(w3 > 1.05 * w1),
                        "realized_normalized_error_rms": cur,
                    })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def decisions(small_dynamic_cfg):
    tasks = fake_tasks(small_dynamic_cfg)
    proxies = build_proxy_table(tasks, small_dynamic_cfg)
    return tasks, proxies, apply_selectors(tasks, proxies, small_dynamic_cfg)


def test_f31_history_selectors_ignore_the_current_outcome(small_dynamic_cfg, decisions):
    """Test 31: D3/D4 depend only on history, never on the current target outcome."""
    tasks, proxies, base = decisions
    poisoned = tasks.copy()
    poisoned["wql_m1"] = poisoned["wql_m1"] * 3.0      # change the present entirely
    poisoned["wql_m3"] = poisoned["wql_m3"] * 0.1
    other = apply_selectors(poisoned, proxies, small_dynamic_cfg)
    for sel in (D3, D4):
        a = base[base["selector"] == sel].sort_values(["base_series_id", "schedule",
                                                       "proxy_mode", "horizon",
                                                       "nominal_covariate_share"])["choice"]
        b = other[other["selector"] == sel].sort_values(["base_series_id", "schedule",
                                                         "proxy_mode", "horizon",
                                                         "nominal_covariate_share"])["choice"]
        assert list(a) == list(b), f"{sel} reacted to the current outcome"


def test_f32_proxy_selectors_ignore_the_current_outcome(small_dynamic_cfg, decisions):
    """Test 32: D5/D6/D7 use the reported proxy (and history), not the current target."""
    tasks, proxies, base = decisions
    poisoned = tasks.copy()
    poisoned["wql_m1"] = poisoned["wql_m1"] * 3.0
    poisoned["wql_m3"] = poisoned["wql_m3"] * 0.1
    other = apply_selectors(poisoned, proxies, small_dynamic_cfg)
    key = ["base_series_id", "schedule", "proxy_mode", "horizon", "nominal_covariate_share"]
    for sel in (D5, D6, D7):
        a = base[base["selector"] == sel].sort_values(key)["choice"]
        b = other[other["selector"] == sel].sort_values(key)["choice"]
        assert list(a) == list(b), f"{sel} reacted to the current outcome"
    # the oracle, by definition, does react
    a = base[base["selector"] == D2].sort_values(key)["choice"]
    b = other[other["selector"] == D2].sort_values(key)["choice"]
    assert list(a) != list(b)


def test_f32b_fixed_policies_are_constant(decisions):
    _, _, d = decisions
    assert (d[d["selector"] == D0]["choice"] == M1).all()
    assert (d[d["selector"] == D1]["choice"] == M3).all()


def test_f32c_threshold_rules_are_applied_as_declared(small_dynamic_cfg, decisions):
    _, _, d = decisions
    th = small_dynamic_cfg.selector_thresholds
    d5 = d[d["selector"] == D5]
    assert (d5.loc[d5["reported_lambda"] < th.use_threshold, "choice"] == M3).all()
    assert (d5.loc[d5["reported_lambda"] >= th.use_threshold, "choice"] == M1).all()
    d7 = d[d["selector"] == D7]
    assert (d7.loc[d7["reported_lambda"] < th.override_low, "choice"] == M3).all()
    assert (d7.loc[d7["reported_lambda"] > th.override_high, "choice"] == M1).all()
    mid = d7[(d7["reported_lambda"] >= th.override_low)
             & (d7["reported_lambda"] <= th.override_high)]
    expected = np.where(mid["hist_wql_m3"] < mid["hist_wql_m1"], M3, M1)
    assert list(mid["choice"]) == list(expected)


def test_f33_false_use_and_false_reject_hand_computation():
    """Test 33: conditional error rates match an arithmetic hand computation."""
    df = pd.DataFrame({
        "m3_is_better": [1, 1, 0, 0, 0],
        "choice": [M3, M1, M3, M1, M1],
    })
    df["false_use"] = ((df["m3_is_better"] == 0) & (df["choice"] == M3)).astype(int)
    df["false_reject"] = ((df["m3_is_better"] == 1) & (df["choice"] == M1)).astype(int)
    out = false_rates(df)
    # M1 was better in 3 rows, M3 was wrongly used in 1 of them -> 1/3
    assert out["false_use_rate"] == pytest.approx(1 / 3)
    # M3 was better in 2 rows, it was wrongly rejected in 1 -> 1/2
    assert out["false_reject_rate"] == pytest.approx(0.5)
    assert out["n_m1_better"] == 3 and out["n_m3_better"] == 2


def test_f33b_oracle_is_an_upper_bound(decisions):
    _, _, d = decisions
    for sel in SELECTORS:
        g = d[d["selector"] == sel]
        assert g["wql_selected"].mean() >= g["wql_oracle"].mean() - 1e-12
        assert (g["regret"] >= -1e-12).all()
    assert d[d["selector"] == D2]["regret"].abs().max() < 1e-12


def test_f33c_proxy_table_covers_every_task_once_per_mode(small_dynamic_cfg, decisions):
    tasks, proxies, d = decisions
    assert len(proxies) == len(tasks) * len(PROXY_MODES)
    assert len(d) == len(tasks) * len(PROXY_MODES) * len(SELECTORS)
    assert set(proxies["proxy_mode"]) == set(PROXY_MODES)


def test_f33d_summaries_keep_conditions_separate(decisions):
    _, _, d = decisions
    cond = condition_summary(d)
    n_sched = d["schedule"].nunique()
    assert len(cond) == n_sched * len(PROXY_MODES) * len(SELECTORS)
    prox = proxy_summary(d)
    assert len(prox) == len(PROXY_MODES) * len(SELECTORS)
    assert {"false_use_rate", "false_reject_rate", "mean_regret", "harm_rate"} <= set(cond.columns)


def test_f33e_run_dynamic_study_visits_every_origin(small_dynamic_cfg):
    """Each (schedule, cell, series) needs 4 historical origins plus the primary one."""
    seen = []

    def fake_predict(inputs, meta):
        seen.append((meta["method"], meta["origin"], meta.get("lam")))
        n = len(small_dynamic_cfg.experiment.quantile_levels)
        return np.tile(np.linspace(-1, 1, n), (inputs.horizon, 1))

    cfg = small_dynamic_cfg
    d = cfg.to_dict()
    d.pop("inherited_from_pilot_yaml")
    d["grid"]["n_series_per_condition"] = 2
    d["grid"]["horizons"] = [24]
    d["grid"]["nominal_covariate_share"] = [0.5]
    from covariate_trust.config import DynamicConfig
    tiny = DynamicConfig.from_dict(d, cfg.inherited)

    tasks = run_dynamic_study(tiny, fake_predict)
    assert len(tasks) == len(tiny.schedules) * 2
    origins = {o for _, o, _ in seen}
    assert origins == {800, 824, 848, 872, 896}
    # M1 is origin-only; M3 varies with the origin's lambda
    m1_keys = {(o, lam) for m, o, lam in seen if m == M1}
    assert all(lam == -1.0 for _, lam in m1_keys)
