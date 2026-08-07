"""Study 4 portfolio and budget checks (items 6, 21-29, 38-39, 43-44)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust import portfolio_selection as ps

ZONES = ["CAPITAL", "LONG_ISLAND", "NYC", "WEST"]


def make_frame(n_days=5, seed=0, drop=None):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2025-07-01 07:00", periods=n_days, freq="D"):
        for z in ZONES:
            if drop and (d.normalize(), z) in drop:
                continue
            base = 0.05 + rng.random() * 0.02
            rows.append({
                "zone": z, "origin_utc": d,
                "wql_m1": base, "wql_m3": base - rng.normal(0.002, 0.004),
                "q90_m1": base * 2, "q90_m3": base * 2 - rng.normal(0.002, 0.004),
                "base_interval_width_mean": rng.random(),
                "revision_rms": rng.random(),
                "recent_base_wql_28": rng.random(),
                "reported_reliability_ratio": rng.random(),
            })
    frame = pd.DataFrame(rows)
    frame["v_wql"] = frame["wql_m1"] - frame["wql_m3"]
    frame["v_q90"] = frame["q90_m1"] - frame["q90_m3"]
    return frame


def test_06_21_only_complete_four_zone_days_are_kept():
    drop = {(pd.Timestamp("2025-07-03"), "NYC")}
    built = ps.build_portfolios(make_frame(5, drop=drop), ZONES)
    assert built.n_days == 4
    assert len(built.excluded) == 1
    assert built.excluded.iloc[0]["reason"] == "missing_zone"
    assert built.frame.groupby(ps.DATE_COLUMN)["zone"].nunique().eq(4).all()


def test_22_23_28_budget_and_abstention():
    scores = np.array([0.5, -0.2, 0.3, -0.1])
    tie = np.arange(4)
    assert top_sum(ps.top_k_mask(scores, 1, False, tie)) == 1
    assert top_sum(ps.top_k_mask(scores, 2, False, tie)) == 2
    # abstention: only the two positive scores can take a slot
    assert top_sum(ps.top_k_mask(scores, 3, True, tie)) == 2
    assert top_sum(ps.top_k_mask(np.array([-1.0, -2.0]), 2, True, np.arange(2))) == 0
    assert top_sum(ps.top_k_mask(scores, 0, False, tie)) == 0


def top_sum(mask):
    return int(np.asarray(mask).sum())


def test_24_tie_break_is_deterministic():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    tie = np.array([3, 1, 0, 2])
    a = ps.top_k_mask(scores, 2, False, tie)
    b = ps.top_k_mask(scores, 2, False, tie)
    assert np.array_equal(a, b)
    # tie_order 0 and 1 sit at positions 2 and 1, so those are the chosen entries
    # (flatnonzero always reports positions in ascending order)
    assert set(np.flatnonzero(a).tolist()) == {1, 2}


def test_25_oracle_picks_the_actual_best_zones():
    built = ps.build_portfolios(make_frame(6, seed=3), ZONES)
    daily = ps.evaluate_policy(built, ps.ORACLE, 1, objective="wql")
    for (_, day), (_, row) in zip(built.frame.groupby(ps.DATE_COLUMN), daily.iterrows()):
        day = day.sort_values("zone")
        best = day.loc[day["v_wql"].idxmax()]
        if best["v_wql"] > 0:
            assert row["selected_zones"] == best["zone"]
        else:
            assert row["selected_zones"] == ""      # abstains when nothing helps


def test_26_random_policy_is_reproducible():
    built = ps.build_portfolios(make_frame(8, seed=4), ZONES)
    a, sa = ps.random_policy_distribution(built, 1, 50, seed=11)
    b, sb = ps.random_policy_distribution(built, 1, 50, seed=11)
    c, sc = ps.random_policy_distribution(built, 1, 50, seed=12)
    pd.testing.assert_frame_equal(a, b)
    assert sa["sd_of_repetition_means"] > 0
    assert not np.allclose(a["loss"], c["loss"])


def test_27_round_robin_respects_the_budget_and_ignores_forecasts():
    built = ps.build_portfolios(make_frame(8, seed=5), ZONES)
    daily = ps.evaluate_policy(built, ps.ROUND_ROBIN, 1)
    assert (daily["n_selected"] == 1).all()
    picks = [s for s in daily["selected_zones"]]
    assert len(set(picks)) > 1       # it must rotate, not sit on one zone


def test_29_budgets_are_computed_independently():
    built = ps.build_portfolios(make_frame(6, seed=6), ZONES)
    k1 = ps.evaluate_policy(built, ps.ORACLE, 1)
    k2 = ps.evaluate_policy(built, ps.ORACLE, 2)
    assert (k1["n_selected"] <= 1).all()
    assert (k2["n_selected"] <= 2).all()
    assert (k2["loss"] <= k1["loss"] + 1e-12).all()   # more budget cannot hurt the oracle


def test_38_39_oracle_recovery_hand_calculation():
    base = np.array([1.0, 1.0, 1.0])
    policy = np.array([0.9, 1.0, 0.8])
    oracle = np.array([0.8, 1.0, 0.6])          # the middle day has zero headroom
    out = ps.oracle_recovery(base, policy, oracle)
    assert out["excluded_fraction"] == pytest.approx(1 / 3)
    assert out["daily_mean_recovery"] == pytest.approx((0.5 + 0.5) / 2)
    assert out["aggregate_recovery"] == pytest.approx((1.0 - 0.9) / (1.0 - 0.8))


def test_43_selection_overlap():
    a = pd.DataFrame({ps.DATE_COLUMN: [1, 2], "k": [1, 1], "selected_zones": ["NYC", "WEST"]})
    b = pd.DataFrame({ps.DATE_COLUMN: [1, 2], "k": [1, 1], "selected_zones": ["NYC", "NYC"]})
    assert ps.selection_overlap(a, b) == pytest.approx(0.5)
    assert ps.selection_overlap(a, a) == pytest.approx(1.0)


def test_44_zone_selection_rates():
    daily = pd.DataFrame({"selected_zones": ["NYC", "NYC,WEST", ""]})
    rates = ps.zone_selection_rates(daily, tuple(ZONES))
    assert rates["NYC"] == pytest.approx(2 / 3)
    assert rates["WEST"] == pytest.approx(1 / 3)
    assert rates["CAPITAL"] == 0.0


def test_no_premium_and_all_premium_bounds():
    built = ps.build_portfolios(make_frame(6, seed=7), ZONES)
    none = ps.evaluate_policy(built, ps.NO_PREMIUM, 1)
    every = ps.evaluate_policy(built, ps.ALL_PREMIUM, 1)
    assert (none["n_selected"] == 0).all()
    assert (every["n_selected"] == 4).all()


def test_empty_split_reports_zero_days():
    empty = ps.PortfolioSet(pd.DataFrame(), pd.DataFrame(), tuple(ZONES))
    assert empty.n_days == 0
