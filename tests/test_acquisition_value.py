"""Study 4 premium-value label checks (items 16-20)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.acquisition_value import (
    build_value_labels,
    compute_q90_losses,
    single_quantile_loss,
    value_distribution_summary,
)
from covariate_trust.metrics import EPSILON, pinball, wql


def test_16_wql_premium_value_by_hand():
    task = pd.DataFrame({"zone": ["NYC"], "origin_utc": pd.to_datetime(["2025-01-10 07:00"]),
                         "wql_m1": [0.10], "wql_m3": [0.06],
                         "nmae_m1": [0.1], "nmae_m3": [0.09]})
    q90 = pd.DataFrame({"zone": ["NYC"], "origin_utc": pd.to_datetime(["2025-01-10 07:00"]),
                        "q90_m1": [0.20], "q90_m3": [0.25]})
    labels = build_value_labels(task, q90)
    assert labels["v_wql"].iloc[0] == pytest.approx(0.04)
    assert labels["premium_positive"].iloc[0] == 1


def test_17_q90_premium_value_by_hand():
    task = pd.DataFrame({"zone": ["NYC"], "origin_utc": pd.to_datetime(["2025-01-10 07:00"]),
                         "wql_m1": [0.1], "wql_m3": [0.1], "nmae_m1": [0.1], "nmae_m3": [0.1]})
    q90 = pd.DataFrame({"zone": ["NYC"], "origin_utc": pd.to_datetime(["2025-01-10 07:00"]),
                        "q90_m1": [0.30], "q90_m3": [0.18]})
    labels = build_value_labels(task, q90)
    assert labels["v_q90"].iloc[0] == pytest.approx(0.12)


def test_18_both_signs_are_kept():
    task = pd.DataFrame({"zone": ["NYC", "WEST"],
                         "origin_utc": pd.to_datetime(["2025-01-10 07:00"] * 2),
                         "wql_m1": [0.10, 0.05], "wql_m3": [0.06, 0.09],
                         "nmae_m1": [0.1, 0.1], "nmae_m3": [0.1, 0.1]})
    q90 = pd.DataFrame({"zone": ["NYC", "WEST"],
                        "origin_utc": pd.to_datetime(["2025-01-10 07:00"] * 2),
                        "q90_m1": [0.2, 0.2], "q90_m3": [0.2, 0.2]})
    labels = build_value_labels(task, q90)
    assert labels["v_wql"].tolist() == pytest.approx([0.04, -0.04])
    assert labels["premium_positive"].tolist() == [1, 0]


def test_19_identical_forecasts_give_zero_value():
    task = pd.DataFrame({"zone": ["NYC"], "origin_utc": pd.to_datetime(["2025-01-10 07:00"]),
                         "wql_m1": [0.123], "wql_m3": [0.123],
                         "nmae_m1": [0.1], "nmae_m3": [0.1]})
    q90 = pd.DataFrame({"zone": ["NYC"], "origin_utc": pd.to_datetime(["2025-01-10 07:00"]),
                        "q90_m1": [0.4], "q90_m3": [0.4]})
    labels = build_value_labels(task, q90)
    assert labels["v_wql"].iloc[0] == 0.0
    assert labels["v_q90"].iloc[0] == 0.0


def test_20_labels_are_separate_from_the_feature_frame():
    task = pd.DataFrame({"zone": ["NYC"], "origin_utc": pd.to_datetime(["2025-01-10 07:00"]),
                         "wql_m1": [0.1], "wql_m3": [0.2], "nmae_m1": [0.1], "nmae_m3": [0.1]})
    q90 = pd.DataFrame({"zone": ["NYC"], "origin_utc": pd.to_datetime(["2025-01-10 07:00"]),
                        "q90_m1": [0.2], "q90_m3": [0.3]})
    labels = build_value_labels(task, q90)
    from covariate_trust.acquisition_features import assert_no_forbidden_columns

    with pytest.raises(Exception):
        assert_no_forbidden_columns(labels, list(labels.columns))


def test_single_quantile_loss_matches_the_wql_normalisation():
    y = np.array([100.0, 120.0, 90.0])
    qhat = np.array([95.0, 130.0, 90.0])
    got = single_quantile_loss(y, qhat, 0.9)
    expected = 2.0 * float(pinball(y, qhat, 0.9).sum()) / (float(np.abs(y).sum()) + EPSILON)
    assert got == pytest.approx(expected)
    # and it agrees with metrics.wql restricted to one level
    assert got == pytest.approx(wql(y, qhat.reshape(-1, 1), [0.9]))


def test_q90_losses_drop_incomplete_tasks():
    origin = pd.Timestamp("2025-01-10 07:00")
    rows = []
    for method in ("M1_past_covariate_only", "M3_forecasted_future_covariate"):
        for h in range(1, 25):
            rows.append({"zone": "NYC", "origin_utc": origin, "method": method,
                         "h_index": h, "q0.9": 100.0})
        for h in range(1, 5):  # a deliberately short task
            rows.append({"zone": "WEST", "origin_utc": origin, "method": method,
                         "h_index": h, "q0.9": 100.0})
    predictions = pd.DataFrame(rows)
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h")
    load = pd.concat([
        pd.DataFrame({"zone": z, "timestamp_utc": times, "load_mw": 90.0})
        for z in ("NYC", "WEST")
    ], ignore_index=True)
    out = compute_q90_losses(predictions, load, "M1_past_covariate_only",
                             "M3_forecasted_future_covariate")
    assert out["zone"].tolist() == ["NYC"]


def test_value_distribution_summary_reports_tails():
    labels = pd.DataFrame({"zone": ["NYC"] * 50 + ["WEST"] * 50,
                           "v_wql": np.concatenate([np.linspace(-1, 1, 50), np.zeros(50)])})
    overall = value_distribution_summary(labels)
    per_zone = value_distribution_summary(labels, "zone")
    assert overall["n"].iloc[0] == 100
    assert set(per_zone["group"]) == {"NYC", "WEST"}
    assert "positive_rate" in overall.columns and "skew" in overall.columns
