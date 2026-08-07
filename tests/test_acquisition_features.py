"""Study 4 feature timing and leakage checks (items 7-15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.acquisition_features import (
    FeatureLeakageError,
    assert_no_forbidden_columns,
    base_forecast_features,
    calendar_features,
    fit_missing_value_fallback,
    recent_load_features,
    rolling_base_loss_features,
    weather_features,
)

ALLOWED = ["zone", "month_sin", "recent_load_mean_24", "recent_base_wql_28"]


def _keys(zone="NYC", origins=("2025-01-10 07:00", "2025-01-11 07:00")):
    return pd.DataFrame({"zone": zone, "origin_utc": pd.to_datetime(list(origins))})


def test_08_09_10_11_12_forbidden_columns_are_refused():
    keys = _keys()
    for bad in ("wql_m3", "current_M3_quantiles", "v_wql", "realized_weather_error_ratio",
                "oracle_gain", "premium_value"):
        frame = keys.copy()
        frame[bad] = 1.0
        with pytest.raises(FeatureLeakageError):
            assert_no_forbidden_columns(frame, ALLOWED + [bad])


def test_allow_list_is_enforced():
    frame = _keys()
    frame["not_declared"] = 1.0
    with pytest.raises(FeatureLeakageError):
        assert_no_forbidden_columns(frame, ALLOWED)


def test_recent_base_wql_is_an_allowed_exception():
    frame = _keys()
    frame["recent_base_wql_28"] = 0.1
    assert_no_forbidden_columns(frame, ALLOWED)  # must not raise


def test_07_recent_load_uses_only_data_before_the_origin():
    origin = pd.Timestamp("2025-01-10 07:00")
    times = pd.date_range(origin - pd.Timedelta(hours=200), origin + pd.Timedelta(hours=50), freq="h")
    load = pd.DataFrame({"zone": "NYC", "timestamp_utc": times,
                         "load_mw": np.arange(len(times), dtype=float)})
    keys = pd.DataFrame({"zone": ["NYC"], "origin_utc": [origin]})
    clean = recent_load_features(load, keys)

    poisoned = load.copy()
    poisoned.loc[poisoned["timestamp_utc"] >= origin, "load_mw"] = 1e9
    dirty = recent_load_features(poisoned, keys)
    pd.testing.assert_frame_equal(clean, dirty)
    # the 24h mean must be the mean of the last 24 values strictly before the origin
    before = load[load["timestamp_utc"] < origin]["load_mw"].to_numpy()[-24:]
    assert clean["recent_load_mean_24"].iloc[0] == pytest.approx(before.mean())


def test_11_base_features_never_read_the_premium_method():
    keys = _keys(origins=("2025-01-10 07:00",))
    rows = []
    for method, level in (("M1_past_covariate_only", 1.0),
                          ("M3_forecasted_future_covariate", 999.0)):
        for h in range(1, 25):
            rows.append({"zone": "NYC", "origin_utc": "2025-01-10 07:00", "method": method,
                         "h_index": h, "q0.1": level, "q0.5": level, "q0.9": level + 1.0})
    predictions = pd.DataFrame(rows)
    out = base_forecast_features(predictions, keys, "M1_past_covariate_only")
    assert out["base_forecast_level_mean"].iloc[0] == pytest.approx(1.0)
    assert out["base_interval_width_mean"].iloc[0] == pytest.approx(1.0)


def test_15_weather_features_use_both_runs():
    origin = pd.Timestamp("2025-01-10 07:00")
    valid = pd.date_range(origin, periods=24, freq="h")
    runs = pd.concat([
        pd.DataFrame({"zone": "NYC", "origin_utc": origin, "valid_time_utc": valid,
                      "temperature_forecast": np.full(24, 10.0), "run_kind": "primary"}),
        pd.DataFrame({"zone": "NYC", "origin_utc": origin, "valid_time_utc": valid,
                      "temperature_forecast": np.full(24, 8.0), "run_kind": "revision"}),
    ], ignore_index=True)
    out = weather_features(runs, pd.DataFrame({"zone": ["NYC"], "origin_utc": [origin]}))
    assert out["primary_weather_mean"].iloc[0] == pytest.approx(10.0)
    assert out["primary_weather_range"].iloc[0] == pytest.approx(0.0)
    assert out["revision_mean"].iloc[0] == pytest.approx(2.0)
    assert out["revision_rms"].iloc[0] == pytest.approx(2.0)


def test_rolling_base_loss_is_shifted_by_one_origin():
    origins = pd.date_range("2025-01-01 07:00", periods=6, freq="D")
    losses = pd.DataFrame({"zone": "NYC", "origin_utc": origins,
                           "wql_m1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                           "q90_m1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
    keys = losses[["zone", "origin_utc"]]
    out = rolling_base_loss_features(losses, keys, window_days=3, minimum_days=2)
    assert np.isnan(out["recent_base_wql_28"].iloc[0])
    assert np.isnan(out["recent_base_wql_28"].iloc[1])
    # third origin sees the first two completed origins only
    assert out["recent_base_wql_28"].iloc[2] == pytest.approx(1.5)
    assert out["recent_base_wql_28"].iloc[3] == pytest.approx(2.0)


def test_13_14_missing_fallback_is_fitted_on_train_only():
    frame = pd.DataFrame({
        "zone": ["NYC"] * 4,
        "origin_utc": pd.to_datetime(["2024-05-01", "2024-05-02", "2026-01-01", "2026-01-02"]),
        "recent_load_mean_24": [10.0, 20.0, np.nan, 1e6],
    })
    train_mask = np.array([True, True, False, False])
    filled = fit_missing_value_fallback(frame, train_mask, ["recent_load_mean_24"])
    # the imputed value comes from the train rows, not from the huge test value
    assert filled["recent_load_mean_24"].iloc[2] == pytest.approx(15.0)
    assert filled["recent_load_mean_24"].iloc[3] == 1e6


def test_calendar_features_are_cyclic():
    out = calendar_features(pd.Series(pd.to_datetime(["2025-01-15 07:00", "2025-07-15 07:00"])))
    assert set(out.columns) == {"month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"}
    assert np.allclose(out["month_sin"] ** 2 + out["month_cos"] ** 2, 1.0)
