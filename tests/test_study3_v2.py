"""Study 3 v2 checks: 07 UTC origin, shared future calendar, naming, model cycle.

Covers the 19 additional items required after the availability fix.  No existing
assertion was weakened to make any of these pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from covariate_trust.real_vintage import (CALENDAR_COLUMNS, LOCAL_TZ, TEMPERATURE_COLUMN,
                                          assert_fair_comparison, build_real_inputs,
                                          calendar_features)
from covariate_trust.schemas import (ID_COLUMN, M0, M1, M2, M3, SchemaError, TARGET_COLUMN,
                                     TIMESTAMP_COLUMN)
from covariate_trust.weather_archive import decision_window
from covariate_trust.weather_proxy import model_cycle_label

ROOT = Path(__file__).resolve().parents[1]


def _inputs(cfg=None):
    # the context must end exactly one hour before the origin, as assemble_origins builds it
    fi = pd.date_range("2025-07-01 07:00", periods=24, freq="h")
    ci = pd.date_range(end=fi[0] - pd.Timedelta(hours=1), periods=64, freq="h")
    load = np.linspace(5000, 6000, 64)
    temp = np.linspace(18, 26, 64)
    x_true = np.linspace(20, 30, 24)
    x_fc = x_true + 0.9
    return {
        "ci": ci, "fi": fi, "x_true": x_true, "x_fc": x_fc,
        # M0 is given the future index for its horizon length only; it gets no future frame
        M0: build_real_inputs(M0, "t", ci, load, temp, fi),
        M1: build_real_inputs(M1, "t", ci, load, temp, fi),
        M2: build_real_inputs(M2, "t", ci, load, temp, fi, x_true),
        M3: build_real_inputs(M3, "t", ci, load, temp, fi, x_fc),
    }


# ------------------------------------------------------- availability -------

def test_v01_decision_origin_is_07_utc(external_cfg):
    """Test 1: the decision origin is fixed at 07 UTC."""
    assert external_cfg.experiment.decision_origin_hour_utc == 7
    assert external_cfg.weather.decision_delay_hours == 7


def test_v02_primary_run_is_usable_only_after_dissemination(external_cfg):
    """Test 2: the 00Z run gets a 7-hour delay, past the ~06:12 UTC dissemination end."""
    day = pd.Timestamp("2025-07-01")
    lo, _ = decision_window(external_cfg, day)
    primary = day.normalize() + pd.Timedelta(hours=external_cfg.weather.primary_run_hour_utc)
    assert (lo - primary) == pd.Timedelta(hours=7)
    assert lo.hour == 7
    # 06:00 UTC would have been inside the dissemination window and is no longer used
    assert lo > day.normalize() + pd.Timedelta(hours=6, minutes=12)


def test_v03_valid_slice_is_exactly_24_hours_07_to_30(external_cfg):
    """Test 3: the slice runs 07 UTC .. 06 UTC next day, 24 steps."""
    lo, hi = decision_window(external_cfg, pd.Timestamp("2025-07-01"))
    idx = pd.date_range(lo, hi, freq="h")
    assert len(idx) == external_cfg.experiment.prediction_length == 24
    assert idx[0] == pd.Timestamp("2025-07-01 07:00")
    assert idx[-1] == pd.Timestamp("2025-07-02 06:00")


def test_v04_previous_12z_run_precedes_the_origin(external_cfg):
    """Test 4: the revision vintage is 19 hours old at the decision origin."""
    day = pd.Timestamp("2025-07-01")
    lo, _ = decision_window(external_cfg, day)
    rev = day.normalize() - pd.Timedelta(days=1) + pd.Timedelta(
        hours=external_cfg.weather.revision_run_hour_utc)
    prim = day.normalize()
    assert rev < prim < lo
    assert (lo - rev) == pd.Timedelta(hours=19)


def test_v05_06utc_and_07utc_processed_files_are_distinct():
    """Test 5: the 07 UTC panel is written beside, not over, the 06 UTC one."""
    proc = ROOT / "data" / "processed"
    if not (proc / "weather_runs_v2_07utc.parquet").exists():
        pytest.skip("07 UTC panel has not been built yet")
    assert (proc / "weather_runs.parquet").exists(), "the 06 UTC artifact was removed"
    old = pd.read_parquet(proc / "weather_runs.parquet")
    new = pd.read_parquet(proc / "weather_runs_v2_07utc.parquet")
    assert pd.Timestamp(old["origin_utc"].iloc[0]).hour == 6
    assert pd.Timestamp(new["origin_utc"].iloc[0]).hour == 7
    assert set(pd.to_datetime(pd.Series(new["origin_utc"].unique())).dt.hour) == {7}


# ---------------------------------------------------------- calendar -------

def test_v06_calendar_features_use_new_york_local_time():
    """Test 6: features come from America/New_York, not from UTC."""
    idx = pd.date_range("2025-07-01 07:00", periods=24, freq="h")
    cal = calendar_features(idx)
    local_hours = idx.tz_localize("UTC").tz_convert(LOCAL_TZ).hour.to_numpy(dtype=float)
    np.testing.assert_allclose(cal["local_hour_sin"], np.sin(2 * np.pi * local_hours / 24))
    # in July New York is UTC-4, so 07 UTC is 03 local, not 07
    assert local_hours[0] == 3
    assert not np.allclose(cal["local_hour_sin"].to_numpy(),
                           np.sin(2 * np.pi * idx.hour.to_numpy(float) / 24))
    assert list(cal.columns) == list(CALENDAR_COLUMNS)


def test_v07_calendar_features_across_a_dst_transition():
    """Test 7: the local hour is correct on both sides of the DST change."""
    # US DST ended 2025-11-02: EDT (UTC-4) before, EST (UTC-5) after
    before = pd.DatetimeIndex(["2025-11-01 12:00"])       # 08:00 EDT
    after = pd.DatetimeIndex(["2025-11-03 12:00"])        # 07:00 EST
    hb = pd.DatetimeIndex(before).tz_localize("UTC").tz_convert(LOCAL_TZ).hour[0]
    ha = pd.DatetimeIndex(after).tz_localize("UTC").tz_convert(LOCAL_TZ).hour[0]
    assert (hb, ha) == (8, 7)
    cb = calendar_features(before)["local_hour_sin"].iloc[0]
    ca = calendar_features(after)["local_hour_sin"].iloc[0]
    assert cb == pytest.approx(np.sin(2 * np.pi * 8 / 24))
    assert ca == pytest.approx(np.sin(2 * np.pi * 7 / 24))
    assert cb != ca
    # weekend flag: 2025-11-01 is a Saturday, 2025-11-03 a Monday
    assert calendar_features(before)["is_weekend"].iloc[0] == 1.0
    assert calendar_features(after)["is_weekend"].iloc[0] == 0.0


def test_v08_contexts_are_identical_across_m1_m2_m3():
    """Test 8: M1, M2 and M3 see exactly the same context."""
    d = _inputs()
    assert d[M1].context_df.equals(d[M2].context_df)
    assert d[M1].context_df.equals(d[M3].context_df)
    expected = [ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN, TEMPERATURE_COLUMN,
                *CALENDAR_COLUMNS]
    assert list(d[M1].context_df.columns) == expected


def test_v09_future_calendar_blocks_are_identical():
    """Test 9: the future calendar is byte-identical across the three methods."""
    d = _inputs()
    cal1 = d[M1].future_df[list(CALENDAR_COLUMNS)]
    assert cal1.equals(d[M2].future_df[list(CALENDAR_COLUMNS)])
    assert cal1.equals(d[M3].future_df[list(CALENDAR_COLUMNS)])
    assert pd.DatetimeIndex(d[M1].future_df[TIMESTAMP_COLUMN]).equals(
        pd.DatetimeIndex(d[M3].future_df[TIMESTAMP_COLUMN]))
    assert_fair_comparison(d[M1], d[M2], d[M3])


def test_v10_m1_has_no_future_temperature():
    """Test 10: M1's future frame is calendar only."""
    d = _inputs()
    assert TEMPERATURE_COLUMN not in d[M1].future_df.columns
    assert list(d[M1].future_df.columns) == [ID_COLUMN, TIMESTAMP_COLUMN, *CALENDAR_COLUMNS]
    with pytest.raises(SchemaError):
        build_real_inputs(M1, "t", d["ci"], np.zeros(64), np.zeros(64), d["fi"], d["x_fc"])


def test_v11_only_m2_carries_verified_future_temperature():
    """Test 11: M2's future temperature is the verification series."""
    d = _inputs()
    np.testing.assert_allclose(d[M2].future_df[TEMPERATURE_COLUMN].to_numpy(), d["x_true"])


def test_v12_only_m3_carries_forecast_future_temperature():
    """Test 12: M3's future temperature is the ECMWF forecast."""
    d = _inputs()
    np.testing.assert_allclose(d[M3].future_df[TEMPERATURE_COLUMN].to_numpy(), d["x_fc"])


def test_v13_m3_temperature_is_not_swapped_for_verification():
    """Test 13: M2 and M3 differ, and only in the temperature column."""
    d = _inputs()
    assert not np.allclose(d[M3].future_df[TEMPERATURE_COLUMN].to_numpy(),
                           d[M2].future_df[TEMPERATURE_COLUMN].to_numpy())
    assert d[M2].future_df.drop(columns=[TEMPERATURE_COLUMN]).equals(
        d[M3].future_df.drop(columns=[TEMPERATURE_COLUMN]))
    swapped = build_real_inputs(M3, "t", d["ci"], np.zeros(64), np.zeros(64), d["fi"], d["x_true"])
    with pytest.raises(SchemaError):
        assert_fair_comparison(d[M1], d[M2], swapped)   # M2 and M3 would be identical


def test_v14_no_future_frame_contains_the_target():
    """Test 14: the target never appears in a future frame."""
    d = _inputs()
    for m in (M1, M2, M3):
        assert TARGET_COLUMN not in d[m].future_df.columns
    assert d[M0].future_df is None


# ------------------------------------------------------------- naming ------

def test_v15_real_ratio_and_synthetic_lambda_are_separate_symbols():
    """Test 15: the real-data ratio does not reuse the synthetic lambda name."""
    import covariate_trust.weather_proxy as wp
    from covariate_trust import dgp, reliability_schedules

    src = Path(wp.__file__).read_text()
    assert "realized_weather_error_ratio" in src
    assert "reported_reliability_ratio" in src
    # the synthetic modules keep their own lambda vocabulary, untouched
    assert "lam" in Path(dgp.__file__).read_text()
    assert "true_current_lambda" in Path(reliability_schedules.__file__).read_text()
    out = wp.origin_weather_errors(np.array([1.0]), np.array([1.0]), np.array([1.0]),
                                   np.array([2.0]), np.array([2.0]))
    assert "realized_weather_error_ratio" in out
    assert "true_lambda" not in out


def test_v16_reported_ratio_excludes_the_current_realized_error():
    """Test 16: the decision-time proxy cannot see the current origin's own error."""
    from covariate_trust.weather_proxy import add_decision_time_features, raw_proxy_score
    rng = np.random.default_rng(0)
    n = 60
    f = pd.DataFrame({
        "zone": "NYC",
        "origin_utc": pd.date_range("2025-07-01", periods=n, freq="D") + pd.Timedelta(hours=7),
        "realized_weather_error_ratio": rng.uniform(0.4, 1.8, n),
        "e_baseline_rmse_168": rng.uniform(2.0, 4.0, n),
        "revision_rms": rng.uniform(0.5, 3.0, n),
    })
    base = add_decision_time_features(f, 28)
    base["raw"] = raw_proxy_score(base["revision_ratio"], base["recent_realized_ratio"], 0.7, 0.3)
    g = f.copy()
    g.loc[g.index[40], "realized_weather_error_ratio"] = 99.0
    other = add_decision_time_features(g, 28)
    other["raw"] = raw_proxy_score(other["revision_ratio"], other["recent_realized_ratio"], 0.7, 0.3)
    assert other["raw"].iloc[40] == pytest.approx(base["raw"].iloc[40])


def test_v17_calibrator_never_sees_held_out_rows(external_cfg):
    """Test 17: the fit window ends strictly before the held-out period starts."""
    from covariate_trust.weather_proxy import split_periods
    idx = pd.date_range("2024-04-01", "2026-06-30", freq="D") + pd.Timedelta(hours=7)
    s = split_periods(pd.DataFrame({"origin_utc": idx, "zone": "NYC"}), external_cfg)
    assert s["train"]["origin_utc"].max() < pd.Timestamp(external_cfg.periods.heldout_test_start)
    assert s["validation"]["origin_utc"].max() < pd.Timestamp(
        external_cfg.periods.heldout_test_start)
    assert len(set(s["train"]["origin_utc"]) & set(s["test"]["origin_utc"])) == 0


# ------------------------------------------------------- model cycle -------

def test_v18_model_cycle_metadata(external_cfg):
    """Test 18: every origin is labelled pre/post 50r1 from the published date."""
    first = external_cfg.weather.model_cycle_50r1_first_00z_run
    assert first == "2026-05-13"          # 06 UTC run on 12 May 2026 -> first 00Z is the 13th
    assert model_cycle_label("2026-05-12 07:00", first) == "pre_50r1"
    assert model_cycle_label("2026-05-13 07:00", first) == "post_50r1"
    assert model_cycle_label("2025-07-01 07:00", first) == "pre_50r1"


def test_v18b_model_cycle_is_in_the_origin_panel(real_panel):
    if "weather_model_cycle" not in real_panel.columns:
        pytest.skip("panel built before the model-cycle column was added")
    assert set(real_panel["weather_model_cycle"]) <= {"pre_50r1", "post_50r1"}


def test_v19_gate_criteria_do_not_reference_the_model_cycle():
    """The cycle label is a secondary diagnostic and must not enter a gate."""
    from covariate_trust import external_gates
    src = Path(external_gates.__file__).read_text()
    assert "weather_model_cycle" not in src
    assert "50r1" not in src
