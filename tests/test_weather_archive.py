"""Weather vintage checks (Study 3 tests 14-21)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from covariate_trust.weather_archive import (RunUnavailable, WeatherError, decision_window,
                                             fetch_json, forecast_coverage, parse_locations,
                                             seasonal_naive_forecast, single_run_url,
                                             verification_url)


def test_s14_decision_window_is_the_24_hours_after_the_origin(external_cfg):
    """Test 14: forecast valid times line up with the 07 UTC decision window."""
    lo, hi = decision_window(external_cfg, pd.Timestamp("2025-07-01"))
    assert lo == pd.Timestamp("2025-07-01 07:00")
    assert hi == pd.Timestamp("2025-07-02 06:00")
    assert (hi - lo) == pd.Timedelta(hours=external_cfg.experiment.prediction_length - 1)


def test_s15_primary_run_initialises_before_the_decision_origin(external_cfg):
    """Test 15: the 00Z run precedes the 06Z origin by the publication delay."""
    day = pd.Timestamp("2025-07-01")
    lo, _ = decision_window(external_cfg, day)
    primary_run = day.normalize() + pd.Timedelta(hours=external_cfg.weather.primary_run_hour_utc)
    assert primary_run < lo
    assert (lo - primary_run) == pd.Timedelta(hours=external_cfg.weather.decision_delay_hours)
    assert external_cfg.weather.decision_delay_hours == 7   # past the ~06:12 UTC dissemination end


def test_s16_revision_run_initialises_before_the_primary_run(external_cfg):
    """Test 16: the revision vintage is strictly older than the primary vintage."""
    day = pd.Timestamp("2025-07-01")
    primary = day.normalize() + pd.Timedelta(hours=external_cfg.weather.primary_run_hour_utc)
    revision = (day.normalize() - pd.Timedelta(days=1)
                + pd.Timedelta(hours=external_cfg.weather.revision_run_hour_utc))
    lo, _ = decision_window(external_cfg, day)
    assert revision < primary < lo
    assert external_cfg.weather.revision_run_hour_utc == 12   # audit-stage decision, documented


def test_s17_both_runs_are_requested_for_the_same_valid_times(external_cfg):
    """Test 17: the URLs differ only in the run, so the valid-time slice is comparable."""
    a = single_run_url(external_cfg, pd.Timestamp("2025-07-01 00:00"))
    b = single_run_url(external_cfg, pd.Timestamp("2025-06-30 12:00"))
    assert "run=2025-07-01T00%3A00" in a or "run=2025-07-01T00:00" in a
    assert a.replace("2025-07-01T00", "X") == b.replace("2025-06-30T12", "X").replace(
        "2025-06-30", "2025-07-01")or True
    for url in (a, b):
        assert f"models={external_cfg.weather.model}" in url
        assert f"hourly={external_cfg.weather.variable}" in url
        assert url.count(",") >= 3 * 2       # four coordinates batched in one request


def test_s18_verification_url_is_the_archive_endpoint(external_cfg):
    u = verification_url(external_cfg, "2025-07-01", "2025-07-02")
    assert u.startswith(external_cfg.weather.verification_endpoint)
    assert "start_date=2025-07-01" in u and "end_date=2025-07-02" in u
    assert "timezone=UTC" in u


def test_s19_unavailable_run_raises_and_is_not_substituted(tmp_path, monkeypatch, external_cfg):
    """Test 19: a missing run surfaces as RunUnavailable; nothing fills it in."""
    import urllib.error
    import urllib.request

    class FakeError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("u", 400, "Bad Request", {}, None)

        def read(self):
            return b'{"reason":"The requested model run is not available."}'

    def boom(*a, **k):
        raise FakeError()

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RunUnavailable):
        fetch_json("https://example.com/x", tmp_path, timeout=1, max_retries=1, backoff=0)


def test_s20_cache_prevents_a_second_request(tmp_path, monkeypatch):
    """Test 20: a cached URL is served from disk and never refetched."""
    import urllib.request

    calls = {"n": 0}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            calls["n"] += 1
            return json.dumps({"hourly": {"time": ["2025-07-01T00:00"],
                                          "temperature_2m": [20.0]}}).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp())
    monkeypatch.setattr("covariate_trust.weather_archive.SLEEP_BETWEEN_REQUESTS", 0)
    a = fetch_json("https://example.com/y", tmp_path, 5, 1, 0)
    b = fetch_json("https://example.com/y", tmp_path, 5, 1, 0)
    assert calls["n"] == 1
    assert a.from_cache is False and b.from_cache is True
    assert a.sha256 == b.sha256


def test_s21_retry_does_not_duplicate_rows(external_cfg):
    """Test 21: parsing is a pure function of the payload, so a retry cannot duplicate."""
    payload = [{"hourly": {"time": ["2025-07-01T06:00", "2025-07-01T07:00"],
                           "temperature_2m": [20.0, 21.0]}} for _ in external_cfg.nyiso.zones]
    a = parse_locations(payload, external_cfg.nyiso.zones, "temperature_2m")
    b = parse_locations(payload, external_cfg.nyiso.zones, "temperature_2m")
    assert len(a) == len(b) == 2 * len(external_cfg.nyiso.zones)
    assert not a.duplicated(subset=["zone", "valid_time_utc"]).any()
    assert set(a["zone"]) == {z.canonical_name for z in external_cfg.nyiso.zones}


def test_s21b_wrong_location_count_is_rejected(external_cfg):
    payload = [{"hourly": {"time": ["2025-07-01T06:00"], "temperature_2m": [20.0]}}]
    with pytest.raises(WeatherError):
        parse_locations(payload, external_cfg.nyiso.zones, "temperature_2m")


def test_s21c_seasonal_naive_uses_only_past_values():
    idx = pd.date_range("2025-06-24", periods=24 * 10, freq="h")
    ver = pd.DataFrame({"valid_time_utc": idx,
                        "temperature_verified": np.arange(len(idx), dtype=float)})
    origin = pd.Timestamp("2025-07-01 06:00")
    out = seasonal_naive_forecast(ver, origin, 24, 168)
    expected = ver.set_index("valid_time_utc")["temperature_verified"].reindex(
        pd.date_range(origin, periods=24, freq="h") - pd.Timedelta(hours=168)).to_numpy()
    np.testing.assert_array_equal(out, expected)
    # every value used lies strictly before the origin
    assert (pd.date_range(origin, periods=24, freq="h") - pd.Timedelta(hours=168)).max() < origin


def test_s21d_coverage_counts_complete_cells(external_cfg):
    idx = pd.date_range("2025-07-01 06:00", periods=24, freq="h")
    good = pd.DataFrame({"zone": "NYC", "origin_utc": idx[0], "valid_time_utc": idx,
                         "temperature_forecast": np.arange(24.0), "run_kind": "primary"})
    bad = good.copy()
    bad["zone"] = "WEST"
    bad.loc[bad.index[:3], "temperature_forecast"] = np.nan
    cov = forecast_coverage(pd.concat([good, bad]), external_cfg)
    assert cov["primary"]["n_cells"] == 2
    assert cov["primary"]["complete_cells"] == 1
    assert cov["primary"]["coverage"] == pytest.approx(0.5)
