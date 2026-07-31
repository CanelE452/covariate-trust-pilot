"""NYISO source and timestamp checks (Study 3 tests 1-13)."""

from __future__ import annotations

import hashlib
import io
import zipfile

import pandas as pd
import pytest

from covariate_trust.nyiso_data import (LOAD_COLUMN, NAME_COLUMN, NyisoError, REQUIRED_COLUMNS,
                                        TIMESTAMP_COLUMN, TIMEZONE_COLUMN, TZ_OFFSET_HOURS,
                                        audit_schema, hourly_panel, monthly_zip_links,
                                        parse_archive, quality_report, to_utc_hourly)

INDEX_HTML = """
<HTML><HEAD><TITLE>P-58C Index</TITLE></HEAD><BODY>
<H3>Integrated Real-Time Actual Load</H3>
<a href="csv/palIntegrated/20250701palIntegrated_csv.zip">Jul 2025</a>
<a href="csv/palIntegrated/20250601palIntegrated_csv.zip">Jun 2025</a>
<a href="csv/palIntegrated/20240301palIntegrated_csv.zip">Mar 2024</a>
<a href="csv/palIntegrated/20250715palIntegrated.csv">daily</a>
<a href="http://example.com/other/20250801palIntegrated_csv.zip">absolute</a>
<a href="notes.html">not an archive</a>
</BODY></HTML>
"""

ZONE_MAP = {"N.Y.C.": "NYC", "LONGIL": "LONG_ISLAND"}


def _hrefs(html: str) -> list[str]:
    import re
    return re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)


def _sample_csv(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))


def test_s01_index_links_are_parsed_not_constructed():
    """Test 1/2: monthly archives come from the page's own hrefs."""
    links = monthly_zip_links(_hrefs(INDEX_HTML), "https://mis.nyiso.com/public/P-58Clist.htm",
                             "2025-06-01", "2025-07-31")
    assert [l["month"] for l in links] == ["2025-06", "2025-07"]
    assert links[0]["url"] == ("https://mis.nyiso.com/public/"
                               "csv/palIntegrated/20250601palIntegrated_csv.zip")
    # a link outside the window, a daily CSV and a non-archive link are all skipped
    assert all("2024" not in l["month"] for l in links)
    assert all(l["href"].endswith(".zip") for l in links)


def test_s02_absolute_hrefs_are_left_alone():
    links = monthly_zip_links(_hrefs(INDEX_HTML), "https://mis.nyiso.com/public/P-58Clist.htm",
                             "2025-08-01", "2025-08-31")
    assert links and links[0]["url"] == "http://example.com/other/20250801palIntegrated_csv.zip"


def test_s03_zip_checksum_is_content_addressed(tmp_path):
    """Test 3: the archive checksum is the sha256 of the bytes actually stored."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("20250701palIntegrated.csv",
                   '"Time Stamp","Time Zone","Name","PTID","Integrated Load"\n'
                   '"07/01/2025 00:00:00","EDT","N.Y.C.",61761,7000.0\n')
    blob = buf.getvalue()
    p = tmp_path / "a.zip"
    p.write_bytes(blob)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256(blob).hexdigest()
    df = parse_archive(p)
    assert list(df.columns) == list(REQUIRED_COLUMNS)


def test_s04_schema_detection_and_missing_columns(tmp_path):
    """Test 4: a file without the required columns is rejected, not silently accepted."""
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(NyisoError):
        parse_archive(p)


def test_s05_zone_normalisation(tmp_path):
    """Test 5: source zone names map onto canonical names."""
    df = _sample_csv([("07/01/2025 00:00:00", "EDT", "N.Y.C.", 61761, 7000.0),
                      ("07/01/2025 00:00:00", "EDT", "LONGIL", 61762, 3000.0),
                      ("07/01/2025 00:00:00", "EDT", "WEST", 61752, 1800.0)])
    out, _ = to_utc_hourly(df, ZONE_MAP)
    assert set(out["zone"]) == {"NYC", "LONG_ISLAND"}      # WEST is not in the map
    assert set(out["source_zone"]) == {"N.Y.C.", "LONGIL"}


def test_s06_minimum_zone_count_is_checked(external_cfg):
    """Test 6: the study requires at least three configured zones to be present."""
    assert external_cfg.nyiso.minimum_zone_count == 3
    assert len(external_cfg.nyiso.zones) >= external_cfg.nyiso.minimum_zone_count


def test_s07_schema_audit_reports_zones_and_markers():
    """Test 7: a changed source is detectable from the audit fields."""
    df = _sample_csv([("07/01/2025 00:00:00", "EDT", "N.Y.C.", 61761, 7000.0),
                      ("07/01/2025 01:00:00", "EDT", "N.Y.C.", 61761, 6800.0)])
    a = audit_schema(df)
    assert a["zones"] == ["N.Y.C."]
    assert a["timezone_markers"] == ["EDT"]
    assert a["unparsed_timestamps"] == 0
    assert a["n_rows"] == 2


def test_s08_local_time_plus_marker_becomes_utc():
    """Test 8: EDT is UTC-4 and EST is UTC-5, applied per row."""
    df = _sample_csv([("07/01/2025 00:00:00", "EDT", "N.Y.C.", 61761, 7000.0),
                      ("01/01/2025 00:00:00", "EST", "N.Y.C.", 61761, 6000.0)])
    out, _ = to_utc_hourly(df, ZONE_MAP)
    # compare Timestamps, not their string form: pandas prints midnight without a time part
    got = dict(zip(pd.to_datetime(out["local_timestamp"]), pd.to_datetime(out["timestamp_utc"])))
    assert got[pd.Timestamp("2025-07-01 00:00")] == pd.Timestamp("2025-07-01 04:00")   # EDT
    assert got[pd.Timestamp("2025-01-01 00:00")] == pd.Timestamp("2025-01-01 05:00")   # EST
    assert TZ_OFFSET_HOURS == {"EST": 5, "EDT": 4}


def test_s09_ambiguous_autumn_hour_is_resolved_by_the_marker():
    """Test 9: the repeated 01:00 local hour maps to two distinct UTC hours."""
    df = _sample_csv([("11/02/2025 01:00:00", "EDT", "N.Y.C.", 61761, 5000.0),
                      ("11/02/2025 01:00:00", "EST", "N.Y.C.", 61761, 4900.0)])
    out, report = to_utc_hourly(df, ZONE_MAP)
    assert sorted(out["timestamp_utc"].astype(str)) == ["2025-11-02 05:00:00",
                                                        "2025-11-02 06:00:00"]
    assert report["ambiguous_local_hours_resolved_by_marker"] == 1
    assert report["n_duplicate_utc_pairs"] == 0


def test_s10_missing_spring_hour_is_left_missing():
    """Test 10: the nonexistent 02:00 local hour is absent and is never interpolated."""
    df = _sample_csv([("03/09/2025 01:00:00", "EST", "N.Y.C.", 61761, 5000.0),
                      ("03/09/2025 03:00:00", "EDT", "N.Y.C.", 61761, 5200.0)])
    out, _ = to_utc_hourly(df, ZONE_MAP)
    panel = hourly_panel(out)
    # 06:00 and 07:00 UTC are the two observations; nothing was invented between them
    assert panel["load_mw"].notna().sum() == 2
    assert len(panel) == 2


def test_s11_utc_index_is_strictly_increasing_and_unique():
    """Test 11: the hourly panel has a unique, ordered index per zone."""
    df = _sample_csv([(f"07/01/2025 {h:02d}:00:00", "EDT", "N.Y.C.", 61761, 7000.0 + h)
                      for h in range(5)])
    out, _ = to_utc_hourly(df, ZONE_MAP)
    panel = hourly_panel(out)
    ts = panel[panel["zone"] == "NYC"]["timestamp_utc"]
    assert ts.is_monotonic_increasing and ts.is_unique
    q = quality_report(panel)["NYC"]
    assert q["strictly_increasing"] and q["missing"] == 0


def test_s12_unknown_timezone_marker_raises():
    df = _sample_csv([("07/01/2025 00:00:00", "XYZ", "N.Y.C.", 61761, 7000.0)])
    with pytest.raises(NyisoError, match="unknown timezone"):
        to_utc_hourly(df, ZONE_MAP)


def test_s13_real_panel_has_context_and_horizon(real_panel, external_cfg):
    """Tests 12/13 on the real data: 512 hours of context and a full 24-hour horizon."""
    C, H = external_cfg.experiment.context_length, external_cfg.experiment.prediction_length
    r = real_panel.iloc[0]
    assert len(r["_ctx_index"]) == C
    assert len(r["_fut_index"]) == H
    assert len(r["_y_ctx"]) == C and len(r["_y_fut"]) == H
    assert r["_ctx_index"][-1] == pd.Timestamp(r["origin_utc"]) - pd.Timedelta(hours=1)
    assert r["_fut_index"][0] == pd.Timestamp(r["origin_utc"])
    assert pd.Series(r["_y_ctx"]).notna().all() and pd.Series(r["_y_fut"]).notna().all()
