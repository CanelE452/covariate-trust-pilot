"""NYISO Integrated Real-Time Actual Load (P-58C) ingestion.

Every URL used here comes from the index page's own ``href`` attributes; nothing is
constructed from a remembered filename pattern.  The measured schema is

    "Time Stamp","Time Zone","Name","PTID","Integrated Load"

which is already hourly (24 rows per zone per day).  The ``Time Zone`` column carries
EDT/EST explicitly, which is what resolves the ambiguous autumn DST hour without any
heuristic: the repeated local hour appears twice with two different offsets.  The
nonexistent spring hour is simply absent and is never interpolated.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

USER_AGENT = "covariate-trust-pilot/0.1 (research; contact via repository)"
LOAD_COLUMN = "Integrated Load"
TIMESTAMP_COLUMN = "Time Stamp"
TIMEZONE_COLUMN = "Time Zone"
NAME_COLUMN = "Name"
REQUIRED_COLUMNS = (TIMESTAMP_COLUMN, TIMEZONE_COLUMN, NAME_COLUMN, "PTID", LOAD_COLUMN)

# NYISO publishes local clock time plus an explicit EDT/EST marker.
TZ_OFFSET_HOURS = {"EST": 5, "EDT": 4}


class NyisoError(RuntimeError):
    """Raised when the NYISO source does not match what the study requires."""


@dataclass(frozen=True)
class Download:
    url: str
    path: Path
    sha256: str
    bytes: int
    from_cache: bool


def _fetch(url: str, timeout: int = 120, max_retries: int = 5,
           backoff: float = 5.0) -> bytes:
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NyisoError(f"404 for {url}") from exc
            last = exc
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(backoff * (2 ** attempt))
    raise NyisoError(f"failed to fetch {url}: {last}")


def fetch_index(url: str) -> list[str]:
    """Return every href on the index page, in page order."""
    html = _fetch(url).decode("utf-8", "replace")
    return re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)


def index_report_title(url: str) -> str:
    html = _fetch(url).decode("utf-8", "replace")
    m = re.search(r"<H3>(.*?)</H3>", html, flags=re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def monthly_zip_links(hrefs: list[str], base_url: str, start: str, end: str) -> list[dict]:
    """Select the monthly ZIP links whose month falls inside [start, end].

    The month is read out of the link text itself, so a link that does not look like a
    dated archive is skipped rather than guessed at.
    """
    root = base_url.rsplit("/", 1)[0]
    start_m = pd.Timestamp(start).to_period("M")
    end_m = pd.Timestamp(end).to_period("M")
    out = []
    for h in hrefs:
        if not h.lower().endswith(".zip"):
            continue
        m = re.search(r"(\d{8})", h)
        if not m:
            continue
        try:
            stamp = pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d"))
        except ValueError:
            continue
        period = stamp.to_period("M")
        if start_m <= period <= end_m:
            out.append({"month": str(period), "href": h,
                        "url": h if h.startswith("http") else f"{root}/{h}"})
    out.sort(key=lambda r: r["month"])
    return out


def download(url: str, cache_dir: Path, timeout: int = 120, max_retries: int = 5,
             backoff: float = 5.0) -> Download:
    """Fetch with an on-disk cache keyed by URL hash.  Never re-downloads."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    suffix = ".zip" if url.lower().endswith(".zip") else ".csv"
    path = cache_dir / f"{key}{suffix}"
    if path.exists():
        blob = path.read_bytes()
        return Download(url, path, hashlib.sha256(blob).hexdigest(), len(blob), True)
    blob = _fetch(url, timeout, max_retries, backoff)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return Download(url, path, hashlib.sha256(blob).hexdigest(), len(blob), False)


def parse_archive(path: Path) -> pd.DataFrame:
    """Read one monthly ZIP (or a single CSV) into a raw frame."""
    path = Path(path)
    frames = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if not name.lower().endswith(".csv"):
                    continue
                frames.append(pd.read_csv(io.BytesIO(z.read(name))))
    else:
        frames.append(pd.read_csv(path))
    if not frames:
        raise NyisoError(f"no CSV inside {path}")
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise NyisoError(f"{path}: missing columns {missing}; got {list(df.columns)}")
    return df


def audit_schema(df: pd.DataFrame) -> dict:
    """What the study needs to know about the source before trusting it."""
    zones = sorted(df[NAME_COLUMN].dropna().unique().tolist())
    tzs = sorted(df[TIMEZONE_COLUMN].dropna().unique().tolist())
    ts = pd.to_datetime(df[TIMESTAMP_COLUMN], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    per_zone = df.assign(_ts=ts).groupby(NAME_COLUMN)["_ts"].apply(
        lambda s: s.sort_values().diff().dropna().value_counts().head(1))
    return {
        "columns": list(df.columns),
        "zones": zones,
        "n_zones": len(zones),
        "timezone_markers": tzs,
        "n_rows": int(len(df)),
        "unparsed_timestamps": int(ts.isna().sum()),
        "dominant_interval_per_zone": {str(k): str(v) for k, v in per_zone.items()},
    }


def to_utc_hourly(df: pd.DataFrame, zone_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Convert local clock time + EDT/EST marker into a UTC hourly panel.

    Returns (panel, dst_report).  The DST report is kept because the conversion is the
    single most likely place for a silent one-hour misalignment.
    """
    df = df[df[NAME_COLUMN].isin(zone_map)].copy()
    local = pd.to_datetime(df[TIMESTAMP_COLUMN], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    bad_tz = sorted(set(df[TIMEZONE_COLUMN].dropna()) - set(TZ_OFFSET_HOURS))
    if bad_tz:
        raise NyisoError(f"unknown timezone markers {bad_tz}")
    offsets = df[TIMEZONE_COLUMN].map(TZ_OFFSET_HOURS)
    utc = local + pd.to_timedelta(offsets, unit="h")

    out = pd.DataFrame({
        "zone": df[NAME_COLUMN].map(zone_map).to_numpy(),
        "source_zone": df[NAME_COLUMN].to_numpy(),
        "timestamp_utc": utc.to_numpy(),
        "local_timestamp": local.to_numpy(),
        "tz_marker": df[TIMEZONE_COLUMN].to_numpy(),
        "load_mw": pd.to_numeric(df[LOAD_COLUMN], errors="coerce").to_numpy(),
    }).dropna(subset=["timestamp_utc", "load_mw"])

    # An ambiguous local hour is one that appears with both markers on the same date.
    amb = (out.groupby(["zone", "local_timestamp"])["tz_marker"].nunique()
           .reset_index().query("tz_marker > 1"))
    dupes = out.duplicated(subset=["zone", "timestamp_utc"], keep=False)
    out = out.sort_values(["zone", "timestamp_utc"])
    deduped = out.drop_duplicates(subset=["zone", "timestamp_utc"], keep="first")

    report = {
        "n_rows_in": int(len(df)),
        "n_rows_out": int(len(deduped)),
        "ambiguous_local_hours_resolved_by_marker": int(len(amb)),
        "duplicate_utc_rows_dropped": int(dupes.sum() - (dupes.sum() > 0)) if dupes.any() else 0,
        "n_duplicate_utc_pairs": int(len(out) - len(deduped)),
        "tz_markers_seen": sorted(out["tz_marker"].unique().tolist()),
        "utc_start": str(deduped["timestamp_utc"].min()),
        "utc_end": str(deduped["timestamp_utc"].max()),
    }
    return deduped.reset_index(drop=True), report


def hourly_panel(records: pd.DataFrame) -> pd.DataFrame:
    """Reindex each zone onto a gap-free hourly UTC index (gaps stay NaN)."""
    frames = []
    for zone, g in records.groupby("zone"):
        g = g.sort_values("timestamp_utc")
        idx = pd.date_range(g["timestamp_utc"].min(), g["timestamp_utc"].max(), freq="h")
        s = g.set_index("timestamp_utc")["load_mw"].reindex(idx)
        frames.append(pd.DataFrame({"zone": zone, "timestamp_utc": idx,
                                    "load_mw": s.to_numpy()}))
    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(subset=["zone", "timestamp_utc"]).any():
        raise NyisoError("duplicate (zone, timestamp) after reindexing")
    return panel


def quality_report(panel: pd.DataFrame) -> dict:
    rows = {}
    for zone, g in panel.groupby("zone"):
        ts = g["timestamp_utc"]
        rows[zone] = {
            "n_hours": int(len(g)),
            "missing": int(g["load_mw"].isna().sum()),
            "missing_rate": float(g["load_mw"].isna().mean()),
            "strictly_increasing": bool(ts.is_monotonic_increasing and ts.is_unique),
            "start": str(ts.min()), "end": str(ts.max()),
            "mean_mw": float(g["load_mw"].mean()),
        }
    return rows


def build_load_panel(cfg, cache_dir: Path, log=lambda *_: None) -> tuple[pd.DataFrame, dict]:
    """Full ingestion: index -> monthly ZIPs -> UTC hourly panel."""
    zone_map = {z.source_name: z.canonical_name for z in cfg.nyiso.zones}
    hrefs = fetch_index(cfg.nyiso.primary_index_url)
    title = index_report_title(cfg.nyiso.primary_index_url)
    links = monthly_zip_links(hrefs, cfg.nyiso.primary_index_url,
                              cfg.periods.requested_start, cfg.periods.requested_end)
    if not links:
        raise NyisoError("no monthly archives found for the requested window")
    log(f"NYISO report '{title}': {len(links)} monthly archives to ingest")

    downloads, frames = [], []
    for i, link in enumerate(links, 1):
        d = download(link["url"], cache_dir)
        downloads.append({"month": link["month"], "url": d.url, "sha256": d.sha256,
                          "bytes": d.bytes, "from_cache": d.from_cache})
        frames.append(parse_archive(d.path))
        if i % 6 == 0 or i == len(links):
            log(f"  NYISO {i}/{len(links)} months")
    raw = pd.concat(frames, ignore_index=True)
    schema = audit_schema(raw)

    present = [z for z in zone_map if z in set(raw[NAME_COLUMN])]
    if len(present) < cfg.nyiso.minimum_zone_count:
        raise NyisoError(f"BLOCKED_ZONE_SCHEMA: only {present} of {list(zone_map)} present")

    records, dst = to_utc_hourly(raw, zone_map)
    panel = hourly_panel(records)
    meta = {"report_title": title, "index_url": cfg.nyiso.primary_index_url,
            "n_months": len(links), "downloads": downloads, "schema": schema,
            "dst": dst, "zones_present": present,
            "quality": quality_report(panel)}
    return panel, meta


def write_download_log(path: Path, entries: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, default=str) + "\n")
