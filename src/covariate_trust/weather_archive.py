"""Open-Meteo ingestion: ECMWF IFS HRES single runs plus a verification series.

Two distinct things are fetched and they are never mixed:

* **forecast vintage** - one specific model run (``run=<ISO datetime>``,
  ``models=ecmwf_ifs``).  This is what a decision maker actually had.  It is a genuine
  single run, not a stitched fixed-lead series.
* **verification** - the archive endpoint's ``temperature_2m`` for the same valid
  times.  This is a reanalysis/model-based verification series, not station
  observations, and is only ever used for scoring, for M1's past covariate and for the
  M2 oracle diagnostic.

A run that is unavailable is recorded as missing.  It is never backfilled with a later
vintage, with verification data, or with an interpolation.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

USER_AGENT = "covariate-trust-pilot/0.1 (research; contact via repository)"
SLEEP_BETWEEN_REQUESTS = 0.25


class WeatherError(RuntimeError):
    """Raised when the weather API cannot supply what the study requires."""


class RunUnavailable(WeatherError):
    """A specific model run does not exist.  Recorded as missing, never substituted."""


@dataclass(frozen=True)
class FetchResult:
    url: str
    payload: object
    sha256: str
    from_cache: bool


def _cache_path(cache_dir: Path, url: str) -> Path:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"{key}.json"


def fetch_json(url: str, cache_dir: Path, timeout: int = 120, max_retries: int = 5,
               backoff: float = 5.0, log=lambda *_: None) -> FetchResult:
    """GET with disk cache, retry and exponential backoff."""
    path = _cache_path(cache_dir, url)
    if path.exists():
        blob = path.read_bytes()
        return FetchResult(url, json.loads(blob), hashlib.sha256(blob).hexdigest(), True)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
            tmp = path.with_suffix(".json.tmp")
            tmp.write_bytes(blob)
            tmp.replace(path)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return FetchResult(url, json.loads(blob), hashlib.sha256(blob).hexdigest(), False)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code == 400 and "not available" in body:
                raise RunUnavailable(f"{url}: {body[:200]}") from exc
            last = exc
            log(f"    HTTP {exc.code} on attempt {attempt + 1}: {body[:120]}")
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(backoff * (2 ** attempt))
    raise WeatherError(f"failed after {max_retries} attempts: {url}: {last}")


def _coord_params(zones) -> dict:
    return {"latitude": ",".join(f"{z.latitude}" for z in zones),
            "longitude": ",".join(f"{z.longitude}" for z in zones)}


def _as_list(payload) -> list:
    """The API returns a bare object for one coordinate and a list for several."""
    return payload if isinstance(payload, list) else [payload]


def single_run_url(cfg, run_utc: pd.Timestamp) -> str:
    w = cfg.weather
    params = {
        **_coord_params(cfg.nyiso.zones),
        "hourly": w.variable,
        "models": w.model,
        "run": run_utc.strftime("%Y-%m-%dT%H:%M"),
    }
    return f"{w.forecast_endpoint}?{urllib.parse.urlencode(params, safe=',:')}"


def verification_url(cfg, start: str, end: str) -> str:
    w = cfg.weather
    params = {
        **_coord_params(cfg.nyiso.zones),
        "hourly": w.variable,
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
    }
    return f"{w.verification_endpoint}?{urllib.parse.urlencode(params, safe=',:')}"


def parse_locations(payload, zones, variable: str) -> pd.DataFrame:
    """Flatten a (possibly multi-location) response into a long frame."""
    blocks = _as_list(payload)
    if len(blocks) != len(zones):
        raise WeatherError(f"expected {len(zones)} locations, got {len(blocks)}")
    frames = []
    for zone, block in zip(zones, blocks):
        hourly = block.get("hourly") or {}
        if variable not in hourly or "time" not in hourly:
            raise WeatherError(f"response for {zone.canonical_name} lacks {variable}")
        frames.append(pd.DataFrame({
            "zone": zone.canonical_name,
            "valid_time_utc": pd.to_datetime(hourly["time"]),
            variable: pd.to_numeric(pd.Series(hourly[variable]), errors="coerce").to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def fetch_run(cfg, run_utc: pd.Timestamp, cache_dir: Path,
              log=lambda *_: None) -> tuple[pd.DataFrame | None, dict]:
    """One model run for every zone.  Returns (frame or None, log entry)."""
    url = single_run_url(cfg, run_utc)
    entry = {"kind": "forecast_run", "run_utc": str(run_utc), "url": url}
    try:
        res = fetch_json(url, cache_dir, cfg.weather.request_timeout_seconds,
                         cfg.weather.max_retries, cfg.weather.retry_backoff_seconds, log)
    except RunUnavailable as exc:
        entry.update({"status": "unavailable", "detail": str(exc)[:200]})
        return None, entry
    except WeatherError as exc:
        entry.update({"status": "failed", "detail": str(exc)[:200]})
        return None, entry
    df = parse_locations(res.payload, cfg.nyiso.zones, cfg.weather.variable)
    df["run_utc"] = run_utc
    entry.update({"status": "ok", "sha256": res.sha256, "from_cache": res.from_cache,
                  "n_rows": int(len(df))})
    return df, entry


def fetch_verification(cfg, start: str, end: str, cache_dir: Path,
                       log=lambda *_: None) -> tuple[pd.DataFrame, dict]:
    url = verification_url(cfg, start, end)
    res = fetch_json(url, cache_dir, cfg.weather.request_timeout_seconds,
                     cfg.weather.max_retries, cfg.weather.retry_backoff_seconds, log)
    df = parse_locations(res.payload, cfg.nyiso.zones, cfg.weather.variable)
    df = df.rename(columns={cfg.weather.variable: "temperature_verified"})
    entry = {"kind": "verification", "url": url, "status": "ok", "sha256": res.sha256,
             "from_cache": res.from_cache, "start": start, "end": end,
             "n_rows": int(len(df))}
    return df, entry


def decision_window(cfg, run_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The 24 valid hours a decision taken at the origin hour is about."""
    origin = run_date.normalize() + pd.Timedelta(hours=cfg.experiment.decision_origin_hour_utc)
    return origin, origin + pd.Timedelta(hours=cfg.experiment.prediction_length - 1)


def build_weather_runs(cfg, cache_dir: Path, log=lambda *_: None,
                       max_workers: int = 6) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch, for each date, the primary run and the previous day's revision run.

    Requests are issued from a small thread pool.  Open-Meteo's free tier allows far
    more than this per minute, and each response is written to its own cache file, so
    concurrency changes throughput only - never which run is fetched for which origin.
    """
    from concurrent.futures import ThreadPoolExecutor

    w = cfg.weather
    dates = pd.date_range(cfg.periods.requested_start, cfg.periods.requested_end, freq="D")

    jobs = []
    for day in dates:
        lo, hi = decision_window(cfg, day)
        primary_run = day.normalize() + pd.Timedelta(hours=w.primary_run_hour_utc)
        revision_run = (day.normalize() - pd.Timedelta(days=1)
                        + pd.Timedelta(hours=w.revision_run_hour_utc))
        for kind, run_utc in (("primary", primary_run), ("revision", revision_run)):
            # a forecast may only be used if its run started before the decision origin
            if run_utc >= lo:
                raise WeatherError(
                    f"{kind} run {run_utc} is not before the decision origin {lo}")
            jobs.append({"kind": kind, "run_utc": run_utc, "lo": lo, "hi": hi})

    done = {"n": 0}

    def work(job):
        df, entry = fetch_run(cfg, job["run_utc"], cache_dir, lambda *_: None)
        entry["kind"] = f"{job['kind']}_run"
        entry["decision_origin_utc"] = str(job["lo"])
        sub = None
        if df is not None:
            sub = df[(df["valid_time_utc"] >= job["lo"])
                     & (df["valid_time_utc"] <= job["hi"])].copy()
            sub["origin_utc"] = job["lo"]
            sub["run_kind"] = job["kind"]
            sub = sub.rename(columns={w.variable: "temperature_forecast"})
        done["n"] += 1
        if done["n"] % 100 == 0:
            log(f"  weather runs {done['n']}/{len(jobs)} requests")
        return sub, entry

    frames, entries = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for sub, entry in pool.map(work, jobs):
            entries.append(entry)
            if sub is not None:
                frames.append(sub)
    log(f"  weather runs {done['n']}/{len(jobs)} requests")
    if not frames:
        raise WeatherError("no forecast runs could be retrieved")
    return pd.concat(frames, ignore_index=True), entries


def forecast_coverage(runs: pd.DataFrame, cfg) -> dict:
    """Fraction of (origin, zone) cells with a complete 24-hour primary forecast."""
    need = cfg.experiment.prediction_length
    out = {}
    for kind, g in runs.groupby("run_kind"):
        counts = g.groupby(["origin_utc", "zone"])["temperature_forecast"].apply(
            lambda s: int(s.notna().sum()))
        out[kind] = {
            "n_cells": int(len(counts)),
            "complete_cells": int((counts == need).sum()),
            "coverage": float((counts == need).mean()) if len(counts) else 0.0,
        }
    return out


def seasonal_naive_forecast(verification: pd.DataFrame, origin: pd.Timestamp,
                            horizon: int, lag_hours: int) -> np.ndarray:
    """x_hat(t+h) = x(t+h-lag): a past-only weather baseline for the lambda denominator."""
    idx = pd.date_range(origin, periods=horizon, freq="h") - pd.Timedelta(hours=lag_hours)
    s = verification.set_index("valid_time_utc")["temperature_verified"]
    return s.reindex(idx).to_numpy(dtype=float)
