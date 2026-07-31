# Study 3 data sources (measured)

Every statement here was produced by fetching the live endpoint, not by recalling a
URL pattern or a parameter name.

## A. Target - NYISO Integrated Real-Time Actual Load (P-58C)

```
index          https://mis.nyiso.com/public/P-58Clist.htm
fallback index https://mis.nyiso.com/public/P-58Blist.htm
page title     "Integrated Real-Time Actual Load"
daily CSV      csv/palIntegrated/YYYYMMDDpalIntegrated.csv
monthly ZIP    csv/palIntegrated/YYYYMM01palIntegrated_csv.zip   (302 links on the index)
```

Links are taken from the index HTML's actual `href` attributes; the filename pattern
above is a description of what was found there, not a template used to construct URLs.

### CSV schema (from `20250701palIntegrated_csv.zip`)

```
"Time Stamp","Time Zone","Name","PTID","Integrated Load"
"07/01/2025 00:00:00","EDT","CAPITL",61757,1618.7514
```

* Already hourly: 24 rows per zone per day, 264 rows per day across 11 zones.
* `Integrated Load` is MW (integrated real-time actual load).
* **`Time Zone` carries EDT/EST explicitly.** This is what resolves the ambiguous
  autumn DST hour: the repeated local hour appears twice with different offsets, so no
  heuristic is needed. The nonexistent spring hour is simply absent and is never
  interpolated.
* Zones present: `CAPITL CENTRL DUNWOD GENESE HUD VL LONGIL MHK VL MILLWD N.Y.C. NORTH WEST`.
  All four configured zones (`N.Y.C.`, `LONGIL`, `CAPITL`, `WEST`) exist.

## B. Forecast covariate - ECMWF IFS HRES single runs

```
endpoint  https://single-runs-api.open-meteo.com/v1/forecast
required  run=<ISO datetime, e.g. 2025-07-01T00:00>     <- mandatory
required  models=ecmwf_ifs                               <- default is dwd_icon, so this
                                                             must be passed explicitly
variable  hourly=temperature_2m
```

Measured behaviour:

* Omitting `run` returns `{"reason":"Parameter 'run' is required"}`.
* `models=ecmwf_ifs025` returns "model run is not available"; `ecmwf_ifs` works.
* A successful response has `timezone: GMT`, `utc_offset_seconds: 0`, and 168 hourly
  steps starting **at the run hour** (e.g. run `2025-08-01T00:00` -> times
  `2025-08-01T00:00 ... 2025-08-07T23:00`).
* 00Z runs were verified available at 2024-04-01, 2024-10-01, 2025-01-15, 2025-07-01,
  2026-01-15 and 2026-06-30, i.e. across the whole requested window.
* The 18Z run of the previous day is available and is used only for the run-to-run
  revision proxy.
* **Multi-coordinate batching is supported**: passing four comma-separated
  `latitude`/`longitude` values returns a JSON *list* of four location objects, each
  with its own `hourly` block. This cuts the request count by 4x.

This is a genuine single model run, not a stitched fixed-lead series. The report says
so explicitly and does not describe it as anything else.

## C. Weather verification

```
endpoint  https://archive-api.open-meteo.com/v1/archive
variable  hourly=temperature_2m, timezone=UTC
```

Returns hourly `temperature_2m` with `timezone: GMT`. This is a
**reanalysis/model-based verification series**, not a station observation record, and
it is called that everywhere in the report. Using it as "truth" for computing the
covariate forecast error is a limitation, stated as such.

## D. What is fetched, and what is not committed

```
data/raw/nyiso/            monthly ZIPs actually linked from the index
data/raw/open_meteo/       one JSON per (run datetime) covering all four zones
data/raw/source_manifest.json, download_log.jsonl, http_checksums.json
data/processed/*.parquet
```

`data/` is added to `.gitignore`. No raw third-party payload is committed.

Requests are cached on disk by URL hash, retried with exponential backoff, and resumed;
a 404 or an unavailable run is recorded as missing and is **never** filled with a later
vintage or with verification data.

## E. Revision run changed from 18Z to 12Z (audit-stage decision)

The instructions specified the previous day's **18Z** run as the revision vintage. A
direct availability probe over 11 dates spanning 2024-04 to 2026-06 measured:

```
previous-day 06Z   complete 24h decision slice in  9/11 dates
previous-day 12Z   complete 24h decision slice in 11/11 dates
previous-day 18Z   complete 24h decision slice in  7/11 dates
```

The 18Z failures are not transient: for 2024-04-01 and 2024-06-15 the API returns
`{"reason":"The requested model run is not available. Model: ecmwf_ifs, run: ...18:00Z"}`,
and for 2024-12-31 and 2025-06-30 it returns HTTP 200 with an all-NaN series. The 2024
H1 gap matters most, because that period is the proxy *training* window; with 18Z the
isotonic calibrator would have had almost no data to fit.

The 00Z primary run, by contrast, was complete in 11/11 sampled dates.

**Decision, taken at the audit stage before any pre-registration was written and before
any held-out quantity was computed:** use the previous day's **12Z** run as the revision
vintage. It initialises 18 hours before the 06Z decision origin, so it remains strictly
a past vintage with no leakage, and it is still a genuine run-to-run revision against
the 00Z primary (12 hours apart). This was reported and confirmed rather than changed
silently, as the instructions require when a config value does not match the API.
