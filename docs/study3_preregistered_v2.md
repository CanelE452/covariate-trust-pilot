# Study 3 pre-registration v2 - real forecast-vintage external validation

Supersedes `docs/study3_preregistered.md` (v1), which is retained unmodified and marked
`INVALID_PRE_EXECUTION_AVAILABILITY_ASSUMPTION` in
`docs/study3_preregistration_v1_status.md`.

Written before any held-out target, M1/M2/M3 score, Gate H or Gate I quantity existed.
At the time of writing there is no `runs/*_real_vintage` directory. The
machine-readable copy is `runs/<run_id>/preregistration_v2.json`; its SHA-256 goes into
that run's `manifest.json`.

Anchor:

```
start commit     ccf629cf67376d8647e08d998ba700092829eac7
start diff hash  1ca88dfcf38823c7b3d2548213328232bdd2e726b42d1cd15b7c0fa02bd0d5fc
```

## 1. What changed from v1, and why

### 1a. Decision origin 06:00 -> 07:00 UTC

[확인] ECMWF disseminates the 00 UTC HRES hourly steps progressively, finishing around
06:12 UTC. At 06:00 UTC the complete 24-hour forecast path is therefore not guaranteed
to be in hand, so v1 handed M3 information a real decision maker might not have had.
v2 moves the origin to 07:00 UTC, one hour after dissemination completes.

Everything downstream is recomputed on that basis:

```
decision origin      07:00 UTC
forecast valid slice 07:00 UTC .. 06:00 UTC next day   (24 hours)
context              the 512 hours ending 06:00 UTC
primary run          same-day 00Z, initialised 7 hours before the origin
revision run         previous-day 12Z, initialised 19 hours before the origin
```

### 1b. A shared future calendar block for M1, M2 and M3

[확인] In v1 only M3 carried a `future_df`. A future frame conveys the future
timestamps themselves, so part of any M1-to-M3 gap could have come from the model
seeing the forecast window's time-of-day and day-of-week structure rather than from
weather content. v2 gives all three methods an identical future calendar block:

```
local_hour_sin, local_hour_cos, day_of_week_sin, day_of_week_cos, is_weekend
```

computed on DST-aware `America/New_York` local time, while the timestamps handed to the
model stay UTC. The same five columns also appear in every context.

```
M0  context: target only                                     future: none
M1  context: target + verified temperature + calendar        future: calendar
M2  context: identical to M1                                 future: calendar + verified temperature
M3  context: identical to M1                                 future: calendar + forecast temperature
```

Asserted on every origin: identical contexts across M1/M2/M3; identical future
timestamps; identical future calendar values; M2 and M3 differ **only** in the
temperature column; M1 has no future temperature column; no future frame contains the
target.

### 1c. Naming: the real ratio is not the synthetic lambda

```
realized_weather_error_ratio = RMSE(00Z forecast, verification)
                             / RMSE(168h seasonal-naive, verification)
reported_reliability_ratio   = frozen isotonic( 0.70 * revision_ratio
                                              + 0.30 * recent realized ratio )
```

The formula is unchanged from v1; only the names are. The synthetic `lambda` was a
multiple of a known AR(1) conditional standard deviation; nothing on real data
reproduces that construction.

Claims **not** made: that the two quantities are mathematically identical; that a ratio
of 1.0 is a theoretical boundary of Chronos WQL; that 0.75 / 1.25 are optimal on real
data. What is claimed: the D7 band fixed on the synthetic scale is applied unchanged to
an analogous real-world error ratio, as a transfer test.

### 1d. Model-cycle metadata

[확인] IFS Cycle 50r1 became operational with the 06 UTC run on 12 May 2026
([ECMWF announcement](https://forum.ecmwf.int/t/confirmation-ifs-cycle-50r1-and-aifs-v2-joint-implementation-on-12-may-2026/14937)),
so the first 00 UTC run on the new cycle is 13 May 2026. Every held-out origin is
labelled `pre_50r1` or `post_50r1`. This is a **secondary diagnostic only**: it enters
no Gate H or Gate I criterion and no weighting. Subsets that are too small are reported
`INCONCLUSIVE_LOW_COUNT`.

## 2. Unchanged from v1

Data sources, zones, the 512/24 context and horizon, the chronological split
(train 2024-04-01..2024-12-31, validation ..2025-06-30, held-out test
2025-07-01..2026-06-30), the 0.70/0.30 proxy weights, the 28-origin window, isotonic
calibration fitted on train only and frozen, the policy set D0/D1/D2/D3/D5/D7 with **D7
primary**, the D7 band 0.75/1.25, the D5 threshold 1.00, calendar-week cluster
bootstrap with 5,000 resamples, and every Gate H and Gate I criterion.

D5 scoring better than D7 on the held-out period is an observation, not a reason to
change the primary policy.

## 3. Gate H

```
H1  >= 3 zones, >= 180 held-out origins per zone, primary forecast coverage >= 95%,
    no leakage, time alignment ok
H2  M2 beats M1 by >= 1% with a week-clustered CI on the improvement side
H3  M3 win rate inside [20%, 80%]
H4  per-origin min(M1, M3) beats best fixed by >= 1% with a CI on the improvement side
H5  held-out Spearman(reported_reliability_ratio, realized_weather_error_ratio) >= 0.20
    and the top reported quartile has >= 20% higher mean realized ratio than the bottom
```

## 4. Gate I

```
I1  D7 beats best fixed by >= 1.5%
I2  calendar-week cluster-bootstrap CI on the improvement side
I3  D7 recovers >= 20% of the oracle admission gap
I4  D7 cuts the harm rate versus always-use by >= 25%
I5  every zone within 2% of best fixed
I6  every season within 2% of best fixed
I7  if >= 20 worsening events, D7 beats D3
I8  if >= 20 improvement events, D7 beats D0
```

Subsets with fewer than 20 events are `NOT_EVALUABLE`, never counted as a pass.

## 5. Verdict mapping

```
Gate H PASS + Gate I PASS          REAL_VINTAGE_EXTERNAL_VALIDATION_GO
Gate H PASS + Gate I INCONCLUSIVE  EXTERNAL_VALIDATION_CONDITIONAL
Gate H PASS + Gate I FAIL          SYNTHETIC_TO_REAL_METHOD_NO_GO
Gate H not PASS                    REAL_DATA_PROBLEM_NOT_ESTABLISHED
environment / API / schema         BLOCKED_EXTERNAL_DATA
```

A PASS is reported as "Real-vintage external validation in NYISO: GO", never as general
validation for all deployments.

## 6. Data continuity

[확인] No cache was deleted when moving from 06 to 07 UTC. Re-deriving the panel at
07 UTC required **1** new API request; everything else was served from the existing
cache. The 06 UTC artifacts are kept; the 07 UTC panel is written to
`weather_runs_v2_07utc.parquet` and `aligned_panel_v2_07utc.parquet`.

## 7. What would invalidate this run

An implementation error forcing a change to this document or to a gate marks the run
`INVALID_IMPLEMENTATION`; it is not overwritten, and the study restarts under a new run
id with the reason recorded. Forbidden once any held-out quantity exists: retuning a
threshold, refitting the proxy on test data, dropping a zone on its test performance,
promoting D5 to primary, or re-running for a better number.
