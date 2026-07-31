# Study 3 pre-registration - real forecast-vintage external validation

Written before any held-out target, M1/M3 score or Gate quantity was computed. The
machine-readable copy is `runs/<run_id>/preregistration.json`; its SHA-256 goes into
that run's `manifest.json`.

Anchor:

```
start commit     ccf629cf67376d8647e08d998ba700092829eac7
start diff hash  1ca88dfcf38823c7b3d2548213328232bdd2e726b42d1cd15b7c0fa02bd0d5fc
```

## 1. Question

Does the admission policy **D7_hybrid_override**, whose thresholds were fixed by the
synthetic studies, reduce forecast error and harmful forecasts on a real
forecast-vintage backtest, compared with always using the weather forecast, never using
it, and a history-only policy?

## 2. Status this study starts from

```
Synthetic method validation   GO           (Gate G PASS on a held-out synthetic sample)
General deployment claim      CONDITIONAL
```

Study 3 can move the second line for **one domain only**. A PASS is reported as
"Real-vintage external validation in NYISO: GO", never as general validation.

## 3. Data, fixed in advance

```
target        NYISO Integrated Real-Time Actual Load (P-58C), hourly, MW
              index https://mis.nyiso.com/public/P-58Clist.htm
forecast      ECMWF IFS HRES via https://single-runs-api.open-meteo.com/v1/forecast
              models=ecmwf_ifs, hourly=temperature_2m, run=<explicit ISO datetime>
verification  https://archive-api.open-meteo.com/v1/archive, hourly temperature_2m
              (a reanalysis/model-based verification series, NOT station observations)
zones         N.Y.C.->NYC, LONGIL->LONG_ISLAND, CAPITL->CAPITAL, WEST->WEST
              with one representative coordinate each
primary run   00 UTC
revision run  previous day 12 UTC   (changed from 18Z at the audit stage; see
                                     docs/study3_data_sources.md section E)
decision origin 06 UTC, i.e. the 00Z run plus a 6-hour publication delay
context       512 hours   horizon 24 hours
```

## 4. Split, fixed in advance

```
proxy train        2024-04-01 .. 2024-12-31
proxy validation   2025-01-01 .. 2025-06-30
held-out test      2025-07-01 .. 2026-06-30
```

The test period is never used to fit the calibrator, to choose a threshold, or to drop
a zone. If real coverage falls short the run reports `PASS_COVERAGE`,
`PARTIAL_COVERAGE` or `BLOCKED_COVERAGE`; the dates are not moved.

## 5. Reliability definitions, fixed in advance

```
true_lambda    RMSE(00Z forecast, verification) / RMSE(168h seasonal-naive, verification)
               over the 24 valid hours.  Needs the future, so it is evaluation-only.

lambda_reported = frozen_isotonic( 0.70 * revision_ratio + 0.30 * recent_lambda )
  revision_ratio  RMSE(00Z primary, previous-day 12Z) / mean past 168h-naive error level
  recent_lambda   mean true_lambda of the previous 28 *completed* origins
```

The 0.70 / 0.30 split, the 28-origin window and the isotonic method are fixed here. The
calibrator is fitted on the train period only and frozen; validation is used to *report*
slope, MAE, Spearman and a calibration plot, never to change anything.

## 6. Policies, fixed in advance

```
D0 always_no_future     always M1
D1 always_use_future    always M3
D2 oracle               per-origin min(M1, M3)  - an upper bound, not a method
D3 historical_utility   28-origin mean WQL(M3) < mean WQL(M1) -> M3, else M1
                        (needs >= 14 previous origins, otherwise M1)
D5 current_proxy        lambda_reported < 1.00 -> M3, else M1
D7 hybrid_override      < 0.75 -> M3;  > 1.25 -> M1;  otherwise the D3 decision
```

**D7 is the primary.** If D5 scores better on the held-out period, that is reported as
an observation; the primary policy is not changed and no threshold is retuned.

## 7. Gate H, fixed in advance

```
H1  >= 3 zones, >= 180 held-out origins per zone, primary forecast coverage >= 95%,
    no leakage, time alignment ok
H2  M2 beats M1 by >= 1% with a week-clustered CI on the improvement side
H3  M3 win rate inside [20%, 80%]
H4  per-origin min(M1, M3) beats best fixed by >= 1% with a CI on the improvement side
H5  held-out Spearman(lambda_reported, true_lambda) >= 0.20 and the top reported
    quartile has >= 20% higher mean true_lambda than the bottom quartile
```

## 8. Gate I, fixed in advance

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

Subsets with fewer than 20 events are reported `NOT_EVALUABLE`, never counted as a pass.

## 9. Bootstrap

Cluster unit is the **ISO calendar week**: all zones, origins and methods inside a week
resample together. Daily origins are autocorrelated and are not independent samples.
5,000 resamples, 95% interval.

## 10. Verdict mapping

```
Gate H PASS + Gate I PASS          REAL_VINTAGE_EXTERNAL_VALIDATION_GO
Gate H PASS + Gate I INCONCLUSIVE  EXTERNAL_VALIDATION_CONDITIONAL
Gate H PASS + Gate I FAIL          SYNTHETIC_TO_REAL_METHOD_NO_GO
Gate H not PASS                    REAL_DATA_PROBLEM_NOT_ESTABLISHED
environment / API / schema         BLOCKED_EXTERNAL_DATA
```

## 11. What would invalidate this run

If an implementation error forces a change to this pre-registration or to a gate, the
run is marked `INVALID_IMPLEMENTATION`, is not overwritten, and the study restarts under
a new run id with the reason recorded.

Forbidden after any held-out quantity exists: retuning a threshold, refitting the proxy
on test data, dropping a zone because of its test performance, promoting D5 to primary,
or re-running with a different window to get a better number.
