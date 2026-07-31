# Status of Study 3 pre-registration v1

```
STATUS: INVALID_PRE_EXECUTION_AVAILABILITY_ASSUMPTION
```

`docs/study3_preregistered.md` (v1) is **superseded and must not be used** for the
held-out analysis. It is kept unmodified as the record of what was registered first;
nothing in it was deleted or rewritten.

## Why it is invalid

[확인] v1 fixed the decision origin at **06:00 UTC**, six hours after the 00 UTC ECMWF
run initialises. ECMWF's published dissemination schedule releases the 00 UTC HRES
hourly steps progressively between roughly 05:45 and 06:12 UTC, so at 06:00 UTC the
*complete* 24-hour forecast path is not guaranteed to have arrived. A decision maker at
06:00 UTC could therefore not reliably have held the entire forecast that v1 gives to
M3. That is an operational-availability error in the design, independent of any result.

[확인] It was found **before** any held-out quantity existed: at the time of this
status note there is no `runs/*_real_vintage` directory, so no M1, M2 or M3 score, no
Gate H and no Gate I had been computed. The change is therefore not a reaction to a
result.

## Second problem fixed in v2

[확인] v1 gave M3 a `future_df` while M1 had none. A future frame carries the future
timestamps themselves, so part of any M1-to-M3 difference could have come from the
model simply knowing the time-of-day and day-of-week pattern of the forecast window
rather than from the weather content. v2 gives **M1, M2 and M3 an identical future
calendar block**, so the only thing that differs between M1 and M3 is the forecasted
temperature column.

## What v2 changes

```
decision origin      06:00 UTC  ->  07:00 UTC   (00Z run + 7 h publication delay)
forecast valid slice 06..29 UTC ->  07..30 UTC  (24 hours)
revision run         previous-day 12Z (unchanged; >= 19 h before the new origin)
future frame         M3 only     ->  M1, M2, M3 all carry the same calendar block
real-data ratio name true_lambda ->  realized_weather_error_ratio
proxy output name    lambda_reported -> reported_reliability_ratio
model cycle          (absent)    ->  pre_50r1 / post_50r1 metadata, secondary only
```

The D7 override band stays 0.75 / 1.25 and the D5 threshold stays 1.00; those come from
the synthetic studies and are not retuned here.

## Continuity of the raw data

[확인] No cache was deleted. The 00Z runs already downloaded contain 168 hourly steps
starting at the run hour, so the 07..30 UTC slice is inside data that was already
fetched; the previous-day 12Z runs likewise cover it. Re-deriving the panel at 07 UTC
required **zero** new API requests. The 06 UTC processed artifacts are kept as they are,
and the 07 UTC panel is written to separate versioned files.
