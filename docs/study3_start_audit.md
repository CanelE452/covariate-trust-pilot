# Study 3 start-state audit

Recorded before any Study 3 code was written and before any external data was fetched
for the experiment itself (the source probes below fetched only a few bytes each, to
establish schema).

## Repository state at start

```
pwd            /home/minjae/Documents/github/timeseries
branch         main
commit         ccf629cf67376d8647e08d998ba700092829eac7
git diff hash  1ca88dfcf38823c7b3d2548213328232bdd2e726b42d1cd15b7c0fa02bd0d5fc
```

`git diff --stat` at start: 5 files changed, 2370 insertions(+), 0 deletions(-)
(`cli.py`, `config.py`, `plotting.py`, `reporting.py`, `tests/conftest.py`).

Untracked at start: `configs/study1b_boundary.yaml`,
`configs/study2_dynamic_reliability.yaml`, `configs/study2b_d7_confirmation.yaml`,
`docs/`, `src/covariate_trust/{baselines,boundary,confirmation,dynamic_admission,followup_gates,reliability_schedules}.py`,
`tests/test_{baselines,boundary,confirmation,dynamic_admission,followup_gates,reliability_schedules}.py`.

The working tree is dirty because Study 1B, Study 2 and Study 2B are implemented but
not committed. That state is preserved: no `git reset`, `restore`, `checkout`, `stash`
or `clean` was run, and nothing under `runs/` or `results/` was modified.

## Existing tests at start

```
.venv/bin/python -m pytest   ->   161 passed
```

No existing regression, so Study 3 may proceed.

## Prior results this study builds on

```
Gate E   PASS          boundary reproduces on an independent synthetic seed
Gate F   INCONCLUSIVE  development study; its primary was D5, not D7
Gate G   PASS          held-out synthetic confirmation of the pre-registered D7
                       (runs/20260730_224528_d7_confirmation, prereg sha256 f9d4cb0e...)
```

Status split carried into this study:

```
Synthetic method validation   GO
General deployment claim      CONDITIONAL
```

D7 thresholds 0.75 / 1.25 and the D5 threshold 1.00 come from the synthetic studies and
are fixed here; they are not re-tuned on real data.

## Source probes (measured, not assumed)

All four probes were executed against the live endpoints before writing any Study 3
module. Details in `docs/study3_data_sources.md`.

```
NYISO P-58C index          HTTP 200, text/html, 312 hrefs, title "Integrated Real-Time Actual Load"
NYISO monthly ZIP          csv/palIntegrated/YYYYMM01palIntegrated_csv.zip  (302 links)
NYISO CSV schema           "Time Stamp","Time Zone","Name","PTID","Integrated Load"
                           already hourly (24 rows per zone per day); Time Zone column
                           carries EDT/EST explicitly
NYISO zones present        CAPITL CENTRL DUNWOD GENESE HUD VL LONGIL MHK VL MILLWD
                           N.Y.C. NORTH WEST   -> all four configured zones exist
Open-Meteo single runs     requires BOTH `run=<ISO datetime>` and `models=ecmwf_ifs`
                           `models=ecmwf_ifs025` is NOT available; the default model is
                           dwd_icon, so the model parameter must be passed explicitly
                           returns 168 hourly steps starting at the run hour, timezone GMT
run archive depth          00Z runs verified available for 2024-04-01 ... 2026-06-30
18Z runs                   available (checked 2025-07-01T18:00)
multi-coordinate batch     supported: passing 4 lat/lon pairs returns a JSON *list* of 4
                           location objects
Archive (verification)     HTTP 200, hourly temperature_2m, timezone GMT
```

The initial guess `models=ecmwf_ifs` happened to be correct, but `run` being mandatory
and `ecmwf_ifs025` being unavailable were both discovered from the API's own error
messages, not assumed.
