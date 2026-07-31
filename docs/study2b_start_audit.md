# Study 2B start-state audit

Recorded before any new code was written and before any Chronos inference was run.

## Repository state at start

```
pwd            /home/minjae/Documents/github/timeseries
branch         main
commit         ccf629cf67376d8647e08d998ba700092829eac7
remote         https://github.com/CanelE452/covariate-trust-pilot.git
git diff hash  9475693fa46cc2d9df77ecad5ef411d413ec17fad044fe68022841e44219cc96
               (sha256 of `git diff`, i.e. of the tracked-file modifications only)
```

`git status --short`

```
 M src/covariate_trust/cli.py
 M src/covariate_trust/config.py
 M src/covariate_trust/plotting.py
 M src/covariate_trust/reporting.py
 M tests/conftest.py
?? configs/study1b_boundary.yaml
?? configs/study2_dynamic_reliability.yaml
?? docs/
?? src/covariate_trust/baselines.py
?? src/covariate_trust/boundary.py
?? src/covariate_trust/dynamic_admission.py
?? src/covariate_trust/followup_gates.py
?? src/covariate_trust/reliability_schedules.py
?? tests/test_baselines.py
?? tests/test_boundary.py
?? tests/test_dynamic_admission.py
?? tests/test_followup_gates.py
?? tests/test_reliability_schedules.py
```

`git diff --stat`: 5 files changed, 1452 insertions(+), 0 deletions(-).

The working tree is dirty because Study 1B and Study 2 are implemented but not
committed. That state is preserved: no `git reset`, `restore`, `checkout`, `stash` or
`clean` was executed at any point, and nothing under `runs/` or `results/` was modified.

## Existing tests at start

```
.venv/bin/python -m pytest   ->   126 passed
```

## Existing results, read from the artifacts (not from memory)

Source runs: `runs/20260730_171941_boundary`, `runs/20260730_172249_dynamic`.

```
Gate E                    PASS                         (tables/gate_e.json)
Gate F                    INCONCLUSIVE                 (tables/gate_f.json)
Gate F primary selector   D5_current_proxy             <- not D7
Gate F primary proxy      P1_calibrated_noisy
```

[확인] Gate E is PASS, which is the precondition for running Study 2B at all.

### D7 as developed in Study 2

From `gate_f.json -> per_selector_calibrated_proxy.D7_hybrid_override`:

```
mean WQL                            0.44075
improvement over best fixed         +0.1041
oracle gap recovery                 0.7690
harm rate                           0.074
```

[확인] In that study D7 satisfied every individual Gate F safety condition, but the
pre-registered rule for Gate F picked whichever of D5/D7 had the lower mean WQL over
the mixture, and that was D5 (0.43377). Gate F was therefore decided on D5 and returned
INCONCLUSIVE because D5 gave up 1.335% against always-no-future in `S1_stable_high`.

[확인] This is exactly why Study 2 is a *development* study for D7 and not a
confirmation: D7's apparent success was observed on the same sample that was used to
compare it against D5.

### Values read out of `runs/20260730_172249_dynamic/config_resolved.yaml`

```
selector thresholds   use 1.0, override_low 0.75, override_high 1.25
proxy sigma           0.20
schedules
   S0_stable_low            hist [0.50 0.50 0.50 0.50]  current 0.50
   S1_stable_high           hist [1.50 1.50 1.50 1.50]  current 1.50
   S2_sudden_worsening      hist [0.50 0.50 0.50 0.50]  current 1.50
   S3_sudden_improvement    hist [1.50 1.50 1.50 1.50]  current 0.50
   S4_gradual_worsening     hist [0.50 0.75 1.00 1.25]  current 1.50
   S5_gradual_improvement   hist [1.50 1.25 1.00 0.75]  current 0.50
```

### Leakage guards verified in code

```
admission.py:23        PSEUDO_ORIGINS = {24: [800, 824, 848, 872], 96: [512, 608, 704, 800]}
admission.py:41-48     assert_no_primary_leak raises unless origin + horizon <= 896
dynamic_admission.py:183-185
                       D7 reads only `reported_lambda` and `hist_supports_m3`;
                       no current-outcome column enters its decision
```

## What Study 2B fixes in advance

Fixed before writing the runner and before any inference:

```
primary policy        D7_hybrid_override          (never replaced, even if D5 scores better)
primary proxy         P1_calibrated_noisy         (Gate G is decided on P1 only)
D7 lower threshold    0.75
D7 upper threshold    1.25
D5 threshold          1.00
proxy sigma           0.20
master seed           20260803                    (Study 2 used 20260802)
schedules             the six above, unchanged
Gate G                8 PASS conditions, fixed in configs/study2b_d7_confirmation.yaml
```

Study 2B inherits the six schedules from `configs/study2_dynamic_reliability.yaml` and
the `gates` block from `configs/pilot.yaml`, so the schedule definitions and the harm
threshold cannot drift from the ones already used.
