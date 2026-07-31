# Study 2B pre-registration

Written **before** the runner was implemented and before any Chronos-2 inference was
executed. The machine-readable copy lives at `runs/<run_id>/preregistration.json`; its
SHA-256 is stored in that run's `manifest.json`.

Start state this pre-registration is anchored to:

```
commit         ccf629cf67376d8647e08d998ba700092829eac7
branch         main
git diff hash  9475693fa46cc2d9df77ecad5ef411d413ec17fad044fe68022841e44219cc96
```

## 1. Question

Does the admission policy **D7_hybrid_override**, fixed in advance, reproduce its
performance *and* its stable-condition safety on a completely fresh synthetic sample?

## 2. Why the previous study cannot answer this

[확인] Gate F chose its primary selector as "whichever of D5/D7 has the lower mean WQL
over the mixture" and evaluated that choice on the same sample. D7's individual success
in Study 2 is therefore a development observation, not a confirmation. Study 2B removes
the selection step: D7 is the only policy Gate G can be decided on.

## 3. Fixed decisions (no post-hoc changes permitted)

```
primary policy        D7_hybrid_override
secondary policies    D5_current_proxy, D3_history_utility, D4_history_reliability,
                      D6_hybrid_conservative
                      (D0_always_no_future, D1_always_use_future, D2_oracle are
                       references, not candidate policies)

primary proxy         P1_calibrated_noisy
secondary proxies     P0_oracle_current, P2_overconfident, P3_underconfident,
                      P4_stale_history

D7 lower threshold    0.75
D7 upper threshold    1.25
D5 threshold          1.00
proxy sigma           0.20

master seed           20260803
shares                0.25, 0.50, 0.75
horizons              24, 96
series per condition  100
bootstrap             5000 resamples, 95%, cluster unit = base_series_id
mixture               all schedules, shares, horizons and series equally weighted
```

Reliability schedules (identical to Study 2, inherited from its config file):

```
S0_stable_low          hist [0.50 0.50 0.50 0.50]  current 0.50
S1_stable_high         hist [1.50 1.50 1.50 1.50]  current 1.50
S2_sudden_worsening    hist [0.50 0.50 0.50 0.50]  current 1.50
S3_sudden_improvement  hist [1.50 1.50 1.50 1.50]  current 0.50
S4_gradual_worsening   hist [0.50 0.75 1.00 1.25]  current 1.50
S5_gradual_improvement hist [1.50 1.25 1.00 0.75]  current 0.50
```

Expected task counts:

```
primary (current-origin) tasks   3 shares x 2 horizons x 6 schedules x 100 series = 3,600
historical tasks                 the same, x 4 pseudo-origins                     = 14,400
logical M1 + M3 forecasts        (3,600 + 14,400) x 2                             = 36,000
distinct model inputs            far fewer; schedules that request an identical
                                 (origin, lambda) share one cached forecast
```

## 4. Gate G, fixed in advance

Gate G is decided on **D7 under P1 only**. All eight conditions must hold for PASS:

```
G1  D7 improves mean WQL over best fixed by >= 5%
G2  the paired cluster-bootstrap 95% CI favours D7
G3  D7 recovers >= 50% of the oracle gap
G4  D7 cuts the harm rate versus D1 always-use by >= 50%
G5  in S0_stable_low, D7 gives up <= 1% versus D1 always-use
G6  in S1_stable_high, D7 gives up <= 1% versus D0 always-no-future
G7  in S2_sudden_worsening and S4_gradual_worsening, D7 beats both D3 and D4
G8  in S3_sudden_improvement and S5_gradual_improvement, D7 beats D0
```

FAIL if any of:

```
overall improvement <= 1%; the CI favours best fixed; oracle recovery <= 20%;
harm reduction < 25%; stable_low or stable_high regression > 2%;
D7 worse than both D3 and D4 in *both* worsening conditions;
D7 worse than D0 in *both* improvement conditions
```

Otherwise INCONCLUSIVE.

Verdict mapping: PASS -> `METHOD GO`; INCONCLUSIVE -> `CONDITIONAL GO`;
FAIL -> `NO-GO CURRENT METHOD`; environment/regression/leakage problem ->
`BLOCKED` or `INVALID_RUN`.

## 5. Independence requirements

New master seed 20260803 regenerates, from scratch: base amplitudes, phases, b and x
innovations, historical eta paths, current eta paths and current proxy noise. No
generated parquet or prediction from any earlier run is read as input. Only the model
weight cache is reused.

The proxy noise namespace (`"proxy"`) is disjoint from the forecast-error namespace
(`"eta"`), so the selector's information is not correlated with the error it judges.

## 6. What would invalidate this run

If an implementation error forces a change to this pre-registration or to Gate G, the
current run is marked `INVALID_IMPLEMENTATION`, is **not** overwritten, and the study is
re-run from the start under a new run id with the reason recorded.

Explicitly forbidden after seeing results: replacing D7 with D5, changing the D7
thresholds, changing sigma_proxy, changing any schedule, changing a Gate G threshold, or
re-running with a different seed to obtain a better outcome.
