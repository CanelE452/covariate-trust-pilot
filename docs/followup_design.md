# Follow-up design: Study 1B and Study 2

Written before the runs.  Every threshold named here is fixed in the config files and
is not revisited after seeing a result.

## Questions

```
Q1  Does the lambda ~ 1 boundary reproduce on an independent seed with many replicates?
Q2  Is that boundary specific to Chronos-2, or a statistical property of any noisy
    plug-in covariate?
Q3  When past and present covariate quality differ, which admission rule actually
    reduces harm?
```

Q1 is Study 1B / Gate E.  Q2 is the B1 and B2 baselines.  Q3 is Study 2 / Gate F.

## What is reused unchanged

Study 1B and Study 2 call the coarse pilot's own functions: `dgp.generate_base_series`,
`build_target`, `covariate_vintage`, `eta_path`, `schemas.build_inputs` and its
assertions, `chronos_adapter.predict_task` (with its literal `cross_learning=False`),
`metrics.wql`, `bootstrap.paired_bootstrap`.

Neither follow-up config restates the `dgp` or `gates` block.  `BoundaryConfig.load`
and `DynamicConfig.load` read those two blocks out of `configs/pilot.yaml` and build a
real `PilotConfig` internally, so the generating equations cannot silently drift from
the ones that produced the existing results.  `test_f09` asserts that with the pilot's
seed the follow-up path reproduces the pilot's series bit for bit, and `test_f36b`
asserts the inherited blocks are equal to the pilot's.

## Study 1B - what changes

```
                  coarse pilot            Study 1B
master_seed       20260730                20260801
lambda grid       0, 0.5, 1, 1.5, 2       0.70, 0.85, 1.00, 1.15, 1.30
shares            0, 0.25, 0.5, 0.75      0.25, 0.5, 0.75
series per cell   30                      150
bootstrap         2000                    5000
```

The boundary estimator interpolates the descending zero crossing of
`V_future(lambda) = WQL(M1) - WQL(M3)` between the two adjacent grid points that
bracket it, and gets its interval from resampling `base_series_id` clusters and
re-estimating the crossing in each resample.  Curves that never cross inside the grid
are labelled `left_censored` / `right_censored` / `unresolved`; nothing is
extrapolated.  `bootstrap_valid_fraction` records how often a resample produced a
usable crossing.

## Baselines - what they are for

`V_future` under Chronos is a WQL quantity.  Under B1 and B2 it is an MSE quantity.
Those are different estimands and the report never equates their numbers; the
comparison is about *direction and location*.

B1 (DGP-aware conditional mean) has a closed form.  The b-process error is identical
under M1 and M3 and cancels, leaving

```
E[MSE(B1-M1) - MSE(B1-M3)] = r * (1 - lambda^2) * mean_h V(h)
```

which is exactly zero at `lambda = 1`.  That is the known answer the implementation is
checked against (`BC1`).  The *point* crossing estimated from realized data is a
finite-sample quantity and drifts below 1 at small n - measured at 0.73 (n=4), 0.87
(n=20), 0.95 (n=80) for H=24 - so the second check (`BC1b`) asks whether the bootstrap
interval covers 1 rather than whether a point estimate lands within a fixed distance.
A fixed +-0.10 tolerance was written first and replaced, because it accepts or rejects
identical code depending only on sample size.

B2 (estimated ARX) is deliberately misspecified and fits every coefficient on the
context window with a fixed ridge of 1e-4.  It is never tuned on the evaluation window;
`test_f14` and `test_f15` poison the future target and the pre-context history to prove
neither can move the fit or the forecast.

## Study 2 - what changes relative to Gate D

Gate D gave the selector the same lambda in history and at the primary origin.  Study 2
assigns each origin its own lambda from a declared schedule:

```
S0 stable_low          hist [0.50 0.50 0.50 0.50]  current 0.50
S1 stable_high         hist [1.50 1.50 1.50 1.50]  current 1.50
S2 sudden_worsening    hist [0.50 0.50 0.50 0.50]  current 1.50
S3 sudden_improvement  hist [1.50 1.50 1.50 1.50]  current 0.50
S4 gradual_worsening   hist [0.50 0.75 1.00 1.25]  current 1.50
S5 gradual_improvement hist [1.50 1.25 1.00 0.75]  current 0.50
```

The target-covariate relationship stays stationary; only reliability moves, so the
effect is isolated.

The selector no longer sees the truth.  A provider reports a proxy:

```
P0 oracle_current    = true current lambda                    (diagnostic only)
P1 calibrated_noisy  = lambda * exp(z - sigma^2/2), sigma=0.2  (Gate F is decided here)
P2 overconfident     = 0.50 * P1
P3 underconfident    = 1.50 * P1
P4 stale_history     = mean of the historical lambda estimates
```

The proxy noise is drawn from a `"proxy"` seed namespace that is disjoint from the
`"eta"` namespace used for the covariate forecast error, so the selector's information
is not correlated with the error it is judging (`test_f25`).

Selectors D0-D7 each pick exactly one of M1 or M3; quantile forecasts are never blended.
`test_f31` and `test_f32` multiply the current outcomes by arbitrary factors and assert
that only the oracle D2 changes its choices - that is the operational proof that no
selector reads the current target.

## Inference budget

Proxy modes change only the decision, so a proxy never triggers a new forecast.  Across
schedules many (origin, lambda) pairs coincide - S0 and S2 share their whole history,
S1 and S3 likewise - so inference is cached by a content hash of the model input
(model id, both frames).  Two tasks share a cache entry only when the model would see
byte-identical input.

```
Study 1B   3 shares x 2 horizons x 150 series x (M1 + M2 + 5 x M3)  = 6,300 calls
Study 2    6 schedules x 3 shares x 2 horizons x 50 series x 5 origins x 2 methods
           = 18,000 logical tasks, of which ~6,000 are distinct inputs
```

## Gate order

Gate E FAIL stops Study 2 (`NO-GO PHENOMENON`).  Gate E INCONCLUSIVE also stops Study 2;
the run reports the replication count each cell would need for a target half-width and
executes nothing extra.  Only Gate E PASS lets Study 2 run.

Gate D keeps its name, its definition and its recorded result.  Gate F is reported as a
separate question and never overwrites it.
