# Follow-up audit of the existing coarse pilot

Audited at commit `ccf629cf67376d8647e08d998ba700092829eac7` (branch `main`, working
tree clean). Everything below was read out of the repository itself, not from memory
of how it was built.

Reference run for the existing numbers: `runs/20260730_161002_diagnostic`, whose
report is the committed snapshot in `results/report.md`.

---

## [확인] Verified directly from code and artifacts

### Method definitions (`src/covariate_trust/schemas.py:21-25`)

```
M0 = "M0_target_only"
M1 = "M1_past_covariate_only"
M2 = "M2_oracle_future_covariate"
M3 = "M3_forecasted_future_covariate"
```

Context frames carry `id, timestamp, target` and, for everything except M0, `x`.
`future_df` carries `id, timestamp, x` and is built only for M2 and M3
(`schemas.py:56-93`).

### Fairness assertions actually execute

* `assert_context_equality(in1, in3)` runs on **every** M3 task in the diagnostic
  loop (`cli.py:511`) and in the smoke test (`cli.py:282`).
* `assert_future_equality(in2, in3)` runs whenever `lam == 0.0` (`cli.py:513`) and
  in the smoke test (`cli.py:283`).
* `future_df` containing a target column raises `SchemaError`
  (`schemas.py:112`, `schemas.py:152`); this is enforced inside
  `assert_task_invariants`, which `build_inputs` always calls.

### cross_learning

* `chronos_adapter.predict_task` passes `cross_learning=False` literally
  (`chronos_adapter.py:138`); there is no code path that passes anything else.
* `PilotModel.__post_init__` raises if a config sets `cross_learning: true`
  (`config.py:200`).
* The installed signature was read with `inspect.signature`; `audit.json` of the
  reference run records `cross_learning_supported: true, default: False`.

### Common random numbers

`generate_base_series` draws amplitudes/phases from
`(master_seed, "series_params", base_series_id)` and the two AR innovation streams
from `"innov_base"` / `"innov_cov"` namespaces (`dgp.py:71-77`). None of these depend
on `nominal_covariate_share` or `lambda`, so the same (b, x) pair is reused across the
whole grid. `PilotDGP` rejects `common_random_numbers: false` outright
(`config.py:172`).

### DGP generating equations (`dgp.py:80-117`)

```
b_raw(t) = A_b1 sin(2 pi t / 24  + phi_b1) + A_b2 sin(2 pi t / 168 + phi_b2) + u_b(t)
x_raw(t) = A_x1 sin(2 pi t / 48  + phi_x1) + A_x2 sin(2 pi t / 96  + phi_x2) + u_x(t)
b, x     = standardized with mean/std of [0, 512) only
y        = sqrt(1 - r) * b + sqrt(r) * x
```

AR(1) residuals start from the stationary distribution (`u[0] = eps[0]/sqrt(1-rho^2)`).

### Covariate error path (`dgp.py:168-196`)

```
V_raw(h)  = sigma_u^2 (1 - rho_x^(2h)) / (1 - rho_x^2)
V(h)      = V_raw(h) / scale_x^2
eta       ~ Normal(0, 1), namespace (master_seed, "eta", series_id, origin, horizon)
error     = lambda * sqrt(V(h)) * eta
x_tilde   = x_true + error
```

`eta` depends on the origin and horizon but **not** on lambda, so the lambda sweep is
paired.

### Bootstrap unit

`bootstrap.BOOTSTRAP_UNIT = "base_series_id"` (`bootstrap.py:17`), and
`paired_bootstrap` resamples unit clusters, summing each unit's observations before
averaging (`bootstrap.py:75-90`).

### Leakage guard for historical admission

`PSEUDO_ORIGINS = {24: [800, 824, 848, 872], 96: [512, 608, 704, 800]}`
(`admission.py:23-26`). `assert_no_primary_leak` raises unless
`origin + horizon <= primary_origin (896)` for every origin (`admission.py:41-52`),
and `pseudo_origins()` calls it on every access.

### Existing gate thresholds (unchanged by this follow-up)

`gates.py:23-26`

```
GATE_A_PRIMARY_SHARES = (0.50, 0.75)
NEGATIVE_CONTROL_SHARE = 0.0
LOW_NOISE_MAX_LAMBDA  = 0.5
HIGH_NOISE_MIN_LAMBDA = 1.5
```

plus the configurable `gates:` block in `configs/pilot.yaml`
(`clean_gain_pass 0.05`, `clean_gain_fail 0.01`, `oracle_headroom_pass 0.03`,
`oracle_headroom_fail 0.01`, `harm_relative_threshold 0.05`,
`high_noise_harm_rate 0.20`).

### Existing results

```
Study 0  PASS      Smoke  PASS
Gate A   PASS      Gate B PASS      Gate C PASS      Gate D PASS
master_seed 20260730     coarse grid: share [0, .25, .5, .75] x lambda [0, .5, 1, 1.5, 2]
                         x horizon [24, 96] x 30 series = 1,200 M3 tasks
```

V_future = WQL(M1) - WQL(M3) at the three lambdas nearest the sign change:

```
share  H     lam=0.5     lam=1.0     lam=1.5
──────────────────────────────────────────────
0.25   24    +0.03995    +0.01798    -0.03046
0.25   96    +0.04027    +0.00074    -0.06585
0.50   24    +0.10757    +0.00366    -0.15073
0.50   96    +0.09762    -0.00867    -0.14426
0.75   24    +0.20926    +0.01912    -0.20685
0.75   96    +0.17285    +0.00185    -0.19423
```

All six curves change sign between lambda 0.5 and 1.5. Four of six are still positive
at lambda 1.0 and two are already negative, so the coarse grid brackets the crossing
but does not locate it.

Baseline test suite at audit time: **69 passed**.

---

## [의심] What the follow-up has to test rather than assume

1. **The crossing location is not measured.** The coarse grid has no lambda between
   0.5 and 1.5, so "the boundary is near lambda = 1" is an interpolation across a
   wide gap, from 30 series per cell. Study 1B must estimate it on an independent
   seed with a dense grid and report an interval, including the possibility that it
   is censored outside [0.70, 1.30].
2. **Sign at lambda = 1.0 is not consistent** across the six curves (four positive,
   two negative), and several coarse cells had CIs covering zero. Whether that is
   sampling noise or a real share/horizon dependence is open.
3. **The coincidence with the Study-0 reference line is unexplained.** Study 0's
   crossing at lambda = 1 is an MSE property of a linear-Gaussian model. Nothing
   verified that a WQL crossing under Chronos-2 must sit in the same place. Study 1B
   plus the statistical baselines (B1, B2) exist to separate "property of noisy
   plug-in covariates in general" from "behaviour of this model".
4. **Gate D used the same lambda at the pseudo-origins and at the primary origin.**
   `run_historical` calls `covariate_vintage(..., origin, h, lam)` with the same
   `lam` that the primary task uses (`admission.py:105-112`). A selector reading
   history therefore observed exactly the reliability it was about to face - the
   easiest possible case. Gate D PASS says nothing about time-varying reliability.
5. **A2's `lambda_hat` is computed from the true error path**
   (`estimate_lambda_hat(v["error"], v["V"])`). In history that is legitimate (both
   the forecast and the realized covariate are observable in the past), but it means
   the existing pipeline has no notion of a *reported* current uncertainty. Study 2
   must add an explicit proxy channel and prove the selector never reads the true
   current lambda outside the P0 oracle diagnostic.
6. **Only one forecast origin** feeds the primary comparison, and only 30 series per
   cell, so cell-level Monte-Carlo SE (0.0033-0.0461) is large relative to several
   effects.

---

## [변경 금지] Must stay identical for the existing results to remain reproducible

* `dgp.generate_base_series`, `build_target`, `conditional_variance_raw`,
  `conditional_variance_path`, `eta_path`, `covariate_vintage` - generating equations
  and seed namespaces.
* `schemas.build_inputs` and every assertion in `assert_task_invariants`.
* `chronos_adapter.predict_task` (including the literal `cross_learning=False`) and
  `task_hash`.
* `metrics.wql`, `pinball`, `nmae`, `mse`, `quantile_crossing_rate`, `is_harm`,
  `relative_delta`.
* `bootstrap.paired_bootstrap` and `BOOTSTRAP_UNIT`.
* `gates.gate_a/gate_b/gate_c` and their constants; `admission.PSEUDO_ORIGINS`,
  `assert_no_primary_leak`, `build_decisions`, `gate_d`.
* `configs/pilot.yaml`, `configs/study0.yaml`.
* Everything under `runs/` and `results/`.

The follow-up therefore **adds** modules and config files and appends CLI commands.
It does not edit any of the functions listed above. New configuration types are added
to `config.py` as new classes; `PilotConfig` and `Study0Config` are untouched, and the
new experiments build a `PilotConfig` internally so that the DGP code path is literally
the same one the coarse pilot used.
