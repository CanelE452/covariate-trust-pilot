# Forecasted future covariates in zero-shot Chronos-2 - coarse pilot

Run directory: `/home/minjae/Documents/github/timeseries/runs/20260730_161002_diagnostic`

## 1. Research question

When does a *forecasted* (imperfect) future covariate add value to a zero-shot
Chronos-2 forecast, and when does it actively hurt?  The incremental value of the
future covariate is defined as the paired difference M1 - M3.  M0 is a sanity
baseline only: the M0 - M3 difference is **not** interpreted as the value of the
future covariate, because it also contains the value of the past covariate. [확인]

## 2. Project and environment

```
project_root       /home/minjae/Documents/github/timeseries
python             3.13.9  (/home/minjae/Documents/github/timeseries/.venv/bin/python)
in_virtualenv      True
torch              2.5.1+cu121  cuda_build 12.1  available True
gpu                NVIDIA GeForce RTX 3080
chronos-forecasting 2.3.1
transformers       4.57.6
git commit         UNBORN
HF_HOME            /home/minjae/Documents/github/timeseries/.cache/huggingface
cross_learning     supported=True default=False  (always passed False)
```

[확인] The `predict_df` signature above was read with `inspect.signature` from the
installed package, not from documentation.

<details><summary>resolved configuration</summary>

```yaml
experiment:
  name: imperfect_future_covariate_coarse_pilot
  master_seed: 20260730
  series_length: 1024
  standardization_end: 512
  primary_origin: 896
  context_length: 512
  frequency: h
  quantile_levels:
  - 0.1
  - 0.2
  - 0.3
  - 0.4
  - 0.5
  - 0.6
  - 0.7
  - 0.8
  - 0.9
grid:
  nominal_covariate_share:
  - 0.0
  - 0.25
  - 0.5
  - 0.75
  lambda_values:
  - 0.0
  - 0.5
  - 1.0
  - 1.5
  - 2.0
  horizons:
  - 24
  - 96
  n_series_per_cell: 30
dgp:
  base_ar: 0.5
  covariate_ar: 0.7
  base_periods:
  - 24
  - 168
  covariate_periods:
  - 48
  - 96
  ar_innovation_std: 0.5
  common_random_numbers: true
model:
  model_id: amazon/chronos-2
  package_version: 2.3.1
  frozen: true
  cross_learning: false
  device: cuda
  allow_cpu_smoke: true
  allow_cpu_diagnostic: false
  attention_implementation: eager
bootstrap:
  n_resamples: 2000
  confidence_level: 0.95
gates:
  clean_gain_pass: 0.05
  clean_gain_fail: 0.01
  oracle_headroom_pass: 0.03
  oracle_headroom_fail: 0.01
  harm_relative_threshold: 0.05
  high_noise_harm_rate: 0.2
  dose_response_curve_fraction: 0.7
  negative_control_ratio: 0.5
  degenerate_policy_share: 0.95
  admission_pass_improvement: 0.02
  admission_fail_improvement: 0.0
  admission_recovery_pass: 0.3
  admission_recovery_fail: 0.1
  admission_harm_reduction_pass: 0.25
  admission_low_noise_regression_max: 0.01
```

</details>

## 3. Study 0 - known-answer simulation

Status: **PASS**

[확인] Maximum relative error between simulated and analytic MSE: 0.00454.

- `C1_analytic_agreement` **PASS** - every simulated MSE within 1% of its closed form
  - max relative error = 0.00454
- `C2_crossing_near_lambda_one` **PASS** - prior-only and plug-in cross in the neighbourhood of lambda = 1
  - plugin<prior for all lambda<1: True; plugin>prior for all lambda>1: True; relative gap at lambda=1: 0.00069
- `C3_posterior_dominance` **PASS** - posterior predictor never worse than either fixed predictor
  - tolerance 1%; violations: 0
- `C4_lambda_zero_identity` **PASS** - at lambda = 0 the posterior predictor equals the plug-in predictor
  - |posterior - plugin| at lambda=0 = 0.000e+00

[확인] w = V / (V + sigma_e^2) = 1 / (1 + lambda^2) is the exact posterior
coefficient of the Study-0 linear-Gaussian model only.  It is not presented as
the optimal way to mix a Chronos quantile forecast.

## 4. Data generating process

[확인] All series are synthetic and generated in-process from the master seed; no
external dataset is read at any point.  Per base series a base process `b` and a
covariate process `x` are built from two sinusoids plus an AR(1) residual, then
standardized using statistics from `t in [0, standardization_end)` only.  The target
is `y = sqrt(1-r) * b + sqrt(r) * x`.

[확인] `r` is the *nominal* covariate share by construction.  It is deliberately not
called a partial R^2; the realized incremental R^2 of `x` given `b` is measured
separately and reported in `generated/series_metadata.parquet` and Figure 6.

[확인] Covariate forecast error: `x_tilde(T+h) = x_true(T+h) + lambda * sqrt(V(h)) * eta_h`
with `V(h)` the h-step AR(1) conditional variance on the standardized scale and
`eta_h` a standard normal path shared by every lambda at the same (series, origin,
horizon).  Error is unbiased and serially uncorrelated by construction; biased and
correlated error models are out of scope for this pilot.

## 5. Estimand

```
V_future = WQL(M1) - WQL(M3)     positive: forecasted future covariate helps
V_oracle = WQL(M1) - WQL(M2)     positive: an accurate future covariate helps
relative_delta = (WQL(M3) - WQL(M1)) / WQL(M1)
harm     = 1 if WQL(M3) > 1.05 * WQL(M1)
```

[확인] Every difference is paired at the level of `base_series_id`: the same base and
covariate processes, phases, amplitudes, AR innovations and eta path are reused
across all grid cells (common random numbers).  The bootstrap unit is therefore also
`base_series_id`, resampled as a cluster.

## 6. Comparison conditions

```
M0 target_only                 context: target                      future: none
M1 past_covariate_only         context: target + x history          future: none
M2 oracle_future_covariate     context: as M1                       future: true x
M3 forecasted_future_covariate context: as M1                       future: x_tilde
```

[확인] Fairness assertions executed on every task: M1 and M3 contexts are compared for
exact equality; M2 and lambda=0 M3 future frames are compared for exact equality; M1
carries no `future_df`; M0 carries no covariate column; no `future_df` ever contains
the target column.  `cross_learning=False` is passed on every call.

## 7. Monte-Carlo precision

```
share    H     lam    n    mean_diff   sd         mc_se      95%_half   
────────────────────────────────────────────────────────────────────────
0.0000   24.00000.0000 30.0000-0.0105     0.0257     0.0047     0.0092     
0.0000   24.00000.5000 30.0000-0.0091     0.0263     0.0048     0.0094     
0.0000   24.00001.0000 30.0000-0.0079     0.0275     0.0050     0.0098     
0.0000   24.00001.5000 30.0000-0.0106     0.0324     0.0059     0.0116     
0.0000   24.00002.0000 30.0000-0.0171     0.0403     0.0074     0.0144     
0.2500   24.00000.0000 30.00000.0423      0.0677     0.0124     0.0242     
0.2500   24.00000.5000 30.00000.0399      0.0631     0.0115     0.0226     
0.2500   24.00001.0000 30.00000.0180      0.0711     0.0130     0.0255     
0.2500   24.00001.5000 30.0000-0.0305     0.0916     0.0167     0.0328     
0.2500   24.00002.0000 30.0000-0.1084     0.1136     0.0207     0.0407     
0.5000   24.00000.0000 30.00000.1421      0.1207     0.0220     0.0432     
0.5000   24.00000.5000 30.00000.1076      0.1206     0.0220     0.0432     
0.5000   24.00001.0000 30.00000.0037      0.1406     0.0257     0.0503     
0.5000   24.00001.5000 30.0000-0.1507     0.1726     0.0315     0.0618     
0.5000   24.00002.0000 30.0000-0.3258     0.2040     0.0372     0.0730     
0.7500   24.00000.0000 30.00000.2960      0.2056     0.0375     0.0736     
0.7500   24.00000.5000 30.00000.2093      0.1997     0.0365     0.0715     
0.7500   24.00001.0000 30.00000.0191      0.1988     0.0363     0.0711     
0.7500   24.00001.5000 30.0000-0.2068     0.2152     0.0393     0.0770     
0.7500   24.00002.0000 30.0000-0.4396     0.2525     0.0461     0.0903     
0.0000   96.00000.0000 30.0000-0.0064     0.0213     0.0039     0.0076     
0.0000   96.00000.5000 30.0000-0.0069     0.0180     0.0033     0.0064     
0.0000   96.00001.0000 30.0000-0.0167     0.0178     0.0033     0.0064     
0.0000   96.00001.5000 30.0000-0.0448     0.0270     0.0049     0.0096     
0.0000   96.00002.0000 30.0000-0.0745     0.0342     0.0062     0.0122     
0.2500   96.00000.0000 30.00000.0505      0.0430     0.0079     0.0154     
0.2500   96.00000.5000 30.00000.0403      0.0435     0.0079     0.0156     
0.2500   96.00001.0000 30.00000.0007      0.0474     0.0086     0.0169     
0.2500   96.00001.5000 30.0000-0.0659     0.0644     0.0118     0.0230     
0.2500   96.00002.0000 30.0000-0.1364     0.0821     0.0150     0.0294     
0.5000   96.00000.0000 30.00000.1367      0.0615     0.0112     0.0220     
0.5000   96.00000.5000 30.00000.0976      0.0599     0.0109     0.0215     
0.5000   96.00001.0000 30.0000-0.0087     0.0643     0.0117     0.0230     
0.5000   96.00001.5000 30.0000-0.1443     0.0891     0.0163     0.0319     
0.5000   96.00002.0000 30.0000-0.2726     0.1189     0.0217     0.0426     
0.7500   96.00000.0000 30.00000.2499      0.0809     0.0148     0.0289     
0.7500   96.00000.5000 30.00000.1729      0.0771     0.0141     0.0276     
0.7500   96.00001.0000 30.00000.0018      0.0745     0.0136     0.0267     
0.7500   96.00001.5000 30.0000-0.1942     0.0897     0.0164     0.0321     
0.7500   96.00002.0000 30.0000-0.3868     0.1212     0.0221     0.0434     
```

[확인] `mean_diff` is the paired V_future = WQL(M1) - WQL(M3) inside the cell.
Cells whose effect is smaller than roughly twice `mc_se` cannot be resolved at
this sample size; that is a precision statement, not a null result.

## 8. Gate A - does an accurate future covariate help at all?

Status: **PASS**

Operative criteria (fixed before the run):

```
comparison: M2 (oracle future covariate) vs M1 (past covariate only)
primary_subset: nominal share in [0.5, 0.75], horizons [24, 96]
pass: both horizons improve, aggregate relative WQL improvement >= 5%, paired bootstrap CI strictly on the improvement side, and no meaningful gain at r=0
fail: aggregate improvement <= 1%, or the CI lies strictly on the degradation side, or the r=0 negative control shows a similar gain (>= 50% of the primary gain and >= 1%)
note: M2 is lambda-invariant; the M2 rows are deduplicated to one per (series, share, horizon) so the aggregate is not lambda-weighted.
```

Measured:

```
all_horizons_improve: True
aggregate_relative_improvement: 0.412676
aggregate_improvement_meets_pass: True
ci_favours_m2: True
ci_favours_m1: False
negative_control_relative_improvement: -0.020582
negative_control_clean: True
negative_control_similar_gain: False
```


## 9. Gate B - is there a benefit-to-harm boundary?

Status: **PASS**

Operative criteria (fixed before the run):

```
comparison: M3 (forecasted future covariate) vs M1 (past covariate only)
v_future: V_future = WQL(M1) - WQL(M3); positive means the future covariate helped
pass: at least 2 (share, horizon) curves with mean V_future > 0 for lambda <= 0.5 and mean V_future < 0 for lambda >= 1.5 on the same curve; >= 70% of curves decreasing in lambda (Spearman rho < 0); at least one high-noise cell with harm rate >= 20%
fail: no curve shows a low-noise benefit, or no curve shows a high-noise harmful region, or fewer than half of the curves decrease in lambda
harm_definition: WQL(M3) > 1.05 * WQL(M1) on a task
note: lambda = 1 is a reference line taken from the Study-0 linear-Gaussian model; it is not a theoretical WQL boundary.
```

Measured:

```
n_curves: 8
n_curves_with_boundary: 6
fraction_curves_decreasing: 1.000000
any_low_noise_benefit: True
any_high_noise_harm: True
max_high_noise_harm_rate: 1.000000
harm_rate_requirement_met: True
```


## 10. Gate C - is there oracle admission headroom?

Status: **PASS**

Operative criteria (fixed before the run):

```
oracle: per task: min(WQL(M1), WQL(M3)) - an upper bound on any selector
fixed_policies: always_no_future = M1 everywhere; always_use_future = M3 everywhere
best_fixed: the fixed policy with the lower mean WQL on this equally weighted grid
pass: oracle improves on best fixed by >= 3%, paired bootstrap CI strictly on the improvement side, and both M1 and M3 win somewhere at >= 2 share levels
fail: oracle headroom <= 1%, or one policy wins on >= 95% of tasks
weighting_caveat: the grid weights every (share, lambda, horizon) cell equally; this is a design choice and does not represent any deployment frequency
```

Measured:

```
oracle_headroom: 0.105597
headroom_meets_pass: True
ci_favours_oracle: True
shares_with_both_winners: 4
m3_overall_win_rate: 0.436667
degenerate_policy: False
```


## 11. Gate D - historical admission

Status: **PASS**

Operative criteria (fixed before the run):

```
selectors:
  A1_historical_utility: use M3 now iff mean historical WQL(M3) < mean historical WQL(M1)
  A2_historical_reliability: use M3 now iff the historical vintage error implies lambda_hat < 1; an analytic-inspired heuristic, not a WQL-optimal rule
pass: at least one selector beats the best fixed policy by >= 2%, recovers >= 30% of the oracle gap, cuts the harm rate versus M3 by >= 25%, gives up <= 1% versus M3 in low-noise cells, and has a paired CI on the improvement side
fail: both selectors are worse than the best fixed policy, or recovery <= 10%, or the low-noise benefit is largely removed
leakage_guard: selection uses pseudo-origins that close at or before the primary origin; the primary future target never enters a decision
```

```
selector                     mean_WQL   impr_vs_fixed  recovery  harm_rate  m3_rate
────────────────────────────────────────────────────────────────────────────────────
A1_historical_utility        0.42694    0.0834         0.7894    0.0467     0.3800
A2_historical_reliability    0.42797    0.0811         0.7685    0.0750     0.4800
```

[확인] A2 is an analytic-inspired reliability heuristic (lambda_hat < 1), not a
WQL-optimal rule.

## 12. Observations

Cell-level summary (mean over base series):

```
share    H     lam    WQL_M1    WQL_M2    WQL_M3    V_future   CI_low     CI_high    harm    m3_win  
─────────────────────────────────────────────────────────────────────────────────────────────────────
0.0000   24.00000.0000 0.4241    0.4346    0.4346    -0.0105    -0.0196    -0.0013    0.3667  0.3000  
0.0000   24.00000.5000 0.4241    0.4346    0.4332    -0.0091    -0.0184    0.0004     0.3000  0.3333  
0.0000   24.00001.0000 0.4241    0.4346    0.4319    -0.0079    -0.0169    0.0024     0.3333  0.3000  
0.0000   24.00001.5000 0.4241    0.4346    0.4347    -0.0106    -0.0220    0.0012     0.4000  0.3000  
0.0000   24.00002.0000 0.4241    0.4346    0.4412    -0.0171    -0.0312    -0.0022    0.5000  0.2667  
0.2500   24.00000.0000 0.4628    0.4205    0.4205    0.0423     0.0189     0.0677     0.1000  0.7667  
0.2500   24.00000.5000 0.4628    0.4205    0.4229    0.0399     0.0194     0.0627     0.1667  0.7000  
0.2500   24.00001.0000 0.4628    0.4205    0.4449    0.0180     -0.0076    0.0436     0.2333  0.6333  
0.2500   24.00001.5000 0.4628    0.4205    0.4933    -0.0305    -0.0621    0.0020     0.5000  0.3333  
0.2500   24.00002.0000 0.4628    0.4205    0.5712    -0.1084    -0.1493    -0.0676    0.7667  0.1000  
0.5000   24.00000.0000 0.4995    0.3574    0.3574    0.1421     0.1046     0.1896     0.0000  1.0000  
0.5000   24.00000.5000 0.4995    0.3574    0.3919    0.1076     0.0676     0.1520     0.1333  0.8333  
0.5000   24.00001.0000 0.4995    0.3574    0.4959    0.0037     -0.0462    0.0528     0.3333  0.5667  
0.5000   24.00001.5000 0.4995    0.3574    0.6502    -0.1507    -0.2099    -0.0897    0.8000  0.1333  
0.5000   24.00002.0000 0.4995    0.3574    0.8253    -0.3258    -0.4007    -0.2567    0.9667  0.0333  
0.7500   24.00000.0000 0.5537    0.2578    0.2578    0.2960     0.2306     0.3741     0.0000  1.0000  
0.7500   24.00000.5000 0.5537    0.2578    0.3445    0.2093     0.1434     0.2854     0.0667  0.9000  
0.7500   24.00001.0000 0.5537    0.2578    0.5346    0.0191     -0.0505    0.0934     0.4333  0.5333  
0.7500   24.00001.5000 0.5537    0.2578    0.7606    -0.2068    -0.2852    -0.1363    0.8333  0.1000  
0.7500   24.00002.0000 0.5537    0.2578    0.9933    -0.4396    -0.5333    -0.3537    0.9667  0.0000  
0.0000   96.00000.0000 0.4005    0.4070    0.4070    -0.0064    -0.0140    0.0003     0.2333  0.4667  
0.0000   96.00000.5000 0.4005    0.4070    0.4074    -0.0069    -0.0133    -0.0006    0.2333  0.4000  
0.0000   96.00001.0000 0.4005    0.4070    0.4173    -0.0167    -0.0231    -0.0105    0.4000  0.1000  
0.0000   96.00001.5000 0.4005    0.4070    0.4453    -0.0448    -0.0542    -0.0355    0.8000  0.0333  
0.0000   96.00002.0000 0.4005    0.4070    0.4751    -0.0745    -0.0862    -0.0629    0.9333  0.0000  
0.2500   96.00000.0000 0.4402    0.3897    0.3897    0.0505     0.0351     0.0658     0.0333  0.8667  
0.2500   96.00000.5000 0.4402    0.3897    0.3999    0.0403     0.0243     0.0550     0.1000  0.8667  
0.2500   96.00001.0000 0.4402    0.3897    0.4395    0.0007     -0.0155    0.0176     0.4333  0.4667  
0.2500   96.00001.5000 0.4402    0.3897    0.5060    -0.0659    -0.0896    -0.0441    0.6667  0.2000  
0.2500   96.00002.0000 0.4402    0.3897    0.5766    -0.1364    -0.1654    -0.1093    0.9333  0.0333  
0.5000   96.00000.0000 0.4649    0.3282    0.3282    0.1367     0.1158     0.1580     0.0000  1.0000  
0.5000   96.00000.5000 0.4649    0.3282    0.3673    0.0976     0.0783     0.1196     0.0333  0.9667  
0.5000   96.00001.0000 0.4649    0.3282    0.4736    -0.0087    -0.0304    0.0134     0.3667  0.3667  
0.5000   96.00001.5000 0.4649    0.3282    0.6092    -0.1443    -0.1745    -0.1149    0.8667  0.0667  
0.5000   96.00002.0000 0.4649    0.3282    0.7375    -0.2726    -0.3151    -0.2330    1.0000  0.0000  
0.7500   96.00000.0000 0.4803    0.2304    0.2304    0.2499     0.2229     0.2792     0.0000  1.0000  
0.7500   96.00000.5000 0.4803    0.2304    0.3075    0.1729     0.1465     0.2011     0.0000  1.0000  
0.7500   96.00001.0000 0.4803    0.2304    0.4785    0.0018     -0.0244    0.0292     0.3333  0.5000  
0.7500   96.00001.5000 0.4803    0.2304    0.6746    -0.1942    -0.2255    -0.1621    0.9000  0.0000  
0.7500   96.00002.0000 0.4803    0.2304    0.8671    -0.3868    -0.4313    -0.3435    1.0000  0.0000  
```

- [확인] Largest measured V_future: +0.29595 at share 0.75, lambda 0, horizon 24 (95% CI +0.23058 to +0.37411).
- [확인] Most negative V_future: -0.43956 at share 0.75, lambda 2, horizon 24 (95% CI -0.53333 to -0.35371).
- [확인] Harm rate ranges from 0.000 to 1.000 across cells.
- [확인] Mean quantile crossing rate of M3 predictions: 0.00000.
- [확인] Oracle future covariate (M2) changes WQL by +0.4127 relative on the Gate A primary subset; the r=0 negative control gives -0.0206.
- [확인] Oracle per-task admission headroom over the best fixed policy: +0.1056 relative.
- [확인] Selector A1_historical_utility chooses M3 on 38.0% of tasks and recovers +0.789 of the oracle gap.
- [확인] Selector A2_historical_reliability chooses M3 on 48.0% of tasks and recovers +0.768 of the oracle gap.

## 13. Verdict

```
Study 0   PASS
Smoke     PASS
Gate A    PASS
Gate B    PASS
Gate C    PASS
Gate D    PASS
```

## 14. Limitations

- [확인] Synthetic data only: two sinusoids plus an AR(1) residual, a linear target and unbiased, serially uncorrelated covariate forecast error.  Nothing here establishes behaviour on real weather or demand data.
- [확인] One forecast origin for the primary comparison and 30 base series per cell; the Monte-Carlo table states which cells are resolvable at this precision.
- [확인] Coarse grid: 4 shares x 5 lambdas x 2 horizons.  Study 1B boundary refinement is explicitly out of scope.
- [확인] Zero-shot Chronos-2 only, frozen, cross_learning=False.  No other TSFM, no fine-tuning, no uncertainty propagation, no predictive quantile mixture.
- [확인] The grid weights every cell equally.  That weighting is a design choice and does not represent any deployment frequency of good or bad covariate forecasts.
- [미검증] The relationship between the nominal covariate share r and what a practitioner would call covariate importance has not been validated against prior work.

## 15. Next steps

- Refine the boundary between benefit and harm on a finer lambda grid (Study 1B) in the region the coarse grid brackets.
- Add independent repetitions for cells reported INCONCLUSIVE rather than changing the DGP, the grid or any threshold.
- Extend the error model to biased and serially correlated covariate forecast error.
- Test nonlinear target-covariate relationships and multiple covariates.

## 16. Reproduction

```bash
python -m covariate_trust.cli study0 --config configs/study0.yaml
python -m covariate_trust.cli smoke --config configs/pilot.yaml
python -m covariate_trust.cli diagnostic --config configs/pilot.yaml
python -m covariate_trust.cli admission --run-dir /home/minjae/Documents/github/timeseries/runs/20260730_161002_diagnostic
python -m covariate_trust.cli report --run-dir /home/minjae/Documents/github/timeseries/runs/20260730_161002_diagnostic
python -m covariate_trust.cli pilot --config configs/pilot.yaml
```

```
master_seed  20260730
model        amazon/chronos-2 (revision None)
started      2026-07-30T16:10:57
finished     2026-07-30T16:10:57
```
