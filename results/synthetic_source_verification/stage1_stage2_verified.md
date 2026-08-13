# Stage 1 and Stage 2, restored from the source

Naming note first, because it is a real discrepancy. The recovered repository calls
its two numerical studies **Stage 1** (the eight-cell factorial) and **Stage 2**
(the rho sweep). The Windows-side protocol and `results/paper_synthesis/` refer to
"Stage 1 / Stage 3 / Stage 4". There is no Stage 3 or Stage 4 in the source. The
mapping is recorded in `paper_synthesis_discrepancy.md`.

---

## Stage 1 — the eight-cell 2×2×2 factorial

Source run `experiments/decomposition_when_helps/results/run_20260802_112655`.
Copied here as `stage1_verified_contrasts.csv` and `stage1_verified_cells.csv`.

### Design

```
factors     sparsity d in {4, 8}
            interval in {I1_INTERVAL_PREDICTABLE, I0_INTERVAL_INDEPENDENT_CONTROL}
            magnitude in {S1_MAGNITUDE_PREDICTABLE, S0_MAGNITUDE_INDEPENDENT_CONTROL}
cells       C01..C08, 8 cells
n_series    80 per cell (40 x 2 data seeds)
models      M0 point (5,856 params), M1 hurdle-mean (5,856), M2 hurdle-ZTNB (5,857)
```

The "predictable" arms are **deterministic period-2 alternation**, i.e. rho = −1.
That single fact is what Stage 2 was built to test.

### Factorial contrasts, M1 vs M0, in percentage points of gain

```
contrast                effect_pp        95% CI              excludes 0
────────────────────────────────────────────────────────────────────────
interval_dependence      +7.831     [ +6.092,  +9.454]          yes
magnitude_dependence     -4.581     [ -6.276,  -2.915]          yes
sparsity                 -6.264     [ -8.014,  -4.574]          yes
sparsity x interval      +3.345     [ +1.619,  +5.091]          yes
sparsity x magnitude     -0.012     [ -1.702,  +1.695]          no
interval x magnitude    -16.739     [-18.448, -15.051]          yes
three_way                -1.965     [ -3.636,  -0.313]          yes
```

M2 vs M0 has the same signs on every contrast that excludes zero.

### Reading

- Making the **occurrence interval** predictable moves the comparison toward the
  factorized model by about 8 points of gain.
- Making the **positive magnitude** predictable moves it the other way, toward the
  direct model, by about 4.6 points.
- The two do not add: the interaction is **−16.7 points**, larger in magnitude than
  either main effect. When both are predictable the factorized advantage largely
  disappears.
- Sparsity strengthens the interval effect (+3.3) and does nothing to the magnitude
  effect (−0.01, interval spans zero).

### Validity audit of Stage 1

`stage1_validity.json`, copied here.

```
status                                 CONDITIONALLY_VALID
marginal_control_pass                  true
template_reuse_detected                false
higher_order_predictability_detected   false
fixed_pattern_detected                 true, scope = predictable arm only (I1/S1)
generator_case                         CASE 2 for the control: independent per-series
                                       uniform stream keyed by BLAKE2b
```

Evidence for the control being genuine: long-motif (L=16) repetition max 0.0035,
40 distinct sequences out of 40 series, zero identical train/test prefixes, and —
decisively — out-of-sample Markov gain **negative at every order** for both control
axes (range [−0.2036, −0.0005]), meaning higher-order predictors lose to the
marginal baseline. That is what a real no-signal control looks like.

**Blocked claims recorded by the audit itself:**
- "temporal predictability in general drives the hurdle advantage"
- "the Stage 1 interval effect measures a graded predictability axis"

Stage 1 alone cannot support either, because its predictable arm is a single
deterministic alternation, not a graded axis.

### Claim status

"Point/Hurdle relative performance differs substantially across temporal-structure
conditions despite controlled marginals" — **CONFIRMED** for Stage 1, with the
boundary that "predictable" there means alternation specifically.

---

## Stage 2 — the stationary Markov rho sweep

Source run `reports/temporal_dependence/rho_sweep/pilot_20260803_051713`.
Copied here as `stage2_verified_cells.csv`, `stage2_verified_factor_effects.csv`,
`stage2_scientific_classification.json`.

### Design

```
grid        d in {4, 8}  x  rho_I in {-0.8, 0.0, +0.8}  x  rho_M in {-0.8, 0.0, +0.8}
cells       18
n_series    80 per cell
models      HURDLE_MEAN, HURDLE_ZTNB, each against the matched point model
held fixed  mean inter-demand interval, interval support, magnitude marginal
```

### Per-cell gain, HURDLE_MEAN, positive favours Hurdle

```
  d    rhoI    rhoM     gain%          CI(%)          excl 0
  4    -0.8    -0.8     10.57   [  8.42,  12.71]       yes
  4    -0.8     0.0     15.92   [ 13.90,  17.86]       yes
  4    -0.8    +0.8     11.79   [  9.35,  14.27]       yes
  4     0.0    -0.8     14.36   [ 12.75,  15.90]       yes
  4     0.0     0.0      7.81   [  5.89,   9.63]       yes
  4     0.0    +0.8      7.48   [  4.64,  10.33]       yes
  4    +0.8    -0.8     19.00   [ 17.10,  21.05]       yes
  4    +0.8     0.0     12.17   [ 10.45,  13.93]       yes
  4    +0.8    +0.8      9.25   [  6.64,  11.75]       yes
  8    -0.8    -0.8      8.66   [  4.96,  12.45]       yes
  8    -0.8     0.0     11.74   [  9.37,  14.40]       yes
  8    -0.8    +0.8     13.94   [ 10.18,  17.67]       yes
  8     0.0    -0.8      9.20   [  6.69,  11.60]       yes
  8     0.0     0.0      2.36   [ -1.20,   5.73]        no
  8     0.0    +0.8    -19.76   [-26.00, -14.53]       yes   <- the only Point-favourable cell
  8    +0.8    -0.8     22.81   [ 20.04,  25.76]       yes
  8    +0.8     0.0     21.66   [ 18.91,  24.37]       yes
  8    +0.8    +0.8     13.95   [  9.19,  18.29]       yes
```

### Factor model, HURDLE_MEAN

```
term            estimate      95% CI               excludes 0
──────────────────────────────────────────────────────────────
abs_rho_I       +0.1904   [+0.1699, +0.2119]          yes
rho_I           +0.0667   [+0.0533, +0.0807]          yes
rho_M           -0.0711   [-0.0851, -0.0570]          yes
abs_rho_M       -0.0228   [-0.0441, -0.0020]          yes
d               -0.0239   [-0.0322, -0.0153]          yes
rho_I x rho_M   -0.0886   [-0.1115, -0.0654]          yes
d x rho_I       +0.0332   [+0.0194, +0.0471]          yes
d x rho_M       -0.0124   [-0.0262, +0.0012]           no
intercept       +0.0621   [+0.0460, +0.0787]          yes
```

### Classification

```
CLASS_A_GENERAL_PREDICTABILITY_SUPPORT

allowed claim   "With occurrence temporal dependence present, the relative advantage
                of Hurdle increased regardless of the sign of the dependence."
blocked claims  "the effect is specific to alternation"
                "the effect is specific to persistence"

C_neg (rho_I -0.8)  +0.1266  [+0.0870, +0.1660]  excludes 0
C_pos (rho_I +0.8)  +0.2008  [+0.1554, +0.2439]  excludes 0
C_pred              +0.1637  [+0.1277, +0.1987]  excludes 0
C_sign              +0.0742  [+0.0300, +0.1194]  excludes 0 but small
```

All five integrity gates pass, including `G1_NO_SIGNAL` (the control's Markov gain
is significantly **negative**, −0.0650 [−0.0743, −0.0562]) and `G0_PROVENANCE`
(`git_commit_at_run` = `git_commit_now` = 03043927…, prereg hash 3b6b5f89…).

### The seven questions, answered from the artifact

```
Q1  |rho_I| more related than signed rho_I?     YES. 0.1904 vs 0.0667, ~3x.
Q2  alternation-specific artifact?              NO. C_pos +0.201 with CI clear of 0.
Q3  same direction under positive dependence?   YES, and slightly stronger than
                                                under negative dependence.
Q4  what does rho_M do?                         Moves toward Point. signed -0.0711,
                                                absolute -0.0228, both CI clear of 0.
Q5  sparsity interaction?                       d x rho_I = +0.0332, CI clear of 0.
                                                Sparsity amplifies the occurrence
                                                effect. d x rho_M is null.
Q6  a Point-favourable cell?                    YES, exactly one of 18:
                                                d=8, rho_I=0, rho_M=+0.8,
                                                gain -19.76% [-26.00, -14.53].
Q7  does it match empirical H2?                 The three axes correspond one to one.
                                                See h1_h2_h3_provenance.md.
```

### What Stage 2 does to Stage 1

The source's own report states it: **"This overturns the reading Stage 1 invited."**
Stage 1 had only an alternating arm, so its interval effect was compatible with an
alternation-specific story. With a positive-dependence arm present, persistence
helps at least as much as alternation, and the operative quantity is `|rho_I|`.

The paper must therefore present Stage 1 and Stage 2 in that order and with that
relationship, not as two interchangeable pieces of support.

---

## Stage 3 — does not exist in the source

There is no Stage 3 or Stage 4 in the recovered repository. The mechanism material
that "Stage 3" referred to on the Windows side is the **hybrid diagnostic columns**
inside the Stage 1 run — `p_true_x_mu_hat`, `p_hat_x_mu_true`, `p_hat_x_mu_hat` —
which substitute the true component for the estimated one and so isolate which head
carries the error. Example, cell C03 (d=4, interval independent, magnitude
predictable), M1:

```
p_true x mu_hat   0.2900     <- magnitude head alone: good
p_hat  x mu_true  0.9030     <- occurrence head alone: bad
p_hat  x mu_hat   0.9652     <- full model, dominated by the occurrence head
```

This is a **within-run diagnostic decomposition, not a separate staged experiment**,
and it is aggregate over 80 series per cell rather than a single-series illustration.

Status: `MECHANISM_AGGREGATE_SUPPORT` **within the controlled study**. It says
nothing about learned occurrence heads on real data, where the separate finding
`LEARNED_OCCURRENCE_HEAD_NOT_SUPPORTED` (Brier skill −0.008 on M5, −0.091 on
Favorita) stands. The two must not be merged into one sentence.
