# Notation registry

Frozen 2026-08-12. Machine-readable twin: `notation_registry.csv` (20 symbols).
Every entry was read from a verification artifact, not from prior prose.

Any Methods, Results, Discussion or caption text that uses a symbol differently from this
table is a defect.

---

## The two formulations

```
symbol        meaning                                     source
----------------------------------------------------------------------------------
y_t           observed demand in period t                 dgp_verification.md
o_t           occurrence indicator, 1[y_t > 0]            derived
y_t^+         positive magnitude, y_t given y_t > 0       dgp_verification.md
h_t           history at origin t (lookback L)            point_hurdle_fairness.md

yhat_t^P      DIRECT prediction of the conditional mean   M0_PARAMETER_MATCHED_POINT
yhat_t^H      FACTORIZED prediction, p_t * mu_t           M1_HURDLE_MEAN
p_t           P(y_t > 0 | h_t)                            p_hat
mu_t          E[y_t | y_t > 0, h_t]                       mu_hat
```

`yhat_t^H = p_t · mu_t` is a **product**. The Croston lineage discussed in Section 2.1
combines a size estimate and an interval estimate as a **ratio**. The two are related
parameterizations of the same conditional mean and are never written with the same
symbols.

## Evaluation

```
m_t              exact oracle conditional mean, p_true * mu_true
                 exact dynamic programming over (steps to next event, next-gap parity,
                 next-magnitude parity), advanced across the horizon
RMSE_P, RMSE_H   root mean squared error of each arm against m_t, per series
G                100 (1 - RMSE_H / RMSE_P), in percentage points
                 G > 0 favours the factorized arm; G < 0 favours the direct arm
```

**`G` is the only performance quantity in the main text.** The source carries three
different absolute-delta sign conventions across the two synthetic stages and the real
data (`metric_sign.md`); `gain` is the one quantity that means the same thing in all
three. Where an absolute delta must appear, its formula goes on the same line.

## Design parameters

```
d          sparsity parameter; gap support {d-1, d+1} with equal long-run share.
           d in {4, 8}.  d is the DESIGN parameter and the mean gap by construction;
           it is not the ADI measured on real data, which is reported separately.
rho_I      serial dependence of the occurrence-interval process, in {-0.8, 0, +0.8}
rho_M      serial dependence of the positive-magnitude process, in {-0.8, 0, +0.8}
```

`rho_I` and `rho_M` exist **only in Stage 2**. Stage 1's structured arms are
deterministic period-2 alternation, i.e. `rho = -1`, which is why every graded
dependence statement rests on Stage 2.

## Protocol

```
L      lookback, 96 periods
H_f    forecast horizon, 24 periods
split  train [0, 384], validation [384, 480], test [480, 576]; retained length 576
n      80 series per synthetic cell (40 per data seed x 2 data seeds)
```

## Study names

```
Stage 1   fixed-marginal 2x2x2 factorial, cells C01..C08, run_20260802_112655
Stage 2   stationary rho sweep, d x rho_I x rho_M, 18 cells, pilot_20260803_051713
```

There is no Stage 3 and no Stage 4 in the source; the earlier synthesis used those
labels and the mapping is recorded in
`../synthetic_source_verification/paper_synthesis_discrepancy.md`.

---

## Collision audit — all clear

```
z_t           NOT used for the occurrence indicator.  Section 2.1 uses z and x for
              Croston's non-zero demand size and inter-demand interval; reusing z for
              occurrence would collide across sections.  The indicator is o_t.
H             NOT used bare.  The horizon is H_f; the factorized arm is a superscript
              on yhat.  A bare H would read as both.
d             disambiguated in the registry: design parameter, not measured ADI.
duplicates    none; 20 symbols, 20 distinct.
```

## Symbols deliberately NOT used in the main text

```
C_neg, C_pos     UNIT-W1.  These are absolute-delta contrasts at rho_M = 0, not G in
                 percentage points.  Using them beside G values would be a unit error.
C_sign           FLAG-W2.  The pooled significance flag disagrees with its own interval
                 in the source artifact.  Not used for any inferential claim.
delta            the absolute difference exists in three mutually inconsistent sign
                 conventions; the main text uses G only.
```
