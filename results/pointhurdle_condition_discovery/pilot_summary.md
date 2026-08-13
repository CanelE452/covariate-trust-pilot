# Gate 1 — M5 ↔ Favorita transfer pilot

Seed 20260813. No model was retrained. ElasticNetCV is the primary predictor; every
imputer, scaler and hyperparameter is fitted on the training dataset only and the held-out
dataset is touched once, at evaluation.

---

## Two tracks, because the required features are not all available

Phase D asks for `S_occ` and `S_mag` computed on **inner validation windows inside outer
train**. A search of `results/external_validity_screen/` found **no cached inner-validation
predictions** — the only cached predictions are at the three outer test origins. So:

```
T1  STRICT TRAIN-ONLY        features from train-split descriptors only.
                             Target: G at all three outer origins.
                             Fully compliant with the train-only rule.

T2  ORIGIN-1-CONDITIONED     adds S_occ, S_mag and the realized paired gain from the
                             FIRST outer origin.  Target: G at origins 2 and 3 only.
                             Causally valid - origin 1 is observed before the origin-2
                             decision - but NOT train-only.  Labelled as such everywhere.
```

Everything in T2 is therefore an **upper bound** on what a strict train-only selector could
do, not an estimate of it.

## Macro results over the two transfer directions

```
track  ablation  features                              macro_rho  macro_improve%  worst%  control_rho
T1     A0        training-domain mean                     n/a          0.000       0.000     -
T1     A1        ADI, CV2                               0.1904        -0.158      -0.316   C0 degenerate
T1     A2        D_gap only                             0.0114         0.000       0.000   C0  0.0114
T1     A3        basic structure + D_gap                0.1824        -0.062      -0.125   C0 -0.0097
T2     A2        D_gap only                             0.0084         0.000       0.000   C0 degenerate
T2     A4        S_occ, S_mag, sparsity                 0.2178        -0.031      -0.031   C0  0.1756
T2     A5        inner (origin-1) gain only             0.2426        +0.105      +0.078   C0 degenerate
T2     A6        component skill + basic structure      0.2078        -0.079      -0.148   C0 -0.1523
T2     A7        component skill + inner gain           0.2779        +0.044      +0.016   C0 degenerate
```

`macro_improve%` is the relative improvement of the selector over the train-domain global
choice, averaged over the two directions; `worst%` is the worse of the two.

## What the controls say

The shuffled-target control was decisive, twice.

**First, it caught a bug in my own harness.** When ElasticNet shrinks to a constant, the
prediction still carries ~1e-16 of floating-point jitter, and Spearman happily ranks that
dust and returns values like 0.22. A degeneracy guard (`pred_sd` below 1e-8 of the target
scale ⇒ report no prediction) now marks those rows `degenerate` instead.

**Second, it caught a spec violation.** The first run imputed `S_mag` with the training
median and no missing indicator, which Phase D3 forbids. `S_mag` is missing for 28% of
series — exactly the sparse ones — so median imputation smuggled a sparsity signal into the
model even when the target was shuffled. That is why `A4_C0` originally scored **0.2212,
above A4 itself at 0.1993**. Adding explicit missing indicators, as Phase D3 requires, moved
A4 to 0.2178 and its control to 0.1756.

**Even after the fix, A4 is only 0.042 above its shuffled control.** Component
forecastability, as computed here, cannot be cleanly separated from a missingness-and-
sparsity artifact. `A5` and `A7`, whose controls are degenerate, are clean.

## Gate 1 verdict

```
criterion                                                              observed            pass
both directions Spearman positive                                      A5, A7 yes           yes
macro Spearman >= 0.10 in both directions                              A5 0.243, A7 0.278   yes
component forecastability beats D_gap-only by >= 0.05 macro Spearman   A4 0.218 vs 0.008    yes*
selector never worse than global by more than 0.5%                     worst -0.148%        yes
selector improves on at least one test dataset                         A5, A7 both          yes
Point and Hurdle candidate regions both adequately sampled             yes, see below       yes
component model separable from its shuffled control                    A4 - A4_C0 = 0.042   NO
```

`*` A4 beats `D_gap`-only comfortably, but so does its own shuffled control, so the
comparison does not establish that component forecastability is what is doing the work.

**GATE 1 = YELLOW.**

Not RED: no direction reverses, nothing is worse than `D_gap`-only, no selector loses more
than 1%, and the gain surface points the same way on both datasets.

Not GREEN: the ablation that carries the hypothesis (A4 / A6, component forecastability)
is not separable from its control and *worsens* policy loss slightly. The ablation that
works (A5) is simply **who won at the previous origin** — outcome persistence, not a
structural condition — and A7 adds essentially nothing to it.

## Candidate regions

Sparsity split at the within-dataset median `zero_ratio_train`, fixed before inspection.

```
dataset    region          rows   series   mean_G   point%   hurdle%
favorita   high_sparsity  8,109    2,703    12.20    28.61     53.22
favorita   low_sparsity   8,106    2,702     4.91    39.28     43.24
m5         high_sparsity  8,541    2,847     7.26    27.64     52.02
m5         low_sparsity   8,538    2,846     0.76    39.87     31.15
```

The direction agrees across the two datasets — sparser series favour Hurdle more — and the
samples are far above the Phase H3 minimums. But this is a **known** relationship, not a
new condition, and no region reaches a Point-favourable mean `G` below −2% on either
dataset. Against the Phase H3 criteria this is a `dataset-consistent gradient`, not a
`stable real-data Point condition`.

## What this does and does not license

```
established   there IS headroom (Gate 0), and gain is partly predictable across datasets
established   the predictable part is dominated by outcome persistence (A5), not by
              component forecastability
established   D_gap alone has essentially no origin-level predictive power (rho ~ 0.01)
NOT shown     that component forecastability is a real-data condition
NOT shown     that any selector meaningfully beats always-Hurdle - the best is +0.105%
              against a series-level ceiling of 1.7-2.0%
```
