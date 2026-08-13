# 3–5 Methods — draft v1

Design only. No outcome appears in this file; every constant resolves to an artifact
through `../paper_methods_verified/methods_claim_source_map.csv`, and every symbol to
`../paper_methods_verified/notation_registry.md`.

---

## 3. Problem Setup and Forecasting Formulations

### 3.1 Intermittent demand and what the standard descriptors retain

Let `y_t` denote demand observed in period `t` for a single series. Intermittent demand is
the regime in which most `y_t` are zero and the remainder are positive and often variable.
Two derived quantities are convenient throughout: the occurrence indicator
`o_t = 1[y_t > 0]`, and the positive magnitude `y_t^+`, meaning `y_t` conditional on
`y_t > 0`. A forecaster observes a history `h_t` — the `L` periods before the origin — and
must produce an estimate of the conditional mean `E[y_t | h_t]` over a horizon of `H_f`
periods.

The descriptors normally used to characterize such a series, the average inter-demand
interval and the squared coefficient of variation of positive sizes, are functionals of
the marginal distributions of intervals and of sizes. They summarize how sparse a series
is and how variable its positive demands are; by construction they do not encode the order
in which intervals and magnitudes arrive. Two series with the same average interval and
the same positive-demand distribution can differ in whether gaps cluster or alternate, and
in whether large orders follow large ones. The question this paper asks is whether that
unencoded ordering changes which of two standard forecasting formulations does better.

### 3.2 Direct prediction of the conditional mean

The direct formulation treats the conditional mean as a single quantity to be regressed.
A model maps the history to a prediction

```
yhat_t^P = f_P(h_t)
```

with no internal separation of occurrence from magnitude. It is trained on the observed
series, zeros included, against a squared-error objective, and is referred to below as the
**direct** arm. In the controlled study this is the `M0_PARAMETER_MATCHED_POINT`
implementation, a DLinear backbone whose output layer emits the horizon directly.

### 3.3 Factorized prediction: occurrence probability times positive mean

The factorized formulation writes the same conditional mean as a product of two
conditional quantities,

```
E[y_t | h_t] = P(y_t > 0 | h_t) · E[y_t | y_t > 0, h_t] = p_t · mu_t
```

and estimates each factor with its own head on a shared backbone. The prediction is

```
yhat_t^H = p_t · mu_t
```

where `p_t` is produced by an occurrence head and `mu_t` by a positive-magnitude head.
This is the `M1_HURDLE_MEAN` implementation. Because the two heads sit on one backbone and
the product is formed at prediction time, no inversion or post-hoc de-biasing step is
required; this distinguishes the formulation from the classical size-over-interval ratio
discussed in Section 2.1, which is a different parameterization of the same conditional
mean.

A third arm, `M2_HURDLE_ZTNB`, replaces the magnitude head's target with a zero-truncated
negative-binomial mean. It carries 5,857 parameters against the other two arms' 5,856 —
one additional scalar, 0.017% — which is inside the pre-registered 1% parameter-match
rule. It is reported as a secondary comparison throughout; the primary comparison is
always `M1` against `M0`.

### 3.4 What a matched comparison measures, and the metric

Both formulations estimate the same target. With unlimited data and unlimited capacity
neither has an advantage that the other cannot represent, since the product `p_t · mu_t`
and the direct output describe the same function. Any difference observed under a matched
budget is therefore a difference in **finite-sample behaviour** — how each parameterization
allocates a fixed number of parameters and a fixed number of gradient steps to a fixed
amount of data — and not a statement about representational capacity. This framing governs
every claim in the paper and is the reason the comparison is held to one backbone family,
one parameter count, one optimizer and one training budget.

Performance is reported as a single relative quantity. Writing `RMSE_P` and `RMSE_H` for
the root mean squared error of the direct and factorized arms against the evaluation
target,

```
G = 100 · (1 − RMSE_H / RMSE_P)   percentage points
```

so that `G > 0` favours the factorized arm and `G < 0` favours the direct arm. `G` is used
throughout the main text and in every figure. The underlying artifacts also record
absolute differences, but under three mutually inconsistent sign conventions across the
two synthetic stages and the real data; the relative quantity is the one that means the
same thing everywhere, and mixing the absolute conventions would silently invert a
statement. Where an absolute difference appears in the appendix, its formula is written on
the same line.

The next question is what `RMSE` is measured against, which depends on the study.

---

## 4. Controlled Synthetic Study

### 4.1 Generating process and marginal control

The synthetic data are generated as a marked point process on a regular time grid. Event
times follow `T_j = T_{j−1} + Q_j`, where `Q_j` is the gap between consecutive demands, and
demand is `y_t = M_j` at `t = T_j` and zero otherwise. Two axes are manipulated.

**Occurrence.** The gap takes one of two values, `d − 1` or `d + 1`, each with long-run
share one half, where `d ∈ {4, 8}` is the sparsity parameter. `d` is therefore the mean gap
by construction, and is a design parameter rather than a measured statistic.

**Magnitude.** The positive magnitude is `M_j = 1 + Poisson(λ_j − 1)`, so `M_j ≥ 1` and
`E[M_j | λ_j] = λ_j`, with `λ_j` taking one of two values, 5 or 15. The long-run positive
mean is 10 under every configuration used.

The design's central property is that **the marginals are held fixed while only the
ordering changes**. On each axis, the structured condition and its control draw from the
same two-point support with the same long-run frequency; they differ only in the sequence
in which the two values appear. The two conditions are additionally paired at the level of
random numbers — the same event index draws the same uniform for the gap choice and the
same Poisson innovation in both modes — so the comparison is not confounded by sampling
noise between arms. Holding the marginals fixed is an experimental control, not a finding;
it is what allows a change in relative performance to be attributed to temporal
organization rather than to sparsity or size variability. The source audit confirms it
empirically, reporting a mode difference of 0.0015 in the empirical average interval at
`d = 4`.

Several factors are excluded by construction rather than controlled statistically: trend,
calendar seasonality, hidden regime switching, heavy tails, test-time distribution shift,
phase jitter, and cross-correlation between interval and magnitude. A burn-in of
`max(128, 8d)` periods precedes the retained series.

### 4.2 Experimental fairness, training protocol and evaluation target

The two arms are matched on every dimension the comparison does not intend to vary
(Table 1). Both use the DLinear backbone, receive the same input window, and carry
**5,856 parameters each** by construction. Both are trained by one procedure, imported
unchanged, with Adam at learning rate `1e-3` and no weight decay, for at most 30 epochs
with patience 5 and batch size 256. Normalization statistics are computed on the training
split only. Series are of length 576 with `L = 96` and `H_f = 24`, split into training
`[0, 384]`, validation `[384, 480]` and test `[480, 576]`.

Model selection is the point at which a comparison of this kind usually leaks, so the
protocol is stated explicitly. The checkpoint criterion is the **validation realized-`y`
mean squared error**, identical for every model and every cell. Selecting on the oracle
target, on test results, on a model-specific metric, or by per-cell tuning is prohibited by
the pre-registration; one setting is used for every cell and every arm.

One continuous trajectory is generated per series and then split, so the gap sequence, the
magnitude parity and the latent process state are never reset at a split boundary.

Evaluation uses an **exact conditional mean** rather than realized demand. The oracle is
computed by dynamic programming over the latent state — steps to the next event, next-gap
parity and next-magnitude parity — advanced across the horizon, and it conditions only on
the state implied by history strictly before the origin. A Monte Carlo computation over
200,000 paths is retained as a cross-check and is never the primary quantity. Because the
synthetic target is an exact conditional mean and the real-data target is realized demand,
the two are not numerically comparable, and no statement in this paper places a synthetic
and a real error side by side.

Each cell contains 80 series, formed as 40 series under each of two data seeds. Two model
seeds are trained per series and averaged within the series before any comparison, so the
comparison unit is the series. Intervals are paired series bootstraps with 2,000 draws at
the 0.95 level.

### 4.3 Stage 1 design: the fixed-marginal factorial

Stage 1 is a 2 × 2 × 2 factorial with eight cells, `C01`–`C08`, crossing sparsity
`d ∈ {4, 8}` with a structured-or-control condition on each of the two axes. In the
structured condition the relevant sequence alternates deterministically: gaps run
`d − 1, d + 1, d − 1, …`, or magnitude parameters run `5, 15, 5, …`, with the initial parity
drawn per series. In the control condition the same two values are drawn independently with
equal probability.

The structured arm is therefore **deterministic period-2 alternation**, a single extreme
point rather than a graded axis. This is a deliberate limitation of the design and it is
what Stage 2 exists to relax: Stage 1 can establish that temporal organization matters
under fixed marginals, and that the two axes interact, but it cannot support any statement
about how the effect varies with the strength or the sign of dependence. The source's own
validity audit records this as a blocked claim, and the paper honours it.

The genuineness of the control arm was audited independently of the main comparison, by
testing whether any higher-order predictor beats the marginal baseline out of sample on the
control series. That check is reported in Section 4.6 with the Stage 1 results.

### 4.4 Component-attribution diagnostic

Stage 1's factorized arm records three additional predictions in which one estimated
component is replaced by its true value: the true occurrence probability with the estimated
magnitude, the estimated occurrence probability with the true magnitude, and both
estimated. Comparing the three isolates which head carries the error in a given cell.

This is an **attribution** diagnostic, not a causal identification. Substituting a true
component tells us where the error concentrates under the fitted model; it does not
establish that the occurrence process causes the factorized arm's advantage. The paper
labels its result accordingly and does not use it to support a mechanism claim.

### 4.5 Stage 2 design: the stationary dependence sweep

Stage 2 replaces the binary structured-or-control contrast with a signed, graded one. Each
of the two sequences is generated by a stationary two-state Markov process whose serial
dependence is set by a parameter: `ρ_I` for the occurrence-interval process and `ρ_M` for
the positive-magnitude process, each taking values in `{−0.8, 0.0, +0.8}`. Negative values
make the sequence alternate; positive values make it persist; zero reproduces independent
draws. Crossed with `d ∈ {4, 8}` this gives **18 cells** of 80 series each.

The marginal control is retained: the mean inter-demand interval, the interval support and
the magnitude marginal are held fixed across all 18 cells, so `ρ_I` and `ρ_M` vary the
ordering alone. Because the two parameters are set independently, the design separates two
questions that Stage 1 could only pose jointly — whether the *strength* of dependence
matters, and whether its *sign* matters — and it does so for each axis separately.

`ρ_I` and `ρ_M` exist only in Stage 2. Stage 1's structured arm corresponds to `ρ = −1`,
outside the swept range, and no Stage 1 quantity is reported as if it were a point on the
Stage 2 axis.

---

## 5. Empirical Validation

### 5.1 Datasets and evaluation protocol

Two public retail datasets provide the core empirical validation: M5 and Favorita
(Table 3). From each, 1,200 series are drawn, balanced at 300 series per SBC regime class,
subject to an eligibility rule of at least 20 positive training observations. M5 series
have length 1,941 with training ending at 1,829 and three evaluation origins at 1,857,
1,885 and 1,913; Favorita series have length 1,688 with training ending at 1,576 and origins
at 1,604, 1,632 and 1,660. Both use a lookback of 96 and a horizon of 28.

The regime-balanced sample is deliberately not representative of either catalogue — M5's
full eligible pool is 23,053 intermittent, 5,942 lumpy, 984 smooth and 496 erratic series —
and is described as balanced rather than representative wherever it appears. Two further
datasets, FreshRetailNet-LT and UCI Online Retail II, are used only as domain- and
time-transfer stress tests for the routing analysis; their protocol is in the appendix and
they are never treated as core validation data.

### 5.2 Operational definitions of the empirical hypotheses

Three hypotheses connect the controlled study to observed demand. Each is defined here in
terms of measured quantities, and each carries a construct boundary that is stated at
definition time rather than at reporting time.

**H1 — the occurrence-dependence relationship.** For each real series, the strength of
serial dependence in the occurrence process is measured as the **absolute** first-order
autocorrelation of the inter-demand interval sequence, and related to that series'
`G`. The absolute value is used because the controlled study identifies magnitude of
dependence, not its direction, as the operative quantity on the occurrence axis; this is a
finding carried over from Stage 2 rather than an analyst's convention, and Section 4.7
reports the coefficients that support it. H1 is a test of whether an analogous relationship
appears in observed demand, not a test of whether the synthetic mechanism reproduces.

**H2 — the frozen selector.** The single configuration that the controlled sweep identifies
as favouring direct prediction is encoded as a rule over three series descriptors —
sparsity, occurrence-dependence strength and magnitude persistence — with the same sign on
each axis as in the synthetic design. The rule is **frozen before it is applied**, and is
then evaluated on a population of M5 series disjoint from the one used to derive it. H2 is
reported as two separate claims, and the separation is definitional: whether the rule
selects series on which direct prediction does better is a question about *prediction*,
while whether the association survives adjustment for confounding is a question about
*mechanism*. The second is tested by re-weighting candidate and control series for overlap
on scale, sparsity, positive-demand variability and occurrence dependence.

**H3 — the sparsity interaction.** The controlled study finds that sparsity amplifies the
occurrence effect, contrasting `d = 4` against `d = 8`. The external test splits series at
the ADI median, which is 1.304 on M5 and 1.317 on Favorita. These are **different
constructs**: the synthetic contrast spans a factor of two in mean interval at values far
above the real median, and the external split separates series that are barely intermittent
from series that are somewhat more so. H3 is therefore reported as a construct mismatch
alongside its numerical result, and its outcome is not read as a refutation of the
synthetic finding.

### 5.3 Uncertainty, bootstrap and seeds

All intervals are 95% percentile intervals from a paired bootstrap resampling the **series**
as the unit, with 2,000 draws. Where several training seeds exist, per-seed estimates are
reported alongside the pooled estimate rather than replaced by it. Classical baselines —
simple exponential smoothing, Croston's method, the Syntetos–Boylan approximation and the
Teunter–Syntetos–Babai method — are evaluated under the identical protocol and are reported
by mean rank, so that the two neural formulations can be placed in absolute context rather
than compared only to each other.

With the design, the protocol and the measured quantities fixed, the remaining question is
what they produce.
