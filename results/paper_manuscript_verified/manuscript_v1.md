# Manuscript v1 — assembled draft

Assembled 2026-08-12 from the frozen section files listed in
`manuscript_section_map.md`. No sentence was rewritten during assembly; each section
is its source file's body verbatim. Section numbering follows
`../paper_outline_verified/final_outline_freeze.md`.

---



<!-- ===== Abstract  (source: abstract_v1.md) ===== -->

Intermittent demand — series that are zero in most periods — can be forecast by predicting
the conditional mean directly, or by factorizing it into an occurrence probability and a
conditional positive magnitude. Both target the same quantity, and on public benchmarks
neither is uniformly better in aggregate, so the useful variation is conditional. What
conditions it is unclear: the descriptors normally used to characterize such series
summarize marginal properties and do not encode the order in which intervals and magnitudes
arrive. We compare the two formulations under matched conditions — one backbone, one
parameter count, one training budget, one evaluation target — on synthetic demand whose
marginals are held fixed while the temporal dependence of occurrence and of positive
magnitude varies along two separate axes. The axes interact
rather than add, and a signed sweep resolves the interaction into an asymmetry: occurrence
effects track the strength of dependence, magnitude effects track its direction, and it is
magnitude persistence that moves the comparison toward direct prediction. Within the
eighteen-cell grid, one configuration — sparse, occurrence-unpredictable,
magnitude-persistent — favours direct prediction by about 20% in relative error. On two public
retail datasets the occurrence relationship appears as an empirical analogue, and a rule
frozen from the sweep shifts unseen series toward direct prediction on an independent
population by 11.87 percentage points of win rate. The isolated association does not survive
overlap adjustment, and a learned router did not transfer across domains despite a
measurable oracle opportunity. We report when factorizing helps or hurts at a fixed budget, and the
boundary of that answer in observed demand.


<!-- ===== 1 Introduction  (source: introduction_v6.md) ===== -->

Intermittent demand — series in which most periods are zero and the remainder carry
positive, often variable quantities — can be forecast in one of two ways. A model can
predict the conditional mean directly, or it can factorize that mean into the
probability of a demand event and the expected size given an event, and multiply the
two. Both target the same quantity, and the choice between them is usually made by
convention. On standard public benchmarks neither formulation is uniformly superior in
aggregate: which of the two leads depends on the error measure. A flat
aggregate does not make the choice unimportant. It suggests that the useful variation
is conditional, and that the question worth asking is not which representation wins but
when each one does.

Answering that requires distinguishing one intermittent series from another, usually
done with the average demand interval and the squared coefficient of variation of demand size
[SBC05; KH06]. Both summarize a marginal distribution and neither retains the order in
which intervals and sizes arrive: two series can match on both and still differ in
whether gaps cluster or alternate. That order is known to matter: simulation work varying
interval autocorrelation, size autocorrelation and their dependence reports
effects on forecast accuracy and inventory performance [ALR12]. Representation has been
examined separately: neural work compares a directly predicted demand rate against a
Croston-style representation that predicts non-zero demand size and inter-demand interval
and combines them as a ratio [Kou13], and single-stage models against two-stage models
that multiply an occurrence probability by a conditional size [NAR26]. These answer
different questions from the comparison here, in which a direct conditional
mean and an occurrence-probability × positive-magnitude factorization are held to one
capacity and training budget while occurrence and magnitude dependence are varied
separately.

We therefore ask a narrow question. Under matched marginal properties, when the
temporal organization of occurrence and of positive magnitude is varied, how does the
finite-sample relative behaviour of direct and factorized forecasting change? The
comparison is deliberately confined to one backbone family, one parameter budget, one
training protocol and one evaluation target. Because both formulations estimate the
same conditional mean, a difference observed under a matched budget is a difference in
finite-sample inductive bias, not a statement about which function class can represent
the target. We do not assume that either representation is generally preferable.

A controlled study answers the first half. In a factorial design that holds the gap
support, the average interval and the positive-demand marginal fixed and varies only
temporal order, occurrence structure and magnitude structure interact: their joint
effect is not the sum of their separate effects. That design contrasts structure
against its independent control, so it shows that order matters without identifying
which property of the order does. Replacing the binary
contrast with a signed dependence sweep resolves this into an asymmetric geometry —
that is, the two axes respond to different features of dependence. Occurrence effects
are primarily associated with the strength of dependence and largely insensitive to its
sign, whereas magnitude effects are primarily associated with its direction, with
persistence rather than structure moving the comparison toward direct prediction. Within the eighteen-cell sweep, one configuration shows statistically clear
superiority for direct prediction — sparse, occurrence-unpredictable,
magnitude-persistent — at a relative gain of about −19.8%.

The second half asks how much of this appears in observed demand. On two public retail
datasets the occurrence relationship appears as an empirical analogue rather than a
direct replication: the strength of interval dependence is positively associated with
the relative advantage of factorization in both, on three error scales, and the
association strengthens within the intermittent regime. Separately, a descriptor rule
encoding the configuration identified in the sweep, frozen before it was applied,
shifts previously unseen series toward direct prediction on an independent population,
by 11.87 percentage points of win rate, and the shift reproduces across three training
seeds. The synthetic-derived condition therefore carries predictive information about
which real series favour which representation.

That predictive transfer does not extend to the mechanism behind it. The rule selects
on three descriptors at once, and in observed demand those descriptors are not
separable in the way the controlled design makes them: candidate and control series
differ substantially in scale as well as in the intended axis. After adjusting for
overlap in scale, sparsity, positive-demand variability and occurrence dependence, the
isolated association is no longer distinguishable from zero. We therefore report two
results rather than one — the configuration transfers as a predictor, and its isolated
mechanism is not recovered — and note that the controlled study, which contains no
scale axis, cannot arbitrate the question either way.

Conditional differences of this kind invite a downstream question: can they be
exploited by a learned routing rule? The two forecasts do make complementary errors,
and an origin-level oracle over them is better than any fixed mixture, so the
opportunity is measurable. Converting it proved unreliable. Gains obtained during
development did not transfer across domains or across time, and one external domain
degraded severely, at which point a pre-registered stopping rule was triggered. This
paper therefore proposes neither a universal winner nor a generally reliable router. It
characterizes when direct and factorized forecasting differ under controlled temporal
structure, and identifies the boundary of that characterization in observed demand.


<!-- ===== 2 Related Work  (source: related_work_v3.md) ===== -->

## 2.1 Classical Intermittent-Demand Forecasting and Decomposition

Forecasting demand that is zero in most periods has been approached, since Croston's
original treatment, by declining to model the demand series directly and instead tracking
two quantities separately: the size of a positive demand when one occurs, and the timing
of occurrences [Cro72]. That method smooths the positive sizes and the inter-demand
intervals only in periods with demand, and forms a demand rate as the ratio of the two
[Cro72].
The construction proved durable enough that most subsequent work is best read as
refinement rather than replacement. The ratio introduces an inversion bias, and the
Syntetos–Boylan approximation supplies the correction factor that renders the estimator
approximately unbiased [SB05]. A second refinement replaces interval updating with the
direct updating of an occurrence probability in every period, including periods without
demand, which allows the forecast to decay when an item stops moving [TSB11].

Two things follow for the present paper. First, treating occurrence and positive
magnitude as separate objects is not a modelling choice this paper introduces; it is the
default of the field and has been for five decades. Second, the classical lineage
combines its two components as a **ratio** of size to interval, whereas the factorized
formulation studied here combines an occurrence probability and a conditional positive
mean as a **product**. The probability-updating variant [TSB11] is the closest classical
ancestor of that product form. The two are related parameterizations of the same
conditional mean rather than the same estimator, and this paper treats them as such.

## 2.2 Intermittency Classification and Marginal Descriptors

Because intermittent series differ widely in how sparse and how variable they are,
practice relies on a small set of descriptors to decide which method to apply to which
item. The standard scheme is built on two statistics: the average inter-demand interval
and the squared coefficient of variation of positive demand sizes. Regions of the plane
these two define are used to separate smooth, erratic, intermittent and lumpy demand and
to select between Croston's method and its bias-corrected variant, with the boundaries
validated on several thousand automotive spare-part series [SBC05]. The placement of one
of those boundaries was subsequently refined on analytical grounds [KH06].

The scheme is economical and it is well established, and this paper uses it as given: the
regime labels reported later follow it, and no alternative classification is proposed.
What is worth stating explicitly is what the two statistics are functions of. Both
summarize a marginal distribution — how long the gaps are on average, and how variable
the positive sizes are — and their definitions therefore do not retain the temporal
ordering of individual occurrences or of positive magnitudes. Two series can agree on
both descriptors and still differ in whether their gaps arrive in clusters or in
alternation, and in whether large orders tend to follow large ones. That observation
motivates the design used here; it is not a criticism of a scheme built for a different
purpose.

## 2.3 Temporal Dependence in Intermittent Demand

The temporal structure of intermittent demand has itself been the subject of sustained
attention, in two related forms.

One line builds dependence into the estimator. Rather than treating occurrences as
independent draws, the lead-time demand distribution can be bootstrapped from a two-state
Markov model over zero and non-zero periods, with sampled positive sizes perturbed to
cover values not seen in a short history; on nine industrial datasets this produced more
accurate distributional forecasts than exponential smoothing or Croston's method [WSS04].
The occurrence process, on this view, carries information that an independence assumption
discards.

A second line treats dependence as something to vary and measure the consequences of.
Working with generated intermittent demand, Altay, Litteral and Rudisill examine three
distinct correlation structures — autocorrelation in demand sizes, autocorrelation in
inter-demand intervals, and cross-correlation between size and interval — and report their
effects on both forecast accuracy and inventory outcomes [ALR12]. The reported effects are
not uniform in sign across the three: negative autocorrelation is associated with higher
achieved service levels than positive autocorrelation, with cost largely unchanged, while
cross-correlation acts in the opposite direction to autocorrelation. The differences are
reported to intensify as intermittency increases.

Taken together, this stream establishes that serial structure in intermittent demand is
consequential for forecasting and for the inventory decisions built on it, and that its
different components need not act in the same direction. The estimators being compared in
that work sit inside a single, already-factorized representation; the question asked is
what dependence does to the performance of such estimators. Temporal dependence in
intermittent demand is therefore established territory, and no part of the present study
is positioned as introducing it.

## 2.4 Neural and Two-Part Forecasting Formulations

Machine-learning and neural methods for intermittent demand form a substantial literature
of their own [GDTP25]; within it, whether a forecast should be
produced as one quantity or as two has been asked directly, and in more than one form
[Kou13; NAR26].

In the neural setting, Kourentzes compares two architectures that differ in exactly this
respect [Kou13]. Both take lagged non-zero demands and inter-demand intervals as inputs. One emits the
demand rate from a single output; the other emits the non-zero demand size and the
inter-demand interval separately and combines them as a ratio, in the manner of Croston's
method [Kou13]. Evaluated over a large simulated population parameterized
from a real spare-parts dataset, each at its own selected configuration, the two rank
differently under accuracy metrics than under inventory metrics [Kou13].

The other pairing — a directly predicted conditional mean against an occurrence
probability multiplied by a conditional size — has been compared in a gradient-boosting
rather than a neural setting [NAR26]. On roughly 1.4 million monthly observations of
automotive spare parts, a LightGBM regressor trained directly on the full feature set is
placed against a two-stage model pairing a LightGBM classifier for the probability of
non-zero demand with a Tweedie-objective LightGBM regressor for the conditional quantity.
The comparison uses a common data preprocessing, feature construction and evaluation
context [NAR26]. The
reported conclusion is that the two-stage form's added complexity does not yield an
aggregate advantage once informative features are supplied. Occurrence and size have also
been modelled jointly, as a deep renewal process that captures
regular and alternating inter-arrival structure on constructed patterns [TJWC21];
hurdle-style decoders continue to appear in current architectures.<sup>1</sup>

These studies provide clear precedents for both comparisons: a directly predicted rate
against a Croston-style ratio in a neural setting [Kou13], and a directly predicted
conditional mean against a probability-times-size product on real data [NAR26]. Accordingly, neither decomposition itself nor the direct-versus-
factorized comparison is the focus of the contribution reported here.

## 2.5 Positioning of the Present Study

The two streams above answer different questions. The dependence stream asks what happens
to forecasting and inventory performance when the serial structure of demand changes,
with the representation held fixed. The representation stream asks which of two ways of
structuring a forecast performs better on a given population.

The present study focuses on a different controlled intersection of these questions: how
the *relative* behaviour of the two representations changes as temporal dependence varies.
A directly predicted conditional mean and an occurrence-probability × positive-magnitude
factorization are compared while the temporal dependence of occurrence and the temporal
dependence of positive magnitude are varied along two separate axes, with the marginal
properties of the generated demand held fixed as an experimental control. That control is a property of the
design rather than a result: it is what allows a change in relative performance to be
attributed to temporal organization rather than to sparsity or size variability. Both
arms receive the same input history, share one backbone family and one parameter budget,
and are trained by one procedure under one budget against one target, so the difference
observed is a difference in finite-sample behaviour under a fixed budget rather than in
what either function class can represent.

The second half of the study asks how far the resulting patterns reach into observed
demand, and reports that boundary as a result in its own right: which parts appear as an
empirical analogue, which transfer as a predictive selector on unseen series, and which do
not survive adjustment for covariates that the controlled design separates but observed
demand does not.

---

<sup>1</sup> For example, a mixture-of-experts encoder with a hurdle decoder
[MC26]; this record is an arXiv preprint and is cited as an indication of current
practice rather than as peer-reviewed evidence.


<!-- ===== 3-5 Methods  (source: methods_v1.md) ===== -->

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


<!-- ===== 4.6-5.9 Results  (source: results_v1.md) ===== -->

## 4.6 Stage 1: temporal organization under fixed marginals

With the marginals held fixed, the relative performance of the two formulations moves
substantially across the eight cells, and it moves in both directions.

The factorial contrasts are given in Table 2. Making the **occurrence interval** structured
rather than independent shifts the comparison toward the factorized arm by **+7.83 pp**
(95% CI [+6.09, +9.45]). Making the **positive magnitude** structured shifts it the other
way, toward the direct arm, by **−4.58 pp** ([−6.28, −2.92]). Increasing sparsity from
`d = 4` to `d = 8` shifts it toward the direct arm by **−6.26 pp** ([−8.01, −4.57]).

The two structural axes do not add. Their interaction is **−16.74 pp**
([−18.45, −15.05]), larger in magnitude than either main effect, so when both sequences
are structured the factorized advantage largely disappears rather than compounding.
Sparsity amplifies the occurrence effect (`sparsity × interval` **+3.35 pp**,
[+1.62, +5.09]) but does not modulate the magnitude effect (**−0.01 pp**,
[−1.70, +1.69], interval spanning zero). A small three-way term remains
(**−1.96 pp**, [−3.64, −0.31]). The secondary factorized arm reproduces the sign of every
contrast whose interval excludes zero.

Two boundaries belong with this result rather than after it.

First, **Stage 1 contains no cell in which the direct arm wins with an interval clear of
zero.** The most direct-favourable cell, `C08` (sparse, both sequences independent), has
`G = −3.01` with a 95% interval of [−6.79, +0.23]. Any statement about a
direct-favourable region therefore belongs to Stage 2, not here.

Second, Stage 1's structured arm is **deterministic period-2 alternation**, so this
experiment establishes that temporal organization matters under fixed marginals and that
the two axes interact, but cannot say how the effect varies with the strength or the sign
of dependence. The source's validity audit classifies Stage 1 as `CONDITIONALLY_VALID` for
exactly this reason and blocks two readings by name: that temporal predictability in
general drives the factorized advantage, and that the interval contrast measures a graded
predictability axis. Both are left to Stage 2.

The control arm itself is not an artifact. Long-motif repetition on the control series
reaches at most 0.0035, all 40 generated sequences are distinct, no train/test prefix is
repeated, and — decisively — the out-of-sample Markov gain is **negative at every order**
on both control axes, ranging over [−0.2036, −0.0005]. A higher-order predictor loses to
the marginal baseline, which is what a genuine no-signal control looks like.

## 4.7 Component attribution within Stage 1

Replacing one estimated component of the factorized arm with its true value localizes where
the error sits. In `C03` — sparse, independent occurrence, structured magnitude — the
factorized arm's error is 0.9652 against the oracle mean; substituting the true occurrence
probability while keeping the estimated magnitude leaves 0.2900, whereas substituting the
true magnitude while keeping the estimated occurrence probability leaves 0.9030. Under this
configuration essentially all of the error is carried by the occurrence head.

This is **component-attribution diagnostic support** and nothing stronger. Substituting a
true component shows where error concentrates under the fitted model; it does not
demonstrate that the occurrence process causes the factorized arm's behaviour, and the
paper does not use it to claim a mechanism.

## 4.8 Stage 2: the dependence sweep, and an asymmetry between the two axes

Replacing the binary contrast with a signed sweep resolves Stage 1's interaction into a
structure in which the two axes respond to different features of dependence (Figure 2).

In the factor model over all 18 cells, the **magnitude** of occurrence dependence carries
the larger coefficient: `|ρ_I|` is **+0.1904** ([+0.1699, +0.2119]) against signed `ρ_I` at
**+0.0667** ([+0.0533, +0.0807]), a ratio of about 2.9. Both signs of occurrence dependence
therefore move the comparison toward the factorized arm. On the **magnitude** axis the
pattern inverts: signed `ρ_M` is **−0.0711** ([−0.0851, −0.0570]) against `|ρ_M|` at
**−0.0228** ([−0.0441, −0.0020]), a ratio of about 3.1, so it is the direction of magnitude
dependence that matters, and persistence specifically that moves the comparison toward
direct prediction. Sparsity contributes **−0.0239** ([−0.0322, −0.0153]) and amplifies the
occurrence effect (`d × ρ_I` **+0.0332**, [+0.0194, +0.0471]) while showing no clear
interaction with the magnitude axis (`d × ρ_M` **−0.0124**, [−0.0262, +0.0012], interval
spanning zero). The two dependence parameters interact
(`ρ_I × ρ_M` **−0.0886**, [−0.1115, −0.0654]).

The marginal patterns in Figure 2B and 2C follow from this. Averaged over the six cells at
each level, occurrence dependence gives **+12.10 / +3.57 / +16.47 pp** at
`ρ_I = −0.8 / 0 / +0.8` — a U-shape in the sign, consistent with strength rather than
direction being operative — while magnitude dependence gives **+14.10 / +11.94 / +6.11 pp**
at `ρ_M = −0.8 / 0 / +0.8`, monotone decreasing. **The vertical extent shown in Figure 2B
and 2C is the spread of the six contributing cells and is not a confidence interval**; the
inferential statements in this subsection rest on the factor-model coefficients and on the
per-cell intervals in Figure 2A.

This asymmetry also resolves the apparent tension between the two stages. Stage 1's
structured magnitude arm is alternation, which on the Stage 2 axis is negative `ρ_M`, and
the magnitude-alternating Stage 1 cells are indeed factorized-favourable. Only positive
`ρ_M` — persistence, which Stage 1 never generated — produces a direct-favourable region.

**Within the 18-cell Stage 2 grid, exactly one cell shows statistically clear superiority
for direct prediction**: `d = 8`, `ρ_I = 0`, `ρ_M = +0.8`, at `G = −19.76`
([−26.00, −14.53]) — sparse, occurrence-unpredictable and magnitude-persistent. One further
cell is not distinguishable from zero (`d = 8`, `ρ_I = 0`, `ρ_M = 0`, `G = +2.36`,
[−1.20, +5.73]); the remaining sixteen favour the factorized arm with intervals clear of
zero. The scope qualifier matters: this is a statement about the tested grid, not about
intermittent demand in general.

All five integrity gates of the sweep pass, including a no-signal control whose
out-of-sample Markov gain is significantly negative (**−0.0650**, [−0.0743, −0.0562]).

---

## 5.4 Overall comparison on real data, and the classical baselines

On M5 and Favorita neither representation dominates in aggregate: the direct arm leads on
RMSE and the factorized arm leads on MAE, on both datasets. More importantly for scope,
both are outranked by classical estimators. Mean ranks on M5 place SBA at **3.152**,
Croston at **3.260**, TSB at **3.411** and simple exponential smoothing at **3.483**, ahead
of the direct arm at **4.202** and the factorized arm at **4.220**.

This paper compares two representations of one backbone under one budget. It is not a claim
about the best available forecaster for intermittent demand, and the aggregate comparison is
reported precisely so that the conditional results below are not read as one.

## 5.5 H1: an occurrence-dependence analogue

The relationship the controlled study identifies on the occurrence axis appears in observed
demand as an analogue. The strength of interval dependence is positively associated with the
relative advantage of the factorized arm on **M5 (+0.1064, [+0.0437, +0.1652])** and on
**Favorita (+0.0789, [+0.0205, +0.1405])**, and the association holds on all three error
scales examined, with six of six estimates having intervals clear of zero.

The association is stronger within the intermittent regime specifically
(**+0.1529**, [+0.0519, +0.2613] on M5); in the lumpy regime the corresponding interval
spans zero.

H1 is reported as `SUPPORTED_WITH_BOUNDARY` and described as an **empirical analogue**, not
a replication. The synthetic and real quantities are not the same measurement: the
controlled study varies a generating parameter and evaluates against an exact conditional
mean, while the empirical test measures an autocorrelation on observed demand and evaluates
against realized demand.

## 5.6 H2: the frozen selector transfers

The configuration that Stage 2 identifies as direct-favourable was encoded as a rule over
three descriptors, frozen, and applied to an M5 population disjoint from the one used to
derive it (675 selected series against 5,018 controls). Series the rule selects show a
relative shift toward direct prediction of **−0.0230** ([−0.0294, −0.0163]), corresponding
to a direct-prediction win rate higher by **+11.87 percentage points**. The shift reproduces
across three training seeds.

The synthetic-derived condition therefore carries predictive information about which real
series favour which representation, on data not used to construct it.

## 5.7 The mechanism does not survive overlap adjustment

The predictive transfer does not extend to the isolated association behind it. The rule
selects on three descriptors simultaneously, and in observed demand those descriptors are
not separable as the controlled design makes them: selected and control series differ
substantially in scale as well as on the intended axis, with an unweighted standardized mean
difference of 1.32 on the log scale. One-to-two matching failed to achieve balance, reaching
only 0.614.

After overlap weighting on scale, sparsity, positive-demand variability and occurrence
dependence, the isolated association is **+0.0032** ([−0.0033, +0.0094]) on 5,693 series —
no longer distinguishable from zero.

These are two results, not one. The configuration transfers as a **predictor**; the
**isolated mechanism** is `NOT_REPLICATED`. The controlled study contains no scale axis and
therefore cannot arbitrate which reading is correct, and this paper does not attribute the
disappearance to scale as a cause — only to the fact that the axes are entangled in observed
demand in a way the design separates.

A related negative belongs here. On real data the factorized arm's occurrence head shows no
skill advantage: the Brier skill score is **−0.0084** on M5, with an interval including
zero, and **−0.0908** ([−0.1401, −0.0438]) on Favorita. The component-attribution diagnostic
of Section 4.7 is a synthetic result and is not supported by a learned occurrence head on
observed demand.

## 5.8 H3: the sparsity interaction does not replicate at the tested split

The controlled study finds that sparsity amplifies the occurrence effect
(`sparsity × interval` +3.35 pp in Stage 1, `d × ρ_I` +0.0332 in Stage 2, both intervals
clear of zero) contrasting `d = 4` against `d = 8`. The pre-registered external test splits
at the ADI median — 1.304 on M5, 1.317 on Favorita — and returns **−0.0305**
([−0.1418, +0.0912]) on M5 and **−0.0428** ([−0.1587, +0.0704]) on Favorita: the wrong sign,
with intervals spanning zero.

H3 is `NOT_REPLICATED` at the pre-registered split. It is also a **construct mismatch**: the
synthetic contrast spans mean intervals of 4 against 8, while the external split separates
series at an ADI near 1.3, so the two tests do not examine the same range. The negative
result is reported as it stands, and it is not presented as a refutation of the synthetic
finding, because the tested contrast is not the synthetic contrast.

## 5.9 The adaptive-use boundary: complementarity without a transferable router

If the advantage is conditional, the natural response is to route between the two
formulations, and this was tested.

The opportunity is real and measurable. On M5, an origin-level convex oracle over the two
forecasts is **4.11%** better than the best static mixture of them, and deliberately
selecting a less correlated pair of experts multiplies that ceiling by **2.15**. The two
forecasts make complementary errors.

Converting the opportunity did not transfer. A gate frozen after development was **−2.43%**
([−2.74, −2.13]) against a static mixture on the first external dataset it was applied to.
Successive changes to the training target, the loss, the aggressiveness, the capacity and
finally the input representation did not recover cross-domain transfer; the last of these
recovered one external domain (**+2.648%**, [+2.068, +3.287] on FreshRetailNet-LT) while
failing catastrophically on another (**−193.9%** on UCI Online Retail II). At that point a
pre-registered stopping rule was triggered and routing development ended.

The UCI result is reported at full strength rather than omitted. A measurable oracle
opportunity does not imply a learnable routing function that transfers across domains and
across time, and this paper proposes no such router.


<!-- ===== 6 Discussion  (source: discussion_v1.md) ===== -->

## 6.1 A finite-sample reading

The two formulations estimate the same quantity. The product `p_t · mu_t` and a directly
regressed conditional mean describe the same function of the history, so nothing in these
results can be read as one formulation being able to represent something the other cannot.
What differs is how a fixed parameter budget, a fixed objective and a fixed number of
gradient steps are spent.

That is the whole content of the comparison, and it constrains the conclusions in a
specific way. When the occurrence sequence carries exploitable structure, dedicating part of
the budget to an explicit occurrence head appears to pay for itself; when the magnitude
sequence is persistent, the same split appears to cost more than it returns, because a
directly regressed mean can absorb a slowly varying level without maintaining two heads.
Both statements are about **estimation under a budget**, not about asymptotic superiority,
and neither is offered as a general claim about intermittent demand.

The practical reading is correspondingly narrow. These results give no reason to prefer one
formulation over the other in the abstract; they give a reason to expect the choice to
matter conditionally, and they indicate which conditions to look at.

## 6.2 Why the two axes behave differently

The most informative pattern is that the two dependence axes respond to different features
of dependence: for occurrence it is the strength of dependence that tracks the comparison,
almost regardless of sign, while for magnitude it is the direction, with persistence
specifically moving the comparison toward direct prediction.

A plausible account is that the two heads face different estimation problems. The occurrence
head estimates a bounded probability, and dependence of either sign — clustering or
alternation — makes that probability more predictable from recent history than the marginal
rate; the *amount* of structure is what helps, not its polarity. The magnitude head
estimates an unbounded conditional level, and persistent magnitude means the level drifts
slowly, which is precisely the regime in which a single directly regressed mean tracks well
without paying for a second head.

This is an interpretation of a controlled observation, not a demonstrated mechanism. The
component-attribution diagnostic of Section 4.7 localizes error to the occurrence head under
one synthetic configuration; it does not identify a cause, and on real data the learned
occurrence head shows no skill advantage at all. The account above should be read as the
hypothesis the results make most natural, and as something a future design could test
directly.

The asymmetry also dissolves an apparent contradiction between the two stages. Stage 1's
structured magnitude arm is alternation, which sits at negative `ρ_M`, and those cells are
factorized-favourable; only persistence, which Stage 1 never generated, produces a
direct-favourable region. Reading Stage 1 alone would have suggested that structured
magnitude is uniformly unfavourable to factorization. It is not; the sign is what matters,
and only the signed sweep can see that.

## 6.3 Synthetic structure and its entanglement in observed demand

Two things transfer to real demand and one does not, and the split is the most useful part
of the empirical section.

What transfers is *conditional information*. The occurrence relationship appears as an
analogue on both datasets and strengthens within the intermittent regime, and a rule frozen
from the controlled sweep selects previously unseen series toward direct prediction on an
independent population, reproducibly across seeds. A pattern found by manipulating a
generating parameter turns out to carry usable signal about which observed series behave
which way.

What does not transfer is the *isolated association*. After adjusting for overlap in scale,
sparsity, positive-demand variability and occurrence dependence, the association is no longer
distinguishable from zero. The reason is visible in the covariates rather than in the
outcome: in the controlled study the three descriptors are set independently by
construction, whereas in observed demand they arrive together, and the selected and control
populations differ substantially in scale as well as on the intended axis.

It is important not to over-read this in either direction. The disappearance is **not**
evidence that the synthetic mechanism is wrong, and the paper does not attribute it to scale
as a cause; the controlled design contains no scale axis and cannot arbitrate the question.
Equally, the selector's success is **not** evidence that the mechanism replicated. These are
two results, reported separately and labelled separately, and a reader who wants one summary
sentence should take the weaker one: the configuration is a useful predictor whose
underlying association is confounded in observed demand.

## 6.4 Complementarity is not a routing function

Section 5.9 reports a measurable opportunity and a failure to convert it. Both belong in the
paper.

The opportunity is genuine: an origin-level oracle over the two forecasts beats the best
static mixture of them, and the ceiling grows when the expert pair is deliberately chosen to
be less correlated. That is the strongest available evidence that the conditional
differences documented in Sections 4 and 5 are real and exploitable in principle.

The conversion did not survive contact with new domains. A gate frozen after development
lost to a static mixture on the first external dataset; successive changes to the target,
the loss, the aggressiveness, the capacity and finally the input representation did not
recover cross-domain transfer, and the final representation experiment improved one external
domain while failing severely on another. A pre-registered stopping rule ended the search
rather than allowing it to continue until something worked, which is the reason this appears
as a boundary and not as an unreported negative.

The general lesson is worth stating plainly because it is easy to get wrong: an oracle
gap measures how much a perfect selector *could* gain, and says nothing about whether a
selector estimable from data will generalize. In this study the gap between the two was the
whole story.

## 6.5 Limitations

**One backbone family.** Every arm in the synthetic study and in the routing chain is
DLinear. The comparison is therefore between two representations *within* one function
class, and it is possible that a different backbone would redistribute the finite-sample
advantage. This is the study's largest single exposure. Testing a second backbone would
strengthen the characterization and is not required to state it; no such experiment was run
for this paper.

**Synthetic simplicity.** The generating process excludes trend, calendar seasonality,
hidden regime switching, heavy tails, test-time shift, phase jitter and interval–magnitude
cross-correlation. That exclusion is what makes the marginal control interpretable, and it
is also why the controlled results are a characterization of a mechanism rather than a
forecast of behaviour on any particular real catalogue.

**Stage 1's conditional validity.** Stage 1's structured arm is deterministic alternation.
It establishes that ordering matters under fixed marginals and that the two axes interact;
every graded statement about the strength or the sign of dependence rests on Stage 2 alone.

**H3's construct mismatch.** The synthetic sparsity contrast is a mean interval of 4 against
8; the pre-registered external test splits at an ADI near 1.3. The non-replication is
reported, and it is not read as a refutation, because the tested contrast is not the
synthetic one. Re-testing at a contrast the real data can support would settle it; the
support exists but was not used as a primary test here.

**No occurrence-head skill on real data.** The synthetic component-attribution result has no
counterpart in the empirical data, where the learned occurrence head shows no skill
advantage. Any mechanism story is therefore synthetic-only.

**Routing instability.** Learned routing is reported as failed, not as unfinished. The
stopping rule was triggered deliberately.

**Absolute accuracy is not the contribution.** Both neural formulations are outranked by
classical estimators on both datasets. The paper compares two representations under one
budget and makes no competitiveness claim.

**A literature limitation, distinct from the above.** One of the closest prior studies could
not be obtained in full text, so what it holds constant across its correlation levels is
unverified. Nothing in this paper's empirical or synthetic results depends on that, and no
statement is made about that study's control design; it is recorded because the novelty
boundary was drawn with it graded conservatively.

## 6.6 What would move this forward

A second backbone would test whether the finite-sample characterization is a property of the
representation pair or of DLinear. A design with a scale axis would let the entanglement
question of Section 6.3 be settled rather than bounded. A per-observation occurrence
diagnostic on real data would test whether the synthetic attribution has any empirical
counterpart. None of these is promised here, and none is required for the claims that are
made.


<!-- ===== 7 Conclusion  (source: conclusion_v1.md) ===== -->

Intermittent demand can be forecast by predicting the conditional mean directly or by
factorizing it into an occurrence probability and a conditional positive magnitude. Both
target the same quantity, and on standard benchmarks neither is uniformly better in
aggregate. This paper asked when each one does better, and answered it by holding the
marginal properties of the demand fixed and varying only its temporal organization.

Under a matched backbone, parameter count, training procedure and budget, the relative
behaviour of the two formulations moves substantially with temporal structure, and the two
structural axes interact rather than add. A signed dependence sweep resolves that
interaction into an asymmetry: on the occurrence axis it is the strength of dependence that
tracks the comparison, while on the magnitude axis it is the direction, with persistence
moving the comparison toward direct prediction. Within the eighteen-cell grid one
configuration — sparse, occurrence-unpredictable, magnitude-persistent — favours direct
prediction with a clear margin.

That characterization reaches observed demand as a conditional signal and stops there. The
occurrence relationship appears on two public retail datasets as an empirical analogue, and
a rule frozen from the controlled sweep shifts previously unseen series toward direct
prediction on an independent population. The isolated association behind it does not survive
adjustment for overlap, because the descriptors the controlled design separates arrive
together in real data. Routing between the two formulations has a measurable oracle
opportunity that did not convert into a router transferring across domains or time.

The contribution is therefore a characterization and its boundary rather than a
recommendation: which conditions make factorizing intermittent demand help or hurt at a
fixed budget, and how far that answer currently travels.


<!-- ===== Figure captions  (source: figure_captions_v1.md) ===== -->

## Figure 1 — the problem and the controlled design

> **Figure 1. The comparison and the design that isolates it.**
> **(a)** Two schematic intermittent series with identical marginal properties — the same
> gap support, the same average interval and the same positive-demand distribution — that
> differ only in the order in which gaps and magnitudes arrive. **(b)** The two forecasting
> formulations compared throughout: a direct arm predicting the conditional mean, and a
> factorized arm predicting an occurrence probability and a conditional positive magnitude
> and multiplying them. **(c)** The matched experimental conditions (Table 1): one backbone,
> 5,856 parameters in each arm, one optimizer, one training budget and one evaluation
> target. Panel (a) is a schematic; the three dependence levels drawn are design levels, and
> the panel is drawn symmetrically in the two axes so that no outcome is implied.

Scope note for the caption file, not for the manuscript: Figure 1 must not hint at either
`|ρ_I|` or the sign of `ρ_M` being the operative quantity. That is a Section 4.8 result.

## Figure 2 — the controlled synthetic result

> **Figure 2. Relative performance across the Stage 2 dependence sweep.**
> `G = 100(1 − RMSE_H / RMSE_P)` in percentage points, measured against the exact oracle
> conditional mean; **positive values favour the factorized arm and negative values favour
> the direct arm**. **(a)** All 18 cells, as two 3 × 3 panels for `d = 4` and `d = 8`, over
> `ρ_I` and `ρ_M`. `G` is printed in every cell. The colour scale is diverging, zero-centred
> and symmetric over [−23, +23], and is shared by both panels. An **open circle marks a cell
> whose 95% interval includes zero**; significance is shown by this marker and not by
> colour. **(b)** Mean `G` at each level of `ρ_I`, averaged over the six contributing cells.
> **(c)** Mean `G` at each level of `ρ_M`, on the same y-axis as (b).
> **In (b) and (c) the vertical extent shows the spread of the six contributing cells
> and is not a confidence interval.**
> Inference in the text rests on the factor-model coefficients and on the per-cell
> intervals in (a).

## Figure 3 — empirical transfer and its boundary

> **Figure 3. What transfers to observed demand, and what does not.**
> **(a)** H1: the association between the strength of interval dependence and the relative
> advantage of the factorized arm, on M5 and Favorita, with the by-regime estimates inset.
> **(b)** H2: the frozen selector applied to an independent M5 population, shown for each of
> three training seeds alongside the pooled estimate. **(c)** The overlap boundary: covariate
> balance before and after weighting, and the isolated association, whose interval crosses
> zero after adjustment. Panels (a) and (b) report transfer; panel (c) reports its limit.
> All intervals are 95% percentile intervals from a paired series bootstrap with 2,000 draws.

No routing panel appears in any main figure.

---

## Appendix figures

```
A1  classical baselines by mean rank, M5 and Favorita
A2  the routing chain: oracle opportunity, expert diversity, and the external results
    including the UCI failure at full scale
A3  Stage 1 per-cell G with intervals, all eight cells
```


<!-- ===== Table captions  (source: table_captions_v1.md) ===== -->

> **Table 1. Matched experimental conditions.** Every dimension the comparison does not
> intend to vary, for the direct and factorized arms of the controlled study. Parameter
> counts are 5,856 in each arm by construction; the secondary zero-truncated
> negative-binomial arm carries 5,857, a difference of 0.017%, inside the pre-registered 1%
> parameter-match rule. The checkpoint criterion is the validation realized-`y` mean squared
> error and is identical for every model and cell; selection on the oracle target, on test
> results, on a model-specific metric or by per-cell tuning is prohibited.

> **Table 2. Stage 1 factorial contrasts.** Effects on `G` in percentage points with 95%
> paired-bootstrap intervals, factorized against direct, over the eight-cell fixed-marginal
> factorial. Positive values favour the factorized arm. The structured arms of this design
> are deterministic period-2 alternation, so these contrasts establish that ordering matters
> and that the two axes interact, and do not measure a graded dependence axis.

> **Table 3. Core empirical validation datasets.** M5 and Favorita only. Series counts,
> lengths, training cut-offs, evaluation origins and the lookback/horizon used. The sample is
> balanced at 300 series per SBC regime class, subject to at least 20 positive training
> observations, and is therefore balanced rather than representative of either catalogue.
> FreshRetailNet-LT and UCI Online Retail II are domain- and time-transfer stress tests used
> only in Section 5.9; their protocol is in the appendix.

> **Table 4. Empirical evidence summary.** For each hypothesis: population, point estimate,
> 95% interval, status and the reading the status licenses. H1 is an empirical analogue
> rather than a replication. H2 is reported as two claims — the frozen selector transfers,
> and the isolated mechanism does not survive overlap adjustment. H3 is a non-replication at
> the pre-registered split and a construct mismatch, since the synthetic contrast is a mean
> interval of 4 against 8 while the external split is at the ADI median.

---

## Appendix tables

```
A0  appendix dataset table: FreshRetailNet-LT and UCI Online Retail II protocol,
    including AVAILABILITY_UNKNOWN
A1  classical baseline mean ranks, both datasets
A2  routing chain results, including the UCI outcome
A3  Stage 1 per-cell G and the component-attribution columns
A4  Stage 2 per-cell G with intervals, all 18 cells
```

## Caption discipline check

```
new scientific claim introduced in a caption          0
number in a caption absent from results_number_source_map.csv   0
"significant" used without naming the interval        0
Figure 2B/2C spread described as a confidence interval 0  (explicitly denied)
routing shown in a main figure                        0
```
