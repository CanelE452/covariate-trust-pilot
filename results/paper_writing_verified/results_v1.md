# 4.6–5.9 Results — draft v1

Every displayed number resolves to a verified artifact through
`results_number_source_map.csv`. No quantity was recomputed and no new inferential
statistic was constructed; the tables and figures are re-aggregations of the same
verified files. `G = 100(1 − RMSE_H / RMSE_P)`, positive favouring the factorized arm.

---

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
