# Claim ledger

Every number below was read from the result artifact named beside it, not from the
history files and not from conversation. Where an artifact does not exist in this
repository the claim is marked so and no number is asserted.

Grades: **CONFIRMED** / **SUPPORTED_WITH_BOUNDARY** / **NOT_REPLICATED** / **REJECTED**.

---

## L1. Neither factorization dominates on real intermittent demand — CONFIRMED

**Support.** `stage_a_results.json`: M5 mean delta −0.00066, Hurdle wins 47.2% of series;
Favorita mean delta −0.03197, Hurdle wins 49.3%. RMSE marginally favours Point on both,
MAE favours Hurdle on both (M5 2.289 vs 2.180; Favorita 3.302 vs 3.166).
`gate_potential.json` M5: point 1.0226, hurdle 1.0214, and a 50:50 mixture reaches 1.0087 —
better than either expert alone.

**Counterevidence.** None. Both datasets and both metrics agree that the gap is small and
its sign depends on the metric.

**Allowed.** "Neither representation dominates on average; which one wins depends on the
metric and on the series."
**Forbidden.** "Point and Hurdle are equivalent." (They are not: they differ per series,
which is the whole point of L2–L4.)

---

## L2. H1 — occurrence-interval dependence is associated with relative Hurdle advantage — SUPPORTED_WITH_BOUNDARY

**Support.**
- `stage_a_results.json`: Spearman(|rho_interval|, delta) M5 **+0.1065** CI [+0.0437, +0.1652];
  Favorita **+0.0789** CI [+0.0205, +0.1405]. Both exclude zero.
- Eligibility threshold 15/20/30 changes it by less than 0.001 on both datasets.
- `posthoc_diagnostic.json`: three scales (raw, relative, scaled) on both datasets — six
  estimates, all positive, all six CI exclude zero.
- `regime_h1.json`, SBC intermittent regime (n=300): relative **+0.153** CI [+0.052, +0.261],
  scaled **+0.123** CI [+0.013, +0.227].

**Counterevidence.**
- `regime_h1.json` lumpy (n=300): +0.014 / +0.032 / +0.028, every CI includes zero. The
  target regime that most resembles the synthetic setting shows nothing.
- `regime_h1.json` raw scale in the intermittent regime: +0.107, CI [−0.009, +0.215] includes zero.
- `posthoc_diagnostic.json` adjusted association: standardized coefficient on |rho_interval|
  is **+0.032** (M5, CI [−0.021, +0.763] on the raw coefficient) and **+0.017** (Favorita),
  small and not separated from zero once log ADI, CV² and log scale are in the model. The
  artifact tags this POSTHOC_DIAGNOSTIC and explicitly not an H1 primary test.
- The Stage A pool is a **regime-balanced** sample (300 per SBC class) while M5's full pool is
  23,053 intermittent / 5,942 lumpy / 984 smooth / 496 erratic. The pooled +0.1065 is therefore
  not a population estimate.

**Allowed.** "In two public datasets the association between occurrence-interval dependence
and the relative advantage of the factorized model is positive and stable across three error
scales, and it is present in the intermittent regime."
**Forbidden.** "Temporal occurrence dependence causes Hurdle superiority." /
"H1 holds across all intermittent regimes." / "|rho_interval| predicts the gap after
controlling for sparsity and scale."

---

## L3. H2 as a predictive selector — CONFIRMED (on M5), SUPPORTED_WITH_BOUNDARY (across datasets)

**Support.**
- `rule_replication/primary_result.json`, an M5 population disjoint from the screen
  (candidate n=675, control n=5,018): rule effect **−0.0230** CI [−0.0294, −0.0163], and the
  Point win rate is **+11.87 pp** higher in the candidate group, CI [+7.85, +15.81]. Both
  exclude zero. The rule was frozen from the screen and nothing was refitted.
- `seed_robustness.json`, seeds 0/1/2: effect −0.0230 / −0.0211 / −0.0277, every CI excludes
  zero; win-rate difference +11.87 / +9.30 / +14.62 pp, every CI excludes zero. Verdict
  `H2_SEED_ROBUST`.
- `favorita_independent/primary_result.json`, unseen Favorita series with the **M5-frozen**
  rule (candidate n=792, control n=4,613): effect CI excludes zero, Point win rate +3.90 pp.
  Verdict `FAVORITA_RULE_PARTIAL_TRANSFER`.

**Counterevidence.**
- `favorita_transfer/primary_transfer.json` on the Stage A Favorita pool (n=18 candidates):
  CI includes zero and the win-rate difference has the **opposite** sign (−14.5 pp). Verdict
  `FAVORITA_RULE_LOW_SUPPORT`. The artifact itself records that this was not the specified
  full-pool replication.
- The screen-stage H2 (`stage_a_results.json`) had CI including zero on both datasets
  (M5 −0.0303 CI [−0.0954, +0.0588]; Favorita −0.0224 CI [−0.1618, +0.1335]).

**Allowed.** "A rule derived from the controlled study and frozen before use shifts unseen M5
series toward the direct model, by about 12 percentage points of win rate, and this survives
three training seeds; the same frozen rule transfers partially to unseen Favorita series."
**Forbidden.** "The rule identifies where Point is better in general." /
"H2 replicates on Favorita." (Two Favorita analyses disagree; say which one.)

---

## L4. H2 as a mechanism — NOT_REPLICATED

**Support for the negative.**
- `rule_replication/secondary_overlap.json`: with overlap weights balancing log ADI, CV²,
  log train scale and |rho_interval| to a worst SMD of 0.0004, the association is
  **+0.0032** CI [−0.0033, +0.0094] — it includes zero and even flips sign relative to the
  unadjusted effect.
- `h2_confirmatory/matching_failed.json`: 1:2 nearest-neighbour matching **failed** with a
  worst absolute SMD of **0.614**, on `log_train_scale`. Candidate and control series differ
  primarily in scale.

**Allowed.** "The rule works as a selector, but once scale, sparsity and occurrence
dependence are balanced the association is not distinguishable from zero, so the rule should
be read as a predictor and not as evidence about magnitude persistence itself."
**Forbidden.** "Magnitude persistence causes the direct model to win." /
"We identify the mechanism behind the Point-favorable regime."

---

## L5. H3 — sparsity strengthens the occurrence effect — NOT_REPLICATED

**Support for the negative.** `stage_a_results.json`: M5 corr_high − corr_low = **−0.0305**
CI [−0.1418, +0.0912]; Favorita **−0.0428** CI [−0.1587, +0.0704]. Both have the sign
opposite to the prediction and both CI include zero.

**Boundary that must be stated.** `posthoc_diagnostic.json` records that Stage A split at the
ADI **median** (M5 1.304) while the synthetic contrast was ADI 4 versus 8; the synthetic-like
groups exist in the data (M5: 127 series with ADI 3–5, 52 with ADI ≥ 8) but were never used
as the primary test. So H3 was not tested at the contrast it was derived from.

**Allowed.** "The sparsity interaction did not replicate at the pre-registered split, and the
pre-registered split did not correspond to the synthetic contrast."
**Forbidden.** "Sparsity does not modulate the occurrence effect." (Not tested at the right
contrast.) / Silently dropping H3.

---

## L6. The occurrence gate has no probabilistic skill in these datasets — CONFIRMED

**Support.** `posthoc_diagnostic.json`: Brier skill score against a per-series constant rate is
**−0.0084** on M5 (CI [−0.0411, +0.0241], includes zero) and **−0.0908** on Favorita
(CI [−0.1401, −0.0438], excludes zero — significantly worse than constant). Per-series BSS is
positive for only 36.8% (M5) and 30.8% (Favorita) of series.

**Boundary.** The artifact records that ROC-AUC, PR-AUC and the Hurdle log loss could not be
computed because Stage A stored per-series aggregates only.

**Allowed.** "In these two datasets the fitted occurrence head does not beat a constant
per-series rate, so whatever drives the H1 association is not a demonstrated gain in
occurrence prediction."
**Forbidden.** "Occurrence predictability explains the Hurdle advantage." (Directly
contradicted.) / Presenting the synthetic gate mechanism as validated on real data.

---

## L7. The backbone is not competitive with classical intermittent methods on M5 — CONFIRMED

**Support.** `classical_benchmark/benchmark.json`, mean rank (lower better), M5:
SBA 3.152, Croston 3.260, TSB 3.411, SES 3.483, **dlinear_point 4.202, dlinear_hurdle 4.220**,
naive 6.820, seasonal_naive 7.453. Favorita: SBA 3.069, Croston 3.787, TSB 3.797, SES 3.940,
dlinear_point 3.947, dlinear_hurdle 4.032. On Favorita `dlinear_point` has the lowest overall
RMSE (20.683) but not the best mean rank.

**Allowed.** "Both neural variants are outranked by SBA on both datasets; the study compares
two representations of one backbone and is not a claim that the backbone is the best
forecaster."
**Forbidden.** Omitting this table. / "Our models are competitive with classical methods."

---

## L8. Point/Hurdle complementarity is real and exploitable in principle — CONFIRMED

**Support.** `convex_oracle.json` M5: best static (50:50) 1.0087, per-origin hard oracle
0.9689, per-origin convex oracle 0.9673 — a **4.11%** convex gain over the best static
mixture. The oracle weight is at 0 for 29.6% of origins, at 1 for 40.3% and interior for
30.1%, so the opportunity is mostly hard selection with a small soft component
(extra soft potential 0.16%).

**Allowed.** "An origin-level oracle over the two experts is about 4% better than the best
static mixture, so the complementarity is real."
**Forbidden.** Reporting the oracle as an achievable result.

---

## L9. Expert diversity enlarges the oracle opportunity — SUPPORTED_WITH_BOUNDARY

**Support.** `expert_set_spec.json`: the selected pair `dlinear_point_plain | naive` has a
geometric convex-ceiling multiplier of **2.15** over the reference Point/Hurdle pair
(worst-dataset multiplier 1.86), with maximum residual correlation 0.855 against the
reference pair's much higher correlation.

**Counterevidence.** `pair_gate_result.json` M5: expert A 1.0366, expert B (naive) 2.1120.
The best static is expert A alone (alpha = 0), and the frozen gate beats it by only
**+0.43%** CI [+0.06, +0.81]. The enlarged ceiling is partly an arithmetic consequence of
pairing a good expert with a much worse one.

**Allowed.** "Choosing a less correlated pair roughly doubles the oracle ceiling, although
the realized gain from routing that pair remains under half a percent."
**Forbidden.** "More diverse experts improve forecasts." (The ceiling grew; the realized
gain did not.)

---

## L10. Learned routing does not survive a change of domain — CONFIRMED

**Support.** `external_benchmark.json`, first TEST scoring of the frozen gate on two datasets
never used in development:
- FreshRetailNet (declared PRIMARY_EXTERNAL_CONFIRMATION before results): proposed gate vs
  the reference static alpha **−2.43%** CI [−2.74, −2.13], CI excludes zero. Worse.
- UCI Online Retail II: **+0.13%** CI [−0.06, +1.13], includes zero; and −0.41% against TSB
  with the CI excluding zero.
- Development results were the opposite: `gate_v2_oof_result.json` `GATE_V2_OOF_GREEN`
  (M5 gate vs point +1.99% CI [+1.69, +2.30]) and `fresh_confirmatory.json`
  `GATE_V2_CONFIRM_GREEN` on 23,513 held-out M5 series.

**Allowed.** "A gate that improved on held-out series of the development datasets was worse
than a static mixture on the first external dataset it was applied to."
**Forbidden.** Letting the later OOF result `GATE_V3_OOF_STRONG` stand in for external
validity — it is an in-train-region cross-fit on the same datasets and does not overturn this.

---

## L11. The handcrafted gate features, not the gate's capacity, are the binding constraint — SUPPORTED_WITH_BOUNDARY

**Support.** `routing_information_ceiling/`: a HistGradientBoosting learner on the identical
feature matrix and identical objective beats the 433-parameter MLP on only 1 of 4 datasets
(M5 +0.286%, CI includes zero) and loses on Favorita (−0.619%), FreshRetailNet (−0.799%) and
UCI (−6.935%, CI excludes zero). Oracle recovery falls on 3 of 4. Diagnosis
`CURRENT_FEATURE_INFORMATION_LIMITED`.

**Counterevidence / boundary.** `capacity_manipulation_check.json`: the frozen bar for
"fits training clearly better" was a 10% gain and the actual in-sample gains were
3.02 / 1.32 / 4.38 / 0.07%. The manipulation was weaker than intended, so this shows
"a far more flexible learner does not generalize better here" more firmly than it shows
"much more usable capacity was supplied".

**Allowed.** "A substantially more flexible learner on the same features does not generalize
better, which points at the features rather than the function class."
**Forbidden.** "We proved the features are the bottleneck."

---

## L12. Raw history carries routing information the summaries discard — SUPPORTED_WITH_BOUNDARY

**Support.** `temporal_routing_encoder/`: on FreshRetailNet the GRU gate goes from the
handcrafted gate's **−0.506%** against the static weight to **+2.648%**, CI [+2.068, +3.287],
beats the handcrafted gate on 3 of 3 folds (+3.138%, CI [+2.557, +3.730]), improves oracle
recovery from −0.028 to +0.148 and lowers p95 tail degradation from 18.24% to 12.81%.

**Counterevidence.** On UCI the same gate is **−193.9%** in aggregate, with one fold at
−265.9%. On that fold expert A's error is 3249.2 against expert B's 86.7, so the optimum is
near g = 1; the handcrafted gate stayed at 0.593 next to its fitted alpha of 0.65 while the
sequence gate moved to 0.295. Its correlation with the oracle weight is the same
(0.381 vs 0.377), so it ranks as well and chooses the wrong level. It also loses to the
handcrafted gate on M5 (−0.220%, CI excludes zero).

**Allowed.** "Reading the raw window recovers routing signal that the summary descriptors do
not carry, on one dataset; on another it leaves the protection of the static weight and fails
catastrophically."
**Forbidden.** "Sequence models solve the routing problem." / Reporting the FreshRetailNet
recovery without the UCI failure in the same sentence.
