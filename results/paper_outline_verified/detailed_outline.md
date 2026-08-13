# Detailed outline

Every subsection carries six fields: **Q** reader question, **P** purpose,
**C** claims, **E** evidence, **A** avoid, **T** transition.

Metric throughout: `G = 100 (1 − RMSE_Hurdle / RMSE_Point)`, G > 0 favours Hurdle.
Stage names: Stage 1 = fixed-marginal 2×2×2 factorial; Stage 2 = stationary ρ sweep.

---

## 1. Introduction

**Q** Why should I care which representation of the conditional mean is used?
**P** Establish that the choice is open, that the usual descriptors cannot settle it,
and that we can settle part of it by construction.
**C** Neither representation dominates on real data; the aggregate is close to a tie
and its sign depends on the metric. The interesting variation is conditional.
**E** M5 mean Δ −0.00066 with Hurdle winning 47.2% of series; Favorita −0.03197,
49.3%; RMSE favours direct, MAE favours factorized on both
(`stage_a_results.json`). Table 2 first block.
**A** "Hurdle models are better for intermittent demand." Any framing where the
paper's answer is a winner.
**T** If the aggregate is a tie, the question becomes *when*, which requires
controlling what the usual descriptors ignore.

---

## 2. Related work

**Q** Hasn't autocorrelation in intermittent demand been studied?
**P** Place the paper narrowly and honestly.
**C** Interval autocorrelation, demand-size autocorrelation and size–interval
cross-correlation are established topics in the intermittent-demand literature. Our
narrower question is the relative *finite-sample inductive bias* of two objectives on
one neural backbone family under controlled temporal structure, and how far it
transfers.
**E** Positioning only.
**A** Any claim to have introduced correlation effects in intermittent demand. Any
suggestion that Croston-family work is a baseline we beat.
**T** To ask the narrow question we first need to say what the descriptors leave out.

---

## 3. Problem setup and forecasting formulations *(merges the original Sections 3 and 4)*

### 3.1 Limits of marginal intermittency summaries

**Q** What exactly is missing from ADI and CV²?
**P** Make the gap concrete and constructive.
**C** ADI and CV² are functionals of the marginal distribution of intervals and
sizes. Two series with identical ADI, identical interval support and identical
positive-demand marginal can still differ in the order of those values. That
difference is what we manipulate.
**E** Design argument; instantiated by the DGP in 4.1, where both arms of each axis
draw from the same two-point support with the same long-run frequency.
**A** Claiming empirically that ADI/CV² fail to predict the gap — we do not measure
that.
**T** With the gap named, the two candidate representations can be stated.

### 3.2 Direct and factorized forecasting

**Q** What are the two things being compared?
**P** Define them and make clear they share a target.
**C** Direct prediction estimates `E[Y|history]`. Factorization estimates
`P(Y>0|history)` and `E[Y|Y>0,history]` and multiplies. They target the same
quantity; they differ in what structure each head can express and in how estimation
error is allocated under a finite budget.
**E** Definitional; the parameter matching in 4.2 is what makes the comparison about
bias rather than capacity.
**A** Asymptotic or function-class superiority claims.
**T** Comparing two estimators of one target requires saying what "relative" means.

### 3.3 Relative finite-sample comparison

**Q** What quantity is reported, and why that one?
**P** Fix the metric and the interpretive frame before any result.
**C** All comparisons use `G = 100 (1 - RMSE_Hurdle / RMSE_Point)`, positive favouring
factorization. Because both estimators target the same conditional mean, differences
under a matched budget are read as differences in finite-sample inductive bias, not
as a statement about which function class can represent the target.
**E** `metric_sign.md`. The three delta conventions in the underlying artifacts are
explained once in Appendix K and never mixed into the main text.
**A** Presenting `G` as a measure of asymptotic superiority.
**T** A comparison of inductive biases requires a setting where the truth is known
and the marginals are held still.

---

## 4. Controlled synthetic study

### 4.1 Data-generating process and marginal control

**Q** How do you change temporal structure without changing anything else?
**P** Establish that the manipulation is clean.
**C** Events are generated as `T_j = T_{j-1} + Q_j`, `Y_t = M_j` at event times.
Gaps take `d−1` or `d+1` with long-run share 0.5 each; magnitudes are
`M_j = 1 + Poisson(λ_j − 1)` with `λ_j ∈ {5,15}` and long-run positive mean 10 in
every arm. The predictable and independent arms of each axis share their marginal and
differ only in order, and are paired by construction: the same event index draws the
same uniform and the same Poisson innovation in both.
**E** `dgp_verification.md`; prereg `GENERATION` block; audit
`marginal_control_pass: true`, d=4 empirical ADI mode difference 0.0015.
Trend, seasonality, hidden regime, heavy tail, test-time shift, phase jitter and
interval–magnitude cross-correlation are forbidden by the design.
**A** Presenting the DGP as realistic. It is a controlled contrast, not a simulator.
**T** A clean manipulation is only half of a controlled comparison; the two models
must also be matched.

### 4.2 Matched comparison and the oracle target

**Q** Is this a fair fight?
**P** Remove capacity and tuning as explanations.
**C** Both arms are DLinear with **5,856 parameters each**, one trainer, identical
optimizer, budget, split, seeds and checkpoint rule, with per-cell tuning prohibited
and the checkpoint chosen on realized-y validation error rather than on the oracle.
Performance is `G` against an **exact dynamic-programming** conditional mean.
**E** `point_hurdle_fairness.md`; `metric_sign.md`; prereg `MODELS`, `TRAINING`,
`ORACLE`. Table 1.
**A** Claiming fairness beyond this backbone family.
**T** With the manipulation clean and the comparison matched, the first question is
whether ordering matters at all.

### 4.3 Stage 1 — fixed-marginal 2×2×2 factorial

**Q** Does temporal structure change the comparison under fixed marginals?
**P** Establish the phenomenon and its joint character.
**C** Yes, substantially, and the two axes do not act independently. Making
occurrence intervals structured moves `G` up by **+7.83 pp** [+6.09, +9.45]; making
magnitudes structured moves it down by **−4.58 pp** [−6.28, −2.92]; sparsity moves it
down by **−6.26 pp** [−8.01, −4.57]. The interaction between the two axes is
**−16.74 pp** [−18.45, −15.05], larger in magnitude than either main effect. Sparsity
amplifies the occurrence effect (+3.35 pp) and does nothing to the magnitude effect
(−0.01, interval spans zero).
**E** `stage1_verified_contrasts.csv`; per-cell table with intervals in
`claim_ledger_frozen.md` §2. Table 2.
**A** A graded reading of the occurrence axis — Stage 1's structured arm is
deterministic period-2 alternation, and its own audit blocks that reading. Also do
not claim a Point-favourable region here: the only negative cell, C08, is −3.01 pp
with the interval touching zero.
**T** The interaction is the largest term in the model, so the next question is what
the two axes are actually measuring.

### 4.4 Stage 1 component-attribution diagnostic

**Q** Which head carries the error?
**P** Give the reader a mechanism-flavoured handle without overclaiming.
**C** Substituting the true component for the estimated one attributes the error. In
the sparse, occurrence-independent, magnitude-structured cell, replacing the
occurrence estimate leaves error 0.2900 while replacing the magnitude estimate leaves
0.9030, against 0.9652 for the full model — the occurrence head dominates there.
**E** Hybrid columns `p_true×mu_hat`, `p_hat×mu_true`, `p_hat×mu_hat` in
`stage1_verified_cells.csv`, cell C03, model M1.
**A** "The mechanism is identified" / "occurrence gating explains the Stage 1
effects". This is `COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT`: it attributes error
under selected conditions, it does not establish a cause. Not a separate Stage.
**T** Attribution says where the error is, not which property of the dependence
matters. That needs a parameterization.

### 4.5 Stage 2 — stationary ρ sweep

**Q** Which property of the dependence actually drives the effect?
**P** Convert a binary contrast into a graded one and resolve the interaction.
**C** Over `d ∈ {4,8} × ρ_I ∈ {−0.8,0,+0.8} × ρ_M ∈ {−0.8,0,+0.8}`, with the mean
interval, interval support and magnitude marginal held fixed, the factorized
advantage rises with occurrence dependence **in both directions**
(C_neg +0.127 [+0.087, +0.166]; C_pos +0.201 [+0.155, +0.244]).
**E** `stage2_verified_cells.csv`, `stage2_verified_factor_effects.csv`,
`stage2_scientific_classification.json`, classification
`CLASS_A_GENERAL_PREDICTABILITY_SUPPORT` with five integrity gates passing including
a no-signal control whose out-of-sample Markov gain is significantly negative.
Figure 2A.
**A** Presenting Stage 2 as a robustness check on Stage 1. It changes the reading.
**T** With both signs available, the two axes can be compared on equal terms.

### 4.6 Occurrence strength versus magnitude direction

**Q** Are the two axes the same kind of thing?
**P** State the paper's sharpest synthetic finding.
**C** No. For occurrence, `|ρ_I|` carries **+0.1904** [+0.1699, +0.2119] against
**+0.0667** for signed `ρ_I` — a factor of 2.9, and the level means trace a U:
+12.10, +3.57, +16.47 pp at ρ_I = −0.8, 0, +0.8. For magnitude the ordering reverses:
signed `ρ_M` carries **−0.0711** [−0.0851, −0.0570] against **−0.0228** for `|ρ_M|` —
a factor of 3.1, and the level means fall monotonically: +14.10, +11.94, +6.11 pp.
Occurrence effects are primarily associated with the strength of dependence;
magnitude effects primarily with its direction.
**E** `stage2_verified_factor_effects.csv`; marginals recomputed from
`stage2_verified_cells.csv`. Figures 2B and 2C.
**A** Stating this in Section 3 or in Figure 1. It is a result, not a premise.
**T** The asymmetry predicts where direct prediction should win, which the sweep can
be checked against.

### 4.7 The Point-favourable region

**Q** Is there anywhere the direct model clearly wins?
**P** Produce the counterexample the empirical section will test.
**C** Within the eighteen-cell grid, only one cell shows statistically clear
superiority for direct prediction: **d = 8, ρ_I = 0, ρ_M = +0.8**, `G = −19.76`
[−26.00, −14.53]. Sparse, occurrence-unpredictable, magnitude-persistent — precisely
what the asymmetry predicts, since occurrence contributes nothing at ρ_I = 0 while
positive magnitude persistence pulls toward the direct model.
**E** `stage2_verified_cells.csv`. Figure 2A, marked cell.
**A** Calling it a Stage 1 finding. Stage 1's C08 does not qualify.
**T** A designed corner is a hypothesis about real data; the next section tests it.

---

## 5. Empirical validation

### 5.1 Datasets and evaluation protocol

**Q** On what, and with what discipline?
**P** Establish that the empirical work was frozen before it was scored.
**C** M5 and Favorita, 1,200 series each, item×store granularity, lookback 96,
horizon 28, three test origins, eligibility `n_positive_train ≥ 20`. The analysis
spec was frozen at 18:06:38 and the results written at 18:11:16.
**E** `stage_a_results.json` manifest; `pre_analysis_spec.json`;
`docs/m5_favorita_data_derivation.md`. Table 3, which lists **M5 and Favorita only**.
FreshRetailNet-LT and UCI Online Retail II are domain- and time-transfer stress tests
used in 5.7; their protocol is in the appendix and they are never described as core
validation data.
**A** Describing the Stage A sample as representative: it is regime-balanced at 300
series per SBC class, while M5's full pool is 23,053 / 5,942 / 984 / 496.
**T** With the protocol fixed, the aggregate comparison sets the baseline.

### 5.2 Overall comparison, and the classical baselines

**Q** How do these models stand in absolute terms?
**P** Pre-empt "why not Croston" and set the paper's scope honestly.
**C** Neither representation dominates. Both are outranked by SBA on both datasets
(M5 mean rank SBA 3.152 < Croston 3.260 < TSB 3.411 < SES 3.483 < direct 4.202 <
factorized 4.220). The study compares two representations of one backbone under one
budget and is not a claim about the best available forecaster.
**E** `classical_benchmark/benchmark.json`. Table 4.
**A** Omitting this table. Any competitiveness claim.
**T** The interesting question is conditional, which is what the synthetic study
predicts.

### 5.3 H1 — the occurrence-dependence analogue

**Q** Does the occurrence relationship appear in real data?
**P** Test the synthetic direction where the manipulation becomes a measurement.
**C** An empirical analogue appears: `|ρ_interval|` is positively associated with the
relative advantage of factorization: Spearman **+0.1065** [+0.0437, +0.1652] on M5 and **+0.0789**
[+0.0205, +0.1405] on Favorita. Positive on all three error scales in both datasets,
six of six intervals clear of zero, insensitive to the eligibility threshold, and
present in the intermittent regime (relative +0.153 [+0.052, +0.261]). The external
use of the **absolute value** is not a convention — it is what Stage 2 found.
**E** `stage_a_results.json`, `posthoc_diagnostic.json`, `regime_h1.json`.
Figure 3A.
**A** "H1 replicates the synthetic result". This is an `EMPIRICAL_ANALOGUE`: the
synthetic ρ is set at three designed levels, the external one is estimated per
series. Numerical comparison across the two is meaningless because the targets
differ.
**T** Direction is one thing; a usable rule is another.

### 5.4 H2 — the frozen Point-favourable selector

**Q** Does the synthetic counterexample pick out real series?
**P** Show out-of-sample transfer of a rule that was frozen before use.
**C** The Stage 2 cell's three axes map one to one onto the frozen rule — high ADI,
low occurrence signal, signed magnitude persistence — including the sign. On an
independent M5 population of 675 candidates against 5,018 controls the rule shifts
series toward direct prediction: effect **−0.0230** [−0.0294, −0.0163], Point win
rate **+11.87 pp** [+7.85, +15.81], reproduced under three training seeds. It
transfers partially to unseen Favorita series.
**E** `rule_replication/primary_result.json`, `seed_robustness.json`,
`favorita_independent/primary_result.json`, and the contradicting
`favorita_transfer/primary_transfer.json`, which is reported beside it.
Figure 3B.
**A** Presenting only the favourable Favorita analysis.
**T** A rule that selects is not a mechanism that explains, and the data can tell
them apart.

### 5.5 The entanglement boundary

**Q** Does the synthetic mechanism transfer too?
**P** This is the paper's most honest moment and should be written as a finding.
**C** No. With overlap weights balancing sparsity, positive-demand variability,
scale and occurrence dependence to a worst standardized difference of 0.0004, the
association is **+0.0032** [−0.0033, +0.0094] — it includes zero and changes sign.
Matching failed outright at a worst SMD of **0.614**, on log scale. Candidate and
control series differ primarily in scale, an axis the synthetic design does not
contain and therefore cannot arbitrate.
**E** `rule_replication/secondary_overlap.json`, `h2_confirmatory/matching_failed.json`.
Figure 3C.
**A** Softening this. The correct statement is that the rule predicts without
explaining, and that the controlled study cannot settle the mechanism because it has
no scale axis.
**T** One synthetic finding remains untested at its own contrast.

### 5.6 H3 and the sparsity-transfer boundary

**Q** Does the sparsity interaction carry over?
**P** Report a non-replication and the reason it is not a refutation.
**C** The interaction is real and significant in both controlled studies — Stage 1
sparsity×interval **+3.35 pp** [+1.62, +5.09], Stage 2 `d×ρ_I` **+0.0332**
[+0.0194, +0.0471]. The external test did not replicate it: M5 −0.0305
[−0.1418, +0.0912], Favorita −0.0428 [−0.1587, +0.0704], both signs opposite to the
prediction. But the external split is at the ADI **median**, 1.304 and 1.317, while
the synthetic contrast is ADI 4 versus 8. The external test therefore compared
ADI ≈ 1.1 against ≈ 2 and did not evaluate the synthetic claim.
**E** `stage_a_results.json`; `posthoc_diagnostic.json` `h3_adi_support` (M5 127
series at ADI 3–5, 52 at ≥ 8; Favorita 84 and 45).
**A** "Sparsity does not modulate the occurrence effect."
**T** If the advantage is conditional but the conditions are entangled, the natural
engineering response is to learn the routing — which is the last thing to test.

---

### 5.7 Adaptive-use boundary: learned routing does not transfer robustly

*Roughly half a page to one page. This is C3, a supporting result, not a section.*

**Q** Can the conditional advantage simply be exploited?
**P** Close the loop the paper's own implication opens, and stop.
**C** The opportunity is real and was deliberately enlarged: a per-origin oracle over
the two experts is **4.11%** better than the best static mixture, and choosing a less
correlated expert pair multiplies that ceiling by **2.15**. It was not converted. A
gate frozen after development was **−2.43%** [−2.74, −2.13] against a static mixture
on the first external dataset it saw. Successive redesigns of the target, the loss,
the aggressiveness, the capacity and the input representation did not recover it; the
last of these recovered FreshRetailNet (+2.648% [+2.068, +3.287]) and failed
catastrophically on UCI (−193.9%). A pre-registered stop rule was triggered.
**E** `structure_gate/convex_oracle.json`, `expert_diversity/expert_set_spec.json`,
`multi_benchmark/external_benchmark.json`, `temporal_routing_encoder/`.
**A** Expanding this into a method contribution, or into an architecture-development
narrative. Presenting the FreshRetailNet recovery without the UCI failure in the same
sentence. Reporting the oracle as achievable. The two stress-test datasets used here
are introduced in one sentence with a pointer to the appendix, so no reader mistakes
them for core validation data.
**T** What remains is to say what all of this licenses.

---

## 6. Discussion

### 6.1 Finite-sample interpretation

### 6.2 Why occurrence and magnitude structure differ

**C** Occurrence structure gives the occurrence head something to estimate that the
direct head must infer implicitly; magnitude persistence gives the direct head a
smoother target while splitting the estimate across two heads costs sample
efficiency. Under a fixed parameter budget these trade off, and the asymmetry
(strength for occurrence, direction for magnitude) is the shape of that trade-off.
**A** Presenting this as demonstrated causation. It is an interpretation consistent
with the component-attribution diagnostic.

### 6.3 Synthetic-to-real entanglement

**C** In a designed factorial the axes are orthogonal by construction. In real demand
they are not: the strongest single obstruction we measured is scale, which is absent
from the synthetic design and which removes the H2 association once balanced. The
generalizable lesson is that a controlled axis can transfer as a *predictor* while
failing to transfer as an *explanation*.

### 6.4 What the routing failures imply

**C** A measurable oracle opportunity is not evidence that a routing function is
learnable. The gap between the two was large enough to be worth reporting, and closing
it is not a matter of tuning: target, loss, aggressiveness, capacity and input
representation were each varied without recovering cross-domain transfer.
**A** Framing this as future work that a better architecture would solve.

### 6.5 Limitations

**C** One backbone family throughout. No scale axis in the controlled design. Stage 1
is `CONDITIONALLY_VALID`, its structured arm being alternation only. H3 was never
tested at its own contrast. The learned occurrence head shows no skill against a
constant per-series rate on either real dataset (Brier skill −0.008 M5, −0.091
Favorita), so the real-data sections do not rest on the occurrence mechanism.
Point metrics only; the pre-registration is on RMSE and MAE.

---

## 7. Conclusion

**C** A measurable, controlled account of *when* factorizing intermittent demand
helps and when it hurts, an asymmetry that names the operative property of each axis,
a counterexample that transfers as a selector, and a boundary where it stops
transferring as a mechanism.
**A** Any closing sentence that reads as advocacy for either representation.

---

## Appendix map

```
A  H1 threshold sensitivity, adjusted partial association
B  occurrence-gate Brier skill
C  H3 in full, and the contrast discrepancy
D  H2 three-seed table; both Favorita analyses
E  Stage 1 full cell table and component-attribution columns
F  Stage 2 full 18-cell table with intervals
G  Stage 1 validity audit; no-signal control evidence
H  routing chain: Gate-v1/v2/v3, P0L1, Safe-P0L1, HGB, sequence gate
I  dataset audit including UCI AVAILABILITY_UNKNOWN
J  pre-registration documents, freeze timestamps, artifact hashes
K  the three delta conventions and why the paper uses G
L  future work: second backbone, per-observation occurrence diagnostic,
   H3 at ADI 3-5 vs >= 8
```
