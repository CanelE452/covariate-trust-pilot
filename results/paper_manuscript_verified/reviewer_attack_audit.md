# Reviewer attack audit — manuscript v1

Twelve objections, each stated at the strength a hostile reviewer would use, with the
manuscript's answer, where the evidence sits, the wording limit that keeps the answer
honest, and whether an additional experiment is required.

---

## R1 — Single backbone

**Objection.** Every arm is DLinear. You have characterized DLinear, not the two
representations. A transformer or an RNN might reverse the asymmetry entirely.

**Answer.** Conceded, and stated three times: Methods 3.4 frames the whole comparison as
finite-sample behaviour within one function class, Discussion 6.5 names it as the study's
largest single exposure, and the contribution wording never generalizes past it.

**Evidence.** Methods 3.4; Table 1; Discussion 6.5 first entry.
**Wording limit.** No sentence says "direct and factorized forecasting" behave a certain
way in general; every claim is scoped to the matched design.
**Additional experiment required.** NO for the claims as written. A second backbone is
`NICE_TO_HAVE` and would strengthen C1 rather than rescue it.
**Severity.** HIGH as a reviewer talking point; LOW as a threat to the stated claim.

## R2 — Synthetic DGP simplicity

**Objection.** Your generator has no trend, no seasonality, no regime switching, no heavy
tails and no cross-correlation. Real demand has all of these. The result is an artefact of
a toy process.

**Answer.** The exclusions are the design, not an oversight: they are what makes the
marginal control interpretable, and they are listed explicitly in Methods 4.1. The paper
then does the honest thing and tests transfer on real data, reporting both what carries
(5.5, 5.6) and what does not (5.7, 5.8).

**Evidence.** Methods 4.1 forbidden-factor list; Sections 5.5–5.8.
**Wording limit.** The synthetic study is described as a characterization of a mechanism
under control, never as a prediction about any catalogue.
**Additional experiment required.** NO.
**Severity.** MEDIUM, and partially answered by the empirical sections.

## R3 — Stage 1 is only `CONDITIONALLY_VALID`

**Objection.** Your headline factorial uses deterministic alternation. That is one extreme
point, not a dependence axis, and you are generalizing from it.

**Answer.** The paper says so before the reviewer does. Section 4.6 states the limitation
in the same subsection as the result, names the classification, and names the two readings
the source audit blocks. Every graded claim is attributed to Stage 2, and `ρ` is not used
in Stage 1 at all.

**Evidence.** Methods 4.3; Results 4.6 closing paragraphs; notation registry (`ρ` is
Stage 2 only).
**Wording limit.** Stage 1 is never cited alone for a graded claim.
**Additional experiment required.** NO — Stage 2 already exists and carries the load.
**Severity.** LOW, because the mitigation is in the design rather than in the prose.

## R4 — Stage 1 and Stage 2 are differently parameterized

**Objection.** Stage 1's structured arm is `ρ = −1`; Stage 2 sweeps `±0.8`. You are
splicing two incompatible experiments and calling the result one finding.

**Answer.** The relationship is stated rather than smoothed over. Results 4.8 explains that
Stage 1's magnitude arm is alternation, which sits at negative `ρ_M`, and that only positive
`ρ_M` — never generated in Stage 1 — produces a direct-favourable region. The apparent
tension between the stages is resolved by the asymmetry rather than hidden by it.

**Evidence.** Results 4.8 penultimate paragraph; Discussion 6.2 closing paragraph.
**Wording limit.** No Stage 1 quantity is plotted or reported as a point on a Stage 2 axis.
**Additional experiment required.** NO.
**Severity.** MEDIUM if unaddressed; the manuscript addresses it directly.

## R5 — Synthetic-to-real construct mismatch

**Objection.** Your synthetic axes and your real-data measurements are not the same
quantities. The synthetic study varies a generating parameter and scores against an exact
oracle; the empirical test measures an autocorrelation and scores against realized demand.

**Answer.** Conceded explicitly, and it is why H1 is labelled an **empirical analogue**
rather than a replication. Methods 4.2 states that the two targets are not numerically
comparable, and no table or sentence places a synthetic and a real error side by side.

**Evidence.** Methods 4.2 closing; Methods 5.2; Results 5.5 closing paragraph.
**Wording limit.** The word "replicates" is never applied to H1.
**Additional experiment required.** NO.
**Severity.** MEDIUM, mitigated by labelling.

## R6 — The selector is not the mechanism

**Objection.** Your rule predicts, but your association vanishes under adjustment. You are
selling a confounded finding as a mechanism.

**Answer.** The paper reaches that conclusion itself, in its own words, and reports the two
as separate results with separate statuses (`CONFIRMED` and `NOT_REPLICATED`). Discussion
6.3 instructs the reader who wants one sentence to take the weaker one.

**Evidence.** Results 5.6 and 5.7; Table 4 rows H2 and H2-isolated; Discussion 6.3.
**Wording limit.** The disappearance is never attributed to scale as a cause; the design has
no scale axis and the manuscript says so.
**Additional experiment required.** NO for the claim as written. A synthetic design with a
scale axis would settle it and is listed in 6.6.
**Severity.** LOW as written; HIGH if the two claims were ever merged.

## R7 — H3 does not replicate

**Objection.** One of your three pre-registered hypotheses fails, with the wrong sign on
both datasets.

**Answer.** Reported at full strength in Section 5.8, including both point estimates and
both intervals, and carried into Table 4 and the limitations. The construct mismatch is
declared in **Methods** 5.2, before any result is shown, so it cannot read as a post-hoc
excuse.

**Evidence.** Methods 5.2 H3 paragraph; Results 5.8; Table 4; Discussion 6.5.
**Wording limit.** The non-replication is not called a refutation, and it is not called a
success either.
**Additional experiment required.** NO. Re-testing at a contrast the data support
(`ADI 3–5` against `≥ 8`, where M5 has 127 against 52 series) is `NICE_TO_HAVE`.
**Severity.** MEDIUM.

## R8 — The occurrence head has no skill on real data

**Objection.** Your synthetic mechanism story rests on the occurrence head, and on real
data that head has negative Brier skill. The mechanism is synthetic-only.

**Answer.** Agreed, and stated in exactly those terms. Results 5.7 reports both Brier skill
values including the interval, and Discussion 6.5 states that any mechanism story is
synthetic-only.

**Evidence.** Results 5.7 final paragraph; Discussion 6.2 and 6.5.
**Wording limit.** The component-attribution result is labelled
`COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT`, synthetic only, and never used to support a
causal claim.
**Additional experiment required.** NO. A per-observation occurrence diagnostic on real
data is `NICE_TO_HAVE`.
**Severity.** MEDIUM.

## R9 — Routing instability

**Objection.** You spent a whole development chain on routing and it failed, including a
−193.9% result. Why is this in the paper?

**Answer.** Because omitting it would misrepresent what was tried. Section 5.9 reports the
oracle opportunity and the failure together, names the UCI figure at full scale, and states
that a pre-registered stopping rule ended the search. Discussion 6.4 draws the general
lesson: an oracle gap does not imply an estimable selector.

**Evidence.** Results 5.9; Discussion 6.4; appendix routing table.
**Wording limit.** No router is proposed; the section is titled as a boundary.
**Additional experiment required.** NO — and routing work is on `DO_NOT_RUN`.
**Severity.** LOW, provided the negative stays visible. It is checked mechanically (F9).

## R10 — Absolute accuracy is poor

**Objection.** Both of your models are beaten by SBA, a method from 2005. Why should anyone
care which of two weak models is weaker?

**Answer.** The comparison is between two representations of one backbone under one budget,
and Section 5.4 puts the classical ranks in the main text specifically so that no reader
can mistake it for a competitiveness claim. The question the paper answers — which
representation to choose, conditionally — is orthogonal to which family is strongest.

**Evidence.** Results 5.4; Discussion 6.5 final entry.
**Wording limit.** No competitiveness or state-of-the-art claim anywhere; checked
mechanically (C-A5).
**Additional experiment required.** NO.
**Severity.** MEDIUM as a framing risk; the mitigation is to keep 5.4 in the main text.

## R11 — The dependence literature got there first

**Objection.** Altay and colleagues varied all three correlation structures in generated
intermittent demand years ago. What is new?

**Answer.** Related Work 2.3 concedes this in as many words and closes by stating that no
part of the study is positioned as introducing temporal dependence. The distinction drawn
is the object of comparison: that work compares estimators inside one already-factorized
representation, while this study compares the representations themselves.

**Evidence.** Related Work 2.3; Related Work 2.5; the frozen novelty boundary.
**Wording limit.** Nothing is said about what that study holds constant — its full text
could not be obtained, and the manuscript's dependency on that unresolved detail is zero,
checked mechanically (L1).
**Additional experiment required.** NO. Obtaining the full text is a submission follow-up.
**Severity.** LOW.

## R12 — The representation comparison also has precedents

**Objection.** Kourentzes compared a direct rate against a decomposed network, and a 2026
paper compared single-stage against a probability-times-size two-stage model. Both of your
representations have been compared before.

**Answer.** Both are conceded without qualification in Related Work 2.4, which ends by
stating that neither decomposition itself nor the direct-versus-factorized comparison is
the focus of the contribution. The paper's claim is the comparison read **as a function of**
separately controlled dependence, at one budget.

**Evidence.** Related Work 2.4 and 2.5; `precedent_intersection_map.md` components 5, 6a
and 6b.
**Wording limit.** Kourentzes' decomposed arm is described as a size-over-interval
**ratio**, never as this paper's product form; the 2026 comparison is described as
gradient-boosting, never as neural. Both checked mechanically (L2, L3). The matched
condition is stated as a property of this design, never as something prior work lacks.
**Additional experiment required.** NO.
**Severity.** HIGH if mishandled; the guards exist precisely because it is.

---

## Summary

```
risk  severity  additional experiment      status
------------------------------------------------------------------
R1    HIGH*     NO   (NICE_TO_HAVE)        conceded, scoped
R2    MEDIUM    NO                         conceded, transfer tested
R3    LOW       NO                         mitigated in design
R4    MEDIUM    NO                         resolved in 4.8 and 6.2
R5    MEDIUM    NO                         labelled analogue
R6    LOW       NO   (NICE_TO_HAVE)        two claims, two statuses
R7    MEDIUM    NO   (NICE_TO_HAVE)        declared in Methods
R8    MEDIUM    NO   (NICE_TO_HAVE)        stated as synthetic-only
R9    LOW       NO   (DO_NOT_RUN)          negative reported in full
R10   MEDIUM    NO                         classical ranks in main text
R11   LOW       NO   (submission follow-up) conceded in 2.3
R12   HIGH*     NO                         conceded in 2.4, guarded

* HIGH means "a reviewer will certainly raise it", not "the claim fails".
```

```
scientific MUST_HAVE experiments     NONE
NICE_TO_HAVE                         second backbone; scale axis in the synthetic design;
                                     per-observation occurrence diagnostic on real data;
                                     H3 re-tested at ADI 3-5 vs >= 8
DO_NOT_RUN                           routing development; synthetic rerun; new dataset;
                                     SOTA expansion; stronger backbone; joint training
```

The `MUST_HAVE = NONE` verdict is unchanged from `paper_readiness_verified.md`, and nothing
found while drafting the manuscript changed it. The largest residual exposure remains R1,
and it is answered by scope rather than by evidence — which is stated in the limitations
rather than argued away.
