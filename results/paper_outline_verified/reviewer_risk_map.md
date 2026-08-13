# Reviewer risk map

Revised 2026-08-11 after outline review. Supersedes the first draft, which classified
several risks as effectively neutralized. That was too strong. Nothing here is
eliminated; risks are **OPEN**, **MITIGATED**, or **ACKNOWLEDGED_BOUNDARY**, and the
distinction is that a mitigated risk is one the text limits, not one the evidence
removes.

```
status                  meaning
────────────────────────────────────────────────────────────────────────────────
OPEN                    no evidence answers it; only a scope statement stands
                        between the paper and the objection
MITIGATED               the text limits the claim so the objection does not land
                        on what is actually asserted, but the underlying gap remains
ACKNOWLEDGED_BOUNDARY   the objection is itself one of the paper's findings and is
                        reported as such
LOW                     unlikely to be raised, or answered by a single sentence
```

---

## R1 — Single backbone only

```
severity   high
status     OPEN
```

**Evidence.** None against it. Every arm in the synthetic study and in the routing
work is DLinear; the recovered source contains no second backbone.

**Wording restriction.** The contribution sentence itself is scoped —
*finite-sample* relative inductive bias under a *matched parameter budget on one
backbone family* — and the restriction is repeated in 4.2 and 6.5. The paper never
claims a property of factorization in general.

**Additional experiment required?** Not to publish the claim as scoped. A second
backbone is the single most valuable optional experiment and remains the largest
residual exposure.

**Manuscript location.** 3.3, 4.2, 6.5, and the abstract's scope clause.

---

## R2 — Synthetic dependence process is too simple

```
severity   medium
status     MITIGATED
```

**Evidence.** Deliberate. Trend, seasonality, hidden regime, heavy tail, test-time
shift, phase jitter and interval–magnitude cross-correlation are forbidden by the
design, so only the manipulated axis varies.

**Wording restriction.** Section 4.1 states that this is a controlled contrast and not
a simulator. The empirical section is presented as the test of whether anything
survives realism, and its answer is partial.

**Additional experiment required?** No. Adding realism would weaken identification,
which is what the empirical section exists to complement.

**Residual.** A hostile reviewer can still call the design a toy. The defence is
structural, not evidential.

**Manuscript location.** 4.1, 6.3.

---

## R3 — Stage 1 carries a conditional verdict

```
severity   medium
status     MITIGATED, NOT ELIMINATED
```

**Evidence.** `stage1_validity.json`: status `CONDITIONALLY_VALID`,
`fixed_pattern_detected: true` scoped to the structured arm, two claims blocked by the
audit. The control itself is genuine — 40 distinct sequences of 40, zero template
reuse, zero identical train/test prefixes, out-of-sample Markov gain negative at every
order.

**Wording restriction.** The condition is named in 4.3 and honoured: Stage 1 is never
cited for a graded claim, and Stage 2 is introduced as the experiment that resolves
it.

**Additional experiment required?** No.

**Residual.** The paper's first experiment carries a qualified verdict and a reviewer
may weigh it accordingly. Citing the audit is the strongest available response, not a
removal of the issue.

**Manuscript location.** 4.3, 4.5, 6.5, Appendix G.

---

## R4 — Stage 1 and Stage 2 appear to disagree

```
severity   medium
status     MITIGATED, NOT ELIMINATED
```

**Evidence.** Both are as reported. Stage 1's magnitude arm is alternation and its
magnitude-structured cells are factorized-favourable (C03 +18.76, C07 +11.10); Stage 2
shows signed `ρ_M` carrying 3.1× the coefficient of `|ρ_M|`, so persistence rather
than structure per se is what moves the comparison toward direct prediction.

**Wording restriction.** Section 4.6 states the asymmetry and 4.7 uses it to derive
where direct prediction should win. The relationship is presented as a resolution, not
as an inconsistency being explained away.

**Additional experiment required?** No.

**Residual.** The two stages use different parameterizations of "dependence" —
a binary alternation contrast and a signed stationary coefficient. A reviewer may
reasonably ask whether they measure the same construct. The paper's answer is that
Stage 2 subsumes Stage 1's contrast as one point in a larger space, which is an
argument rather than a measurement.

**Manuscript location.** 4.5, 4.6, 6.2.

---

## R5 — Synthetic axes versus real descriptors

```
severity   medium
status     ACKNOWLEDGED_BOUNDARY
```

**Evidence.** Synthetic `ρ` is set at three designed levels; the external quantity is
estimated per series with no controlled support. Targets differ too: exact oracle
conditional mean against realized `y`.

**Wording restriction.** H1 and H2 are classified `EMPIRICAL_ANALOGUE` throughout and
the word "replicates" is not used for either. Only directions are compared; no
numerical comparison is made across the two settings.

**Additional experiment required?** No.

**Effect on the claim.** This is what caps C2's strength. C2 asserts transfer of a
*relationship* and of a *selector*, not of a measured quantity.

**Manuscript location.** 5.3, 5.4, 6.3.

---

## R6 — The H2 selector is not mechanism evidence

```
severity   medium
status     ACKNOWLEDGED_BOUNDARY
```

**Evidence.** The paper's own result: +0.0032 [−0.0033, +0.0094] after overlap
weighting to a worst SMD of 0.0004, and matching failure at SMD 0.614 on log scale.

**Wording restriction.** H2 is written as two claims with different statuses
throughout, and 5.5 is titled as a boundary. Figure 3C exists to make the distinction
visible rather than textual.

**Additional experiment required?** No. The synthetic design has no scale axis and so
cannot arbitrate the mechanism in either direction; that is stated.

**Manuscript location.** 5.4, 5.5, 6.3, Figure 3C.

---

## R7 — H3 did not replicate

```
severity   medium
status     ACKNOWLEDGED_BOUNDARY
```

**Evidence.** M5 −0.0305 [−0.1418, +0.0912] and Favorita −0.0428 [−0.1587, +0.0704],
both signs opposite to the prediction, at the pre-registered median split. The
synthetic interaction is significant in Stage 1 (+3.35 pp) and Stage 2 (+0.0332). The
external test was never run at the ADI 4-vs-8 contrast the prediction came from, and
the support exists (M5 127 series at ADI 3–5, 52 at ≥ 8).

**Wording restriction.** Reported as a non-replication at the tested contrast, never
as a refutation and never as a confirmation.

**Additional experiment required?** No, but re-testing at ADI 3–5 versus ≥ 8 would
convert an ambiguity into a result and is listed as future work.

**Manuscript location.** 5.6, 6.5, Appendix C.

---

## R8 — The learned occurrence head shows no skill on real data

```
severity   medium
status     OPEN / ACKNOWLEDGED
```

**Evidence.** Brier skill −0.0084 on M5, interval including zero; −0.0908 on Favorita,
interval clear of zero, i.e. worse than a constant per-series rate. The diagnostic is
incomplete: ROC-AUC, PR-AUC and log loss were unavailable because Stage A stored
per-series aggregates only.

**Wording restriction.** The mechanism claim is confined to the controlled study and
labelled `COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT`. Section 6.5 states outright that
the real-data sections do not rest on the occurrence mechanism.

**Additional experiment required?** Not for the claims as written. A per-observation
diagnostic would let the paper say something positive or firmly negative about the
mechanism in real data, and is listed as future work.

**Residual.** It remains true that a mechanism-flavoured story is available in the
synthetic study and unavailable in the real one, and a reviewer may read that as a
weakness of the whole framing.

**Manuscript location.** 4.4, 6.5, Appendix B.

---

## R9 — Adaptive routing failed externally

```
severity   low to medium
status     ACKNOWLEDGED_BOUNDARY
```

**Evidence.** −2.43% [−2.74, −2.13] on the pre-declared primary external dataset, and
−193.9% on UCI for the final representation experiment, against an oracle opportunity
of 4.11% and a diversity multiplier of 2.15.

**Wording restriction.** Section 5.7 is half a page to one page, framed as a boundary
on the paper's own implication rather than as a method, and reports that a
pre-registered stop rule was triggered. It is C3, a supporting result. No routing
panel appears in the main figures.

**Additional experiment required?** No, and further routing work is explicitly
`DO_NOT_RUN`.

**Manuscript location.** 5.7, 6.4, Appendix H.

---

## R10 — Absolute performance and classical competitiveness

```
severity   high
status     ACKNOWLEDGED_BOUNDARY
```

**Evidence.** Both representations are outranked by SBA on both datasets (M5 mean rank
SBA 3.152 against direct 4.202 and factorized 4.220).

**Wording restriction.** The paper states that the novelty is not absolute forecasting
accuracy, prints the classical benchmark rather than omitting it, and frames the study
as a comparison of two representations sharing one backbone and one budget. A stronger
backbone changes both arms and does not test the claim.

**Additional experiment required?** No. Publishing Table 4 Block B is what makes the
scope statement credible.

**Residual.** Honest but uncomfortable; a reviewer who wants a competitive forecaster
will not be satisfied, and should not be.

**Manuscript location.** 2, 5.2, 6.5, Table 4 Block B.

---

## Ranked residual exposure

```
rank  risk                                    status
──────────────────────────────────────────────────────────────────────
 1    R1   single backbone                    OPEN
 2    R10  absolute performance               ACKNOWLEDGED_BOUNDARY
 3    R8   occurrence head skill on real data OPEN / ACKNOWLEDGED
 4    R2   synthetic simplicity               MITIGATED
 5    R4   Stage 1 vs Stage 2 parameterization MITIGATED, NOT ELIMINATED
 6    R3   Stage 1 conditional verdict        MITIGATED, NOT ELIMINATED
 7    R7   H3 non-replication                 ACKNOWLEDGED_BOUNDARY
 8    R5   construct mismatch                 ACKNOWLEDGED_BOUNDARY
 9    R6   selector vs mechanism              ACKNOWLEDGED_BOUNDARY
10    R9   routing instability                ACKNOWLEDGED_BOUNDARY
```

None is eliminated. R5, R6, R7 and R9 are arguments the paper makes about itself, which
lowers their cost but does not remove them: a reviewer may still judge that a paper
whose own findings include this many boundaries is a paper with a narrow result. That
is a fair reading, and the correct response is to make the narrow result precise rather
than to broaden it.
