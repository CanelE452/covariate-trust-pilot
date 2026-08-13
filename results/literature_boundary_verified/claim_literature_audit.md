# Claim–literature audit (LIT-C1 … LIT-C8)

**Revised 2026-08-12 (second build).** Re-numbered from seven to eight so that the two
halves of the old LIT-C6 are graded separately, and so that "what ALR12 controls" and
"whether we claim marginal control" are no longer one entry.

Verdicts — **evidence** vocabulary:

```
SUPPORTED             a verified record backs it as written
PRIOR_ART_EXISTS      prior work does this; the paper concedes it and claims nothing
NOT_FOUND_IN_AUDIT    this search located nothing that does it
UNRESOLVED            evidence could not be obtained
NOVELTY_OVERLAP       a located paper does enough of it that the claim must qualify
```

Verdicts — **policy** vocabulary, recorded separately (see
`evidence_policy_separation.md`):

```
EXCLUDED_FROM_NOVELTY_BY_POLICY    we choose not to claim it, regardless of evidence
CLAIM_ONLY_IN_CONJUNCTION          may appear only as one condition of the intersection
```

---

## LIT-C1 — ADI and CV² do not encode temporal ordering

**SUPPORTED.** [SBC05] states the categorization in the average inter-demand interval
and the squared coefficient of variation of demand sizes; [KH06] corrects the boundary.
Order-invariance follows from the definitions and needs no citation.

v6 attributes order-blindness to **the statistics**, not to the scheme, and does not
criticise the scheme. Nothing is claimed as new.

---

## LIT-C2 — temporal-dependence literature exists

**PRIOR_ART_EXISTS.** [ALR12] varies interval autocorrelation, size autocorrelation and
their cross-correlation in generated demand and reports effects on forecast accuracy and
on service level and cost. [WSS04] builds a two-state Markov occurrence model into a
bootstrap.

v6 states this as a specific result rather than a nod to a subject area. It says nothing
about what [ALR12] holds constant — that is LIT-C5 and it is unresolved.

---

## LIT-C3 — decomposition literature exists

**PRIOR_ART_EXISTS.** [Cro72] separates size and interval; [SB05] corrects the bias;
[TSB11] updates the occurrence **probability** every period, which is the closest
classical ancestor of this paper's parameterization.

The Introduction claims nothing here — P1 presents factorization as one of two standard
options chosen "usually by convention".

---

## LIT-C4 — neural direct-vs-decomposed comparison exists

**PRIOR_ART_EXISTS, with a mandatory wording constraint (LIT-W-KOU13).**

[Kou13] compares NN-Rate against NN-Dual on 1000 simulated intermittent series. Full
text verified. The precedent is real and v6 gives it its own clause.

The constraint: NN-Dual predicts non-zero demand size and inter-demand interval and
combines them **as a ratio**, then removes the resulting inversion bias with a fitted
coefficient. This paper's factorized arm is a **product** of an occurrence probability
and a conditional positive mean. v6 therefore describes [Kou13] as *"a Croston-style
representation that predicts non-zero demand size and inter-demand interval and combines
them as a ratio"*.

Getting this wrong in either direction is a defect: overstating the difference hides a
real precedent; understating it concedes the paper.

---

## LIT-C5 — what ALR12 holds constant

**UNRESOLVED.** Six access routes failed
(`alr12_fulltext_verification.md`). Whether ADI, CV², zero proportion, mean demand, the
size distribution or the interval distribution are held constant across correlation
levels is not established. The matrix records `U` and **does not promote it to `Y`**.

`LIT-W3` stays OPEN. Manuscript dependency on this unresolved detail: **0** — no sentence
in `introduction_v6.md` says what [ALR12] holds constant, and `related_work_outline.md`
2.3 prohibits one while the warning is open.

---

## LIT-C5b — fixed-marginal control as *our* novelty

**EXCLUDED_FROM_NOVELTY_BY_POLICY.** This is a policy entry, not an evidence entry, and
it is deliberately separated from LIT-C5.

Holding the marginals fixed is an **experimental control** in this paper's design. The
project's frozen sources already treat it that way:

```
claim_ledger_frozen.md reader chain 1-2   the control is the PREMISE of step 2
final_outline_freeze.md wording decisions "related work: no novelty claim over the
                                          correlation literature"
```

The exclusion therefore does **not** depend on how LIT-C5 resolves. Both directions are
closed: we do not assert that [ALR12] fixed the marginals, and we do not claim marginal
control as ours.

---

## LIT-C6a — direct / single-stage vs hurdle / two-stage precedent

**PRIOR_ART_EXISTS.** Verified at full text (`nar26_matched_comparison_verification.md`).

[NAR26] compares *"a LightGBM regressor … trained directly on the full feature set"*
against a two-stage model in which *"a LightGBM classifier estimates the probability of
observing non-zero demand"* and *"a LightGBM regressor with a Tweedie objective predicts
the expected demand quantity"* — the **product** form — under *"identical data
preprocessing, feature construction, and evaluation protocols"*.

This is the same pair of forms this paper compares. It is conceded without qualification,
and **it is not weakened by [NAR26] being non-neural**: the comparison of the two forms
has been done. v6 acknowledges it in its own clause, which the earlier drafts did not.

---

## LIT-C6b — the same comparison under matched capacity and matched training

**NOT_FOUND_IN_AUDIT**, policy **CLAIM_ONLY_IN_CONJUNCTION**.

```
[NAR26]  NAR-E capacity / tree counts / depths matched     NOT STATED  -> U
         NAR-F training budget, optimizer, early stopping  NOT STATED  -> U
         NAR-G per-formulation tuning                      NOT STATED  -> U
         what IS stated is a match on FEATURES and on the DATA and EVALUATION pipeline,
         which is not a match on capacity or training (LIT-W-NAR26)
[Kou13]  parameters not matched -- two output nodes against one, and each arm reported
         at its own best (I,H) by in-sample MAE rank
         training partially matched -- shared trainer, unmatched capacity
```

**The policy matters here.** Because the nearest neighbour's match status is `UNKNOWN`
rather than known-absent, this component may never carry a claim standalone. It appears
only as one condition of the intersection. Symmetrically to LIT-C5, the gap is not
spendable in our favour.

---

## LIT-C7 — representation choice crossed with separate temporal-dependence axes

**NOT_FOUND_IN_AUDIT.** `representation_x_dependence_interaction` is `N` for every prior
row in the matrix. [ALR12] has the dependence axes and no representation contrast;
[Kou13] and [NAR26] have a representation contrast and no dependence factor
(NAR-H verified negative: no manipulation, no breakdown).

This is the load-bearing component. v6 states it as a difference in question — *"These
answer different questions from the comparison here, in which … occurrence and magnitude
dependence are varied separately"* — bounded to the three cited works, not to all
literature.

---

## LIT-C8 — synthetic-to-real conditional transfer boundary

**NOT_FOUND_IN_AUDIT.** No located paper follows a controlled synthetic condition into
real demand through an empirical analogue, a pre-frozen selector on an independent
population, and an overlap-adjusted mechanism test that fails. [TJWC21] is nearest and
its synthetic experiments are illustrative rather than transferred.

Not asserted in the Introduction as a novelty claim at all — P5 and P6 simply report the
two results. No change needed.

---

## Totals

```
evidence
  SUPPORTED             1     LIT-C1
  PRIOR_ART_EXISTS      4     LIT-C2, LIT-C3, LIT-C4, LIT-C6a
  UNRESOLVED            1     LIT-C5
  NOT_FOUND_IN_AUDIT    3     LIT-C6b, LIT-C7, LIT-C8

policy
  EXCLUDED_FROM_NOVELTY_BY_POLICY   1   LIT-C5b
  CLAIM_ONLY_IN_CONJUNCTION         1   LIT-C6b
```

Five of nine evidence entries are concessions or unresolved. That ratio is the audit
working.

---

## Changes from the previous build

```
LIT-C5   was "PRIOR_ART_EXISTS -- graded conservatively".  That conflated an unresolved
         fact with a policy decision and contradicted the matrix, which said U.
         Now UNRESOLVED, with the policy split out as LIT-C5b.
LIT-C6   was one entry ("NOT_FOUND_IN_AUDIT with a NOVELTY_OVERLAP qualification").
         Split into 6a (PRIOR, verified at full text) and 6b (NOT_FOUND_IN_AUDIT,
         CLAIM_ONLY_IN_CONJUNCTION).
LIT-C7   was LIT-C6's second half; now standalone and identified as load-bearing.
LIT-C8   renumbered from LIT-C7.
LIT-C6a  new; it is the reason introduction_v6 exists.
```
