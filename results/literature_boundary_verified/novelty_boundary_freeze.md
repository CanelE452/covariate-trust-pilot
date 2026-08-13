# Novelty boundary freeze

Frozen 2026-08-12 (second build). Status: **`LITERATURE_BOUNDARY_VERIFIED_INTRODUCTION_READY`**.

Supersedes the first build. Two evidence-label defects were corrected; neither changed the
novelty story, both changed what the audit is allowed to assert.

```
defect 1   component 3 graded PRIOR while the matrix recorded UNKNOWN.
           -> evidence_status UNKNOWN, novelty_policy EXCLUDED_FROM_NOVELTY.
defect 2   [NAR26] recorded as "matched features and model family", read as a partial
           capacity match, graded N2.
           -> component 6 split into 6a (PRIOR, verified) and 6b (NOT_FOUND_IN_AUDIT,
              CLAIM_ONLY_IN_CONJUNCTION); [NAR26] regraded N3; match fields U, not P/Y.
```

Root cause and the fix: `evidence_policy_separation.md`.

---

## What is frozen

```
two-field schema                                   evidence_status + novelty_policy
novelty is an INTERSECTION, not a component        precedent_intersection_map.md I1..I4
fixed marginals                                    EXPERIMENTAL CONTROL, excluded by policy
matched budget                                     a design property, CLAIM_ONLY_IN_CONJUNCTION
[Kou13] NN-Dual                                    size / interval RATIO, never a hurdle
[NAR26]                                            non-neural PRODUCT-form precedent, PRIOR
LEVEL A / B / C wordings                           novelty_wording_options.md
Introduction P2                                    introduction_v6.md, 167 words
"first" / "first to combine" / "no prior work"     never used
```

---

## What is already prior — concede every time

```
1   ADI / CV^2 classification                        PRIOR    [SBC05] [KH06]
2   temporal dependence manipulated                  PRIOR    [ALR12]
4   occurrence / magnitude decomposition             PRIOR    [Cro72] [SB05] [TSB11]
4b  the occurrence-PROBABILITY parameterization      PRIOR    [TSB11]
5   neural direct vs decomposed (ratio form)         PRIOR    [Kou13]
6a  direct vs the PRODUCT form, identical features   PRIOR    [NAR26]
```

## What we do NOT claim

```
3   controlled marginals + dependence variation      UNKNOWN evidence,
                                                     EXCLUDED_FROM_NOVELTY by policy
6b  the same comparison at matched capacity and      NOT_FOUND_IN_AUDIT,
    matched training                                 CLAIM_ONLY_IN_CONJUNCTION
                                                     ([NAR26] match status UNKNOWN,
                                                      not absent)
```

## What remains

```
7   occurrence and magnitude dependence on SEPARATE axes,      NOT_FOUND_IN_AUDIT
    CROSSED with representation choice
8   finite-sample relative inductive bias at a fixed budget    NOT_FOUND_IN_AUDIT
9   controlled synthetic condition followed to a real-data     NOT_FOUND_IN_AUDIT
    transfer boundary
```

`NOT_FOUND_IN_AUDIT` is about this search, never about the world.

---

## Final novelty intersection, one paragraph

Three questions have already been asked, by three literatures, for three purposes.
[ALR12] varies interval autocorrelation, size autocorrelation and their cross-correlation
in generated demand and asks what happens to forecast accuracy and inventory performance —
of Croston-family estimators inside one already-factorized form. [Kou13] asks whether a
neural forecaster should output a demand rate directly or output size and interval and
combine them as a ratio — on one generated population, with each arm reported at its own
best configuration. [NAR26] asks whether a single-stage model or a two-stage model that
multiplies an occurrence probability by a conditional size does better on real spare-parts
data at an identical feature set. This paper asks the representation question **as a
function of** the dependence question: with the backbone, the parameter count
(5,856 = 5,856), the trainer and the budget held fixed, and with occurrence and magnitude
dependence varied on separate axes, how does the relative finite-sample behaviour of a
direct conditional mean and an occurrence-probability × positive-magnitude factorization
move — and how far does the answer survive in real demand? What the crossing returns is
not what any of the three strands predicts: the two axes respond to different features of
dependence, occurrence to its strength and magnitude to its direction.

**No claim is made that combining the questions is itself new.** Recall is unquantified.

---

## LEVEL A / B / C

**LEVEL A** — *We characterize the finite-sample relative behaviour of matched direct
conditional-mean and occurrence–magnitude factorized forecasts while varying occurrence
and magnitude dependence separately, and examine how the resulting patterns transfer to
real intermittent demand.*

**LEVEL B** *(abstract, C1)* — *We compare matched direct and occurrence–magnitude
factorized forecasting formulations across controlled temporal dependence structures and
examine the empirical boundary of their conditional advantages.*

**LEVEL C** *(smallest defensible)* — *We study how controlled temporal dependence changes
the relative behaviour of matched direct and factorized forecasting formulations and
whether these patterns are visible in real intermittent demand.*

In all three, "matched" describes **our design**. It is never a statement about what prior
work lacks.

---

## Reviewer-facing answers

**Q1 — "Altay et al. already studies temporal dependence."**
Yes, and we cite them for exactly that. [ALR12] varies size autocorrelation, interval
autocorrelation and size–interval cross-correlation in generated demand and reports the
consequences for forecast accuracy and for service level and cost. What it compares are
Croston-family **estimators inside one already-factorized representation** — there is no
direct conditional-mean arm, so no representation contrast. Our question is what
dependence does to the **choice between** representations.

**Q2 — "Did Altay et al. also fix the marginals?"**
Unresolved. Six access routes to the full text failed, so which marginal characteristics
are held constant across correlation levels is not established. We do not assert that they
did, and we do not use the gap the other way: marginal control is excluded from our
novelty by policy, because it is an experimental control in our design rather than a
finding.

**Q3 — "Kourentzes already compares direct and neural decomposition."**
Yes. NN-Rate outputs the demand rate from a single linear node; NN-Dual outputs size and
interval and combines them **as a ratio**, carrying an inversion bias removed afterwards
by a fitted coefficient. That is not this paper's occurrence-probability × magnitude
product. Nor is it a matched budget — each arm is reported at its own best (I, H), and the
output widths differ by construction — and there is no dependence factor at all.

**Q4 — "Other work already compares hurdle / two-stage against direct."**
Yes. [NAR26] compares a LightGBM regressor trained directly on the full feature set
against a LightGBM classifier for P(non-zero) times a Tweedie regressor for the
conditional size, under identical preprocessing, feature construction and evaluation
protocols. That is the same pair of forms, and we concede it without qualification. It is
non-neural, has no dependence factor and no dependence breakdown, and its capacity and
training match are not stated — so we do not place our contribution on the comparison
itself.

**Q5 — "Then what exactly remains?"**
The crossing: matched representation comparison × separately varied occurrence and
magnitude dependence × relative finite-sample behaviour × the empirical transfer boundary.
No located row in the evidence matrix carries
`representation_x_dependence_interaction`.

**Q6 — "Is fixed-marginal control novel?"**
We do not claim that.

**Q7 — "Are you claiming first?"**
No. "First" appears nowhere, and neither does "first to combine". Recall is unquantified —
no Scopus or Web of Science sweep, and 2025–2026 preprints are thinly indexed.

**Q8 — "What is your smallest defensible contribution?"**
LEVEL C, above. It claims nothing about the literature and cannot be refuted by naming a
paper.

---

## Evidence consistency — EC1 … EC5

Checked mechanically by `verify_consistency.py` against
`literature_evidence_matrix.csv` and `novelty_component_map.csv`.

```
EC1  a paper's status for a component is identical in every file          0 violations
EC2  one formulation per model everywhere (ratio vs product; neural       0 violations
     vs non-neural)
EC3  same family / same features never recorded as matched capacity       0 violations
EC4  NOT_FOUND_IN_AUDIT never rendered as non-existence                   0 violations
EC5  evidence_status and novelty_policy never substitute for one another  0 violations
```

---

## NBF gate — NBF1 … NBF32

```
NBF1  ALR12 full-text status OPEN                               PASS  SUBMISSION_FOLLOWUP
NBF2  ALR12 marginal-control evidence UNKNOWN                   PASS  U in both files
NBF3  ALR12 fixed-marginal novelty policy EXCLUDED              PASS
NBF4  evidence status and novelty policy separated              PASS  two fields
NBF5  manuscript free of unresolved ALR12 control detail        PASS  0 dependencies
NBF6  Kou13 ratio formulation correct                           PASS  z'/x', de-biased
NBF7  Kou13 not labelled as exact Hurdle                        PASS
NBF8  Kou13 neural precedent acknowledged                       PASS  P2 S8
NBF9  component 6a / 6b split                                   PASS
NBF10 NAR26 exact formulation verified                          PASS  NAR-A .. NAR-H
NBF11 same-family != matched-capacity guard                     PASS  LIT-W-NAR26
NBF12 component 7 evidence status                               PASS  NOT_FOUND_IN_AUDIT
NBF13 component 8 evidence status                               PASS  NOT_FOUND_IN_AUDIT
NBF14 component 9 evidence status                               PASS  NOT_FOUND_IN_AUDIT
NBF15 NOT_FOUND_IN_AUDIT wording preserved                      PASS
NBF16 fixed-marginal novelty claim                              PASS  0
NBF17 decomposition novelty claim                               PASS  0
NBF18 direct-vs-neural-decomposition novelty claim              PASS  0
NBF19 hurdle-first novelty claim                                PASS  0
NBF20 final intersection defined                                PASS  I1..I4
NBF21 LEVEL A / B / C re-verified                               PASS
NBF22 Introduction P2 evidence-consistent                       PASS  5/5 citations
NBF23 P1 / P3-P7 substantive story unchanged                    PASS  byte-identical
NBF24 collision grades reassessed                               PASS  ALR12 / Kou13 /
                                                                      NAR26 all N3
NBF25 N4 = 0, else stop                                         PASS  0
NBF26 EC1-EC5 = 0                                               PASS
NBF27 OVERCLAIM = 0                                             PASS
NBF28 UNSUPPORTED = 0                                           PASS
NBF29 no broad new literature search                            PASS  targeted re-reads
NBF30 new experiment / training / TEST scoring = 0              PASS
NBF31 frozen scientific artifacts unchanged                     PASS  mtime 2026-08-11
NBF32 commit / push / merge = 0                                 PASS
```

```
PASS 32    FAIL 0    OPEN WARNINGS 3
```

---

## Open warnings — none is a blocker

```
LIT-W3   [ALR12] full text unobtained.  OPEN / SUBMISSION_FOLLOWUP_DESIRABLE.
         Blocks exactly one thing: any sentence stating what [ALR12] holds constant.
         Does not block Related Work drafting.
LIT-W1   Croston 1972 container name (ORQ vs post-1978 JORS).  Cosmetic.
LIT-W2   [GDTP25] volume / pages unassigned (online first).  Re-check before submission.
```

Two permanent guards remain active: `LIT-W-KOU13` (ratio, never hurdle) and
`LIT-W-NAR26` (same family is not matched capacity).

---

## Next step, not started here

**Related Work drafting from the verified literature boundary**, following
`related_work_outline.md` sections 2.1–2.5.

Methods, Results and Discussion are **not** to be written until `introduction_v6.md` and
this novelty boundary have been reviewed by the user.
