# Novelty wording — LEVEL A / B / C

**Revised 2026-08-12 (second build)** after the NAR26 full-text re-read. The first build
still let LEVEL A/B be read as claiming a matched direct-versus-hurdle comparison; that
comparison exists ([NAR26], component 6a = PRIOR), so the matched condition may now
appear only inside the conjunction (component 6b, CLAIM_ONLY_IN_CONJUNCTION).

```
LEVEL A   strongest defensible.  Use where space allows a full statement.
LEVEL B   conservative.  Use in the abstract and in C1.
LEVEL C   reviewer-safe fallback.  Use if a reviewer challenges A or B.
```

All three are hedged or falsifiable. None uses "first". None claims a combination is new.

---

## LEVEL A — strongest defensible

> We characterize the finite-sample relative behaviour of matched direct conditional-mean
> and occurrence–magnitude factorized forecasts while varying occurrence and magnitude
> dependence separately, and examine how the resulting patterns transfer to real
> intermittent demand.

```
carries       matched comparison (as a CONDITION, not a claim); separate dependence axes;
              relative finite-sample outcome; the transfer boundary
concedes      nothing is called new; "characterize" describes what was done
refutable by  a paper crossing representation choice with separately varied occurrence
              and magnitude dependence.  None located (component 7).
guards        "matched" appears as a property of OUR design, never as something prior
              work lacks -- [NAR26]'s match status is UNKNOWN, not absent
guards        "finite-sample" retained; the factorization named by its product form so
              it cannot be read as [Kou13]'s ratio form
```

---

## LEVEL B — conservative *(abstract, C1)*

> We compare matched direct and occurrence–magnitude factorized forecasting formulations
> across controlled temporal dependence structures and examine the empirical boundary of
> their conditional advantages.

```
shorter, and drops "separate axes" and "finite-sample" -- both recoverable from the body
still safe: "compare" and "examine" are process verbs with no precedence content
```

---

## LEVEL C — reviewer-safe fallback

> We study how controlled temporal dependence changes the relative behaviour of matched
> direct and factorized forecasting formulations and whether these patterns are visible
> in real intermittent demand.

```
claims nothing about the literature at all.  Cannot be refuted by naming a paper.
this is also the answer to "what is the smallest defensible contribution?"
```

---

## The in-text form, for Introduction P2

Not a novelty sentence but a **description of prior work**, which is the safest possible
shape for a gap:

> These answer different questions from the comparison here, in which a direct
> conditional mean and an occurrence-probability × positive-magnitude factorization are
> held to one capacity and training budget while occurrence and magnitude dependence are
> varied separately.

"These" is bounded to the three works just cited, not to all literature, and the sentence
states our design rather than their absence. In place in `introduction_v6.md`.

---

## Rejected, with the paper that refutes each

```
"We are the first to show that the optimal forecasting representation depends on
 temporal dependence structure."
    REFUTED twice.  "first" is unsupportable at this recall; "optimal representation"
    is a function-class claim that P3 explicitly disclaims.

"No prior work has examined how temporal dependence affects the choice between direct
 and factorized forecasting."
    RISKY to REFUTED.  [ALR12] examines dependence effects; [Kou13] examines the choice.
    A reviewer pairs them and calls the conjunction a small step.

"We vary temporal dependence under fixed marginals, which prior work does not."
    WITHDRAWN.  Two reasons, either sufficient.  (i) Fixed marginals are an experimental
    control, not a finding -- claim_literature_audit LIT-C5b, policy
    EXCLUDED_FROM_NOVELTY.  (ii) What [ALR12] holds constant is UNRESOLVED (LIT-C5), so
    "which prior work does not" asserts something we cannot check.

"We provide the first controlled comparison of direct and hurdle forecasting."
    REFUTED.  [NAR26] compares a single-stage LightGBM regressor against a two-stage
    classifier x Tweedie-regressor product form at an identical feature set, under
    identical preprocessing and evaluation protocols, on real data.

"Ours is the first MATCHED comparison of the two forms."
    NOT USED.  [NAR26]'s capacity and training match is NOT STATED -- unknown, not
    absent.  A claim of matched-first would be spending an evidence gap in our favour.
    The matched condition appears only inside the conjunction (component 6b).

"Prior work compares the same two representations."
    REFUTED IN THE OPPOSITE DIRECTION -- this would concede too much.  [Kou13]'s dual
    DIVIDES size by interval; ours MULTIPLIES a probability by a conditional mean.
    See LIT-W-KOU13.

"We are the first to combine the dependence and representation questions."
    NOT USED.  Recall is unquantified (WARN_FAIL G2, G3); a combination claim is as
    unfalsifiable as a precedence claim.
```

---

## Vocabulary rules

```
never    "first", "the first study to", "first to combine"
never    "no prior work", "nobody has", "has never been"
never    "optimal representation", "the right representation"
never    "we introduce decomposition" / "we introduce neural factorization"
never    "fixed marginals are our contribution"
never    describe [Kou13]'s NN-Dual as a hurdle, or as "the same two representations"
never    claim a matched direct-vs-hurdle comparison as standalone novelty (6b)
never    describe [NAR26] as neural, or its feature match as a capacity match
never    present [MC26] as peer-reviewed

always   "we are not aware of" for any absence claim
always   state what prior work DID when narrowing a gap
always   keep "finite-sample" and "matched" attached to the comparison claim
always   name the factorization by its product form (occurrence probability x
         positive magnitude) so it cannot be confused with the Croston ratio form
always   cite [Kou13] wherever a direct-vs-decomposed neural comparison is introduced
always   cite [ALR12] wherever prior dependence manipulation is conceded
always   cite [NAR26] wherever the direct-vs-product-form comparison is introduced
```

---

## Placement

```
abstract          LEVEL B
introduction P2   the in-text form above          (in place, introduction_v6.md)
contributions C1  LEVEL B, trimmed to one clause  (contributions_v3.md unchanged;
                  its wording is already compatible -- see below)
related work 2.5  LEVEL A, expanded per related_work_outline.md
rebuttal          LEVEL C
```

`contributions_v3.md` C1 reads *"Controlled characterization of the finite-sample
relative inductive bias of direct and factorized forecasting under temporal occurrence
and magnitude dependence."* This is LEVEL-B-compatible as written and was **not**
edited: it claims no precedence, keeps "finite-sample", and matches the frozen ledger.
