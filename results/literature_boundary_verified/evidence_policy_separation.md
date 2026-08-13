# Evidence status vs novelty policy — the separation

The audit was conflating two different things under one label. This file defines the
separation and applies it. It is the root cause of both defects corrected on 2026-08-12.

---

## The two fields

```
evidence_status     WHAT THE LITERATURE SHOWS.  A statement about verified fact.
                    PRIOR | PARTIAL_OVERLAP | NOT_FOUND_IN_AUDIT | UNKNOWN

novelty_policy      WHAT WE DECIDE TO CLAIM.  A statement about our own conduct.
                    CLAIMED_IN_CONJUNCTION | CLAIM_ONLY_IN_CONJUNCTION |
                    EXCLUDED_FROM_NOVELTY | NOT_CLAIMED
```

They are independent. A component can be `UNKNOWN` in evidence and still
`EXCLUDED_FROM_NOVELTY` in policy — that is precisely the honest handling of a gap we
cannot close.

**The rule that was violated:** a conservative *policy* must never be recorded as a
*finding*. Choosing not to claim something is not evidence that someone else did it.

---

## Defect 1 — ALR12 marginal control

```
what the matrix said        marginal_characteristics_controlled = U   (correct)
what the component map said component 3 = PRIOR, "graded from UNKNOWN ... because
                            grading an unverified gap in our own favour is exactly the
                            error this audit exists to prevent"
the problem                 the reasoning was right, the encoding was wrong.  Recording
                            UNKNOWN as PRIOR does not make the audit conservative; it
                            manufactures a precedent that has not been verified, and it
                            makes two files disagree about the same fact (EC1).
```

**Corrected to:**

```
component 3   controlled marginal / intermittency characteristics + dependence variation
              evidence_status   UNKNOWN
              reason            [ALR12] full text unobtainable; which marginal
                                characteristics are held constant across correlation
                                levels is not established
              novelty_policy    EXCLUDED_FROM_NOVELTY
              reason            holding the marginals fixed is an EXPERIMENTAL CONTROL
                                in our design, not a finding -- see
                                claim_ledger_frozen.md reader chain 1-2 and
                                final_outline_freeze.md "no novelty claim over the
                                correlation literature".  This holds regardless of what
                                [ALR12] turns out to do.
```

Neither direction is now spendable. We do not write "ALR12 held the marginals fixed",
and we do not write "we are the first to control marginals".

---

## Defect 2 — NAR26 matched comparison

```
what the matrix said        matched_training_protocol = Y, matched_parameter_budget = P,
                            direct_neural_prediction = Y
what the article states     NAR-E, NAR-F, NAR-G: NOT STATED.  And the model is LightGBM,
                            not neural -- the row's own note already said so.
the problem                 "identical data preprocessing, feature construction, and
                            evaluation protocols" is a DATA-pipeline statement.  It was
                            read as a TRAINING match (EC3), and a non-neural learner was
                            flagged as neural (EC2).
```

**Corrected to:** `matched_parameter_budget = U`, `matched_training_protocol = U`,
`direct_neural_prediction` renamed `direct_prediction_arm` = Y with a new
`neural_model = N`, and `matched_feature_set = Y` added (that one *is* stated).

Component 6 splits into 6a (PRIOR, verified) and 6b (NOT_FOUND_IN_AUDIT, nearest
neighbour UNKNOWN → `CLAIM_ONLY_IN_CONJUNCTION`).

---

## Why the two defects are symmetric

They are the same error pointing in opposite directions.

```
ALR12   an unverified gap was recorded as PRIOR       -- over-conceding, in the belief
                                                         that over-conceding is safe
NAR26   an unstated match was recorded as MATCHED     -- over-conceding again
```

Both felt conservative. Neither was: both replaced "we do not know" with a definite
label, and a definite label can be checked and found wrong. Over-conceding is not free —
it corrupts the record just as under-conceding does, and it makes the audit unusable as
a source of truth for the manuscript.

**The correct conservative move is to record the uncertainty and constrain the claim
separately.** That is what the two fields are for.

---

## Consistency invariants — must hold across every file

```
EC1  a given paper's status for a given component is IDENTICAL in every file
EC2  a given model is described by ONE formulation everywhere
       [Kou13] NN-Dual        = size / interval RATIO, always
       [NAR26] two-stage      = probability x magnitude PRODUCT, non-neural, always
       this paper's Hurdle    = probability x magnitude PRODUCT, neural, always
EC3  "same model family" / "same feature set" NEVER implies matched capacity or matched
     training.  Matching is recorded only where the source STATES it.
EC4  NOT_FOUND_IN_AUDIT is never rendered as "no prior work exists" in any text
EC5  evidence_status and novelty_policy never substitute for one another
```

Enforced mechanically: `verify_consistency.py` reads the matrix and the component map
and fails on any disagreement. Result at freeze: **EC1–EC5 all 0 violations.**

---

## What this changes in the manuscript

Nothing in P1 or P3–P7. In P2, one addition and one rewording, both driven by evidence
rather than by story: the hurdle / two-stage precedent ([NAR26], component 6a = PRIOR)
must be acknowledged, and [Kou13] must be described in its own terms. See
`introduction_v6.md`.

No novelty claim moved in either direction. Component 3 was already excluded from the
novelty claim before this correction; the correction only stops the audit from asserting
a precedent it never verified.
