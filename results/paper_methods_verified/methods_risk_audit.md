# Methods risk audit

Risks that arise specifically in writing Methods, and how the structure handles each.
Scientific reviewer risks live in `../paper_outline_verified/reviewer_risk_map.md`.

---

## Notation risks — all closed before prose

```
MR1  z_t reused for the occurrence indicator
     Section 2.1 already uses z and x for Croston's size and interval.  Reuse would
     make the ratio/product distinction unreadable.
     CLOSED: the indicator is o_t; z appears only in Section 2.1.

MR2  H used for both the horizon and the Hurdle arm
     CLOSED: horizon is H_f; the factorized arm is a superscript on yhat.

MR3  d read as the measured ADI
     d is the design parameter and equals the mean gap by construction; the real-data
     ADI is a measured quantity with different values (median 1.304 / 1.317).
     CLOSED: the registry states the distinction, and Section 4.1 says "by
     construction".

MR4  G confused with an absolute delta
     The source carries three inconsistent delta sign conventions.
     CLOSED: G is the only performance quantity in the main text; the registry records
     the three conventions and the rule.

MR5  rho used in Stage 1
     Stage 1's structured arm is deterministic alternation, rho = -1, not a swept
     parameter.  Writing rho in Stage 1 would imply a graded axis the design cannot
     support.
     CLOSED: rho_I and rho_M appear only from Section 4.5.
```

## Leakage statements that must be made, not assumed

```
MR6  checkpoint selection could see the oracle
     The prereg forbids it explicitly (checkpoint_prohibitions: oracle mean, test
     results, per-model metric, per-cell tuning).  Methods 4.2 states the checkpoint
     metric IS the validation realized-y MSE, and that it is identical for every model.
     Omitting this invites the strongest possible reviewer objection.

MR7  the split could break the latent process
     One continuous trajectory is generated and then split; the gap sequence, magnitude
     parity and latent state are never reset at a boundary.  Methods 4.2 says so.

MR8  the oracle could read the future
     It conditions on the latent state implied by history strictly before the origin.
     Methods 4.2 quotes the constraint.
```

## Result-leakage risk

```
MR9  design and outcome in one subsection
     The frozen outline merges Stage 1 design and Stage 1 results into 4.3, and the
     same for 4.5.  During drafting this produced one paragraph containing both.
     HANDLED: methods_structure.md proposes a design/result split that keeps the
     outline's numbering.  A mechanical check (METH-P1) scans the Methods text for
     outcome vocabulary and for any G value.
```

## Fairness statements that are weaker than they look

```
MR10 "parameter matched" is exact for M1 and approximate for M2
     M1 is 5,856 = 5,856 by construction.  M2 is 5,857, one scalar more, +0.017%,
     below the prereg's 1% rule.  Methods states both numbers rather than rounding
     them together.

MR11 "same trainer" is a reuse claim, not a re-implementation claim
     TRAINING.reuse points at om_factorization_killtest.train.train_one, imported
     unchanged.  That repository's DGP is NOT the paper DGP; its models and trainer
     are.  Methods states the reuse; Related Work never confuses the two.
```

## Empirical-construct risks

```
MR12 H1's external statistic is |rho_interval|, not signed rho_interval
     This is a finding, not a convention: Stage 2 gives |rho_I| 0.1904 against signed
     0.0667.  Methods 5.2 must say the absolute value is used AND why, or a reviewer
     reads it as a post-hoc choice.

MR13 H3's synthetic contrast and its external split are different constructs
     Synthetic is d = 4 versus d = 8; external splits at the ADI median.  Methods 5.2
     states the mismatch at definition time, not only in the Results.
```

---

## METH-A gate

```
METH-A1   notation conflicts                                   0    (MR1, MR2, MR3, MR5)
METH-A2   Point formulation verified                           PASS point_hurdle_fairness
METH-A3   Hurdle formulation verified                          PASS point_hurdle_fairness
METH-A4   fairness verified                                    PASS 5,856 = 5,856, one
                                                                    trainer, one budget
METH-A5   DGP verified                                         PASS dgp_verification
METH-A6   Stage 1 terminology verified                         PASS stage1_stage2_verified
METH-A7   Stage 2 terminology verified                         PASS stage1_stage2_verified
METH-A8   G definition verified                                PASS metric_sign
METH-A9   oracle definition verified                           PASS exact DP
METH-A10  split / index verified                               PASS 384/480/576, 96/24
METH-A11  source map complete                                  PASS 36 rows, 11 subsections
METH-A12  no Methods prose written at this point               PASS
```

Status: **`METHODS_STRUCTURE_NOTATION_FROZEN`**.
