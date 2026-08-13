# WARN / FAIL — synthetic source verification

No experiment was run, no model trained, no TEST scored, nothing committed or pushed.

## FAIL

None. `SYNTHETIC_STUDY_ARTIFACTS_ABSENT`, the single critical failure carried since
the first synthesis, is **resolved**: the source is recovered, hash-verified end to
end, and its HEAD matches the commit recorded in the Stage A results.

## WARN

**W1 `STAGE_NAMING_MISMATCH`.** The source has Stage 1 and Stage 2. `paper_synthesis`
says Stage 1 / Stage 3 / Stage 4. Rename in the verified derivative; nothing
scientific changes.

**W2 `STAGE1_PREDICTABLE_ARM_IS_ALTERNATION_ONLY`.** Stage 1's predictable arm is
deterministic period-2 alternation (rho = -1). The source's own audit blocks the
claims "temporal predictability in general drives the hurdle advantage" and "the
Stage 1 interval effect measures a graded predictability axis". Only Stage 2
licenses the graded reading, and the source states plainly that Stage 2 "overturns
the reading Stage 1 invited". Stage 1 must never be cited alone for the graded claim.

**W3 `STAGE1_STATUS_IS_CONDITIONALLY_VALID`.** Not "VALID". The condition is W2.

**W4 `THREE_DELTA_CONVENTIONS`.** Stage 1's absolute delta is `hurdle - point`;
Stage 2's and the real-data one are `point - hurdle`. `gain` is consistent
everywhere. Any table mixing studies must use `gain`, or print the formula inline.

**W5 `SINGLE_BACKBONE_FAMILY`.** Both arms are DLinear, in the synthetic study as
well as in the routing chain. The recovery does not remove this limitation; it
confirms it extends to the paper's most upstream contribution.

**W6 `NO_SCALE_AXIS_IN_THE_SYNTHETIC_DESIGN`.** The controlled study manipulates
sparsity, occurrence dependence and magnitude dependence. It has no scale axis, so
it cannot address the confound that made the external H2 association vanish under
overlap weighting. C1 and the H2 mechanism question are therefore not connected by
evidence in either direction.

**W7 `H3_TESTED_AT_THE_WRONG_CONTRAST`.** Synthetic contrast is ADI 4 vs 8; the
external split is the ADI median at 1.304 (M5) and 1.317 (Favorita). The synthetic
interaction is significant in both Stage 1 (+3.35 pp) and Stage 2 (d x rho_I
+0.0332); the external null is not a refutation of it.

**W8 `DEAD_CODE_NEAR_MISS`.** The source audit found that `_event_stream` would have
let magnitude read `gap_uniform`, creating the interval-magnitude cross-dependence
the design forbids. It has no call site, so executed runs are unaffected. Recorded
because the "no cross-correlation by construction" claim rests on it.

**W9 `OCCURRENCE_MECHANISM_NEEDS_TWO_LABELS`.** `SYNTHETIC_DIAGNOSTIC_SUPPORTED`
inside the controlled study; `REAL_LEARNED_GATE_NOT_SUPPORTED` on M5 and Favorita.
The existing single label "REJECTED" merges two different questions.

**W10 `KILLTEST_PHRASING_CORRECTED`.** My earlier Windows-side statement that
`om_factorization_killtest` "is not the paper's" was too strong. Its DGP is not the
paper DGP, exactly as its own prereg says, but the paper study imports its models
and trainer unchanged.

**W11 `RECOVERY_AREA_IS_LARGE_AND_UNBACKED`.** `E:\research_recovery` holds about
27 GB (download + reconstructed tar + extracted tree). It is a working copy, not a
backup. The durable copy is the GitHub private release
`CanelE452/m5dataset-recovery @ recovery-20260811_135203`.

**W12 `SOURCE_REPO_STILL_HAS_NO_REMOTE`.** The Ubuntu original remains a local-only
repository with 6,962 untracked files. The release is now its only off-machine copy.

**W13 `NO_TEST_SCORED`.** No existing TEST was scored and no new dataset was used.
