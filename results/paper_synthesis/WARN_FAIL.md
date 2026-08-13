# WARN / FAIL

No experiment was run, no model trained, no TEST scored.

---

## FAIL

**F1. `SYNTHETIC_STUDY_ARTIFACTS_ABSENT` — critical for C1.**
The controlled numerical study is not in this repository. Its source
(`~/Documents/github/m5dataset`) has no git remote, was last committed 2026-06-25, and sits on a
different machine from the one this work now runs on. Only the external-validity closure was
migrated. Consequence: no number from Stage 1, Stage 3 or Stage 4 is asserted anywhere in this
synthesis, Fig 2 cannot be produced, and abstract sentence 4 is a placeholder. This is the
single reason the readiness verdict is not `PAPER_READY_WITH_CURRENT_EVIDENCE`.

---

## WARN

**W1. `STAGE_A_SAMPLE_IS_REGIME_BALANCED`.** The screen uses 300 series per SBC class while
M5's full pool is 23,053 intermittent, 5,942 lumpy, 984 smooth and 496 erratic. The pooled
H1 estimates are therefore not population estimates and must not be quoted without the caveat.

**W2. `H1_ADJUSTED_ASSOCIATION_NOT_SEPARATED_FROM_ZERO`.** Standardized partial coefficient
+0.032 (M5) and +0.017 (Favorita), intervals containing zero. The artifact itself tags the
analysis `POSTHOC_DIAGNOSTIC / exploratory; not an H1 primary test`.

**W3. `H1_ABSENT_IN_LUMPY`.** +0.014 / +0.032 / +0.028 with every interval spanning zero, in a
regime that is one of the two the synthetic study targeted.

**W4. `OCCURRENCE_MECHANISM_REJECTED_ON_REAL_DATA`.** Brier skill against a per-series constant
rate is −0.008 on M5 and −0.091 on Favorita, the latter with an interval excluding zero. The
synthetic mechanism story cannot be carried into the real-data sections.

**W5. `OCCURRENCE_DIAGNOSTIC_INCOMPLETE`.** ROC-AUC, PR-AUC and the Hurdle log loss are
unavailable because Stage A stored per-series aggregates only, so W4 rests on Brier score alone.

**W6. `H3_NOT_TESTED_AT_ITS_DERIVED_CONTRAST`.** The pre-registered split used the ADI median
(M5 1.304); the synthetic contrast was ADI 4 versus 8. Series at the intended contrast exist
(M5: 127 at ADI 3–5, 52 at ADI ≥ 8) but were never used as a primary test. H3 is reported as a
non-replication at the pre-registered split, not as a refutation.

**W7. `FAVORITA_H2_ANALYSES_DISAGREE`.** `FAVORITA_RULE_LOW_SUPPORT` (n=18, interval spans
zero, win-rate difference −14.5 pp) against `FAVORITA_RULE_PARTIAL_TRANSFER` (n=792, interval
excludes zero, +3.9 pp). Both must appear; neither may be reported alone.

**W8. `FAVORITA_TRANSFER_DESIGN_DEVIATION`.** The specified full-pool replication was not
executed because the repository contains only the Stage A Favorita sample. The artifact records
`independence_from_stage_a_sample: false` and
`not_the_specified_external_replication: true`.

**W9. `BACKBONE_IS_OUTRANKED_BY_CLASSICAL_METHODS`.** SBA leads both datasets by mean rank and
both neural variants trail four classical methods on M5. Must be published as Table 3.

**W10. `SINGLE_BACKBONE_FAMILY`.** Every expert in every experiment, including the whole routing
chain, comes from one DLinear family plus `naive`. Whether the routing instability is a property
of routing or of this expert family is untested.

**W11. `UCI_AVAILABILITY_UNKNOWN`.** Recorded in `dataset_audit.json`. UCI was admitted with
role `EXTERNAL_ROBUSTNESS`, not as a confirmation dataset, and its −193.9% should be used as a
failure-mode counterexample rather than as a benchmark ranking.

**W12. `OPERATIONAL_LABELS_ARE_NOT_EVIDENCE`.** `GATE_V3_OOF_STRONG`, `P0L1_TEMPORAL_STRONG`
and `DIVERSE_GATE_GREEN` are all in-development or cross-fitted-OOF labels. None of them
qualifies the external failure, and the paper must never present them as if they did.

**W13. `IETS_NOT_AVAILABLE`.** No verified implementation in the environment and it was
deliberately not written from scratch, so the classical benchmark omits iETS. Must be footnoted.

**W14. `SEQUENCE_GATE_SINGLE_SEED`.** Frozen before training as one canonical seed plus a
reproducibility rerun, matching the handcrafted gate's single-seed structure. The rerun
reproduced to `max|dg| = 0`, but seed variability is not characterized.

**W15. `REPOSITORY_NAME_IS_UNRELATED`.** `covariate-trust-pilot`, its README,
`results/report.md` and `results/figures/*` belong to a separate Chronos-2 covariate pilot.
None of that material is part of this paper and it should not be confused with it in any
data-availability statement.
