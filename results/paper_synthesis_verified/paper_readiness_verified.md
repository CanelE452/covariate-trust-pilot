# Paper readiness — verified against the recovered synthetic source

`results/paper_synthesis/` is preserved unmodified. This directory records only what
the recovery changes.

---

## Verdict

**PAPER_READY_FOR_OUTLINE**

The one blocking item — the controlled synthetic study having no citable artifact —
is resolved. The source was recovered with an unbroken hash chain, its paper DGP was
identified by provenance rather than by filename, and every C1 gate passes. No claim
in the existing synthesis is contradicted by the source.

---

## What changed from `PAPER_READY_AFTER_MINIMAL_EXPERIMENTS`

```
                                    before                    after
────────────────────────────────────────────────────────────────────────────────
C1                          PARTIALLY_VERIFIED          CONFIRMED
Figure 2                    BLOCKED                     SOURCE_READY
Abstract sentence 4         placeholder                 fillable, draft in
                                                        paper_synthesis_discrepancy.md
P3-G1 (all numbers from     PARTIAL FAIL                PASS
  artifacts)
MUST_HAVE                   1 (artifact recovery)       0
```

C2 and C3 are untouched. Their evidence never depended on the synthetic study.

---

## Final hypothesis status

```
H1                          SUPPORTED_WITH_BOUNDARY
  synthetic                 strong; |rho_I| coefficient +0.1904 against +0.0667
                            signed, so the external use of the absolute value is
                            the finding rather than a convention
  external                  +0.1065 (M5) and +0.0789 (Favorita), six of six scale
                            estimates with intervals clear of zero; intermittent
                            regime yes, lumpy regime no; adjusted partial
                            association not separated from zero
  relation                  EMPIRICAL_ANALOGUE

H2 predictive selector      CONFIRMED
  synthetic                 origin is Stage 2, not Stage 1: Stage 1 has no cell where
                            Point wins with an interval clear of zero (C08 is -3.01,
                            CI [-6.79, +0.23]).  Stage 2 has exactly one of eighteen:
                            d=8, rho_I=0, rho_M=+0.8, gain -19.76% [-26.00, -14.53]
  external                  frozen rule, independent M5 population: -0.0230
                            [-0.0294, -0.0163], Point win rate +11.87 pp, three seeds
  relation                  EMPIRICAL_ANALOGUE, one-to-one on all three axes

H2 isolated mechanism       NOT_REPLICATED
                            overlap-weighted association +0.0032 [-0.0033, +0.0094];
                            matching failed at SMD 0.614 on log scale; the synthetic
                            design has no scale axis and cannot speak to it

H3                          NOT_REPLICATED at the pre-registered split,
                            CONSTRUCT_MISMATCH on the contrast
                            synthetic sparsity x interval +3.345 pp and d x rho_I
                            +0.0332, both intervals clear of zero, at ADI 4 vs 8;
                            external split at ADI median 1.304 / 1.317

Occurrence mechanism        COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT (synthetic)
                            (hybrid columns: C03 M1 occurrence head 0.9030 against
                            magnitude head 0.2900)
                            REAL_LEARNED_GATE_NOT_SUPPORTED
                            (Brier skill -0.008 M5, -0.091 Favorita)
                            these are two questions and take two labels

Adaptive routing            opportunity CONFIRMED (convex oracle 4.11% over the best
                            static mixture; diversity multiplier 2.15x)
                            stable learned routing NOT_REPLICATED (-2.43% on the
                            first external dataset; -193.9% on UCI for the sequence gate)
```

---

## Contributions, final

```
C1  Controlled characterization of the FINITE-SAMPLE relative inductive bias of
    direct and factorized forecasting under temporal occurrence and magnitude
    dependence.  CONFIRMED.  Boundary: one backbone family; no scale axis; Stage 1's
    predictable arm is alternation only, so the graded claim rests on Stage 2.
    Wording frozen in claim_ledger_frozen.md section 1.

C2  Real-data validation and the entanglement boundary: the structure transfers as
    a predictive selector, and the association does not survive balancing.
    SUPPORTED. Unchanged by the recovery.

C3  Adaptive-use boundary: a measurable oracle opportunity does not imply a
    learnable routing function. SUPPORTED. Unchanged by the recovery.
```

---

## Remaining experiments

```
MUST_HAVE        NONE
NICE_TO_HAVE     second backbone           (still the largest reviewer exposure,
                                            and the recovery confirms it reaches C1)
                 occurrence-skill diagnostic with per-observation p_hat on real data
                 H3 re-tested at ADI 3-5 vs >= 8, where the support exists
                 (M5 127 vs 52 series; Favorita 84 vs 45)
DO_NOT_RUN       synthetic rerun (the source is intact and hash-verified),
                 any routing work (ROUTING_MODEL_DEVELOPMENT_STOP stands),
                 new dataset, SOTA benchmark, stronger backbone, joint training
```

---

## Next step

Paper detailed outline, then figure and table specification, then drafting. No new
experiment is required to write the paper.


---

## Frozen terminology

Six terms are frozen in `claim_ledger_frozen.md` section 0: stage names, the single
performance quantity `G = 100(1 - RMSE_H / RMSE_P)`, Stage 2 as the generalization
basis, the two-claim split for H2, `COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT` for the
mechanism, and `PRIMARY SOURCE PAYLOAD INTEGRITY PASS` for the transport audit.

Two findings established during the freeze, both sharpening the story:

- Stage 1 contains no Point-favourable cell with an interval clear of zero, so the
  Point-favourable region and therefore H2's synthetic origin belong to Stage 2.
- The two axes are asymmetric: for occurrence the magnitude of dependence governs
  (|rho_I| 2.9x signed rho_I, both signs positive); for magnitude the sign governs
  (signed rho_M 3.1x |rho_M|, and it is persistence that favours the direct model).
  This resolves why Stage 1's magnitude-alternating cells are Hurdle-favourable
  while Stage 2's magnitude-persistent cell is Point-favourable.
