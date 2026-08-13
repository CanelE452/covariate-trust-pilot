# Related Work freeze

Frozen 2026-08-12. Status: **`RELATED_WORK_V2_FROZEN_READY_FOR_METHODS_AUDIT`**.

Current text: `related_work_v2.md`.
Audit: `related_work_v2_claim_audit.md`. Map: `related_work_v2_reference_map.csv`.
Diff: `related_work_v1_v2_diff.md`. Summary: `related_work_v2_section_summary.md`.
v1 and its three companion files are retained unmodified.

---

## What this revision was, and was not

```
was      a style-and-density pass: 2.4 compressed, two concession sentences made less
         declarative, one verb and one construction in 2.5 tightened
was NOT  a re-derivation of the literature boundary, the citation set, the component
         grades, or the novelty intersection.  None of those moved.
```

```
verified literature boundary   unchanged
citation set                   unchanged (11 keys in prose + [MC26] in a footnote)
component evidence / policy    unchanged
novelty intersection           unchanged
Introduction v6                unmodified
frozen scientific files        unmodified
```

---

## Frozen state

```
scientific boundary   UNCHANGED.  literature_evidence_matrix.*, novelty_component_map.csv,
                      precedent_intersection_map.md and novelty_boundary_freeze.md were
                      not touched by this step.
2.4                   COMPRESSED 387 -> 333 words.  All ten precedent-concession elements
                      retained, checked mechanically rather than by reading (RWF5).
Kou13                 RATIO maintained.  "a Croston-style ... combines them as a ratio, in
                      the manner of Croston's method [Kou13]".  Never "hurdle", never
                      "the same two representations".  LIT-W-KOU13 active, permanent.
NAR26                 PRODUCT maintained, and non-neural maintained: "in a
                      gradient-boosting rather than a neural setting", LightGBM named
                      three times.  Capacity and training budget across the two arms are
                      "not reported" -- UNKNOWN, never "not matched".  LIT-W-NAR26 active,
                      permanent.
ALR12                 LIT-W3 remains OPEN.  Zero sentences state what it holds constant.
absence claims        0.  No "first", "no prior work", "unexplored", "novel combination",
                      and no claim about what prior literature can or cannot explain.
                      The word "asymmetry" does not appear in Related Work at all.
novelty ceiling       below LEVEL A.  The strongest sentence is "The present study focuses
                      on a different controlled intersection of these questions."
```

## Word counts

```
        v1     v2
2.1    247    247    byte-identical
2.2    234    234    byte-identical
2.3    276    276    byte-identical
2.4    387    333    compressed
2.5    262    265    two wording edits
TOTAL 1406   1355
```

## Gate

```
RWF1 .. RWF31    PASS 31 / FAIL 0
EC1 .. EC6       0 violations (verify_consistency.py, EC6 added this cycle)
reference map    21 rows, unmapped 0
claim audit      LITERATURE_SUPPORTED 21, BOUNDARY_WORDING 13, POSITIONING 16,
                 OVERCLAIM 0, UNSUPPORTED 0
Intro v6         consistency PASS on all four checked pairs
new search / experiment / training / TEST scoring   0
commit / push / merge                              0
```

## One item recorded rather than smoothed over

Compression detached `[Kou13]` from the sentence carrying the **ratio** distinction, which
the regenerated reference map caught as unmapped. The citation was restored before the map
was accepted. This is written into `related_work_v1_v2_diff.md` rather than silently
repaired, because the reference-map rebuild exists precisely to catch it.

## Open warnings — none blocks the next step

```
LIT-W3        [ALR12] full text unobtained.  OPEN / SUBMISSION_FOLLOWUP_DESIRABLE.
              Blocks only sentences that would state what that study holds constant.
LIT-W-KOU13   ratio, never hurdle.  ACTIVE, permanent.
LIT-W-NAR26   same family is not matched capacity.  ACTIVE, permanent.
LIT-W1 / W2   bibliography cosmetics; re-check at typesetting.
LIT-W6        closed; its generalization is now the derivative-artifact error policy
              (TYPE A / B / C) in WARN_FAIL.md.
```

---

## Next step, not started here

**Methods structure + notation audit.** Before any Methods prose, a single notation
registry must be fixed:

```
Point formulation                 Hurdle formulation
p_t                               mu_t
y_t / positive demand             occurrence indicator
d                                 rho_I
rho_M                             G
oracle target                     train / validation / test indexing
Stage 1 / Stage 2 terminology
```

Only after that registry is frozen does the Methods section hierarchy get designed.
Methods, Results, Discussion and Conclusion prose are **not** to be written until the user
has reviewed `related_work_v2.md` and approved the next step.
