# Introduction structure freeze

Frozen 2026-08-11. Status: **`INTRODUCTION_STRUCTURE_FROZEN_PENDING_CITATIONS`**.

Current text: `introduction_v3.md`, `contributions_v3.md`.
Audit: `introduction_v3_claim_audit.md`. Diff: `introduction_v2_v3_diff.md`.

---

## Paragraph roles

```
P1  128  problem and motivation; neither formulation uniformly superior in aggregate
P2  156  gap: marginal summaries discard order; prior temporal-dependence literature
         exists and is conceded
P3   96  the controlled finite-sample question, with its scope
P4  156  Stage 1 interaction, Stage 2 asymmetric dependence geometry, the scoped
         direct-favourable configuration
P5  114  empirical analogue and frozen-selector predictive transfer
P6  112  overlap-adjusted mechanism boundary and synthetic-to-real entanglement
P7  108  adaptive-use boundary and positioning
```

One logical role per paragraph. The order is frozen.

---

## What is frozen

```
P1-P6 scientific content     frozen at v2, carried into v3.  P2-P6 are byte-identical
                             to v2; P1 lost one clause ("the two are close") that
                             implied an equivalence the artifacts do not support.
P7                           compressed 158 -> 108 words; routing chronology removed;
                             contains no digits
H3                           intentionally omitted from the Introduction; reported in
                             Section 5.6, Table 4 and the limitations
numeric results              three: "eighteen-cell", "-19.8%", "+11.87 percentage
                             points".  Do not add a fourth.
citation placeholders        two, both in P2, unresolved by design
contributions                C1 / C2 / C3 unchanged; contributions_v3 body is
                             byte-identical to contributions_v2
```

---

## What is deliberately still open

```
[CITATION NEEDED] 1   ADI / CV^2 and the intermittent-demand classification scheme
[CITATION NEEDED] 2   prior temporal-dependence work: interval autocorrelation,
                      demand-size autocorrelation, size-interval dependence, and the
                      estimators built on them
```

Both sit in P2, which is why P2 is the one paragraph likely to move again. Its length
(156 words) was not trimmed here: trimming before the citations land would mean
rewriting it twice.

No web search, citation research, Related Work drafting, author or year guessing was
performed in this step.

---

## Claim guards carried forward

```
required     "finite-sample"; "primarily associated with"; "empirical analogue";
             "predictive transfer"; the direct-favourable configuration always scoped
             to the eighteen-cell sweep
never        an aggregate tie or equivalence claim
never        attributing the empirical boundary to scale as a cause
never        "the synthetic mechanism replicates"; a universal winner; a reliable
             router asserted rather than denied
never        C_neg / C_pos (UNIT-W1) or C_sign pooled significance (FLAG-W2)
```

---

## Audit at freeze

```
SUPPORTED 20   BOUNDARY_WORDING 13   CITATION_NEEDED 2   OVERCLAIM 0   UNSUPPORTED 0
IV3-1 .. IV3-32   all PASS
reviewer simulation Q1 .. Q9   all read as intended
new experiment / training / TEST scoring   0
frozen scientific artifacts modified       0
commit / push / merge                      0
```

---

## Next step, not started here

**Citation resolution and Related Work boundary audit**, covering:

1. ADI / CV² and intermittent-demand classification
2. temporal correlation and dependence in intermittent demand
3. the Croston and factorization lineage
4. the novelty boundary between direct and factorized neural forecasting

Until that step runs, the Introduction stays structure-frozen: paragraph roles,
paragraph order, the three numbers and the claim guards do not change, and only P2 is
expected to move.
