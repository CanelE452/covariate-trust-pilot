# Visual and table QA

Each item was checked against the rendered draft, not against the specification.

## Figure QA

```
V1   text readable when reduced          PASS   7 pt base at 7.16 in; cell numbers in
                                                Fig 2a stay legible at 50% scale
V2   grayscale intelligible              PASS   every cell prints its signed value, so
                                                sign and magnitude survive without hue;
                                                Fig 3 uses position and fill, not colour
V3   zero is the heatmap midpoint        PASS   diverging scale centred at 0, limits
                                                [-23, +23] from |max| = 22.81
V4   d=4 and d=8 share one scale         PASS   one imshow norm, one shared colourbar
V5   negative cell identifiable, not
     exaggerated                         PASS   the only blue cell; no added highlight
V6   cell text contrast sufficient       PASS   text switches to white above 0.55*limit
V7   significance encoding not cluttered PASS   with Option B, one open circle total
                                                (Option A rendered for comparison: 17)
V8   Fig 2b/2c share a y-axis            PASS   common ylim, right panel's tick labels
                                                suppressed
V9   uncertainty semantics correct       PASS   spread is labelled spread; no marginal
                                                CI is implied, because none exists
V10  Figure 1 leaks no result            PASS   schematic labelled as such; both axes
                                                symmetric; only 5,856 is artifact-derived
V11  Fig 3 separates selector from
     mechanism                           PASS   (b) transfer, (c) the same estimate
                                                before and after adjustment crossing zero
V12  axes readable without the legend    PASS   colourbar carries "factorized better" /
                                                "direct better"; 3b/3c say "negative =
                                                shifted toward direct"
V13  captions do not overstate           PASS   "within this grid", "empirical analogue";
                                                no causal or mechanism language
V14  notation consistent across panels   PASS   rho_I, rho_M, G, direct, factorized
V15  no colour-only encoding             PASS   colour is redundant with printed values
                                                everywhere it appears
```

Two cosmetic items remain, neither affecting interpretation: in Figure 2b the spread
note sits close to the lowest open circle, and Figure 1a's schematic would read
slightly better with the two rows further apart.

## Table QA

```
TQ1  decimal precision consistent        PASS   Table 2 at 2 dp; Table 4 at 4 dp for
                                                estimates and intervals alike
TQ2  G convention consistent             PASS   Table 2 is in G percentage points and
                                                says so in the header
TQ3  CI notation consistent              PASS   [low, high] with explicit signs
TQ4  main and appendix roles clear       PASS   Table 3 lists M5 and Favorita and states
                                                that the two stress-test datasets are in
                                                the appendix
TQ5  fairness auditable at a glance      PASS   Table 1 is item / direct / factorized /
                                                matched, with 5,856 = 5,856
TQ6  no legacy delta in Table 2          PASS   only effect_pp, which is G in pp
TQ7  no "replication" wording for H1     PASS   "empirical analogue, not replication"
TQ8  H2 split into two rows              PASS   selector CONFIRMED, mechanism
                                                NOT_REPLICATED, as separate rows
TQ9  H3 negative result visible          PASS   both datasets, both signs opposite,
                                                labelled CONSTRUCT_MISMATCH
TQ10 tables agree with figures           PASS   Table 4's H1, H2 and adjusted estimates
                                                are the same values plotted in Figure 3,
                                                read from the same fields
```

## Reviewer simulation, figures and tables only

```
Q1  why are ADI and CV2 insufficient?              Fig 1a: same multisets, different
                                                   order.  ANSWERED
Q2  is the comparison fair?                        Fig 1c and Table 1: 5,856 = 5,856,
                                                   one trainer, one checkpoint rule.
                                                   ANSWERED
Q3  what did Stage 1 find?                         Table 2: both axes matter and the
                                                   interaction is the largest term.
                                                   ANSWERED
Q4  how does Stage 2 make Stage 1 readable?        PARTIAL.  Figure 2 shows the graded
                                                   sweep, but nothing on the page states
                                                   that Stage 1's structured arm was
                                                   alternation only.  This has to be
                                                   carried by the caption and Section 4.5.
Q5  how do the two axes differ?                    Fig 2b U-shaped against 2c monotone,
                                                   shared axis.  ANSWERED
Q6  is there a direct-favourable region?           Fig 2a: one blue cell, labelled.
                                                   ANSWERED
Q7  did it replicate in real data?                 Fig 3a says "empirical analogue";
                                                   Table 4 says "not replication".
                                                   ANSWERED as intended, i.e. no
Q8  does the isolated mechanism survive?           Fig 3c: the estimate crosses zero
                                                   after weighting.  ANSWERED, i.e. no
Q9  did routing succeed?                           NOT ANSWERABLE from the figures, by
                                                   design.  Routing is Section 5.7 text
                                                   plus an appendix; no main figure.
```

Q4 is the one gap. It is a caption and text obligation rather than a rendering defect,
and it is already required by the frozen outline, but a reader who looks only at the
figures could still read Stage 1 and Stage 2 as interchangeable.
