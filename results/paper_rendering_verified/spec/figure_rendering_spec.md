# Figure rendering specification

Panel-level specification for the three main figures. Numeric values are not restated
here; they are in `spec/source_map.csv`, which the renderer writes itself so it cannot
drift from the figures.

---

## Figure 1 — problem and controlled design

```
draft        drafts/figure1_draft.{pdf,svg,png}
size         7.16 x 1.95 in
claim        marginal summaries do not determine temporal order, and the comparison
             we run holds the marginals and the budget fixed
```

**(a) same marginals, different order.** Two event trains with an identical gap
multiset and an identical magnitude multiset, differing only in arrangement. Stems at
event times, height proportional to magnitude. Hand-built, and labelled inside the
panel as *schematic illustration; not experimental data*.

**(b) two dependence axes.** Occurrence and magnitude on two identical horizontal
axes, each with markers at −0.8, 0.0, +0.8 labelled alternating / iid / persistent.
Equal weight, equal ordering, equal styling. **No result may appear here.**

**(c) two representations.** History branches to direct and to factorized, both
returning to a shared node reading *same conditional mean target, matched budget:
5,856 parameters each*. That count is the only artifact-derived number in Figure 1.

**Caption obligation.** State that the levels shown are design levels rather than
findings, and that panel (a) is schematic.

---

## Figure 2 — controlled synthetic discovery

```
draft        drafts/figure2_draft_markerB.{pdf,svg,png}   recommended
             drafts/figure2_draft_markerA.{pdf,svg,png}   comparison only
size         7.16 x 4.5 in
source       stage2_verified_cells.csv, stage2_verified_factor_effects.csv
claim        occurrence effects track dependence strength, magnitude effects track
             dependence direction, and one configuration favours direct prediction
```

**(a) the sweep.** Two 3x3 heatmaps, `d = 4` and `d = 8`. Rows `rho_I`, columns
`rho_M`, both ordered −0.8, 0.0, +0.8 with the origin at lower left. Each cell prints
its `G` to one decimal with an explicit sign. One diverging norm shared by both
panels, symmetric about zero at ±23 pp, derived from the observed |max| of 22.81. A
single open circle marks the one cell whose interval includes zero; the caption states
that every unmarked cell excludes it.

**(b) occurrence marginal.** Mean `G` over the six cells at each `rho_I` level as a
large filled marker, with the six contributing cells as open circles. A thin
connecting line, deliberately faint so three levels are not read as a continuous
function. One compact annotation comparing the `|rho_I|` and signed `rho_I`
coefficients.

**(c) magnitude marginal.** Identical construction over `rho_M`. **Shares (b)'s
y-axis**, with its own tick labels suppressed, so a U-shape and a monotone line are
compared directly rather than through two rescalings.

**Caption obligations.** State the metric and its sign. Scope the direct-favourable
cell to this grid. Describe the open circles as spread, not intervals. State that
Stage 1's structured arm was alternation only, which is why the graded sweep is
needed — without this the two stages read as interchangeable.

---

## Figure 3 — empirical transfer and boundary

```
draft        drafts/figure3_draft.{pdf,svg,png}
size         7.16 x 2.75 in
claim        an analogue appears and the frozen configuration selects out of sample,
             but the association does not survive adjustment
```

**(a) H1 analogue.** Horizontal forest. M5 and Favorita first, then the four M5 SBC
regimes indented and drawn smaller. Filled markers where the interval excludes zero,
open where it does not, so the lumpy null is visible rather than omitted. Axis:
Spearman. The panel title says *empirical analogue*.

**(b) H2 selector.** Forest of the three seeds and the 3-seed aggregate, in percentage
points of relative error, with a note that seed 0 is the primary run. No dual axis:
the win-rate difference is reported in Table 4 rather than crowded onto this panel.

**(c) mechanism boundary.** The same estimate unadjusted and overlap-adjusted, in the
same units as (b), with a zero reference line the adjusted interval crosses. A note
below gives the worst |SMD| on log scale at three stages: unweighted, after 1:2
matching, after weighting.

**Caption obligations.** H1 is an analogue, not a replication. Panels (b) and (c) are
the same quantity under two adjustments, and the difference between them is the
finding. No routing result appears in this figure.

---

## What is deliberately absent

```
routing            no main-figure panel; Section 5.7 text plus Appendix H
mechanism figure   the component-attribution diagnostic stays in the appendix beside
   for real data   Stage 1; the learned occurrence head has no skill on real data
stars              significance is a discrete marker or nothing
dual axes          never; two units go in two panels or one goes to a table
```
