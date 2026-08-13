# Final rendering freeze

Frozen 2026-08-11. Anything later that contradicts this is a defect. Full reasoning is
in `review/rendering_decisions.md`; this page is the short list.

---

## Decisions

**Figure 2a: Option B.** Only the cell whose interval includes zero carries a mark.
Re-read from `stage2_verified_cells.csv` at freeze time: that is exactly one cell of
eighteen — `d = 8, ρ_I = 0, ρ_M = 0`, `G = +2.36` [−1.20, +5.73]. Every other cell's
interval excludes zero.

Legend and caption wording, fixed:
> Open circle denotes a cell whose confidence interval includes zero; all unmarked
> cells have intervals excluding zero.

The open circle is an **uncertainty encoding**, not a marker of interest. No stars.
The 17-marker variant is retained only as
`drafts/figure2_comparison_markerA_NOT_FINAL.*` and is not a manuscript figure. The
manuscript figure is `drafts/figure2_final.*`.

**Symmetric zero-centred colour.** Re-read at freeze time: `min = −19.7643`,
`max = +22.8073`, so `|max| = 22.81` and the limits stay `[−23, +23]`, shared by the
`d = 4` and `d = 8` panels, with zero at the exact neutral midpoint. Each cell prints
its signed `G`. The direct-favourable cell — `d = 8, ρ_I = 0, ρ_M = +0.8`,
`G = −19.76` [−26.00, −14.53], interval excluding zero — is left to emerge from the
opposite hue and its printed value. No added highlight.

**Classical benchmark moves to the appendix.** Table 4 is the empirical hypothesis
summary: H1, H2 selector, H2 overlap-adjusted mechanism, H3. Verified at freeze time
that no classical method appears in `table4_draft`. The full eight-method ranking on
both datasets goes to Appendix L, referenced from Section 5.2 by one sentence.

This is not concealment. The paper states plainly that both representations are
outranked by SBA on both datasets, and that its contribution is conditional relative
behaviour and transfer boundary rather than absolute accuracy.

**No single-column variant.** Figures 1–3 are frozen at the double-column candidate
(7.16 in); Figure 2 requires full width. A 3.5 in re-layout is built only if a target
venue demands it. No 2×2 or vertical-stack alternative was generated.

---

## Warnings that must survive into the manuscript

**`UNIT-W1` — `C_neg` / `C_pos` are absolute RMSE-difference contrasts, not `G` (pp).**

Verified numerically at `d = 4`:
```
delta(ρ_I=-0.8, ρ_M=0)      0.22672404
delta(ρ_I= 0.0, ρ_M=0)      0.08599591
difference                  0.14072813
C_neg (stored)              0.14072813     exact match
the same difference in gain 0.08110814     does not match
```
They are also computed **at the `ρ_M = 0` slice only**, so they are not marginal
effects over `ρ_M`.

Forbidden: displaying them on a `G` axis; labelling them pp; using them as the
Figure 2b/2c marginal effect; quoting them as a C1 effect size. They appear in no
figure and no table, and must stay out.

**`FLAG-W2` — `C_sign` POOLED carries an internal flag inconsistency.**
```
d          value      95% CI                 interval excludes 0?   stored flag
4         -0.02750   [-0.06653, +0.01507]    no                     False
8         +0.17590   [+0.12651, +0.22380]    yes                    True
POOLED    +0.07420   [+0.02999, +0.11943]    yes                    False
```
The run's own `final_report.md` reads the interval and says "yes, but small". Two
readings are defensible — the pooled interval directly, or an AND rule over `d` — and
the artifacts disagree about which was used. Classified
`ARTIFACT_INTERNAL_FLAG_INCONSISTENCY`. The cause is **not** guessed at and no
bootstrap is re-run. `C_sign` pooled significance is
`NOT USED FOR MANUSCRIPT INFERENTIAL CLAIM`, in text, figures or tables.

**`CAPTION-O1` — Stage 1's structured magnitude arm was alternation only.**

Reviewer simulation Q4 was PARTIAL: from the figures alone, Stage 1 and Stage 2 read
as interchangeable, and Stage 1's magnitude-structured cells being
factorized-favourable looks inconsistent with Stage 2's magnitude-persistent cell being
direct-favourable. The fix is one sentence, not a busier figure. Required in the
Stage 1 → Stage 2 transition and in the Figure 2 caption, to this effect:

> Stage 1 contrasts an alternating magnitude structure with its iid counterpart,
> whereas Stage 2 separately varies the sign of magnitude dependence; this distinction
> motivates the signed dependence sweep.

---

## Retained from the previous freeze

```
Figure 2b/2c   no marginal bootstrap CI exists in the artifacts, so none is drawn;
               cells appear as open circles labelled "spread, not CI"; the two panels
               share one y-axis
Figure 3       (b) and (c) both in percentage points of relative error; the duplicate
               "primary = seed 0" row is gone, replaced by the 3-seed aggregate;
               no routing panel
Figure 1       schematic, labelled as such; no result leakage; 5,856 is its only
               artifact-derived number
FreshRetailNet-LT and UCI Online Retail II stay in the appendix dataset table
```

---

## QA status at freeze

```
V1-V15    PASS      checked against the rendered drafts, not the specification
TQ1-TQ10  PASS
source_map            88 displayed quantities, 0 unmapped, 0 sourced from prose
reviewer simulation   Q1-Q3, Q5-Q9 answerable from figures alone
                      Q4 addressed by CAPTION-O1 rather than by the figure
```

Regenerate with:
```
python results/paper_rendering_verified/render_drafts.py
python results/paper_rendering_verified/render_tables.py
```
