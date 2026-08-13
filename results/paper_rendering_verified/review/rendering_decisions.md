# Rendering decisions

Every decision below was made against a rendered draft, not in the abstract. Three of
them changed what the figures show; two are corrections to earlier advice of mine.

---

## D1. Figure 2a significance marking: Option B

**Counted first.** Of the 18 Stage 2 cells, **17 have intervals excluding zero and
exactly one does not** (`d=8, rho_I=0, rho_M=0`, `G=+2.36` [-1.20, +5.73]).

Both options were rendered and compared side by side
(`figure2_draft_markerA.png`, `figure2_draft_markerB.png`).

```
Option A  mark the significant cells        17 dots.  Cluttered; the dots read as a
                                            texture and add no information, because
                                            "significant" is the default state here.
Option B  mark the ONE non-significant cell   1 open circle.  Clean, and the legend
                                            line does the work:
                                            "all unmarked cells have intervals
                                             excluding zero"
```

**Recommendation: Option B.** This inverts the visual encoding but not the science —
the same fact is conveyed with 1 mark instead of 17. Stars were avoided throughout:
they read as "special discovery" rather than "interval excludes zero".

---

## D2. Figure 2a colour: symmetric, and my earlier advice was wrong

Actual range read from the artifact: `min = -19.7643`, `max = +22.8073`, so
`|max| = 22.81` and the symmetric limit `[-23, +23]` in the frozen spec is correct.
One scale for both panels, zero exactly at the neutral midpoint.

An earlier draft of the figure specification recommended an **asymmetric** scale, on
the theory that a symmetric one would wash out the single negative cell. Rendering
shows the opposite: with a diverging scale centred at zero, the seventeen positive
cells occupy one hue family and the negative cell is the only one in the other, so it
is the most conspicuous cell on the page without any added emphasis. The asymmetric
advice was withdrawn.

---

## D3. Figure 2b/2c uncertainty: cell spread, never called a CI

`primary_contrasts.csv` was checked for a valid bootstrap interval on the marginal
means. It does not contain one. What it contains is **contrasts between rho_I levels**
(`C_neg`, `C_pos`, `C_pred`, `C_sign`) with their own intervals — a different quantity.

CASE 2 of the specification therefore applies: the panels show each level's mean over
its six cells as a large filled marker, the six individual cell values as open
circles, and the label states explicitly *"open circles: individual cells (spread, not
CI)"*. No interval was synthesised from cell-level intervals.

---

## D4. `C_neg` / `C_pos` are absolute-delta contrasts, not gains

Checked numerically rather than assumed. At `d = 4`:

```
delta(rho_I=-0.8, rho_M=0)  0.22672404
delta(rho_I= 0.0, rho_M=0)  0.08599591
difference                  0.14072813
C_neg (stored)              0.14072813      exact match
the same difference in gain 0.08110814      does NOT match
```

So `C_neg` and `C_pos` are contrasts in the **absolute normalized delta**, evaluated
**at rho_M = 0 only**, and are not marginal means over rho_M and not percentages of G.
`C_pred` is their mean and `C_sign` their difference.

**Consequence.** These four quantities must never be placed on a G axis or quoted in
percentage points beside `G` values. The direction conclusion drawn from them —
occurrence dependence raises the factorized advantage for both signs — is unaffected,
because both contrasts are positive with intervals clear of zero. Earlier summary
documents quoted `C_neg +0.1266` and `C_pos +0.2008` next to G figures without noting
the unit change; that is a presentation defect in those documents and is corrected
here. They are not used in any figure.

---

## D5. `C_sign` POOLED carries an internal inconsistency

```
d          value      95% CI                 CI excludes 0?   stored flag
4         -0.02750   [-0.06653, +0.01507]    no               False
8         +0.17590   [+0.12651, +0.22380]    yes              True
POOLED    +0.07420   [+0.02999, +0.11943]    yes              False
```

The POOLED row's interval excludes zero but its stored flag says it does not. The
run's own `final_report.md` reads the interval and says "yes, but small". Two
defensible definitions are in play — reading the pooled interval directly, or
requiring both `d` levels to agree — and the artifacts disagree about which was used.

`C_sign` appears in no figure and in no main table. The paper should not cite its
significance in either direction without resolving this, and the discrepancy is
recorded rather than silently resolved.

---

## D6. Figure 3b and 3c now share one unit

The first draft showed the same quantity twice in two different units: panel (b) had
the rule effect at `-2.30` (x100) and panel (c) had the identical number at `-0.023`
(raw), side by side. Both panels now report percentage points of relative error, so
`-2.30` in (b) and `-2.30` in (c) are visibly the same estimate.

---

## D7. Figure 3b: the "primary" row was a duplicate

`by_seed.0.effect` and `H2_rule_effect` are the same number (`-0.02304`). The first
draft plotted both, implying four independent estimates where there were three.
Replaced with the three seeds plus the **3-seed aggregate**
(`-0.02394` [-0.03045, -0.01721]), and a note that seed 0 is the primary run.

---

## D8. Figure 1 carries no result

Verified by construction: the toy sequences in 1a are hand-built with an identical gap
multiset and an identical magnitude multiset, and are labelled *"schematic
illustration; not experimental data"* inside the panel. Panel 1b shows the three rho
levels as design levels with symmetric treatment of both axes and no indication that
|rho_I| or the sign of rho_M matters. The only artifact-derived number anywhere in
Figure 1 is the matched parameter count, 5,856.

---

## D9. Column width

No venue template exists in the repository, so no journal dimensions were invented.
All drafts are rendered at the double-column candidate width (7.16 in) with 7 pt base
type. A single-column candidate (3.5 in) would require Figure 2 to become a 2x2 or a
vertical stack; that decision is deferred until a venue is chosen.
