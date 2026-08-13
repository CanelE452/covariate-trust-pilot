# Figure specification

Three main figures. No figure is rendered here. Every panel names the artifact it
draws from. Metric throughout: `G = 100 (1 − RMSE_Hurdle / RMSE_Point)`, G > 0
favours the factorized model.

---

## Figure 1 — Problem and controlled design

```
number        1
section       3 (Problem setup and forecasting formulations)
placement     main
purpose       state the gap in marginal descriptors and the design that closes it
message       "Two series can share every marginal summary and still differ in the
              order of their events; we hold the marginals fixed and vary only that
              order, then compare two representations of the same conditional mean."
source        conceptual; parameters quoted from prereg GENERATION and DATA blocks
new plot      yes, entirely
```

**Panels.**

- **1A — the discarded axis.** Two short synthetic series drawn with the *same* gap
  support (`d−1`, `d+1`, share 0.5 each) and the *same* positive-magnitude marginal
  (`λ ∈ {5,15}`, long-run mean 10), one with structured order and one iid. Annotate
  that ADI, CV² and the interval support are identical between them. Values may be
  redrawn illustratively from the DGP definition; do not copy real cells, and say so
  in the caption.
- **1B — the two process axes, presented symmetrically.** Occurrence dependence
  `ρ_I < 0 / 0 / > 0` labelled alternating / iid / persistent; magnitude dependence
  `ρ_M < 0 / 0 / > 0` with the same three labels. Equal visual weight, equal
  treatment.
- **1C — the two estimators.** Direct: `E[Y | history]`. Factorized:
  `P(Y>0 | history) × E[Y | Y>0, history]`. Show that both target the same quantity
  and that the factorized path splits one budget across two heads. Annotate the
  matched parameter count (5,856 each).

**Hard constraint — no result leakage.** Figure 1 must not indicate that `|ρ_I|` is
what matters, that the sign of `ρ_M` is what matters, or where either model wins.
The three ρ levels appear as design levels only. Axis 1B must not be drawn with any
asymmetry in emphasis, ordering or colour.

**Caption skeleton.** *"Controlled temporal structures with matched marginal
properties. Both arms of each axis draw from the same support with the same long-run
frequency and differ only in ordering; the two estimators target the same conditional
mean under an identical parameter budget. Levels shown are design levels, not
results."*

---

## Figure 2 — Controlled synthetic discovery

```
number        2
section       4.5 - 4.7
placement     main; this is the paper's central result figure
source        results/synthetic_source_verification/stage2_verified_cells.csv
              (model == HURDLE_MEAN; 18 cells, each with gain and a bootstrap CI)
              results/synthetic_source_verification/stage2_verified_factor_effects.csv
aggregation   per cell: paired series bootstrap, 2000 draws, 80 series per cell
new plot      yes, all panels
message       "Occurrence effects are primarily associated with dependence strength,
              magnitude effects with dependence direction; within this grid only one
              configuration shows statistically clear superiority for direct
              prediction."
```

### 2A — the full sweep

```
layout        two 3x3 heatmaps side by side, one per sparsity level (d = 4, d = 8)
x             rho_M = -0.8, 0.0, +0.8
y             rho_I = -0.8, 0.0, +0.8
cell value    G in percentage points, printed in the cell
uncertainty   see "statistical marking" below; encoded as a marker, not as colour
colour        diverging, ZERO-CENTRED and SYMMETRIC
range         symmetric about zero, covering the observed extremes:
              min -19.76, max +22.81  ->  use [-23, +23] (or the rounded
              max absolute value), identical for both panels
```

**Colour normalization, fixed.** Zero must be exactly the neutral midpoint. The two
panels (d = 4 and d = 8) share one scale. Forbidden: min/max linear normalization that
puts zero off-centre; an asymmetric range that exaggerates the negative cell; separate
scales per panel or per sign.

With a symmetric zero-centred scale the seventeen positive cells occupy one hue family
and the single negative cell is the only one in the opposite hue, so it separates on
its own. No highlight colour is added to force it.

**Cell text.** Print `G` in every cell, signed, one decimal: `+12.4`, `-19.8`.
Numeric clarity is carried by the text, not by the colour.

**Statistical marking is separate from colour.** Cells whose CI excludes zero carry one
minimal marker — a border, dot or star, choose one — and the legend states
`marker = CI excludes zero`. Exactly one cell lacks it: `d=8, ρ_I=0, ρ_M=0`,
`G = +2.36` [−1.20, +5.73]. Do not encode significance in the colour.

### 2B — occurrence marginal

```
x             rho_I = -0.8, 0.0, +0.8
y             mean G over the six cells at each level, in pp
values        +12.10, +3.57, +16.47
uncertainty   cell spread at each level shown as light points or a range band
              (-0.8: [+8.66, +15.92];  0: [-19.76, +14.36];  +0.8: [+9.25, +22.81])
shape         a U, minimum at zero
annotation    "|rho_I| coefficient +0.190 [+0.170, +0.212];
               signed rho_I +0.067 [+0.053, +0.081]"
wording       "primarily associated with dependence strength" -- never a causal claim
```

### 2C — magnitude marginal

```
x             rho_M = -0.8, 0.0, +0.8
y             mean G over the six cells at each level, in pp
values        +14.10, +11.94, +6.11
uncertainty   cell spread as in 2B
              (-0.8: [+8.66, +22.81];  0: [+2.36, +21.66];  +0.8: [-19.76, +13.95])
shape         monotone decreasing
annotation    "signed rho_M coefficient -0.071 [-0.085, -0.057];
               |rho_M| -0.023 [-0.044, -0.002]"
wording       "primarily associated with dependence direction" -- never a causal claim
```

**2B and 2C must share a y-axis** so the reader compares a U against a line rather
than two rescaled shapes. The wide spread at `ρ_I = 0` and `ρ_M = +0.8` is the same
cell in both panels; that is the point, and the caption should say so rather than
hide it.

**Caption skeleton.** *"Relative performance across the stationary dependence sweep.
(A) All eighteen conditions; within this grid the only cell with statistically clear
superiority for direct prediction is sparse, occurrence-unpredictable and
magnitude-persistent. Markers indicate cells whose interval excludes zero. (B) Averaging over magnitude, the
factorized advantage rises with the strength of occurrence dependence and is
insensitive to its sign. (C) Averaging over occurrence, the advantage falls
monotonically with the direction of magnitude dependence. Bars/bands show the spread
across the cells averaged at each level; coefficients are from the factor model."*

---

## Figure 3 — Empirical transfer and boundary

```
number        3
section       5.3 - 5.5
placement     main
message       "An empirical analogue of the occurrence relationship appears, and the
              frozen configuration shows predictive transfer; the association does not
              survive overlap adjustment."
new plot      3A restyled from an existing figure; 3B partly; 3C new
```

### 3A — H1 analogue

```
panels        two, M5 and Favorita
x             |rho_interval| measured on the training window
y             per-series relative performance (state the scale in the axis label)
overlay       Spearman with CI: M5 +0.1065 [+0.0437, +0.1652];
              Favorita +0.0789 [+0.0205, +0.1405]
inset         forest of the twelve regime x scale estimates from regime_h1.json, so
              the intermittent intervals clear of zero and the lumpy ones straddling
              it are both visible
source        stage_a_results.json, posthoc_diagnostic.json, regime_h1/regime_h1.json
              existing figA_H1_rho_interval_vs_delta.* is a starting point
caption note  must say "empirical analogue", never "replication"
```

### 3B — H2 selector transfer

```
panels        candidate vs control distributions on the independent M5 population
              (675 vs 5,018)
overlay       rule effect -0.0230 [-0.0294, -0.0163];
              Point win-rate difference +11.87 pp [+7.85, +15.81]
strip         the three seeds side by side (-0.0230 / -0.0211 / -0.0277, all intervals
              clear of zero)
source        rule_replication/primary_result.json, seed_robustness.json
              existing figB_H2_point_candidate_vs_control.* is a starting point
link          a small inset or marginal note tying this panel to the Figure 2A cell
              it was derived from; the axes correspond one to one
```

### 3C — the entanglement boundary

```
content       covariate balance before and after overlap weighting, with
              log_train_scale going from SMD 0.614 (matching failed) to 0.0004,
              beside the overlap-adjusted association +0.0032 [-0.0033, +0.0094]
              crossing zero
source        rule_replication/secondary_overlap.json,
              h2_confirmatory/matching_failed.json
```

**This is the most important new panel in the paper.** It is what separates a
predictor from an explanation, and it should be drawn so that the crossing of zero is
the first thing the eye lands on.

**Caption skeleton.** *"Transfer and its boundary. (A) The occurrence-dependence
relationship reproduces in direction on both datasets and strengthens in the
intermittent regime. (B) A rule frozen from the controlled study shifts unseen series
toward direct prediction, across three training seeds. (C) After balancing sparsity,
variability, scale and occurrence dependence, the same association is not
distinguishable from zero; the selector predicts without explaining."*

---

## Deliberately not a figure

A mechanism figure for real data. The learned occurrence head has no skill against a
constant per-series rate (Brier skill −0.008 on M5, −0.091 on Favorita with the
interval clear of zero). The component-attribution diagnostic belongs inside the
controlled study and, if drawn at all, belongs in the appendix beside Stage 1 — never
captioned as if validated on M5 or Favorita.

---

## Figure-only reading test

```
Fig 1   here is the problem and here is what we control
Fig 2   here is what the control reveals, and it is asymmetric
Fig 3   here is how much survives in real data, and where it stops
```


---

## Routing is not a main figure

Section 5.7 reports the adaptive-use boundary in text and, if space allows, a small
table. No routing panel appears in Figures 1–3. Detailed routing figures, if any,
belong in Appendix H. This keeps the figure sequence at
problem -> discovery -> transfer and boundary, and prevents C3 from reading as a
second paper.
