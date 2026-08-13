# Figure captions — draft v1

Captions describe what is plotted and how to read it. **No caption states a scientific
claim that is not already made in the body**, and no caption introduces a number that is
not in `results_number_source_map.csv`.

---

## Figure 1 — the problem and the controlled design

> **Figure 1. The comparison and the design that isolates it.**
> **(a)** Two schematic intermittent series with identical marginal properties — the same
> gap support, the same average interval and the same positive-demand distribution — that
> differ only in the order in which gaps and magnitudes arrive. **(b)** The two forecasting
> formulations compared throughout: a direct arm predicting the conditional mean, and a
> factorized arm predicting an occurrence probability and a conditional positive magnitude
> and multiplying them. **(c)** The matched experimental conditions (Table 1): one backbone,
> 5,856 parameters in each arm, one optimizer, one training budget and one evaluation
> target. Panel (a) is a schematic; the three dependence levels drawn are design levels, and
> the panel is drawn symmetrically in the two axes so that no outcome is implied.

Scope note for the caption file, not for the manuscript: Figure 1 must not hint at either
`|ρ_I|` or the sign of `ρ_M` being the operative quantity. That is a Section 4.8 result.

## Figure 2 — the controlled synthetic result

> **Figure 2. Relative performance across the Stage 2 dependence sweep.**
> `G = 100(1 − RMSE_H / RMSE_P)` in percentage points, measured against the exact oracle
> conditional mean; **positive values favour the factorized arm and negative values favour
> the direct arm**. **(a)** All 18 cells, as two 3 × 3 panels for `d = 4` and `d = 8`, over
> `ρ_I` and `ρ_M`. `G` is printed in every cell. The colour scale is diverging, zero-centred
> and symmetric over [−23, +23], and is shared by both panels. An **open circle marks a cell
> whose 95% interval includes zero**; significance is shown by this marker and not by
> colour. **(b)** Mean `G` at each level of `ρ_I`, averaged over the six contributing cells.
> **(c)** Mean `G` at each level of `ρ_M`, on the same y-axis as (b).
> **In (b) and (c) the vertical extent shows the spread of the six contributing cells
> and is not a confidence interval.**
> Inference in the text rests on the factor-model coefficients and on the per-cell
> intervals in (a).

## Figure 3 — empirical transfer and its boundary

> **Figure 3. What transfers to observed demand, and what does not.**
> **(a)** H1: the association between the strength of interval dependence and the relative
> advantage of the factorized arm, on M5 and Favorita, with the by-regime estimates inset.
> **(b)** H2: the frozen selector applied to an independent M5 population, shown for each of
> three training seeds alongside the pooled estimate. **(c)** The overlap boundary: covariate
> balance before and after weighting, and the isolated association, whose interval crosses
> zero after adjustment. Panels (a) and (b) report transfer; panel (c) reports its limit.
> All intervals are 95% percentile intervals from a paired series bootstrap with 2,000 draws.

No routing panel appears in any main figure.

---

## Appendix figures

```
A1  classical baselines by mean rank, M5 and Favorita
A2  the routing chain: oracle opportunity, expert diversity, and the external results
    including the UCI failure at full scale
A3  Stage 1 per-cell G with intervals, all eight cells
```
