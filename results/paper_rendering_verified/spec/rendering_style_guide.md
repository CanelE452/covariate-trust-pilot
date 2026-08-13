# Rendering style guide

Applies to every figure and table in the manuscript. Enforced in `render_drafts.py`
and `render_tables.py`, which regenerate every draft from verified artifacts.

## Output

```
formats        PDF and SVG for the manuscript, PNG for preview
text           never rasterized:  pdf.fonttype 42, svg.fonttype "none"
width          double-column candidate 7.16 in; single-column candidate 3.5 in
               no venue template exists in the repository, so no journal-specific
               dimension was invented
base type      7 pt; axis labels 7 pt; ticks 6.5 pt; in-panel notes 5.6-6.2 pt
line weight    axes and ticks 0.7 pt
spines         top and right removed except on heatmaps
panel labels   (a), (b), (c), left-aligned in the axes title
```

## The performance quantity

```
G = 100 (1 - RMSE_Hurdle / RMSE_Point)      units: percentage points, "pp"
G > 0 favours factorization,  G < 0 favours direct prediction
```

Written the same way everywhere: axis label `G (pp)`, colourbar annotated
"factorized better" at the top and "direct better" at the bottom. The legacy absolute
deltas of the underlying artifacts never appear in a main figure or table; they are
explained once in the appendix.

## Notation

```
rho_I    occurrence-interval dependence      italic subscript I
rho_M    positive-magnitude dependence       italic subscript M
d        mean inter-demand interval (sparsity level), d in {4, 8}
direct / factorized     never "Point model" / "Hurdle model" in axis text
```

## Colour

```
diverging     RdBu_r, zero at the neutral midpoint, symmetric limits
sequential    not used
red/green     never a semantic pair
redundancy    every coloured cell also prints its value, so hue is never the sole
              channel; grayscale printing loses nothing essential
```

## Uncertainty

```
intervals     a line through the estimate; never a shaded band that could be
              mistaken for a density
spread        when several cells are summarized and no interval exists for the
              summary, individual cells are shown as open circles and labelled
              "spread, not CI"
significance  a discrete marker, never colour saturation, and never the same visual
              channel as effect magnitude
stars         not used
```

## In-panel text

Minimal. One short note per panel at most. Everything else belongs in the caption.
No prose sentences inside axes.

## Regeneration

```
python results/paper_rendering_verified/render_drafts.py
python results/paper_rendering_verified/render_tables.py
```

Both read only from `results/synthetic_source_verification/` and
`results/external_validity_screen/`. Neither trains anything, scores anything, or
writes outside `results/paper_rendering_verified/`. Every displayed quantity is
appended to `spec/source_map.csv` at render time, so the map cannot drift from the
figures.
