# Figure 2 readiness

**Status: `FIG2_SOURCE_READY`.**

The 18-cell Stage 2 table is exactly the shape the planned heatmap needs, with a
bootstrap interval per cell. Nothing has to be re-simulated and nothing has to be
re-aggregated beyond a pivot.

```
source        results/synthetic_source_verification/stage2_verified_cells.csv
              (copied from reports/temporal_dependence/rho_sweep/
               pilot_20260803_051713/cell_metrics.csv)
rows          36 = 18 cells x 2 hurdle models; use model == HURDLE_MEAN for the primary
axes          x = rho_interval in {-0.8, 0.0, +0.8}
              y = rho_magnitude in {-0.8, 0.0, +0.8}
panels        one per sparsity level, d = 4 and d = 8
cell value    gain, positive favours Hurdle
uncertainty   gain_ci_low, gain_ci_high, and delta_ci_excludes_zero per cell
annotation    mark d=8, rho_I=0, rho_M=+0.8 as the only Point-favourable cell
              (gain -19.76%, CI [-26.00, -14.53])
```

A companion panel for Stage 1 is also available if the paper wants the alternation
contrast beside the graded one: `stage1_verified_contrasts.csv` has all seven
factorial effects with intervals.

No figure was rendered in this task.

## One caution for whoever renders it

The colour scale must be centred on zero and must not be symmetric-clipped: 17 of
18 cells lie between +2.4% and +22.8%, and the single negative cell is -19.8%.
A naive diverging scale will make the one cell that matters look like noise.
