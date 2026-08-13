# Metric and sign convention

Read from the recovered prereg and from the actual result columns. This section
matters because **the three studies do not all use the same delta sign**, and mixing
them would silently invert a claim.

---

## Primary metric

```
PRIMARY_METRIC        "rmse_mean_truth"
target                mean_true = p_true * mu_true, from the exact DP oracle
aggregation           per series, then bootstrapped across series
PRIMARY_COMPARISON    M1_HURDLE_MEAN vs M0_PARAMETER_MATCHED_POINT
SECONDARY_COMPARISON  M2_HURDLE_ZTNB vs M0_PARAMETER_MATCHED_POINT
bootstrap             unit = series, draws = 2000, level = 0.95, paired,
                      seed = 20260802, model seeds averaged within a series first
```

---

## The three sign conventions in play

```
study                        formula                             positive means
──────────────────────────────────────────────────────────────────────────────────
Stage 1 absolute (prereg)    delta = RMSE_hurdle - RMSE_point     Point better
Stage 1 relative (prereg)    gain  = 1 - RMSE_hurdle/RMSE_point   Hurdle better
Stage 2 delta (cell_metrics) delta = loss_point - loss_hurdle     Hurdle better
Stage 2 relative             gain  = delta / loss_point           Hurdle better
Real data, Stage A           delta = RMSE_Point - RMSE_Hurdle     Hurdle better
```

Verified numerically on the first Stage 2 row: `loss_point 1.34639`,
`loss_hurdle 1.20409`, `delta 0.14231` = point − hurdle, and
`gain 0.10569` = delta / loss_point.

**Consequence.** `gain` is consistent across all three and always means "positive
favours Hurdle". `delta` is **not**: Stage 1's absolute delta is the negative of
Stage 2's and of the real-data one.

**Rule for the paper.** Use `gain` (or an explicitly named relative improvement)
whenever synthetic and real numbers appear in the same table or sentence. If an
absolute delta is shown, state its formula on the same line. The existing
`results/paper_synthesis/` never quotes a synthetic delta, so nothing there is
inverted — but Figure 2 and the Stage 1 table are where this would first bite.

---

## Both scales are reported by design

`REPORT_BOTH` in the prereg requires the absolute delta and the relative gain
side by side, with the stated reason that "앞선 실험에서 분모 변화가 비율을
오도한 전례" — a prior experiment where a moving denominator made the ratio
misleading. The paper should keep that pairing.

---

## Secondary metrics available in the recovered run

```
realized_y_mse, realized_y_mae, occurrence_brier, rmse_p_truth,
positive_mean_rmse, train_seconds, n_parameters
hybrid diagnostics: p_true_x_mu_hat, p_hat_x_mu_true, p_hat_x_mu_hat
```

The three hybrid columns are the mechanism decomposition: they isolate whether the
occurrence head or the magnitude head carries the error, by substituting the true
component for the estimated one.

---

## Verdict

```
C1-G7  metric, sign and statistics verified from source   PASS
       with the standing requirement that `gain` is used for cross-study comparison
       DELTA_SIGN_VERIFIED, three conventions documented
```
