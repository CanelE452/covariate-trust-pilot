# Phase A — repository and artifact audit

Read-only. Nothing in `results/external_validity_screen/` or any other existing directory
was modified, moved or deleted.

---

## A1 repository state

```
root                 E:/CODING/proj/covariate-trust-pilot
git status           22 entries, all untracked new work or one pre-existing modified
                     history file; this task added only two untracked directories:
                       experiments/pointhurdle_condition_discovery/
                       results/pointhurdle_condition_discovery/
real-data pipeline   experiments/external_validity_screen/   (screen, prereg, rule
                     replication, favorita transfer, classical benchmark, seeds)
protocol of record   results/external_validity_screen/pre_analysis_spec.json
```

## A2 artifact inventory

Full table in `artifact_inventory.csv`. The decisive finding:

```
dataset   paired series  origins  horizon  primary metric        raw preds  head preds  natural sample
m5              5,693        3       28    RMSE on realized y      yes       p_hat,mu    independent pool
favorita        5,405        3       28    RMSE on realized y      yes       p_hat,mu    independent pool
Stage A         2,400        0       28    RMSE on realized y      NO        aggregated  NO - regime-balanced
Fresh / UCI         0        0        -    routing gain only       n/a       no          n/a
```

Three points matter.

**The Stage A table cannot support this analysis.** `per_series_metrics.csv` carries one row
per series with the three origins already collapsed, and no raw predictions. It is also the
SBC regime-balanced 300-per-regime sample, so its policy numbers are not natural-distribution
numbers.

**The independent-population parquets can.** They carry `series_id, group, origin, step,
y_observed, occurrence, target_mask, point_mean_prediction, hurdle_mean_prediction,
hurdle_p_prediction, hurdle_mu_prediction` — origin-level, paired, with both Hurdle heads.
This is what the panel is built from.

**FreshRetailNet-LT and UCI carry no paired Point/Hurdle RMSE in this schema.** They exist
only as routing stress tests. A four-dataset leave-one-dataset-out design is therefore not
available from cached artifacts today; see `missing_requirements.md`.

## A3 model comparison integrity

Full table in `protocol_comparison.csv`. Target, inputs, lookback (96), horizon (28), splits,
backbone, trainer, seed, evaluation metric and all three test origins match between the two
arms. The only difference is the one the study is about: the output factorization and its
objective. Paired causal interpretation is therefore supported by the cached artifacts.

## A4 stop conditions — none triggered

```
Point and Hurdle test ranges differ            no  - identical origins and horizon
cannot match >= 95% of series-origin pairs     no  - 100.00% paired coverage
metric definition not reproducible             no  - RMSE on realized y, per the prereg
test target or series mapping unclear          no  - series_id and target_mask explicit
outer-test information inside cached features  no  - population descriptors are train-window
                                                     only, by the prereg descriptor_rules
only aggregate artifacts available             no  - origin-level predictions exist
```

Audit passes. The panel was built without retraining anything.
