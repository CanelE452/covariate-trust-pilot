# Phase A — audit and reproduction

Read-only. No existing file was modified, moved or deleted; `git status` shows one
pre-existing modified history file and only new untracked directories from this work.

---

## A1 repository state

```
root                    E:/CODING/proj/covariate-trust-pilot
protocol of record      results/external_validity_screen/pre_analysis_spec.json
Point implementation    M0PM_point_mse_param_matched
Hurdle implementation   M1_factorized_mean  (robustness pair M2 hurdle-ZTNB exists)
trainer                 experiments.om_factorization_killtest.train.train_one, imported
                        unchanged
primary metric          RMSE against realized y on the 28-step test window, per series
                        x origin; delta = RMSE_Point - RMSE_Hurdle, positive favours Hurdle
natural population      independent eligible pools: m5 5,693, favorita 5,405
inventory evaluation    NOT FOUND.  No lead time, review period, cost or service target is
                        defined anywhere in the repository, so no inventory metric was
                        computed and none was invented.
```

## A2 dataset compatibility

```
dataset            freq   target   history  horizon  origins  eligible  raw preds  head preds  natural  compatible
m5                 daily  units     1,941       28        3     5,693        yes        yes      yes    YES
favorita           daily  units     1,688       28        3     5,405        yes        yes      yes    YES
FreshRetailNet-LT  -      -             -        -        0         -         no         no       -    NO
UCI Online Retail  -      -             -        -        0         -         no         no       -    NO
```

FreshRetailNet-LT and UCI appear only as routing stress tests. They carry routing-gain
artifacts, not paired Point/Hurdle per-series RMSE, and no origin-level predictions in this
schema. **Leave-one-dataset-out over four datasets is therefore not available**, and the
analysis is reported as a two-domain transfer pilot throughout.

## A3 Point-Hurdle fairness

Verified identical: target, train window, forecast origins (m5 1857/1885/1913, favorita
1604/1632/1660), horizon 28, lookback 96, covariates, DLinear backbone, parameter count,
training budget, canonical model seed 0, evaluation metric. The only difference is the
output factorization and its objective, which is the comparison itself. Paired causal
interpretation is supported.

## A4 reproduction of the previously reported 3-origin numbers

Recomputed independently from the raw prediction parquets.

```
quantity          recomputed   previously reported   abs diff
point win %          33.8469               33.8500     0.0031
hurdle win %         44.8249               44.8200     0.0049
neutral %            21.3282               21.3300     0.0018
max |G - G_prior| over all 33,294 rows                 3.40e-05
```

Tolerance for the win shares was set at 0.05 pp and for per-row `G` at 1e-3. Both pass with
large margin; the residual is float formatting in the earlier CSV. **Reproduction PASS** —
the analysis proceeds.
