# Point / Hurdle fairness

From the recovered `experiments/decomposition_when_helps/prereg.py` and the
per-cell parameter counts actually recorded in the run.

---

## Side by side

```
                        M0_PARAMETER_MATCHED_POINT        M1_HURDLE_MEAN
────────────────────────────────────────────────────────────────────────────────────
implementation          om_factorization_killtest         om_factorization_killtest
                          .models.PointDLinearParamMatched  .models.FactorizedDLinear
backbone                DLinear                           DLinear
input                   same window                       same window
output                  y_hat directly                    p_hat and mu_hat, combined
                                                          as mean_hat = p_hat * mu_hat
parameters (recorded)   5,856                             5,856
lookback / horizon      96 / 24                           96 / 24
split                   train [0,384] val [384,480]       identical
                          test [480,576]
optimizer               Adam                              Adam
learning rate           1e-3                              1e-3
weight decay            0.0                               0.0
max epochs              30                                30
patience                5                                 5
batch size              256                               256
checkpoint metric       validation realized-y MSE         validation realized-y MSE
                        (identical for every model)       (identical for every model)
trainer                 train_one, imported unchanged     train_one, imported unchanged
data seeds              (0, 1)                            (0, 1)
model seeds             (0, 1)                            (0, 1)
normalization           train split only                  train split only
```

The third model, `M2_HURDLE_ZTNB`, carries **5,857** parameters — one extra scalar,
+0.017%. The prereg's `PARAMETER_MATCH_RULE` states this is below its 1% rule so no
separate control was added, and the counts are reported. `M2` is the secondary
comparison; the primary is `M1 vs M0`.

## Checkpoint prohibitions, as written

```
checkpoint_prohibitions = ["oracle mean", "test results",
                           "per-model metric", "per-cell tuning"]
```

So the model selected for evaluation was never chosen using the oracle target, the
test split, a model-specific metric, or per-cell tuning. One setting for every cell
and every model.

## Verdict

```
same backbone                 yes
same input information        yes
same training budget          yes
same optimizer and schedule   yes
same split and seeds          yes
parameter matched             yes, 5,856 = 5,856 by construction
checkpoint rule identical     yes, and it is the realized-y metric, not the oracle
per-cell tuning               none

C1-G4  CONTROLLED_FAIRNESS_PASS
```

## Boundary that must still be stated in the paper

The comparison is fair **within one backbone family**. Both arms are DLinear. The
recovered source contains no second backbone, so every C1 claim remains conditional
on this family. That limitation was already recorded on the Windows side
(`WARN_FAIL.md` W10) and the recovery does not remove it.
