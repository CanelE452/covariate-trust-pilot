# Runtime plan — what a 12-origin full run would cost

Nothing below was executed. Figures are derived from the cached artifacts and the frozen
split definitions, not from a timing run.

---

## Available origins under the frozen protocol

```
                length  train_end  horizon  stride  origins now  max origins at stride 28
m5               1,941      1,829       28      28            3   (1941-1829)/28 = 4
favorita         1,688      1,576       28      28            3   (1688-1576)/28 = 4
```

**[확인] The target of 12 origins is not reachable without moving `train_end` backwards.**
The frozen split leaves only 112 post-train periods on both datasets, which is four
non-overlapping 28-step windows. Three are cached; a fourth exists.

To reach 8-12 origins the training window must be shortened:

```
origins needed   train_end must move back by      m5 train_end      favorita train_end
8                5 x 28 = 140                     1,689             1,436
12               9 x 28 = 252                     1,577             1,324
```

That is a different experiment, not an extension of this one: every model would be
retrained on less history, so the losses are not comparable to the cached ones and the
frozen `pre_analysis_spec.json` split would no longer hold.

## Cost if that experiment were run

```
scope            2 datasets x 2 arms x 12 origins x 3 seeds
runs             144 training runs (each origin needs its own expanding-window fit)
series per run   m5 5,693 · favorita 5,405 (natural eligible populations)
storage          12 origins x 28 steps x 11,098 series x 11 columns
                 ~3.7M rows per seed per arm; roughly 12x the current 932k-row parquet,
                 order 1-2 GB for the full grid
GPU time         NOT ESTIMATED.  No timing artifact exists for a single origin fit on the
                 natural population; the cached runs record no wall-clock.  A smoke of
                 500 series x 1 origin x 1 arm is the minimum needed before quoting a number.
```

## Cheaper alternatives, in order of cost

```
1  add the 4th cached-protocol origin        no retraining if predictions can be scored
   (m5 1941, favorita 1688)                  from the existing checkpoints; 1 scoring pass
                                             per dataset per arm.  Raises origins 3 -> 4.
2  seed x origin decomposition               re-score the 3 existing origins with the two
                                             additional seeds already on disk
                                             (seed_robustness/models/seed1,seed2).
                                             4 scoring passes.  This is what would separate
                                             origin instability from seed noise, which is
                                             the open question in R3.
3  8-origin shortened-train experiment       96 training runs, new split, not comparable
                                             to the frozen results.
```

**Recommendation.** Item 2 is the highest value per unit cost: it is inference only, the
checkpoints exist, and it directly resolves the one decomposition this analysis could not
perform. Item 1 is nearly free and adds a fourth origin. Item 3 should not be run without a
separate pre-registration, because it changes the split.
