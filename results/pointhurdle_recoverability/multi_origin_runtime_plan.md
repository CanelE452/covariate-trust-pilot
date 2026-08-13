# Multi-origin runtime plan  (supersedes runtime_plan.md)

**Correction first.** `runtime_plan.md` stated that four origins are available under the
frozen split. That was wrong: it divided the 112 post-train periods by the stride without
removing the validation window.

```
m5         length 1941  train_end 1829  val_end 1857
           validation window  [1829, 1857)   28 periods, NOT a test origin
           test origins       1857, 1885, 1913        -> [1913, 1941) ends exactly at length
           remaining periods after the last origin: 0

favorita   length 1688  train_end 1576  val_end 1604
           validation window  [1576, 1604)   28 periods, NOT a test origin
           test origins       1604, 1632, 1660        -> [1660, 1688) ends exactly at length
           remaining periods after the last origin: 0
```

**[확인] Exactly three test origins exist on each dataset, and there is no fourth.** The
"add the 4th cached-protocol origin" item in `runtime_plan.md` is withdrawn.

---

## What can still be done without a new split

```
1  RE-SCORE the three existing origins with seed1 and seed2      NOT RUN in this pass
   checkpoints    seed_robustness/models/{seed1,seed2}_{point,hurdle}.pt, 31.7 KB each
                  provenance recorded with sha256; seed0 reuses rule_replication/models
   train cost     already paid: 32.1-44.0 s per model at training time
   what it buys   the series x origin x seed decomposition that R3 could not perform.
                  Today the seed audit is series-level only, so origin instability and
                  seed noise are NOT separated.
   scope          M5 only - seed1/seed2 checkpoints exist for M5, not for Favorita
   passes         2 seeds x 2 arms x 3 origins = 12 forward passes over 5,693 series
   entrypoint     seed_robustness.py already has _load(seed, role, device) and
                  cmd_evaluate; the missing piece is emitting per-origin rows instead of
                  the series-level delta it currently writes
   GPU time       NOT MEASURED.  Inference only, no gradient steps.  A 500-series smoke
                  is the minimum before quoting a number.
   storage        3 origins x 28 steps x 5,693 series x 2 seeds x 11 cols
                  ~957k rows, order 30-60 MB
```

## What requires a new split, and is therefore out of scope

```
2  8-12 origins on the same datasets
   train_end must move back by 5x28=140 (8 origins) or 9x28=252 (12 origins)
   m5 train_end 1829 -> 1689 or 1577;  favorita 1576 -> 1436 or 1324
   every arm retrains on less history, so the losses are NOT comparable to the cached
   ones and pre_analysis_spec.json no longer holds.  This is a different experiment and
   needs its own pre-registration.
   runs          2 datasets x 2 arms x N origins x 3 seeds = 96 (8 origins) or 144 (12)
   storage       order 1-2 GB
   GPU time      NOT ESTIMATED

3  FreshRetailNet-LT and UCI paired panels
   no paired Point/Hurdle per-series RMSE exists in this schema; both arms would have to
   be trained on both datasets under the frozen protocol.  Larger than item 2.
```

## Recommendation

Item 1 only, and only on approval. It is inference on existing checkpoints, it is the
cheapest artifact in the list, and it answers the one question this analysis could not:
whether the origin-to-origin winner instability is temporal variation or estimation noise.
Items 2 and 3 change the split or require new training and are outside the stated scope.
