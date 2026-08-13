# Integrity summary — what changed when the required corrections were applied

Seed 20260813. No model was trained. No existing file was modified.

---

## Three defects found in my own previous pass

### D1 · capture was measured against the wrong family oracle — CORRECTED

The previous pass divided **every** policy's improvement by the **hard-selection** origin
oracle, including the convex-combination policies. A convex policy can exceed the hard
oracle (hard selection is the `w ∈ {0,1}` special case of convex), which is why the
previous table reported `ORACLE origin convex ... 105.7%` capture — a number above 100%
should have been the tell.

Family-separated ladders are now in `oracle_family_ladders.csv`:

```
dataset   family        train_domain_global  series_oracle  origin_oracle  H_total  H_static  SRR_hard/convex
favorita  hard                     2.4097         2.3624         2.3330    3.18%     1.96%           0.617
favorita  convex                   2.4146         2.3559         2.3282    3.58%     2.43%           0.679
m5        hard                     1.6407         1.6129         1.5947    2.80%     1.70%           0.605
m5        convex                   1.6351         1.6096         1.5924    2.61%     1.56%           0.596
```

`SRR` is renamed **static recoverability ratio**, reported per family, never mixed.

**The two families do not share a global baseline.** The train-domain convex weight is
`w = 0.60` for Favorita-as-test and `w = 0.85` for M5-as-test, and applying the transferred
weight makes Favorita *worse* than always-Hurdle (2.4146 vs 2.4097) while making M5 *better*
(1.6351 vs 1.6407). Any single number that pools the families is meaningless.

### D2 · the fourth origin does not exist — CORRECTED

`runtime_plan.md` claimed four origins were available. It divided the 112 post-train
periods by the stride without removing the 28-period validation window. **Exactly three
test origins exist on each dataset**, and `[1913,1941)` / `[1660,1688)` end precisely at
the series length. The "add a fourth origin" recommendation is withdrawn; see
`multi_origin_runtime_plan.md`.

### D3 · η and normalization were chosen after seeing held-out results — CORRECTED

The previous pass reported `D3 exp-weighted η=8` as the best policy, but η and the
normalization were selected by comparing held-out outcomes. That is exactly the leakage
the protocol forbids ("training datasets에서만 탐색").

Re-run with the grid `η ∈ {0.5, 2, 8, 32} × norm ∈ {raw, per-series, per-origin}` scored
**on the training dataset only**, then applied once:

```
train     test      selected            rel_improve   95% CI            capture(convex)  vs 50:50
m5     -> favorita  eta=8, per_series      +0.537%   [+0.401, +0.677]        15.84%       +1.004%
favorita -> m5      eta=8, per_series      +0.558%   [+0.467, +0.658]        18.89%       -0.004%
```

**Both directions independently selected the same configuration**, and the honest numbers
are *better* than the leaked ones (+0.54/+0.56 against +0.36/+0.45). That is the opposite
of what I expected and is reported as found. **[추정]** the leaked selection over-fitted a
smaller grid; it is not evidence that leakage helps.

---

## Sensitivity that survives the corrections

### Normalization matters, and the margin is not comfortable

`exp_weight_normalization_sensitivity.csv`, held-out-selected variants, macro over the two
directions:

```
policy                                  macro_improve   worst(dataset x origin)   capture
D3 eta=8.0  per_series_scale                  +0.553%                  +0.018%    17.61%
D3 eta=8.0  raw                               +0.410%                  +0.018%    12.98%
D3 eta=8.0  per_origin_scale                  +0.397%                  +0.018%    12.63%
D3 eta=2.0  raw                               +0.357%                  +0.018%    11.46%
D3 eta=2.0  per_series_scale                  +0.337%                  +0.007%    11.06%
D3 eta=2.0  per_origin_scale                  +0.244%                  -0.228%     8.07%
```

Two things follow. The macro improvement varies by **2.3×** across the grid, so the choice
is not incidental. And when the unit is `dataset × origin` rather than `dataset`, the
**worst cell falls to about +0.02%** — on at least one origin the policy is indistinguishable
from the global choice, even for the best configuration.

### 50:50 is not a safe default, and per-origin it is worse than reported

```
P3 50:50 average    macro +0.042%    worst (dataset x origin)  -0.748%
                    m5  +0.561% [+0.432, +0.694]   favorita  -0.476% [-0.710, -0.257]
```

Both dataset-level intervals exclude zero **in opposite directions**.

### Why the practical label is fragile

`seed_margin_flip.csv`:

```
dataset    within 0.5pp of the +-2% boundary   within 2pp   |G| <= 1
m5                                    11.61%       44.75%     12.49%
favorita                               8.52%       33.71%      9.02%
```

Nearly half of M5 rows sit within 2 pp of the threshold. That is the mechanism behind the
36.6% cross-seed label flip observed at series level with only 0.10% seed variance:
**the underlying quantity is stable, the discretized label is not.** [추정]

---

## Bootstrap semantics — stated explicitly

All intervals resample **series clusters**, carrying all three origins of a chosen series
together, 2,000 draws. They therefore express **series uncertainty conditional on the three
observed origins**, and they do **not** express origin uncertainty: with three origins there
is no basis for a distribution over origins. Any statement about how the policy would
behave at other origins is outside what these intervals cover.

---

## Not done in this pass

```
seed_origin_panel.parquet    NOT PRODUCED.  Re-scoring the three origins with the seed1
                             and seed2 checkpoints was in scope but not executed: it
                             requires emitting per-origin rows from a code path that
                             currently writes series-level deltas, and the scope forbade
                             starting a run without a cost report.  Cost and entrypoint
                             are in multi_origin_runtime_plan.md item 1.
                             CONSEQUENCE: origin instability and seed noise remain
                             unseparated, which is the single largest open question here.
```
