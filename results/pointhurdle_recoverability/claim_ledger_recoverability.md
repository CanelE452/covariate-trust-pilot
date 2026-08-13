# Claim ledger — recoverability

Seed 20260813. Two datasets, three origins, one model seed for the panel; three model
seeds for the seed audit. No model was trained. No existing file was modified.

---

## R1 · Complementarity exists

```
claim              Neither formulation wins everywhere on real intermittent demand.
estimand           practical winner share at tau = 2%, over series x origin rows
artifact           multi_origin_paired_panel.parquet; reproduction_check.csv
observed           Point 33.85%, neutral 21.33%, Hurdle 44.82%
                   Point rows 11,269 · Hurdle rows 14,924 · 11,098 series · 100% paired
status             SUPPORTED
allowed wording    "both formulations have substantial practical win regions"
forbidden wording  "the winner is predictable"; any implication that share = recoverable value
```

## R2 · Most headroom is series-fixed, but a large minority is not

```
claim              About 60% of the oracle headroom is expressible as one fixed choice per
                   series; the remaining ~40% requires changing the choice per origin.
estimand           H_static / H_total and H_dynamic / H_total
artifact           recoverability_decomposition.csv
observed           m5        H_total 2.80%  H_static 1.70%  H_dynamic 1.11%  R_static 0.605
                   favorita  H_total 3.18%  H_static 1.96%  H_dynamic 1.22%  R_static 0.617
status             SUPPORTED, two datasets, consistent to within 0.012 in R_static
allowed wording    "roughly three fifths of the headroom is series-fixed"
forbidden wording  "the series oracle is the ceiling for any adaptive policy" - it is not;
                   the origin convex oracle beats the series hard oracle
```

## R3 · The winner is unstable across origins, and the label is fragile across seeds

```
claim              Relative advantage is largely origin-specific, and near the neutral band
                   the winner label also flips with the training seed.
estimand           same-winner-all-origins share; gain lag-1 Spearman; cross-seed label flip
artifact           winner_stability.csv; variance_components.csv; seed_gate.json
observed           same winner at all 3 origins   m5 18.1%   favorita 25.9%
                   pairwise agreement             m5 39.2%   favorita 45.8%
                   gain lag-1 Spearman            m5 0.135   favorita 0.372
                   seed variance share            0.10%  (series 97.3%, residual 2.6%)
                   seed / residual ratio          0.037   (gate threshold 0.20)  PASS
                   practical winner flips across seeds   36.62% of series   FAILS the 20%
                                                                            gate criterion
status             SPLIT.  Variance-based seed gate PASSES; label-flip gate FAILS.
allowed wording    "the winner label is fragile: seed variance is negligible (0.10%) yet
                    36.6% of series change practical winner across seeds, because many
                    series sit near the +-2% neutral boundary"
forbidden wording  "origin instability is temporal variation" - not separated;
                   "seed noise explains the instability" - seed variance is 0.10%
```

## R4 · Hard selection recovers almost nothing; soft combination recovers some

```
claim              Under a feasible sequential protocol on an unseen dataset, hard
                   selection is near-worthless while dynamic soft combination improves in
                   both transfer directions.
estimand           relative loss improvement over the train-domain global choice
artifact           policy_macro_results.csv; policy_bootstrap_dynamic.csv
observed (macro over the two directions, worst direction in brackets)
                   D3 exp-weighted avg eta=8    +0.405%  [+0.357%]   capture 13.7%
                   D2 rolling convex K=2        +0.296%  [+0.254%]   capture 10.0%
                   P4 train-domain convex        +0.343%  [-0.203%]  direction-dependent
                   P3 50:50 average             +0.043%  [-0.475%]   direction-dependent
                   D1 discounted-loss selector  +0.030%  [-0.050%]   hard selection
                   P0 always Point              -1.822%  [-3.121%]
                   ORACLE origin hard           +2.994%  (diagnostic, not deployable)
                   ORACLE origin convex         +3.166%  (diagnostic, not deployable)
                   D3 bootstrap  m5->fav +0.359% [+0.201, +0.516]  capture 11.2% [6.5, 15.7]
                                 fav->m5 +0.451% [+0.350, +0.555]  capture 16.1% [12.8, 19.3]
status             SUPPORTED as a two-domain pilot only
allowed wording    "a time-varying convex combination improved on the global choice in both
                    directions, with cluster-bootstrap intervals excluding zero"
forbidden wording  "external validation"; "method contribution" - only two datasets, and
                   3 of 4 held-out datasets cannot be checked
```

## R5 · 50:50 is not a safe default here

```
claim              The equal-weight average, the primary baseline every complex strategy
                   must beat, helps on one dataset and hurts on the other.
estimand           relative improvement of the 50:50 forecast average over global choice
artifact           policy_bootstrap.csv
observed           m5       +0.561%  [+0.432, +0.694]   CI excludes zero, positive
                   favorita -0.476%  [-0.710, -0.257]   CI excludes zero, NEGATIVE
                   held-out-best convex weight   m5 w=0.60   favorita w=0.85
status             SUPPORTED
allowed wording    "the best fixed combination weight differs materially by dataset
                    (0.60 vs 0.85), so a transferred fixed weight can hurt"
forbidden wording  "simple averaging is a robust default"
```

## R6 · Not claimed

```
NOT claimed   that any static feature selector works.  The earlier pilot found the
              component-forecastability ablation inseparable from its shuffled control.
NOT claimed   that origin instability is temporal rather than estimation noise.  The two
              are not separated: the seed audit is series-level, so a series x origin x
              seed decomposition does not exist.
NOT claimed   external validity.  FreshRetailNet-LT and UCI carry no paired Point/Hurdle
              RMSE in this schema.
NOT claimed   any inventory or newsvendor result.  No verified implementation was found.
```
