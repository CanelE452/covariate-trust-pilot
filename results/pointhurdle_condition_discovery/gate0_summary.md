# Gate 0 — is there any selection headroom between Point and Hurdle on real data?

Seed 20260813. No model was retrained; every number is computed from cached predictions.

---

## Panel

```
source        rule_replication/independent_raw_predictions.parquet          (M5)
              favorita_independent/independent_raw_predictions.parquet      (Favorita)
unit          dataset x series x outer_origin
rows          33,294        series 11,098        origins 3 per series
paired_valid  100.00%       loss_point == 0: 0 rows
sample        independent eligible population (M5 5,693; Favorita 5,405)
              NOT the SBC regime-balanced 300-per-regime Stage A sample
metric        RMSE of each arm against realized y on the 28-step test window,
              per series x origin.  Sign convention: G = 100(1 - RMSE_H / RMSE_P),
              G > 0 favours Hurdle.  This is the estimand already fixed in
              external_validity_screen/pre_analysis_spec.json.
```

## C1 / C2 — static models, oracles and headroom

```
dataset    always_point  always_hurdle  global   test_static_best  series_oracle  origin_oracle
favorita        2.4849         2.4097   hurdle             2.4097         2.3624         2.3330
m5              1.6493         1.6407   hurdle             1.6407         1.6129         1.5947
MACRO           2.0563         2.0152        -             2.0152         1.9779         1.9543

headroom vs test-static-best      m5 2.804%   [2.668, 2.945]      favorita 3.183%  [3.034, 3.349]
headroom vs train-domain global   m5 2.804%                       favorita 3.183%
SERIES-oracle headroom            m5 1.696%                       favorita 1.963%
```

95% intervals are cluster bootstraps that resample series and carry all three origins of a
series together, 2,000 draws.

**The two oracles differ, and the difference is the most important number here.** The
origin-level oracle recovers 2.8–3.2%, but a *series*-level oracle — one choice per series,
which is the most any train-only selector can express — recovers only 1.7–2.0%. Roughly
**40% of the apparent headroom is origin-level variation that no series-level rule can
capture.**

## C3 — winner distribution, practical threshold τ = 2%

```
dataset    rows   series  point%  neutral%  hurdle%  point_rows  hurdle_rows  series_consistent%
favorita  16,215   5,405   33.94     17.82    48.23       5,504        7,821              25.86
m5        17,079   5,693   33.75     24.66    41.59       5,765        7,103              18.07
MACRO     33,294  11,098   33.85     21.33    44.82      11,269       14,924              21.87

tau sensitivity        point%   neutral%   hurdle%    point_rows   hurdle_rows
  1%                    39.63      10.80     49.57        13,193        16,505
  2%  (primary)         33.85      21.33     44.82        11,269        14,924
  5%                    19.31      46.66     34.04         6,428        11,332
```

Winner stability across the three origins of the same series:

```
dataset    rho(o1,o2)  rho(o1,o3)  rho(o2,o3)   all-three-same-sign %
favorita        0.383       0.322       0.360                   35.2
m5              0.184       0.123       0.086                   28.7
```

Only 18–26% of series have the same practical winner at all three origins, and the
cross-origin rank correlation of `G` is 0.09–0.38. The winner is real but noisy.

---

## Gate 0 verdict

```
criterion                                             required        observed        pass
paired coverage                                       >= 95%          100.00%          yes
origin-oracle headroom vs test-static-best            >= 2%           2.80 / 3.18%     yes
Point practical win share                             >= 10%          33.85%           yes
Hurdle practical win share                            >= 10%          44.82%           yes
Point practical rows                                  >= 100          11,269           yes
Hurdle practical rows                                 >= 100          14,924           yes
both observed in at least 2 datasets                  yes             yes              yes
```

**GATE 0 = GREEN.**

There is real headroom and both models genuinely win on substantial subsets. Proceed to the
condition-feature pilot.

**Carried forward as a constraint, not a caveat:** the series-level ceiling is 1.7–2.0%,
not 2.8–3.2%, and only ~22% of series have a stable winner. Any train-only selector is
competing for the smaller number.
