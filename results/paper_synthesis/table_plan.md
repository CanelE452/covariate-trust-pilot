# Table plan

Four main tables. Every gate development variant stays out of the main text.

---

## Table 1 — Datasets and experimental setup

**Contents.** Per dataset: series count, series definition, span, train/val/test boundaries,
lookback 96, horizon 28, number of test origins, eligibility rule
(`n_positive_train >= 20`), and the availability handling.

```
dataset          series    T      train_end  test origins        availability
M5                1,200   1,941      1,829   1857/1885/1913      sell_prices-derived
Favorita          1,200   1,688      1,576   1604/1632/1660      raw
FreshRetailNet    7,528     770        658   686/714/742         explicit observed mask
UCI Online Retail 2,158     739        627    655/683/711        AVAILABILITY_UNKNOWN
```

The UCI row must carry its `AVAILABILITY_UNKNOWN` flag in the table, not only in the text.

**Source.** `stage_a_results.json` manifest, `multi_benchmark/dataset_audit.json`,
`multi_benchmark/external_benchmark.json`, `docs/m5_favorita_data_derivation.md`.

---

## Table 2 — Point versus Hurdle on real data

**Contents.** Overall RMSE and MAE for both representations on M5 and Favorita, mean and median
delta, win rates; then the three pre-registered hypotheses with effect and CI.

```
                     M5                          Favorita
RMSE  Point / Hurdle  2.9209 / 2.9215            5.1124 / 5.1443
MAE   Point / Hurdle  2.2890 / 2.1797            3.3017 / 3.1660
mean delta            -0.00066                   -0.03197
Hurdle win %          47.17                      49.25
H1 spearman           +0.1065 [+0.0437,+0.1652]  +0.0789 [+0.0205,+0.1405]
H2 screen difference  -0.0303 [-0.0954,+0.0588]  -0.0224 [-0.1618,+0.1335]
H3 difference         -0.0305 [-0.1418,+0.0912]  -0.0428 [-0.1587,+0.0704]
```

**Source.** `stage_a_results.json`.

---

## Table 3 — Classical intermittent-demand benchmark

**Contents.** All eight methods on both datasets: overall RMSE, overall MAE, mean per-series
RMSE, mean rank, win percentage. Sorted by mean rank so the reader sees immediately that SBA
leads and both neural variants trail.

```
M5 mean rank      SBA 3.152 < Croston 3.260 < TSB 3.411 < SES 3.483
                  < dlinear_point 4.202 < dlinear_hurdle 4.220
                  < naive 6.820 < seasonal_naive 7.453
Favorita          SBA 3.069 < Croston 3.787 < TSB 3.797 < SES 3.940
                  < dlinear_point 3.947 < dlinear_hurdle 4.032
                  < naive 6.306 < seasonal_naive 7.123
```

A footnote must record that iETS was not run — `IETS_NOT_AVAILABLE`, no verified
implementation in the environment, and it was deliberately not written from scratch.

**Source.** `classical_benchmark/benchmark.json`, `classical_benchmark/implementation_audit.json`.

---

## Table 4 — Claim replication and boundary summary

**Contents.** One row per claim: the claim, where it was derived, where it was tested, the
effect, whether the interval excludes zero, and the final status. This is the table a sceptical
reader will read first and it is where the non-replications live in plain sight.

```
claim                       derived    tested            effect                 status
H1 occurrence dependence    synthetic  M5, Favorita      +0.1065 / +0.0789      SUPPORTED_WITH_BOUNDARY
H1 in intermittent regime   synthetic  M5 SBC subset     +0.153 (relative)      SUPPORTED
H1 in lumpy regime          synthetic  M5 SBC subset     +0.014, CI spans 0     NOT_SUPPORTED
H2 as selector              synthetic  independent M5    -0.0230, +11.87 pp     CONFIRMED
H2 as mechanism             synthetic  overlap-weighted  +0.0032, CI spans 0    NOT_REPLICATED
H3 sparsity interaction     synthetic  M5, Favorita      -0.0305 / -0.0428      NOT_REPLICATED
occurrence gate skill       synthetic  M5, Favorita      BSS -0.008 / -0.091    REJECTED on real data
routing improves forecasts  development  external TEST   -2.43%                 NOT_REPLICATED
raw history carries signal  -          FreshRetailNet    +2.648%                SUPPORTED (1 dataset)
routing generalizes         -          UCI               -193.9%                NOT_SUPPORTED
```

**Source.** `results/paper_synthesis/claim_ledger.md` and every artifact it cites.

---

## Appendix tables (not main)

Gate-v1 kill test, Gate-v2 selection and OOF, Gate-v2 fresh-holdout, expert diversity pairwise
matrix, Gate-v3 2×2 factorial effects, P0L1 per-fold table, Safe-P0L1 per-fold and tail table,
HGB per-dataset table, sequence-gate per-fold table, H1 threshold sensitivity, H2 three-seed
table, both Favorita transfer analyses.

That is twelve appendix tables against four main ones, which is the intended ratio: the paper
argues from four, and is auditable from sixteen.
