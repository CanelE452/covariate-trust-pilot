# Table specification

Four main tables. Every gate-development variant stays out of the main text.

---

## Table 1 — Controlled comparison: design and fairness

```
section     4.1 - 4.2
placement   main
purpose     let a reviewer confirm in one glance that the manipulation is clean and
            the comparison is matched
```

Two blocks in one table.

**Block A, the design.**
```
occurrence      Q_j in {d-1, d+1}, long-run share 0.5 each
                structured: alternating;  control: iid
magnitude       M_j = 1 + Poisson(lambda_j - 1), lambda_j in {5, 15}
                long-run positive mean 10 in both arms
sparsity        d in {4, 8}
Stage 1 grid    2 x 2 x 2 = 8 cells, 80 series per cell
Stage 2 grid    d x rho_I x rho_M = 2 x 3 x 3 = 18 cells, 80 series per cell
length/split    576, train [0,384] val [384,480] test [480,576]
lookback/horizon 96 / 24
seeds           data (0,1) x model (0,1); model seeds averaged within a series
forbidden       trend, seasonality, hidden regime, heavy tail, test-time shift,
                phase jitter, interval-magnitude cross-correlation
```

**Block B, the two estimators.**
```
                        direct                  factorized
parameters              5,856                   5,856
backbone                DLinear                 DLinear
optimizer / lr          Adam / 1e-3             Adam / 1e-3
epochs / patience       30 / 5                  30 / 5
batch                   256                     256
checkpoint              validation realized-y MSE, identical for both
per-cell tuning         prohibited              prohibited
target for scoring      exact DP conditional mean
```

Source: `dgp_verification.md`, `point_hurdle_fairness.md`, prereg blocks.

---

## Table 2 — Stage 1 factorial effects

```
section     4.3
placement   main
metric      G = 100(1 - RMSE_Hurdle / RMSE_Point), percentage points
uncertainty paired series bootstrap, 2000 draws
source      results/synthetic_source_verification/stage1_verified_contrasts.csv
```

```
contrast                 effect (pp)      95% CI            excludes 0
interval_dependence         +7.83     [ +6.09,  +9.45]        yes
magnitude_dependence        -4.58     [ -6.28,  -2.92]        yes
sparsity                    -6.26     [ -8.01,  -4.57]        yes
sparsity x interval         +3.35     [ +1.62,  +5.09]        yes
sparsity x magnitude        -0.01     [ -1.70,  +1.69]         no
interval x magnitude       -16.74     [-18.45, -15.05]        yes
three_way                   -1.96     [ -3.64,  -0.31]        yes
```

A second column block may carry the ZTNB variant, whose signs agree on every contrast
that excludes zero. The per-cell table (C01–C08 with intervals) goes to Appendix E,
including the fact that **C08 is −3.01 pp with the interval touching zero**, i.e.
Stage 1 has no clearly direct-favourable cell.

---

## Table 3 — Core empirical validation datasets and protocol

```
section     5.1
placement   main
scope       M5 and Favorita ONLY
```

```
dataset     series      T    train_end   test origins      availability          sampling
M5           1,200   1,941      1,829    1857/1885/1913    sell_prices-derived   SBC-balanced,
                                                                                  300 per regime
Favorita     1,200   1,688      1,576    1604/1632/1660    raw                    SBC-balanced,
                                                                                  300 per regime
```

Plus: lookback 96, horizon 28, eligibility `n_positive_train >= 20`, spec frozen at
18:06:38 with results written at 18:11:16, and the note that the sample is
regime-balanced against a full M5 pool of 23,053 / 5,942 / 984 / 496.

**FreshRetailNet-LT and UCI Online Retail II are deliberately excluded from this
table.** They are domain- and time-transfer stress tests used only in Section 5.7, and
listing them here would let a reader believe they contributed to H1 or H2. Their
protocol, including UCI's `AVAILABILITY_UNKNOWN` status, goes to the appendix dataset
table, and Section 5.7 refers to it in one sentence.

Source: `stage_a_results.json`, `docs/m5_favorita_data_derivation.md`.
Appendix source: `multi_benchmark/dataset_audit.json`,
`multi_benchmark/external_benchmark.json`.

---

## Table 4 — Empirical results and the classical baselines

```
section     5.2 - 5.6
placement   main
```

**Block A, the two representations.**
```
                        M5                          Favorita
RMSE  direct/factorized 2.9209 / 2.9215             5.1124 / 5.1443
MAE   direct/factorized 2.2890 / 2.1797             3.3017 / 3.1660
mean delta              -0.00066                    -0.03197
factorized win %        47.17                       49.25
```

**Block B, classical baselines by mean rank, lower is better.**
```
M5        SBA 3.152 < Croston 3.260 < TSB 3.411 < SES 3.483
          < direct 4.202 < factorized 4.220 < naive 6.820 < seasonal naive 7.453
Favorita  SBA 3.069 < Croston 3.787 < TSB 3.797 < SES 3.940
          < direct 3.947 < factorized 4.032 < naive 6.306 < seasonal naive 7.123
```
Footnote: iETS not run — no verified implementation in the environment, and it was
deliberately not written from scratch.

**Block C, the three hypotheses.**
```
claim                    derived    tested             effect                    status
H1 occurrence            Stage 2    M5, Favorita       +0.1065 / +0.0789         analogue, supported
H1 in intermittent       Stage 2    M5 SBC subset      +0.153 (relative)         supported
H1 in lumpy              Stage 2    M5 SBC subset      +0.014, CI spans 0        not supported
H2 as selector           Stage 2    independent M5     -0.0230, +11.87 pp        confirmed
H2 as mechanism          Stage 2    overlap-weighted   +0.0032, CI spans 0       not replicated
H3 sparsity interaction  Stage 1+2  M5, Favorita       -0.0305 / -0.0428         not replicated at
                                                                                  the tested contrast
```

Source: `stage_a_results.json`, `classical_benchmark/benchmark.json`,
`regime_h1.json`, `rule_replication/*.json`.

---

## Appendix tables

```
A0  stress-test datasets: FreshRetailNet-LT and UCI Online Retail II, full protocol,
    including UCI AVAILABILITY_UNKNOWN.  Referenced from Section 5.7.
A   H1 eligibility-threshold sensitivity (15 / 20 / 30) and the adjusted partial
    association, tagged exploratory
B   occurrence-gate Brier skill against a constant per-series rate
C   H3 in full, with the ADI-median versus ADI 4-vs-8 discrepancy and the available
    support (M5 127 / 52, Favorita 84 / 45)
D   H2 three-seed table; both Favorita analyses side by side
E   Stage 1 full per-cell table with intervals and the component-attribution columns
F   Stage 2 full 18-cell table with intervals
G   Stage 1 validity audit and the no-signal control evidence
H   routing chain: Gate-v1/v2/v3, P0L1, Safe-P0L1, HGB, sequence gate
J   pre-registration documents, freeze timestamps, artifact hashes
K   the three delta conventions and why the paper reports G
```

Four main tables against eleven appendix tables: the paper argues from four and is
auditable from fifteen.  No gate-development variant appears in the main text.
