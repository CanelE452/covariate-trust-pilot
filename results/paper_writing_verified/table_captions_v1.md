# Table captions — draft v1

---

> **Table 1. Matched experimental conditions.** Every dimension the comparison does not
> intend to vary, for the direct and factorized arms of the controlled study. Parameter
> counts are 5,856 in each arm by construction; the secondary zero-truncated
> negative-binomial arm carries 5,857, a difference of 0.017%, inside the pre-registered 1%
> parameter-match rule. The checkpoint criterion is the validation realized-`y` mean squared
> error and is identical for every model and cell; selection on the oracle target, on test
> results, on a model-specific metric or by per-cell tuning is prohibited.

> **Table 2. Stage 1 factorial contrasts.** Effects on `G` in percentage points with 95%
> paired-bootstrap intervals, factorized against direct, over the eight-cell fixed-marginal
> factorial. Positive values favour the factorized arm. The structured arms of this design
> are deterministic period-2 alternation, so these contrasts establish that ordering matters
> and that the two axes interact, and do not measure a graded dependence axis.

> **Table 3. Core empirical validation datasets.** M5 and Favorita only. Series counts,
> lengths, training cut-offs, evaluation origins and the lookback/horizon used. The sample is
> balanced at 300 series per SBC regime class, subject to at least 20 positive training
> observations, and is therefore balanced rather than representative of either catalogue.
> FreshRetailNet-LT and UCI Online Retail II are domain- and time-transfer stress tests used
> only in Section 5.9; their protocol is in the appendix.

> **Table 4. Empirical evidence summary.** For each hypothesis: population, point estimate,
> 95% interval, status and the reading the status licenses. H1 is an empirical analogue
> rather than a replication. H2 is reported as two claims — the frozen selector transfers,
> and the isolated mechanism does not survive overlap adjustment. H3 is a non-replication at
> the pre-registered split and a construct mismatch, since the synthetic contrast is a mean
> interval of 4 against 8 while the external split is at the ADI median.

---

## Appendix tables

```
A0  appendix dataset table: FreshRetailNet-LT and UCI Online Retail II protocol,
    including AVAILABILITY_UNKNOWN
A1  classical baseline mean ranks, both datasets
A2  routing chain results, including the UCI outcome
A3  Stage 1 per-cell G and the component-attribution columns
A4  Stage 2 per-cell G with intervals, all 18 cells
```

## Caption discipline check

```
new scientific claim introduced in a caption          0
number in a caption absent from results_number_source_map.csv   0
"significant" used without naming the interval        0
Figure 2B/2C spread described as a confidence interval 0  (explicitly denied)
routing shown in a main figure                        0
```
