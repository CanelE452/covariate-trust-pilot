# Methods structure

Frozen 2026-08-12, after `notation_registry.md`. Hierarchy follows
`../paper_outline_verified/final_outline_freeze.md`; one refinement is proposed and
justified below.

---

## Hierarchy

```
3  Problem Setup and Forecasting Formulations
   3.1  Intermittent demand and what the standard descriptors retain     M-Q1
   3.2  Direct prediction of the conditional mean                        M-Q2
   3.3  Factorized prediction: occurrence probability x positive mean    M-Q2
   3.4  What a matched comparison measures, and the metric               M-Q3, M-Q8

4  Controlled Synthetic Study
   4.1  Generating process and marginal control                          M-Q4
   4.2  Experimental fairness, training protocol and evaluation target   M-Q3, M-Q9
   4.3  Stage 1: fixed-marginal factorial  -- design                     M-Q5
   4.4  Component-attribution diagnostic   -- design                     M-Q5
   4.5  Stage 2: stationary dependence sweep -- design                   M-Q6

5  Empirical Validation
   5.1  Datasets and evaluation protocol                                 M-Q7
   5.2  Operational definitions of H1, H2 and H3                         M-Q7
   5.3  Uncertainty, bootstrap and seeds                                 M-Q9
```

Results live in Sections 4.6–4.8 and 5.4–5.9 under the frozen outline's numbering; the
**design** subsections above contain no outcome.

### The one refinement proposed

`final_outline_freeze.md` places Stage 1 and Stage 2 **results** inside 4.3 and 4.5, i.e.
design and outcome share a subsection. That is workable in a journal format but it makes
the Methods/Results boundary impossible to audit mechanically, and it was the source of a
near-miss in drafting: a design sentence and a result sentence sitting in one paragraph.

**Proposal, minimal:** keep the outline's section numbers, but split each of 4.3 and 4.5
into a design half (Methods) and a result half (Results), and give the empirical
hypotheses their own design subsection (5.2) ahead of their results. No section is added
or removed; no figure or table moves.

```
outline freeze          this document
4.3 Stage 1             4.3 Stage 1 design      -> results reported in Section 4.6
4.5 Stage 2             4.5 Stage 2 design      -> results reported in Section 4.7
5.3/5.4/5.6 H1/H2/H3    5.2 operational defs    -> results reported in 5.4/5.5/5.6/5.7
```

This is a presentation change only. If the venue prefers the original merged form, the
prose recombines without editing a sentence.

---

## Reader chain, and where each question is answered

```
M-Q1  what is being predicted?                     3.1   the conditional mean of y_t
M-Q2  how do the two formulations differ?          3.2, 3.3
M-Q3  why is the comparison fair?                  3.4, 4.2
M-Q4  what is controlled and what is varied?       4.1
M-Q5  what does Stage 1 test?                      4.3, 4.4
M-Q6  how does Stage 2 extend it?                  4.5
M-Q7  how do H1/H2/H3 connect to the constructs?   5.1, 5.2
M-Q8  how is G defined?                            3.4
M-Q9  how is uncertainty computed?                 4.2, 5.3
```

Each subsection ends on the question the next one opens.

---

## What Methods must NOT contain

```
any G value, any win rate, any confidence interval on an outcome
"we find", "we show that", "as expected"
which formulation won, in any cell, on any dataset
any interpretation of the asymmetry between the two axes
the routing chain's outcome (its design belongs to the appendix, its result to 5.9)
```

Setup constants (5,856 parameters; 96/24; 384/480/576; 80 series; 18 cells; 2000
bootstrap draws) are **not** results and belong in Methods.

---

## Source coverage required before prose

Every constant below must resolve to an artifact. Coverage table:
`methods_claim_source_map.csv`.

```
5,856 = 5,856 parameters              point_hurdle_fairness.md
same trainer, imported unchanged      point_hurdle_fairness.md
Adam, lr 1e-3, wd 0.0                 point_hurdle_fairness.md
30 epochs, patience 5, batch 256      point_hurdle_fairness.md
checkpoint on validation realized-y   point_hurdle_fairness.md  (oracle/test forbidden)
train-split-only normalization        point_hurdle_fairness.md
split 384 / 480 / 576                 point_hurdle_fairness.md, dgp_verification.md
lookback 96, horizon 24               point_hurdle_fairness.md
gap support {d-1, d+1}, share 0.5     dgp_verification.md
lambda levels {5, 15}, mean 10 both   dgp_verification.md
burn-in max(128, 8d)                  dgp_verification.md
paired random numbers                 dgp_verification.md
forbidden factors list                dgp_verification.md
exact DP oracle                       dgp_verification.md
Stage 1: 8 cells, 80 series           stage1_stage2_verified.md
Stage 2: 18 cells, 80 series          stage1_stage2_verified.md
bootstrap 2000, paired, seed          metric_sign.md
M5 / Favorita protocol                table3_draft.md
```
