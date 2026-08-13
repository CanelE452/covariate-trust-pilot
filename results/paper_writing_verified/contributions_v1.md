# Contributions — draft v1

Wording follows `paper_synthesis_verified/claim_ledger_frozen.md`. C3 is deliberately
the shortest.

---

## As a list, for the end of the Introduction

**A controlled characterization of the finite-sample relative inductive bias of direct
and factorized forecasting under temporal occurrence and magnitude dependence.**
Holding the gap support, the mean interval and the positive-demand marginal fixed and
varying only temporal order, we show that the relative behaviour of the two
representations changes substantially and that the occurrence and magnitude axes
interact. A signed dependence sweep resolves that interaction into an asymmetry:
occurrence effects are primarily associated with the strength of dependence, magnitude
effects primarily with its direction. Both representations are matched on backbone,
parameter count, training budget and evaluation target, so the comparison is of
inductive bias under a fixed budget rather than of representational capacity.

**An empirical transfer result with an explicit boundary.** On two public retail
datasets the occurrence relationship appears as an empirical analogue, and a rule
encoding the synthetic direct-favourable configuration, frozen before use, shows
predictive transfer on an independent population and across three training seeds.
The isolated mechanism does not follow: once sparsity, variability, scale and
occurrence dependence are balanced, the association is not distinguishable from zero.
We report predictive transfer and mechanism replication as two results, and identify
scale — absent from the controlled design — as the covariate that separates them.

**An adaptive-use boundary.** Complementarity between the two representations is real
and can be enlarged on purpose, but a learned router over it did not transfer robustly
across domains or across time, and a pre-registered stopping rule was triggered rather
than the search continued.

---

## As a paragraph, if the venue prefers prose

This paper contributes a controlled characterization of the finite-sample relative
inductive bias of direct and factorized intermittent-demand forecasting under temporal
occurrence and magnitude dependence, showing that the two axes interact and that they
have different response geometry — occurrence effects associated primarily with the
strength of dependence, magnitude effects primarily with its direction. It then
establishes how far that characterization transfers: the occurrence relationship
appears as an empirical analogue on two public datasets, and a frozen rule derived from
the controlled study selects unseen series toward direct prediction, while the isolated
mechanism does not survive balancing on scale and sparsity. Finally it reports the
boundary of adaptive use, where a measurable complementarity did not yield a router
that transfers across domains or time.

---

## Wording constraints carried from the frozen ledger

```
required     "finite-sample"; "primarily associated with"; "empirical analogue";
             "predictive transfer"; scope every statement about the
             direct-favourable configuration to the eighteen-cell sweep
forbidden    "proves"; "universally"; "first"; "novel correlation effect";
             "establishes a causal mechanism"; "the synthetic mechanism replicates";
             "Hurdle is superior for intermittent demand"; "Point is superior in
             sparse demand"
scope        one backbone family; one parameter budget; point metrics only; the
             controlled design has no scale axis
```
