# Contributions — draft v3

Scientific content is unchanged from v1 and v2; wording follows
`paper_synthesis_verified/claim_ledger_frozen.md`. C3 is deliberately the shortest and
is not expanded.

---

## As a list, for the end of the Introduction

**A controlled characterization of the finite-sample relative inductive bias of direct
and factorized forecasting under temporal occurrence and magnitude dependence.**
Holding the gap support, the average interval and the positive-demand marginal fixed
and varying only temporal order, we show that occurrence structure and magnitude
structure interact, and that a signed dependence sweep resolves that interaction into
an asymmetric geometry: occurrence effects are primarily associated with the strength
of dependence, magnitude effects primarily with its direction. Both representations are
matched on backbone, parameter count, training budget and evaluation target, so the
comparison concerns inductive bias under a fixed budget rather than representational
capacity.

**An empirical transfer result with an explicit boundary.** On two public retail
datasets the occurrence relationship appears as an empirical analogue, and a descriptor
rule encoding the configuration identified in the controlled sweep, frozen before use,
shows predictive transfer on an independent population and across three training seeds.
The isolated mechanism does not follow: after adjusting for overlap in scale, sparsity,
positive-demand variability and occurrence dependence, the association is no longer
distinguishable from zero. We report predictive transfer and mechanism replication as
two separate results, and note that the synthetic axes are not cleanly separable in
observed demand.

**An adaptive-use boundary.** The two forecasts make complementary errors and that
complementarity can be enlarged deliberately, but a learned router over it did not
transfer robustly across domains or across time, and a pre-registered stopping rule was
triggered rather than the search continued.

---

## As a paragraph, if the venue prefers prose

This paper contributes a controlled characterization of the finite-sample relative
inductive bias of direct and factorized intermittent-demand forecasting under temporal
occurrence and magnitude dependence, showing that the two axes interact and that they
respond to different features of dependence — occurrence effects associated primarily
with its strength, magnitude effects primarily with its direction. It then establishes
how far that characterization carries: the occurrence relationship appears as an
empirical analogue on two public datasets, and a rule frozen from the controlled study
selects unseen series toward direct prediction, while the isolated association does not
survive adjustment for overlap in scale and sparsity. Finally it reports the boundary of
adaptive use, where a measurable complementarity did not yield a router that transfers
across domains or time.

---

## Wording constraints carried from the frozen ledger

```
required     "finite-sample";  "primarily associated with";  "empirical analogue";
             "predictive transfer";  every statement about the direct-favourable
             configuration scoped to the eighteen-cell sweep
avoid        any verb attributing the boundary to scale as a cause; use
             "not separable", "entangled with", "does not survive adjustment"
forbidden    "proves";  "universally";  "first";  "novel correlation effect";
             "establishes a causal mechanism";  "the synthetic mechanism replicates";
             "Hurdle is superior for intermittent demand";  "Point is superior in
             sparse demand"
scope        one backbone family;  one parameter budget;  point metrics only;  the
             controlled design has no scale axis
```
