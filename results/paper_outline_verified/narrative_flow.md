# Narrative flow

Frozen 2026-08-11 after outline review. Supersedes the first draft of this file.

## The paper in one paragraph

Intermittent-demand series are routinely summarized by ADI and CV², statistics of the
marginal distribution that discard the order in which zeros and positive values
arrive. Two series can share those summaries and still differ in whether occurrence
intervals or positive magnitudes are temporally structured. We ask whether that
discarded structure changes which representation of the conditional mean a forecaster
should use — direct prediction, or a factorization into occurrence probability times
positive magnitude. Holding the marginals fixed and varying only the temporal
ordering, the finite-sample relative behaviour of the two representations moves
substantially, and the two axes interact. Parameterizing the dependence directly then
resolves that interaction into an asymmetry: occurrence effects are primarily
associated with the *strength* of dependence, magnitude effects primarily with its
*direction*. Within the eighteen-cell sweep, only one configuration shows
statistically clear superiority for direct prediction — sparse,
occurrence-unpredictable, magnitude-persistent. On two public retail datasets the
occurrence-dependence relationship appears as an empirical analogue, and a rule frozen
from that configuration shows predictive transfer on unseen series; its isolated
mechanism, however, does not survive overlap adjustment, and an attempt to exploit the
conditional advantage with a learned router does not transfer across domains. The
contribution is a controlled characterization of *when* factorization helps or hurts,
together with an explicit account of how far that characterization carries.

---

## Reader chain, with the section that carries each step

```
step                                                      section
────────────────────────────────────────────────────────────────────────
1  ADI/CV2 summarize the marginal and discard ordering      3.1
2  the two representations differ in what structure         3.2
   each can express, under one budget
3  what a relative finite-sample comparison means           3.3
4  under fixed marginals, relative behaviour changes,       4.3
   and the axes interact
5  component substitution says where the error sits         4.4
6  parameterizing dependence resolves the interaction       4.5
7  the resolution is an asymmetry                           4.6
8  the asymmetry implies one clearly direct-favourable      4.7
   configuration within the sweep
9  does an analogue appear in real demand?                  5.3
10 does the frozen configuration select real series?        5.4
11 does its mechanism survive adjustment?                   5.5
12 was the sparsity interaction tested at its contrast?     5.6
13 can the conditional advantage be exploited?              5.7
14 what all of this does and does not license               6
```

## The three sentences the paper must earn

1. **Fixed marginals, changed ordering, changed relative behaviour.** Section 4.3.
2. **Occurrence effects are primarily associated with dependence strength; magnitude
   effects primarily with dependence direction.** Section 4.6.
3. **The configuration transfers as a predictor and not as a mechanism.** Sections
   5.4–5.5.

Everything else is support, boundary, or negative result.

## What the figures alone must convey

```
Figure 1   here is the problem and here is what we control
Figure 2   here is what the control reveals, and it is asymmetric
Figure 3   here is what appears in real data, and where it stops
```

A reader who looks only at the three figures should be able to state the paper's
claim and its limit. Routing does not appear in any main figure.

---

## Wording that was removed in this revision

```
removed                                    replaced by
────────────────────────────────────────────────────────────────────────────────
"real data reproduce the magnitude-        "the frozen configuration shows predictive
 direction effect"                          transfer; its isolated mechanism does not
                                            survive overlap adjustment"
"exactly one counterexample"               "within the eighteen-cell Stage 2 grid,
                                            only one cell shows statistically clear
                                            superiority for direct prediction"
"Stage 1 identifies the Point regime"      removed entirely; Stage 1 has no cell whose
                                            interval excludes zero on that side
"the direction reproduces in real data"    "an empirical analogue appears"
"mechanism" used unqualified               "component-attribution diagnostic" in the
                                            synthetic study; nothing in real data
```

The general rule: the synthetic study licenses statements about a designed contrast;
the empirical study licenses statements about association and out-of-sample selection.
Neither licenses a causal mechanism claim, and no sentence may borrow strength across
the two.

---

## Structural decisions, now closed

**Sections 3 and 4 of the original plan are merged** into Section 3, so that "what do
the descriptors discard" runs straight into "why that matters for these two
estimators" without a section that ends on no question.

**Routing is Section 5.7, not an independent section.** It is a boundary on the
paper's own implication — that a conditional advantage invites adaptive selection —
and belongs at the end of the empirical work, at roughly half a page to one page. It
is C3, a supporting result. The paper must not read as two papers.
