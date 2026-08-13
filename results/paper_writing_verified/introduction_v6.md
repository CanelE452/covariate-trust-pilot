# Introduction — draft v6

Seven paragraphs, one logical role each. Three numeric results, listed with their
sources in `introduction_v6_claim_audit.md`. Citation keys resolve to
`../literature_boundary_verified/core_reference_list.md`; every one was verified through
Crossref. Supersedes `introduction_v5.md`; v1–v5 are retained. **Only P2 changed**: the
hurdle / two-stage precedent [NAR26] is acknowledged, now that component 6a is graded
PRIOR on verified full text, and [Kou13] is described in its own terms as a size /
interval ratio. Both changes reduce what the paragraph asserts.

---


Intermittent demand — series in which most periods are zero and the remainder carry
positive, often variable quantities — can be forecast in one of two ways. A model can
predict the conditional mean directly, or it can factorize that mean into the
probability of a demand event and the expected size given an event, and multiply the
two. Both target the same quantity, and the choice between them is usually made by
convention. On standard public benchmarks neither formulation is uniformly superior in
aggregate: which of the two leads depends on the error measure. A flat
aggregate does not make the choice unimportant. It suggests that the useful variation
is conditional, and that the question worth asking is not which representation wins but
when each one does.

Answering that requires distinguishing one intermittent series from another, usually
done with the average demand interval and the squared coefficient of variation of demand size
[SBC05; KH06]. Both summarize a marginal distribution and neither retains the order in
which intervals and sizes arrive: two series can match on both and still differ in
whether gaps cluster or alternate. That order is known to matter: simulation work varying
interval autocorrelation, size autocorrelation and their dependence reports
effects on forecast accuracy and inventory performance [ALR12]. Representation has been
examined separately: neural work compares a directly predicted demand rate against a
Croston-style representation that predicts non-zero demand size and inter-demand interval
and combines them as a ratio [Kou13], and single-stage models against two-stage models
that multiply an occurrence probability by a conditional size [NAR26]. These answer
different questions from the comparison here, in which a direct conditional
mean and an occurrence-probability × positive-magnitude factorization are held to one
capacity and training budget while occurrence and magnitude dependence are varied
separately.

We therefore ask a narrow question. Under matched marginal properties, when the
temporal organization of occurrence and of positive magnitude is varied, how does the
finite-sample relative behaviour of direct and factorized forecasting change? The
comparison is deliberately confined to one backbone family, one parameter budget, one
training protocol and one evaluation target. Because both formulations estimate the
same conditional mean, a difference observed under a matched budget is a difference in
finite-sample inductive bias, not a statement about which function class can represent
the target. We do not assume that either representation is generally preferable.

A controlled study answers the first half. In a factorial design that holds the gap
support, the average interval and the positive-demand marginal fixed and varies only
temporal order, occurrence structure and magnitude structure interact: their joint
effect is not the sum of their separate effects. That design contrasts structure
against its independent control, so it shows that order matters without identifying
which property of the order does. Replacing the binary
contrast with a signed dependence sweep resolves this into an asymmetric geometry —
that is, the two axes respond to different features of dependence. Occurrence effects
are primarily associated with the strength of dependence and largely insensitive to its
sign, whereas magnitude effects are primarily associated with its direction, with
persistence rather than structure moving the comparison toward direct prediction. Within the eighteen-cell sweep, one configuration shows statistically clear
superiority for direct prediction — sparse, occurrence-unpredictable,
magnitude-persistent — at a relative gain of about −19.8%.

The second half asks how much of this appears in observed demand. On two public retail
datasets the occurrence relationship appears as an empirical analogue rather than a
direct replication: the strength of interval dependence is positively associated with
the relative advantage of factorization in both, on three error scales, and the
association strengthens within the intermittent regime. Separately, a descriptor rule
encoding the configuration identified in the sweep, frozen before it was applied,
shifts previously unseen series toward direct prediction on an independent population,
by 11.87 percentage points of win rate, and the shift reproduces across three training
seeds. The synthetic-derived condition therefore carries predictive information about
which real series favour which representation.

That predictive transfer does not extend to the mechanism behind it. The rule selects
on three descriptors at once, and in observed demand those descriptors are not
separable in the way the controlled design makes them: candidate and control series
differ substantially in scale as well as in the intended axis. After adjusting for
overlap in scale, sparsity, positive-demand variability and occurrence dependence, the
isolated association is no longer distinguishable from zero. We therefore report two
results rather than one — the configuration transfers as a predictor, and its isolated
mechanism is not recovered — and note that the controlled study, which contains no
scale axis, cannot arbitrate the question either way.

Conditional differences of this kind invite a downstream question: can they be
exploited by a learned routing rule? The two forecasts do make complementary errors,
and an origin-level oracle over them is better than any fixed mixture, so the
opportunity is measurable. Converting it proved unreliable. Gains obtained during
development did not transfer across domains or across time, and one external domain
degraded severely, at which point a pre-registered stopping rule was triggered. This
paper therefore proposes neither a universal winner nor a generally reliable router. It
characterizes when direct and factorized forecasting differ under controlled temporal
structure, and identifies the boundary of that characterization in observed demand.
