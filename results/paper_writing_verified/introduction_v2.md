# Introduction — draft v2

Seven paragraphs, one logical role each. Three numeric results, listed with their
sources in `introduction_v2_claim_audit.md`. Citation keys are placeholders because the
repository contains no bibliography. Supersedes `introduction_v1.md`, which is retained.

---

Intermittent demand — series in which most periods are zero and the remainder carry
positive, often variable quantities — can be forecast in one of two ways. A model can
predict the conditional mean directly, or it can factorize that mean into the
probability of a demand event and the expected size given an event, and multiply the
two. Both target the same quantity, and the choice between them is usually made by
convention. On standard public benchmarks neither formulation is uniformly superior in
aggregate: the two are close, and which one leads depends on the error measure. A flat
aggregate does not make the choice unimportant. It suggests that the useful variation
is conditional, and that the question worth asking is not which representation wins but
when each one does.

Answering that requires saying what distinguishes one intermittent series from another.
In practice this is done with the average demand interval and the squared coefficient
of variation of demand size, the two statistics behind the standard categorization
scheme [CITATION NEEDED]. Both are functionals of a marginal distribution, and neither
is affected by the order in which intervals and sizes arrive. Two series can share
their average interval, their interval support and their positive-demand distribution
and still differ in whether short and long gaps alternate or cluster, and in whether
large orders tend to follow large ones. Temporal dependence in intermittent demand is
not a new subject: autocorrelation in demand intervals, autocorrelation in demand
sizes, and dependence between the two have all been studied, and estimators have been
proposed that exploit them [CITATION NEEDED]. What has not been isolated is whether
such dependence should change the *representation* a forecaster uses, rather than the
estimator applied within it.

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

If the advantage is conditional, the natural response is to route between the two
representations. We tested this and report the boundary. The two forecasts do make
complementary errors, and an origin-level oracle over them is meaningfully better than
any fixed mixture; deliberately choosing a less correlated pair enlarges that
opportunity further. Converting it proved unreliable. A gate frozen after development
was worse than a static mixture on the first external dataset it was applied to, and
successive changes to the training target, the loss, the aggressiveness, the capacity
and the input representation did not recover cross-domain transfer; the last of these
improved one external domain and degraded severely on another, at which point a
pre-registered stopping rule was triggered. This paper therefore proposes neither a
universal winner nor a robust routing rule. It characterizes when factorizing
intermittent demand helps and when it hurts under controlled temporal structure, and
identifies the boundary of that characterization in observed demand.
