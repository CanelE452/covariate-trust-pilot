# Introduction — draft v1

Six paragraphs. Numbers appear only where a verified artifact supports them; sources
are listed in `introduction_claim_audit.md`. Citation keys are placeholders because the
repository contains no bibliography.

---

Intermittent demand — series in which most periods are zero and the remainder carry
positive, often variable quantities — is forecast in one of two ways. A model can
predict the conditional mean directly, or it can factorize that mean into the
probability of a demand event and the expected size given an event, and multiply the
two. Both target the same quantity. The choice between them is made routinely, usually
by convention rather than by evidence, and on two standard public benchmarks it is
close to a coin flip: across 1,200 series each, the mean difference between the two is
−0.0007 on M5 and −0.0320 on Favorita, with RMSE marginally favouring the direct model
and MAE the factorized one on both. An aggregate this flat does not mean the choice is
unimportant. It means the interesting variation is conditional, and that the useful
question is not which representation wins but when each one does.

Answering that question requires saying what distinguishes one intermittent series
from another. In practice this is done with the average demand interval and the squared
coefficient of variation of demand size, the two statistics behind the standard
categorization scheme [CITATION NEEDED]. Both are functionals of a marginal
distribution. Neither is affected by the order in which intervals and sizes arrive.
Two series can share their ADI, their interval support and their positive-demand
distribution and still differ in whether short and long gaps alternate or cluster, and
in whether large orders tend to follow large ones. Temporal dependence in intermittent
demand is not a new subject: autocorrelation in demand intervals, autocorrelation in
demand sizes, and dependence between the two have all been studied, and estimators have
been proposed that exploit them [CITATION NEEDED]. What has not been isolated is
whether that dependence should change the *representation* a forecaster uses, as
opposed to the estimator.

We therefore ask a narrow question. Under matched marginal properties, when the
temporal organization of occurrence and of positive magnitude is varied, how does the
finite-sample relative behaviour of direct and factorized forecasting change? The
comparison is deliberately confined: one backbone family, one parameter budget, one
training protocol, one evaluation target. Because both formulations estimate the same
conditional mean, any difference under a matched budget is a difference in
finite-sample inductive bias, not a statement about which function class can represent
the target. We do not assume that either representation is generally preferable.

A controlled study answers the first half. In a factorial design that holds the gap
support, the mean interval and the positive-demand marginal fixed and varies only
temporal order, relative performance moves substantially, and the two axes do not act
independently: their interaction is larger in magnitude than either main effect. That
design contrasts structure against its independent control, so it establishes that
order matters without saying which property of the order is responsible. Replacing the
binary contrast with a signed dependence sweep resolves this. Occurrence effects are
primarily associated with the *strength* of dependence and are insensitive to its
sign, while magnitude effects are primarily associated with its *direction*, with
persistence — not structure as such — moving the comparison toward direct prediction.
The two axes therefore have different response geometry, which is what makes their
interaction interpretable rather than merely large. Within the eighteen conditions of
that sweep, exactly one shows statistically clear superiority for direct prediction:
sparse, occurrence-unpredictable, magnitude-persistent, at a relative gain of −19.8%.

How much of this survives real demand is the second half. On M5 and Favorita the
occurrence relationship appears as an empirical analogue: the strength of interval
dependence is positively associated with the relative advantage of factorization in
both datasets, on three error scales, and it strengthens within the intermittent
regime while being indistinguishable from zero in the lumpy one. A rule encoding the
synthetic direct-favourable configuration, frozen before use, shifts unseen series
toward direct prediction on an independent population by nearly twelve percentage
points of win rate, and reproduces across three training seeds. The mechanism does not
follow the selector. Once sparsity, variability, scale and occurrence dependence are
balanced, the same association is not distinguishable from zero, and the covariate
responsible is scale — an axis the controlled design does not contain and therefore
cannot arbitrate. Predictive transfer and mechanism replication come apart here, and
the paper reports them as two results rather than one.

If the advantage is conditional, the natural engineering response is to route between
the two representations per series or per origin. We tried, and report the boundary. An
origin-level oracle over the two is about four percent better than the best static
mixture, and deliberately diversifying the expert pair roughly doubles that ceiling,
so the opportunity is real and measurable. A gate frozen after development was worse
than a static mixture on the first external dataset it saw, and successive changes to
the target, the loss, the aggressiveness, the capacity and the input representation did
not recover it; the last of these repaired one dataset and failed catastrophically on
another. A pre-registered stopping rule was triggered. What this work offers is
therefore not a universal winner and not a router, but a controlled characterization of
when factorizing intermittent demand helps and when it hurts, together with an explicit
account of how far that characterization carries into observational data.
