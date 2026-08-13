# 7. Conclusion — draft v1

---

Intermittent demand can be forecast by predicting the conditional mean directly or by
factorizing it into an occurrence probability and a conditional positive magnitude. Both
target the same quantity, and on standard benchmarks neither is uniformly better in
aggregate. This paper asked when each one does better, and answered it by holding the
marginal properties of the demand fixed and varying only its temporal organization.

Under a matched backbone, parameter count, training procedure and budget, the relative
behaviour of the two formulations moves substantially with temporal structure, and the two
structural axes interact rather than add. A signed dependence sweep resolves that
interaction into an asymmetry: on the occurrence axis it is the strength of dependence that
tracks the comparison, while on the magnitude axis it is the direction, with persistence
moving the comparison toward direct prediction. Within the eighteen-cell grid one
configuration — sparse, occurrence-unpredictable, magnitude-persistent — favours direct
prediction with a clear margin.

That characterization reaches observed demand as a conditional signal and stops there. The
occurrence relationship appears on two public retail datasets as an empirical analogue, and
a rule frozen from the controlled sweep shifts previously unseen series toward direct
prediction on an independent population. The isolated association behind it does not survive
adjustment for overlap, because the descriptors the controlled design separates arrive
together in real data. Routing between the two formulations has a measurable oracle
opportunity that did not convert into a router transferring across domains or time.

The contribution is therefore a characterization and its boundary rather than a
recommendation: which conditions make factorizing intermittent demand help or hurt at a
fixed budget, and how far that answer currently travels.
