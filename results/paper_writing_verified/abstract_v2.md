# Abstract — draft v2

---

Intermittent demand — mostly zeros — can be forecast by predicting the conditional mean
directly, or by factorizing it into an occurrence probability and a
conditional magnitude. Both target the same quantity, and on public benchmarks neither is
uniformly better in aggregate, so the useful variation is conditional. Standard descriptors of such
series summarize marginal properties and do not encode the order in which intervals and
magnitudes arrive. We compare the two formulations under
matched conditions — one backbone, one parameter count, one training budget — on synthetic
demand whose marginals are held fixed while the temporal dependence of occurrence and of
magnitude varies along two separate axes. The axes interact rather than add, and a signed
sweep resolves the interaction into an asymmetry: occurrence effects track the strength of
dependence, magnitude effects its direction, with persistence moving the comparison toward
direct prediction. In the eighteen-cell grid one configuration — sparse,
occurrence-unpredictable, magnitude-persistent — favours direct prediction by about −19.8%
in relative error. On two public retail datasets the occurrence relationship appears as an
empirical analogue, and a rule frozen from the sweep shifts unseen series toward direct
prediction by 11.87 percentage points of win rate. The isolated
association does not survive overlap adjustment, and a learned router did not transfer
across domains despite a measurable oracle opportunity. Both formulations are outranked by
classical estimators in absolute accuracy; the study isolates relative behaviour rather than
establishing forecasting accuracy. We report when factorizing helps or hurts at a fixed
budget, and where that stops holding.
