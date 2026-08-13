# Claim wording

Fourteen pairs. Every allowed sentence is checkable against a named artifact.

---

**1. Overall comparison**

ALLOWED — "Across 1,200 series in each of two public datasets neither representation dominates:
the mean difference is −0.0007 on M5 and −0.0320 on Favorita, and the sign of the aggregate
depends on whether RMSE or MAE is used."
FORBIDDEN — "The two models perform equivalently."

**2. H1 direction**

ALLOWED — "Occurrence-interval dependence is positively associated with the relative advantage
of the factorized model in both datasets (Spearman +0.107 and +0.079, both intervals excluding
zero)."
FORBIDDEN — "Temporal occurrence dependence causes the factorized model to win."

**3. H1 robustness**

ALLOWED — "The association is positive on all three error scales in both datasets, six
estimates in total, with every interval excluding zero, and is unchanged under eligibility
thresholds of 15, 20 and 30 positive training observations."
FORBIDDEN — "The association is robust." (Say to what.)

**4. H1 adjusted**

ALLOWED — "Once sparsity, positive-demand variability and scale are entered jointly, the
standardized partial coefficient on occurrence dependence is +0.03 with an interval containing
zero; we therefore report a marginal association rather than a controlled effect."
FORBIDDEN — "The effect survives adjustment for confounders."

**5. H1 by regime**

ALLOWED — "The association is present in the intermittent regime on the relative and scaled
metrics (+0.153 and +0.123, intervals excluding zero) and is not distinguishable from zero in
the lumpy regime."
FORBIDDEN — "H1 holds in intermittent demand." (Only two of three scales, and lumpy fails.)

**6. Sample composition**

ALLOWED — "The screen uses a regime-balanced sample of 300 series per SBC class; the M5 full
pool is 23,053 intermittent, 5,942 lumpy, 984 smooth and 496 erratic, so these estimates are not
population estimates."
FORBIDDEN — Quoting +0.1065 as M5's effect without the balancing caveat.

**7. H2 as a selector**

ALLOWED — "A rule derived from the controlled study and frozen before use shifts unseen M5
series toward the direct model by 11.9 percentage points of win rate (interval [7.9, 15.8]) on
675 candidates against 5,018 controls, and the effect reproduces under three training seeds."
FORBIDDEN — "Magnitude persistence causes the direct model to outperform the factorized model."

**8. H2 as a mechanism**

ALLOWED — "After overlap weighting that balances sparsity, positive-demand variability, scale
and occurrence dependence to a worst standardized difference of 0.0004, the association is
+0.003 with an interval containing zero; the rule should be read as a predictor rather than as
evidence about magnitude persistence."
FORBIDDEN — "We isolate the mechanism behind the Point-favorable regime."

**9. H2 across datasets**

ALLOWED — "The M5-frozen rule transfers partially to unseen Favorita series; on the smaller
Favorita screen pool of 18 candidates the interval contains zero and the win-rate difference has
the opposite sign, and we report both analyses."
FORBIDDEN — "The rule replicates on Favorita."

**10. H3**

ALLOWED — "The predicted sparsity interaction did not replicate at the pre-registered median
split, where both datasets show the opposite sign with intervals containing zero; we note that
this split does not correspond to the ADI 4-versus-8 contrast from which the prediction was
derived."
FORBIDDEN — "Sparsity does not modulate the occurrence effect."

**11. Occurrence mechanism on real data**

ALLOWED — "In both datasets the fitted occurrence head does not improve on a per-series constant
rate (Brier skill −0.008 on M5 and −0.091 on Favorita), so we do not attribute the observed
association to improved occurrence prediction."
FORBIDDEN — "Occurrence predictability explains the factorization advantage."

**12. Absolute positioning**

ALLOWED — "Both neural variants are outranked by SBA on both datasets; this study compares two
representations of one backbone under one training budget and is not a claim about the best
available forecaster."
FORBIDDEN — "Our models are competitive with classical intermittent-demand methods."

**13. Complementarity and diversity**

ALLOWED — "A per-origin oracle over the two experts is 4.1% better than the best static mixture,
and selecting a less correlated pair multiplies that ceiling by 2.15; the realized gain from
routing the diverse pair was 0.4%."
FORBIDDEN — "Greater expert diversity improves forecast accuracy." (The ceiling rose; the
realized gain did not.)

**14. Routing**

ALLOWED — "A gate that improved on held-out series of the development datasets was 2.4% worse
than a static mixture on the first external dataset it was applied to, and successive
modifications of the target, the loss, the aggressiveness, the capacity and the input
representation did not recover it."
ALLOWED — "Reading the raw window recovers routing signal the summary descriptors do not carry
on one dataset (+2.6%, interval [2.1, 3.3]), while the same model is catastrophically worse on
another (−193.9%); we therefore separate the claim that raw history contains routing information
from the claim that routing generalizes."
FORBIDDEN — "Our adaptive router generalizes across retail domains."
FORBIDDEN — "Sequence models solve the routing problem."
