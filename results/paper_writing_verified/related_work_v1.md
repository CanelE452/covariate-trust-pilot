# 2. Related Work — draft v1

Prose draft of Sections 2.1–2.5. Structure follows
`../literature_boundary_verified/related_work_outline.md`; every literature statement is
mapped to a verified source in `related_work_reference_map.csv` and tagged in
`related_work_claim_audit.md`. Citation keys resolve to
`../literature_boundary_verified/core_reference_list.md`.

Introduction v6 is frozen and was not modified.

---

## 2.1 Classical Intermittent-Demand Forecasting and Decomposition

Forecasting demand that is zero in most periods has been approached, since Croston's
original treatment, by declining to model the demand series directly and instead tracking
two quantities separately: the size of a positive demand when one occurs, and the timing
of occurrences [Cro72]. That method smooths the positive sizes and the inter-demand
intervals only in periods with demand, and forms a demand rate as the ratio of the two
[Cro72].
The construction proved durable enough that most subsequent work is best read as
refinement rather than replacement. The ratio introduces an inversion bias, and the
Syntetos–Boylan approximation supplies the correction factor that renders the estimator
approximately unbiased [SB05]. A second refinement replaces interval updating with the
direct updating of an occurrence probability in every period, including periods without
demand, which allows the forecast to decay when an item stops moving [TSB11].

Two things follow for the present paper. First, treating occurrence and positive
magnitude as separate objects is not a modelling choice this paper introduces; it is the
default of the field and has been for five decades. Second, the classical lineage
combines its two components as a **ratio** of size to interval, whereas the factorized
formulation studied here combines an occurrence probability and a conditional positive
mean as a **product**. The probability-updating variant [TSB11] is the closest classical
ancestor of that product form. The two are related parameterizations of the same
conditional mean rather than the same estimator, and this paper treats them as such.

## 2.2 Intermittency Classification and Marginal Descriptors

Because intermittent series differ widely in how sparse and how variable they are,
practice relies on a small set of descriptors to decide which method to apply to which
item. The standard scheme is built on two statistics: the average inter-demand interval
and the squared coefficient of variation of positive demand sizes. Regions of the plane
these two define are used to separate smooth, erratic, intermittent and lumpy demand and
to select between Croston's method and its bias-corrected variant, with the boundaries
validated on several thousand automotive spare-part series [SBC05]. The placement of one
of those boundaries was subsequently refined on analytical grounds [KH06].

The scheme is economical and it is well established, and this paper uses it as given: the
regime labels reported later follow it, and no alternative classification is proposed.
What is worth stating explicitly is what the two statistics are functions of. Both
summarize a marginal distribution — how long the gaps are on average, and how variable
the positive sizes are — and their definitions therefore do not retain the temporal
ordering of individual occurrences or of positive magnitudes. Two series can agree on
both descriptors and still differ in whether their gaps arrive in clusters or in
alternation, and in whether large orders tend to follow large ones. That observation
motivates the design used here; it is not a criticism of a scheme built for a different
purpose.

## 2.3 Temporal Dependence in Intermittent Demand

The temporal structure of intermittent demand has itself been the subject of sustained
attention, in two related forms.

One line builds dependence into the estimator. Rather than treating occurrences as
independent draws, the lead-time demand distribution can be bootstrapped from a two-state
Markov model over zero and non-zero periods, with sampled positive sizes perturbed to
cover values not seen in a short history; on nine industrial datasets this produced more
accurate distributional forecasts than exponential smoothing or Croston's method [WSS04].
The occurrence process, on this view, carries information that an independence assumption
discards.

A second line treats dependence as something to vary and measure the consequences of.
Working with generated intermittent demand, Altay, Litteral and Rudisill examine three
distinct correlation structures — autocorrelation in demand sizes, autocorrelation in
inter-demand intervals, and cross-correlation between size and interval — and report their
effects on both forecast accuracy and inventory outcomes [ALR12]. The reported effects are
not uniform in sign across the three: negative autocorrelation is associated with higher
achieved service levels than positive autocorrelation, with cost largely unchanged, while
cross-correlation acts in the opposite direction to autocorrelation. The differences are
reported to intensify as intermittency increases.

Taken together, this stream establishes that serial structure in intermittent demand is
consequential for forecasting and for the inventory decisions built on it, and that its
different components need not act in the same direction. The estimators being compared in
that work sit inside a single, already-factorized representation; the question asked is
what dependence does to the performance of such estimators. Temporal dependence in
intermittent demand is therefore established territory, and no part of the present study
is positioned as introducing it.

## 2.4 Neural and Two-Part Forecasting Formulations

Machine-learning and neural methods for intermittent demand now form a substantial
literature of their own, recently surveyed in [GDTP25]. Within it, whether the forecast
should be produced as one quantity or as two combined has been asked directly, and in more
than one form [Kou13; NAR26].

In the neural setting, Kourentzes compares two architectures that differ precisely in
this respect [Kou13]. One takes lagged non-zero demands and lagged inter-demand intervals
as inputs and emits the demand rate from a single output. The other takes the same inputs
and emits two quantities, the non-zero demand size and the inter-demand interval, which
are combined as a ratio in the manner of Croston's method and then corrected for the
resulting inversion bias [Kou13]. Both are evaluated over
a large simulated population parameterized from a real spare-parts dataset, each at its
own best configuration of input lags and hidden nodes, selected on in-sample error rank.
It reports that the two formulations rank differently under accuracy metrics than under
inventory metrics, and favours the directly predicted rate once service levels are
considered [Kou13].

The other pairing — a directly predicted conditional mean against an occurrence
probability multiplied by a conditional size — has been compared in a gradient-boosting
rather than a neural setting [NAR26]. On roughly 1.4 million monthly observations of automotive
spare parts, a LightGBM regressor trained directly on the full feature set is placed
against a two-stage model in which a LightGBM classifier estimates the probability of
non-zero demand and a Tweedie-objective LightGBM regressor predicts the conditional
quantity, both operating under identical data preprocessing, feature construction and
evaluation protocols [NAR26]. That study [NAR26] reports that the added
architectural complexity of the two-stage form does not translate into an aggregate
advantage over the single-stage model once informative features are supplied. Occurrence
and size have also been modelled jointly rather than comparatively, by casting
intermittent demand as a deep renewal process that captures regular and alternating
inter-arrival structure on constructed patterns [TJWC21];
hurdle-style decoders continue to appear in current architectures.<sup>1</sup>

Both comparisons therefore already exist: a directly predicted rate against a
Croston-style ratio in a neural setting [Kou13], and a directly predicted conditional mean
against a probability-times-size product in a tree-ensemble setting on real data [NAR26].
Neither the factorized formulation nor the act of comparing it against a direct one
originates here.

## 2.5 Positioning of the Present Study

The two streams above answer different questions. The dependence stream asks what happens
to forecasting and inventory performance when the serial structure of demand changes,
with the representation held fixed. The representation stream asks which of two ways of
structuring a forecast performs better on a given population.

The present study focuses on a different controlled intersection of these questions: how
the *relative* behaviour of the two representations moves as temporal dependence changes.
A directly predicted conditional mean and an occurrence-probability × positive-magnitude
factorization are compared while the temporal dependence of occurrence and that of
positive magnitude are varied along separate axes, with the marginal properties of the
generated demand held fixed as an experimental control. That control is a property of the
design rather than a result: it is what allows a change in relative performance to be
attributed to temporal organization rather than to sparsity or size variability. Both
arms receive the same input history, share one backbone family and one parameter budget,
and are trained by one procedure under one budget against one target, so the difference
observed is a difference in finite-sample behaviour under a fixed budget rather than in
what either function class can represent.

The second half of the study asks how far the resulting patterns reach into observed
demand, and reports that boundary as a result in its own right: which parts appear as an
empirical analogue, which transfer as a predictive selector on unseen series, and which do
not survive adjustment for covariates that the controlled design separates but observed
demand does not.

---

<sup>1</sup> For example, a mixture-of-experts encoder with a hurdle decoder
[MC26]; this record is an arXiv preprint and is cited as an indication of current
practice rather than as peer-reviewed evidence.
