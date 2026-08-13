# Related Work v1 — claim audit

Sentence-level audit of `related_work_v1.md`. Mapping of every literature-factual sentence
to a verified source is in `related_work_reference_map.csv` (22 rows, unmapped 0).

```
LITERATURE_SUPPORTED   a verified record backs it, and it is cited
BOUNDARY_WORDING       supported, and the sentence exists to limit a claim
POSITIONING            a statement about THIS study or about section structure, not a
                       claim about the literature
METADATA_UNRESOLVED    the sentence depends on metadata this audit could not verify
                       -- must be 0 in main prose
OVERCLAIM              wording exceeds the evidence            -- must be 0
UNSUPPORTED            no verified record backs it             -- must be 0
```

---

## 2.1 Classical Intermittent-Demand Forecasting and Decomposition *(247 words)*

```
P1S1  since Croston, occurrence timing and positive size are tracked
      separately rather than the series modelled directly              LITERATURE_SUPPORTED
                                                                       [Cro72]
P1S2  smoothing occurs only in periods with demand; the rate is the
      ratio of size to interval                                        LITERATURE_SUPPORTED
                                                                       [Cro72]
P1S3  most subsequent work is refinement rather than replacement       BOUNDARY_WORDING
      a characterization of the lineage, not a claim about any one paper
P1S4  the ratio induces an inversion bias; SBA supplies the
      correction that renders the estimator approximately unbiased     LITERATURE_SUPPORTED
                                                                       [SB05]
P1S5  a later refinement updates an occurrence probability every
      period, including zero periods, allowing decay under
      obsolescence                                                     LITERATURE_SUPPORTED
                                                                       [TSB11]
P2S1  two things follow                                                POSITIONING
P2S2  separating occurrence and positive magnitude is not introduced
      here; it is the field default and has been for five decades      BOUNDARY_WORDING
      the sentence exists to block a decomposition-novelty reading
P2S3  the classical lineage combines by RATIO; the formulation here
      combines by PRODUCT                                              BOUNDARY_WORDING
      distinction verified in kou13_representation_verification.md and [Cro72]
P2S4  the probability-updating variant is the closest classical
      ancestor of the product form                                     LITERATURE_SUPPORTED
                                                                       [TSB11]
P2S5  related parameterizations of one conditional mean rather than
      the same estimator                                               BOUNDARY_WORDING
```

## 2.2 Intermittency Classification and Marginal Descriptors *(234 words)*

```
P1S1  practice uses descriptors to route items to methods            POSITIONING
P1S2  the scheme is built on the average inter-demand interval and
      the squared coefficient of variation of positive sizes         LITERATURE_SUPPORTED
                                                                     [SBC05]
P1S3  the regions separate smooth / erratic / intermittent / lumpy
      demand and select between Croston's method and its bias-
      corrected variant, validated on several thousand automotive
      spare-part series                                              LITERATURE_SUPPORTED
                                                                     [SBC05]
P1S4  one boundary was subsequently refined on analytical grounds    LITERATURE_SUPPORTED
                                                                     [KH06]
P2S1  the paper uses the scheme as given; no alternative proposed    POSITIONING
P2S2  worth stating what the two statistics are functions of        POSITIONING
P2S3  both summarize a marginal distribution, so their definitions
      do not retain the temporal ordering of occurrences or
      magnitudes                                                     BOUNDARY_WORDING
      definitional; follows from [SBC05]'s definitions.  Deliberately says
      "do not retain", never "fail", "ignore" or "are inadequate".
P2S4  two series can agree on both and still differ in clustering
      versus alternation, and in whether large orders follow large    BOUNDARY_WORDING
P2S5  that motivates the design; it is not a criticism of a scheme
      built for a different purpose                                  BOUNDARY_WORDING
      present specifically to prevent RW-Q2 from failing
```

## 2.3 Temporal Dependence in Intermittent Demand *(276 words)*

```
P1S1  the temporal structure has had sustained attention, in two
      forms                                                          POSITIONING
P2S1  one line builds dependence into the estimator                  POSITIONING
P2S2  lead-time demand bootstrapped from a two-state Markov model
      over zero and non-zero periods, with sampled positive sizes
      perturbed; on nine industrial datasets this beat exponential
      smoothing and Croston's method on distributional accuracy      LITERATURE_SUPPORTED
                                                                     [WSS04]
P2S3  the occurrence process carries information an independence
      assumption discards                                            BOUNDARY_WORDING
P3S1  a second line varies dependence and measures consequences      POSITIONING
P3S2  three correlation structures -- size autocorrelation, interval
      autocorrelation, size-interval cross-correlation -- examined
      in generated intermittent demand, with effects reported on
      forecast accuracy and inventory outcomes                       LITERATURE_SUPPORTED
                                                                     [ALR12]
      *** LIT-W3 GUARD: says WHAT WAS VARIED and WHAT WAS REPORTED.  Says nothing
      about what was held constant.  See the scope note below. ***
P3S3  the effects are not uniform in sign: negative autocorrelation
      is associated with higher achieved service levels than
      positive, cost largely unchanged; cross-correlation acts in
      the opposite direction                                         LITERATURE_SUPPORTED
                                                                     [ALR12]
P3S4  the differences are reported to intensify as intermittency
      increases                                                      LITERATURE_SUPPORTED
                                                                     [ALR12], attributed
                                                                     ("are reported to")
P4S1  serial structure is consequential for forecasting and for the
      inventory decisions built on it, and its components need not
      act in the same direction                                      LITERATURE_SUPPORTED
                                                                     [ALR12] [WSS04]
P4S2  the estimators compared in that work sit inside a single,
      already-factorized representation                              BOUNDARY_WORDING
      graded high-confidence from title and abstract; see the scope note
P4S3  temporal dependence is established territory and no part of
      this study is positioned as introducing it                     BOUNDARY_WORDING
```

**LIT-W3 scope note.** `[ALR12]`'s full text could not be obtained (six routes; see
`../literature_boundary_verified/alr12_fulltext_verification.md`). Section 2.3 therefore
states only what is established at abstract and record level: which dependence structures
are varied, that the demand is generated, and what outcomes are reported. **No sentence
states what the study holds constant**, and no sentence describes its marginal-control
design. Checked mechanically (RW7, RW8). If the full text is later obtained, 2.3 can gain
detail; nothing in it needs retraction.

## 2.4 Neural and Two-Part Forecasting Formulations *(387 words)*

```
P1S1  ML and neural methods for intermittent demand form a
      substantial literature, recently surveyed                      LITERATURE_SUPPORTED
                                                                     [GDTP25]
P1S2  whether the forecast should be one quantity or two combined
      has been asked directly, in more than one form                 LITERATURE_SUPPORTED
                                                                     [Kou13; NAR26]
P2S1  Kourentzes compares two architectures differing in this
      respect                                                        LITERATURE_SUPPORTED
                                                                     [Kou13]
P2S2  one emits the demand rate from a single output                 LITERATURE_SUPPORTED
                                                                     [Kou13], full text
P2S3  the other emits size and interval, combined as a RATIO in the
      manner of Croston's method and then corrected for the
      resulting inversion bias                                       LITERATURE_SUPPORTED
                                                                     [Kou13], full text
      *** LIT-W-KOU13 GUARD: ratio, never "hurdle", never "the same two
      representations".  Checked mechanically (RW10). ***
P2S4  both evaluated on a large simulated population parameterized
      from a real spare-parts dataset, each at its own best
      configuration of lags and hidden nodes, selected on in-sample
      error rank                                                     LITERATURE_SUPPORTED
                                                                     [Kou13], sec 3.1 / 3.4
P2S5  the two rank differently under accuracy than under inventory
      metrics; the study favours the direct rate once service
      levels are considered                                          LITERATURE_SUPPORTED
                                                                     [Kou13], sec 4
P3S1  the other pairing -- direct conditional mean against
      probability times conditional size -- has been compared in a
      gradient-boosting rather than a neural setting                  LITERATURE_SUPPORTED
                                                                     [NAR26]
      *** LIT-W-NAR26 GUARD: the setting is named as gradient-boosting, so the
      paper is never implied to be neural.  Checked mechanically (RW11). ***
P3S2  on ~1.4M monthly observations, a LightGBM regressor trained
      directly on the full feature set against a two-stage LightGBM
      classifier plus Tweedie-objective regressor, both under
      identical data preprocessing, feature construction and
      evaluation protocols                                           LITERATURE_SUPPORTED
                                                                     [NAR26], full text
      *** the match reported is on FEATURES and PIPELINE only.  Capacity, training
      budget and per-formulation tuning are NOT STATED and are not mentioned in
      either direction.  Checked mechanically (RW13). ***
P3S3  the study reports that the two-stage form's added complexity
      does not translate into an aggregate advantage once
      informative features are supplied                              LITERATURE_SUPPORTED
                                                                     [NAR26], attributed
P3S4  occurrence and size have also been modelled jointly as a deep
      renewal process capturing regular and alternating inter-
      arrival structure on constructed patterns                      LITERATURE_SUPPORTED
                                                                     [TJWC21]
P4S1  both comparisons already exist: direct against a Croston-style
      ratio in a neural setting, and direct against a probability-
      times-size product in a tree-ensemble setting on real data     LITERATURE_SUPPORTED
                                                                     [Kou13] [NAR26]
P4S2  neither the factorized formulation nor the act of comparing it
      against a direct one originates here                           BOUNDARY_WORDING
      the load-bearing concession of the whole section
FN1   a mixture-of-experts encoder with a hurdle decoder, labelled
      in the footnote as an arXiv preprint cited for current
      practice rather than as peer-reviewed evidence                 LITERATURE_SUPPORTED
                                                                     [MC26], preprint
```

## 2.5 Positioning of the Present Study *(293 words)*

```
P1S1  the two streams answer different questions                     POSITIONING
P1S2  the dependence stream asks what happens to forecasting and
      inventory performance when serial structure changes, with the
      representation held fixed                                      BOUNDARY_WORDING
      a characterization of 2.3, not a deficiency claim
P1S3  the representation stream asks which of two ways of
      structuring a forecast performs better on a given population   BOUNDARY_WORDING
      a characterization of 2.4, not a deficiency claim
P2S1  the present study focuses on a different controlled
      intersection: how the RELATIVE behaviour moves as dependence
      changes                                                        POSITIONING
P2S2  the two formulations are compared while occurrence dependence
      and magnitude dependence are varied along separate axes, with
      the marginals held fixed as an experimental control            POSITIONING
P2S3  that control is a property of the design rather than a result POSITIONING
      present specifically to block a fixed-marginal novelty reading (RW15)
P2S4  same input history, one backbone family, one parameter budget,
      one training procedure and budget, one target -- so the
      difference is finite-sample behaviour under a fixed budget
      rather than representational capacity                          POSITIONING
      "matched" is stated as a property of OUR design; nothing is said about
      whether prior work matched (component 6b, CLAIM_ONLY_IN_CONJUNCTION)
P3S1  the second half asks how far the patterns reach into observed
      demand and reports that boundary as a result: analogue,
      predictive selector, and what does not survive adjustment      POSITIONING
```

---

## Totals

```
LITERATURE_SUPPORTED   22
BOUNDARY_WORDING       14
POSITIONING            16
METADATA_UNRESOLVED     0
OVERCLAIM               0
UNSUPPORTED             0
```

52 sentences. Every one of the 22 literature-factual sentences carries a citation key
that resolves to `core_reference_list.md`; `related_work_reference_map.csv` has 22 rows
and 0 unmapped.

`METADATA_UNRESOLVED = 0` in main prose. `LIT-W2` ([GDTP25] volume and pages unassigned)
does not surface, because no sentence depends on that metadata; it will matter only when
the bibliography is typeset.

---

## Absence-claim scan

```
"first" / "first to"                       0
"no prior work" / "no previous"            0
"nobody has" / "has never been"            0
"unexplored" / "remains unexplored"        0
"novel combination"                        0
"have never been combined"                 0
prior work cannot explain / does not
  predict our result                       0
```

The section makes **no negative existential statement of any scope.** Section 2.5
positions the study by describing what it does, and the contrast with 2.3 and 2.4 is left
to the reader. Verified mechanically (RW16, RW17).

---

## Guards discharged

```
LIT-W3        2.3 states what [ALR12] varied and reported; nothing about what it held
              constant.  RW7, RW8 PASS.
LIT-W-KOU13   2.4 P2S3 describes the RATIO form; "hurdle" never attached to [Kou13];
              "the same two representations" absent.  RW10 PASS.
LIT-W-NAR26   2.4 names the setting as gradient-boosting and the learner as LightGBM;
              the stated match is on features and pipeline only.  RW11, RW13 PASS.
MC26          appears only in a footnote, labelled as an arXiv preprint.
LIT-W1        [Cro72] is cited by key; the container-name discrepancy is a bibliography
              matter and does not surface in prose.
```

---

## Reviewer simulation

```
RW-Q1   Does Croston / decomposition read as our invention?           NO
        2.1 P2S2: "not a modelling choice this paper introduces; it is the default of
        the field and has been for five decades."
RW-Q2   Are ADI / CV^2 disparaged?                                    NO
        2.2 P2S5: "it is not a criticism of a scheme built for a different purpose."
        The scheme is used as given for the regime labels.
RW-Q3   Is the temporal-dependence precedent adequately conceded?     YES
        2.3 is 276 words, cites [ALR12] four times and [WSS04] once, and closes with
        "established territory ... no part of the present study is positioned as
        introducing it."
RW-Q4   Are [ALR12]'s marginal controls written as verified?          NO
        No sentence states what it holds constant.  RW7, RW8.
RW-Q5   Is [Kou13]'s ratio distinguished from our product?            YES
        2.1 P2S3 draws the ratio / product distinction; 2.4 P2S3 describes the ratio
        and its inversion bias.
RW-Q6   Is [NAR26] called neural?                                     NO
        "in a gradient-boosting rather than a neural setting"; all arms named LightGBM.
RW-Q7   Are same features read as matched capacity?                   NO
        Only "identical data preprocessing, feature construction and evaluation
        protocols" is stated -- the article's own words.  Capacity and training are not
        mentioned in either direction.
RW-Q8   Is direct-vs-hurdle itself made the novelty?                  NO
        2.4 P4S2: "Neither the factorized formulation nor the act of comparing it
        against a direct one originates here."
RW-Q9   Any first / no-prior-work / unexplored claim?                 NO   0 occurrences
RW-Q10  Is it claimed that prior work cannot predict our asymmetry?   NO
        The asymmetry is not mentioned in Related Work at all; it belongs to Section 4.
RW-Q11  Does the contribution still read?                             YES
        2.5: matched representation-relative finite-sample behaviour as occurrence and
        magnitude dependence vary separately, plus the empirical transfer boundary.
RW-Q12  Is any closest precedent made a strawman?                     NO
        [Kou13] and [NAR26] are each given their own reported conclusion, in their own
        favour.  "Unlike prior work" appears 0 times.
```

All twelve pass.

---

## Novelty ceiling check

The strongest statement in the section is 2.5 P2S1–S2, which asserts a focus, not a
priority: *"The present study focuses on a different controlled intersection of these
questions."* That sits below LEVEL A of
`../literature_boundary_verified/novelty_wording_options.md`, which is the ceiling this
step was given. LEVEL A itself is not used in the prose; it is reserved for the abstract
and the contributions list.
