# Related Work — section summary

One block per subsection: the question a reader arrives with, what the literature
establishes, the core references, and what that implies for positioning. Numbers are word
counts of `related_work_v1.md`.

---

## 2.1 Classical Intermittent-Demand Forecasting and Decomposition — 247 words

```
reader question    "Aren't you just proposing that demand be split into occurrence and
                    size?  Hasn't that been done?"
what the           Splitting intermittent demand into a positive-size component and an
literature         occurrence-timing component is the field's default and dates to 1972.
establishes        Croston smooths sizes and inter-demand intervals only in periods with
                   demand and forms a rate as their RATIO.  SBA corrects the inversion
                   bias that the ratio induces.  TSB replaces interval updating with
                   direct updating of an occurrence PROBABILITY in every period.
core references    [Cro72]  [SB05]  [TSB11]
positioning        Decomposition is conceded outright.  The one distinction retained is
implication        formal, not credit-related: the classical lineage combines by ratio,
                   this paper's factorized arm by product, and [TSB11]'s probability
                   updating is named as the closest classical ancestor of that product.
```

## 2.2 Intermittency Classification and Marginal Descriptors — 234 words

```
reader question    "Why isn't the standard ADI / CV^2 description enough?"
what the           The standard scheme routes items to methods using the average
literature         inter-demand interval and the squared coefficient of variation of
establishes        positive sizes, separating smooth, erratic, intermittent and lumpy
                   demand, validated on several thousand automotive spare-part series;
                   one boundary was later refined analytically.
core references    [SBC05]  [KH06]
positioning        The scheme is used as given -- the paper's regime labels follow it and
implication        no alternative is proposed.  The observation made is definitional
                   rather than critical: both statistics summarize a marginal
                   distribution, so their definitions do not retain temporal ordering.
                   The subsection closes by saying so explicitly ("not a criticism of a
                   scheme built for a different purpose") to prevent a strawman reading.
```

## 2.3 Temporal Dependence in Intermittent Demand — 276 words

```
reader question    "Altay et al. already studied correlation in intermittent demand.
                    What is left?"
what the           Two things.  Dependence can be built INTO an estimator: lead-time
literature         demand bootstrapped from a two-state Markov occurrence model beat
establishes        exponential smoothing and Croston's method on nine industrial
                   datasets.  And dependence can be VARIED and its consequences measured:
                   size autocorrelation, interval autocorrelation and size-interval
                   cross-correlation each affect forecast accuracy and inventory
                   outcomes, and not uniformly in sign.
core references    [ALR12]  [WSS04]
positioning        Temporal dependence is conceded as established territory in as many
implication        words.  The only distinction drawn is the object of comparison: that
                   work compares estimators inside a single already-factorized
                   representation, so the question asked is what dependence does to those
                   estimators.
guard              LIT-W3 is OPEN.  No sentence states what [ALR12] holds constant; only
                   what it varies and what it reports.
```

## 2.4 Neural and Two-Part Forecasting Formulations — 387 words

```
reader question    "Hasn't someone already compared a direct forecast against a
                    decomposed one -- and specifically against a probability-times-size
                    model?"
what the           Yes, twice, in different settings.  In a neural setting, a directly
literature         emitted demand rate is compared against a Croston-style network that
establishes        emits size and interval and combines them as a RATIO, de-biased
                   afterwards; both are evaluated on a large simulated population, each
                   at its own best configuration.  In a gradient-boosting setting, a
                   LightGBM regressor trained directly on the full feature set is
                   compared against a LightGBM classifier times a Tweedie regressor --
                   the PRODUCT form -- under identical preprocessing, feature
                   construction and evaluation protocols, on ~1.4M real observations.
                   Occurrence and size have also been modelled jointly as a deep renewal
                   process.
core references    [Kou13]  [NAR26]  [TJWC21]  [GDTP25];  [MC26] in a footnote, labelled
                   as a preprint
positioning        Both comparisons are conceded without qualification, and the
implication        subsection ends by saying so: neither the factorized formulation nor
                   the act of comparing it against a direct one originates here.  This is
                   the section's load-bearing concession.
guards             LIT-W-KOU13: ratio, never "hurdle".
                   LIT-W-NAR26: gradient-boosting, never neural; the stated match is on
                   features and pipeline, and capacity / training / tuning are not
                   mentioned in either direction.
```

## 2.5 Positioning of the Present Study — 293 words

```
reader question    "Given all of that, what is this paper for?"
what the section   Not a new fact about the literature.  It restates the two streams as
does               two different questions -- what dependence does to performance with the
                   representation fixed, and which representation performs better on a
                   given population -- and then names a third: how the RELATIVE behaviour
                   of the two representations moves as dependence changes.
core references    none new; refers back to 2.3 and 2.4
the design, as     occurrence dependence and magnitude dependence varied on separate axes;
stated             marginals held fixed AS AN EXPERIMENTAL CONTROL, stated as a property
                   of the design rather than a result; same input history, one backbone
                   family, one parameter budget, one training procedure and budget, one
                   target -- hence a finite-sample difference rather than a
                   representational one.
the second half    how far the patterns reach into observed demand, reported as a result
                   in its own right: empirical analogue, predictive selector on unseen
                   series, and what does not survive adjustment for covariates the
                   controlled design separates but observed demand does not.
absence claims     0.  The contrast with 2.3 and 2.4 is left to the reader; the section
                   states what this study does, never what others did not.
ceiling            below LEVEL A of novelty_wording_options.md.  The strongest phrase is
                   "focuses on a different controlled intersection of these questions".
```

---

## Totals

```
2.1  247 w     2.2  234 w     2.3  276 w     2.4  387 w     2.5  293 w
TOTAL 1,437 words of prose (1,406 excluding headings and the footnote)
52 sentences, 22 literature-factual
citations in prose: 21 in-text occurrences + 1 footnote, 12 unique keys
```

## How the section answers the reader without an absence claim

Each subsection concedes its own topic in full, and each concession narrows the space by
one dimension:

```
2.1  decomposition is not ours                -> what remains is not "should we factorize"
2.2  the descriptors are marginal summaries   -> what remains is temporal ordering
2.3  dependence effects are established, for  -> what remains is the effect on the CHOICE
     estimators inside one representation        BETWEEN representations
2.4  both representation comparisons exist,   -> what remains is that comparison read AS A
     each on a fixed population                  FUNCTION of dependence, at one budget
2.5  states the design that does exactly that, and the boundary at which it stops holding
```

The contribution is legible because four things have been removed, not because anything
was declared missing from the literature.
