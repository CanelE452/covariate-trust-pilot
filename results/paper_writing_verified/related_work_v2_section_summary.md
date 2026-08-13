# Related Work v2 — section summary

Supersedes `related_work_section_summary.md`, which is retained for v1. Sections 2.1–2.3
are byte-identical to v1, so their blocks are unchanged; 2.4 and 2.5 are updated.

---

## 2.1 Classical Intermittent-Demand Forecasting and Decomposition — 247 words *(unchanged)*

```
reader question    "Aren't you just proposing that demand be split into occurrence and
                    size?  Hasn't that been done?"
what the           Splitting intermittent demand into a positive-size component and an
literature         occurrence-timing component is the field's default and dates to 1972.
establishes        Croston smooths sizes and inter-demand intervals only in periods with
                   demand and forms a rate as their RATIO.  SBA corrects the inversion
                   bias the ratio induces.  TSB replaces interval updating with direct
                   updating of an occurrence PROBABILITY in every period.
core references    [Cro72]  [SB05]  [TSB11]
positioning        Decomposition is conceded outright.  The one distinction retained is
implication        formal: the classical lineage combines by ratio, this paper's
                   factorized arm by product, with [TSB11] named as the closest classical
                   ancestor of that product.
```

## 2.2 Intermittency Classification and Marginal Descriptors — 234 words *(unchanged)*

```
reader question    "Why isn't the standard ADI / CV^2 description enough?"
what the           The standard scheme routes items to methods using the average
literature         inter-demand interval and the squared coefficient of variation of
establishes        positive sizes, separating smooth, erratic, intermittent and lumpy
                   demand, validated on several thousand automotive spare-part series;
                   one boundary was later refined analytically.
core references    [SBC05]  [KH06]
positioning        The scheme is used as given.  The observation made is definitional
implication        rather than critical -- both statistics summarize a marginal
                   distribution, so their definitions do not retain temporal ordering --
                   and the subsection says so explicitly to avoid a strawman reading.
```

## 2.3 Temporal Dependence in Intermittent Demand — 276 words *(unchanged)*

```
reader question    "Altay et al. already studied correlation in intermittent demand.
                    What is left?"
what the           Dependence can be built INTO an estimator (a two-state Markov
literature         occurrence model inside a bootstrap beat exponential smoothing and
establishes        Croston's method on nine industrial datasets), and dependence can be
                   VARIED and its consequences measured (size autocorrelation, interval
                   autocorrelation and size-interval cross-correlation each affect
                   forecast accuracy and inventory outcomes, and not uniformly in sign).
core references    [ALR12]  [WSS04]
positioning        Conceded as established territory in as many words.  The only
implication        distinction drawn is the object of comparison: that work compares
                   estimators inside a single already-factorized representation.
guard              LIT-W3 OPEN.  No sentence states what [ALR12] holds constant.
```

## 2.4 Neural and Two-Part Forecasting Formulations — 333 words *(compressed from 387)*

```
reader question    "Hasn't someone already compared a direct forecast against a
                    decomposed one -- and specifically against a probability-times-size
                    model?"
what the           Yes, twice, in different settings.  In a neural setting, a directly
literature         emitted demand rate against a Croston-style network that emits size
establishes        and interval separately and combines them as a RATIO, both evaluated
                   on a large simulated population, each at its own selected
                   configuration.  In a gradient-boosting setting, a LightGBM regressor
                   trained directly on the full feature set against a LightGBM classifier
                   times a Tweedie regressor -- the PRODUCT form -- under identical
                   preprocessing, feature construction and evaluation protocols, on
                   ~1.4M real observations, with capacity and training budget across the
                   two arms not reported.  Occurrence and size have also been modelled
                   jointly as a deep renewal process.
core references    [Kou13]  [NAR26]  [TJWC21]  [GDTP25];  [MC26] in a footnote, labelled
                   as a preprint
positioning        Both precedents are conceded, now attributed rather than declared:
implication        "These studies provide clear precedents for both comparisons ...
                   Accordingly, neither decomposition itself nor the direct-versus-
                   factorized comparison is the focus of the contribution reported here."
what v2 moved      [Kou13]'s inversion-bias correction, its (I,H) grid and selection
out of prose       criterion, and its inventory-metric conclusion -- all retained in
                   related_work_v2_reference_map.csv and
                   kou13_representation_verification.md.
what v2 added      one bounding clause: capacity and training budget across [NAR26]'s two
                   arms are NOT REPORTED.  This prevents "identical preprocessing,
                   feature construction and evaluation protocols" from being over-read as
                   a fully matched experiment.
guards             LIT-W-KOU13 ratio, never "hurdle".
                   LIT-W-NAR26 gradient-boosting, never neural; "not reported", never
                   "not matched".
```

## 2.5 Positioning of the Present Study — 265 words *(two wording edits)*

```
reader question    "Given all of that, what is this paper for?"
what the section   Restates the two streams as two different questions -- what dependence
does               does to performance with the representation fixed, and which
                   representation performs better on a given population -- and names a
                   third: how the RELATIVE behaviour of the two representations CHANGES
                   as temporal dependence VARIES.
core references    none new; refers back to 2.3 and 2.4
the design, as     occurrence dependence and magnitude dependence varied along TWO
stated             SEPARATE AXES; marginals held fixed AS AN EXPERIMENTAL CONTROL, stated
                   as a property of the design rather than a result; same input history,
                   one backbone family, one parameter budget, one training procedure and
                   budget, one target -- hence a finite-sample difference rather than a
                   representational one.
the second half    how far the patterns reach into observed demand, reported as a result
                   in its own right: empirical analogue, predictive selector on unseen
                   series, and what does not survive adjustment.
v2 edits           "moves as temporal dependence changes" -> "changes as temporal
                   dependence varies"; "that of positive magnitude ... separate axes" ->
                   "the temporal dependence of positive magnitude ... two separate axes".
absence claims     0.  The contrast with 2.3 and 2.4 is left to the reader.
ceiling            below LEVEL A of novelty_wording_options.md.
```

---

## Totals

```
2.1  247 w     2.2  234 w     2.3  276 w     2.4  333 w     2.5  265 w
TOTAL 1,355 words of prose, 50 sentences, 21 literature-factual
citations: 20 in-text occurrences + 1 footnote, 12 unique keys
reference map: 21 rows, unmapped 0
```

## How the section reaches its positioning without an absence claim *(unchanged in v2)*

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
