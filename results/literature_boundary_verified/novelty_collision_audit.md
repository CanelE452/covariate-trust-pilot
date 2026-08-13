# Novelty collision audit

**Revised 2026-08-12 (second build).** Three gradings have now been corrected across
two revisions. The first revision fixed two:

```
1  it treated "marginals held fixed while dependence varies" as a novelty component.
   It is an EXPERIMENTAL CONTROL.  Removed from every novelty claim.
2  it called [Kou13]'s NN-Dual "decomposed" and stopped there, which let a reader
   equate it with this paper's occurrence-probability x magnitude hurdle.  It is a
   Croston-style size / interval ratio.  Corrected; see
   kou13_representation_verification.md.
```

The second revision fixed a third, and it changed a verdict:

```
3  it recorded [NAR26] as "matched features and model family", read that as a partial
   capacity match, and graded the paper N2.  Full-text re-read: the article does NOT
   state a capacity match, a training-budget match, or per-formulation tuning
   (NAR-E/F/G).  What it does state is a genuine direct-versus-product-form comparison
   at an identical feature set.  [NAR26] is therefore regraded N2 -> N3, and the
   "matched" fields are recorded as UNKNOWN rather than partial.
   See nar26_matched_comparison_verification.md.
```

Component-level grading now lives in `precedent_intersection_map.md` under two separate
fields, evidence and policy; the evidence table is `literature_evidence_matrix.md`. This
file grades **papers** and decides the collision verdict.

---

## Grading scale

```
N0  unrelated
N1  same field, different question.  Cite as background.
N2  overlaps one component.  Cite and concede that component.
N3  strong partial overlap, but the paper's intersection remains distinct.
N4  the paper's intersection has substantially already been performed.  STOP.
```

---

## Verdict

```
N4 count           0
N3                 [ALR12]  [Kou13]  [NAR26]      -- [NAR26] upgraded from N2
N2                 [WSS04]  [TJWC21]
N1                 [SBC05] [KH06] [Cro72] [SB05] [TSB11] [GDTP25] [MC26]*
N0                 none listed
```

**No collision.** Detail below, then the reasoning for each N3 remaining N3.

---

## [Kou13] — re-assessed, stays N3

Kourentzes (2013), IJPE 143(1) 198-206. **Full text verified** (28 pp, sections 2, 3.1,
3.4, 4). Formulation detail in `kou13_representation_verification.md`.

**Conceded, without qualification: a neural direct-versus-decomposed precedent exists.**
NN-Rate outputs the demand rate from a single linear node. NN-Dual outputs size and
interval from two linear nodes. Both are compared, on 1000 simulated intermittent series,
in the same tables. This paper does not claim that comparison as new.

**What keeps it at N3 rather than N4, in order of strength.**

```
1  no dependence factor.  Section 3.1 builds ONE simulated population from marginals
   fitted to the [SB05] automotive data.  There is no factorial, no sweep, no
   dependence level.  The representation question is never asked AS A FUNCTION OF
   anything.
2  not a matched budget.  Section 3.4: I and H each range over 1..3 and "only the best
   performing parameters of each model will be presented ... The criteria for selecting
   the model is minimum in-sample mean MAE rank."  Each arm is reported at its own best
   configuration.  Independently, NN-Dual has two output nodes and NN-Rate one, so even
   at identical (I,H) the two differ by H + 1 weights.
3  a different decomposition.  NN-Dual computes z'/x' -- size DIVIDED BY interval --
   and then removes the resulting inversion bias with a fitted coefficient c.  This
   paper's factorized arm computes P(Y>0|h) * E[Y|Y>0,h] -- a product of a probability
   and a conditional mean, period-indexed, with no inversion to remove.  Related
   parameterizations of one conditional mean; not the same estimator.
4  a different dependence.  Kou13's motivating dependence is CONTEMPORANEOUS
   size-interval association ("the distribution of non-zero demand changes for different
   inter-demand intervals"), fitted as a BIC-selected quadratic.  This paper's axes are
   SERIAL.
```

Point 3 is a qualification, not a defence: it means the precedent is adjacent rather
than identical. Points 1 and 2 are what actually separate the work.

---

## [ALR12] — re-assessed, stays N3

Altay, Litteral & Rudisill (2012), IJPE 135(1) 275-283. **Full-text verification
FAILED** across six routes; `LIT-W3` stays OPEN. See `alr12_fulltext_verification.md`.

**Conceded: temporal dependence in intermittent demand has been manipulated before.**
All three constructs — size autocorrelation, interval autocorrelation, size-interval
cross-correlation — are varied in generated compound Poisson demand, with effects
reported on forecast accuracy and on service level and cost, strengthening with
intermittency.

**Not conceded, and not claimed either:** whether ALR12 holds ADI, CV², zero proportion
or the size and interval distributions fixed across correlation levels is **unknown**. The
matrix records `U` and the component map records `evidence_status = UNKNOWN`. The previous
build promoted this to `PRIOR` and called the promotion conservative; it was not.
Recording UNKNOWN as PRIOR manufactures a precedent nobody verified, and it made two files
disagree about one fact.

The claim is closed off by **policy** instead: component 3 carries
`novelty_policy = EXCLUDED_FROM_NOVELTY`, because holding the marginals fixed is an
experimental control in our design rather than a finding. That holds whatever ALR12 turns
out to do. See `evidence_policy_separation.md`.

**What keeps it at N3 rather than N4.** The compared objects are Croston-family
**estimators inside a single, already-factorized representation**. There is no direct
conditional-mean arm, so there is no representation contrast to collide with. The outcome
is method accuracy and inventory performance, not the relative behaviour of two
representations. Neither is there a neural or matched-capacity element.

Confidence that a representation contrast is absent: **high**. It would be the paper's
headline and would appear in the title and abstract; neither mentions one. This is a
judgement from title and abstract, and it is labelled as such rather than as full-text
evidence.

---

## [NAR26] — upgraded to N3

Nathan et al. (2026), *Scientific Reports* 16, 4792. **Full text re-read** via PMC;
detail in `nar26_matched_comparison_verification.md`.

**Conceded without qualification: the two forms this paper compares have been compared
before.** A LightGBM regressor *"trained directly on the full feature set"* against a
two-stage model in which *"a LightGBM classifier estimates the probability of observing
non-zero demand"* and *"a LightGBM regressor with a Tweedie objective predicts the
expected demand quantity"* — a product of an occurrence probability and a conditional
size — under *"identical data preprocessing, feature construction, and evaluation
protocols"*.

That is component 6a, and it is graded **PRIOR**. Any earlier reading in which the
product-form comparison was ours is withdrawn.

**What keeps it at N3 rather than N4.**

```
1  no dependence factor.  NAR-H is verified negative: autocorrelation of intervals or of
   sizes is neither manipulated nor reported as a breakdown.  Results are dataset-wide
   aggregates across validation folds.
2  no synthetic study at all; real automotive aftermarket data only.
3  not neural.  LightGBM, XGBoost, Random Forest, Ridge, ElasticNet.
4  the match is on FEATURES and on the data and evaluation pipeline.  Capacity, training
   budget and per-formulation tuning are NOT STATED (NAR-E, NAR-F, NAR-G) -- recorded as
   UNKNOWN, not as "not matched".
```

Point 4 cuts both ways and is recorded that way: it prevents us from claiming a
matched-comparison first (component 6b is `CLAIM_ONLY_IN_CONJUNCTION`), and it prevents
[NAR26] from being credited with a match it never claims.

---

## N2

```
[TJWC21] Turkmen et al. (2021), PLOS ONE 16(11) e0259764.  Deep renewal processes;
         occurrence and size modelled jointly.  Its synthetic experiments -- periodic
         demand, ALTERNATING inter-demand times, random arrivals -- are the nearest miss
         on controlled synthetic dependence.  They are illustrative demonstrations that
         a model captures a pattern, not a factorial with fixed marginals, and there is
         no matched representation contrast.  Named rather than buried.
[WSS04]  Willemain et al. (2004).  A two-state Markov occurrence model inside a
         bootstrap: an estimator that EXPLOITS occurrence dependence.  Does not
         manipulate it, does not contrast representations.
```

## N1

```
[SBC05] [KH06]   the categorization scheme and its correction; the paper's premise
[Cro72] [SB05]   the classical size/interval lineage
[TSB11]          TSB updates the occurrence PROBABILITY every period rather than the
                 interval -- the closest classical ancestor of the hurdle
                 parameterization, and cited as such.  Not a representation comparison.
[GDTP25]         the 2025 ML review; checked for a prior statement that the question is
                 settled.  None located.
[MC26]*          arXiv preprint, labelled; hurdle decoder; context only, never cited in
                 the manuscript.
```

---

## What survives — and how it must be phrased

The claim is an intersection, not a component. Full statement in
`precedent_intersection_map.md`; wordings in `novelty_wording_options.md`.

```
SAFE (hedged):
  "We are not aware of a study that varies occurrence and magnitude dependence on
   separate axes and reads the relative finite-sample behaviour of a matched direct
   conditional mean against an occurrence-probability x positive-magnitude
   factorization."

SAFE (falsifiable, preferred in body text):
  "Prior work has asked three questions separately: [ALR12] varies dependence and
   compares estimators; [Kou13] compares a direct demand rate against a Croston-style
   size-and-interval ratio; [NAR26] compares a single-stage model against a
   probability-times-size two-stage model on real data."

REFUTED -- each by a named paper:
  "first to study temporal dependence in intermittent demand"          -> [ALR12]
  "first to control marginals while varying dependence"                -> component 3
  "first to compare direct and decomposed neural forecasters"          -> [Kou13]
  "first to compare direct against the probability x magnitude form"   -> [NAR26]
  "first MATCHED comparison of the two forms"        NOT USED: [NAR26]'s match status is
                                                     UNKNOWN, not absent (component 6b)
  "we introduce the occurrence/magnitude decomposition"                -> [Cro72]
  "we introduce a neural hurdle for demand"                            -> [NAR26], [MC26]*
  "no prior neural study compares representations"                     -> [Kou13]
  "prior work compares the same two representations"  <-- FALSE IN THE OTHER DIRECTION:
        it would equate [Kou13]'s size/interval ratio with our probability x magnitude
        product.  See LIT-W-KOU13.
```

"First" is not used. "First to combine" is not used either: recall is unquantified, and
a combination claim is as unfalsifiable as a precedence claim.

---

## Residual risk

```
R1  [ALR12] full text unobtained.  Handled by grading against ourselves; the outcome of
    LIT-W3 cannot weaken any claim we make.  OPEN.
R2  2025-2026 preprints thinly indexed; a same-idea preprint cannot be excluded.  This
    is why absence claims are hedged with "we are not aware of".
R3  no Scopus / Web of Science sweep; recall unquantified.
R4  [ALR12]'s "no representation contrast" rests on title and abstract, at high
    confidence but not on methodology.  Labelled at every use.
R5  [NAR26]'s capacity and training match is UNKNOWN, not absent.  If a later reading
    establishes that it WAS matched, component 6b moves from NOT_FOUND_IN_AUDIT to
    PRIOR.  The intersection is unaffected: components 7, 8 and 9 do not depend on 6b,
    which is why 6b is CLAIM_ONLY_IN_CONJUNCTION.
```
