# Related Work — outline only

**Revised 2026-08-12 (second build).** 2.4 now carries [NAR26] as a *verified*
precedent for the direct-versus-product-form comparison (component 6a = PRIOR), not as a
mention; 2.5 is rewritten around three prior strands rather than two. **No prose was
drafted**; this is headings, records and the single job each subsection does.

---

## 2.1 — Classical decomposition

```
records   [Cro72]  [SB05]  [TSB11]
job       establish that the occurrence/magnitude decomposition is the field's default
          and is fifty years old.
must say  Croston separates size and interval and divides them; SBA corrects the
          resulting bias with (1 - alpha/2); TSB updates the occurrence PROBABILITY every
          period instead of the interval.
must say  TSB's probability form is the closest classical ancestor of this paper's
          factorized arm.  Saying so costs nothing and pre-empts the question.
must not  present decomposition, or the probability parameterization, as our idea.
```

## 2.2 — Intermittency classification

```
records   [SBC05]  [KH06]
job       the paper's premise: the standard descriptors summarize marginals and do not
          retain temporal ordering.
must say  the rules are stated in ADI and CV^2; KH06 corrects the boundary.
must not  criticise the scheme.  It is used as-is.  The point is what it does not encode.
```

## 2.3 — Temporal dependence

```
records   [ALR12]  [WSS04]
job       concession one.  A reviewer checks here whether we know the field.
must say  ALR12 varies interval autocorrelation, size autocorrelation and their
          cross-correlation in generated demand, and reports effects on forecast accuracy
          and on service level and cost, strengthening with intermittency.
must say  WSS04 builds a two-state Markov occurrence model into a bootstrap -- an
          estimator that exploits occurrence dependence rather than assuming it away.
must say  what neither does: contrast two representations.  ALR12's compared objects are
          Croston-family estimators inside one already-factorized form.
must not  understate ALR12.
must not  describe what ALR12 holds constant.  LIT-C5 is UNRESOLVED and LIT-W3 is OPEN;
          no sentence may assert its control design until the full text is read.  The
          prohibition is checked mechanically (NBF5).
```

## 2.4 — Neural and two-part / decomposed approaches

```
records   [Kou13]  [TJWC21]  [NAR26]  [GDTP25]   and, labelled as a preprint, [MC26]*
job       concession two, and the one that fixes our contribution wording.
must say  Kou13 compares NN-Rate, which outputs the demand rate directly, against
          NN-Dual, which outputs size and interval and DIVIDES them, on 1000 simulated
          series -- a genuine neural direct-versus-decomposed precedent.
must say  its qualifications, plainly: one generated population with no dependence
          factor; each arm reported at its own best (I,H); NN-Dual carries an inversion
          bias removed by a fitted coefficient.
must say  NAR26 compares a single-stage LightGBM regressor trained directly on the
          full feature set against a two-stage model -- a LightGBM classifier for
          P(non-zero) times a Tweedie LightGBM regressor for the conditional size --
          under "identical data preprocessing, feature construction, and evaluation
          protocols", on ~1.4M real observations.  This is the SAME PAIR OF FORMS this
          paper compares, and the precedent is conceded without qualification.
must say  what NAR26 does NOT state: matched capacity, matched training budget, or
          whether hyperparameters were tuned per formulation (NAR-E/F/G).  Report these
          as NOT STATED, never as "not matched".
must say  what NAR26 verifiably lacks: any dependence factor or dependence breakdown
          (NAR-H), any synthetic study, any neural model.
must say  TJWC21 unifies occurrence and size in a deep renewal process, with illustrative
          synthetic patterns including alternating inter-demand times.
must say  GDTP25 is the current ML review.
must say  MC26 is an arXiv preprint, stated as such.
must not  call NN-Dual a hurdle, or write "the same two representations" --
          LIT-W-KOU13.
must not  call NAR26 neural, or read its feature match as a capacity match --
          LIT-W-NAR26.
must not  claim the direct-versus-decomposed comparison is new.
must not  claim the direct-versus-product-form comparison is new.
```

## 2.5 — Positioning

```
records   none new; refers back to 2.3 and 2.4
job       define the present question at the intersection of the two preceding
          subsections.
structure ALR12  -> the DEPENDENCE question:      does serial dependence change
                                                   forecasting and inventory outcomes?
          Kou13  -> the REPRESENTATION question:  direct rate, or size and interval
                                                   combined as a ratio?
          NAR26  -> the FACTORIZATION question:   direct, or occurrence probability
                                                   times conditional size?
          ours   -> the representation question ASKED AS A FUNCTION OF the dependence
                    question, with occurrence and magnitude on separate axes, both arms
                    held to one capacity and training budget -- plus how far the answer
                    survives in real demand.
must say  fixed marginals are an EXPERIMENTAL CONTROL, named as one.  Never a claim.
must say  the matched budget as a property of OUR design, never as something the prior
          work lacks -- NAR26's match status is UNKNOWN, not absent (component 6b).
must say  the empirical boundary, so the section does not read as a novelty pitch.
must not  "first", "no prior work", "first to combine", "optimal representation".
must not  present the matched comparison as standalone novelty (6b is
          CLAIM_ONLY_IN_CONJUNCTION).
wording   LEVEL A from novelty_wording_options.md.
```

---

## Ordering rationale

2.1 → 2.2 → 2.3 → 2.4 → 2.5 moves from the representation, to how series are described,
to what that description misses, to who has already compared representations, to the
remaining question. Both concessions ([ALR12] in 2.3, [Kou13] in 2.4) land **before** 2.5,
which is the order that makes 2.5 credible rather than defensive.

## Length guidance

```
2.1  short        settled background
2.2  short        a premise, not a debate
2.3  longest      concession one; a reviewer will test it
2.4  long         concession two; contains the decisive prior comparison
2.5  one para     the positioning, stated once
```

## Changes from the previous outline

```
RW1..RW5 renumbered to 2.1..2.5 to match final_outline_freeze.md section 2
2.1  TSB's probability form added as the classical ancestor of our factorized arm
2.3  explicit prohibition on describing ALR12's control design while LIT-W3 is open
2.4  NN-Dual's ratio form, inversion bias and per-arm tuning made mandatory content;
     NAR26's exact two arms quoted, and its three NOT STATED items made mandatory
2.5  restructured around THREE strands -- dependence, representation, factorization --
     with fixed marginals demoted to an experimental control and the matched budget
     demoted to a design property
```

## Out of scope for this file

No sentences, no bibliography formatting, no `.bib`, no claims about our own results.
Those belong to the drafting step that follows.
