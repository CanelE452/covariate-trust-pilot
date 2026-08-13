# Related Work v2 — claim audit

Sentence-level audit of `related_work_v2.md`. Supersedes `related_work_claim_audit.md`,
which is retained for v1. Mapping is in `related_work_v2_reference_map.csv` (21 rows,
unmapped 0).

Sections 2.1, 2.2 and 2.3 are **byte-identical to v1**, so their tags carry over unchanged
and are summarized rather than restated. Only 2.4 and 2.5 were re-derived.

```
LITERATURE_SUPPORTED   a verified record backs it, and it is cited
BOUNDARY_WORDING       supported, and the sentence exists to limit a claim
POSITIONING            a statement about THIS study or about section structure
OVERCLAIM              wording exceeds the evidence          -- must be 0
UNSUPPORTED            no verified record backs it           -- must be 0
```

---

## 2.1 / 2.2 / 2.3 — carried over unchanged *(757 words, 30 sentences)*

```
2.1  10 sentences   LITERATURE_SUPPORTED 5  [Cro72] x2, [SB05], [TSB11] x2
                    BOUNDARY_WORDING 4      POSITIONING 1
2.2   9 sentences   LITERATURE_SUPPORTED 3  [SBC05] x2, [KH06]
                    BOUNDARY_WORDING 3      POSITIONING 3
2.3  11 sentences   LITERATURE_SUPPORTED 5  [WSS04], [ALR12] x3, [ALR12]+[WSS04]
                    BOUNDARY_WORDING 3      POSITIONING 3
```

The `LIT-W3` scope note from the v1 audit stands in full: 2.3 states only which dependence
structures `[ALR12]` varies, that the demand is generated, and what outcomes are reported.
**No sentence states what that study holds constant.** Verified mechanically (RWF11).

---

## 2.4 Neural and Two-Part Forecasting Formulations *(333 words, 11 sentences)*

```
P1S1  ML and neural methods form a substantial literature of their own,
      recently surveyed; within it, whether a forecast should be one
      quantity or two has been asked directly and in more than one form
                                                     LITERATURE_SUPPORTED
                                                     [GDTP25]; [Kou13; NAR26]
P2S1  Kourentzes compares two architectures differing in exactly this
      respect                                        LITERATURE_SUPPORTED  [Kou13]
P2S2  both take lagged non-zero demands and inter-demand intervals as
      inputs                                         LITERATURE_SUPPORTED  [Kou13]
P2S3  one emits the demand rate from a single output; the other emits
      size and interval separately and combines them as a RATIO, in the
      manner of Croston's method                     LITERATURE_SUPPORTED  [Kou13]
      *** LIT-W-KOU13.  The load-bearing sentence of the subsection.  Its citation
      was detached during compression and restored before the map was accepted;
      see related_work_v1_v2_diff.md. ***
P2S4  evaluated over a large simulated population parameterized from a
      real spare-parts dataset, each at its own selected configuration;
      the two rank differently under accuracy than under inventory
      metrics                                        LITERATURE_SUPPORTED  [Kou13]
P3S1  the other pairing -- direct conditional mean against occurrence
      probability multiplied by conditional size -- has been compared in
      a gradient-boosting rather than a neural setting
                                                     LITERATURE_SUPPORTED  [NAR26]
      *** LIT-W-NAR26.  "gradient-boosting rather than a neural setting" is the
      non-neural marker and must survive any future trim. ***
P3S2  ~1.4M monthly observations; a LightGBM regressor trained directly
      on the full feature set against a two-stage model pairing a
      LightGBM classifier for P(non-zero) with a Tweedie-objective
      regressor for the conditional quantity, both under identical data
      preprocessing, feature construction and evaluation protocols;
      capacity and training budget across the two arms are NOT REPORTED
                                                     LITERATURE_SUPPORTED  [NAR26]
      *** NEW IN v2.  The clause bounds what is matched.  "not reported" is the
      article's evidence status (NAR-E/F/G NOT STATED = UNKNOWN).  It is NOT
      "not matched", and it is NOT phrased as a deficiency relative to this study.
      Checked mechanically (RWF10). ***
P3S3  the reported conclusion is that the two-stage form's added
      complexity does not yield an aggregate advantage once informative
      features are supplied                          LITERATURE_SUPPORTED  [NAR26]
                                                     attributed ("the reported
                                                     conclusion is")
P3S4  occurrence and size have also been modelled jointly as a deep
      renewal process capturing regular and alternating inter-arrival
      structure on constructed patterns              LITERATURE_SUPPORTED  [TJWC21]
P4S1  these studies provide clear precedents for both comparisons: a
      direct rate against a Croston-style ratio in a neural setting, and
      a direct conditional mean against a probability-times-size product
      on real data                                   LITERATURE_SUPPORTED
                                                     [Kou13] [NAR26]
      *** v2 rewording: "Both comparisons therefore already exist" -> "These studies
      provide clear precedents".  Same concession, attributed rather than declared. ***
P4S2  accordingly, neither decomposition itself nor the direct-versus-
      factorized comparison is the focus of the contribution reported
      here                                           BOUNDARY_WORDING
      *** the load-bearing concession.  v2 names BOTH conceded objects explicitly
      (decomposition, and the comparison), where v1 named the formulation twice. ***
FN1   a mixture-of-experts encoder with a hurdle decoder, labelled in
      the footnote as an arXiv preprint cited for current practice
      rather than as peer-reviewed evidence          LITERATURE_SUPPORTED  [MC26]
```

## 2.5 Positioning of the Present Study *(265 words, 9 sentences)*

```
P1S1  the two streams answer different questions                POSITIONING
P1S2  the dependence stream asks what happens to forecasting and
      inventory performance when serial structure changes, with
      the representation held fixed                             BOUNDARY_WORDING
P1S3  the representation stream asks which of two ways of
      structuring a forecast performs better on a given
      population                                                BOUNDARY_WORDING
P2S1  the present study focuses on a different controlled
      intersection: how the RELATIVE behaviour of the two
      representations CHANGES as temporal dependence VARIES     POSITIONING
      *** v2: "moves ... changes" -> "changes ... varies". ***
P2S2  the two formulations are compared while the temporal
      dependence of occurrence and the temporal dependence of
      positive magnitude are varied along TWO SEPARATE AXES,
      with the marginals held fixed as an experimental control  POSITIONING
      *** v2 spells out both axes rather than using "that of". ***
P2S3  that control is a property of the design rather than a
      result                                                    POSITIONING
P2S4  same input history, one backbone family, one parameter
      budget, one training procedure and budget, one target --
      so the difference is finite-sample behaviour under a
      fixed budget rather than representational capacity        POSITIONING
      *** "matched" appears only as a property of OUR design.  Nothing is said about
      whether prior work matched -- component 6b is CLAIM_ONLY_IN_CONJUNCTION. ***
P3S1  the second half asks how far the patterns reach into
      observed demand and reports that boundary as a result:
      analogue, predictive selector, and what does not survive
      adjustment                                                POSITIONING
```

---

## Totals

```
                        v1     v2
LITERATURE_SUPPORTED    22     21
BOUNDARY_WORDING        14     13
POSITIONING             16     16
METADATA_UNRESOLVED      0      0
OVERCLAIM                0      0
UNSUPPORTED              0      0
sentences               52     50
```

The single reduction in each supported category is the removed `[Kou13]`
inventory-conclusion sentence and the merged `[NAR26]` setting marker. No concession was
lost; see `related_work_v1_v2_diff.md` for the 10/10 retention check.

---

## Absence-claim scan

```
"first" / "first to"                        0
"no prior work" / "no previous"             0
"nobody has" / "has never been"             0
"unexplored" / "remains unexplored"         0
"novel combination" / "never been combined" 0
"asymmetr*" anywhere in Related Work        0
prior work cannot explain / does not
  predict our result                        0
```

The section still makes **no negative existential statement of any scope**, and the word
*asymmetry* does not appear — that result belongs to Section 4. Verified mechanically
(RWF16, RWF17).

---

## Guards discharged in v2

```
LIT-W3        2.3 unchanged; nothing stated about what [ALR12] holds constant.  RWF11.
LIT-W-KOU13   "combines them as a ratio" retained with its citation; "hurdle" never
              attached to [Kou13]; "the same two representations" absent.  RWF6, RWF7.
LIT-W-NAR26   "gradient-boosting rather than a neural setting" retained; LightGBM named
              three times; the new clause says "not reported", never "not matched".
              RWF8, RWF9, RWF10.
MC26          footnote only, labelled as a preprint.
component 6b  "matched" appears in 2.5 only as a property of this study's design.
```

---

## Reviewer reverse validation

```
Q1   Is decomposition itself the novelty?          NO.  2.1 P2S2 and 2.4 P4S2.
Q2   Is temporal dependence itself the novelty?    NO.  2.3 closes with "established
                                                   territory ... no part of the present
                                                   study is positioned as introducing it."
Q3   Is direct-vs-decomposed comparison itself
     the novelty?                                  NO.  2.4 P4S1-S2.
Q4   Is probability x size two-stage itself the
     novelty?                                      NO.  2.4 P3, conceded to [NAR26].
Q5   What is [Kou13]'s formulation?                Size and inter-demand interval emitted
                                                   separately and combined as a RATIO, in
                                                   the manner of Croston's method.
Q6   What is [NAR26]'s formulation?                A direct LightGBM regressor against a
                                                   LightGBM classifier for P(non-zero)
                                                   times a Tweedie regressor for the
                                                   conditional size -- a PRODUCT.
Q7   Is [NAR26] a matched-capacity experiment?     NOT ESTABLISHED.  Capacity and training
                                                   budget across the two arms are not
                                                   reported.  UNKNOWN, not negative.
Q8   Did [ALR12] fix the marginals?                Unresolved (LIT-W3 OPEN).  No sentence
                                                   in the manuscript relies on it.
Q9   What does the present study actually ask?     How the relative finite-sample
                                                   behaviour of the two representations
                                                   changes as occurrence and magnitude
                                                   dependence vary along two separate
                                                   axes, under matched conditions.
Q10  What does the real data do?                   Establishes the transfer boundary of
                                                   the controlled synthetic pattern:
                                                   analogue, predictive selector, and what
                                                   does not survive adjustment.
Q11  Is a "first" claim needed?                    NO.  The section reaches its
                                                   positioning by conceding four things,
                                                   not by declaring anything missing.
```

All eleven read as intended.

---

## Novelty ceiling

The strongest statement in v2 remains 2.5 P2S1: *"The present study focuses on a different
controlled intersection of these questions."* That is below LEVEL A of
`../literature_boundary_verified/novelty_wording_options.md`, which was the ceiling for
this step. LEVEL A itself is reserved for the abstract and the contributions list and does
not appear in Related Work.
