# introduction_v4 → v5

`introduction_v1.md` … `v4.md` and `contributions_v1..v3.md` are retained unmodified.
`contributions_v4` was not created: C1/C2/C3 are unchanged, so `contributions_v3.md`
remains current.

One paragraph changed. Verified by direct string comparison.

---

## Paragraph-level

```
        v4 words   v5 words   substantive change
P1        128        128      none -- byte-identical
P2        209        165      rewritten; see below
P3         96         96      none -- byte-identical
P4        156        156      none -- byte-identical
P5        114        114      none -- byte-identical
P6        112        112      none -- byte-identical
P7        108        108      none -- byte-identical
```

**P1 and P3–P7 substantive story: preserved, no diff at all.** No source contradiction
arose that would require touching them. P2 is now 165 words, inside the 130–165 target,
and is no longer the longest paragraph.

---

## P2 — four changes, three of which reduce what is claimed

### 1. Fixed marginals demoted from an implied novelty to a control

```
v4   "Both are functionals of a marginal distribution, and neither is affected by the
      order in which intervals and sizes arrive. Two series can share their average
      interval, their interval support and their positive-demand distribution and still
      differ in whether short and long gaps alternate or cluster, and in whether large
      orders tend to follow large ones."                              (2 sentences, 76 w)

v5   "Both summarize a marginal distribution, and neither retains the order in which
      intervals and sizes arrive: two series can match on both and still differ in
      whether gaps cluster or alternate."                              (1 sentence, 34 w)
```

The v4 pair reads as setting up a controlled design as the paper's distinctive move.
Holding the marginals fixed is an **experimental control**, not a finding, and the
project's own frozen sources already treat it that way — `claim_ledger_frozen.md` reader
chain steps 1–2 make it the premise of step 2, and `final_outline_freeze.md` records "no
novelty claim over the correlation literature". v5 keeps the motivation and drops the
implication. **Claim reduced.**

### 2. [ALR12] concession restated as a specific result

```
v4   "Temporal dependence in intermittent demand is not a new subject. Interval
      autocorrelation, size autocorrelation and dependence between the two have been
      varied in simulated demand and shown to change forecast and inventory performance
      [ALR12]..."

v5   "That order is already known to matter. Simulation work varying interval
      autocorrelation, size autocorrelation and the dependence between them reports
      effects on forecast accuracy and inventory performance [ALR12]..."
```

Same force, tighter, and the topic sentence now carries the concession instead of a
meta-statement about the subject. **Claim unchanged.**

### 3. [Kou13] described by what it actually does

```
v4   "...and neural forecasters have been compared in direct and decomposed form on
      simulated series [Kou13]."

v5   "Representation has been examined too: neural work compares a directly predicted
      demand rate against a Croston-style network that forecasts size and interval and
      divides them [Kou13]."
```

**This is the most important change in the revision.** Full-text verification established
that NN-Dual outputs size and interval and **divides** them (Croston's equation 1), then
removes the resulting inversion bias with a fitted coefficient. This paper's factorized
arm **multiplies** an occurrence probability by a conditional positive mean. "Decomposed"
is true but lets a reader equate the two. Registered as **LIT-W-KOU13**.

The concession is not weakened — it now has its own sentence and its own clause of
emphasis ("too"). **Precision increased, claim unchanged.**

### 4. The gap restated as an intersection, not as a budget

```
v4   "What has not been isolated is whether such dependence should change the
      *representation* a forecaster uses rather than the estimator applied within it.
      Prior comparisons of the two representations tune each separately on a single
      generated population, rather than holding the budget fixed across a controlled
      range of dependence structures."                                 (2 sentences, 56 w)

v5   "These two lines of work answer different questions. Neither isolates how a direct
      conditional mean and an occurrence-probability × positive-magnitude factorization
      behave, at matched capacity, as occurrence and magnitude dependence are varied
      separately."                                                     (2 sentences, 47 w)
```

Two defects fixed:

```
"the two representations"   repeated the LIT-W-KOU13 error inside the gap sentence.
                            v5 names the factorization by its product form.
"holding the budget fixed"  the matched budget cannot carry the claim alone: [NAR26]
                            compares single-stage against two-stage HURDLE models at
                            identical features and model family on 1.4M real
                            observations.  v5 rests the gap on the CROSSING --
                            separately varied occurrence and magnitude dependence --
                            with matched capacity as one condition among several.
```

Also: "The two strands have stayed apart" was drafted and then replaced with "These two
lines of work answer different questions", because the first is a mild absence claim over
all literature while the second is a description of two named works. **Claim reduced.**

---

## Unchanged across v4 → v5

```
seven-paragraph structure and paragraph roles
five citations in P2, the same five: [SBC05] [KH06] [ALR12] [WSS04] [Kou13]
three numeric results: "eighteen-cell", "-19.8%", "+11.87 percentage points"
P7 contains no digits
H3 absent from the Introduction
C_neg / C_pos and C_sign absent
contributions C1 / C2 / C3 -- contributions_v3.md current and untouched
no aggregate tie or equivalence claim; no causal wording for scale
no universal winner, no reliable router asserted
no new experiment, training or scoring
```

---

## What did NOT change, deliberately

```
[NAR26] still not cited in the Introduction
        It is now load-bearing in the audit -- it is why component 6 is PARTIAL_OVERLAP
        rather than new.  It belongs in Related Work 2.4, not in a seven-paragraph
        Introduction that already carries five citations.  Recorded so the omission is a
        decision, not an oversight.

contributions_v3.md unedited
        C1 reads "Controlled characterization of the finite-sample relative inductive
        bias of direct and factorized forecasting under temporal occurrence and magnitude
        dependence."  It claims no precedence, keeps "finite-sample", and matches the
        frozen ledger.  It is LEVEL-B compatible as written.

P1's "neither formulation is uniformly superior in aggregate"
        now independently consistent with [NAR26], which reports that two-stage hurdle
        models do not outperform a single-stage model.  Citation not added: P1 is frozen
        and the claim rests on our own artifacts.
```

---

## Net effect

Three of the four P2 changes **reduce** what the Introduction asserts; the fourth
increases precision without changing force. v5 is the least-claiming version of this
paragraph so far, and it is the first one whose every literature statement has been
checked against a full text or explicitly graded against us.
