# introduction_v5 → v6

`introduction_v1.md` … `v5.md` and `contributions_v1..v3.md` are retained unmodified.
`contributions_v4` was not created: C1/C2/C3 are unchanged, so `contributions_v3.md`
remains current.

One paragraph changed. Verified by direct string comparison.

---

## Paragraph-level

```
        v5 words   v6 words   substantive change
P1        128        128      none -- byte-identical
P2        165        167      one citation added, two descriptions corrected
P3         96         96      none -- byte-identical
P4        156        156      none -- byte-identical
P5        114        114      none -- byte-identical
P6        112        112      none -- byte-identical
P7        108        108      none -- byte-identical
```

**P1 and P3–P7 substantive story: preserved, no diff at all.** No source contradiction
arose that would require touching them.

---

## Why v6 exists

Two evidence-label defects were found in the audit, not in the prose. Fixing them changed
what P2 is allowed to imply.

```
defect 1   the component map graded "controlled marginals + dependence variation" as
           PRIOR while the evidence matrix recorded the same fact as UNKNOWN.
           -> no P2 change needed.  v5 already said nothing about what [ALR12] holds
              constant, and the fixed-marginal novelty claim was already withdrawn.
              Fixed in the audit only.

defect 2   [NAR26] was recorded as "matched features and model family" and graded N2.
           Full-text re-read: it compares a direct LightGBM regressor against a
           two-stage classifier x Tweedie-regressor PRODUCT form -- the same pair of
           forms this paper compares -- under identical preprocessing, feature
           construction and evaluation protocols.
           -> component 6a becomes PRIOR, [NAR26] becomes N3, and P2 MUST acknowledge
              the precedent.  This is what v6 changes.
```

---

## P2 — three changes, all of which reduce or sharpen

### 1. The hurdle / two-stage precedent is acknowledged *(the reason for v6)*

```
v5   "...neural work compares a directly predicted demand rate against a Croston-style
      network that forecasts size and interval and divides them [Kou13]."

v6   "...neural work compares a directly predicted demand rate against a Croston-style
      representation that predicts non-zero demand size and inter-demand interval and
      combines them as a ratio [Kou13], and single-stage models against two-stage models
      that multiply an occurrence probability by a conditional size [NAR26]."
```

v5 named only a **ratio-form** precedent. A reviewer holding [NAR26] could reply that the
**product-form** comparison — exactly the pair this paper compares — has also been done,
and v5's gap sentence gave no sign that we knew. v6 says it outright. **Claim reduced.**

### 2. [Kou13] is described in the source's own terms

`forecasts size and interval and divides them` → `predicts non-zero demand size and
inter-demand interval and combines them as a ratio`. Same fact, phrased as the paper
phrases it, and it survives being quoted out of context. **Precision increased.**

### 3. The closing sentence states our design, not their absence

```
v5   "These two lines of work answer different questions. Neither isolates how a direct
      conditional mean and an occurrence-probability × positive-magnitude factorization
      behave, at matched capacity, as occurrence and magnitude dependence are varied
      separately."

v6   "These answer different questions from the comparison here, in which a direct
      conditional mean and an occurrence-probability × positive-magnitude factorization
      are held to one capacity and training budget while occurrence and magnitude
      dependence are varied separately."
```

v5's "Neither isolates …" is a negative existential over the cited works. v6 replaces it
with a positive description of what **this** comparison does, so the sentence carries no
absence claim at all — not even a bounded one. This matters more than it did in v5,
because with [NAR26] in the list the reader now knows three prior works, and a negative
statement over three is easier to misread as a statement over the field. **Claim
reduced.**

### 4. [WSS04] moved out of P2

Its concession — estimators exist that exploit occurrence dependence — does not touch this
paper's claim, while [NAR26]'s does. It is retained in Related Work 2.3. No sentence
became uncited: the clause it supported left with it. Citation count in P2 is unchanged at
five.

---

## Word count

```
v5   165
v6   167   (+2)
```

`LIT-W4` reopened and closed at 167. The two words are stated rather than shaved: a
fourth precedent was added on verified evidence, and three sentences were tightened to
absorb most of the cost. Dropping the precedent to hit a self-imposed number would have
been the wrong trade.

---

## Unchanged across v5 → v6

```
seven-paragraph structure and paragraph roles
five citations in P2
three numeric results: "eighteen-cell", "-19.8%", "+11.87 percentage points"
P7 contains no digits
H3 absent from the Introduction
C_neg / C_pos and C_sign absent
contributions C1 / C2 / C3 -- contributions_v3.md current and untouched
no aggregate tie or equivalence claim; no causal wording for scale
no universal winner, no reliable router asserted
nothing said about what [ALR12] holds constant  (LIT-C5 UNRESOLVED, NBF5 checked)
no new experiment, training or scoring
```

---

## What did NOT change, deliberately

```
the matched budget is never claimed as novel
        [NAR26]'s capacity and training match is NOT STATED -- unknown, not absent.
        Component 6b is CLAIM_ONLY_IN_CONJUNCTION, so "matched" appears in P2 as a
        property of OUR design ("are held to one capacity and training budget"), never
        as something prior work lacks.

P1's "neither formulation is uniformly superior in aggregate"
        now independently consistent with [NAR26], which reports that two-stage hurdle
        models do not outperform a single-stage model.  Citation still not added: P1 is
        frozen and the claim rests on our own artifacts.

contributions_v3.md unedited
        C1 claims no precedence, keeps "finite-sample", matches the frozen ledger, and
        is LEVEL-B compatible as written.
```

---

## Net effect

v6 asserts less than v5. It names a third precedent that touches the paper's own
comparison directly, and it removes the last negative existential from the paragraph. The
gap now rests entirely on what the study **does** — separate dependence axes, one
capacity and training budget — rather than on what anyone else did not do.
