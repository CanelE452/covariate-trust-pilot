# Citation resolution

**Revised 2026-08-12 (second build).** The v3 → v4 step resolved the two
`[CITATION NEEDED]` placeholders. The v4 → v5 step corrected two descriptions and
shortened the paragraph. The v5 → v6 step adds a **fifth precedent citation** that the
earlier builds had graded too weakly to cite.

---

## Placeholder 1 — the categorization scheme *(unchanged since v4)*

```
v3   "...the two statistics behind the standard categorization scheme [CITATION NEEDED]."
v5   "...the average demand interval and the squared coefficient of variation of demand
      size [SBC05; KH06]."
```

[SBC05] Syntetos, Boylan & Croston (2005), JORS 56(5) 495-503 — the categorization rules.
[KH06] Kostenko & Hyndman (2006), JORS 57(10) 1256-1257 — the correction.

No claim changed. v5 drops the phrase "the two statistics behind the standard
categorization scheme" only for length; the citation carries it.

---

## Placeholder 2 — prior temporal-dependence work *(corrected in v5)*

v4 resolved it to three citations. v5 keeps all three and fixes **how two of them are
described**.

### Correction 1 — [Kou13] was described too loosely

```
v4   "...and neural forecasters have been compared in direct and decomposed form on
      simulated series [Kou13]."
v5   "Representation has been examined too: neural work compares a directly predicted
      demand rate against a Croston-style network that forecasts size and interval and
      divides them [Kou13]."
```

**Why.** "Decomposed" is true but uninformative, and a reader carrying it forward into P3
and P4 would take [Kou13]'s NN-Dual for this paper's Hurdle. They are different objects:

```
[Kou13] NN-Dual      z' / x'         size DIVIDED BY interval; inversion bias present
                                     and removed afterwards by a fitted coefficient
this paper's Hurdle  P(Y>0) * E[Y|Y>0]   probability TIMES conditional mean; period-
                                     indexed; no inversion
```

Registered as **LIT-W-KOU13**; evidence in `kou13_representation_verification.md`
(full text verified).

The concession is unchanged in force: a neural direct-versus-decomposed precedent exists,
and v5 gives it its own sentence.

### Correction 2 — the gap sentence no longer leans on "matched budget" alone

```
v4   "Prior comparisons of the two representations tune each separately on a single
      generated population, rather than holding the budget fixed across a controlled
      range of dependence structures."
v5   "These two lines of work answer different questions. Neither isolates how a direct
      conditional mean and an occurrence-probability x positive-magnitude factorization
      behave, at matched capacity, as occurrence and magnitude dependence are varied
      separately."
```

**Why.** Two reasons.

```
1  "the two representations" repeated the LIT-W-KOU13 error in the gap sentence itself.
   v5 names the factorization by its product form, so the sentence cannot be read as
   claiming [Kou13] compared what we compare.
2  the matched budget alone will not carry the claim.  [NAR26] compares single-stage
   against two-stage HURDLE models at identical features and the same model family on
   1.4M real observations.  So "matched direct-vs-hurdle comparison" is PARTIAL_OVERLAP,
   not new.  v5 therefore rests the gap on the CROSSING -- separately varied occurrence
   and magnitude dependence -- with matched capacity as one condition among several.
```

`representation_x_dependence_interaction` is empty for every prior row in the matrix;
that column, not the budget, is what the sentence stands on.

### What v5 no longer implies

```
withdrawn   any suggestion that holding the marginals fixed is what is new.
            v4's P2 carried a 44-word sentence about matching interval support and the
            positive-demand distribution; v5 compresses it to a clause -- "two series
            can match on both and still differ in whether gaps cluster or alternate" --
            which states the motivation without inviting a novelty reading.
            Fixed marginals are an EXPERIMENTAL CONTROL.  See claim_literature_audit
            LIT-C5 and WARN_FAIL LIT-W5.
```

---

---

## v5 → v6 — one citation added, one description corrected

### Added: [NAR26], because component 6a is PRIOR on verified full text

```
v5   "...neural work compares a directly predicted demand rate against a Croston-style
      network that forecasts size and interval and divides them [Kou13]."

v6   "...neural work compares a directly predicted demand rate against a Croston-style
      representation that predicts non-zero demand size and inter-demand interval and
      combines them as a ratio [Kou13], and single-stage models against two-stage models
      that multiply an occurrence probability by a conditional size [NAR26]."
```

**Why.** The full-text re-read established that [NAR26] compares *"a LightGBM regressor
… trained directly on the full feature set"* against *"a LightGBM classifier [that]
estimates the probability of observing non-zero demand"* times *"a LightGBM regressor
with a Tweedie objective"* — the **same pair of forms this paper compares** — under
identical preprocessing, feature construction and evaluation protocols.

Without this clause, v5's gap sentence was reachable by a reviewer holding [NAR26]: it
named only a ratio-form precedent, so a reader could infer that the product-form
comparison was ours. It is not. **Claim reduced.**

### Corrected: [Kou13] described in the source's own terms

`divides them` → `predicts non-zero demand size and inter-demand interval and combines
them as a ratio`. Same fact, stated the way the paper states it, and it survives being
quoted out of context.

### Removed: [WSS04] from P2

Moved to Related Work 2.3. Its concession — that estimators exploit occurrence dependence
— does not touch this paper's claim, whereas [NAR26]'s does. Nothing in P2 became uncited
by the removal: the clause it supported was removed with it.

### Not changed: the ALR12 sentence

v6 still says only that simulation work varying the three dependence types *reports
effects on forecast accuracy and inventory performance*. It says nothing about what
[ALR12] holds constant, because `LIT-C5` is UNRESOLVED. Checked mechanically by NBF5.

---

## Cost accounting

```
                        v3     v4     v5     v6
P2 words               156    209    165    167
citations in P2          0      5      5      5
[CITATION NEEDED]        2      0      0      0
numeric results in P2    0      0      0      0
paragraphs changed      P7     P2     P2     P2
```

The citation count is unchanged at five: [NAR26] entered as [WSS04] left.

LIT-W4 was closed at v5 and briefly reopened at v6, then closed at 167 words. The two
words over the v5 band are stated rather than shaved: a fourth precedent was added on
verified evidence, and dropping it to hit a self-imposed number would have been the wrong
trade.

---

## Not done here

No Related Work prose. No Methods, Results or Discussion text. No `.bib` file — the twelve
records live in `core_reference_list.md` and `reference_metadata.csv` until the manuscript
needs one. `contributions_v3.md` was not edited; its C1 wording is already LEVEL-B
compatible and claims no precedence. No new literature search was run in this build: the
[NAR26] evidence came from re-reading a source already in the reference list.
