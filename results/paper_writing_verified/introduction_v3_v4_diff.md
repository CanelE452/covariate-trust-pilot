# introduction_v3 → v4

`introduction_v1.md`, `v2.md`, `v3.md` and `contributions_v1..v3.md` are retained
unmodified. `contributions_v4` was not created: the contributions text is unchanged, so
`contributions_v3.md` remains current.

This revision touches **one paragraph**. Verified by direct string comparison, not by
inspection.

---

## Paragraph-level

```
        v3 words   v4 words   substantive change
P1        128        128      none -- byte-identical
P2        156        209      citations resolved; gap sentence narrowed
P3         96         96      none -- byte-identical
P4        156        156      none -- byte-identical
P5        114        114      none -- byte-identical
P6        112        112      none -- byte-identical
P7        108        108      none -- byte-identical
```

Six of seven paragraphs are byte-identical to v3. The seven-paragraph flow, the
paragraph roles and the paragraph order are unchanged.

---

## P2 — what changed

### Resolved placeholder 1

```
v3   "...the standard categorization scheme [CITATION NEEDED]."
v4   "...the standard categorization scheme [SBC05; KH06]."
```

No claim change. [SBC05] is the categorization paper; [KH06] is the note correcting its
boundary.

### Resolved placeholder 2, and split into three attributions

```
v3   "Temporal dependence in intermittent demand is not a new subject: autocorrelation
      in demand intervals, autocorrelation in demand sizes, and dependence between the
      two have all been studied, and estimators have been proposed that exploit them
      [CITATION NEEDED]."

v4   "Temporal dependence in intermittent demand is not a new subject. Interval
      autocorrelation, size autocorrelation and dependence between the two have been
      varied in simulated demand and shown to change forecast and inventory performance
      [ALR12]; estimators have been built that exploit occurrence dependence rather
      than assume it away [WSS04]; and neural forecasters have been compared in direct
      and decomposed form on simulated series [Kou13]."
```

Three changes of substance, all of them concessions:

```
"have all been studied"          -> a specific result: dependence was VARIED in
                                    simulated demand and CHANGED forecast and inventory
                                    performance [ALR12].  A stronger concession.
"estimators ... exploit them"    -> narrowed to OCCURRENCE dependence, because [WSS04]
                                    is the estimator actually verified.  Claiming
                                    size-autocorrelation estimators would need a record
                                    this audit did not obtain.
(new clause)                     -> [Kou13] added.  It was not anticipated by the v3
                                    freeze.  It concedes that direct-vs-decomposed
                                    NEURAL comparison on simulated intermittent series
                                    already exists.
```

### Narrowed the gap sentence

```
v3   "What has not been isolated is whether such dependence should change the
      *representation* a forecaster uses, rather than the estimator applied within it."

v4   same sentence, followed by:
     "Prior comparisons of the two representations tune each separately on a single
      generated population, rather than holding the budget fixed across a controlled
      range of dependence structures."
```

v3's sentence could be read as claiming nobody had compared the two representations.
[Kou13] refutes that reading. v4 states instead what prior comparisons *did* — a claim a
reviewer can check against [Kou13] section 3.1 and 3.4 — so the gap now rests on three
verifiable properties rather than on an absence.

**This is a reduction.** v4 asserts less than v3 did.

---

## Unchanged across v3 → v4

```
seven-paragraph structure and paragraph roles
three numeric results: "eighteen-cell", "-19.8%", "+11.87 percentage points"
P7 contains no digits
H3 absent from the Introduction
C_neg / C_pos and C_sign absent
contributions C1 / C2 / C3 -- contributions_v3.md is current and untouched
no aggregate tie or equivalence claim
no causal wording for scale
no universal winner, no reliable router asserted
no new experiment, training or scoring
```

---

## What did NOT get added, deliberately

```
[NAR26] as a supporting citation in P1
        It reports that two-stage hurdle models do not outperform a single-stage model
        on 1.4M real observations, which is consistent with P1's "neither formulation is
        uniformly superior in aggregate".  Not added: P1 is frozen, and the claim
        already rests on our own artifacts.  Recorded so the omission is a decision.

[Cro72] [SB05] [TSB11] anywhere in the Introduction
        They belong to Related Work RW1 and to the Methods baselines, not to a
        seven-paragraph Introduction that already carries five citations.

[MC26] anywhere at all
        arXiv preprint.  Used in the audit as evidence of current practice; not cited
        in the manuscript text.
```

---

## Nothing unexpected

The one outcome that was not predicted by the v3 freeze is [Kou13]. The freeze expected
placeholder 2 to resolve to "prior temporal-dependence work"; it also turned out to
require conceding a prior *representation* comparison. That concession is the reason P2
grew, and it is the most important single result of the citation step.
