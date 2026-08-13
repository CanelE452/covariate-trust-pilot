# introduction_v2 → v3

`introduction_v1.md`, `introduction_v2.md`, `contributions_v1.md` and
`contributions_v2.md` are retained unmodified.

This revision was deliberately narrow: one paragraph compressed, one clause tightened,
nothing else touched.

---

## Paragraph-level

```
        v2 words   v3 words   substantive change
P1        131        128      one clause removed, see below
P2        156        156      none -- byte-identical to v2
P3         96         96      none -- byte-identical to v2
P4        156        156      none -- byte-identical to v2
P5        114        114      none -- byte-identical to v2
P6        112        112      none -- byte-identical to v2
P7        158        108      rewritten and compressed
```

P2 through P6 were verified byte-identical to v2 by direct string comparison, not by
inspection. The seven-paragraph flow and each paragraph's logical role are unchanged.

---

## P1 — one clause removed

```
v2   "neither formulation is uniformly superior in aggregate: the two are close, and
      which one leads depends on the error measure."
v3   "neither formulation is uniformly superior in aggregate: which of the two leads
      depends on the error measure."
```

"The two are close" edges toward an equivalence claim that the artifacts do not make:
the aggregates are not equal, and the sentence does not need them to be. Removing the
clause leaves the substantive point — RMSE and MAE disagree about which formulation
leads — and drops the implied tie. This is the only change to P1–P6, and it removes a
claim rather than adding one.

---

## P7 — compressed from 158 to 108 words

**Removed.**
```
"successive changes to the training target, the loss, the aggressiveness, the
 capacity and the input representation did not recover cross-domain transfer"
        -> development chronology; belongs in Section 5.7
"the last of these improved one external domain and degraded severely on another"
        -> reduced to "one external domain degraded severely"
"A gate frozen after development was worse than a static mixture on the first
 external dataset it was applied to"
        -> reduced to "gains obtained during development did not transfer"
"deliberately choosing a less correlated pair enlarges that opportunity further"
        -> removed; the expert-diversity result is Section 5.7 material
```

**Retained**, matching the four required contents:
```
A  conditional differences invite a downstream question about a learned router
B  the forecasts make complementary errors; an origin-level oracle beats any fixed
   mixture, so the opportunity is measurable
C  development gains did not transfer across domains or time, and one external domain
   degraded severely; a pre-registered stopping rule was triggered
D  neither a universal winner nor a generally reliable router is proposed; the
   contribution is the characterization and its transfer boundary
```

No numeric result appears in P7, in v2 or in v3. Verified: the paragraph contains no
digits at all, and none of `Gate-v`, `P0L1`, `HGB`, `GRU`, `Safe-`, `sequence gate`,
`redesign`, `FreshRetailNet` or `UCI`.

---

## Unchanged across v2 → v3

```
seven-paragraph structure and paragraph roles
three numeric results: "eighteen-cell", "-19.8%", "+11.87 percentage points"
two [CITATION NEEDED] placeholders in P2
H3 absent from the Introduction
C_neg / C_pos and C_sign absent
C1 / C2 / C3 scientific content -- contributions_v3 body is byte-identical to v2
no new experiment, training or scoring
```

---

## Nothing unexpected

No substantive difference appeared in P2–P6, which is the outcome the freeze required.
The only two edits are the P1 clause removal and the P7 rewrite, both of which reduce
what the Introduction asserts rather than extend it.
