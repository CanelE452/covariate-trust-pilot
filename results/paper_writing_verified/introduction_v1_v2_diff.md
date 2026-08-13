# introduction_v1 → v2

`introduction_v1.md` and `contributions_v1.md` are retained unmodified.

---

## Structure

```
v1   6 paragraphs.  P5 carried three roles at once:
     H1 analogue, H2 selector transfer, and the overlap-adjusted boundary.
v2   7 paragraphs.  P5 splits into P5 (transfer) and P6 (boundary);
     routing and positioning move to P7.
```

| | v1 | v2 |
|---|---|---|
| P1 | problem, opens with two aggregate figures | problem, no figures |
| P2 | gap, prior literature conceded | unchanged in role, lightly reworded |
| P3 | question, finite-sample framing | unchanged in role |
| P4 | synthetic, quotes the interaction size | synthetic, interaction stated qualitatively |
| P5 | analogue + selector + boundary | **analogue + selector only** |
| P6 | routing + positioning | **boundary only** |
| P7 | — | routing + positioning |

---

## Numbers

```
v1   11 numeric results
v2    3
```

Removed, with the reason:

```
-0.0007, -0.0320   P1 aggregate near-tie.  Replaced with "neither formulation is
                   uniformly superior in aggregate; which one leads depends on the
                   error measure."  The point is qualitative; two four-decimal
                   figures in the opening sentence read as a results table.
-16.74 pp          Stage 1 interaction.  Replaced with "their joint effect is not the
                   sum of their separate effects."  The magnitude is a Results number;
                   the Introduction needs only that an interaction exists.
+0.1064, +0.0789   H1 on both datasets.  Replaced with "positively associated ... in
                   both".  The direction is the claim; the coefficients are small and
                   inviting a reader to weigh them in the Introduction is unhelpful.
+0.1529 vs +0.0323 intermittent against lumpy.  Kept as prose ("strengthens within the
                   intermittent regime"); the lumpy null stays in Results and Table 4.
+0.0032            overlap-adjusted association.  Replaced with "no longer
                   distinguishable from zero".
4.11%, 2.15x       oracle gap and diversity multiplier.  Replaced with "meaningfully
                   better than any fixed mixture" and "enlarges that opportunity".
-2.43%             external gate result.  Replaced with "worse than a static mixture".
+2.648%, -193.9%   sequence gate.  Replaced with "improved one external domain and
                   degraded severely on another".  Routing is C3; its numeric weight
                   must not exceed C1 and C2 in the Introduction.
```

Retained:

```
eighteen-cell sweep     needed to scope the direct-favourable claim
-19.8%                  the claim turns on the magnitude; "the direct model wins"
                        without a size is uninformative
+11.87 pp               the strongest single piece of empirical transfer evidence and
                        the one a sceptical reader will want quantified
```

---

## H3

Absent from v1 and still absent from v2, now recorded as a deliberate decision rather
than an omission. Reporting it accurately requires the construct mismatch — the
synthetic contrast is d = 4 against d = 8, the external test split at the ADI median —
and that caveat does not fit the P5/P6 flow. Stating it without the caveat would read
as a plain failure and would misdescribe what was tested. It is reported in Section 5.6,
Table 4 and the limitations.

---

## Scale wording

```
v1   "the covariate responsible is scale -- an axis the controlled design does not
      contain and therefore cannot arbitrate"
v2   "candidate and control series differ substantially in scale as well as in the
      intended axis ... the isolated association is no longer distinguishable from
      zero ... the controlled study, which contains no scale axis, cannot arbitrate
      the question either way"
```

"Responsible" attributes the boundary to scale. The v2 wording states what was
measured — the groups differ on scale, and adjustment removes the association — without
asserting that scale produces the effect. "Confounded" was considered and not used: it
carries a causal-inference commitment the design does not license here.

---

## Other wording changes

```
"asymmetric geometry"     kept, but now immediately glossed: "that is, the two axes
                          respond to different features of dependence"
"exactly one" (cell)      -> "one configuration ... within the eighteen-cell sweep"
"we tried"                -> "we tested this and report the boundary"
paragraph lengths         v1 P4 168 words -> v2 P4 within target; all paragraphs
                          under 160
```

---

## Unchanged

```
C1 / C2 / C3 scientific content
the frozen vocabulary: finite-sample, empirical analogue, predictive transfer
the two [CITATION NEEDED] placeholders
absence of C_neg / C_pos and of C_sign
no new experiment, training or scoring
```
