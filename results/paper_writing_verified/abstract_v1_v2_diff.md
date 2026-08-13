# abstract_v1 → v2

```
v1  265 words   numbers: 20%, 11.87 percentage points
v2  254 words   numbers: -19.8%, 11.87 percentage points
```

## Change 1 — number rendering unified

```
v1  "favours direct prediction by about 20% in relative error"
v2  "favours direct prediction by about -19.8% in relative error"
```

One quantity had three renderings across the paper: `20%` in the Abstract, `-19.8%` in the
Introduction and `-19.76` in Results 4.8. The Abstract's form also dropped the sign, which
is what tells a reader *which* formulation wins. Both surviving renderings map to the same
row of `../paper_manuscript_verified/manuscript_v2_number_map.csv`.

## Change 2 — the absolute-accuracy result is now conceded in the Abstract

```
added  "Both formulations are outranked by classical estimators in absolute accuracy; the
        study isolates relative behaviour rather than establishing forecasting accuracy."
```

The v1 abstract reported only conditional findings. Sections 5.4 and 6.5 state the
classical result plainly, but a reader who sees only the abstract could infer a
competitiveness claim the paper does not make. That is the single most likely misreading of
the whole manuscript, and it now cannot survive the first paragraph.

## Change 3 — trims paying for change 2

```
"- series that are zero in most periods -"            -> "- mostly zeros -"
"The descriptors normally used to characterize
 such series"                                          -> "Standard descriptors of such
                                                          series"
"...by 11.87 percentage points of win rate on an       -> "...of win rate."
 independent population."                                 the independent population is
                                                          stated in 5.6
"Within the eighteen-cell grid"                        -> "In the eighteen-cell grid"
"and the boundary of that answer in observed demand"   -> "and where that stops holding"
"one evaluation target" dropped from the matched-conditions list; it remains in 3.4
```

## Retained — the seven required elements

```
1 problem              intermittent demand, two formulations, one target
2 controlled question  neither uniformly better, so the variation is conditional; the
                       standard descriptors do not encode ordering
3 matched design       one backbone, one parameter count, one training budget; marginals
                       fixed, two dependence axes varied
4 synthetic asymmetry  occurrence tracks strength, magnitude tracks direction; persistence
                       moves toward direct; one cell at about -19.8%
5 empirical transfer   analogue on two datasets; frozen selector +11.87 pp; the isolated
                       association does not survive overlap adjustment
6 routing boundary     one clause: a learned router did not transfer despite a measurable
                       oracle opportunity
7 implication          when factorizing helps or hurts at a fixed budget, and where that
                       stops holding
```

## Word target

The target for this pass was 240–250. v2 is **254**, four over. The overage is the
classical-baseline sentence, and it is reported rather than met by deleting it. Under a hard
venue limit the first trim candidates are the routing clause (23 words, already minimal) and
the "Standard descriptors" sentence (20 words).
