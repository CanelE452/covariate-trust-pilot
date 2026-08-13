# Introduction v3 — claim audit

Sentence-level audit of `introduction_v3.md` and `contributions_v3.md`. Tags live here
only. Supersedes `introduction_v2_claim_audit.md`, which is retained.

Because P2–P6 are byte-identical to v2, their sentence tags carry over unchanged and
are restated here so this file stands alone. Only P1 and P7 were re-derived.

```
SUPPORTED          a verified artifact backs it and the wording does not exceed it
BOUNDARY_WORDING   supported, and the sentence exists to limit a claim
CITATION_NEEDED    a literature statement with no bibliography in the repository
OVERCLAIM          wording exceeds the evidence          -- must be 0
UNSUPPORTED        no artifact backs it                  -- must be 0
```

---

## P1 — problem (128 words) *(one clause removed from v2)*

```
S1  two formulations; both target the same quantity            SUPPORTED (definitional)
S2  the choice is usually made by convention                   BOUNDARY_WORDING
S3  neither is uniformly superior in aggregate; which of the
    two leads depends on the error measure                     SUPPORTED
    stage_a_results.json manifest.*.overall -- RMSE favours direct, MAE favours
    factorized on both datasets.  The v2 clause "the two are close" was removed:
    the aggregates are not equal and the sentence does not require them to be.
S4  the useful variation is conditional                        BOUNDARY_WORDING
```

## P2 — gap (156 words, unchanged)

```
S5  ADI and CV^2 underpin the standard categorization scheme   CITATION_NEEDED
S6  both are marginal functionals, unaffected by order         SUPPORTED (definitional)
S7  two series can match on all three and still differ in
    ordering                                                   SUPPORTED
S8  interval autocorrelation, size autocorrelation and their
    dependence are established subjects with estimators that
    exploit them                                               CITATION_NEEDED
S9  what has not been isolated is representation rather than
    estimator                                                  BOUNDARY_WORDING
```

## P3 — question (96 words, unchanged)

```
S10 the question under matched marginals                       SUPPORTED (design)
S11 confined to one backbone, budget, protocol, target         BOUNDARY_WORDING
S12 a matched-budget difference is a difference in
    finite-sample inductive bias                               BOUNDARY_WORDING
S13 no assumption that either is preferable                    BOUNDARY_WORDING
```

## P4 — controlled study (156 words, unchanged)

```
S14 the two structures interact; their joint effect is not
    the sum of their separate effects                          SUPPORTED
    stage1_verified_contrasts.csv: interaction -16.74 against +7.83 and -4.58;
    magnitude deliberately not quoted
S15 the factorial shows order matters without identifying
    which property does                                        BOUNDARY_WORDING
S16 the sweep resolves this into an asymmetric geometry, i.e.
    the axes respond to different features of dependence       SUPPORTED
S17 occurrence primarily strength, magnitude primarily
    direction, persistence moving toward direct prediction     SUPPORTED
    stage2_verified_factor_effects.csv; association wording only
S18 within the eighteen-cell sweep, one configuration shows
    statistically clear superiority for direct prediction,
    about -19.8%                                               SUPPORTED
    stage2_verified_cells.csv: -19.76 [-26.00,-14.53]
```

## P5 — empirical transfer (114 words, unchanged)

```
S19 an empirical analogue rather than a direct replication     SUPPORTED
S20 the association strengthens within the intermittent
    regime                                                     SUPPORTED
    regime_h1.json: +0.1529 [+0.0519,+0.2613]
S21 a frozen descriptor rule shifts unseen series toward
    direct prediction by 11.87 percentage points of win rate   SUPPORTED
    rule_replication/primary_result.json 0.11872 [0.07853,0.15813]
S22 the shift reproduces across three training seeds           SUPPORTED
S23 the condition carries predictive information               BOUNDARY_WORDING
```

## P6 — empirical boundary (112 words, unchanged)

```
S24 the three descriptors are not separable in observed
    demand                                                     SUPPORTED
    unweighted |SMD| 1.32 on log scale; 1:2 matching failed at 0.61
S25 candidate and control differ substantially in scale as
    well as in the intended axis                               SUPPORTED
    stated as a measured difference, not as a cause
S26 after adjusting for overlap the isolated association is
    no longer distinguishable from zero                        SUPPORTED
    secondary_overlap.json +0.0032 [-0.0033,+0.0094]; figure not quoted
S27 two results rather than one                                BOUNDARY_WORDING
S28 the controlled study has no scale axis and cannot
    arbitrate either way                                       BOUNDARY_WORDING
```

## P7 — adaptive-use boundary (108 words) *(rewritten)*

```
S29 conditional differences invite a downstream question
    about a learned routing rule                               BOUNDARY_WORDING
S30 the forecasts make complementary errors, and an
    origin-level oracle beats any fixed mixture, so the
    opportunity is measurable                                  SUPPORTED
    structure_gate/convex_oracle.json 4.11%; figure not quoted
S31 gains obtained during development did not transfer across
    domains or across time                                     SUPPORTED
    multi_benchmark/external_benchmark.json -2.43%; figure not quoted
S32 one external domain degraded severely                      SUPPORTED
    temporal_routing_encoder -193.9%; figure not quoted
S33 a pre-registered stopping rule was triggered               SUPPORTED
S34 neither a universal winner nor a generally reliable
    router is proposed                                         BOUNDARY_WORDING
S35 the contribution is the characterization and its
    transfer boundary                                          BOUNDARY_WORDING
```

## Contributions v3

```
C1  finite-sample retained; matched-budget scope stated        SUPPORTED
C2  analogue and predictive transfer used; replication absent  SUPPORTED
C3  three sentences; stopping rule reported                    SUPPORTED
```

Body byte-identical to `contributions_v2.md`.

---

## Totals

```
SUPPORTED          20
BOUNDARY_WORDING   13
CITATION_NEEDED     2
OVERCLAIM           0
UNSUPPORTED         0
```

v2 had 20 / 12 / 2 / 0 / 0. The single additional BOUNDARY_WORDING is P7's opening
sentence, which now states the downstream question explicitly instead of implying it.

---

## Numbers retained — three, unchanged from v2

```
value            sentence  source file / field                        role
"eighteen-cell"  S18       stage2_verified_cells.csv, 18 rows,        scopes the
                           model HURDLE_MEAN                          direct-favourable
                                                                      claim
"-19.8%"         S18       stage2_verified_cells.csv                  the claim turns on
                           d=8, rho_I=0, rho_M=+0.8                   the magnitude
                           gain -0.1976 CI [-0.2600, -0.1453]
"11.87           S21       rule_replication/primary_result.json       strongest single
 percentage                point_win_rate_difference 0.11872          transfer result
 points"                   CI [0.07853, 0.15813]
```

P7 contains no digits.

---

## IV3 checks

```
IV3-1   seven paragraphs                                                       PASS
IV3-2   P1 problem and motivation                                              PASS
IV3-3   P2 gap and literature boundary                                         PASS
IV3-4   P3 controlled finite-sample question                                   PASS
IV3-5   P4 Stage 1 into Stage 2                                                PASS
IV3-6   P5 transfer only                                                       PASS
IV3-7   P6 boundary only                                                       PASS
IV3-8   P7 within 90-110 words                                                 PASS  108
IV3-9   P7 free of routing chronology                                          PASS
IV3-10  P7 free of numeric routing results                                     PASS  no digits
IV3-11  H3 absent from the Introduction                                        PASS
IV3-12  H3 omission rationale recorded                                         PASS  see below
IV3-13  numeric results <= 3                                                   PASS  3
IV3-14  eighteen-cell scope stated                                             PASS
IV3-15  -19.8% source-linked                                                   PASS
IV3-16  +11.87 pp source-linked                                                PASS
IV3-17  H1 called an empirical analogue                                        PASS
IV3-18  selector and mechanism separated                                       PASS
IV3-19  no causal wording for scale                                            PASS
IV3-20  no aggregate tie claim                                                 PASS  clause removed
IV3-21  "finite-sample" retained                                               PASS
IV3-22  no universal winner claim                                              PASS
IV3-23  no robust router claim; the phrase appears only under negation         PASS
IV3-24  no correlation-first novelty                                           PASS
IV3-25  C_neg / C_pos absent                                                   PASS
IV3-26  C_sign absent                                                          PASS
IV3-27  no invented citation                                                   PASS
IV3-28  two [CITATION NEEDED] retained                                         PASS
IV3-29  no new experiment, training or scoring                                 PASS
IV3-30  frozen scientific artifacts unmodified                                 PASS
IV3-31  v1 and v2 preserved                                                    PASS
IV3-32  no commit, push or merge                                               PASS
```

**IV3-12 — H3 omission rationale.** `INTENTIONALLY OMITTED FROM INTRODUCTION.`
Reporting it accurately requires the construct mismatch — the synthetic contrast is
`d = 4` against `d = 8` while the external test split at the ADI median (1.304 on M5,
1.317 on Favorita) — and that caveat does not fit the P5/P6 flow. A caveat-free
sentence would read as a plain failure and would misdescribe what was tested. It is
reported in Section 5.6, Table 4 and the limitations.

---

## Reviewer simulation, Introduction v3 only

```
Q1  is Hurdle always better?                       NO.  P1 denies uniform superiority;
                                                    P7 denies a universal winner.
Q2  is Point always better in sparse demand?       NO.  P4 scopes the direct-favourable
                                                    configuration to one cell of the
                                                    eighteen-cell sweep and names three
                                                    conditions, not sparsity alone.
Q3  is correlation studied here first?             NO.  P2 concedes the prior literature.
Q4  what is the synthetic study for?               To hold the marginals fixed and vary
                                                    only temporal order.  P2 into P3.
Q5  how do Stage 1 and Stage 2 relate?             P4: the factorial finds an
                                                    interaction, the signed sweep
                                                    resolves which property it tracks.
Q6  did real data reproduce the synthetic
    mechanism?                                     NO.  P6 states the isolated
                                                    association does not survive
                                                    adjustment.
Q7  what did transfer?                             P5: an empirical analogue and a
                                                    predictive selector signal.
Q8  what did not transfer?                         P6 and P7: the isolated mechanism,
                                                    and robust learned routing.
Q9  what is the contribution?                      P7 closing: a controlled
                                                    finite-sample characterization and
                                                    its empirical transfer boundary.
```

All nine read as intended.
