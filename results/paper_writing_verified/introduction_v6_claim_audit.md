# Introduction v6 — claim audit

Sentence-level audit of `introduction_v6.md`. Supersedes `introduction_v5_claim_audit.md`,
which is retained. `contributions_v3.md` is current and unchanged; its audit stands.

P1 and P3–P7 are byte-identical to v5, so their tags carry over and are restated here so
this file stands alone. Only P2 was re-derived.

```
ARTIFACT_SUPPORTED    a verified repository artifact backs it
LITERATURE_SUPPORTED  a verified peer-reviewed record backs it, and it is cited
CITATION_SUPPORTED    definitional or standard practice, with an adequate citation
BOUNDARY_WORDING      supported, and the sentence exists to limit a claim
OVERCLAIM             wording exceeds the evidence         -- must be 0
UNSUPPORTED           no artifact or record backs it       -- must be 0
```

---

## P1 — problem (128 words, unchanged)

```
S1  two formulations; both target the same quantity          CITATION_SUPPORTED
                                                             (definitional)
S2  the choice is usually made by convention                 BOUNDARY_WORDING
S3  neither is uniformly superior in aggregate; which of
    the two leads depends on the error measure               ARTIFACT_SUPPORTED
    stage_a_results.json manifest.*.overall -- RMSE favours direct, MAE favours
    factorized on both datasets.  Independently consistent with [NAR26], which reports
    that two-stage hurdle models do not outperform a single-stage model; citation
    deliberately not added, P1 is frozen.
S4  the useful variation is conditional                      BOUNDARY_WORDING
```

## P2 — gap (167 words) *(rewritten)*

```
S5  ADI and CV^2 are what distinguishes intermittent series
    in practice                                              LITERATURE_SUPPORTED
                                                             [SBC05; KH06]
    LIT-C1.
S6  both summarize a marginal distribution and neither
    retains ordering; two series can match on both and
    still differ in whether gaps cluster or alternate        CITATION_SUPPORTED
                                                             (definitional)
    LIT-C1.  Order-invariance attributed to the STATISTICS, not to the scheme.
    This is a MOTIVATION, not a contribution: fixed marginals are an experimental
    control -- LIT-C5b, policy EXCLUDED_FROM_NOVELTY.
S7  that order is known to matter; simulation work varying
    interval AC, size AC and their dependence reports
    effects on forecast accuracy and inventory performance   LITERATURE_SUPPORTED
                                                             [ALR12]
    LIT-C2.  A concession.  Says NOTHING about what [ALR12] holds constant, because
    LIT-C5 is UNRESOLVED and LIT-W3 is OPEN.  Checked mechanically by NBF5.
S8  neural work compares a directly predicted demand rate
    against a Croston-style representation that predicts
    non-zero demand size and inter-demand interval and
    combines them as a RATIO                                 LITERATURE_SUPPORTED
                                                             [Kou13]
    LIT-C4.  A concession, stated against our own interest.  Full text verified.
    LIT-W-KOU13: never "hurdle", never "the same two representations".
S9  single-stage models have been compared against two-
    stage models that multiply an occurrence probability
    by a conditional size                                    LITERATURE_SUPPORTED
                                                             [NAR26]
    LIT-C6a.  NEW IN v6, and the reason v6 exists.  Full text verified: a LightGBM
    regressor "trained directly on the full feature set" against a LightGBM classifier
    for P(non-zero) times a Tweedie LightGBM regressor, under "identical data
    preprocessing, feature construction, and evaluation protocols".
    This is the SAME PAIR OF FORMS the paper compares.  Conceded without qualification.
S10 these answer different questions from the comparison
    here, in which a direct conditional mean and an
    occurrence-probability x positive-magnitude
    factorization are held to one capacity and training
    budget while occurrence and magnitude dependence are
    varied separately                                        BOUNDARY_WORDING
    LIT-C7.  The paper's one literature-relative statement, and it contains NO absence
    claim -- not even a bounded one.  It states what THIS comparison does; the
    difference from the three cited works follows by contrast, not by assertion.
    "held to one capacity and training budget" describes OUR design, never a deficiency
    in prior work: [NAR26]'s match status is UNKNOWN, not absent (LIT-C6b, policy
    CLAIM_ONLY_IN_CONJUNCTION).
```

## P3 — question (96 words, unchanged)

```
S11 the question under matched marginal properties           ARTIFACT_SUPPORTED (design)
S12 confined to one backbone, budget, protocol, target       BOUNDARY_WORDING
S13 a matched-budget difference is a difference in
    finite-sample inductive bias                             BOUNDARY_WORDING
S14 no assumption that either is preferable                  BOUNDARY_WORDING
```

## P4 — controlled study (156 words, unchanged)

```
S15 the two structures interact; their joint effect is not
    the sum of their separate effects                        ARTIFACT_SUPPORTED
    stage1_verified_contrasts.csv: interaction -16.74 [-18.45,-15.05] against +7.83
    and -4.58
S16 the factorial shows order matters without identifying
    which property does                                      BOUNDARY_WORDING
S17 the sweep resolves this into an asymmetric geometry      ARTIFACT_SUPPORTED
S18 occurrence primarily strength, magnitude primarily
    direction, persistence moving toward direct prediction   ARTIFACT_SUPPORTED
    stage2_verified_factor_effects.csv; association wording only
S19 within the eighteen-cell sweep, one configuration shows
    statistically clear superiority for direct prediction,
    about -19.8%                                             ARTIFACT_SUPPORTED
    stage2_verified_cells.csv: d=8, rho_I=0, rho_M=+0.8, -19.76 [-26.00,-14.53]
```

## P5 — empirical transfer (114 words, unchanged)

```
S20 an empirical analogue rather than a direct replication   ARTIFACT_SUPPORTED
S21 the association strengthens within the intermittent
    regime                                                   ARTIFACT_SUPPORTED
    regime_h1.json: +0.1529 [+0.0519,+0.2613]
S22 a frozen descriptor rule shifts unseen series toward
    direct prediction by 11.87 percentage points of win rate ARTIFACT_SUPPORTED
    rule_replication/primary_result.json 0.11872 [0.07853,0.15813]
S23 the shift reproduces across three training seeds         ARTIFACT_SUPPORTED
S24 the condition carries predictive information             BOUNDARY_WORDING
```

## P6 — empirical boundary (112 words, unchanged)

```
S25 the three descriptors are not separable in observed
    demand                                                   ARTIFACT_SUPPORTED
    unweighted |SMD| 1.32 on log scale; 1:2 matching failed at 0.614
S26 candidate and control differ substantially in scale as
    well as in the intended axis                             ARTIFACT_SUPPORTED
S27 after adjusting for overlap the isolated association is
    no longer distinguishable from zero                      ARTIFACT_SUPPORTED
    secondary_overlap.json +0.0032 [-0.0033,+0.0094]
S28 two results rather than one                              BOUNDARY_WORDING
S29 the controlled study has no scale axis and cannot
    arbitrate either way                                     BOUNDARY_WORDING
```

## P7 — adaptive-use boundary (108 words, unchanged)

```
S30 conditional differences invite a downstream question
    about a learned routing rule                             BOUNDARY_WORDING
S31 complementary errors; an origin-level oracle beats any
    fixed mixture                                            ARTIFACT_SUPPORTED
    structure_gate/convex_oracle.json 4.11%
S32 development gains did not transfer across domains or
    across time                                              ARTIFACT_SUPPORTED
    multi_benchmark/external_benchmark.json -2.43%
S33 one external domain degraded severely                    ARTIFACT_SUPPORTED
    temporal_routing_encoder -193.9%
S34 a pre-registered stopping rule was triggered             ARTIFACT_SUPPORTED
S35 neither a universal winner nor a generally reliable
    router is proposed                                       BOUNDARY_WORDING
S36 the contribution is the characterization and its
    transfer boundary                                        BOUNDARY_WORDING
```

---

## Totals

```
ARTIFACT_SUPPORTED     19
LITERATURE_SUPPORTED    4     S5, S7, S8, S9
CITATION_SUPPORTED      2     S1, S6
BOUNDARY_WORDING       11
OVERCLAIM               0
UNSUPPORTED             0
```

**Literature factual claims in P2 without a citation: 0.** Every P2 sentence is either
cited ([SBC05; KH06], [ALR12], [Kou13], [NAR26]), definitional (S6), or a statement about
this paper's own design (S10).

**Absence claims in P2: 0.** v5's "Neither isolates …" is gone; S10 makes no negative
existential of any scope.

---

## Numbers retained — three, unchanged since v2

```
"eighteen-cell"          S19   stage2_verified_cells.csv, 18 rows, HURDLE_MEAN
"-19.8%"                 S19   d=8, rho_I=0, rho_M=+0.8; -0.1976 [-0.2600,-0.1453]
"11.87 percentage        S22   rule_replication/primary_result.json
 points"                       point_win_rate_difference 0.11872 [0.07853,0.15813]
```

Citation keys contain year digits; these are reference labels, not results. P7 contains
no digits at all.

---

## IV6 checks

```
IV6-1   seven paragraphs                                                       PASS
IV6-2   only P2 differs from v5; P1 and P3-P7 byte-identical                   PASS
IV6-3   P2 within 130-170 words                                                PASS  167
IV6-4   zero [CITATION NEEDED]                                                 PASS  0
IV6-5   all five P2 citation keys resolve to core_reference_list.md            PASS  5/5
IV6-6   no arXiv-only record cited in the manuscript text                      PASS  [MC26] absent
IV6-7   descriptors described as marginal summaries that do not retain order   PASS  S6
IV6-8   temporal-dependence precedent conceded with a citation                 PASS  S7
IV6-9   nothing asserted about what [ALR12] holds constant                     PASS  NBF5
IV6-10  [Kou13] described as a size / interval RATIO                           PASS  S8
IV6-11  [Kou13] never called a hurdle anywhere in the Introduction             PASS
IV6-12  "the same two representations" absent                                  PASS
IV6-13  hurdle / two-stage precedent conceded with a citation                  PASS  S9
IV6-14  [NAR26] never described as neural                                      PASS
IV6-15  matched budget stated as OUR design, not as a prior-work deficiency    PASS  S10
IV6-16  no absence claim of any scope in P2                                    PASS
IV6-17  fixed marginals never presented as the novelty                         PASS
IV6-18  "no prior work" / "nobody has" / "has never been" absent               PASS
IV6-19  "first" not used as a novelty claim                                    PASS
        (the string "the first half" appears in P4 as a structural phrase)
IV6-20  "we introduce decomposition" / "neural factorization" absent           PASS
IV6-21  numeric results <= 3                                                   PASS  3
IV6-22  P7 free of digits                                                      PASS
IV6-23  H3 absent from the Introduction                                        PASS
IV6-24  no aggregate tie or equivalence claim                                  PASS
IV6-25  no causal wording for scale                                            PASS
IV6-26  "finite-sample" retained                                               PASS  S13
IV6-27  no universal winner claim                                              PASS
IV6-28  no reliable router asserted; phrase appears only under negation        PASS
IV6-29  C_neg / C_pos / C_sign absent                                          PASS
IV6-30  no invented citation                                                   PASS
IV6-31  contributions_v3 unmodified                                            PASS
IV6-32  v1, v2, v3, v4, v5 preserved                                           PASS
IV6-33  frozen scientific artifacts unmodified                                 PASS
IV6-34  no new experiment, training or scoring                                 PASS
IV6-35  no commit, push or merge                                               PASS
```

---

## Reviewer simulation, Introduction v6

```
Q1  is factorization always better?              NO.  P1 denies uniform superiority;
                                                 P7 denies a universal winner.
Q2  is direct prediction always better in
    sparse demand?                               NO.  P4 scopes it to one cell of the
                                                 eighteen-cell sweep, under three
                                                 conditions, not sparsity alone.
Q3  is temporal dependence studied here first?   NO.  S7 cites [ALR12].
Q4  did Altay et al. also fix the marginals?     UNRESOLVED, and the paper says nothing
                                                 about it.  We do not use it either way.
Q5  hasn't someone already compared direct and
    decomposed neural forecasters?               YES, and S8 says so: [Kou13].
Q6  is Kourentzes' NN-Dual your factorization?   NO.  S8 says it combines size and
                                                 interval as a RATIO.  Ours multiplies a
                                                 probability by a conditional size.
Q7  hasn't someone compared direct against the
    probability x size form?                     YES, and S9 says so: [NAR26], on real
                                                 data at an identical feature set.
Q8  so is the matched comparison your novelty?   NOT ON ITS OWN.  [NAR26]'s capacity and
                                                 training match is not stated, so we
                                                 claim the match only as one condition
                                                 of the crossing.
Q9  is holding the marginals fixed your
    contribution?                                NO.  S6 uses it to explain what the
                                                 descriptors miss; it is a control.
Q10 then what remains?                           S10: the two forms held to one capacity
                                                 and training budget while occurrence and
                                                 magnitude dependence are varied
                                                 separately -- plus P5 and P6's transfer
                                                 boundary.
Q11 are you claiming first?                      NO.  The word is not used.
Q12 what is the contribution?                    P7 closing: a controlled finite-sample
                                                 characterization and its empirical
                                                 transfer boundary.
```

Q4, Q7 and Q8 are new in v6 and are the three questions this revision exists to answer.
All twelve read as intended.
