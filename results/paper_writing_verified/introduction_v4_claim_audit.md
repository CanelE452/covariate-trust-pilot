# Introduction v4 — claim audit

Sentence-level audit of `introduction_v4.md`. Supersedes `introduction_v3_claim_audit.md`,
which is retained. `contributions_v3.md` is current and unchanged; its audit stands.

Because P1 and P3–P7 are byte-identical to v3, their tags carry over unchanged and are
restated so this file stands alone. Only P2 was re-derived.

```
SUPPORTED          a verified artifact backs it and the wording does not exceed it
CITED              a verified peer-reviewed record backs it
BOUNDARY_WORDING   supported, and the sentence exists to limit a claim
CITATION_NEEDED    a literature statement with no reference          -- must now be 0
OVERCLAIM          wording exceeds the evidence                      -- must be 0
UNSUPPORTED        no artifact or record backs it                    -- must be 0
```

---

## P1 — problem (128 words, unchanged)

```
S1  two formulations; both target the same quantity            SUPPORTED (definitional)
S2  the choice is usually made by convention                   BOUNDARY_WORDING
S3  neither is uniformly superior in aggregate; which of the
    two leads depends on the error measure                     SUPPORTED
    stage_a_results.json manifest.*.overall -- RMSE favours direct, MAE favours
    factorized on both datasets.  Independently consistent with [NAR26]; citation
    deliberately not added, see introduction_v3_v4_diff.md.
S4  the useful variation is conditional                        BOUNDARY_WORDING
```

## P2 — gap (209 words) *(rewritten; citations resolved)*

```
S5  ADI and CV^2 underpin the standard categorization scheme   CITED  [SBC05; KH06]
    LIT-C1.  SBC05 states the rules in terms of the average inter-demand interval and
    the squared coefficient of variation of demand sizes; KH06 corrects the boundary.
S6  both are marginal functionals, unaffected by order         SUPPORTED (definitional)
    LIT-C2.  Follows from the definitions; attributed to the statistics, not to the
    scheme.
S7  two series can match on all three and still differ in
    ordering                                                   SUPPORTED
    instantiated by the DGP: both arms of each axis draw the same two-point support
    with the same long-run share
S8a dependence has been varied in simulated demand and shown
    to change forecast and inventory performance               CITED  [ALR12]
    LIT-C3.  A concession, and stronger than v3's.  ALR12 manipulates size
    autocorrelation, interval autocorrelation and size-interval cross-correlation.
S8b estimators exploit occurrence dependence                   CITED  [WSS04]
    LIT-C4.  Two-state (zero / non-zero) Markov occurrence model.  Narrowed from v3's
    unqualified "them" to occurrence dependence specifically.
S8c neural forecasters have been compared in direct and
    decomposed form on simulated series                        CITED  [Kou13]
    LIT-C5.  NEW IN v4, and added against our own interest.  NN-Rate (single linear
    output, demand rate) vs NN-Dual (two linear outputs, demand and interval), on a
    simulation of 1000 items.
S9a what has not been isolated is representation rather than
    estimator                                                  BOUNDARY_WORDING
S9b prior comparisons tune each separately on a single
    generated population, rather than holding the budget fixed
    across a controlled range of dependence structures         CITED  [Kou13]
    LIT-C6.  Both halves verified in the full text: section 3.4 selects the best I,H
    per model by in-sample MAE rank; section 3.1 builds one simulated population from
    fitted marginals, with no dependence factor.
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
    stage1_verified_contrasts.csv: interaction -16.74 against +7.83 and -4.58
S15 the factorial shows order matters without identifying
    which property does                                        BOUNDARY_WORDING
S16 the sweep resolves this into an asymmetric geometry        SUPPORTED
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
S26 after adjusting for overlap the isolated association is
    no longer distinguishable from zero                        SUPPORTED
    secondary_overlap.json +0.0032 [-0.0033,+0.0094]
S27 two results rather than one                                BOUNDARY_WORDING
S28 the controlled study has no scale axis and cannot
    arbitrate either way                                       BOUNDARY_WORDING
```

## P7 — adaptive-use boundary (108 words, unchanged)

```
S29 conditional differences invite a downstream question
    about a learned routing rule                               BOUNDARY_WORDING
S30 complementary errors; an origin-level oracle beats any
    fixed mixture                                              SUPPORTED
    structure_gate/convex_oracle.json 4.11%; figure not quoted
S31 development gains did not transfer across domains or time  SUPPORTED
    multi_benchmark/external_benchmark.json -2.43%; figure not quoted
S32 one external domain degraded severely                      SUPPORTED
    temporal_routing_encoder -193.9%; figure not quoted
S33 a pre-registered stopping rule was triggered               SUPPORTED
S34 neither a universal winner nor a generally reliable
    router is proposed                                         BOUNDARY_WORDING
S35 the contribution is the characterization and its
    transfer boundary                                          BOUNDARY_WORDING
```

## Contributions

`contributions_v3.md` unchanged and current. C1 / C2 / C3 tags stand as audited in
`introduction_v3_claim_audit.md`. C1 is not edited: the Introduction now blocks, via S8c,
any reading in which C1 claims the direct-vs-factorized comparison is itself new.

---

## Totals

```
SUPPORTED          20
BOUNDARY_WORDING   13
CITED               5     (was CITATION_NEEDED 2)
CITATION_NEEDED     0
OVERCLAIM           0
UNSUPPORTED         0
```

v3 had 20 / 13 / 0 / 2 / 0 / 0. The two placeholders became five cited statements
because the single v3 placeholder was covering three distinct prior results.

---

## Numbers retained — three, unchanged from v3

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

Citation keys contain year digits ([SBC05], [Kou13] and so on). These are reference
labels, not numeric results, and do not count against the three-number budget. P7
still contains no digits at all.

---

## IV4 checks

```
IV4-1   seven paragraphs                                                       PASS
IV4-2   only P2 differs from v3; P1, P3-P7 byte-identical                      PASS
IV4-3   zero [CITATION NEEDED] remaining                                       PASS  0
IV4-4   every citation key resolves to core_reference_list.md                   PASS  5/5
IV4-5   every cited record verified through Crossref                           PASS  11/11 OK
IV4-6   no arXiv-only record cited in the manuscript text                      PASS  [MC26] absent
IV4-7   "first" not used as a novelty claim                                    PASS
        (the string "the first half" appears in P4 as a structural phrase)
IV4-8   "no prior work" / "nobody has" absent                                  PASS
IV4-9   the gap is stated as what prior work DID, not as an absence            PASS  S9b
IV4-10  [Kou13] cited where direct-vs-decomposed comparison is introduced      PASS  S8c
IV4-11  [ALR12] cited where temporal-dependence prior work is conceded         PASS  S8a
IV4-12  numeric results <= 3                                                   PASS  3
IV4-13  P7 free of digits                                                      PASS
IV4-14  H3 absent from the Introduction                                        PASS
IV4-15  no aggregate tie or equivalence claim                                  PASS
IV4-16  no causal wording for scale                                            PASS
IV4-17  "finite-sample" retained                                               PASS  S12
IV4-18  no universal winner claim                                              PASS
IV4-19  no reliable router asserted; phrase appears only under negation        PASS
IV4-20  C_neg / C_pos absent                                                   PASS
IV4-21  C_sign absent                                                          PASS
IV4-22  no invented citation                                                   PASS
IV4-23  contributions_v3 unmodified                                            PASS
IV4-24  v1, v2, v3 preserved                                                   PASS
IV4-25  no new experiment, training or scoring                                 PASS
IV4-26  frozen scientific artifacts unmodified                                 PASS
IV4-27  no commit, push or merge                                               PASS
```

---

## Reviewer simulation, Introduction v4

```
Q1  is factorization always better?           NO.  P1 denies uniform superiority; P7
                                              denies a universal winner.
Q2  is direct prediction always better in
    sparse demand?                            NO.  P4 scopes it to one cell of the
                                              eighteen-cell sweep and names three
                                              conditions, not sparsity alone.
Q3  is temporal dependence studied here
    first?                                    NO.  P2 cites [ALR12], which varied all
                                              three dependence types in simulated
                                              demand.
Q4  hasn't someone already compared direct
    and decomposed neural forecasters?        YES, and P2 says so: [Kou13].  The paper's
                                              claim is about dependence as a manipulated
                                              factor at a matched budget, not about the
                                              comparison existing.
Q5  what is the synthetic study for?          To hold the marginals fixed and vary only
                                              temporal order.  P2 into P3.
Q6  how do Stage 1 and Stage 2 relate?        P4: the factorial finds an interaction,
                                              the signed sweep resolves which property
                                              it tracks.
Q7  did real data reproduce the synthetic
    mechanism?                                NO.  P6 states the isolated association
                                              does not survive adjustment.
Q8  what transferred, and what did not?       P5: an analogue and a predictive selector.
                                              P6 and P7: not the isolated mechanism, not
                                              robust learned routing.
Q9  what is the contribution?                 P7 closing: a controlled finite-sample
                                              characterization and its empirical
                                              transfer boundary.
```

Q4 is new in v4 and is the question the whole citation step exists to answer. All nine
read as intended.
