# Introduction v5 — claim audit

Sentence-level audit of `introduction_v5.md`. Supersedes `introduction_v4_claim_audit.md`,
which is retained. `contributions_v3.md` is current and unchanged; its audit stands.

P1 and P3–P7 are byte-identical to v4, so their tags carry over and are restated here so
this file stands alone. Only P2 was re-derived.

```
ARTIFACT_SUPPORTED    a verified repository artifact backs it
LITERATURE_SUPPORTED  a verified peer-reviewed record backs it, and it is cited
CITATION_SUPPORTED    a definitional or standard-practice statement whose citation is
                      present and adequate
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
    factorized on both datasets.  Independently consistent with [NAR26]; citation
    deliberately not added (P1 frozen).
S4  the useful variation is conditional                      BOUNDARY_WORDING
```

## P2 — gap (165 words) *(rewritten)*

```
S5  ADI and CV^2 are what distinguishes intermittent
    series in practice                                       LITERATURE_SUPPORTED
                                                             [SBC05; KH06]
    LIT-C1.  SBC05 states the rules in these two statistics; KH06 corrects the
    boundary.  No claim of novelty; this is the premise.
S6  both summarize a marginal distribution and neither
    retains ordering; two series can match on both and
    still differ in whether gaps cluster or alternate        CITATION_SUPPORTED
                                                             (definitional)
    LIT-C1.  Order-invariance is attributed to the STATISTICS, not to the scheme.
    The clause instantiating it is also instantiated by the DGP: both arms of each
    axis draw the same two-point support with the same long-run share.
    NOTE: this sentence is a MOTIVATION, not a contribution.  Fixed marginals are an
    experimental control -- LIT-C5, LIT-W5.
S7  that order is already known to matter; simulation
    work varying interval AC, size AC and their
    dependence reports effects on forecast accuracy and
    inventory performance                                    LITERATURE_SUPPORTED
                                                             [ALR12]
    LIT-C2.  A concession.  Deliberately states a result, not a subject area.
    Nothing is said about what ALR12 holds constant -- LIT-W3 is open.
S8  estimators have been built that exploit occurrence
    dependence rather than assume it away                    LITERATURE_SUPPORTED
                                                             [WSS04]
    LIT-C2.  Two-state (zero / non-zero) Markov occurrence model inside a bootstrap.
    Scoped to OCCURRENCE dependence, which is what was verified.
S9  neural work compares a directly predicted demand rate
    against a Croston-style network that forecasts size
    and interval and divides them                            LITERATURE_SUPPORTED
                                                             [Kou13]
    LIT-C4.  A concession, stated against our own interest.  Full text verified:
    NN-Rate one linear output = demand rate; NN-Dual two linear outputs = z' and x',
    then z'/x' with a fitted de-biasing coefficient.
    LIT-W-KOU13: never "hurdle", never "the same two representations".
S10 these two lines of work answer different questions       BOUNDARY_WORDING
    Bounded to the two named strands.  An earlier draft read "the two strands have
    stayed apart", a mild absence claim over all literature; replaced.
S11 neither isolates how a direct conditional mean and an
    occurrence-probability x positive-magnitude
    factorization behave, at matched capacity, as
    occurrence and magnitude dependence are varied
    separately                                               BOUNDARY_WORDING
    LIT-C6.  The paper's one literature-relative claim.  "Neither" refers to the two
    strands just cited, not to all literature.  Every element is checkable:
      matched capacity        [Kou13] sec 3.4 tunes each arm separately; two output
                              nodes vs one
      separate axes           [Kou13] sec 3.1 has one generated population, no
                              dependence factor
      product form            [Kou13] uses the ratio form
    QUALIFIED by [NAR26]: matched direct-vs-hurdle alone is PARTIAL_OVERLAP, so the
    sentence rests on the CROSSING, not on the match.  See LIT-C6.
```

## P3 — question (96 words, unchanged)

```
S12 the question under matched marginal properties           ARTIFACT_SUPPORTED (design)
S13 confined to one backbone, budget, protocol, target       BOUNDARY_WORDING
S14 a matched-budget difference is a difference in
    finite-sample inductive bias                             BOUNDARY_WORDING
S15 no assumption that either is preferable                  BOUNDARY_WORDING
```

## P4 — controlled study (156 words, unchanged)

```
S16 the two structures interact; their joint effect is
    not the sum of their separate effects                    ARTIFACT_SUPPORTED
    stage1_verified_contrasts.csv: interaction -16.74 [-18.45,-15.05] against main
    effects +7.83 and -4.58
S17 the factorial shows order matters without identifying
    which property does                                      BOUNDARY_WORDING
S18 the sweep resolves this into an asymmetric geometry      ARTIFACT_SUPPORTED
S19 occurrence primarily strength, magnitude primarily
    direction, persistence moving toward direct prediction   ARTIFACT_SUPPORTED
    stage2_verified_factor_effects.csv; association wording only
S20 within the eighteen-cell sweep, one configuration
    shows statistically clear superiority for direct
    prediction, about -19.8%                                 ARTIFACT_SUPPORTED
    stage2_verified_cells.csv: d=8, rho_I=0, rho_M=+0.8, -19.76 [-26.00,-14.53]
```

## P5 — empirical transfer (114 words, unchanged)

```
S21 an empirical analogue rather than a direct replication   ARTIFACT_SUPPORTED
S22 the association strengthens within the intermittent
    regime                                                   ARTIFACT_SUPPORTED
    regime_h1.json: +0.1529 [+0.0519,+0.2613]
S23 a frozen descriptor rule shifts unseen series toward
    direct prediction by 11.87 percentage points of win rate ARTIFACT_SUPPORTED
    rule_replication/primary_result.json 0.11872 [0.07853,0.15813]
S24 the shift reproduces across three training seeds         ARTIFACT_SUPPORTED
S25 the condition carries predictive information             BOUNDARY_WORDING
```

## P6 — empirical boundary (112 words, unchanged)

```
S26 the three descriptors are not separable in observed
    demand                                                   ARTIFACT_SUPPORTED
    unweighted |SMD| 1.32 on log scale; 1:2 matching failed at 0.614
S27 candidate and control differ substantially in scale
    as well as in the intended axis                          ARTIFACT_SUPPORTED
S28 after adjusting for overlap the isolated association
    is no longer distinguishable from zero                   ARTIFACT_SUPPORTED
    secondary_overlap.json +0.0032 [-0.0033,+0.0094]
S29 two results rather than one                              BOUNDARY_WORDING
S30 the controlled study has no scale axis and cannot
    arbitrate either way                                     BOUNDARY_WORDING
```

## P7 — adaptive-use boundary (108 words, unchanged)

```
S31 conditional differences invite a downstream question
    about a learned routing rule                             BOUNDARY_WORDING
S32 complementary errors; an origin-level oracle beats
    any fixed mixture                                        ARTIFACT_SUPPORTED
    structure_gate/convex_oracle.json 4.11%
S33 development gains did not transfer across domains
    or across time                                           ARTIFACT_SUPPORTED
    multi_benchmark/external_benchmark.json -2.43%
S34 one external domain degraded severely                    ARTIFACT_SUPPORTED
    temporal_routing_encoder -193.9%
S35 a pre-registered stopping rule was triggered             ARTIFACT_SUPPORTED
S36 neither a universal winner nor a generally reliable
    router is proposed                                       BOUNDARY_WORDING
S37 the contribution is the characterization and its
    transfer boundary                                        BOUNDARY_WORDING
```

---

## Totals

```
ARTIFACT_SUPPORTED     19
LITERATURE_SUPPORTED    4     S7, S8, S9, and S5
CITATION_SUPPORTED      2     S1, S6
BOUNDARY_WORDING       12
OVERCLAIM               0
UNSUPPORTED             0
```

**Literature factual claims in P2 without a citation: 0.** Every P2 sentence is either
cited ([SBC05; KH06], [ALR12], [WSS04], [Kou13]), definitional (S6), or a bounded
statement about the works just cited (S10, S11).

---

## Numbers retained — three, unchanged since v2

```
"eighteen-cell"          S20   stage2_verified_cells.csv, 18 rows, HURDLE_MEAN
"-19.8%"                 S20   d=8, rho_I=0, rho_M=+0.8; -0.1976 [-0.2600,-0.1453]
"11.87 percentage        S23   rule_replication/primary_result.json
 points"                       point_win_rate_difference 0.11872 [0.07853,0.15813]
```

Citation keys contain year digits; these are reference labels, not results. P7 contains
no digits at all.

---

## IV5 checks

```
IV5-1   seven paragraphs                                                       PASS
IV5-2   only P2 differs from v4; P1 and P3-P7 byte-identical                   PASS
IV5-3   P2 within 130-165 words                                                PASS  165
IV5-4   zero [CITATION NEEDED]                                                 PASS  0
IV5-5   all five P2 citation keys resolve to core_reference_list.md            PASS  5/5
IV5-6   no arXiv-only record cited in the manuscript text                      PASS  [MC26] absent
IV5-7   P2-A present: descriptors summarize marginals, do not retain order     PASS  S6
IV5-8   P2-B present: dependence precedent conceded with a citation            PASS  S7, S8
IV5-9   P2-C present: [Kou13] described as Croston-style size/interval,
        divided                                                                PASS  S9
IV5-10  [Kou13] never called a hurdle anywhere in the Introduction             PASS
IV5-11  "the same two representations" absent                                  PASS
IV5-12  P2-D present: gap stated as the crossing at matched capacity           PASS  S11
IV5-13  fixed marginals never presented as the novelty                         PASS
IV5-14  "prior work ignores temporal dependence" absent                        PASS
IV5-15  "temporal dependence has not been studied" absent                      PASS
IV5-16  "no prior neural study compares representations" absent                PASS
IV5-17  "no prior work" / "nobody has" / "has never been" absent               PASS
IV5-18  "first" not used as a novelty claim                                    PASS
        (the string "the first half" appears in P4 as a structural phrase)
IV5-19  "we introduce decomposition" / "neural factorization" absent           PASS
IV5-20  numeric results <= 3                                                   PASS  3
IV5-21  P7 free of digits                                                      PASS
IV5-22  H3 absent from the Introduction                                        PASS
IV5-23  no aggregate tie or equivalence claim                                  PASS
IV5-24  no causal wording for scale                                            PASS
IV5-25  "finite-sample" retained                                               PASS  S14
IV5-26  no universal winner claim                                              PASS
IV5-27  no reliable router asserted; phrase appears only under negation        PASS
IV5-28  C_neg / C_pos / C_sign absent                                          PASS
IV5-29  no invented citation                                                   PASS
IV5-30  contributions_v3 unmodified                                            PASS
IV5-31  v1, v2, v3, v4 preserved                                               PASS
IV5-32  frozen scientific artifacts unmodified                                 PASS
IV5-33  no new experiment, training or scoring                                 PASS
IV5-34  no commit, push or merge                                               PASS
```

---

## Reviewer simulation, Introduction v5

```
Q1  is factorization always better?              NO.  P1 denies uniform superiority;
                                                 P7 denies a universal winner.
Q2  is direct prediction always better in
    sparse demand?                               NO.  P4 scopes it to one cell of the
                                                 eighteen-cell sweep, under three
                                                 conditions, not sparsity alone.
Q3  is temporal dependence studied here first?   NO.  S7 cites [ALR12], which varied all
                                                 three dependence types in simulated
                                                 demand.
Q4  hasn't someone already compared direct and
    decomposed neural forecasters?               YES, and S9 says so.  [Kou13] compared
                                                 NN-Rate against NN-Dual.  Our claim is
                                                 about the crossing with dependence at
                                                 matched capacity.
Q5  is Kourentzes' NN-Dual your Hurdle?          NO.  S9 says it forecasts size and
                                                 interval and DIVIDES them.  Ours
                                                 multiplies a probability by a
                                                 conditional mean.
Q6  is holding the marginals fixed your
    contribution?                                NO.  S6 uses it to explain what the
                                                 descriptors miss.  It is an
                                                 experimental control; P3 and P4 treat
                                                 it as design.
Q7  what is the synthetic study for?             To hold the marginals fixed and vary only
                                                 temporal order.  S6 into P3.
Q8  did real data reproduce the synthetic
    mechanism?                                   NO.  P6 states the isolated association
                                                 does not survive adjustment.
Q9  what is the contribution?                    P7 closing: a controlled finite-sample
                                                 characterization and its empirical
                                                 transfer boundary.
```

Q5 and Q6 are new in v5 and are the two questions this revision exists to answer. All
nine read as intended.
