# Introduction v2 — claim audit

Sentence-level audit of `introduction_v2.md` and `contributions_v2.md`. Tags live here
only. Supersedes `introduction_claim_audit.md`, which is retained for v1.

```
SUPPORTED          a verified artifact backs it and the wording does not exceed it
BOUNDARY_WORDING   supported, and the sentence exists to limit a claim
CITATION_NEEDED    a literature statement with no bibliography in the repository
OVERCLAIM          wording exceeds the evidence          -- must be 0
UNSUPPORTED        no artifact backs it                  -- must be 0
```

---

## P1 — problem (131 words)

```
S1  two formulations; both target the same quantity            SUPPORTED (definitional)
S2  the choice is usually made by convention                   BOUNDARY_WORDING
S3  neither is uniformly superior in aggregate; which leads
    depends on the error measure                               SUPPORTED
    stage_a_results.json manifest.*.overall -- RMSE favours direct, MAE favours
    factorized on both datasets.  Stated qualitatively; figures moved to Results.
S4  the useful variation is conditional                        BOUNDARY_WORDING
```

## P2 — gap (156 words)

```
S5  ADI and CV^2 underpin the standard categorization scheme   CITATION_NEEDED
S6  both are marginal functionals, unaffected by order         SUPPORTED (definitional)
S7  two series can match on all three and still differ in
    ordering                                                   SUPPORTED
    instantiated by the DGP: both arms of each axis draw the same two-point support
    with the same long-run share
S8  interval autocorrelation, size autocorrelation and their
    dependence are established subjects with estimators that
    exploit them                                               CITATION_NEEDED
    -- present specifically to prevent a novelty overclaim
S9  what has not been isolated is representation rather than
    estimator                                                  BOUNDARY_WORDING
```

## P3 — question (96 words)

```
S10 the question under matched marginals                       SUPPORTED (design)
S11 confined to one backbone, budget, protocol, target         BOUNDARY_WORDING
S12 a difference under a matched budget is a difference in
    finite-sample inductive bias                               BOUNDARY_WORDING
S13 no assumption that either is preferable                    BOUNDARY_WORDING
```

## P4 — controlled study (156 words)

```
S14 under fixed marginals the two structures interact; their
    joint effect is not the sum of their separate effects      SUPPORTED
    stage1_verified_contrasts.csv: interaction -16.74
    [-18.45,-15.05] against main effects +7.83 and -4.58.
    Magnitude deliberately not quoted here.
S15 the factorial shows order matters without identifying
    which property does                                        BOUNDARY_WORDING
    -- discharges CAPTION-O1: Stage 1 contrasts structure against its control only
S16 the sweep resolves this into an asymmetric geometry, i.e.
    the axes respond to different features of dependence       SUPPORTED
    glossed immediately, as required
S17 occurrence primarily strength, magnitude primarily
    direction, persistence moving toward direct prediction     SUPPORTED
    stage2_verified_factor_effects.csv; association wording only
S18 within the eighteen-cell sweep, one configuration shows
    statistically clear superiority for direct prediction,
    about -19.8%                                               SUPPORTED
    stage2_verified_cells.csv: -19.76 [-26.00,-14.53].  Scope stated.
```

## P5 — empirical transfer (114 words)

```
S19 the occurrence relationship appears as an empirical
    analogue rather than a direct replication                  SUPPORTED
    positive on both datasets and on three error scales
S20 the association strengthens within the intermittent
    regime                                                     SUPPORTED
    regime_h1.json: intermittent relative +0.1529 [+0.0519,+0.2613]
S21 a frozen descriptor rule shifts unseen series toward
    direct prediction by 11.87 percentage points of win rate   SUPPORTED
    rule_replication/primary_result.json  +0.11872 [+0.0785,+0.1581]
S22 the shift reproduces across three training seeds          SUPPORTED
    seed_robustness.json  by_seed 0/1/2, all intervals clear of zero
S23 the condition carries predictive information               BOUNDARY_WORDING
```

## P6 — empirical boundary (112 words)

```
S24 the rule selects on three descriptors at once and they
    are not separable in observed demand                       SUPPORTED
    unweighted |SMD| 1.32 on log scale; 1:2 matching failed at 0.61
S25 candidate and control differ substantially in scale as
    well as in the intended axis                               SUPPORTED
    stated as a measured difference, not as a cause
S26 after adjusting for overlap the isolated association is
    no longer distinguishable from zero                        SUPPORTED
    secondary_overlap.json  +0.0032 [-0.0033,+0.0094];
    figure omitted from the Introduction
S27 two results rather than one                                BOUNDARY_WORDING
S28 the controlled study has no scale axis and cannot
    arbitrate either way                                       BOUNDARY_WORDING
```

## P7 — adaptive-use boundary and positioning (158 words)

```
S29 the two forecasts make complementary errors and an
    origin-level oracle is meaningfully better than any
    fixed mixture                                              SUPPORTED
    structure_gate/convex_oracle.json  4.11%; figure not quoted
S30 choosing a less correlated pair enlarges the opportunity   SUPPORTED
    expert_diversity/expert_set_spec.json  2.15x; figure not quoted
S31 a frozen gate was worse than a static mixture on the
    first external dataset it was applied to                   SUPPORTED
    multi_benchmark/external_benchmark.json  -2.43%; figure not quoted
S32 successive changes did not recover cross-domain transfer;
    the last improved one domain and degraded severely on
    another                                                    SUPPORTED
    temporal_routing_encoder  +2.648% and -193.9%; figures not quoted
S33 a pre-registered stopping rule was triggered               SUPPORTED
S34 neither a universal winner nor a robust routing rule;
    a characterization and its boundary                        BOUNDARY_WORDING
```

## Contributions v2

```
C1  finite-sample retained; matched-budget scope stated        SUPPORTED
C2  analogue and predictive transfer used; replication absent;
    the boundary stated without attributing it to a cause      SUPPORTED
C3  three sentences; stopping rule reported                    SUPPORTED
```

---

## Totals

```
SUPPORTED          20
BOUNDARY_WORDING   12
CITATION_NEEDED     2
OVERCLAIM           0
UNSUPPORTED         0
```

---

## Numbers retained — three

```
value          sentence  source file / field                        why it is needed
"eighteen-     S18       stage2_verified_cells.csv                  without it the
 cell"                   18 rows, model HURDLE_MEAN                 direct-favourable
                                                                    claim has no scope
"-19.8%"       S18       stage2_verified_cells.csv                  "direct wins" with
                         d=8, rho_I=0, rho_M=+0.8, gain -0.1976     no size is
                         CI [-0.2600, -0.1453]                      uninformative; the
                                                                    claim turns on the
                                                                    magnitude
"11.87         S21       rule_replication/primary_result.json       the strongest single
 percentage              point_win_rate_difference 0.11872          piece of transfer
 points"                 CI [0.07853, 0.15813]                      evidence; a sceptical
                                                                    reader will want it
                                                                    quantified
```

Eight further quantities are cited in this audit as support for sentences that state
them qualitatively. They belong in Results, not in the Introduction; the reasoning is
in `introduction_v1_v2_diff.md`.

---

## IR checks

```
IR1   seven paragraphs                                                          PASS
IR2   P5 carries transfer only -- no mechanism sentence                         PASS
IR3   P6 carries the boundary only                                              PASS
IR4   H3 absent from the Introduction                                           PASS
IR5   omission rationale recorded                                               PASS  see below
IR6   numeric results <= 4                                                      PASS  3
IR7   no routing numbers                                                        PASS
IR8   no aggregate near-tie figures                                             PASS
IR9   Stage 2 scope stated as "within the eighteen-cell sweep"                  PASS
IR10  H1 called an empirical analogue, "replication" explicitly denied          PASS
IR11  selector and mechanism separated across P5 and P6                         PASS
IR12  no causal claim for scale; "responsible" removed                          PASS
IR13  "finite-sample" retained in S12 and C1                                    PASS
IR14  no universal winner claim                                                 PASS
IR15  no correlation-first novelty; S8 concedes the literature                  PASS
IR16  C1/C2/C3 match the frozen ledger                                          PASS
IR17  C_neg / C_pos absent                                                      PASS
IR18  C_sign absent                                                             PASS
IR19  no invented citation; two placeholders                                    PASS
IR20  no new experiment, training or scoring                                    PASS
IR21  frozen scientific files unmodified                                        PASS
IR22  no commit, push or merge                                                  PASS
```

**IR5 — H3 omission rationale.** `NOT INCLUDED IN INTRODUCTION BY DESIGN; explicitly
reported in Section 5.6, Table 4 and the limitations.` Reporting it accurately in the
Introduction requires the construct mismatch — the synthetic contrast is `d = 4`
against `d = 8` while the external test split at the ADI median (1.304 on M5, 1.317 on
Favorita) — and that caveat does not fit the P5/P6 flow. A single caveat-free sentence
would read as a plain failure and would misdescribe what was tested.

---

## Reviewer simulation, Introduction only

```
Q1  does the paper claim factorization is superior?     NO.  P1 states neither is
                                                        uniformly superior; P7 states
                                                        no universal winner.
Q2  does it claim to study correlation first?           NO.  P2 concedes the prior
                                                        literature and narrows the
                                                        claim to representation.
Q3  why is a synthetic study needed?                    To hold the marginals fixed and
                                                        vary only order.  P2 into P3.
Q4  how do Stage 1 and Stage 2 relate?                  P4: the factorial shows an
                                                        interaction, the sweep resolves
                                                        which property it tracks.
Q5  did the synthetic mechanism replicate?              NO.  P5 gives an analogue and a
                                                        predictive transfer; P6 states
                                                        the isolated mechanism is not
                                                        recovered.
Q6  why does routing appear?                            P7: a downstream test of whether
                                                        the conditional advantage can be
                                                        exploited, reported as a boundary.
Q7  what is the contribution?                           P7 closing: a finite-sample
                                                        characterization and its transfer
                                                        boundary.
```

All seven read as intended.
