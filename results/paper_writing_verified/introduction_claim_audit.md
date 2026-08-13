# Introduction claim audit

Sentence-level audit of `introduction_v1.md` and `contributions_v1.md`. Tags live here
only; the drafts carry none.

```
SUPPORTED          a verified artifact backs it, and the wording does not exceed it
BOUNDARY_WORDING   supported, and the sentence exists to limit a claim
CITATION_NEEDED    a literature statement with no bibliography in the repository
OVERCLAIM          wording exceeds the evidence            -- must be 0
UNSUPPORTED        no artifact backs it                    -- must be 0
```

---

## Paragraph 1 — problem

```
S1  two ways to forecast; both target the same quantity          SUPPORTED (definitional)
S2  the choice is made by convention rather than evidence        BOUNDARY_WORDING
    -- phrased as a description of practice, not a literature claim
S3  mean difference -0.0007 (M5) and -0.0320 (Favorita), RMSE
    favours direct and MAE favours factorized on both            SUPPORTED
    stage_a_results.json, manifest.*.overall
S4  a flat aggregate means the variation is conditional          BOUNDARY_WORDING
```

## Paragraph 2 — gap

```
S5  ADI and CV^2 are the two statistics behind the standard
    categorization scheme                                        CITATION_NEEDED
S6  both are functionals of a marginal distribution and are
    unaffected by order                                          SUPPORTED (definitional)
S7  two series can match on all three and still differ in
    ordering                                                     SUPPORTED
    instantiated by the DGP: both arms of each axis draw the
    same two-point support with the same long-run share
S8  interval autocorrelation, size autocorrelation and their
    dependence are established subjects, with estimators that
    exploit them                                                 CITATION_NEEDED
    -- this sentence exists to prevent a novelty overclaim
S9  what has not been isolated is whether dependence should
    change the representation rather than the estimator          BOUNDARY_WORDING
    -- the novelty claim, deliberately narrow
```

## Paragraph 3 — question

```
S10 the question, stated under matched marginals               SUPPORTED (design)
S11 confined to one backbone, budget, protocol, target         BOUNDARY_WORDING
S12 differences under a matched budget are differences in
    finite-sample inductive bias, not representational capacity BOUNDARY_WORDING
    -- carries the frozen "finite-sample" requirement
S13 we do not assume either representation is preferable        BOUNDARY_WORDING
```

## Paragraph 4 — controlled finding

```
S14 relative performance moves substantially under fixed
    marginals                                                   SUPPORTED
    stage1_verified_contrasts.csv
S15 the interaction is larger in magnitude than either main
    effect                                                      SUPPORTED
    -16.74 against +7.83 and -4.58
S16 the factorial establishes that order matters without
    identifying which property                                  BOUNDARY_WORDING
    -- discharges CAPTION-O1 in the Introduction
S17 occurrence effects primarily associated with strength,
    magnitude effects primarily with direction                  SUPPORTED
    stage2_verified_factor_effects.csv; association wording,
    no causal verb
S18 persistence, not structure as such, moves the comparison
    toward direct prediction                                    SUPPORTED
    signed rho_M -0.0711 against |rho_M| -0.0228
S19 within the eighteen conditions, exactly one shows
    statistically clear superiority for direct prediction,
    at -19.8%                                                   SUPPORTED
    scope stated explicitly, as the freeze requires
```

## Paragraph 5 — empirical transfer

```
S20 the occurrence relationship appears as an empirical
    analogue on both datasets, on three error scales            SUPPORTED
    +0.1064 and +0.0789; six of six scale estimates positive
S21 it strengthens in the intermittent regime and is
    indistinguishable from zero in the lumpy one                SUPPORTED
    +0.1529 [+0.0519,+0.2613] against +0.0323 [-0.0793,+0.1463]
S22 a frozen rule shifts unseen series toward direct
    prediction by nearly twelve points of win rate, across
    three seeds                                                 SUPPORTED
    +11.87 pp; seeds -0.0230 / -0.0211 / -0.0277
S23 after balancing, the association is not distinguishable
    from zero                                                   SUPPORTED
    +0.0032 [-0.0033,+0.0094]
S24 the covariate responsible is scale, which the controlled
    design does not contain                                     SUPPORTED
    unweighted |SMD| 1.32 on log scale; matching failed at 0.61
S25 predictive transfer and mechanism replication come apart    BOUNDARY_WORDING
```

## Paragraph 6 — boundary

```
S26 an origin-level oracle is about four percent better than
    the best static mixture                                     SUPPORTED  4.11%
S27 diversifying the pair roughly doubles that ceiling          SUPPORTED  2.15x
S28 a frozen gate was worse than a static mixture on the
    first external dataset                                      SUPPORTED  -2.43%
S29 successive changes did not recover it; the last repaired
    one dataset and failed catastrophically on another          SUPPORTED
    +2.648% and -193.9%
S30 a pre-registered stopping rule was triggered                SUPPORTED
S31 what this offers is a characterization and a boundary,
    not a winner or a router                                    BOUNDARY_WORDING
```

## Contributions

```
C1  finite-sample kept; matched-budget scope stated             SUPPORTED
C2  "empirical analogue" and "predictive transfer" used;
    "replication" absent; scale named as the separator          SUPPORTED
C3  two sentences; stopping rule reported                       SUPPORTED
```

---

## Totals

```
SUPPORTED          24
BOUNDARY_WORDING   11
CITATION_NEEDED     2
OVERCLAIM           0
UNSUPPORTED         0
```

---

## IA checks

```
IA1   ADI/CV2 limitation not overstated -- stated as a definitional property
      of marginal functionals, with no empirical claim that they fail to predict     PASS
IA2   no correlation-first novelty; S8 concedes the prior literature explicitly
      and S9 narrows the claim to representation versus estimator                    PASS
IA3   "finite-sample" present in S12 and in C1                                       PASS
IA4   Stage 1 and Stage 2 given distinct roles in S16-S17; the alternating-arm
      caveat is discharged in prose rather than in a figure                          PASS
IA5   association wording throughout S17-S18; no causal verb                         PASS
IA6   the direct-favourable cell is scoped to the eighteen conditions in S19          PASS
IA7   H1 is called an empirical analogue in S20; "replicate" never applied to it      PASS
IA8   selector and mechanism separated across S22-S25 and in C2                       PASS
IA9   H3 -- not mentioned in the Introduction.  Deliberate: it is a
      non-replication at a contrast the external test did not evaluate, and
      compressing it into one sentence would either overstate or understate it.
      It is reported in Section 5.6 and in Table 4.  Flagged for the user below.      SEE NOTE
IA10  routing failure stated with both numbers, including -193.9%                     PASS
IA11  no universal claim for either representation                                    PASS
IA12  C1/C2/C3 match the frozen ledger                                                PASS
IA13  C_neg / C_pos absent                                                            PASS
IA14  C_sign absent                                                                   PASS
IA15  no invented citation; two [CITATION NEEDED] placeholders                        PASS
```

**Note on IA9.** The instruction was that H3's negative result must not be hidden. It is
not hidden — it is in Section 5.6, Table 4 and the appendix — but it is currently absent
from the Introduction. That is a judgement call, and it is the user's to make. Adding it
would cost a sentence in the already-long fifth paragraph and would need its own caveat
(the external test used the ADI median, not the ADI 4-versus-8 contrast the prediction
came from), or it would read as a plain failure. Leaving it out keeps the Introduction to
one clean transfer story. Flagged rather than decided.

---

## Numbers used, with sources

```
value                          where     source artifact / field
-0.0007, -0.0320               S3        stage_a_results.json  manifest.*.overall.mean_delta
-16.74 (implied "larger than") S15       stage1_verified_contrasts.csv  interval_x_magnitude
"eighteen conditions", -19.8%  S19       stage2_verified_cells.csv  d=8, rho_I=0, rho_M=+0.8
+0.1064, +0.0789 (as "both")   S20       stage_a_results.json  results.*.20.H1
intermittent vs lumpy          S21       regime_h1.json  regimes.*.relative
"nearly twelve points"         S22       rule_replication/primary_result.json  +0.11872
"three seeds"                  S22       seed_robustness.json  by_seed
"not distinguishable from 0"   S23       secondary_overlap.json  +0.0032 [-0.0033,+0.0094]
"about four percent"           S26       structure_gate/convex_oracle.json  4.11%
"roughly doubles"              S27       expert_diversity/expert_set_spec.json  2.15x
-2.43%                         S28       multi_benchmark/external_benchmark.json
+2.648%, -193.9%               S29       temporal_routing_encoder/aggregate_results.json
```

Eleven quantities, all artifact-linked. The Introduction quotes exact figures only
where the sentence turns on the number; elsewhere it uses a rounded verbal form
("about four percent", "nearly twelve points") so the paragraph does not read as a
results table.
