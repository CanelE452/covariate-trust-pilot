# Abstract v2 — claim audit

```
S1   two formulations, same target                         ARTIFACT_SUPPORTED (definitional)
S2   neither uniformly better in aggregate                  ARTIFACT_SUPPORTED
     stage_a_results.json manifest.*.overall: RMSE favours direct, MAE favours factorized
S3   standard descriptors summarize marginals, not order    CITATION_SUPPORTED [SBC05; KH06]
     definitional; cited in Introduction P2 and Related Work 2.2
S4   matched conditions; marginals fixed; two axes varied   ARTIFACT_SUPPORTED (design)
     point_hurdle_fairness.md; dgp_verification.md
S5   the axes interact rather than add                      ARTIFACT_SUPPORTED
     stage1_verified_contrasts.csv: interaction -16.74 against +7.83 and -4.58
S6   asymmetry: occurrence strength, magnitude direction    ARTIFACT_SUPPORTED
     stage2_verified_factor_effects.csv; association wording only, no mechanism claim
S7   one configuration favours direct by about -19.8%       ARTIFACT_SUPPORTED
     stage2_verified_cells.csv d=8, rho_I=0, rho_M=+0.8: -19.76 [-26.00, -14.53];
     scoped by "in the eighteen-cell grid"
S8   an empirical analogue on two public retail datasets    ARTIFACT_SUPPORTED
     table4_draft.md; "analogue" is required wording, "replicates" is forbidden
S9   a frozen rule shifts unseen series by 11.87 pp         ARTIFACT_SUPPORTED
     rule_replication/primary_result.json 0.11872 [0.07853, 0.15813]
S10  the isolated association does not survive overlap
     adjustment                                             ARTIFACT_SUPPORTED
     secondary_overlap.json +0.0032 [-0.0033, +0.0094]
S11  a learned router did not transfer despite a
     measurable oracle opportunity                          ARTIFACT_SUPPORTED
     convex_oracle.json 4.11%; external_benchmark.json -2.43%
S12  both formulations are outranked by classical
     estimators; the study isolates relative behaviour      ARTIFACT_SUPPORTED
     classical_benchmark/benchmark.json; M5 mean ranks SBA 3.152 < Croston 3.260 <
     TSB 3.411 < SES 3.483 < direct 4.202 < factorized 4.220.   NEW IN v2.
S13  closing: when factorizing helps or hurts at a fixed
     budget, and where that stops holding                   BOUNDARY_WORDING
```

## Totals

```
ARTIFACT_SUPPORTED   10
CITATION_SUPPORTED    2
BOUNDARY_WORDING      1
OVERCLAIM             0
UNSUPPORTED           0
```

## Gate

```
ABS1  unsupported claim                                0
ABS2  result numbers source-linked                     2 of 2  (-19.8%, 11.87 pp)
ABS3  finite-sample boundary retained                  "at a fixed budget" in S13;
                                                       "matched conditions" in S4
ABS4  first / SOTA / optimal / universal claim         0
ABS5  routing given more than one clause               no; exactly one clause, S11
ABS6  a competitiveness reading available              no; blocked by S12
ABS7  H1 described as a replication                    no; "empirical analogue"
ABS8  mechanism claimed to replicate                   no; S10 states the opposite
ABS9  word count                                       254
```
