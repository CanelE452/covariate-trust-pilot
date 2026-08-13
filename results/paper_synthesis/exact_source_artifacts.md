# Exact source artifacts

Every number used in this synthesis, with the file it came from. Machine-readable form with
hashes is in `artifact_audit.json` (56 entries).

---

## Present and verified in this repository

```
number                                          artifact
─────────────────────────────────────────────────────────────────────────────────────────────
M5 / Favorita overall RMSE, MAE, mean delta,    results/external_validity_screen/
win rates; H1 / H2 / H3 at thresholds 15/20/30    stage_a_results.json
H1 raw / relative / scaled, adjusted partial     .../posthoc_diagnostic/posthoc_diagnostic.json
association, occurrence-gate Brier skill,
ADI support counts, H2 full-pool counts
H1 by SBC regime, regime label reproduction,     .../regime_h1/regime_h1.json
full-pool regime counts
H2 frozen-rule effect, win-rate difference,      .../rule_replication/primary_result.json
candidate and control means
H2 overlap-adjusted association, weighted SMD,   .../rule_replication/secondary_overlap.json
effective sample sizes
H2 matching failure, worst SMD 0.614             .../h2_confirmatory/matching_failed.json
H2 per-seed effects and intervals                .../seed_robustness/seed_robustness.json
Favorita transfer, low support                   .../favorita_transfer/primary_transfer.json
Favorita independent, partial transfer           .../favorita_independent/primary_result.json
classical benchmark tables, hand check, iETS     .../classical_benchmark/benchmark.json
absence                                          .../classical_benchmark/implementation_audit.json
static mixture and expert losses                 results/structure_gate/gate_potential.json
hard and convex oracle, g* distribution          results/structure_gate/convex_oracle.json
Gate-v1 verdict                                  results/structure_gate/killtest.json
Gate-v2 OOF improvements and intervals           results/structure_gate/gate_v2_oof_result.json
Gate-v2 fresh-holdout confirmation               results/structure_gate/fresh_confirmatory.json
ceiling multipliers, selected pair               results/expert_diversity/expert_set_spec.json
frozen gate on the diverse pair                  results/expert_diversity/pair_gate_result.json
external benchmark, spec and scoring times       results/multi_benchmark/external_benchmark.json
dataset roles, UCI AVAILABILITY_UNKNOWN          results/multi_benchmark/dataset_audit.json
2x2 factorial effects, identity checks           results/gate_v3/aggregate_results.json
P0L1 per-fold and aggregate                      results/gate_p0l1_robustness/aggregate_result.json
Safe-P0L1 aggregates, tails, lambda              results/gate_safe_p0l1/*.json
HGB comparison, capacity caveat, diagnosis       results/routing_information_ceiling/*.json
sequence gate aggregates, folds, stop rule       results/temporal_routing_encoder/*.json
data derivation, split boundaries                docs/m5_favorita_data_derivation.md
freeze timestamps                                .../pre_analysis_spec.json (18:06:38)
                                                 stage_a_results.json (18:11:16)
```

---

## Absent from this repository

```
Stage 1 / Stage 3 / Stage 4 controlled synthetic study
  claimed location  ~/Documents/github/m5dataset
  git remotes       none
  last commit       2026-06-25
  machine           the Linux host referenced in results/report.md and
                    _docs/history/2026-08-07-migration.md, not the current host
  what was migrated only the external-validity screen closure: 21 python files, the two
                    processed parquet files, three raw CSVs and 17 result artifacts
  consequence       no number from the controlled study can be cited from an artifact here,
                    and it cannot be recomputed from this repository
```

---

## Numbers deliberately not used

- Anything appearing only in `_docs/history/*.md` without a backing JSON or CSV. History files
  were used for navigation and for provenance statements, never as a source for an effect size.
- Anything from the conversation record.
- `results/report.md` and `results/figures/*` — these belong to the unrelated Chronos-2
  covariate pilot that gives this repository its name, and none of it is part of this paper.
