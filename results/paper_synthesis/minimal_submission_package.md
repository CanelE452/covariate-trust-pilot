# Minimal submission package

What a submission would contain if it were assembled from the evidence that exists today, plus
the one recovery task.

---

## Main body

```
section                              content                                    evidence
─────────────────────────────────────────────────────────────────────────────────────────────
1 Introduction                       the conditioning question                  Table 2 overall row
2 Related work                       Croston family vs representation choice    -
3 Controlled numerical study         design, the two dependence axes,           BLOCKED: synthetic
                                     the relative-gain surface                  artifacts absent
4 Real-data validation               Stage A on M5 and Favorita;                stage_a_results.json
                                     H1 across scales and regimes               posthoc_diagnostic.json
                                                                                regime_h1.json
5 The entanglement boundary          H2 as a selector on an independent         rule_replication/
                                     population; the same association           primary_result.json
                                     vanishing under overlap weighting          secondary_overlap.json
                                                                                h2_confirmatory/
                                                                                matching_failed.json
6 Positioning                        classical intermittent benchmark           classical_benchmark/
                                                                                benchmark.json
7 The adaptive-use boundary          oracle gap, deliberate diversification,    structure_gate/
                                     external failure, final representation     convex_oracle.json
                                     experiment                                 expert_diversity/
                                                                                multi_benchmark/
                                                                                temporal_routing_encoder/
8 Discussion and limitations         one backbone family; regime-balanced       -
                                     sample; RMSE/MAE only; UCI availability
```

Figures 1–5 as in `figure_plan.md`. Tables 1–4 as in `table_plan.md`.

---

## Appendix

```
A  H1 eligibility-threshold sensitivity (15 / 20 / 30)
B  H1 adjusted partial association, tagged exploratory
C  occurrence-gate Brier skill against a constant rate
D  H3 in full, with the ADI-median versus ADI 4-vs-8 discrepancy
E  H2 three-seed table
F  both Favorita analyses side by side: LOW_SUPPORT and PARTIAL_TRANSFER
G  Gate-v1 kill test
H  Gate-v2 cross-fitted OOF and fresh-holdout confirmation
I  expert diversity pairwise screen
J  Gate-v3 2x2 factorial
K  P0L1 expanded temporal robustness, with the operational-label caveat
L  Safe-P0L1 shrinkage, per-fold and tail
M  routing-information ceiling (HGB), including the weak-manipulation caveat
N  sequence-gate design, unit tests and per-fold results
O  dataset audit, including UCI AVAILABILITY_UNKNOWN
P  pre-registration documents and freeze timestamps
```

---

## Reproducibility statement

Every main-text number can be traced to a JSON or CSV under `results/`, listed in
`exact_source_artifacts.md`, **except** Section 3, which depends on artifacts not currently in
this repository.

Freeze evidence that should be cited explicitly, because it is unusually strong and reviewers
rarely see it:

- `pre_analysis_spec.json` frozen at 18:06:38 with Stage A results written at 18:11:16.
- The external benchmark spec frozen at `2026-08-07T16:29:27Z`, first TEST scoring at
  `2026-08-07T16:38:04Z`.
- P0L1 fold boundaries hashed before any error was computed.
- The sequence-gate stop rule frozen in `spec.py` before the first training run, together with
  its seed policy.

---

## What is missing and what it costs

Only Section 3. Without the recovery task, Sections 4–7 still stand on their own artifacts, but
the paper would have to be reframed as a purely empirical study — which would remove the one
place where the causal manipulation is real, and with it the reason to believe the temporal
axes matter at all. That is why the recovery is MUST_HAVE rather than NICE_TO_HAVE.
