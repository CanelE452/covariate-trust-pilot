# Which experiments go in the paper

Rule of thumb applied throughout: an experiment earns MAIN only if removing it would weaken a
contribution's claim. Everything that only rules out an alternative explanation goes to
APPENDIX. Everything that documents how the work was built goes to development history.

---

## MAIN

```
experiment                                  role                       reason
──────────────────────────────────────────────────────────────────────────────────────────
Controlled numerical study (Stage 1 + 4)    C1 foundation              the only place the
                                                                       cause is manipulated
Stage A: M5 + Favorita, Point vs Hurdle     C2 setup + L1              establishes that
                                                                       neither dominates
H1 in two datasets, three scales            C2 primary                 6/6 CI clear of zero
H1 within SBC regimes                       C2 boundary                intermittent yes,
                                                                       lumpy no - this is the
                                                                       honest limit
H2 frozen rule, independent M5 population   C2 primary                 strongest single result
                                                                       in the project
H2 overlap adjustment + matching failure    C2 primary                 turns a selector into a
                                                                       boundary; without it the
                                                                       paper overclaims
Classical intermittent benchmark            C2 positioning             pre-empts "why not
                                                                       Croston" and is honest
                                                                       about the backbone
Convex-oracle complementarity               C3 setup                   quantifies what is on
                                                                       the table (4.11%)
Expert diversity -> ceiling multiplier      C3 setup                   opportunity enlarged
                                                                       on purpose (2.15x)
Multi-dataset external benchmark            C3 primary                 the -2.43% that ends
                                                                       the method reading
Raw-history sequence gate (final)           C3 closing                 Fresh +2.648% and
                                                                       UCI -193.9% in one
                                                                       result
```

Eleven MAIN items, of which the last four are one compressed section.

---

## APPENDIX

```
experiment                                  keep because
──────────────────────────────────────────────────────────────────────────
H1 eligibility threshold sensitivity        reviewers will ask; 15/20/30 barely moves it
H1 adjusted partial association             exploratory, tagged as such in the artifact
Occurrence-gate Brier skill diagnostic      needed to justify not claiming the mechanism
H3 full result + synthetic-like support     a non-replication that is still a result
H2 three-seed robustness                    supports the main H2 number
Favorita rule transfer (low support)        contradicts the Favorita independent result;
                                            both must be visible
Favorita independent pool result            partial transfer
Gate-v1 kill test                           shows the first gate was already YELLOW
Gate-v2 OOF + fresh-holdout confirmation    the development success that later failed
                                            externally - the contrast is the point
Gate-v3 2x2 factorial                       direct loss supported, alpha anchor not
P0L1 expanded temporal robustness           the operational-STRONG vs interpretation-B split
Safe-P0L1 shrinkage                         eliminates "just be less aggressive"
HGB routing-information ceiling             eliminates "just use a bigger gate"
Sequence-gate details and unit tests        causality and parity checks
Dataset audit incl. UCI AVAILABILITY_UNKNOWN  honest about the weakest benchmark
```

---

## DROP_FROM_PAPER (development history only)

```
Gate-v1 variant search internals (gate_selection.json, gate_spec.json)
Gate-v2 selection sweep (gate_v2_selection.json), gradient checks
posthoc_gate_report.json and the Stage A integrity hash blocks
P0L1 fold_alpha.csv, gate_weight_stability.json, exposure_audit.json
Safe-P0L1 identity tests, lambda calibration CSV, WARN files
Favorita full-pool reconstruction mechanics (full_pool_manifest.json)
routing_information_ceiling feature-parity hashes and quadratic identity table
temporal_routing_encoder row_parity.json, seed_stability.json
every gate_report.json / WARN_FAIL.json in every area
```

These are what make the results auditable, not what makes them interesting. They belong in the
repository and in a data-availability statement, not in the paper.

---

## The compression that matters most

The routing chain is fourteen steps and produced roughly twenty artifacts. In the paper it is
**one section with four numbers**: the oracle gap (4.11%), the diversity multiplier (2.15×),
the external failure (−2.43%), and the final split result (+2.648% on FreshRetailNet against
−193.9% on UCI). Everything else in that chain is an appendix entry whose only job is to close
an alternative explanation.
