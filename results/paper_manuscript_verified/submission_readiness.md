# Submission readiness

Status: **`MANUSCRIPT_V2_HUMAN_REVIEW_PACKAGE_READY`**.
Blockers are separated by kind, because mixing them is how a bibliographic task turns into
a perceived scientific one.

---

## Scientific blockers

```
NONE.
```

`MUST_HAVE` has been `NONE` since `paper_readiness_verified.md`, and nothing found while
drafting or auditing the manuscript changed it. The largest open exposure is R1 (single
backbone) and it is answered by scope in the limitations, not by evidence.

## Editorial items — inside our control, no new science

```
E1  abstract is 254 words against a 240-250 target        four over; the overage buys the
                                                          classical-baseline sentence
E2  6.6 could fold into 6.5                               cosmetic; user decision B3
E3  working title not final                               user decision B2
E4  appendix prose not written                            roles are specified in the
                                                          caption files; text is not drafted
E5  no author block, no acknowledgements                  venue-dependent
```

## Operational follow-ups — not scientific, not blocking

```
O1  ALR12 full text via institutional subscription or ILL  internal_submission_followups FU-1
O2  venue selection                                        drives O3, O4, O6
O3  word / page limits                                     see human_review_issue_package C2
O4  BibTeX finalization from the twelve verified records   core_reference_list.md
O5  figure dimensions, fonts, template                     drafts exist at three sizes
O6  anonymization                                          artifacts name a machine and a
                                                           user path; the manuscript does not
O7  supplementary packaging                                synthetic source is hash-verified
O8  code / data availability statement                     text not written
O9  private artifact backup                                already on a private release
O10 commit / push                                          after user approval only
```

---

## Second-backbone decision rule

Current status: **not run, `NICE_TO_HAVE`, not a submission blocker.**

Run one, and only one, if **all four** hold:

```
1  the advisor explicitly requests it, or
2  the target venue strongly requires architecture generality, and
3  a verified matched implementation already exists, and
4  it can be pre-registered before any outcome is inspected
```

Permitted scope if it runs:

```
one backbone only, chosen before seeing any result
the same Point/Hurdle parameter-matching principle, stated as a rule not tuned to fit
the core Stage 2 grid, or a pre-registered representative subset of it
the same G metric and the same bootstrap protocol
no model search, no result-driven hyperparameter tuning, no per-cell tuning
stop as soon as the robustness decision is answerable
```

Explicitly `DO_NOT_RUN`:

```
open-ended stronger-backbone search        backbone cherry-picking
SOTA expansion                             many-backbone benchmarking
result-driven tuning                       re-running the synthetic study
routing development                        new datasets
```

**Manuscript wording either way.** No sentence claims the result is backbone-general. Every
claim is scoped to the matched DLinear setup, and `reviewer_attack_audit.md` keeps R1 as the
highest open risk regardless of whether the check is run.

---

## What is verified and will not be redone

```
synthetic source recovered, hash-verified, provenance-identified   dgp_verification.md
C1 gates                                                          c1_gate_report.json
Stage 1 / Stage 2 numbers                                         *_verified_*.csv
empirical hypothesis statuses                                     table4_draft.md
literature boundary and novelty components                        novelty_boundary_freeze.md
notation registry                                                 notation_registry.md
every displayed number mapped                                     manuscript_v2_number_map.csv
```
