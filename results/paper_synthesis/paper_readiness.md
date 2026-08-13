# Paper readiness

## Verdict

**PAPER_READY_AFTER_MINIMAL_EXPERIMENTS**

The claim chain is coherent and three non-overlapping contributions are each supported by at
least two independent artifacts, with one exception: the controlled study that C1 rests on is
not in this repository, so its numbers cannot currently be cited or verified. That is a
recovery task rather than an experiment, and it is the only blocking item. Nothing in the
empirical or routing evidence needs to be re-run, and no new experiment is required to protect
any claim.

---

## Phase 3 gates

```
gate   question                                              status
────────────────────────────────────────────────────────────────────────────────────────────
P3-G1  every number read from an artifact?                   PARTIAL FAIL
                                                             all empirical and routing numbers
                                                             yes (56 audited entries); the
                                                             controlled study has no artifact
                                                             here and no number from it is
                                                             asserted anywhere in this synthesis
P3-G2  negative results hidden?                              PASS
                                                             H3 non-replication, occurrence-gate
                                                             BSS, external -2.43%, UCI -193.9%,
                                                             Safe-P0L1 mean loss and the
                                                             classical benchmark all in MAIN
                                                             or APPENDIX, none dropped
P3-G3  operational PASS separated from science?              PASS
                                                             GATE_V3_OOF_STRONG, P0L1_TEMPORAL
                                                             _STRONG and DIVERSE_GATE_GREEN are
                                                             all labelled as OOF or operational
                                                             and none is used as external evidence
P3-G4  synthetic and real-data claim scopes separated?       PASS
                                                             wording pairs 2, 4, 10 and 11 in
                                                             claim_wording.md enforce it; the
                                                             mechanism claim is confined to the
                                                             controlled study
P3-G5  predictive selector separated from mechanism?         PASS
                                                             H2 is reported as two results with
                                                             different statuses
P3-G6  routing success separated from generalization?        PASS
                                                             hypothesis_final_status.md states
                                                             the two sentences separately and
                                                             claim_wording.md forbids merging them
P3-G7  main experiment count compressed?                     PASS
                                                             11 MAIN items, of which the whole
                                                             14-step routing chain is 4;
                                                             15 APPENDIX; the rest dropped
P3-G8  MUST_HAVE genuinely necessary?                        PASS
                                                             exactly one, and it is artifact
                                                             recovery, not an experiment
P3-G9  ROUTING_MODEL_DEVELOPMENT_STOP respected?             PASS
                                                             every routing candidate is
                                                             DO_NOT_RUN, including the tempting
                                                             one
P3-G10 new experiments, training or TEST scoring?            PASS
                                                             zero
```

One partial failure, on P3-G1, and it is the reason the verdict is not
`PAPER_READY_WITH_CURRENT_EVIDENCE`.

---

## Why not the other three verdicts

**Not `PAPER_READY_WITH_CURRENT_EVIDENCE`** — C1's evidence cannot be cited from an artifact in
this repository, and the source repository has no remote and lives on another machine.

**Not `RESEARCH_STORY_NEEDS_REFOCUS`** — the chain is already coherent under Scope B. The
results that look like scattered failures are a single elimination sequence: target, loss,
aggressiveness, capacity, representation. That is one argument, not several.

**Not `NOT_READY`** — the two most defensible results in the project are complete and
independently replicated: the H2 selector on 5,693 held-out series across three seeds, and the
external routing failure measured against a pre-declared primary dataset with the spec frozen
nine minutes before the first scoring.

---

## Standing risk that is not blocking

The single-backbone limitation is real, unaddressed, and the most likely reviewer objection.
It does not block submission provided the limitation is stated at abstract-level scope rather
than buried in the discussion. A second backbone is the highest-value optional experiment and
is classified NICE_TO_HAVE.
