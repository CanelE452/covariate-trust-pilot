# Additional-experiment triage

No experiment is executed in this task. `ROUTING_MODEL_DEVELOPMENT_STOP` is in force and no
candidate below is allowed to reopen it.

Test applied to each candidate: **does a contribution collapse without it, or is it a
robustness nicety?**

---

## MUST_HAVE

### M1. Recover and verify the controlled synthetic study artifacts

**Not an experiment.** No model is trained, no dataset is scored. It is a recovery and
verification task.

**Why it is mandatory.** C1 is the foundation of the paper and its artifacts are not in this
repository. `artifact_audit.json` entry `A_numerical` records the source as
`~/Documents/github/m5dataset`, a repository with **no git remote**, last commit 2026-06-25, on
a different machine from the one this work now runs on. Every downstream document — the frozen
H1/H2/H3 definitions in `pre_analysis_spec.json`, the Point-favorable cell that H2's rule
encodes, the ADI 4-versus-8 contrast that H3 was derived from — points at numbers that cannot
currently be cited from an artifact, and cannot be recomputed if that machine is lost.

**Which claim it protects.** All of C1, and the derivation provenance of H1, H2 and H3 in C2.
Without it the paper's first contribution is a citation to unavailable work and its
pre-registration story has a hole.

**Minimum scope.** Copy the study's result artifacts and preregistration into this repository
under a new area; verify that the H1/H2/H3 definitions and the Point-favorable cell in
`pre_analysis_spec.json` match what the synthetic artifacts actually contain; record hashes.
No re-running of the synthetic study is required if the artifacts are intact.

**Requires user approval before proceeding.**

---

## NICE_TO_HAVE

```
candidate                              value                                    why not MUST
─────────────────────────────────────────────────────────────────────────────────────────────
second backbone for Point/Hurdle       answers the single largest reviewer      claims survive as
                                       objection; would let C1 and C3 drop      "for this backbone
                                       "for this backbone family"               family", stated openly
occurrence-skill diagnostic with       would let the paper say something        the paper does not
per-observation p_hat (ROC/PR/logloss) positive or firmly negative about the    need the mechanism
                                       mechanism in real data; currently        claim on real data;
                                       blocked because Stage A stored only      it withdraws it
                                       per-series aggregates
H3 tested at the synthetic contrast    the current non-replication used the     H3 is already excluded
(ADI 3-5 vs >=8)                       ADI median, not ADI 4 vs 8; support      from the contributions
                                       exists (M5 127 vs 52 series)             and reported as a
                                                                                non-replication
more seeds for the sequence gate       would firm up the UCI catastrophe        the failure is large
                                                                                (-193.9%) and the
                                                                                mechanism is explained
                                                                                by the gate weight
additional external dataset            more evidence on generalization          forbidden while not
                                                                                GREEN, and the stop
                                                                                rule already fired
inventory / newsvendor evaluation      would broaden the practical reading      changes the estimand;
                                                                                a different paper
probabilistic metrics (CRPS, pinball)  the Hurdle model is naturally            the pre-registration
                                       probabilistic; point metrics under-      is on RMSE/MAE and
                                       serve it                                 changing it now would
                                                                                be outcome-driven
more synthetic rho grid points         smoother heatmaps                        cosmetic
```

---

## DO_NOT_RUN

```
candidate                          reason
──────────────────────────────────────────────────────────────────────────────────
more gate tuning of any kind       ROUTING_MODEL_DEVELOPMENT_STOP, frozen before results
bigger GRU / TCN / Transformer     explicitly named in the stop rule
hybrid G-NOSCALE + sequence gate   explicitly named in the stop rule
new expert search / reselection    explicitly named in the stop rule
alpha anchor, shrinkage retry      already eliminated; retrying is outcome-driven
new clean dataset scoring          only permitted on SEQUENCE_ROUTING_GREEN, which did not occur
SOTA benchmark                     does not test the comparative claim; named in the stop rule
joint expert-gate training         named in the stop rule
re-running any existing TEST       would destroy the one-shot property of the external result
```

---

## The trap this triage is designed to avoid

The tempting MUST_HAVE is "one more routing experiment, because FreshRetailNet finally worked".
FreshRetailNet did work: −0.506% became +2.648% with the interval clear of zero and 3 of 3
folds beating the handcrafted gate. That is exactly the result that makes another attempt feel
justified, and it is exactly why the stop rule was frozen before any of it was visible. The
same model was −193.9% on UCI. Reopening now would be selecting on the outcome, and it is
classified DO_NOT_RUN for that reason and not because the idea is uninteresting.
