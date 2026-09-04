# PH-ONLINE-MEMORY-GONO-v1

RUN STATUS: `PHASE0_AUDIT_STOP`

Scientific verdict: **not issued**. The attempt stopped before preregistration, Gate 0, smoke training, or any full run, so none of the experiment's exact final-verdict strings applies.

## Confirmed stop reasons

- `[확인]` Frozen records say the canonical `train_one` trainer was imported unchanged for external data, but the actual M5/Favorita path uses copied wrappers. The canonical trainer constructs its own dense synthetic split and cannot directly consume the availability-masked external split.
- `[확인]` The wrapper comment names `test_loop_matches_train_one()` as its equivalence proof, but no such function or test exists in the repository.
- `[확인]` The current base Python environment cannot normally import PyTorch because of Intel OpenMP Error #15. The `KMP_DUPLICATE_LIB_OK=TRUE` bypass was rejected because it can produce crashes or silently incorrect results.

These conditions meet the request's failure-first rule: a records-versus-code conflict or unsafe execution environment must stop the pilot immediately.

## Preserved state

- `[확인]` Existing stop rules remain unchanged: `HANDCRAFTED_FEATURE_GATE_STOP`, `RAW_SEQUENCE_GATE_STOP`, `ROUTING_MODEL_DEVELOPMENT_STOP`, and `DO_NOT_CONSUME_NEW_CONFIRMATORY_DATASET`.
- `[확인]` No existing result, checkpoint, prediction, claim ledger, or stop rule was overwritten.
- `[확인]` Pre-existing dirty-worktree changes were left untouched.
- Only this status file and [`audit.json`](./audit.json) were created under the new result namespace.

## Important audit findings for a restarted attempt

- Existing M5/Favorita checkpoints cross the newly requested cutoff by 112 training days and therefore cannot be reused; fresh pre-cutoff training is required.
- M5 availability masks protect targets, but history and scaling retain pre-availability zero padding; this must be frozen as a known availability/scale confound.
- The request contains inconsistent verdict-token spellings and does not explicitly freeze the external training-origin stride. Both must be resolved before a truthful preregistration.

## Not run

No preregistration, repository implementation, Gate 0 decision, 200-series smoke run, full seed, additional seed, prediction export, checkpoint, or figure was produced.

## Exact next action

Resolve and re-freeze the external trainer contract, add a real equivalence test, select a clean PyTorch environment, and normalize the verdict vocabulary plus training-origin stride; then start a newly audited attempt without reusing post-cutoff checkpoints.
