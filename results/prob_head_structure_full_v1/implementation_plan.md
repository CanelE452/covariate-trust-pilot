# PROB-HEAD-STRUCTURE-FULL-v1 implementation plan

Date: 2026-09-05

Authority: the user's 70-section protocol in the task that created this plan. This
plan resolves only implementation details left open by that protocol. Scientific
thresholds, dataset priorities, seeds, and the diagnostic-continuation policy are
not negotiable after preregistration freezes.

## Global constraints

- Work only on branch `prob-head-structure-full-v1` in the isolated worktree.
- Never edit any of the 14 protected result directories. Capture a full relative
  path, byte count, and SHA-256 manifest before fitting and compare it after all
  work.
- Treat future-target use, index overlap, prediction-key mismatch, source mutation,
  support mismatch, invalid probability distributions, unvalidated full Tweedie,
  and post-result tuning as integrity failures exactly as the protocol states.
- Scientific gate failures never stop later scientific stages. They set downstream
  `confirmatory_eligible=false` and the required
  `DIAGNOSTIC_CONTINUATION_AFTER_<GATE>` role.
- Follow strict TDD: write a behavior test, run it and observe the expected failure,
  add the smallest implementation, then rerun the focused test.
- Do not commit during implementation. The user's explicit contract requires all
  tests and the final protected-artifact comparison to pass before the first
  commit. Review task diffs without intermediate commits.
- Keep large predictions, samples, checkpoints, source downloads, and the isolated
  environment under `runs/prob_head_structure_full_v1/`.
- Keep `results/prob_head_structure_full_v1/` compact and append-only. Never replace
  a successful stage attempt.
- Do not use `KMP_DUPLICATE_LIB_OK`, Tweedie deviance, rounded continuous targets,
  validation/test targets in sampling or scaling, or cross-family NLL ranking.

## Frozen implementation rulings

These decisions are made before any new-model fit.

1. The recovered Stage-2 generator at source commit
   `03043927c1ea89e8569ab4e9d0bfe0b6d041a47f` is the generator authority. Vendor
   its symmetric event-index Markov construction, independent RNG streams, gap
   support `{d-1,d+1}`, and `1+Poisson(lambda-1)` magnitude law. Adapt only the
   user-specified split to `[0,380)`, `[380,408)`, `[408,436)`, five origins from
   436, horizon 28. Cross-cell base innovations are explicitly keyed without rho,
   then transformed by the rho-specific Markov transition, so the new experiment
   has stronger common-random-number pairing than the recovered run; this is
   reported as an adaptation, not attributed to the old run. The common base is
   d-specific: cluster `(data_seed,d,base_series_index)` attaches all nine rho cells,
   model seeds and origins at that d, but never pairs d=4 with d=8. Canonical output
   columns are `data_seed,d,rho_I,rho_M,base_innovation_id` and inspected generator
   arguments `rho_interval/rho_magnitude` map explicitly to `rho_I/rho_M`.
2. DGP balance tolerances are exactly 0.02 absolute zero-rate, 5% relative positive
   mean, and 10% relative positive variance. Failed cells remain in diagnostic
   output and lose confirmatory eligibility.
3. Reuse the verified DLinear decomposition with moving-average kernel 25. Each
   probabilistic teacher has the same two `Linear(lookback,horizon)` trunk maps and
   one deterministic hidden head adapter. Adapter widths are NB=19, HSNB=14,
   Tweedie=29 for output multiplicities 2, 3, and 1; the two Tweedie global scalars
   are counted. Construction must prove the maximum parameter-count spread is at
   most 5% before fitting.
4. Teacher initialization is `phi=1`, `p=1.5`, epsilon `1e-6`; Tweedie uses
   `phi=softplus(raw_phi)+eps` and `p=1.05+0.90*sigmoid(raw_p)` exactly.
5. Reuse Adam, learning rate 1e-3, weight decay 0, effective batch 256, maximum 30
   epochs, and patience 5. Synthetic training origins are dense. Real teacher
   training origins use the verified stride 7. Student training uses non-overlap
   stride 28. Teacher loss is the sum of full NLL over valid cells divided by their
   count; all-masked rows are removed deterministically. Validation checks are the
   1-based epochs 2,4,...,30, with an additional last executed/max-epoch check only
   if not already checked. Five consecutive scheduled checks without strict
   `score < best` improvement stop training at that check; ties retain the earliest
   checkpoint. An OOM discards partial state and restarts the same seed/order once
   at half microbatch and double accumulation; a second OOM blocks that model branch.
6. The common CRPS quantile grid is
   `[.01,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,.99]`.
   CRPS is `2 * sum(midpoint-cell-width * pinball)` with probability cells bounded
   by 0 and 1. All scientific checkpoint, gate, and pool-selection metrics use
   deterministic native head quantiles or numerical inversion of a linear CDF pool;
   student/P3 metrics use their deterministic monotone common-grid representation.
   The 256/1024 draws are diagnostic sample/moment-stability checks only and cannot
   create a confirmatory evaluation result. No scientific metric uses an empirical
   sample CDF.
7. The full Tweedie training density is minimally adapted from
   `StefanoDamato/TweedieGP@2567d1322c8cc65f19df4f2d1774c610b167fb66`, with
   Apache-2.0 license and modification notice. CPU-only helper tensors and the
   float32-only output buffer are fixed. `FixedDispersionTweedie.log_prob` is never
   used. `thequackdaddy/tweedie@f14a189...` / PyPI `tweedie==0.0.9` is reference
   only. An independent compound-Poisson–Gamma expansion is a second oracle.
8. The Tweedie validation grid crosses all 5 mu, 4 phi, and 5 p levels (100
   parameter points), each at y=`[0,.05mu,.25mu,mu,2mu,5mu]`. The user's finite,
   zero-mass, log-density, and monotone-CDF tolerances are applied without dropping
   or clamping failed points.
9. P2 global CDF weights use every 0.1-simplex triple (66 states). P3 uses the same
   states at each quantile. For each fixed penalty `[0, 0.01, 0.1, 1.0]`, minimize
   validation mean pinball plus mean adjacent squared weight change, breaking an
   exact tie by the lexicographically smallest full path. Select a separate p0
   simplex by validation Brier. Postprocess by cumulative maximum, q<=p0 zeroing,
   then cumulative maximum; select the penalty by postprocessed validation sCRPS,
   tail loss, smaller penalty, then path. Report pre/post crossing and zero-adjustment.
10. The student reuses MA-kernel 25 and two `Linear(96,horizon)` maps, followed by
    one shared scalar-per-horizon `1->16->22` SiLU MLP. Training emits a softplus
    base and 20 softplus increments; q<=p0 zeroing occurs only at validation/eval.
    BCE means divide by valid cells and quantile means by valid cells times 21;
    numerators and denominators, not microbatch means, accumulate across microbatches.
    Parameter count is checked against the smallest teacher before fitting. Student
    optimizer, batch, epochs, patience, train rows, and seeds are identical within
    each declared A or B comparison.
11. Router soft labels are fit by expanding each row once per class and passing the
    soft probability as sample weight to multinomial logistic regression. Median
    imputation is always paired with missingness indicators; an all-missing fold
    feature uses raw zero, indicator 1, scaler mean 0/scale 1 and records
    `ALL_MISSING_TRAIN_FEATURE`. At k=2 temperature is fixed to 0.25; at k>=3 each
    temperature is evaluated by strictly nested OOF origins 2..k-1, then the chosen
    temperature fits origins <k. Earlier heldout weights are never backfilled.
    Router HGB regressors and sensor HGB classifiers remain secondary.
12. M5 and OnlineRetail are the expected primary real datasets. Auto (L=24,h=6),
    Carparts (L=51,h=6), and RAF (L=84,h=12) are audited in the fixed priority but
    cannot supply six evaluation horizons plus warmup, validation, and a 48-step
    lookback. OnlineRetail uses the exact TweedieGP UCI-352 aggregation; the local
    second sheet reproduces 2,036 series x 374 days and is accepted only if hashes,
    dates, row count, support, and reproduced dimensions match. Its frozen split is
    train `[0,150)`, validation `[150,178)`, warmup `[178,206)`, evaluation origins
    206..346, lookback 96, horizon 28. A dataset becomes eligible only after literal
    source hash, construction, COUNT SUPPORT, exact PMF-index and fixed-geometry
    audits all pass; a length/geometry check alone cannot select it.
13. The rounded Favorita artifact is not eligible for the count-primary track even
    if its stored values look integral, because its provenance explicitly applies
    `rint`. A continuous-support Favorita diagnostic is eligible only in FULL tier
    and must read the raw target without rounding.
14. The 200-series timing smoke uses a synthetic `d=4,rho_I=0,rho_M=0` block and is
    never scientific evidence. Measure head-specific train rates plus end-to-end
    21-grid native, pooled-inversion, P3-direct and student-direct GPU case rates.
    Project the complete FULL inventory workload (M5 N<=4000, OnlineRetail N<=2036),
    every 30-epoch fit, 15 validation checks, candidate pool path and required outer/
    train prediction; add 25%. The `N*W` inventory geometry is a conservative upper
    bound and is not the later support audit. Select the tier once and never recompute.
15. B router origins are the last eight non-overlapping valid model-train origins if
    available, otherwise eight deterministic evenly spaced valid origins; fewer than
    four blocks B. Each inner descriptor uses only its origin prefix. C sensor pairs
    instead require `t>=lookback` and `[t,t+2h)` wholly in model_train. M5 uses
    `[1465,1493,1521,1549,1577,1605,1633,1661]`; OnlineRetail has no valid pair and is
    `REAL_C_SENSOR_GEOMETRY_BLOCKED`, so C1 deterministically cannot pass two datasets.
    Fixed teacher checkpoints are the sole explicit full-model-train exception.
16. A sensor decision at t occurs after `[t,t+h)` is observed and predicts the next
    horizon, never reading index >=t+h. C1 is target-1/C2-logistic only; targets 2–4
    are diagnostic. The validation q80(method=higher) threshold flags only scores
    strictly greater; zero flagged rows fails C3. Widening factors are selected on
    flagged validation rows by mean(CE90,CE95), sCRPS, smaller factor and apply also
    to q=.025/.975. Safest fallback is chosen over all validation rows by the same
    coverage composite, then sCRPS/head order. The final action is selected on flagged
    validation rows by sCRPS, coverage composite, and fixed A0<A1<A2<A3 order.
17. C-SYN uses 24 series per `(d,shift-type)` at minimum and the same tier multiplier
    as S1 otherwise. The calendar changepoint is 288; event-state continuity is
    preserved while the transition kernel changes. Its exact features are the four
    user-listed disagreement levels plus their four deltas; the first delta is
    missing+indicator and all 16 origins remain. Train-half medians/moments alone
    transform heldout rows. Interval/magnitude component separation uses strict
    standardized contrasts; zero/nonfinite train-pre SD fails separation. The
    no-change control applies a calendar pseudo-post label to the frozen sensor score.
18. A branch can be confirmatory only if every named upstream gate passed when it
    began. Later diagnostic success cannot mutate that field or an earlier verdict.
    Every real aggregate, latency and negative-control gate must carry the identical
    content-bound primary-dataset manifest; Tweedie validation and synthetic C2 are
    the only declared exemptions.

## Task 1 — audit, immutable preregistration, and append-only execution contracts

Files:

- `tests/prob_head_structure_full_v1/test_integrity.py`
- `tests/prob_head_structure_full_v1/test_preregistration.py`
- `experiments/prob_head_structure_full_v1/integrity.py`
- `experiments/prob_head_structure_full_v1/preregistration.py`
- package initializers and third-party notices

Implement path-safe protected-directory manifests, payload/file hashing, immutable
preregistration freeze with companion file hash, seed derivation, split/support
contracts, atomic attempt directories, completion markers, resume idempotence,
hard/scientific status types, and branch eligibility propagation. Generate the
actual before manifest and frozen preregistration only after focused tests pass.

## Task 2 — distributions and numerical validation

Files:

- `tests/prob_head_structure_full_v1/test_distributions.py`
- `tests/prob_head_structure_full_v1/test_tweedie_reference.py`
- `experiments/prob_head_structure_full_v1/distributions.py`
- `experiments/prob_head_structure_full_v1/vendor/tweediegp/*`
- `experiments/prob_head_structure_full_v1/numerical_validation.py`

Implement NB, shifted-hurdle NB, and full Tweedie under one `[batch,horizon]`
interface (`log_prob`, `mean`, `p_zero`, `cdf`, `quantile`, `sample`). Cover T04-T09,
T12-T13 and the independent 600-evaluation grid. Persist compact validation rows
and block only the Tweedie branch if its preregistered tolerances fail.

## Task 3 — verified DGP/data adapters, models, windows, and training

Files:

- `tests/prob_head_structure_full_v1/test_data.py`
- `tests/prob_head_structure_full_v1/test_models_training.py`
- `experiments/prob_head_structure_full_v1/synthetic.py`
- `experiments/prob_head_structure_full_v1/data.py`
- `experiments/prob_head_structure_full_v1/models.py`
- `experiments/prob_head_structure_full_v1/training.py`

Vendor/adapt the recovered generator, implement known-change sequences, support
audits, deterministic stratified sampling, fixed splits and origin-key frames.
Implement parameter-matched heads and sCRPS checkpoint training with one OOM retry
that halves microbatch and doubles accumulation. Cover T01-T03, T10-T11, checkpoint
restore, identical keys, DGP balance, and source-hash guards.

## Task 4 — probabilistic metrics, pools, bootstrap, and gates

Files:

- `tests/prob_head_structure_full_v1/test_evaluation.py`
- `tests/prob_head_structure_full_v1/test_gates.py`
- `experiments/prob_head_structure_full_v1/evaluation.py`
- `experiments/prob_head_structure_full_v1/pooling.py`
- `experiments/prob_head_structure_full_v1/bootstrap.py`
- `experiments/prob_head_structure_full_v1/gates.py`

Implement all common metrics, exact prediction-key joins, deterministic native and linear-CDF inversion,
P0-P3 selection, series/base-innovation cluster bootstrap, dataset macro effects,
S/R/A/B/C thresholds, practical winners, temporal contrasts, and immutable upstream
verdict propagation. Cover T12-T13 and T18-T21 with hand-derived fixtures.

## Task 5 — students, routing, sensors, actions, and controls

Files:

- `tests/prob_head_structure_full_v1/test_methods.py`
- `experiments/prob_head_structure_full_v1/student.py`
- `experiments/prob_head_structure_full_v1/temporal_features.py`
- `experiments/prob_head_structure_full_v1/routing.py`
- `experiments/prob_head_structure_full_v1/sensor.py`
- `experiments/prob_head_structure_full_v1/controls.py`

Implement A0-A4, B0-B2, C0-C3, C-A0..A3 and every named negative control. Ensure
student non-crossing, training-row parity, inner expanding cross-fit, no outer-label
fit, next-origin alignment, deterministic shuffles, and component-wise disagreement.
Cover T14-T17 and T19.

## Task 6 — end-to-end orchestration, reports, figures, and regeneration

Files:

- `tests/prob_head_structure_full_v1/test_orchestration.py`
- `experiments/prob_head_structure_full_v1/run.py`
- `experiments/prob_head_structure_full_v1/reporting.py`
- `experiments/prob_head_structure_full_v1/figures.py`
- `experiments/prob_head_structure_full_v1/__main__.py`

Implement stages S0, S1/S2, C-SYN, real audit/R1/R2/pools, A, B, C, controls,
bootstrap, gates, compact Tables A-T, at most eight figures, the exact STATUS
section order, source-artifact recomputation, 24-hour cap, retry/resume, and concise
final console output. Cover T22-T24 on synthetic fixtures without fitting.

## Task 7 — execute, verify, document, commit, and push

1. Run focused tests and full suite.
2. Freeze the pre-fit protected-artifact manifest and preregistration.
3. Run likelihood validation, DGP audit, 200-series smoke, and freeze the runtime
   tier before any scientific stage.
4. Execute every remaining stage in protocol order, continuing after scientific
   failures and resuming only incomplete attempts.
5. Generate Tables A-T, no more than eight figures, and `STATUS.md` from persisted
   source artifacts without re-running experiments.
6. Recompute every compact number, compare the protected manifest, run the full
   suite, and perform whole-branch scientific/code review.
7. Stage only user-authorized paths by explicit filename/pathspec, inspect the
   staged diff for secrets and large blobs, create the final commit, then push the
   experiment branch if origin/auth/LFS checks remain safe. Record SHA/push outcome
   in STATUS without rewriting frozen scientific results.
