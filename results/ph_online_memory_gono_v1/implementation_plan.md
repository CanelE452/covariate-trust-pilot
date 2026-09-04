# PH-ONLINE-MEMORY-GONO-v1 Implementation Plan

> **For agentic workers:** implement task-by-task with tests first. Do not commit, push, modify prior artifacts, or continue after a failed gate.

**Goal:** Execute the failure-first Point/Hurdle online-memory Go/No-Go pilot without changing the frozen routing chain.

**Architecture:** A new package owns split construction, the canonical-trainer adapter, pure metric/policy/retrieval logic, artifact validation, gates, and orchestration. The adapter calls the real canonical `train_one`; only its internal split-builder lookup is scoped to the already validated external `Split`. All scientific output is append-only under `results/ph_online_memory_gono_v1/`.

**Tech stack:** Python 3.10.20, PyTorch 2.1.1+cu118, NumPy 1.26.4, pandas 2.3.3, SciPy 1.12.0, scikit-learn 1.4.1.post1, PyArrow 25.0.1, standard-library `unittest`.

**Spec:** User attachment SHA-256 `B606BD0E115E4170E5AB844E73D3CDB988D89FF9139DF3CF270E6E7E977E9C4E`; Phase 0 evidence is in `results/ph_online_memory_gono_v1/audit.json` and authorized rulings are in `audit_resolution.json`.

## Global constraints

- Create code only under `experiments/ph_online_memory_gono_v1/` and results only under `results/ph_online_memory_gono_v1/`.
- Never modify the listed frozen result directories, prior stop rules, canonical model/trainer source, or user dirty-worktree files.
- Never reuse checkpoints or predictions trained with the old, later cutoffs for the six-origin run.
- Freeze `preregistered_spec.json` and its content hash before any model fit, including trainer-equivalence fitting.
- Use model seed 0 first; run seeds 1 and 2 only under the exact conditional rules.
- Use no `KMP_DUPLICATE_LIB_OK`, no package installation, no concurrent in-process fits, and no commit or push.
- Stop immediately at the first failed reproduction, runtime, leakage, pairing, or scientific gate.

## Preflight cross-impact matrix

```text
Area                  Producer                                Consumer                         Ruling
trainer split         data.build_split                       trainer.train_one_on_split       validate masks/origins before delegation
canonical execution   scoped build_splits substitution       canonical train_one              sequential/process-isolated only
model scale           canonical train_scale                  model forward                     preserve existing mean-scale behavior
policy loss scale     metrics.policy_scale_squared           B3/B4/M1/gates                   separate RMS-squared contract
prediction schema     prediction_frame                       metrics/reproduction             unique dataset/series/origin/step keys
online memory         resolved cases with end <= query       B4/M1/C0/C1                      update only after scoring current origin
source tuning         source-only frames                     target scoring                   target frames absent from tuner signatures
artifacts             exclusive-create writer                every stage                      refuse overwrite
GPU lifecycle         sequential Point then Hurdle fit        smoke/full runner                clear references/cache between fits
```

### Task 1: Contract tests and canonical trainer adapter

**Files:**

- Create `experiments/ph_online_memory_gono_v1/__init__.py`
- Create `experiments/ph_online_memory_gono_v1/trainer.py`
- Create `experiments/ph_online_memory_gono_v1/tests/test_trainer.py`

**Interface:**

```python
def train_one_on_split(
    model_name: str,
    split: km_train.Split,
    cfg: ExperimentConfig,
    model_seed: int,
    device: torch.device,
) -> dict:
    """Return the canonical train_one result unchanged."""
```

- [ ] Write a real equivalence test for both canonical model IDs using more than 256 training windows and masked targets.
- [ ] Run it and confirm RED because the new module is absent.
- [ ] Implement only the scoped split substitution and canonical call.
- [ ] Defer GREEN execution until preregistration exists because the test performs model fitting.

### Task 2: Pure split, metric, policy, retrieval, bootstrap, and gate contracts

**Files:**

- Create `contracts.py`, `data.py`, `metrics.py`, `policies.py`, `retrieval.py`, `bootstrap.py`, `gates.py`, and `artifacts.py` in the new package.
- Create focused tests under `experiments/ph_online_memory_gono_v1/tests/`.

**Required observable tests:**

```text
T1  retrieval key reads only history positions < query origin
T2  every memory case satisfies case origin + horizon <= query origin
T3  source tuners accept no target-evaluation argument
T4  same-series neighbors are excluded
T5  Point/Hurdle predictions pair completely on dataset/series/origin/step
T6  final origin + horizon equals the actual series length
T7  policy scale uses only model_train values
T8  eligibility uses only model_train positive count
T9  bootstrap resamples whole six-origin series clusters
T10 exponential weighting remains finite for extreme losses
T11 constant Spearman predictor returns DEGENERATE
T12 hard and convex oracle ladders use family-matched denominators
```

- [ ] Add one failing behavioral test at a time and observe the intended failure.
- [ ] Add the minimal pure implementation and observe GREEN before moving on.
- [ ] Keep source tuning signatures source-only by construction.

### Task 3: Read-only three-origin reproduction

**Files:**

- Create `reproduction.py` and `tests/test_reproduction.py`.

**Interface:**

```python
def reproduce_three_origin(raw_paths: dict[str, Path], reference_root: Path) -> dict:
    """Read, validate, recompute, and return a report without writing."""
```

- [ ] Validate exact schema, row counts, keys, masks, finite values, occurrence identity, and Hurdle factorization.
- [ ] Rebuild the 33,294-row panel and compare to the frozen panel with absolute tolerance `1e-7`.
- [ ] Reproduce frozen aggregates with their audited tolerances.
- [ ] Hash every frozen input before and after; fail on any change.

### Task 4: Freeze preregistration before any fit

**Files:**

- Create `prereg.py` and `cli.py`.
- Create `results/ph_online_memory_gono_v1/preregistered_spec.json` exactly once.

- [ ] Record repository, environment, model/trainer identities, source artifacts, split, seeds, grids, retrieval transformations, gates, stop rules, and all rulings.
- [ ] Hash the canonical JSON payload excluding the hash field; record the algorithm and scope next to the digest.
- [ ] Record hashes of every implementation and test file because the Git worktree is intentionally uncommitted.
- [ ] Refuse to overwrite the frozen file.

### Task 5: Verify code and run Stage 0 reproduction

- [ ] Run all pure tests in the clean environment.
- [ ] Run the trainer equivalence test after preregistration; require exact selected weights and predictions for Point and Hurdle on CPU.
- [ ] Execute the read-only three-origin reproduction and exclusively create `stage0_reproduction.json`.
- [ ] Stop on mismatch, NaN/Inf, underflow, degeneracy mishandling, or any frozen-input hash change.

### Task 6: M5 200-series smoke and runtime gate

**Files:**

- Create `smoke.py` and smoke-specific tests.
- Exclusively create `runtime_estimate.json` and a `smoke/` artifact directory.

- [ ] Load full M5, recompute cutoff eligibility, remove Stage-A IDs, and select 200 deterministic two-dimensional strata over zero ratio and log model train scale.
- [ ] Build stride-7 train, dense validation, warmup, and first-evaluation windows; apply availability masks only to train/validation targets.
- [ ] Fit Point then Hurdle seed 0 through the canonical adapter.
- [ ] Generate warmup and first-origin paired predictions, normalized metrics, one causal B4 update, and one same-series-excluded retrieval query.
- [ ] Record wall time, inference/origin, peak CUDA memory, output-size estimate, projected full seed-0 GPU-hours, and storage.
- [ ] If projected total exceeds six GPU-hours, calculate the per-dataset 2,000-series alternative and stop for user approval.

### Task 7: Conditional full seed-0 and Gate 0/1A/1B

- [ ] Run only if the runtime gate passes.
- [ ] Train four fresh experts on the full independent eligible pools and save full provenance/checkpoints.
- [ ] Predict warmup plus six origins with 100% Point/Hurdle pairing.
- [ ] Evaluate family-separated Gate 0; on failure run only the available TSB/SBA diagnostic and issue the exact section-40 verdict.
- [ ] If Gate 0 passes, evaluate recurrence; stop on Gate 1A failure.
- [ ] Tune B3/B4 on source only and score target prequentially; stop on Gate 1B failure.

### Task 8: Conditional retrieval, controls, bootstrap, safety, and seeds

- [ ] Tune `k` and `lambda_max` on source only with B4 hyperparameters fixed.
- [ ] Score target M1 causally, then C0 and C1 with frozen seeds.
- [ ] Run 2,000 series-cluster bootstrap draws and Gate 2/3.
- [ ] Run seed 1 and possibly seed 2 only under the exact borderline rules.
- [ ] Stop immediately on any failed gate.

### Task 9: Final report and verification

- [ ] Exclusively create tables A–G, at most three figures, `final_gate_report.json`, and the exact ordered `STATUS.md` content appropriate to the reached terminal state.
- [ ] Re-hash all forbidden result directories and require equality with the Phase 0 baseline.
- [ ] Run the full new-package test suite and verify every created artifact exists, parses, and traces to its source.
- [ ] Report the exact verdict first, followed by gates, transfer values, worst origin, controls, GPU time, files, and one next action.

## Execution command convention

```powershell
$env:PYTHONPATH = "E:\CODING\proj\covariate-trust-pilot;E:\CODING\proj\covariate-trust-pilot\src"
Remove-Item Env:KMP_DUPLICATE_LIB_OK -ErrorAction SilentlyContinue
& 'C:\Users\User\anaconda3\envs\pallet-pose\python.exe' -B -m unittest discover -s experiments/ph_online_memory_gono_v1/tests -v
```

Every writer uses exclusive creation and every training command is single-process and sequential.
