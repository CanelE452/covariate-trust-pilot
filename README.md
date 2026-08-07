# covariate-trust-pilot

A controlled numerical study of **forecasted future covariates in zero-shot Chronos-2**.

## Research question

When does an *imperfect, forecasted* future covariate add value to a zero-shot
Chronos-2 forecast, and when does it actively hurt?

This pilot only checks whether three conditions exist:

1. a region where an accurate future covariate beats a past-covariate-only forecast;
2. a region where an inaccurate future covariate is worse than a past-covariate-only forecast;
3. oracle headroom for choosing per task rather than always using or never using it.

## Methods compared

```
M0  target_only                  target history only                       (sanity baseline)
M1  past_covariate_only          target history + covariate history        (primary baseline)
M2  oracle_future_covariate      M1 + the true future covariate            (upper bound)
M3  forecasted_future_covariate  M1 + a noisy forecast of the covariate    (the object of study)
```

The incremental value of the future covariate is the **paired M1 - M3 difference**.
M0 exists only as a sanity check: the M0 - M3 difference also contains the value of
the *past* covariate and is never interpreted as the value of the future covariate.

Covariate forecast error is controlled by `lambda`:

```
x_tilde(T+h) = x_true(T+h) + lambda * sqrt(V(h)) * eta_h
```

`V(h)` is the h-step AR(1) conditional variance of the covariate on the standardized
scale, and `eta_h` is a standard-normal path shared by every `lambda` at the same
(series, origin, horizon), which makes the lambda sweep paired.

## Study 0 versus Study 1A

* **Study 0** is a known-answer linear-Gaussian simulation with closed-form MSE. It
  verifies the DGP wiring, the `lambda` definition, the posterior shrinkage formula
  and the metric implementation. It never calls Chronos-2. Study 1A does not run if
  Study 0 fails.
* **Study 1A** is the coarse Chronos-2 grid: 4 nominal covariate shares x 5 lambdas x
  2 horizons x 30 base series (1,200 M3 tasks).

## Data

**No external data is used and none is downloaded.** Every series is generated in
process from the master seed and stored under `runs/<run_id>/generated/`. The only
thing fetched from the network is the public model `amazon/chronos-2` and the Python
packages needed to run it. The model cache lives inside the project at
`.cache/huggingface` (`HF_HOME`).

## Installation

```bash
python -m venv .venv --system-site-packages   # reuses an existing CUDA PyTorch
.venv/bin/python -m pip install --upgrade pip wheel setuptools
.venv/bin/python -m pip install -e ".[dev]"
```

`--system-site-packages` is deliberate: it keeps an already working CUDA PyTorch
instead of guessing a CUDA wheel. If PyTorch is missing or incompatible, the pipeline
stops at `BLOCKED_CHRONOS_ENV` rather than installing an arbitrary build.

Requires Python >= 3.10.

## Running

```bash
.venv/bin/python -m covariate_trust.cli audit
.venv/bin/python -m covariate_trust.cli study0     --config configs/study0.yaml
.venv/bin/python -m covariate_trust.cli smoke      --config configs/pilot.yaml
.venv/bin/python -m covariate_trust.cli diagnostic --config configs/pilot.yaml
.venv/bin/python -m covariate_trust.cli admission  --run-dir runs/<diagnostic_run_id>
.venv/bin/python -m covariate_trust.cli report     --run-dir runs/<run_id>

# everything in order, stopping at the first failure
.venv/bin/python -m covariate_trust.cli pilot --config configs/pilot.yaml
```

`pilot` runs: audit -> pytest -> Study 0 -> smoke -> diagnostic -> Gates A/B/C ->
admission (only if A, B and C all PASS) -> report -> pytest.

Exit codes: `0` ok, `2` environment or input error, `3` Study 0 FAIL, `4` smoke FAIL,
`5` a gate returned FAIL or INCONCLUSIVE and downstream work stopped.

`diagnostic` is the only resumable command:

```bash
.venv/bin/python -m covariate_trust.cli diagnostic --config configs/pilot.yaml \
    --resume runs/<diagnostic_run_id>
```

## Gates

```
Gate A   Does Chronos-2 exploit an accurate future covariate?   M2 vs M1
Gate B   Is there a benefit-to-harm boundary in lambda?         M3 vs M1
Gate C   Is there oracle headroom for per-task admission?       min(M1, M3) vs best fixed
Gate D   Does a history-only selector recover part of it?       A1 / A2 vs best fixed
```

Each gate returns `PASS`, `FAIL`, `INCONCLUSIVE` or `NOT_RUN`. Gate A FAIL stops B and
C; admission and Gate D run only when A, B and C all PASS. A failed gate never triggers
an automatic change to the DGP, the grid, a threshold or the model.

## Results layout

```
runs/<run_id>/
├── config_resolved.yaml
├── manifest.json          seeds, versions, git commit, runtime, peak GPU memory
├── audit.json             environment and the inspected Chronos API signature
├── environment.txt
├── generated/             series.parquet, series_metadata.parquet, covariate_vintages.parquet
├── predictions/           parts/<task_hash>.parquet and the merged predictions.parquet
├── tables/                task_metrics.parquet, cell_summary.csv, monte_carlo_se.csv,
│                          bootstrap_summary.csv, gate_report.json, gate_d_report.json
├── figures/               figure1..figure6 (one figure per file, no subplots)
├── reports/report.md
└── logs/run.log
```

Run directories are never reused or overwritten, and every file is written to `.tmp`
first and renamed on success.

## Committed results

`runs/` is not tracked (it holds hundreds of megabytes of parquet and per-task
predictions). A snapshot of one complete run is committed instead:

```
results/report.md      the full report for the 2026-07-30 pilot run
results/figures/       the ten figures from that run
```

Everything else in that run is reproducible with `cli pilot`; two independent
diagnostic runs produced bit-identical WQL values.

## Out of scope for this pilot

External datasets, real weather or demand data, any data download, fine-tuning, other
TSFMs, uncertainty propagation, predictive quantile mixtures, nonlinear
target-covariate relationships, biased or serially correlated covariate forecast error,
inventory cost, Study 1B boundary refinement, and neural selectors.

The report states outcomes as measured. Nothing about the result is asserted here in
advance.

## External-validity SCREEN (M5 / Favorita)

`experiments/external_validity_screen/` — 합성 study의 Point 대 Hurdle 상대우위 조건이
두 공개 소매 벤치마크에서 재현되는지 확인하는 SCREEN.

**진행 상태·결과·다음 단계는 [`docs/external_validity_STATUS.md`](docs/external_validity_STATUS.md)
를 먼저 읽으세요.** 데이터 유도 과정은 [`docs/m5_favorita_data_derivation.md`](docs/m5_favorita_data_derivation.md).
