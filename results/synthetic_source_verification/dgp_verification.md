# DGP verification

Read from the recovered source, not from history or from `results/paper_synthesis/`.

```
source repository   /home/minjae/Documents/github/m5dataset  (Ubuntu, minjae-MS-7D17)
HEAD                03043927c1ea89e8569ab4e9d0bfe0b6d041a47f   branch master
extracted to        E:/research_recovery/m5dataset_recovery-20260811_135203/extracted/m5dataset
paper DGP           experiments/decomposition_when_helps/
prereg              experiments/decomposition_when_helps/prereg.py
prereg hash         3b6b5f89ba24d42aaab6f65dc39526096578dbfcc4820e0284bbf92736bab9f9
```

---

## 1. Why this is the paper DGP, on provenance rather than filename

Three independent identifiers, recorded in the source's own
`reports/temporal_dependence/stage1_audit/provenance.md` and re-checked here:

1. `run_20260802_112655` is the **only** run in the repository carrying
   `factorial_contrasts.csv`.
2. It is the **only** run whose cells are the eight-cell `C01..C08` 2×2×2 layout.
   `om_factorization_killtest` runs use a four-cell `C1_adi2_magOFF..` layout.
3. Its stored contrasts reproduce the presented numbers exactly:
   interval **+7.83**, magnitude **−4.58**, sparsity **−6.26**,
   interval×magnitude **−16.74**. Verified against
   `stage1_verified_contrasts.csv` in this directory.

A fourth identifier, found independently on the Windows side before the archive
arrived: the migrated audit note recorded the synthetic study's geometry as
`lookback 96, horizon 24, length 576, train/val/test = 384/480/576`. The
recovered `prereg.py` has `retained_length 576`, `lookback 96`, `horizon 24`,
`train [0,384]`, `validation [384,480]`, `test [480,576]` — an exact match.

### Candidates excluded, with reasons

```
candidate                              excluded because
────────────────────────────────────────────────────────────────────────────────────
om_factorization_killtest              its own prereg says verbatim "exploratory
                                       kill-test DGP, not the paper DGP"; four-cell
                                       layout; sinusoidal occurrence
unified_temporal_27_v2 / v3 / v4       27-scenario catalogue, length 512, split
                                       358/409; no factorial contrast file
sparsity_factorization_boundary        no C01..C08 layout, not linked to the
occurrence_predictability_boundary     presented contrasts
paired_temporal_hurdle_v1              per-epoch logs only; the "16.74" string hits
                                       there are numeric coincidences
```

**Important nuance.** `om_factorization_killtest` is excluded as a *DGP*, but the
paper study **imports its models and its trainer unchanged**
(`MODELS` point at `om_factorization_killtest.models.*`, and `TRAINING.reuse` is
`om_factorization_killtest.train.train_one, imported unchanged`). So the earlier
Windows-side reading — "the kill test disclaims itself, therefore nothing in it is
the paper's" — was too strong. Its DGP is not the paper's; its model and trainer
code are.

---

## 2. The generating process, as written

```
event time      T_j = T_{j-1} + Q_j ;  Y_t = M_j at t = T_j, else 0
gap support     short = d-1, long = d+1, each with long-run share 0.5
sparsity        d in {4, 8}
magnitude law   M_j = 1 + Poisson(lambda_j - 1), so M_j >= 1 and E[M_j|lambda_j] = lambda_j
lambda levels   {5, 15};  long-run positive mean is 10 in BOTH modes
burn-in         max(128, 8*d)
```

| axis | predictable arm | independent control |
|---|---|---|
| occurrence interval | `Q_j` alternates `d-1, d+1, …`, first parity drawn per series | `Q_j` iid, `P(d-1) = P(d+1) = 0.5` |
| positive magnitude | `lambda_j` alternates `5, 15, …`, first parity drawn per series | `lambda_j` iid, `P(5) = P(15) = 0.5` |

**Marginal control (C1-G2): verified.** Both arms of each axis draw from the same
two-point support with the same long-run frequency. Only the *order* differs. The
prereg states it for magnitude explicitly — "the long-run positive mean is 10 in
both modes" — and the audit measured it: `marginal_control_pass: true`, with the
d=4 empirical ADI mode difference at 0.0015 and the mean-gap mode difference of
the same order.

**Pairing.** `paired_random_numbers`: the same event index draws the same uniform
for the gap choice and the same Poisson innovation in both modes, so predictable
and independent conditions are paired without redefining either.

**Excluded by construction.** `forbidden`: trend, calendar seasonality, hidden
regime, heavy tail, test-time shift, phase jitter, interval–magnitude
cross-correlation.

**Leakage.** One continuous trajectory is generated and then split; the gap
sequence, magnitude parity and latent state are never reset at a split boundary.
Normalization uses the train split only. The oracle conditions on "the latent
process state implied by the history strictly before the origin … Nothing at or
after the origin is read."

---

## 3. The oracle the metric is measured against

```
method        exact dynamic programming over (steps to next event, next-gap parity,
              next-magnitude parity), advanced across the horizon
independent   in-progress gap conditioned on survival: having seen no event for u
  interval    steps rules out any gap shorter than u; remaining support integrated
              with renormalised probabilities
independent   conditional positive mean is 10
  magnitude
Monte Carlo   cross-check only, never primary; 200,000 paths
```

This is why the synthetic delta and the real-data delta are **not numerically
comparable**: the synthetic target is an exact conditional mean, the real-data
target is realized `y`.

---

## 4. Replication and seeds (C1-G8)

```
n_series_per_cell_per_seed   40
data_seeds                   (0, 1)
model_seeds                  (0, 1)
n_series per cell            80        (40 x 2 data seeds; verified in metrics_by_cell.csv)
model seed handling          averaged within a series before comparing
bootstrap                    series unit, 2000 draws, paired, level 0.95, seed 20260802
seed key                     BLAKE2b over (DATASET_VERSION, name, d, data_seed, series_i)
```

---

## 5. One defect the source found and recorded

`experiments/temporal_dependence/` audit A2 found that the `_event_stream` helper
would have let magnitude read `gap_uniform`, creating an interval–magnitude
cross-dependence that the design forbids. It is **dead code with no call site**,
so the executed runs are unaffected. Recorded here because the paper's
"no cross-correlation by construction" claim rests on it.

---

## 6. Verdict

```
C1-G1  paper DGP identified by provenance          PASS
C1-G2  marginal control verified                   PASS
C1-G3  temporal dependence manipulation verified   PASS
C1-G8  seed / replication provenance               PASS
```
