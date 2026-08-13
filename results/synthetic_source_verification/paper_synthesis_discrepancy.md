# paper_synthesis vs the recovered source

`results/paper_synthesis/` is unmodified. Every claim in it that touches the
synthetic study is compared here against the recovered artifacts.

Grades: **MATCH** / **MINOR_MISMATCH** / **MAJOR_MISMATCH** / **UNVERIFIABLE**.

---

## 1. Contribution C1 — "controlled characterization of relative Point/Hurdle inductive bias under temporal occurrence and magnitude dependence"

**MATCH.** The source manipulates exactly those two axes with the marginals held
fixed, on a parameter-matched pair of the same backbone. Everything C1 asserts is
now backed by `run_20260802_112655` and the rho sweep.

C1 was carried at `PARTIALLY_VERIFIED` only because the artifacts were absent. That
reason is gone.

---

## 2. Stage naming — "Stage 1 / Stage 3 / Stage 4"

**MINOR_MISMATCH.** The source has **Stage 1** (eight-cell factorial) and
**Stage 2** (rho sweep). There is no Stage 3 or Stage 4.

```
paper_synthesis says     actual source
─────────────────────────────────────────────────────────────────────
Stage 1                  Stage 1 — decomposition_when_helps,
                         run_20260802_112655, C01..C08
Stage 4                  Stage 2 — temporal_dependence rho sweep,
                         pilot_20260803_051713, 18 cells
Stage 3 (mechanism)      not a stage. The mechanism material is the hybrid
                         diagnostic columns inside the Stage 1 run
                         (p_true_x_mu_hat / p_hat_x_mu_true / p_hat_x_mu_hat)
```

Fix: rename throughout the verified derivative. Nothing scientific changes.

---

## 3. "the synthetic delta is measured against an exact DP oracle and is not numerically comparable to the real-data delta"

**MATCH**, and now provable. `PRIMARY_METRIC = rmse_mean_truth`, target
`mean_true = p_true * mu_true` from exact dynamic programming; the real-data
estimand is RMSE against realized y.

---

## 4. "H3 was derived from an ADI 4 versus 8 contrast, and the external test split at the ADI median"

**MATCH.** `SPARSITY = (4, 8)` in the source; the external split is at 1.304 / 1.317.
The concern recorded in `claim_ledger.md` L5 is correct.

---

## 5. H1 wording — "occurrence-interval dependence is positively associated with the relative advantage of the factorized model"

**MATCH in direction, and the source is stronger than `paper_synthesis` claims.**

`paper_synthesis` treats the external use of `abs(rho_interval)` as an operational
choice. The source shows it is the *finding*: `|rho_I|` carries a coefficient of
+0.1904 against +0.0667 for signed `rho_I`, and both signs of dependence produce a
positive gain. This is an under-claim to be corrected upward, not an error.

---

## 6. H2 motivation — "a Point-favorable condition exists in the controlled study"

**MATCH, and sharper than stated.** Exactly one of eighteen cells is
Point-favourable: `d=8, rho_I=0, rho_M=+0.8`, gain **−19.76% [−26.00, −14.53]**.
Its three axes map one to one onto the frozen external rule, sign included.

---

## 7. Occurrence-mechanism wording

**MINOR_MISMATCH — needs a sharper split.** `hypothesis_final_status.md` currently
records the occurrence mechanism as "REJECTED as a real-data mechanism claim". That
is right about real data, but the recovered source shows the mechanism operating
inside the controlled study (hybrid diagnostics: C03 M1, occurrence head 0.9030
against magnitude head 0.2900). The verified derivative should carry two labels:

```
SYNTHETIC_DIAGNOSTIC_SUPPORTED          within the controlled study
REAL_LEARNED_GATE_NOT_SUPPORTED         Brier skill -0.008 (M5), -0.091 (Favorita)
```

The single word "REJECTED" collapses two different questions.

---

## 8. Figure 2 plan — "heatmap of relative gain over the (rho_interval × rho_magnitude) grid, one panel per sparsity level, with the Point-favourable cell marked"

**MATCH, and the source data exists exactly in that shape.** 18 cells =
2 sparsity × 3 rho_I × 3 rho_M, each with gain and a bootstrap CI. The
Point-favourable cell is identifiable. Status moves from `BLOCKED` to
`SOURCE_READY`.

---

## 9. Abstract sentence 4 placeholder

**Now fillable.** Proposed, using only verified numbers and no cell-by-cell listing:

> Holding the interval support, the mean inter-demand interval and the magnitude
> marginal fixed, temporal dependence in the occurrence process moves the comparison
> toward the factorized model — the coefficient on the strength of that dependence
> is +0.19 and its sign does not matter — while magnitude persistence moves it back
> toward the direct model, leaving a single configuration in eighteen where the
> direct model wins by about twenty percent.

---

## 10. `exact_source_artifacts.md` — "Stage 1 / Stage 3 / Stage 4 controlled synthetic study: ABSENT from this repository"

**Superseded.** True when written, false now. The verified derivative replaces it
with the recovered paths and hashes in `source_manifest.json`.

---

## 11. Earlier Windows-side reading: "`om_factorization_killtest` disclaims itself, therefore it is not the paper's"

**MINOR_MISMATCH in my own earlier reasoning, recorded rather than quietly fixed.**
Its *DGP* is not the paper's, exactly as its prereg says. But the paper study
**imports its models and its trainer unchanged**. The statement should have been
"its DGP is not the paper DGP", not "it is not the paper's".

---

## Verdict

```
MAJOR_MISMATCH        none
MINOR_MISMATCH        4   (stage naming, occurrence wording split,
                           H1 under-claim, my own killtest phrasing)
MATCH                 6
SUPERSEDED            1
UNVERIFIABLE          none
```

No claim in `results/paper_synthesis/` is contradicted by the source. Four need
sharpening, and one of those sharpenings makes a claim **stronger** rather than
weaker. `PAPER_SYNTHESIS_SYNTHETIC_MISMATCH` is **not** triggered.
