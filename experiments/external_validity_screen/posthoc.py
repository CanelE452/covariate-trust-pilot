"""Post-hoc diagnostics for Stage A. Reads only; never trains, never predicts.

Stage A's primary results stay exactly as reported. Everything added here is
tagged POSTHOC_DIAGNOSTIC and none of it replaces H1, H2 or H3.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats

from ..unified_temporal_27_v3.training import train_scale
from . import prereg, screen

STAGE_A = screen.OUT
OUT = screen.OUT / "posthoc_diagnostic"
THRESHOLD = prereg.ELIGIBILITY["primary_threshold"]
DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]
#: A per-series RMSE below this is treated as degenerate rather than divided by.
RELATIVE_FLOOR = 1e-6
#: Synthetic sparsity contrast the SCREEN is trying to find support for.
SYNTHETIC_LOW = (3.0, 5.0)
SYNTHETIC_HIGH = 8.0


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boot_ci(fn, n: int, rng):
    draws = np.array([fn(rng.integers(0, n, size=n)) for _ in range(DRAWS)],
                     dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size < 100:
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))


def _spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


# ------------------------------------------------------------ step 1-2 ----


def scale_robustness(pool: pd.DataFrame, scale: np.ndarray) -> dict:
    """H1 under raw, relative and train-scale-normalised delta."""
    rng = np.random.default_rng(SEED)
    x = pool["rho_interval_abs_train"].to_numpy(float)
    raw = pool["delta_rmse"].to_numpy(float)
    point = pool["rmse_point"].to_numpy(float)

    degenerate = point <= RELATIVE_FLOOR
    relative = np.where(degenerate, np.nan, raw / np.where(degenerate, 1.0, point))
    scaled = raw / scale

    out = {"n": int(len(pool)),
           "n_excluded_from_relative": int(degenerate.sum()),
           "relative_exclusion_reason": (
               f"RMSE_Point <= {RELATIVE_FLOOR}; excluded rather than "
               "epsilon-padded")}
    for label, d in (("raw", raw), ("relative", relative), ("scaled", scaled)):
        rho = _spearman(x, d)
        out[label] = {
            "spearman": rho,
            "ci": _boot_ci(lambda i: _spearman(x[i], d[i]), len(pool), rng),
            "n_finite": int((np.isfinite(x) & np.isfinite(d)).sum()),
            "sign": "positive" if rho > 0 else ("negative" if rho < 0 else "zero"),
        }
    return out


def confound_table(pool: pd.DataFrame, scale: np.ndarray) -> dict:
    """[exploratory] Is H1 just a proxy for sparsity, dispersion or size?"""
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame({
        "abs_rho_interval": pool["rho_interval_abs_train"].to_numpy(float),
        "log_ADI": np.log(pool["ADI_train"].to_numpy(float)),
        "CV2_positive": pool["CV2_positive_train"].to_numpy(float),
        "log_scale": np.log(scale),
        "zero_ratio": pool["zero_ratio_train"].to_numpy(float),
    })
    delta = pool["delta_rmse"].to_numpy(float)
    pairwise = frame.corr(method="spearman").round(4)

    terms = ["abs_rho_interval", "log_ADI", "CV2_positive", "log_scale"]
    design = frame[terms].to_numpy(float)
    ok = np.isfinite(design).all(axis=1) & np.isfinite(delta)
    design, y = design[ok], delta[ok]
    z = (design - design.mean(0)) / design.std(0)

    def fit(idx, matrix, target):
        a = np.column_stack([np.ones(len(idx)), matrix[idx]])
        beta, *_ = np.linalg.lstsq(a, target[idx], rcond=None)
        return beta

    n = ok.sum()
    beta = fit(np.arange(n), design, y)
    beta_std = fit(np.arange(n), z, y)
    coefficients = {}
    for j, term in enumerate(terms, start=1):
        coefficients[term] = {
            "coefficient": float(beta[j]),
            "standardized": float(beta_std[j]),
            "ci": _boot_ci(lambda i, j=j: fit(i, design, y)[j], n, rng),
        }
    condition = float(np.linalg.cond(np.column_stack([np.ones(n), z])))
    return {"n": int(n), "pairwise_spearman": pairwise.to_dict(),
            "coefficients": coefficients, "design_condition_number": condition,
            "unstable": bool(condition > 30),
            "tag": "POSTHOC_DIAGNOSTIC / exploratory; not an H1 primary test"}


# -------------------------------------------------------------- step 3 ----


def adi_support(pool: pd.DataFrame) -> dict:
    adi = pool["ADI_train"].to_numpy(float)
    low = int(((adi >= SYNTHETIC_LOW[0]) & (adi <= SYNTHETIC_LOW[1])).sum())
    high = int((adi >= SYNTHETIC_HIGH).sum())
    verdict = ("H3_EXTERNAL_SUPPORT_AVAILABLE" if low >= 30 and high >= 30
               else "H3_EXTERNAL_NOT_IDENTIFIABLE")
    return {
        "quantiles": {f"p{q}": float(np.nanpercentile(adi, q))
                      for q in (0, 10, 25, 50, 75, 90, 95, 100)},
        "n_ge_4": int((adi >= 4).sum()), "n_ge_6": int((adi >= 6).sum()),
        "n_ge_8": int((adi >= 8).sum()),
        "LOW_SYNTHETIC_LIKE_3_to_5": low,
        "HIGH_SYNTHETIC_LIKE_ge_8": high,
        "verdict": verdict,
        "note": ("descriptive support check only; these groups are not a new "
                 "hypothesis and do not replace H3"),
    }


# -------------------------------------------------------------- step 4 ----


def gate_skill(name: str, pool: pd.DataFrame) -> dict:
    """Hurdle occurrence head against a train-prevalence constant predictor.

    Brier and the Brier skill score are exact from stored aggregates. ROC-AUC,
    PR-AUC and the Hurdle log loss need per-observation p_hat, which Stage A
    did not persist; regenerating it would require retraining, which is
    forbidden here.
    """
    data = screen.load_dataset(name)
    cfg = screen.config_for(name)
    order = pd.Index(data["series_id"]).get_indexer(pool["series_id"])
    if (order < 0).any():
        raise RuntimeError("series_id mismatch between Stage A and the source")
    z = data["z"][order]

    stride = prereg.SPLITS[name]["test_origin_stride"]
    origins = np.arange(cfg.val_end, cfg.length - cfg.horizon + 1, stride)
    test_idx = np.concatenate([np.arange(o, o + cfg.horizon) for o in origins])

    #: train-only, never the test prevalence
    p_const = z[:, :cfg.train_end].mean(axis=1)
    rate = z[:, test_idx].mean(axis=1)

    brier_const = (1 - rate) * p_const ** 2 + rate * (1 - p_const) ** 2
    brier_hurdle = pool["hurdle_occurrence_brier"].to_numpy(float)
    clipped = np.clip(p_const, 1e-7, 1 - 1e-7)
    logloss_const = -(rate * np.log(clipped) + (1 - rate) * np.log(1 - clipped))

    bss_pooled = 1.0 - brier_hurdle.mean() / brier_const.mean()
    rng = np.random.default_rng(SEED)
    ci = _boot_ci(lambda i: 1.0 - brier_hurdle[i].mean() / brier_const[i].mean(),
                  len(pool), rng)
    per_series_bss = 1.0 - brier_hurdle / np.maximum(brier_const, 1e-12)

    if bss_pooled > 0.01:
        verdict = "OCCURRENCE_GATE_HAS_SKILL"
    elif bss_pooled < -0.01:
        verdict = "OCCURRENCE_GATE_WORSE_THAN_CONSTANT"
    else:
        verdict = "OCCURRENCE_GATE_WEAK_SKILL"
    return {
        "p_const_definition": "mean(z) over the train split, per series",
        "mean_p_const_train": float(p_const.mean()),
        "mean_test_positive_rate": float(rate.mean()),
        "brier_constant": float(brier_const.mean()),
        "brier_hurdle": float(brier_hurdle.mean()),
        "brier_skill_score_pooled": float(bss_pooled),
        "brier_skill_score_ci": ci,
        "median_per_series_bss": float(np.median(per_series_bss)),
        "pct_series_bss_positive": float(np.mean(per_series_bss > 0) * 100),
        "logloss_constant": float(logloss_const.mean()),
        "logloss_hurdle": None,
        "roc_auc": None,
        "pr_auc": None,
        "unavailable_reason": (
            "Stage A stored per-series aggregates only; ROC-AUC, PR-AUC and the "
            "Hurdle log loss need per-observation p_hat, and regenerating it "
            "would require retraining, which this task forbids"),
        "mean_p_hat_on_zero": float(pool["hurdle_mean_p_hat_on_zero"].mean()),
        "mean_p_hat_on_positive": float(pool["hurdle_mean_p_hat_on_positive"].mean()),
        "discrimination_gap": float(pool["hurdle_mean_p_hat_on_positive"].mean()
                                    - pool["hurdle_mean_p_hat_on_zero"].mean()),
        "verdict": verdict,
        "aggregation": ("Brier averaged over series after averaging within a "
                        "series; the pooled BSS uses the ratio of those means"),
    }


# -------------------------------------------------------------- step 5 ----


def frozen_cutoffs(pool: pd.DataFrame) -> dict:
    """The numeric cutoffs Stage A actually used, recovered deterministically.

    pre_analysis_spec stores the rule, not the value; analyse() computed the
    quantiles inside the 1200-series pool and never persisted them. Applying
    the identical expressions to the identical pool reproduces them exactly.
    """
    x = pool["rho_interval_abs_train"].to_numpy(float)
    return {"HIGH_ADI_min": float(np.nanmedian(pool["ADI_train"].to_numpy(float))),
            "LOW_OCC_max": float(np.nanquantile(x, 1 / 3)),
            "MAG_PERSISTENT_min": float(np.nanquantile(
                pool["rho_magnitude_train"].to_numpy(float), 2 / 3)),
            "recovered_from": "reports/external_validity_screen/per_series_metrics.csv",
            "recomputed_on_full_pool": False}


def full_pool_expansion(name: str, cutoffs: dict) -> dict:
    """How many candidates the untouched full pool would yield. No test metric."""
    cfg = screen.config_for(name)
    if name == "m5":
        wide = pd.read_csv(screen.REPO / "data" / "sales_train_evaluation.csv")
        day_cols = [c for c in wide.columns if c.startswith("d_")]
        y = wide[day_cols].to_numpy(dtype=np.float32)
        meta = wide[["item_id", "store_id"]]
        starts = screen.m5_availability(meta)
        ids = wide["id"].to_numpy().astype(str)
    else:
        frame = pd.read_parquet(screen.REPO / "data" / "processed"
                                / "favorita_series.parquet")
        pivot = frame.pivot_table(index="series_id", columns="timestamp",
                                  values="target").sort_index()
        y = pivot.to_numpy(dtype=np.float32)
        starts = np.zeros(len(pivot), dtype=int)
        ids = pivot.index.to_numpy().astype(str)

    rows = [screen.describe_series(y[i], int(starts[i]), cfg.train_end)
            for i in range(len(ids))]
    table = pd.DataFrame(rows)
    table["series_id"] = ids
    eligible = table[table["n_positive_train"] >= THRESHOLD]

    adi = eligible["ADI_train"].to_numpy(float)
    occ = eligible["rho_interval_abs_train"].to_numpy(float)
    mag = eligible["rho_magnitude_train"].to_numpy(float)
    high = adi >= cutoffs["HIGH_ADI_min"]
    low_occ = occ <= cutoffs["LOW_OCC_max"]
    persistent = mag >= cutoffs["MAG_PERSISTENT_min"]
    candidate = high & low_occ & persistent
    control = high & low_occ & ~persistent

    n_cand = int(candidate.sum())
    verdict = ("H2_EXPANSION_HIGH_VALUE" if n_cand >= 200 and control.sum() >= 200
               else "H2_EXPANSION_POSSIBLE" if n_cand >= 50
               else "H2_EXPANSION_LOW_YIELD")
    describe = lambda sel, col: (  # noqa: E731
        {"median": float(np.nanmedian(eligible[col].to_numpy(float)[sel])),
         "p25": float(np.nanpercentile(eligible[col].to_numpy(float)[sel], 25)),
         "p75": float(np.nanpercentile(eligible[col].to_numpy(float)[sel], 75))}
        if sel.any() else None)
    return {
        "total_series_in_source": int(len(ids)),
        "descriptor_eligible": int(len(eligible)),
        "point_candidate": n_cand,
        "control": int(control.sum()),
        "candidate_descriptors": {c: describe(candidate, c) for c in
                                  ("ADI_train", "rho_interval_abs_train",
                                   "rho_magnitude_train")},
        "control_descriptors": {c: describe(control, c) for c in
                                ("ADI_train", "rho_interval_abs_train",
                                 "rho_magnitude_train")},
        "verdict": verdict,
        "note": ("operational thresholds for sizing the next run, not a "
                 "scientific cutoff; no test metric was computed here"),
    }


# --------------------------------------------------------------- driver ----


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(STAGE_A / "per_series_metrics.csv")
    integrity = {f.name: {"sha256": _sha(f), "bytes": f.stat().st_size}
                 for f in sorted(STAGE_A.iterdir()) if f.suffix in (".json", ".csv")}

    report = {"stage_a_integrity_before": integrity, "threshold": THRESHOLD,
              "datasets": {}}
    for name in ("m5", "favorita"):
        block = series[series["dataset"] == name].copy()
        pool = block[(block["n_positive_train"] >= THRESHOLD)
                     & np.isfinite(block["delta_rmse"])].reset_index(drop=True)
        data = screen.load_dataset(name)
        cfg = screen.config_for(name)
        order = pd.Index(data["series_id"]).get_indexer(pool["series_id"])
        scale = train_scale({"y": data["y"], "z": data["z"]}, cfg)[order]

        cutoffs = frozen_cutoffs(pool)
        report["datasets"][name] = {
            "n_pool": int(len(pool)),
            "h1_scale_robustness": scale_robustness(pool, scale),
            "h1_confound": confound_table(pool, scale),
            "h3_adi_support": adi_support(pool),
            "occurrence_gate_skill": gate_skill(name, pool),
            "frozen_cutoffs_recovered": cutoffs,
            "h2_full_pool": full_pool_expansion(name, cutoffs),
        }
        print(f"[{name}] done")

    report["stage_a_integrity_after"] = {
        f.name: {"sha256": _sha(f), "bytes": f.stat().st_size}
        for f in sorted(STAGE_A.iterdir()) if f.suffix in (".json", ".csv")}
    report["stage_a_unmodified"] = (
        report["stage_a_integrity_before"] == report["stage_a_integrity_after"])
    (OUT / "posthoc_diagnostic.json").write_text(json.dumps(report, indent=2,
                                                            default=str))
    print(f"wrote {OUT.relative_to(screen.REPO)}/posthoc_diagnostic.json  "
          f"stage_a_unmodified={report['stage_a_unmodified']}")


if __name__ == "__main__":
    main()
