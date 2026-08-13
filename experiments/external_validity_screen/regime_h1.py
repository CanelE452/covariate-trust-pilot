"""H1 re-analysed inside each SBC regime. Reads only; never trains.

Stage A tested H1 on the pooled 1200 M5 series and found a positive rank
correlation between |rho_interval| and delta.  The post-hoc diagnostic then
showed that signal is carried by the near-dense series and vanishes at high
ADI, which is the opposite of where the synthetic study lives.  This module
asks the same question one regime at a time, so "which regime carries H1" is
answered directly rather than inferred from an ADI split.

Nothing here re-derives a cutoff.  The SBC thresholds (ADI 1.32, CV2 0.49) and
the eligibility threshold come from the frozen record, and the delta
definitions, the Spearman helper and the bootstrap are imported from posthoc
rather than rewritten.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..unified_temporal_27_v3.training import train_scale
from . import posthoc, prereg, screen

OUT = screen.OUT / "regime_h1"
REGIMES = ("smooth", "erratic", "intermittent", "lumpy")
#: The two regimes the synthetic study actually modelled.
TARGET_REGIMES = ("intermittent", "lumpy")

#: Frozen in ``docs/m5_favorita_data_derivation.md`` section 4, used by
#: ``prepare_m5.py`` to draw the stratified sample. Not recomputed here.
SBC = {"adi_threshold": 1.32, "cv2_threshold": 0.49,
       "min_positive_train": 2, "min_train_length": 184}


def classify(adi: np.ndarray, cv2: np.ndarray) -> np.ndarray:
    """The four SBC quadrants, exactly as classify_sbc defines them."""
    out = np.full(len(adi), "excluded", dtype=object)
    ok = np.isfinite(adi) & np.isfinite(cv2)
    low_adi, low_cv2 = adi < SBC["adi_threshold"], cv2 < SBC["cv2_threshold"]
    out[ok & low_adi & low_cv2] = "smooth"
    out[ok & low_adi & ~low_cv2] = "erratic"
    out[ok & ~low_adi & low_cv2] = "intermittent"
    out[ok & ~low_adi & ~low_cv2] = "lumpy"
    return out


def regime_table(raw_csv, train_end: int) -> pd.DataFrame:
    """Train-only SBC labels for every M5 series in the raw file.

    prepare_m5.py was not migrated with the SCREEN, so the labels it wrote are
    not in the repository.  They are recomputed here from the same raw file and
    the same frozen rule; ``verify_against_design`` is what makes that a
    reproduction rather than a guess.
    """
    frame = pd.read_csv(raw_csv)
    day_cols = [c for c in frame.columns if c.startswith("d_")]
    train = frame[day_cols[:train_end]].to_numpy(np.float64)
    n_positive = (train > 0).sum(1)
    length = train.shape[1]

    adi = np.full(len(train), np.nan)
    cv2 = np.full(len(train), np.nan)
    eligible = (n_positive >= SBC["min_positive_train"]) & (length >= SBC["min_train_length"])
    for i in np.flatnonzero(eligible):
        positive = train[i][train[i] > 0]
        adi[i] = length / n_positive[i]
        cv2[i] = (positive.std(ddof=1) / positive.mean()) ** 2

    return pd.DataFrame({"series_id": frame["id"].to_numpy(),
                         "regime": classify(adi, cv2),
                         "ADI_selection": adi, "CV2_selection": cv2,
                         "n_positive_selection": n_positive})


def verify_against_design(labels: pd.DataFrame, stage_a_ids) -> dict:
    """The stratified draw took exactly 300 per regime; that is the check.

    If the recomputed labels put anything other than 300 of the Stage A series
    in each regime, the recomputation did not reproduce prepare_m5.py and every
    number downstream of it is unsafe.
    """
    selected = labels[labels["series_id"].isin(stage_a_ids)]
    counts = selected["regime"].value_counts().to_dict()
    per_regime_300 = all(counts.get(r) == 300 for r in REGIMES) and len(counts) == len(REGIMES)
    return {"stage_a_regime_counts": counts,
            "full_pool_regime_counts": labels["regime"].value_counts().to_dict(),
            "expected_per_regime": 300,
            "reproduced": bool(per_regime_300)}


def h1_within(pool: pd.DataFrame, scale: np.ndarray) -> dict:
    """posthoc.scale_robustness, plus the location and win-rate summaries."""
    rng = np.random.default_rng(posthoc.SEED)
    x = pool["rho_interval_abs_train"].to_numpy(float)
    raw = pool["delta_rmse"].to_numpy(float)
    point = pool["rmse_point"].to_numpy(float)

    degenerate = point <= posthoc.RELATIVE_FLOOR
    deltas = {"raw": raw,
              "relative": np.where(degenerate, np.nan,
                                   raw / np.where(degenerate, 1.0, point)),
              "scaled": raw / scale}

    out = {"n": int(len(pool)), "n_excluded_from_relative": int(degenerate.sum())}
    for label, d in deltas.items():
        rho = posthoc._spearman(x, d)
        finite = d[np.isfinite(d)]
        out[label] = {
            "spearman": rho,
            "ci": posthoc._boot_ci(lambda i, d=d: posthoc._spearman(x[i], d[i]),
                                   len(pool), rng),
            "n_finite": int(finite.size),
            "mean_delta": float(finite.mean()) if finite.size else float("nan"),
            "median_delta": float(np.median(finite)) if finite.size else float("nan"),
            # delta = RMSE_Point - RMSE_Hurdle, so a negative delta is a Point win.
            "point_win_rate": float((finite < 0).mean()) if finite.size else float("nan"),
            "hurdle_win_rate": float((finite > 0).mean()) if finite.size else float("nan"),
            "sign": "positive" if rho > 0 else ("negative" if rho < 0 else "zero"),
        }
    return out


def verdict(per_regime: dict) -> str:
    signs = [per_regime[r]["relative"]["spearman"] > 0 for r in TARGET_REGIMES
             if r in per_regime]
    if len(signs) < len(TARGET_REGIMES):
        return "H1_TARGET_REGIME_INCOMPUTABLE"
    if all(signs):
        return "H1_TARGET_REGIME_DIRECTION_SUPPORTED"
    if any(signs):
        return "H1_TARGET_REGIME_MIXED"
    return "H1_TARGET_REGIME_NOT_SUPPORTED"


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    threshold = prereg.ELIGIBILITY["primary_threshold"]
    series = pd.read_csv(screen.OUT / "per_series_metrics.csv")
    block = series[series["dataset"] == "m5"].copy()
    pool = block[(block["n_positive_train"] >= threshold)
                 & np.isfinite(block["delta_rmse"])].reset_index(drop=True)

    labels = regime_table(screen.REPO / "data" / "sales_train_evaluation.csv",
                          screen.config_for("m5").train_end)
    check = verify_against_design(labels, set(block["series_id"]))
    if not check["reproduced"]:
        raise screen.ScreenFailure(
            "REGIME_LABEL_NOT_REPRODUCED: the recomputed SBC labels do not put "
            f"300 Stage A series in each regime; got {check['stage_a_regime_counts']}")

    data = screen.load_dataset("m5")
    cfg = screen.config_for("m5")
    order = pd.Index(data["series_id"]).get_indexer(pool["series_id"])
    scale = train_scale({"y": data["y"], "z": data["z"]}, cfg)[order]

    pool = pool.merge(labels[["series_id", "regime"]], on="series_id", how="left")
    per_regime = {}
    for regime in REGIMES:
        take = (pool["regime"] == regime).to_numpy()
        if take.sum() < 3:
            continue
        per_regime[regime] = h1_within(pool[take].reset_index(drop=True), scale[take])

    report = {"analysis": "H1 within SBC regime, no retraining",
              "eligibility_threshold": threshold,
              "n_pool": int(len(pool)),
              "bootstrap": {"draws": posthoc.DRAWS, "seed": posthoc.SEED,
                            "unit": "series"},
              "sbc_rule": SBC,
              "label_reproduction": check,
              "regimes": per_regime,
              "target_regimes": list(TARGET_REGIMES),
              "verdict": verdict(per_regime)}
    (OUT / "regime_h1.json").write_text(json.dumps(report, indent=2, default=str))
    pool[["series_id", "regime", "rho_interval_abs_train", "delta_rmse",
          "rmse_point", "rmse_hurdle", "ADI_train", "CV2_positive_train",
          "n_positive_train"]].to_csv(OUT / "per_series_regime.csv", index=False)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"verdict": result["verdict"],
                      "n_pool": result["n_pool"],
                      "counts": {r: v["n"] for r, v in result["regimes"].items()}},
                     indent=2))
