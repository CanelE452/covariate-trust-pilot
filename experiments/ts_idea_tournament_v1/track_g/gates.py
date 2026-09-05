"""Track G verdict. Reads only Track G artifacts plus the frozen preregistration.

Never opens a Track X, Track F or Track V result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np

import paths
import bootstrap as B


def _gates() -> dict:
    return json.loads((paths.RESULTS / "preregistration.json").read_text())["gates"]


def phenomenon_verdict(diag: dict) -> dict:
    """Gate on the best-validation checkpoints, aggregating the two seeds."""
    g = _gates()["G_phenomenon"]
    per_ds = {}
    for dsname in sorted({v["dataset"] for v in diag["checkpoints"].values()}):
        entries = {k: v for k, v in diag["checkpoints"].items()
                   if v["dataset"] == dsname and v["checkpoint_tag"] == "best"}
        # Unbiased harm rate: the randomly drawn half of the exact-checked pairs.
        rates, by_block, gains, fps, fns, seeds_dir = [], {}, [], [], [], []
        unbiased = []
        for k, v in entries.items():
            rates.append(v["exact_harm_rate"])
            unbiased.append(v["exact_harm_rate_random_subset"])
            gains.append(v["auprc_gain_cross_probe_over_cosine"])
            fps.append(v["same_batch_cosine_detector"]["false_positive_rate"])
            fns.append(v["same_batch_cosine_detector"]["false_negative_rate"])
            seeds_dir.append(v["auprc_cross_probe_affinity"] > v["auprc_same_batch_cosine"])
            for r in v["rows"]:
                by_block.setdefault((v["seed"], r["pair"]), []).append(float(r["harm_label"]))
        bs = B.cluster_bootstrap(by_block, stat=np.mean, block=2)
        checks = {
            "exact_harm_rate_ge_10pct": bool(np.mean(rates) >= g["exact_harm_rate_ge"]),
            "bootstrap_lower_ge_5pct": bool(bs["ci_lower"] >= g["bootstrap_lower_ge"]),
            "cosine_detector_fp_or_fn_ge_20pct": bool(
                max(np.mean(fps), np.mean(fns)) >= g["cosine_detector_fp_or_fn_ge"]),
            "cross_probe_auprc_gain_ge_0p10": bool(np.mean(gains) >= g["cross_probe_auprc_gain_ge"]),
            "seed_direction_stable": bool(all(seeds_dir)),
        }
        per_ds[dsname] = {
            "exact_harm_rate_mean": float(np.mean(rates)),
            "exact_harm_rate_unbiased_random_subset": float(np.mean(unbiased)),
            "bootstrap_harm_rate": bs,
            "cosine_detector_fp_mean": float(np.mean(fps)),
            "cosine_detector_fn_mean": float(np.mean(fns)),
            "auprc_same_batch_cosine_mean": float(np.mean(
                [v["auprc_same_batch_cosine"] for v in entries.values()])),
            "auprc_cross_probe_mean": float(np.mean(
                [v["auprc_cross_probe_affinity"] for v in entries.values()])),
            "auprc_gain_mean": float(np.mean(gains)),
            "control_random_task_pairing_auprc_mean": float(np.mean(
                [v["control_random_task_pairing_auprc"] for v in entries.values()])),
            "control_gradient_sign_randomised_auprc_mean": float(np.mean(
                [v["control_gradient_sign_randomised_auprc"] for v in entries.values()])),
            "checks": checks, "passes": all(checks.values()),
        }
    ok = all(v["passes"] for v in per_ds.values())
    return {"per_dataset": per_ds, "gate": g,
            "verdict": "G_PHENOMENON_GO" if ok else "G_NO_HARMFUL_INTERFERENCE_SIGNAL"}


def arm_macro(inter: dict, dsname: str, arm: str) -> float:
    v = [r["macro_mse"] for r in inter["runs"].values()
         if r["dataset"] == dsname and r["arm"] == arm]
    return float(np.mean(v)) if v else float("nan")


def method_verdict(inter: dict, per_origin: dict, phenomenon_passed: bool) -> dict:
    """Compare probe-gated PCGrad with ERM, PCGrad and the norm-balanced control."""
    g = _gates()["G_method_go"]
    per_ds = {}
    for dsname in sorted({r["dataset"] for r in inter["runs"].values()}):
        m = {a: arm_macro(inter, dsname, a)
             for a in ("erm", "pcgrad", "norm_balanced", "probe_gated")}
        gain_erm = (m["erm"] - m["probe_gated"]) / m["erm"]
        gain_pcg = (m["pcgrad"] - m["probe_gated"]) / m["pcgrad"]
        gain_nb = (m["norm_balanced"] - m["probe_gated"]) / m["norm_balanced"]
        bs = B.paired_difference_bootstrap(
            per_origin[dsname]["probe_gated"], per_origin[dsname]["erm"],
            relative=True, block=2)
        # Random control: the same comparison against a task-order-shuffled arm
        # is unavailable, so the norm-balanced control stands in as the "is it
        # just gradient scaling?" test, as preregistered in G9.
        per_ds[dsname] = {
            "macro_mse": m,
            "gain_vs_erm": gain_erm, "gain_vs_pcgrad": gain_pcg,
            "gain_vs_norm_balanced": gain_nb,
            "bootstrap_vs_erm": bs,
            "extra_probe_cost": {
                a: float(np.mean([r["extra_probe_forward"] for r in inter["runs"].values()
                                  if r["dataset"] == dsname and r["arm"] == a]))
                for a in ("erm", "pcgrad", "norm_balanced", "probe_gated")},
            "wall_s": {a: float(np.mean([r["wall_s"] for r in inter["runs"].values()
                                         if r["dataset"] == dsname and r["arm"] == a]))
                       for a in ("erm", "pcgrad", "norm_balanced", "probe_gated")},
        }
    gains_erm = [v["gain_vs_erm"] for v in per_ds.values()]
    gains_pcg = [v["gain_vs_pcgrad"] for v in per_ds.values()]
    checks = {
        "vs_erm_ge_0p7pct": bool(min(gains_erm) >= g["vs_erm_macro_mse_gain_ge"]),
        "vs_pcgrad_ge_0p3pct": bool(min(gains_pcg) >= g["vs_pcgrad_gain_ge"]),
        "both_datasets_positive": bool(all(v > 0 for v in gains_erm)),
        "no_dataset_worse_than_minus_0p5pct": bool(min(gains_erm) > g["no_dataset_worse_than"]),
        "bootstrap_lower_gt_0": bool(all(v["bootstrap_vs_erm"]["ci_lower"] > g["bootstrap_lower_gt"]
                                         for v in per_ds.values())),
        "beats_norm_balanced_control": bool(
            all(v["gain_vs_norm_balanced"] > 0 for v in per_ds.values())),
    }
    method_go = all(checks.values())
    existing_solves = any(
        (v["macro_mse"]["pcgrad"] < v["macro_mse"]["erm"] * (1 - g["vs_erm_macro_mse_gain_ge"])
         or v["macro_mse"]["norm_balanced"] < v["macro_mse"]["erm"] * (1 - g["vs_erm_macro_mse_gain_ge"]))
        for v in per_ds.values()) and all(
        (v["macro_mse"]["pcgrad"] < v["macro_mse"]["erm"]
         or v["macro_mse"]["norm_balanced"] < v["macro_mse"]["erm"])
        for v in per_ds.values())
    if not phenomenon_passed:
        verdict = "G_NO_HARMFUL_INTERFERENCE_SIGNAL"
        role = "DIAGNOSTIC_CONTINUATION_AFTER_G_PHENOMENON_FAIL"
    elif method_go:
        verdict, role = "G_HARMFUL_INTERFERENCE_METHOD_GO", "CONFIRMATORY_SCREEN"
    elif existing_solves:
        verdict, role = "G_EXISTING_BASELINE_SOLVES", "CONFIRMATORY_SCREEN"
    else:
        verdict, role = "G_CHARACTERIZATION_ONLY", "CONFIRMATORY_SCREEN"
    return {"per_dataset": per_ds, "checks": checks, "method_go": method_go,
            "existing_baseline_solves": existing_solves,
            "verdict": verdict, "result_role": role}


def track_verdict(v: str) -> str:
    return {
        "G_HARMFUL_INTERFERENCE_METHOD_GO": "METHOD_GO",
        "G_EXISTING_BASELINE_SOLVES": "SIMPLE_BASELINE_SOLVES",
        "G_CHARACTERIZATION_ONLY": "CHARACTERIZATION_ONLY",
        "G_NO_HARMFUL_INTERFERENCE_SIGNAL": "NO_PHENOMENON",
    }.get(v, "NOT_EVALUATED")
