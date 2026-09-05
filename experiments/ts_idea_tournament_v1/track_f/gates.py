"""Track F verdict. Reads only Track F artifacts plus the frozen preregistration.

Never opens a Track X, Track G or Track V result.
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

EXISTING_FILTERS = ["high_loss_removal", "rho_loss", "adarho"]
RANDOM_BASELINE_REMOVAL = 0.20


def _gates() -> dict:
    return json.loads((paths.RESULTS / "preregistration.json").read_text())["gates"]


def mean_over_seeds(retrain: dict, method: str, field: str) -> float:
    v = [r[field] for r in retrain.values() if r["method"] == method]
    return float(np.mean(v)) if v else float("nan")


def phenomenon_verdict(sel: dict) -> dict:
    g = _gates()["F_phenomenon"]
    per_ds = {}
    for dsname, e in sel["datasets"].items():
        diag = e["selection_diagnostic"]
        rt = e["retraining"]
        nof = mean_over_seeds(rt, "no_filter", "shifted_test_mse")
        hits = {}
        for m in EXISTING_FILTERS:
            d = diag[m]
            shifted = mean_over_seeds(rt, m, "shifted_test_mse")
            hits[m] = {
                "corruption_removal": d["corruption_removal_rate"],
                "legitimate_shift_removal": d["legitimate_shift_removal_rate"],
                "shift_retention": d["shift_retention"],
                "clean_removal": d["clean_removal_rate"],
                "shifted_test_mse": shifted,
                "shifted_test_degradation_vs_no_filter": (shifted - nof) / nof,
                "checks": {
                    "corruption_removal_ge_50pct": bool(
                        d["corruption_removal_rate"] >= g["corruption_removal_ge"]),
                    "legit_shift_removal_ge_35pct": bool(
                        d["legitimate_shift_removal_rate"] >= g["legit_shift_removal_ge"]),
                    "legit_shift_removal_over_random_ge_15pp": bool(
                        (d["legitimate_shift_removal_rate"] - RANDOM_BASELINE_REMOVAL) * 100
                        >= g["legit_shift_removal_minus_random_ge_pp"]),
                },
            }
            hits[m]["passes"] = all(hits[m]["checks"].values())
        any_pass = any(v["passes"] for v in hits.values())
        strong = any(v["passes"] and v["shifted_test_degradation_vs_no_filter"] >= 0.03
                     for v in hits.values())
        per_ds[dsname] = {"filters": hits, "passes": any_pass,
                          "strong_evidence_shifted_test_degradation": strong,
                          "no_filter_shifted_test_mse": nof}
    ok = all(v["passes"] for v in per_ds.values())
    return {"per_dataset": per_ds, "gate": g,
            "verdict": "F_SELECTION_CONFOUNDING_PRESENT" if ok else "F_NO_SELECTION_CONFOUNDING"}


def method_verdict(sel: dict, per_window: dict, phenomenon_passed: bool) -> dict:
    g = _gates()["F_method_go"]
    per_ds = {}
    for dsname, e in sel["datasets"].items():
        diag = e["selection_diagnostic"]
        rt = e["retraining"]
        # Best existing filter: the one retaining most legitimate shift among
        # those meeting the 50% corruption-removal bar; otherwise the best
        # corruption remover. Chosen from selection diagnostics, not test MSE.
        eligible = [m for m in EXISTING_FILTERS if diag[m]["corruption_removal_rate"] >= 0.50]
        pool = eligible or EXISTING_FILTERS
        best = max(pool, key=lambda m: diag[m]["shift_retention"])
        co = diag["coherence_aware"]
        bd = diag[best]
        clean_new = mean_over_seeds(rt, "coherence_aware", "clean_test_mse")
        clean_old = mean_over_seeds(rt, best, "clean_test_mse")
        shift_new = mean_over_seeds(rt, "coherence_aware", "shifted_test_mse")
        shift_old = mean_over_seeds(rt, best, "shifted_test_mse")
        bs = B.paired_difference_bootstrap(
            per_window[dsname]["coherence_aware"], per_window[dsname][best],
            relative=True, block=2)
        per_ds[dsname] = {
            "best_existing_filter": best,
            "shift_retention_gain_pp": (co["shift_retention"] - bd["shift_retention"]) * 100,
            "corruption_removal_drop_pp": (bd["corruption_removal_rate"]
                                           - co["corruption_removal_rate"]) * 100,
            "shifted_test_gain": (shift_old - shift_new) / shift_old,
            "clean_test_degradation": (clean_new - clean_old) / clean_old,
            "coherence_aware": co, "best_existing": bd,
            "clean_test_mse": {"coherence_aware": clean_new, best: clean_old},
            "shifted_test_mse": {"coherence_aware": shift_new, best: shift_old},
            "bootstrap_shifted_gain": bs,
            "controls": e["controls"],
            "removal_budget_equal": len(set(e["removal_budget"].values())) == 1
            or max(e["removal_budget"].values()) - min(
                v for k, v in e["removal_budget"].items() if k != "no_filter") <= 1,
        }
    checks = {
        "shift_retention_gain_ge_15pp": bool(
            min(v["shift_retention_gain_pp"] for v in per_ds.values()) >= g["shift_retention_gain_pp_ge"]),
        "corruption_removal_drop_le_5pp": bool(
            max(v["corruption_removal_drop_pp"] for v in per_ds.values()) <= g["corruption_removal_drop_le_pp"]),
        "shifted_test_gain_ge_2pct": bool(
            min(v["shifted_test_gain"] for v in per_ds.values()) >= g["shifted_test_mse_gain_ge"]),
        "clean_test_degradation_lt_1pct": bool(
            max(v["clean_test_degradation"] for v in per_ds.values()) < g["clean_test_mse_degradation_lt"]),
        "same_direction_both_datasets": bool(
            all(v["shifted_test_gain"] > 0 for v in per_ds.values())),
        "bootstrap_lower_gt_0": bool(
            all(v["bootstrap_shifted_gain"]["ci_lower"] > 0 for v in per_ds.values())),
    }
    method_go = all(checks.values())
    adarho_solves = all(
        e["selection_diagnostic"]["adarho"]["shift_retention"]
        >= e["selection_diagnostic"]["coherence_aware"]["shift_retention"] - 0.05
        and e["selection_diagnostic"]["adarho"]["corruption_removal_rate"]
        >= e["selection_diagnostic"]["rho_loss"]["corruption_removal_rate"]
        for e in sel["datasets"].values())
    if not phenomenon_passed:
        verdict = "F_NO_SELECTION_CONFOUNDING"
        role = "DIAGNOSTIC_CONTINUATION_AFTER_F_PHENOMENON_FAIL"
    elif method_go:
        verdict, role = "F_SELECTION_CONFOUNDING_METHOD_GO", "CONFIRMATORY_SCREEN"
    elif adarho_solves:
        verdict, role = "F_EXISTING_ADAPTIVE_FILTER_SOLVES", "CONFIRMATORY_SCREEN"
    else:
        verdict, role = "F_SELECTION_CHARACTERIZATION", "CONFIRMATORY_SCREEN"
    return {"per_dataset": per_ds, "checks": checks, "method_go": method_go,
            "existing_adaptive_filter_solves": adarho_solves,
            "verdict": verdict, "result_role": role}


def track_verdict(v: str) -> str:
    return {
        "F_SELECTION_CONFOUNDING_METHOD_GO": "METHOD_GO",
        "F_EXISTING_ADAPTIVE_FILTER_SOLVES": "SIMPLE_BASELINE_SOLVES",
        "F_SELECTION_CHARACTERIZATION": "CHARACTERIZATION_ONLY",
        "F_NO_SELECTION_CONFOUNDING": "NO_PHENOMENON",
    }.get(v, "NOT_EVALUATED")
