"""Track X verdict. Reads only Track X artifacts plus the frozen preregistration.

Never opens a Track G, Track F or Track V result.
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

ALLOWED_INPUTS = ("preregistration.json", "runtime_tier.json", "clean_baselines.json")


def _gates() -> dict:
    return json.loads((paths.RESULTS / "preregistration.json").read_text())["gates"]


def offdiag_by_origin(recs, family, severity, C) -> dict:
    """Off-diagonal spillover values grouped by evaluation origin."""
    out = {}
    for r in recs:
        if r["family"] != family or r["severity"] != severity:
            continue
        rel = r["rel"]
        j = r["j"]
        out.setdefault(r["origin"], []).extend(rel[i] for i in range(C) if i != j)
    return out


def phenomenon_verdict(phen: dict, C_by_ds: dict) -> dict:
    g = _gates()["X_phenomenon"]
    per_ds = {}
    for dsname, entry in phen["datasets"].items():
        C = C_by_ds[dsname]
        fams = {}
        for key, s in entry["summary"]["TQNet"].items():
            byo = offdiag_by_origin(entry["raw"]["TQNet"], s["family"], s["severity"], C)
            bs = B.cluster_bootstrap(byo, stat=np.median, block=2)
            ratio = entry["tqnet_over_dlinear_spillover"][key]
            checks = {
                "median_offdiag_ge_3pct": bool(s["median_offdiag_spillover"] >= g["median_offdiag_spillover_ge"]),
                "bootstrap_lower_gt_0": bool(bs["ci_lower"] > g["bootstrap_ci_lower_gt"]),
                "tqnet_over_dlinear_ge_3x": bool(ratio >= g["tqnet_over_dlinear_spillover_ge"]),
                "no_dominant_source_channel": bool(
                    s["max_single_source_channel_share"] < g["no_single_source_channel_share_ge"]),
            }
            fams[key] = {**s, "bootstrap": bs, "tqnet_over_dlinear": ratio,
                         "checks": checks, "passes": all(checks.values())}
        # A family counts once if any severity of that family passes.
        passing_families = sorted({v["family"] for v in fams.values() if v["passes"]})
        per_ds[dsname] = {"families": fams, "passing_families": passing_families,
                          "n_passing_families": len(passing_families),
                          "passes": len(passing_families) >= g["min_corruption_families"]}
    ok = all(v["passes"] for v in per_ds.values())
    return {"per_dataset": per_ds, "gate": g,
            "verdict": "X_PHENOMENON_GO" if ok else "X_NO_SPILLOVER"}


def method_verdict(meth: dict, phenomenon_passed: bool, shared_weak: dict) -> dict:
    """Simple-baseline and quarantine gates, from the mitigation artifact."""
    gs = _gates()["X_simple_solves"]
    gm = _gates()["X_method_go"]
    per_ds = {}
    for dsname, e in meth["datasets"].items():
        base = e["baseline_spillover"]
        simple = {k: e["arms"][k] for k in ("clipping", "channel_dropout") if k in e["arms"]}
        red = {k: (base - v["median_offdiag_spillover"]) / (abs(base) + 1e-12)
               for k, v in simple.items()}
        simple_ok = {k: bool(red[k] >= gs["spillover_reduction_ge"]
                             and v["clean_mse_degradation"] < gs["clean_mse_degradation_lt"])
                     for k, v in simple.items()}
        best_simple = max(simple, key=lambda k: red[k]) if simple else None
        q = e["arms"].get("quarantine")
        extra = None
        if q is not None and best_simple is not None:
            bs = simple[best_simple]["median_offdiag_spillover"]
            extra = (bs - q["median_offdiag_spillover"]) / (abs(bs) + 1e-12)
        per_ds[dsname] = {
            "baseline_spillover": base,
            "simple_reduction": red, "simple_solves": simple_ok,
            "best_simple": best_simple,
            "quarantine": q,
            "quarantine_extra_reduction_vs_best_simple": extra,
            "quarantine_bootstrap": e.get("quarantine_bootstrap"),
            "shared_baseline_weak": bool(shared_weak.get(dsname, False)),
        }
    simple_solves = (all(any(v["simple_solves"].values()) for v in per_ds.values())
                     and len({tuple(sorted(k for k, ok in v["simple_solves"].items() if ok))
                              for v in per_ds.values()}) >= 1)
    m_checks = {}
    for dsname, v in per_ds.items():
        q = v["quarantine"]
        bsr = v.get("quarantine_bootstrap") or {}
        m_checks[dsname] = {
            "extra_reduction_ge_20pct": bool(v["quarantine_extra_reduction_vs_best_simple"] is not None
                                             and v["quarantine_extra_reduction_vs_best_simple"]
                                             >= gm["extra_spillover_reduction_vs_best_simple_ge"]),
            "clean_mse_degradation_lt_1pct": bool(q is not None
                                                  and q["clean_mse_degradation"] < gm["clean_mse_degradation_lt"]),
            "direct_damage_degradation_lt_5pct": bool(q is not None
                                                      and q["direct_damage_degradation"] < gm["direct_damage_degradation_lt"]),
            "bootstrap_lower_gt_0": bool(bsr.get("ci_lower", -1) > gm["bootstrap_lower_gt"]),
            "shared_baseline_not_weak": not v["shared_baseline_weak"],
        }
    method_go = all(all(c.values()) for c in m_checks.values())
    if not phenomenon_passed:
        verdict = "X_SIMPLE_BASELINE_SOLVES" if simple_solves else "NO_PHENOMENON"
        role = "DIAGNOSTIC_CONTINUATION_AFTER_X_PHENOMENON_FAIL"
    elif method_go:
        verdict, role = "X_CORRUPTION_QUARANTINE_METHOD_GO", "CONFIRMATORY_SCREEN"
    elif simple_solves:
        verdict, role = "X_SIMPLE_BASELINE_SOLVES", "CONFIRMATORY_SCREEN"
    else:
        verdict, role = "X_SPILLOVER_CHARACTERIZATION", "CONFIRMATORY_SCREEN"
    return {"per_dataset": per_ds, "method_checks": m_checks,
            "simple_baseline_solves": simple_solves, "method_go": method_go,
            "verdict": verdict, "result_role": role}


def track_verdict(verdict_method: str) -> str:
    return {
        "X_CORRUPTION_QUARANTINE_METHOD_GO": "METHOD_GO",
        "X_SIMPLE_BASELINE_SOLVES": "SIMPLE_BASELINE_SOLVES",
        "X_SPILLOVER_CHARACTERIZATION": "CHARACTERIZATION_ONLY",
        "NO_PHENOMENON": "NO_PHENOMENON",
    }.get(verdict_method, "NOT_EVALUATED")
