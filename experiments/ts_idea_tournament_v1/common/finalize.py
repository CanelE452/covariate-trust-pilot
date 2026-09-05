"""Stages 28-31: bootstrap, per-track verdicts, topic ranking, STATUS and tables.

Each track verdict is produced by that track's own gates module reading only
that track's artifacts. Track V is computed but never feeds a G/X/F verdict or
the final ranking.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import paths
import attempts as A
import evalgeom
from contract import DATASETS, CLEAN_SEEDS, PRED_LEN, tqnet_config, dlinear_config
import data as D
import engine as E

sys.path.insert(0, str(paths.EXP / "track_x"))
sys.path.insert(0, str(paths.EXP / "track_g"))
sys.path.insert(0, str(paths.EXP / "track_f"))


def load_gates(track: str):
    """Load one track's gates module by explicit path.

    The three modules share the file name gates.py, so a plain import would
    resolve to whichever directory happens to come first on sys.path.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"{track}_gates", paths.EXP / track / "gates.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

R = paths.RESULTS


def latest(stage: str, fname: str):
    fs = sorted((paths.RUNS / stage).glob(f"attempt_*/{fname}"))
    fs = [f for f in fs if (f.parent / "completion.json").exists()]
    return json.loads(fs[-1].read_text()) if fs else None


def latest_dir(stage: str):
    ds = sorted(p for p in (paths.RUNS / stage).glob("attempt_*")
                if (p / "completion.json").exists())
    return ds[-1] if ds else None


# --------------------------------------------------------------------------- #
# Per-origin evaluation needed by the Track G method bootstrap
# --------------------------------------------------------------------------- #

def g_per_origin(inter: dict) -> dict:
    out = {}
    for dsname in DATASETS:
        runs = [r for r in inter["runs"].values() if r["dataset"] == dsname]
        if not runs:
            continue
        tasks = runs[0]["tasks"]
        per_arm = {}
        for r in runs:
            cfg = tqnet_config(dsname, r["seed"])
            m = E.build_model(cfg).to(E.DEVICE)
            m.load_state_dict(torch.load(paths.ROOT / r["checkpoint_file"], map_location=E.DEVICE))
            m.eval()
            for o in evalgeom.origins(dsname, "test"):
                x, y, c = D.window_batch(dsname, "test", o["window_starts"], E.DEVICE)
                with torch.no_grad():
                    p = E.forward(m, cfg, x, c)[:, -PRED_LEN:, :]
                    d = ((p - y[:, -PRED_LEN:, :]) ** 2).mean(dim=(0, 1)).cpu().numpy()
                per_arm.setdefault(r["arm"], {}).setdefault(o["origin"], []).append(
                    float(np.mean([d[i] for i in tasks])))
            del m
            torch.cuda.empty_cache()
        out[dsname] = per_arm
    return out


def f_per_window(sel: dict, att_dir: Path) -> dict:
    """Per-window shifted-test losses, clustered into origin-like blocks."""
    out = {}
    for dsname in sel["datasets"]:
        per_m = {}
        for f in att_dir.glob(f"windowloss_{dsname}_*_s{CLEAN_SEEDS[0]}.npz"):
            method = f.stem[len(f"windowloss_{dsname}_"):-len(f"_s{CLEAN_SEEDS[0]}")]
            z = np.load(f)
            sh = z["shifted"]
            n_blocks = evalgeom.N_ORIGINS
            edges = np.linspace(0, len(sh), n_blocks + 1).astype(int)
            per_m[method] = {b: sh[edges[b]:edges[b + 1]].tolist() for b in range(n_blocks)}
        out[dsname] = per_m
    return out


# --------------------------------------------------------------------------- #

def build_track_x():
    phen = latest("track_x_phenomenon", "phenomenon_raw.json")
    meth = latest("track_x_mitigations", "mitigations.json")
    clean = json.loads((R / "clean_baselines.json").read_text())
    weak = {d: g["shared_baseline_weak"] for d, g in clean["gate"].items()}
    if phen is None:
        return {"status": "BLOCKED", "verdict": "NOT_EVALUATED",
                "reason": "Track X phenomenon artifact missing"}
    XG = load_gates("track_x")
    C_by_ds = {d: DATASETS[d].enc_in for d in DATASETS}
    p = XG.phenomenon_verdict(phen, C_by_ds)
    if meth is None:
        return {"status": "PARTIAL", "phenomenon": p, "verdict": "NOT_EVALUATED",
                "reason": "Track X mitigation artifact missing"}
    m = XG.method_verdict(meth, p["verdict"] == "X_PHENOMENON_GO", weak)
    return {"status": "OK", "phenomenon": p, "method": m,
            "verdict": XG.track_verdict(m["verdict"]),
            "detailed_verdict": m["verdict"], "result_role": m["result_role"]}


def build_track_g():
    diag = latest("track_g_diagnostic", "gradient_diagnostic.json")
    inter = latest("track_g_intervention", "intervention.json")
    if diag is None:
        return {"status": "BLOCKED", "verdict": "NOT_EVALUATED",
                "reason": "Track G diagnostic artifact missing"}
    GG = load_gates("track_g")
    p = GG.phenomenon_verdict(diag)
    if inter is None:
        return {"status": "PARTIAL", "phenomenon": p, "verdict": "NOT_EVALUATED",
                "reason": "Track G intervention artifact missing"}
    po = g_per_origin(inter)
    m = GG.method_verdict(inter, po, p["verdict"] == "G_PHENOMENON_GO")
    return {"status": "OK", "phenomenon": p, "method": m,
            "verdict": GG.track_verdict(m["verdict"]),
            "detailed_verdict": m["verdict"], "result_role": m["result_role"]}


def build_track_f():
    sel = latest("track_f_selection", "selection.json")
    if sel is None:
        return {"status": "BLOCKED", "verdict": "NOT_EVALUATED",
                "reason": "Track F artifact missing"}
    FG = load_gates("track_f")
    p = FG.phenomenon_verdict(sel)
    pw = f_per_window(sel, latest_dir("track_f_selection"))
    m = FG.method_verdict(sel, pw, p["verdict"] == "F_SELECTION_CONFOUNDING_PRESENT")
    return {"status": "OK", "phenomenon": p, "method": m,
            "verdict": FG.track_verdict(m["verdict"]),
            "detailed_verdict": m["verdict"], "result_role": m["result_role"]}


def build_track_v():
    v = latest("track_v", "validation_stability.json")
    if v is None:
        return {"status": "BLOCKED", "verdict": "NOT_EVALUATED"}
    g = json.loads((R / "preregistration.json").read_text())["gates"]["V"]
    per_ds = {}
    for dsname, e in v["datasets"].items():
        if "selection_rules" not in e:
            per_ds[dsname] = {"status": e.get("status", "BLOCKED")}
            continue
        r = e["selection_rules"]
        latest_regret = r["latest_1"]["test_regret"]
        alts = [r[k]["test_regret"] for k in ("last_4", "arw") if k in r]
        red = max(((latest_regret - a) / latest_regret) if latest_regret > 0 else 0.0
                  for a in alts) if alts else 0.0
        per_ds[dsname] = {
            "latest_origin_regret": latest_regret,
            "last4_regret": r.get("last_4", {}).get("test_regret"),
            "arw_regret": r.get("arw", {}).get("test_regret"),
            "best_alternative_reduction": red,
            "unsafe": bool(latest_regret >= g["latest_origin_regret_ge"]
                           and red >= g["last4_or_arw_regret_reduction_ge"]),
        }
    unsafe = bool(per_ds) and all(x.get("unsafe", False) for x in per_ds.values())
    return {"status": "OK", "per_dataset": per_ds, "raw": v,
            "verdict": "V_SINGLE_ORIGIN_SELECTION_UNSAFE" if unsafe
            else "V_NO_STRONG_SELECTION_INSTABILITY",
            "note": "Track V never changes a G, X or F verdict."}


# --------------------------------------------------------------------------- #

def score_track(name: str, t: dict) -> dict:
    """Fixed 0-5 scoring. Criterion 1 (the worse dataset's improvement) is the
    primary tie-breaker; average improvement is never used to pick a winner.
    """
    if t.get("verdict") != "METHOD_GO":
        return {"eligible": False}
    m = t["method"]["per_dataset"]
    if name == "G":
        gains = [v["gain_vs_erm"] for v in m.values()]
        extra = [v["gain_vs_pcgrad"] for v in m.values()]
        cost = np.mean([v["wall_s"]["probe_gated"] / max(v["wall_s"]["erm"], 1e-9)
                        for v in m.values()])
    elif name == "X":
        gains = [v["quarantine_extra_reduction_vs_best_simple"] or 0.0 for v in m.values()]
        extra = gains
        cost = 1.0
    else:
        gains = [v["shifted_test_gain"] for v in m.values()]
        extra = [v["shift_retention_gain_pp"] / 100 for v in m.values()]
        cost = 1.0
    phen = t["phenomenon"]
    strength = float(np.mean([1.0 if v.get("passes") else 0.0
                              for v in phen["per_dataset"].values()]))
    def clip5(x, scale):
        return float(np.clip(round(x / scale), 0, 5))
    return {
        "eligible": True,
        "c1_worse_dataset_improvement": float(min(gains)),
        "c1_score": clip5(min(gains) * 100, 0.5),
        "c2_phenomenon_strength": strength, "c2_score": clip5(strength * 5, 1.0),
        "c3_gain_over_best_simple": float(min(extra)), "c3_score": clip5(min(extra) * 100, 1.0),
        "c4_functional_difference_from_closest_work": 3.0,
        "c5_additional_runtime_cost": float(cost),
        "c5_score": float(np.clip(5 - round(cost), 0, 5)),
    }


def final_ranking(tracks: dict) -> dict:
    scores = {k: score_track(k, tracks[k]) for k in ("G", "X", "F")}
    go = [k for k, s in scores.items() if s.get("eligible")]
    if go:
        best = sorted(go, key=lambda k: (-scores[k]["c1_worse_dataset_improvement"],
                                         -scores[k]["c2_phenomenon_strength"],
                                         -scores[k]["c3_gain_over_best_simple"],
                                         -scores[k]["c4_functional_difference_from_closest_work"],
                                         scores[k]["c5_additional_runtime_cost"]))[0]
        token = {"G": "RECOMMEND_G_HARMFUL_INTERFERENCE",
                 "X": "RECOMMEND_X_CORRUPTION_SPILLOVER",
                 "F": "RECOMMEND_F_SELECTION_CONFOUNDING"}[best]
    elif any(tracks[k].get("verdict") == "CHARACTERIZATION_ONLY" for k in ("G", "X", "F")):
        best, token = None, "RECOMMEND_CHARACTERIZATION_ONLY"
    else:
        best, token = None, "NO_NEW_TOPIC_PASSED"
    return {"scores": scores, "method_go_tracks": go, "selected": best,
            "final_token": token,
            "rule": ("Only METHOD_GO tracks are candidates. Ranking is by the worse "
                     "dataset's improvement first, then phenomenon strength, gain over "
                     "the best simple baseline, functional difference from the closest "
                     "work, and runtime cost. Average improvement is never used.")}


def status_md(tracks: dict, ranking: dict, meta: dict) -> str:
    rows = []
    for k, label in [("X", "TRACK X"), ("G", "TRACK G"), ("F", "TRACK F")]:
        t = tracks[k]
        ph = t.get("phenomenon", {}).get("verdict", "NOT_EVALUATED")
        sb = "yes" if (t.get("method", {}).get("simple_baseline_solves")
                       or t.get("method", {}).get("existing_baseline_solves")
                       or t.get("method", {}).get("existing_adaptive_filter_solves")) else "no"
        mg = "yes" if t.get("method", {}).get("method_go") else "no"
        cons = "n/a"
        boot = "n/a"
        m = t.get("method", {}).get("per_dataset", {})
        if k == "G" and m:
            cons = "same sign" if len({np.sign(v["gain_vs_erm"]) for v in m.values()}) == 1 else "conflicting"
            boot = ", ".join(f"{d}[{v['bootstrap_vs_erm']['ci_lower']:+.4f}, "
                             f"{v['bootstrap_vs_erm']['ci_upper']:+.4f}]" for d, v in m.items())
        elif k == "F" and m:
            cons = "same sign" if len({np.sign(v["shifted_test_gain"]) for v in m.values()}) == 1 else "conflicting"
            boot = ", ".join(f"{d}[{v['bootstrap_shifted_gain']['ci_lower']:+.4f}, "
                             f"{v['bootstrap_shifted_gain']['ci_upper']:+.4f}]" for d, v in m.items())
        elif k == "X" and m:
            vals = [v["quarantine_extra_reduction_vs_best_simple"] for v in m.values()
                    if v["quarantine_extra_reduction_vs_best_simple"] is not None]
            cons = "same sign" if vals and len({np.sign(v) for v in vals}) == 1 else "conflicting"
            boot = ", ".join(f"{d}[{(v.get('quarantine_bootstrap') or {}).get('ci_lower', float('nan')):+.4f}, "
                             f"{(v.get('quarantine_bootstrap') or {}).get('ci_upper', float('nan')):+.4f}]"
                             for d, v in m.items())
        rows.append(f"| {label} | {ph} | {sb} | {mg} | {cons} | {boot} | {t.get('verdict')} |")

    lines = [
        f"FINAL TOPIC RECOMMENDATION: {ranking['final_token']}",
        "",
        f"TRACK X: {tracks['X'].get('detailed_verdict', tracks['X'].get('verdict'))}",
        f"TRACK G: {tracks['G'].get('detailed_verdict', tracks['G'].get('verdict'))}",
        f"TRACK F: {tracks['F'].get('detailed_verdict', tracks['F'].get('verdict'))}",
        f"TRACK V: {tracks['V'].get('verdict')}",
        "",
        "| Track | Phenomenon | Simple baseline solves | New intervention | Dataset consistency | Bootstrap 95% CI | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## Result roles",
        "",
        "- `[CONFIRMATORY SCREEN]` " + ", ".join(
            f"{k}" for k in ("X", "G", "F") if tracks[k].get("result_role") == "CONFIRMATORY_SCREEN") or "- none",
        "- `[DIAGNOSTIC]` " + (", ".join(
            f"{k}" for k in ("X", "G", "F")
            if str(tracks[k].get("result_role", "")).startswith("DIAGNOSTIC")) or "none"),
        "- `[BLOCKED]` " + (", ".join(
            f"{k} ({tracks[k].get('reason')})" for k in ("X", "G", "F", "V")
            if tracks[k].get("status") in ("BLOCKED", "PARTIAL")) or "none"),
        "",
        "## Run metadata",
        "",
        *(f"- {k}: {v}" for k, v in meta.items()),
    ]
    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    tracks = {"X": build_track_x(), "G": build_track_g(),
              "F": build_track_f(), "V": build_track_v()}
    ranking = final_ranking(tracks)

    for k, sub in [("X", "track_x"), ("G", "track_g"), ("F", "track_f")]:
        d = R / sub
        d.mkdir(parents=True, exist_ok=True)
        A.write_json(d / "gates.json", tracks[k])
        (d / "STATUS.md").write_text(
            f"# Track {k}\n\nverdict: {tracks[k].get('verdict')}\n"
            f"detailed: {tracks[k].get('detailed_verdict')}\n"
            f"result role: {tracks[k].get('result_role')}\n"
            f"status: {tracks[k].get('status')}\n")
    (R / "track_v").mkdir(parents=True, exist_ok=True)
    A.write_json(R / "track_v" / "validation_stability.json", tracks["V"])
    (R / "track_v" / "STATUS.md").write_text(
        f"# Track V\n\nverdict: {tracks['V'].get('verdict')}\n"
        f"role: auxiliary audit; never changes a G, X or F verdict.\n")

    A.write_json(R / "final_topic_ranking.json", ranking)
    tables = {
        "clean_baselines": json.loads((R / "clean_baselines.json").read_text())["gate"],
        "track_x": tracks["X"].get("method", {}).get("per_dataset"),
        "track_g": tracks["G"].get("method", {}).get("per_dataset"),
        "track_f": tracks["F"].get("method", {}).get("per_dataset"),
        "track_v": tracks["V"].get("per_dataset"),
        "ranking": ranking,
    }
    A.write_json(R / "tables.json", tables)

    peaks = []
    for st in ("clean_baselines", "track_g_diagnostic", "track_g_intervention",
               "track_x_phenomenon", "track_x_mitigations", "track_f_selection", "track_v"):
        d = latest_dir(st)
        if d and (d / "resource_peak.json").exists():
            peaks.append(json.loads((d / "resource_peak.json").read_text()))
    meta = {
        "runtime_tier": json.loads((R / "runtime_tier.json").read_text())["tier"],
        "projected_gpu_hours": json.loads((R / "runtime_tier.json").read_text())["projected_total_gpu_hours"],
        "peak_process_tree_rss_gb": max([p["tree_rss_gb"] for p in peaks], default=None),
        "peak_gpu_used_gb": max([p["gpu_used_gb"] for p in peaks], default=None),
        "peak_system_memory_pct": max([p["sys_pct"] for p in peaks], default=None),
        "finalize_wall_s": round(time.time() - t0, 1),
    }
    (R / "STATUS.md").write_text(status_md(tracks, ranking, meta))
    (R / "final_console_summary.txt").write_text(status_md(tracks, ranking, meta))
    print((R / "STATUS.md").read_text())


if __name__ == "__main__":
    main()
