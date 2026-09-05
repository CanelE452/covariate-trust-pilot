"""Stage 30: at most eight figures, each drawn only when its payload exists.

1 spillover heatmap                       (Track X)
2 corruption-family spillover bars        (Track X)
3 same-batch cosine vs actual harm        (Track G)
4 cross-probe affinity vs cosine, PR      (Track G)
5 ERM / PCGrad / norm-balanced / gated    (Track G)
6 removed-window composition              (Track F)
7 clean vs shifted test performance       (Track F)
8 topic tournament scorecard              (final)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths

OUT = paths.RESULTS / "figures"
plt.rcParams.update({"figure.dpi": 140, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
made = []


def latest(stage, fname):
    fs = sorted((paths.RUNS / stage).glob(f"attempt_*/{fname}"))
    fs = [f for f in fs if (f.parent / "completion.json").exists()]
    return json.loads(fs[-1].read_text()) if fs else None


def latest_dir(stage):
    ds = sorted(p for p in (paths.RUNS / stage).glob("attempt_*")
                if (p / "completion.json").exists())
    return ds[-1] if ds else None


def save(fig, n, name):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"fig{n}_{name}.png"
    fig.savefig(f)
    plt.close(fig)
    made.append(str(f.relative_to(paths.ROOT)))
    print("wrote", f.name)


# --------------------------------------------------------------------------- #

def fig1_fig2_track_x(phen):
    if not phen:
        return
    dss = list(phen["datasets"])
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3.6))
    axes = np.atleast_1d(axes)
    drew = False
    for ax, dsname in zip(axes, dss):
        e = phen["datasets"][dsname]
        recs = [r for r in e["raw"]["TQNet"] if r["family"] == "combined"]
        if not recs:
            continue
        chans = e["spec_channels"]
        C = len(recs[0]["rel"])
        M = np.full((len(chans), C), np.nan)
        for a, j in enumerate(chans):
            sub = [r["rel"] for r in recs if r["j"] == j]
            if sub:
                M[a] = np.median(np.array(sub), axis=0)
        v = np.nanpercentile(np.abs(M), 98) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_title(f"{dsname}: median loss inflation, combined corruption")
        ax.set_xlabel("affected output channel i")
        ax.set_ylabel("corrupted input channel j")
        ax.set_yticks(range(len(chans)))
        ax.set_yticklabels(chans)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label="relative MSE increase")
        drew = True
    if drew:
        save(fig, 1, "spillover_heatmap")
    else:
        plt.close(fig)

    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    drew = False
    for ax, dsname in zip(axes, dss):
        s = phen["datasets"][dsname]["summary"]
        keys = sorted(s["TQNet"])
        if not keys:
            continue
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [s["TQNet"][k]["median_offdiag_spillover"] for k in keys],
               0.4, label="TQNet (shared)")
        ax.bar(x + 0.2, [s["DLinear"][k]["median_offdiag_spillover"] for k in keys],
               0.4, label="DLinear (channel-independent)")
        ax.axhline(0.03, ls="--", lw=0.8, color="k")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_title(dsname)
        ax.set_ylabel("median off-diagonal spillover")
        drew = True
    if drew:
        axes[0].legend(frameon=False, fontsize=7)
        save(fig, 2, "corruption_family_spillover")
    else:
        plt.close(fig)


def fig3_fig4_track_g(diag):
    if not diag:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, dsname in zip(axes, sorted({v["dataset"] for v in diag["checkpoints"].values()})):
        rows = [r for k, v in diag["checkpoints"].items() if v["dataset"] == dsname
                and v["checkpoint_tag"] == "best" for r in v["rows"]]
        c = np.array([r["cos_same_batch"] for r in rows])
        h = np.array([r["exact_harm"] for r in rows])
        lab = np.array([r["harm_label"] for r in rows])
        ax.scatter(c[~lab], h[~lab], s=7, alpha=0.55, label="no harm")
        ax.scatter(c[lab], h[lab], s=7, alpha=0.55, label="harm > 0.5%")
        ax.axhline(0.005, ls="--", lw=0.8, color="k")
        ax.axvline(0.0, ls=":", lw=0.8, color="k")
        ax.set_xlabel("same-batch gradient cosine")
        ax.set_ylabel("exact probe-loss change")
        ax.set_title(dsname)
        ax.legend(frameon=False, fontsize=7)
    save(fig, 3, "cosine_vs_actual_harm")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, dsname in zip(axes, sorted({v["dataset"] for v in diag["checkpoints"].values()})):
        rows = [r for k, v in diag["checkpoints"].items() if v["dataset"] == dsname
                and v["checkpoint_tag"] == "best" for r in v["rows"]
                if r["selection"] == "random"]
        lab = np.array([r["harm_label"] for r in rows], float)
        for name, sc in [("same-batch cosine", np.array([-r["cos_same_batch"] for r in rows])),
                         ("cross-probe affinity",
                          np.array([r["cross_probe_affinity"] for r in rows]))]:
            o = np.argsort(-sc)
            l = lab[o]
            tp = np.cumsum(l)
            prec = tp / np.arange(1, len(l) + 1)
            rec = tp / max(l.sum(), 1)
            ax.plot(rec, prec, lw=1.4, label=name)
        ax.axhline(lab.mean(), ls="--", lw=0.8, color="k", label="base rate")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{dsname}: detecting real harm (unbiased random subset)")
        ax.legend(frameon=False, fontsize=7)
    save(fig, 4, "harm_detector_precision_recall")


def fig5_track_g(inter):
    if not inter:
        return
    dss = sorted({r["dataset"] for r in inter["runs"].values()})
    arms = inter["arms"]
    fig, axes = plt.subplots(1, len(dss), figsize=(4.0 * len(dss), 3.2))
    axes = np.atleast_1d(axes)
    for ax, dsname in zip(axes, dss):
        vals, errs = [], []
        for a in arms:
            v = [r["macro_mse"] for r in inter["runs"].values()
                 if r["dataset"] == dsname and r["arm"] == a]
            vals.append(np.mean(v) if v else np.nan)
            errs.append(np.std(v) if len(v) > 1 else 0.0)
        base = vals[arms.index("erm")]
        ax.bar(range(len(arms)), vals, yerr=errs, capsize=3)
        ax.axhline(base, ls="--", lw=0.8, color="k")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=30, ha="right")
        ax.set_ylabel("macro MSE over task variables")
        ax.set_ylim(min(vals) * 0.98, max(vals) * 1.02)
        ax.set_title(f"{dsname} (dashed = ERM)")
    save(fig, 5, "gradient_rule_comparison")


def fig6_fig7_track_f(sel):
    if not sel:
        return
    dss = list(sel["datasets"])
    methods = [m for m in sel["methods"] if m != "no_filter"]
    fig, axes = plt.subplots(1, len(dss), figsize=(4.4 * len(dss), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dsname in zip(axes, dss):
        d = sel["datasets"][dsname]["selection_diagnostic"]
        x = np.arange(len(methods))
        w = 0.27
        ax.bar(x - w, [d[m]["corruption_removal_rate"] for m in methods], w,
               label="corruption removed")
        ax.bar(x, [d[m]["legitimate_shift_removal_rate"] for m in methods], w,
               label="legitimate shift removed")
        ax.bar(x + w, [d[m]["clean_removal_rate"] for m in methods], w, label="clean removed")
        ax.axhline(0.20, ls="--", lw=0.8, color="k")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("fraction of that class removed")
        ax.set_title(f"{dsname} (dashed = random 20% budget)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, 6, "removed_window_composition")

    fig, axes = plt.subplots(1, len(dss), figsize=(4.4 * len(dss), 3.4))
    axes = np.atleast_1d(axes)
    for ax, dsname in zip(axes, dss):
        rt = sel["datasets"][dsname]["retraining"]
        ms = sel["methods"]
        cl = [np.mean([r["clean_test_mse"] for r in rt.values() if r["method"] == m]) for m in ms]
        sh = [np.mean([r["shifted_test_mse"] for r in rt.values() if r["method"] == m]) for m in ms]
        x = np.arange(len(ms))
        ax.bar(x - 0.2, cl, 0.4, label="clean test")
        ax.bar(x + 0.2, sh, 0.4, label="shifted test")
        ax.set_xticks(x)
        ax.set_xticklabels(ms, rotation=30, ha="right")
        ax.set_ylabel("test MSE")
        ax.set_title(dsname)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, 7, "clean_vs_shifted_test")


def fig8_scorecard(rank, tracks):
    crit = ["c1_score", "c2_score", "c3_score",
            "c4_functional_difference_from_closest_work", "c5_score"]
    labels = ["worse-dataset\nimprovement", "phenomenon\nstrength",
              "gain over best\nsimple baseline", "difference from\nclosest work",
              "runtime cost\n(higher = cheaper)"]
    names = ["G", "X", "F"]
    M = np.zeros((len(names), len(crit)))
    for i, n in enumerate(names):
        s = rank["scores"].get(n, {})
        for j, c in enumerate(crit):
            M[i, j] = s.get(c, 0.0) if s.get("eligible") else 0.0
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=5)
    for i in range(len(names)):
        for j in range(len(crit)):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(crit)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}: {tracks[n].get('verdict')}" for n in names], fontsize=7)
    ax.grid(False)
    ax.set_title(f"Topic tournament scorecard - {rank['final_token']}", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, label="score (0-5)")
    save(fig, 8, "topic_tournament_scorecard")


def main():
    phen = latest("track_x_phenomenon", "phenomenon_raw.json")
    diag = latest("track_g_diagnostic", "gradient_diagnostic.json")
    inter = latest("track_g_intervention", "intervention.json")
    sel = latest("track_f_selection", "selection.json")
    fig1_fig2_track_x(phen)
    fig3_fig4_track_g(diag)
    fig5_track_g(inter)
    fig6_fig7_track_f(sel)
    rf = paths.RESULTS / "final_topic_ranking.json"
    if rf.exists():
        rank = json.loads(rf.read_text())
        tracks = {k: json.loads((paths.RESULTS / f"track_{k.lower()}" / "gates.json").read_text())
                  for k in ("G", "X", "F")}
        fig8_scorecard(rank, tracks)
    print(json.dumps({"figures": made, "n": len(made)}, indent=1))


if __name__ == "__main__":
    main()
