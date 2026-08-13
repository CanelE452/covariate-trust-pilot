"""Draft renderer for Figures 1-3.

Every number is read from a verified artifact at run time; nothing is typed in from
a document. The only hand-made values are the toy sequences in Figure 1a, which are
schematic by construction and labelled as such.

Run from the repository root:
    python results/paper_rendering_verified/render_drafts.py
"""

from __future__ import annotations

import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = pathlib.Path(".")
SYN = ROOT / "results/synthetic_source_verification"
EMP = ROOT / "results/external_validity_screen"
OUT = ROOT / "results/paper_rendering_verified/drafts"
OUT.mkdir(parents=True, exist_ok=True)

SINGLE, DOUBLE = 3.5, 7.16          # inches, candidate column widths
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 200,
})
SRC_ROWS: list[dict] = []


def note(fig, panel, quantity, source, field, agg, unc, value):
    SRC_ROWS.append({"figure_or_table": fig, "panel_or_row": panel,
                     "displayed_quantity": quantity, "source_file": source,
                     "source_field_or_columns": field, "aggregation": agg,
                     "uncertainty_source": unc, "verified_value": value})


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.svg/.png")


# --------------------------------------------------------------------------- data
def stage2():
    rows = [r for r in csv.DictReader(open(SYN / "stage2_verified_cells.csv"))
            if r["model"] == "HURDLE_MEAN"]
    for r in rows:
        r["G"] = float(r["gain"]) * 100
        r["lo"] = float(r["gain_ci_low"]) * 100
        r["hi"] = float(r["gain_ci_high"]) * 100
        r["sig"] = r["delta_ci_excludes_zero"] == "True"
        r["d"] = int(r["d"])
        r["rI"] = float(r["rho_interval"])
        r["rM"] = float(r["rho_magnitude"])
    return rows


def stage2_effects():
    return {r["term"]: (float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"]))
            for r in csv.DictReader(open(SYN / "stage2_verified_factor_effects.csv"))
            if r["model"] == "HURDLE_MEAN"}


# ------------------------------------------------------------------------ Figure 1
def figure1():
    fig = plt.figure(figsize=(DOUBLE, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.34)

    # (a) schematic: identical multiset, different order
    ax = fig.add_subplot(gs[0])
    mags = [5, 15, 5, 15, 5, 15]
    gaps_alt = [3, 5, 3, 5, 3, 5]          # alternating
    gaps_clu = [3, 3, 3, 5, 5, 5]          # clustered; same multiset
    for row, (gaps, lab) in enumerate([(gaps_alt, "ordering A"), (gaps_clu, "ordering B")]):
        y = 1 - row
        t = 0
        ts, hs = [], []
        for g, m in zip(gaps, mags if row == 0 else mags[::-1]):
            t += g
            ts.append(t)
            hs.append(m)
        ax.hlines(y, 0, 30, color="0.85", lw=0.8, zorder=0)
        ax.vlines(ts, y, [y + h / 45 for h in hs], color="0.25", lw=1.6)
        ax.scatter(ts, [y + h / 45 for h in hs], s=7, color="0.15", zorder=3)
        ax.text(-1.2, y, lab, ha="right", va="center", fontsize=6.5)
    ax.set_xlim(-9, 31)
    ax.set_ylim(-0.35, 1.75)
    ax.set_yticks([])
    ax.set_xlabel("time")
    ax.set_title("(a) same marginals, different order", loc="left")
    ax.text(0.5, -0.42, "identical gap multiset, identical magnitude multiset\n"
                        "schematic illustration; not experimental data",
            transform=ax.transAxes, ha="center", va="top", fontsize=6, color="0.3")

    # (b) the two axes, symmetric
    ax = fig.add_subplot(gs[1])
    for i, (name, sym) in enumerate([("occurrence  " + r"$\rho_I$", ""),
                                     ("magnitude  " + r"$\rho_M$", "")]):
        y = 1 - i
        ax.annotate("", xy=(1.05, y), xytext=(-1.05, y),
                    arrowprops=dict(arrowstyle="-", lw=0.9, color="0.4"))
        for x, lab in [(-0.8, "alternating"), (0.0, "iid"), (0.8, "persistent")]:
            ax.scatter([x], [y], s=16, facecolor="white", edgecolor="0.25", lw=0.9, zorder=3)
            ax.text(x, y - 0.24, lab, ha="center", va="top", fontsize=6, color="0.3")
            ax.text(x, y + 0.13, f"{x:+.1f}".replace("+0.0", " 0.0"),
                    ha="center", va="bottom", fontsize=6, color="0.45")
        ax.text(0, y + 0.42, name, ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.55, 1.85)
    ax.axis("off")
    ax.set_title("(b) two dependence axes", loc="left")

    # (c) the two formulations
    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    ax.set_title("(c) two representations of one target", loc="left")
    box = dict(boxstyle="round,pad=0.32", fc="white", ec="0.35", lw=0.8)
    ax.text(0.5, 0.86, "history", ha="center", va="center", fontsize=7, bbox=box)
    ax.text(0.22, 0.52, "direct\n" + r"$E[Y\,|\,\mathcal{H}]$", ha="center", va="center",
            fontsize=6.8, bbox=box)
    ax.text(0.78, 0.52, "factorized\n" + r"$P(Y{>}0\,|\,\mathcal{H})\times E[Y\,|\,Y{>}0,\mathcal{H}]$",
            ha="center", va="center", fontsize=6.2, bbox=box)
    for x0 in (0.22, 0.78):
        ax.add_patch(FancyArrowPatch((0.5, 0.78), (x0, 0.62), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=6, lw=0.7, color="0.4"))
        ax.add_patch(FancyArrowPatch((x0, 0.42), (0.5, 0.26), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=6, lw=0.7, color="0.4"))
    ax.text(0.5, 0.20, "same conditional mean target\nmatched budget: 5,856 parameters each",
            ha="center", va="center", fontsize=6.4, bbox=box)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save(fig, "figure1_draft")
    note("Figure1", "a", "toy sequences", "(schematic)", "-", "-", "-", "not data")
    note("Figure1", "c", "parameter count", "point_hurdle_fairness.md / prereg",
         "PARAMETER_MATCH_RULE", "-", "-", "5856 = 5856")


# ------------------------------------------------------------------------ Figure 2
def figure2(marker_option="B"):
    rows = stage2()
    eff = stage2_effects()
    G = [r["G"] for r in rows]
    lim = float(np.ceil(max(abs(min(G)), abs(max(G)))))
    lim = max(lim, 23.0)

    fig = plt.figure(figsize=(DOUBLE, 4.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.52, wspace=0.26)

    # ---- 2a heatmaps
    levels = [-0.8, 0.0, 0.8]
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, d in zip(axes, (4, 8)):
        M = np.full((3, 3), np.nan)
        for r in rows:
            if r["d"] == d:
                M[levels.index(r["rI"]), levels.index(r["rM"])] = r["G"]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                shade = "white" if abs(v) > 0.55 * lim else "black"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7, color=shade)
        for r in rows:
            if r["d"] != d:
                continue
            i, j = levels.index(r["rI"]), levels.index(r["rM"])
            if marker_option == "B" and not r["sig"]:
                ax.scatter([j], [i - 0.33], s=13, facecolor="none",
                           edgecolor="0.15", lw=0.8, zorder=4)
            if marker_option == "A" and r["sig"]:
                ax.scatter([j], [i - 0.33], s=5, color="0.15", zorder=4)
        ax.set_xticks(range(3), [f"{v:+.1f}" for v in levels])
        ax.set_yticks(range(3), [f"{v:+.1f}" for v in levels])
        ax.set_xlabel(r"magnitude dependence  $\rho_M$")
        if d == 4:
            ax.set_ylabel(r"occurrence dependence  $\rho_I$")
        ax.set_title(("(a)  " if d == 4 else "") + f"$d={d}$", loc="left")
        for s in ax.spines.values():
            s.set_visible(True)
    cb = fig.colorbar(im, ax=axes, fraction=0.032, pad=0.02)
    cb.set_label("G  (pp)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6)
    cb.ax.text(0.5, 1.03, "factorized better", transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=5.8)
    cb.ax.text(0.5, -0.05, "direct better", transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=5.8)

    # ---- 2b / 2c marginals, shared y
    means = {}
    for key, lab in (("rI", "b"), ("rM", "c")):
        means[key] = [(lv, float(np.mean([r["G"] for r in rows if r[key] == lv])),
                       [r["G"] for r in rows if r[key] == lv]) for lv in levels]
    allpts = [g for k in means for _, _, pts in means[k] for g in pts]
    ylim = (min(allpts) - 4, max(allpts) + 3)

    for idx, (key, panel, xlab, ann) in enumerate([
            ("rI", "(b) occurrence marginal", r"$\rho_I$",
             r"$|\rho_I|$ %+.3f  >  signed %+.3f" % (eff["abs_rho_I"][0], eff["rho_I"][0])),
            ("rM", "(c) magnitude marginal", r"$\rho_M$",
             r"signed $\rho_M$ %+.3f  >  $|\rho_M|$ %+.3f" % (eff["rho_M"][0], eff["abs_rho_M"][0]))]):
        ax = fig.add_subplot(gs[1, idx])
        xs = [m[0] for m in means[key]]
        ys = [m[1] for m in means[key]]
        for x, _, pts in means[key]:
            ax.scatter([x] * len(pts), pts, s=7, facecolor="none",
                       edgecolor="0.6", lw=0.6, zorder=2)
        ax.plot(xs, ys, color="0.25", lw=0.9, zorder=3, alpha=0.7)
        ax.scatter(xs, ys, s=26, color="0.15", zorder=4, label="mean over 6 cells")
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_xticks(levels, [f"{v:+.1f}" for v in levels])
        ax.set_xlabel(xlab)
        ax.set_ylim(*ylim)
        if idx == 0:
            ax.set_ylabel("G  (pp)")
        else:
            ax.set_yticklabels([])
        ax.set_title(panel, loc="left")
        ax.text(0.5, 0.16, ann, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.2, color="0.2")
        if idx == 0:
            ax.text(0.5, 0.05, "open circles: individual cells (spread, not CI)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=5.6, color="0.45")

    save(fig, "figure2_final" if marker_option == "B"
         else "figure2_comparison_markerA_NOT_FINAL")

    for r in rows:
        note("Figure2", f"a d={r['d']} rI={r['rI']:+.1f} rM={r['rM']:+.1f}", "G (pp)",
             "stage2_verified_cells.csv", "gain*100", "per-cell",
             "gain_ci_low/high (bootstrap 2000, series)", f"{r['G']:+.4f}")
    for key, panel in (("rI", "b"), ("rM", "c")):
        for lv, mu, pts in means[key]:
            note("Figure2", f"{panel} {key}={lv:+.1f}", "mean G over 6 cells",
                 "stage2_verified_cells.csv", "gain*100", "mean of 6 cells",
                 "NONE - cell spread shown, no marginal CI exists", f"{mu:+.4f}")
    for t in ("abs_rho_I", "rho_I", "rho_M", "abs_rho_M"):
        note("Figure2", "b/c annotation", f"factor coefficient {t}",
             "stage2_verified_factor_effects.csv", "estimate", "factor model",
             "ci_low/ci_high", f"{eff[t][0]:+.4f}")
    return lim


# ------------------------------------------------------------------------ Figure 3
def figure3():
    sa = json.load(open(EMP / "stage_a_results.json"))
    reg = json.load(open(EMP / "regime_h1/regime_h1.json"))
    rr = json.load(open(EMP / "rule_replication/primary_result.json"))
    sr = json.load(open(EMP / "seed_robustness/seed_robustness.json"))
    so = json.load(open(EMP / "rule_replication/secondary_overlap.json"))
    mf = json.load(open(EMP / "h2_confirmatory/matching_failed.json"))

    fig = plt.figure(figsize=(DOUBLE, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.42)

    # ---- 3a H1 forest
    ax = fig.add_subplot(gs[0])
    items = []
    for d, lab in (("m5", "M5"), ("favorita", "Favorita")):
        h = sa["results"][d]["20"]["H1"]
        items.append((lab, h["spearman"], h["ci"][0], h["ci"][1], True))
        note("Figure3", f"a {lab}", "Spearman(|rho_interval|, delta)",
             "stage_a_results.json", "results.*.20.H1", "per dataset",
             "bootstrap ci", f"{h['spearman']:+.4f}")
    items.append((None, None, None, None, None))
    for k in ("smooth", "erratic", "intermittent", "lumpy"):
        v = reg["regimes"][k]["relative"]
        items.append((f"  {k}", v["spearman"], v["ci"][0], v["ci"][1],
                      v["ci"][0] > 0 or v["ci"][1] < 0))
        note("Figure3", f"a regime {k}", "Spearman, relative scale",
             "regime_h1/regime_h1.json", "regimes.*.relative", "M5 SBC subset",
             "bootstrap ci", f"{v['spearman']:+.4f}")
    ys = list(range(len(items)))[::-1]
    for y, (lab, v, lo, hi, sig) in zip(ys, items):
        if lab is None:
            continue
        ax.plot([lo, hi], [y, y], color="0.3", lw=1.0)
        ax.scatter([v], [y], s=22 if not lab.startswith("  ") else 13,
                   color="0.1" if sig else "white",
                   edgecolor="0.1", lw=0.8, zorder=3)
    ax.axvline(0, color="0.6", lw=0.7, ls=":")
    ax.set_yticks(ys, [i[0] if i[0] else "" for i in items])
    ax.set_xlabel("Spearman")
    ax.set_title("(a) H1 empirical analogue", loc="left")
    ax.text(0.02, 0.02, "filled: interval excludes zero\nindented: M5 SBC regimes",
            transform=ax.transAxes, fontsize=5.6, color="0.4", va="bottom")

    # ---- 3b H2 selector
    ax = fig.add_subplot(gs[1])
    labs, vals, los, his = [], [], [], []
    for s in ("0", "1", "2"):
        b = sr["by_seed"][s]
        labs.append(f"seed {s}")
        vals.append(b["effect"] * 100)
        los.append(b["effect_ci"][0] * 100)
        his.append(b["effect_ci"][1] * 100)
        note("Figure3", f"b seed {s}", "rule effect x100",
             "seed_robustness/seed_robustness.json", "by_seed.*.effect", "per seed",
             "bootstrap ci", f"{b['effect']*100:+.4f}")
    agg = sr["aggregate_3seed"]
    labs.append("3-seed")
    vals.append(agg["effect"] * 100)
    los.append(agg["effect_ci"][0] * 100)
    his.append(agg["effect_ci"][1] * 100)
    note("Figure3", "b 3-seed aggregate", "rule effect x100",
         "seed_robustness/seed_robustness.json", "aggregate_3seed.effect",
         f"{rr['candidate']['n']} vs {rr['control']['n']} per seed", "bootstrap ci",
         f"{agg['effect']*100:+.4f}")
    ys = list(range(len(labs)))[::-1]
    for y, v, lo, hi in zip(ys, vals, los, his):
        ax.plot([lo, hi], [y, y], color="0.3", lw=1.0)
        ax.scatter([v], [y], s=22, color="0.1", zorder=3)
    ax.axvline(0, color="0.6", lw=0.7, ls=":")
    ax.set_yticks(ys, labs)
    ax.set_xlabel("rule effect  (pp of relative error)")
    ax.set_title("(b) H2 selector transfer", loc="left")
    ax.set_ylim(-0.7, len(labs) - 0.3)
    ax.text(0.5, -0.235, "negative = shifted toward direct.  seed 0 is the primary run.",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color="0.35")

    # ---- 3c overlap boundary
    ax = fig.add_subplot(gs[2])
    pts = [("unadjusted", rr["H2_rule_effect"] * 100,
            [c * 100 for c in rr["H2_rule_effect_ci"]]),
           ("overlap-adjusted", so["overlap_adjusted_association"] * 100,
            [c * 100 for c in so["overlap_adjusted_association_ci"]])]
    ys = [1, 0]
    for y, (lab, v, ci) in zip(ys, pts):
        ax.plot([ci[0], ci[1]], [y, y], color="0.3", lw=1.1)
        ax.scatter([v], [y], s=26, color="0.1", zorder=3)
    ax.axvline(0, color="0.55", lw=0.8, ls=":")
    ax.set_yticks(ys, [p[0] for p in pts])
    ax.set_ylim(-1.25, 1.5)
    ax.set_xlabel("association  (pp of relative error)")
    ax.set_title("(c) mechanism boundary", loc="left")
    worst_un = max(abs(v) for v in so["unweighted_smd"].values())
    ax.text(0.5, -0.55,
            "worst |SMD| on log scale\n"
            f"unweighted {worst_un:.2f}   after 1:2 matching {mf['worst_abs_smd']:.2f} (failed)\n"
            f"after overlap weighting {so['worst_abs_weighted_smd']:.4f}",
            transform=ax.get_yaxis_transform(), ha="center", va="top",
            fontsize=5.8, color="0.3")
    note("Figure3", "c unadjusted", "rule effect",
         "rule_replication/primary_result.json", "H2_rule_effect", "-", "bootstrap ci",
         f"{rr['H2_rule_effect']:+.5f}")
    note("Figure3", "c adjusted", "overlap-adjusted association",
         "rule_replication/secondary_overlap.json", "overlap_adjusted_association",
         f"n_used {so['n_used']}", "bootstrap ci",
         f"{so['overlap_adjusted_association']:+.6f}")
    note("Figure3", "c SMD", "worst |SMD| unweighted / matched / weighted",
         "secondary_overlap.json + matching_failed.json",
         "unweighted_smd, worst_abs_smd, worst_abs_weighted_smd", "-", "-",
         f"{worst_un:.4f} / {mf['worst_abs_smd']:.4f} / {so['worst_abs_weighted_smd']:.6f}")

    save(fig, "figure3_draft")


if __name__ == "__main__":
    print("rendering figure 1 ...")
    figure1()
    print("rendering figure 2 (final = Option B; A kept only for comparison) ...")
    lim = figure2("B")
    figure2("A")
    print(f"  symmetric colour limit: +/-{lim:.0f} pp")
    print("rendering figure 3 ...")
    figure3()
    spec = pathlib.Path("results/paper_rendering_verified/spec")
    spec.mkdir(parents=True, exist_ok=True)
    with open(spec / "source_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(SRC_ROWS[0]))
        w.writeheader()
        w.writerows(SRC_ROWS)
    print(f"source_map.csv: {len(SRC_ROWS)} displayed quantities mapped")
