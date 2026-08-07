"""The three SCREEN figures. Tables and bootstrap remain the primary evidence."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import prereg, screen  # noqa: E402

OUT = screen.OUT
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#666666"
DPI = 300
AXIS = r"$\Delta$ = RMSE$_{\mathrm{Point}}$ - RMSE$_{\mathrm{Hurdle}}$  (realized $y$)"


def _load():
    frame = pd.read_csv(OUT / "per_series_metrics.csv")
    results = json.loads((OUT / "stage_a_results.json").read_text())["results"]
    thr = prereg.ELIGIBILITY["primary_threshold"]
    return frame[frame["n_positive_train"] >= thr].copy(), results


def _save(fig, stem):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=DPI, facecolor="white")
    plt.close(fig)


def figure_a(frame, results):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.20, wspace=0.08)
    for ax, (name, block) in zip(axes, frame.groupby("dataset")):
        x = block["rho_interval_abs_train"].to_numpy(float)
        y = block["delta_rmse"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=9, alpha=0.35, color=BLUE, edgecolor="none")
        order = np.argsort(x[ok])
        window = max(25, ok.sum() // 20)
        smooth = pd.Series(y[ok][order]).rolling(window, center=True).mean()
        ax.plot(x[ok][order], smooth, color=ORANGE, lw=2.4,
                label=f"rolling mean ({window} series)")
        ax.axhline(0, color="#333333", lw=1.0)
        r = results[name][str(prereg.ELIGIBILITY["primary_threshold"])]["H1"]
        ax.set_title(f"{name}   Spearman {r['spearman']:+.4f}  "
                     f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]   n = {r['n']}",
                     fontsize=11.5)
        ax.set_xlabel(r"$|\rho_{\mathrm{interval}}|$  (train only)", fontsize=11.5)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(AXIS, fontsize=11.5)
    lo, hi = np.nanpercentile(frame["delta_rmse"], [1, 99])
    axes[0].set_ylim(lo, hi)
    fig.text(0.07, 0.055,
             "H1: stronger inter-arrival dependence should move the comparison "
             "towards Hurdle (positive slope). Axis clipped to the 1st-99th "
             "percentile; the correlation uses every eligible series.",
             fontsize=9, color=GREY, va="top")
    _save(fig, "figA_H1_rho_interval_vs_delta")


def _groups(block):
    x = block["rho_interval_abs_train"].to_numpy(float)
    adi = block["ADI_train"].to_numpy(float)
    mag = block["rho_magnitude_train"].to_numpy(float)
    high = adi >= np.nanmedian(adi)
    low_occ = x <= np.nanquantile(x, 1 / 3)
    persistent = mag >= np.nanquantile(mag, 2 / 3)
    return high & low_occ & persistent, high & low_occ & ~persistent, high


def figure_b(frame, results):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.22, wspace=0.08)
    for ax, (name, block) in zip(axes, frame.groupby("dataset")):
        cand, ctrl, _ = _groups(block)
        d = block["delta_rmse"].to_numpy(float)
        parts = ax.violinplot([d[ctrl], d[cand]], positions=[0, 1],
                              showextrema=False, widths=0.75)
        for body, colour in zip(parts["bodies"], (BLUE, ORANGE)):
            body.set_facecolor(colour); body.set_alpha(0.35)
        for pos, sel, colour in ((0, ctrl, BLUE), (1, cand, ORANGE)):
            ax.scatter(np.random.default_rng(0).normal(pos, 0.045, sel.sum()),
                       d[sel], s=14, color=colour, alpha=0.8, edgecolor="none")
            ax.hlines(np.nanmean(d[sel]), pos - 0.3, pos + 0.3, color="#111111",
                      lw=2.2, zorder=5)
        ax.axhline(0, color="#333333", lw=1.0)
        h = results[name][str(prereg.ELIGIBILITY["primary_threshold"])]["H2"]
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Control\nn = {h['n_control']}",
                            f"PointCandidate\nn = {h['n_candidate']}"], fontsize=10.5)
        ax.set_title(f"{name}   mean {h['mean_control']:+.4f} vs "
                     f"{h['mean_candidate']:+.4f}   diff {h['difference']:+.4f}",
                     fontsize=11.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(AXIS, fontsize=11.5)
    fig.text(0.07, 0.085,
             "H2: both groups are HIGH_ADI and LOW_OCC_SIGNAL; PointCandidate "
             "adds persistent positive magnitude. The prediction is that "
             "PointCandidate sits lower.\nBlack bar is the group mean. Cutoffs "
             "were frozen before any prediction existed.",
             fontsize=9, color=GREY, va="top")
    _save(fig, "figB_H2_point_candidate_vs_control")


def figure_c(frame, results):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.22, wspace=0.08)
    for ax, (name, block) in zip(axes, frame.groupby("dataset")):
        _, _, high = _groups(block)
        x = block["rho_interval_abs_train"].to_numpy(float)
        d = block["delta_rmse"].to_numpy(float)
        for sel, colour, label in ((~high, BLUE, "LOW_ADI"),
                                   (high, ORANGE, "HIGH_ADI")):
            ok = sel & np.isfinite(x) & np.isfinite(d)
            order = np.argsort(x[ok])
            window = max(25, ok.sum() // 12)
            ax.scatter(x[ok], d[ok], s=7, alpha=0.20, color=colour, edgecolor="none")
            ax.plot(x[ok][order],
                    pd.Series(d[ok][order]).rolling(window, center=True).mean(),
                    color=colour, lw=2.6, label=label)
        ax.axhline(0, color="#333333", lw=1.0)
        h = results[name][str(prereg.ELIGIBILITY["primary_threshold"])]["H3"]
        ax.set_title(f"{name}   HIGH_ADI {h['corr_high_ADI']:+.4f}  vs  "
                     f"LOW_ADI {h['corr_low_ADI']:+.4f}   diff {h['difference']:+.4f}",
                     fontsize=11.5)
        ax.set_xlabel(r"$|\rho_{\mathrm{interval}}|$  (train only)", fontsize=11.5)
        ax.legend(fontsize=10, frameon=False, loc="upper left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(AXIS, fontsize=11.5)
    lo, hi = np.nanpercentile(frame["delta_rmse"], [1, 99])
    axes[0].set_ylim(lo, hi)
    fig.text(0.07, 0.085,
             "H3 predicted the HIGH_ADI curve to rise more steeply than LOW_ADI. "
             "It does not in either dataset — the observed difference is negative "
             "in both, though\nneither difference separates from zero. Axis "
             "clipped to the 1st-99th percentile.",
             fontsize=9, color=GREY, va="top")
    _save(fig, "figC_H3_sparsity_interaction")


def main() -> None:
    frame, results = _load()
    figure_a(frame, results)
    figure_b(frame, results)
    figure_c(frame, results)
    print(f"wrote three figures to {OUT.relative_to(screen.REPO)}")


if __name__ == "__main__":
    main()
