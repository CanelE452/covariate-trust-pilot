"""Poster figure 1, and the poster-format re-render of figures 4 and 5.

Figure 1 draws the Stage 1 factorial: per-cell G on the left, the factor
contrasts on the right.  The requested left panel was a per-series box plot,
which this repository cannot draw -- the recovered Stage 1 artifacts are
cell-level (eighty series collapsed into one RMSE per cell per model), so no
per-series improvement column exists.  The paired-series bootstrap interval is
drawn instead: it is the uncertainty the source study actually reported, and
it is the quantity the contrasts on the right are built from.

Cell point estimates are recomputed here from the cell RMSEs rather than read
off the ledger, and checked against the ledger table, so the panel cannot
drift from the frozen claim.  The intervals themselves are only available in
the ledger and are transcribed.

Figures 4 and 5 are redrawn from the stored poster_stage2 artifacts.  Nothing
is refitted: the Spearman estimates, their bootstrap intervals and the support
diagnostic all come off disk, and only the canvas size and the axis line width
change.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import poster_concept, poster_stage2, screen

OUT = screen.OUT / "poster_stage2"
VERIFIED = screen.OUT.parent / "synthetic_source_verification"
CELLS = VERIFIED / "stage1_verified_cells.csv"
CONTRASTS = VERIFIED / "stage1_verified_contrasts.csv"

SERIES = VERIFIED / "stage1_verified_series.csv"

MODEL = "M1_HURDLE_MEAN"
POINT = "M0_PARAMETER_MATCHED_POINT"

#: Paired-series bootstrap intervals for G, in percentage points, from
#: results/paper_synthesis_verified/claim_ledger_frozen.md section 2.  The
#: point estimates in that table are reproduced from the cell RMSEs below.
CELL_CI_PP = {"C01": (6.36, 10.57), "C02": (25.46, 30.27),
              "C03": (17.04, 20.50), "C04": (7.14, 9.82),
              "C05": (2.18, 5.00), "C06": (23.78, 30.18),
              "C07": (8.37, 13.78), "C08": (-6.79, 0.23)}
LEDGER_G_PP = {"C01": 8.47, "C02": 27.81, "C03": 18.76, "C04": 8.55,
               "C05": 3.57, "C06": 26.87, "C07": 11.10, "C08": -3.01}

#: Four of the seven contrasts: the three main effects and the interaction the
#: poster text discusses.  The remaining three are not on the poster.
CONTRAST_LABEL = {"interval_dependence": "Interval dep.",
                  "magnitude_dependence": "Magnitude dep.",
                  "sparsity": "Sparsity",
                  "interval_x_magnitude": "Interval × Magnitude"}

HURDLE_COLOUR = "#1f6fb4"
POINT_COLOUR = "#d1642a"

FONT = {"font.size": 22, "axes.labelsize": 24, "axes.titlesize": 24,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
        "axes.linewidth": 2.0, "lines.linewidth": 2.5}


def stage1_cells() -> pd.DataFrame:
    """Per-cell G, recomputed, with the ledger interval attached."""
    raw = pd.read_csv(CELLS)
    wide = raw[raw["model"].isin((MODEL, POINT))].pivot_table(
        index=["cell", "d", "interval", "magnitude"], columns="model",
        values="rmse_mean_truth").reset_index()
    wide["G_pp"] = 100.0 * (1.0 - wide[MODEL] / wide[POINT])

    drift = {row.cell: round(row.G_pp - LEDGER_G_PP[row.cell], 2)
             for row in wide.itertuples() if
             abs(row.G_pp - LEDGER_G_PP[row.cell]) > 0.01}
    if drift:
        raise ValueError(f"recomputed G disagrees with the frozen ledger: {drift}")

    wide["ci_low"] = wide["cell"].map(lambda c: CELL_CI_PP[c][0])
    wide["ci_high"] = wide["cell"].map(lambda c: CELL_CI_PP[c][1])
    wide["label"] = [
        f"{row.cell}\n{'A' if 'I1' in row.interval else 'I'} / "
        f"{'A' if 'S1' in row.magnitude else 'I'}"
        for row in wide.itertuples()]
    return wide.sort_values("cell").reset_index(drop=True)


def stage1_contrasts() -> pd.DataFrame:
    raw = pd.read_csv(CONTRASTS)
    block = raw[(raw["model"] == MODEL)
                & raw["contrast"].isin(CONTRAST_LABEL)].copy()
    block["label"] = block["contrast"].map(CONTRAST_LABEL)
    order = list(CONTRAST_LABEL)
    block["rank"] = block["contrast"].map(order.index)
    return block.sort_values("rank").reset_index(drop=True)


def _panel_cells(ax, cells: pd.DataFrame) -> None:
    x = np.arange(len(cells))
    colours = [HURDLE_COLOUR if g > 0 else POINT_COLOUR for g in cells["G_pp"]]
    ax.axhline(0.0, color="0.30", linewidth=2.0, zorder=2)
    ax.axvspan(3.5, len(cells) - 0.4, color="0.93", zorder=0)
    for xi, row, colour in zip(x, cells.itertuples(), colours):
        ax.errorbar(xi, row.G_pp,
                    yerr=[[row.G_pp - row.ci_low], [row.ci_high - row.G_pp]],
                    fmt="o", markersize=14, capsize=9, capthick=2.5,
                    color=colour, elinewidth=2.5, zorder=4)
    low, high = cells["ci_low"].min(), cells["ci_high"].max()
    ax.set_ylim(low - 3.0, high + 6.0)
    for xi, text in ((1.5, "d = 4"), (5.5, "d = 8")):
        ax.text(xi, high + 4.4, text, ha="center", va="center", fontsize=21,
                color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(cells["label"])
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylabel("G  (pp)")
    ax.set_xlabel("interval / magnitude\nA = alternating,   I = independent",
                  fontsize=21)
    ax.grid(axis="y", color="0.88", linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _panel_contrasts(ax, contrasts: pd.DataFrame) -> None:
    y = np.arange(len(contrasts))
    for yi, row in zip(y, contrasts.itertuples()):
        colour = HURDLE_COLOUR if row.effect_pp > 0 else POINT_COLOUR
        ax.barh(yi, row.effect_pp, height=0.62, color=colour, zorder=3)
        ax.errorbar(row.effect_pp, yi,
                    xerr=[[row.effect_pp - row.ci_low],
                          [row.ci_high - row.effect_pp]],
                    fmt="none", ecolor="0.20", elinewidth=2.5, capsize=9,
                    capthick=2.5, zorder=4)
        offset = 1.1 if row.effect_pp > 0 else -1.7
        ax.text(row.ci_high + offset if row.effect_pp > 0
                else row.ci_low + offset, yi, f"{row.effect_pp:+.2f}",
                ha="left" if row.effect_pp > 0 else "right", va="center",
                fontsize=21)
    ax.axvline(0.0, color="0.30", linewidth=2.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(contrasts["label"])
    ax.set_ylim(len(contrasts) - 0.45, -0.55)
    ax.set_xlim(-32.0, 15.5)
    ax.set_xlabel("effect on G  (pp, 95% CI)")
    ax.grid(axis="x", color="0.88", linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure1() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cells, contrasts = stage1_cells(), stage1_contrasts()

    with plt.rc_context(FONT):
        #: constrained layout, and no tight bbox on save, so the file is
        #: exactly the slot the poster reserves.  A tight bbox crops the
        #: canvas to the artists and the aspect ratio stops matching.
        fig, axes = plt.subplots(1, 2, figsize=(16.84, 5.10),
                                 layout="constrained",
                                 gridspec_kw={"width_ratios": [6, 4]})
        _panel_cells(axes[0], cells)
        _panel_contrasts(axes[1], contrasts)
        for ax, tag in zip(axes, ("(a)", "(b)")):
            ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=24, fontweight="bold")
        fig.savefig(OUT / "fig1_synthetic_results.png", dpi=220)
        plt.close(fig)

    return {"figure": "fig1_synthetic_results.png",
            "panel_a": {"source": str(CELLS),
                        "intervals": "claim_ledger_frozen.md section 2",
                        "note": "per-series improvements do not exist in the "
                                "recovered artifacts; the box plot the request "
                                "asked for is replaced by the paired-series "
                                "bootstrap interval",
                        "cells": cells[["cell", "d", "G_pp", "ci_low",
                                        "ci_high"]].to_dict("records")},
            "panel_b": {"source": str(CONTRASTS),
                        "contrasts": contrasts[["contrast", "effect_pp",
                                                "ci_low", "ci_high",
                                                "excludes_zero"]]
                        .to_dict("records")}}


def stage1_series_gain() -> dict:
    """Per-series gain per cell, under the source study's own definition.

    `visualize_predictions._series_gain_table` computes 1 - RMSE_M1 / RMSE_M0
    series by series, sorted on the series index so the two models line up.
    That is reproduced here rather than reinvented.  Note this is NOT the cell
    aggregate: the aggregate is a ratio of cell RMSEs, so it is not the mean of
    these per-series ratios, which is exactly what the spread panel shows.
    """
    raw = pd.read_csv(SERIES)
    out = {}
    for cell in sorted(raw["cell"].unique()):
        block = raw[raw["cell"] == cell]
        m0 = block[block["model"] == POINT].sort_values("series")
        m1 = block[block["model"] == MODEL].sort_values("series")
        if len(m0) != len(m1) or len(m0) == 0:
            raise ValueError(f"{cell}: {len(m0)} point vs {len(m1)} hurdle series")
        out[cell] = 100.0 * (1.0 - m1["rmse_mean_truth"].to_numpy()
                             / m0["rmse_mean_truth"].to_numpy())
    return out


def _panel_cells_box(ax, cells: pd.DataFrame, gains: dict) -> None:
    order = list(cells["cell"])
    data = [gains[c] for c in order]
    x = np.arange(len(order))
    ax.axhline(0.0, color="0.30", linewidth=2.0, zorder=2)
    ax.axvspan(3.5, len(order) - 0.4, color="0.93", zorder=0)

    parts = ax.boxplot(data, positions=x, widths=0.62, showfliers=False,
                       patch_artist=True, zorder=3)
    for cell, box in zip(order, parts["boxes"]):
        point_wins = cells.loc[cells["cell"] == cell, "G_pp"].iloc[0] < 0
        box.set_facecolor("#f6d9c8" if point_wins else "#cfe0f0")
        box.set_edgecolor(POINT_COLOUR if point_wins else HURDLE_COLOUR)
        box.set_linewidth(2.2)
    for key, width in (("whiskers", 2.0), ("caps", 2.0), ("medians", 2.6)):
        for artist in parts[key]:
            artist.set_color("0.25")
            artist.set_linewidth(width)

    ax.scatter(x, cells["G_pp"], marker="D", s=70, color="#b3202c", zorder=5,
               label="cell aggregate (reported)")
    ax.set_ylim(-48.0, 74.0)
    ax.set_xticks(x)
    ax.set_xticklabels(cells["label"])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("per-series G  (pp)")
    ax.set_xlabel("interval / magnitude:   A = alternating,  I = independent",
                  fontsize=21)
    for xi, text in ((1.5, "d = 4"), (5.5, "d = 8")):
        ax.text(xi, 68.5, text, ha="center", va="center", fontsize=21,
                color="0.35")
    ax.legend(loc="lower left", frameon=False, fontsize=19)
    ax.grid(axis="y", color="0.88", linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure1_v2() -> dict:
    """Figure 1 with the per-series spread the aggregate interval cannot show."""
    OUT.mkdir(parents=True, exist_ok=True)
    cells, contrasts = stage1_cells(), stage1_contrasts()
    gains = stage1_series_gain()

    counts = {c: int(v.size) for c, v in gains.items()}
    if set(counts) != set(cells["cell"]) or set(counts.values()) != {80}:
        raise ValueError(f"per-series counts are not 80 per cell: {counts}")

    with plt.rc_context(FONT):
        fig, axes = plt.subplots(1, 2, figsize=(16.84, 4.70),
                                 layout="constrained",
                                 gridspec_kw={"width_ratios": [6, 4]})
        _panel_cells_box(axes[0], cells, gains)
        _panel_contrasts(axes[1], contrasts)
        for ax, tag in zip(axes, ("(a)", "(b)")):
            ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=24, fontweight="bold")
        fig.savefig(OUT / "fig1_synthetic_results_v2.png", dpi=220)
        plt.close(fig)

    spread = {c: {"n": counts[c], "sd_pp": float(np.std(v, ddof=1)),
                  "median_pp": float(np.median(v)),
                  "aggregate_pp": float(cells.loc[cells["cell"] == c,
                                                  "G_pp"].iloc[0])}
              for c, v in gains.items()}
    return {"figure": "fig1_synthetic_results_v2.png",
            "panel_a": {"source": str(SERIES),
                        "definition": "per series, 100 * (1 - RMSE_M1 / RMSE_M0)",
                        "showfliers": False, "spread": spread}}


#: The poster body now says D_gap, iid-null and "strong-dependence reference",
#: so the figure has to say the same words.  |rho_I| is deliberately absent
#: from the real-data axis: rho_I is the synthetic latent-state parameter and
#: D_gap is what is measured on real series.
V2_XLABEL = ("$D_{gap}$ = |ACF$_1$(inter-arrival gaps)|\n"
             "(operational proxy for interval dependence)")
V2_PANEL = {"m5": "M5 (n = 1,200)", "favorita": "Favorita (n = 1,195)"}
V2_REFERENCE = 0.80


def figure5_v2(pool: pd.DataFrame, support: dict) -> dict:
    """Figure 5 relabelled for the poster's vocabulary. Bins and counts unchanged."""
    bins = np.linspace(0, 1, 51)
    heights = {}
    with plt.rc_context(FONT):
        fig, axes = plt.subplots(1, 2, figsize=(16.84, 5.55), sharey=True,
                                 layout="constrained")
        for ax, name in zip(axes, poster_stage2.DATASETS):
            block = pool[(pool["dataset"] == name) & pool["eligible"]]
            x = block[poster_stage2.X_COLUMN].to_numpy(float)
            x = x[np.isfinite(x)]
            counts, _, _ = ax.hist(x, bins=bins, color="0.55",
                                   edgecolor="white", linewidth=0.6)
            heights[name] = counts.astype(int).tolist()

            floor = support[name]["noise_floor"][
                "expected_median_abs_rho_if_independent"]
            ax.axvline(floor, color="#2e7d32", linewidth=2.5,
                       label=f"iid-null median ({floor:.3f})")
            ax.axvline(V2_REFERENCE, color="#b3202c", linewidth=2.5,
                       linestyle="--")
            n_at = int((x >= V2_REFERENCE).sum())
            ax.annotate(f"synthetic strong-dependence\nreference ≈ 0.80\n"
                        f"{n_at} series in this range",
                        xy=(V2_REFERENCE, 0.0), xytext=(0.10, 0.50),
                        textcoords="axes fraction", fontsize=19,
                        color="#b3202c", ha="left")
            ax.set_xlim(0, 1)
            ax.set_title(V2_PANEL[name])
            ax.legend(loc="upper right", frameon=False, fontsize=19)
        axes[0].set_ylabel("series")
        fig.supxlabel(V2_XLABEL, fontsize=22)

        #: Read the retired wording back off the drawn artists rather than
        #: trusting that it was edited out of the source.
        drawn = " ".join(t.get_text()
                         for t in fig.findobj(matplotlib.text.Text))
        left = [bad for bad in ("noise floor", "synthetic contrast",
                                "occurrence dependence", "lag-1 autocorr",
                                r"\rho_I", "ρI", "|ρ") if bad in drawn]
        if left:
            raise ValueError(f"retired wording still drawn in fig5 v2: {left}")

        fig.savefig(OUT / "fig5_support_vs_synthetic_v2.png", dpi=220)
        plt.close(fig)

    return {"figure": "fig5_support_vs_synthetic_v2.png",
            "retired_wording_absent": True,
            "n_bins": len(bins) - 1, "bin_heights": heights,
            "reference_line": V2_REFERENCE,
            "n_at_or_above_reference": {
                name: int((pool[(pool["dataset"] == name) & pool["eligible"]]
                           [poster_stage2.X_COLUMN].to_numpy(float)
                           >= V2_REFERENCE).sum())
                for name in poster_stage2.DATASETS}}


# ------------------------------------------------- Stage 2 rho sweep heatmap ----

STAGE2_CELLS = VERIFIED / "stage2_verified_cells.csv"
HEAT_MODEL = "HURDLE_MEAN"
RHO_I = (-0.8, 0.0, 0.8)
#: Rows run top to bottom, so rho_M descends.
RHO_M = (0.8, 0.0, -0.8)

#: Transcribed from the request, and compared against the stored cells before
#: anything is drawn. Keyed (d, rho_M) -> the three rho_I values in RHO_I order.
EXPECTED_DELTA = {
    (4, 0.8): (0.164, 0.085, 0.143),
    (4, 0.0): (0.227, 0.086, 0.199),
    (4, -0.8): (0.142, 0.180, 0.351),
    (8, 0.8): (0.137, -0.174, 0.176),
    (8, 0.0): (0.135, 0.023, 0.311),
    (8, -0.8): (0.088, 0.099, 0.352),
}
#: The one cell the poster points at. Highlighted, not interpreted: this figure
#: shows conditional RMSE only and measures nothing about how p-hat gates.
CANDIDATE = (8, 0.0, 0.8)


def stage2_grid() -> dict:
    """The 18 Hurdle-Mean cells as two 3x3 grids, checked against the request.

    Stage 2's delta is RMSE_point - RMSE_hurdle (claim_ledger_frozen.md F2),
    which is the orientation the colourbar label states. Stage 1's absolute
    delta has the opposite sign and is never mixed in here.
    """
    raw = pd.read_csv(STAGE2_CELLS)
    block = raw[raw["model"] == HEAT_MODEL]
    grids, drift = {}, []
    for d in (4, 8):
        grid = np.full((len(RHO_M), len(RHO_I)), np.nan)
        for r, rm in enumerate(RHO_M):
            for c, ri in enumerate(RHO_I):
                hit = block[(block["d"] == d)
                            & np.isclose(block["rho_interval"], ri)
                            & np.isclose(block["rho_magnitude"], rm)]
                if len(hit) != 1:
                    raise ValueError(f"d={d} rho_I={ri} rho_M={rm}: {len(hit)} rows")
                value = float(hit["delta"].iloc[0])
                grid[r, c] = value
                want = EXPECTED_DELTA[(d, rm)][c]
                if abs(round(value, 3) - want) > 1e-9:
                    drift.append((d, ri, rm, round(value, 3), want))
        grids[d] = grid
    if drift:
        raise ValueError(f"stored delta disagrees with the requested values: {drift}")
    return grids


#: The full note. The middle two lines are dropped in the poster build: the
#: sign convention is already in the colourbar label, and the highlighted cell
#: is named in the poster body, so the outline alone carries it here.
HEAT_NOTE_FULL = ("Absolute RMSE difference, not percentage gain"
                  "        positive: Hurdle better  ·  negative: Point better\n"
                  "bold outline = Point-favorable candidate "
                  r"($d$ = 8, $\rho_I$ = 0, $\rho_M$ = +0.8)" "\n"
                  "Cell means are exploratory; cell-level CIs are not "
                  "multiplicity-adjusted")
HEAT_NOTE_TRIMMED = ("Absolute RMSE difference, not percentage gain\n"
                     "Cell means are exploratory; cell-level CIs are not "
                     "multiplicity-adjusted")


def figure_markov_2panel(size: tuple[float, float] = (16.84, 6.05),
                         filename: str = "fig_markov_2panel.png",
                         note: str = HEAT_NOTE_FULL) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    grids = stage2_grid()
    limit = float(max(abs(g).max() for g in grids.values()))

    with plt.rc_context(FONT):
        fig, axes = plt.subplots(1, 2, figsize=size, layout="constrained")
        for ax, d in zip(axes, (4, 8)):
            grid = grids[d]
            #: aspect="auto" so the two grids fill the poster slot; square
            #: cells would leave most of the 16.84in width empty.
            mesh = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit,
                             aspect="auto")
            for r in range(len(RHO_M)):
                for c in range(len(RHO_I)):
                    value = grid[r, c]
                    shade = "white" if abs(value) > 0.62 * limit else "0.10"
                    ax.text(c, r, f"{value:+.3f}", ha="center", va="center",
                            fontsize=23, color=shade, fontweight="bold")
            if d == CANDIDATE[0]:
                c = RHO_I.index(CANDIDATE[1])
                r = RHO_M.index(CANDIDATE[2])
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=5.0,
                                       zorder=5))
            ax.set_xticks(range(len(RHO_I)), [f"{v:+.1f}" for v in RHO_I])
            ax.set_yticks(range(len(RHO_M)), [f"{v:+.1f}" for v in RHO_M])
            ax.set_xlabel(r"$\rho_I$  (interval dependence)")
            ax.set_title(f"Hurdle-Mean    d = {d}")
            ax.tick_params(length=0)
        axes[0].set_ylabel(r"$\rho_M$  (magnitude dependence)")
        axes[1].set_yticklabels([])

        bar = fig.colorbar(mesh, ax=axes, fraction=0.038, pad=0.02)
        bar.set_label(r"$\Delta$ = RMSE$_{Point}$ - RMSE$_{Hurdle}$",
                      fontsize=21)
        bar.ax.tick_params(labelsize=19)

        fig.supxlabel(note, fontsize=20)

        drawn = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
        required = ["Absolute RMSE difference", "exploratory"]
        missing = [need for need in required if need not in drawn]
        if missing:
            raise ValueError(f"required wording absent from heatmap: {missing}")

        fig.savefig(OUT / filename, dpi=220)
        plt.close(fig)

    return {"figure": filename, "model": HEAT_MODEL,
            "note_lines": note.count("\n") + 1,
            "dropped_wording": [bad for bad in ("positive: Hurdle better",
                                                "bold outline")
                                if bad not in drawn],
            "source": str(STAGE2_CELLS),
            "delta_definition": "RMSE_point - RMSE_hurdle",
            "colour_limit": limit, "cells_checked": 18,
            "ztnb_panels_excluded": True,
            "grids": {str(d): g.tolist() for d, g in grids.items()}}


# ------------------------------------------- Stage 2 contrast forest plot ----

STAGE2_CONTRASTS = VERIFIED / "stage2_verified_contrasts.csv"

#: Each row is a contrast between conditions, NOT an absolute delta. The
#: internal names are replaced by the conditions they compare, because
#: "C_sign" tells a reader nothing and its negative value at d = 4 is easily
#: misread as "Point wins" when it means "alternating beat persistent".
#: "independent" is the interval process, never "iid" -- the demand series as
#: a whole is not iid.
FOREST_ROWS = (
    ("C_neg", "POOLED", "Alternating − independent"),
    ("C_pos", "POOLED", "Persistent − independent"),
    ("C_pred", "POOLED", "Any dependence − independent"),
    ("C_sign", "4", "Persistent − alternating,  d = 4"),
    ("C_sign", "8", "Persistent − alternating,  d = 8"),
)
FOREST_MODELS = (("HURDLE_MEAN", "Hurdle-Mean", "#0072B2", "o"),
                 ("HURDLE_ZTNB", "Hurdle-ZTNB", "#E69F00", "s"))

#: (contrast, d, model) -> estimate, ci_low, ci_high, at the four decimals the
#: request states. Three decimals would put 0.2405 on a rounding boundary and
#: fail a comparison that is actually exact; the labels still print three.
EXPECTED_FOREST = {
    ("C_neg", "POOLED", "HURDLE_MEAN"): (0.1266, 0.0992, 0.1541),
    ("C_neg", "POOLED", "HURDLE_ZTNB"): (0.1125, 0.0799, 0.1456),
    ("C_pos", "POOLED", "HURDLE_MEAN"): (0.2008, 0.1689, 0.2319),
    ("C_pos", "POOLED", "HURDLE_ZTNB"): (0.1762, 0.1386, 0.2132),
    ("C_pred", "POOLED", "HURDLE_MEAN"): (0.1637, 0.1381, 0.1883),
    ("C_pred", "POOLED", "HURDLE_ZTNB"): (0.1444, 0.1148, 0.1750),
    ("C_sign", "4", "HURDLE_MEAN"): (-0.0275, -0.0665, 0.0151),
    ("C_sign", "4", "HURDLE_ZTNB"): (-0.0572, -0.1035, -0.0073),
    ("C_sign", "8", "HURDLE_MEAN"): (0.1759, 0.1265, 0.2238),
    ("C_sign", "8", "HURDLE_ZTNB"): (0.1845, 0.1273, 0.2405),
}
FOREST_TOLERANCE = 5e-5

FOREST_XLABEL = (r"Contrast in $\Delta$RMSE" "        "
                 r"$\Delta$RMSE = RMSE$_{Point}$ - RMSE$_{Hurdle}$" "\n"
                 "Positive values mean the first-named condition increases "
                 "the Hurdle advantage relative to the reference.")
FOREST_NOTE = (
    "95% CIs from 2,000 paired series-cluster bootstrap draws.    "
    "Open markers denote CIs crossing zero.\n"
    "The persistence-alternation contrast is reported separately by sparsity "
    "because its direction differs between d = 4 and d = 8.\n"
    "Raw RMSE units; not percentage points.")
#: The open-marker sentence is dropped here: the legend row above the plot
#: already states it, so keeping both spends a line on nothing.
FOREST_NOTE_TRIMMED = (
    "95% CIs from 2,000 paired series-cluster bootstrap draws.\n"
    "The persistence-alternation contrast is reported separately by sparsity "
    "because its direction differs between d = 4 and d = 8.\n"
    "Raw RMSE units; not percentage points.")


def forest_rows() -> list[dict]:
    """The ten points, checked against the requested values before drawing."""
    raw = pd.read_csv(STAGE2_CONTRASTS, dtype={"d": str})
    out, drift = [], []
    for contrast, d, label in FOREST_ROWS:
        for model, model_label, colour, marker in FOREST_MODELS:
            hit = raw[(raw["contrast"] == contrast) & (raw["d"] == d)
                      & (raw["model"] == model)]
            if len(hit) != 1:
                raise ValueError(f"{contrast}/{d}/{model}: {len(hit)} rows")
            row = hit.iloc[0]
            got = (row["value"], row["ci_low"], row["ci_high"])
            want = EXPECTED_FOREST[(contrast, d, model)]
            if any(abs(g - w) > FOREST_TOLERANCE for g, w in zip(got, want)):
                drift.append((contrast, d, model,
                              tuple(round(g, 4) for g in got), want))
            out.append({"label": label, "model": model_label, "colour": colour,
                        "marker": marker, "estimate": float(row["value"]),
                        "ci_low": float(row["ci_low"]),
                        "ci_high": float(row["ci_high"]),
                        "excludes_zero": bool(row["excludes_zero"]),
                        "ci_source": row["ci_source"]})
    if drift:
        raise ValueError(f"stored contrasts disagree with the request: {drift}")
    return out


def figure_markov_forest_v2(size: tuple[float, float] = (16.84, 4.80),
                            filename: str = "fig_markov_forest_v2.png",
                            note: str = FOREST_NOTE) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    points = forest_rows()
    labels = [label for _, _, label in FOREST_ROWS]
    open_markers = [p for p in points if not p["excludes_zero"]]
    if len(open_markers) != 1:
        raise ValueError(f"expected exactly one CI crossing zero, got "
                         f"{[(p['label'], p['model']) for p in open_markers]}")

    with plt.rc_context(FONT):
        fig, (ax, side) = plt.subplots(
            1, 2, figsize=size, layout="constrained",
            gridspec_kw={"width_ratios": [5.2, 1]})
        ax.axvline(0.0, color="0.35", linewidth=2.0, linestyle="--", zorder=1)
        #: Wide enough that the two rows of numbers on the right do not touch.
        offset = {"Hurdle-Mean": -0.23, "Hurdle-ZTNB": 0.23}
        for point in points:
            y = labels.index(point["label"]) + offset[point["model"]]
            filled = point["excludes_zero"]
            ax.errorbar(point["estimate"], y,
                        xerr=[[point["estimate"] - point["ci_low"]],
                              [point["ci_high"] - point["estimate"]]],
                        fmt=point["marker"], markersize=15,
                        markerfacecolor=point["colour"] if filled else "white",
                        markeredgecolor=point["colour"], markeredgewidth=3.0,
                        ecolor=point["colour"], elinewidth=2.5, capsize=7,
                        capthick=2.5, zorder=3)
            side.text(0.0, y, f"{point['estimate']:+.3f}"
                      f"   [{point['ci_low']:.3f}, {point['ci_high']:.3f}]",
                      ha="left", va="center", fontsize=16,
                      color=point["colour"])

        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlim(-0.15, 0.30)
        ax.set_xlabel(FOREST_XLABEL, fontsize=20)
        ax.set_title(r"$\rho_M$ = 0     ·     top three contrasts averaged "
                     r"over $d \in \{4, 8\}$", fontsize=22, pad=12)
        ax.grid(axis="x", color="0.88", linewidth=1.0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        #: One legend row under the title: model identity and the open-marker
        #: convention read together, so a grey marker never has to stand in
        #: for "not significant" at the cost of losing which model it is.
        handles = [Line2D([], [], color=colour, marker=marker, linestyle="",
                          markersize=15, markerfacecolor=colour,
                          markeredgecolor=colour, markeredgewidth=3.0)
                   for _, _, colour, marker in FOREST_MODELS]
        handles.append(Line2D([], [], color="0.35", marker="o", linestyle="",
                              markersize=15, markerfacecolor="white",
                              markeredgecolor="0.35", markeredgewidth=3.0))
        #: On the figure, not the axes: this legend row is wider than the plot
        #: box, and anchoring it to the axes makes the layout engine shrink
        #: the plot to fit it.
        fig.legend(handles,
                   [label for _, label, _, _ in FOREST_MODELS]
                   + ["Open markers indicate 95% CIs crossing zero"],
                   loc="outside upper center", ncol=3, frameon=False,
                   fontsize=19, handletextpad=0.4, columnspacing=2.2)

        side.set_ylim(ax.get_ylim())
        side.set_xlim(0, 1)
        side.axis("off")

        fig.supxlabel(note, fontsize=18)

        drawn = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
        left = [bad for bad in ("C_neg", "C_pos", "C_pred", "C_sign", "iid",
                                "favours") if bad in drawn]
        if left:
            raise ValueError(f"retired wording still drawn: {left}")
        required = ["Positive values mean", "2,000 paired",
                    "reported separately by sparsity", "Raw RMSE units",
                    "Open markers indicate"]
        missing = [need for need in required if need not in drawn]
        if missing:
            raise ValueError(f"required wording absent from forest: {missing}")

        fig.savefig(OUT / filename, dpi=220)
        #: Measured after layout, because the row labels sit outside the axes
        #: and eat into the width the plot actually gets.
        fig.canvas.draw()
        plot_share = ax.get_position().width
        widest = max(
            side.transData.transform((0, 0))[0]
            + t.get_window_extent(fig.canvas.get_renderer()).width
            for t in side.texts) / (fig.get_size_inches()[0] * fig.dpi)
        plt.close(fig)

    return {"figure": filename,
            "source": str(STAGE2_CONTRASTS),
            "note_lines": note.count("\n") + 1,
            "dropped_wording": ["Open markers denote"]
            if "Open markers denote" not in drawn else [],
            "values": [(p["label"], p["model"], p["estimate"], p["ci_low"],
                        p["ci_high"]) for p in points],
            "plot_share_of_width": round(float(plot_share), 4),
            "value_column_right_edge": round(float(widest), 4),
            "points": len(points), "open_markers": 1,
            "open_marker_row": (open_markers[0]["label"],
                                open_markers[0]["model"]),
            "retired_wording_absent": True,
            "pooled_ci_source": "recomputed_pooled_bootstrap"}


#: The small slot next to the forest plot needs its own type sizes; the poster
#: FONT block is set for figures three times this wide.
SMALL_FONT = {"font.size": 20, "axes.labelsize": 22, "axes.titlesize": 22,
              "xtick.labelsize": 19, "ytick.labelsize": 19,
              "axes.linewidth": 2.0}


def figure_heat_d8() -> dict:
    """The d = 8 face alone, so the one Point-favourable cell has a location.

    Shares stage2_grid()'s check of all eighteen cells, and the colour limit is
    still taken over both d: the largest absolute delta in the sweep sits at
    d = 8, so this panel's scale is identical to the two-panel figure's.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    grids = stage2_grid()
    limit = float(max(abs(g).max() for g in grids.values()))
    grid = grids[8]

    with plt.rc_context(SMALL_FONT):
        fig, ax = plt.subplots(figsize=(6.25, 3.60), layout="constrained")
        mesh = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit,
                         aspect="auto")
        for r in range(len(RHO_M)):
            for c in range(len(RHO_I)):
                value = grid[r, c]
                shade = "white" if abs(value) > 0.62 * limit else "0.10"
                ax.text(c, r, f"{value:+.3f}", ha="center", va="center",
                        fontsize=22, color=shade, fontweight="bold")

        c = RHO_I.index(CANDIDATE[1])
        r = RHO_M.index(CANDIDATE[2])
        #: Inset slightly so the white rule between cells stays visible under
        #: the highlight instead of being painted over.
        ax.add_patch(Rectangle((c - 0.46, r - 0.46), 0.92, 0.92, fill=False,
                               edgecolor="black", linewidth=4.5, zorder=6))
        highlights = [p for p in ax.patches if isinstance(p, Rectangle)
                      and not p.get_fill()]
        if len(highlights) != 1:
            raise ValueError(f"expected one highlighted cell, got {len(highlights)}")

        ax.set_xticks(range(len(RHO_I)), [f"{v:+.1f}" for v in RHO_I])
        ax.set_yticks(range(len(RHO_M)), [f"{v:+.1f}" for v in RHO_M])
        #: At this width a 22pt number nearly spans its cell, so without a
        #: rule between cells the three values in a row read as one string.
        ax.set_xticks(np.arange(-0.5, len(RHO_I)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(RHO_M)), minor=True)
        ax.grid(which="minor", color="white", linewidth=3.0)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel(r"$\rho_I$  (interval dependence)")
        #: Two lines: rotated upright, this label runs along the 3.60in height,
        #: and one line of it does not fit inside the axes box.
        ax.set_ylabel(r"$\rho_M$  (magnitude" "\n" "dependence)", fontsize=20)
        ax.set_title("Hurdle-Mean,  d = 8")
        ax.tick_params(length=0)

        bar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
        bar.set_label(r"$\Delta$ = RMSE$_{Point}$" "\n" r"$-$ RMSE$_{Hurdle}$",
                      fontsize=16)
        bar.ax.tick_params(labelsize=16)
        fig.savefig(OUT / "fig_heat_d8.png", dpi=220)
        plt.close(fig)

    return {"figure": "fig_heat_d8.png", "model": HEAT_MODEL, "d": 8,
            "cells": 9, "colour_limit": limit,
            "colour_limit_matches_two_panel": True,
            "highlighted_cells": 1,
            "highlighted": {"rho_I": CANDIDATE[1], "rho_M": CANDIDATE[2],
                            "delta": float(grid[r, c])},
            "grid": grid.tolist()}


# ------------------------------------------------- forest, poster-scale v4 ----

#: The five rows regrouped: the shared half of each label becomes a header, so
#: the per-row text shrinks and the left margin stops eating the plot.
FOREST_GROUPS = (
    ("vs. independent baseline",
     (("C_neg", "POOLED", "Alternating"),
      ("C_pos", "POOLED", "Persistent"),
      ("C_pred", "POOLED", "Any dependence"))),
    ("Persistent vs. alternating",
     (("C_sign", "4", "d = 4"),
      ("C_sign", "8", "d = 8"))),
)
#: The axis label carries only the axis definition here. Its second line in
#: FOREST_XLABEL is the same sentence the note keeps, and drawing it twice
#: costs a row of plot height for nothing.
FOREST_XLABEL_V4 = (r"Contrast in $\Delta$RMSE" "        "
                    r"$\Delta$RMSE = RMSE$_{Point}$ - RMSE$_{Hurdle}$")
#: One line, not two: at 4.70in every text row costs the plot about 6% of its
#: height, and the height target is the binding constraint here.
FOREST_NOTE_V4 = ("Positive values mean the first-named condition increases "
                  "the Hurdle advantage relative to the reference.     "
                  "Raw RMSE units; not percentage points.")
FOREST_TOP_V4 = (r"Hurdle-Mean ●  ·  Hurdle-ZTNB ■  ·  open = CI crosses zero"
                 r"      $\rho_M$ = 0,  averaged over $d \in \{4, 8\}$")


def figure_markov_forest_v4() -> dict:
    """Same ten estimates, sized for a 1-2 m viewing distance.

    The right-hand column of interval text is gone: on a forest plot the
    whisker already IS the interval, so printing it again costs width twice
    over. Point estimates stay, next to their own whisker.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    by_key = {}
    for point in forest_rows():
        by_key.setdefault(point["label"], {})[point["model"]] = point

    #: Row order and y positions, with a gap between the two groups.
    rows, headers, y = [], [], 0.0
    for header, members in FOREST_GROUPS:
        headers.append((header, y - 0.62))
        for contrast, d, short in members:
            full = next(label for c, dd, label in FOREST_ROWS
                        if c == contrast and dd == d)
            rows.append({"y": y, "short": short, "points": by_key[full]})
            y += 1.0
        y += 0.75

    open_markers = [p for row in rows for p in row["points"].values()
                    if not p["excludes_zero"]]
    if len(open_markers) != 1:
        raise ValueError(f"expected one CI crossing zero, got {len(open_markers)}")

    with plt.rc_context(FONT):
        fig, ax = plt.subplots(figsize=(16.84, 4.70), layout="constrained")
        ax.axvline(0.0, color="0.35", linewidth=2.5, linestyle="--", zorder=1)
        offset = {"Hurdle-Mean": -0.21, "Hurdle-ZTNB": 0.21}
        for row in rows:
            for name, point in row["points"].items():
                yy = row["y"] + offset[name]
                filled = point["excludes_zero"]
                ax.errorbar(point["estimate"], yy,
                            xerr=[[point["estimate"] - point["ci_low"]],
                                  [point["ci_high"] - point["estimate"]]],
                            fmt=point["marker"], markersize=27,
                            markerfacecolor=point["colour"] if filled else "white",
                            markeredgecolor=point["colour"],
                            #: An open marker has to survive being read from
                            #: two metres away, so its ring is heavier.
                            markeredgewidth=3.5 if filled else 6.0,
                            ecolor=point["colour"], elinewidth=4.0,
                            capsize=11, capthick=4.0, zorder=3)
                ax.annotate(f"{point['estimate']:+.3f}",
                            xy=(point["ci_high"], yy), xytext=(11, 0),
                            textcoords="offset points", ha="left",
                            va="center", fontsize=21, color=point["colour"],
                            fontweight="bold")

        for header, yy in headers:
            ax.text(-0.005, yy, header, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=19, color="0.35",
                    style="italic")

        ax.set_yticks([row["y"] for row in rows], [row["short"] for row in rows])
        ax.set_ylim(rows[-1]["y"] + 0.7, headers[0][1] - 0.45)
        ax.set_xlim(-0.15, 0.30)
        ax.set_xlabel(FOREST_XLABEL_V4, fontsize=20)
        ax.set_title(FOREST_TOP_V4, fontsize=20, pad=8)
        ax.grid(axis="x", color="0.88", linewidth=1.0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.supxlabel(FOREST_NOTE_V4, fontsize=16)

        drawn = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        joined = " ".join(drawn)
        brackets = [t for t in drawn if "[" in t]
        if brackets:
            raise ValueError(f"interval text still drawn: {brackets}")
        left = [bad for bad in ("2,000 paired",
                                "reported separately by sparsity",
                                "C_neg", "C_sign", "iid", "favours")
                if bad in joined]
        if left:
            raise ValueError(f"retired wording still drawn: {left}")

        fig.canvas.draw()
        box = ax.get_position()
        fig.savefig(OUT / "fig_markov_forest_v4.png", dpi=220)
        plt.close(fig)

    return {"figure": "fig_markov_forest_v4.png",
            "plot_width_share": round(float(box.width), 4),
            "plot_height_share": round(float(box.height), 4),
            "points": sum(len(r["points"]) for r in rows),
            "open_markers": len(open_markers),
            "interval_text_removed": True,
            "values": [(r["short"], n, p["estimate"])
                       for r in rows for n, p in r["points"].items()]}


def stored_pool() -> tuple[pd.DataFrame, dict]:
    """The eligible pool and the stored report, with the counts cross-checked."""
    report = json.loads((OUT / "poster_stage2.json").read_text())
    pool = pd.read_csv(OUT / "per_series_poster.csv")
    #: The stored pool is already the eligible set -- its per-dataset counts
    #: equal the support diagnostic's n -- so the flag figure5 filters on is
    #: restored rather than recomputed.
    for name in poster_stage2.DATASETS:
        stored = report["support_diagnostic"][name]["n"]
        found = int((pool["dataset"] == name).sum())
        if stored != found:
            raise ValueError(f"{name}: stored pool has {found} series, the "
                             f"support diagnostic reports {stored}")
    pool["eligible"] = True
    return pool, report


def rerender_stage2() -> dict:
    """Redraw figures 4 and 5 at poster size from the stored artifacts."""
    pool, report = stored_pool()
    poster_stage2.figure5(pool, report["support_diagnostic"])
    poster_stage2.figure4(report["fits"])
    return {"redrawn": ["fig5_support_vs_synthetic.png",
                        "fig4_spearman_forest.png", "fig4_slope_forest.png"],
            "refitted": False}


def run() -> dict:
    pool, report = stored_pool()
    v2 = figure5_v2(pool, report["support_diagnostic"])

    #: The v2 canvas changed; the histogram must not have. Recount the same
    #: bins independently of the drawing and compare.
    for name in poster_stage2.DATASETS:
        x = pool[(pool["dataset"] == name) & pool["eligible"]][
            poster_stage2.X_COLUMN].to_numpy(float)
        expected, _ = np.histogram(x[np.isfinite(x)], bins=np.linspace(0, 1, 51))
        if expected.tolist() != v2["bin_heights"][name]:
            raise ValueError(f"{name}: fig5 v2 bin heights moved")
    v2["bin_heights_match_original"] = True

    return {"figure1_v2": figure1_v2(), "figure5_v2": v2}


def run_v3() -> dict:
    """The two figures the reflowed poster still needs."""
    return {"markov_2panel": figure_markov_2panel(),
            "figureA_v2": poster_concept.figure_a(
                size=(16.84, 4.60),
                filename="figA_matched_marginals_v2.png")}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
