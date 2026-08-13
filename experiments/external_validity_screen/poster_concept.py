"""Poster concept figure: matched marginals, different order.

Separate from poster_stage2.py because it analyses nothing.  It draws a toy
pair of series that share every marginal summary an intermittent-demand
taxonomy looks at — ADI, CV-squared, the multiset of inter-arrival gaps, the
event sizes — and differ only in the ORDER of the gaps.  That is the claim the
synthetic study rests on, and a scatter of real data cannot make it.

The sequences are hand-chosen, not sampled: the alternating order gives
rho_I exactly -1 and the shuffled order gives rho_I exactly 0, which a random
draw would only approach.  Both numbers are recomputed here from the drawn
gaps rather than asserted, so the panel labels cannot drift from the picture.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import screen

OUT = screen.OUT / "poster_stage2"

#: Five 3s and four 5s. Alternating -> rho_I = -1; this shuffle -> rho_I = 0.
#: Found by exhausting the 126 distinct orders of that multiset.
GAPS_ALTERNATING = (3, 5, 3, 5, 3, 5, 3, 5, 3)
GAPS_INDEPENDENT = (3, 3, 3, 5, 3, 5, 5, 5, 3)
#: Identical event sizes, in identical order, so rho_M is identical too and the
#: only thing that differs between the panels is the gap order. The order is
#: also chosen so rho_M is exactly 0 in both panels, and so the sizes carry no
#: linear trend either: if the sizes had structure of their own the figure
#: would show two kinds of structure at once.
#:
#: The multiset is not arbitrary. The Stage 1 DGP alternates the positive
#: mean between lambda 5 and 15, giving a theoretical positive mean of 10 and
#: variance 34 (stage1_validity.json), i.e. CV-squared 0.34.  These ten values
#: reproduce that exactly under the same estimator the SBC rule uses
#: ((std/mean)^2, ddof=1), so the drawn series sits where the real design
#: sits rather than at an invented dispersion.
SIZES = (7, 12, 11, 5, 17, 19, 3, 4, 6, 16)
FIRST_EVENT = 2
LENGTH = 40

#: The frozen SBC quadrant cutoffs, so the panel can name its own regime
#: instead of leaving "which quadrant is this" to the reader.
SBC_ADI, SBC_CV2 = 1.32, 0.49
#: Stage 1's theoretical positive mean and variance, from stage1_validity.json.
DESIGN_CV2 = 34.0 / 10.0 ** 2

COLOUR = {"alternating": "#1f6fb4", "independent": "#d1642a"}

FONT = {"font.size": 22, "axes.labelsize": 24, "axes.titlesize": 24,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
        "axes.linewidth": 2.0, "lines.linewidth": 2.5}


def lag1(values) -> float:
    v = np.asarray(values, dtype=float)
    r = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    #: The zero cases are exact by construction but land on -1e-17, which
    #: reaches the label as "-0.00". Snap them so the sign is not read as real.
    return 0.0 if abs(r) < 1e-9 else r


def series_stats(gaps) -> dict:
    events = FIRST_EVENT + np.concatenate([[0], np.cumsum(gaps)])
    sizes = np.asarray(SIZES, dtype=float)
    if events.max() >= LENGTH:
        raise ValueError("the toy series does not fit in the drawn window")
    adi = LENGTH / len(events)
    cv2 = float((sizes.std(ddof=1) / sizes.mean()) ** 2)
    sparse, erratic = adi >= SBC_ADI, cv2 >= SBC_CV2
    regime = ("lumpy" if sparse and erratic else "intermittent" if sparse
              else "erratic" if erratic else "smooth")
    return {"events": events, "sizes": sizes,
            "ADI": adi, "CV2": cv2, "regime": regime,
            "mean_gap": float(np.mean(gaps)),
            "rho_interval": lag1(gaps),
            "rho_magnitude": lag1(sizes)}


def _panel(ax, gaps, stats, colour, label):
    events, sizes = stats["events"], stats["sizes"]
    ax.vlines(events, 0, sizes, color=colour, linewidth=6.0, zorder=3)
    ax.scatter(events, sizes, s=90, color=colour, zorder=4)
    ax.axhline(0.0, color="0.3", linewidth=2.0, zorder=2)

    top = sizes.max() * 1.62
    arrow_y, text_y = -top * 0.10, -top * 0.24
    for start, gap in zip(events[:-1], gaps):
        ax.annotate("", xy=(start, arrow_y), xytext=(start + gap, arrow_y),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=2.0))
        ax.text(start + gap / 2, text_y, str(gap), ha="center", va="center",
                fontsize=20, color="0.25")

    ax.text(0.006, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=23, color=colour, fontweight="bold")
    ax.set_xlim(-0.5, LENGTH - 0.5)
    ax.set_ylim(text_y * 1.45, top)
    ax.set_yticks([0, 10])
    ax.set_ylabel("demand")
    ax.grid(axis="x", color="0.90", linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_a(size: tuple[float, float] = (16.84, 6.00),
             filename: str = "figA_matched_marginals.png") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    alt = series_stats(GAPS_ALTERNATING)
    ind = series_stats(GAPS_INDEPENDENT)
    if sorted(GAPS_ALTERNATING) != sorted(GAPS_INDEPENDENT):
        raise ValueError("the two panels no longer share a gap multiset")
    if abs(alt["CV2"] - DESIGN_CV2) > 0.005:
        raise ValueError(f"drawn CV2 {alt['CV2']:.4f} has drifted from the "
                         f"Stage 1 design value {DESIGN_CV2:.4f}")

    with plt.rc_context(FONT):
        fig, axes = plt.subplots(2, 1, figsize=size, sharex=True,
                                 layout="constrained")
        _panel(axes[0], GAPS_ALTERNATING, alt, COLOUR["alternating"],
               f"Series A  —  alternating   " r"$\rho_I$ = "
               f"{alt['rho_interval']:+.2f}")
        _panel(axes[1], GAPS_INDEPENDENT, ind, COLOUR["independent"],
               f"Series B  —  independent   " r"$\rho_I$ = "
               f"{ind['rho_interval']:+.2f}")
        axes[1].set_xlabel("t")
        axes[1].set_xticks([0, 10, 20, 30, 40])
        #: mean gap is deliberately not printed next to ADI. ADI is
        #: length / events (40 / 10) and the mean gap averages the nine gaps
        #: between ten events, so the two cannot coincide for this sequence --
        #: matching them would need a length of 38.9. Printing both invites
        #: the reader to reconcile a difference that is pure bookkeeping.
        shared = (f"identical marginals:   ADI = {alt['ADI']:.1f}"
                  f"    CV$^2$ = {alt['CV2']:.2f}  ({alt['regime']})"
                  f"    same gap multiset    same event sizes    "
                  r"$\rho_M$ = " f"{alt['rho_magnitude']:.2f}")
        fig.supxlabel(shared, fontsize=21)
        fig.align_ylabels(axes)
        fig.savefig(OUT / filename, dpi=220)
        plt.close(fig)

    report = {"figure": filename,
              "data": "hand-chosen toy sequences, not sampled from the DGP",
              "window_length": LENGTH, "n_events": len(SIZES),
              "gaps": {"alternating": list(GAPS_ALTERNATING),
                       "independent": list(GAPS_INDEPENDENT),
                       "multiset_identical": True},
              "event_sizes": list(SIZES),
              "series_a": {k: v for k, v in alt.items()
                           if k not in ("events", "sizes")},
              "series_b": {k: v for k, v in ind.items()
                           if k not in ("events", "sizes")}}
    (OUT / filename.replace(".png", ".json")).write_text(
        json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    print(json.dumps(figure_a(), indent=2, default=str))
