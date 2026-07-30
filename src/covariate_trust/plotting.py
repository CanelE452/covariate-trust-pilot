"""Figures.

One figure per file, no subplots, default matplotlib colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .storage import atomic_savefig  # noqa: E402


def _close(fig, path: Path) -> Path:
    out = atomic_savefig(path, fig)
    plt.close(fig)
    return out


def study0_mse_curve(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in summary.groupby("predictor"):
        g = g.sort_values("lam")
        ax.plot(g["lam"], g["mse_simulated"], marker="o", label=f"{name} (simulated)")
        ax.plot(g["lam"], g["mse_analytic"], linestyle="--", label=f"{name} (analytic)")
    ax.axvline(1.0, linestyle=":", color="gray")
    ax.annotate("lambda = 1 reference", xy=(1.0, ax.get_ylim()[1]), xytext=(3, -12),
                textcoords="offset points", fontsize=8, color="gray")
    ax.set_xlabel("lambda (covariate forecast error multiplier)")
    ax.set_ylabel("MSE")
    ax.set_title("Study 0: simulated vs analytic MSE")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figure1_heatmap(cells: pd.DataFrame, horizon: int, path: Path) -> Path:
    """WQL(M3) - WQL(M1) over (nominal share, lambda) with the zero contour."""
    c = cells[cells["horizon"] == horizon]
    shares = sorted(c["nominal_covariate_share"].unique())
    lams = sorted(c["lam"].unique())
    z = np.full((len(shares), len(lams)), np.nan)
    for i, s in enumerate(shares):
        for j, l in enumerate(lams):
            sel = c[(c["nominal_covariate_share"] == s) & (c["lam"] == l)]
            if len(sel):
                # v_future = WQL(M1) - WQL(M3); the figure shows WQL(M3) - WQL(M1)
                z[i, j] = -float(sel["v_future_mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim,
                   extent=(-0.5, len(lams) - 0.5, -0.5, len(shares) - 0.5))
    if np.isfinite(z).all() and np.nanmin(z) < 0 < np.nanmax(z):
        ax.contour(np.arange(len(lams)), np.arange(len(shares)), z, levels=[0.0],
                   colors="black", linewidths=1.5)
    for i in range(len(shares)):
        for j in range(len(lams)):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:+.4f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(lams)), [f"{l:g}" for l in lams])
    ax.set_yticks(range(len(shares)), [f"{s:g}" for s in shares])
    ax.set_xlabel("lambda")
    ax.set_ylabel("nominal covariate share")
    ax.set_title(f"WQL(M3) - WQL(M1), horizon {horizon}\n(negative = forecasted future covariate helps)")
    fig.colorbar(im, ax=ax, label="WQL(M3) - WQL(M1)")
    return _close(fig, path)


def figure2_v_future(cells: pd.DataFrame, horizon: int, path: Path) -> Path:
    c = cells[cells["horizon"] == horizon]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for share, g in c.groupby("nominal_covariate_share"):
        g = g.sort_values("lam")
        ax.plot(g["lam"], g["v_future_mean"], marker="o", label=f"share = {share:g}")
        ax.fill_between(g["lam"], g["v_future_ci_low"], g["v_future_ci_high"], alpha=0.15)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(1.0, linestyle=":", color="gray")
    ax.set_xlabel("lambda")
    ax.set_ylabel("V_future = WQL(M1) - WQL(M3)")
    ax.set_title(f"Incremental value of the forecasted future covariate, horizon {horizon}\n"
                 "(shaded: 95% paired bootstrap CI; dotted line: lambda = 1 reference)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figure3_method_wql(task_metrics: pd.DataFrame, cis: dict, horizon: int, path: Path) -> Path:
    t = task_metrics[task_metrics["horizon"] == horizon]
    labels = ["M0", "M1", "M2", "M3"]
    means = [float(t[f"wql_m{i}"].mean()) for i in range(4)]
    err_low, err_high = [], []
    for i in range(4):
        key = f"h{horizon}_m{i}"
        ci = cis.get(key)
        if ci is None:
            err_low.append(0.0)
            err_high.append(0.0)
        else:
            err_low.append(max(0.0, means[i] - ci[0]))
            err_high.append(max(0.0, ci[1] - means[i]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, means, yerr=[err_low, err_high], capsize=5)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean WQL")
    ax.set_title(f"Mean WQL by method, horizon {horizon}\n"
                 "(error bars: 95% CI of the paired difference against M1)")
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figure4_harm_rate(cells: pd.DataFrame, admission_harm: pd.DataFrame | None,
                      horizon: int, path: Path) -> Path:
    c = cells[cells["horizon"] == horizon].sort_values("lam")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    agg = c.groupby("lam")["harm_rate"].mean()
    ax.plot(agg.index, agg.values, marker="o", label="M3 (always use future covariate)")
    if admission_harm is not None and len(admission_harm):
        a = admission_harm[admission_harm["horizon"] == horizon]
        for sel, g in a.groupby("selector"):
            g = g.sort_values("lam")
            ax.plot(g["lam"], g["harm_rate"], marker="s", label=f"admission {sel}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("harm rate (WQL > 1.05 x WQL(M1))")
    ax.set_title(f"Harm rate versus covariate error, horizon {horizon}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figure5_example_series(series_df: pd.DataFrame, vintage_row: dict, origin: int,
                           horizon: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    window = series_df[(series_df["t"] >= origin - 168) & (series_df["t"] < origin + horizon)]
    ax.plot(window["t"], window["y"], label="target y")
    ax.plot(window["t"], window["x"], label="covariate x (true)", alpha=0.8)
    fut_t = np.arange(origin, origin + horizon)
    ax.plot(fut_t, vintage_row["x_true"], linewidth=2.5, label="x true future")
    ax.plot(fut_t, vintage_row["x_tilde"], linewidth=2.5, linestyle="--",
            label=f"x forecast (lambda = {vintage_row['lam']:g})")
    ax.axvline(origin, color="black", linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel("standardized value")
    ax.set_title(f"Representative series (base_series_id "
                 f"{vintage_row['base_series_id']}, horizon {horizon})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figure6_share_vs_r2(meta: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(meta["nominal_covariate_share"], meta["realized_incremental_r2"], alpha=0.6, s=18)
    lo = float(meta["nominal_covariate_share"].min())
    hi = float(meta["nominal_covariate_share"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="identity")
    means = meta.groupby("nominal_covariate_share")["realized_incremental_r2"].mean()
    ax.plot(means.index, means.values, marker="o", label="cell mean")
    ax.set_xlabel("nominal covariate share r")
    ax.set_ylabel("realized incremental R^2 of x given b")
    ax.set_title("Nominal covariate share versus realized incremental R^2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)
