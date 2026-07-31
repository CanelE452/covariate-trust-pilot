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


# ============================================================================
# Follow-up studies (Study 1B and Study 2).  Additive: the figures above are
# unchanged.  One figure per file, no subplots, default matplotlib colours.
# ============================================================================

def figB1_v_future_curve(cells: pd.DataFrame, share: float, horizon: int, path: Path) -> Path:
    c = cells[(cells["nominal_covariate_share"] == share)
              & (cells["horizon"] == horizon)].sort_values("lam")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(c["lam"], c["v_future_mean"], marker="o")
    ax.fill_between(c["lam"], c["v_future_ci_low"], c["v_future_ci_high"], alpha=0.2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(1.0, linestyle=":", color="gray")
    ax.set_xlabel("lambda")
    ax.set_ylabel("V_future = WQL(M1) - WQL(M3)")
    ax.set_title(f"Study 1B: refined V_future curve, share {share:g}, horizon {horizon}\n"
                 "(shaded: 95% paired bootstrap CI; dotted: lambda = 1 reference line)")
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figB2_boundary_pointrange(boundaries: pd.DataFrame, path: Path) -> Path:
    b = boundaries.sort_values(["horizon", "nominal_covariate_share"]).reset_index(drop=True)
    labels = [f"r={r.nominal_covariate_share:g}, H={int(r.horizon)}" for r in b.itertuples()]
    ypos = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    finite = b["status"] == "finite"
    ax.errorbar(b.loc[finite, "boundary_lambda"], ypos[finite.to_numpy()],
                xerr=[(b.loc[finite, "boundary_lambda"] - b.loc[finite, "ci_low"]).to_numpy(),
                      (b.loc[finite, "ci_high"] - b.loc[finite, "boundary_lambda"]).to_numpy()],
                fmt="o", capsize=4, label="finite crossing (95% CI)")
    for i, r in b.iterrows():
        if r["status"] != "finite":
            ax.annotate(r["status"], xy=(np.nanmean(b["boundary_lambda"]), i), fontsize=8,
                        ha="center", va="center")
    ax.axvline(1.0, linestyle=":", color="gray")
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("boundary lambda (V_future = 0)")
    ax.set_title("Study 1B: estimated boundary per curve\n(dotted: lambda = 1 reference line)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    return _close(fig, path)


def figB3_coarse_vs_refined(coarse_cells: pd.DataFrame, refined_cells: pd.DataFrame,
                            horizon: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for share, g in coarse_cells[(coarse_cells["horizon"] == horizon)
                                 & (coarse_cells["nominal_covariate_share"] > 0)].groupby(
            "nominal_covariate_share"):
        g = g.sort_values("lam")
        ax.plot(g["lam"], g["v_future_mean"], marker="o", linestyle="--", alpha=0.55,
                label=f"coarse r={share:g}")
    for share, g in refined_cells[refined_cells["horizon"] == horizon].groupby(
            "nominal_covariate_share"):
        g = g.sort_values("lam")
        ax.plot(g["lam"], g["v_future_mean"], marker="s", label=f"refined r={share:g}")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(1.0, linestyle=":", color="gray")
    ax.set_xlabel("lambda")
    ax.set_ylabel("V_future = WQL(M1) - WQL(M3)")
    ax.set_title(f"Coarse pilot (seed 20260730) vs independent refinement (seed 20260801)\n"
                 f"horizon {horizon}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figB4_method_boundaries(all_boundaries: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = list(dict.fromkeys(all_boundaries["metric"]))
    curves = (all_boundaries[["nominal_covariate_share", "horizon"]]
              .drop_duplicates().sort_values(["horizon", "nominal_covariate_share"]))
    labels = [f"r={r.nominal_covariate_share:g}\nH={int(r.horizon)}" for r in curves.itertuples()]
    xpos = np.arange(len(curves))
    width = 0.8 / max(1, len(metrics))
    for k, m in enumerate(metrics):
        vals, xs = [], []
        for j, r in enumerate(curves.itertuples()):
            row = all_boundaries[(all_boundaries["metric"] == m)
                                 & (all_boundaries["nominal_covariate_share"] == r.nominal_covariate_share)
                                 & (all_boundaries["horizon"] == r.horizon)]
            if len(row) and row.iloc[0]["status"] == "finite":
                vals.append(float(row.iloc[0]["boundary_lambda"]))
                xs.append(j + (k - len(metrics) / 2 + 0.5) * width)
        if vals:
            ax.scatter(xs, vals, label=m, s=45)
    ax.axhline(1.0, linestyle=":", color="gray")
    ax.set_xticks(xpos, labels, fontsize=8)
    ax.set_ylabel("boundary lambda")
    ax.set_title("Boundary by method and metric\n"
                 "(WQL and MSE boundaries are different quantities, shown side by side only)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figB5_harm_rate(cells: pd.DataFrame, horizon: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for share, g in cells[cells["horizon"] == horizon].groupby("nominal_covariate_share"):
        g = g.sort_values("lam")
        ax.plot(g["lam"], g["harm_rate"], marker="o", label=f"share = {share:g}")
    ax.axvline(1.0, linestyle=":", color="gray")
    ax.set_xlabel("lambda")
    ax.set_ylabel("harm rate (WQL(M3) > 1.05 x WQL(M1))")
    ax.set_title(f"Study 1B: harm rate versus covariate error, horizon {horizon}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)


# ------------------------------------------------------------- Study 2 -------

def figD1_lambda_trajectories(schedules: pd.DataFrame, horizon: int, path: Path) -> Path:
    s = schedules[schedules["horizon"] == horizon]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in s.groupby("schedule"):
        g = g.sort_values("origin_index")
        ax.plot(g["origin_index"], g["true_lambda"], marker="o", label=name)
    ax.axhline(1.0, linestyle=":", color="gray")
    n = int(s["origin_index"].max())
    ax.set_xticks(range(n + 1), [f"hist {i+1}" for i in range(n)] + ["current"])
    ax.set_ylabel("true lambda")
    ax.set_title(f"Study 2: reliability schedules, horizon {horizon}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figD2_selector_wql_by_schedule(condition_summary: pd.DataFrame, proxy_mode: str,
                                   path: Path) -> Path:
    c = condition_summary[condition_summary["proxy_mode"] == proxy_mode]
    schedules = sorted(c["schedule"].unique())
    selectors = sorted(c["selector"].unique())
    xpos = np.arange(len(schedules))
    width = 0.8 / len(selectors)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, sel in enumerate(selectors):
        vals = [float(c[(c["schedule"] == s) & (c["selector"] == sel)]["mean_wql"].iloc[0])
                for s in schedules]
        ax.bar(xpos + (k - len(selectors) / 2 + 0.5) * width, vals, width, label=sel)
    ax.set_xticks(xpos, [s.replace("_", "\n", 1) for s in schedules], fontsize=8)
    ax.set_ylabel("mean WQL")
    ax.set_title(f"Study 2: selector WQL by reliability schedule ({proxy_mode})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figD_regret(condition_summary: pd.DataFrame, schedule: str, proxy_mode: str,
                path: Path) -> Path:
    c = condition_summary[(condition_summary["schedule"] == schedule)
                          & (condition_summary["proxy_mode"] == proxy_mode)].sort_values("selector")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(c["selector"], c["mean_regret"])
    for i, v in enumerate(c["mean_regret"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("mean regret vs per-task oracle (WQL)")
    ax.set_title(f"Study 2: selector regret under {schedule} ({proxy_mode})")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figD5_calibration_vs_harm(proxy_summary: pd.DataFrame, selector: str, path: Path) -> Path:
    p = proxy_summary[proxy_summary["selector"] == selector]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(p["mean_calibration_ratio"], p["harm_rate"], s=60)
    for _, r in p.iterrows():
        ax.annotate(r["proxy_mode"], (r["mean_calibration_ratio"], r["harm_rate"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axvline(1.0, linestyle=":", color="gray")
    ax.set_xlabel("mean calibration ratio (reported lambda / true lambda)")
    ax.set_ylabel("harm rate")
    ax.set_title(f"Study 2: proxy calibration versus harm rate ({selector})")
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figD6_false_rates(proxy_summary: pd.DataFrame, selectors: list[str], path: Path) -> Path:
    p = proxy_summary[proxy_summary["selector"].isin(selectors)]
    modes = sorted(p["proxy_mode"].unique())
    xpos = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / (2 * len(selectors))
    k = 0
    for sel in selectors:
        for metric in ("false_use_rate", "false_reject_rate"):
            vals = [float(p[(p["proxy_mode"] == m) & (p["selector"] == sel)][metric].iloc[0])
                    for m in modes]
            ax.bar(xpos + (k - len(selectors) + 0.5) * width, vals, width,
                   label=f"{sel} {metric.replace('_rate', '')}")
            k += 1
    ax.set_xticks(xpos, modes, fontsize=8, rotation=15)
    ax.set_ylabel("conditional rate")
    ax.set_title("Study 2: false-use and false-reject by proxy mode")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figD7_selector_comparison(proxy_summary: pd.DataFrame, proxy_mode: str, path: Path) -> Path:
    p = proxy_summary[proxy_summary["proxy_mode"] == proxy_mode].sort_values("mean_wql")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(p["selector"], p["mean_wql"])
    for i, v in enumerate(p["mean_wql"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("mean WQL over the whole mixture")
    ax.set_title(f"Study 2: fixed policies, history-only, proxy and hybrid selectors "
                 f"({proxy_mode})")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


# ============================================================================
# Study 2B - held-out D7 confirmation.  Additive; earlier figures unchanged.
# ============================================================================

def figure_g1_overall_wql(selector_summary: pd.DataFrame, path: Path) -> Path:
    s = selector_summary.sort_values("mean_wql")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(s["selector"], s["mean_wql"])
    for i, (v, sel) in enumerate(zip(s["mean_wql"], s["selector"])):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("mean WQL over the held-out mixture")
    ax.set_title("Study 2B: overall WQL by policy (P1 calibrated proxy)\n"
                 "primary policy D7_hybrid_override was fixed before the run")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figure_g2_schedule_wql(condition_summary: pd.DataFrame, proxy_mode: str,
                           selectors: list[str], path: Path) -> Path:
    c = condition_summary[(condition_summary["proxy_mode"] == proxy_mode)
                          & (condition_summary["selector"].isin(selectors))]
    schedules = sorted(c["schedule"].unique())
    xpos = np.arange(len(schedules))
    width = 0.8 / len(selectors)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, sel in enumerate(selectors):
        vals = [float(c[(c["schedule"] == s) & (c["selector"] == sel)]["mean_wql"].iloc[0])
                for s in schedules]
        ax.bar(xpos + (k - len(selectors) / 2 + 0.5) * width, vals, width, label=sel)
    ax.set_xticks(xpos, [s.replace("_", "\n", 1) for s in schedules], fontsize=8)
    ax.set_ylabel("mean WQL")
    ax.set_title(f"Study 2B: WQL by reliability schedule ({proxy_mode})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figure_g3_harm_rate(condition_summary: pd.DataFrame, proxy_mode: str,
                        selectors: list[str], path: Path) -> Path:
    c = condition_summary[(condition_summary["proxy_mode"] == proxy_mode)
                          & (condition_summary["selector"].isin(selectors))]
    schedules = sorted(c["schedule"].unique())
    xpos = np.arange(len(schedules))
    width = 0.8 / len(selectors)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, sel in enumerate(selectors):
        vals = [float(c[(c["schedule"] == s) & (c["selector"] == sel)]["harm_rate"].iloc[0])
                for s in schedules]
        ax.bar(xpos + (k - len(selectors) / 2 + 0.5) * width, vals, width, label=sel)
    ax.set_xticks(xpos, [s.replace("_", "\n", 1) for s in schedules], fontsize=8)
    ax.set_ylabel("harm rate (WQL > 1.05 x WQL(M1))")
    ax.set_title(f"Study 2B: harm rate by schedule ({proxy_mode})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figure_g4_false_use_reject(proxy_stress: pd.DataFrame, selector: str, path: Path) -> Path:
    p = proxy_stress[proxy_stress["selector"] == selector].sort_values("proxy_mode")
    modes = list(p["proxy_mode"])
    xpos = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xpos - 0.2, p["false_use_rate"], 0.4, label="false-use")
    ax.bar(xpos + 0.2, p["false_reject_rate"], 0.4, label="false-reject")
    ax.set_xticks(xpos, modes, fontsize=8, rotation=15)
    ax.set_ylabel("conditional rate")
    ax.set_title(f"Study 2B: {selector} false-use and false-reject by proxy mode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figure_g5_proxy_calibration(proxies: pd.DataFrame, proxy_mode: str, path: Path) -> Path:
    p = proxies[proxies["proxy_mode"] == proxy_mode]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.012, size=len(p))
    ax.scatter(p["true_current_lambda"] + jitter, p["reported_lambda"], s=6, alpha=0.25)
    lim = [0.0, float(max(p["true_current_lambda"].max(), p["reported_lambda"].max())) * 1.05]
    ax.plot(lim, lim, linestyle="--", color="gray", label="perfect report")
    ax.axhline(1.0, linestyle=":", color="black", linewidth=1)
    for t, g in p.groupby("true_current_lambda"):
        ax.plot([t], [g["reported_lambda"].median()], marker="o", markersize=9,
                color="black", linestyle="none")
    ax.set_xlabel("true current lambda")
    ax.set_ylabel("reported lambda")
    ax.set_title(f"Study 2B: reported versus true current uncertainty ({proxy_mode})\n"
                 "black dots: median report; dotted line: the 1.0 use threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figure_g6_stable_high_tail(decisions: pd.DataFrame, proxy_mode: str, selector: str,
                               schedule: str, path: Path) -> Path:
    d = decisions[(decisions["proxy_mode"] == proxy_mode)
                  & (decisions["selector"] == selector)
                  & (decisions["schedule"] == schedule)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(d["reported_lambda"], d["regret"], s=8, alpha=0.35)
    ax.axvline(1.0, linestyle=":", color="black", linewidth=1)
    ax.axvline(0.75, linestyle="--", color="gray", linewidth=1)
    ax.axvline(1.25, linestyle="--", color="gray", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("reported lambda (true lambda is 1.5 in this schedule)")
    ax.set_ylabel("regret versus the per-task oracle (WQL)")
    ax.set_title(f"Study 2B: {schedule} tail behaviour of {selector} ({proxy_mode})\n"
                 "dashed: D7 override band 0.75 / 1.25; dotted: the 1.0 use threshold")
    ax.grid(alpha=0.3)
    return _close(fig, path)


# ============================================================================
# Study 3 - real forecast-vintage external validation.  Additive.
# ============================================================================

def figure_r1_lambda_vs_value(tasks: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(tasks["realized_weather_error_ratio"], -tasks["v_future"], s=7, alpha=0.25)
    b = pd.cut(tasks["realized_weather_error_ratio"], bins=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 10])
    med = (-tasks["v_future"]).groupby(b, observed=True).median()
    centres = [iv.mid for iv in med.index]
    ax.plot(centres, med.values, marker="o", color="black", label="binned median")
    ax.axhline(0.0, color="black", linewidth=1)
    for x in (0.75, 1.25):
        ax.axvline(x, linestyle="--", color="gray", linewidth=1)
    ax.axvline(1.0, linestyle=":", color="gray", linewidth=1)
    ax.set_xlim(0, min(3.0, float(tasks["realized_weather_error_ratio"].quantile(0.99))))
    ax.set_xlabel("realized weather error ratio (ECMWF RMSE / seasonal-naive RMSE)")
    ax.set_ylabel("WQL(M3) - WQL(M1)   (negative = forecast helped)")
    ax.set_title("Study 3: does real forecast accuracy change the sign of the value?\n"
                 "dashed: the pre-registered D7 override band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figure_r2_revision_vs_lambda(features: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(features["revision_ratio"], features["realized_weather_error_ratio"], s=7, alpha=0.25)
    ax.set_xlabel("raw run-to-run revision ratio (decision-time)")
    ax.set_ylabel("realized weather error ratio (known only afterwards)")
    ax.set_xlim(0, float(features["revision_ratio"].quantile(0.99)))
    ax.set_ylim(0, float(features["realized_weather_error_ratio"].quantile(0.99)))
    ax.set_title("Study 3: raw revision proxy versus realized forecast error")
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figure_r3_calibration(features: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, g in features.groupby("split"):
        ax.scatter(g["reported_reliability_ratio"], g["realized_weather_error_ratio"], s=7, alpha=0.3, label=split)
    lim = [0, float(np.nanquantile(features["realized_weather_error_ratio"], 0.99))]
    ax.plot(lim, lim, linestyle="--", color="gray", label="perfect calibration")
    ax.axvline(1.0, linestyle=":", color="black", linewidth=1)
    ax.set_xlim(0, float(np.nanquantile(features["reported_reliability_ratio"], 0.99)))
    ax.set_ylim(*lim)
    ax.set_xlabel("calibrated reported reliability ratio (decision-time)")
    ax.set_ylabel("realized weather error ratio")
    ax.set_title("Study 3: calibrated reliability proxy versus truth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _close(fig, path)


def figure_r4_selector_wql(selector_summary: pd.DataFrame, path: Path) -> Path:
    s = selector_summary.sort_values("mean_wql")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(s["selector"], s["mean_wql"])
    for i, v in enumerate(s["mean_wql"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("mean WQL on the held-out period")
    ax.set_title("Study 3: held-out WQL by policy (NYISO, real ECMWF vintages)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def _grouped_bars(df: pd.DataFrame, key: str, value: str, selectors: list[str],
                  title: str, ylabel: str, path: Path) -> Path:
    keys = sorted(df[key].unique())
    xpos = np.arange(len(keys))
    width = 0.8 / max(1, len(selectors))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, sel in enumerate(selectors):
        vals = []
        for k in keys:
            sub = df[(df[key] == k) & (df["selector"] == sel)]
            vals.append(float(sub[value].iloc[0]) if len(sub) else np.nan)
        ax.bar(xpos + (i - len(selectors) / 2 + 0.5) * width, vals, width, label=sel)
    ax.set_xticks(xpos, [str(k) for k in keys], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figure_r5_zone_wql(zone_summary: pd.DataFrame, selectors: list[str], path: Path) -> Path:
    return _grouped_bars(zone_summary, "zone", "mean_wql", selectors,
                         "Study 3: held-out WQL by NYISO zone", "mean WQL", path)


def figure_r6_season_wql(season_summary: pd.DataFrame, selectors: list[str], path: Path) -> Path:
    return _grouped_bars(season_summary, "season", "mean_wql", selectors,
                         "Study 3: held-out WQL by season", "mean WQL", path)


def figure_r7_harm_rate(zone_summary: pd.DataFrame, selectors: list[str], path: Path) -> Path:
    return _grouped_bars(zone_summary, "zone", "harm_rate", selectors,
                         "Study 3: harm rate by zone (WQL > 1.05 x WQL(M1))", "harm rate", path)


def figure_r8_false_rates(selector_summary: pd.DataFrame, path: Path) -> Path:
    s = selector_summary.sort_values("selector")
    xpos = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xpos - 0.2, s["false_use_rate"], 0.4, label="false-use")
    ax.bar(xpos + 0.2, s["false_reject_rate"], 0.4, label="false-reject")
    ax.set_xticks(xpos, s["selector"], rotation=30, fontsize=7)
    ax.set_ylabel("conditional rate")
    ax.set_title("Study 3: false-use and false-reject by policy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figure_r9_event_subset(event_summary: pd.DataFrame, event: str, path: Path) -> Path:
    e = event_summary[event_summary["event"] == event].sort_values("mean_wql")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if len(e):
        ax.bar(e["selector"], e["mean_wql"])
        for i, v in enumerate(e["mean_wql"]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        n = int(e["n_origins"].iloc[0])
    else:
        n = 0
    ax.set_ylabel("mean WQL")
    ax.set_title(f"Study 3: {event} subset (n = {n} origins)"
                 + ("" if n >= 20 else "  - NOT_EVALUABLE, fewer than 20 events"))
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    return _close(fig, path)


def figure_r11_example_vintage(primary: pd.DataFrame, revision: pd.DataFrame,
                               verification: pd.DataFrame, zone: str, origin,
                               path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(primary["valid_time_utc"], primary["temperature_forecast"], marker="o",
            label="00Z primary forecast (used by M3)")
    ax.plot(revision["valid_time_utc"], revision["temperature_forecast"], marker="s",
            linestyle="--", label="previous-day 12Z run (revision proxy only)")
    ax.plot(verification["valid_time_utc"], verification["temperature_verified"],
            marker="^", label="verification (used by M1 history and the M2 oracle)")
    ax.set_xlabel("valid time (UTC)")
    ax.set_ylabel("temperature_2m (C)")
    ax.set_title(f"Study 3: one real vintage, {zone}, decision origin {origin}")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.grid(alpha=0.3)
    return _close(fig, path)
