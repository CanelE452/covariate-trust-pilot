"""Poster stage-2 tables and figures. Reads only; never trains.

regime_h1 answered "which SBC regime carries H1" with a Spearman rank
correlation on M5.  A poster needs a slope with a confidence interval, and it
needs Favorita on the same axes, so this module

  * reuses the M5 regime labels regime_h1 already reproduced,
  * derives the Favorita labels from the frozen SBC rule and refuses to
    continue unless the stratified 300-per-regime draw is reproduced exactly,
  * fits delta_relative ~ |rho_interval| by OLS inside each dataset x regime,
  * reports both the HC3 interval and the frozen series bootstrap interval,
  * and draws the scatter and the forest plot.

Nothing here re-derives a cutoff, a regime rule or an eligibility threshold.
The SBC thresholds, the eligibility threshold, the bootstrap protocol and the
delta definition all come from the frozen record.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import posthoc, prereg, regime_h1, screen

OUT = screen.OUT / "poster_stage2"
DATASETS = ("m5", "favorita")
#: The two regimes the synthetic study modelled; the poster reports these.
TARGET_REGIMES = ("intermittent", "lumpy")
COLOUR = {"intermittent": "#1f6fb4", "lumpy": "#d1642a"}
DATASET_LABEL = {"m5": "M5", "favorita": "Favorita"}

X_COLUMN = "rho_interval_abs_train"
X_LABEL = ("occurrence dependence\n"
           r"|lag-1 autocorr. of inter-arrival gaps|")
Y_LABEL = ("hurdle relative gain\n"
           r"(RMSE$_{P}$-RMSE$_{H}$)/RMSE$_{P}$")

FONT = {"font.size": 22, "axes.labelsize": 24, "axes.titlesize": 24,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
        "lines.linewidth": 2.5, "axes.linewidth": 2.0}


# ------------------------------------------------------------- labelling ----


def favorita_regimes(block: pd.DataFrame) -> tuple[pd.Series, dict]:
    """SBC labels for Favorita from its own train-only selection descriptors.

    Favorita's primary descriptor variant is ``raw`` and prepare_favorita.py
    computed its selection descriptors on the same train segment, so the
    columns already in per_series_metrics.csv are the selection descriptors.
    The 300-per-regime check is what turns that argument into a verified
    reproduction rather than an assumption.
    """
    regime = pd.Series(regime_h1.classify(block["ADI_train"].to_numpy(float),
                                          block["CV2_positive_train"].to_numpy(float)),
                       index=block.index)
    counts = regime.value_counts().to_dict()
    check = {"regime_counts": counts, "expected_per_regime": 300,
             "descriptor_source": "per_series_metrics.csv, descriptor_variant=raw",
             "reproduced": bool(all(counts.get(r) == 300 for r in regime_h1.REGIMES)
                                and len(counts) == len(regime_h1.REGIMES))}
    return regime, check


def test_positive_count(name: str) -> pd.Series:
    """Positive observations inside the evaluation window, per series.

    A series with none of them is scored against an all-zero target, so the
    relative gain degenerates to "whichever model predicts closer to zero" and
    carries no information about occurrence structure. Counting them is the
    only way to say how much of a dataset's gain is that artefact.
    """
    split = prereg.SPLITS[name]
    frame = pd.read_parquet(screen.DATASETS[name]["parquet"])
    wide = frame.pivot_table(index="series_id", columns="timestamp",
                             values="target").sort_index()
    test = wide.to_numpy(np.float32)[:, split["val_end"]:split["length"]]
    return pd.Series((test > 0).sum(axis=1), index=wide.index.astype(str),
                     name="n_positive_test")


def labelled_pool() -> tuple[pd.DataFrame, dict]:
    """Every eligible series with its frozen-rule regime label, both datasets."""
    threshold = prereg.ELIGIBILITY["primary_threshold"]
    series = pd.read_csv(screen.OUT / "per_series_metrics.csv")
    checks = {}
    frames = []

    m5 = series[series["dataset"] == "m5"].copy()
    m5_labels = regime_h1.regime_table(
        screen.REPO / "data" / "sales_train_evaluation.csv",
        screen.config_for("m5").train_end)
    checks["m5"] = regime_h1.verify_against_design(m5_labels, set(m5["series_id"]))
    if not checks["m5"]["reproduced"]:
        raise screen.ScreenFailure(
            "REGIME_LABEL_NOT_REPRODUCED (m5): "
            f"{checks['m5']['stage_a_regime_counts']}")
    m5 = m5.merge(m5_labels[["series_id", "regime", "ADI_selection",
                             "CV2_selection"]], on="series_id", how="left")
    frames.append(m5)

    fav = series[series["dataset"] == "favorita"].copy()
    regime, checks["favorita"] = favorita_regimes(fav)
    if not checks["favorita"]["reproduced"]:
        raise screen.ScreenFailure(
            "REGIME_LABEL_NOT_REPRODUCED (favorita): "
            f"{checks['favorita']['regime_counts']}")
    fav["regime"] = regime
    fav["ADI_selection"] = fav["ADI_train"]
    fav["CV2_selection"] = fav["CV2_positive_train"]
    frames.append(fav)

    pool = pd.concat(frames, ignore_index=True)
    counts = pd.concat([test_positive_count(name) for name in DATASETS])
    pool["n_positive_test"] = pool["series_id"].astype(str).map(counts).to_numpy()
    if pool["n_positive_test"].isna().any():
        raise screen.ScreenFailure("a series has no evaluation-window row")
    pool["eligible"] = ((pool["n_positive_train"] >= threshold)
                        & np.isfinite(pool["delta_rmse"])
                        & (pool["rmse_point"] > posthoc.RELATIVE_FLOOR))
    pool["relative_gain"] = np.where(pool["eligible"],
                                     pool["delta_rmse"] / pool["rmse_point"],
                                     np.nan)
    return pool, checks


# ---------------------------------------------------------------- models ----


def ols_slope(x: np.ndarray, y: np.ndarray) -> dict:
    """delta_relative ~ 1 + |rho_interval|, HC3 heteroskedasticity-robust."""
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    lo, hi = fit.conf_int(alpha=0.05)[1]
    return {"n": int(fit.nobs),
            "beta": float(fit.params[1]), "se_hc3": float(fit.bse[1]),
            "ci_hc3": [float(lo), float(hi)],
            "t": float(fit.tvalues[1]), "p": float(fit.pvalues[1]),
            "r2": float(fit.rsquared),
            "intercept": float(fit.params[0])}


def boot_slope_ci(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Frozen protocol: resample series, 2000 draws, seed 20260807."""
    rng = np.random.default_rng(posthoc.SEED)

    def slope(idx):
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0:
            return np.nan
        return float(np.polyfit(xi, yi, 1)[0])

    return list(posthoc._boot_ci(slope, len(x), rng))


def regime_fit(block: pd.DataFrame) -> dict:
    x = block[X_COLUMN].to_numpy(float)
    y = block["relative_gain"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rng = np.random.default_rng(posthoc.SEED)
    out = ols_slope(x, y)
    out["ci_bootstrap_series"] = boot_slope_ci(x, y)
    out["spearman"] = posthoc._spearman(x, y)
    out["spearman_ci_bootstrap"] = list(
        posthoc._boot_ci(lambda i: posthoc._spearman(x[i], y[i]), len(x), rng))
    out["mean_relative_gain"] = float(y.mean())
    out["median_relative_gain"] = float(np.median(y))
    out["hurdle_win_rate"] = float((y > 0).mean())
    out["x_quantiles"] = {k: float(v) for k, v in zip(
        ("min", "q25", "median", "q75", "max"),
        np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0]))}
    out["y_quantiles"] = {k: float(v) for k, v in zip(
        ("min", "q05", "q25", "median", "q75", "q95", "max"),
        np.quantile(y, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]))}
    out["quintile_profile"] = quintile_profile(x, y)
    out["n_dropped_nonfinite"] = int((~ok).sum())
    return out


def quintile_profile(x: np.ndarray, y: np.ndarray) -> list[dict]:
    """Mean gain per quintile of x — shows whether the slope is even linear.

    |rho_interval| is strongly right skewed, so an OLS slope is leveraged by a
    thin tail. This table is the diagnostic that says so out loud.
    """
    edges = np.quantile(x, np.linspace(0, 1, 6))
    bucket = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, 4)
    return [{"quintile": int(k), "n": int((bucket == k).sum()),
             "x_mean": float(x[bucket == k].mean()),
             "gain_mean": float(y[bucket == k].mean()),
             "gain_median": float(np.median(y[bucket == k]))}
            for k in range(5)]


#: The synthetic study's interval contrast, confirmed against its generator in
#: m5dataset (gap = d + s is affine in the latent state, so the gap sequence's
#: lag-1 autocorrelation IS the rho_I knob; measured -0.8026 for a -0.80
#: target). rho_M does not survive the same way: M = 1 + Poisson(lambda - 1)
#: adds within-level noise, so an observed magnitude autocorrelation is the
#: knob times Var_between / Var_total = 25/34 for lambda in {5, 15}.
SYNTHETIC = {"abs_rho_interval_contrast": (0.0, 0.80),
             "rho_magnitude_knob": 0.80,
             "magnitude_attenuation": 25 / 34,
             "rho_magnitude_on_observed_scale": 0.80 * 25 / 34,
             "attenuation_depends_on": "lambda_levels = (5, 15)",
             "source": "m5dataset dgp.py:43/80/162, verified by regenerating "
                       "d=4 and d=8 (9 cells each, 200 series)"}


def support_diagnostic(pool: pd.DataFrame) -> dict:
    """Does the real data even reach the synthetic contrast?

    The descriptor matches the synthetic knob exactly, which makes the next
    question the binding one: an axis measured only where it is near zero
    cannot confirm a mechanism that was demonstrated at |rho_I| = 0.8.

    The noise floor matters as much as the counts. A lag-1 autocorrelation
    estimated from n gaps has standard error ~1/sqrt(n) under independence, so
    the median |rho_I| is only evidence of real dependence if it clears that.
    """
    out = {}
    target = SYNTHETIC["abs_rho_interval_contrast"][1]
    magnitude_target = SYNTHETIC["rho_magnitude_on_observed_scale"]
    for name in DATASETS:
        block = pool[(pool["dataset"] == name) & pool["eligible"]]
        x = block[X_COLUMN].to_numpy(float)
        mag = block["rho_magnitude_train"].to_numpy(float)
        gaps = block["n_positive_train"].to_numpy(float) - 1.0
        ok = np.isfinite(x) & (gaps > 3)
        # Median |r| of a white-noise lag-1 autocorrelation at this n.
        floor = 0.6745 / np.sqrt(gaps[ok])
        significant = np.abs(x[ok]) > 1.96 / np.sqrt(gaps[ok])
        good = np.isfinite(mag)
        out[name] = {
            "n": int(ok.sum()),
            "abs_rho_interval": {
                "median": float(np.median(x[ok])),
                "p90": float(np.quantile(x[ok], 0.9)),
                "max": float(x[ok].max()),
                "n_at_or_above_synthetic_contrast": int((x[ok] >= target).sum()),
                "n_at_or_above_half_contrast": int((x[ok] >= target / 2).sum()),
            },
            "noise_floor": {
                "n_gaps_median": float(np.median(gaps[ok])),
                "expected_median_abs_rho_if_independent": float(np.median(floor)),
                "n_exceeding_two_sided_5pct": int(significant.sum()),
                "share_exceeding_two_sided_5pct": float(significant.mean()),
            },
            "rho_magnitude": {
                "median": float(np.median(mag[good])),
                "max": float(mag[good].max()),
                "n_at_or_above_synthetic_on_observed_scale":
                    int((mag[good] >= magnitude_target).sum()),
                "upper_tertile": float(np.quantile(mag[good], 2 / 3)),
                "upper_tertile_as_share_of_synthetic":
                    float(np.quantile(mag[good], 2 / 3) / magnitude_target),
            },
        }
    return out


def figure5(pool: pd.DataFrame, support: dict) -> None:
    """Where the real data sits relative to the synthetic contrast."""
    with plt.rc_context(FONT):
        fig, axes = plt.subplots(1, 2, figsize=(16.84, 6.50), sharey=True,
                                 layout="constrained")
        for ax, name in zip(axes, DATASETS):
            block = pool[(pool["dataset"] == name) & pool["eligible"]]
            x = block[X_COLUMN].to_numpy(float)
            x = x[np.isfinite(x)]
            ax.hist(x, bins=np.linspace(0, 1, 51), color="0.55",
                    edgecolor="white", linewidth=0.6)
            floor = support[name]["noise_floor"][
                "expected_median_abs_rho_if_independent"]
            ax.axvline(floor, color="#2e7d32", linewidth=2.5,
                       label=f"noise floor ({floor:.3f})")
            for value in SYNTHETIC["abs_rho_interval_contrast"]:
                ax.axvline(value, color="#b3202c", linewidth=2.5,
                           linestyle="--")
            n_at = support[name]["abs_rho_interval"][
                "n_at_or_above_synthetic_contrast"]
            ax.annotate(f"synthetic contrast 0 vs 0.80\n{n_at} series at or above 0.80",
                        xy=(0.80, 0.0), xytext=(0.30, 0.55),
                        textcoords="axes fraction", fontsize=19,
                        color="#b3202c", ha="left")
            ax.set_xlim(0, 1)
            ax.set_title(DATASET_LABEL[name])
            ax.legend(loc="upper right", frameon=False, fontsize=19)
        axes[0].set_ylabel("series")
        fig.supxlabel(X_LABEL, fontsize=FONT["axes.labelsize"])
        fig.savefig(OUT / "fig5_support_vs_synthetic.png", dpi=220)
        plt.close(fig)


def bridge_diagnostic(pool: pd.DataFrame) -> dict:
    """The two candidate occurrence-dependence measures, kept apart on purpose.

    The synthetic knob is settled: rho_I is the latent interval state's lag-1
    autocorrelation, and the gap is affine in that state, so the frozen
    descriptor |lag-1 autocorrelation of the inter-arrival gaps| IS the knob.
    The 0/1 series' autocorrelation, which screen.py also stores, is a
    different quantity and is NOT what the synthetic study varied.

    It is reported anyway because it points the other way, and a reader who
    hears "occurrence dependence" and pictures occurrence persistence should
    see that the two do not agree rather than discover it later.
    """
    out = {}
    for name in DATASETS:
        out[name] = {}
        for regime in TARGET_REGIMES:
            block = pool[(pool["dataset"] == name) & pool["eligible"]
                         & (pool["regime"] == regime)]
            gap = block[X_COLUMN].to_numpy(float)
            binary = np.abs(block["occurrence_binary_acf1_train"].to_numpy(float))
            y = block["relative_gain"].to_numpy(float)
            ok = np.isfinite(gap) & np.isfinite(binary) & np.isfinite(y)
            rng = np.random.default_rng(posthoc.SEED)
            b, yy = binary[ok], y[ok]
            out[name][regime] = {
                "n": int(ok.sum()),
                "spearman_between_measures": posthoc._spearman(gap[ok], binary[ok]),
                "pearson_between_measures": float(
                    np.corrcoef(gap[ok], binary[ok])[0, 1]),
                "binary_acf1_quantiles": {k: float(v) for k, v in zip(
                    ("min", "q25", "median", "q75", "max"),
                    np.quantile(b, [0.0, 0.25, 0.5, 0.75, 1.0]))},
                "h1_with_binary_acf1": {
                    "spearman": posthoc._spearman(b, yy),
                    "ci_bootstrap": list(posthoc._boot_ci(
                        lambda i: posthoc._spearman(b[i], yy[i]), len(b), rng)),
                    "ols": ols_slope(b, yy),
                },
            }
    return out


# ---------------------------------------------------------------- tables ----


def dataset_table(pool: pd.DataFrame) -> dict:
    """Poster table 2: one column per dataset, medians over the used series."""
    out = {}
    for name in DATASETS:
        block = pool[pool["dataset"] == name]
        used = block[block["eligible"]]
        split = prereg.SPLITS[name]
        frame = pd.read_parquet(screen.DATASETS[name]["parquet"],
                                columns=["timestamp"])
        stamps = pd.to_datetime(frame["timestamp"].unique())
        out[name] = {
            "n_series_selected": int(len(block)),
            "n_series_used": int(len(used)),
            "n_series_excluded": int(len(block) - len(used)),
            "exclusion_rule": (f"n_positive_train >= "
                               f"{prereg.ELIGIBILITY['primary_threshold']}"),
            "period": {"start": str(stamps.min().date()),
                       "end": str(stamps.max().date()),
                       "length_days": int(split["length"])},
            "split": {"train_end": split["train_end"], "val_end": split["val_end"],
                      "train_days": split["train_end"],
                      "validation_days": split["val_end"] - split["train_end"],
                      "test_days": split["length"] - split["val_end"],
                      "test_origins": 3, "test_origin_stride": 28},
            "horizon": split["horizon"], "lookback": split["lookback"],
            "n_all_zero_test_window": int((used["n_positive_test"] == 0).sum()),
            "zero_ratio_train_median": float(used["zero_ratio_train"].median()),
            "ADI_selection_median": float(used["ADI_selection"].median()),
            "CV2_selection_median": float(used["CV2_selection"].median()),
            "descriptor_variant": str(block["descriptor_variant"].iloc[0]),
        }
    return out


def regime_counts(pool: pd.DataFrame) -> dict:
    out = {}
    for name in DATASETS:
        block = pool[pool["dataset"] == name]
        out[name] = {r: {"selected": int((block["regime"] == r).sum()),
                         "used": int(((block["regime"] == r)
                                      & block["eligible"]).sum())}
                     for r in regime_h1.REGIMES}
    return out


# --------------------------------------------------------------- figures ----


def _band(ax, x, y, colour, label):
    """OLS line with an HC3 95% band for the fitted mean."""
    design = sm.add_constant(x)
    fit = sm.OLS(y, design).fit(cov_type="HC3")
    grid = np.linspace(x.min(), x.max(), 100)
    g = sm.add_constant(grid)
    mean = g @ fit.params
    se = np.sqrt(np.einsum("ij,jk,ik->i", g, fit.cov_params(), g))
    ax.plot(grid, mean, color=colour, linewidth=2.5, label=label, zorder=4)
    ax.fill_between(grid, mean - 1.96 * se, mean + 1.96 * se, color=colour,
                    alpha=0.20, linewidth=0, zorder=3)


def figure3(pool: pd.DataFrame, ylim: tuple[float, float]) -> dict:
    """Series-level scatter, one panel per dataset, regimes coloured."""
    clipped = {}
    with plt.rc_context(FONT):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        for ax, name in zip(axes, DATASETS):
            block = pool[(pool["dataset"] == name) & pool["eligible"]
                         & pool["regime"].isin(TARGET_REGIMES)]
            for regime in TARGET_REGIMES:
                sub = block[block["regime"] == regime]
                x = sub[X_COLUMN].to_numpy(float)
                y = sub["relative_gain"].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                ax.scatter(x, np.clip(y, *ylim), s=26, alpha=0.45,
                           color=COLOUR[regime], edgecolors="none", zorder=2)
                _band(ax, x, y, COLOUR[regime], f"{regime}  (n={len(x)})")
                clipped[f"{name}_{regime}"] = int(((y < ylim[0])
                                                   | (y > ylim[1])).sum())
            ax.axhline(0.0, color="0.35", linewidth=2.0, linestyle="--", zorder=1)
            ax.set_title(DATASET_LABEL[name])
            ax.set_ylim(*ylim)
            ax.legend(loc="upper left", frameon=False, handlelength=1.4)
        axes[0].set_ylabel(Y_LABEL)
        fig.supxlabel(X_LABEL, fontsize=FONT["axes.labelsize"], y=-0.09)
        fig.savefig(OUT / "fig3_occurrence_vs_gain.png", dpi=220,
                    bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    with plt.rc_context(FONT):
        fig, ax = plt.subplots(figsize=(16, 5))
        block = pool[(pool["dataset"] == "m5") & pool["eligible"]
                     & pool["regime"].isin(TARGET_REGIMES)]
        for regime in TARGET_REGIMES:
            sub = block[block["regime"] == regime]
            x = sub[X_COLUMN].to_numpy(float)
            y = sub["relative_gain"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            ax.scatter(x, np.clip(y, *ylim), s=30, alpha=0.45,
                       color=COLOUR[regime], edgecolors="none", zorder=2)
            _band(ax, x, y, COLOUR[regime], f"{regime}  (n={len(x)})")
        ax.axhline(0.0, color="0.35", linewidth=2.0, linestyle="--", zorder=1)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.set_ylim(*ylim)
        ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.4)
        fig.savefig(OUT / "fig3_occurrence_vs_gain_m5.png", dpi=220,
                    bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return {"y_axis_display_limits": list(ylim),
            "points_outside_display_limits": clipped,
            "note": "clipping is display only; every fit uses all points"}


def _forest(fits: dict, estimate_key: str, ci_key: str, xlabel: str,
            filename: str) -> None:
    rows = [(name, regime) for name in DATASETS for regime in TARGET_REGIMES
            if regime in fits[name]]
    with plt.rc_context(FONT):
        fig, ax = plt.subplots(figsize=(16.84, 4.40), layout="constrained")
        for i, (name, regime) in enumerate(rows):
            f = fits[name][regime]
            value, (lo, hi) = f[estimate_key], f[ci_key]
            ax.errorbar(value, i, xerr=[[value - lo], [hi - value]],
                        fmt="o", markersize=13, capsize=8, capthick=2.5,
                        color=COLOUR[regime], elinewidth=2.5, zorder=3)
        ax.axvline(0.0, color="0.35", linewidth=2.0, linestyle="--", zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{DATASET_LABEL[n]}  {r}" for n, r in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        fig.savefig(OUT / filename, dpi=220)
        plt.close(fig)


def figure4(fits: dict) -> None:
    """Two forest plots: the requested OLS slope, and the pre-registered test.

    H1's frozen test is a Spearman rank correlation, not a slope. Both are
    drawn because they do not tell the same story: the rank statistic separates
    intermittent from lumpy, the raw-scale slope does not separate anything.
    """
    _forest(fits, "beta", "ci_hc3",
            r"$\beta$  (OLS slope, HC3 95% CI)", "fig4_slope_forest.png")
    _forest(fits, "spearman", "spearman_ci_bootstrap",
            r"Spearman $\rho$  (95% bootstrap CI)",
            "fig4_spearman_forest.png")


# ------------------------------------------------------------------- run ----


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    pool, checks = labelled_pool()

    fits = {}
    for name in DATASETS:
        fits[name] = {}
        for regime in regime_h1.REGIMES:
            block = pool[(pool["dataset"] == name) & pool["eligible"]
                         & (pool["regime"] == regime)]
            if len(block) < 3:
                continue
            entry = regime_fit(block)
            alive = block[block["n_positive_test"] > 0]
            entry["n_all_zero_test_window"] = int(len(block) - len(alive))
            entry["sensitivity_drop_all_zero_test"] = (
                regime_fit(alive) if len(alive) >= 3 else None)
            fits[name][regime] = entry

    values = pool.loc[pool["eligible"] & pool["regime"].isin(TARGET_REGIMES),
                      "relative_gain"].to_numpy(float)
    values = values[np.isfinite(values)]
    ylim = tuple(float(v) for v in np.quantile(values, [0.01, 0.99]))

    support = support_diagnostic(pool)
    figure_note = figure3(pool, ylim)
    figure4(fits)
    figure5(pool, support)

    report = {
        "analysis": "poster stage 2: OLS slope of hurdle relative gain on "
                    "occurrence dependence, inside SBC regime, no retraining",
        "x": {"column": X_COLUMN,
              "definition": prereg.DESCRIPTOR_RULES["rho_interval_train"]
                            + ", absolute value",
              "window": "train split only"},
        "y": {"column": "relative_gain",
              "definition": "(RMSE_point - RMSE_hurdle) / RMSE_point, per series, "
                            "realized y on the test window",
              "sign": "positive favours Hurdle"},
        "eligibility": prereg.ELIGIBILITY,
        "sbc_rule": regime_h1.SBC,
        "label_reproduction": checks,
        "bootstrap": {"draws": posthoc.DRAWS, "seed": posthoc.SEED,
                      "unit": "series"},
        "regression": "OLS with HC3 robust standard errors; the bootstrap "
                      "interval resamples series under the frozen protocol",
        "synthetic_reference": SYNTHETIC,
        "support_diagnostic": support,
        "bridge_diagnostic": bridge_diagnostic(pool),
        "dataset_table": dataset_table(pool),
        "regime_counts": regime_counts(pool),
        "fits": fits,
        "figures": figure_note,
    }
    (OUT / "poster_stage2.json").write_text(
        json.dumps(report, indent=2, default=str))
    pool.loc[pool["eligible"], ["dataset", "series_id", "regime", X_COLUMN,
                               "relative_gain", "delta_rmse", "rmse_point",
                               "rmse_hurdle", "ADI_selection", "CV2_selection",
                               "zero_ratio_train", "n_positive_train",
                               "n_positive_test"]].to_csv(
        OUT / "per_series_poster.csv", index=False)
    return report


if __name__ == "__main__":
    result = run()
    for dataset, per_regime in result["fits"].items():
        for regime, f in per_regime.items():
            print(f"{dataset:9s} {regime:13s} n={f['n']:4d} "
                  f"beta={f['beta']:+.4f} "
                  f"CI[{f['ci_hc3'][0]:+.4f},{f['ci_hc3'][1]:+.4f}] "
                  f"p={f['p']:.4f} R2={f['r2']:.4f}")
