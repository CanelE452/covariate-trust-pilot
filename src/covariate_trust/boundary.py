"""Study 1B - independent refinement of the benefit-to-harm boundary.

Re-uses the coarse pilot's DGP, schemas, adapter, metrics and bootstrap unchanged;
only the seed, the lambda grid and the sample size differ.

The boundary is the lambda where the paired curve
``V_future(lambda) = WQL(M1) - WQL(M3)`` crosses zero from positive to negative.
It is estimated by linear interpolation between the two adjacent grid points that
bracket the sign change, and its interval comes from a cluster bootstrap over
``base_series_id``.  Nothing is extrapolated outside the grid: a curve that never
crosses inside [min(lambda), max(lambda)] is reported as censored or unresolved.

lambda = 1 is the Study-0 linear-Gaussian reference line only.  It is not a
theoretical boundary of Chronos WQL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .bootstrap import BOOTSTRAP_UNIT
from .config import BoundaryConfig
from .dgp import build_target, covariate_vintage, generate_base_series
from .metrics import is_harm, mse, nmae, quantile_crossing_rate, relative_delta, wql
from .schemas import M1, M2, M3, assert_context_equality, assert_future_equality, build_inputs
from .seeds import make_rng

FINITE, LEFT_CENSORED, RIGHT_CENSORED, UNRESOLVED = (
    "finite", "left_censored", "right_censored", "unresolved")


# --------------------------------------------------------------- crossing ----

def zero_crossing(lams: np.ndarray, values: np.ndarray) -> tuple[float, str]:
    """Interpolate where ``values`` crosses zero downward, or report why it does not.

    Only descending crossings (positive -> non-positive) count: the estimand is the
    lambda beyond which the forecasted future covariate stops helping.
    """
    lams = np.asarray(lams, dtype=float)
    values = np.asarray(values, dtype=float)
    if lams.shape != values.shape or lams.size < 2:
        raise ValueError("lams and values must be 1-D arrays of equal length >= 2")
    order = np.argsort(lams)
    lams, values = lams[order], values[order]

    desc = (values[:-1] > 0) & (values[1:] <= 0)
    if desc.any():
        i = int(np.argmax(desc))
        v0, v1 = values[i], values[i + 1]
        frac = v0 / (v0 - v1)            # v0 > 0 >= v1, so the denominator is positive
        return float(lams[i] + frac * (lams[i + 1] - lams[i])), FINITE
    if np.all(values > 0):
        return float("nan"), RIGHT_CENSORED   # still beneficial at the largest lambda
    if np.all(values <= 0):
        return float("nan"), LEFT_CENSORED    # already harmful at the smallest lambda
    return float("nan"), UNRESOLVED           # non-monotone sign pattern


def bootstrap_crossing(unit_curves: np.ndarray, lams: np.ndarray, n_resamples: int,
                       confidence_level: float, seed_parts: tuple) -> dict:
    """Cluster bootstrap of the crossing over rows (one row per base series)."""
    unit_curves = np.asarray(unit_curves, dtype=float)
    n_units = unit_curves.shape[0]
    rng = make_rng(*seed_parts, n_resamples, n_units)
    idx = rng.integers(0, n_units, size=(n_resamples, n_units))
    boot = unit_curves[idx].mean(axis=1)                       # (n_resamples, n_lambda)

    order = np.argsort(lams)
    lams_sorted = np.asarray(lams, dtype=float)[order]
    boot = boot[:, order]

    desc = (boot[:, :-1] > 0) & (boot[:, 1:] <= 0)
    has = desc.any(axis=1)
    first = np.argmax(desc, axis=1)
    rows = np.arange(boot.shape[0])
    v0 = boot[rows, first]
    v1 = boot[rows, first + 1]
    l0 = lams_sorted[first]
    l1 = lams_sorted[first + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings = l0 + (v0 / (v0 - v1)) * (l1 - l0)
    valid = crossings[has & np.isfinite(crossings)]

    alpha = 1.0 - confidence_level
    if valid.size < 2:
        return {"ci_low": float("nan"), "ci_high": float("nan"),
                "valid_fraction": float(valid.size) / n_resamples,
                "n_resamples": n_resamples}
    return {
        "ci_low": float(np.percentile(valid, 100 * alpha / 2)),
        "ci_high": float(np.percentile(valid, 100 * (1 - alpha / 2))),
        "valid_fraction": float(valid.size) / n_resamples,
        "n_resamples": n_resamples,
    }


def boundary_estimates(task_metrics: pd.DataFrame, cfg: BoundaryConfig,
                       value_col: str = "v_future", tag: str = "wql") -> pd.DataFrame:
    """One boundary estimate per (share, horizon) curve."""
    rows = []
    lams = np.array(sorted(task_metrics["lam"].unique()), dtype=float)
    for (share, horizon), g in task_metrics.groupby(
            ["nominal_covariate_share", "horizon"], sort=True):
        pivot = g.pivot_table(index=BOOTSTRAP_UNIT, columns="lam", values=value_col,
                              aggfunc="mean").reindex(columns=lams)
        if pivot.isna().any().any():
            raise RuntimeError(f"incomplete curve for share {share}, horizon {horizon}")
        cell_means = pivot.to_numpy().mean(axis=0)
        point, status = zero_crossing(lams, cell_means)
        boot = bootstrap_crossing(pivot.to_numpy(), lams, cfg.bootstrap.n_resamples,
                                  cfg.bootstrap.confidence_level,
                                  seed_parts=(cfg.experiment.master_seed, "crossing", tag,
                                              float(share), int(horizon)))
        rho = float(stats.spearmanr(lams, cell_means).statistic)
        rows.append({
            "metric": tag,
            "nominal_covariate_share": float(share),
            "horizon": int(horizon),
            "boundary_lambda": point,
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "ci_width": boot["ci_high"] - boot["ci_low"],
            "bootstrap_valid_fraction": boot["valid_fraction"],
            "status": status,
            "n_series": int(pivot.shape[0]),
            "spearman_rho": rho,
            "low_lambda_mean": float(cell_means[:2].mean()),
            "high_lambda_mean": float(cell_means[-2:].mean()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- run -----

def run_boundary_study(cfg: BoundaryConfig, predict_fn, log=lambda *_: None) -> pd.DataFrame:
    """Run the dense-lambda grid.  ``predict_fn(inputs) -> (H, Q)`` is injected."""
    pilot = cfg.to_pilot_config()
    exp, grid = cfg.experiment, cfg.grid
    q_levels = exp.quantile_levels
    origin = exp.primary_origin
    rows = []
    n_done, total = 0, (len(cfg.base_series_ids) * len(grid.nominal_covariate_share)
                        * len(grid.horizons) * len(grid.lambda_values))

    for b_id in cfg.base_series_ids:
        s = generate_base_series(b_id, pilot)
        for share in grid.nominal_covariate_share:
            share = float(share)
            y = build_target(s, share)
            for h in grid.horizons:
                item = f"b{b_id}_s{share:g}_h{h}"
                y_true = y[origin:origin + h]

                in1 = build_inputs(M1, item, y, s.x, origin, h, exp.context_length, exp.frequency)
                v_true = covariate_vintage(pilot, s, origin, h, 0.0)
                in2 = build_inputs(M2, item, y, s.x, origin, h, exp.context_length,
                                   exp.frequency, x_future=v_true["x_true"])
                # M1 and M2 do not depend on lambda: computed once, reused for every lambda
                q1 = predict_fn(in1, {"base_series_id": b_id, "nominal_covariate_share": share,
                                      "origin": origin, "horizon": h, "method": M1, "lam": -1.0})
                q2 = predict_fn(in2, {"base_series_id": b_id, "nominal_covariate_share": share,
                                      "origin": origin, "horizon": h, "method": M2, "lam": -1.0})
                w1, w2 = wql(y_true, q1, q_levels), wql(y_true, q2, q_levels)
                mse1, mse2 = mse(y_true, q1, q_levels), mse(y_true, q2, q_levels)

                for lam in grid.lambda_values:
                    lam = float(lam)
                    v = covariate_vintage(pilot, s, origin, h, lam)
                    in3 = build_inputs(M3, item, y, s.x, origin, h, exp.context_length,
                                       exp.frequency, x_future=v["x_tilde"])
                    assert_context_equality(in1, in3)
                    if lam == 0.0:
                        assert_future_equality(in2, in3)
                    q3 = predict_fn(in3, {"base_series_id": b_id,
                                          "nominal_covariate_share": share, "origin": origin,
                                          "horizon": h, "method": M3, "lam": lam})
                    w3 = wql(y_true, q3, q_levels)
                    mse3 = mse(y_true, q3, q_levels)
                    rows.append({
                        "base_series_id": b_id, "nominal_covariate_share": share,
                        "horizon": h, "origin": origin, "lam": lam,
                        "wql_m1": w1, "wql_m2": w2, "wql_m3": w3,
                        "mse_m1": mse1, "mse_m2": mse2, "mse_m3": mse3,
                        "nmae_m1": nmae(y_true, q1, q_levels),
                        "nmae_m3": nmae(y_true, q3, q_levels),
                        "crossing_m3": quantile_crossing_rate(q3),
                        "v_future": w1 - w3,
                        "v_future_mse": mse1 - mse3,
                        "v_oracle": w1 - w2,
                        "relative_delta_m3": relative_delta(w1, w3),
                        "harm_m3": int(is_harm(w1, w3, pilot.gates.harm_relative_threshold)),
                        "m3_wins": int(w3 < w1),
                        "realized_normalized_error_rms": v["realized_normalized_error_rms"],
                    })
                    n_done += 1
                    if n_done % 250 == 0:
                        log(f"  boundary: {n_done}/{total} M3 tasks")
    return pd.DataFrame(rows)


def curve_summary(task_metrics: pd.DataFrame, cfg: BoundaryConfig) -> pd.DataFrame:
    """Per (share, horizon, lambda) summary with paired bootstrap intervals."""
    from .bootstrap import paired_bootstrap
    rows = []
    for (share, horizon, lam), g in task_metrics.groupby(
            ["nominal_covariate_share", "horizon", "lam"], sort=True):
        b = paired_bootstrap(g[BOOTSTRAP_UNIT].to_numpy(), g["wql_m1"].to_numpy(),
                             g["wql_m3"].to_numpy(), cfg.bootstrap.n_resamples,
                             cfg.bootstrap.confidence_level,
                             seed_parts=(cfg.experiment.master_seed, "curve", share, horizon, lam))
        rows.append({
            "nominal_covariate_share": share, "horizon": horizon, "lam": lam,
            "n_series": int(g[BOOTSTRAP_UNIT].nunique()),
            "wql_m1": float(g["wql_m1"].mean()), "wql_m2": float(g["wql_m2"].mean()),
            "wql_m3": float(g["wql_m3"].mean()),
            "v_future_mean": b.mean_diff, "v_future_median": b.median_diff,
            "v_future_ci_low": b.ci_low, "v_future_ci_high": b.ci_high,
            "v_future_mc_se": b.monte_carlo_se, "v_future_sd": b.sd_diff,
            "m3_win_rate": b.win_rate,
            "harm_rate": float(g["harm_m3"].mean()),
            "mse_m1": float(g["mse_m1"].mean()), "mse_m3": float(g["mse_m3"].mean()),
            "v_future_mse_mean": float(g["v_future_mse"].mean()),
            "crossing_rate_m3": float(g["crossing_m3"].mean()),
        })
    return pd.DataFrame(rows)


def required_replications(cell_summary_df: pd.DataFrame, target_half_width: float) -> pd.DataFrame:
    """How many series a cell would need for a given CI half-width, at the observed SD.

    Reported when Gate E is INCONCLUSIVE.  Nothing is executed automatically.
    """
    rows = []
    for _, r in cell_summary_df.iterrows():
        sd, n = r["v_future_sd"], r["n_series"]
        need = int(np.ceil((1.96 * sd / target_half_width) ** 2)) if target_half_width > 0 else -1
        rows.append({"nominal_covariate_share": r["nominal_covariate_share"],
                     "horizon": r["horizon"], "lam": r["lam"], "n_series": n,
                     "observed_sd": sd, "monte_carlo_se": r["v_future_mc_se"],
                     "target_half_width": target_half_width,
                     "required_n_series": need,
                     "additional_series_needed": max(0, need - int(n))})
    return pd.DataFrame(rows)
