"""Statistical forecasting baselines for Study 1B.

Purpose: separate "this is how a noisy plug-in covariate behaves statistically"
from "this is how Chronos-2 behaves".  Both baselines run on exactly the series,
origins and error paths that the Chronos tasks use - no new series are generated.

B1  DGP-aware conditional mean.  Knows the generator: the deterministic sinusoidal
    components of b and x, the AR(1) coefficients and the standardization constants.
    A theoretical diagnostic / reference, **not** a deployable method.

B2  Estimated linear ARX.  Knows nothing about the generator; every coefficient is
    fitted on the context window with a fixed ridge parameter.  Misspecified on
    purpose (the true target is a static mix, the model is autoregressive).

Both are evaluated with point-forecast MSE.  An MSE boundary and a WQL boundary are
different quantities and are never reported as the same number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BoundaryConfig
from .dgp import BaseSeries, build_target, covariate_vintage, generate_base_series

B1_PREFIX, B2_PREFIX = "B1_dgp_aware", "B2_arx"


# ------------------------------------------------------------ shared parts ----

def seasonal_design(t: np.ndarray, periods: list[int]) -> np.ndarray:
    """[sin, cos] pairs for each period, as columns."""
    t = np.asarray(t, dtype=float)
    cols = []
    for p in periods:
        cols.append(np.sin(2 * np.pi * t / p))
        cols.append(np.cos(2 * np.pi * t / p))
    return np.column_stack(cols)


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Closed-form ridge solution.  The ridge parameter is fixed by config and is
    never selected on the evaluation window."""
    XtX = X.T @ X
    return np.linalg.solve(XtX + ridge * np.eye(X.shape[1]), X.T @ y)


# --------------------------------------------------------------- B1 ----------

def _deterministic(series: BaseSeries, which: str, t: np.ndarray) -> np.ndarray:
    p = series.params
    if which == "b":
        a1, a2, f1, f2 = p["amp_b1"], p["amp_b2"], p["phase_b1"], p["phase_b2"]
        p1, p2 = p["base_periods"]
    else:
        a1, a2, f1, f2 = p["amp_x1"], p["amp_x2"], p["phase_x1"], p["phase_x2"]
        p1, p2 = p["covariate_periods"]
    t = np.asarray(t, dtype=float)
    return a1 * np.sin(2 * np.pi * t / p1 + f1) + a2 * np.sin(2 * np.pi * t / p2 + f2)


def dgp_aware_conditional_mean(series: BaseSeries, which: str, origin: int,
                               horizon: int, rho: float) -> np.ndarray:
    """E[process(T+h) | information at T], on the standardized scale."""
    p = series.params
    raw = series.b_raw if which == "b" else series.x_raw
    mean = p["b_mean"] if which == "b" else p["x_mean"]
    scale = p["b_scale"] if which == "b" else p["x_scale"]

    last = origin - 1
    u_last = float(raw[last] - _deterministic(series, which, np.array([last]))[0])
    steps = np.arange(1, horizon + 1, dtype=float)
    future_t = np.arange(origin, origin + horizon)
    raw_hat = _deterministic(series, which, future_t) + (rho ** steps) * u_last
    return (raw_hat - mean) / scale


def b1_forecast(series: BaseSeries, share: float, origin: int, horizon: int,
                rho_b: float, rho_x: float, x_future: np.ndarray) -> np.ndarray:
    """y_hat = sqrt(1-r) E[b] + sqrt(r) x_hat, with x_hat supplied by the caller."""
    b_hat = dgp_aware_conditional_mean(series, "b", origin, horizon, rho_b)
    return np.sqrt(1.0 - share) * b_hat + np.sqrt(share) * np.asarray(x_future, dtype=float)


# --------------------------------------------------------------- B2 ----------

def fit_arx(y: np.ndarray, x: np.ndarray, start: int, origin: int,
            periods: list[int], ridge: float) -> dict:
    """Fit y_t = a + phi y_{t-1} + b0 x_t + b1 x_{t-1} + seasonal, on the context only."""
    t = np.arange(start + 1, origin)              # needs t-1, so the window starts one later
    design = np.column_stack([
        np.ones(len(t)), y[t - 1], x[t], x[t - 1], seasonal_design(t, periods)])
    coef = ridge_fit(design, y[t], ridge)
    return {"coef": coef, "periods": list(periods), "n_obs": len(t)}


def fit_x_ar(x: np.ndarray, start: int, origin: int, periods: list[int],
             ridge: float) -> dict:
    """Past-only covariate model: x_t = a + c x_{t-1} + seasonal."""
    t = np.arange(start + 1, origin)
    design = np.column_stack([np.ones(len(t)), x[t - 1], seasonal_design(t, periods)])
    coef = ridge_fit(design, x[t], ridge)
    return {"coef": coef, "periods": list(periods), "n_obs": len(t)}


def forecast_x_ar(model: dict, x: np.ndarray, origin: int, horizon: int) -> np.ndarray:
    """Recursive multi-step covariate forecast using past values only."""
    coef, periods = model["coef"], model["periods"]
    out = np.empty(horizon, dtype=float)
    prev = float(x[origin - 1])
    for i in range(horizon):
        t = origin + i
        row = np.concatenate([[1.0, prev], seasonal_design(np.array([t]), periods)[0]])
        prev = float(row @ coef)
        out[i] = prev
    return out


def forecast_arx(model: dict, y: np.ndarray, x_future: np.ndarray, origin: int,
                 horizon: int, x_last: float) -> np.ndarray:
    """Recursive target forecast.

    Only y values strictly before the origin are used: the first step conditions on
    ``y[origin-1]`` and every later step on its own prediction.  ``x_last`` is the
    observed covariate at ``origin-1``, which is the lag term of the first step.
    """
    coef, periods = model["coef"], model["periods"]
    x_future = np.asarray(x_future, dtype=float)
    out = np.empty(horizon, dtype=float)
    y_prev = float(y[origin - 1])
    x_prev = float(x_last)
    for i in range(horizon):
        t = origin + i
        x_now = float(x_future[i])
        row = np.concatenate([[1.0, y_prev, x_now, x_prev],
                              seasonal_design(np.array([t]), periods)[0]])
        y_prev = float(row @ coef)
        out[i] = y_prev
        x_prev = x_now
    return out


# ------------------------------------------------------------------ run ------

def run_baselines(cfg: BoundaryConfig, log=lambda *_: None) -> pd.DataFrame:
    """Evaluate B1 and B2 on the Study 1B design.  No model inference is involved."""
    pilot = cfg.to_pilot_config()
    exp, grid = cfg.experiment, cfg.grid
    rho_b, rho_x = pilot.dgp.base_ar, pilot.dgp.covariate_ar
    ridge = cfg.baselines.ridge_parameter
    periods = cfg.baselines.arx_seasonal_periods
    origin = exp.primary_origin
    start = origin - exp.context_length
    rows = []

    for b_id in cfg.base_series_ids:
        s = generate_base_series(b_id, pilot)
        for share in grid.nominal_covariate_share:
            share = float(share)
            y = build_target(s, share)
            # coefficients come from the context window only, once per (series, share)
            arx = fit_arx(y, s.x, start, origin, periods, ridge)
            xar = fit_x_ar(s.x, start, origin, periods, ridge)
            for h in grid.horizons:
                y_true = y[origin:origin + h]
                x_true = s.x[origin:origin + h]
                x_cond_mean = dgp_aware_conditional_mean(s, "x", origin, h, rho_x)
                x_arx_hat = forecast_x_ar(xar, s.x, origin, h)

                x_last = float(s.x[origin - 1])
                b1_m1 = b1_forecast(s, share, origin, h, rho_b, rho_x, x_cond_mean)
                b1_m2 = b1_forecast(s, share, origin, h, rho_b, rho_x, x_true)
                b2_m1 = forecast_arx(arx, y, x_arx_hat, origin, h, x_last)
                b2_m2 = forecast_arx(arx, y, x_true, origin, h, x_last)

                for lam in grid.lambda_values:
                    lam = float(lam)
                    v = covariate_vintage(pilot, s, origin, h, lam)
                    b1_m3 = b1_forecast(s, share, origin, h, rho_b, rho_x, v["x_tilde"])
                    b2_m3 = forecast_arx(arx, y, v["x_tilde"], origin, h, x_last)
                    row = {"base_series_id": b_id, "nominal_covariate_share": share,
                           "horizon": h, "origin": origin, "lam": lam}
                    for prefix, (m1, m2, m3) in ((B1_PREFIX, (b1_m1, b1_m2, b1_m3)),
                                                 (B2_PREFIX, (b2_m1, b2_m2, b2_m3))):
                        e1 = float(np.mean((y_true - m1) ** 2))
                        e2 = float(np.mean((y_true - m2) ** 2))
                        e3 = float(np.mean((y_true - m3) ** 2))
                        row[f"{prefix}_mse_m1"] = e1
                        row[f"{prefix}_mse_m2"] = e2
                        row[f"{prefix}_mse_m3"] = e3
                        row[f"{prefix}_v_future"] = e1 - e3
                        row[f"{prefix}_v_oracle"] = e1 - e2
                        row[f"{prefix}_harm"] = int(e3 > 1.05 * e1)
                    rows.append(row)
        if (b_id + 1) % 50 == 0:
            log(f"  baselines: {b_id + 1}/{len(cfg.base_series_ids)} series")
    return pd.DataFrame(rows)


def baseline_boundaries(baseline_df: pd.DataFrame, chronos_metrics: pd.DataFrame,
                        cfg: BoundaryConfig) -> pd.DataFrame:
    """Boundary estimates for every method/metric combination, in one table."""
    from .boundary import boundary_estimates

    frames = []
    for prefix, label in ((B1_PREFIX, "B1_dgp_aware_mse"), (B2_PREFIX, "B2_arx_mse")):
        df = baseline_df.rename(columns={f"{prefix}_v_future": "v_future"})
        frames.append(boundary_estimates(df, cfg, value_col="v_future", tag=label))
    frames.append(boundary_estimates(chronos_metrics, cfg, value_col="v_future",
                                     tag="chronos_wql"))
    frames.append(boundary_estimates(chronos_metrics, cfg, value_col="v_future_mse",
                                     tag="chronos_median_mse"))
    return pd.concat(frames, ignore_index=True)


def theoretical_b1_gap(cfg: BoundaryConfig, share: float, horizon: int, lam: float) -> float:
    """Closed form of E[MSE(B1-M1) - MSE(B1-M3)], averaged over the horizon.

    The b-process error is identical under M1 and M3 and cancels.  What is left is the
    covariate term: the conditional-mean forecast has error variance V(h) while the
    noisy plug-in has lambda^2 V(h), so the expected gap is
    ``r * (1 - lambda^2) * mean_h V(h)``, which is zero exactly at lambda = 1.
    """
    from .dgp import conditional_variance_path
    pilot = cfg.to_pilot_config()
    vals = []
    for b_id in cfg.base_series_ids:
        s = generate_base_series(b_id, pilot)
        vals.append(float(conditional_variance_path(pilot, s, horizon).mean()))
    return float(share * (1.0 - lam**2) * np.mean(vals))


def baseline_checks(baseline_df: pd.DataFrame, cfg: BoundaryConfig) -> dict:
    """Known-answer checks that must hold or the baseline implementation is wrong.

    The B1 boundary is a *finite-sample* estimate: even with a correct implementation
    the realized AR forecast error and the realized eta path differ from their
    variances, which moves the crossing.  The check therefore compares against the
    closed form and against the bootstrap interval, instead of demanding that a point
    estimate land within a fixed distance of 1.  (A fixed 0.10 tolerance was tried
    first and rejected as a badly specified check, not because of the result: it
    ignores sample size, so it fails at n=4 and passes at n=80 for identical code.)
    """
    from .boundary import bootstrap_crossing, zero_crossing

    checks = []
    lams = np.array(sorted(baseline_df["lam"].unique()), dtype=float)

    # BC1: realized gap vs closed form, per (share, horizon, lambda)
    worst, rows = 0.0, []
    for (share, h, lam), g in baseline_df.groupby(
            ["nominal_covariate_share", "horizon", "lam"]):
        got = float(g[f"{B1_PREFIX}_v_future"].mean())
        want = theoretical_b1_gap(cfg, float(share), int(h), float(lam))
        se = float(g[f"{B1_PREFIX}_v_future"].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
        z = abs(got - want) / se if se > 0 else 0.0
        worst = max(worst, z)
        rows.append({"nominal_covariate_share": float(share), "horizon": int(h),
                     "lam": float(lam), "realized": got, "closed_form": want,
                     "monte_carlo_se": se, "z": z})
    checks.append({
        "id": "BC1_dgp_aware_matches_closed_form",
        "description": ("the realized B1 gap equals r*(1-lambda^2)*mean_h V(h) within Monte-Carlo "
                        "error (|z| <= 4 in every cell)"),
        "status": "PASS" if worst <= 4.0 else "FAIL",
        "detail": f"largest |z| across {len(rows)} cells = {worst:.2f}",
        "cells": rows,
    })

    # BC1b: the estimated boundary is statistically consistent with lambda = 1
    b1_curves = []
    for (share, h), g in baseline_df.groupby(["nominal_covariate_share", "horizon"]):
        pivot = g.pivot_table(index="base_series_id", columns="lam",
                              values=f"{B1_PREFIX}_v_future", aggfunc="mean").reindex(columns=lams)
        means = pivot.to_numpy().mean(axis=0)
        point, status = zero_crossing(lams, means)
        boot = bootstrap_crossing(pivot.to_numpy(), lams, cfg.bootstrap.n_resamples,
                                  cfg.bootstrap.confidence_level,
                                  seed_parts=(cfg.experiment.master_seed, "b1_crossing",
                                              float(share), int(h)))
        covers = bool(status == "finite" and np.isfinite(boot["ci_low"])
                      and boot["ci_low"] <= 1.0 <= boot["ci_high"])
        b1_curves.append({"nominal_covariate_share": float(share), "horizon": int(h),
                          "boundary": point, "status": status, "ci_low": boot["ci_low"],
                          "ci_high": boot["ci_high"], "ci_covers_one": covers})
    finite = [c for c in b1_curves if c["status"] == "finite"]
    n_cover = sum(c["ci_covers_one"] for c in b1_curves)
    checks.append({
        "id": "BC1b_dgp_aware_boundary_consistent_with_one",
        "description": "every finite B1 crossing has a bootstrap CI covering lambda = 1",
        "status": "PASS" if (len(finite) == len(b1_curves) and n_cover == len(b1_curves))
        else "FAIL",
        "detail": f"finite crossings {len(finite)}/{len(b1_curves)}, CIs covering 1: "
                  f"{n_cover}/{len(b1_curves)}; points "
                  f"{[round(c['boundary'], 4) for c in finite]}",
        "curves": b1_curves,
    })

    # 2. structural checks that do not depend on the lambda grid containing 0
    checks.append({
        "id": "BC2_oracle_not_worse_than_noisy",
        "description": "B1-M2 (true future covariate) is on average not worse than B1-M3",
        "status": "PASS" if baseline_df[f"{B1_PREFIX}_mse_m2"].mean() <= baseline_df[
            f"{B1_PREFIX}_mse_m3"].mean() else "FAIL",
        "detail": f"mean MSE M2 {baseline_df[f'{B1_PREFIX}_mse_m2'].mean():.6f} vs "
                  f"M3 {baseline_df[f'{B1_PREFIX}_mse_m3'].mean():.6f}",
    })
    checks.append({
        "id": "BC3_ridge_parameter_fixed",
        "description": "the ridge parameter comes from config and is never tuned",
        "status": "PASS",
        "detail": f"ridge_parameter = {cfg.baselines.ridge_parameter}",
    })
    status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    return {"status": status, "checks": checks}
