"""Synthetic data generating process and covariate forecast-error model.

No external data is ever read.  Everything below is generated from the master
seed via :mod:`covariate_trust.seeds`.

Design notes
------------
* ``b`` (base process) and ``x`` (covariate process) are generated once per
  ``base_series_id`` and reused across every grid cell (common random numbers).
  Only the mixing weight ``r`` and the error multiplier ``lambda`` vary.
* Standardization statistics use the window ``[0, standardization_end)`` only.
  No forecast-window information enters the scaling.
* ``nominal_covariate_share`` is the *nominal* variance weight of ``x`` in ``y``
  by construction; it is deliberately not called a partial R^2.  The realized
  incremental R^2 is measured separately and stored in the metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PilotConfig
from .seeds import make_rng, stable_seed

AMPLITUDE_RANGE = (0.5, 1.5)


@dataclass(frozen=True)
class BaseSeries:
    """One realization of the shared (b, x) pair, standardized on the early window."""

    base_series_id: int
    b: np.ndarray            # standardized base process
    x: np.ndarray            # standardized covariate process
    b_raw: np.ndarray
    x_raw: np.ndarray
    params: dict

    @property
    def scale_x(self) -> float:
        return float(self.params["x_scale"])


def _ar1(rho: float, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) path: u_t = rho * u_{t-1} + eps_t, eps ~ N(0, sigma^2)."""
    eps = rng.normal(0.0, sigma, size=n)
    u = np.empty(n, dtype=float)
    # start from the stationary distribution so there is no burn-in transient
    u[0] = eps[0] / np.sqrt(1.0 - rho**2)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + eps[t]
    return u


def generate_base_series(base_series_id: int, cfg: PilotConfig) -> BaseSeries:
    """Generate the (b, x) pair for one ``base_series_id``.

    Independent of ``nominal_covariate_share`` and ``lambda`` by construction,
    which is what makes the grid comparisons paired.
    """
    exp, dg = cfg.experiment, cfg.dgp
    n = exp.series_length
    t = np.arange(n, dtype=float)

    prng = make_rng(exp.master_seed, "series_params", base_series_id)
    amp = prng.uniform(*AMPLITUDE_RANGE, size=4)
    phase = prng.uniform(0.0, 2.0 * np.pi, size=4)

    # innovations for b and x come from separate, independent streams
    u_b = _ar1(dg.base_ar, dg.ar_innovation_std, n,
               make_rng(exp.master_seed, "innov_base", base_series_id))
    u_x = _ar1(dg.covariate_ar, dg.ar_innovation_std, n,
               make_rng(exp.master_seed, "innov_cov", base_series_id))

    pb1, pb2 = dg.base_periods
    px1, px2 = dg.covariate_periods
    b_raw = (amp[0] * np.sin(2 * np.pi * t / pb1 + phase[0])
             + amp[1] * np.sin(2 * np.pi * t / pb2 + phase[1])
             + u_b)
    x_raw = (amp[2] * np.sin(2 * np.pi * t / px1 + phase[2])
             + amp[3] * np.sin(2 * np.pi * t / px2 + phase[3])
             + u_x)

    w = exp.standardization_end
    b_mean, b_scale = float(b_raw[:w].mean()), float(b_raw[:w].std(ddof=0))
    x_mean, x_scale = float(x_raw[:w].mean()), float(x_raw[:w].std(ddof=0))
    if b_scale <= 0 or x_scale <= 0:
        raise RuntimeError(f"degenerate standardization scale for series {base_series_id}")

    b = (b_raw - b_mean) / b_scale
    x = (x_raw - x_mean) / x_scale

    params = {
        "base_series_id": base_series_id,
        "amp_b1": float(amp[0]), "amp_b2": float(amp[1]),
        "amp_x1": float(amp[2]), "amp_x2": float(amp[3]),
        "phase_b1": float(phase[0]), "phase_b2": float(phase[1]),
        "phase_x1": float(phase[2]), "phase_x2": float(phase[3]),
        "base_ar": dg.base_ar, "covariate_ar": dg.covariate_ar,
        "ar_innovation_std": dg.ar_innovation_std,
        "base_periods": list(dg.base_periods), "covariate_periods": list(dg.covariate_periods),
        "b_mean": b_mean, "b_scale": b_scale, "x_mean": x_mean, "x_scale": x_scale,
        "standardization_end": w,
        "seed_series_params": stable_seed(exp.master_seed, "series_params", base_series_id),
        "seed_innov_base": stable_seed(exp.master_seed, "innov_base", base_series_id),
        "seed_innov_cov": stable_seed(exp.master_seed, "innov_cov", base_series_id),
    }
    return BaseSeries(base_series_id, b, x, b_raw, x_raw, params)


def build_target(series: BaseSeries, nominal_covariate_share: float) -> np.ndarray:
    """y_t = sqrt(1-r) * b_t + sqrt(r) * x_t."""
    r = float(nominal_covariate_share)
    return np.sqrt(1.0 - r) * series.b + np.sqrt(r) * series.x


def _incremental_r2(y: np.ndarray, b: np.ndarray, x: np.ndarray) -> dict:
    """R^2 of x alone, and the incremental R^2 of x given b, by OLS."""
    n = len(y)
    ones = np.ones(n)

    def _r2(design: np.ndarray) -> tuple[float, float]:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), ss_res / n)

    r2_x, _ = _r2(np.column_stack([ones, x]))
    r2_b, _ = _r2(np.column_stack([ones, b]))
    r2_bx, resid_var_bx = _r2(np.column_stack([ones, b, x]))
    return {
        "r2_x_alone": r2_x,
        "r2_b_alone": r2_b,
        "r2_b_and_x": r2_bx,
        "realized_incremental_r2": r2_bx - r2_b,
        "regression_residual_variance": resid_var_bx,
    }


def series_metadata(series: BaseSeries, y: np.ndarray, nominal_covariate_share: float,
                    cfg: PilotConfig) -> dict:
    """Per-(series, share) diagnostics recorded regardless of what they show."""
    exp = cfg.experiment
    o, c = exp.primary_origin, exp.context_length
    ctx = slice(o - c, o)
    meta = dict(series.params)
    meta.update({
        "nominal_covariate_share": float(nominal_covariate_share),
        "context_corr_x_y": float(np.corrcoef(series.x[ctx], y[ctx])[0, 1]),
        "full_corr_x_y": float(np.corrcoef(series.x, y)[0, 1]),
        "context_corr_b_x": float(np.corrcoef(series.b[ctx], series.x[ctx])[0, 1]),
        "primary_origin": o,
        "context_length": c,
        "series_length": exp.series_length,
    })
    meta.update(_incremental_r2(y, series.b, series.x))
    return meta


# --------------------------------------------------------------------------
# covariate forecast error
# --------------------------------------------------------------------------

def conditional_variance_raw(rho: float, sigma: float, h: int) -> float:
    """h-step conditional variance of the AR(1) residual, raw scale."""
    return float(sigma**2 * (1.0 - rho ** (2 * h)) / (1.0 - rho**2))


def conditional_variance_path(cfg: PilotConfig, series: BaseSeries, horizon: int) -> np.ndarray:
    """V(h) for h = 1..horizon on the standardized scale."""
    dg = cfg.dgp
    raw = np.array([conditional_variance_raw(dg.covariate_ar, dg.ar_innovation_std, h)
                    for h in range(1, horizon + 1)])
    return raw / series.scale_x**2


def eta_path(cfg: PilotConfig, base_series_id: int, origin: int, horizon: int) -> np.ndarray:
    """Standard-normal error path shared by every lambda at this (series, origin, h)."""
    rng = make_rng(cfg.experiment.master_seed, "eta", base_series_id, origin, horizon)
    return rng.normal(0.0, 1.0, size=horizon)


def covariate_vintage(cfg: PilotConfig, series: BaseSeries, origin: int, horizon: int,
                      lam: float) -> dict:
    """Build x_tilde for one (series, origin, horizon, lambda)."""
    v = conditional_variance_path(cfg, series, horizon)
    eta = eta_path(cfg, series.base_series_id, origin, horizon)
    err = lam * np.sqrt(v) * eta
    x_true = series.x[origin:origin + horizon]
    x_tilde = x_true + err
    scale = series.scale_x
    return {
        "x_true": x_true,
        "x_tilde": x_tilde,
        "error": err,
        "V": v,
        "eta": eta,
        "eta_path_id": f"b{series.base_series_id}_o{origin}_h{horizon}",
        "requested_lambda": float(lam),
        "realized_normalized_error_rms": float(np.sqrt(np.mean((err / np.sqrt(v)) ** 2))),
        "raw_rmse": float(np.sqrt(np.mean((err * scale) ** 2))),
        "standardized_rmse": float(np.sqrt(np.mean(err**2))),
        "error_bias": float(err.mean()),
        "error_autocorr_lag1": float(
            np.corrcoef(err[:-1], err[1:])[0, 1]) if horizon > 2 and np.std(err) > 0 else float("nan"),
    }


def estimate_lambda_hat(error: np.ndarray, v: np.ndarray) -> float:
    """Method-of-moments lambda estimate from a realized error path."""
    z = error / np.sqrt(v)
    return float(np.sqrt(np.mean(z**2)))


# --------------------------------------------------------------------------
# dataset assembly
# --------------------------------------------------------------------------

def generate_dataset(cfg: PilotConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, BaseSeries]]:
    """Generate every series/share combination plus its metadata."""
    rows, meta_rows = [], []
    series_map: dict[int, BaseSeries] = {}
    for b_id in cfg.base_series_ids:
        s = generate_base_series(b_id, cfg)
        series_map[b_id] = s
        for r in cfg.grid.nominal_covariate_share:
            y = build_target(s, r)
            rows.append(pd.DataFrame({
                "base_series_id": b_id,
                "nominal_covariate_share": float(r),
                "t": np.arange(cfg.experiment.series_length),
                "b": s.b,
                "x": s.x,
                "y": y,
            }))
            meta_rows.append(series_metadata(s, y, r, cfg))
    return (pd.concat(rows, ignore_index=True),
            pd.DataFrame(meta_rows),
            series_map)


def vintage_table(cfg: PilotConfig, series_map: dict[int, BaseSeries],
                  origins: list[int] | None = None) -> pd.DataFrame:
    """Long table of every covariate vintage actually used."""
    origins = origins or [cfg.experiment.primary_origin]
    rows = []
    for b_id, s in series_map.items():
        for origin in origins:
            for h in cfg.grid.horizons:
                for lam in cfg.grid.lambda_values:
                    v = covariate_vintage(cfg, s, origin, h, lam)
                    rows.append({
                        "base_series_id": b_id,
                        "origin": origin,
                        "horizon": h,
                        "lam": float(lam),
                        "eta_path_id": v["eta_path_id"],
                        "requested_lambda": v["requested_lambda"],
                        "realized_normalized_error_rms": v["realized_normalized_error_rms"],
                        "standardized_rmse": v["standardized_rmse"],
                        "raw_rmse": v["raw_rmse"],
                        "error_bias": v["error_bias"],
                        "error_autocorr_lag1": v["error_autocorr_lag1"],
                        "lambda_hat": estimate_lambda_hat(v["error"], v["V"]),
                        "x_true": v["x_true"].tolist(),
                        "x_tilde": v["x_tilde"].tolist(),
                    })
    return pd.DataFrame(rows)
