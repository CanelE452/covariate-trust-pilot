"""Time-varying covariate reliability and the reported-uncertainty proxy.

Study 2 removes the easy assumption behind Gate D, where the pseudo-origins and the
primary origin shared one lambda.  Here each origin carries its own lambda taken from
a schedule declared in the config, and the selector never sees the true current
lambda: it sees a *reported* proxy with its own error model.

Two independent random streams are used:
  * ``"eta"``   - the covariate forecast error path (already used by the coarse pilot)
  * ``"proxy"`` - the reported-uncertainty noise
They must not share a namespace, otherwise the selector's information would be
correlated with the very error it is supposed to be judging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .admission import PSEUDO_ORIGINS
from .config import DynamicConfig
from .seeds import make_rng

P0_ORACLE = "P0_oracle_current"
P1_CALIBRATED = "P1_calibrated_noisy"
P2_OVERCONFIDENT = "P2_overconfident"
P3_UNDERCONFIDENT = "P3_underconfident"
P4_STALE = "P4_stale_history"
PROXY_MODES = (P0_ORACLE, P1_CALIBRATED, P2_OVERCONFIDENT, P3_UNDERCONFIDENT, P4_STALE)

# The only mode allowed to read the true current lambda.  Everything else is a
# diagnostic-free deployment analogue.
ORACLE_MODES = frozenset({P0_ORACLE})


def schedule_origins(cfg: DynamicConfig, horizon: int) -> tuple[list[int], int]:
    """(historical pseudo-origins, primary origin) for a horizon."""
    hist = list(PSEUDO_ORIGINS[horizon])
    if len(hist) != cfg.n_historical_origins:
        raise ValueError(
            f"horizon {horizon} has {len(hist)} pseudo-origins but the schedules declare "
            f"{cfg.n_historical_origins} historical lambdas")
    primary = cfg.experiment.primary_origin
    for o in hist:
        if o + horizon > primary:
            raise AssertionError(
                f"pseudo-origin {o} with horizon {horizon} would reach past the primary "
                f"origin {primary}: future target leakage")
    return hist, primary


def lambda_at(cfg: DynamicConfig, schedule_name: str, horizon: int, origin: int) -> float:
    """The true lambda that applies at this origin under this schedule.

    Historical and current lambdas are never mixed or averaged: each origin gets the
    single value the schedule assigns to it.
    """
    sched = next((s for s in cfg.schedules if s.name == schedule_name), None)
    if sched is None:
        raise KeyError(f"unknown schedule {schedule_name!r}")
    hist, primary = schedule_origins(cfg, horizon)
    if origin == primary:
        return float(sched.current)
    if origin in hist:
        return float(sched.historical[hist.index(origin)])
    raise KeyError(f"origin {origin} is not part of the horizon-{horizon} schedule")


def schedule_table(cfg: DynamicConfig) -> pd.DataFrame:
    """Flat, inspectable view of every (schedule, horizon, origin) -> lambda."""
    rows = []
    for s in cfg.schedules:
        for h in cfg.grid.horizons:
            hist, primary = schedule_origins(cfg, h)
            for i, o in enumerate(hist):
                rows.append({"schedule": s.name, "horizon": h, "origin": o,
                             "origin_index": i, "is_primary": False,
                             "true_lambda": float(s.historical[i])})
            rows.append({"schedule": s.name, "horizon": h, "origin": primary,
                         "origin_index": len(hist), "is_primary": True,
                         "true_lambda": float(s.current)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ proxy ----

def calibrated_proxy(cfg: DynamicConfig, base_series_id: int, share: float, horizon: int,
                     schedule_name: str, true_lambda: float) -> float:
    """Median-unbiased lognormal proxy: lambda * exp(z - sigma^2/2), z ~ N(0, sigma^2).

    Drawn from the "proxy" namespace, which is disjoint from the "eta" namespace used
    for the covariate forecast error.
    """
    sigma = cfg.proxy.sigma_proxy
    rng = make_rng(cfg.experiment.master_seed, "proxy", base_series_id, round(float(share), 6),
                   int(horizon), schedule_name)
    z = float(rng.normal(0.0, sigma))
    return float(true_lambda * np.exp(z - 0.5 * sigma**2))


def reported_lambda(cfg: DynamicConfig, mode: str, *, base_series_id: int, share: float,
                    horizon: int, schedule_name: str, true_current_lambda: float,
                    historical_lambda_estimates: list[float]) -> float:
    """What the forecast provider reports to the selector for the current origin.

    ``true_current_lambda`` is only consumed by P0.  P1-P3 use it solely to *build*
    the noisy proxy value here inside the provider; the selector receives the returned
    number and never the truth.  P4 ignores it entirely.
    """
    if mode not in PROXY_MODES:
        raise KeyError(f"unknown proxy mode {mode!r}")
    if mode == P0_ORACLE:
        return float(true_current_lambda)
    if mode == P4_STALE:
        if not historical_lambda_estimates:
            raise ValueError("P4 requires historical lambda estimates")
        return float(np.mean(historical_lambda_estimates))
    base = calibrated_proxy(cfg, base_series_id, share, horizon, schedule_name,
                            true_current_lambda)
    if mode == P1_CALIBRATED:
        return base
    if mode == P2_OVERCONFIDENT:
        return float(cfg.proxy.overconfident_multiplier * base)
    return float(cfg.proxy.underconfident_multiplier * base)


def proxy_record(cfg: DynamicConfig, mode: str, reported: float, true_lambda: float) -> dict:
    return {
        "proxy_mode": mode,
        "true_current_lambda": float(true_lambda),
        "reported_lambda": float(reported),
        "absolute_error": float(abs(reported - true_lambda)),
        "relative_error": float((reported - true_lambda) / true_lambda) if true_lambda > 0
        else float("nan"),
        "calibration_ratio": float(reported / true_lambda) if true_lambda > 0 else float("nan"),
        "uses_true_current_lambda": mode in ORACLE_MODES,
    }


# Namespace constants, kept next to the two draws they label so a future edit that
# merges them is visible here.  `dgp.eta_path` draws from ETA_NAMESPACE and
# `calibrated_proxy` from PROXY_NAMESPACE; they must never coincide, otherwise the
# selector's reported uncertainty would be correlated with the error it judges.
ETA_NAMESPACE = "eta"
PROXY_NAMESPACE = "proxy"


def eta_namespace_check() -> dict:
    return {"eta": ETA_NAMESPACE, "proxy": PROXY_NAMESPACE,
            "disjoint": ETA_NAMESPACE != PROXY_NAMESPACE}
