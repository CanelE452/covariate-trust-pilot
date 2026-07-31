"""Study 2 - admission under time-varying reliability and an imperfect proxy.

Every selector chooses **one of M1 or M3**.  Quantile forecasts are never blended.

Information available to each selector:
  D0, D1, D2   no decision / oracle upper bound
  D3, D4       history only (pseudo-origins strictly before the primary origin)
  D5           the reported current uncertainty proxy only
  D6, D7       history plus the reported proxy

No selector may read the current future target or (outside the P0 oracle diagnostic)
the true current lambda.  That is enforced by construction here and asserted in the
tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import BOOTSTRAP_UNIT
from .config import DynamicConfig
from .dgp import build_target, covariate_vintage, estimate_lambda_hat, generate_base_series
from .metrics import is_harm, nmae, quantile_crossing_rate, wql
from .reliability_schedules import (P0_ORACLE, P4_STALE, PROXY_MODES, lambda_at,
                                    proxy_record, reported_lambda, schedule_origins)
from .schemas import M1, M3, assert_context_equality, build_inputs

D0 = "D0_always_no_future"
D1 = "D1_always_use_future"
D2 = "D2_oracle_per_task"
D3 = "D3_history_utility"
D4 = "D4_history_reliability"
D5 = "D5_current_proxy"
D6 = "D6_hybrid_conservative"
D7 = "D7_hybrid_override"

SELECTORS = (D0, D1, D2, D3, D4, D5, D6, D7)
FIXED_POLICIES = (D0, D1)
HISTORY_ONLY = (D3, D4)
PROXY_USING = (D5, D6, D7)


# ------------------------------------------------------------------ run ------

def run_dynamic_study(cfg: DynamicConfig, predict_fn, log=lambda *_: None) -> pd.DataFrame:
    """Run every (schedule, share, horizon, series) at all origins.

    ``predict_fn(inputs, meta) -> (H, Q)`` is injected and is expected to cache by
    content hash, so schedules that request an identical (origin, lambda) task reuse
    one inference instead of repeating it.
    """
    pilot = cfg.to_pilot_config()
    exp, grid = cfg.experiment, cfg.grid
    q_levels = exp.quantile_levels
    rows = []
    n_done = 0
    total = (len(cfg.schedules) * len(grid.nominal_covariate_share) * len(grid.horizons)
             * len(cfg.base_series_ids))

    for b_id in cfg.base_series_ids:
        s = generate_base_series(b_id, pilot)
        for share in grid.nominal_covariate_share:
            share = float(share)
            y = build_target(s, share)
            for h in grid.horizons:
                hist_origins, primary = schedule_origins(cfg, h)
                for sched in cfg.schedules:
                    hist_rows = []
                    for origin in hist_origins:
                        lam = lambda_at(cfg, sched.name, h, origin)
                        ctx_len = min(exp.context_length, origin)
                        item = f"b{b_id}_s{share:g}_h{h}_o{origin}"
                        y_true = y[origin:origin + h]
                        in1 = build_inputs(M1, item, y, s.x, origin, h, ctx_len, exp.frequency)
                        v = covariate_vintage(pilot, s, origin, h, lam)
                        in3 = build_inputs(M3, item, y, s.x, origin, h, ctx_len,
                                           exp.frequency, x_future=v["x_tilde"])
                        assert_context_equality(in1, in3)
                        meta = {"base_series_id": b_id, "nominal_covariate_share": share,
                                "origin": origin, "horizon": h}
                        q1 = predict_fn(in1, {**meta, "method": M1, "lam": -1.0})
                        q3 = predict_fn(in3, {**meta, "method": M3, "lam": lam})
                        hist_rows.append({
                            "origin": origin, "lam": lam,
                            "wql_m1": wql(y_true, q1, q_levels),
                            "wql_m3": wql(y_true, q3, q_levels),
                            "lambda_hat": estimate_lambda_hat(v["error"], v["V"]),
                        })

                    lam_cur = lambda_at(cfg, sched.name, h, primary)
                    item = f"b{b_id}_s{share:g}_h{h}_o{primary}"
                    y_true = y[primary:primary + h]
                    in1 = build_inputs(M1, item, y, s.x, primary, h, exp.context_length,
                                       exp.frequency)
                    v_cur = covariate_vintage(pilot, s, primary, h, lam_cur)
                    in3 = build_inputs(M3, item, y, s.x, primary, h, exp.context_length,
                                       exp.frequency, x_future=v_cur["x_tilde"])
                    assert_context_equality(in1, in3)
                    meta = {"base_series_id": b_id, "nominal_covariate_share": share,
                            "origin": primary, "horizon": h}
                    q1 = predict_fn(in1, {**meta, "method": M1, "lam": -1.0})
                    q3 = predict_fn(in3, {**meta, "method": M3, "lam": lam_cur})
                    w1, w3 = wql(y_true, q1, q_levels), wql(y_true, q3, q_levels)

                    hist = pd.DataFrame(hist_rows)
                    rows.append({
                        "base_series_id": b_id, "nominal_covariate_share": share,
                        "horizon": h, "schedule": sched.name, "origin": primary,
                        "true_current_lambda": lam_cur,
                        "hist_wql_m1": float(hist["wql_m1"].mean()),
                        "hist_wql_m3": float(hist["wql_m3"].mean()),
                        "hist_lambda_hat": float(hist["lambda_hat"].mean()),
                        "hist_lambda_hat_last": float(hist["lambda_hat"].iloc[-1]),
                        "hist_lambda_true_mean": float(hist["lam"].mean()),
                        "wql_m1": w1, "wql_m3": w3,
                        "nmae_m1": nmae(y_true, q1, q_levels),
                        "nmae_m3": nmae(y_true, q3, q_levels),
                        "crossing_m3": quantile_crossing_rate(q3),
                        "wql_oracle": min(w1, w3),
                        "m3_is_better": int(w3 < w1),
                        "harm_m3": int(is_harm(w1, w3, pilot.gates.harm_relative_threshold)),
                        "realized_normalized_error_rms": v_cur["realized_normalized_error_rms"],
                    })
                    n_done += 1
                    if n_done % 100 == 0:
                        log(f"  dynamic: {n_done}/{total} (schedule, cell, series) tasks")
    return pd.DataFrame(rows)


# ------------------------------------------------------------- decisions ----

def build_proxy_table(tasks: pd.DataFrame, cfg: DynamicConfig) -> pd.DataFrame:
    """Reported lambda for every task and proxy mode.

    The forecasts themselves are unaffected: a proxy only changes the decision, never
    the model input, so no inference is repeated per proxy mode.
    """
    rows = []
    for _, t in tasks.iterrows():
        # historical lambda estimates are observable in the past: both the historical
        # covariate forecast and the realized covariate are known there
        hist_est = [t["hist_lambda_hat"]]
        for mode in PROXY_MODES:
            rep = reported_lambda(
                cfg, mode,
                base_series_id=int(t[BOOTSTRAP_UNIT]),
                share=float(t["nominal_covariate_share"]),
                horizon=int(t["horizon"]),
                schedule_name=str(t["schedule"]),
                true_current_lambda=float(t["true_current_lambda"]),
                historical_lambda_estimates=hist_est)
            rows.append({
                BOOTSTRAP_UNIT: int(t[BOOTSTRAP_UNIT]),
                "nominal_covariate_share": float(t["nominal_covariate_share"]),
                "horizon": int(t["horizon"]), "schedule": str(t["schedule"]),
                **proxy_record(cfg, mode, rep, float(t["true_current_lambda"])),
            })
    return pd.DataFrame(rows)


def apply_selectors(tasks: pd.DataFrame, proxies: pd.DataFrame,
                    cfg: DynamicConfig) -> pd.DataFrame:
    """Long table: one row per (task, proxy_mode, selector)."""
    th = cfg.selector_thresholds
    keys = [BOOTSTRAP_UNIT, "nominal_covariate_share", "horizon", "schedule"]
    merged = tasks.merge(proxies, on=keys, how="inner", validate="one_to_many")
    if len(merged) != len(tasks) * len(PROXY_MODES):
        raise RuntimeError("proxy table does not cover every task exactly once per mode")

    hist_supports_m3 = merged["hist_wql_m3"] < merged["hist_wql_m1"]
    hist_reliable = merged["hist_lambda_hat"] < th.use_threshold
    proxy_ok = merged["reported_lambda"] < th.use_threshold

    choice = {
        D0: np.full(len(merged), M1),
        D1: np.full(len(merged), M3),
        D2: np.where(merged["wql_m3"] < merged["wql_m1"], M3, M1),
        D3: np.where(hist_supports_m3, M3, M1),
        D4: np.where(hist_reliable, M3, M1),
        D5: np.where(proxy_ok, M3, M1),
        D6: np.where(hist_supports_m3 & proxy_ok, M3, M1),
        D7: np.where(merged["reported_lambda"] < th.override_low, M3,
                     np.where(merged["reported_lambda"] > th.override_high, M1,
                              np.where(hist_supports_m3, M3, M1))),
    }

    out = []
    for sel in SELECTORS:
        d = merged.copy()
        d["selector"] = sel
        d["choice"] = choice[sel]
        d["wql_selected"] = np.where(d["choice"] == M3, d["wql_m3"], d["wql_m1"])
        d["nmae_selected"] = np.where(d["choice"] == M3, d["nmae_m3"], d["nmae_m1"])
        d["regret"] = d["wql_selected"] - d["wql_oracle"]
        d["chose_m3"] = (d["choice"] == M3).astype(int)
        d["harm"] = [int(is_harm(b, s, 0.05)) for s, b in zip(d["wql_selected"], d["wql_m1"])]
        # false-use: M1 was actually better but M3 was chosen; false-reject: the reverse
        d["false_use"] = ((d["m3_is_better"] == 0) & (d["choice"] == M3)).astype(int)
        d["false_reject"] = ((d["m3_is_better"] == 1) & (d["choice"] == M1)).astype(int)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def false_rates(decisions: pd.DataFrame) -> dict:
    """Conditional rates: denominators are the cases where each error is possible."""
    n_m1_better = int((decisions["m3_is_better"] == 0).sum())
    n_m3_better = int((decisions["m3_is_better"] == 1).sum())
    return {
        "false_use_rate": float(decisions["false_use"].sum() / n_m1_better) if n_m1_better else float("nan"),
        "false_reject_rate": float(decisions["false_reject"].sum() / n_m3_better) if n_m3_better else float("nan"),
        "n_m1_better": n_m1_better,
        "n_m3_better": n_m3_better,
    }


def _summarize(g: pd.DataFrame) -> dict:
    oracle = float(g["wql_oracle"].mean())
    m1, m3 = float(g["wql_m1"].mean()), float(g["wql_m3"].mean())
    best_fixed = min(m1, m3)
    sel = float(g["wql_selected"].mean())
    gap = best_fixed - oracle
    return {
        "n_tasks": int(len(g)),
        "n_series": int(g[BOOTSTRAP_UNIT].nunique()),
        "mean_wql": sel,
        "median_nmae": float(g["nmae_selected"].median()),
        "mean_regret": float(g["regret"].mean()),
        "m3_choice_rate": float(g["chose_m3"].mean()),
        "harm_rate": float(g["harm"].mean()),
        "win_rate_vs_best_fixed": float((g["wql_selected"] < (
            g["wql_m1"] if m1 <= m3 else g["wql_m3"])).mean()),
        "mean_wql_m1": m1, "mean_wql_m3": m3, "mean_wql_oracle": oracle,
        "best_fixed_mean": best_fixed,
        "relative_improvement_over_best_fixed": (best_fixed - sel) / best_fixed,
        "oracle_gap_recovery": (best_fixed - sel) / gap if gap > 0 else float("nan"),
        **false_rates(g),
    }


def condition_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Per (schedule, proxy mode, selector).  Never collapse conditions into one mean."""
    rows = []
    for (sched, mode, sel), g in decisions.groupby(["schedule", "proxy_mode", "selector"],
                                                   sort=True):
        rows.append({"schedule": sched, "proxy_mode": mode, "selector": sel, **_summarize(g)})
    return pd.DataFrame(rows)


def proxy_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Per (proxy mode, selector), pooled over schedules."""
    rows = []
    for (mode, sel), g in decisions.groupby(["proxy_mode", "selector"], sort=True):
        rows.append({"proxy_mode": mode, "selector": sel,
                     "mean_calibration_ratio": float(g["calibration_ratio"].mean()),
                     **_summarize(g)})
    return pd.DataFrame(rows)


def proxy_calibration_summary(proxies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, sched), g in proxies.groupby(["proxy_mode", "schedule"], sort=True):
        rows.append({
            "proxy_mode": mode, "schedule": sched,
            "mean_true_lambda": float(g["true_current_lambda"].mean()),
            "mean_reported_lambda": float(g["reported_lambda"].mean()),
            "mean_absolute_error": float(g["absolute_error"].mean()),
            "mean_relative_error": float(g["relative_error"].mean()),
            "mean_calibration_ratio": float(g["calibration_ratio"].mean()),
            "uses_true_current_lambda": bool(g["uses_true_current_lambda"].iloc[0]),
        })
    return pd.DataFrame(rows)
