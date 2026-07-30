"""Historical admission diagnostic and Gate D.

Only run when Gates A, B and C all PASS.

The selectors may only use information available *before* the primary forecast
origin.  Concretely: every pseudo-origin satisfies
``pseudo_origin + horizon <= primary_origin``, so no primary future target value
can ever enter a selection decision.  This is asserted, not assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import BOOTSTRAP_UNIT, paired_bootstrap
from .config import PilotConfig
from .dgp import BaseSeries, build_target, covariate_vintage, estimate_lambda_hat
from .gates import FAIL, INCONCLUSIVE, NOT_RUN, PASS
from .metrics import is_harm, quantile_crossing_rate, wql
from .schemas import M1, M3, build_inputs

PSEUDO_ORIGINS: dict[int, list[int]] = {
    24: [800, 824, 848, 872],
    96: [512, 608, 704, 800],
}

A1 = "A1_historical_utility"
A2 = "A2_historical_reliability"
SELECTORS = (A1, A2)


def pseudo_origins(horizon: int, cfg: PilotConfig) -> list[int]:
    if horizon not in PSEUDO_ORIGINS:
        raise ValueError(f"no pseudo-origins defined for horizon {horizon}")
    origins = PSEUDO_ORIGINS[horizon]
    assert_no_primary_leak(origins, horizon, cfg)
    return origins


def assert_no_primary_leak(origins: list[int], horizon: int, cfg: PilotConfig) -> None:
    """Every historical window must close at or before the primary origin."""
    primary = cfg.experiment.primary_origin
    for o in origins:
        if o + horizon > primary:
            raise AssertionError(
                f"pseudo-origin {o} with horizon {horizon} reaches t={o + horizon}, "
                f"which is beyond the primary origin {primary}: future target leakage"
            )
        if o - min(cfg.experiment.context_length, o) < 0:
            raise AssertionError(f"pseudo-origin {o} has no usable context")


def historical_tasks(cfg: PilotConfig) -> list[dict]:
    """Enumerate every historical forecast task (method-agnostic key list)."""
    tasks = []
    for h in cfg.grid.horizons:
        for o in pseudo_origins(h, cfg):
            for b in cfg.base_series_ids:
                for share in cfg.grid.nominal_covariate_share:
                    tasks.append({"base_series_id": b, "nominal_covariate_share": float(share),
                                  "horizon": int(h), "origin": int(o)})
    return tasks


def run_historical(cfg: PilotConfig, series_map: dict[int, BaseSeries], predict_fn,
                   log=lambda *_: None) -> pd.DataFrame:
    """Run M1 and M3 at every pseudo-origin.

    ``predict_fn(inputs) -> (H, Q) quantile matrix`` is injected so this module
    stays testable without the model.
    """
    exp = cfg.experiment
    q_levels = exp.quantile_levels
    rows = []
    total = 0
    for task in historical_tasks(cfg):
        b, share = task["base_series_id"], task["nominal_covariate_share"]
        h, origin = task["horizon"], task["origin"]
        s = series_map[b]
        y = build_target(s, share)
        ctx_len = min(exp.context_length, origin)
        y_true = y[origin:origin + h]

        m1 = build_inputs(M1, f"b{b}_s{share}_h{h}_o{origin}", y, s.x, origin, h, ctx_len,
                          exp.frequency)
        q_m1 = predict_fn(m1)
        wql_m1 = wql(y_true, q_m1, q_levels)

        for lam in cfg.grid.lambda_values:
            v = covariate_vintage(cfg, s, origin, h, float(lam))
            m3 = build_inputs(M3, f"b{b}_s{share}_h{h}_o{origin}", y, s.x, origin, h, ctx_len,
                              exp.frequency, x_future=v["x_tilde"])
            q_m3 = predict_fn(m3)
            rows.append({
                "base_series_id": b,
                "nominal_covariate_share": share,
                "horizon": h,
                "origin": origin,
                "lam": float(lam),
                "wql_m1": wql_m1,
                "wql_m3": wql(y_true, q_m3, q_levels),
                "crossing_m3": quantile_crossing_rate(q_m3),
                "lambda_hat": estimate_lambda_hat(v["error"], v["V"]),
                "realized_normalized_error_rms": v["realized_normalized_error_rms"],
            })
            total += 1
        if total % 200 == 0:
            log(f"historical admission: {total} M3 tasks done")
    return pd.DataFrame(rows)


def build_decisions(historical: pd.DataFrame, task_metrics: pd.DataFrame,
                    cfg: PilotConfig) -> pd.DataFrame:
    """Apply both selectors and attach the resulting primary-origin WQL."""
    keys = [BOOTSTRAP_UNIT, "nominal_covariate_share", "horizon", "lam"]
    agg = (historical.groupby(keys, as_index=False)
           .agg(hist_wql_m1=("wql_m1", "mean"),
                hist_wql_m3=("wql_m3", "mean"),
                hist_lambda_hat=("lambda_hat", "mean"),
                n_pseudo_origins=("origin", "nunique")))

    df = task_metrics.merge(agg, on=keys, how="left", validate="one_to_one")
    if df[["hist_wql_m1", "hist_wql_m3", "hist_lambda_hat"]].isna().any().any():
        raise RuntimeError("historical diagnostic does not cover every primary task")

    df[f"choice_{A1}"] = np.where(df["hist_wql_m3"] < df["hist_wql_m1"], M3, M1)
    df[f"choice_{A2}"] = np.where(df["hist_lambda_hat"] < 1.0, M3, M1)
    for sel in SELECTORS:
        df[f"wql_{sel}"] = np.where(df[f"choice_{sel}"] == M3, df["wql_m3"], df["wql_m1"])
    df["wql_oracle"] = np.minimum(df["wql_m1"], df["wql_m3"])
    return df


def _harm_rate(selected: np.ndarray, baseline: np.ndarray, threshold: float) -> float:
    return float(np.mean([is_harm(b, s, threshold) for s, b in zip(selected, baseline)]))


def gate_d(decisions: pd.DataFrame, cfg: PilotConfig) -> dict:
    """Gate D: does a history-only selector recover part of the oracle headroom?"""
    g = cfg.gates
    criteria = {
        "selectors": {
            A1: "use M3 now iff mean historical WQL(M3) < mean historical WQL(M1)",
            A2: ("use M3 now iff the historical vintage error implies lambda_hat < 1; "
                 "an analytic-inspired heuristic, not a WQL-optimal rule"),
        },
        "pass": (f"at least one selector beats the best fixed policy by >= "
                 f"{g.admission_pass_improvement:.0%}, recovers >= {g.admission_recovery_pass:.0%} of the "
                 f"oracle gap, cuts the harm rate versus M3 by >= {g.admission_harm_reduction_pass:.0%}, "
                 f"gives up <= {g.admission_low_noise_regression_max:.0%} versus M3 in low-noise cells, "
                 f"and has a paired CI on the improvement side"),
        "fail": (f"both selectors are worse than the best fixed policy, or recovery <= "
                 f"{g.admission_recovery_fail:.0%}, or the low-noise benefit is largely removed"),
        "leakage_guard": ("selection uses pseudo-origins that close at or before the primary origin; "
                          "the primary future target never enters a decision"),
    }

    mean_m1 = float(decisions["wql_m1"].mean())
    mean_m3 = float(decisions["wql_m3"].mean())
    mean_oracle = float(decisions["wql_oracle"].mean())
    best_fixed_name = "always_no_future" if mean_m1 <= mean_m3 else "always_use_future"
    best_fixed_col = "wql_m1" if best_fixed_name == "always_no_future" else "wql_m3"
    best_fixed_mean = min(mean_m1, mean_m3)
    oracle_gap = best_fixed_mean - mean_oracle

    harm_m3 = _harm_rate(decisions["wql_m3"].to_numpy(), decisions["wql_m1"].to_numpy(),
                         g.harm_relative_threshold)
    low = decisions[decisions["lam"] <= 0.5]

    results = {}
    for sel in SELECTORS:
        sel_mean = float(decisions[f"wql_{sel}"].mean())
        improvement = (best_fixed_mean - sel_mean) / best_fixed_mean
        recovery = (best_fixed_mean - sel_mean) / oracle_gap if oracle_gap > 0 else float("nan")
        harm_sel = _harm_rate(decisions[f"wql_{sel}"].to_numpy(),
                              decisions["wql_m1"].to_numpy(), g.harm_relative_threshold)
        harm_reduction = (harm_m3 - harm_sel) / harm_m3 if harm_m3 > 0 else float("nan")
        low_regression = ((low[f"wql_{sel}"].mean() - low["wql_m3"].mean()) / low["wql_m3"].mean()
                          if len(low) else float("nan"))
        boot = paired_bootstrap(
            decisions[BOOTSTRAP_UNIT].to_numpy(),
            decisions[best_fixed_col].to_numpy(),
            decisions[f"wql_{sel}"].to_numpy(),
            n_resamples=cfg.bootstrap.n_resamples,
            confidence_level=cfg.bootstrap.confidence_level,
            seed_parts=(cfg.experiment.master_seed, "bootstrap", f"gateD_{sel}"))
        passes = {
            "improvement_over_best_fixed": bool(improvement >= g.admission_pass_improvement),
            "oracle_gap_recovery": bool(np.isfinite(recovery) and recovery >= g.admission_recovery_pass),
            "harm_reduction_vs_m3": bool(np.isfinite(harm_reduction)
                                         and harm_reduction >= g.admission_harm_reduction_pass),
            "low_noise_regression_within_bound": bool(
                np.isfinite(low_regression) and low_regression <= g.admission_low_noise_regression_max),
            "ci_favours_selector": bool(boot.ci_favours_treatment),
        }
        results[sel] = {
            "mean_wql": sel_mean,
            "relative_improvement_over_best_fixed": float(improvement),
            "oracle_gap_recovery": float(recovery),
            "harm_rate": harm_sel,
            "harm_reduction_vs_m3": float(harm_reduction),
            "low_noise_relative_regression_vs_m3": float(low_regression),
            "m3_choice_rate": float((decisions[f"choice_{sel}"] == M3).mean()),
            "bootstrap": boot.to_dict(),
            "pass_conditions": passes,
            "all_pass_conditions_met": all(passes.values()),
        }

    any_pass = any(r["all_pass_conditions_met"] for r in results.values())
    both_worse = all(r["relative_improvement_over_best_fixed"] < 0 for r in results.values())
    recovery_low = all((not np.isfinite(r["oracle_gap_recovery"]))
                       or r["oracle_gap_recovery"] <= g.admission_recovery_fail
                       for r in results.values())
    # "removes most of the low-noise benefit": in low-noise cells M3 gains
    # (m1 - m3)/m3 over M1; a selector that gives back more than half of that gain
    # relative to M3 has destroyed it.  If there is no low-noise gain to begin
    # with, this failure mode does not apply.
    low_noise_benefit = (float((low["wql_m1"].mean() - low["wql_m3"].mean()) / low["wql_m3"].mean())
                         if len(low) else float("nan"))
    low_noise_destroyed = bool(
        len(low) and np.isfinite(low_noise_benefit) and low_noise_benefit > 0
        and all(np.isfinite(r["low_noise_relative_regression_vs_m3"])
                and r["low_noise_relative_regression_vs_m3"] > 0.5 * low_noise_benefit
                for r in results.values()))

    if any_pass:
        status = PASS
    elif both_worse or recovery_low or low_noise_destroyed:
        status = FAIL
    else:
        status = INCONCLUSIVE

    return {
        "gate": "D",
        "status": status,
        "criteria": criteria,
        "reference": {
            "always_no_future_m1": mean_m1,
            "always_use_future_m3": mean_m3,
            "oracle": mean_oracle,
            "best_fixed": best_fixed_name,
            "best_fixed_mean": best_fixed_mean,
            "oracle_gap": oracle_gap,
            "harm_rate_m3": harm_m3,
            "low_noise_benefit_m3_vs_m1": low_noise_benefit,
        },
        "selectors": results,
    }


def gate_d_not_run(reason: str) -> dict:
    return {"gate": "D", "status": NOT_RUN, "reason": reason}
