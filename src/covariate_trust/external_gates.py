"""Gate H and Gate I - real forecast-vintage external validation.

Gate H asks whether the real problem even exists: does a perfect future weather
covariate help, do real forecasts vary enough that the choice matters, is there
admission headroom, and does the decision-time proxy carry signal?

Gate I asks the pre-registered question about D7 only.  A secondary policy scoring
better is an observation, never a reason to promote it.

Gates A-G are untouched and are never recomputed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gates import FAIL, INCONCLUSIVE, NOT_RUN, PASS
from .real_vintage import D0, D1, D2, D3, D5, D7, summarize, week_cluster_bootstrap

NOT_EVALUABLE = "NOT_EVALUABLE"
MIN_EVENTS = 20


def _cond(value: bool | None) -> bool | str:
    return NOT_EVALUABLE if value is None else bool(value)


# ------------------------------------------------------------------ Gate H ---

def gate_h(tasks: pd.DataFrame, features: pd.DataFrame, coverage: dict,
           data_quality: dict, cfg) -> dict:
    """Is the admission problem real in this domain?"""
    gh = cfg.gate_h
    criteria = {
        "question": ("does future weather carry value in real NYISO load forecasting, and do "
                     "real ECMWF vintages vary enough that using them is a genuine choice?"),
        "H1": (f">= {cfg.nyiso.minimum_zone_count} zones, >= {gh.minimum_test_origins_per_zone} "
               f"held-out origins per zone, forecast coverage >= "
               f"{cfg.weather.minimum_forecast_coverage:.0%}, no leakage, time alignment ok"),
        "H2": f"M2 (verified future weather) beats M1 by >= {gh.oracle_gain_pass:.0%} with a CI on "
              f"the improvement side",
        "H3": f"M3 win rate strictly inside [{gh.minimum_m3_win_rate:.0%}, "
              f"{gh.maximum_m3_win_rate:.0%}]",
        "H4": f"per-origin min(M1, M3) beats the best fixed policy by >= "
              f"{gh.oracle_headroom_pass:.0%} with a CI on the improvement side",
        "H5": f"held-out Spearman(reported_reliability_ratio, realized_weather_error_ratio) >= {gh.proxy_spearman_pass}, and "
              f"the top reported quartile has >= 20% higher mean realized_weather_error_ratio than the bottom",
        "note": "the bootstrap clusters on ISO calendar week, not on individual origins",
    }

    zones = sorted(tasks["zone"].unique().tolist())
    per_zone = tasks.groupby("zone")["origin_utc"].nunique().to_dict()
    zones_ok = [z for z, n in per_zone.items() if n >= gh.minimum_test_origins_per_zone]
    fc = data_quality.get("forecast_coverage", {}).get("primary", {}).get("coverage", 0.0)
    h1 = {
        "n_zones": len(zones),
        "origins_per_zone": {k: int(v) for k, v in per_zone.items()},
        "zones_meeting_minimum": zones_ok,
        "primary_forecast_coverage": float(fc),
        "coverage_status": coverage.get("status"),
        "leakage_ok": bool(data_quality.get("leakage_ok", True)),
        "time_alignment_ok": bool(data_quality.get("time_alignment_ok", True)),
    }
    h1_pass = (len(zones_ok) >= cfg.nyiso.minimum_zone_count
               and fc >= cfg.weather.minimum_forecast_coverage
               and h1["leakage_ok"] and h1["time_alignment_ok"])

    b_m2 = week_cluster_bootstrap(tasks.assign(calendar_week=tasks["calendar_week"]),
                                  "wql_m1", "wql_m2", cfg,
                                  (cfg.experiment.master_seed, "gateH_m2"))
    h2_pass = bool(b_m2["relative_improvement"] >= gh.oracle_gain_pass
                   and b_m2["ci_favours_treatment"])
    per_zone_m2 = {}
    for z, g in tasks.groupby("zone"):
        per_zone_m2[z] = float((g["wql_m1"].mean() - g["wql_m2"].mean()) / g["wql_m1"].mean())

    win = float(tasks["m3_is_better"].mean())
    h3_pass = bool(gh.minimum_m3_win_rate <= win <= gh.maximum_m3_win_rate)

    t = tasks.copy()
    m1, m3 = float(t["wql_m1"].mean()), float(t["wql_m3"].mean())
    best_fixed = D0 if m1 <= m3 else D1
    best_col = "wql_m1" if best_fixed == D0 else "wql_m3"
    b_oracle = week_cluster_bootstrap(t, best_col, "wql_oracle", cfg,
                                      (cfg.experiment.master_seed, "gateH_oracle"))
    h4_pass = bool(b_oracle["relative_improvement"] >= gh.oracle_headroom_pass
                   and b_oracle["ci_favours_treatment"])

    from .weather_proxy import calibration_diagnostics
    cal = calibration_diagnostics(features["reported_reliability_ratio"].to_numpy(),
                                  features["realized_weather_error_ratio"].to_numpy())
    ratio = cal.get("quartile_ratio", np.nan)
    h5_pass = bool(np.isfinite(cal["spearman"]) and cal["spearman"] >= gh.proxy_spearman_pass
                   and np.isfinite(ratio) and ratio >= 1.20)

    conditions = {"H1_data_integrity": h1_pass, "H2_oracle_weather_value": h2_pass,
                  "H3_forecast_heterogeneity": h3_pass, "H4_admission_headroom": h4_pass,
                  "H5_proxy_relevance": h5_pass}

    fail_reasons = []
    if b_m2["relative_improvement"] <= gh.oracle_gain_fail or b_m2["ci_favours_baseline"]:
        fail_reasons.append("no oracle future-weather value")
    if win < gh.minimum_m3_win_rate or win > gh.maximum_m3_win_rate:
        fail_reasons.append("one fixed policy wins almost always")
    if not h4_pass and b_oracle["relative_improvement"] <= 0:
        fail_reasons.append("no oracle admission headroom")
    if np.isfinite(cal["spearman"]) and cal["spearman"] <= -gh.proxy_spearman_pass:
        fail_reasons.append("the proxy is significantly inverted")

    status = PASS if all(conditions.values()) else (FAIL if fail_reasons else INCONCLUSIVE)
    return {
        "gate": "H", "status": status, "criteria": criteria,
        "checks": {**conditions,
                   "m3_win_rate": win,
                   "m2_relative_improvement": b_m2["relative_improvement"],
                   "oracle_headroom": b_oracle["relative_improvement"],
                   "proxy_spearman": cal["spearman"],
                   "proxy_quartile_ratio": ratio},
        "h1_detail": h1,
        "m2_bootstrap": b_m2, "m2_per_zone_relative_gain": per_zone_m2,
        "oracle_bootstrap": b_oracle, "best_fixed": best_fixed,
        "proxy_calibration_heldout": cal,
        "fail_reasons": fail_reasons,
    }


# ------------------------------------------------------------------ Gate I ---

def gate_i(decisions: pd.DataFrame, events: pd.DataFrame, cfg) -> dict:
    """Does the pre-registered D7 beat the fixed policies on real held-out vintages?"""
    gi = cfg.gate_i
    criteria = {
        "primary_policy": f"{D7}, fixed by the synthetic studies; thresholds 0.75 / 1.25 unchanged",
        "best_fixed": f"the better of {D0} and {D1} on the held-out mixture",
        "I1": f"D7 beats best fixed by >= {gi.improvement_pass:.1%}",
        "I2": "calendar-week cluster-bootstrap CI on the improvement side",
        "I3": f"recovers >= {gi.oracle_recovery_pass:.0%} of the oracle admission gap",
        "I4": f"cuts the harm rate versus always-use by >= {gi.harm_reduction_pass:.0%}",
        "I5": f"every zone within {gi.maximum_subset_regression:.0%} of best fixed",
        "I6": f"every season within {gi.maximum_subset_regression:.0%} of best fixed",
        "I7": f"if >= {MIN_EVENTS} worsening events, D7 beats {D3}",
        "I8": f"if >= {MIN_EVENTS} improvement events, D7 beats {D0}",
        "scope": ("this is one real domain (NYISO load, ECMWF temperature, Chronos-2); it is not "
                  "a general deployment claim"),
    }

    per_selector = {sel: summarize(g) for sel, g in decisions.groupby("selector")}
    fixed = {sel: per_selector[sel]["mean_wql"] for sel in (D0, D1)}
    best_fixed_name = min(fixed, key=fixed.get)
    best_fixed_mean = fixed[best_fixed_name]
    best_col = "wql_m1" if best_fixed_name == D0 else "wql_m3"
    oracle_mean = per_selector[D2]["mean_wql"]
    gap = best_fixed_mean - oracle_mean

    d7 = decisions[decisions["selector"] == D7]
    prim = per_selector[D7]
    boot = week_cluster_bootstrap(d7, best_col, "wql_selected", cfg,
                                  (cfg.experiment.master_seed, "gateI", D7))
    improvement = (best_fixed_mean - prim["mean_wql"]) / best_fixed_mean
    recovery = (best_fixed_mean - prim["mean_wql"]) / gap if gap > 0 else float("nan")
    harm_always_use = per_selector[D1]["harm_rate"]
    harm_reduction = ((harm_always_use - prim["harm_rate"]) / harm_always_use
                      if harm_always_use > 0 else float("nan"))

    def subset_regression(by: str) -> dict:
        out = {}
        for key, g in decisions[decisions["selector"] == D7].groupby(by):
            ref = decisions[(decisions["selector"] == best_fixed_name) & (decisions[by] == key)]
            if not len(ref):
                continue
            out[str(key)] = float((g["wql_selected"].mean() - ref["wql_selected"].mean())
                                  / ref["wql_selected"].mean())
        return out

    zone_reg = subset_regression("zone")
    season_reg = subset_regression("season")
    zone_bad = [z for z, v in zone_reg.items() if v > gi.maximum_subset_regression]
    season_bad = [s for s, v in season_reg.items() if v > gi.maximum_subset_regression]

    ev = decisions.merge(events, on=["zone", "origin_utc"], how="left")
    n_wors = int(ev[(ev["selector"] == D7) & (ev["worsening_event"] == 1)].shape[0])
    n_impr = int(ev[(ev["selector"] == D7) & (ev["improvement_event"] == 1)].shape[0])

    def event_compare(flag: str, other: str) -> tuple[bool | None, dict]:
        sub = ev[ev[flag] == 1]
        n = int(sub[sub["selector"] == D7].shape[0])
        if n < MIN_EVENTS:
            return None, {"n_events": n, "status": NOT_EVALUABLE}
        a = float(sub[sub["selector"] == D7]["wql_selected"].mean())
        b = float(sub[sub["selector"] == other]["wql_selected"].mean())
        return (a < b), {"n_events": n, "d7_mean_wql": a, f"{other}_mean_wql": b}

    i7, i7_detail = event_compare("worsening_event", D3)
    i8, i8_detail = event_compare("improvement_event", D0)

    conditions = {
        "I1_improvement": bool(improvement >= gi.improvement_pass),
        "I2_ci_favours_d7": bool(boot["ci_favours_treatment"]),
        "I3_oracle_recovery": bool(np.isfinite(recovery) and recovery >= gi.oracle_recovery_pass),
        "I4_harm_reduction": bool(np.isfinite(harm_reduction)
                                  and harm_reduction >= gi.harm_reduction_pass),
        "I5_zone_safety": bool(not zone_bad),
        "I6_season_safety": bool(not season_bad),
        "I7_worsening_beats_history": _cond(i7),
        "I8_improvement_beats_no_future": _cond(i8),
    }
    evaluable = {k: v for k, v in conditions.items() if v is not NOT_EVALUABLE}

    fail_reasons = []
    if improvement <= gi.improvement_fail:
        fail_reasons.append("aggregate improvement at or below zero")
    if boot["ci_favours_baseline"]:
        fail_reasons.append("the CI favours the best fixed policy")
    if np.isfinite(recovery) and recovery <= gi.oracle_recovery_fail:
        fail_reasons.append("oracle recovery at or below the FAIL bound")
    if np.isfinite(harm_reduction) and harm_reduction <= 0:
        fail_reasons.append("harm rate not reduced versus always-use")
    if len(zone_bad) >= 2:
        fail_reasons.append(f"more than one zone regresses beyond the bound: {zone_bad}")
    if len(season_bad) >= 2:
        fail_reasons.append(f"more than one season regresses beyond the bound: {season_bad}")

    status = PASS if all(evaluable.values()) else (FAIL if fail_reasons else INCONCLUSIVE)
    return {
        "gate": "I", "status": status, "criteria": criteria,
        "primary_selector": D7,
        "reference": {"best_fixed": best_fixed_name, "best_fixed_mean": best_fixed_mean,
                      "oracle_mean": oracle_mean, "oracle_gap": gap,
                      "harm_rate_always_use": harm_always_use},
        "primary_metrics": prim,
        "bootstrap": boot,
        "checks": {"relative_improvement": float(improvement),
                   "oracle_gap_recovery": float(recovery),
                   "harm_reduction_vs_always_use": float(harm_reduction),
                   "zone_regression": zone_reg, "season_regression": season_reg,
                   "worsening": i7_detail, "improvement": i8_detail,
                   "n_worsening_events": n_wors, "n_improvement_events": n_impr,
                   **conditions},
        "not_evaluable": [k for k, v in conditions.items() if v is NOT_EVALUABLE],
        "failed_conditions": [k for k, v in evaluable.items() if not v],
        "fail_reasons": fail_reasons,
        "per_selector": per_selector,
    }


def external_verdict(gate_h_result: dict | None, gate_i_result: dict | None,
                     blocked_reason: str | None = None) -> dict:
    """The four outcomes this study is allowed to reach, plus BLOCKED."""
    if blocked_reason:
        return {"verdict": "BLOCKED_EXTERNAL_DATA", "reason": blocked_reason}
    if gate_h_result is None:
        return {"verdict": "BLOCKED_EXTERNAL_DATA", "reason": "Gate H was not evaluated"}
    if gate_h_result["status"] != PASS:
        return {"verdict": "REAL_DATA_PROBLEM_NOT_ESTABLISHED",
                "reason": f"Gate H is {gate_h_result['status']}: "
                          f"{gate_h_result.get('fail_reasons') or gate_h_result['checks']}",
                "gate_h": gate_h_result["status"], "gate_i": NOT_RUN}
    if gate_i_result is None:
        return {"verdict": "BLOCKED_EXTERNAL_DATA", "reason": "Gate I was not evaluated"}
    s = gate_i_result["status"]
    if s == PASS:
        return {"verdict": "REAL_VINTAGE_EXTERNAL_VALIDATION_GO",
                "reason": ("in one real NYISO weather-vintage domain the pre-registered D7 beat "
                           "the fixed policies on both mean WQL and harm rate"),
                "gate_h": PASS, "gate_i": s,
                "scope_note": ("Supported in a real NYISO weather-vintage case study. This is NOT "
                               "a claim of general validation for all deployments.")}
    if s == FAIL:
        return {"verdict": "SYNTHETIC_TO_REAL_METHOD_NO_GO",
                "reason": "; ".join(gate_i_result["fail_reasons"]) or "Gate I FAIL",
                "gate_h": PASS, "gate_i": s}
    return {"verdict": "EXTERNAL_VALIDATION_CONDITIONAL",
            "reason": f"Gate I is {s}; unmet: {gate_i_result['failed_conditions']}, "
                      f"not evaluable: {gate_i_result['not_evaluable']}",
            "gate_h": PASS, "gate_i": s}
