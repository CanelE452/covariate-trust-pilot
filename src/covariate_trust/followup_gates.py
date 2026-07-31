"""Gate E (independent boundary) and Gate F (dynamic admission).

Both use only PASS / FAIL / INCONCLUSIVE / NOT_RUN.  Thresholds come from the config
and are fixed before the run; nothing here reacts to a result by changing a criterion.

Gate D (existing) and Gate F (new) answer different questions and are never merged:
    Gate D   is historical admission possible under *stationary* reliability?
    Gate F   is admission possible under *time-varying* reliability with an imperfect
             current uncertainty proxy?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .boundary import FINITE
from .bootstrap import BOOTSTRAP_UNIT, paired_bootstrap
from .config import BoundaryConfig, DynamicConfig
from .dynamic_admission import D0, D1, D2, D3, D5, D7, FIXED_POLICIES, _summarize
from .gates import FAIL, INCONCLUSIVE, NOT_RUN, PASS
from .reliability_schedules import P0_ORACLE, P1_CALIBRATED, P2_OVERCONFIDENT, P3_UNDERCONFIDENT, P4_STALE

LOW_LAMBDA_MAX = 0.85
HIGH_LAMBDA_MIN = 1.15

# schedule names the gate refers to by role
STABLE_LOW, STABLE_HIGH = "S0_stable_low", "S1_stable_high"
WORSENING = ("S2_sudden_worsening", "S4_gradual_worsening")
IMPROVING = ("S3_sudden_improvement", "S5_gradual_improvement")


# ------------------------------------------------------------------ Gate E ---

def gate_e(task_metrics: pd.DataFrame, boundaries: pd.DataFrame, cfg: BoundaryConfig,
           coarse_reference: dict | None = None) -> dict:
    """Does the coarse pilot's benefit-to-harm boundary reproduce on an independent seed?"""
    ge = cfg.gate_e
    criteria = {
        "comparison": "M3 (forecasted future covariate) vs M1 (past covariate only), independent seed",
        "boundary": "descending zero crossing of V_future(lambda) = WQL(M1) - WQL(M3), "
                    "linearly interpolated inside the grid, never extrapolated",
        "pass": (f">= {ge.min_finite_crossings} of the (share, horizon) curves give a finite "
                 f"crossing; every finite crossing has a positive low-lambda mean and a negative "
                 f"high-lambda mean; >= {ge.min_narrow_crossings} crossings have CI width <= "
                 f"{ge.max_ci_width}; every curve has Spearman rho < 0; the pooled low-lambda and "
                 f"high-lambda paired CIs point to benefit and harm respectively; and the effect "
                 f"direction matches the coarse pilot"),
        "fail": (f"<= {ge.fail_max_finite_crossings} finite crossings, or the low/high direction is "
                 f"opposite to the coarse pilot, or >= {ge.fail_min_non_decreasing_curves} curves "
                 f"are not decreasing in lambda"),
        "note": ("lambda = 1 is the Study-0 linear-Gaussian reference line, not a theoretical "
                 "boundary of Chronos WQL"),
    }

    n_curves = len(boundaries)
    finite = boundaries[boundaries["status"] == FINITE]
    n_finite = int(len(finite))
    narrow = finite[finite["ci_width"] <= ge.max_ci_width]
    n_narrow = int(len(narrow))
    directions_ok = bool(len(finite) and (finite["low_lambda_mean"] > 0).all()
                         and (finite["high_lambda_mean"] < 0).all())
    n_decreasing = int((boundaries["spearman_rho"] < 0).sum())
    all_decreasing = bool(n_decreasing == n_curves)

    low = task_metrics[task_metrics["lam"] <= LOW_LAMBDA_MAX]
    high = task_metrics[task_metrics["lam"] >= HIGH_LAMBDA_MIN]
    b_low = paired_bootstrap(low[BOOTSTRAP_UNIT].to_numpy(), low["wql_m1"].to_numpy(),
                             low["wql_m3"].to_numpy(), cfg.bootstrap.n_resamples,
                             cfg.bootstrap.confidence_level,
                             seed_parts=(cfg.experiment.master_seed, "gateE_low"))
    b_high = paired_bootstrap(high[BOOTSTRAP_UNIT].to_numpy(), high["wql_m1"].to_numpy(),
                              high["wql_m3"].to_numpy(), cfg.bootstrap.n_resamples,
                              cfg.bootstrap.confidence_level,
                              seed_parts=(cfg.experiment.master_seed, "gateE_high"))
    low_benefit = bool(b_low.ci_low > 0)
    high_harm = bool(b_high.ci_high < 0)

    coarse_match, coarse_detail = True, "no coarse reference supplied"
    if coarse_reference:
        want_low = coarse_reference.get("low_lambda_v_future", 0.0) > 0
        want_high = coarse_reference.get("high_lambda_v_future", 0.0) < 0
        got_low, got_high = b_low.mean_diff > 0, b_high.mean_diff < 0
        coarse_match = bool((want_low == got_low) and (want_high == got_high))
        coarse_detail = (f"coarse low {coarse_reference.get('low_lambda_v_future'):+.5f} / high "
                         f"{coarse_reference.get('high_lambda_v_future'):+.5f}; refinement low "
                         f"{b_low.mean_diff:+.5f} / high {b_high.mean_diff:+.5f}")

    passed = (n_finite >= ge.min_finite_crossings and directions_ok
              and n_narrow >= ge.min_narrow_crossings and all_decreasing
              and low_benefit and high_harm and coarse_match)
    failed = (n_finite <= ge.fail_max_finite_crossings or not coarse_match
              or (n_curves - n_decreasing) >= ge.fail_min_non_decreasing_curves)

    status = PASS if passed else (FAIL if failed else INCONCLUSIVE)
    return {
        "gate": "E",
        "status": status,
        "criteria": criteria,
        "checks": {
            "n_curves": n_curves,
            "n_finite_crossings": n_finite,
            "n_narrow_crossings": n_narrow,
            "max_ci_width_allowed": ge.max_ci_width,
            "all_finite_directions_ok": directions_ok,
            "n_curves_decreasing": n_decreasing,
            "all_curves_decreasing": all_decreasing,
            "low_lambda_mean_diff": b_low.mean_diff,
            "low_lambda_ci": [b_low.ci_low, b_low.ci_high],
            "low_lambda_benefit": low_benefit,
            "high_lambda_mean_diff": b_high.mean_diff,
            "high_lambda_ci": [b_high.ci_low, b_high.ci_high],
            "high_lambda_harm": high_harm,
            "coarse_direction_match": coarse_match,
        },
        "coarse_comparison": coarse_detail,
        "boundaries": boundaries.to_dict("records"),
    }


def coarse_reference_from_cells(coarse_cells: pd.DataFrame) -> dict:
    """Direction of the coarse pilot's effect, for the Gate E consistency check."""
    pos = coarse_cells[coarse_cells["nominal_covariate_share"] > 0]
    return {
        "low_lambda_v_future": float(pos[pos["lam"] <= 0.5]["v_future_mean"].mean()),
        "high_lambda_v_future": float(pos[pos["lam"] >= 1.5]["v_future_mean"].mean()),
        "source": "coarse pilot cell_summary.csv (share > 0)",
    }


# ------------------------------------------------------------------ Gate F ---

def gate_f(decisions: pd.DataFrame, cfg: DynamicConfig) -> dict:
    """Does a proxy-using selector stay safe when reliability changes over time?"""
    gf = cfg.gate_f
    criteria = {
        "primary_proxy": f"{P1_CALIBRATED} (the calibrated noisy proxy)",
        "primary_selector": f"the better of {D5} and {D7} under that proxy, both fixed in advance",
        "best_fixed": f"the better of {D0} and {D1} over the whole experimental mixture",
        "pass": (f"primary selector beats best fixed by >= {gf.pass_improvement:.0%} with a paired CI "
                 f"on the improvement side; oracle gap recovery >= {gf.pass_recovery:.0%}; harm rate "
                 f"cut by >= {gf.pass_harm_reduction:.0%} versus always-use; beats history-only "
                 f"{D3} in both worsening conditions; gives up <= "
                 f"{gf.improvement_condition_regression_max:.0%} versus always-no-future in the "
                 f"improvement conditions; gives up <= {gf.stable_regression_max:.0%} versus "
                 f"always-use in stable_low and versus always-no-future in stable_high"),
        "fail": (f"every calibrated-proxy selector is worse than best fixed, or recovery <= "
                 f"{gf.fail_recovery:.0%}, or the worsening conditions are not improved over "
                 f"history-only, or the stable low-noise benefit is largely removed"),
        "caveat": ("the mixture weights all six schedules and all cells equally; that is a design "
                   "choice, not a claim about how often reliability shifts in deployment"),
    }

    p1 = decisions[decisions["proxy_mode"] == P1_CALIBRATED]
    if p1.empty:
        return {"gate": "F", "status": NOT_RUN, "criteria": criteria,
                "reason": "calibrated proxy rows are missing"}

    per_selector = {sel: _summarize(g) for sel, g in p1.groupby("selector")}
    fixed_means = {sel: per_selector[sel]["mean_wql"] for sel in FIXED_POLICIES}
    best_fixed_name = min(fixed_means, key=fixed_means.get)
    best_fixed_mean = fixed_means[best_fixed_name]
    oracle_mean = per_selector[D2]["mean_wql"]
    gap = best_fixed_mean - oracle_mean

    candidates = {sel: per_selector[sel]["mean_wql"] for sel in (D5, D7)}
    primary = min(candidates, key=candidates.get)
    prim = per_selector[primary]
    prim_mean = prim["mean_wql"]

    best_fixed_col = "wql_m1" if best_fixed_name == D0 else "wql_m3"
    prim_rows = p1[p1["selector"] == primary]
    boot = paired_bootstrap(prim_rows[BOOTSTRAP_UNIT].to_numpy(),
                            prim_rows[best_fixed_col].to_numpy(),
                            prim_rows["wql_selected"].to_numpy(),
                            cfg.bootstrap.n_resamples, cfg.bootstrap.confidence_level,
                            seed_parts=(cfg.experiment.master_seed, "gateF", primary))

    improvement = (best_fixed_mean - prim_mean) / best_fixed_mean
    recovery = (best_fixed_mean - prim_mean) / gap if gap > 0 else float("nan")
    harm_always_use = per_selector[D1]["harm_rate"]
    harm_reduction = ((harm_always_use - prim["harm_rate"]) / harm_always_use
                      if harm_always_use > 0 else float("nan"))

    def cond_mean(selector: str, schedule: str) -> float:
        g = p1[(p1["selector"] == selector) & (p1["schedule"] == schedule)]
        return float(g["wql_selected"].mean()) if len(g) else float("nan")

    worsening_beats_history = {
        s: bool(cond_mean(primary, s) < cond_mean(D3, s)) for s in WORSENING}
    improving_regression = {
        s: float((cond_mean(primary, s) - cond_mean(D0, s)) / cond_mean(D0, s))
        for s in IMPROVING}
    stable_low_reg = float((cond_mean(primary, STABLE_LOW) - cond_mean(D1, STABLE_LOW))
                           / cond_mean(D1, STABLE_LOW))
    stable_high_reg = float((cond_mean(primary, STABLE_HIGH) - cond_mean(D0, STABLE_HIGH))
                            / cond_mean(D0, STABLE_HIGH))

    conditions = {
        "improvement_over_best_fixed": bool(improvement >= gf.pass_improvement),
        "ci_favours_selector": bool(boot.ci_favours_treatment),
        "oracle_gap_recovery": bool(np.isfinite(recovery) and recovery >= gf.pass_recovery),
        "harm_reduction_vs_always_use": bool(np.isfinite(harm_reduction)
                                             and harm_reduction >= gf.pass_harm_reduction),
        "beats_history_only_in_worsening": bool(all(worsening_beats_history.values())),
        "improvement_conditions_within_bound": bool(
            all(v <= gf.improvement_condition_regression_max for v in improving_regression.values())),
        "stable_low_within_bound": bool(stable_low_reg <= gf.stable_regression_max),
        "stable_high_within_bound": bool(stable_high_reg <= gf.stable_regression_max),
    }

    all_proxy_selectors_worse = all(
        per_selector[s]["mean_wql"] > best_fixed_mean for s in (D5, D7))
    low_noise_destroyed = bool(stable_low_reg > 0.5 * max(
        1e-12, (cond_mean(D0, STABLE_LOW) - cond_mean(D1, STABLE_LOW)) / cond_mean(D1, STABLE_LOW)))

    if all(conditions.values()):
        status = PASS
    elif (all_proxy_selectors_worse
          or (np.isfinite(recovery) and recovery <= gf.fail_recovery)
          or not any(worsening_beats_history.values())
          or low_noise_destroyed):
        status = FAIL
    else:
        status = INCONCLUSIVE

    diagnostics = {}
    for mode in (P0_ORACLE, P2_OVERCONFIDENT, P3_UNDERCONFIDENT, P4_STALE):
        g = decisions[(decisions["proxy_mode"] == mode) & (decisions["selector"] == primary)]
        if not len(g):
            continue
        entry = _summarize(g)
        if mode == P4_STALE:
            for s in WORSENING + IMPROVING:
                sub = g[g["schedule"] == s]
                entry[f"mean_wql_{s}"] = float(sub["wql_selected"].mean()) if len(sub) else float("nan")
                entry[f"m3_choice_rate_{s}"] = float(sub["chose_m3"].mean()) if len(sub) else float("nan")
        diagnostics[mode] = entry

    return {
        "gate": "F",
        "status": status,
        "criteria": criteria,
        "primary_selector": primary,
        "primary_proxy": P1_CALIBRATED,
        "reference": {
            "best_fixed": best_fixed_name, "best_fixed_mean": best_fixed_mean,
            "oracle_mean": oracle_mean, "oracle_gap": gap,
            "harm_rate_always_use": harm_always_use,
        },
        "primary_metrics": prim,
        "bootstrap": boot.to_dict(),
        "checks": {
            "relative_improvement": float(improvement),
            "oracle_gap_recovery": float(recovery),
            "harm_reduction_vs_always_use": float(harm_reduction),
            "worsening_beats_history_only": worsening_beats_history,
            "improvement_condition_relative_regression": improving_regression,
            "stable_low_relative_regression": stable_low_reg,
            "stable_high_relative_regression": stable_high_reg,
            **conditions,
        },
        "per_selector_calibrated_proxy": per_selector,
        "proxy_diagnostics": diagnostics,
    }


# ------------------------------------------------------------- final call ---

def go_no_go(gate_e_result: dict | None, gate_f_result: dict | None,
             baseline_result: dict | None, leakage_ok: bool,
             regression_ok: bool) -> dict:
    """FINAL GO / CONDITIONAL GO / NO-GO METHOD / NO-GO PHENOMENON / BLOCKED."""
    if gate_e_result is None:
        return {"verdict": "BLOCKED", "reason": "Gate E was not evaluated"}
    e = gate_e_result["status"]
    f = gate_f_result["status"] if gate_f_result else NOT_RUN
    b = baseline_result["status"] if baseline_result else NOT_RUN

    if e == FAIL:
        return {"verdict": "NO-GO PHENOMENON",
                "reason": "the coarse pilot's boundary did not reproduce on an independent sample",
                "gate_e": e, "gate_f": f}
    if e != PASS:
        return {"verdict": "CONDITIONAL GO",
                "reason": f"Gate E is {e}; Study 2 was not run and no additional samples were "
                          f"executed automatically",
                "gate_e": e, "gate_f": f}
    if f == FAIL:
        return {"verdict": "NO-GO METHOD",
                "reason": "the boundary exists but the current admission methods fail under "
                          "time-varying reliability; they need redesign",
                "gate_e": e, "gate_f": f}
    if f == PASS and b == PASS and leakage_ok and regression_ok:
        return {"verdict": "FINAL GO", "reason": "Gate E PASS, Gate F PASS, baseline checks PASS, "
                                                 "no leakage, no test regression",
                "gate_e": e, "gate_f": f}
    reasons = []
    if f != PASS:
        reasons.append(f"Gate F is {f}")
    if b != PASS:
        reasons.append(f"baseline checks are {b}")
    if not leakage_ok:
        reasons.append("a leakage check did not pass")
    if not regression_ok:
        reasons.append("an existing test regressed")
    return {"verdict": "CONDITIONAL GO", "reason": "; ".join(reasons), "gate_e": e, "gate_f": f}


# ------------------------------------------------------------------ Gate G ---
# Study 2B: held-out confirmation of the pre-registered D7 policy.  Gate E and
# Gate F above are untouched and are never recomputed from this data.

def gate_g(decisions, cfg) -> dict:
    """Does the pre-registered D7 policy reproduce on a fresh, independent sample?

    Decided on D7 under the P1 calibrated proxy only.  A secondary policy scoring
    better is recorded as an observation and never promoted.
    """
    import numpy as np
    from .dynamic_admission import D0, D1, D2, D3, D4, D7, _summarize

    gg = cfg.gate_g
    criteria = {
        "decided_on": f"{cfg.selectors.primary} under {cfg.proxy.primary_mode} only, fixed before "
                      f"the run; a better-scoring secondary policy does not replace it",
        "G1": f"mean WQL improvement over best fixed >= {gg.overall_improvement_pass:.0%}",
        "G2": "paired cluster-bootstrap 95% CI on the improvement side",
        "G3": f"oracle gap recovery >= {gg.oracle_recovery_pass:.0%}",
        "G4": f"harm rate cut versus always-use by >= {gg.harm_reduction_pass:.0%}",
        "G5": f"S0_stable_low: gives up <= {gg.stable_condition_regression_max:.0%} versus always-use",
        "G6": f"S1_stable_high: gives up <= {gg.stable_condition_regression_max:.0%} versus "
              f"always-no-future",
        "G7": "beats both D3 and D4 in S2_sudden_worsening and in S4_gradual_worsening",
        "G8": "beats D0 in S3_sudden_improvement and in S5_gradual_improvement",
        "fail": (f"improvement <= {gg.overall_improvement_fail:.0%}; or the CI favours best fixed; "
                 f"or recovery <= {gg.oracle_recovery_fail:.0%}; or harm reduction < "
                 f"{gg.harm_reduction_fail:.0%}; or a stable-condition regression > "
                 f"{gg.stable_condition_fail:.0%}; or D7 loses to both history-only policies in "
                 f"both worsening schedules; or D7 loses to D0 in both improvement schedules"),
        "mixture_caveat": ("all schedules, shares, horizons and series are weighted equally; that "
                           "is a design choice, not a deployment prevalence"),
    }

    p1 = decisions[decisions["proxy_mode"] == cfg.proxy.primary_mode]
    if p1.empty:
        return {"gate": "G", "status": NOT_RUN, "criteria": criteria,
                "reason": "no rows for the primary proxy"}

    per_selector = {sel: _summarize(g) for sel, g in p1.groupby("selector")}
    fixed = {sel: per_selector[sel]["mean_wql"] for sel in (D0, D1)}
    best_fixed_name = min(fixed, key=fixed.get)
    best_fixed_mean = fixed[best_fixed_name]
    best_fixed_col = "wql_m1" if best_fixed_name == D0 else "wql_m3"
    oracle_mean = per_selector[D2]["mean_wql"]
    gap = best_fixed_mean - oracle_mean

    prim = per_selector[D7]
    prim_mean = prim["mean_wql"]
    d7_rows = p1[p1["selector"] == D7]
    boot = paired_bootstrap(d7_rows[BOOTSTRAP_UNIT].to_numpy(),
                            d7_rows[best_fixed_col].to_numpy(),
                            d7_rows["wql_selected"].to_numpy(),
                            cfg.bootstrap.n_resamples, cfg.bootstrap.confidence_level,
                            seed_parts=(cfg.experiment.master_seed, "gateG", D7))

    improvement = (best_fixed_mean - prim_mean) / best_fixed_mean
    recovery = (best_fixed_mean - prim_mean) / gap if gap > 0 else float("nan")
    harm_always_use = per_selector[D1]["harm_rate"]
    harm_reduction = ((harm_always_use - prim["harm_rate"]) / harm_always_use
                      if harm_always_use > 0 else float("nan"))

    def cond(selector: str, schedule: str) -> float:
        g = p1[(p1["selector"] == selector) & (p1["schedule"] == schedule)]
        return float(g["wql_selected"].mean()) if len(g) else float("nan")

    stable_low_reg = (cond(D7, STABLE_LOW) - cond(D1, STABLE_LOW)) / cond(D1, STABLE_LOW)
    stable_high_reg = (cond(D7, STABLE_HIGH) - cond(D0, STABLE_HIGH)) / cond(D0, STABLE_HIGH)
    worsening = {s: {"d7": cond(D7, s), "d3": cond(D3, s), "d4": cond(D4, s),
                     "beats_both": bool(cond(D7, s) < cond(D3, s) and cond(D7, s) < cond(D4, s))}
                 for s in WORSENING}
    improving = {s: {"d7": cond(D7, s), "d0": cond(D0, s),
                     "beats_d0": bool(cond(D7, s) < cond(D0, s))} for s in IMPROVING}

    conditions = {
        "G1_overall_improvement": bool(improvement >= gg.overall_improvement_pass),
        "G2_ci_favours_primary": bool(boot.ci_favours_treatment),
        "G3_oracle_recovery": bool(np.isfinite(recovery) and recovery >= gg.oracle_recovery_pass),
        "G4_harm_reduction": bool(np.isfinite(harm_reduction)
                                  and harm_reduction >= gg.harm_reduction_pass),
        "G5_stable_low_safety": bool(stable_low_reg <= gg.stable_condition_regression_max),
        "G6_stable_high_safety": bool(stable_high_reg <= gg.stable_condition_regression_max),
        "G7_worsening_beats_history_only": bool(all(v["beats_both"] for v in worsening.values())),
        "G8_improvement_beats_no_future": bool(all(v["beats_d0"] for v in improving.values())),
    }

    fail_reasons = []
    if improvement <= gg.overall_improvement_fail:
        fail_reasons.append("overall improvement at or below the FAIL bound")
    if boot.ci_favours_baseline:
        fail_reasons.append("the paired CI favours the best fixed policy")
    if np.isfinite(recovery) and recovery <= gg.oracle_recovery_fail:
        fail_reasons.append("oracle recovery at or below the FAIL bound")
    if np.isfinite(harm_reduction) and harm_reduction < gg.harm_reduction_fail:
        fail_reasons.append("harm reduction below the FAIL bound")
    if stable_low_reg > gg.stable_condition_fail:
        fail_reasons.append("stable_low regression beyond the FAIL bound")
    if stable_high_reg > gg.stable_condition_fail:
        fail_reasons.append("stable_high regression beyond the FAIL bound")
    if all(not v["beats_both"] for v in worsening.values()):
        fail_reasons.append("D7 loses to the history-only policies in both worsening schedules")
    if all(not v["beats_d0"] for v in improving.values()):
        fail_reasons.append("D7 loses to always-no-future in both improvement schedules")

    if all(conditions.values()):
        status = PASS
    elif fail_reasons:
        status = FAIL
    else:
        status = INCONCLUSIVE

    return {
        "gate": "G",
        "status": status,
        "criteria": criteria,
        "primary_selector": cfg.selectors.primary,
        "primary_proxy": cfg.proxy.primary_mode,
        "reference": {"best_fixed": best_fixed_name, "best_fixed_mean": best_fixed_mean,
                      "oracle_mean": oracle_mean, "oracle_gap": gap,
                      "harm_rate_always_use": harm_always_use},
        "primary_metrics": prim,
        "bootstrap": boot.to_dict(),
        "checks": {
            "overall_relative_improvement": float(improvement),
            "oracle_gap_recovery": float(recovery),
            "harm_reduction_vs_always_use": float(harm_reduction),
            "stable_low_relative_regression": float(stable_low_reg),
            "stable_high_relative_regression": float(stable_high_reg),
            "worsening": worsening,
            "improving": improving,
            **conditions,
        },
        "failed_conditions": [k for k, v in conditions.items() if not v],
        "fail_reasons": fail_reasons,
        "per_selector_primary_proxy": per_selector,
    }


def gate_g_verdict(gate_g_result: dict | None, existing_regression_ok: bool,
                   leakage_ok: bool, independence_ok: bool) -> dict:
    """METHOD GO / CONDITIONAL GO / NO-GO CURRENT METHOD / BLOCKED / INVALID_RUN."""
    if not existing_regression_ok:
        return {"verdict": "BLOCKED", "reason": "an existing test regressed"}
    if not (leakage_ok and independence_ok):
        return {"verdict": "INVALID_RUN",
                "reason": "a leakage or independence check did not pass"}
    if gate_g_result is None:
        return {"verdict": "BLOCKED", "reason": "Gate G was not evaluated"}
    s = gate_g_result["status"]
    if s == PASS:
        return {"verdict": "METHOD GO",
                "reason": "the pre-registered D7 policy reproduced both its average performance "
                          "and its stable-condition safety on an independent held-out sample",
                "gate_g": s}
    if s == FAIL:
        return {"verdict": "NO-GO CURRENT METHOD",
                "reason": "; ".join(gate_g_result.get("fail_reasons", [])) or "Gate G FAIL",
                "gate_g": s}
    return {"verdict": "CONDITIONAL GO",
            "reason": f"Gate G is {s}; unmet conditions: "
                      f"{gate_g_result.get('failed_conditions')}",
            "gate_g": s}
