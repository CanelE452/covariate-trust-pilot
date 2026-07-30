"""Gate A / B / C evaluation.

Every gate returns exactly one of PASS, FAIL, INCONCLUSIVE, NOT_RUN together
with the numbers behind the verdict.  Nothing here changes the DGP, the grid or
a threshold in response to a result.

Verbal parts of the specification are made executable through explicit,
configurable constants (``gates.*`` in the config), and the operative definition
is repeated in the returned ``criteria`` field so the report can state it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .bootstrap import BOOTSTRAP_UNIT, paired_bootstrap
from .config import PilotConfig

PASS, FAIL, INCONCLUSIVE, NOT_RUN = "PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN"

GATE_A_PRIMARY_SHARES = (0.50, 0.75)
NEGATIVE_CONTROL_SHARE = 0.0
LOW_NOISE_MAX_LAMBDA = 0.5
HIGH_NOISE_MIN_LAMBDA = 1.5


def _boot(df: pd.DataFrame, baseline: str, treatment: str, cfg: PilotConfig, tag: str):
    return paired_bootstrap(
        df[BOOTSTRAP_UNIT].to_numpy(),
        df[baseline].to_numpy(),
        df[treatment].to_numpy(),
        n_resamples=cfg.bootstrap.n_resamples,
        confidence_level=cfg.bootstrap.confidence_level,
        seed_parts=(cfg.experiment.master_seed, "bootstrap", tag),
    )


def cell_summary(task_metrics: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    """Per (share, horizon, lambda) summary with paired bootstrap intervals."""
    rows = []
    keys = ["nominal_covariate_share", "horizon", "lam"]
    for (share, horizon, lam), g in task_metrics.groupby(keys, sort=True):
        tag = f"cell_s{share}_h{horizon}_l{lam}"
        b31 = _boot(g, "wql_m1", "wql_m3", cfg, tag + "_m3")
        b21 = _boot(g, "wql_m1", "wql_m2", cfg, tag + "_m2")
        rows.append({
            "nominal_covariate_share": share,
            "horizon": horizon,
            "lam": lam,
            "n_series": int(g[BOOTSTRAP_UNIT].nunique()),
            "wql_m0": float(g["wql_m0"].mean()),
            "wql_m1": float(g["wql_m1"].mean()),
            "wql_m2": float(g["wql_m2"].mean()),
            "wql_m3": float(g["wql_m3"].mean()),
            "v_future_mean": b31.mean_diff,
            "v_future_median": b31.median_diff,
            "v_future_sd": b31.sd_diff,
            "v_future_mc_se": b31.monte_carlo_se,
            "v_future_ci_low": b31.ci_low,
            "v_future_ci_high": b31.ci_high,
            "v_future_rel": b31.relative_improvement,
            "m3_win_rate": b31.win_rate,
            "v_oracle_mean": b21.mean_diff,
            "v_oracle_ci_low": b21.ci_low,
            "v_oracle_ci_high": b21.ci_high,
            "v_oracle_rel": b21.relative_improvement,
            "m2_win_rate": b21.win_rate,
            "harm_rate": float(g["harm_m3"].mean()),
            "relative_delta_mean": float(g["relative_delta_m3"].mean()),
            "median_nmae_m1": float(g["nmae_m1"].median()),
            "median_nmae_m3": float(g["nmae_m3"].median()),
            "median_mse_m1": float(g["mse_m1"].median()),
            "median_mse_m3": float(g["mse_m3"].median()),
            "crossing_rate_m3": float(g["crossing_m3"].mean()),
        })
    return pd.DataFrame(rows)


def monte_carlo_table(task_metrics: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    """Monte-Carlo precision of the primary paired contrast per cell."""
    rows = []
    for (share, horizon, lam), g in task_metrics.groupby(
            ["nominal_covariate_share", "horizon", "lam"], sort=True):
        d = (g["wql_m1"] - g["wql_m3"]).to_numpy()
        n = g[BOOTSTRAP_UNIT].nunique()
        rows.append({
            "nominal_covariate_share": share,
            "horizon": horizon,
            "lam": lam,
            "n_units": int(n),
            "paired_mean_diff": float(d.mean()),
            "paired_median_diff": float(np.median(d)),
            "paired_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "monte_carlo_se": float(d.std(ddof=1) / np.sqrt(n)) if len(d) > 1 else float("nan"),
            "half_width_95_normal": float(1.96 * d.std(ddof=1) / np.sqrt(n)) if len(d) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Gate A ---

def gate_a(task_metrics: pd.DataFrame, cfg: PilotConfig) -> dict:
    """Does Chronos-2 exploit an accurate future covariate?  M2 vs M1."""
    g = cfg.gates
    criteria = {
        "comparison": "M2 (oracle future covariate) vs M1 (past covariate only)",
        "primary_subset": f"nominal share in {list(GATE_A_PRIMARY_SHARES)}, horizons {cfg.grid.horizons}",
        "pass": (f"both horizons improve, aggregate relative WQL improvement >= "
                 f"{g.clean_gain_pass:.0%}, paired bootstrap CI strictly on the improvement side, "
                 f"and no meaningful gain at r=0"),
        "fail": (f"aggregate improvement <= {g.clean_gain_fail:.0%}, or the CI lies strictly on the "
                 f"degradation side, or the r=0 negative control shows a similar gain "
                 f"(>= {g.negative_control_ratio:.0%} of the primary gain and >= {g.clean_gain_fail:.0%})"),
        "note": ("M2 is lambda-invariant; the M2 rows are deduplicated to one per "
                 "(series, share, horizon) so the aggregate is not lambda-weighted."),
    }

    # M1/M2 do not depend on lambda -> deduplicate to avoid fake replication
    dedup = task_metrics.drop_duplicates(
        subset=[BOOTSTRAP_UNIT, "nominal_covariate_share", "horizon"])
    primary = dedup[dedup["nominal_covariate_share"].round(6).isin(
        [round(s, 6) for s in GATE_A_PRIMARY_SHARES])]
    if primary.empty:
        return {"gate": "A", "status": NOT_RUN, "criteria": criteria,
                "detail": "primary subset is empty for this grid"}

    per_horizon = {}
    for h, gh in primary.groupby("horizon"):
        b = _boot(gh, "wql_m1", "wql_m2", cfg, f"gateA_h{h}")
        per_horizon[int(h)] = b.to_dict()

    agg = _boot(primary, "wql_m1", "wql_m2", cfg, "gateA_agg")
    nc = dedup[dedup["nominal_covariate_share"].round(6) == round(NEGATIVE_CONTROL_SHARE, 6)]
    nc_boot = _boot(nc, "wql_m1", "wql_m2", cfg, "gateA_nc") if not nc.empty else None

    all_horizons_improve = all(v["relative_improvement"] > 0 for v in per_horizon.values())
    aggregate_ok = agg.relative_improvement >= g.clean_gain_pass
    ci_ok = agg.ci_favours_treatment
    nc_rel = nc_boot.relative_improvement if nc_boot else 0.0
    nc_clean = nc_rel < g.clean_gain_pass
    nc_similar = (nc_rel >= g.negative_control_ratio * agg.relative_improvement
                  and nc_rel >= g.clean_gain_fail)

    if all_horizons_improve and aggregate_ok and ci_ok and nc_clean:
        status = PASS
    elif (agg.relative_improvement <= g.clean_gain_fail) or agg.ci_favours_baseline or nc_similar:
        status = FAIL
    else:
        status = INCONCLUSIVE

    return {
        "gate": "A",
        "status": status,
        "criteria": criteria,
        "aggregate": agg.to_dict(),
        "per_horizon": per_horizon,
        "negative_control": nc_boot.to_dict() if nc_boot else None,
        "checks": {
            "all_horizons_improve": bool(all_horizons_improve),
            "aggregate_relative_improvement": float(agg.relative_improvement),
            "aggregate_improvement_meets_pass": bool(aggregate_ok),
            "ci_favours_m2": bool(ci_ok),
            "ci_favours_m1": bool(agg.ci_favours_baseline),
            "negative_control_relative_improvement": float(nc_rel),
            "negative_control_clean": bool(nc_clean),
            "negative_control_similar_gain": bool(nc_similar),
        },
    }


# ------------------------------------------------------------------ Gate B ---

def gate_b(task_metrics: pd.DataFrame, cells: pd.DataFrame, cfg: PilotConfig) -> dict:
    """Is there a benefit-to-harm boundary as covariate error grows?  M3 vs M1."""
    g = cfg.gates
    criteria = {
        "comparison": "M3 (forecasted future covariate) vs M1 (past covariate only)",
        "v_future": "V_future = WQL(M1) - WQL(M3); positive means the future covariate helped",
        "pass": (f"at least 2 (share, horizon) curves with mean V_future > 0 for lambda <= "
                 f"{LOW_NOISE_MAX_LAMBDA} and mean V_future < 0 for lambda >= {HIGH_NOISE_MIN_LAMBDA} "
                 f"on the same curve; >= {g.dose_response_curve_fraction:.0%} of curves decreasing in "
                 f"lambda (Spearman rho < 0); at least one high-noise cell with harm rate >= "
                 f"{g.high_noise_harm_rate:.0%}"),
        "fail": ("no curve shows a low-noise benefit, or no curve shows a high-noise harmful region, "
                 "or fewer than half of the curves decrease in lambda"),
        "harm_definition": f"WQL(M3) > {1 + g.harm_relative_threshold:.2f} * WQL(M1) on a task",
        "note": ("lambda = 1 is a reference line taken from the Study-0 linear-Gaussian model; "
                 "it is not a theoretical WQL boundary."),
    }

    curves = []
    for (share, horizon), c in cells.groupby(["nominal_covariate_share", "horizon"], sort=True):
        c = c.sort_values("lam")
        low = c[c["lam"] <= LOW_NOISE_MAX_LAMBDA]["v_future_mean"]
        high = c[c["lam"] >= HIGH_NOISE_MIN_LAMBDA]["v_future_mean"]
        rho = np.nan
        if len(c) >= 3 and c["lam"].nunique() >= 3:
            rho = float(stats.spearmanr(c["lam"], c["v_future_mean"]).statistic)
        curves.append({
            "nominal_covariate_share": float(share),
            "horizon": int(horizon),
            "low_noise_mean_v_future": float(low.mean()) if len(low) else float("nan"),
            "high_noise_mean_v_future": float(high.mean()) if len(high) else float("nan"),
            "low_noise_benefit": bool(len(low) and low.mean() > 0),
            "high_noise_harmful": bool(len(high) and high.mean() < 0),
            "spearman_rho_lambda_vs_v_future": rho,
            "decreasing": bool(rho < 0) if np.isfinite(rho) else False,
            "max_harm_rate_high_noise": float(
                c[c["lam"] >= HIGH_NOISE_MIN_LAMBDA]["harm_rate"].max())
            if (c["lam"] >= HIGH_NOISE_MIN_LAMBDA).any() else float("nan"),
        })
    cur = pd.DataFrame(curves)

    n_boundary = int((cur["low_noise_benefit"] & cur["high_noise_harmful"]).sum())
    frac_decreasing = float(cur["decreasing"].mean()) if len(cur) else 0.0
    any_low_benefit = bool(cur["low_noise_benefit"].any())
    any_high_harm = bool(cur["high_noise_harmful"].any())
    high_noise_cells = cells[cells["lam"] >= HIGH_NOISE_MIN_LAMBDA]
    max_harm = float(high_noise_cells["harm_rate"].max()) if len(high_noise_cells) else float("nan")
    harm_ok = bool(np.isfinite(max_harm) and max_harm >= g.high_noise_harm_rate)

    if (n_boundary >= 2 and frac_decreasing >= g.dose_response_curve_fraction and harm_ok):
        status = PASS
    elif (not any_low_benefit) or (not any_high_harm) or (frac_decreasing < 0.5):
        status = FAIL
    else:
        status = INCONCLUSIVE

    return {
        "gate": "B",
        "status": status,
        "criteria": criteria,
        "curves": cur.to_dict("records"),
        "checks": {
            "n_curves": int(len(cur)),
            "n_curves_with_boundary": n_boundary,
            "fraction_curves_decreasing": frac_decreasing,
            "any_low_noise_benefit": any_low_benefit,
            "any_high_noise_harm": any_high_harm,
            "max_high_noise_harm_rate": max_harm,
            "harm_rate_requirement_met": harm_ok,
        },
    }


# ------------------------------------------------------------------ Gate C ---

def gate_c(task_metrics: pd.DataFrame, cfg: PilotConfig) -> dict:
    """Is there oracle headroom for per-task admission over any fixed policy?"""
    g = cfg.gates
    criteria = {
        "oracle": "per task: min(WQL(M1), WQL(M3)) - an upper bound on any selector",
        "fixed_policies": "always_no_future = M1 everywhere; always_use_future = M3 everywhere",
        "best_fixed": "the fixed policy with the lower mean WQL on this equally weighted grid",
        "pass": (f"oracle improves on best fixed by >= {g.oracle_headroom_pass:.0%}, paired bootstrap CI "
                 f"strictly on the improvement side, and both M1 and M3 win somewhere at >= 2 share levels"),
        "fail": (f"oracle headroom <= {g.oracle_headroom_fail:.0%}, or one policy wins on >= "
                 f"{g.degenerate_policy_share:.0%} of tasks"),
        "weighting_caveat": ("the grid weights every (share, lambda, horizon) cell equally; this is a "
                             "design choice and does not represent any deployment frequency"),
    }

    df = task_metrics.copy()
    df["wql_oracle"] = np.minimum(df["wql_m1"], df["wql_m3"])
    mean_m1 = float(df["wql_m1"].mean())
    mean_m3 = float(df["wql_m3"].mean())
    mean_oracle = float(df["wql_oracle"].mean())
    best_fixed_name = "always_no_future" if mean_m1 <= mean_m3 else "always_use_future"
    best_fixed_col = "wql_m1" if best_fixed_name == "always_no_future" else "wql_m3"
    best_fixed_mean = min(mean_m1, mean_m3)

    boot = _boot(df.assign(_bf=df[best_fixed_col]), "_bf", "wql_oracle", cfg, "gateC")
    headroom = (best_fixed_mean - mean_oracle) / best_fixed_mean

    m3_wins = df["wql_m3"] < df["wql_m1"]
    per_share = []
    for share, gs in df.groupby("nominal_covariate_share"):
        w = float((gs["wql_m3"] < gs["wql_m1"]).mean())
        per_share.append({"nominal_covariate_share": float(share), "m3_win_rate": w,
                          "both_present": bool(0.0 < w < 1.0)})
    shares_with_both = int(sum(p["both_present"] for p in per_share))
    m3_win_overall = float(m3_wins.mean())
    degenerate = bool(max(m3_win_overall, 1.0 - m3_win_overall) >= g.degenerate_policy_share)

    if (headroom >= g.oracle_headroom_pass and boot.ci_favours_treatment and shares_with_both >= 2):
        status = PASS
    elif headroom <= g.oracle_headroom_fail or degenerate:
        status = FAIL
    else:
        status = INCONCLUSIVE

    return {
        "gate": "C",
        "status": status,
        "criteria": criteria,
        "means": {"always_no_future_m1": mean_m1, "always_use_future_m3": mean_m3,
                  "oracle": mean_oracle, "best_fixed": best_fixed_name,
                  "best_fixed_mean": best_fixed_mean},
        "bootstrap": boot.to_dict(),
        "per_share_win_rates": per_share,
        "checks": {
            "oracle_headroom": float(headroom),
            "headroom_meets_pass": bool(headroom >= g.oracle_headroom_pass),
            "ci_favours_oracle": bool(boot.ci_favours_treatment),
            "shares_with_both_winners": shares_with_both,
            "m3_overall_win_rate": m3_win_overall,
            "degenerate_policy": degenerate,
        },
    }


def gates_abc(task_metrics: pd.DataFrame, cells: pd.DataFrame, cfg: PilotConfig) -> dict:
    a = gate_a(task_metrics, cfg)
    if a["status"] == FAIL:
        return {"A": a,
                "B": {"gate": "B", "status": NOT_RUN,
                      "reason": "Gate A FAIL - downstream gates are not evaluated"},
                "C": {"gate": "C", "status": NOT_RUN,
                      "reason": "Gate A FAIL - downstream gates are not evaluated"},
                "all_pass": False}
    b = gate_b(task_metrics, cells, cfg)
    c = gate_c(task_metrics, cfg)
    return {"A": a, "B": b, "C": c,
            "all_pass": all(x["status"] == PASS for x in (a, b, c))}
