"""Study 0 - known-answer linear-Gaussian simulation.

Verifies the DGP wiring, the ``lambda`` definition, the posterior shrinkage
formula and the MSE implementation against closed-form answers.  Chronos-2 is
never called here.

Model
-----
    x        ~ Normal(m, V)
    eta      ~ Normal(0, 1)
    e        = lambda * sqrt(V) * eta          so  sigma_e^2 = lambda^2 * V
    x_tilde  = x + e
    y        = beta * x + eps_y,  eps_y ~ Normal(0, sigma_y^2)

Predictors (all use y_hat = beta * x_hat)
    S0 prior-only     x_hat = m
    S1 noisy plug-in  x_hat = x_tilde
    S2 exact posterior w = V / (V + sigma_e^2);  x_hat = m + w * (x_tilde - m)

The shrinkage weight w = 1 / (1 + lambda^2) is the exact posterior coefficient
*of this linear-Gaussian model*.  It is not claimed to be the optimal way to mix
a Chronos quantile forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Study0Config
from .seeds import make_rng

REL_TOL = 0.01           # required agreement between simulation and closed form
DOMINANCE_TOL = 0.01     # Monte-Carlo slack for the "never worse" check
PREDICTORS = ("S0_prior", "S1_plugin", "S2_posterior")


def analytic_mse(beta: float, V: float, sigma_y: float, lam: float) -> dict[str, float]:
    sigma_e2 = lam**2 * V
    return {
        "S0_prior": beta**2 * V + sigma_y**2,
        "S1_plugin": beta**2 * sigma_e2 + sigma_y**2,
        "S2_posterior": beta**2 * V * sigma_e2 / (V + sigma_e2) + sigma_y**2,
    }


def posterior_weight(V: float, sigma_e2: float) -> float:
    return V / (V + sigma_e2)


def run_study0(cfg: Study0Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (raw repetition-level table, summary table, verdict dict)."""
    e, d = cfg.experiment, cfg.dgp
    n = e.n_repetitions
    m, V, beta, sigma_y = d.prior_mean, d.prior_variance, d.beta, d.target_noise_std

    # common random numbers across lambda: one draw of (x, eta, eps_y) reused
    rng = make_rng(e.seed, "study0")
    x = rng.normal(m, np.sqrt(V), size=n)
    eta = rng.normal(0.0, 1.0, size=n)
    eps_y = rng.normal(0.0, sigma_y, size=n) if sigma_y > 0 else np.zeros(n)
    y = beta * x + eps_y

    raw_frames, summary_rows = [], []
    for lam in cfg.grid.lambda_values:
        lam = float(lam)
        sigma_e2 = lam**2 * V
        x_tilde = x + lam * np.sqrt(V) * eta
        w = posterior_weight(V, sigma_e2)

        x_hat = {
            "S0_prior": np.full(n, m, dtype=float),
            "S1_plugin": x_tilde,
            "S2_posterior": m + w * (x_tilde - m),
        }
        ana = analytic_mse(beta, V, sigma_y, lam)
        for name in PREDICTORS:
            sq = (y - beta * x_hat[name]) ** 2
            mse = float(sq.mean())
            mc_se = float(sq.std(ddof=1) / np.sqrt(n))
            summary_rows.append({
                "lam": lam,
                "predictor": name,
                "mse_simulated": mse,
                "mse_analytic": ana[name],
                "relative_error": abs(mse - ana[name]) / ana[name],
                "monte_carlo_se": mc_se,
                "posterior_weight": w,
                "sigma_e2": sigma_e2,
                "n_repetitions": n,
            })
            raw_frames.append(pd.DataFrame({
                "lam": np.float64(lam),
                "predictor": pd.Categorical([name] * n, categories=list(PREDICTORS)),
                "repetition": np.arange(n, dtype=np.int32),
                "squared_error": sq,
            }))

    raw = pd.concat(raw_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    verdict = evaluate_study0(summary)
    return raw, summary, verdict


def evaluate_study0(summary: pd.DataFrame) -> dict:
    """Apply the four PASS conditions.  Any failure blocks Study 1A."""
    checks: list[dict] = []
    wide = summary.pivot(index="lam", columns="predictor", values="mse_simulated")
    ana = summary.pivot(index="lam", columns="predictor", values="mse_analytic")

    worst = float(summary["relative_error"].max())
    bad = summary.loc[summary["relative_error"] > REL_TOL, ["lam", "predictor", "relative_error"]]
    checks.append({
        "id": "C1_analytic_agreement",
        "description": f"every simulated MSE within {REL_TOL:.0%} of its closed form",
        "status": "PASS" if bad.empty else "FAIL",
        "detail": f"max relative error = {worst:.5f}",
        "offenders": bad.to_dict("records"),
    })

    lams = list(wide.index)
    below = [l for l in lams if l < 1.0]
    above = [l for l in lams if l > 1.0]
    ok_below = all(wide.loc[l, "S1_plugin"] < wide.loc[l, "S0_prior"] for l in below)
    ok_above = all(wide.loc[l, "S1_plugin"] > wide.loc[l, "S0_prior"] for l in above)
    at_one = None
    if 1.0 in wide.index:
        at_one = abs(wide.loc[1.0, "S1_plugin"] - wide.loc[1.0, "S0_prior"]) / wide.loc[1.0, "S0_prior"]
    ok_at_one = at_one is None or at_one <= REL_TOL
    checks.append({
        "id": "C2_crossing_near_lambda_one",
        "description": "prior-only and plug-in cross in the neighbourhood of lambda = 1",
        "status": "PASS" if (ok_below and ok_above and ok_at_one) else "FAIL",
        "detail": (f"plugin<prior for all lambda<1: {ok_below}; plugin>prior for all lambda>1: "
                   f"{ok_above}; relative gap at lambda=1: "
                   f"{'n/a' if at_one is None else f'{at_one:.5f}'}"),
    })

    viol = []
    for l in lams:
        post = wide.loc[l, "S2_posterior"]
        ref = min(wide.loc[l, "S0_prior"], wide.loc[l, "S1_plugin"])
        if post > ref * (1.0 + DOMINANCE_TOL):
            viol.append({"lam": float(l), "posterior": float(post), "best_fixed": float(ref)})
    checks.append({
        "id": "C3_posterior_dominance",
        "description": "posterior predictor never worse than either fixed predictor",
        "status": "PASS" if not viol else "FAIL",
        "detail": f"tolerance {DOMINANCE_TOL:.0%}; violations: {len(viol)}",
        "offenders": viol,
    })

    if 0.0 in wide.index:
        gap = abs(wide.loc[0.0, "S2_posterior"] - wide.loc[0.0, "S1_plugin"])
        ok_zero = gap <= 1e-12 * max(1.0, abs(wide.loc[0.0, "S1_plugin"]))
        detail = f"|posterior - plugin| at lambda=0 = {gap:.3e}"
    else:
        ok_zero, detail = False, "lambda = 0 not present in the grid"
    checks.append({
        "id": "C4_lambda_zero_identity",
        "description": "at lambda = 0 the posterior predictor equals the plug-in predictor",
        "status": "PASS" if ok_zero else "FAIL",
        "detail": detail,
    })

    status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "max_relative_error": worst,
        "analytic_table": ana.reset_index().to_dict("records"),
    }


def study0_report(cfg: Study0Config, summary: pd.DataFrame, verdict: dict) -> str:
    lines = [
        "# Study 0 - known-answer linear-Gaussian simulation",
        "",
        f"Status: **{verdict['status']}**",
        "",
        "## Setup",
        "",
        "```",
        f"name            {cfg.experiment.name}",
        f"seed            {cfg.experiment.seed}",
        f"n_repetitions   {cfg.experiment.n_repetitions}",
        f"prior_mean m    {cfg.dgp.prior_mean}",
        f"prior_var  V    {cfg.dgp.prior_variance}",
        f"beta            {cfg.dgp.beta}",
        f"sigma_y         {cfg.dgp.target_noise_std}",
        f"lambda grid     {list(cfg.grid.lambda_values)}",
        "```",
        "",
        "Common random numbers: one draw of (x, eta, eps_y) is reused across every",
        "lambda, so differences across lambda are paired. [확인]",
        "",
        "## Simulated vs analytic MSE",
        "",
        "```",
        "lam    predictor      mse_sim     mse_analytic   rel_err     mc_se",
        "──────────────────────────────────────────────────────────────────────",
    ]
    for _, r in summary.iterrows():
        lines.append(f"{r['lam']:<6.2f} {r['predictor']:<14s} {r['mse_simulated']:<11.6f} "
                     f"{r['mse_analytic']:<14.6f} {r['relative_error']:<11.6f} "
                     f"{r['monte_carlo_se']:.6f}")
    lines += ["```", "", "## Checks", ""]
    for c in verdict["checks"]:
        lines.append(f"- **{c['id']}** - {c['status']} - {c['description']}")
        lines.append(f"  - {c['detail']}")
    lines += [
        "",
        "## Interpretation",
        "",
        "[확인] w = V / (V + sigma_e^2) = 1 / (1 + lambda^2) is the exact posterior mean",
        "coefficient of this linear-Gaussian model.  It is a check on the simulation",
        "machinery only; it is *not* claimed to be the optimal way to blend a Chronos",
        "quantile forecast with a prior.",
        "",
        "[확인] The prior-only and plug-in predictors cross at lambda = 1 because at that",
        "point sigma_e^2 = V: the covariate forecast is exactly as uncertain as the",
        "unconditional prior.  This is a property of squared error in this model and is",
        "used as a reference line only.",
    ]
    return "\n".join(lines) + "\n"
