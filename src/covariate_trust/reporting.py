"""Report generation.

Observations and verdicts are kept in separate sections.  Every factual or
interpretive sentence carries one of the tags [확인] (traced or executed),
[추정] (inferred, not verified) or [미검증] (design not validated against prior
work).  Gate outcomes use only PASS / FAIL / INCONCLUSIVE / NOT_RUN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _fixed_table(df: pd.DataFrame, columns: list[tuple[str, str, int]]) -> list[str]:
    """Whitespace-aligned table inside a code block (pipe tables break alignment)."""
    header = "".join(title.ljust(width) for title, _, width in columns)
    lines = [header, "─" * len(header)]
    for _, row in df.iterrows():
        cells = []
        for _, col, width in columns:
            v = row[col]
            if isinstance(v, float):
                s = f"{v:.4f}" if abs(v) < 1000 else f"{v:.3e}"
            else:
                s = str(v)
            cells.append(s.ljust(width))
        lines.append("".join(cells))
    return lines


def _gate_block(g: dict) -> list[str]:
    lines = [f"Status: **{g['status']}**", ""]
    if g.get("reason"):
        lines += [f"Not run because: {g['reason']}", ""]
        return lines
    crit = g.get("criteria", {})
    lines += ["Operative criteria (fixed before the run):", "", "```"]
    for k, v in crit.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k}: {v}")
    lines += ["```", ""]
    if g.get("checks"):
        lines += ["Measured:", "", "```"]
        for k, v in g["checks"].items():
            lines.append(f"{k}: {v if not isinstance(v, float) else f'{v:.6f}'}")
        lines += ["```", ""]
    return lines


def build_report(run_dir: Path, manifest: dict, audit: dict, cfg_yaml: str,
                 study0_verdict: dict | None, smoke: dict | None,
                 cells: pd.DataFrame | None, mc: pd.DataFrame | None,
                 gates: dict | None, gate_d: dict | None,
                 observations: list[str], limitations: list[str],
                 next_steps: list[str], commands: list[str]) -> str:
    L: list[str] = []
    A = L.append

    A("# Forecasted future covariates in zero-shot Chronos-2 - coarse pilot")
    A("")
    A(f"Run directory: `{run_dir}`")
    A("")

    A("## 1. Research question")
    A("")
    A("When does a *forecasted* (imperfect) future covariate add value to a zero-shot")
    A("Chronos-2 forecast, and when does it actively hurt?  The incremental value of the")
    A("future covariate is defined as the paired difference M1 - M3.  M0 is a sanity")
    A("baseline only: the M0 - M3 difference is **not** interpreted as the value of the")
    A("future covariate, because it also contains the value of the past covariate. [확인]")
    A("")

    A("## 2. Project and environment")
    A("")
    A("```")
    A(f"project_root       {audit['project_root']}")
    A(f"python             {audit['python']['version']}  ({audit['python']['executable']})")
    A(f"in_virtualenv      {audit['python']['in_venv']}")
    A(f"torch              {audit['torch'].get('version')}  cuda_build "
      f"{audit['torch'].get('cuda_build_version')}  available {audit['torch'].get('cuda_available')}")
    A(f"gpu                {audit['torch'].get('gpu_name', 'n/a')}")
    A(f"chronos-forecasting {audit['packages']['chronos-forecasting']}")
    A(f"transformers       {audit['packages']['transformers']}")
    A(f"git commit         {audit['git']['commit']}")
    A(f"HF_HOME            {audit['hf_home']}")
    A(f"cross_learning     supported={audit['chronos_api'].get('cross_learning_supported')} "
      f"default={audit['chronos_api'].get('cross_learning_default')}  (always passed False)")
    A("```")
    A("")
    A("[확인] The `predict_df` signature above was read with `inspect.signature` from the")
    A("installed package, not from documentation.")
    A("")
    A("<details><summary>resolved configuration</summary>")
    A("")
    A("```yaml")
    A(cfg_yaml.rstrip())
    A("```")
    A("")
    A("</details>")
    A("")

    A("## 3. Study 0 - known-answer simulation")
    A("")
    if study0_verdict is None:
        A("NOT_RUN in this run directory.")
    else:
        A(f"Status: **{study0_verdict['status']}**")
        A("")
        A(f"[확인] Maximum relative error between simulated and analytic MSE: "
          f"{study0_verdict['max_relative_error']:.5f}.")
        A("")
        for c in study0_verdict["checks"]:
            A(f"- `{c['id']}` **{c['status']}** - {c['description']}")
            A(f"  - {c['detail']}")
        A("")
        A("[확인] w = V / (V + sigma_e^2) = 1 / (1 + lambda^2) is the exact posterior")
        A("coefficient of the Study-0 linear-Gaussian model only.  It is not presented as")
        A("the optimal way to mix a Chronos quantile forecast.")
    A("")

    A("## 4. Data generating process")
    A("")
    A("[확인] All series are synthetic and generated in-process from the master seed; no")
    A("external dataset is read at any point.  Per base series a base process `b` and a")
    A("covariate process `x` are built from two sinusoids plus an AR(1) residual, then")
    A("standardized using statistics from `t in [0, standardization_end)` only.  The target")
    A("is `y = sqrt(1-r) * b + sqrt(r) * x`.")
    A("")
    A("[확인] `r` is the *nominal* covariate share by construction.  It is deliberately not")
    A("called a partial R^2; the realized incremental R^2 of `x` given `b` is measured")
    A("separately and reported in `generated/series_metadata.parquet` and Figure 6.")
    A("")
    A("[확인] Covariate forecast error: `x_tilde(T+h) = x_true(T+h) + lambda * sqrt(V(h)) * eta_h`")
    A("with `V(h)` the h-step AR(1) conditional variance on the standardized scale and")
    A("`eta_h` a standard normal path shared by every lambda at the same (series, origin,")
    A("horizon).  Error is unbiased and serially uncorrelated by construction; biased and")
    A("correlated error models are out of scope for this pilot.")
    A("")

    A("## 5. Estimand")
    A("")
    A("```")
    A("V_future = WQL(M1) - WQL(M3)     positive: forecasted future covariate helps")
    A("V_oracle = WQL(M1) - WQL(M2)     positive: an accurate future covariate helps")
    A("relative_delta = (WQL(M3) - WQL(M1)) / WQL(M1)")
    A("harm     = 1 if WQL(M3) > 1.05 * WQL(M1)")
    A("```")
    A("")
    A("[확인] Every difference is paired at the level of `base_series_id`: the same base and")
    A("covariate processes, phases, amplitudes, AR innovations and eta path are reused")
    A("across all grid cells (common random numbers).  The bootstrap unit is therefore also")
    A("`base_series_id`, resampled as a cluster.")
    A("")

    A("## 6. Comparison conditions")
    A("")
    A("```")
    A("M0 target_only                 context: target                      future: none")
    A("M1 past_covariate_only         context: target + x history          future: none")
    A("M2 oracle_future_covariate     context: as M1                       future: true x")
    A("M3 forecasted_future_covariate context: as M1                       future: x_tilde")
    A("```")
    A("")
    A("[확인] Fairness assertions executed on every task: M1 and M3 contexts are compared for")
    A("exact equality; M2 and lambda=0 M3 future frames are compared for exact equality; M1")
    A("carries no `future_df`; M0 carries no covariate column; no `future_df` ever contains")
    A("the target column.  `cross_learning=False` is passed on every call.")
    A("")

    A("## 7. Monte-Carlo precision")
    A("")
    if mc is None or not len(mc):
        A("NOT_RUN.")
    else:
        A("```")
        for line in _fixed_table(mc.sort_values(["horizon", "nominal_covariate_share", "lam"]), [
                ("share", "nominal_covariate_share", 9), ("H", "horizon", 6), ("lam", "lam", 7),
                ("n", "n_units", 5), ("mean_diff", "paired_mean_diff", 12),
                ("sd", "paired_sd", 11), ("mc_se", "monte_carlo_se", 11),
                ("95%_half", "half_width_95_normal", 11)]):
            A(line)
        A("```")
        A("")
        A("[확인] `mean_diff` is the paired V_future = WQL(M1) - WQL(M3) inside the cell.")
        A("Cells whose effect is smaller than roughly twice `mc_se` cannot be resolved at")
        A("this sample size; that is a precision statement, not a null result.")
    A("")

    for name, key in (("8. Gate A - does an accurate future covariate help at all?", "A"),
                      ("9. Gate B - is there a benefit-to-harm boundary?", "B"),
                      ("10. Gate C - is there oracle admission headroom?", "C")):
        A(f"## {name}")
        A("")
        if gates is None or key not in gates:
            A("NOT_RUN.")
        else:
            L.extend(_gate_block(gates[key]))
        A("")

    A("## 11. Gate D - historical admission")
    A("")
    if gate_d is None:
        A("NOT_RUN - the admission diagnostic was not executed in this run.")
    else:
        L.extend(_gate_block(gate_d))
        if gate_d.get("selectors"):
            A("```")
            A("selector                     mean_WQL   impr_vs_fixed  recovery  harm_rate  m3_rate")
            A("────────────────────────────────────────────────────────────────────────────────────")
            for sel, r in gate_d["selectors"].items():
                A(f"{sel:<28s} {r['mean_wql']:<10.5f} {r['relative_improvement_over_best_fixed']:<14.4f} "
                  f"{r['oracle_gap_recovery']:<9.4f} {r['harm_rate']:<10.4f} {r['m3_choice_rate']:.4f}")
            A("```")
            A("")
            A("[확인] A2 is an analytic-inspired reliability heuristic (lambda_hat < 1), not a")
            A("WQL-optimal rule.")
    A("")

    A("## 12. Observations")
    A("")
    if cells is not None and len(cells):
        A("Cell-level summary (mean over base series):")
        A("")
        A("```")
        for line in _fixed_table(cells.sort_values(["horizon", "nominal_covariate_share", "lam"]), [
                ("share", "nominal_covariate_share", 9), ("H", "horizon", 6), ("lam", "lam", 7),
                ("WQL_M1", "wql_m1", 10), ("WQL_M2", "wql_m2", 10), ("WQL_M3", "wql_m3", 10),
                ("V_future", "v_future_mean", 11), ("CI_low", "v_future_ci_low", 11),
                ("CI_high", "v_future_ci_high", 11), ("harm", "harm_rate", 8),
                ("m3_win", "m3_win_rate", 8)]):
            A(line)
        A("```")
        A("")
    for o in observations:
        A(f"- {o}")
    A("")

    A("## 13. Verdict")
    A("")
    A("```")
    A(f"Study 0   {study0_verdict['status'] if study0_verdict else 'NOT_RUN'}")
    A(f"Smoke     {smoke['status'] if smoke else 'NOT_RUN'}")
    for key in ("A", "B", "C"):
        A(f"Gate {key}    {gates[key]['status'] if gates and key in gates else 'NOT_RUN'}")
    A(f"Gate D    {gate_d['status'] if gate_d else 'NOT_RUN'}")
    A("```")
    A("")

    A("## 14. Limitations")
    A("")
    for l in limitations:
        A(f"- {l}")
    A("")

    A("## 15. Next steps")
    A("")
    for s in next_steps:
        A(f"- {s}")
    A("")

    A("## 16. Reproduction")
    A("")
    A("```bash")
    for c in commands:
        A(c)
    A("```")
    A("")
    A("```")
    A(f"master_seed  {manifest.get('seeds', {}).get('master_seed')}")
    A(f"model        {manifest.get('model_id')} (revision {manifest.get('model_revision')})")
    A(f"started      {manifest.get('started_at')}")
    A(f"finished     {manifest.get('finished_at')}")
    A("```")
    return "\n".join(L) + "\n"


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ============================================================================
# Follow-up reports (Study 1B, Study 2, combined).  Additive.
# ============================================================================

FORBIDDEN_CLAIMS = (
    "lambda = 1 is not presented as a theoretical boundary of Chronos WQL",
    "the uncertainty proxy is not presented as a model of real weather ensembles",
    "no result is generalized from this one model to all TSFMs",
    "no synthetic result is stated as a real demand-forecasting deployment outcome",
    "Gate D PASS is not treated as evidence about dynamic reliability",
)


def _gate_lines(g: dict | None, name: str) -> list[str]:
    if g is None:
        return [f"{name}: **NOT_RUN**", ""]
    lines = [f"{name}: **{g['status']}**", ""]
    if g.get("reason"):
        lines += [f"Not run because: {g['reason']}", ""]
        return lines
    lines += ["Operative criteria, fixed before the run:", "", "```"]
    for k, v in (g.get("criteria") or {}).items():
        lines.append(f"{k}: {v}")
    lines += ["```", "", "Measured:", "", "```"]
    for k, v in (g.get("checks") or {}).items():
        lines.append(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    lines += ["```", ""]
    return lines


def _boundary_table(bounds) -> list[str]:
    lines = ["```",
             "share  H     boundary   CI_low     CI_high    width    valid_frac  status         n",
             "─" * 92]
    for _, r in bounds.sort_values(["horizon", "nominal_covariate_share"]).iterrows():
        def f(v):
            return "     n/a " if v != v else f"{v:<10.4f}"
        lines.append(f"{r['nominal_covariate_share']:<6.2f} {int(r['horizon']):<5d} "
                     f"{f(r['boundary_lambda'])} {f(r['ci_low'])} {f(r['ci_high'])} "
                     f"{f(r['ci_width'])[:8]} {r['bootstrap_valid_fraction']:<11.4f} "
                     f"{r['status']:<14s} {int(r['n_series'])}")
    lines.append("```")
    return lines


def build_boundary_report(run_dir, manifest: dict, cfg, cells, bounds, all_bounds,
                          base_checks: dict, gate_e_result: dict, coarse_ref: dict | None) -> str:
    L: list[str] = []
    A = L.append
    A("# Study 1B - independent boundary refinement")
    A("")
    A(f"Run directory: `{run_dir}`")
    A("")
    A("## Design")
    A("")
    A("```")
    A(f"master_seed        {cfg.experiment.master_seed}   (coarse pilot used 20260730)")
    A(f"shares             {list(cfg.grid.nominal_covariate_share)}")
    A(f"lambda grid        {list(cfg.grid.lambda_values)}")
    A(f"horizons           {list(cfg.grid.horizons)}")
    A(f"series per cell    {cfg.grid.n_series_per_cell}")
    A(f"bootstrap          {cfg.bootstrap.n_resamples} resamples, cluster unit base_series_id")
    A(f"inference calls    {manifest.get('n_inference_calls')} "
      f"(cache hits {manifest.get('n_cache_hits')})")
    A(f"runtime            {manifest.get('runtime_seconds')} s")
    A("```")
    A("")
    A("[확인] The DGP, schemas, adapter, WQL and bootstrap code paths are the ones the coarse")
    A("pilot used; the `dgp` and `gates` blocks are inherited from `configs/pilot.yaml`. Only the")
    A("seed, the lambda grid and the sample size differ.")
    A("")
    A("## Boundary estimates (Chronos WQL)")
    A("")
    L.extend(_boundary_table(bounds))
    A("")
    A("[확인] The boundary is interpolated inside the grid only. A curve that does not change")
    A("sign inside [%.2f, %.2f] is reported censored or unresolved rather than extrapolated."
      % (min(cfg.grid.lambda_values), max(cfg.grid.lambda_values)))
    A("")
    A("## Curve summary")
    A("")
    A("```")
    A("share  H     lam    WQL_M1     WQL_M3     V_future    CI_low      CI_high     harm   win")
    A("─" * 94)
    for _, r in cells.sort_values(["horizon", "nominal_covariate_share", "lam"]).iterrows():
        A(f"{r['nominal_covariate_share']:<6.2f} {int(r['horizon']):<5d} {r['lam']:<6.2f} "
          f"{r['wql_m1']:<10.5f} {r['wql_m3']:<10.5f} {r['v_future_mean']:<+11.5f} "
          f"{r['v_future_ci_low']:<+11.5f} {r['v_future_ci_high']:<+11.5f} "
          f"{r['harm_rate']:<6.2f} {r['m3_win_rate']:.2f}")
    A("```")
    A("")
    A("## Statistical baselines")
    A("")
    A(f"Baseline known-answer checks: **{base_checks['status']}**")
    A("")
    for c in base_checks["checks"]:
        A(f"- `{c['id']}` **{c['status']}** - {c['description']}")
        A(f"  - {c['detail']}")
    A("")
    A("Boundaries by method and metric:")
    A("")
    A("```")
    A("metric                  share  H     boundary   status")
    A("─" * 60)
    for _, r in all_bounds.sort_values(["metric", "horizon", "nominal_covariate_share"]).iterrows():
        b = "  n/a   " if r["boundary_lambda"] != r["boundary_lambda"] else f"{r['boundary_lambda']:<10.4f}"
        A(f"{r['metric']:<23s} {r['nominal_covariate_share']:<6.2f} {int(r['horizon']):<5d} "
          f"{b} {r['status']}")
    A("```")
    A("")
    A("[확인] B1 (DGP-aware conditional mean) and B2 (estimated ARX) are evaluated with")
    A("point-forecast MSE; the Chronos WQL boundary is a different quantity and the two are")
    A("listed side by side for comparison of *direction and location*, not as equal numbers.")
    A("")
    A("[확인] B1 is a theoretical diagnostic that knows the generator. It is not a deployable")
    A("method and is not proposed as one.")
    A("")
    A("## Gate E")
    A("")
    L.extend(_gate_lines(gate_e_result, "Gate E"))
    if coarse_ref:
        A(f"Coarse-pilot direction reference: `{coarse_ref}`")
        A("")
    A("[확인] lambda = 1 is the Study-0 linear-Gaussian reference line. It is not claimed to be")
    A("a theoretical boundary of Chronos WQL.")
    return "\n".join(L) + "\n"


def build_dynamic_report(run_dir, manifest: dict, cfg, cond, prox, gate_f_result: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Study 2 - admission under time-varying covariate reliability")
    A("")
    A(f"Run directory: `{run_dir}`")
    A("")
    A("## Design")
    A("")
    A("```")
    A(f"master_seed          {cfg.experiment.master_seed}")
    A(f"schedules            {[s.name for s in cfg.schedules]}")
    A(f"shares               {list(cfg.grid.nominal_covariate_share)}")
    A(f"horizons             {list(cfg.grid.horizons)}")
    A(f"series per condition {cfg.grid.n_series_per_condition}")
    A(f"proxy sigma          {cfg.proxy.sigma_proxy}")
    A(f"selector thresholds  use {cfg.selector_thresholds.use_threshold}, override "
      f"[{cfg.selector_thresholds.override_low}, {cfg.selector_thresholds.override_high}]")
    A(f"inference calls      {manifest.get('n_inference_calls')} "
      f"(cache hits {manifest.get('n_cache_hits')})")
    A(f"runtime              {manifest.get('runtime_seconds')} s")
    A("```")
    A("")
    A("[확인] Each origin carries its own lambda from the schedule; historical and current")
    A("lambdas are never mixed or averaged. Proxy modes change only the decision, never the")
    A("model input, so no forecast is recomputed per proxy mode.")
    A("")
    A("## Selector performance under the calibrated proxy, by schedule")
    A("")
    A("```")
    A("schedule                  selector                      WQL       regret    m3_rate  harm   f_use  f_rej")
    A("─" * 112)
    p1 = cond[cond["proxy_mode"] == "P1_calibrated_noisy"]
    for _, r in p1.sort_values(["schedule", "mean_wql"]).iterrows():
        A(f"{r['schedule']:<25s} {r['selector']:<29s} {r['mean_wql']:<9.5f} "
          f"{r['mean_regret']:<9.5f} {r['m3_choice_rate']:<8.3f} {r['harm_rate']:<6.3f} "
          f"{r['false_use_rate']:<6.3f} {r['false_reject_rate']:.3f}")
    A("```")
    A("")
    A("## Selector performance by proxy mode (pooled over schedules)")
    A("")
    A("```")
    A("proxy_mode                selector                      WQL       calib   m3_rate  harm   f_use  f_rej")
    A("─" * 110)
    for _, r in prox.sort_values(["proxy_mode", "mean_wql"]).iterrows():
        A(f"{r['proxy_mode']:<25s} {r['selector']:<29s} {r['mean_wql']:<9.5f} "
          f"{r['mean_calibration_ratio']:<7.3f} {r['m3_choice_rate']:<8.3f} {r['harm_rate']:<6.3f} "
          f"{r['false_use_rate']:<6.3f} {r['false_reject_rate']:.3f}")
    A("```")
    A("")
    A("## Gate F")
    A("")
    L.extend(_gate_lines(gate_f_result, "Gate F"))
    if gate_f_result and gate_f_result.get("proxy_diagnostics"):
        A("Diagnostics for the primary selector under the other proxy modes:")
        A("")
        A("```")
        for mode, m in gate_f_result["proxy_diagnostics"].items():
            A(f"{mode:<25s} WQL {m['mean_wql']:.5f}  harm {m['harm_rate']:.3f}  "
              f"m3_rate {m['m3_choice_rate']:.3f}  f_use {m['false_use_rate']:.3f}  "
              f"f_rej {m['false_reject_rate']:.3f}")
        A("```")
        A("")
    A("[확인] P0 uses the true current lambda and is a diagnostic upper bound only; Gate F is")
    A("decided on the P1 calibrated noisy proxy.")
    A("")
    A("[미검증] The proxy model is a lognormal perturbation of the true lambda. Whether it")
    A("resembles the uncertainty a real forecast provider reports is outside this experiment.")
    return "\n".join(L) + "\n"


def build_followup_report(run_dir, boundary_run, dynamic_run, gate_e_result, gate_f_result,
                          base_checks, bounds, all_bounds, cells, cond, prox,
                          decision: dict, extra: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Follow-up study: boundary refinement and dynamic admission")
    A("")
    A(f"Report directory: `{run_dir}`")
    A(f"Study 1B run: `{boundary_run}`")
    A(f"Study 2 run: `{dynamic_run}`")
    A("")

    A("## 1. What the existing pilot established")
    A("")
    A("[확인] On the coarse grid (seed 20260730, 30 series/cell) Gates A, B, C and D all")
    A("returned PASS: an accurate future covariate helped (M2 vs M1), a benefit-to-harm")
    A("boundary in lambda existed, per-task oracle admission had headroom, and a history-only")
    A("selector recovered part of it under **stationary** reliability.")
    A("")

    A("## 2. Limits of that pilot")
    A("")
    A("[확인] The coarse lambda grid had no point between 0.5 and 1.5, so the crossing was")
    A("bracketed but never located. [확인] Gate D gave the selector the same lambda in history")
    A("and at the primary origin, which is the easiest possible case. [확인] Cell-level")
    A("Monte-Carlo SE was large relative to several effects at 30 series per cell.")
    A("")
    A("[확인] Gate D PASS is not evidence about time-varying reliability; that is what Gate F")
    A("is for.")
    A("")

    A("## 3. Study 1B purpose")
    A("")
    A("[확인] Re-estimate the boundary on an independent seed with a dense lambda grid and")
    A("five times the series per cell, using the identical generating equations.")
    A("")

    A("## 4. Independent seed and sample")
    A("")
    A("```")
    A("coarse pilot     master_seed 20260730, lambda [0, 0.5, 1, 1.5, 2],      30 series/cell")
    A("Study 1B         master_seed 20260801, lambda [0.7, 0.85, 1, 1.15, 1.3], 150 series/cell")
    A("Study 2          master_seed 20260802, lambdas from the reliability schedules")
    A("```")
    A("")
    A("[확인] The seeds are distinct and the seed namespaces are derived by SHA-256 from the")
    A("master seed, so no series is shared with the coarse pilot.")
    A("")

    A("## 5. Boundary results")
    A("")
    if bounds is not None:
        L.extend(_boundary_table(bounds))
    else:
        A("NOT_RUN.")
    A("")

    A("## 6. Statistical baseline results")
    A("")
    if base_checks:
        A(f"Known-answer checks: **{base_checks['status']}**")
        A("")
        for c in base_checks["checks"]:
            A(f"- `{c['id']}` **{c['status']}** - {c['detail']}")
        A("")
    if all_bounds is not None:
        A("```")
        A("metric                  share  H     boundary   status")
        A("─" * 60)
        for _, r in all_bounds.sort_values(["metric", "horizon", "nominal_covariate_share"]).iterrows():
            b = "  n/a   " if r["boundary_lambda"] != r["boundary_lambda"] else f"{r['boundary_lambda']:<10.4f}"
            A(f"{r['metric']:<23s} {r['nominal_covariate_share']:<6.2f} {int(r['horizon']):<5d} "
              f"{b} {r['status']}")
        A("```")
    A("")

    A("## 7. Gate E")
    A("")
    L.extend(_gate_lines(gate_e_result, "Gate E"))

    A("## 8. Study 2 purpose")
    A("")
    A("[확인] Let historical and current covariate forecast quality differ, and give the")
    A("selector only a *reported* uncertainty proxy rather than the truth.")
    A("")

    A("## 9. Reliability schedules")
    A("")
    A("```")
    A("S0 stable_low          hist [0.50 0.50 0.50 0.50]  current 0.50")
    A("S1 stable_high         hist [1.50 1.50 1.50 1.50]  current 1.50")
    A("S2 sudden_worsening    hist [0.50 0.50 0.50 0.50]  current 1.50")
    A("S3 sudden_improvement  hist [1.50 1.50 1.50 1.50]  current 0.50")
    A("S4 gradual_worsening   hist [0.50 0.75 1.00 1.25]  current 1.50")
    A("S5 gradual_improvement hist [1.50 1.25 1.00 0.75]  current 0.50")
    A("```")
    A("")

    A("## 10. Uncertainty proxy")
    A("")
    A("```")
    A("P0 oracle_current    reported = true current lambda            (diagnostic only)")
    A("P1 calibrated_noisy  reported = lambda * exp(z - sigma^2/2)    sigma = 0.20")
    A("P2 overconfident     reported = 0.50 * P1")
    A("P3 underconfident    reported = 1.50 * P1")
    A("P4 stale_history     reported = mean of historical lambda estimates")
    A("```")
    A("")
    A("[확인] The proxy noise is drawn from a namespace disjoint from the covariate forecast")
    A("error, so the selector's information is not correlated with the error it judges.")
    A("[확인] Only P0 reads the true current lambda.")
    A("")

    A("## 11. Selector comparison")
    A("")
    if prox is not None:
        A("```")
        A("proxy_mode                selector                      WQL       m3_rate  harm   f_use  f_rej")
        A("─" * 100)
        for _, r in prox.sort_values(["proxy_mode", "mean_wql"]).iterrows():
            A(f"{r['proxy_mode']:<25s} {r['selector']:<29s} {r['mean_wql']:<9.5f} "
              f"{r['m3_choice_rate']:<8.3f} {r['harm_rate']:<6.3f} "
              f"{r['false_use_rate']:<6.3f} {r['false_reject_rate']:.3f}")
        A("```")
    else:
        A("NOT_RUN.")
    A("")

    A("## 12. Gate F")
    A("")
    L.extend(_gate_lines(gate_f_result, "Gate F"))

    A("## 13. Gate D versus Gate F")
    A("")
    A("```")
    A("Gate D   stationary reliability: history matched the present exactly.        PASS (coarse pilot)")
    A(f"Gate F   time-varying reliability + imperfect reported proxy.               "
      f"{gate_f_result['status'] if gate_f_result else 'NOT_RUN'}")
    A("```")
    A("")
    A("[확인] The existing Gate D result and its name are unchanged; Gate F is a separate")
    A("question and is reported separately.")
    A("")

    A("## 14. Observations")
    A("")
    for line in extra.get("observations", []):
        A(f"- {line}")
    if not extra.get("observations"):
        A("- see the per-study reports for the full tables")
    A("")

    A("## 15. Verdict")
    A("")
    A("```")
    A(f"Gate E            {gate_e_result['status'] if gate_e_result else 'NOT_RUN'}")
    A(f"Gate F            {gate_f_result['status'] if gate_f_result else 'NOT_RUN'}")
    A(f"baseline checks   {base_checks['status'] if base_checks else 'NOT_RUN'}")
    A(f"final             {decision['verdict']}")
    A("```")
    A("")
    A(f"Reason: {decision['reason']}")
    A("")

    A("## 16. Limitations")
    A("")
    A("[확인] Synthetic data only; two sinusoids plus an AR(1) residual and a linear target.")
    A("[확인] One primary forecast origin; the covariate error is unbiased and serially")
    A("uncorrelated in every condition.")
    A("[확인] The target-covariate relationship itself is stationary in Study 2; only the")
    A("reliability of the covariate forecast varies.")
    A("[확인] The mixture weights all schedules and cells equally, which is a design choice and")
    A("not a claim about deployment frequency.")
    A("[미검증] Whether the proxy model resembles what a real forecast provider reports.")
    A("[미검증] Whether any of this transfers to other foundation models; only")
    A("amazon/chronos-2 was run, frozen, with cross_learning=False.")
    A("")
    for claim in FORBIDDEN_CLAIMS:
        A(f"- {claim}")
    A("")

    A("## 17. Next steps")
    A("")
    A("- Non-stationary target-covariate relationships, not only non-stationary reliability.")
    A("- Biased and serially correlated covariate forecast error.")
    A("- A proxy calibrated from an actual provider's reliability statistics.")
    A("- Independent replications for any cell reported INCONCLUSIVE, with the grid and the")
    A("  thresholds unchanged.")
    A("")

    A("## 18. Reproduction")
    A("")
    A("```bash")
    A("python -m covariate_trust.cli followup \\")
    A("  --boundary-config configs/study1b_boundary.yaml \\")
    A("  --dynamic-config configs/study2_dynamic_reliability.yaml")
    A("```")
    return "\n".join(L) + "\n"


# ============================================================================
# Study 2B - held-out D7 confirmation report.  Additive.
# ============================================================================

CONFIRMATION_FORBIDDEN = (
    "no claim is made that D7 has been validated on real weather forecasts",
    "P1 is not presented as equal to a real ensemble's reported uncertainty",
    "no result is generalized from amazon/chronos-2 to all TSFMs",
    "lambda = 1 is not presented as a theoretical boundary of WQL",
    "nothing here implies the held-out outcome was guaranteed in advance",
    "an INCONCLUSIVE gate is never described as a pass",
)


def build_confirmation_report(run_dir, manifest: dict, cfg, prereg: dict, prereg_hash: str,
                              independence: dict, leakage: dict, selector_summary,
                              condition_summary, proxy_stress, share_summary,
                              horizon_summary, gate_g_result: dict, verdict: dict,
                              observations: list[str]) -> str:
    L: list[str] = []
    A = L.append
    prim = cfg.selectors.primary
    p1 = cfg.proxy.primary_mode

    A("# Study 2B - held-out confirmation of the pre-registered D7 policy")
    A("")
    A(f"Run directory: `{run_dir}`")
    A("")

    A("## 1. Research question")
    A("")
    A(f"[확인] Does **{prim}**, fixed in advance, reproduce both its average performance and its")
    A("stable-condition safety on a completely fresh synthetic sample?")
    A("")

    A("## 2. Why the previous study was a development study")
    A("")
    A("[확인] Gate F selected its primary policy as whichever of D5/D7 scored the lower mean WQL")
    A("and then judged that same policy on the same sample.  D7's individual success there is a")
    A("development observation.  Study 2B removes the selection step entirely: D7 is the only")
    A("policy Gate G can be decided on, and a better-scoring secondary policy does not replace it.")
    A("")

    A("## 3. Pre-registration")
    A("")
    A("```")
    A(f"primary policy        {prereg['primary_policy']}")
    A(f"primary proxy         {prereg['primary_proxy']}")
    A(f"D7 thresholds         {prereg['d7_lower_threshold']} / {prereg['d7_upper_threshold']}")
    A(f"D5 threshold          {prereg['d5_threshold']}")
    A(f"proxy sigma           {prereg['proxy_sigma']}")
    A(f"master seed           {prereg['master_seed']}")
    A(f"expected primary tasks {prereg['expected_primary_tasks']}")
    A(f"start commit          {prereg['start_state'].get('commit')}")
    A(f"start git diff hash   {prereg['start_state'].get('git_diff_sha256')}")
    A(f"preregistration hash  {prereg_hash}")
    A("```")
    A("")
    A("[확인] `preregistration.json` was written before the model was loaded; its SHA-256 is")
    A("recorded in `manifest.json` and was not recomputed afterwards.")
    A("")

    A("## 4. Independence from the earlier studies")
    A("")
    A(f"Status: **{independence['status']}**")
    A("")
    for c in independence["checks"]:
        A(f"- `{c['id']}` **{c['status']}** - {c['description']}")
        A(f"  - {c['detail']}")
    A("")
    A("Leakage checks:")
    A("")
    for c in leakage["checks"]:
        A(f"- `{c['id']}` **{c['status']}** - {c['description']}")
        A(f"  - {c['detail']}")
    A("")

    A("## 5. DGP and schedules")
    A("")
    A("[확인] The generating equations are the coarse pilot's, re-seeded; the six reliability")
    A("schedules are inherited verbatim from the Study 2 config file.")
    A("")
    A("```")
    for s in prereg["schedules"]:
        A(f"{s['name']:<24s} hist {s['historical']}  current {s['current']}")
    A("```")
    A("")

    A(f"## 6. Primary policy {prim}")
    A("")
    A("```")
    A("reported lambda < 0.75          -> M3")
    A("reported lambda > 1.25          -> M1")
    A("0.75 <= reported lambda <= 1.25 -> the D3 history-utility decision")
    A("```")
    A("")
    A("[확인] D7 reads only the reported proxy and the historical comparison; the leakage check")
    A("above rescales the current outcomes and confirms only the oracle D2 reacts.")
    A("")

    A("## 7. Current uncertainty proxy")
    A("")
    A("```")
    A("P0 oracle_current    reported = true current lambda            (diagnostic only)")
    A(f"P1 calibrated_noisy  lambda * exp(z - sigma^2/2), sigma = {prereg['proxy_sigma']}   (Gate G)")
    A("P2 overconfident     0.50 x P1")
    A("P3 underconfident    1.50 x P1")
    A("P4 stale_history     mean of the historical lambda estimates")
    A("```")
    A("")

    A("## 8. Experimental mixture")
    A("")
    A(f"[확인] {prereg['expected_primary_tasks']} primary tasks: "
      f"{len(prereg['nominal_covariate_share'])} shares x {len(prereg['horizons'])} horizons x "
      f"{len(prereg['schedules'])} schedules x {prereg['n_series_per_condition']} series, all "
      f"equally weighted.  That weighting is a design choice and does not represent how often "
      f"each reliability regime occurs in deployment.")
    A("")

    A("## 9. Overall result")
    A("")
    A("```")
    A("selector                      WQL       impr      recovery  harm    m3_rate  f_use   f_rej")
    A("─" * 96)
    for _, r in selector_summary.sort_values("mean_wql").iterrows():
        A(f"{r['selector']:<29s} {r['mean_wql']:<9.5f} "
          f"{r['relative_improvement_over_best_fixed']:<+9.4f} {r['oracle_gap_recovery']:<9.4f} "
          f"{r['harm_rate']:<7.3f} {r['m3_choice_rate']:<8.3f} {r['false_use_rate']:<7.3f} "
          f"{r['false_reject_rate']:.3f}")
    A("```")
    A("")

    A("## 10. Stable-condition safety")
    A("")
    ch = gate_g_result["checks"]
    A("```")
    A(f"S0_stable_low  regression versus always-use      {ch['stable_low_relative_regression']:+.5f}")
    A(f"S1_stable_high regression versus always-no-future {ch['stable_high_relative_regression']:+.5f}")
    A(f"allowance                                        {cfg.gate_g.stable_condition_regression_max:.5f}")
    A("```")
    A("")

    A("## 11. Worsening conditions")
    A("")
    A("```")
    A("schedule                  D7         D3         D4         D7 beats both")
    A("─" * 72)
    for s, v in ch["worsening"].items():
        A(f"{s:<25s} {v['d7']:<10.5f} {v['d3']:<10.5f} {v['d4']:<10.5f} {v['beats_both']}")
    A("```")
    A("")

    A("## 12. Improvement conditions")
    A("")
    A("```")
    A("schedule                  D7         D0         D7 beats D0")
    A("─" * 60)
    for s, v in ch["improving"].items():
        A(f"{s:<25s} {v['d7']:<10.5f} {v['d0']:<10.5f} {v['beats_d0']}")
    A("```")
    A("")

    A("## 13. Proxy stress diagnostics")
    A("")
    A("[확인] Secondary only.  Gate G is decided on P1; P0 succeeding on its own never grants a pass.")
    A("")
    A("```")
    A("selector                  proxy_mode                WQL       calib   harm    f_use   f_rej")
    A("─" * 96)
    for _, r in proxy_stress.iterrows():
        A(f"{r['selector']:<25s} {r['proxy_mode']:<25s} {r['mean_wql']:<9.5f} "
          f"{r['mean_calibration_ratio']:<7.3f} {r['harm_rate']:<7.3f} "
          f"{r['false_use_rate']:<7.3f} {r['false_reject_rate']:.3f}")
    A("```")
    A("")

    A("## 14. False-use and false-reject")
    A("")
    A("```")
    A("false-use     M1 was actually better and the policy chose M3")
    A("false-reject  M3 was actually better and the policy chose M1")
    A("both are conditional rates: the denominator is the number of tasks where that error")
    A("was possible at all")
    A("```")
    A("")
    prim_row = selector_summary[selector_summary["selector"] == prim].iloc[0]
    A(f"[확인] {prim} under {p1}: false-use {prim_row['false_use_rate']:.4f} "
      f"(n={int(prim_row['n_m1_better'])} tasks where M1 was better), false-reject "
      f"{prim_row['false_reject_rate']:.4f} (n={int(prim_row['n_m3_better'])}).")
    A("")

    A("## 15. Gate G")
    A("")
    L.extend(_gate_lines(gate_g_result, "Gate G"))
    if gate_g_result.get("failed_conditions"):
        A(f"Unmet conditions: `{gate_g_result['failed_conditions']}`")
        A("")
    if gate_g_result.get("fail_reasons"):
        A(f"FAIL reasons: `{gate_g_result['fail_reasons']}`")
        A("")

    A("## 16. Observations")
    A("")
    for o in observations:
        A(f"- {o}")
    A("")

    A("## 17. Verdict")
    A("")
    A("```")
    A(f"Gate G   {gate_g_result['status']}")
    A(f"final    {verdict['verdict']}")
    A("```")
    A("")
    A(f"Reason: {verdict['reason']}")
    A("")

    A("## 18. Limitations")
    A("")
    A("[확인] Synthetic data only: two sinusoids plus an AR(1) residual and a linear target.")
    A("[확인] Covariate forecast error is unbiased and serially uncorrelated in every schedule.")
    A("[확인] The target-covariate relationship is stationary; only reliability varies.")
    A("[확인] One primary forecast origin, one frozen model (amazon/chronos-2, cross_learning=False).")
    A("[미검증] Whether the lognormal proxy resembles what a real forecast provider reports.")
    A("[미검증] Whether this transfers to other foundation models or to real data.")
    A("")
    for c in CONFIRMATION_FORBIDDEN:
        A(f"- {c}")
    A("")

    A("## 19. Next steps")
    A("")
    A("- Biased and serially correlated covariate forecast error.")
    A("- Non-stationary target-covariate relationships, not only non-stationary reliability.")
    A("- A proxy calibrated from an actual provider's reliability statistics.")
    A("- If a threshold is ever revisited, it must be re-registered and confirmed on another")
    A("  fresh sample, not on this one.")
    A("")

    A("## 20. Reproduction")
    A("")
    A("```bash")
    A(f"python -m covariate_trust.cli confirm-d7 --config {cfg.source_path or 'configs/study2b_d7_confirmation.yaml'}")
    A(f"python -m covariate_trust.cli confirmation-report --run-dir {run_dir}")
    A("```")
    A("")
    A("```")
    A(f"master_seed          {prereg['master_seed']}")
    A(f"preregistration hash {prereg_hash}")
    A(f"inference calls      {manifest.get('n_inference_calls')} "
      f"(cache hits {manifest.get('n_cache_hits')})")
    A(f"runtime              {manifest.get('runtime_seconds')} s")
    A(f"peak GPU             {manifest.get('peak_gpu_memory_gb')} GB")
    A("```")
    return "\n".join(L) + "\n"


# ============================================================================
# Study 3 - real forecast-vintage external validation report.  Additive.
# ============================================================================

REAL_VINTAGE_FORBIDDEN = (
    "this is not a claim that D7 is validated for all deployments",
    "the verification series is reanalysis/model-based, not station observations",
    "no result is generalized from amazon/chronos-2 to other models",
    "lambda = 1 is not presented as a theoretical boundary of WQL",
    "an INCONCLUSIVE or NOT_EVALUABLE outcome is never described as a pass",
)


def build_real_vintage_report(run_dir, manifest: dict, cfg, prereg: dict, prereg_hash: str,
                              data_quality: dict, coverage: dict, calibration: dict,
                              gate_h_result: dict, gate_i_result: dict | None,
                              verdict: dict, selector_summary, zone_summary,
                              season_summary, event_summary, observations: list[str]) -> str:
    L: list[str] = []
    A = L.append

    A("# Study 3 - real forecast-vintage external validation (NYISO load + ECMWF IFS)")
    A("")
    A(f"Run directory: `{run_dir}`")
    A("")

    A("## 1. Research question")
    A("")
    A("[확인] Does the pre-registered D7 admission policy, whose thresholds were fixed by the")
    A("synthetic studies, reduce forecast error and harmful forecasts on a real forecast-vintage")
    A("backtest, against always using the weather forecast, never using it, and a history-only rule?")
    A("")

    A("## 2. Why the synthetic GO was not a deployment claim")
    A("")
    A("[확인] Gate G established the policy on synthetic data where the covariate error was")
    A("constructed, unbiased and serially uncorrelated, and where the reliability proxy was a")
    A("lognormal perturbation of a known lambda.  None of that is guaranteed by a real numerical")
    A("weather prediction system.  Study 3 replaces every one of those pieces with the real thing")
    A("for one domain.")
    A("")

    A("## 3. Pre-registration")
    A("")
    A("```")
    A(f"primary policy       {prereg['primary_policy']}")
    A(f"D7 thresholds        {prereg['d7_lower_threshold']} / {prereg['d7_upper_threshold']}")
    A(f"proxy weights        {prereg['proxy_revision_weight']} revision + "
      f"{prereg['proxy_recent_weight']} recent")
    A(f"held-out test        {prereg['heldout_test_start']} .. {prereg['heldout_test_end']}")
    A(f"start commit         {prereg['start_state']['commit']}")
    A(f"start diff hash      {prereg['start_state']['git_diff_sha256']}")
    A(f"preregistration hash {prereg_hash}")
    A("```")
    A("")
    A("[확인] Written before any held-out target, M1/M3 score or gate quantity was computed.")
    A("")

    A("## 4. Data sources")
    A("")
    A("```")
    A(f"target        {prereg['target_source']}")
    A(f"report        {data_quality.get('nyiso_report_title')}")
    A(f"forecast      {prereg['forecast_source']}  models={cfg.weather.model}")
    A(f"verification  {prereg['verification_source']}")
    A(f"zones         {[z.canonical_name for z in cfg.nyiso.zones]}")
    A("```")
    A("")
    A("[확인] NYISO URLs were taken from the index page's own href attributes.  The ECMWF")
    A("endpoint requires an explicit `run` datetime and an explicit `models=ecmwf_ifs`; both were")
    A("discovered from the API's error responses, not assumed.")
    A("")
    A("[미검증] The verification series is a reanalysis/model-based product, not station")
    A("observations, and one representative coordinate stands in for a whole load zone.")
    A("")

    A("## 5. Forecast vintage")
    A("")
    A("```")
    A(f"primary run      {cfg.weather.primary_run_hour_utc:02d}Z, used by M3")
    A(f"revision run     previous day {cfg.weather.revision_run_hour_utc:02d}Z, proxy only")
    A(f"decision origin  {cfg.experiment.decision_origin_hour_utc:02d}Z "
      f"(= run hour + {cfg.weather.decision_delay_hours}h publication delay)")
    A(f"context          {cfg.experiment.context_length} h    horizon "
      f"{cfg.experiment.prediction_length} h")
    A("```")
    A("")
    A("[확인] Each forecast used at an origin comes from a single model run that initialised")
    A("strictly before that origin.  This is a genuine single run, not a stitched fixed-lead")
    A("series, and the code raises if a run is not earlier than its decision origin.")
    A("")

    A("## 6. Decision-time information set")
    A("")
    A("[확인] At 06 UTC the policy may use: load and verified temperature up to 05 UTC, the 00Z")
    A("run issued six hours earlier, the previous day's 12Z run, and statistics of *completed*")
    A("earlier origins.  It may not use the current 24-hour target or the current true_lambda.")
    A("Both restrictions are enforced by tests that perturb those quantities and assert the")
    A("decisions do not move.")
    A("")

    A("## 7. Timezone and DST")
    A("")
    dst = data_quality.get("dst", {})
    A("```")
    for k, v in dst.items():
        A(f"{k}: {v}")
    A("```")
    A("")
    A("[확인] NYISO publishes local clock time together with an explicit EDT/EST marker, so the")
    A("repeated autumn hour is resolved from the source's own metadata rather than a heuristic.")
    A("The nonexistent spring hour is absent and was never interpolated.")
    A("")

    A("## 8. Split and coverage")
    A("")
    A("```")
    A(f"proxy train       {cfg.periods.requested_start} .. {cfg.periods.proxy_train_end}")
    A(f"proxy validation  .. {cfg.periods.proxy_validation_end}")
    A(f"held-out test     {cfg.periods.heldout_test_start} .. {cfg.periods.heldout_test_end}")
    A(f"coverage status   {coverage.get('status')}")
    A(f"test origins/zone {coverage.get('test_origins_per_zone')}")
    A("```")
    A("")

    A("## 9. M0 / M1 / M2 / M3")
    A("")
    A("```")
    A("M0  load history only")
    A("M1  load history + verified temperature history            <- primary baseline")
    A("M2  M1 + verified future temperature                       <- oracle bound, not a method")
    A("M3  M1 + the 00Z ECMWF forecast of future temperature      <- the object of study")
    A("```")
    A("")
    A("[확인] M1 and M3 contexts are asserted byte-identical at every origin; M2 and M3 future")
    A("frames share timestamps; no future frame contains the target.")
    A("")

    A("## 10. Reliability proxy")
    A("")
    A("```")
    A("true_lambda      RMSE(00Z forecast, verification) / RMSE(168h seasonal-naive, verification)")
    A("revision_ratio   RMSE(00Z, previous-day 12Z) / mean past 168h-naive error level")
    A("recent_lambda    mean true_lambda over the previous 28 completed origins")
    A("raw_proxy        0.70 * revision_ratio + 0.30 * recent_lambda")
    A("lambda_reported  frozen isotonic calibrator applied to raw_proxy")
    A("```")
    A("")

    A("## 11. Proxy calibration")
    A("")
    A("```")
    for split in ("train", "validation", "test"):
        c = calibration.get(split)
        if c:
            A(f"{split:<11s} n={c['n']:<6d} spearman={c['spearman']:+.4f} mae={c['mae']:.4f} "
              f"slope={c['slope']:+.4f} quartile_ratio={c['quartile_ratio']:.3f}")
    A("```")
    A("")
    A("[확인] The isotonic calibrator was fitted on the training period only and frozen; the")
    A("validation numbers are reported, never used to change a weight or a threshold.")
    A("")

    A("## 12. Gate H - is the problem real here?")
    A("")
    L.extend(_gate_lines(gate_h_result, "Gate H"))

    A("## 13. D7 external validation")
    A("")
    A("```")
    A("selector                      WQL       impr      recovery  harm    m3_rate  f_use   f_rej")
    A("─" * 96)
    for _, r in selector_summary.sort_values("mean_wql").iterrows():
        A(f"{r['selector']:<29s} {r['mean_wql']:<9.5f} "
          f"{r['relative_improvement_over_best_fixed']:<+9.4f} {r['oracle_gap_recovery']:<9.4f} "
          f"{r['harm_rate']:<7.3f} {r['m3_choice_rate']:<8.3f} {r['false_use_rate']:<7.3f} "
          f"{r['false_reject_rate']:.3f}")
    A("```")
    A("")

    A("## 14. Gate I")
    A("")
    if gate_i_result is None:
        A("NOT_RUN - Gate H did not pass, so Gate I was not evaluated.")
        A("")
    else:
        L.extend(_gate_lines(gate_i_result, "Gate I"))
        if gate_i_result.get("not_evaluable"):
            A(f"NOT_EVALUABLE conditions: `{gate_i_result['not_evaluable']}`")
            A("")
        if gate_i_result.get("failed_conditions"):
            A(f"Unmet conditions: `{gate_i_result['failed_conditions']}`")
            A("")

    A("## 15. Results by zone")
    A("")
    A("```")
    A("zone            selector                      WQL       harm    m3_rate  n_origins")
    A("─" * 82)
    for _, r in zone_summary.sort_values(["zone", "mean_wql"]).iterrows():
        A(f"{r['zone']:<15s} {r['selector']:<29s} {r['mean_wql']:<9.5f} {r['harm_rate']:<7.3f} "
          f"{r['m3_choice_rate']:<8.3f} {int(r['n_origins'])}")
    A("```")
    A("")

    A("## 16. Results by season")
    A("")
    A("```")
    A("season  selector                      WQL       harm    m3_rate  n_origins")
    A("─" * 74)
    for _, r in season_summary.sort_values(["season", "mean_wql"]).iterrows():
        A(f"{r['season']:<7s} {r['selector']:<29s} {r['mean_wql']:<9.5f} {r['harm_rate']:<7.3f} "
          f"{r['m3_choice_rate']:<8.3f} {int(r['n_origins'])}")
    A("```")
    A("")

    A("## 17. Reliability-shift subsets")
    A("")
    if event_summary is not None and len(event_summary):
        A("```")
        A("event                 selector                      WQL       n_origins")
        A("─" * 72)
        for _, r in event_summary.sort_values(["event", "mean_wql"]).iterrows():
            A(f"{r['event']:<21s} {r['selector']:<29s} {r['mean_wql']:<9.5f} {int(r['n_origins'])}")
        A("```")
    else:
        A("No reliability-shift events were identified.")
    A("")
    A("[확인] Shift events are labelled after the fact from true_lambda and are never used in any")
    A("decision.  Subsets with fewer than 20 events are reported NOT_EVALUABLE.")
    A("")

    A("## 18. False-use and false-reject")
    A("")
    A("```")
    A("false-use     M1 was actually better and the policy chose M3")
    A("false-reject  M3 was actually better and the policy chose M1")
    A("```")
    A("")

    A("## 19. Observations")
    A("")
    for o in observations:
        A(f"- {o}")
    A("")

    A("## 20. Verdict")
    A("")
    A("```")
    A(f"Gate H   {gate_h_result['status']}")
    A(f"Gate I   {gate_i_result['status'] if gate_i_result else 'NOT_RUN'}")
    A(f"final    {verdict['verdict']}")
    A("```")
    A("")
    A(f"Reason: {verdict['reason']}")
    A("")
    if verdict.get("scope_note"):
        A(f"[확인] Scope: {verdict['scope_note']}")
        A("")

    A("## 21. Limitations")
    A("")
    A("[확인] The weather verification series is a reanalysis/model-based product, not station")
    A("observations, so 'forecast error' here is error against that product.")
    A("[확인] One representative coordinate per zone does not represent a whole load zone's weather.")
    A("[확인] One system operator (NYISO) and one covariate (2 m temperature).")
    A("[확인] One forecasting model (amazon/chronos-2, frozen, cross_learning=False).")
    A("[확인] The D7 thresholds were fixed by the synthetic studies and were not tuned here.")
    A("[확인] The reliability proxy is built from run-to-run revision plus recent realized error;")
    A("a real provider's own uncertainty product was not used.")
    A("[미검증] Production latency, operating cost and downstream decision value were not evaluated.")
    A("")
    for c in REAL_VINTAGE_FORBIDDEN:
        A(f"- {c}")
    A("")

    A("## 22. Reproduction")
    A("")
    A("```bash")
    A("python -m covariate_trust.cli external-validation \\")
    A("  --config configs/study3_real_vintage.yaml")
    A("```")
    A("")
    A("```")
    A(f"inference calls {manifest.get('n_inference_calls')} "
      f"(cache hits {manifest.get('n_cache_hits')})")
    A(f"runtime         {manifest.get('runtime_seconds')} s")
    A(f"peak GPU        {manifest.get('peak_gpu_memory_gb')} GB")
    A("```")
    return "\n".join(L) + "\n"
