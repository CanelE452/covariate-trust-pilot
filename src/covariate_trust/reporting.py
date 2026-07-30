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
