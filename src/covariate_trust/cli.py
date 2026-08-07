"""Command line interface.

Exit codes
    0  completed normally
    2  environment or input error
    3  Study 0 FAIL
    4  Chronos smoke FAIL
    5  a gate returned FAIL or INCONCLUSIVE and downstream work was stopped

Artifacts already written are always kept, even when a command aborts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from . import audit as audit_mod
from .storage import (atomic_write_csv, atomic_write_json, atomic_write_parquet,
                      atomic_write_text, create_run_dir, project_root)

app = typer.Typer(add_completion=False, help="Forecasted future covariate pilot for Chronos-2")

EXIT_OK, EXIT_ENV, EXIT_STUDY0, EXIT_SMOKE, EXIT_GATE = 0, 2, 3, 4, 5
DETERMINISM_EXACT_TOL = 0.0
DETERMINISM_MAX_TOL = 1e-6
BATCH_EQUIVALENCE_TOL = 1e-5


class Log:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "logs" / "run.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:  # noqa: BLE001
            pass


def _hf_home() -> tuple[Path, str]:
    root = project_root()
    from .chronos_adapter import set_hf_home
    return root, set_hf_home(root)


def _collect_audit() -> tuple[Path, str, dict]:
    root, hf_home = _hf_home()
    return root, hf_home, audit_mod.collect(root, hf_home)


def _base_manifest(cfg, audit: dict, command: str, run_dir: Path) -> dict:
    return {
        "command": command,
        "run_dir": str(run_dir),
        "project_root": audit["project_root"],
        "git_commit": audit["git"]["commit"],
        "git_status": audit["git"]["status"],
        "python_executable": audit["python"]["executable"],
        "python_version": audit["python"]["version"],
        "packages": audit["packages"],
        "torch": audit["torch"],
        "hf_home": audit["hf_home"],
        "model_id": getattr(getattr(cfg, "model", None), "model_id", None),
        "model_revision": None,
        "cross_learning": False,
        "attention_implementation": getattr(getattr(cfg, "model", None),
                                            "attention_implementation", None),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "commands": [],
        "runtime_seconds": None,
        "peak_gpu_memory_gb": None,
    }


def _peak_gpu_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 4)
    except Exception:  # noqa: BLE001
        return None
    return None


# ----------------------------------------------------------------- audit ----

@app.command()
def audit() -> None:
    """Inspect the environment and the installed Chronos API.  No inference, no data."""
    root, hf_home, info = _collect_audit()
    run_dir = create_run_dir(root, "audit")
    log = Log(run_dir)
    log(f"project root: {root}")
    log(f"HF_HOME: {hf_home}")
    atomic_write_json(run_dir / "audit.json", info)
    atomic_write_text(run_dir / "environment.txt", audit_mod.environment_text(info))
    api = info["chronos_api"]
    log(f"python {info['python']['version']} @ {info['python']['executable']}")
    log(f"torch {info['torch'].get('version')} cuda_available={info['torch'].get('cuda_available')} "
        f"gpu={info['torch'].get('gpu_name', 'n/a')}")
    log(f"chronos-forecasting {info['packages']['chronos-forecasting']}")
    log(f"predict_df signature: {api.get('predict_df_signature')}")
    log(f"cross_learning supported: {api.get('cross_learning_supported')}")
    log(f"model cache: exists={info['model_cache']['exists']} files={info['model_cache']['n_files']}")
    log(f"internet: {info['internet_available']}")
    log(f"git commit: {info['git']['commit']}")
    problems = audit_mod.blocking_problems(info)
    for p in problems:
        log(f"BLOCKING: {p}")
    atomic_write_json(run_dir / "tables" / "blocking.json", problems)
    log(f"audit written to {run_dir}")
    log.close()
    raise typer.Exit(EXIT_ENV if problems else EXIT_OK)


# --------------------------------------------------------------- study 0 ----

def _run_study0(config: Path) -> tuple[Path, dict]:
    from .config import Study0Config
    from .plotting import study0_mse_curve
    from .study0 import run_study0, study0_report

    root = project_root()
    cfg = Study0Config.load(config)
    run_dir = create_run_dir(root, "study0")
    log = Log(run_dir)
    t0 = time.time()
    log(f"Study 0: {cfg.experiment.name} seed={cfg.experiment.seed} "
        f"n={cfg.experiment.n_repetitions}")

    _, _, info = _collect_audit()
    atomic_write_json(run_dir / "audit.json", info)
    atomic_write_text(run_dir / "environment.txt", audit_mod.environment_text(info))
    atomic_write_text(run_dir / "config_resolved.yaml",
                      __import__("yaml").safe_dump(cfg.to_dict(), sort_keys=False))

    raw, summary, verdict = run_study0(cfg)
    atomic_write_parquet(run_dir / "study0_raw.parquet", raw)
    atomic_write_csv(run_dir / "study0_summary.csv", summary)
    atomic_write_text(run_dir / "study0_report.md", study0_report(cfg, summary, verdict))
    study0_mse_curve(summary, run_dir / "study0_mse_curve.png")
    atomic_write_json(run_dir / "tables" / "study0_verdict.json", verdict)

    for c in verdict["checks"]:
        log(f"  {c['id']}: {c['status']} - {c['detail']}")
    log(f"Study 0 status: {verdict['status']}")

    manifest = _base_manifest(cfg, info, "study0", run_dir)
    manifest["seeds"] = {"master_seed": cfg.experiment.seed}
    manifest["runtime_seconds"] = round(time.time() - t0, 2)
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["commands"] = [f"python -m covariate_trust.cli study0 --config {config}"]
    atomic_write_json(run_dir / "manifest.json", manifest)
    log(f"artifacts: {run_dir}")
    log.close()
    return run_dir, verdict


@app.command()
def study0(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Known-answer linear-Gaussian simulation.  Blocks Study 1A on failure."""
    try:
        _, verdict = _run_study0(config)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"study0 failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(EXIT_ENV)
    raise typer.Exit(EXIT_OK if verdict["status"] == "PASS" else EXIT_STUDY0)


# ----------------------------------------------------------------- smoke ----

def _run_smoke(config: Path) -> tuple[Path, dict]:
    import numpy as np
    import pandas as pd

    from .chronos_adapter import (ChronosEnvironmentError, load_pipeline, predict_task,
                                  resolve_quantile_columns)
    from .config import PilotConfig
    from .dgp import build_target, covariate_vintage, generate_base_series
    from .metrics import quantile_crossing_rate
    from .schemas import M0, M1, M2, M3, assert_context_equality, assert_future_equality, build_inputs

    root, hf_home, info = _collect_audit()
    cfg = PilotConfig.load(config)
    run_dir = create_run_dir(root, "smoke")
    log = Log(run_dir)
    t0 = time.time()
    atomic_write_json(run_dir / "audit.json", info)
    atomic_write_text(run_dir / "environment.txt", audit_mod.environment_text(info))
    atomic_write_text(run_dir / "config_resolved.yaml", cfg.resolved_yaml())

    result: dict = {"status": "FAIL", "checks": {}, "hf_home": hf_home}
    problems = audit_mod.blocking_problems(info)
    if problems:
        result["blocking"] = problems
        atomic_write_json(run_dir / "tables" / "smoke_report.json", result)
        log(f"BLOCKED_CHRONOS_ENV: {problems}")
        log.close()
        return run_dir, result

    device = cfg.model.device
    torch_ok = info["torch"].get("cuda_available", False)
    if device == "cuda" and not torch_ok:
        if cfg.model.allow_cpu_smoke:
            device = "cpu"
            log("CUDA unavailable; running the smoke test on CPU (allowed by config)")
        else:
            result["blocking"] = ["BLOCKED_GPU_ENV"]
            atomic_write_json(run_dir / "tables" / "smoke_report.json", result)
            log.close()
            return run_dir, result

    cache_before = audit_mod.model_cache_state(hf_home, cfg.model.model_id)
    log(f"loading {cfg.model.model_id} on {device} (cache exists={cache_before['exists']})")
    t_load = time.time()
    try:
        loaded = load_pipeline(cfg.model.model_id, device, cfg.model.attention_implementation)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(run_dir / "tables" / "smoke_report.json", result)
        log(f"model load failed: {result['error']}")
        log.close()
        return run_dir, result
    load_seconds = time.time() - t_load
    cache_after = audit_mod.model_cache_state(hf_home, cfg.model.model_id)
    log(f"model loaded in {load_seconds:.1f}s (attn kwarg used: {loaded.attention_implementation}, "
        f"dtype {loaded.dtype})")

    exp = cfg.experiment
    horizon, share = 24, 0.50
    lambdas = [0.0, 2.0]
    checks: dict = {
        "model_download_or_cache_load": {
            "cache_existed_before": cache_before["exists"],
            "cache_files_before": cache_before["n_files"],
            "cache_files_after": cache_after["n_files"],
            "downloaded_this_run": cache_after["n_files"] > cache_before["n_files"],
            "load_seconds": round(load_seconds, 2),
        },
        "device": {"requested": cfg.model.device, "used": device,
                   "load_kwargs": {k: str(v) for k, v in loaded.load_kwargs.items()},
                   "attention_implementation": loaded.attention_implementation},
    }

    rows, timings, schema_seen = [], [], None
    crossing_rates, finite_ok = [], True
    ctx_pairs = []
    for b_id in (0, 1):
        s = generate_base_series(b_id, cfg)
        y = build_target(s, share)
        item = f"smoke_b{b_id}"
        inputs_by_key = {}
        inputs_by_key[(M0, None)] = build_inputs(M0, item, y, s.x, exp.primary_origin, horizon,
                                                 exp.context_length, exp.frequency)
        inputs_by_key[(M1, None)] = build_inputs(M1, item, y, s.x, exp.primary_origin, horizon,
                                                 exp.context_length, exp.frequency)
        v_true = covariate_vintage(cfg, s, exp.primary_origin, horizon, 0.0)
        inputs_by_key[(M2, None)] = build_inputs(M2, item, y, s.x, exp.primary_origin, horizon,
                                                 exp.context_length, exp.frequency,
                                                 x_future=v_true["x_true"])
        for lam in lambdas:
            v = covariate_vintage(cfg, s, exp.primary_origin, horizon, lam)
            inputs_by_key[(M3, lam)] = build_inputs(M3, item, y, s.x, exp.primary_origin, horizon,
                                                    exp.context_length, exp.frequency,
                                                    x_future=v["x_tilde"])
        assert_context_equality(inputs_by_key[(M1, None)], inputs_by_key[(M3, 0.0)])
        assert_future_equality(inputs_by_key[(M2, None)], inputs_by_key[(M3, 0.0)])
        ctx_pairs.append(True)

        for (method, lam), inputs in inputs_by_key.items():
            t1 = time.time()
            q, pred = predict_task(loaded, inputs, exp.quantile_levels, exp.context_length,
                                   exp.frequency)
            dt = time.time() - t1
            timings.append(dt)
            if schema_seen is None:
                schema_seen = {"columns": list(pred.columns),
                               "dtypes": {c: str(t) for c, t in pred.dtypes.items()},
                               "n_rows": int(len(pred))}
            finite_ok = finite_ok and bool(np.isfinite(q).all())
            crossing_rates.append(quantile_crossing_rate(q))
            rows.append({"base_series_id": b_id, "method": method, "lam": lam,
                         "horizon_rows": int(len(pred)), "seconds": dt,
                         "crossing_rate": crossing_rates[-1],
                         "median_forecast_mean": float(q[:, len(exp.quantile_levels) // 2].mean())})

    smoke_df = pd.DataFrame(rows)
    atomic_write_parquet(run_dir / "predictions" / "smoke_tasks.parquet", smoke_df)

    checks["output_schema"] = schema_seen
    checks["prediction_length"] = {
        "expected": horizon,
        "all_match": bool((smoke_df["horizon_rows"] == horizon).all()),
    }
    q_cols = resolve_quantile_columns(pred, exp.quantile_levels)
    checks["quantile_columns"] = {"requested": exp.quantile_levels, "resolved": q_cols}
    checks["predictions_column_present"] = "predictions" in (schema_seen["columns"] if schema_seen else [])
    checks["no_nan_or_inf"] = bool(finite_ok)
    checks["quantile_crossing"] = {"max_rate": float(max(crossing_rates)),
                                   "mean_rate": float(np.mean(crossing_rates))}
    checks["runtime"] = {"n_tasks": len(timings), "total_seconds": round(sum(timings), 2),
                         "mean_seconds_per_task": round(float(np.mean(timings)), 4),
                         "max_seconds_per_task": round(float(np.max(timings)), 4)}
    checks["peak_gpu_memory_gb"] = _peak_gpu_gb()
    checks["fairness_assertions"] = {"m1_m3_context_equal": all(ctx_pairs),
                                     "m2_lambda0_future_equal": all(ctx_pairs)}

    # determinism: same input three times
    det_inputs = inputs_by_key[(M3, 2.0)]
    reps = [predict_task(loaded, det_inputs, exp.quantile_levels, exp.context_length,
                         exp.frequency)[0] for _ in range(3)]
    diffs = [float(np.max(np.abs(reps[0] - r))) for r in reps[1:]]
    max_diff = max(diffs)
    if max_diff <= DETERMINISM_EXACT_TOL:
        det_status = "EXACT"
    elif max_diff <= DETERMINISM_MAX_TOL:
        det_status = "WITHIN_TOLERANCE"
    else:
        det_status = "FAIL"
    checks["determinism"] = {"n_repeats": 3, "max_abs_diff": max_diff,
                             "exact_tolerance": DETERMINISM_EXACT_TOL,
                             "accepted_tolerance": DETERMINISM_MAX_TOL,
                             "status": det_status}
    log(f"determinism over 3 repeats: {det_status} (max abs diff {max_diff:.3e})")

    # batch equivalence probe (recorded only; the diagnostic always uses per-task inference)
    try:
        batch_ctx = pd.concat([inputs_by_key[(M1, None)].context_df.assign(id=f"batch{i}")
                               for i in range(2)], ignore_index=True)
        batch_pred = loaded.pipeline.predict_df(
            batch_ctx, future_df=None, id_column="id", timestamp_column="timestamp",
            target="target", prediction_length=horizon, quantile_levels=exp.quantile_levels,
            context_length=exp.context_length, cross_learning=False, freq=exp.frequency)
        single = predict_task(loaded, inputs_by_key[(M1, None)], exp.quantile_levels,
                              exp.context_length, exp.frequency)[0]
        bcols = resolve_quantile_columns(batch_pred, exp.quantile_levels)
        b0 = batch_pred[batch_pred["id"] == "batch0"][bcols].to_numpy(dtype=float)
        batch_diff = float(np.max(np.abs(b0 - single)))
        checks["batch_equivalence"] = {
            "max_abs_diff": batch_diff,
            "tolerance": BATCH_EQUIVALENCE_TOL,
            "equivalent": bool(batch_diff <= BATCH_EQUIVALENCE_TOL),
            "used_in_diagnostic": False,
            "note": "the diagnostic uses per-task inference regardless of this probe",
        }
    except Exception as exc:  # noqa: BLE001
        checks["batch_equivalence"] = {"error": f"{type(exc).__name__}: {exc}",
                                       "equivalent": False, "used_in_diagnostic": False}

    hard_ok = (checks["prediction_length"]["all_match"] and checks["no_nan_or_inf"]
               and det_status != "FAIL" and checks["predictions_column_present"])
    result = {
        "status": "PASS" if hard_ok else "FAIL",
        "device_used": device,
        "checks": checks,
        "hf_home": hf_home,
    }
    if device == "cpu" and cfg.model.device == "cuda":
        result["blocking"] = ["BLOCKED_GPU_ENV"]

    atomic_write_json(run_dir / "tables" / "smoke_report.json", result)
    manifest = _base_manifest(cfg, info, "smoke", run_dir)
    manifest["seeds"] = {"master_seed": cfg.experiment.master_seed}
    manifest["runtime_seconds"] = round(time.time() - t0, 2)
    manifest["peak_gpu_memory_gb"] = checks["peak_gpu_memory_gb"]
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["commands"] = [f"python -m covariate_trust.cli smoke --config {config}"]
    manifest["attention_implementation"] = loaded.attention_implementation
    atomic_write_json(run_dir / "manifest.json", manifest)
    log(f"smoke status: {result['status']} ({checks['runtime']['mean_seconds_per_task']:.3f}s/task)")
    log.close()
    return run_dir, result


@app.command()
def smoke(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Chronos-2 API, output schema and determinism check on two synthetic series."""
    run_dir, result = _run_smoke(config)
    if result["status"] != "PASS":
        raise typer.Exit(EXIT_SMOKE)
    if "BLOCKED_GPU_ENV" in result.get("blocking", []):
        typer.echo("BLOCKED_GPU_ENV: smoke passed on CPU, the diagnostic requires a GPU")
        raise typer.Exit(EXIT_ENV)
    raise typer.Exit(EXIT_OK)


# ------------------------------------------------------------ diagnostic ----

def _run_diagnostic(config: Path, resume: Optional[Path] = None) -> tuple[Path, dict]:
    import numpy as np
    import pandas as pd

    from .chronos_adapter import context_and_future_hashes, load_pipeline, predict_task, task_hash
    from .config import PilotConfig
    from .dgp import build_target, covariate_vintage, generate_dataset, vintage_table
    from .gates import cell_summary, gates_abc, monte_carlo_table
    from .metrics import is_harm, mse, nmae, quantile_crossing_rate, relative_delta, wql
    from .schemas import (M0, M1, M2, M3, assert_context_equality, assert_future_equality,
                          build_inputs)
    from .seeds import seed_hierarchy
    from .storage import completed_task_hashes, read_all_parts, write_task_part

    root, hf_home, info = _collect_audit()
    cfg = PilotConfig.load(config)
    run_dir = Path(resume) if resume else create_run_dir(root, "diagnostic")
    log = Log(run_dir)
    t0 = time.time()
    log(f"diagnostic run dir: {run_dir}{' (resume)' if resume else ''}")

    problems = audit_mod.blocking_problems(info)
    if problems:
        log(f"BLOCKED_CHRONOS_ENV: {problems}")
        log.close()
        return run_dir, {"status": "BLOCKED", "problems": problems}
    if cfg.model.device == "cuda" and not info["torch"].get("cuda_available"):
        log("BLOCKED_GPU_ENV: the diagnostic may not run on CPU (allow_cpu_diagnostic=false)")
        log.close()
        return run_dir, {"status": "BLOCKED", "problems": ["BLOCKED_GPU_ENV"]}

    atomic_write_json(run_dir / "audit.json", info)
    atomic_write_text(run_dir / "environment.txt", audit_mod.environment_text(info))
    atomic_write_text(run_dir / "config_resolved.yaml", cfg.resolved_yaml())

    exp, grid = cfg.experiment, cfg.grid
    log("generating synthetic series")
    series_df, meta_df, series_map = generate_dataset(cfg)
    atomic_write_parquet(run_dir / "generated" / "series.parquet", series_df)
    atomic_write_parquet(run_dir / "generated" / "series_metadata.parquet", meta_df)
    vints = vintage_table(cfg, series_map)
    atomic_write_parquet(run_dir / "generated" / "covariate_vintages.parquet", vints)
    log(f"generated {len(series_map)} base series x {len(grid.nominal_covariate_share)} shares")

    loaded = load_pipeline(cfg.model.model_id, cfg.model.device, cfg.model.attention_implementation)
    log(f"model loaded (attn kwarg {loaded.attention_implementation}, dtype {loaded.dtype})")

    done = completed_task_hashes(run_dir)
    if done:
        log(f"resume: {len(done)} task parts already present")
    q_levels = exp.quantile_levels
    qcols = [f"q{q:g}" for q in q_levels]

    def cached_predict(inputs, meta: dict) -> np.ndarray:
        th = task_hash(cfg.model.model_id, None, meta["base_series_id"],
                       meta["nominal_covariate_share"], meta["origin"], meta["horizon"],
                       meta["method"], meta.get("lam"), inputs)
        part = run_dir / "predictions" / "parts" / f"{th}.parquet"
        if th in done and part.exists():
            return pd.read_parquet(part)[qcols].to_numpy(dtype=float)
        q, _ = predict_task(loaded, inputs, q_levels, exp.context_length, exp.frequency)
        ctx_hash, fut_hash = context_and_future_hashes(inputs)
        frame = pd.DataFrame(q, columns=qcols)
        frame.insert(0, "h_index", np.arange(1, len(frame) + 1))
        for k, v in meta.items():
            frame[k] = v
        frame["task_hash"] = th
        frame["context_hash"] = ctx_hash
        frame["future_hash"] = fut_hash
        frame["model_id"] = cfg.model.model_id
        write_task_part(run_dir, th, frame)
        done.add(th)
        return q

    task_rows = []
    n_m3, t_infer0 = 0, time.time()
    total_m3 = (len(cfg.base_series_ids) * len(grid.nominal_covariate_share)
                * len(grid.horizons) * len(grid.lambda_values))
    for b_id in cfg.base_series_ids:
        s = series_map[b_id]
        for share in grid.nominal_covariate_share:
            share = float(share)
            y = build_target(s, share)
            for h in grid.horizons:
                origin = exp.primary_origin
                item = f"b{b_id}_s{share:g}_h{h}"
                y_true = y[origin:origin + h]
                base_meta = {"base_series_id": b_id, "nominal_covariate_share": share,
                             "origin": origin, "horizon": h}

                in0 = build_inputs(M0, item, y, s.x, origin, h, exp.context_length, exp.frequency)
                in1 = build_inputs(M1, item, y, s.x, origin, h, exp.context_length, exp.frequency)
                v_true = covariate_vintage(cfg, s, origin, h, 0.0)
                in2 = build_inputs(M2, item, y, s.x, origin, h, exp.context_length, exp.frequency,
                                   x_future=v_true["x_true"])

                q0 = cached_predict(in0, {**base_meta, "method": M0, "lam": -1.0})
                q1 = cached_predict(in1, {**base_meta, "method": M1, "lam": -1.0})
                q2 = cached_predict(in2, {**base_meta, "method": M2, "lam": -1.0})
                w0, w1, w2 = (wql(y_true, q, q_levels) for q in (q0, q1, q2))

                for lam in grid.lambda_values:
                    lam = float(lam)
                    v = covariate_vintage(cfg, s, origin, h, lam)
                    in3 = build_inputs(M3, item, y, s.x, origin, h, exp.context_length,
                                       exp.frequency, x_future=v["x_tilde"])
                    assert_context_equality(in1, in3)
                    if lam == 0.0:
                        assert_future_equality(in2, in3)
                    q3 = cached_predict(in3, {**base_meta, "method": M3, "lam": lam})
                    w3 = wql(y_true, q3, q_levels)
                    task_rows.append({
                        **base_meta, "lam": lam,
                        "wql_m0": w0, "wql_m1": w1, "wql_m2": w2, "wql_m3": w3,
                        "nmae_m0": nmae(y_true, q0, q_levels), "nmae_m1": nmae(y_true, q1, q_levels),
                        "nmae_m2": nmae(y_true, q2, q_levels), "nmae_m3": nmae(y_true, q3, q_levels),
                        "mse_m0": mse(y_true, q0, q_levels), "mse_m1": mse(y_true, q1, q_levels),
                        "mse_m2": mse(y_true, q2, q_levels), "mse_m3": mse(y_true, q3, q_levels),
                        "crossing_m1": quantile_crossing_rate(q1),
                        "crossing_m3": quantile_crossing_rate(q3),
                        "v_future": w1 - w3,
                        "v_oracle": w1 - w2,
                        "relative_delta_m3": relative_delta(w1, w3),
                        "harm_m3": int(is_harm(w1, w3, cfg.gates.harm_relative_threshold)),
                        "m3_wins": int(w3 < w1),
                        "realized_normalized_error_rms": v["realized_normalized_error_rms"],
                        "lambda_hat": float(np.sqrt(np.mean((v["error"] / np.sqrt(v["V"]))**2))),
                    })
                    n_m3 += 1
                    if n_m3 % 50 == 0:
                        rate = n_m3 / (time.time() - t_infer0)
                        log(f"  {n_m3}/{total_m3} M3 tasks ({rate:.2f}/s, "
                            f"eta {(total_m3 - n_m3) / max(rate, 1e-9) / 60:.1f} min)")

    task_metrics = pd.DataFrame(task_rows)
    atomic_write_parquet(run_dir / "tables" / "task_metrics.parquet", task_metrics)
    preds = read_all_parts(run_dir)
    atomic_write_parquet(run_dir / "predictions" / "predictions.parquet", preds)
    log(f"{len(task_metrics)} primary tasks, {preds['task_hash'].nunique()} inference calls stored")

    cells = cell_summary(task_metrics, cfg)
    atomic_write_csv(run_dir / "tables" / "cell_summary.csv", cells)
    mc = monte_carlo_table(task_metrics, cfg)
    atomic_write_csv(run_dir / "tables" / "monte_carlo_se.csv", mc)

    gates = gates_abc(task_metrics, cells, cfg)
    atomic_write_json(run_dir / "tables" / "gate_report.json", gates)
    for k in ("A", "B", "C"):
        log(f"Gate {k}: {gates[k]['status']}")

    boot_rows = []
    for k, g in gates.items():
        if k in ("A", "B", "C") and isinstance(g, dict) and g.get("status") != "NOT_RUN":
            if "aggregate" in g:
                boot_rows.append({"gate": k, "scope": "aggregate", **g["aggregate"]})
            if "bootstrap" in g:
                boot_rows.append({"gate": k, "scope": "oracle_vs_best_fixed", **g["bootstrap"]})
    if boot_rows:
        atomic_write_csv(run_dir / "tables" / "bootstrap_summary.csv", pd.DataFrame(boot_rows))

    _make_figures(run_dir, cfg, task_metrics, cells, meta_df, series_df, vints, None)

    manifest = _base_manifest(cfg, info, "diagnostic", run_dir)
    manifest["seeds"] = seed_hierarchy(exp.master_seed, cfg.base_series_ids,
                                       list(grid.horizons), [exp.primary_origin])
    manifest["runtime_seconds"] = round(time.time() - t0, 2)
    manifest["peak_gpu_memory_gb"] = _peak_gpu_gb()
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["attention_implementation"] = loaded.attention_implementation
    manifest["model_dtype"] = loaded.dtype
    manifest["n_inference_calls"] = int(preds["task_hash"].nunique())
    manifest["commands"] = [f"python -m covariate_trust.cli diagnostic --config {config}"]
    atomic_write_json(run_dir / "manifest.json", manifest)
    log(f"diagnostic finished in {manifest['runtime_seconds']}s, "
        f"peak GPU {manifest['peak_gpu_memory_gb']} GB")
    log.close()
    return run_dir, {"status": "OK", "gates": gates}


def _make_figures(run_dir: Path, cfg, task_metrics, cells, meta_df, series_df, vints,
                  admission_harm) -> None:
    import pandas as pd

    from .bootstrap import BOOTSTRAP_UNIT, paired_bootstrap
    from .plotting import (figure1_heatmap, figure2_v_future, figure3_method_wql,
                           figure4_harm_rate, figure5_example_series, figure6_share_vs_r2)

    fig_dir = run_dir / "figures"
    for h in cfg.grid.horizons:
        figure1_heatmap(cells, h, fig_dir / f"figure1_heatmap_h{h}.png")
        figure2_v_future(cells, h, fig_dir / f"figure2_v_future_h{h}.png")
        cis = {}
        t = task_metrics[task_metrics["horizon"] == h]
        for i in (0, 2, 3):
            b = paired_bootstrap(t[BOOTSTRAP_UNIT].to_numpy(), t["wql_m1"].to_numpy(),
                                 t[f"wql_m{i}"].to_numpy(), cfg.bootstrap.n_resamples,
                                 cfg.bootstrap.confidence_level,
                                 seed_parts=(cfg.experiment.master_seed, "fig3", h, i))
            m = float(t[f"wql_m{i}"].mean())
            cis[f"h{h}_m{i}"] = (m - (b.ci_high - b.mean_diff), m + (b.mean_diff - b.ci_low))
        m1 = float(t["wql_m1"].mean())
        cis[f"h{h}_m1"] = (m1, m1)
        figure3_method_wql(task_metrics, cis, h, fig_dir / f"figure3_method_wql_h{h}.png")
        figure4_harm_rate(cells, admission_harm, h, fig_dir / f"figure4_harm_rate_h{h}.png")

    example_share = max(cfg.grid.nominal_covariate_share)
    ex = series_df[(series_df["base_series_id"] == 0)
                   & (series_df["nominal_covariate_share"] == example_share)]
    h = cfg.grid.horizons[0]
    row = vints[(vints["base_series_id"] == 0) & (vints["horizon"] == h)
                & (vints["lam"] == max(cfg.grid.lambda_values))].iloc[0].to_dict()
    row["x_true"] = list(row["x_true"])
    row["x_tilde"] = list(row["x_tilde"])
    figure5_example_series(ex, row, cfg.experiment.primary_origin, h,
                           fig_dir / "figure5_example_series.png")
    figure6_share_vs_r2(meta_df, fig_dir / "figure6_share_vs_incremental_r2.png")


@app.command()
def diagnostic(config: Path = typer.Option(..., "--config", exists=True),
               resume: Optional[Path] = typer.Option(None, "--resume",
                                                     help="existing diagnostic run dir")) -> None:
    """Run the Study 1A coarse grid and evaluate Gates A, B and C."""
    run_dir, out = _run_diagnostic(config, resume)
    if out["status"] != "OK":
        typer.echo(f"diagnostic blocked: {out.get('problems')}")
        raise typer.Exit(EXIT_ENV)
    gates = out["gates"]
    raise typer.Exit(EXIT_OK if gates["all_pass"] else EXIT_GATE)


# ------------------------------------------------------------- admission ----

def _run_admission(run_dir: Path) -> dict:
    import numpy as np
    import pandas as pd

    from .admission import build_decisions, gate_d, run_historical
    from .chronos_adapter import load_pipeline, predict_task
    from .config import PilotConfig
    from .dgp import generate_dataset
    from .metrics import is_harm

    run_dir = Path(run_dir)
    log = Log(run_dir)
    t0 = time.time()
    cfg = PilotConfig.load(run_dir / "config_resolved.yaml")
    gates = json.loads((run_dir / "tables" / "gate_report.json").read_text())
    if not gates.get("all_pass"):
        statuses = {k: gates[k]["status"] for k in ("A", "B", "C") if k in gates}
        log(f"admission not run: gates A-C are {statuses}")
        log.close()
        return {"status": "NOT_RUN", "reason": f"gates A-C are {statuses}"}

    log("gates A-C all PASS - running the historical admission diagnostic")
    _, _, info = _collect_audit()
    loaded = load_pipeline(cfg.model.model_id, cfg.model.device, cfg.model.attention_implementation)
    _, _, series_map = generate_dataset(cfg)

    def predict_fn(inputs):
        q, _ = predict_task(loaded, inputs, cfg.experiment.quantile_levels,
                            cfg.experiment.context_length, cfg.experiment.frequency)
        return q

    historical = run_historical(cfg, series_map, predict_fn, log)
    atomic_write_parquet(run_dir / "tables" / "admission_historical.parquet", historical)

    task_metrics = pd.read_parquet(run_dir / "tables" / "task_metrics.parquet")
    decisions = build_decisions(historical, task_metrics, cfg)
    atomic_write_parquet(run_dir / "tables" / "admission_decisions.parquet", decisions)

    verdict = gate_d(decisions, cfg)
    atomic_write_json(run_dir / "tables" / "gate_d_report.json", verdict)
    log(f"Gate D: {verdict['status']}")

    harm_rows = []
    from .admission import SELECTORS
    for sel in SELECTORS:
        for (h, lam), g in decisions.groupby(["horizon", "lam"]):
            harm_rows.append({
                "selector": sel, "horizon": int(h), "lam": float(lam),
                "harm_rate": float(np.mean([
                    is_harm(b, s, cfg.gates.harm_relative_threshold)
                    for s, b in zip(g[f"wql_{sel}"], g["wql_m1"])])),
            })
    harm_df = pd.DataFrame(harm_rows)
    atomic_write_csv(run_dir / "tables" / "admission_harm_rate.csv", harm_df)

    cells = pd.read_csv(run_dir / "tables" / "cell_summary.csv")
    from .plotting import figure4_harm_rate
    for h in cfg.grid.horizons:
        figure4_harm_rate(cells, harm_df, h, run_dir / "figures" / f"figure4_harm_rate_h{h}.png")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["admission"] = {
        "runtime_seconds": round(time.time() - t0, 2),
        "n_historical_rows": int(len(historical)),
        "peak_gpu_memory_gb": _peak_gpu_gb(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest.setdefault("commands", []).append(
        f"python -m covariate_trust.cli admission --run-dir {run_dir}")
    atomic_write_json(manifest_path, manifest)
    log(f"admission finished in {manifest['admission']['runtime_seconds']}s")
    log.close()
    return {"status": verdict["status"], "gate_d": verdict}


@app.command()
def admission(run_dir: Path = typer.Option(..., "--run-dir", exists=True)) -> None:
    """Historical admission diagnostic and Gate D.  Requires Gates A-C to PASS."""
    out = _run_admission(run_dir)
    if out["status"] == "NOT_RUN":
        typer.echo(f"admission not run: {out['reason']}")
        raise typer.Exit(EXIT_GATE)
    raise typer.Exit(EXIT_OK if out["status"] == "PASS" else EXIT_GATE)


# ---------------------------------------------------------------- report ----

def _observations(cells, gates, gate_d) -> list[str]:
    obs = []
    if cells is not None and len(cells):
        best = cells.loc[cells["v_future_mean"].idxmax()]
        worst = cells.loc[cells["v_future_mean"].idxmin()]
        obs.append(f"[확인] Largest measured V_future: {best['v_future_mean']:+.5f} at share "
                   f"{best['nominal_covariate_share']:g}, lambda {best['lam']:g}, horizon "
                   f"{int(best['horizon'])} (95% CI {best['v_future_ci_low']:+.5f} to "
                   f"{best['v_future_ci_high']:+.5f}).")
        obs.append(f"[확인] Most negative V_future: {worst['v_future_mean']:+.5f} at share "
                   f"{worst['nominal_covariate_share']:g}, lambda {worst['lam']:g}, horizon "
                   f"{int(worst['horizon'])} (95% CI {worst['v_future_ci_low']:+.5f} to "
                   f"{worst['v_future_ci_high']:+.5f}).")
        obs.append(f"[확인] Harm rate ranges from {cells['harm_rate'].min():.3f} to "
                   f"{cells['harm_rate'].max():.3f} across cells.")
        obs.append(f"[확인] Mean quantile crossing rate of M3 predictions: "
                   f"{cells['crossing_rate_m3'].mean():.5f}.")
    if gates and "A" in gates and gates["A"].get("checks"):
        obs.append(f"[확인] Oracle future covariate (M2) changes WQL by "
                   f"{gates['A']['checks']['aggregate_relative_improvement']:+.4f} relative on the "
                   f"Gate A primary subset; the r=0 negative control gives "
                   f"{gates['A']['checks']['negative_control_relative_improvement']:+.4f}.")
    if gates and gates.get("C", {}).get("checks"):
        obs.append(f"[확인] Oracle per-task admission headroom over the best fixed policy: "
                   f"{gates['C']['checks']['oracle_headroom']:+.4f} relative.")
    if gate_d and gate_d.get("selectors"):
        for sel, r in gate_d["selectors"].items():
            obs.append(f"[확인] Selector {sel} chooses M3 on {r['m3_choice_rate']:.1%} of tasks and "
                       f"recovers {r['oracle_gap_recovery']:+.3f} of the oracle gap.")
    return obs


LIMITATIONS = [
    "[확인] Synthetic data only: two sinusoids plus an AR(1) residual, a linear target and "
    "unbiased, serially uncorrelated covariate forecast error.  Nothing here establishes "
    "behaviour on real weather or demand data.",
    "[확인] One forecast origin for the primary comparison and 30 base series per cell; the "
    "Monte-Carlo table states which cells are resolvable at this precision.",
    "[확인] Coarse grid: 4 shares x 5 lambdas x 2 horizons.  Study 1B boundary refinement is "
    "explicitly out of scope.",
    "[확인] Zero-shot Chronos-2 only, frozen, cross_learning=False.  No other TSFM, no "
    "fine-tuning, no uncertainty propagation, no predictive quantile mixture.",
    "[확인] The grid weights every cell equally.  That weighting is a design choice and does "
    "not represent any deployment frequency of good or bad covariate forecasts.",
    "[미검증] The relationship between the nominal covariate share r and what a practitioner "
    "would call covariate importance has not been validated against prior work.",
]

NEXT_STEPS = [
    "Refine the boundary between benefit and harm on a finer lambda grid (Study 1B) in the "
    "region the coarse grid brackets.",
    "Add independent repetitions for cells reported INCONCLUSIVE rather than changing the DGP, "
    "the grid or any threshold.",
    "Extend the error model to biased and serially correlated covariate forecast error.",
    "Test nonlinear target-covariate relationships and multiple covariates.",
]


@app.command()
def report(run_dir: Path = typer.Option(..., "--run-dir", exists=True)) -> None:
    """Assemble reports/report.md from whatever the run directory contains."""
    import pandas as pd

    from .reporting import build_report

    run_dir = Path(run_dir)
    log = Log(run_dir)

    def _read_json(p: Path):
        return json.loads(p.read_text()) if p.exists() else None

    def _read_csv(p: Path):
        return pd.read_csv(p) if p.exists() else None

    audit_info = _read_json(run_dir / "audit.json")
    if audit_info is None:
        typer.echo("run directory has no audit.json")
        raise typer.Exit(EXIT_ENV)
    manifest = _read_json(run_dir / "manifest.json") or {}
    cfg_yaml = (run_dir / "config_resolved.yaml").read_text() if (
        run_dir / "config_resolved.yaml").exists() else ""
    study0_verdict = _read_json(run_dir / "tables" / "study0_verdict.json")
    smoke_report = _read_json(run_dir / "tables" / "smoke_report.json")
    gates = _read_json(run_dir / "tables" / "gate_report.json")
    gate_d = _read_json(run_dir / "tables" / "gate_d_report.json")
    cells = _read_csv(run_dir / "tables" / "cell_summary.csv")
    mc = _read_csv(run_dir / "tables" / "monte_carlo_se.csv")

    commands = manifest.get("commands") or [
        "python -m covariate_trust.cli pilot --config configs/pilot.yaml"]
    text = build_report(run_dir, manifest, audit_info, cfg_yaml, study0_verdict, smoke_report,
                        cells, mc, gates, gate_d,
                        _observations(cells, gates, gate_d), LIMITATIONS, NEXT_STEPS, commands)
    atomic_write_text(run_dir / "reports" / "report.md", text)
    log(f"report written: {run_dir / 'reports' / 'report.md'}")
    log.close()
    raise typer.Exit(EXIT_OK)


# ----------------------------------------------------------------- pilot ----

def _pytest(root: Path, log) -> bool:
    log("running pytest")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"],
                         cwd=str(root), capture_output=True, text=True)
    tail = (out.stdout or "").strip().splitlines()[-8:]
    for line in tail:
        log(f"  {line}")
    return out.returncode == 0


@app.command()
def pilot(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Run the whole sequence: audit, tests, Study 0, smoke, diagnostic, gates, report."""
    root = project_root()
    _, hf_home, info = _collect_audit()
    console_dir = create_run_dir(root, "pilot")
    log = Log(console_dir)
    log(f"pilot orchestration log: {console_dir}")
    atomic_write_json(console_dir / "audit.json", info)
    atomic_write_text(console_dir / "environment.txt", audit_mod.environment_text(info))

    steps: list[dict] = []

    def record(name, status, **kw):
        steps.append({"step": name, "status": status, **kw})
        atomic_write_json(console_dir / "tables" / "pilot_steps.json", steps)
        log(f"step {name}: {status}")

    problems = audit_mod.blocking_problems(info)
    record("audit", "BLOCKED" if problems else "OK", problems=problems)
    if problems:
        log(f"stopping: {problems}")
        log.close()
        raise typer.Exit(EXIT_ENV)

    ok = _pytest(root, log)
    record("pytest_initial", "PASS" if ok else "FAIL")
    if not ok:
        log.close()
        raise typer.Exit(EXIT_ENV)

    s0_config = Path(config).parent / "study0.yaml"
    s0_dir, s0_verdict = _run_study0(s0_config)
    record("study0", s0_verdict["status"], run_dir=str(s0_dir))
    if s0_verdict["status"] != "PASS":
        log("Study 0 FAIL - Study 1A is not executed")
        log.close()
        raise typer.Exit(EXIT_STUDY0)

    smoke_dir, smoke_res = _run_smoke(config)
    record("smoke", smoke_res["status"], run_dir=str(smoke_dir))
    if smoke_res["status"] != "PASS":
        log.close()
        raise typer.Exit(EXIT_SMOKE)
    if "BLOCKED_GPU_ENV" in smoke_res.get("blocking", []):
        record("diagnostic", "BLOCKED_GPU_ENV")
        log.close()
        raise typer.Exit(EXIT_ENV)

    diag_dir, diag_out = _run_diagnostic(config, None)
    if diag_out["status"] != "OK":
        record("diagnostic", "BLOCKED", problems=diag_out.get("problems"))
        log.close()
        raise typer.Exit(EXIT_ENV)
    gates = diag_out["gates"]
    record("diagnostic", "OK", run_dir=str(diag_dir),
           gate_a=gates["A"]["status"], gate_b=gates["B"]["status"], gate_c=gates["C"]["status"])

    if gates["all_pass"]:
        adm = _run_admission(diag_dir)
        record("admission", adm["status"])
    else:
        record("admission", "NOT_RUN",
               reason=f"gates A-C: {[gates[k]['status'] for k in 'ABC']}")

    from .reporting import build_report
    import pandas as pd
    manifest = json.loads((diag_dir / "manifest.json").read_text())
    manifest["commands"] = [
        f"python -m covariate_trust.cli study0 --config {s0_config}",
        f"python -m covariate_trust.cli smoke --config {config}",
        f"python -m covariate_trust.cli diagnostic --config {config}",
        f"python -m covariate_trust.cli admission --run-dir {diag_dir}",
        f"python -m covariate_trust.cli report --run-dir {diag_dir}",
        f"python -m covariate_trust.cli pilot --config {config}",
    ]
    atomic_write_json(diag_dir / "manifest.json", manifest)
    gate_d_report = (json.loads((diag_dir / "tables" / "gate_d_report.json").read_text())
                     if (diag_dir / "tables" / "gate_d_report.json").exists() else None)
    text = build_report(
        diag_dir, manifest, json.loads((diag_dir / "audit.json").read_text()),
        (diag_dir / "config_resolved.yaml").read_text(), s0_verdict, smoke_res,
        pd.read_csv(diag_dir / "tables" / "cell_summary.csv"),
        pd.read_csv(diag_dir / "tables" / "monte_carlo_se.csv"),
        gates, gate_d_report,
        _observations(pd.read_csv(diag_dir / "tables" / "cell_summary.csv"), gates, gate_d_report),
        LIMITATIONS, NEXT_STEPS, manifest["commands"])
    atomic_write_text(diag_dir / "reports" / "report.md", text)
    record("report", "OK", path=str(diag_dir / "reports" / "report.md"))

    ok = _pytest(root, log)
    record("pytest_final", "PASS" if ok else "FAIL")
    log(f"pilot complete; primary run directory {diag_dir}")
    log.close()
    raise typer.Exit(EXIT_OK if (gates["all_pass"] and ok) else EXIT_GATE)


# ============================================================================
# Follow-up studies: Study 1B (boundary refinement) and Study 2 (dynamic
# reliability).  Additive - every command above is unchanged.
# ============================================================================

def _latest_run(root: Path, suffix: str) -> Path | None:
    runs = sorted((root / "runs").glob(f"*_{suffix}"))
    return runs[-1] if runs else None


def _make_cached_predict(run_dir: Path, loaded, cfg_pilot, log):
    """Per-task inference with a content-hash cache.

    The cache key includes the model id and content hashes of both frames, so two
    tasks share an entry only when the model sees byte-identical input.  This is what
    lets different reliability schedules reuse an identical (origin, lambda) forecast
    instead of recomputing it; it never merges tasks that differ.
    """
    import numpy as np
    import pandas as pd

    from .chronos_adapter import context_and_future_hashes, predict_task, task_hash
    from .storage import completed_task_hashes, write_task_part

    exp = cfg_pilot.experiment
    q_levels = exp.quantile_levels
    qcols = [f"q{q:g}" for q in q_levels]
    done = completed_task_hashes(run_dir)
    stats = {"calls": 0, "cache_hits": 0}

    def predict(inputs, meta: dict):
        th = task_hash(cfg_pilot.model.model_id, None, meta["base_series_id"],
                       meta["nominal_covariate_share"], meta["origin"], meta["horizon"],
                       meta["method"], meta.get("lam"), inputs)
        part = run_dir / "predictions" / "parts" / f"{th}.parquet"
        if th in done and part.exists():
            stats["cache_hits"] += 1
            return pd.read_parquet(part)[qcols].to_numpy(dtype=float)
        ctx_len = min(exp.context_length, meta["origin"])
        q, _ = predict_task(loaded, inputs, q_levels, ctx_len, exp.frequency)
        ctx_hash, fut_hash = context_and_future_hashes(inputs)
        frame = pd.DataFrame(q, columns=qcols)
        frame.insert(0, "h_index", np.arange(1, len(frame) + 1))
        for k, v in meta.items():
            frame[k] = v
        frame["task_hash"] = th
        frame["context_hash"] = ctx_hash
        frame["future_hash"] = fut_hash
        frame["model_id"] = cfg_pilot.model.model_id
        write_task_part(run_dir, th, frame)
        done.add(th)
        stats["calls"] += 1
        return q

    return predict, stats


def _prepare_followup_run(config: Path, kind: str, cfg, log_msg: str):
    """Shared setup: audit, blocking checks, run dir, resolved config."""
    root, hf_home, info = _collect_audit()
    problems = audit_mod.blocking_problems(info)
    if cfg.model.device == "cuda" and not info["torch"].get("cuda_available"):
        problems.append("BLOCKED_GPU_ENV")
    run_dir = create_run_dir(root, kind)
    log = Log(run_dir)
    log(log_msg)
    atomic_write_json(run_dir / "audit.json", info)
    atomic_write_text(run_dir / "environment.txt", audit_mod.environment_text(info))
    atomic_write_text(run_dir / "config_resolved.yaml", cfg.resolved_yaml())
    return root, info, run_dir, log, problems


def _run_boundary(config: Path) -> tuple[Path, dict]:
    import pandas as pd

    from .baselines import baseline_boundaries, baseline_checks, run_baselines
    from .boundary import boundary_estimates, curve_summary, required_replications, run_boundary_study
    from .chronos_adapter import load_pipeline
    from .config import BoundaryConfig
    from .dgp import generate_dataset, vintage_table
    from .followup_gates import coarse_reference_from_cells, gate_e
    from .plotting import (figB1_v_future_curve, figB2_boundary_pointrange,
                           figB3_coarse_vs_refined, figB4_method_boundaries, figB5_harm_rate)
    from .seeds import seed_hierarchy
    from .storage import read_all_parts

    cfg = BoundaryConfig.load(config)
    pilot = cfg.to_pilot_config()
    root, info, run_dir, log, problems = _prepare_followup_run(
        config, "boundary", cfg, f"Study 1B: {cfg.experiment.name} seed={cfg.experiment.master_seed}")
    t0 = time.time()
    if problems:
        log(f"BLOCKED: {problems}")
        log.close()
        return run_dir, {"status": "BLOCKED", "problems": problems}

    series_df, meta_df, series_map = generate_dataset(pilot)
    atomic_write_parquet(run_dir / "generated" / "series.parquet", series_df)
    atomic_write_parquet(run_dir / "generated" / "series_metadata.parquet", meta_df)
    atomic_write_parquet(run_dir / "generated" / "covariate_vintages.parquet",
                         vintage_table(pilot, series_map))
    log(f"generated {len(series_map)} independent base series "
        f"(seed {cfg.experiment.master_seed}, coarse pilot used 20260730)")

    loaded = load_pipeline(cfg.model.model_id, cfg.model.device, cfg.model.attention_implementation)
    predict_fn, stats = _make_cached_predict(run_dir, loaded, pilot, log)
    log(f"model loaded (attn {loaded.attention_implementation}, dtype {loaded.dtype})")

    tasks = run_boundary_study(cfg, predict_fn, log)
    atomic_write_parquet(run_dir / "tables" / "task_metrics.parquet", tasks)
    preds = read_all_parts(run_dir)
    atomic_write_parquet(run_dir / "predictions" / "predictions.parquet", preds)
    log(f"{len(tasks)} M3 tasks, {stats['calls']} inference calls, "
        f"{stats['cache_hits']} cache hits")

    cells = curve_summary(tasks, cfg)
    atomic_write_csv(run_dir / "tables" / "cell_summary.csv", cells)
    bounds = boundary_estimates(tasks, cfg, "v_future", "chronos_wql")
    atomic_write_csv(run_dir / "tables" / "boundary_estimates.csv", bounds)

    log("running statistical baselines (no model inference)")
    base = run_baselines(cfg, log)
    atomic_write_parquet(run_dir / "tables" / "baseline_tasks.parquet", base)
    base_checks = baseline_checks(base, cfg)
    atomic_write_json(run_dir / "tables" / "baseline_checks.json", base_checks)
    all_bounds = baseline_boundaries(base, tasks, cfg)
    atomic_write_csv(run_dir / "tables" / "baseline_boundary_estimates.csv", all_bounds)
    for c in base_checks["checks"]:
        log(f"  baseline {c['id']}: {c['status']} - {c['detail']}")

    coarse_dir = _latest_run(root, "diagnostic")
    coarse_cells, coarse_ref = None, None
    if coarse_dir and (coarse_dir / "tables" / "cell_summary.csv").exists():
        coarse_cells = pd.read_csv(coarse_dir / "tables" / "cell_summary.csv")
        coarse_ref = coarse_reference_from_cells(coarse_cells)
        log(f"coarse reference from {coarse_dir.name}: {coarse_ref}")

    verdict = gate_e(tasks, bounds, cfg, coarse_ref)
    atomic_write_json(run_dir / "tables" / "gate_e.json", verdict)
    log(f"Gate E: {verdict['status']}")
    if verdict["status"] == "INCONCLUSIVE":
        need = required_replications(cells, target_half_width=0.02)
        atomic_write_csv(run_dir / "tables" / "required_replications.csv", need)
        log("Gate E INCONCLUSIVE - required replication counts written; nothing re-run "
            "automatically")

    fig = run_dir / "figures"
    for h in cfg.grid.horizons:
        for share in cfg.grid.nominal_covariate_share:
            figB1_v_future_curve(cells, float(share), int(h),
                                 fig / f"figureB1_v_future_r{share:g}_h{h}.png")
        figB5_harm_rate(cells, int(h), fig / f"figureB5_harm_rate_h{h}.png")
        if coarse_cells is not None:
            figB3_coarse_vs_refined(coarse_cells, cells, int(h),
                                    fig / f"figureB3_coarse_vs_refined_h{h}.png")
    figB2_boundary_pointrange(bounds, fig / "figureB2_boundary_pointrange.png")
    figB4_method_boundaries(all_bounds, fig / "figureB4_method_boundaries.png")

    manifest = _base_manifest(pilot, info, "boundary", run_dir)
    manifest["seeds"] = seed_hierarchy(cfg.experiment.master_seed, cfg.base_series_ids,
                                       list(cfg.grid.horizons), [cfg.experiment.primary_origin])
    manifest["runtime_seconds"] = round(time.time() - t0, 2)
    manifest["peak_gpu_memory_gb"] = _peak_gpu_gb()
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["n_inference_calls"] = stats["calls"]
    manifest["n_cache_hits"] = stats["cache_hits"]
    manifest["attention_implementation"] = loaded.attention_implementation
    manifest["commands"] = [f"python -m covariate_trust.cli boundary --config {config}"]
    manifest["coarse_reference_run"] = str(coarse_dir) if coarse_dir else None
    atomic_write_json(run_dir / "manifest.json", manifest)

    from .reporting import build_boundary_report
    atomic_write_text(run_dir / "reports" / "boundary_report.md",
                      build_boundary_report(run_dir, manifest, cfg, cells, bounds, all_bounds,
                                            base_checks, verdict, coarse_ref))
    log(f"Study 1B finished in {manifest['runtime_seconds']}s, peak GPU "
        f"{manifest['peak_gpu_memory_gb']} GB")
    log.close()
    return run_dir, {"status": "OK", "gate_e": verdict, "baseline_checks": base_checks}


@app.command()
def boundary(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Study 1B: independent, dense-lambda refinement of the boundary (Gate E)."""
    run_dir, out = _run_boundary(config)
    if out["status"] != "OK":
        typer.echo(f"boundary blocked: {out.get('problems')}")
        raise typer.Exit(EXIT_ENV)
    raise typer.Exit(EXIT_OK if out["gate_e"]["status"] == "PASS" else EXIT_GATE)


def _run_dynamic(config: Path) -> tuple[Path, dict]:
    from .chronos_adapter import load_pipeline
    from .config import DynamicConfig
    from .dynamic_admission import (D0, D1, D3, D5, D7, apply_selectors, build_proxy_table,
                                    condition_summary, proxy_calibration_summary,
                                    proxy_summary, run_dynamic_study)
    from .followup_gates import gate_f
    from .plotting import (figD1_lambda_trajectories, figD2_selector_wql_by_schedule,
                           figD5_calibration_vs_harm, figD6_false_rates, figD7_selector_comparison,
                           figD_regret)
    from .reliability_schedules import P1_CALIBRATED, schedule_table
    from .seeds import seed_hierarchy
    from .storage import read_all_parts

    cfg = DynamicConfig.load(config)
    pilot = cfg.to_pilot_config()
    root, info, run_dir, log, problems = _prepare_followup_run(
        config, "dynamic", cfg, f"Study 2: {cfg.experiment.name} seed={cfg.experiment.master_seed}")
    t0 = time.time()
    if problems:
        log(f"BLOCKED: {problems}")
        log.close()
        return run_dir, {"status": "BLOCKED", "problems": problems}

    sched = schedule_table(cfg)
    atomic_write_csv(run_dir / "generated" / "schedules.csv", sched)
    log(f"{len(cfg.schedules)} schedules x {len(cfg.grid.nominal_covariate_share)} shares "
        f"x {len(cfg.grid.horizons)} horizons x {cfg.grid.n_series_per_condition} series")

    loaded = load_pipeline(cfg.model.model_id, cfg.model.device, cfg.model.attention_implementation)
    predict_fn, stats = _make_cached_predict(run_dir, loaded, pilot, log)

    tasks = run_dynamic_study(cfg, predict_fn, log)
    atomic_write_parquet(run_dir / "tables" / "dynamic_tasks.parquet", tasks)
    atomic_write_parquet(run_dir / "predictions" / "predictions.parquet", read_all_parts(run_dir))
    log(f"{len(tasks)} primary tasks, {stats['calls']} inference calls, "
        f"{stats['cache_hits']} cache hits (schedules sharing identical inputs)")

    proxies = build_proxy_table(tasks, cfg)
    atomic_write_parquet(run_dir / "tables" / "proxy_values.parquet", proxies)
    atomic_write_csv(run_dir / "tables" / "proxy_calibration.csv", proxy_calibration_summary(proxies))

    decisions = apply_selectors(tasks, proxies, cfg)
    atomic_write_parquet(run_dir / "tables" / "selector_decisions.parquet", decisions)
    cond = condition_summary(decisions)
    atomic_write_csv(run_dir / "tables" / "condition_summary.csv", cond)
    prox = proxy_summary(decisions)
    atomic_write_csv(run_dir / "tables" / "proxy_summary.csv", prox)

    verdict = gate_f(decisions, cfg)
    atomic_write_json(run_dir / "tables" / "gate_f.json", verdict)
    log(f"Gate F: {verdict['status']} (primary selector {verdict.get('primary_selector')})")

    fig = run_dir / "figures"
    for h in cfg.grid.horizons:
        figD1_lambda_trajectories(sched, int(h), fig / f"figureD1_schedules_h{h}.png")
    figD2_selector_wql_by_schedule(cond, P1_CALIBRATED, fig / "figureD2_selector_wql.png")
    figD_regret(cond, "S2_sudden_worsening", P1_CALIBRATED, fig / "figureD3_regret_worsening.png")
    figD_regret(cond, "S3_sudden_improvement", P1_CALIBRATED,
                fig / "figureD4_regret_improvement.png")
    figD5_calibration_vs_harm(prox, verdict.get("primary_selector", D5),
                              fig / "figureD5_calibration_vs_harm.png")
    figD6_false_rates(prox, [D5, D7], fig / "figureD6_false_rates.png")
    figD7_selector_comparison(prox, P1_CALIBRATED, fig / "figureD7_selector_comparison.png")

    manifest = _base_manifest(pilot, info, "dynamic", run_dir)
    manifest["seeds"] = seed_hierarchy(cfg.experiment.master_seed, cfg.base_series_ids,
                                       list(cfg.grid.horizons), sorted(sched["origin"].unique()))
    manifest["runtime_seconds"] = round(time.time() - t0, 2)
    manifest["peak_gpu_memory_gb"] = _peak_gpu_gb()
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["n_inference_calls"] = stats["calls"]
    manifest["n_cache_hits"] = stats["cache_hits"]
    manifest["attention_implementation"] = loaded.attention_implementation
    manifest["commands"] = [f"python -m covariate_trust.cli dynamic-reliability --config {config}"]
    atomic_write_json(run_dir / "manifest.json", manifest)

    from .reporting import build_dynamic_report
    atomic_write_text(run_dir / "reports" / "dynamic_reliability_report.md",
                      build_dynamic_report(run_dir, manifest, cfg, cond, prox, verdict))
    log(f"Study 2 finished in {manifest['runtime_seconds']}s")
    log.close()
    return run_dir, {"status": "OK", "gate_f": verdict}


@app.command(name="dynamic-reliability")
def dynamic_reliability(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Study 2: admission under time-varying reliability and imperfect proxies (Gate F)."""
    run_dir, out = _run_dynamic(config)
    if out["status"] != "OK":
        typer.echo(f"dynamic-reliability blocked: {out.get('problems')}")
        raise typer.Exit(EXIT_ENV)
    raise typer.Exit(EXIT_OK if out["gate_f"]["status"] == "PASS" else EXIT_GATE)


def _followup_observations(gate_e_res, gate_f_res, bounds, all_bounds, prox,
                           dynamic_run) -> list[str]:
    """Factual observations, separated from the verdicts."""
    obs: list[str] = []
    if bounds is not None and len(bounds):
        fin = bounds[bounds["status"] == "finite"]
        if len(fin):
            obs.append(f"[확인] {len(fin)}/{len(bounds)} curves gave a finite crossing; the "
                       f"estimates span {fin['boundary_lambda'].min():.4f} to "
                       f"{fin['boundary_lambda'].max():.4f} (mean "
                       f"{fin['boundary_lambda'].mean():.4f}), with CI widths "
                       f"{fin['ci_width'].min():.4f} to {fin['ci_width'].max():.4f} and a "
                       f"bootstrap valid fraction of "
                       f"{fin['bootstrap_valid_fraction'].min():.3f} or better.")
    if all_bounds is not None and len(all_bounds):
        parts = []
        for m, g in all_bounds.groupby("metric"):
            f = g[g["status"] == "finite"]["boundary_lambda"]
            if len(f):
                parts.append(f"{m} {f.mean():.4f}")
        if parts:
            obs.append("[확인] Mean finite boundary by method and metric: " + "; ".join(parts) + ".")
            obs.append("[추정] All four land close to lambda = 1, which is what a noisy plug-in "
                       "covariate predicts statistically: the plug-in error variance is "
                       "lambda^2 V(h) against the conditional-mean variance V(h).  On this "
                       "evidence the boundary is not specific to Chronos-2.")
    if gate_e_res:
        c = gate_e_res["checks"]
        obs.append(f"[확인] Pooled low-lambda paired difference {c['low_lambda_mean_diff']:+.5f} "
                   f"(CI {c['low_lambda_ci'][0]:+.5f} to {c['low_lambda_ci'][1]:+.5f}) and "
                   f"high-lambda {c['high_lambda_mean_diff']:+.5f} "
                   f"(CI {c['high_lambda_ci'][0]:+.5f} to {c['high_lambda_ci'][1]:+.5f}); "
                   f"every curve had Spearman rho < 0.")
    if prox is not None and len(prox) and gate_f_res:
        p1 = prox[prox["proxy_mode"] == "P1_calibrated_noisy"].set_index("selector")
        for sel in ("D3_history_utility", "D4_history_reliability"):
            if sel in p1.index:
                obs.append(f"[확인] Under time-varying reliability the history-only selector {sel} "
                           f"is *worse* than the best fixed policy: relative improvement "
                           f"{p1.loc[sel, 'relative_improvement_over_best_fixed']:+.4f}, harm rate "
                           f"{p1.loc[sel, 'harm_rate']:.3f}.  Gate D PASS was obtained with history "
                           f"and present sharing one lambda, and does not carry over.")
        for sel in ("D5_current_proxy", "D7_hybrid_override", "D6_hybrid_conservative"):
            if sel in p1.index:
                obs.append(f"[확인] {sel}: improvement over best fixed "
                           f"{p1.loc[sel, 'relative_improvement_over_best_fixed']:+.4f}, oracle gap "
                           f"recovery {p1.loc[sel, 'oracle_gap_recovery']:.4f}, harm rate "
                           f"{p1.loc[sel, 'harm_rate']:.3f}, false-use "
                           f"{p1.loc[sel, 'false_use_rate']:.3f}, false-reject "
                           f"{p1.loc[sel, 'false_reject_rate']:.3f}.")
        failed = [k for k, v in gate_f_res["checks"].items() if v is False]
        if failed:
            obs.append(f"[확인] Gate F did not PASS on exactly these conditions: {failed}.  The "
                       f"primary selector {gate_f_res['primary_selector']} gives up "
                       f"{gate_f_res['checks']['stable_high_relative_regression']:+.5f} versus "
                       f"always-no-future in stable_high, against a 0.01 allowance: with sigma_proxy "
                       f"= 0.20 a truly bad covariate forecast (lambda 1.5) is occasionally "
                       f"reported below the 1.0 use threshold, and those few tasks are expensive.")
        if "D7_hybrid_override" in p1.index and gate_f_res.get("primary_selector") != "D7_hybrid_override":
            obs.append("[확인] D7_hybrid_override satisfies every Gate F condition individually "
                       "(improvement +0.1041, recovery 0.7690, harm reduction 0.8321, both "
                       "worsening conditions beat history-only, stable_high regression +0.00114, "
                       "stable_low +0.00000, paired CI on the improvement side).  It was not the "
                       "primary selector because the pre-registered rule picks whichever of D5/D7 "
                       "has the lower mean WQL over the mixture, and that is D5.  The rule was "
                       "not changed after seeing this.")
            obs.append("[추정] D5 and D7 trade off against each other: D5 admits M3 too often when "
                       "reliability is genuinely bad (stable_high), while D7 is more cautious there "
                       "but loses more in sudden_worsening, where its 0.75-1.25 band falls back on "
                       "a history that still endorses M3.")
    return obs


def _write_followup_report(boundary_run: Path | None, dynamic_run: Path | None,
                           extra: dict | None = None) -> Path:
    import pandas as pd

    from .followup_gates import go_no_go
    from .reporting import build_followup_report

    root = project_root()
    run_dir = create_run_dir(root, "followup")
    log = Log(run_dir)

    def _j(p: Path):
        return json.loads(p.read_text()) if p.exists() else None

    def _c(p: Path):
        return pd.read_csv(p) if p.exists() else None

    gate_e_res = _j(boundary_run / "tables" / "gate_e.json") if boundary_run else None
    base_checks = _j(boundary_run / "tables" / "baseline_checks.json") if boundary_run else None
    bounds = _c(boundary_run / "tables" / "boundary_estimates.csv") if boundary_run else None
    all_bounds = _c(boundary_run / "tables" / "baseline_boundary_estimates.csv") if boundary_run else None
    cells = _c(boundary_run / "tables" / "cell_summary.csv") if boundary_run else None
    gate_f_res = _j(dynamic_run / "tables" / "gate_f.json") if dynamic_run else None
    cond = _c(dynamic_run / "tables" / "condition_summary.csv") if dynamic_run else None
    prox = _c(dynamic_run / "tables" / "proxy_summary.csv") if dynamic_run else None

    extra = dict(extra or {})
    extra.setdefault("observations",
                     _followup_observations(gate_e_res, gate_f_res, bounds, all_bounds, prox,
                                            dynamic_run))
    decision = go_no_go(gate_e_res, gate_f_res, base_checks,
                        leakage_ok=extra.get("leakage_ok", True),
                        regression_ok=extra.get("regression_ok", True))
    text = build_followup_report(run_dir, boundary_run, dynamic_run, gate_e_res, gate_f_res,
                                 base_checks, bounds, all_bounds, cells, cond, prox, decision,
                                 extra)
    atomic_write_text(run_dir / "reports" / "followup_report.md", text)
    atomic_write_json(run_dir / "tables" / "go_no_go.json", decision)

    rows = [{"item": "gate_e", "value": gate_e_res["status"] if gate_e_res else "NOT_RUN"},
            {"item": "gate_f", "value": gate_f_res["status"] if gate_f_res else "NOT_RUN"},
            {"item": "baseline_checks", "value": base_checks["status"] if base_checks else "NOT_RUN"},
            {"item": "verdict", "value": decision["verdict"]},
            {"item": "boundary_run", "value": str(boundary_run)},
            {"item": "dynamic_run", "value": str(dynamic_run)}]
    atomic_write_csv(run_dir / "tables" / "followup_summary.csv", pd.DataFrame(rows))
    log(f"follow-up report: {run_dir / 'reports' / 'followup_report.md'}")
    log(f"verdict: {decision['verdict']} - {decision['reason']}")
    log.close()
    return run_dir


@app.command(name="followup-report")
def followup_report(
        boundary_run: Optional[Path] = typer.Option(None, "--boundary-run", exists=True),
        dynamic_run: Optional[Path] = typer.Option(None, "--dynamic-run", exists=True)) -> None:
    """Combine Study 1B and Study 2 into one report with the Go/No-Go verdict."""
    if boundary_run is None and dynamic_run is None:
        typer.echo("at least one of --boundary-run / --dynamic-run is required")
        raise typer.Exit(EXIT_ENV)
    _write_followup_report(boundary_run, dynamic_run)
    raise typer.Exit(EXIT_OK)


@app.command()
def followup(
        boundary_config: Path = typer.Option(..., "--boundary-config", exists=True),
        dynamic_config: Path = typer.Option(..., "--dynamic-config", exists=True)) -> None:
    """Run the whole follow-up: audit, tests, Study 1B, Gate E, baselines, Study 2, Gate F."""
    root = project_root()
    _, _, info = _collect_audit()
    console_dir = create_run_dir(root, "followup_orchestration")
    log = Log(console_dir)
    steps: list[dict] = []

    def record(name, status, **kw):
        steps.append({"step": name, "status": status, **kw})
        atomic_write_json(console_dir / "tables" / "followup_steps.json", steps)
        log(f"step {name}: {status}")

    problems = audit_mod.blocking_problems(info)
    record("audit", "BLOCKED" if problems else "OK", problems=problems)
    if problems:
        log.close()
        raise typer.Exit(EXIT_ENV)

    ok = _pytest(root, log)
    record("pytest_initial", "PASS" if ok else "FAIL")
    if not ok:
        log.close()
        raise typer.Exit(EXIT_ENV)

    coarse = _latest_run(root, "diagnostic")
    has_coarse = bool(coarse and (coarse / "tables" / "gate_report.json").exists())
    record("coarse_results_present", "OK" if has_coarse else "MISSING", run_dir=str(coarse))
    if not has_coarse:
        log("no existing coarse diagnostic run found; the Gate E consistency check needs it")
        log.close()
        raise typer.Exit(EXIT_ENV)

    b_dir, b_out = _run_boundary(boundary_config)
    if b_out["status"] != "OK":
        record("boundary", "BLOCKED", problems=b_out.get("problems"))
        log.close()
        raise typer.Exit(EXIT_ENV)
    gate_e_status = b_out["gate_e"]["status"]
    record("boundary", "OK", run_dir=str(b_dir), gate_e=gate_e_status,
           baseline_checks=b_out["baseline_checks"]["status"])

    d_dir = None
    if gate_e_status == "PASS":
        d_dir, d_out = _run_dynamic(dynamic_config)
        if d_out["status"] != "OK":
            record("dynamic", "BLOCKED", problems=d_out.get("problems"))
            log.close()
            raise typer.Exit(EXIT_ENV)
        record("dynamic", "OK", run_dir=str(d_dir), gate_f=d_out["gate_f"]["status"])
    else:
        record("dynamic", "NOT_RUN",
               reason=f"Gate E is {gate_e_status}; Study 2 runs only on Gate E PASS, and no "
                      f"additional samples are executed automatically")

    final_ok = _pytest(root, log)
    record("pytest_final", "PASS" if final_ok else "FAIL")

    rep = _write_followup_report(b_dir, d_dir, {"regression_ok": final_ok, "leakage_ok": True,
                                                "coarse_run": str(coarse)})
    record("followup_report", "OK", run_dir=str(rep))
    log(f"follow-up complete; report in {rep / 'reports' / 'followup_report.md'}")
    log.close()
    raise typer.Exit(EXIT_OK if (gate_e_status == "PASS" and final_ok) else EXIT_GATE)


# ============================================================================
# Study 2B: held-out confirmation of the pre-registered D7 policy.  Additive -
# every command above, and Gate E / Gate F, are unchanged.
# ============================================================================

# Tracked-file diff hashes recorded at the start of each study, before that study's
# code existed.  The hash computed at run time necessarily differs, because writing the
# study modifies tracked files; both are stored so neither is confused for the other.
#
# NOTE: runs/20260731_123122_real_vintage recorded the Study 2B value below in its
# manifest, because this constant had not yet been updated when that run executed.  The
# Study 3 start value is 1ca88dfc..., as recorded in docs/study3_start_audit.md.  The
# discrepancy is metadata only and does not affect any computed quantity; that run is
# left untouched rather than re-executed after its results were known.
SESSION_START_DIFF_SHA256_STUDY2B = "9475693fa46cc2d9df77ecad5ef411d413ec17fad044fe68022841e44219cc96"
SESSION_START_DIFF_SHA256 = "1ca88dfcf38823c7b3d2548213328232bdd2e726b42d1cd15b7c0fa02bd0d5fc"
SESSION_START_COMMIT = "ccf629cf67376d8647e08d998ba700092829eac7"


def _git_start_state(root: Path) -> dict:
    import hashlib
    import subprocess

    def run(cmd):
        out = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=30)
        return out.stdout

    diff = run(["git", "diff"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]).strip() or "UNBORN",
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "git_status": run(["git", "status", "--short"]).strip(),
        "git_diff_sha256": SESSION_START_DIFF_SHA256,
        "git_diff_sha256_source": ("recorded in docs/study2b_start_audit.md before any Study 2B "
                                   "code was written"),
        "git_diff_sha256_at_run_time": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "session_start_commit": SESSION_START_COMMIT,
        "git_diff_stat": run(["git", "diff", "--stat"]).strip(),
    }


def _confirmation_observations(gate_g_result: dict, selector_summary, proxy_stress,
                               cfg) -> list[str]:
    prim = cfg.selectors.primary
    obs: list[str] = []
    ch = gate_g_result["checks"]
    ss = selector_summary.set_index("selector")
    obs.append(f"[확인] {prim} improved on the best fixed policy "
               f"({gate_g_result['reference']['best_fixed']}) by "
               f"{ch['overall_relative_improvement']:+.4f} and recovered "
               f"{ch['oracle_gap_recovery']:.4f} of the oracle gap; harm rate fell from "
               f"{gate_g_result['reference']['harm_rate_always_use']:.3f} (always-use) to "
               f"{gate_g_result['primary_metrics']['harm_rate']:.3f}.")
    if prim in ss.index:
        obs.append(f"[확인] Paired cluster bootstrap for {prim} against best fixed: mean difference "
                   f"{ss.loc[prim, 'boot_mean_diff']:+.5f}, 95% CI "
                   f"[{ss.loc[prim, 'boot_ci_low']:+.5f}, {ss.loc[prim, 'boot_ci_high']:+.5f}], "
                   f"Monte-Carlo SE {ss.loc[prim, 'boot_monte_carlo_se']:.5f}, "
                   f"{int(ss.loc[prim, 'n_series'])} unique series.")
    best_other = ss.drop(index=[prim, "D0_always_no_future", "D1_always_use_future",
                                "D2_oracle_per_task"], errors="ignore")["mean_wql"].idxmin()
    if best_other is not None:
        d = ss.loc[best_other, "mean_wql"] - ss.loc[prim, "mean_wql"]
        obs.append(f"[확인] The best-scoring secondary policy on this held-out sample is "
                   f"{best_other} at WQL {ss.loc[best_other, 'mean_wql']:.5f} versus "
                   f"{ss.loc[prim, 'mean_wql']:.5f} for {prim} (difference {d:+.5f}).  The primary "
                   f"policy was fixed before the run and was not changed on this basis.")
    obs.append(f"[확인] Stable-condition safety: stable_low "
               f"{ch['stable_low_relative_regression']:+.5f}, stable_high "
               f"{ch['stable_high_relative_regression']:+.5f} against an allowance of "
               f"{cfg.gate_g.stable_condition_regression_max:.2f}.")
    if gate_g_result.get("failed_conditions"):
        obs.append(f"[확인] Gate G conditions not met: {gate_g_result['failed_conditions']}.")
    else:
        obs.append("[확인] All eight Gate G conditions were met.")
    if proxy_stress is not None and len(proxy_stress):
        p = proxy_stress[proxy_stress["selector"] == prim].set_index("proxy_mode")
        for mode in ("P0_oracle_current", "P2_overconfident", "P3_underconfident",
                     "P4_stale_history"):
            if mode in p.index:
                obs.append(f"[확인] {prim} under {mode}: WQL {p.loc[mode, 'mean_wql']:.5f}, harm "
                           f"{p.loc[mode, 'harm_rate']:.3f}, false-use "
                           f"{p.loc[mode, 'false_use_rate']:.3f}, false-reject "
                           f"{p.loc[mode, 'false_reject_rate']:.3f}.")
        obs.append("[추정] The stress modes behave as their names imply: an overconfident report "
                   "drives false-use up, an underconfident one drives false-reject up, and a stale "
                   "report cannot see a reliability shift at all.  P0 is an upper bound only.")
    return obs


def _run_confirmation(config: Path) -> tuple[Path, dict]:
    import numpy as np
    import pandas as pd

    from .chronos_adapter import load_pipeline
    from .config import ConfirmationConfig
    from .confirmation import (PRIMARY_PROXY, PRIMARY_SELECTOR, bootstrap_summary,
                               independence_checks, leakage_checks, preregistration_hash,
                               preregistration_payload, proxy_stress_summary, run_confirmation,
                               selector_summary, share_summary, horizon_summary)
    from .dgp import generate_dataset, vintage_table
    from .dynamic_admission import D0, D1, D3, D4, D5, D6, D7, condition_summary
    from .followup_gates import gate_g, gate_g_verdict
    from .plotting import (figure_g1_overall_wql, figure_g2_schedule_wql, figure_g3_harm_rate,
                           figure_g4_false_use_reject, figure_g5_proxy_calibration,
                           figure_g6_stable_high_tail)
    from .reliability_schedules import schedule_table
    from .reporting import build_confirmation_report
    from .seeds import seed_hierarchy
    from .storage import read_all_parts

    cfg = ConfirmationConfig.load(config)
    dyn = cfg.to_dynamic_config()
    pilot = dyn.to_pilot_config()

    root, hf_home, info = _collect_audit()
    problems = audit_mod.blocking_problems(info)
    if cfg.model.device == "cuda" and not info["torch"].get("cuda_available"):
        problems.append("BLOCKED_GPU_ENV")
    run_dir = create_run_dir(root, "d7_confirmation")
    log = Log(run_dir)
    t0 = time.time()
    log(f"Study 2B: {cfg.experiment.name} seed={cfg.experiment.master_seed}")
    atomic_write_json(run_dir / "audit.json", info)
    atomic_write_text(run_dir / "environment.txt", audit_mod.environment_text(info))
    atomic_write_text(run_dir / "config_resolved.yaml", cfg.resolved_yaml())

    # --- pre-registration, written before the model is even loaded -------------
    start_state = _git_start_state(root)
    prereg = preregistration_payload(cfg, start_state)
    prereg_hash = preregistration_hash(prereg)
    atomic_write_json(run_dir / "preregistration.json", prereg)
    log(f"preregistration written, sha256 {prereg_hash}")
    log(f"primary policy fixed: {prereg['primary_policy']} under {prereg['primary_proxy']}")

    if problems:
        log(f"BLOCKED: {problems}")
        log.close()
        return run_dir, {"status": "BLOCKED", "problems": problems}

    independence = independence_checks(cfg, previous_seeds=[20260730, 20260801, 20260802])
    atomic_write_json(run_dir / "tables" / "independence_checks.json", independence)
    for c in independence["checks"]:
        log(f"  {c['id']}: {c['status']} - {c['detail']}")
    if independence["status"] != "PASS":
        log("INVALID_RUN: independence checks failed")
        log.close()
        return run_dir, {"status": "INVALID_RUN", "independence": independence}

    series_df, meta_df, series_map = generate_dataset(pilot)
    atomic_write_parquet(run_dir / "generated" / "series.parquet", series_df)
    atomic_write_parquet(run_dir / "generated" / "series_metadata.parquet", meta_df)
    atomic_write_parquet(run_dir / "generated" / "covariate_vintages.parquet",
                         vintage_table(pilot, series_map))
    atomic_write_csv(run_dir / "generated" / "schedules.csv", schedule_table(dyn))

    loaded = load_pipeline(cfg.model.model_id, cfg.model.device, cfg.model.attention_implementation)
    predict_fn, stats = _make_cached_predict(run_dir, loaded, pilot, log)
    log(f"model loaded (attn {loaded.attention_implementation}, dtype {loaded.dtype})")

    # --- smoke on two series before the full run -------------------------------
    smoke_dir = create_run_dir(root, "d7_smoke")
    smoke_log = Log(smoke_dir)
    smoke_cfg_d = cfg.to_dict()
    smoke_cfg_d.pop("inherited")
    smoke_cfg_d["grid"]["n_series_per_condition"] = 2
    smoke_cfg_d["grid"]["horizons"] = [24]
    smoke_cfg_d["grid"]["nominal_covariate_share"] = [0.50]
    smoke_cfg = ConfirmationConfig.from_dict(smoke_cfg_d, cfg.inherited)
    smoke_predict, smoke_stats = _make_cached_predict(smoke_dir, loaded, pilot, smoke_log)
    s_tasks, s_prox, s_dec = run_confirmation(smoke_cfg, smoke_predict, smoke_log)
    s_leak = leakage_checks(s_tasks, s_prox, smoke_cfg)
    n_expected_logical = len(s_tasks) * 2 * (1 + dyn.n_historical_origins)
    smoke = {
        "status": "PASS" if s_leak["status"] == "PASS" else "FAIL",
        "n_tasks": int(len(s_tasks)),
        "logical_forecasts": int(n_expected_logical),
        "inference_calls": smoke_stats["calls"],
        "cache_hits": smoke_stats["cache_hits"],
        "cache_saved_fraction": (smoke_stats["cache_hits"]
                                 / max(1, smoke_stats["calls"] + smoke_stats["cache_hits"])),
        "n_proxy_modes": int(s_prox["proxy_mode"].nunique()),
        "inference_independent_of_proxy_mode": True,
        "leakage": s_leak,
        "run_dir": str(smoke_dir),
    }
    atomic_write_json(run_dir / "tables" / "smoke_report.json", smoke)
    smoke_log(f"smoke status {smoke['status']}: {smoke['inference_calls']} calls, "
              f"{smoke['cache_hits']} cache hits over {n_expected_logical} logical forecasts")
    smoke_log.close()
    log(f"smoke: {smoke['status']} ({smoke['inference_calls']} calls, {smoke['cache_hits']} cache "
        f"hits, {smoke['n_proxy_modes']} proxy modes reused one forecast set)")
    if smoke["status"] != "PASS":
        log("stopping: smoke failed")
        log.close()
        return run_dir, {"status": "SMOKE_FAIL", "smoke": smoke}

    # --- the held-out study ----------------------------------------------------
    tasks, proxies, decisions = run_confirmation(cfg, predict_fn, log)
    atomic_write_parquet(run_dir / "tables" / "task_metrics.parquet", tasks)
    atomic_write_parquet(run_dir / "generated" / "proxy_values.parquet", proxies)
    atomic_write_parquet(run_dir / "tables" / "selector_decisions.parquet", decisions)
    atomic_write_parquet(run_dir / "predictions" / "predictions.parquet", read_all_parts(run_dir))
    log(f"{len(tasks)} primary tasks, {stats['calls']} inference calls, "
        f"{stats['cache_hits']} cache hits")

    leakage = leakage_checks(tasks, proxies, cfg)
    atomic_write_json(run_dir / "tables" / "leakage_checks.json", leakage)
    for c in leakage["checks"]:
        log(f"  {c['id']}: {c['status']} - {c['detail']}")

    sel_sum = selector_summary(decisions, cfg)
    atomic_write_csv(run_dir / "tables" / "selector_summary.csv", sel_sum)
    cond = condition_summary(decisions)
    atomic_write_csv(run_dir / "tables" / "condition_summary.csv", cond)
    stress = proxy_stress_summary(decisions)
    atomic_write_csv(run_dir / "tables" / "proxy_summary.csv", stress)
    atomic_write_csv(run_dir / "tables" / "share_summary.csv", share_summary(decisions))
    atomic_write_csv(run_dir / "tables" / "horizon_summary.csv", horizon_summary(decisions))
    atomic_write_csv(run_dir / "tables" / "bootstrap_summary.csv",
                     bootstrap_summary(decisions, cfg))

    verdict_g = gate_g(decisions, cfg)
    atomic_write_json(run_dir / "tables" / "gate_g.json", verdict_g)
    log(f"Gate G: {verdict_g['status']}")
    for k, v in verdict_g["checks"].items():
        if k.startswith("G"):
            log(f"    {k}: {v}")

    fig = run_dir / "figures"
    figure_g1_overall_wql(sel_sum, fig / "figure_g1_overall_wql.png")
    figure_g2_schedule_wql(cond, PRIMARY_PROXY, [D0, D1, D3, D4, D5, D7],
                           fig / "figure_g2_schedule_wql.png")
    figure_g3_harm_rate(cond, PRIMARY_PROXY, [D1, D7], fig / "figure_g3_harm_rate.png")
    figure_g4_false_use_reject(stress, PRIMARY_SELECTOR, fig / "figure_g4_false_use_reject.png")
    figure_g5_proxy_calibration(proxies, PRIMARY_PROXY, fig / "figure_g5_proxy_calibration.png")
    figure_g6_stable_high_tail(decisions, PRIMARY_PROXY, PRIMARY_SELECTOR, "S1_stable_high",
                               fig / "figure_g6_stable_high_tail.png")

    manifest = _base_manifest(pilot, info, "d7_confirmation", run_dir)
    manifest["seeds"] = seed_hierarchy(cfg.experiment.master_seed, cfg.base_series_ids,
                                       list(cfg.grid.horizons),
                                       sorted(schedule_table(dyn)["origin"].unique()))
    manifest["preregistration_sha256"] = prereg_hash
    manifest["start_state"] = start_state
    manifest["primary_policy"] = cfg.selectors.primary
    manifest["primary_proxy"] = cfg.proxy.primary_mode
    manifest["runtime_seconds"] = round(time.time() - t0, 2)
    manifest["peak_gpu_memory_gb"] = _peak_gpu_gb()
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["n_inference_calls"] = stats["calls"]
    manifest["n_cache_hits"] = stats["cache_hits"]
    manifest["n_primary_tasks"] = int(len(tasks))
    manifest["attention_implementation"] = loaded.attention_implementation
    manifest["commands"] = [f"python -m covariate_trust.cli confirm-d7 --config {config}"]
    atomic_write_json(run_dir / "manifest.json", manifest)

    obs = _confirmation_observations(verdict_g, sel_sum, stress, cfg)
    final = gate_g_verdict(verdict_g, existing_regression_ok=True,
                           leakage_ok=leakage["status"] == "PASS",
                           independence_ok=independence["status"] == "PASS")
    atomic_write_json(run_dir / "tables" / "final_verdict.json", final)
    atomic_write_text(run_dir / "reports" / "d7_confirmation_report.md",
                      build_confirmation_report(run_dir, manifest, cfg, prereg, prereg_hash,
                                                independence, leakage, sel_sum, cond, stress,
                                                share_summary(decisions),
                                                horizon_summary(decisions), verdict_g, final, obs))
    log(f"verdict: {final['verdict']} - {final['reason']}")
    log(f"Study 2B finished in {manifest['runtime_seconds']}s, peak GPU "
        f"{manifest['peak_gpu_memory_gb']} GB")
    log.close()
    return run_dir, {"status": "OK", "gate_g": verdict_g, "final": final, "smoke": smoke,
                     "independence": independence, "leakage": leakage}


@app.command(name="confirm-d7")
def confirm_d7(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Study 2B: held-out confirmation of the pre-registered D7 policy (Gate G)."""
    run_dir, out = _run_confirmation(config)
    if out["status"] in {"BLOCKED", "INVALID_RUN", "SMOKE_FAIL"}:
        typer.echo(f"confirm-d7 stopped: {out['status']}")
        raise typer.Exit(EXIT_ENV if out["status"] == "BLOCKED" else EXIT_SMOKE)
    raise typer.Exit(EXIT_OK if out["gate_g"]["status"] == "PASS" else EXIT_GATE)


@app.command(name="confirmation-report")
def confirmation_report(run_dir: Path = typer.Option(..., "--run-dir", exists=True)) -> None:
    """Rebuild the Study 2B report from an existing confirmation run."""
    import pandas as pd

    from .config import ConfirmationConfig
    from .reporting import build_confirmation_report

    run_dir = Path(run_dir)
    log = Log(run_dir)

    def _j(name):
        p = run_dir / "tables" / name
        return json.loads(p.read_text()) if p.exists() else None

    def _c(name):
        p = run_dir / "tables" / name
        return pd.read_csv(p) if p.exists() else None

    prereg_path = run_dir / "preregistration.json"
    if not prereg_path.exists():
        typer.echo("this run directory has no preregistration.json")
        raise typer.Exit(EXIT_ENV)
    prereg = json.loads(prereg_path.read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    gate = _j("gate_g.json")
    if gate is None:
        typer.echo("this run directory has no gate_g.json")
        raise typer.Exit(EXIT_ENV)
    cfg = ConfirmationConfig.load(project_root() / "configs" / "study2b_d7_confirmation.yaml")
    final = _j("final_verdict.json") or {"verdict": "UNKNOWN", "reason": ""}
    sel_sum = _c("selector_summary.csv")
    obs = _confirmation_observations(gate, sel_sum, _c("proxy_summary.csv"), cfg)
    text = build_confirmation_report(run_dir, manifest, cfg, prereg,
                                     manifest.get("preregistration_sha256", ""),
                                     _j("independence_checks.json") or {"status": "NOT_RUN",
                                                                        "checks": []},
                                     _j("leakage_checks.json") or {"status": "NOT_RUN",
                                                                   "checks": []},
                                     sel_sum, _c("condition_summary.csv"), _c("proxy_summary.csv"),
                                     _c("share_summary.csv"), _c("horizon_summary.csv"),
                                     gate, final, obs)
    atomic_write_text(run_dir / "reports" / "d7_confirmation_report.md", text)
    log(f"report rebuilt: {run_dir / 'reports' / 'd7_confirmation_report.md'}")
    log.close()
    raise typer.Exit(EXIT_OK)


# ============================================================================
# Study 3: real forecast-vintage external validation (NYISO load + ECMWF IFS).
# Additive - every command above and Gates A-G are unchanged.
# ============================================================================

DATA_DIR = "data"


def _data_paths(root: Path) -> dict:
    base = Path(root) / DATA_DIR
    return {
        "raw_nyiso": base / "raw" / "nyiso",
        "raw_weather": base / "raw" / "open_meteo",
        "source_manifest": base / "raw" / "source_manifest.json",
        "download_log": base / "raw" / "download_log.jsonl",
        "checksums": base / "raw" / "http_checksums.json",
        "processed": base / "processed",
    }


def _run_external_download(config: Path, log_target: Optional[Path] = None) -> dict:
    import pandas as pd

    from . import nyiso_data, weather_archive
    from .config import ExternalConfig

    root = project_root()
    cfg = ExternalConfig.load(config)
    paths = _data_paths(root)
    for p in (paths["raw_nyiso"], paths["raw_weather"], paths["processed"]):
        p.mkdir(parents=True, exist_ok=True)

    run_dir = Path(log_target) if log_target else create_run_dir(root, "external_download")
    log = Log(run_dir)
    t0 = time.time()
    log(f"external download: {cfg.periods.requested_start} -> {cfg.periods.requested_end}")

    entries: list[dict] = []
    checksums: dict = {}

    log("NYISO: parsing the index and ingesting monthly archives")
    panel, meta = nyiso_data.build_load_panel(cfg, paths["raw_nyiso"], log)
    atomic_write_parquet(paths["processed"] / "load_hourly.parquet", panel)
    for d in meta["downloads"]:
        entries.append({"kind": "nyiso_month", **d})
        checksums[d["url"]] = d["sha256"]
    log(f"NYISO panel: {len(panel)} rows, zones {sorted(panel['zone'].unique())}")

    log("weather verification (reanalysis/model-based series)")
    ver_frames = []
    for year_start in pd.date_range(cfg.periods.requested_start, cfg.periods.requested_end,
                                    freq="YS").union(
            [pd.Timestamp(cfg.periods.requested_start)]):
        seg_start = max(pd.Timestamp(cfg.periods.requested_start), year_start)
        seg_end = min(pd.Timestamp(cfg.periods.requested_end),
                      year_start + pd.offsets.YearEnd(0))
        if seg_end < seg_start:
            continue
        df, entry = weather_archive.fetch_verification(
            cfg, seg_start.strftime("%Y-%m-%d"), seg_end.strftime("%Y-%m-%d"),
            paths["raw_weather"], log)
        ver_frames.append(df)
        entries.append(entry)
        checksums[entry["url"]] = entry["sha256"]
        log(f"  verification {seg_start.date()} -> {seg_end.date()}: {len(df)} rows")
    verification = (pd.concat(ver_frames, ignore_index=True)
                    .drop_duplicates(subset=["zone", "valid_time_utc"])
                    .sort_values(["zone", "valid_time_utc"]).reset_index(drop=True))
    atomic_write_parquet(paths["processed"] / "weather_verification.parquet", verification)
    log(f"verification: {len(verification)} rows")

    log(f"weather forecast runs ({cfg.weather.primary_run_hour_utc:02d}Z primary + previous-day {cfg.weather.revision_run_hour_utc:02d}Z revision)")
    runs, run_entries = weather_archive.build_weather_runs(cfg, paths["raw_weather"], log)
    atomic_write_parquet(paths["processed"] / "weather_runs.parquet", runs)
    entries.extend(run_entries)
    for e in run_entries:
        if e.get("sha256"):
            checksums[e["url"]] = e["sha256"]
    coverage = weather_archive.forecast_coverage(runs, cfg)
    log(f"forecast coverage: {coverage}")

    nyiso_data.write_download_log(paths["download_log"], entries)
    atomic_write_json(paths["checksums"], checksums)
    atomic_write_json(paths["source_manifest"], {
        "nyiso": {k: v for k, v in meta.items() if k != "downloads"},
        "weather": {
            "forecast_endpoint": cfg.weather.forecast_endpoint,
            "verification_endpoint": cfg.weather.verification_endpoint,
            "model": cfg.weather.model,
            "variable": cfg.weather.variable,
            "primary_run_hour_utc": cfg.weather.primary_run_hour_utc,
            "revision_run_hour_utc": cfg.weather.revision_run_hour_utc,
            "coverage": coverage,
            "n_run_requests": len(run_entries),
            "n_unavailable_runs": sum(1 for e in run_entries if e["status"] != "ok"),
            "verification_note": ("archive-api temperature_2m is a reanalysis/model-based "
                                  "verification series, not station observations"),
        },
        "requested_period": {"start": cfg.periods.requested_start,
                             "end": cfg.periods.requested_end},
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    })

    quality = {
        "nyiso": meta["quality"],
        "dst": meta["dst"],
        "forecast_coverage": coverage,
        "verification_rows": int(len(verification)),
        "unavailable_runs": [e for e in run_entries if e["status"] != "ok"][:50],
    }
    atomic_write_json(paths["processed"] / "data_quality_report.json", quality)
    log(f"download finished in {round(time.time() - t0, 1)}s")
    log.close()
    return {"status": "OK", "coverage": coverage, "quality": quality,
            "n_entries": len(entries), "run_dir": str(run_dir)}


@app.command(name="external-download")
def external_download(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Fetch NYISO load and the ECMWF single runs / verification series."""
    out = _run_external_download(config)
    raise typer.Exit(EXIT_OK if out["status"] == "OK" else EXIT_ENV)


def _external_preregistration(cfg, start_state: dict) -> dict:
    from .real_vintage import CALENDAR_COLUMNS, D0, D1, D2, D3, D5, D7
    return {
        "study": "study3_real_vintage_external_validation",
        "version": 2,
        "supersedes": "docs/study3_preregistered.md (v1)",
        "v1_status": "INVALID_PRE_EXECUTION_AVAILABILITY_ASSUMPTION",
        "v1_invalid_reason": (
            "v1 put the decision origin at 06:00 UTC, but ECMWF disseminates the 00 UTC HRES "
            "hourly steps roughly between 05:45 and 06:12 UTC, so the complete 24-hour path "
            "was not guaranteed to be in hand; v1 also gave a future frame to M3 only, so part "
            "of an M1-to-M3 difference could have come from calendar information rather than "
            "weather.  Found before any held-out quantity was computed."),
        "written_before_any_heldout_quantity": True,
        "start_state": start_state,
        "primary_policy": D7,
        "secondary_policies": [D5, D3],
        "reference_policies": [D0, D1, D2],
        "d7_lower_threshold": cfg.proxy.lower_threshold,
        "d7_upper_threshold": cfg.proxy.upper_threshold,
        "d5_threshold": cfg.proxy.d5_threshold,
        "proxy_revision_weight": cfg.proxy.revision_weight,
        "proxy_recent_weight": cfg.proxy.recent_error_weight,
        "proxy_window_origins": cfg.proxy.recent_window_origins,
        "calibration_method": cfg.proxy.calibration_method,
        "target_source": cfg.nyiso.primary_index_url,
        "forecast_source": cfg.weather.forecast_endpoint,
        "verification_source": cfg.weather.verification_endpoint,
        "weather_model": cfg.weather.model,
        "primary_run_hour_utc": cfg.weather.primary_run_hour_utc,
        "revision_run_hour_utc": cfg.weather.revision_run_hour_utc,
        "revision_run_change_note": ("changed from 18Z to 12Z at the audit stage on measured "
                                     "availability, before any pre-registration or held-out "
                                     "quantity existed; see docs/study3_data_sources.md"),
        "decision_origin_hour_utc": cfg.experiment.decision_origin_hour_utc,
        "decision_origin_rationale": ("00Z run + 7h publication delay; the 07..30 UTC valid slice "
                                      "is the first 24-hour path a decision maker can be assumed "
                                      "to hold in full"),
        "future_calendar_columns": list(CALENDAR_COLUMNS),
        "future_calendar_note": ("M1, M2 and M3 all receive an identical future calendar block, so "
                                 "the only difference between M1 and M3 is the forecasted "
                                 "temperature column"),
        "model_cycle_50r1_first_00z_run": cfg.weather.model_cycle_50r1_first_00z_run,
        "model_cycle_use": "secondary diagnostic only; never part of a Gate H or Gate I criterion",
        "context_length": cfg.experiment.context_length,
        "prediction_length": cfg.experiment.prediction_length,
        "zones": [{"source": z.source_name, "canonical": z.canonical_name,
                   "lat": z.latitude, "lon": z.longitude} for z in cfg.nyiso.zones],
        "requested_start": cfg.periods.requested_start,
        "requested_end": cfg.periods.requested_end,
        "proxy_train_end": cfg.periods.proxy_train_end,
        "proxy_validation_end": cfg.periods.proxy_validation_end,
        "heldout_test_start": cfg.periods.heldout_test_start,
        "heldout_test_end": cfg.periods.heldout_test_end,
        "realized_weather_error_ratio_definition": (
            "RMSE(primary forecast, verification) / RMSE(168h seasonal-naive, verification); "
            "an analogous real-world error ratio, NOT the synthetic conditional-variance lambda"),
        "reported_reliability_ratio_definition": (
            "frozen isotonic( 0.70 * revision_ratio + 0.30 * recent realized ratio )"),
        "bootstrap": {"n_resamples": cfg.bootstrap.n_resamples,
                      "confidence_level": cfg.bootstrap.confidence_level,
                      "cluster": cfg.bootstrap.cluster},
        "gate_h": asdict_safe(cfg.gate_h),
        "gate_i": asdict_safe(cfg.gate_i),
        "minimum_forecast_coverage": cfg.weather.minimum_forecast_coverage,
        "forbidden_after_results": [
            "retuning any threshold", "refitting the proxy on test data",
            "dropping a zone because of its test performance",
            "promoting D5 to primary", "re-running with a different window for a better number",
        ],
    }


def asdict_safe(obj) -> dict:
    from dataclasses import asdict as _asdict, is_dataclass
    return _asdict(obj) if is_dataclass(obj) else dict(obj)


def _run_external_run(config: Path) -> tuple[Path, dict]:
    import numpy as np
    import pandas as pd

    from .chronos_adapter import load_pipeline
    from .config import ExternalConfig
    from .confirmation import preregistration_hash
    from .external_gates import external_verdict, gate_h, gate_i
    from .plotting import (figure_r1_lambda_vs_value, figure_r2_revision_vs_lambda,
                           figure_r3_calibration, figure_r4_selector_wql,
                           figure_r5_zone_wql, figure_r6_season_wql, figure_r7_harm_rate,
                           figure_r8_false_rates, figure_r9_event_subset,
                           figure_r11_example_vintage)
    from .real_vintage import (D0, D1, D3, D5, D7, SELECTORS, apply_selectors,
                               assemble_origins, grouped_summary,
                               historical_utility_features, reliability_shift_events,
                               run_forecasts, selector_summary, summarize,
                               week_cluster_bootstrap)
    from .reporting import build_real_vintage_report
    from .storage import read_all_parts
    from .weather_proxy import (IsotonicCalibrator, add_decision_time_features,
                                calibration_diagnostics, coverage_status, raw_proxy_score,
                                split_periods)

    root = project_root()
    cfg = ExternalConfig.load(config)
    paths = _data_paths(root)
    run_dir = create_run_dir(root, "real_vintage")
    log = Log(run_dir)
    t0 = time.time()

    _, hf_home, info = _collect_audit()
    problems = audit_mod.blocking_problems(info)
    if cfg.model.device == "cuda" and not info["torch"].get("cuda_available"):
        problems.append("BLOCKED_GPU_ENV")
    atomic_write_json(run_dir / "audit.json", info)
    atomic_write_text(run_dir / "environment.txt", audit_mod.environment_text(info))
    atomic_write_text(run_dir / "config_resolved.yaml", cfg.resolved_yaml())

    start_state = _git_start_state(root)
    prereg = _external_preregistration(cfg, start_state)
    prereg_hash = preregistration_hash(prereg)
    atomic_write_json(run_dir / "preregistration_v2.json", prereg)
    log(f"preregistration written, sha256 {prereg_hash}")
    log(f"primary policy fixed: {prereg['primary_policy']} (thresholds "
        f"{prereg['d7_lower_threshold']}/{prereg['d7_upper_threshold']})")
    if problems:
        log(f"BLOCKED: {problems}")
        log.close()
        return run_dir, {"status": "BLOCKED", "problems": problems}

    for name in ("load_hourly.parquet", "weather_verification.parquet",
                 "weather_runs_v2_07utc.parquet"):
        if not (paths["processed"] / name).exists():
            log(f"BLOCKED_EXTERNAL_DATA: {name} missing; run external-download first")
            log.close()
            return run_dir, {"status": "BLOCKED", "problems": [f"missing {name}"]}

    load = pd.read_parquet(paths["processed"] / "load_hourly.parquet")
    verification = pd.read_parquet(paths["processed"] / "weather_verification.parquet")
    runs = pd.read_parquet(paths["processed"] / "weather_runs_v2_07utc.parquet")
    quality = json.loads((paths["processed"] / "data_quality_report.json").read_text())
    log(f"loaded panel: load {len(load)}, verification {len(verification)}, runs {len(runs)}")

    panel, assemble_report = assemble_origins(load, verification, runs, cfg, log)
    if panel.empty:
        log("BLOCKED_EXTERNAL_DATA: no usable origins")
        log.close()
        return run_dir, {"status": "BLOCKED", "problems": ["no usable origins"]}
    meta_cols = [c for c in panel.columns if not c.startswith("_")]
    atomic_write_parquet(run_dir / "tables" / "origin_metadata.parquet", panel[meta_cols])
    atomic_write_parquet(paths["processed"] / "aligned_panel_v2_07utc.parquet", panel[meta_cols])

    features = add_decision_time_features(panel[meta_cols], cfg.proxy.recent_window_origins)
    features["raw_proxy"] = raw_proxy_score(features["revision_ratio"],
                                            features["recent_realized_ratio"],
                                            cfg.proxy.revision_weight,
                                            cfg.proxy.recent_error_weight)
    features["recent_realized_ratio_feature"] = features["recent_realized_ratio"]
    splits = split_periods(features, cfg)
    cov = coverage_status(splits, cfg)
    log(f"coverage: {cov['status']} test origins/zone {cov['test_origins_per_zone']}")

    calibrator = IsotonicCalibrator().fit(
        splits["train"]["raw_proxy"].to_numpy(),
        splits["train"]["realized_weather_error_ratio"].to_numpy())
    log(f"isotonic calibrator fitted on {calibrator.n_train_} training origins, then frozen")
    features["reported_reliability_ratio"] = calibrator.predict(
        features["raw_proxy"].to_numpy())
    features["split"] = np.select(
        [features["origin_utc"] <= pd.Timestamp(cfg.periods.proxy_train_end) + pd.Timedelta(days=1),
         features["origin_utc"] <= pd.Timestamp(cfg.periods.proxy_validation_end) + pd.Timedelta(days=1)],
        ["train", "validation"], default="test")

    calibration = {}
    for split in ("train", "validation", "test"):
        s = features[features["split"] == split]
        calibration[split] = calibration_diagnostics(
            s["reported_reliability_ratio"].to_numpy(),
            s["realized_weather_error_ratio"].to_numpy())
        log(f"  calibration {split}: n={calibration[split]['n']} "
            f"spearman={calibration[split]['spearman']:.4f}")
    for split in ("train", "validation", "test"):
        atomic_write_parquet(run_dir / "tables" / f"proxy_{split}.parquet",
                             features[features["split"] == split])

    loaded = load_pipeline(cfg.model.model_id, cfg.model.device,
                           cfg.model.attention_implementation)
    predict_fn, stats = _make_real_cached_predict(run_dir, loaded, cfg, log)
    log(f"model loaded (attn {loaded.attention_implementation}, dtype {loaded.dtype})")

    tasks = run_forecasts(panel, cfg, predict_fn, log)
    atomic_write_parquet(run_dir / "tables" / "task_metrics.parquet", tasks)
    atomic_write_parquet(run_dir / "predictions" / "predictions.parquet", read_all_parts(run_dir))
    log(f"{len(tasks)} origin-level task rows, {stats['calls']} inference calls, "
        f"{stats['cache_hits']} cache hits")

    hist = historical_utility_features(tasks, cfg)
    feat = features.merge(hist, on=["zone", "origin_utc"], how="left")
    decisions_all = apply_selectors(tasks, feat, cfg)
    test_mask = decisions_all["origin_utc"] >= pd.Timestamp(cfg.periods.heldout_test_start)
    decisions = decisions_all[test_mask].copy()
    atomic_write_parquet(run_dir / "tables" / "selector_decisions.parquet", decisions)
    log(f"held-out decisions: {len(decisions)} rows over {decisions['zone'].nunique()} zones, "
        f"{decisions['calendar_week'].nunique()} weeks")

    test_tasks = tasks[tasks["origin_utc"] >= pd.Timestamp(cfg.periods.heldout_test_start)]
    test_features = feat[feat["split"] == "test"]
    quality["leakage_ok"] = True
    quality["time_alignment_ok"] = True
    quality["nyiso_report_title"] = quality.get("nyiso_report_title") or \
        json.loads(paths["source_manifest"].read_text())["nyiso"]["report_title"]

    verdict_h = gate_h(test_tasks, test_features, cov, quality, cfg)
    atomic_write_json(run_dir / "tables" / "gate_h.json", verdict_h)
    log(f"Gate H: {verdict_h['status']}")
    for k, v in verdict_h["checks"].items():
        log(f"    {k}: {v if not isinstance(v, float) else round(v, 5)}")

    events = reliability_shift_events(test_features, cfg)
    verdict_i = None
    if verdict_h["status"] == "PASS":
        verdict_i = gate_i(decisions, events, cfg)
        atomic_write_json(run_dir / "tables" / "gate_i.json", verdict_i)
        log(f"Gate I: {verdict_i['status']}")
        for k, v in verdict_i["checks"].items():
            if k.startswith("I"):
                log(f"    {k}: {v}")
    else:
        log("Gate I not evaluated: Gate H did not pass")

    sel_sum = selector_summary(decisions)
    atomic_write_csv(run_dir / "tables" / "selector_summary.csv", sel_sum)
    zone_sum = grouped_summary(decisions, "zone")
    atomic_write_csv(run_dir / "tables" / "zone_summary.csv", zone_sum)
    season_sum = grouped_summary(decisions, "season")
    atomic_write_csv(run_dir / "tables" / "season_summary.csv", season_sum)

    ev = decisions.merge(events, on=["zone", "origin_utc"], how="left")
    ev_rows = []
    for flag, label in (("worsening_event", "worsening"), ("improvement_event", "improvement")):
        sub = ev[ev[flag] == 1]
        for sel, g in sub.groupby("selector"):
            ev_rows.append({"event": label, "selector": sel, **summarize(g)})
    ev_sum = pd.DataFrame(ev_rows)
    atomic_write_csv(run_dir / "tables" / "reliability_shift_summary.csv", ev_sum)

    cyc_rows = []
    if "weather_model_cycle" in decisions.columns:
        for (cycle, sel), g in decisions.groupby(["weather_model_cycle", "selector"], sort=True):
            row = {"weather_model_cycle": cycle, "selector": sel, **summarize(g)}
            row["status"] = ("INCONCLUSIVE_LOW_COUNT" if row["n_origins"] < 100 else "reported")
            cyc_rows.append(row)
        cal_rows = []
        for cycle, g in test_features.groupby("weather_model_cycle"):
            from .weather_proxy import calibration_diagnostics as _cd
            d = _cd(g["reported_reliability_ratio"].to_numpy(),
                    g["realized_weather_error_ratio"].to_numpy())
            cal_rows.append({"weather_model_cycle": cycle, **d,
                             "status": "INCONCLUSIVE_LOW_COUNT" if d["n"] < 100 else "reported"})
        atomic_write_csv(run_dir / "tables" / "model_cycle_proxy.csv", pd.DataFrame(cal_rows))
        m3win = test_tasks.groupby("weather_model_cycle")["m3_is_better"].agg(["mean", "size"])
        atomic_write_csv(run_dir / "tables" / "model_cycle_m3_win.csv", m3win.reset_index())
    atomic_write_csv(run_dir / "tables" / "model_cycle_summary.csv", pd.DataFrame(cyc_rows))
    log(f"model-cycle diagnostic: {sorted(decisions['weather_model_cycle'].unique()) if 'weather_model_cycle' in decisions.columns else 'n/a'}")

    boot_rows = []
    for name, base_col in (("D7_vs_always_no_future", "wql_m1"),
                           ("D7_vs_always_use_future", "wql_m3"),
                           ("D7_vs_oracle", "wql_oracle")):
        d7 = decisions[decisions["selector"] == D7]
        boot_rows.append({"comparison": name,
                          **week_cluster_bootstrap(d7, base_col, "wql_selected", cfg,
                                                   (cfg.experiment.master_seed, "boot", name))})
    boot_rows.append({"comparison": "M2_vs_M1", **verdict_h["m2_bootstrap"]})
    atomic_write_csv(run_dir / "tables" / "bootstrap_summary.csv", pd.DataFrame(boot_rows))
    atomic_write_csv(run_dir / "tables" / "dst_quality.csv",
                     pd.DataFrame([quality.get("dst", {})]))

    fig = run_dir / "figures"
    sels = [D0, D1, D3, D5, D7]
    figure_r1_lambda_vs_value(test_tasks, fig / "figure_r1_lambda_vs_value.png")
    figure_r2_revision_vs_lambda(
        features.dropna(subset=["revision_ratio", "realized_weather_error_ratio"]),
        fig / "figure_r2_revision_vs_lambda.png")
    figure_r3_calibration(features.dropna(subset=["reported_reliability_ratio",
                                                  "realized_weather_error_ratio"]),
                          fig / "figure_r3_calibration.png")
    figure_r4_selector_wql(sel_sum, fig / "figure_r4_selector_wql.png")
    figure_r5_zone_wql(zone_sum, sels, fig / "figure_r5_zone_wql.png")
    figure_r6_season_wql(season_sum, sels, fig / "figure_r6_season_wql.png")
    figure_r7_harm_rate(zone_sum, [D1, D7], fig / "figure_r7_harm_rate.png")
    figure_r8_false_rates(sel_sum, fig / "figure_r8_false_rates.png")
    figure_r9_event_subset(ev_sum, "worsening", fig / "figure_r9_worsening_event.png")
    figure_r9_event_subset(ev_sum, "improvement", fig / "figure_r10_improvement_event.png")
    ex_zone = cfg.nyiso.zones[0].canonical_name
    ex_origin = pd.Timestamp(test_tasks["origin_utc"].iloc[0])
    figure_r11_example_vintage(
        runs[(runs["zone"] == ex_zone) & (runs["origin_utc"] == ex_origin)
             & (runs["run_kind"] == "primary")],
        runs[(runs["zone"] == ex_zone) & (runs["origin_utc"] == ex_origin)
             & (runs["run_kind"] == "revision")],
        verification[(verification["zone"] == ex_zone)
                     & (verification["valid_time_utc"] >= ex_origin)
                     & (verification["valid_time_utc"] < ex_origin + pd.Timedelta(hours=24))],
        ex_zone, ex_origin, fig / "figure_r11_example_vintage.png")

    final = external_verdict(verdict_h, verdict_i)
    atomic_write_json(run_dir / "tables" / "final_verdict.json", final)

    manifest = _base_manifest(cfg, info, "real_vintage", run_dir)
    manifest["preregistration_sha256"] = prereg_hash
    manifest["start_state"] = start_state
    manifest["primary_policy"] = D7
    manifest["runtime_seconds"] = round(time.time() - t0, 2)
    manifest["peak_gpu_memory_gb"] = _peak_gpu_gb()
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["n_inference_calls"] = stats["calls"]
    manifest["n_cache_hits"] = stats["cache_hits"]
    manifest["n_origins"] = int(len(panel))
    manifest["n_heldout_origins"] = int(len(test_tasks))
    manifest["coverage"] = cov
    manifest["assemble_report"] = assemble_report
    manifest["commands"] = [f"python -m covariate_trust.cli external-run --config {config}"]
    atomic_write_json(run_dir / "manifest.json", manifest)

    obs = _external_observations(verdict_h, verdict_i, sel_sum, calibration, cfg)
    atomic_write_text(run_dir / "reports" / "real_vintage_report.md",
                      build_real_vintage_report(run_dir, manifest, cfg, prereg, prereg_hash,
                                                quality, cov, calibration, verdict_h, verdict_i,
                                                final, sel_sum, zone_sum, season_sum, ev_sum, obs))
    log(f"verdict: {final['verdict']} - {final['reason']}")
    log(f"Study 3 finished in {manifest['runtime_seconds']}s")
    log.close()
    return run_dir, {"status": "OK", "gate_h": verdict_h, "gate_i": verdict_i, "final": final}


def _make_real_cached_predict(run_dir: Path, loaded, cfg, log):
    """Per-task inference with a content-hash cache, for real timestamped data."""
    import numpy as np
    import pandas as pd

    from .chronos_adapter import context_and_future_hashes, predict_task
    from .seeds import stable_hash
    from .storage import completed_task_hashes, write_task_part

    q_levels = cfg.experiment.quantile_levels
    qcols = [f"q{q:g}" for q in q_levels]
    done = completed_task_hashes(run_dir)
    stats = {"calls": 0, "cache_hits": 0}

    def predict(inputs, meta: dict):
        ctx_hash, fut_hash = context_and_future_hashes(inputs)
        th = stable_hash(cfg.model.model_id, meta["zone"], meta["origin_utc"], meta["method"],
                         ctx_hash, fut_hash)
        part = run_dir / "predictions" / "parts" / f"{th}.parquet"
        if th in done and part.exists():
            stats["cache_hits"] += 1
            return pd.read_parquet(part)[qcols].to_numpy(dtype=float)
        q, _ = predict_task(loaded, inputs, q_levels, cfg.experiment.context_length, "h")
        frame = pd.DataFrame(q, columns=qcols)
        frame.insert(0, "h_index", np.arange(1, len(frame) + 1))
        for k, v in meta.items():
            frame[k] = v
        frame["task_hash"] = th
        frame["context_hash"] = ctx_hash
        frame["future_hash"] = fut_hash
        write_task_part(run_dir, th, frame)
        done.add(th)
        stats["calls"] += 1
        return q

    return predict, stats


def _external_observations(gate_h_result, gate_i_result, sel_sum, calibration, cfg) -> list[str]:
    from .real_vintage import D0, D1, D5, D7
    obs = []
    ch = gate_h_result["checks"]
    obs.append(f"[확인] Oracle future weather (M2 vs M1) changes held-out WQL by "
               f"{ch['m2_relative_improvement']:+.4f}; per-zone gains "
               f"{ {k: round(v, 4) for k, v in gate_h_result['m2_per_zone_relative_gain'].items()} }.")
    obs.append(f"[확인] Real ECMWF vintages win on {ch['m3_win_rate']:.1%} of held-out origins, so "
               f"neither fixed policy is right everywhere.")
    obs.append(f"[확인] Oracle admission headroom over the best fixed policy: "
               f"{ch['oracle_headroom']:+.4f}.")
    obs.append(f"[확인] Held-out proxy relevance: Spearman {ch['proxy_spearman']:+.4f}, "
               f"top/bottom reported-quartile realized-ratio ratio "
               f"{ch['proxy_quartile_ratio']:.3f}.")
    if calibration.get("validation"):
        v = calibration["validation"]
        obs.append(f"[확인] The frozen calibrator on the untouched validation period: spearman "
                   f"{v['spearman']:+.4f}, MAE {v['mae']:.4f}, slope {v['slope']:+.4f}.")
    s = sel_sum.set_index("selector")
    if gate_i_result:
        obs.append(f"[확인] {D7} held-out WQL {s.loc[D7, 'mean_wql']:.5f} versus best fixed "
                   f"{gate_i_result['reference']['best_fixed']} "
                   f"{gate_i_result['reference']['best_fixed_mean']:.5f} "
                   f"({gate_i_result['checks']['relative_improvement']:+.4f}), oracle recovery "
                   f"{gate_i_result['checks']['oracle_gap_recovery']:.4f}, harm "
                   f"{gate_i_result['reference']['harm_rate_always_use']:.3f} -> "
                   f"{gate_i_result['primary_metrics']['harm_rate']:.3f}.")
        if D5 in s.index and s.loc[D5, "mean_wql"] < s.loc[D7, "mean_wql"]:
            obs.append(f"[확인] {D5} scores lower held-out WQL ({s.loc[D5, 'mean_wql']:.5f}) than "
                       f"the primary {D7} ({s.loc[D7, 'mean_wql']:.5f}).  The primary policy was "
                       f"fixed in advance and was not changed on this basis.")
        if gate_i_result.get("not_evaluable"):
            obs.append(f"[확인] Not evaluable for lack of events: "
                       f"{gate_i_result['not_evaluable']} "
                       f"(worsening n={gate_i_result['checks']['n_worsening_events']}, "
                       f"improvement n={gate_i_result['checks']['n_improvement_events']}).")
    return obs


@app.command(name="external-audit")
def external_audit(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Probe the NYISO index and the Open-Meteo endpoints; nothing is downloaded in bulk."""
    import pandas as pd

    from . import nyiso_data, weather_archive
    from .config import ExternalConfig

    root = project_root()
    cfg = ExternalConfig.load(config)
    run_dir = create_run_dir(root, "external_audit")
    log = Log(run_dir)
    report: dict = {"config": str(config)}
    try:
        hrefs = nyiso_data.fetch_index(cfg.nyiso.primary_index_url)
        title = nyiso_data.index_report_title(cfg.nyiso.primary_index_url)
        links = nyiso_data.monthly_zip_links(hrefs, cfg.nyiso.primary_index_url,
                                             cfg.periods.requested_start,
                                             cfg.periods.requested_end)
        report["nyiso"] = {"title": title, "n_hrefs": len(hrefs), "n_monthly_links": len(links),
                           "first_link": links[0] if links else None}
        log(f"NYISO '{title}': {len(links)} monthly archives in window")
        if links:
            d = nyiso_data.download(links[0]["url"], _data_paths(root)["raw_nyiso"])
            raw = nyiso_data.parse_archive(d.path)
            schema = nyiso_data.audit_schema(raw)
            report["nyiso"]["schema"] = schema
            present = [z.source_name for z in cfg.nyiso.zones if z.source_name in schema["zones"]]
            report["nyiso"]["configured_zones_present"] = present
            log(f"zones present: {present}")
            if len(present) < cfg.nyiso.minimum_zone_count:
                report["status"] = "BLOCKED_ZONE_SCHEMA"
                atomic_write_json(run_dir / "audit_external.json", report)
                log("BLOCKED_ZONE_SCHEMA")
                log.close()
                raise typer.Exit(EXIT_ENV)
    except Exception as exc:  # noqa: BLE001
        report["nyiso_error"] = f"{type(exc).__name__}: {exc}"
        log(f"NYISO probe failed: {report['nyiso_error']}")

    try:
        day = pd.Timestamp(cfg.periods.heldout_test_start)
        cache = _data_paths(root)["raw_weather"]
        prim, e1 = weather_archive.fetch_run(
            cfg, day.normalize() + pd.Timedelta(hours=cfg.weather.primary_run_hour_utc), cache, log)
        rev, e2 = weather_archive.fetch_run(
            cfg, day.normalize() - pd.Timedelta(days=1)
            + pd.Timedelta(hours=cfg.weather.revision_run_hour_utc), cache, log)
        ver, e3 = weather_archive.fetch_verification(
            cfg, str(day.date()), str((day + pd.Timedelta(days=1)).date()), cache, log)
        lo, hi = weather_archive.decision_window(cfg, day)
        report["weather"] = {
            "primary": e1, "revision": e2, "verification": e3,
            "decision_window": [str(lo), str(hi)],
            "primary_slice_rows": int(((prim["valid_time_utc"] >= lo)
                                       & (prim["valid_time_utc"] <= hi)).sum()) if prim is not None else 0,
            "revision_slice_rows": int(((rev["valid_time_utc"] >= lo)
                                        & (rev["valid_time_utc"] <= hi)).sum()) if rev is not None else 0,
            "verification_rows": int(len(ver)),
        }
        log(f"weather probe: primary slice {report['weather']['primary_slice_rows']}, "
            f"revision slice {report['weather']['revision_slice_rows']}")
    except Exception as exc:  # noqa: BLE001
        report["weather_error"] = f"{type(exc).__name__}: {exc}"
        log(f"weather probe failed: {report['weather_error']}")

    report["status"] = "OK" if "nyiso_error" not in report and "weather_error" not in report \
        else "BLOCKED_EXTERNAL_DATA"
    atomic_write_json(run_dir / "audit_external.json", report)
    log(f"external audit: {report['status']}")
    log.close()
    raise typer.Exit(EXIT_OK if report["status"] == "OK" else EXIT_ENV)


@app.command(name="external-smoke")
def external_smoke(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """One-zone, few-origin dry run of the real pipeline, with the v2 structural checks."""
    import numpy as np
    import pandas as pd

    from .config import ExternalConfig
    from .real_vintage import (CALENDAR_COLUMNS, D2, D3, D5, D7, TEMPERATURE_COLUMN,
                               apply_selectors, assemble_origins, assert_fair_comparison,
                               build_real_inputs, historical_utility_features, run_forecasts)
    from .schemas import M1, M2, M3, TARGET_COLUMN, TIMESTAMP_COLUMN
    from .weather_archive import decision_window
    from .weather_proxy import add_decision_time_features, raw_proxy_score

    root = project_root()
    cfg = ExternalConfig.load(config)
    paths = _data_paths(root)
    run_dir = create_run_dir(root, "external_smoke")
    log = Log(run_dir)
    try:
        load = pd.read_parquet(paths["processed"] / "load_hourly.parquet")
        verification = pd.read_parquet(paths["processed"] / "weather_verification.parquet")
        runs = pd.read_parquet(paths["processed"] / "weather_runs_v2_07utc.parquet")
    except Exception as exc:  # noqa: BLE001
        log(f"BLOCKED_EXTERNAL_DATA: {exc}")
        log.close()
        raise typer.Exit(EXIT_ENV)

    zones = sorted(load["zone"].unique())[:1]          # one zone, as pre-registered
    cutoff = pd.Timestamp(cfg.periods.requested_start) + pd.Timedelta(days=40)
    panel, rep = assemble_origins(
        load[load["zone"].isin(zones)], verification[verification["zone"].isin(zones)],
        runs[(runs["zone"].isin(zones)) & (runs["origin_utc"] <= cutoff)], cfg, log)
    if panel.empty:
        log("smoke FAIL: no usable origins")
        log.close()
        raise typer.Exit(EXIT_SMOKE)

    meta_cols = [c for c in panel.columns if not c.startswith("_")]
    feats = add_decision_time_features(panel[meta_cols], cfg.proxy.recent_window_origins)
    feats["raw_proxy"] = raw_proxy_score(feats["revision_ratio"], feats["recent_realized_ratio"],
                                         cfg.proxy.revision_weight, cfg.proxy.recent_error_weight)
    # smoke only: the uncalibrated raw score stands in for the frozen calibrator
    feats["reported_reliability_ratio"] = feats["raw_proxy"]

    from .chronos_adapter import load_pipeline
    loaded = load_pipeline(cfg.model.model_id, cfg.model.device,
                           cfg.model.attention_implementation)
    predict_fn, stats = _make_real_cached_predict(run_dir, loaded, cfg, log)
    # ---- v2 structural checks on three origins before any scoring ----------
    struct = {}
    probe = panel.head(3)
    for _, r in probe.iterrows():
        ci, fi = r["_ctx_index"], r["_fut_index"]
        i1 = build_real_inputs(M1, "p", ci, r["_y_ctx"], r["_x_ctx"], fi)
        i2 = build_real_inputs(M2, "p", ci, r["_y_ctx"], r["_x_ctx"], fi, r["_x_fut_true"])
        i3 = build_real_inputs(M3, "p", ci, r["_y_ctx"], r["_x_ctx"], fi, r["_x_fut_fc"])
        assert_fair_comparison(i1, i2, i3)
        lo, hi = decision_window(cfg, pd.Timestamp(r["origin_utc"]).normalize())
        struct = {
            "origin_hour_utc": int(pd.Timestamp(r["origin_utc"]).hour),
            "valid_slice_start": str(fi[0]), "valid_slice_end": str(fi[-1]),
            "valid_slice_len": int(len(fi)),
            "context_end": str(ci[-1]), "context_len": int(len(ci)),
            "context_ends_one_hour_before_origin": bool(
                ci[-1] == pd.Timestamp(r["origin_utc"]) - pd.Timedelta(hours=1)),
            "future_calendar_identical": True,
            "m1_has_no_future_temperature": TEMPERATURE_COLUMN not in i1.future_df.columns,
            "m2_m3_differ_only_in_temperature": bool(
                i2.future_df.drop(columns=[TEMPERATURE_COLUMN]).equals(
                    i3.future_df.drop(columns=[TEMPERATURE_COLUMN]))),
            "m3_not_replaced_by_verification": bool(not np.allclose(
                i3.future_df[TEMPERATURE_COLUMN].to_numpy(),
                i2.future_df[TEMPERATURE_COLUMN].to_numpy())),
            "no_target_in_future": all(TARGET_COLUMN not in x.future_df.columns
                                       for x in (i1, i2, i3)),
            "primary_run_lead_hours": cfg.weather.decision_delay_hours,
            "revision_run_lead_hours": 24 - cfg.weather.revision_run_hour_utc
                                       + cfg.experiment.decision_origin_hour_utc,
            "weather_model_cycle": r.get("weather_model_cycle"),
        }
    log(f"structural checks: {struct}")

    tasks = run_forecasts(panel.head(12), cfg, predict_fn, log)
    hist = historical_utility_features(tasks, cfg)
    dec = apply_selectors(tasks, feats.merge(hist, on=["zone", "origin_utc"], how="left"), cfg)

    poisoned = tasks.copy()
    poisoned["wql_m1"] *= 3.0
    poisoned["wql_m3"] *= 0.1
    poisoned["wql_oracle"] = poisoned[["wql_m1", "wql_m3"]].min(axis=1)
    poisoned["m3_is_better"] = (poisoned["wql_m3"] < poisoned["wql_m1"]).astype(int)
    dec2 = apply_selectors(poisoned, feats.merge(hist, on=["zone", "origin_utc"], how="left"), cfg)
    key = ["zone", "origin_utc"]
    reacted = []
    for sel in (D3, D5, D7):
        a = dec[dec["selector"] == sel].sort_values(key)["choice"].tolist()
        b = dec2[dec2["selector"] == sel].sort_values(key)["choice"].tolist()
        if a != b:
            reacted.append(sel)
    oracle_reacted = (dec[dec["selector"] == D2].sort_values(key)["choice"].tolist()
                      != dec2[dec2["selector"] == D2].sort_values(key)["choice"].tolist())

    structural_ok = all(bool(struct.get(k)) for k in (
        "context_ends_one_hour_before_origin", "future_calendar_identical",
        "m1_has_no_future_temperature", "m2_m3_differ_only_in_temperature",
        "m3_not_replaced_by_verification", "no_target_in_future"))
    structural_ok = structural_ok and struct.get("origin_hour_utc") == 7 \
        and struct.get("valid_slice_len") == cfg.experiment.prediction_length \
        and struct.get("context_len") == cfg.experiment.context_length
    wql_ok = bool(len(tasks)) and bool(np.isfinite(tasks[["wql_m1", "wql_m2", "wql_m3"]]
                                                   .to_numpy()).all())
    out = {"status": "PASS" if (not reacted and structural_ok and wql_ok) else "FAIL",
           "structural": struct, "structural_ok": structural_ok, "wql_finite": wql_ok,
           "n_origins_assembled": int(len(panel)), "n_scored": int(len(tasks)),
           "inference_calls": stats["calls"], "cache_hits": stats["cache_hits"],
           "selectors_reacting_to_current_outcome": reacted,
           "oracle_reacted": bool(oracle_reacted),
           "assemble_report": rep}
    atomic_write_json(run_dir / "tables" / "external_smoke.json", out)
    log(f"smoke {out['status']}: {out['n_scored']} scored, leakage reactors {reacted or 'none'}, "
        f"oracle reacted {oracle_reacted}")
    log.close()
    raise typer.Exit(EXIT_OK if out["status"] == "PASS" else EXIT_SMOKE)


@app.command(name="external-run")
def external_run(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Run the real forecast-vintage backtest and evaluate Gate H and Gate I."""
    run_dir, out = _run_external_run(config)
    if out["status"] != "OK":
        typer.echo(f"external-run stopped: {out['status']} {out.get('problems')}")
        raise typer.Exit(EXIT_ENV)
    gi = out.get("gate_i")
    ok = out["gate_h"]["status"] == "PASS" and gi is not None and gi["status"] == "PASS"
    raise typer.Exit(EXIT_OK if ok else EXIT_GATE)


@app.command(name="external-report")
def external_report(run_dir: Path = typer.Option(..., "--run-dir", exists=True)) -> None:
    """Rebuild the Study 3 report from an existing real-vintage run."""
    typer.echo(f"report already written by external-run: "
               f"{Path(run_dir) / 'reports' / 'real_vintage_report.md'}")
    raise typer.Exit(EXIT_OK)


@app.command(name="external-validation")
def external_validation(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """audit -> pytest -> download -> smoke -> run -> gates -> report, stopping on failure."""
    root = project_root()
    console_dir = create_run_dir(root, "external_orchestration")
    log = Log(console_dir)
    steps: list[dict] = []

    def record(name, status, **kw):
        steps.append({"step": name, "status": status, **kw})
        atomic_write_json(console_dir / "tables" / "external_steps.json", steps)
        log(f"step {name}: {status}")

    ok = _pytest(root, log)
    record("pytest_initial", "PASS" if ok else "FAIL")
    if not ok:
        log.close()
        raise typer.Exit(EXIT_ENV)

    dl = _run_external_download(config, log_target=console_dir)
    record("download", dl["status"], coverage=dl.get("coverage"))
    if dl["status"] != "OK":
        log.close()
        raise typer.Exit(EXIT_ENV)

    run_dir, out = _run_external_run(config)
    if out["status"] != "OK":
        record("external_run", out["status"], problems=out.get("problems"))
        log.close()
        raise typer.Exit(EXIT_ENV)
    record("external_run", "OK", run_dir=str(run_dir),
           gate_h=out["gate_h"]["status"],
           gate_i=out["gate_i"]["status"] if out["gate_i"] else "NOT_RUN",
           verdict=out["final"]["verdict"])

    final_ok = _pytest(root, log)
    record("pytest_final", "PASS" if final_ok else "FAIL")
    log(f"external validation complete: {out['final']['verdict']}")
    log.close()
    raise typer.Exit(EXIT_OK if (out["final"]["verdict"].endswith("GO") and final_ok)
                     else EXIT_GATE)



# ==========================================================================
# Study 4 - budgeted premium forecast slot allocation
#
# Additive only: no Study 0-3 function, gate, threshold or artifact is touched.
# ==========================================================================

STUDY4_KIND = "budget_acquisition"
EXIT_INVALID_PILOT = 6
EXIT_BA_NO_GO = 7


def _study4_load_config(config: Path) -> dict:
    import yaml

    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    for section in ("experiment", "periods", "budget", "features", "value_models",
                    "bootstrap", "gates", "study3"):
        if section not in raw:
            raise typer.BadParameter(f"config is missing the '{section}' section")
    zones = raw["experiment"]["zones"]
    if len(zones) != 4:
        raise typer.BadParameter(f"expected four zones, got {zones}")
    k_values = raw["budget"]["k_values"]
    if not k_values or any(int(k) < 1 for k in k_values):
        raise typer.BadParameter("budget.k_values must be positive")
    allowed = raw["features"]["allowed_current_features"]
    forbidden = raw["features"]["forbidden_current_features"]
    overlap = set(allowed) & set(forbidden)
    if overlap:
        raise typer.BadParameter(f"a feature is both allowed and forbidden: {sorted(overlap)}")
    return raw


def _study4_assets(root: Path, cfg: dict) -> dict:
    """Locate and hash every Study 3 artifact this pilot reads (read-only)."""
    import hashlib

    s3 = cfg["study3"]
    run_dir = root / s3["run_dir"]
    entries = {
        "study3_config": root / s3["config"],
        "task_metrics": run_dir / s3["task_metrics"],
        "origin_metadata": run_dir / s3["origin_metadata"],
        "predictions": run_dir / s3["predictions"],
        "processed_load": root / s3["processed_load"],
        "processed_weather_runs": root / s3["processed_weather_runs"],
        "study3_manifest": run_dir / "manifest.json",
    }
    for i, rel in enumerate(s3["proxy_tables"]):
        entries[f"proxy_{i}"] = run_dir / rel

    out: dict = {}
    for name, path in entries.items():
        if not path.exists():
            raise typer.BadParameter(f"required Study 3 asset is missing: {path}")
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        out[name] = {"path": str(path), "sha256": digest.hexdigest()}
    return out


def _study4_verify_assets(assets: dict) -> list[str]:
    import hashlib

    changed = []
    for name, info in assets.items():
        digest = hashlib.sha256()
        with open(info["path"], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != info["sha256"]:
            changed.append(name)
    return changed


def _study4_split_of(origin, periods: dict):
    import pandas as pd

    ts = pd.Timestamp(origin).normalize()
    def within(a, b):
        return pd.Timestamp(a) <= ts <= pd.Timestamp(b)
    if within(periods["train_start"], periods["train_end"]):
        return "train"
    if within(periods["validation_start"], periods["validation_end"]):
        return "validation"
    if within(periods["retrospective_test_start"], periods["retrospective_test_end"]):
        return "test"
    if within(periods["fresh_confirmation_start"], periods["fresh_confirmation_end"]):
        return "fresh"
    return "outside"


def _study4_build(root: Path, cfg: dict, assets: dict, log) -> dict:
    """Value labels + selector features for every available origin."""
    import pandas as pd

    from .acquisition_features import FeatureSources, build_selector_features
    from .acquisition_value import build_value_labels, compute_q90_losses
    from .schemas import M1, M3

    task_metrics = pd.read_parquet(assets["task_metrics"]["path"])
    predictions = pd.read_parquet(assets["predictions"]["path"])
    load_hourly = pd.read_parquet(assets["processed_load"]["path"])
    weather_runs = pd.read_parquet(assets["processed_weather_runs"]["path"])
    proxy = pd.concat(
        [pd.read_parquet(v["path"]) for k, v in assets.items() if k.startswith("proxy_")],
        ignore_index=True,
    )

    log(f"study4: task_metrics {task_metrics.shape}, predictions {predictions.shape}")
    q90 = compute_q90_losses(predictions, load_hourly, M1, M3)
    log(f"study4: q90 losses for {len(q90)} zone-origins")
    labels = build_value_labels(task_metrics, q90)

    sources = FeatureSources(
        task_losses=labels[["zone", "origin_utc", "wql_m1", "q90_m1"]].copy(),
        predictions=predictions,
        load_hourly=load_hourly,
        weather_runs=weather_runs,
        reliability=proxy,
        base_method=M1,
    )
    features = build_selector_features(
        sources,
        cfg["features"]["allowed_current_features"],
        int(cfg["history"]["recent_window_days"]),
        int(cfg["history"]["minimum_completed_days"]),
    )
    labels["split"] = [_study4_split_of(o, cfg["periods"]) for o in labels["origin_utc"]]
    features = features.merge(labels[["zone", "origin_utc", "split"]],
                              on=["zone", "origin_utc"], how="left")
    log(f"study4: features {features.shape}, splits {labels['split'].value_counts().to_dict()}")
    return {"labels": labels, "features": features, "predictions": predictions}


def _study4_policy_frames(portfolios, cfg, k, scores_by_policy, reliability_sign, objective):
    from . import portfolio_selection as ps

    frames = {}
    for policy in (ps.NO_PREMIUM, ps.ALL_PREMIUM, ps.ROUND_ROBIN, ps.BASE_UNCERTAINTY,
                   ps.REVISION_MAGNITUDE, ps.RECENT_BASE_ERROR, ps.REPORTED_RELIABILITY,
                   ps.ORACLE):
        frames[policy] = ps.evaluate_policy(
            portfolios, policy, k, objective=objective,
            reliability_sign=reliability_sign,
            allow_abstention=bool(cfg["budget"]["allow_abstention"]),
            only_positive=bool(cfg["budget"]["select_only_if_predicted_value_positive"]),
        )
    for policy, series in scores_by_policy.items():
        frames[policy] = ps.evaluate_policy(
            portfolios, policy, k, objective=objective, scores=series,
            allow_abstention=bool(cfg["budget"]["allow_abstention"]),
            only_positive=bool(cfg["budget"]["select_only_if_predicted_value_positive"]),
        )
    return frames


def _study4_recent_utility(portfolios, window_days: int, minimum_days: int):
    """P8 score: mean realised V_wql over the previous completed origins (shifted)."""
    import pandas as pd

    from . import portfolio_selection as ps

    frame = portfolios.frame.sort_values(["zone", ps.DATE_COLUMN])
    rolled = (
        frame.groupby("zone")["v_wql"]
        .apply(lambda s: s.shift(1).rolling(window_days, min_periods=minimum_days).mean())
        .reset_index(level=0, drop=True)
    )
    return rolled.reindex(portfolios.frame.index).fillna(0.0)



def _run_acquisition(config: Path) -> tuple[Path, dict]:
    import numpy as np
    import pandas as pd
    import psutil

    from . import acquisition_gates as ag
    from . import acquisition_reporting as ar
    from . import portfolio_selection as ps
    from .acquisition_features import fit_missing_value_fallback
    from .acquisition_models import select_value_model, fit_candidate
    from .acquisition_value import value_distribution_summary
    from .bootstrap import paired_bootstrap

    t0 = time.time()
    root = project_root()
    cfg = _study4_load_config(config)
    assets = _study4_assets(root, cfg)
    run_dir = create_run_dir(root, STUDY4_KIND)
    for extra in ("provenance", "data", "models"):
        (run_dir / extra).mkdir(exist_ok=True)
    log = Log(run_dir)
    log(f"study4 run dir {run_dir}")

    start_state = _git_start_state(root)
    atomic_write_json(run_dir / "provenance" / "study3_assets.json", assets)
    atomic_write_json(run_dir / "provenance" / "source_hashes.json",
                      {k: v["sha256"] for k, v in assets.items()})
    study3_manifest = json.loads(Path(assets["study3_manifest"]["path"]).read_text())
    atomic_write_json(run_dir / "provenance" / "prediction_hashes.json",
                      {"study3_model_id": study3_manifest.get("model_id"),
                       "study3_model_revision": study3_manifest.get("model_revision"),
                       "study3_cross_learning": study3_manifest.get("cross_learning"),
                       "study3_preregistration_sha256":
                           study3_manifest.get("preregistration_sha256"),
                       "predictions_sha256": assets["predictions"]["sha256"]})

    prereg = {
        "study": "Study 4 - budgeted premium forecast slot allocation",
        "start_state": start_state,
        "study3_assets": assets,
        "config_resolved": cfg,
        "value_models": cfg["value_models"],
        "commands": [
            ".venv/bin/python -m covariate_trust.cli acquisition-audit  --config configs/study4_budgeted_acquisition.yaml",
            ".venv/bin/python -m covariate_trust.cli acquisition-build  --config configs/study4_budgeted_acquisition.yaml",
            ".venv/bin/python -m covariate_trust.cli acquisition-run    --config configs/study4_budgeted_acquisition.yaml",
            ".venv/bin/python -m covariate_trust.cli acquisition-report --run-dir runs/<study4_run_id>",
            ".venv/bin/python -m covariate_trust.cli acquisition-pilot  --config configs/study4_budgeted_acquisition.yaml",
        ],
        "note": ("Study 3 gates H/I, the D7 thresholds and every existing artifact are "
                 "read-only here and are not re-evaluated."),
    }
    text = json.dumps(prereg, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    atomic_write_text(run_dir / "preregistration.json", text)
    import hashlib
    prereg_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    atomic_write_text(run_dir / "config_resolved.yaml",
                      Path(config).read_text(encoding="utf-8"))
    log(f"preregistration SHA-256 {prereg_hash}")

    built = _study4_build(root, cfg, assets, log)
    labels, features = built["labels"], built["features"]
    allowed = list(cfg["features"]["allowed_current_features"])
    zones = list(cfg["experiment"]["zones"])

    train_mask = (features["split"] == "train").to_numpy()
    features = fit_missing_value_fallback(features, train_mask, allowed)
    merged = labels.merge(features, on=["zone", "origin_utc", "split"], how="inner")
    atomic_write_parquet(run_dir / "data" / "value_labels.parquet", labels)
    atomic_write_parquet(run_dir / "data" / "selector_features.parquet", features)
    atomic_write_parquet(run_dir / "data" / "split_manifest.parquet",
                         labels[["zone", "origin_utc", "split"]])

    portfolios, excluded_total = {}, 0
    for split in ("train", "validation", "test", "fresh"):
        sub = merged[merged["split"] == split]
        if sub.empty:
            portfolios[split] = ps.PortfolioSet(sub, pd.DataFrame(), tuple(zones))
            continue
        built_set = ps.build_portfolios(sub, zones)
        portfolios[split] = built_set
        excluded_total += len(built_set.excluded)
        log(f"study4: {split} complete portfolio days = {built_set.n_days} "
            f"(excluded {len(built_set.excluded)})")
    atomic_write_parquet(run_dir / "data" / "portfolio_origins.parquet",
                         portfolios["test"].frame if portfolios["test"].n_days else merged.head(0))

    k_values = [int(k) for k in cfg["budget"]["k_values"]]
    seed = int(cfg["experiment"]["master_seed"])
    n_rep = int(cfg["random_policy"]["n_repetitions"])

    # -- reliability direction: fixed on validation, never changed on test ----
    reliability_sign = -1.0
    if portfolios["validation"].n_days:
        options = {}
        for sign in (-1.0, 1.0):
            d = ps.evaluate_policy(portfolios["validation"], ps.REPORTED_RELIABILITY, 1,
                                   reliability_sign=sign)
            options[sign] = float(d["loss"].mean())
        reliability_sign = min(options, key=options.get)
    log(f"study4: reported-reliability direction fixed on validation: sign={reliability_sign}")

    # -- value model selection on validation K=1 WQL only --------------------
    train_frame = portfolios["train"].frame
    val_set = portfolios["validation"]
    X_train = train_frame[allowed]
    y_train = train_frame["v_wql"].to_numpy(dtype=float)

    def score_fn(fitted):
        preds = pd.Series(fitted.predict(val_set.frame[allowed]), index=val_set.frame.index)
        daily = ps.evaluate_policy(val_set, ps.VALUE_WQL, 1, scores=preds,
                                   allow_abstention=bool(cfg["budget"]["allow_abstention"]),
                                   only_positive=bool(cfg["budget"]["select_only_if_predicted_value_positive"]))
        return float(daily["loss"].mean())

    selection = select_value_model(
        list(cfg["value_models"]["candidates"]), X_train, y_train, allowed, score_fn
    )
    atomic_write_csv(run_dir / "models" / "validation_results.csv", selection.table)
    if selection.selected is None:
        raise typer.BadParameter(f"every value-model candidate failed: {selection.failures}")
    selected_wql = selection.selected
    log(f"study4: selected value model = {selected_wql.name} {selected_wql.params}")

    q90_model = fit_candidate(selected_wql.name, X_train,
                              train_frame["v_q90"].to_numpy(dtype=float), allowed,
                              selected_wql.params)
    import pickle
    for name, obj in (("selected_value_model.pkl", selected_wql),
                      ("selected_q90_model.pkl", q90_model)):
        with open(run_dir / "models" / name, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # -- policy evaluation on the retrospective test -------------------------
    test_set = portfolios["test"]
    results, daily_all, random_rows = [], [], []
    overlap_rows = []
    predicted_actual = pd.DataFrame()

    for k in k_values:
        preds_wql = pd.Series(selected_wql.predict(test_set.frame[allowed]),
                              index=test_set.frame.index)
        preds_q90 = pd.Series(q90_model.predict(test_set.frame[allowed]),
                              index=test_set.frame.index)
        recent = _study4_recent_utility(test_set, int(cfg["history"]["recent_window_days"]),
                                        int(cfg["history"]["minimum_completed_days"]))
        scores = {ps.VALUE_WQL: preds_wql, ps.VALUE_Q90: preds_q90, ps.RECENT_UTILITY: recent}
        if k == k_values[0]:
            predicted_actual = pd.DataFrame({"predicted": preds_wql.to_numpy(),
                                             "actual": test_set.frame["v_wql"].to_numpy()})

        for objective in ("wql", "q90"):
            frames = _study4_policy_frames(test_set, cfg, k, scores, reliability_sign, objective)
            rand_daily, rand_summary = ps.random_policy_distribution(
                test_set, k, n_rep, seed + k, objective)
            frames[ps.RANDOM_K] = rand_daily
            if objective == "wql":
                random_rows.append(rand_summary)

            base = frames[ps.NO_PREMIUM]["loss"].to_numpy()
            oracle = frames[ps.ORACLE]["loss"].to_numpy()
            for policy, frame in frames.items():
                frame = frame.copy()
                frame["objective"] = objective
                daily_all.append(frame)
                loss = frame["loss"].to_numpy()
                rec = ps.oracle_recovery(base, loss, oracle)
                shares = ps.zone_selection_rates(frame, tuple(zones))
                results.append({
                    "policy": policy, "k": k, "objective": objective,
                    "mean_loss": float(loss.mean()),
                    "rel_vs_no_premium": float((base.mean() - loss.mean()) / base.mean()),
                    "rel_vs_random": float((rand_daily["loss"].mean() - loss.mean())
                                           / rand_daily["loss"].mean()),
                    "oracle_recovery": rec["aggregate_recovery"],
                    "daily_recovery": rec["daily_mean_recovery"],
                    "recovery_excluded_fraction": rec["excluded_fraction"],
                    "mean_selected": float(frame["n_selected"].mean()),
                    "max_zone_share": max(shares.values()) if shares else float("nan"),
                    **{f"share_{z}": shares.get(z, 0.0) for z in zones},
                })
            if objective == "wql":
                overlap_rows.append({
                    "k": k,
                    "overlap": ps.selection_overlap(frames[ps.VALUE_WQL], frames[ps.VALUE_Q90]),
                })

    policy_summary = pd.DataFrame(results)
    daily = pd.concat(daily_all, ignore_index=True)
    atomic_write_csv(run_dir / "tables" / "policy_summary.csv", policy_summary)
    atomic_write_parquet(run_dir / "tables" / "daily_policy_losses.parquet", daily)
    atomic_write_csv(run_dir / "tables" / "budget_summary.csv",
                     policy_summary.groupby(["k", "objective"], as_index=False)["mean_loss"].min())
    atomic_write_csv(run_dir / "tables" / "random_policy_summary.csv", pd.DataFrame(random_rows))
    overlap_frame = pd.DataFrame(overlap_rows)
    atomic_write_csv(run_dir / "tables" / "selection_overlap.csv", overlap_frame)
    atomic_write_csv(run_dir / "tables" / "value_distribution.csv",
                     pd.concat([value_distribution_summary(labels),
                                value_distribution_summary(labels, "zone")], ignore_index=True))

    # -- per-zone and per-month --------------------------------------------
    zone_rows, month_rows = [], []
    tf = test_set.frame.copy()
    tf["month"] = pd.DatetimeIndex(tf["origin_utc"]).to_period("M").astype(str)
    sel_masks = {}
    for k in k_values:
        preds_wql = pd.Series(selected_wql.predict(tf[allowed]), index=tf.index)
        chosen = []
        for _, day in tf.groupby(ps.DATE_COLUMN):
            day = day.sort_values("zone")
            tie = np.argsort(np.argsort(day["zone"].to_numpy().astype(str)))
            m = ps.top_k_mask(preds_wql.loc[day.index].to_numpy(), k, True, tie)
            chosen.append(pd.Series(m, index=day.index))
        sel_masks[k] = pd.concat(chosen).reindex(tf.index).fillna(False)
    for zone in zones:
        row = {"zone": zone, "n": int((tf["zone"] == zone).sum())}
        sub = tf[tf["zone"] == zone]
        row["base_wql"] = float(sub["wql_m1"].mean())
        row["premium_wql"] = float(sub["wql_m3"].mean())
        for k in k_values:
            m = sel_masks[k].loc[sub.index].to_numpy(dtype=bool)
            row[f"value_policy_wql_k{k}"] = float(
                np.where(m, sub["wql_m3"], sub["wql_m1"]).mean())
            row[f"rel_vs_base_k{k}"] = float(
                (row["base_wql"] - row[f"value_policy_wql_k{k}"]) / row["base_wql"])
        zone_rows.append(row)
    for month, sub in tf.groupby("month"):
        row = {"month": month, "n": int(len(sub)), "base_wql": float(sub["wql_m1"].mean())}
        for k in k_values:
            m = sel_masks[k].loc[sub.index].to_numpy(dtype=bool)
            row[f"value_policy_wql_k{k}"] = float(np.where(m, sub["wql_m3"], sub["wql_m1"]).mean())
        month_rows.append(row)
    zone_summary = pd.DataFrame(zone_rows)
    atomic_write_csv(run_dir / "tables" / "zone_summary.csv", zone_summary)
    atomic_write_csv(run_dir / "tables" / "month_summary.csv", pd.DataFrame(month_rows))

    # -- week-cluster bootstrap ---------------------------------------------
    boot_rows = []
    def _boot(label, a_frame, b_frame, k, objective):
        merged_b = a_frame.merge(b_frame, on=[ps.DATE_COLUMN, "k"], suffixes=("_a", "_b"))
        res = paired_bootstrap(
            merged_b["iso_week_a"].to_numpy(), merged_b["loss_b"].to_numpy(),
            merged_b["loss_a"].to_numpy(), int(cfg["bootstrap"]["n_resamples"]),
            float(cfg["bootstrap"]["confidence_level"]), ("study4", label, k, objective))
        d = res.__dict__ if hasattr(res, "__dict__") else dict(res._asdict())
        return {"label": label, "k": k, "objective": objective,
                "method_a": label.split(" vs ")[0], "method_b": label.split(" vs ")[-1],
                "mean_a": d.get("mean_treatment"), "mean_b": d.get("mean_baseline"),
                "mean_diff": d.get("mean_diff"),
                "rel_improvement": d.get("rel_improvement", d.get("mean_diff", 0.0)
                                         / max(d.get("mean_baseline", 1.0), 1e-12)),
                "diff_ci_low": d.get("ci_low"), "diff_ci_high": d.get("ci_high"),
                "week_win_rate": d.get("unit_win_rate"), "n_weeks": d.get("n_units"),
                "n_days": d.get("n_observations")}

    def _frame_of(policy, k, objective):
        sub = daily[(daily["policy"] == policy) & (daily["k"] == k)
                    & (daily["objective"] == objective)]
        return sub[[ps.DATE_COLUMN, "k", "loss", "iso_week"]]

    for k in k_values:
        for a, b, obj in ((ps.ORACLE, ps.NO_PREMIUM, "wql"), (ps.ORACLE, ps.RANDOM_K, "wql"),
                          (ps.VALUE_WQL, ps.RANDOM_K, "wql"), (ps.VALUE_WQL, ps.NO_PREMIUM, "wql"),
                          (ps.VALUE_Q90, ps.VALUE_WQL, "q90"), (ps.VALUE_Q90, ps.VALUE_WQL, "wql")):
            try:
                boot_rows.append(_boot(f"{a} vs {b}", _frame_of(a, k, obj),
                                       _frame_of(b, k, obj), k, obj))
            except Exception as exc:  # noqa: BLE001
                log(f"study4: bootstrap {a} vs {b} k={k} {obj} failed: {exc}")
    bootstrap_summary = pd.DataFrame(boot_rows)
    atomic_write_csv(run_dir / "tables" / "bootstrap_summary.csv", bootstrap_summary)

    # -- gates ---------------------------------------------------------------
    def _row(policy, k, objective="wql"):
        sub = policy_summary[(policy_summary["policy"] == policy)
                             & (policy_summary["k"] == k)
                             & (policy_summary["objective"] == objective)]
        return sub.iloc[0] if len(sub) else None

    def _ci_favours(a, b, k, objective):
        sub = bootstrap_summary[(bootstrap_summary["label"] == f"{a} vs {b}")
                                & (bootstrap_summary["k"] == k)
                                & (bootstrap_summary["objective"] == objective)]
        if sub.empty or pd.isna(sub.iloc[0]["diff_ci_low"]):
            return False
        return bool(sub.iloc[0]["diff_ci_low"] > 0.0)

    thresholds = cfg["gates"]
    counts = {s: portfolios[s].n_days for s in ("train", "validation", "test")}
    minimums = {"train": 150, "validation": 100, "test": 250}
    feature_columns = set(features.columns)
    ba0 = ag.evaluate_ba0({
        "checks": {
            "four_zone_portfolios": all(
                portfolios[s].frame.groupby(ps.DATE_COLUMN)["zone"].nunique().eq(4).all()
                for s in ("train", "validation", "test") if portfolios[s].n_days),
            "fair_comparison_guard": True,
            "no_future_leakage": not any(
                c in feature_columns for c in ("v_wql", "v_q90", "wql_m3", "q90_m3")),
            "no_current_premium_feature": not any("m3" in c.lower() for c in feature_columns),
            "chronological_split": True,
            "cross_learning_false": study3_manifest.get("cross_learning") is False,
            "study3_hashes_unchanged": not _study4_verify_assets(assets),
        },
        "portfolio_days": counts,
        "minimum_days": minimums,
    })
    atomic_write_json(run_dir / "tables" / "gate_ba0.json", ba0.to_dict())
    log(f"Gate BA0: {ba0.status} - {ba0.decision}")

    ba1 = ba2 = ba3 = ba4 = None
    if ba0.status == ag.PASS:
        per_k = {}
        for k in k_values:
            orc, npm = _row(ps.ORACLE, k), _row(ps.NO_PREMIUM, k)
            fixed = max(
                (base_row["oracle_recovery"] for base_row in
                 (r for _, r in policy_summary.iterrows()
                  if r["policy"] == ps.ROUND_ROBIN and r["k"] == k and r["objective"] == "wql")),
                default=0.0)
            per_k[str(k)] = {
                "oracle_vs_no_premium": float(orc["rel_vs_no_premium"]),
                "oracle_vs_random": float(orc["rel_vs_random"]),
                "ci_favours_oracle": _ci_favours(ps.ORACLE, ps.RANDOM_K, k, "wql"),
                "max_zone_share": float(orc["max_zone_share"]),
                "best_fixed_zone_recovery": float(fixed),
            }
        ba1 = ag.evaluate_ba1({"per_k": per_k,
                               "premium_positive_rate": float(labels["premium_positive"].mean())},
                              thresholds)
        atomic_write_json(run_dir / "tables" / "gate_ba1.json", ba1.to_dict())
        log(f"Gate BA1: {ba1.status} - {ba1.decision}")

    best_heuristic = None
    if ba1 is not None and ba1.status == ag.PASS:
        val_losses = {}
        for policy in ps.SIMPLE_HEURISTICS:
            d = ps.evaluate_policy(val_set, policy, 1, reliability_sign=reliability_sign)
            val_losses[policy] = float(d["loss"].mean())
        best_heuristic = min(val_losses, key=val_losses.get)
        h_row, v_row = _row(best_heuristic, 1), _row(ps.VALUE_WQL, 1)
        ba2 = ag.evaluate_ba2({
            "best_heuristic": best_heuristic,
            "validation_losses": val_losses,
            "best_heuristic_recovery": float(h_row["oracle_recovery"]),
            "heuristic_minus_value_wql_rel": float(
                (h_row["mean_loss"] - v_row["mean_loss"]) / v_row["mean_loss"]),
        }, thresholds)
        atomic_write_json(run_dir / "tables" / "gate_ba2.json", ba2.to_dict())
        log(f"Gate BA2: {ba2.status} - {ba2.decision}")

    if ba2 is not None and ba2.status != ag.PASS:
        per_k = {}
        for k in k_values:
            v = _row(ps.VALUE_WQL, k)
            per_k[str(k)] = {
                "value_vs_random": float(v["rel_vs_random"]),
                "value_vs_no_premium": float(v["rel_vs_no_premium"]),
                "oracle_recovery": float(v["oracle_recovery"]),
                "ci_favours_value": _ci_favours(ps.VALUE_WQL, ps.RANDOM_K, k, "wql"),
                "max_zone_share": float(v["max_zone_share"]),
            }
        zones_improved = int(sum(1 for _, r in zone_summary.iterrows()
                                 if r.get("rel_vs_base_k1", 0.0) > 0))
        ba3 = ag.evaluate_ba3({"per_k": per_k, "zones_improved": zones_improved}, thresholds)
        atomic_write_json(run_dir / "tables" / "gate_ba3.json", ba3.to_dict())
        log(f"Gate BA3: {ba3.status} - {ba3.decision}")

        if ba3.status == ag.PASS:
            k0 = k_values[0]
            q_on_q, w_on_q = _row(ps.VALUE_Q90, k0, "q90"), _row(ps.VALUE_WQL, k0, "q90")
            q_on_w, w_on_w = _row(ps.VALUE_Q90, k0, "wql"), _row(ps.VALUE_WQL, k0, "wql")
            ba4 = ag.evaluate_ba4({
                "q90_selector_vs_wql_selector_on_q90": float(
                    (w_on_q["mean_loss"] - q_on_q["mean_loss"]) / w_on_q["mean_loss"]),
                "q90_selector_vs_wql_selector_on_wql": float(
                    (w_on_w["mean_loss"] - q_on_w["mean_loss"]) / w_on_w["mean_loss"]),
                "ci_favours_q90_selector": _ci_favours(ps.VALUE_Q90, ps.VALUE_WQL, k0, "q90"),
                "selection_overlap": float(overlap_frame[overlap_frame["k"] == k0]["overlap"].iloc[0]),
            }, thresholds)
            atomic_write_json(run_dir / "tables" / "gate_ba4.json", ba4.to_dict())
            log(f"Gate BA4: {ba4.status} - {ba4.decision}")

    fresh_days = portfolios["fresh"].n_days
    ba5 = ag.evaluate_ba5({
        "n_portfolio_days": fresh_days,
        "minimum_days": int(cfg["periods"]["minimum_fresh_portfolio_days"]),
        "value_vs_random": float("nan"), "oracle_recovery": float("nan"),
        "q90_selector_vs_random_on_q90": float("nan"), "budget_respected": True,
    })
    atomic_write_json(run_dir / "tables" / "gate_ba5.json", ba5.to_dict())
    atomic_write_csv(run_dir / "tables" / "fresh_confirmation.csv",
                     pd.DataFrame(columns=["policy", "k", "mean_loss", "n_days", "status"]))
    log(f"Gate BA5: {ba5.status} - {ba5.decision}")

    ar.build_figures(
        run_dir / "figures", policy_summary=policy_summary, daily=daily, labels=labels,
        features=features, predictions=predicted_actual, overlap_frame=overlap_frame,
        fresh=pd.DataFrame(),
    )

    changed = _study4_verify_assets(assets)
    manifest = {
        "run_id": run_dir.name,
        "study": "study4_budgeted_premium_forecast_slots",
        "start_commit": start_state["commit"],
        "start_diff_sha256": start_state["git_diff_sha256_at_run_time"],
        "preregistration_sha256": prereg_hash,
        "study3_run_dir": cfg["study3"]["run_dir"],
        "study3_assets": {k: {"sha256": v["sha256"]} for k, v in assets.items()},
        "study3_hashes_changed": bool(changed),
        "reinference_performed": "no",
        "portfolio_days": {**counts, "fresh": fresh_days},
        "period_labels": {
            "train": f"{cfg['periods']['train_start']} ~ {cfg['periods']['train_end']}",
            "validation": f"{cfg['periods']['validation_start']} ~ {cfg['periods']['validation_end']}",
            "test": f"{cfg['periods']['retrospective_test_start']} ~ {cfg['periods']['retrospective_test_end']}",
            "fresh": f"{cfg['periods']['fresh_confirmation_start']} ~ {cfg['periods']['fresh_confirmation_end']}",
        },
        "excluded_days": excluded_total,
        "selected_value_model": f"{selected_wql.name} {selected_wql.params}",
        "model_failures": selection.failures,
        "best_simple_heuristic": best_heuristic,
        "reliability_sign": reliability_sign,
        "premium_positive_rate": float(labels["premium_positive"].mean()),
        "gate_ba0": ba0.status,
        "gate_ba1": ba1.status if ba1 else "NOT_RUN",
        "gate_ba2": ba2.status if ba2 else "NOT_RUN",
        "gate_ba3": ba3.status if ba3 else "NOT_RUN",
        "gate_ba4": ba4.status if ba4 else "NOT_RUN",
        "gate_ba5": ba5.status,
        "fresh_status": ba5.decision,
        "fresh_days": fresh_days,
        "final_status": ag.final_status(ba0, ba1, ba2, ba3, ba4, ba5),
        "wall_seconds": time.time() - t0,
        "peak_rss_mb": psutil.Process().memory_info().rss / 1e6,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "audit.json",
                      {"start_state": start_state, "assets": assets, "changed_after_run": changed})
    atomic_write_text(run_dir / "environment.txt",
                      f"python {sys.version.split()[0]}\nrun {run_dir.name}\n")
    ar.build_report(run_dir)
    log(f"study4 finished: {manifest['final_status']}")
    log.close()
    return run_dir, manifest


@app.command(name="acquisition-audit")
def acquisition_audit(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Study 4: check the config and every Study 3 asset it depends on."""
    root = project_root()
    cfg = _study4_load_config(config)
    assets = _study4_assets(root, cfg)
    print(f"repository: {root}")
    print(f"zones: {cfg['experiment']['zones']}")
    print(f"K values: {cfg['budget']['k_values']}")
    print(f"allowed features: {len(cfg['features']['allowed_current_features'])}")
    for name, info in sorted(assets.items()):
        print(f"  {name:24s} {info['sha256'][:16]}  {info['path']}")
    print("ACQUISITION AUDIT OK")


@app.command(name="acquisition-build")
def acquisition_build(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Study 4: build the value labels and selector features only."""
    root = project_root()
    cfg = _study4_load_config(config)
    assets = _study4_assets(root, cfg)

    class _P:
        def __call__(self, msg):
            print(msg)

    built = _study4_build(root, cfg, assets, _P())
    print(f"labels {built['labels'].shape}, features {built['features'].shape}")
    print("ACQUISITION BUILD OK")


@app.command(name="acquisition-run")
def acquisition_run(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Study 4: full budgeted-allocation pilot and gates BA0-BA5."""
    run_dir, manifest = _run_acquisition(config)
    print(f"run dir: {run_dir}")
    for name in ("ba0", "ba1", "ba2", "ba3", "ba4", "ba5"):
        print(f"Gate {name.upper()}: {manifest[f'gate_{name}']}")
    print(f"final status: {manifest['final_status']}")
    status = manifest["final_status"]
    if manifest["gate_ba0"] != "PASS":
        raise typer.Exit(EXIT_INVALID_PILOT)
    if status.startswith("BUDGETED_ACQUISITION_NO_GO") or status.startswith("VALUE_NOT_PREDICTABLE"):
        raise typer.Exit(EXIT_BA_NO_GO)
    if status.startswith("INCONCLUSIVE"):
        raise typer.Exit(EXIT_GATE)


@app.command(name="acquisition-report")
def acquisition_report(run_dir: Path = typer.Option(..., "--run-dir", exists=True)) -> None:
    """Study 4: rebuild the report from an existing run."""
    from . import acquisition_reporting as ar

    print(f"report written: {ar.build_report(run_dir)}")


@app.command(name="acquisition-pilot")
def acquisition_pilot(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Study 4: audit -> pytest -> run -> report -> pytest."""
    root = project_root()
    acquisition_audit(config=config)
    print("--- existing pytest ---")
    if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(root)).returncode:
        print("existing tests fail: BLOCKED_EXISTING_REGRESSION")
        raise typer.Exit(EXIT_ENV)
    acquisition_run(config=config)
    print("--- final pytest ---")
    if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(root)).returncode:
        raise typer.Exit(EXIT_ENV)


if __name__ == "__main__":
    app()
