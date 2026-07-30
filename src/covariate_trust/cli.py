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


if __name__ == "__main__":
    app()
