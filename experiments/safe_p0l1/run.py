"""Shrink P0L1's local correction toward the static mixture by a calibrated lambda.

P0L1 is untouched: same experts, same G-NOSCALE features, same MLP, same direct
mixture loss.  The only addition is one scalar per evaluation fold:

    g_safe = alpha + lambda * (g_raw - alpha),    0 <= lambda <= 1

lambda is not searched.  Under the direct loss the mixture error is quadratic in
the weight, so writing Y_safe = Y_alpha + lambda * Z gives a weighted
least-squares optimum in closed form.  In terms of the per-origin coefficients
this project already carries:

    Z_j        = (g_j - alpha) * (B - A)
    Y - Y_alpha= (Y - A) - alpha * (B - A)
    numerator  = (g_j - alpha) * (num_j - alpha * den_j) / n_j
    denominator= (g_j - alpha)^2 * den_j / n_j

so the weighting is exactly the direct loss's own mask and train-scale
normalisation, not a new definition.

The information boundary is the point of the exercise.  lambda for fold k comes
only from folds strictly earlier than k, and only from predictions those folds
produced while they were held out.  The first evaluation fold has no such
history and runs at lambda = 0, which is the static mixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..external_validity_screen import cli, prereg, screen
from ..gate_v3.diagnose import fit_alpha, mixture_mse
from ..p0l1_robustness.folds import DATASETS, OUT as P0L1_OUT
from ..p0l1_robustness.run import fit_direct, fold_table, freeze_spec
from ..structure_gate import features as F, gate as G
from ..structure_gate.convex_oracle import FLAT_THRESHOLD

OUT = screen.OUT.parent / "gate_safe_p0l1"
DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]
#: Reuses the project's existing numerical floor rather than inventing one.
DENOMINATOR_FLOOR = FLAT_THRESHOLD
#: Operational diagnostic threshold, fixed before any lambda was computed.
LAMBDA_RANGE_WARN = 0.8


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def calibrate(frames: list[pd.DataFrame]) -> dict:
    """Closed-form lambda from past cross-fitted folds; 0 when there is no signal."""
    if not frames:
        return {"lambda": 0.0, "lambda_raw": None, "n_calibration_origins": 0,
                "status": "COLD_START", "denominator": 0.0}
    block = pd.concat(frames, ignore_index=True)
    delta = block["g_raw"].to_numpy(np.float64) - block["alpha"].to_numpy(np.float64)
    n = block["n"].to_numpy(np.float64)
    numerator = float((delta * (block["num"].to_numpy(np.float64)
                                - block["alpha"].to_numpy(np.float64)
                                * block["den"].to_numpy(np.float64)) / n).sum())
    denominator = float(((delta ** 2) * block["den"].to_numpy(np.float64) / n).sum())
    if denominator <= DENOMINATOR_FLOOR:
        return {"lambda": 0.0, "lambda_raw": None, "n_calibration_origins": int(len(block)),
                "status": "SAFE_GATE_NO_CALIBRATION_SIGNAL", "denominator": denominator}
    raw = numerator / denominator
    return {"lambda": float(np.clip(raw, 0.0, 1.0)), "lambda_raw": float(raw),
            "n_calibration_origins": int(len(block)), "status": "CALIBRATED",
            "denominator": denominator}


def identity_tests() -> dict:
    """The safe gate must collapse to its endpoints exactly."""
    rng = np.random.default_rng(0)
    n = 64
    frame = pd.DataFrame({"n": rng.integers(5, 28, n).astype(float)})
    a = rng.normal(size=(n, 28)); b = a + rng.normal(size=(n, 28))
    y = a + rng.normal(size=(n, 28))
    d, r = b - a, y - a
    frame["num"] = (r * d).sum(1); frame["den"] = (d * d).sum(1)
    frame["sse_a"] = (r ** 2).sum(1)
    frame["n"] = 28.0
    g_raw = rng.uniform(size=n)
    alpha = 0.3
    checks = {}
    # alpha + 1*(g - alpha) is g only up to floating-point associativity, so the
    # endpoints are checked to a tight tolerance rather than bit equality.
    tolerance = 1e-12
    for lam, reference in ((0.0, np.full(n, alpha)), (1.0, g_raw)):
        g_safe = alpha + lam * (g_raw - alpha)
        weight_error = float(np.abs(g_safe - reference).max())
        loss_error = float(np.abs(mixture_mse(frame, g_safe)
                                  - mixture_mse(frame, reference)).max())
        checks[f"lambda_{lam}"] = {
            "max_abs_weight_error": weight_error, "max_abs_loss_error": loss_error,
            "tolerance": tolerance,
            "ok": bool(weight_error <= tolerance and loss_error <= tolerance)}
    same = np.full(n, alpha)
    checks["g_raw_equals_alpha"] = {
        "max_abs_error": float(max(np.abs(alpha + lam * (same - alpha) - same).max()
                                   for lam in (0.0, 0.37, 1.0))),
        "ok": bool(all(np.abs(alpha + lam * (same - alpha) - same).max() <= tolerance
                       for lam in (0.0, 0.37, 1.0)))}
    flat = frame.copy(); flat["den"] = 0.0; flat["num"] = 0.0
    checks["experts_identical_invariant"] = {
        "ok": bool(np.allclose(mixture_mse(flat, np.zeros(n)), mixture_mse(flat, np.ones(n))))}
    g_safe = alpha + 0.5 * (g_raw - alpha)
    checks["g_safe_in_unit_interval"] = {"min": float(g_safe.min()), "max": float(g_safe.max()),
                                         "ok": bool((g_safe >= 0).all() and (g_safe <= 1).all())}
    checks["no_signal_gives_zero"] = {
        "ok": bool(calibrate([flat.assign(g_raw=g_raw, alpha=alpha)])["lambda"] == 0.0)}
    checks["cold_start_gives_zero"] = {"ok": bool(calibrate([])["lambda"] == 0.0)}
    checks["passed"] = bool(all(v.get("ok", False) for v in checks.values() if isinstance(v, dict)))
    return checks


def boot(per_series: pd.DataFrame, a: str, b: str) -> dict:
    x, y = per_series[a].to_numpy(np.float64), per_series[b].to_numpy(np.float64)
    rng = np.random.default_rng(SEED)
    n = len(x)
    v = np.empty(DRAWS)
    for i in range(DRAWS):
        idx = rng.integers(0, n, n)
        v[i] = (y[idx].mean() - x[idx].mean()) / y[idx].mean()
    lo, hi = np.quantile(v, [0.025, 0.975])
    return {"relative_improvement": float((y.mean() - x.mean()) / y.mean()),
            "ci": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = identity_tests()
    (OUT / "identity_tests.json").write_text(json.dumps(checks, indent=2, default=str))
    if not checks["passed"]:
        raise SystemExit("IDENTITY_TESTS_FAILED")
    print("identity tests: PASS", flush=True)

    manifest = json.loads((P0L1_OUT / "expanded_fold_manifest.json").read_text())
    spec = freeze_spec()
    safe_spec = dict(spec)
    safe_spec.update({
        "name": "SAFE_P0L1_V1", "base_method": "FINAL_GATE_CANDIDATE_V1",
        "base_method_sha256": spec["spec_sha256"],
        "addition": "g_safe = alpha + lambda * (g_raw - alpha), lambda in [0,1]",
        "lambda_fitting": ("weighted least squares in closed form under the direct loss's own "
                           "mask and train-scale weighting; never a grid search"),
        "lambda_information_boundary": "cross-fitted folds strictly earlier than the evaluation fold",
        "cold_start_rule": "lambda = 0 when no earlier cross-fitted fold exists",
        "denominator_floor": DENOMINATOR_FLOOR,
        "no_signal_rule": "lambda = 0 (SAFE_GATE_NO_CALIBRATION_SIGNAL)"})
    safe_spec["safe_spec_sha256"] = hashlib.sha256(
        json.dumps(safe_spec, sort_keys=True, default=str).encode()).hexdigest()
    (OUT / "safe_gate_spec.json").write_text(json.dumps(safe_spec, indent=2, default=str))

    a, b = spec["expert_a"], spec["expert_b"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"analysis": "Safe-P0L1 shrinkage calibration, TRAIN/OOF only",
              "test_used": False, "existing_test_scored": False,
              "safe_spec_sha256": safe_spec["safe_spec_sha256"],
              "fold_boundary_sha256": manifest["fold_boundary_sha256"],
              "identity_tests_passed": checks["passed"],
              "started_at_utc": _utc(), "datasets": {}}
    lambda_rows, all_frames = [], {}

    for name in args.datasets:
        block = manifest["datasets"][name]
        tables = {f["fold"]: fold_table(name, f, a, b, device) for f in block["folds"]}
        print(f"[{name}] fold tables built", flush=True)
        columns = F.feature_columns(spec["feature_variant"], list(tables[0].columns))

        crossfitted, history, results = [], [], {}
        for fold in block["folds"][1:]:
            k = fold["fold"]
            past = pd.concat([tables[j] for j in range(k)], ignore_index=True)
            held = tables[k].copy()
            alpha = fit_alpha(past)
            medians = F.fit_imputer(past, F.STRUCTURE_COLUMNS)
            xp = F.apply_imputer(past, medians)[columns].to_numpy(np.float64)
            centre, spread = xp.mean(0), np.where(xp.std(0) > 0, xp.std(0), 1.0)
            model = fit_direct(((xp - centre) / spread).astype(np.float32), past, device)
            xh = F.apply_imputer(held, medians)[columns].to_numpy(np.float64)
            g_raw = G.gate_weights(model, ((xh - centre) / spread).astype(np.float32),
                                   device).astype(np.float64)

            calibration = calibrate(history)          # strictly earlier cross-fitted folds
            lam = calibration["lambda"]
            g_safe = alpha + lam * (g_raw - alpha)

            e_alpha = mixture_mse(held, np.full(len(held), alpha))
            e_raw = mixture_mse(held, g_raw)
            e_safe = mixture_mse(held, g_safe)
            held["g_raw"], held["alpha"] = g_raw, alpha
            history.append(held[["g_raw", "alpha", "num", "den", "n"]].copy())

            row = {"dataset": name, "fold": k, "alpha": alpha, "lambda": lam,
                   "lambda_raw": calibration["lambda_raw"], "status": calibration["status"],
                   "n_calibration_origins": calibration["n_calibration_origins"],
                   "n_eval_origins": int(len(held)),
                   "E_alpha": float(e_alpha.mean()), "E_raw": float(e_raw.mean()),
                   "E_safe": float(e_safe.mean()),
                   "safe_vs_alpha": float((e_alpha.mean() - e_safe.mean()) / e_alpha.mean()),
                   "raw_vs_alpha": float((e_alpha.mean() - e_raw.mean()) / e_alpha.mean()),
                   "safe_vs_raw": float((e_raw.mean() - e_safe.mean()) / e_raw.mean()),
                   "mean_g_raw": float(g_raw.mean()), "mean_g_safe": float(g_safe.mean())}
            results[k] = row
            lambda_rows.append(row)
            crossfitted.append(pd.DataFrame({
                "series_id": held["series_id"].to_numpy(), "fold": k,
                "alpha_loss": e_alpha, "raw_loss": e_raw, "safe_loss": e_safe,
                "convex": held["mse_convex"].to_numpy(),
                "g_raw": g_raw, "g_safe": g_safe, "lambda": lam,
                "log_ADI": held["f_log_ADI"].to_numpy()}))
            print(f"   fold {k}: alpha {alpha:.2f}  lambda {lam:.3f} ({calibration['status']})  "
                  f"safe_vs_alpha {row['safe_vs_alpha']*100:+.3f}%  "
                  f"raw_vs_alpha {row['raw_vs_alpha']*100:+.3f}%", flush=True)

        joined = pd.concat(crossfitted, ignore_index=True)
        joined.to_parquet(OUT / f"per_origin_safe_{name}.parquet", index=False)
        all_frames[name] = joined
        per_series = joined.groupby("series_id")[
            ["alpha_loss", "raw_loss", "safe_loss", "convex"]].mean()
        per_series.to_csv(OUT / f"per_series_{name}.csv")
        calibrated = [r for r in results.values() if r["status"] == "CALIBRATED"]
        cold = [r for r in results.values() if r["status"] != "CALIBRATED"]
        lambdas = [r["lambda"] for r in calibrated]
        report["datasets"][name] = {
            "n_folds": len(results), "n_calibrated": len(calibrated), "n_cold_start": len(cold),
            "fold_results": results,
            "aggregate": {
                "E_alpha": float(per_series["alpha_loss"].mean()),
                "E_raw": float(per_series["raw_loss"].mean()),
                "E_safe": float(per_series["safe_loss"].mean()),
                "E_convex": float(per_series["convex"].mean()),
                "safe_vs_alpha": float((per_series["alpha_loss"].mean()
                                        - per_series["safe_loss"].mean())
                                       / per_series["alpha_loss"].mean()),
                "raw_vs_alpha": float((per_series["alpha_loss"].mean()
                                       - per_series["raw_loss"].mean())
                                      / per_series["alpha_loss"].mean()),
                "safe_vs_raw": float((per_series["raw_loss"].mean()
                                      - per_series["safe_loss"].mean())
                                     / per_series["raw_loss"].mean())},
            "bootstrap": {"safe_vs_alpha": boot(per_series, "safe_loss", "alpha_loss"),
                          "safe_vs_raw": boot(per_series, "safe_loss", "raw_loss")},
            "calibrated_fold_stability": {
                "n": len(calibrated),
                "positive": int(sum(r["safe_vs_alpha"] > 0 for r in calibrated)),
                "non_negative": int(sum(r["safe_vs_alpha"] >= 0 for r in calibrated)),
                "worst": float(min((r["safe_vs_alpha"] for r in calibrated), default=float("nan"))),
                "median": float(np.median([r["safe_vs_alpha"] for r in calibrated])
                                if calibrated else float("nan")),
                "mean": float(np.mean([r["safe_vs_alpha"] for r in calibrated])
                              if calibrated else float("nan"))},
            "lambda_stability": {
                "min": float(np.min(lambdas)) if lambdas else None,
                "max": float(np.max(lambdas)) if lambdas else None,
                "mean": float(np.mean(lambdas)) if lambdas else None,
                "median": float(np.median(lambdas)) if lambdas else None,
                "std": float(np.std(lambdas)) if lambdas else None,
                "warn_instability": bool(lambdas and len(lambdas) >= 3
                                         and (max(lambdas) - min(lambdas)) > LAMBDA_RANGE_WARN)},
        }

    pd.DataFrame(lambda_rows).to_csv(OUT / "lambda_calibration_by_fold.csv", index=False)
    report["finished_at_utc"] = _utc()
    report["git_commit"] = cli._git_commit()
    (OUT / "aggregate_results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({n: {"safe_vs_alpha": b["aggregate"]["safe_vs_alpha"],
                          "raw_vs_alpha": b["aggregate"]["raw_vs_alpha"],
                          "lambda_mean": b["lambda_stability"]["mean"]}
                      for n, b in report["datasets"].items()}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser("Safe-P0L1 shrinkage calibration")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=list(DATASETS))
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
