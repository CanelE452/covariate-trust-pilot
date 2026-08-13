"""Where do Point and Hurdle sit against classical intermittent-demand methods?

This is a context table, not evidence for H1 or H2.  The mechanism contrast
stays with the paired bootstrap results; what this answers is the reviewer's
question of whether the learned models are competitive at all.

TSB and seasonal naive already exist in the repository
(``om_factorization_killtest.models.tsb_state`` / ``seasonal_naive``) and are
reused rather than reimplemented, including the frozen ``TSB_GRID``.  Croston
and SBA do not exist anywhere in the tree and no forecasting package is
installed, so they are written here against the published recursions and
checked by ``hand_check()`` before any benchmark uses them.

Every method sees the same series, the same lookback windows and the same test
origins as Point and Hurdle, and every smoothing parameter is chosen on the
validation split with the same grid TSB was given.  Nothing is tuned on test.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ..om_factorization_killtest import models as km_models
from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from . import cli, prereg, rule_replication as rr, screen

OUT = screen.OUT / "classical_benchmark"

#: Reuses the grid TSB was frozen with rather than inventing a second one.
ALPHA_GRID = tuple(km_prereg.TSB_GRID["alpha"])

LEARNED_CHECKPOINTS = {
    "m5": screen.OUT / "rule_replication" / "models",
    "favorita": screen.OUT / "favorita_transfer" / "models",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------- Croston -----


def croston_state(series: np.ndarray, alpha: float) -> tuple[float, float]:
    """Croston's recursion: SES on demand size and on inter-arrival interval.

    Both states are seeded from the history's own statistics before the
    recursion runs, which is how ``tsb_state`` seeds p and m; the alternative of
    starting from the first event makes the forecast depend on where the window
    happens to begin.  A window with no demand has no interval to estimate and
    returns a zero forecast rather than dividing by zero.
    """
    series = np.asarray(series, dtype=np.float64)
    events = np.flatnonzero(series > 0)
    if events.size == 0:
        return 0.0, 1.0
    z = float(series[events].mean())
    x = float(np.diff(events).mean()) if events.size > 1 else float(series.size / events.size)
    x = max(x, 1.0)
    gap = 1
    for value in series:
        if value > 0:
            z += alpha * (value - z)
            x += alpha * (gap - x)
            gap = 1
        else:
            gap += 1
    return z, max(x, 1e-9)


def croston_forecast(history: np.ndarray, horizon: int, alpha: float,
                     variant: str = "croston") -> np.ndarray:
    """Flat per-period rate z/x over the horizon; SBA scales it by 1 - alpha/2."""
    history = np.asarray(history, dtype=np.float64)
    out = np.empty(history.shape[0])
    factor = 1.0 - alpha / 2.0 if variant == "sba" else 1.0
    for i, row in enumerate(history):
        z, x = croston_state(row, alpha)
        out[i] = factor * z / x
    return np.repeat(out[:, None], horizon, axis=1)


def ses_forecast(history: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    """Simple exponential smoothing on the raw series, seeded at its mean."""
    history = np.asarray(history, dtype=np.float64)
    level = history.mean(axis=1)
    for t in range(history.shape[1]):
        level = level + alpha * (history[:, t] - level)
    return np.repeat(level[:, None], horizon, axis=1)


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[:, -1:], horizon, axis=1)


def hand_check() -> dict:
    """Croston and SBA against a recursion worked out by hand.

    series = [0, 2, 0, 0, 4], alpha = 0.5.
    seed: events at 1 and 4, z = (2+4)/2 = 3, intervals = [3], x = 3.
    t=0 y=0            gap 1 -> 2
    t=1 y=2  z = 3 + .5(2-3) = 2.5    x = 3 + .5(2-3) = 2.5    gap -> 1
    t=2 y=0            gap 1 -> 2
    t=3 y=0            gap 2 -> 3
    t=4 y=4  z = 2.5 + .5(4-2.5) = 3.25  x = 2.5 + .5(3-2.5) = 2.75  gap -> 1
    croston = 3.25 / 2.75 = 1.1818...   sba = 0.75 * that = 0.8863...
    """
    series = np.array([0.0, 2.0, 0.0, 0.0, 4.0])
    z, x = croston_state(series, 0.5)
    expected_z, expected_x = 3.25, 2.75
    croston = croston_forecast(series[None, :], 2, 0.5, "croston")
    sba = croston_forecast(series[None, :], 2, 0.5, "sba")
    checks = {
        "z": {"expected": expected_z, "got": z, "ok": abs(z - expected_z) < 1e-12},
        "x": {"expected": expected_x, "got": x, "ok": abs(x - expected_x) < 1e-12},
        "croston": {"expected": expected_z / expected_x, "got": float(croston[0, 0]),
                    "ok": abs(croston[0, 0] - expected_z / expected_x) < 1e-12},
        "sba": {"expected": 0.75 * expected_z / expected_x, "got": float(sba[0, 0]),
                "ok": abs(sba[0, 0] - 0.75 * expected_z / expected_x) < 1e-12},
        "sba_is_scaled_croston": {"expected": 0.75, "got": float(sba[0, 0] / croston[0, 0]),
                                  "ok": abs(sba[0, 0] / croston[0, 0] - 0.75) < 1e-12},
        "flat_over_horizon": {"expected": True,
                              "got": bool(np.allclose(croston[0, 0], croston[0, 1])),
                              "ok": bool(np.allclose(croston[0, 0], croston[0, 1]))},
        "empty_window_is_zero": {"expected": 0.0,
                                 "got": float(croston_forecast(np.zeros((1, 5)), 1, 0.5)[0, 0]),
                                 "ok": croston_forecast(np.zeros((1, 5)), 1, 0.5)[0, 0] == 0.0},
    }
    return {"passed": all(c["ok"] for c in checks.values()), "checks": checks}


# ------------------------------------------------------------ selection ----


def _masked_mse(prediction: np.ndarray, windows) -> float:
    mask = windows.target_mask.astype(np.float64)
    target = windows.target.astype(np.float64)
    return float((((prediction - target) ** 2) * mask).sum() / max(mask.sum(), 1.0))


def select_alpha(validation, horizon: int, method: str) -> tuple[float, float]:
    """Grid search on the VALIDATION split only, same grid TSB was given."""
    history = validation.history.astype(np.float64)
    best = (None, float("inf"))
    for alpha in ALPHA_GRID:
        if method in ("croston", "sba"):
            pred = croston_forecast(history, horizon, alpha, method)
        elif method == "ses":
            pred = ses_forecast(history, horizon, alpha)
        else:
            raise ValueError(method)
        mse = _masked_mse(pred, validation)
        if mse < best[1]:
            best = (alpha, mse)
    return best


# ----------------------------------------------------------- evaluation ----


def per_series_metrics(mean_prediction: np.ndarray, windows, n_series: int) -> dict:
    return screen.test_metrics({"mean_prediction": mean_prediction}, windows, n_series)


def run_dataset(name: str, device) -> dict:
    cfg = screen.config_for(name)
    data = screen.load_dataset(name)
    split = screen.build_split(data, cfg, prereg.SPLITS[name]["test_origin_stride"])
    n_series = data["y"].shape[0]
    horizon = cfg.horizon
    test_history = split.test.history.astype(np.float64)

    chosen, predictions, failures = {}, {}, {}

    for method in ("croston", "sba", "ses"):
        alpha, val_mse = select_alpha(split.validation, horizon, method)
        chosen[method] = {"alpha": alpha, "validation_mse": val_mse,
                          "grid": list(ALPHA_GRID), "selected_on": "validation split"}
        predictions[method] = (croston_forecast(test_history, horizon, alpha, method)
                               if method in ("croston", "sba")
                               else ses_forecast(test_history, horizon, alpha))

    from ..om_factorization_killtest import evaluate as km_evaluate
    alpha, beta, val_mse = km_evaluate.select_tsb(split.validation)
    chosen["tsb"] = {"alpha": alpha, "beta": beta, "validation_mse": val_mse,
                     "grid": {k: list(v) for k, v in km_prereg.TSB_GRID.items()},
                     "selected_on": "validation split",
                     "implementation": "om_factorization_killtest.models.tsb_state"}
    p, mu = km_models.tsb_forecast(test_history, horizon, alpha, beta)
    predictions["tsb"] = p * mu

    predictions["seasonal_naive"] = km_models.seasonal_naive(
        test_history, horizon, cfg.period)
    chosen["seasonal_naive"] = {"period": cfg.period, "selected_on": "no parameter",
                                "implementation": "om_factorization_killtest.models.seasonal_naive"}
    predictions["naive"] = naive_forecast(test_history, horizon)
    chosen["naive"] = {"selected_on": "no parameter"}

    for role in ("point", "hurdle"):
        path = LEARNED_CHECKPOINTS[name] / f"{role}.pt"
        if not path.exists():
            failures[f"dlinear_{role}"] = f"checkpoint missing: {path}"
            continue
        payload = torch.load(path, map_location=device, weights_only=False)
        model = km_models.BUILDERS[payload["builder_key"]](payload["lookback"],
                                                           payload["horizon"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        predictions[f"dlinear_{role}"] = km_train.predict(model, split.test, device)["mean_prediction"]
        chosen[f"dlinear_{role}"] = {"checkpoint": str(path.relative_to(screen.REPO)),
                                     "model_seed": payload["model_seed"],
                                     "trained_on": payload["trained_on"]}

    mask = split.test.target_mask.astype(np.float64)
    target = split.test.target.astype(np.float64)
    rows, per_series = [], {}
    for method, pred in predictions.items():
        metrics = per_series_metrics(pred.astype(np.float64), split.test, n_series)
        per_series[method] = metrics["rmse_realized"]
        rows.append({
            "method": method,
            "overall_rmse": float(np.sqrt((((pred - target) ** 2) * mask).sum() / mask.sum())),
            "overall_mae": float(((np.abs(pred - target)) * mask).sum() / mask.sum()),
            "mean_per_series_rmse": float(np.mean(metrics["rmse_realized"])),
            "median_per_series_rmse": float(np.median(metrics["rmse_realized"])),
            "mean_per_series_mae": float(np.mean(metrics["mae_realized"]))})

    table = pd.DataFrame(rows)
    matrix = np.column_stack([per_series[m] for m in table["method"]])
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    table["mean_rank"] = ranks.mean(axis=0)
    table["win_pct"] = [(matrix.argmin(axis=1) == i).mean() * 100 for i in range(len(table))]
    table = table.sort_values("mean_per_series_rmse").reset_index(drop=True)

    friedman = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    posthoc = {}
    if friedman.pvalue < 0.05:
        methods = list(table["method"])
        order = [list(per_series).index(m) for m in methods]
        pairs, pvals = [], []
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                res = stats.wilcoxon(matrix[:, order[a]], matrix[:, order[b]],
                                     zero_method="zsplit")
                pairs.append(f"{methods[a]} vs {methods[b]}")
                pvals.append(res.pvalue)

        adjusted = _holm(pvals)
        posthoc = {"method": "pairwise Wilcoxon signed-rank, Holm-corrected; the "
                             "repository has no standard post-hoc and no package "
                             "providing Nemenyi is installed",
                   "results": {p: {"p_raw": float(r), "p_holm": float(h),
                                   "significant_at_0.05": bool(h < 0.05)}
                               for p, r, h in zip(pairs, pvals, adjusted)}}

    return {"dataset": name, "n_series": int(n_series),
            "test_origins": split.test.origins.tolist(),
            "horizon": cfg.horizon, "lookback": cfg.lookback,
            "table": table.to_dict("records"), "method_config": chosen,
            "failures": failures,
            "friedman": {"statistic": float(friedman.statistic),
                         "pvalue": float(friedman.pvalue),
                         "unit": "series", "n_series": int(n_series),
                         "k_methods": int(matrix.shape[1])},
            "posthoc": posthoc,
            "per_series_rmse_columns": list(table["method"])}


def _holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    n = len(pvals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def cmd_run(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = hand_check()
    implementation_audit = {
        "hand_check": audit,
        "reused_from_repository": {
            "tsb": "om_factorization_killtest.models.tsb_state / tsb_forecast",
            "seasonal_naive": "om_factorization_killtest.models.seasonal_naive",
            "tsb_grid": {k: list(v) for k, v in km_prereg.TSB_GRID.items()}},
        "implemented_here": {
            "croston": "SES on demand size and interval, forecast z/x",
            "sba": "Croston scaled by 1 - alpha/2",
            "ses": "SES on the raw series",
            "naive": "last observed value"},
        "packages_available": {"statsforecast": False, "sktime": False, "darts": False},
        "iets": "IETS_NOT_AVAILABLE -- no verified implementation in the repository "
                "or the environment; not written from scratch",
        "alpha_grid": list(ALPHA_GRID),
        "tuning": "validation split only; test never used for selection"}
    (OUT / "implementation_audit.json").write_text(
        json.dumps(implementation_audit, indent=2, default=str))
    if not audit["passed"]:
        raise SystemExit("CLASSICAL_BENCHMARK_BLOCKED: Croston hand-check failed")
    print("hand_check: PASS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"analysis": "classical intermittent-demand benchmark, Stage A populations",
              "not_evidence_for": ["H1", "H2"],
              "population": "each dataset's Stage A 1200 series, same test origins as Stage A",
              "learned_seed": prereg.MODELS["canonical_model_seed"],
              "seed_note": "canonical seed only; the M5 3-seed result is a separate table",
              "implementation_audit": implementation_audit,
              "datasets": {}, "run_at_utc": _utc(), "device": str(device),
              "git_commit": cli._git_commit()}
    for name in ("m5", "favorita"):
        print(f"--- {name} ---")
        report["datasets"][name] = run_dataset(name, device)
        for row in report["datasets"][name]["table"]:
            print(f"  {row['method']:<16} rmse={row['mean_per_series_rmse']:.4f} "
                  f"rank={row['mean_rank']:.2f} win={row['win_pct']:.1f}%")

    complete = all(not report["datasets"][d]["failures"] for d in report["datasets"])
    report["verdict"] = ("CLASSICAL_BENCHMARK_COMPLETE" if complete
                         else "CLASSICAL_BENCHMARK_PARTIAL")
    (OUT / "benchmark.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": report["verdict"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("classical intermittent benchmark")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("hand-check").set_defaults(
        func=lambda _a: print(json.dumps(hand_check(), indent=2, default=str)))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
