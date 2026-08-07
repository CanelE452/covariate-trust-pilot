"""Is the regret surrogate aimed at the right thing? Answered on OOF alone.

Gate-v2's target is q = sigmoid(regret / tau), which is 0.5 wherever the two
experts tie.  The quantity a mixture forecast actually wants is g*, the exact
convex minimiser.  If q and g* disagree, the gate was optimising a proxy that
does not point where the loss does -- and the tell is not that mean(g) differs
from alpha, which can happen for innocent reasons, but that using q itself as a
mixing weight loses to simply using the static alpha.

That is the test here: plug q, alpha and g* straight in as weights, on OOF rows,
and compare the mixture errors they produce.  No gate is trained, so nothing in
this file can be confounded by how well a network fits.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from ..external_validity_screen import cli, screen
from ..structure_gate import gate as G
from .oof import DATASETS, OUT

#: Numerical safety only, so logit(alpha) stays finite. Not a tuned quantity.
EPS_ALPHA = 1e-4
ALPHA_GRID = np.arange(0.0, 1.0001, 0.05)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def mixture_mse(table: pd.DataFrame, g) -> np.ndarray:
    return ((table["sse_a"].to_numpy(np.float64) - 2 * g * table["num"].to_numpy(np.float64)
             + g ** 2 * table["den"].to_numpy(np.float64)) / table["n"].to_numpy(np.float64))


def fit_alpha(table: pd.DataFrame) -> float:
    """The repository's global-alpha rule: a grid search on the given rows."""
    return float(min(ALPHA_GRID, key=lambda w: mixture_mse(table, np.full(len(table), w)).mean()))


def anchor_logit(alpha: float) -> float:
    safe = float(np.clip(alpha, EPS_ALPHA, 1 - EPS_ALPHA))
    return float(np.log(safe / (1 - safe)))


def alpha_by_fold(table: pd.DataFrame) -> dict:
    """Expanding time: each validation fold's alpha comes from strictly earlier folds."""
    folds = sorted(table["fold"].unique())
    out = {}
    for validation in folds[1:]:
        past = table[table["fold"] < validation]
        alpha = fit_alpha(past)
        held = table[table["fold"] == validation]
        out[int(validation)] = {
            "alpha_train": alpha,
            "alpha_safe": float(np.clip(alpha, EPS_ALPHA, 1 - EPS_ALPHA)),
            "anchor_logit": anchor_logit(alpha),
            "n_train_rows": int(len(past)), "n_validation_rows": int(len(held)),
            "validation_alpha_loss": float(mixture_mse(held, np.full(len(held), alpha)).mean())}
    return out


def alignment(table: pd.DataFrame, tau: float) -> dict:
    q = G.regret_target(table["regret"].to_numpy(np.float64), tau)
    g = table["g_star"].to_numpy(np.float64)
    non_flat = ~table["flat"].to_numpy()
    r = table["regret"].to_numpy(np.float64)
    band = np.quantile(np.abs(r), 0.1)
    groups = {"regret_negative": r < -band, "regret_near_zero": np.abs(r) <= band,
              "regret_positive": r > band}
    return {
        "tau": tau,
        "q": {"mean": float(q.mean()), "median": float(np.median(q)), "std": float(q.std())},
        "g_star": {"mean": float(g.mean()), "median": float(np.median(g)), "std": float(g.std())},
        "pearson_q_gstar": float(np.corrcoef(q, g)[0, 1]),
        "spearman_q_gstar": float(stats.spearmanr(q, g).statistic),
        "mae_q_gstar": float(np.abs(q - g).mean()),
        "rmse_q_gstar": float(np.sqrt(((q - g) ** 2).mean())),
        "g_star_distribution": {
            "flat_share": float(table["flat"].mean()),
            "at_zero_share": float(((g == 0) & non_flat).mean()),
            "interior_share": float(((g > 0) & (g < 1) & non_flat).mean()),
            "at_one_share": float(((g == 1) & non_flat).mean())},
        "g_star_by_regret_sign": {
            name: {"n": int(sel.sum()), "mean_g_star": float(g[sel].mean()),
                   "median_g_star": float(np.median(g[sel])),
                   "mean_q": float(q[sel].mean())}
            for name, sel in groups.items() if sel.any()},
    }


def target_as_predictor(table: pd.DataFrame, tau: float, alpha: float) -> dict:
    """Use q, alpha and g* directly as weights. q is a diagnostic, not deployable."""
    q = G.regret_target(table["regret"].to_numpy(np.float64), tau)
    e_q = float(mixture_mse(table, q).mean())
    e_alpha = float(mixture_mse(table, np.full(len(table), alpha)).mean())
    e_gstar = float(table["mse_convex"].mean())
    return {"E_q": e_q, "E_alpha": e_alpha, "E_gstar": e_gstar,
            "alpha_used": alpha,
            "q_worse_than_alpha": bool(e_q > e_alpha),
            "q_vs_alpha_relative": float((e_q - e_alpha) / e_alpha),
            "note": ("q is computed from the true OOF regret, so it is an upper bound on "
                     "what a regret-BCE gate could learn, not a deployable predictor")}


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"analysis": "regret-surrogate alignment, OOF only",
              "test_used": False, "existing_test_scored": False,
              "eps_alpha": EPS_ALPHA, "computed_at_utc": _utc(), "datasets": {}}
    rows = []
    for name in args.datasets:
        table = pd.read_parquet(OUT / f"oof_{name}.parquet")
        tau = G.choose_tau(table["regret"].to_numpy(np.float64))
        alpha_all = fit_alpha(table)
        folds = alpha_by_fold(table)
        block = {"n_origins": int(len(table)), "tau": tau,
                 "alpha_full_oof": alpha_all,
                 "alpha_by_validation_fold": folds,
                 "alignment": alignment(table, tau),
                 "target_as_predictor": target_as_predictor(table, tau, alpha_all)}
        report["datasets"][name] = block
        for fold, info in folds.items():
            rows.append({"dataset": name, "validation_fold": fold, **info})
        a = block["alignment"]
        t = block["target_as_predictor"]
        print(f"[{name}] alpha {alpha_all:.2f}  corr(q,g*) {a['pearson_q_gstar']:+.3f}  "
              f"spearman {a['spearman_q_gstar']:+.3f}  MAE {a['mae_q_gstar']:.3f}  "
              f"E_q/E_alpha {t['q_vs_alpha_relative']*100:+.2f}%", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "oof_alpha_by_dataset_fold.csv", index=False)
    report["git_commit"] = cli._git_commit()
    (OUT / "target_alignment_summary.json").write_text(json.dumps(report, indent=2, default=str))
    (OUT / "target_as_predictor_result.json").write_text(json.dumps(
        {n: b["target_as_predictor"] for n, b in report["datasets"].items()}, indent=2))
    print("wrote target_alignment_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser("Gate-v3 target diagnostic")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=list(DATASETS))
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
