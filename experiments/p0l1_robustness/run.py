"""P0L1 across every expanding validation block the train regions can hold.

One method, frozen: the diversity screen's expert pair, G-NOSCALE features, the
Gate-v2 MLP, a neutral sigmoid output, and the direct mixture MSE.  Nothing is
searched here.  The question is only whether the two-fold result that selected
it survives more of the time axis.

Each fold fits its alpha and its gate on strictly earlier data and is scored on
the block that follows, so a fold's own target never touches what predicts it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..expert_diversity.oof_experts import OUT as DIVERSITY_OUT
from ..external_validity_screen import cli, prereg, rule_replication as rr, screen
from ..gate_v3.diagnose import ALPHA_GRID, fit_alpha, mixture_mse
from ..gate_v3.oof import add_targets, quadratic_from_arrays
from ..multi_benchmark import run as MB
from ..structure_gate import features as F, gate as G, gate_v2 as V2
from ..structure_gate.oof import build_oof_split
from ..unified_temporal_27_v3.training import train_scale
from .folds import DATASETS, OUT

DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]
SPEC = OUT / "final_candidate_spec.json"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def freeze_spec() -> dict:
    """FINAL_GATE_CANDIDATE_V1, recovered from the artifacts rather than restated."""
    experts = json.loads((DIVERSITY_OUT / "expert_set_spec.json").read_text())
    gate = json.loads((screen.OUT.parent / "structure_gate" / "gate_v2_spec.json").read_text())
    spec = {
        "name": "FINAL_GATE_CANDIDATE_V1",
        "frozen_at_utc": _utc(),
        "expert_a": experts["top_pair"][0], "expert_b": experts["top_pair"][1],
        "expert_source": "results/expert_diversity/expert_set_spec.json",
        "experts_frozen_during_gate_fitting": True,
        "feature_variant": gate["selected"]["variant"],
        "feature_policy": "as-of-origin only; structure recomputed per origin",
        "architecture": "small MLP, hidden %d" % V2.G.MLP_HIDDEN,
        "output": "g = sigmoid(f(x))  (neutral parameterization)",
        "mixture": "y = (1 - g) * A + g * B",
        "loss": "direct normalized mixture MSE",
        "normalization": "per-series train-window mean (train_scale), train only",
        "gate_training": "TRAIN-internal OOF only, expanding time",
        "global_alpha": "grid over [0,1] step 0.05, fitted on strictly earlier folds",
        "optimizer": {"epochs": V2.EPOCHS, "lr": V2.LR, "seed": V2.SEED},
        "candidate_search": "none; P0L1 is the only condition run",
    }
    payload = json.dumps(spec, indent=2, sort_keys=True, default=str)
    spec["spec_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return spec


def fold_table(name: str, fold: dict, a: str, b: str, device) -> pd.DataFrame:
    """Expert predictions and gate features for one fold's validation block."""
    if name in ("m5", "favorita"):
        data = screen.load_dataset(name)
        cfg = screen.config_for(name)
        stride = prereg.SPLITS[name]["test_origin_stride"]
        fold_cfg = replace(cfg, train_end=fold["train_end"],
                           val_end=fold["train_end"] + cfg.horizon,
                           length=fold["validation_end"])
        split = build_oof_split(data, fold_cfg, stride)
        # MB.expert_predictions already routes neural keys to the frozen trainer and
        # classical ids to their own fitter, so both dataset families share one path.
        experts = {e: MB.expert_predictions(MB.ID_TO_KEY.get(e, e), fold_cfg, split, device, data)
                   for e in (a, b)}
    else:
        data = MB.load_grid(name)
        cfg = MB.config_for(data)
        fold_cfg = replace(cfg, train_end=fold["train_end"],
                           val_end=fold["train_end"] + cfg.horizon,
                           length=fold["validation_end"])
        split = MB.build_split(data, fold_cfg, MB.STRIDE)
        experts = {e: MB.expert_predictions(MB.ID_TO_KEY.get(e, e), fold_cfg, split, device, data)
                   for e in (a, b)}

    n_series = data["y"].shape[0]
    scale = split.scale
    axis = np.repeat(np.arange(n_series), split.test.n_origins)
    s = np.maximum(scale[axis].astype(np.float64), 1e-9)[:, None]
    mask = split.test.target_mask.astype(np.float64)
    yn = split.test.target.astype(np.float64) / s
    pa, pb = experts[a].astype(np.float64) / s, experts[b].astype(np.float64) / s
    table = quadratic_from_arrays(pa, pb, yn, mask, axis)
    keep = mask.sum(axis=1) > 0
    frame = MB.gate_frame(data, fold_cfg, split, scale, n_series)[keep].reset_index(drop=True)
    state = MB.pair_state(experts, a, b, np.maximum(scale[axis], 1e-9))[keep].reset_index(drop=True)
    block = pd.concat([table, frame.drop(columns=["series_id", "origin"]), state], axis=1)
    block["series_id"] = frame["series_id"].to_numpy()
    block["fold"] = fold["fold"]
    return F.add_missing_flags(add_targets(block), F.STRUCTURE_COLUMNS)


def fit_direct(x: np.ndarray, block: pd.DataFrame, device):
    """P0L1's optimiser. gate_v2.fit expects Gate-v2's column names, not these.

    The loss, architecture, epochs, learning rate and seed are all the frozen
    ones; only the column lookup differs, so this is a rename rather than a
    second implementation of the objective.
    """
    torch.manual_seed(V2.SEED)
    np.random.seed(V2.SEED)
    model = G.build_gate("mlp", x.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=V2.LR)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    sse = torch.tensor(block["sse_a"].to_numpy(np.float64), dtype=torch.float32, device=device)
    num = torch.tensor(block["num"].to_numpy(np.float64), dtype=torch.float32, device=device)
    den = torch.tensor(block["den"].to_numpy(np.float64), dtype=torch.float32, device=device)
    n_obs = torch.tensor(block["n"].to_numpy(np.float64), dtype=torch.float32, device=device)
    for _ in range(V2.EPOCHS):
        optimiser.zero_grad(set_to_none=True)
        g = model(xt)
        (((sse - 2 * g * num + g ** 2 * den) / n_obs).mean()).backward()
        optimiser.step()
    model.eval()
    return model


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
    manifest = json.loads((OUT / "expanded_fold_manifest.json").read_text())
    spec = freeze_spec()
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    a, b = spec["expert_a"], spec["expert_b"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"analysis": "P0L1 temporal robustness on expanded OOF folds",
              "test_used": False, "existing_test_scored": False,
              "final_candidate_sha256": spec["spec_sha256"],
              "fold_boundary_sha256": manifest["fold_boundary_sha256"],
              "started_at_utc": _utc(), "datasets": {}}
    fold_rows, alpha_rows = [], []

    for name in args.datasets:
        block = manifest["datasets"][name]
        tables = {}
        for fold in block["folds"]:
            tables[fold["fold"]] = fold_table(name, fold, a, b, device)
            print(f"[{name}] fold {fold['fold']}: {len(tables[fold['fold']]):,} origins", flush=True)
        columns = F.feature_columns(spec["feature_variant"], list(tables[0].columns))

        results, crossfitted = {}, []
        for fold in block["folds"][1:]:
            k = fold["fold"]
            past = pd.concat([tables[j] for j in range(k)], ignore_index=True)
            held = tables[k]
            alpha = fit_alpha(past)
            medians = F.fit_imputer(past, F.STRUCTURE_COLUMNS)
            xp = F.apply_imputer(past, medians)[columns].to_numpy(np.float64)
            centre, spread = xp.mean(0), np.where(xp.std(0) > 0, xp.std(0), 1.0)
            model = fit_direct(((xp - centre) / spread).astype(np.float32), past, device)
            xh = F.apply_imputer(held, medians)[columns].to_numpy(np.float64)
            g = G.gate_weights(model, ((xh - centre) / spread).astype(np.float32),
                               device).astype(np.float64)
            e_gate = mixture_mse(held, g)
            e_alpha = mixture_mse(held, np.full(len(held), alpha))
            e_convex = held["mse_convex"].to_numpy()
            denominator = e_alpha.mean() - e_convex.mean()
            row = {"dataset": name, "fold": k, "n_origins": int(len(held)), "alpha": alpha,
                   "E_a": float(held["mse_a"].mean()), "E_b": float(held["mse_b"].mean()),
                   "E_equal": float(mixture_mse(held, np.full(len(held), 0.5)).mean()),
                   "E_alpha": float(e_alpha.mean()), "E_gate": float(e_gate.mean()),
                   "E_convex": float(e_convex.mean()),
                   "improvement": float((e_alpha.mean() - e_gate.mean()) / e_alpha.mean()),
                   "oracle_recovery": (float((e_alpha.mean() - e_gate.mean()) / denominator)
                                       if denominator > 0 else None),
                   "mean_g": float(g.mean()), "median_g": float(np.median(g)),
                   "std_g": float(g.std()), "p10_g": float(np.quantile(g, .1)),
                   "p90_g": float(np.quantile(g, .9)),
                   "frac_g_below_0.1": float((g < 0.1).mean()),
                   "frac_g_above_0.9": float((g > 0.9).mean()),
                   "corr_g_gstar": float(np.corrcoef(g, held["g_star"].to_numpy())[0, 1]),
                   "corr_g_q": float(np.corrcoef(
                       g, G.regret_target(held["regret"].to_numpy(float),
                                          G.choose_tau(past["regret"].to_numpy(float))))[0, 1])}
            results[k] = row
            fold_rows.append(row)
            alpha_rows.append({"dataset": name, "fold": k, "alpha": alpha,
                               "E_alpha": row["E_alpha"]})
            crossfitted.append(pd.DataFrame({
                "series_id": held["series_id"].to_numpy(), "fold": k, "g": g,
                "gate": e_gate, "alpha_loss": e_alpha, "convex": e_convex,
                "mse_a": held["mse_a"].to_numpy(), "mse_b": held["mse_b"].to_numpy(),
                "ADI_train": held["f_log_ADI"].to_numpy()}))
            print(f"   fold {k}: alpha {alpha:.2f}  improvement {row['improvement']*100:+.3f}%  "
                  f"recovery {row['oracle_recovery']}", flush=True)

        joined = pd.concat(crossfitted, ignore_index=True)
        joined.to_parquet(OUT / f"crossfitted_{name}.parquet", index=False)
        per_series = joined.groupby("series_id")[["gate", "alpha_loss", "convex"]].mean()
        improvements = [r["improvement"] for r in results.values()]
        alphas = [r["alpha"] for r in results.values()]
        denominator = per_series["alpha_loss"].mean() - per_series["convex"].mean()
        report["datasets"][name] = {
            "n_validation_folds": len(results),
            "warn_limited_temporal_folds": block["warn_limited"],
            "fold_results": results,
            "stability": {"mean": float(np.mean(improvements)),
                          "median": float(np.median(improvements)),
                          "worst": float(np.min(improvements)),
                          "best": float(np.max(improvements)),
                          "std": float(np.std(improvements)),
                          "positive_folds": int(sum(v > 0 for v in improvements)),
                          "negative_folds": int(sum(v <= 0 for v in improvements)),
                          "sign_consistency": float(np.mean([v > 0 for v in improvements]))},
            "alpha_stability": {"mean": float(np.mean(alphas)), "std": float(np.std(alphas)),
                                "min": float(np.min(alphas)), "max": float(np.max(alphas)),
                                "max_fold_to_fold_change": float(
                                    np.max(np.abs(np.diff(alphas))) if len(alphas) > 1 else 0.0)},
            "aggregate": {"E_gate": float(per_series["gate"].mean()),
                          "E_alpha": float(per_series["alpha_loss"].mean()),
                          "E_convex": float(per_series["convex"].mean())},
            "bootstrap": boot(per_series, "gate", "alpha_loss"),
            "oracle_recovery": (float((per_series["alpha_loss"].mean()
                                       - per_series["gate"].mean()) / denominator)
                                if denominator > 0 else "ORACLE_RECOVERY_UNDEFINED"),
        }

    pd.DataFrame(fold_rows).to_csv(OUT / "fold_results.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(OUT / "fold_alpha.csv", index=False)
    report["finished_at_utc"] = _utc()
    report["git_commit"] = cli._git_commit()
    (OUT / "aggregate_result.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({n: {"mean": b["stability"]["mean"],
                          "positive": f"{b['stability']['positive_folds']}/{b['n_validation_folds']}",
                          "ci_excludes_zero": b["bootstrap"]["ci_excludes_zero"]}
                      for n, b in report["datasets"].items()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("P0L1 temporal robustness")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=list(DATASETS))
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
