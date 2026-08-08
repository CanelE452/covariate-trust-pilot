"""Is the routing information there and unlearned, or not there?

P0L1 gains are small, FreshRetailNet is negative, and shrinking the gate toward
the static weight bought tail safety at the cost of the mean.  None of that says
whether the ceiling is the 529-parameter MLP or the features it reads.  So this
run changes exactly one thing -- the function that maps gate features to g --
and leaves the experts, the G-NOSCALE features, the expanding folds, the
normalization and the direct mixture objective at whatever the P0L1 artifacts
already say they are.

The comparison is only possible because the direct mixture MSE is quadratic in
g.  Per origin it equals a constant plus (den/n)(g - num/den)^2, so a boosted
tree fitted to num/den with weight den/n optimises the same objective the MLP
does, and the two can be scored on the same rows.

HistGradientBoostingRegressor here is a diagnostic instrument, not a proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from ..expert_diversity.oof_experts import OUT as DIVERSITY_OUT
from ..external_validity_screen import cli, prereg, screen
from ..gate_v3.diagnose import fit_alpha, mixture_mse
from ..p0l1_robustness.folds import DATASETS, OUT as P0L1_OUT
from ..p0l1_robustness.run import fit_direct, fold_table, freeze_spec
from ..structure_gate import features as F, gate as G
from ..structure_gate.convex_oracle import FLAT_THRESHOLD
from . import spec as S

OUT = screen.OUT.parent / "routing_information_ceiling"
DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]

# Only what this run actually reads.  Unrelated repository churn is reported
# separately and does not touch this list.
DEPENDENCY_ARTIFACTS = [
    "results/expert_diversity/expert_set_spec.json",
    "results/structure_gate/gate_v2_spec.json",
    "results/gate_p0l1_robustness/expanded_fold_manifest.json",
    "results/gate_p0l1_robustness/final_candidate_spec.json",
    "results/gate_p0l1_robustness/aggregate_result.json",
    "results/gate_safe_p0l1/uci_sensitivity.json",
    "data/processed/series.parquet",
    "data/processed/favorita_series.parquet",
    "data/processed/freshretailnet_grid.npz",
    "data/processed/uci_grid.npz",
    "data/calendar.csv",
    "data/sell_prices.csv",
    "experiments/p0l1_robustness/run.py",
    "experiments/p0l1_robustness/folds.py",
    "experiments/structure_gate/features.py",
    "experiments/structure_gate/gate.py",
    "experiments/structure_gate/gate_v2.py",
    "experiments/structure_gate/convex_oracle.py",
    "experiments/gate_v3/oof.py",
    "experiments/gate_v3/diagnose.py",
    "experiments/multi_benchmark/run.py",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root():
    return screen.REPO


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency_seal() -> dict:
    root = _repo_root()
    seal, missing = {}, []
    for rel in DEPENDENCY_ARTIFACTS:
        path = root / rel
        if path.exists():
            seal[rel] = {"sha256": _sha(path), "bytes": path.stat().st_size}
        else:
            missing.append(rel)
    return {"sealed_at_utc": _utc(), "n_artifacts": len(seal),
            "missing": missing, "artifacts": seal}


def verify_seal(seal: dict) -> dict:
    root = _repo_root()
    changed = [rel for rel, meta in seal["artifacts"].items()
               if not (root / rel).exists() or _sha(root / rel) != meta["sha256"]]
    return {"n_checked": len(seal["artifacts"]), "changed": changed,
            "passed": not changed and not seal["missing"]}


def repository_change_warning(seal: dict) -> dict:
    """Everything else git considers untracked or modified, kept out of G0a."""
    root = _repo_root()
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                             capture_output=True, text=True, check=True).stdout
    except Exception as exc:                              # pragma: no cover
        return {"available": False, "error": str(exc)}
    entries = []
    for line in out.splitlines():
        status, path = line[:2].strip(), line[3:].strip()
        entries.append({"status": status, "path": path,
                        "is_scientific_dependency": path in seal["artifacts"]})
    unrelated = [e for e in entries if not e["is_scientific_dependency"]]
    return {"available": True, "n_entries": len(entries),
            "n_unrelated": len(unrelated), "entries": entries,
            "note": ("unrelated untracked or modified files do not promote to a "
                     "G0a failure; only a changed dependency artifact does")}


def equivalence_test(tables: dict) -> dict:
    """The quadratic rewrite has to be the same function, not merely close.

    Checked on the real OOF rows rather than on synthetic numbers, at several g
    values including the endpoints, on the origins where the target is
    identified.
    """
    rows, worst, checked = [], 0.0, 0
    for name, per_fold in tables.items():
        block = pd.concat(per_fold.values(), ignore_index=True)
        live = block[block["den"].to_numpy(np.float64) > FLAT_THRESHOLD]
        num = live["num"].to_numpy(np.float64)
        den = live["den"].to_numpy(np.float64)
        n = live["n"].to_numpy(np.float64)
        sse = live["sse_a"].to_numpy(np.float64)
        u = num / den
        constant = (sse - num ** 2 / den) / n
        for g in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            direct = mixture_mse(live, np.full(len(live), g))
            quadratic = constant + (den / n) * (g - u) ** 2
            error = float(np.max(np.abs(direct - quadratic)))
            relative = float(np.max(np.abs(direct - quadratic)
                                    / np.maximum(np.abs(direct), 1e-12)))
            rows.append({"dataset": name, "g": g, "n_rows": int(len(live)),
                         "max_abs_error": error, "max_rel_error": relative})
            worst = max(worst, error)
            checked += len(live)
    tolerance = 1e-6
    return {"rows": rows, "max_abs_error": worst, "n_comparisons": checked,
            "tolerance": tolerance, "passed": bool(worst <= tolerance),
            "note": ("absolute tolerance rather than bit equality: the two "
                     "expressions differ in floating-point association, and the "
                     "errors scale with the size of the loss itself")}


def fit_hgb(x: np.ndarray, block: pd.DataFrame):
    """The same objective the MLP minimises, written as weighted least squares."""
    den = block["den"].to_numpy(np.float64)
    num = block["num"].to_numpy(np.float64)
    n = block["n"].to_numpy(np.float64)
    live = den > FLAT_THRESHOLD
    target = num[live] / den[live]                 # unconstrained optimum, not clipped
    weight = den[live] / n[live]
    model = HistGradientBoostingRegressor(**S.HGB_CONFIG)
    model.fit(x[live].astype(np.float64), target, sample_weight=weight)
    return model, int(live.sum()), int((~live).sum())


def predict_hgb(model, x: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(x.astype(np.float64)), 0.0, 1.0)


def boot(per_series: pd.DataFrame, a: str, b: str) -> dict:
    """Relative improvement of column a over column b, resampling series."""
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


def three_scales(per_series: pd.DataFrame, a: str, b: str) -> dict:
    x, y = per_series[a].to_numpy(np.float64), per_series[b].to_numpy(np.float64)
    safe = np.maximum(y, 1e-12)
    return {"normalized_mse": float((y.mean() - x.mean()) / y.mean()),
            "mean_per_series_relative": float(np.mean((y - x) / safe)),
            "rmse": float((np.sqrt(y.mean()) - np.sqrt(x.mean())) / np.sqrt(y.mean()))}


def recovery(e_alpha: float, e_model: float, e_convex: float):
    denominator = e_alpha - e_convex
    if denominator <= 0:
        return None
    return float((e_alpha - e_model) / denominator)


def method_identity(frozen: dict) -> dict:
    """The stored P0L1 spec hash cannot be matched, so the fields are compared.

    freeze_spec puts frozen_at_utc inside the payload it hashes, so re-freezing
    the same method yields a different sha256 every time.  That is a property of
    the original code, not a change of method, so identity is established from
    the fields themselves and then confirmed numerically by reproducing the
    MLP's cross-fitted aggregate.
    """
    stored = json.loads((P0L1_OUT / "final_candidate_spec.json").read_text())
    volatile = {"frozen_at_utc", "spec_sha256"}
    diff = {k: {"stored": stored.get(k), "now": frozen.get(k)}
            for k in set(stored) | set(frozen)
            if k not in volatile and stored.get(k) != frozen.get(k)}
    return {"stored_sha256": stored.get("spec_sha256"),
            "recomputed_sha256": frozen.get("spec_sha256"),
            "sha256_match": stored.get("spec_sha256") == frozen.get("spec_sha256"),
            "sha256_is_timestamp_dependent": True,
            "field_differences": diff, "fields_identical": not diff}


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostic = S.diagnostic_spec()
    (OUT / "hgb_diagnostic_spec.json").write_text(json.dumps(diagnostic, indent=2, default=str))

    seal = dependency_seal()
    (OUT / "scientific_dependency_seal.json").write_text(json.dumps(seal, indent=2, default=str))
    warning = repository_change_warning(seal)
    (OUT / "repository_change_warning.json").write_text(json.dumps(warning, indent=2, default=str))

    manifest = json.loads((P0L1_OUT / "expanded_fold_manifest.json").read_text())
    frozen = freeze_spec()
    a, b = frozen["expert_a"], frozen["expert_b"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = json.loads((P0L1_OUT / "aggregate_result.json").read_text())

    report = {"analysis": "routing information ceiling: same feature, stronger learner",
              "test_used": False, "existing_test_scored": False, "new_dataset_used": False,
              "diagnostic_spec_sha256": diagnostic["spec_sha256"],
              "final_candidate_sha256": frozen["spec_sha256"],
              "p0l1_final_candidate_sha256_reference": reference["final_candidate_sha256"],
              "fold_boundary_sha256": manifest["fold_boundary_sha256"],
              "fold_boundary_sha256_reference": reference["fold_boundary_sha256"],
              "expert_pair": [a, b], "started_at_utc": _utc(), "datasets": {}}

    tables_by_dataset, fold_rows, parity, training_fit = {}, [], {}, {}
    crossfitted_frames, gate_frames = {}, {}

    for name in args.datasets:
        block = manifest["datasets"][name]
        tables = {}
        for fold in block["folds"]:
            tables[fold["fold"]] = fold_table(name, fold, a, b, device)
            print(f"[{name}] fold {fold['fold']}: {len(tables[fold['fold']]):,} origins", flush=True)
        tables_by_dataset[name] = tables
        columns = F.feature_columns(frozen["feature_variant"], list(tables[0].columns))

        results, crossfitted, gates, parity[name], training_fit[name] = {}, [], [], [], []
        for fold in block["folds"][1:]:
            k = fold["fold"]
            past = pd.concat([tables[j] for j in range(k)], ignore_index=True)
            held = tables[k]
            alpha = fit_alpha(past)
            medians = F.fit_imputer(past, F.STRUCTURE_COLUMNS)
            xp = F.apply_imputer(past, medians)[columns].to_numpy(np.float64)
            centre, spread = xp.mean(0), np.where(xp.std(0) > 0, xp.std(0), 1.0)
            # One matrix object, handed to both learners: parity by construction.
            x_train = ((xp - centre) / spread).astype(np.float32)
            xh = F.apply_imputer(held, medians)[columns].to_numpy(np.float64)
            x_valid = ((xh - centre) / spread).astype(np.float32)

            mlp = fit_direct(x_train, past, device)
            hgb, n_fit, n_flat_fit = fit_hgb(x_train, past)

            g_mlp = G.gate_weights(mlp, x_valid, device).astype(np.float64)
            g_hgb = predict_hgb(hgb, x_valid)
            g_mlp_train = G.gate_weights(mlp, x_train, device).astype(np.float64)
            g_hgb_train = predict_hgb(hgb, x_train)

            e_alpha = mixture_mse(held, np.full(len(held), alpha))
            e_mlp = mixture_mse(held, g_mlp)
            e_hgb = mixture_mse(held, g_hgb)
            e_convex = held["mse_convex"].to_numpy(np.float64)

            parity[name].append({
                "fold": k, "n_features": len(columns),
                "train_rows": int(len(past)), "valid_rows": int(len(held)),
                "x_train_sha256": hashlib.sha256(x_train.tobytes()).hexdigest(),
                "x_valid_sha256": hashlib.sha256(x_valid.tobytes()).hexdigest(),
                "shared_matrix_object": True,
                "n_flat_excluded_from_fit": n_flat_fit,
                "n_rows_fitted": n_fit,
                "n_flat_in_validation": int((held["den"].to_numpy() <= FLAT_THRESHOLD).sum())})

            e_train_mlp = float(mixture_mse(past, g_mlp_train).mean())
            e_train_hgb = float(mixture_mse(past, g_hgb_train).mean())
            e_train_alpha = float(mixture_mse(past, np.full(len(past), alpha)).mean())
            training_fit[name].append({
                "fold": k, "rows": int(len(past)), "E_train_alpha": e_train_alpha,
                "E_train_mlp": e_train_mlp, "E_train_hgb": e_train_hgb,
                "train_fit_gain": float((e_train_mlp - e_train_hgb) / e_train_mlp)})

            gstar = held["g_star"].to_numpy(np.float64)
            live = held["den"].to_numpy(np.float64) > FLAT_THRESHOLD
            weight = (held["den"].to_numpy(np.float64) / held["n"].to_numpy(np.float64))[live]
            u = (held["num"].to_numpy(np.float64) / held["den"].to_numpy(np.float64))[live]

            def weighted_rmse(g, target):
                return float(np.sqrt(np.sum(weight * (g[live] - target) ** 2) / np.sum(weight)))

            row = {"dataset": name, "fold": k, "n_origins": int(len(held)), "alpha": alpha,
                   "E_alpha": float(e_alpha.mean()), "E_mlp": float(e_mlp.mean()),
                   "E_hgb": float(e_hgb.mean()), "E_convex": float(e_convex.mean()),
                   "gain_capacity": float((e_mlp.mean() - e_hgb.mean()) / e_mlp.mean()),
                   "gain_static_hgb": float((e_alpha.mean() - e_hgb.mean()) / e_alpha.mean()),
                   "gain_static_mlp": float((e_alpha.mean() - e_mlp.mean()) / e_alpha.mean()),
                   "recovery_mlp": recovery(e_alpha.mean(), e_mlp.mean(), e_convex.mean()),
                   "recovery_hgb": recovery(e_alpha.mean(), e_hgb.mean(), e_convex.mean()),
                   "mean_g_mlp": float(g_mlp.mean()), "mean_g_hgb": float(g_hgb.mean()),
                   "std_g_mlp": float(g_mlp.std()), "std_g_hgb": float(g_hgb.std()),
                   "corr_mlp_gstar": float(np.corrcoef(g_mlp, gstar)[0, 1]),
                   "corr_hgb_gstar": float(np.corrcoef(g_hgb, gstar)[0, 1]),
                   "spearman_mlp_gstar": float(spearmanr(g_mlp, gstar).statistic),
                   "spearman_hgb_gstar": float(spearmanr(g_hgb, gstar).statistic),
                   "wrmse_mlp_u": weighted_rmse(g_mlp, u),
                   "wrmse_hgb_u": weighted_rmse(g_hgb, u),
                   "wrmse_mlp_gstar": weighted_rmse(g_mlp, gstar[live]),
                   "wrmse_hgb_gstar": weighted_rmse(g_hgb, gstar[live]),
                   "train_fit_gain": training_fit[name][-1]["train_fit_gain"]}
            results[k] = row
            fold_rows.append(row)
            crossfitted.append(pd.DataFrame({
                "series_id": held["series_id"].to_numpy(), "fold": k,
                "alpha_loss": e_alpha, "mlp": e_mlp, "hgb": e_hgb, "convex": e_convex}))
            gates.append(pd.DataFrame({
                "series_id": held["series_id"].to_numpy(), "fold": k,
                "g_mlp": g_mlp, "g_hgb": g_hgb, "g_star": gstar,
                "den": held["den"].to_numpy(np.float64), "n": held["n"].to_numpy(np.float64)}))
            print(f"   fold {k}: alpha {alpha:.2f}  hgb-vs-mlp {row['gain_capacity']*100:+.3f}%  "
                  f"hgb-vs-alpha {row['gain_static_hgb']*100:+.3f}%  "
                  f"train gain {row['train_fit_gain']*100:+.2f}%", flush=True)

        joined = pd.concat(crossfitted, ignore_index=True)
        joined.to_parquet(OUT / f"crossfitted_{name}.parquet", index=False)
        pd.concat(gates, ignore_index=True).to_parquet(
            OUT / f"crossfitted_gate_{name}.parquet", index=False)
        crossfitted_frames[name] = joined
        gate_frames[name] = pd.concat(gates, ignore_index=True)

        per_series = joined.groupby("series_id")[["alpha_loss", "mlp", "hgb", "convex"]].mean()
        per_series.to_csv(OUT / f"per_series_{name}.csv")
        e = {c: float(per_series[c].mean()) for c in ("alpha_loss", "mlp", "hgb", "convex")}
        report["datasets"][name] = {
            "n_validation_folds": len(results),
            "warn_limited_temporal_folds": block["warn_limited"],
            "n_series": int(len(per_series)),
            "fold_results": results,
            "aggregate": {"E_alpha": e["alpha_loss"], "E_mlp": e["mlp"],
                          "E_hgb": e["hgb"], "E_convex": e["convex"],
                          "hgb_vs_mlp": float((e["mlp"] - e["hgb"]) / e["mlp"]),
                          "hgb_vs_alpha": float((e["alpha_loss"] - e["hgb"]) / e["alpha_loss"]),
                          "mlp_vs_alpha": float((e["alpha_loss"] - e["mlp"]) / e["alpha_loss"])},
            "scales": {"hgb_vs_mlp": three_scales(per_series, "hgb", "mlp"),
                       "hgb_vs_alpha": three_scales(per_series, "hgb", "alpha_loss"),
                       "mlp_vs_alpha": three_scales(per_series, "mlp", "alpha_loss")},
            "bootstrap": {"hgb_vs_mlp": boot(per_series, "hgb", "mlp"),
                          "hgb_vs_alpha": boot(per_series, "hgb", "alpha_loss"),
                          "mlp_vs_alpha": boot(per_series, "mlp", "alpha_loss")},
            "oracle_recovery": {"mlp": recovery(e["alpha_loss"], e["mlp"], e["convex"]),
                                "hgb": recovery(e["alpha_loss"], e["hgb"], e["convex"])},
        }

    pd.DataFrame(fold_rows).to_csv(OUT / "per_fold_results.csv", index=False)
    (OUT / "feature_parity.json").write_text(json.dumps(
        {"per_dataset": parity,
         "note": ("both learners receive the identical standardized G-NOSCALE "
                  "matrix; the hashes are of that one array")}, indent=2, default=str))
    (OUT / "training_fit_diagnostic.json").write_text(json.dumps(training_fit, indent=2, default=str))

    equivalence = equivalence_test(tables_by_dataset)
    (OUT / "quadratic_equivalence_test.json").write_text(json.dumps(equivalence, indent=2, default=str))

    report["method_identity"] = method_identity(frozen)
    report["mlp_reference_check"] = {
        name: {"reproduced_mlp_vs_alpha": report["datasets"][name]["aggregate"]["mlp_vs_alpha"],
               "p0l1_artifact_improvement": reference["datasets"][name]["bootstrap"][
                   "relative_improvement"],
               "abs_difference": abs(report["datasets"][name]["aggregate"]["mlp_vs_alpha"]
                                     - reference["datasets"][name]["bootstrap"][
                                         "relative_improvement"]),
               "reproduced": bool(abs(report["datasets"][name]["aggregate"]["mlp_vs_alpha"]
                                      - reference["datasets"][name]["bootstrap"][
                                          "relative_improvement"]) < 1e-9)}
        for name in report["datasets"] if name in reference["datasets"]}
    report["seal_verification"] = verify_seal(seal)
    report["quadratic_equivalence_passed"] = equivalence["passed"]
    report["finished_at_utc"] = _utc()
    report["git_commit"] = cli._git_commit()
    (OUT / "aggregate_results.json").write_text(json.dumps(report, indent=2, default=str))

    print()
    for name, blockk in report["datasets"].items():
        agg = blockk["aggregate"]
        print(f"{name:<15} hgb-vs-mlp {agg['hgb_vs_mlp']*100:+8.3f}%   "
              f"hgb-vs-alpha {agg['hgb_vs_alpha']*100:+8.3f}%   "
              f"mlp-vs-alpha {agg['mlp_vs_alpha']*100:+8.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser("routing information ceiling diagnostic")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=list(DATASETS))
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
