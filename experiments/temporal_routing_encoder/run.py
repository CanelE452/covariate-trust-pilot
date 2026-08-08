"""Does reading the raw window recover routing the summaries threw away?

Every earlier step on this axis kept the handcrafted G-NOSCALE descriptors and
changed something else: the loss, the parameterisation, the shrinkage, the
learner.  The last diagnostic said the features, not the learner, were the
binding constraint.  So this changes the representation and only the
representation -- a GRU over the raw window plus the two expert forecasts --
against the same experts, folds, objective and normalisation.

The stop rule is in spec.py and was fixed before the first error was computed.
If this does not clear it, the gate architecture axis ends here.
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

from ..expert_diversity.oof_experts import OUT as DIVERSITY_OUT
from ..external_validity_screen import cli, prereg, screen
from ..gate_v3.diagnose import fit_alpha, mixture_mse
from ..p0l1_robustness.folds import DATASETS, OUT as P0L1_OUT
from ..p0l1_robustness.run import freeze_spec
from ..routing_information_ceiling.run import OUT as RIC_OUT
from . import spec as S
from .data import fold_bundle
from .model import count_parameters, fit_sequence_gate, gate_weights

OUT = screen.OUT.parent / "temporal_routing_encoder"
DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]

DEPENDENCY_ARTIFACTS = [
    "results/expert_diversity/expert_set_spec.json",
    "results/gate_p0l1_robustness/expanded_fold_manifest.json",
    "results/gate_p0l1_robustness/final_candidate_spec.json",
    "results/gate_p0l1_robustness/aggregate_result.json",
    "results/routing_information_ceiling/aggregate_results.json",
    "results/routing_information_ceiling/crossfitted_m5.parquet",
    "results/routing_information_ceiling/crossfitted_favorita.parquet",
    "results/routing_information_ceiling/crossfitted_freshretailnet.parquet",
    "results/routing_information_ceiling/crossfitted_uci.parquet",
    "results/gate_p0l1_robustness/crossfitted_m5.parquet",
    "results/gate_p0l1_robustness/crossfitted_favorita.parquet",
    "results/gate_p0l1_robustness/crossfitted_freshretailnet.parquet",
    "results/gate_p0l1_robustness/crossfitted_uci.parquet",
    "data/processed/series.parquet",
    "data/processed/favorita_series.parquet",
    "data/processed/freshretailnet_grid.npz",
    "data/processed/uci_grid.npz",
    "data/calendar.csv",
    "data/sell_prices.csv",
    "experiments/p0l1_robustness/run.py",
    "experiments/p0l1_robustness/folds.py",
    "experiments/gate_v3/oof.py",
    "experiments/gate_v3/diagnose.py",
    "experiments/multi_benchmark/run.py",
    "experiments/unified_temporal_27_v3/training.py",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency_seal() -> dict:
    root = screen.REPO
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
    root = screen.REPO
    changed = [rel for rel, meta in seal["artifacts"].items()
               if not (root / rel).exists() or _sha(root / rel) != meta["sha256"]]
    return {"n_checked": len(seal["artifacts"]), "changed": changed,
            "passed": not changed and not seal["missing"]}


def repository_change_warning(seal: dict) -> dict:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=screen.REPO,
                             capture_output=True, text=True, check=True).stdout
    except Exception as exc:                              # pragma: no cover
        return {"available": False, "error": str(exc)}
    entries = [{"status": line[:2].strip(), "path": line[3:].strip(),
                "is_scientific_dependency": line[3:].strip() in seal["artifacts"]}
               for line in out.splitlines()]
    unrelated = [e for e in entries if not e["is_scientific_dependency"]]
    return {"available": True, "n_entries": len(entries), "n_unrelated": len(unrelated),
            "entries": entries,
            "note": "unrelated paths are recorded but never promote to a dependency failure"}


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


def three_scales(per_series: pd.DataFrame, a: str, b: str) -> dict:
    x, y = per_series[a].to_numpy(np.float64), per_series[b].to_numpy(np.float64)
    return {"normalized_mse": float((y.mean() - x.mean()) / y.mean()),
            "mean_per_series_relative": float(np.mean((y - x) / np.maximum(y, 1e-12))),
            "rmse": float((np.sqrt(y.mean()) - np.sqrt(x.mean())) / np.sqrt(y.mean()))}


def recovery(e_alpha: float, e_model: float, e_convex: float):
    denominator = e_alpha - e_convex
    return None if denominator <= 0 else float((e_alpha - e_model) / denominator)


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate_spec = S.sequence_gate_spec()
    (OUT / "sequence_gate_spec.json").write_text(json.dumps(gate_spec, indent=2, default=str))
    seal = dependency_seal()
    (OUT / "scientific_dependency_seal.json").write_text(json.dumps(seal, indent=2, default=str))
    (OUT / "repository_change_warning.json").write_text(
        json.dumps(repository_change_warning(seal), indent=2, default=str))

    manifest = json.loads((P0L1_OUT / "expanded_fold_manifest.json").read_text())
    (OUT / "fold_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    frozen = freeze_spec()
    a, b = frozen["expert_a"], frozen["expert_b"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = {"analysis": "raw-history sequence routing gate against the handcrafted P0L1 gate",
              "test_used": False, "existing_test_scored": False, "new_dataset_used": False,
              "sequence_gate_sha256": gate_spec["spec_sha256"],
              "expert_pair": [a, b],
              "fold_boundary_sha256": manifest["fold_boundary_sha256"],
              "device": str(device), "started_at_utc": _utc(), "datasets": {}}

    fold_rows, schema, parameters, row_parity, seed_check = [], {}, {}, {}, {}

    for name in args.datasets:
        block_manifest = manifest["datasets"][name]
        reference = pd.read_parquet(RIC_OUT / f"crossfitted_{name}.parquet")
        # The ADI quartiles come from P0L1's own artifact so no new cutoff is made.
        adi_source = pd.read_parquet(P0L1_OUT / f"crossfitted_{name}.parquet")

        bundles = {}
        for fold in block_manifest["folds"]:
            bundles[fold["fold"]] = fold_bundle(name, fold, a, b, device)
            print(f"[{name}] fold {fold['fold']}: {len(bundles[fold['fold']].block):,} origins",
                  flush=True)

        first = bundles[0]
        schema[name] = {
            "lookback": first.lookback, "horizon": first.horizon,
            "channels": [c["name"] for c in S.CHANNELS],
            "context_dim": int(first.context.shape[1]),
            "sequence_shape_example": list(first.sequence.shape),
            "scale_source_end_by_fold": {int(f["fold"]): int(f["train_end"])
                                         for f in block_manifest["folds"]},
            "max_history_timestamp_below_origin": bool(
                (first.origin - 1 < first.origin).all()),
        }

        results, crossfitted, gates = {}, [], []
        for fold in block_manifest["folds"][1:]:
            k = fold["fold"]
            past_blocks = [bundles[j] for j in range(k)]
            held = bundles[k]
            past_block = pd.concat([p.block for p in past_blocks], ignore_index=True)
            past_seq = np.concatenate([p.sequence for p in past_blocks], axis=0)
            past_ctx = np.concatenate([p.context for p in past_blocks], axis=0)
            alpha = fit_alpha(past_block)

            model = fit_sequence_gate(past_seq, past_ctx, past_block, device)
            if name not in parameters:
                parameters[name] = {"sequence_gate_params": count_parameters(model)}
            g = gate_weights(model, held.sequence, held.context, device)

            e_alpha = mixture_mse(held.block, np.full(len(held.block), alpha))
            e_seq = mixture_mse(held.block, g)
            e_convex = held.block["mse_convex"].to_numpy(np.float64)

            held_reference = reference[reference["fold"] == k].reset_index(drop=True)
            held_adi = adi_source[adi_source["fold"] == k].reset_index(drop=True)
            aligned = bool(len(held_reference) == len(held.block)
                           and np.array_equal(held_reference["series_id"].to_numpy().astype(str),
                                              held.series_id.astype(str)))
            adi_aligned = bool(len(held_adi) == len(held.block)
                               and np.array_equal(held_adi["series_id"].to_numpy().astype(str),
                                                  held.series_id.astype(str)))
            row_parity.setdefault(name, []).append(
                {"fold": k, "rows": int(len(held.block)),
                 "reference_rows": int(len(held_reference)),
                 "series_id_order_matches": aligned,
                 "adi_source_order_matches": adi_aligned,
                 "alpha_matches_reference": bool(np.allclose(
                     e_alpha, held_reference["alpha_loss"].to_numpy(), rtol=0, atol=1e-9))})
            if not (aligned and adi_aligned
                    and row_parity[name][-1]["alpha_matches_reference"]):
                raise RuntimeError(f"row parity failed for {name} fold {k}")
            e_mlp = held_reference["mlp"].to_numpy(np.float64)

            row = {"dataset": name, "fold": k, "n_origins": int(len(held.block)), "alpha": alpha,
                   "E_alpha": float(e_alpha.mean()), "E_p0l1": float(e_mlp.mean()),
                   "E_seq": float(e_seq.mean()), "E_convex": float(e_convex.mean()),
                   "seq_vs_alpha": float((e_alpha.mean() - e_seq.mean()) / e_alpha.mean()),
                   "seq_vs_p0l1": float((e_mlp.mean() - e_seq.mean()) / e_mlp.mean()),
                   "p0l1_vs_alpha": float((e_alpha.mean() - e_mlp.mean()) / e_alpha.mean()),
                   "recovery_p0l1": recovery(e_alpha.mean(), e_mlp.mean(), e_convex.mean()),
                   "recovery_seq": recovery(e_alpha.mean(), e_seq.mean(), e_convex.mean()),
                   "mean_g": float(g.mean()), "std_g": float(g.std()),
                   "min_g": float(g.min()), "max_g": float(g.max()),
                   "corr_g_gstar": float(np.corrcoef(
                       g, held.block["g_star"].to_numpy())[0, 1])}
            results[k] = row
            fold_rows.append(row)
            crossfitted.append(pd.DataFrame({
                "series_id": held.series_id.astype(str), "origin": held.origin, "fold": k,
                "alpha_loss": e_alpha, "p0l1": e_mlp, "seq": e_seq, "convex": e_convex,
                "g_seq": g, "g_star": held.block["g_star"].to_numpy(np.float64),
                "ADI_train": held_adi["ADI_train"].to_numpy(np.float64)}))
            gates.append(g)
            print(f"   fold {k}: alpha {alpha:.2f}  seq-vs-alpha {row['seq_vs_alpha']*100:+.3f}%  "
                  f"seq-vs-p0l1 {row['seq_vs_p0l1']*100:+.3f}%  mean g {row['mean_g']:.3f}",
                  flush=True)

        joined = pd.concat(crossfitted, ignore_index=True)
        joined.to_parquet(OUT / f"crossfitted_sequence_{name}.parquet", index=False)
        per_series = joined.groupby("series_id")[["alpha_loss", "p0l1", "seq", "convex"]].mean()
        per_series.to_csv(OUT / f"per_series_{name}.csv")
        e = {c: float(per_series[c].mean()) for c in per_series.columns}
        report["datasets"][name] = {
            "n_validation_folds": len(results),
            "warn_limited_temporal_folds": block_manifest["warn_limited"],
            "n_series": int(len(per_series)), "fold_results": results,
            "aggregate": {"E_alpha": e["alpha_loss"], "E_p0l1": e["p0l1"],
                          "E_seq": e["seq"], "E_convex": e["convex"],
                          "seq_vs_alpha": float((e["alpha_loss"] - e["seq"]) / e["alpha_loss"]),
                          "seq_vs_p0l1": float((e["p0l1"] - e["seq"]) / e["p0l1"]),
                          "p0l1_vs_alpha": float((e["alpha_loss"] - e["p0l1"]) / e["alpha_loss"])},
            "scales": {"seq_vs_alpha": three_scales(per_series, "seq", "alpha_loss"),
                       "seq_vs_p0l1": three_scales(per_series, "seq", "p0l1")},
            "bootstrap": {"seq_vs_alpha": boot(per_series, "seq", "alpha_loss"),
                          "seq_vs_p0l1": boot(per_series, "seq", "p0l1")},
            "oracle_recovery": {"p0l1": recovery(e["alpha_loss"], e["p0l1"], e["convex"]),
                                "seq": recovery(e["alpha_loss"], e["seq"], e["convex"])},
        }

        # Reproducibility rerun: the frozen policy is one canonical seed, so this
        # re-fits the last fold with the same seed and checks it lands identically.
        last = block_manifest["folds"][-1]["fold"]
        past_blocks = [bundles[j] for j in range(last)]
        repeat = fit_sequence_gate(
            np.concatenate([p.sequence for p in past_blocks], axis=0),
            np.concatenate([p.context for p in past_blocks], axis=0),
            pd.concat([p.block for p in past_blocks], ignore_index=True), device)
        again = gate_weights(repeat, bundles[last].sequence, bundles[last].context, device)
        seed_check[name] = {"fold": last, "max_abs_g_difference": float(
            np.max(np.abs(again - gates[-1]))), "identical": bool(
            np.allclose(again, gates[-1], rtol=0, atol=1e-6))}
        print(f"   reproducibility rerun fold {last}: max |dg| "
              f"{seed_check[name]['max_abs_g_difference']:.2e}", flush=True)

    pd.DataFrame(fold_rows).to_csv(OUT / "per_fold_results.csv", index=False)
    (OUT / "raw_sequence_schema.json").write_text(json.dumps(
        {"per_dataset": schema, "channels": S.CHANNELS,
         "excluded_inputs": S.EXCLUDED_INPUTS}, indent=2, default=str))
    (OUT / "row_parity.json").write_text(json.dumps(row_parity, indent=2, default=str))
    (OUT / "seed_stability.json").write_text(json.dumps(
        {"policy": S.SEED_POLICY, "reproducibility": seed_check}, indent=2, default=str))

    reference_report = json.loads((P0L1_OUT / "aggregate_result.json").read_text())
    report["p0l1_identity"] = {
        name: {"reproduced_p0l1_vs_alpha": report["datasets"][name]["aggregate"]["p0l1_vs_alpha"],
               "p0l1_artifact": reference_report["datasets"][name]["bootstrap"][
                   "relative_improvement"],
               "matches": bool(abs(report["datasets"][name]["aggregate"]["p0l1_vs_alpha"]
                                   - reference_report["datasets"][name]["bootstrap"][
                                       "relative_improvement"]) < 1e-9)}
        for name in report["datasets"] if name in reference_report["datasets"]}
    report["parameter_audit_partial"] = parameters
    report["seal_verification"] = verify_seal(seal)
    report["finished_at_utc"] = _utc()
    report["git_commit"] = cli._git_commit()
    (OUT / "aggregate_results.json").write_text(json.dumps(report, indent=2, default=str))

    print()
    for name, blockk in report["datasets"].items():
        agg = blockk["aggregate"]
        print(f"{name:<15} seq-vs-alpha {agg['seq_vs_alpha']*100:+8.3f}%   "
              f"seq-vs-p0l1 {agg['seq_vs_p0l1']*100:+8.3f}%   "
              f"p0l1-vs-alpha {agg['p0l1_vs_alpha']*100:+8.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser("raw-history temporal routing encoder")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=list(DATASETS))
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
