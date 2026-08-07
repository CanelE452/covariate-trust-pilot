"""Confirm Gate-v2 on series whose test outcome has never been computed.

The M5 and Favorita test splits Gate-v1 used are burned: their numbers have been
read, and reusing them would let that reading leak into a design decision.  This
audits every artifact that ever produced a test metric, subtracts those series,
and evaluates the frozen Gate-v2 on what is left.

Nothing is fitted here.  The experts are the Stage A checkpoints, the gate is
the frozen GateSpec-v2, and the fresh series appear in no training split of
either.  Selection of the fresh set uses eligibility and exposure only -- never
a forecast, an error, or a gate weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..external_validity_screen import cli, confirmatory_h2 as conf, posthoc, prereg
from ..external_validity_screen import rule_replication as rr, screen
from ..external_validity_screen import favorita_independent as fi
from ..om_factorization_killtest import models as km_models
from ..om_factorization_killtest import train as km_train
from ..unified_temporal_27_v3.training import train_scale
from . import convex_oracle as CO, features as F, gate as G
from . import gate_v2 as V2, killtest as KT
from .oof import OUT

MANIFEST = OUT / "fresh_manifest.json"
DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(values) -> str:
    return hashlib.sha256("\n".join(sorted(str(v) for v in values)).encode()).hexdigest()


def exposure_audit(name: str) -> dict:
    """Every series for which a test metric was ever computed, by artifact."""
    E = screen.OUT
    exposed, sources = set(), {}

    stage_a = set(pd.read_csv(E / "per_series_metrics.csv")
                  .query("dataset == @name")["series_id"].astype(str))
    sources["stage_a"] = len(stage_a)
    exposed |= stage_a
    # Stage A's series are also what the classical benchmark and Gate-v1 test used.
    sources["classical_benchmark"] = len(stage_a)
    sources["gate_v1_test"] = len(stage_a)

    if name == "m5":
        population = pd.read_csv(E / "rule_replication" / "independent_population.csv")
        h2 = set(population["series_id"].astype(str))
        sources["h2_independent_m5"] = len(h2)
        exposed |= h2
    else:
        transfer = pd.read_csv(E / "favorita_transfer" / "transfer_population.csv")
        independent = pd.read_csv(E / "favorita_independent" / "independent_population.csv")
        block = set(transfer["series_id"].astype(str)) | set(independent["series_id"].astype(str))
        sources["favorita_transfer_and_independent"] = len(block)
        exposed |= block

    seed_rows = OUT / f"test_gate_weights_{name}.csv"
    if seed_rows.exists():
        sources["gate_v1_test_weights"] = int(
            pd.read_csv(seed_rows)["series_id"].astype(str).nunique())
    return {"n_exposed": len(exposed), "by_source": sources, "_exposed": exposed}


def full_population(name: str) -> dict:
    """Every series the dataset offers, in load_dataset's shape."""
    if name == "m5":
        return conf.m5_full()
    return fi.load_pool()


def cmd_audit(args) -> None:
    report = {"analysis": "fresh-holdout exposure audit", "audited_at_utc": _utc(),
              "datasets": {}}
    for name in args.datasets:
        audit = exposure_audit(name)
        exposed = audit.pop("_exposed")
        data = full_population(name)
        cfg = screen.config_for(name)
        rows = [screen.describe_series(data["y"][i], int(data["available_from"][i]),
                                       cfg.train_end)
                for i in range(len(data["series_id"]))]
        table = pd.DataFrame(rows)
        table["series_id"] = data["series_id"]
        eligible = table[table["n_positive_train"] >= prereg.ELIGIBILITY["primary_threshold"]]
        fresh = sorted(set(eligible["series_id"].astype(str)) - exposed)
        audit["n_total"] = int(len(table))
        audit["n_eligible"] = int(len(eligible))
        audit["n_fresh"] = len(fresh)
        audit["fresh_available"] = len(fresh) > 0
        audit["selection_inputs"] = ["descriptor eligibility (n_positive_train >= 20)",
                                     "exposure to any previously computed test metric"]
        audit["fresh_id_sha256"] = _sha(fresh)
        report["datasets"][name] = audit
        pd.DataFrame({"series_id": fresh}).to_csv(OUT / f"fresh_ids_{name}.csv", index=False)
        print(f"[{name}] total {len(table):,} eligible {len(eligible):,} "
              f"exposed {audit['n_exposed']:,} -> fresh {len(fresh):,}")
    report["verdict"] = ("FRESH_HOLDOUT_AVAILABLE"
                         if all(b["fresh_available"] for b in report["datasets"].values())
                         else "FRESH_HOLDOUT_NOT_AVAILABLE")
    report["git_commit"] = cli._git_commit()
    MANIFEST.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": report["verdict"]}, indent=2))


def cmd_freeze(args) -> None:
    """Fit the selected GateSpec on every OOF fold and freeze it, before fresh scoring."""
    if V2.SPEC.exists():
        raise SystemExit(f"{V2.SPEC} already frozen; refusing to overwrite")
    selection = json.loads((OUT / "gate_v2_selection.json").read_text())
    manifest = json.loads((OUT / "oof_manifest.json").read_text())
    chosen = selection["selected"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = {"study": "Gate-v2, frozen for fresh-holdout confirmation",
            "frozen_at_utc": _utc(), "selected": chosen,
            "selection_criterion": selection["selection_criterion"],
            "cv_scheme": selection["cv_scheme"],
            "fitted_on": "all OOF folds; the cross-fold estimates that chose it came "
                         "from gates fitted on strictly earlier folds",
            "experts_frozen": True, "existing_test_reused": False,
            "gradient_checks": selection["gradient_checks"], "datasets": {}}
    for name in args.datasets:
        frame = V2.assemble_v2(name, manifest)
        columns = F.feature_columns(chosen["variant"], list(frame.columns))
        medians = F.fit_imputer(frame, F.STRUCTURE_COLUMNS)
        filled = F.apply_imputer(frame, medians)
        x = filled[columns].to_numpy(np.float64)
        centre = x.mean(0)
        spread = np.where(x.std(0) > 0, x.std(0), 1.0)
        xz = ((x - centre) / spread).astype(np.float32)
        model = V2.fit(chosen["loss"], xz, frame, chosen["architecture"], device)
        torch.save({"state_dict": model.state_dict(), "architecture": chosen["architecture"],
                    "variant": chosen["variant"], "loss": chosen["loss"], "columns": columns,
                    "centre": centre.tolist(), "spread": spread.tolist(),
                    "medians": medians}, OUT / f"gate_v2_{name}.pt")
        spec["datasets"][name] = {
            "checkpoint": f"gate_v2_{name}.pt", "columns": columns,
            "centre": centre.tolist(), "spread": spread.tolist(), "medians": medians,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "expert_checkpoints": str(G.FINAL_CHECKPOINTS[name].relative_to(screen.REPO)),
            "expert_trained_on": G.TRAINED_ON[name]}
        print(f"[{name}] frozen {len(columns)} features, "
              f"{spec['datasets'][name]['n_parameters']} parameters")
    spec["git_commit"] = cli._git_commit()
    V2.SPEC.write_text(json.dumps(spec, indent=2, default=str))
    print(f"froze {V2.SPEC}")


def score_fresh(name: str, ids: list[str], device) -> dict:
    """Frozen experts on fresh series; no fitting of any kind happens here."""
    cfg = screen.config_for(name)
    pool = full_population(name)
    index = pd.Index(pool["series_id"]).get_indexer(ids)
    if (index < 0).any():
        raise screen.ScreenFailure("a fresh id is absent from the source pool")
    data = {"name": name, "y": pool["y"][index], "z": pool["z"][index],
            "series_id": np.asarray(ids), "available_from": pool["available_from"][index],
            "first_positive": pool["first_positive"][index]}
    windows = rr.test_windows_only(data, cfg, prereg.SPLITS[name]["test_origin_stride"])
    predictions = {}
    for role in ("point", "hurdle"):
        payload = torch.load(G.FINAL_CHECKPOINTS[name] / f"{role}.pt", map_location=device,
                             weights_only=False)
        if payload["trained_on"] != G.TRAINED_ON[name]:
            raise screen.ScreenFailure("unexpected expert training population")
        model = km_models.BUILDERS[payload["builder_key"]](payload["lookback"],
                                                           payload["horizon"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        predictions[role] = km_train.predict(model, windows, device)
    scale = pd.Series(train_scale({"y": data["y"], "z": data["z"]}, cfg),
                      index=pd.Index(data["series_id"]).astype(str))
    return {"data": data, "windows": windows, "predictions": predictions, "scale": scale}


def cmd_confirm(args) -> None:
    if not V2.SPEC.exists():
        raise SystemExit("freeze the GateSpec-v2 before scoring fresh series")
    spec = json.loads(V2.SPEC.read_text())
    audit = json.loads(MANIFEST.read_text())
    convex = json.loads((OUT / "convex_oracle.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = _utc()
    if not (spec["frozen_at_utc"] < started):
        raise screen.ScreenFailure("GateSpec timestamp is not before fresh scoring")

    report = {"analysis": "Gate-v2 fresh-holdout confirmation", "scored_at_utc": started,
              "spec_frozen_at_utc": spec["frozen_at_utc"], "pooled": False,
              "existing_test_reused": False, "datasets": {}}

    for name in args.datasets:
        ids = pd.read_csv(OUT / f"fresh_ids_{name}.csv")["series_id"].astype(str).tolist()
        if _sha(ids) != audit["datasets"][name]["fresh_id_sha256"]:
            raise screen.ScreenFailure("fresh ids do not match the frozen audit")
        if args.limit:
            ids = ids[:args.limit]
        print(f"[{name}] scoring {len(ids):,} fresh series ...", flush=True)
        scored = score_fresh(name, ids, device)
        windows, predictions = scored["windows"], scored["predictions"]
        n_series, horizon = len(ids), windows.target.shape[1]
        n_origins = windows.n_origins
        series_axis = np.repeat(np.arange(n_series), n_origins)

        frame = pd.DataFrame({
            "series_id": np.repeat(scored["data"]["series_id"][series_axis], horizon),
            "origin": np.repeat(np.tile(np.asarray(windows.origins), n_series), horizon),
            "y_observed": windows.target.reshape(-1),
            "target_mask": windows.target_mask.reshape(-1),
            "point": predictions["point"]["mean_prediction"].reshape(-1),
            "hurdle": predictions["hurdle"]["mean_prediction"].reshape(-1),
            "hurdle_p": predictions["hurdle"]["p_prediction"].reshape(-1),
            "hurdle_mu": predictions["hurdle"]["mu_prediction"].reshape(-1)})

        origin_table = frame[["series_id", "origin"]].drop_duplicates().reset_index(drop=True)
        bundle = {"origin_table": origin_table, "data": scored["data"],
                  "expert_state": KT.expert_state_at_test(frame, scored["scale"])}
        block = spec["datasets"][name]
        inputs = KT.gate_inputs(bundle, block["medians"])
        checkpoint = torch.load(OUT / block["checkpoint"], map_location=device,
                                weights_only=False)
        model = G.build_gate(checkpoint["architecture"], len(checkpoint["columns"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        g = KT.apply_gate(inputs, checkpoint["columns"], block["centre"], block["spread"],
                          model, device)
        origin_table["g"] = g

        keys = pd.MultiIndex.from_arrays([frame["series_id"], frame["origin"]])
        pairs = pd.MultiIndex.from_arrays([origin_table["series_id"], origin_table["origin"]])
        frame["g"] = pd.Series(g, index=pairs).reindex(keys).to_numpy()
        frame["gate_v2"] = (1 - frame["g"]) * frame["point"] + frame["g"] * frame["hurdle"]
        frame["fifty"] = 0.5 * (frame["point"] + frame["hurdle"])
        alpha = float(convex["datasets"][name]["global_alpha"])
        frame["global_alpha"] = (1 - alpha) * frame["point"] + alpha * frame["hurdle"]

        v1 = torch.load(OUT / f"gate_{name}.pt", map_location=device, weights_only=False)
        v1_inputs = F.apply_imputer(inputs, v1["medians"])
        v1_model = G.build_gate(v1["architecture"], len(v1["columns"])).to(device)
        v1_model.load_state_dict(v1["state_dict"])
        v1_model.eval()
        v1_spec = json.loads(G.SPEC.read_text())["datasets"][name]
        g1 = KT.apply_gate(v1_inputs, v1["columns"], v1_spec["centre"], v1_spec["spread"],
                           v1_model, device)
        frame["g_v1"] = pd.Series(g1, index=pairs).reindex(keys).to_numpy()
        frame["gate_v1"] = (1 - frame["g_v1"]) * frame["point"] + frame["g_v1"] * frame["hurdle"]

        methods = ["point", "hurdle", "fifty", "global_alpha", "gate_v1", "gate_v2"]
        per_series = KT.per_series_metrics(frame, methods)

        observed = frame[frame["target_mask"] > 0].copy()
        s = np.maximum(scored["scale"].loc[observed["series_id"]].to_numpy(np.float64), 1e-9)
        observed["d"] = (observed["hurdle"] - observed["point"]) / s
        observed["r"] = (observed["y_observed"] - observed["point"]) / s
        grouped = observed.groupby(["series_id", "origin"])
        num = grouped.apply(lambda d: float((d["r"] * d["d"]).sum()), include_groups=False)
        den = grouped.apply(lambda d: float((d["d"] * d["d"]).sum()), include_groups=False)
        gstar = np.clip(np.divide(num.to_numpy(),
                                  np.where(den.to_numpy() <= CO.FLAT_THRESHOLD, 1.0,
                                           den.to_numpy())), 0, 1)
        gstar = np.where(den.to_numpy() <= CO.FLAT_THRESHOLD, 0.5, gstar)
        se_p = grouped.apply(lambda d: float(((d["point"] - d["y_observed"]) ** 2).sum()),
                             include_groups=False)
        se_h = grouped.apply(lambda d: float(((d["hurdle"] - d["y_observed"]) ** 2).sum()),
                             include_groups=False)
        pick = pd.Series(se_h.to_numpy() < se_p.to_numpy(), index=se_p.index)
        gmap = pd.Series(gstar, index=se_p.index)
        observed["oracle_hard"] = np.where(
            pick.reindex(pd.MultiIndex.from_arrays(
                [observed["series_id"], observed["origin"]])).to_numpy(),
            observed["hurdle"], observed["point"])
        gg = gmap.reindex(pd.MultiIndex.from_arrays(
            [observed["series_id"], observed["origin"]])).to_numpy()
        observed["oracle_convex"] = (1 - gg) * observed["point"] + gg * observed["hurdle"]
        oracle_series = KT.per_series_metrics(observed, ["oracle_hard", "oracle_convex"])
        per_series = per_series.join(oracle_series)

        table = {m: KT.overall(frame, m) | {
            "mean_per_series_rmse": float(per_series["rmse_" + m].mean()),
            "median_per_series_rmse": float(per_series["rmse_" + m].median()),
            "mean_per_series_mae": float(per_series["mae_" + m].mean())} for m in methods}
        for m in ("oracle_hard", "oracle_convex"):
            table[m] = KT.overall(observed, m) | {
                "mean_per_series_rmse": float(per_series["rmse_" + m].mean()),
                "median_per_series_rmse": float(per_series["rmse_" + m].median()),
                "upper_bound_only": True}

        best_static = min(("point", "hurdle", "fifty", "global_alpha"),
                          key=lambda m: table[m]["overall_rmse"])
        comparisons = {"gate_v2_vs_" + m: KT.paired_bootstrap(per_series, "gate_v2", m)
                       for m in ("point", "hurdle", "fifty", "global_alpha", "gate_v1")}
        comparisons["gate_v2_vs_best_static"] = comparisons["gate_v2_vs_" + best_static]
        denominator = table[best_static]["overall_rmse"] - table["oracle_convex"]["overall_rmse"]
        recovery = ((table[best_static]["overall_rmse"] - table["gate_v2"]["overall_rmse"])
                    / denominator if denominator > 0 else "CONVEX_GAP_UNAVAILABLE")

        origin_table.to_csv(OUT / f"fresh_gate_weights_{name}.csv", index=False)
        per_series.to_csv(OUT / f"fresh_per_series_{name}.csv")
        report["datasets"][name] = {
            "n_series": int(len(ids)), "origins": windows.origins.tolist(),
            "fresh_id_sha256": _sha(ids),
            "deployable": {m: table[m] for m in methods},
            "oracle_upper_bounds": {m: table[m] for m in ("oracle_hard", "oracle_convex")},
            "best_static": best_static, "paired_bootstrap": comparisons,
            "convex_oracle_recovery": recovery,
            "gate_weight": {"mean": float(g.mean()), "sd": float(g.std()),
                            "extreme_share": float(((g < 0.05) | (g > 0.95)).mean())}}
        print(f"[{name}] best_static={best_static} "
              f"gate_v2={table['gate_v2']['overall_rmse']:.4f} "
              f"static={table[best_static]['overall_rmse']:.4f} recovery={recovery}")

    directions = {n: b["paired_bootstrap"]["gate_v2_vs_best_static"]["relative_improvement"] > 0
                  for n, b in report["datasets"].items()}
    excludes = {n: (b["paired_bootstrap"]["gate_v2_vs_best_static"]["ci_excludes_zero"]
                    and b["paired_bootstrap"]["gate_v2_vs_best_static"]["relative_improvement"] > 0)
                for n, b in report["datasets"].items()}
    reversal = {n: b["paired_bootstrap"]["gate_v2_vs_best_static"]["relative_improvement"] < -0.005
                for n, b in report["datasets"].items()}
    if not any(directions.values()):
        verdict = "GATE_V2_CONFIRM_FAIL"
    elif all(directions.values()) and any(excludes.values()) and not any(reversal.values()):
        verdict = "GATE_V2_CONFIRM_GREEN"
    else:
        verdict = "GATE_V2_CONFIRM_PARTIAL"
    report["verdict"] = verdict
    report["direction_positive"] = directions
    report["ci_excludes_zero"] = excludes
    report["git_commit"] = cli._git_commit()
    (OUT / "fresh_confirmatory.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "direction": directions,
                      "ci_excludes_zero": excludes}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("fresh holdout")
    sub = parser.add_subparsers(required=True)
    a = sub.add_parser("audit")
    a.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    a.set_defaults(func=cmd_audit)
    f = sub.add_parser("freeze")
    f.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    f.set_defaults(func=cmd_freeze)
    c = sub.add_parser("confirm")
    c.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    c.add_argument("--limit", type=int, default=0)
    c.set_defaults(func=cmd_confirm)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
