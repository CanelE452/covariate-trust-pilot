"""Open the test split once, with the GateSpec already frozen.

Refuses to run without `gate_spec.json`, so no test number can have influenced
the gate's features, architecture, loss or temperature.  The experts are the
same frozen checkpoints the empirical work verified, and nothing here retrains
anything.

Deployable methods and oracles are kept in separate tables.  The per-origin
oracle reads the test target to choose an expert; it bounds what routing could
achieve and is never a result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..external_validity_screen import cli, confirmatory_h2 as conf, prereg
from ..external_validity_screen import rule_replication as rr, screen
from ..om_factorization_killtest import models as km_models
from ..om_factorization_killtest import train as km_train
from ..unified_temporal_27_v3.training import train_scale
from . import features as F, gate as G
from .oof import OUT

DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]
COLLAPSE_SHARE = 0.90


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_expert(name: str, role: str, device):
    payload = torch.load(G.FINAL_CHECKPOINTS[name] / f"{role}.pt",
                         map_location=device, weights_only=False)
    if payload["trained_on"] != G.TRAINED_ON[name]:
        raise screen.ScreenFailure(
            f"{name}/{role}: unexpected training population {payload['trained_on']}")
    model = km_models.BUILDERS[payload["builder_key"]](payload["lookback"],
                                                       payload["horizon"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def test_frame(name: str, device) -> dict:
    """Per-(series, origin, step) test predictions from both frozen experts."""
    cfg = screen.config_for(name)
    data = screen.load_dataset(name)
    split = screen.build_split(data, cfg, prereg.SPLITS[name]["test_origin_stride"])
    windows = split.test
    n_series, horizon = data["y"].shape[0], windows.target.shape[1]
    n_origins = windows.n_origins
    series_axis = np.repeat(np.arange(n_series), n_origins)

    predictions, provenance = {}, {}
    for role in ("point", "hurdle"):
        model, payload = load_expert(name, role, device)
        predictions[role] = km_train.predict(model, windows, device)
        provenance[role] = {"trained_on": payload["trained_on"],
                            "model_seed": payload["model_seed"]}

    frame = pd.DataFrame({
        "series_id": np.repeat(data["series_id"][series_axis], horizon),
        "origin": np.repeat(np.tile(np.asarray(windows.origins), n_series), horizon),
        "step": np.tile(np.arange(horizon), len(series_axis)),
        "y_observed": windows.target.reshape(-1),
        "occurrence": windows.occurrence.reshape(-1),
        "target_mask": windows.target_mask.reshape(-1),
        "point": predictions["point"]["mean_prediction"].reshape(-1),
        "hurdle": predictions["hurdle"]["mean_prediction"].reshape(-1),
        "hurdle_p": predictions["hurdle"]["p_prediction"].reshape(-1),
        "hurdle_mu": predictions["hurdle"]["mu_prediction"].reshape(-1)})
    scale = pd.Series(train_scale({"y": data["y"], "z": data["z"]}, cfg),
                      index=pd.Index(data["series_id"]).astype(str))
    return {"frame": frame, "data": data, "cfg": cfg, "scale": scale,
            "provenance": provenance, "origins": windows.origins.tolist()}


def expert_state_at_test(frame: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """The same expert-state summaries the OOF table carried, recomputed here."""
    keys = ["series_id", "origin"]
    work = frame.copy()
    work["scale"] = np.maximum(scale.loc[work["series_id"]].to_numpy(np.float64), 1e-9)
    work["gap"] = work["hurdle"] - work["point"]
    grouped = work.groupby(keys)
    out = pd.DataFrame({
        "xs_point_mean": grouped.apply(lambda d: (d["point"] / d["scale"]).mean(),
                                       include_groups=False),
        "xs_hurdle_mean": grouped.apply(lambda d: (d["hurdle"] / d["scale"]).mean(),
                                        include_groups=False),
        "xs_abs_disagreement": grouped.apply(lambda d: (d["gap"].abs() / d["scale"]).mean(),
                                             include_groups=False),
        "xs_max_disagreement": grouped.apply(lambda d: (d["gap"].abs() / d["scale"]).max(),
                                             include_groups=False),
        "xs_signed_disagreement": grouped.apply(lambda d: (d["gap"] / d["scale"]).mean(),
                                                include_groups=False),
        "xs_hurdle_p_mean": grouped["hurdle_p"].mean(),
        "xs_hurdle_p_min": grouped["hurdle_p"].min(),
        "xs_hurdle_p_max": grouped["hurdle_p"].max(),
        "xs_hurdle_mu_mean": grouped.apply(lambda d: (d["hurdle_mu"] / d["scale"]).mean(),
                                           include_groups=False),
        "xs_hurdle_mu_min": grouped.apply(lambda d: (d["hurdle_mu"] / d["scale"]).min(),
                                          include_groups=False),
        "xs_hurdle_mu_max": grouped.apply(lambda d: (d["hurdle_mu"] / d["scale"]).max(),
                                          include_groups=False)}).reset_index()
    return out


def gate_inputs(bundle: dict, medians: dict) -> pd.DataFrame:
    """As-of-origin structure plus expert state, for every test origin."""
    pairs = bundle["origin_table"][["series_id", "origin"]]
    structure = F.structure_table(bundle["data"], pairs)
    merged = pairs.merge(structure, on=["series_id", "origin"], how="left")
    merged = merged.merge(bundle["expert_state"], on=["series_id", "origin"], how="left")
    merged = F.add_missing_flags(merged, F.STRUCTURE_COLUMNS)
    return F.apply_imputer(merged, medians)


def apply_gate(inputs: pd.DataFrame, columns: list[str], centre, spread,
               model, device) -> np.ndarray:
    x = inputs[columns].to_numpy(np.float64)
    xz = ((x - np.asarray(centre)) / np.asarray(spread)).astype(np.float32)
    return G.gate_weights(model, xz, device)


def per_series_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    observed = frame[frame["target_mask"] > 0]
    out = {}
    for column in columns:
        error = (observed[column].to_numpy(np.float64)
                 - observed["y_observed"].to_numpy(np.float64))
        block = pd.DataFrame({"series_id": observed["series_id"].to_numpy(),
                              "se": error ** 2, "ae": np.abs(error)})
        grouped = block.groupby("series_id")
        out[f"rmse_{column}"] = np.sqrt(grouped["se"].mean())
        out[f"mae_{column}"] = grouped["ae"].mean()
    return pd.DataFrame(out)


def overall(frame: pd.DataFrame, column: str) -> dict:
    observed = frame[frame["target_mask"] > 0]
    error = (observed[column].to_numpy(np.float64)
             - observed["y_observed"].to_numpy(np.float64))
    return {"overall_rmse": float(np.sqrt((error ** 2).mean())),
            "overall_mae": float(np.abs(error).mean())}


def paired_bootstrap(per_series: pd.DataFrame, a: str, b: str) -> dict:
    """Relative RMSE difference of a against b, resampling series."""
    x = per_series[f"rmse_{a}"].to_numpy(np.float64)
    y = per_series[f"rmse_{b}"].to_numpy(np.float64)
    rng = np.random.default_rng(SEED)
    n = len(x)
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        idx = rng.integers(0, n, n)
        draws[i] = (y[idx].mean() - x[idx].mean()) / y[idx].mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"relative_improvement": float((y.mean() - x.mean()) / y.mean()),
            "ci": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def frozen_rule_route(name: str, origin_table: pd.DataFrame) -> np.ndarray:
    """B3: the frozen H2 rule as a hard router. Candidate -> Point, else Hurdle."""
    cutoffs = conf.frozen_cutoffs_from_artifact()
    series = pd.read_csv(screen.OUT / "per_series_metrics.csv")
    block = series[series["dataset"] == name].set_index("series_id")
    have = block.index.intersection(origin_table["series_id"].unique())
    candidate = pd.Series(
        (block.loc[have, "ADI_train"].to_numpy(float) >= cutoffs["HIGH_ADI_min"])
        & (block.loc[have, "rho_interval_abs_train"].to_numpy(float) <= cutoffs["LOW_OCC_max"])
        & (block.loc[have, "rho_magnitude_train"].to_numpy(float) >= cutoffs["MAG_PERSISTENT_min"]),
        index=have)
    point_candidate = origin_table["series_id"].map(candidate).fillna(False).to_numpy()
    return np.where(point_candidate, 0.0, 1.0)


def oracle_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Upper bounds only. These read the test target to choose an expert."""
    observed = frame[frame["target_mask"] > 0].copy()
    observed["se_point"] = (observed["point"] - observed["y_observed"]) ** 2
    observed["se_hurdle"] = (observed["hurdle"] - observed["y_observed"]) ** 2
    keys = pd.MultiIndex.from_arrays([observed["series_id"], observed["origin"]])
    by_origin = observed.groupby(["series_id", "origin"])[["se_point", "se_hurdle"]].mean()
    by_series = observed.groupby("series_id")[["se_point", "se_hurdle"]].mean()
    pick_origin = (by_origin["se_hurdle"] < by_origin["se_point"]).reindex(keys).to_numpy()
    pick_series = observed["series_id"].map(
        by_series["se_hurdle"] < by_series["se_point"]).to_numpy()
    observed["oracle_origin"] = np.where(pick_origin, observed["hurdle"], observed["point"])
    observed["oracle_series"] = np.where(pick_series, observed["hurdle"], observed["point"])
    return observed


def cmd_run(args) -> None:
    if not G.SPEC.exists():
        raise SystemExit("freeze the GateSpec before opening test")
    spec = json.loads(G.SPEC.read_text())
    manifest = json.loads((OUT / "oof_manifest.json").read_text())
    potential = json.loads((OUT / "gate_potential.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = spec["selected"]
    report = {"analysis": "structure-gated Point/Hurdle kill test",
              "spec_frozen_at_utc": spec["frozen_at_utc"], "tested_at_utc": _utc(),
              "selected": selected, "experts_frozen": True, "joint_training": False,
              "datasets": {}}

    for name in args.datasets:
        bundle = test_frame(name, device)
        frame = bundle["frame"]
        origin_table = frame[["series_id", "origin"]].drop_duplicates().reset_index(drop=True)
        bundle["origin_table"] = origin_table
        bundle["expert_state"] = expert_state_at_test(frame, bundle["scale"])
        block = spec["datasets"][name]
        inputs = gate_inputs(bundle, block["medians"])

        checkpoint = torch.load(OUT / block["checkpoint"], map_location=device,
                                weights_only=False)
        model = G.build_gate(checkpoint["architecture"], len(checkpoint["columns"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        g = apply_gate(inputs, checkpoint["columns"], block["centre"], block["spread"],
                       model, device)
        origin_table["g"] = g

        ablation = {}
        for variant in spec["ablation_variants"]:
            if variant == selected["variant"]:
                continue
            fitted = G.train_final_gate(name, manifest, variant, selected["architecture"],
                                        selected["loss_mode"], device)
            ablation[variant] = apply_gate(inputs, fitted["columns"], fitted["centre"],
                                           fitted["spread"], fitted["model"], device)

        keys = pd.MultiIndex.from_arrays([frame["series_id"], frame["origin"]])
        pairs = pd.MultiIndex.from_arrays([origin_table["series_id"], origin_table["origin"]])
        frame["g"] = pd.Series(g, index=pairs).reindex(keys).to_numpy()
        frame["gate"] = (1 - frame["g"]) * frame["point"] + frame["g"] * frame["hurdle"]
        frame["fifty"] = 0.5 * frame["point"] + 0.5 * frame["hurdle"]
        alpha = float(potential["datasets"][name]["B2_global_alpha"]["alpha"])
        frame["global_alpha"] = (1 - alpha) * frame["point"] + alpha * frame["hurdle"]
        rule = pd.Series(frozen_rule_route(name, origin_table), index=pairs).reindex(keys).to_numpy()
        frame["frozen_rule"] = np.where(rule < 0.5, frame["point"], frame["hurdle"])
        for variant, weights in ablation.items():
            gv = pd.Series(weights, index=pairs).reindex(keys).to_numpy()
            frame[f"gate_{variant}"] = (1 - gv) * frame["point"] + gv * frame["hurdle"]

        methods = ["point", "hurdle", "fifty", "global_alpha", "frozen_rule", "gate"]
        methods += [f"gate_{v}" for v in ablation]
        per_series = per_series_metrics(frame, methods)

        observed = oracle_columns(frame)
        oracle_series_metrics = per_series_metrics(observed, ["oracle_origin", "oracle_series"])
        per_series = per_series.join(oracle_series_metrics)

        table = {}
        for method in methods:
            table[method] = overall(frame, method) | {
                "mean_per_series_rmse": float(per_series["rmse_" + method].mean()),
                "median_per_series_rmse": float(per_series["rmse_" + method].median()),
                "mean_per_series_mae": float(per_series["mae_" + method].mean())}
        for method in ("oracle_series", "oracle_origin"):
            table[method] = overall(observed, method) | {
                "mean_per_series_rmse": float(per_series["rmse_" + method].mean()),
                "median_per_series_rmse": float(per_series["rmse_" + method].median()),
                "mean_per_series_mae": float(per_series["mae_" + method].mean()),
                "upper_bound_only": True}

        best_static = min(("point", "hurdle", "fifty", "global_alpha"),
                          key=lambda m: table[m]["overall_rmse"])
        targets = ["point", "hurdle", "fifty", "global_alpha", "frozen_rule"]
        comparisons = {"gate_vs_" + m: paired_bootstrap(per_series, "gate", m) for m in targets}
        comparisons["gate_vs_best_static"] = comparisons["gate_vs_" + best_static]

        e_static = table[best_static]["overall_rmse"]
        e_gate = table["gate"]["overall_rmse"]
        e_oracle = table["oracle_origin"]["overall_rmse"]
        denominator = e_static - e_oracle
        recovery = ((e_static - e_gate) / denominator if denominator > 0
                    else "ORACLE_GAP_UNAVAILABLE")

        extreme = float(((g < 0.05) | (g > 0.95)).mean())
        winner = (observed.groupby(["series_id", "origin"])
                  .apply(lambda d: d["se_hurdle"].mean() < d["se_point"].mean(),
                         include_groups=False)
                  .reindex(pairs).to_numpy().astype(float))
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(winner, g)) if len(np.unique(winner)) > 1 else float("nan")
        routing = {"mean_g": float(g.mean()), "median_g": float(np.median(g)),
                   "sd_g": float(g.std()), "min_g": float(g.min()), "max_g": float(g.max()),
                   "fraction_below_0.1": float((g < 0.1).mean()),
                   "fraction_above_0.9": float((g > 0.9).mean()),
                   "extreme_share": extreme,
                   "collapse_warn": bool(extreme >= COLLAPSE_SHARE),
                   "expert_winner_auc": auc,
                   "correlation_with_winner": float(np.corrcoef(g, winner)[0, 1]),
                   "hurdle_win_share": float(winner.mean())}

        descriptors = pd.read_csv(screen.OUT / "per_series_metrics.csv")
        descriptors = descriptors[descriptors["dataset"] == name].set_index("series_id")
        joined = origin_table.join(descriptors[["ADI_train", "rho_interval_abs_train",
                                                "rho_magnitude_train"]], on="series_id")
        joined["train_scale"] = bundle["scale"].reindex(joined["series_id"]).to_numpy()
        by_bin = {}
        for column in ("ADI_train", "rho_interval_abs_train", "rho_magnitude_train",
                       "train_scale"):
            edges = np.unique(np.nanquantile(joined[column].to_numpy(float),
                                             [0, .25, .5, .75, 1.0]))
            if len(edges) < 3:
                continue
            labels = pd.cut(joined[column].to_numpy(float), edges, include_lowest=True)
            by_bin[column] = {str(k): float(v) for k, v in
                              joined.groupby(labels, observed=True)["g"].mean().items()}
        routing["gate_by_bin"] = by_bin

        origin_table.to_csv(OUT / ("test_gate_weights_" + name + ".csv"), index=False)
        per_series.to_csv(OUT / ("test_per_series_" + name + ".csv"))
        frame.to_parquet(OUT / ("test_predictions_" + name + ".parquet"), index=False)
        report["datasets"][name] = {
            "n_series": int(per_series.shape[0]), "origins": bundle["origins"],
            "expert_provenance": bundle["provenance"],
            "deployable": {m: table[m] for m in methods},
            "oracle_upper_bounds": {m: table[m] for m in ("oracle_series", "oracle_origin")},
            "best_static": best_static, "global_alpha": alpha,
            "paired_bootstrap": comparisons, "oracle_recovery": recovery,
            "routing": routing}
        print("[" + name + "] best_static=" + best_static
              + " gate=%.4f static=%.4f oracle=%.4f recovery=%s"
              % (e_gate, e_static, e_oracle, recovery), flush=True)

    beats = {}
    for n, b in report["datasets"].items():
        cmp = b["paired_bootstrap"]
        beats[n] = bool(cmp["gate_vs_fifty"]["relative_improvement"] > 0
                        and cmp["gate_vs_global_alpha"]["relative_improvement"] > 0
                        and cmp["gate_vs_best_static"]["relative_improvement"] > 0)
    gains = {n: b["paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"]
             for n, b in report["datasets"].items()}
    recoveries = {n: b["oracle_recovery"] for n, b in report["datasets"].items()}
    numeric = [v for v in recoveries.values() if isinstance(v, float)]
    same_direction = len({v > 0 for v in gains.values()}) == 1
    magnitude_ok = any(v >= 0.02 for v in gains.values()) or all(v >= 0.01 for v in gains.values())
    if not any(beats.values()):
        verdict = "GATE_KILL_RED"
    elif (all(beats.values()) and same_direction and magnitude_ok
          and any(v >= 0.25 for v in numeric)):
        verdict = "GATE_KILL_GREEN"
    else:
        verdict = "GATE_KILL_YELLOW"
    report["verdict"] = verdict
    report["beats_all_static"] = beats
    report["relative_gain_vs_best_static"] = gains
    report["oracle_recovery_by_dataset"] = recoveries
    report["git_commit"] = cli._git_commit()
    (OUT / "killtest.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "gains": gains, "recovery": recoveries},
                     indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser("structure gate kill test")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
