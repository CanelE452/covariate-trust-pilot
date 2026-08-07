"""Is there any complementarity for a gate to exploit? Answered without test.

Everything here reads the OOF regret table only.  If the best an omniscient
per-origin selector can do is indistinguishable from just picking one expert and
keeping it, then a learned gate cannot help either and the work stops before a
gate is written.

The oracle rows are upper bounds and are never reported as methods.  The
per-origin oracle in particular reads the OOF target to choose, which no
deployable model can do.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..external_validity_screen import cli, confirmatory_h2 as conf, posthoc, screen
from .oof import OUT

#: Operational go/no-go, not a statistical threshold: below this the compute a
#: gate would cost is not worth spending.
MIN_ORACLE_IMPROVEMENT = 0.005


def load_predictions(name: str, folds: int) -> pd.DataFrame:
    frames = []
    for k in range(folds):
        path = OUT / f"oof_predictions_{name}_fold{k}.parquet"
        block = pd.read_parquet(path)
        block["fold"] = k
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def per_origin_losses(predictions: pd.DataFrame, scale: pd.Series,
                      weights=None) -> pd.DataFrame:
    """Per-origin normalised MSE for point, hurdle and any requested mixtures."""
    observed = predictions[predictions["target_mask"] > 0].copy()
    observed["scale"] = scale.loc[observed["series_id"]].to_numpy()
    y = observed["y_observed"].to_numpy(np.float64)
    p = observed["point_mean_prediction"].to_numpy(np.float64)
    h = observed["hurdle_mean_prediction"].to_numpy(np.float64)
    s = np.maximum(observed["scale"].to_numpy(np.float64), 1e-9)
    keys = ["fold", "series_id", "origin"]
    out = {}
    columns = {"point": p, "hurdle": h}
    for w in (weights or []):
        columns[f"mix_{w:.2f}"] = (1 - w) * p + w * h
    for label, yhat in columns.items():
        observed[f"se_{label}"] = ((yhat - y) / s) ** 2
        out[label] = observed.groupby(keys)[f"se_{label}"].mean()
    frame = pd.DataFrame(out).reset_index()
    return frame


def summarise(frame: pd.DataFrame, weights) -> dict:
    point = frame["point"].to_numpy()
    hurdle = frame["hurdle"].to_numpy()
    best_single_name = "point" if point.mean() <= hurdle.mean() else "hurdle"
    best_single = min(point.mean(), hurdle.mean())

    fifty = frame["mix_0.50"].to_numpy()
    alpha_losses = {w: frame[f"mix_{w:.2f}"].to_numpy().mean() for w in weights}
    best_alpha = min(alpha_losses, key=alpha_losses.get)

    per_series = frame.groupby("series_id")[["point", "hurdle"]].mean()
    oracle_series_pick = (per_series["hurdle"] < per_series["point"]).rename("pick_hurdle")
    joined = frame.join(oracle_series_pick, on="series_id")
    oracle_series = np.where(joined["pick_hurdle"], joined["hurdle"], joined["point"]).mean()
    oracle_origin = np.minimum(point, hurdle).mean()

    static_best = min(best_single, fifty.mean(), alpha_losses[best_alpha])
    improvement = (static_best - oracle_origin) / static_best
    return {
        "n_origins": int(len(frame)),
        "n_series": int(frame["series_id"].nunique()),
        "B0_best_single": {"name": best_single_name, "loss": float(best_single),
                           "point_loss": float(point.mean()),
                           "hurdle_loss": float(hurdle.mean())},
        "B1_fifty_fifty": {"loss": float(fifty.mean())},
        "B2_global_alpha": {"alpha": float(best_alpha),
                            "loss": float(alpha_losses[best_alpha]),
                            "grid": [float(w) for w in weights],
                            "losses": {f"{w:.2f}": float(v) for w, v in alpha_losses.items()}},
        "B4_oracle_series": {"loss": float(oracle_series),
                             "hurdle_share": float(oracle_series_pick.mean()),
                             "upper_bound_only": True},
        "B5_oracle_origin": {"loss": float(oracle_origin),
                             "hurdle_share": float((hurdle < point).mean()),
                             "upper_bound_only": True},
        "best_static_loss": float(static_best),
        "oracle_origin_relative_improvement": float(improvement),
        "oracle_series_relative_improvement": float((static_best - oracle_series) / static_best),
        "threshold": MIN_ORACLE_IMPROVEMENT,
        "passes_threshold": bool(improvement >= MIN_ORACLE_IMPROVEMENT),
    }


def frozen_rule_baseline(name: str, frame: pd.DataFrame) -> dict:
    """B3: the frozen H2 rule, applied with its own cutoffs and no refitting.

    The rule was written for M5 descriptors computed on the Stage A train split;
    it is applied here to the same series, so this is a static routing baseline
    rather than a new selection study.  Series the rule does not name a
    candidate for take Hurdle, which is the rule's own complement.
    """
    cutoffs = conf.frozen_cutoffs_from_artifact()
    series = pd.read_csv(screen.OUT / "per_series_metrics.csv")
    block = series[series["dataset"] == name].set_index("series_id")
    have = block.index.intersection(frame["series_id"].unique())
    adi = block.loc[have, "ADI_train"].to_numpy(float)
    occ = block.loc[have, "rho_interval_abs_train"].to_numpy(float)
    mag = block.loc[have, "rho_magnitude_train"].to_numpy(float)
    candidate = pd.Series((adi >= cutoffs["HIGH_ADI_min"])
                          & (occ <= cutoffs["LOW_OCC_max"])
                          & (mag >= cutoffs["MAG_PERSISTENT_min"]), index=have)
    joined = frame.join(candidate.rename("point_candidate"), on="series_id")
    missing = int(joined["point_candidate"].isna().sum())
    pick_point = joined["point_candidate"].fillna(False).to_numpy()
    loss = np.where(pick_point, joined["point"], joined["hurdle"]).mean()
    return {"loss": float(loss), "n_point_routed": int(pick_point.sum()),
            "point_share": float(pick_point.mean()),
            "n_series_without_descriptor": missing,
            "cutoffs_refitted": False}


def cmd_run(args) -> None:
    manifest = json.loads((OUT / "oof_manifest.json").read_text())
    weights = [round(w, 2) for w in np.arange(0.0, 1.0001, 0.05)]
    report = {"analysis": "OOF gate potential; test never read",
              "test_used": False,
              "computed_at_utc": datetime.now(timezone.utc).isoformat(),
              "mixture_grid": weights, "datasets": {}}

    for name in args.datasets:
        block = manifest["datasets"][name]
        data = screen.load_dataset(name)
        from ..unified_temporal_27_v3.training import train_scale
        cfg = screen.config_for(name)
        scale = pd.Series(train_scale({"y": data["y"], "z": data["z"]}, cfg),
                          index=pd.Index(data["series_id"]).astype(str))
        predictions = load_predictions(name, len(block["folds"]))
        frame = per_origin_losses(predictions, scale, weights)
        summary = summarise(frame, weights)
        summary["B3_frozen_rule"] = frozen_rule_baseline(name, frame)
        summary["folds"] = [f["origins"] for f in block["folds"]]
        frame.to_parquet(OUT / f"oof_origin_losses_{name}.parquet", index=False)
        report["datasets"][name] = summary
        print(f"[{name}] best_static={summary['best_static_loss']:.6f} "
              f"oracle_origin={summary['B5_oracle_origin']['loss']:.6f} "
              f"improvement={summary['oracle_origin_relative_improvement']*100:.2f}%")

    passes = {n: b["passes_threshold"] for n, b in report["datasets"].items()}
    if all(passes.values()):
        verdict = "GATE_POTENTIAL_PASS"
    elif any(passes.values()):
        verdict = "GATE_POTENTIAL_DATASET_SPECIFIC"
    else:
        verdict = "GATE_POTENTIAL_RED"
    report["verdict"] = verdict
    report["per_dataset_passes"] = passes
    report["git_commit"] = cli._git_commit()
    (OUT / "gate_potential.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "passes": passes}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("gate potential from OOF only")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
