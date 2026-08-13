"""How much room does a soft mixture have that a hard choice does not?

Gate-v1 learned to answer "which expert wins" and came back YELLOW.  Two
explanations survive that result and this module separates them without
touching test:

    A  the binary objective was wrong for a problem whose answer is a weight
    B  the two experts are too alike for any weight to matter

The discriminator is the exact convex oracle.  For one origin with predictions
P and H and truth Y, the squared error of (1-g)P + gH is a quadratic in g, so
the minimiser is available in closed form:

    g* = clip( <Y - P, H - P> / <H - P, H - P>, 0, 1 )

If that beats the hard per-origin oracle by a wide margin and g* often lands
strictly inside (0, 1), the binary formulation was throwing information away.
If convex and hard nearly coincide, the ceiling itself is low and no gate
objective rescues it.

Everything here runs on OOF rows.  Computing g* on test would hand the answer
to a model that has to be developed without it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..external_validity_screen import cli, screen
from ..unified_temporal_27_v3.training import train_scale
from .oof import OUT
from .potential import load_predictions

#: An origin whose two experts agree to within this (on the normalised scale)
#: has no identifiable g*: the quadratic is flat and the closed form divides by
#: nearly nothing. Fixed before any g* was inspected.
FLAT_THRESHOLD = 1e-8


def origin_arrays(predictions: pd.DataFrame, scale: pd.Series) -> dict:
    """Group the OOF rows into per-origin vectors on the train-normalised scale."""
    observed = predictions[predictions["target_mask"] > 0].copy()
    s = np.maximum(scale.loc[observed["series_id"]].to_numpy(np.float64), 1e-9)
    observed["yn"] = observed["y_observed"].to_numpy(np.float64) / s
    observed["pn"] = observed["point_mean_prediction"].to_numpy(np.float64) / s
    observed["hn"] = observed["hurdle_mean_prediction"].to_numpy(np.float64) / s
    observed["d"] = observed["hn"] - observed["pn"]
    observed["r"] = observed["yn"] - observed["pn"]
    keys = ["fold", "series_id", "origin"]
    grouped = observed.groupby(keys)
    frame = pd.DataFrame({
        "n_observed": grouped.size(),
        "num": grouped.apply(lambda d: float((d["r"] * d["d"]).sum()), include_groups=False),
        "den": grouped.apply(lambda d: float((d["d"] * d["d"]).sum()), include_groups=False),
        "sse_point": grouped.apply(lambda d: float((d["r"] ** 2).sum()), include_groups=False),
        "sse_hurdle": grouped.apply(lambda d: float(((d["yn"] - d["hn"]) ** 2).sum()),
                                    include_groups=False),
        "sse_fifty": grouped.apply(
            lambda d: float(((d["yn"] - 0.5 * (d["pn"] + d["hn"])) ** 2).sum()),
            include_groups=False),
        "mean_abs_disagreement": grouped["d"].apply(lambda v: float(v.abs().mean())),
        "corr_ph": grouped.apply(
            lambda d: float(np.corrcoef(d["pn"], d["hn"])[0, 1])
            if d["pn"].std() > 0 and d["hn"].std() > 0 else np.nan,
            include_groups=False),
    }).reset_index()
    return {"frame": frame, "observed": observed}


def convex_oracle(frame: pd.DataFrame) -> pd.DataFrame:
    """g* per origin, with flat origins flagged rather than forced to an end point."""
    den = frame["den"].to_numpy(np.float64)
    num = frame["num"].to_numpy(np.float64)
    flat = den <= FLAT_THRESHOLD
    raw = np.divide(num, np.where(flat, 1.0, den))
    g_star = np.clip(raw, 0.0, 1.0)
    # A flat origin gets a storage placeholder, never a claim about its optimum.
    g_star = np.where(flat, 0.5, g_star)

    out = frame.copy()
    out["g_star"] = g_star
    out["g_star_unclipped"] = np.where(flat, np.nan, raw)
    out["flat"] = flat
    # sse((1-g)P + gH) = sse_point - 2 g num + g^2 den
    out["sse_convex"] = np.where(
        flat, frame["sse_point"],
        frame["sse_point"] - 2 * g_star * num + g_star ** 2 * den)
    out["sse_hard"] = np.minimum(frame["sse_point"], frame["sse_hurdle"])
    out["hurdle_wins"] = (frame["sse_hurdle"] < frame["sse_point"]).astype(int)
    for column in ("point", "hurdle", "fifty", "convex", "hard"):
        out[f"mse_{column}"] = out[f"sse_{column}"] / out["n_observed"]
    return out


def global_alpha(frame: pd.DataFrame) -> tuple[float, float]:
    """The single best constant weight on these OOF rows, on the same grid."""
    best = (None, np.inf)
    for alpha in np.arange(0.0, 1.0001, 0.05):
        sse = (frame["sse_point"].to_numpy() - 2 * alpha * frame["num"].to_numpy()
               + alpha ** 2 * frame["den"].to_numpy())
        loss = float((sse / frame["n_observed"].to_numpy()).mean())
        if loss < best[1]:
            best = (round(float(alpha), 2), loss)
    return best


def diversity_report(frame: pd.DataFrame) -> dict:
    """Does exploitable gain actually concentrate where the experts disagree?"""
    disagreement = frame["mean_abs_disagreement"].to_numpy(np.float64)
    edges = np.unique(np.nanquantile(disagreement, [0, .25, .5, .75, 1.0]))
    bins = pd.cut(disagreement, edges, include_lowest=True)
    static = frame["mse_fifty"].to_numpy()
    rows = {}
    for label, block in frame.groupby(bins, observed=True):
        base = float(block["mse_fifty"].mean())
        rows[str(label)] = {
            "n": int(len(block)),
            "mean_abs_disagreement": float(block["mean_abs_disagreement"].mean()),
            "mse_fifty": base,
            "hard_gain": float((base - block["mse_hard"].mean()) / base) if base > 0 else np.nan,
            "convex_gain": float((base - block["mse_convex"].mean()) / base) if base > 0 else np.nan,
            "hurdle_win_share": float(block["hurdle_wins"].mean()),
            "interior_share": float(((block["g_star"] > 0) & (block["g_star"] < 1)
                                     & ~block["flat"]).mean())}
    per_series = frame.groupby("series_id")["hurdle_wins"]
    flip = per_series.apply(lambda v: float((v.to_numpy()[1:] != v.to_numpy()[:-1]).mean())
                            if len(v) > 1 else 0.0)
    share = per_series.mean()
    entropy = -(share * np.log(share.clip(1e-9)) + (1 - share) * np.log((1 - share).clip(1e-9)))
    return {"disagreement_bins": rows,
            "mean_corr_point_hurdle": float(frame["corr_ph"].mean(skipna=True)),
            "median_corr_point_hurdle": float(frame["corr_ph"].median(skipna=True)),
            "per_series_winner_flip_rate": float(flip.mean()),
            "mean_origin_winner_entropy": float(entropy.mean()),
            "share_series_always_same_winner": float(((share == 0) | (share == 1)).mean())}


def cmd_run(args) -> None:
    manifest = json.loads((OUT / "oof_manifest.json").read_text())
    report = {"analysis": "convex-mixture oracle diagnostic, OOF only",
              "test_used": False, "flat_threshold": FLAT_THRESHOLD,
              "computed_at_utc": datetime.now(timezone.utc).isoformat(),
              "datasets": {}}
    for name in args.datasets:
        data = screen.load_dataset(name)
        cfg = screen.config_for(name)
        scale = pd.Series(train_scale({"y": data["y"], "z": data["z"]}, cfg),
                          index=pd.Index(data["series_id"]).astype(str))
        predictions = load_predictions(name, len(manifest["datasets"][name]["folds"]))
        grouped = origin_arrays(predictions, scale)
        frame = convex_oracle(grouped["frame"])
        frame.to_parquet(OUT / f"oof_gstar_{name}.parquet", index=False)

        alpha, alpha_loss = global_alpha(frame)
        losses = {"point": float(frame["mse_point"].mean()),
                  "hurdle": float(frame["mse_hurdle"].mean()),
                  "fifty": float(frame["mse_fifty"].mean()),
                  "global_alpha": alpha_loss}
        per_series = frame.groupby("series_id")[["sse_point", "sse_hurdle", "n_observed"]].sum()
        pick_hurdle = (per_series["sse_hurdle"] < per_series["sse_point"])
        chosen = np.where(frame["series_id"].map(pick_hurdle).to_numpy(),
                          frame["sse_hurdle"], frame["sse_point"])
        losses["oracle_series_hard"] = float((chosen / frame["n_observed"]).mean())
        losses["oracle_origin_hard"] = float(frame["mse_hard"].mean())
        losses["oracle_origin_convex"] = float(frame["mse_convex"].mean())

        best_static_name = min(("point", "hurdle", "fifty", "global_alpha"),
                               key=lambda k: losses[k])
        best_static = losses[best_static_name]
        non_flat = ~frame["flat"].to_numpy()
        g = frame["g_star"].to_numpy()
        interior = (g > 0) & (g < 1) & non_flat
        report["datasets"][name] = {
            "n_origins": int(len(frame)), "n_series": int(frame["series_id"].nunique()),
            "global_alpha": alpha, "best_static": best_static_name,
            "losses": losses,
            "hard_gain": float((best_static - losses["oracle_origin_hard"]) / best_static),
            "convex_gain": float((best_static - losses["oracle_origin_convex"]) / best_static),
            "extra_soft_potential": float(
                (losses["oracle_origin_hard"] - losses["oracle_origin_convex"]) / best_static),
            "g_star_distribution": {
                "flat_share": float(frame["flat"].mean()),
                "at_zero_share": float(((g == 0) & non_flat).mean()),
                "interior_share": float(interior.mean()),
                "at_one_share": float(((g == 1) & non_flat).mean()),
                "INTERIOR_RATE_given_non_flat": float(interior.sum() / max(non_flat.sum(), 1)),
                "mean": float(g[non_flat].mean()), "median": float(np.median(g[non_flat])),
                "p10": float(np.quantile(g[non_flat], .10)),
                "p25": float(np.quantile(g[non_flat], .25)),
                "p75": float(np.quantile(g[non_flat], .75)),
                "p90": float(np.quantile(g[non_flat], .90)),
                "per_series_mean_sd": float(
                    frame[non_flat].groupby("series_id")["g_star"].mean().std()),
                "within_series_sd": float(
                    frame[non_flat].groupby("series_id")["g_star"].std().mean())},
            "diversity": diversity_report(frame)}
        block = report["datasets"][name]
        print(f"[{name}] best_static={best_static_name} hard={block['hard_gain']*100:.2f}% "
              f"convex={block['convex_gain']*100:.2f}% "
              f"extra_soft={block['extra_soft_potential']*100:.2f}% "
              f"interior={block['g_star_distribution']['INTERIOR_RATE_given_non_flat']*100:.1f}%")

    report["git_commit"] = cli._git_commit()
    (OUT / "convex_oracle.json").write_text(json.dumps(report, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser("convex mixture oracle diagnostic")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
