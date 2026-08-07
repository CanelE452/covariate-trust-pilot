"""Which expert pair fails differently enough for routing to matter?

Prediction correlation is the wrong question.  Two experts can predict very
different numbers and still be wrong at the same origins, in which case no
mixture helps.  So the ranking here is driven by *residual* correlation and by
the convex-oracle ceiling each pair achieves, with prediction correlation kept
only as a contrast.

Everything runs on the OOF rows that Gate-v2 already fixed.  The M5 and Favorita
test splits, and the fresh holdout Gate-v2 was confirmed on, are outcomes that
have been read; using them to pick an expert pair would make the pick unfalsifiable.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..external_validity_screen import cli, screen
from ..structure_gate.convex_oracle import FLAT_THRESHOLD
from ..structure_gate.oof import OUT as GATE_OUT
from ..unified_temporal_27_v3.training import train_scale
from .oof_experts import CLASSICAL, NEURAL, OUT, REFERENCE_PAIR

EXPERTS = list(NEURAL) + list(CLASSICAL)

#: Operational go/no-go thresholds, fixed before any pair was scored.
CEILING_MULTIPLIER_GREEN = 1.5
ABSOLUTE_GREEN = {"m5": 0.06, "favorita": 0.035}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalised(name: str) -> pd.DataFrame:
    """OOF rows with every expert prediction divided by the train scale."""
    frame = pd.read_parquet(OUT / f"oof_experts_{name}.parquet")
    frame = frame[frame["target_mask"] > 0].copy()
    frame["series_id"] = frame["series_id"].astype(str)
    data = screen.load_dataset(name)
    cfg = screen.config_for(name)
    scale = pd.Series(train_scale({"y": data["y"], "z": data["z"]}, cfg),
                      index=pd.Index(data["series_id"]).astype(str))
    s = np.maximum(scale.loc[frame["series_id"]].to_numpy(np.float64), 1e-9)
    frame["yn"] = frame["y_observed"].to_numpy(np.float64) / s
    for expert in EXPERTS:
        frame[f"n_{expert}"] = frame[expert].to_numpy(np.float64) / s
        frame[f"e_{expert}"] = frame[f"n_{expert}"] - frame["yn"]
    return frame


def pair_table(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per-origin quadratic coefficients and losses for the mixture of a and b."""
    work = pd.DataFrame({
        "fold": frame["fold"].to_numpy(), "series_id": frame["series_id"].to_numpy(),
        "origin": frame["origin"].to_numpy(),
        "ea": frame[f"e_{a}"].to_numpy(), "eb": frame[f"e_{b}"].to_numpy(),
        "d": frame[f"n_{b}"].to_numpy() - frame[f"n_{a}"].to_numpy()})
    work["r"] = -work["ea"]                      # y - a
    grouped = work.groupby(["fold", "series_id", "origin"])
    table = pd.DataFrame({
        "n": grouped.size(),
        "sse_a": grouped["ea"].apply(lambda v: float((v ** 2).sum())),
        "sse_b": grouped["eb"].apply(lambda v: float((v ** 2).sum())),
        "num": grouped.apply(lambda d: float((d["r"] * d["d"]).sum()), include_groups=False),
        "den": grouped["d"].apply(lambda v: float((v ** 2).sum())),
        "mean_abs_d": grouped["d"].apply(lambda v: float(v.abs().mean())),
    }).reset_index()
    flat = table["den"].to_numpy() <= FLAT_THRESHOLD
    raw = np.divide(table["num"].to_numpy(),
                    np.where(flat, 1.0, table["den"].to_numpy()))
    g = np.where(flat, 0.5, np.clip(raw, 0.0, 1.0))
    table["g_star"] = g
    table["flat"] = flat
    table["sse_fifty"] = (table["sse_a"] - table["num"]
                          + 0.25 * table["den"])
    table["sse_convex"] = np.where(
        flat, table["sse_a"],
        table["sse_a"] - 2 * g * table["num"] + g ** 2 * table["den"])
    table["sse_hard"] = np.minimum(table["sse_a"], table["sse_b"])
    for column in ("a", "b", "fifty", "convex", "hard"):
        table[f"mse_{column}"] = table[f"sse_{column}"] / table["n"]
    return table


def best_alpha(table: pd.DataFrame) -> tuple[float, float]:
    best = (None, np.inf)
    n = table["n"].to_numpy(np.float64)
    for alpha in np.arange(0.0, 1.0001, 0.05):
        sse = (table["sse_a"].to_numpy() - 2 * alpha * table["num"].to_numpy()
               + alpha ** 2 * table["den"].to_numpy())
        loss = float((sse / n).mean())
        if loss < best[1]:
            best = (round(float(alpha), 2), loss)
    return best


def diversity(frame: pd.DataFrame, table: pd.DataFrame, a: str, b: str) -> dict:
    """Prediction-level and error-level agreement, reported side by side."""
    pa, pb = frame[f"n_{a}"].to_numpy(), frame[f"n_{b}"].to_numpy()
    ea, eb = frame[f"e_{a}"].to_numpy(), frame[f"e_{b}"].to_numpy()
    b_wins = (table["sse_b"] < table["sse_a"]).astype(int)
    by_series = b_wins.groupby(table["series_id"]).apply(
        lambda v: float((v.to_numpy()[1:] != v.to_numpy()[:-1]).mean()) if len(v) > 1 else 0.0)
    share = b_wins.groupby(table["series_id"]).mean()
    return {"prediction_correlation": float(np.corrcoef(pa, pb)[0, 1]),
            "residual_correlation": float(np.corrcoef(ea, eb)[0, 1]),
            "squared_error_correlation": float(np.corrcoef(ea ** 2, eb ** 2)[0, 1]),
            "mean_abs_prediction_difference": float(table["mean_abs_d"].mean()),
            "b_win_probability": float(b_wins.mean()),
            "origin_winner_flip_rate": float(by_series.mean()),
            "series_always_same_winner": float(((share == 0) | (share == 1)).mean())}


def screen_pair(frame: pd.DataFrame, a: str, b: str) -> dict:
    table = pair_table(frame, a, b)
    alpha, alpha_loss = best_alpha(table)
    losses = {"a": float(table["mse_a"].mean()), "b": float(table["mse_b"].mean()),
              "fifty": float(table["mse_fifty"].mean()), "global_alpha": alpha_loss}
    by_series = table.groupby("series_id")[["sse_a", "sse_b", "n"]].sum()
    pick = (by_series["sse_b"] < by_series["sse_a"])
    chosen = np.where(table["series_id"].map(pick).to_numpy(),
                      table["sse_b"], table["sse_a"])
    losses["oracle_series_hard"] = float((chosen / table["n"]).mean())
    losses["oracle_origin_hard"] = float(table["mse_hard"].mean())
    losses["oracle_origin_convex"] = float(table["mse_convex"].mean())
    best_static_name = min(("a", "b", "fifty", "global_alpha"), key=lambda k: losses[k])
    best_static = losses[best_static_name]
    non_flat = ~table["flat"].to_numpy()
    g = table["g_star"].to_numpy()
    return {"a": a, "b": b, "global_alpha": alpha,
            "best_static": {"a": a, "b": b, "fifty": "fifty",
                            "global_alpha": "global_alpha"}[best_static_name],
            "losses": losses, "best_static_loss": best_static,
            "hard_gain": float((best_static - losses["oracle_origin_hard"]) / best_static),
            "convex_gain": float((best_static - losses["oracle_origin_convex"]) / best_static),
            "extra_soft_potential": float(
                (losses["oracle_origin_hard"] - losses["oracle_origin_convex"]) / best_static),
            "interior_g_rate": float(((g > 0) & (g < 1) & non_flat).sum() / max(non_flat.sum(), 1)),
            "flat_share": float(table["flat"].mean()),
            "diversity": diversity(frame, table, a, b)}


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"analysis": "pairwise expert diversity and oracle screening, OOF only",
              "test_used": False, "fresh_holdout_used": False,
              "computed_at_utc": _utc(),
              "reference_pair": list(REFERENCE_PAIR),
              "thresholds": {"ceiling_multiplier_green": CEILING_MULTIPLIER_GREEN,
                             "absolute_green": ABSOLUTE_GREEN},
              "datasets": {}}
    for name in args.datasets:
        frame = normalised(name)
        pairs = {}
        for a, b in itertools.combinations(EXPERTS, 2):
            pairs[f"{a}|{b}"] = screen_pair(frame, a, b)
            print(f"[{name}] {a}|{b:<22} convex={pairs[f'{a}|{b}']['convex_gain']*100:6.2f}% "
                  f"resid_corr={pairs[f'{a}|{b}']['diversity']['residual_correlation']:+.3f}",
                  flush=True)
        reference = pairs["|".join(REFERENCE_PAIR)]
        for key, block in pairs.items():
            block["ceiling_multiplier"] = (block["convex_gain"] / reference["convex_gain"]
                                           if reference["convex_gain"] > 0 else np.nan)
            block["ceiling_increment_pp"] = block["convex_gain"] - reference["convex_gain"]
        report["datasets"][name] = {
            "n_origins": int(frame.groupby(["fold", "series_id", "origin"]).ngroups),
            "reference_convex_gain": reference["convex_gain"],
            "pairs": pairs}
    (OUT / "diversity_screen.json").write_text(json.dumps(report, indent=2, default=str))
    print("wrote diversity_screen.json")


def main() -> None:
    parser = argparse.ArgumentParser("expert diversity screen")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
