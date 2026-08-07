"""Pick one expert pair by a pre-registered OOF rule, then route it with Gate-v2.

The pair is chosen before the gate runs and by a formula fixed in advance:
geometric mean of the convex-oracle ceiling multiplier across datasets, then the
worse dataset, then residual correlation, then static performance, then size.
No test or fresh-holdout outcome enters, because both have been read.

Gate-v2 is reused as it was frozen -- G-NOSCALE features, the same small MLP, the
regret BCE, expanding-time cross-fold fitting.  Only the expert-state features
change, because they describe whichever two experts are being routed.  Changing
the gate and the expert pool at once would leave the result unattributable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..external_validity_screen import cli, prereg, screen
from ..structure_gate import features as F, gate as G, gate_v2 as V2
from ..structure_gate.convex_oracle import FLAT_THRESHOLD
from ..structure_gate.oof import OUT as GATE_OUT
from ..unified_temporal_27_v3.training import train_scale
from .oof_experts import CLASSICAL, NEURAL, OUT, REFERENCE_PAIR
from .screen import EXPERTS, normalised, pair_table, best_alpha

SPEC = OUT / "expert_set_spec.json"
#: A third expert has to move the ceiling by more than this to justify a
#: softmax gate over a scalar one. Fixed before any triple was scored.
TRIPLE_MIN_INCREMENT = 0.01


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(values) -> str:
    return hashlib.sha256("\n".join(str(v) for v in values).encode()).hexdigest()


def rank_pairs(report: dict, datasets: list[str], max_static_degradation: float = 0.02):
    """The pre-registered ordering, with the static-degradation guard applied."""
    reference = {d: report["datasets"][d]["pairs"]["|".join(REFERENCE_PAIR)] for d in datasets}
    rows = []
    for key in report["datasets"][datasets[0]]["pairs"]:
        blocks = {d: report["datasets"][d]["pairs"][key] for d in datasets}
        degradation = {d: blocks[d]["best_static_loss"] / reference[d]["best_static_loss"] - 1
                       for d in datasets}
        if any(v > max_static_degradation for v in degradation.values()):
            continue
        multipliers = [blocks[d]["ceiling_multiplier"] for d in datasets]
        rows.append({
            "pair": key,
            "geometric_ceiling_multiplier": float(np.sqrt(np.prod([max(m, 0) for m in multipliers]))),
            "worst_ceiling_multiplier": float(min(multipliers)),
            "max_residual_correlation": float(max(
                blocks[d]["diversity"]["residual_correlation"] for d in datasets)),
            "max_static_degradation": float(max(degradation.values())),
            "convex_gain": {d: blocks[d]["convex_gain"] for d in datasets},
            "convex_oracle_loss": {d: blocks[d]["losses"]["oracle_origin_convex"] for d in datasets},
            "best_static_loss": {d: blocks[d]["best_static_loss"] for d in datasets},
            "residual_correlation": {d: blocks[d]["diversity"]["residual_correlation"]
                                     for d in datasets}})
    rows.sort(key=lambda r: (-r["geometric_ceiling_multiplier"],
                             -r["worst_ceiling_multiplier"],
                             r["max_residual_correlation"],
                             max(r["best_static_loss"].values())))
    return rows


def simplex_oracle(frame: pd.DataFrame, experts: list[str]) -> dict:
    """Per-origin best convex combination of three experts, on a coarse simplex.

    An exact quadratic-programme per origin would need a solver dependency; a
    fixed grid over the simplex is enough for an upper bound that only has to
    answer "does a third expert move the ceiling at all".
    """
    keys = ["fold", "series_id", "origin"]
    values = {e: frame[f"n_{e}"].to_numpy(np.float64) for e in experts}
    y = frame["yn"].to_numpy(np.float64)
    grid = [w for w in itertools.product(np.arange(0, 1.0001, 0.1), repeat=len(experts) - 1)
            if sum(w) <= 1.0 + 1e-9]
    best = None
    group = frame.groupby(keys).ngroup().to_numpy()
    counts = np.bincount(group)
    for partial in grid:
        weights = list(partial) + [1.0 - sum(partial)]
        blend = sum(w * values[e] for w, e in zip(weights, experts))
        sse = np.bincount(group, weights=(blend - y) ** 2)
        best = sse if best is None else np.minimum(best, sse)
    single = {e: np.bincount(group, weights=(values[e] - y) ** 2) / counts for e in experts}
    equal = sum(values[e] for e in experts) / len(experts)
    return {"oracle_origin_simplex": float((best / counts).mean()),
            "equal_weight": float((np.bincount(group, weights=(equal - y) ** 2) / counts).mean()),
            "singles": {e: float(v.mean()) for e, v in single.items()},
            "grid_step": 0.1, "n_origins": int(len(counts))}


def cmd_select(args) -> None:
    report = json.loads((OUT / "diversity_screen.json").read_text())
    ranked = rank_pairs(report, args.datasets)
    top = ranked[0]
    a, b = top["pair"].split("|")
    print(f"TOP_DIVERSE_PAIR = {a} | {b}")
    for row in ranked[:5]:
        print(f"  {row['pair']:<40} geo={row['geometric_ceiling_multiplier']:.2f} "
              f"worst={row['worst_ceiling_multiplier']:.2f} "
              f"maxRC={row['max_residual_correlation']:+.3f}")

    triples = {}
    for name in args.datasets:
        frame = normalised(name)
        pair_loss = simplex_oracle(frame, [a, b])["oracle_origin_simplex"]
        best_third, best_value = None, pair_loss
        for third in EXPERTS:
            if third in (a, b):
                continue
            result = simplex_oracle(frame, [a, b, third])
            improvement = (pair_loss - result["oracle_origin_simplex"]) / pair_loss
            triples.setdefault(name, {})[third] = {
                "simplex_oracle": result["oracle_origin_simplex"],
                "improvement_over_pair": float(improvement)}
            if result["oracle_origin_simplex"] < best_value:
                best_third, best_value = third, result["oracle_origin_simplex"]
        triples[name]["_pair_simplex_oracle"] = pair_loss
        triples[name]["_best_third"] = best_third
        triples[name]["_best_improvement"] = float((pair_loss - best_value) / pair_loss)
        print(f"[{name}] third expert best = {best_third} "
              f"(+{(pair_loss - best_value) / pair_loss * 100:.2f}% over the pair)")

    increments = [triples[d]["_best_improvement"] for d in args.datasets]
    thirds = {triples[d]["_best_third"] for d in args.datasets}
    use_triple = bool(min(increments) >= TRIPLE_MIN_INCREMENT and len(thirds) == 1)

    spec = {"study": "expert diversity screen, selection frozen before any gate",
            "frozen_at_utc": _utc(), "test_used": False, "fresh_holdout_used": False,
            "selection_formula": ("geometric mean of the convex-oracle ceiling multiplier "
                                  "across datasets, then the worse dataset, then residual "
                                  "correlation, then static loss; pairs whose best static "
                                  "is more than 2% worse than the reference are excluded"),
            "reference_pair": list(REFERENCE_PAIR),
            "reference_convex_gain": {d: report["datasets"][d]["reference_convex_gain"]
                                      for d in args.datasets},
            "top_pair": [a, b], "top_pair_metrics": top,
            "ranking_top10": ranked[:10],
            "triple_screen": triples,
            "triple_min_increment": TRIPLE_MIN_INCREMENT,
            "use_triple": use_triple,
            "expert_pool": EXPERTS,
            "oof_manifest_sha256": _sha([json.dumps(
                json.loads((OUT / "oof_experts_manifest.json").read_text()), sort_keys=True)]),
            "gate_reused_from": str(V2.SPEC.relative_to(screen.REPO)),
            "git_commit": cli._git_commit()}
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    print(json.dumps({"top_pair": [a, b], "use_triple": use_triple,
                      "triple_increments": increments}, indent=2))


def pair_expert_state(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Expert-state features for an arbitrary pair, same shape as Gate-v2's.

    Gate-v2's block described Point and Hurdle, including the hurdle's p and mu
    heads.  A pair without those heads simply has fewer expert-state columns;
    that is the interface adaptation the design allows, not a redesign.
    """
    keys = ["fold", "series_id", "origin"]
    work = frame[keys].copy()
    work["na"] = frame[f"n_{a}"].to_numpy()
    work["nb"] = frame[f"n_{b}"].to_numpy()
    work["gap"] = work["nb"] - work["na"]
    grouped = work.groupby(keys)
    out = pd.DataFrame({
        "xs_a_mean": grouped["na"].mean(),
        "xs_b_mean": grouped["nb"].mean(),
        "xs_abs_disagreement": grouped["gap"].apply(lambda v: float(v.abs().mean())),
        "xs_max_disagreement": grouped["gap"].apply(lambda v: float(v.abs().max())),
        "xs_signed_disagreement": grouped["gap"].mean()}).reset_index()
    return out


def build_pair_frame(name: str, a: str, b: str) -> pd.DataFrame:
    """Per-origin losses, regret, as-of-origin structure and pair expert state."""
    frame = normalised(name)
    table = pair_table(frame, a, b)
    table["regret"] = table["mse_a"] - table["mse_b"]
    table = table.merge(pair_expert_state(frame, a, b),
                        on=["fold", "series_id", "origin"], how="left")
    data = screen.load_dataset(name)
    structure = F.structure_table(data, table[["series_id", "origin"]])
    table = pd.concat([table.reset_index(drop=True),
                       structure.drop(columns=["series_id", "origin"])], axis=1)
    return F.add_missing_flags(table, F.STRUCTURE_COLUMNS)


def mixture_mse_pair(table: pd.DataFrame, g: np.ndarray) -> np.ndarray:
    sse = (table["sse_a"].to_numpy(np.float64) - 2 * g * table["num"].to_numpy(np.float64)
           + g ** 2 * table["den"].to_numpy(np.float64))
    return sse / table["n"].to_numpy(np.float64)


def cmd_gate(args) -> None:
    if not SPEC.exists():
        raise SystemExit("run `select` first; the expert set must be frozen before the gate")
    spec = json.loads(SPEC.read_text())
    gate_spec = json.loads(V2.SPEC.read_text())
    chosen = gate_spec["selected"]
    a, b = spec["top_pair"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    draws, seed = prereg.BOOTSTRAP["draws"], prereg.BOOTSTRAP["seed"]

    reference = json.loads((GATE_OUT / "convex_oracle.json").read_text())
    screen_report = json.loads((OUT / "diversity_screen.json").read_text())
    reference_gate = json.loads((GATE_OUT / "gate_v2_oof_result.json").read_text())

    report = {"analysis": "frozen Gate-v2 applied to the selected diverse pair, OOF only",
              "test_used": False, "fresh_holdout_used": False,
              "expert_set_frozen_at_utc": spec["frozen_at_utc"],
              "gate_evaluated_at_utc": _utc(),
              "pair": [a, b], "gate_design": chosen,
              "gate_design_source": str(V2.SPEC.relative_to(screen.REPO)),
              "experts_frozen": True, "gate_retuned": False, "datasets": {}}

    for name in args.datasets:
        table = build_pair_frame(name, a, b)
        columns = F.feature_columns(chosen["variant"], list(table.columns))
        folds = sorted(table["fold"].unique())
        pieces, improvements = [], {}
        for validation in folds[1:]:
            train_mask = table["fold"].to_numpy() < validation
            valid_mask = table["fold"].to_numpy() == validation
            block = table[train_mask]
            medians = F.fit_imputer(block, F.STRUCTURE_COLUMNS)
            filled = F.apply_imputer(table, medians)
            x = filled[columns].to_numpy(np.float64)
            centre = x[train_mask].mean(0)
            spread = np.where(x[train_mask].std(0) > 0, x[train_mask].std(0), 1.0)
            xz = ((x - centre) / spread).astype(np.float32)
            model = V2.fit(chosen["loss"], xz[train_mask], block, chosen["architecture"], device)
            held = table[valid_mask].copy()
            held["g"] = V2.G.gate_weights(model, xz[valid_mask], device).astype(np.float64)
            held["mse_gate"] = mixture_mse_pair(held, held["g"].to_numpy())
            static = min(float(held["mse_a"].mean()), float(held["mse_b"].mean()),
                         float(held["mse_fifty"].mean()))
            improvements[int(validation)] = (static - float(held["mse_gate"].mean())) / static
            pieces.append(held)
        cross = pd.concat(pieces, ignore_index=True)

        alpha, alpha_loss = best_alpha(cross)
        cross["mse_global_alpha"] = mixture_mse_pair(cross, np.full(len(cross), alpha))
        means = {"expert_a": float(cross["mse_a"].mean()),
                 "expert_b": float(cross["mse_b"].mean()),
                 "fifty": float(cross["mse_fifty"].mean()),
                 "global_alpha": float(cross["mse_global_alpha"].mean()),
                 "gate": float(cross["mse_gate"].mean()),
                 "oracle_hard": float(cross["mse_hard"].mean()),
                 "oracle_convex": float(cross["mse_convex"].mean())}
        best_static = min(("expert_a", "expert_b", "fifty", "global_alpha"),
                          key=lambda k: means[k])
        column = {"expert_a": "mse_a", "expert_b": "mse_b", "fifty": "mse_fifty",
                  "global_alpha": "mse_global_alpha"}
        per_series = cross.groupby("series_id")[
            ["mse_a", "mse_b", "mse_fifty", "mse_global_alpha", "mse_gate",
             "mse_convex", "mse_hard"]].mean()

        def boot(target: str) -> dict:
            x = per_series["mse_gate"].to_numpy(np.float64)
            y = per_series[target].to_numpy(np.float64)
            rng = np.random.default_rng(seed)
            n = len(x)
            values = np.empty(draws)
            for i in range(draws):
                idx = rng.integers(0, n, n)
                values[i] = (y[idx].mean() - x[idx].mean()) / y[idx].mean()
            lo, hi = np.quantile(values, [0.025, 0.975])
            return {"relative_improvement": float((y.mean() - x.mean()) / y.mean()),
                    "ci": [float(lo), float(hi)],
                    "ci_excludes_zero": bool(lo > 0 or hi < 0)}

        comparisons = {f"gate_vs_{k}": boot(v) for k, v in column.items()}
        comparisons["gate_vs_best_static"] = comparisons[f"gate_vs_{best_static}"]
        denominator = means[best_static] - means["oracle_convex"]
        recovery = ((means[best_static] - means["gate"]) / denominator
                    if denominator > 0 else "CONVEX_GAP_UNAVAILABLE")
        g = cross["g"].to_numpy()
        report["datasets"][name] = {
            "n_origins": int(len(cross)), "n_series": int(per_series.shape[0]),
            "means": means, "best_static": best_static, "global_alpha": alpha,
            "fold_improvement": improvements,
            "folds_positive": int(sum(v > 0 for v in improvements.values())),
            "paired_bootstrap": comparisons, "convex_oracle_recovery": recovery,
            "gate_weight": {"mean": float(g.mean()), "sd": float(g.std()),
                            "extreme_share": float(((g < 0.05) | (g > 0.95)).mean())},
            "reference_pair": {
                # Ceilings are compared on the diversity screen's rows, which cover
                # all three folds for both pairs. The cross-fitted block above omits
                # fold 0 (it has no validation fold), so its levels are not
                # comparable to a three-fold reference.
                "convex_gain": reference["datasets"][name]["convex_gain"],
                "best_static_loss": reference["datasets"][name]["losses"][
                    reference["datasets"][name]["best_static"]],
                "convex_oracle_loss": reference["datasets"][name]["losses"]["oracle_origin_convex"],
                "pair_convex_oracle_loss_same_rows": screen_report["datasets"][name][
                    "pairs"][f"{a}|{b}"]["losses"]["oracle_origin_convex"],
                "pair_best_static_loss_same_rows": screen_report["datasets"][name][
                    "pairs"][f"{a}|{b}"]["best_static_loss"],
                "gate_v2_oof_improvement": reference_gate["datasets"][name][
                    "paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"],
                "gate_v2_oof_recovery": reference_gate["datasets"][name][
                    "convex_oracle_recovery"]}}
        cross.to_parquet(OUT / f"pair_gate_oof_{name}.parquet", index=False)
        print(f"[{name}] best_static={best_static} gate={means['gate']:.6f} "
              f"static={means[best_static]:.6f} "
              f"improvement={(means[best_static]-means['gate'])/means[best_static]*100:+.3f}% "
              f"recovery={recovery}")

    # Same-row comparison: both pairs scored over all three folds.
    ceiling_up = {n: (b["reference_pair"]["pair_convex_oracle_loss_same_rows"]
                      < b["reference_pair"]["convex_oracle_loss"])
                  for n, b in report["datasets"].items()}
    directions = {n: b["paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"] > 0
                  for n, b in report["datasets"].items()}
    excludes = {n: (b["paired_bootstrap"]["gate_vs_best_static"]["ci_excludes_zero"]
                    and b["paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"] > 0)
                for n, b in report["datasets"].items()}
    reversal = {n: b["paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"] < -0.005
                for n, b in report["datasets"].items()}
    better_than_reference = {
        n: (b["paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"]
            > b["reference_pair"]["gate_v2_oof_improvement"])
        for n, b in report["datasets"].items()}
    if not all(ceiling_up.values()):
        verdict = "DIVERSE_GATE_RED"
    elif (all(directions.values()) and any(excludes.values()) and not any(reversal.values())
          and any(better_than_reference.values())):
        verdict = "DIVERSE_GATE_GREEN"
    else:
        verdict = "DIVERSE_GATE_YELLOW"
    report["verdict"] = verdict
    report["ceiling_increased"] = ceiling_up
    report["direction_positive"] = directions
    report["ci_excludes_zero"] = excludes
    report["better_than_reference_gate"] = better_than_reference
    report["git_commit"] = cli._git_commit()
    (OUT / "pair_gate_result.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "ceiling_increased": ceiling_up,
                      "ci_excludes_zero": excludes,
                      "better_than_reference": better_than_reference}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("expert set selection")
    sub = parser.add_subparsers(required=True)
    s = sub.add_parser("select")
    s.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    s.set_defaults(func=cmd_select)
    g = sub.add_parser("gate")
    g.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    g.set_defaults(func=cmd_gate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
