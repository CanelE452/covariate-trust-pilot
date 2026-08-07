"""Gate-v2: learn the mixture weight by minimising the mixture's own error.

Gate-v1 minimised a binary cross-entropy against "which expert won".  That
answers a different question from the one the forecast asks, so this fits the
same gate against the loss it will actually be judged on:

    Y_hat = (1 - g) P + g H,     L = normalised MSE(Y_hat, Y)

P and H enter as constants.  The experts are frozen and never receive gradient,
so a change in error belongs to the routing.

Three losses are compared on OOF: the v1 regret BCE, this direct mixture MSE,
and a regression onto the exact convex-oracle weight.  Selection uses every OOF
fold in an expanding-time scheme rather than v1's single latest fold, because
that single fold picked a variant that later looked like an overfit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..external_validity_screen import cli, prereg, screen
from ..unified_temporal_27_v3.training import train_scale
from . import features as F, gate as G
from .oof import OUT
from .potential import load_predictions

SPEC = OUT / "gate_v2_spec.json"
LOSSES = ("regret_bce", "direct_mixture", "gstar_regression")
EPOCHS = 400
LR = 0.01
SEED = prereg.MODELS["canonical_model_seed"]

#: g* is only meaningful where the experts actually differ. Weight each
#: regression sample by its disagreement, capped so one origin cannot dominate.
#: Fixed here, before any OOF result was inspected.
GSTAR_WEIGHT_CLIP_QUANTILE = 0.95


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def gradient_integrity(device) -> dict:
    """Step 8: the mixture must be exact and the gradient must reach only the gate."""
    from ..om_factorization_killtest import models as km_models
    torch.manual_seed(SEED)
    point = km_models.BUILDERS[prereg.MODELS["point"]](96, 28).to(device)
    hurdle = km_models.BUILDERS[prereg.MODELS["hurdle"]](96, 28).to(device)
    for expert in (point, hurdle):
        for parameter in expert.parameters():
            parameter.requires_grad_(False)

    history = torch.rand(16, 96, device=device) * 3
    scale = torch.ones(16, 1, device=device)
    y = torch.rand(16, 28, device=device) * 3
    with torch.no_grad():
        p = point(history, scale)["mean_prediction"]
        h = hurdle(history, scale)["mean_prediction"]

    checks = {}
    for value, expected in ((0.0, p), (1.0, h), (0.5, 0.5 * (p + h))):
        mixed = (1 - value) * p + value * h
        checks[f"identity_g_{value}"] = {
            "max_abs_error": float((mixed - expected).abs().max()),
            "ok": bool(torch.allclose(mixed, expected, atol=0, rtol=0))}

    gate = G.LinearGate(4).to(device)
    x = torch.randn(16, 4, device=device)
    g = gate(x).unsqueeze(1)
    loss = (((1 - g) * p + g * h - y) ** 2).mean()
    loss.backward()
    checks["expert_grads_absent"] = {
        "point": all(q.grad is None or float(q.grad.abs().sum()) == 0 for q in point.parameters()),
        "hurdle": all(q.grad is None or float(q.grad.abs().sum()) == 0 for q in hurdle.parameters()),
        "gate_receives_gradient": bool(
            sum(float(q.grad.abs().sum()) for q in gate.parameters()) > 0)}

    # A gate cannot be pushed anywhere when the two experts agree.
    magnitudes = {}
    for label, spread in (("identical", 0.0), ("different", 1.0)):
        gate2 = G.LinearGate(4).to(device)
        h2 = p + spread * torch.rand_like(p)
        g2 = gate2(x).unsqueeze(1)
        ((((1 - g2) * p + g2 * h2 - y) ** 2).mean()).backward()
        magnitudes[label] = float(sum(float(q.grad.abs().sum()) for q in gate2.parameters()))
    checks["gradient_vanishes_when_experts_agree"] = {
        "identical": magnitudes["identical"], "different": magnitudes["different"],
        "ok": magnitudes["identical"] < magnitudes["different"]}

    checks["passed"] = bool(
        all(checks[f"identity_g_{v}"]["ok"] for v in (0.0, 1.0, 0.5))
        and checks["expert_grads_absent"]["point"]
        and checks["expert_grads_absent"]["hurdle"]
        and checks["expert_grads_absent"]["gate_receives_gradient"]
        and checks["gradient_vanishes_when_experts_agree"]["ok"])
    return checks


def assemble_v2(name: str, manifest: dict) -> pd.DataFrame:
    """Per-origin quadratic coefficients, g*, and the v1 feature block."""
    gstar = pd.read_parquet(OUT / f"oof_gstar_{name}.parquet")
    gstar["series_id"] = gstar["series_id"].astype(str)
    regret = pd.read_parquet(OUT / f"oof_regret_{name}.parquet")
    regret["series_id"] = regret["series_id"].astype(str)
    expert_columns = [c for c in regret.columns if c.startswith(F.EXPERT_PREFIX)]
    frame = gstar.merge(regret[["fold", "series_id", "origin", "regret"] + expert_columns],
                        on=["fold", "series_id", "origin"], how="inner")
    data = screen.load_dataset(name)
    structure = F.structure_table(data, frame[["series_id", "origin"]])
    frame = pd.concat([frame.reset_index(drop=True),
                       structure.drop(columns=["series_id", "origin"])], axis=1)
    return F.add_missing_flags(frame, F.STRUCTURE_COLUMNS)


def mixture_mse(frame: pd.DataFrame, g: np.ndarray) -> np.ndarray:
    """sse((1-g)P + gH) / n, from the stored quadratic coefficients."""
    sse = (frame["sse_point"].to_numpy(np.float64)
           - 2 * g * frame["num"].to_numpy(np.float64)
           + g ** 2 * frame["den"].to_numpy(np.float64))
    return sse / frame["n_observed"].to_numpy(np.float64)


def fit(loss_name: str, x: np.ndarray, block: pd.DataFrame, architecture: str,
        device) -> torch.nn.Module:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = G.build_gate(architecture, x.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    xt = torch.tensor(x, dtype=torch.float32, device=device)

    if loss_name == "regret_bce":
        tau = G.choose_tau(block["regret"].to_numpy(float))
        target = torch.tensor(G.regret_target(block["regret"].to_numpy(float), tau),
                              dtype=torch.float32, device=device)
    elif loss_name == "direct_mixture":
        sse_p = torch.tensor(block["sse_point"].to_numpy(np.float64), dtype=torch.float32, device=device)
        num = torch.tensor(block["num"].to_numpy(np.float64), dtype=torch.float32, device=device)
        den = torch.tensor(block["den"].to_numpy(np.float64), dtype=torch.float32, device=device)
        n_obs = torch.tensor(block["n_observed"].to_numpy(np.float64), dtype=torch.float32, device=device)
    else:
        keep = ~block["flat"].to_numpy()
        weight = block["mean_abs_disagreement"].to_numpy(np.float64)
        clip = float(np.quantile(weight[keep], GSTAR_WEIGHT_CLIP_QUANTILE)) if keep.any() else 1.0
        weight = np.where(keep, np.minimum(weight, clip), 0.0)
        weight = weight / max(weight.mean(), 1e-12)
        target = torch.tensor(block["g_star"].to_numpy(np.float64), dtype=torch.float32, device=device)
        weight_t = torch.tensor(weight, dtype=torch.float32, device=device)

    for _ in range(EPOCHS):
        optimiser.zero_grad(set_to_none=True)
        g = model(xt)
        if loss_name == "regret_bce":
            gc = g.clamp(1e-6, 1 - 1e-6)
            loss = (-(target * torch.log(gc) + (1 - target) * torch.log(1 - gc))).mean()
        elif loss_name == "direct_mixture":
            loss = ((sse_p - 2 * g * num + g ** 2 * den) / n_obs).mean()
        else:
            loss = (weight_t * (g - target) ** 2).mean()
        loss.backward()
        optimiser.step()
    model.eval()
    return model


def cmd_gradcheck(_args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = gradient_integrity(device)
    (OUT / "gate_v2_gradient_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    print(json.dumps(checks, indent=2, default=str))
    if not checks["passed"]:
        raise SystemExit("GRADIENT_INTEGRITY_FAIL")


def cmd_select(args) -> None:
    manifest = json.loads((OUT / "oof_manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gradients = gradient_integrity(device)
    if not gradients["passed"]:
        raise SystemExit("GRADIENT_INTEGRITY_FAIL")

    report = {"analysis": "Gate-v2 cross-fold selection on OOF only",
              "test_used": False, "existing_test_reused": False,
              "selected_at_utc": _utc(), "gradient_checks": gradients,
              "cv_scheme": ("expanding time: a gate is fitted only on folds strictly "
                            "earlier than the one it is scored on, so fold 0 has no "
                            "validation row and leave-one-fold-out is not used"),
              "candidates": {}, "datasets": {}}
    scores = {}

    for name in args.datasets:
        frame = assemble_v2(name, manifest)
        folds = sorted(frame["fold"].unique())
        report["datasets"][name] = {"folds": [int(f) for f in folds],
                                    "n_origins": int(len(frame))}
        for variant in G.VARIANTS:
            columns = F.feature_columns(variant, list(frame.columns))
            for architecture in G.ARCHITECTURES:
                for loss_name in LOSSES:
                    key = f"{variant}|{architecture}|{loss_name}"
                    improvements, n_parameters = {}, None
                    for validation in folds[1:]:
                        train_mask = frame["fold"].to_numpy() < validation
                        valid_mask = frame["fold"].to_numpy() == validation
                        block = frame[train_mask]
                        medians = F.fit_imputer(block, F.STRUCTURE_COLUMNS)
                        filled = F.apply_imputer(frame, medians)
                        x = filled[columns].to_numpy(np.float64)
                        centre = x[train_mask].mean(0)
                        spread = np.where(x[train_mask].std(0) > 0, x[train_mask].std(0), 1.0)
                        xz = ((x - centre) / spread).astype(np.float32)
                        model = fit(loss_name, xz[train_mask], block, architecture, device)
                        n_parameters = int(sum(p.numel() for p in model.parameters()))
                        g = G.gate_weights(model, xz[valid_mask], device)
                        held = frame[valid_mask]
                        gate_loss = float(mixture_mse(held, g.astype(np.float64)).mean())
                        static = min(float(held["mse_fifty"].mean()),
                                     float(held["mse_point"].mean()),
                                     float(held["mse_hurdle"].mean()))
                        improvements[int(validation)] = (static - gate_loss) / static
                    values = list(improvements.values())
                    report["candidates"].setdefault(key, {})[name] = {
                        "fold_improvement": improvements,
                        "mean": float(np.mean(values)), "median": float(np.median(values)),
                        "worst": float(np.min(values)),
                        "sign_consistent": bool(len({v > 0 for v in values}) == 1),
                        "n_parameters": n_parameters}
                    scores.setdefault(key, {})[name] = {
                        "mean": float(np.mean(values)), "worst": float(np.min(values)),
                        "n_parameters": n_parameters}
                    print(f"[{name}] {key:<42} mean={np.mean(values)*100:+.3f}% "
                          f"worst={np.min(values)*100:+.3f}%", flush=True)

    def rank_key(key: str):
        means = [scores[key][n]["mean"] for n in scores[key]]
        worsts = [scores[key][n]["worst"] for n in scores[key]]
        size = max(scores[key][n]["n_parameters"] for n in scores[key])
        return (-float(np.mean(means)), -float(np.min(worsts)), size)

    ranked = sorted(scores, key=rank_key)
    selected = ranked[0]
    variant, architecture, loss_name = selected.split("|")
    report["selection_criterion"] = ("mean improvement over best static across datasets, "
                                     "then worst fold, then fewer parameters; test never used")
    report["selected"] = {"variant": variant, "architecture": architecture,
                          "loss": loss_name, "key": selected}
    report["ranking"] = ranked[:10]
    report["git_commit"] = cli._git_commit()
    (OUT / "gate_v2_selection.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["selected"], indent=2))


def cross_fitted(name: str, frame: pd.DataFrame, columns, architecture: str,
                 loss_name: str, device) -> pd.DataFrame:
    """One held-out gate weight per origin, from a gate that never saw its fold."""
    folds = sorted(frame["fold"].unique())
    pieces = []
    for validation in folds[1:]:
        train_mask = frame["fold"].to_numpy() < validation
        valid_mask = frame["fold"].to_numpy() == validation
        block = frame[train_mask]
        medians = F.fit_imputer(block, F.STRUCTURE_COLUMNS)
        filled = F.apply_imputer(frame, medians)
        x = filled[columns].to_numpy(np.float64)
        centre = x[train_mask].mean(0)
        spread = np.where(x[train_mask].std(0) > 0, x[train_mask].std(0), 1.0)
        xz = ((x - centre) / spread).astype(np.float32)
        model = fit(loss_name, xz[train_mask], block, architecture, device)
        held = frame[valid_mask].copy()
        held["g"] = G.gate_weights(model, xz[valid_mask], device).astype(np.float64)
        pieces.append(held)
    return pd.concat(pieces, ignore_index=True)


def series_bootstrap(per_series: pd.DataFrame, a: str, b: str, draws: int, seed: int) -> dict:
    x = per_series[a].to_numpy(np.float64)
    y = per_series[b].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        out[i] = (y[idx].mean() - x[idx].mean()) / y[idx].mean()
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {"relative_improvement": float((y.mean() - x.mean()) / y.mean()),
            "ci": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def cmd_aggregate(args) -> None:
    manifest = json.loads((OUT / "oof_manifest.json").read_text())
    selection = json.loads((OUT / "gate_v2_selection.json").read_text())
    convex = json.loads((OUT / "convex_oracle.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chosen = selection["selected"]
    draws, seed = prereg.BOOTSTRAP["draws"], prereg.BOOTSTRAP["seed"]

    report = {"analysis": "Gate-v2 aggregate on cross-fitted OOF; existing TEST untouched",
              "test_used": False, "existing_test_reused": False,
              "selected": chosen, "computed_at_utc": _utc(), "datasets": {}}

    for name in args.datasets:
        frame = assemble_v2(name, manifest)
        columns = F.feature_columns(chosen["variant"], list(frame.columns))
        held = {}
        for loss_name in LOSSES:
            held[loss_name] = cross_fitted(name, frame, columns, chosen["architecture"],
                                           loss_name, device)
        base = held[chosen["loss"]][["fold", "series_id", "origin", "n_observed",
                                     "mse_point", "mse_hurdle", "mse_fifty",
                                     "mse_convex", "mse_hard"]].copy()
        alpha = float(convex["datasets"][name]["global_alpha"])
        source = held[chosen["loss"]]
        base["mse_global_alpha"] = mixture_mse(source, np.full(len(source), alpha))
        for loss_name, block in held.items():
            base[f"mse_gate_{loss_name}"] = mixture_mse(block, block["g"].to_numpy(np.float64))
        base["g_selected"] = source["g"].to_numpy(np.float64)

        per_series = base.groupby("series_id")[
            [c for c in base.columns if c.startswith("mse_")]].mean()
        means = {c.replace("mse_", ""): float(base[c].mean())
                 for c in base.columns if c.startswith("mse_")}
        best_static = min(("point", "hurdle", "fifty", "global_alpha"), key=lambda k: means[k])
        gate_column = f"mse_gate_{chosen['loss']}"
        comparisons = {f"gate_vs_{m}": series_bootstrap(per_series, gate_column, f"mse_{m}",
                                                        draws, seed)
                       for m in ("point", "hurdle", "fifty", "global_alpha", best_static)}
        comparisons["gate_vs_best_static"] = comparisons[f"gate_vs_{best_static}"]

        e_static, e_gate = means[best_static], means[f"gate_{chosen['loss']}"]
        e_convex = means["convex"]
        denominator = e_static - e_convex
        recovery = ((e_static - e_gate) / denominator if denominator > 0
                    else "CONVEX_GAP_UNAVAILABLE")
        g = base["g_selected"].to_numpy()
        fold_improvement = selection["candidates"][chosen["key"]][name]["fold_improvement"]
        report["datasets"][name] = {
            "n_origins": int(len(base)), "n_series": int(per_series.shape[0]),
            "means": means, "best_static": best_static, "global_alpha": alpha,
            "paired_bootstrap": comparisons,
            "convex_oracle_recovery": recovery,
            "fold_improvement": fold_improvement,
            "folds_positive": int(sum(v > 0 for v in fold_improvement.values())),
            "n_folds_scored": len(fold_improvement),
            "gate_weight": {"mean": float(g.mean()), "sd": float(g.std()),
                            "min": float(g.min()), "max": float(g.max()),
                            "extreme_share": float(((g < 0.05) | (g > 0.95)).mean())}}
        base.to_parquet(OUT / f"gate_v2_oof_{name}.parquet", index=False)
        print(f"[{name}] best_static={best_static} gate={e_gate:.6f} static={e_static:.6f} "
              f"improvement={(e_static-e_gate)/e_static*100:+.3f}% recovery={recovery}")

    directions = {n: b["paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"] > 0
                  for n, b in report["datasets"].items()}
    excludes = {n: b["paired_bootstrap"]["gate_vs_best_static"]["ci_excludes_zero"]
                and b["paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"] > 0
                for n, b in report["datasets"].items()}
    folds_ok = {n: b["folds_positive"] >= max(1, b["n_folds_scored"] - 1)
                for n, b in report["datasets"].items()}
    reversal = {n: b["paired_bootstrap"]["gate_vs_best_static"]["relative_improvement"] < -0.005
                for n, b in report["datasets"].items()}
    if not any(directions.values()):
        verdict = "GATE_V2_OOF_RED"
    elif all(directions.values()) and any(excludes.values()) and not any(reversal.values()) \
            and all(folds_ok.values()):
        verdict = "GATE_V2_OOF_GREEN"
    else:
        verdict = "GATE_V2_OOF_YELLOW"
    report["verdict"] = verdict
    report["direction_positive"] = directions
    report["ci_excludes_zero"] = excludes
    report["fold_consistency_ok"] = folds_ok
    report["git_commit"] = cli._git_commit()
    (OUT / "gate_v2_oof_result.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "direction": directions,
                      "ci_excludes_zero": excludes}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("Gate-v2")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("gradcheck").set_defaults(func=cmd_gradcheck)
    s = sub.add_parser("select")
    s.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    s.set_defaults(func=cmd_select)
    a = sub.add_parser("aggregate")
    a.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    a.set_defaults(func=cmd_aggregate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
