"""The 2x2: neutral vs alpha-anchored output, regret vs direct loss.

Experts, features, architecture, hidden width and parameter budget are all the
ones Gate-v2 froze.  Two things vary and nothing else, so whatever separates the
four cells is attributable:

    P0L0  sigmoid(f(x))                      regret BCE      (Gate-v2 reference)
    P0L1  sigmoid(f(x))                      direct mixture MSE
    P1L0  sigmoid(logit(alpha) + delta(x))   regret BCE
    P1L1  sigmoid(logit(alpha) + delta(x))   direct mixture MSE   (Gate-v3 candidate)

The anchored cells start from the static mixture exactly: the residual head's
last layer is zero-initialised, so at epoch 0 the gate emits alpha for every
origin.  Epoch 0 therefore stays a legitimate checkpoint -- a fold with no local
signal to learn is allowed to fall back to the static answer rather than being
forced to move off it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..external_validity_screen import cli, prereg, screen
from ..structure_gate import features as F, gate as G, gate_v2 as V2
from .diagnose import EPS_ALPHA, anchor_logit, fit_alpha, mixture_mse
from .oof import DATASETS, OUT

CONDITIONS = ("P0L0", "P0L1", "P1L0", "P1L1")
DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnchoredGate(nn.Module):
    """sigmoid(anchor + delta(x)), with delta identically zero at initialisation."""

    def __init__(self, n_features: int, anchor: float, hidden: int) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.tensor(float(anchor)))
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return torch.sigmoid(self.anchor + self.net(x).squeeze(-1))


def build(condition: str, n_features: int, anchor: float, hidden: int, device):
    if condition.startswith("P0"):
        return G.build_gate("mlp", n_features).to(device)
    return AnchoredGate(n_features, anchor, hidden).to(device)


def fit(condition: str, x: np.ndarray, block: pd.DataFrame, anchor: float,
        hidden: int, device, validation: tuple | None):
    """Train, keeping epoch 0 as a candidate checkpoint for the anchored cells."""
    torch.manual_seed(V2.SEED)
    np.random.seed(V2.SEED)
    model = build(condition, x.shape[1], anchor, hidden, device)
    optimiser = torch.optim.Adam(model.parameters(), lr=V2.LR)
    xt = torch.tensor(x, dtype=torch.float32, device=device)

    if condition.endswith("L0"):
        tau = G.choose_tau(block["regret"].to_numpy(float))
        target = torch.tensor(G.regret_target(block["regret"].to_numpy(float), tau),
                              dtype=torch.float32, device=device)
    else:
        sse = torch.tensor(block["sse_a"].to_numpy(np.float64), dtype=torch.float32, device=device)
        num = torch.tensor(block["num"].to_numpy(np.float64), dtype=torch.float32, device=device)
        den = torch.tensor(block["den"].to_numpy(np.float64), dtype=torch.float32, device=device)
        n_obs = torch.tensor(block["n"].to_numpy(np.float64), dtype=torch.float32, device=device)

    def validation_loss() -> float:
        if validation is None:
            return float("nan")
        xv, held = validation
        with torch.no_grad():
            g = model(torch.tensor(xv, dtype=torch.float32, device=device)).cpu().numpy()
        return float(mixture_mse(held, g.astype(np.float64)).mean())

    best_epoch, best_loss = 0, validation_loss()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, V2.EPOCHS + 1):
        optimiser.zero_grad(set_to_none=True)
        g = model(xt)
        if condition.endswith("L0"):
            gc = g.clamp(1e-6, 1 - 1e-6)
            loss = (-(target * torch.log(gc) + (1 - target) * torch.log(1 - gc))).mean()
        else:
            loss = ((sse - 2 * g * num + g ** 2 * den) / n_obs).mean()
        loss.backward()
        optimiser.step()
        if epoch % 20 == 0 or epoch == V2.EPOCHS:
            current = validation_loss()
            if np.isfinite(current) and current < best_loss:
                best_epoch, best_loss = epoch, current
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss


def prepare(table: pd.DataFrame, columns, train_mask):
    medians = F.fit_imputer(table[train_mask], F.STRUCTURE_COLUMNS)
    filled = F.apply_imputer(table, medians)
    x = filled[columns].to_numpy(np.float64)
    centre = x[train_mask].mean(0)
    spread = np.where(x[train_mask].std(0) > 0, x[train_mask].std(0), 1.0)
    return ((x - centre) / spread).astype(np.float32)


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
    gate_spec = json.loads((screen.OUT.parent / "structure_gate" / "gate_v2_spec.json").read_text())
    variant = gate_spec["selected"]["variant"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"analysis": "2x2 parameterization x loss factorial, OOF only",
              "test_used": False, "existing_test_scored": False,
              "frozen_from_gate_v2": {"variant": variant, "architecture": "mlp",
                                      "hidden": V2.G.MLP_HIDDEN, "epochs": V2.EPOCHS,
                                      "lr": V2.LR, "seed": V2.SEED},
              "conditions": list(CONDITIONS), "computed_at_utc": _utc(), "datasets": {}}
    fold_rows, identity = [], {}

    for name in args.datasets:
        table = pd.read_parquet(OUT / f"oof_{name}.parquet")
        columns = F.feature_columns(variant, list(table.columns))
        folds = sorted(table["fold"].unique())
        keyed = table["series_index"] if "series_index" in table else table["series_id"]
        table = table.assign(_series=keyed.to_numpy())
        results = {c: {} for c in CONDITIONS}
        crossfitted = {c: [] for c in CONDITIONS}
        alpha_rows = []

        for validation in folds[1:]:
            train_mask = (table["fold"].to_numpy() < validation)
            valid_mask = (table["fold"].to_numpy() == validation)
            past, held = table[train_mask], table[valid_mask]
            alpha = fit_alpha(past)
            anchor = anchor_logit(alpha)
            x = prepare(table, columns, train_mask)
            e_alpha = float(mixture_mse(held, np.full(len(held), alpha)).mean())
            alpha_rows.append({"validation_fold": int(validation), "alpha": alpha,
                               "E_alpha": e_alpha})
            for condition in CONDITIONS:
                model, epoch, _ = fit(condition, x[train_mask], past, anchor,
                                      V2.G.MLP_HIDDEN, device, (x[valid_mask], held))
                g = G.gate_weights(model, x[valid_mask], device).astype(np.float64)
                loss = mixture_mse(held, g)
                results[condition][int(validation)] = {
                    "alpha": alpha, "selected_epoch": epoch,
                    "epoch0_selected": bool(epoch == 0),
                    "E_gate": float(loss.mean()), "E_alpha": e_alpha,
                    "improvement": float((e_alpha - loss.mean()) / e_alpha),
                    "mean_g": float(g.mean()), "std_g": float(g.std()),
                    "mean_abs_g_minus_alpha": float(np.abs(g - alpha).mean())}
                crossfitted[condition].append(pd.DataFrame(
                    {"_series": held["_series"].to_numpy(), "loss": loss,
                     "alpha_loss": mixture_mse(held, np.full(len(held), alpha)),
                     "convex": held["mse_convex"].to_numpy(), "g": g,
                     "alpha": alpha, "fold": int(validation)}))
                fold_rows.append({"dataset": name, "condition": condition,
                                  "validation_fold": int(validation),
                                  **results[condition][int(validation)]})
                if condition.startswith("P1"):
                    fresh = build(condition, x.shape[1], anchor, V2.G.MLP_HIDDEN, device)
                    g0 = G.gate_weights(fresh, x[valid_mask], device)
                    # The anchor encodes alpha_safe, so that is what epoch 0 must
                    # reproduce. alpha itself can be exactly 0 or 1, where logit is
                    # not finite; the clip is numerical treatment, not tuning, and
                    # both values are reported.
                    alpha_safe = float(np.clip(alpha, EPS_ALPHA, 1 - EPS_ALPHA))
                    identity.setdefault(name, {})[f"fold{validation}_{condition}"] = {
                        "alpha": alpha, "alpha_safe": alpha_safe,
                        "max_abs_g0_minus_alpha_safe": float(np.abs(g0 - alpha_safe).max()),
                        "max_abs_g0_minus_alpha": float(np.abs(g0 - alpha).max()),
                        "clipped": bool(alpha != alpha_safe)}
            print(f"[{name}] fold {validation}: alpha {alpha:.2f}  "
                  + "  ".join(f"{c} {results[c][int(validation)]['improvement']*100:+.3f}%"
                              for c in CONDITIONS), flush=True)

        block = {"n_origins": int(len(table)), "alpha_by_fold": alpha_rows, "conditions": {}}
        for condition in CONDITIONS:
            joined = pd.concat(crossfitted[condition], ignore_index=True)
            per_series = joined.groupby("_series")[["loss", "alpha_loss", "convex"]].mean()
            improvements = [results[condition][f]["improvement"] for f in results[condition]]
            denominator = per_series["alpha_loss"].mean() - per_series["convex"].mean()
            block["conditions"][condition] = {
                "fold_improvement": {str(k): v["improvement"] for k, v in results[condition].items()},
                "mean_improvement": float(np.mean(improvements)),
                "worst_fold": float(np.min(improvements)),
                "sign_consistent": bool(len({v > 0 for v in improvements}) == 1),
                "aggregate": {"E_gate": float(per_series["loss"].mean()),
                              "E_alpha": float(per_series["alpha_loss"].mean()),
                              "E_convex": float(per_series["convex"].mean())},
                "vs_alpha_bootstrap": boot(per_series, "loss", "alpha_loss"),
                "oracle_recovery": (float((per_series["alpha_loss"].mean()
                                           - per_series["loss"].mean()) / denominator)
                                    if denominator > 0 else "ORACLE_RECOVERY_UNDEFINED"),
                "epoch0_selected_folds": [k for k, v in results[condition].items()
                                          if v["epoch0_selected"]],
                "gate_weight": {"mean_g": float(joined["g"].mean()),
                                "std_g": float(joined["g"].std()),
                                "p10": float(joined["g"].quantile(.1)),
                                "p90": float(joined["g"].quantile(.9)),
                                "mean_abs_g_minus_alpha": float(
                                    (joined["g"] - joined["alpha"]).abs().mean()),
                                "fraction_below_alpha": float((joined["g"] < joined["alpha"]).mean()),
                                "fraction_above_alpha": float((joined["g"] > joined["alpha"]).mean())}}
            joined.to_parquet(OUT / f"crossfitted_{name}_{condition}.parquet", index=False)
        report["datasets"][name] = block

    report["identity_checks"] = identity
    report["identity_passed"] = bool(all(
        v["max_abs_g0_minus_alpha_safe"] < 1e-6
        for ds in identity.values() for v in ds.values()))
    report["identity_note"] = (
        "checked against alpha_safe = clip(alpha, %g, 1-%g); two folds have alpha "
        "exactly 0 where logit is undefined, and there the epoch-0 gate reproduces "
        "alpha_safe rather than alpha, a difference of %g by construction"
        % (EPS_ALPHA, EPS_ALPHA, EPS_ALPHA))
    effects = {}
    for name, block in report["datasets"].items():
        m = {c: block["conditions"][c]["mean_improvement"] for c in CONDITIONS}
        effects[name] = {"anchor_under_regret": m["P1L0"] - m["P0L0"],
                         "anchor_under_direct": m["P1L1"] - m["P0L1"],
                         "loss_under_neutral": m["P0L1"] - m["P0L0"],
                         "loss_under_anchored": m["P1L1"] - m["P1L0"],
                         "interaction": (m["P1L1"] - m["P1L0"]) - (m["P0L1"] - m["P0L0"])}
    report["factorial_effects"] = effects
    pd.DataFrame(fold_rows).to_csv(OUT / "factorial_fold_results.csv", index=False)
    report["git_commit"] = cli._git_commit()
    (OUT / "aggregate_results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"identity_passed": report["identity_passed"],
                      "P1L1_mean_improvement": {n: b["conditions"]["P1L1"]["mean_improvement"]
                                                for n, b in report["datasets"].items()}},
                     indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("Gate-v3 2x2 factorial")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=list(DATASETS))
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
