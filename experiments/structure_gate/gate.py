"""Train a routing gate on OOF regret, choose one spec, freeze it, then open test.

The gate emits one scalar per forecast origin:

    y_mix = (1 - g) * y_point + g * y_hurdle

so g near zero trusts Point and g near one trusts Hurdle.  Horizon-wise gating
is deliberately not attempted here; the kill test asks whether routing helps at
all, and a per-step gate would change two things at once.

Experts are frozen throughout.  Nothing about Point or Hurdle is retrained or
tuned while the gate learns, so any difference in test error belongs to the
routing and not to a better forecaster.

Selection discipline: twelve configurations (3 feature variants x 2
architectures x 2 losses) are scored on the LATEST OOF fold, which no gate was
trained on, and exactly one is written into a GateSpec.  `test` refuses to run
without that file, so the ordering cannot be reversed after the fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..external_validity_screen import cli, prereg, rule_replication as rr, screen
from ..om_factorization_killtest import train as km_train
from ..unified_temporal_27_v3.training import train_scale
from . import features as F
from .oof import OUT
from .potential import load_predictions, per_origin_losses

SPEC = OUT / "gate_spec.json"
VARIANTS = ("G-STRUCT", "G-FULL", "G-NOSCALE")
ARCHITECTURES = ("linear", "mlp")
LOSS_MODES = ("softbce", "regret_weighted")

#: The gate must stay far smaller than the pair of experts it routes between,
#: so a win cannot be explained by capacity. Point + Hurdle carry 7056 each.
MLP_HIDDEN = 16
GATE_EPOCHS = 200
GATE_LR = 0.01
GATE_SEED = prereg.MODELS["canonical_model_seed"]
WEIGHT_CLIP_QUANTILE = 0.95

FINAL_CHECKPOINTS = {
    "m5": screen.OUT / "rule_replication" / "models",
    "favorita": screen.OUT / "favorita_transfer" / "models",
}
TRAINED_ON = {"m5": "stage_a_m5_1200", "favorita": "stage_a_favorita_1200"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class LinearGate(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.layer = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x)).squeeze(-1)


class MLPGate(nn.Module):
    def __init__(self, n_features: int, hidden: int = MLP_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def build_gate(architecture: str, n_features: int) -> nn.Module:
    return LinearGate(n_features) if architecture == "linear" else MLPGate(n_features)


def regret_target(regret: np.ndarray, tau: float) -> np.ndarray:
    """q = sigmoid(r / tau); r > 0 means Hurdle was better at this origin."""
    return 1.0 / (1.0 + np.exp(-regret / tau))


def choose_tau(regret: np.ndarray) -> float:
    """median(|r|) on the gate-train rows, floored so a degenerate fold cannot
    divide by ~0 and turn every soft target into a hard 0/1 label."""
    tau = float(np.median(np.abs(regret)))
    floor = float(np.quantile(np.abs(regret), 0.75)) * 1e-3
    return max(tau, floor, 1e-8)


def fit_gate(x: np.ndarray, q: np.ndarray, w: np.ndarray, architecture: str,
             device) -> nn.Module:
    torch.manual_seed(GATE_SEED)
    np.random.seed(GATE_SEED)
    model = build_gate(architecture, x.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=GATE_LR)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    qt = torch.tensor(q, dtype=torch.float32, device=device)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    for _ in range(GATE_EPOCHS):
        optimiser.zero_grad(set_to_none=True)
        g = model(xt).clamp(1e-6, 1 - 1e-6)
        bce = -(qt * torch.log(g) + (1 - qt) * torch.log(1 - g))
        (bce * wt).mean().backward()
        optimiser.step()
    model.eval()
    return model


def gate_weights(model: nn.Module, x: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def mixture_loss(losses: pd.DataFrame, predictions: pd.DataFrame,
                 scale: pd.Series, g: pd.Series) -> np.ndarray:
    """Per-origin normalised MSE of the gated mixture.

    Squaring does not commute with mixing, so the mixture error is recomputed
    from the two prediction paths rather than blended from the stored losses.
    """
    observed = predictions[predictions["target_mask"] > 0].copy()
    key = list(zip(observed["fold"], observed["series_id"], observed["origin"]))
    observed["g"] = g.reindex(pd.MultiIndex.from_tuples(key)).to_numpy()
    s = np.maximum(scale.loc[observed["series_id"]].to_numpy(np.float64), 1e-9)
    y = observed["y_observed"].to_numpy(np.float64)
    p = observed["point_mean_prediction"].to_numpy(np.float64)
    h = observed["hurdle_mean_prediction"].to_numpy(np.float64)
    gg = observed["g"].to_numpy(np.float64)
    observed["se"] = (((1 - gg) * p + gg * h - y) / s) ** 2
    return observed.groupby(["fold", "series_id", "origin"])["se"].mean()


def assemble(name: str, manifest: dict) -> dict:
    """OOF losses, per-origin features and the raw predictions, aligned."""
    data = screen.load_dataset(name)
    cfg = screen.config_for(name)
    scale = pd.Series(train_scale({"y": data["y"], "z": data["z"]}, cfg),
                      index=pd.Index(data["series_id"]).astype(str))
    predictions = load_predictions(name, len(manifest["datasets"][name]["folds"]))
    losses = per_origin_losses(predictions, scale, [0.5])
    regret = pd.read_parquet(OUT / f"oof_regret_{name}.parquet")
    regret["series_id"] = regret["series_id"].astype(str)
    expert_columns = [c for c in regret.columns if c.startswith(F.EXPERT_PREFIX)]
    losses = losses.merge(regret[["fold", "series_id", "origin", "regret"] + expert_columns],
                          on=["fold", "series_id", "origin"], how="inner")
    structure = F.structure_table(data, losses[["series_id", "origin"]])
    frame = pd.concat([losses.reset_index(drop=True),
                       structure.drop(columns=["series_id", "origin"])], axis=1)
    frame = F.add_missing_flags(frame, F.STRUCTURE_COLUMNS)
    return {"frame": frame, "predictions": predictions, "scale": scale,
            "data": data, "cfg": cfg}


def cmd_select(args) -> None:
    manifest = json.loads((OUT / "oof_manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"analysis": "gate variant selection on OOF only; test never read",
              "test_used": False, "selected_at_utc": _utc(),
              "candidates": {}, "datasets": {}}

    bundles = {name: assemble(name, manifest) for name in args.datasets}
    scores = {}
    for name, bundle in bundles.items():
        frame = bundle["frame"]
        folds = sorted(frame["fold"].unique())
        train_folds, validation_fold = folds[:-1], folds[-1]
        is_train = frame["fold"].isin(train_folds).to_numpy()
        report["datasets"][name] = {
            "gate_train_folds": [int(f) for f in train_folds],
            "gate_validation_fold": int(validation_fold),
            "split_rule": "time ordered; the latest OOF fold validates, never a random split",
            "n_train_rows": int(is_train.sum()),
            "n_validation_rows": int((~is_train).sum())}

        medians = F.fit_imputer(frame[is_train], F.STRUCTURE_COLUMNS)
        filled = F.apply_imputer(frame, medians)
        tau = choose_tau(frame.loc[is_train, "regret"].to_numpy(float))
        report["datasets"][name]["tau"] = tau
        report["datasets"][name]["imputation_medians"] = medians

        regret = frame["regret"].to_numpy(float)
        q = regret_target(regret, tau)
        magnitude = np.abs(regret)
        clip = float(np.quantile(magnitude[is_train], WEIGHT_CLIP_QUANTILE))
        scale_w = float(np.median(magnitude[is_train])) or 1.0
        weights = {"softbce": np.ones_like(regret),
                   "regret_weighted": np.minimum(magnitude, clip) / scale_w}
        report["datasets"][name]["weight_clip"] = clip

        for variant in VARIANTS:
            columns = F.feature_columns(variant, list(filled.columns))
            x = filled[columns].to_numpy(np.float64)
            centre, spread = x[is_train].mean(0), x[is_train].std(0)
            spread = np.where(spread > 0, spread, 1.0)
            xz = ((x - centre) / spread).astype(np.float32)
            for architecture in ARCHITECTURES:
                for mode in LOSS_MODES:
                    model = fit_gate(xz[is_train], q[is_train], weights[mode][is_train],
                                     architecture, device)
                    g = pd.Series(gate_weights(model, xz, device),
                                  index=pd.MultiIndex.from_arrays(
                                      [frame["fold"], frame["series_id"], frame["origin"]]))
                    per_origin = mixture_loss(frame, bundle["predictions"],
                                              bundle["scale"], g)
                    validation = frame["fold"].to_numpy() == validation_fold
                    key = f"{variant}|{architecture}|{mode}"
                    loss = float(per_origin.to_numpy()[validation].mean())
                    scores.setdefault(key, {})[name] = loss
                    report["candidates"].setdefault(key, {})[name] = {
                        "gate_validation_loss": loss,
                        "n_features": int(xz.shape[1]),
                        "n_parameters": int(sum(p.numel() for p in model.parameters())),
                        "mean_g": float(g.to_numpy()[validation].mean())}
                    print(f"[{name}] {key:<34} val={loss:.6f} "
                          f"mean_g={g.to_numpy()[validation].mean():.3f}", flush=True)

    # One spec for both datasets: the kill test asks whether a single routing
    # design transfers, not whether each dataset can be fitted its own gate.
    best_per_dataset = {n: min(scores[k][n] for k in scores) for n in args.datasets}

    def penalty(key: str) -> float:
        """Mean ratio to each dataset's own best candidate.

        Raw losses are on different scales between M5 and Favorita, so summing
        them would let Favorita decide alone.
        """
        return float(np.mean([scores[key][n] / best_per_dataset[n] for n in scores[key]]))

    ranked = sorted(scores, key=penalty)
    selected = ranked[0]
    report["candidate_penalty"] = {k: penalty(k) for k in ranked}
    variant, architecture, mode = selected.split("|")
    report["selection_rule"] = ("mean across datasets of the ratio to that dataset's best "
                                "candidate; one spec must serve both")
    report["selected"] = {"variant": variant, "architecture": architecture,
                          "loss_mode": mode, "key": selected}
    report["ranking"] = ranked
    report["git_commit"] = cli._git_commit()
    (OUT / "gate_selection.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["selected"], indent=2))


def train_final_gate(name: str, manifest: dict, variant: str, architecture: str,
                     mode: str, device):
    """Refit the selected configuration on exactly the rows it was validated from.

    Not on all three folds: the gate that goes to test has to be the one whose
    held-out loss justified choosing it, otherwise the selection evidence
    describes a different model than the one being reported.
    """
    bundle = assemble(name, manifest)
    frame = bundle["frame"]
    folds = sorted(frame["fold"].unique())
    train_folds, validation_fold = folds[:-1], folds[-1]
    is_train = frame["fold"].isin(train_folds).to_numpy()

    medians = F.fit_imputer(frame[is_train], F.STRUCTURE_COLUMNS)
    filled = F.apply_imputer(frame, medians)
    tau = choose_tau(frame.loc[is_train, "regret"].to_numpy(float))
    regret = frame["regret"].to_numpy(float)
    q = regret_target(regret, tau)
    magnitude = np.abs(regret)
    clip = float(np.quantile(magnitude[is_train], WEIGHT_CLIP_QUANTILE))
    scale_w = float(np.median(magnitude[is_train])) or 1.0
    weights = (np.ones_like(regret) if mode == "softbce"
               else np.minimum(magnitude, clip) / scale_w)

    columns = F.feature_columns(variant, list(filled.columns))
    x = filled[columns].to_numpy(np.float64)
    centre, spread = x[is_train].mean(0), x[is_train].std(0)
    spread = np.where(spread > 0, spread, 1.0)
    xz = ((x - centre) / spread).astype(np.float32)
    model = fit_gate(xz[is_train], q[is_train], weights[is_train], architecture, device)
    return {"model": model, "columns": columns, "centre": centre.tolist(),
            "spread": spread.tolist(), "medians": medians, "tau": tau,
            "weight_clip": clip, "weight_scale": scale_w,
            "gate_train_folds": [int(f) for f in train_folds],
            "gate_validation_fold": int(validation_fold),
            "n_parameters": int(sum(p.numel() for p in model.parameters()))}


def cmd_freeze(args) -> None:
    if SPEC.exists():
        raise SystemExit(f"{SPEC} already frozen; refusing to overwrite")
    selection = json.loads((OUT / "gate_selection.json").read_text())
    manifest = json.loads((OUT / "oof_manifest.json").read_text())
    variant = selection["selected"]["variant"]
    architecture = selection["selected"]["architecture"]
    mode = selection["selected"]["loss_mode"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = {"study": "structure-gated Point-Hurdle routing, kill test",
            "frozen_at_utc": _utc(), "frozen_before_any_test_prediction": True,
            "selected": selection["selected"],
            "selection_rule": selection["selection_rule"],
            "experts_frozen": True, "joint_training": False,
            "gate_unit": "one scalar per forecast origin, shared across the horizon",
            "mixture": "y = (1-g) * point + g * hurdle",
            "hyperparameters": {"epochs": GATE_EPOCHS, "lr": GATE_LR,
                                "seed": GATE_SEED, "mlp_hidden": MLP_HIDDEN,
                                "weight_clip_quantile": WEIGHT_CLIP_QUANTILE},
            "ablation_variants": list(VARIANTS), "datasets": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        fitted = train_final_gate(name, manifest, variant, architecture, mode, device)
        torch.save({"state_dict": fitted["model"].state_dict(),
                    "architecture": architecture, "variant": variant,
                    "loss_mode": mode, "columns": fitted["columns"],
                    "centre": fitted["centre"], "spread": fitted["spread"],
                    "medians": fitted["medians"], "tau": fitted["tau"]},
                   OUT / f"gate_{name}.pt")
        spec["datasets"][name] = {k: v for k, v in fitted.items() if k != "model"}
        spec["datasets"][name]["checkpoint"] = f"gate_{name}.pt"
        spec["datasets"][name]["expert_checkpoints"] = str(
            FINAL_CHECKPOINTS[name].relative_to(screen.REPO))
        spec["datasets"][name]["expert_trained_on"] = TRAINED_ON[name]
        print(f"[{name}] frozen: {len(fitted['columns'])} features, "
              f"{fitted['n_parameters']} gate parameters, tau={fitted['tau']:.6f}")
    spec["git_commit"] = cli._git_commit()
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    print(f"froze {SPEC}")


def main() -> None:
    parser = argparse.ArgumentParser("structure gate")
    sub = parser.add_subparsers(required=True)
    s = sub.add_parser("select")
    s.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    s.set_defaults(func=cmd_select)
    f = sub.add_parser("freeze")
    f.add_argument("--datasets", nargs="*", default=["m5", "favorita"])
    f.set_defaults(func=cmd_freeze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
