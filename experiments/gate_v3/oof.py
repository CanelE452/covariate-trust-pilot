"""Assemble one OOF table per dataset: quadratic coefficients, q, g*, features.

M5 and Favorita already have expert OOF predictions from the Gate-v2 work and
are read back rather than retrained.  FreshRetailNet and UCI have none, so their
folds are generated here with the same frozen experts, inside the train region
and never touching a test window.

Everything downstream -- the alignment diagnostic, the 2x2 factorial, the
alpha fitting -- reads this one table, so the four datasets are treated
identically by construction rather than by discipline.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..expert_diversity.oof_experts import OUT as DIVERSITY_OUT
from ..external_validity_screen import cli, prereg, screen
from ..multi_benchmark import run as MB
from ..structure_gate import features as F, gate as G
from ..structure_gate.convex_oracle import FLAT_THRESHOLD
from ..structure_gate.oof import OUT as GATE_OUT, build_oof_split, fold_config, fold_cutoffs
from ..unified_temporal_27_v3.training import train_scale

OUT = screen.OUT.parent / "gate_v3"
DATASETS = ("m5", "favorita", "freshretailnet", "uci")
LEGACY = ("m5", "favorita")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pair() -> tuple[str, str]:
    spec = json.loads((DIVERSITY_OUT / "expert_set_spec.json").read_text())
    return tuple(spec["top_pair"])


def quadratic_from_arrays(pa, pb, yn, mask, index) -> pd.DataFrame:
    """Per-origin coefficients of sse((1-g)A + gB) as a quadratic in g."""
    d, r = pb - pa, yn - pa
    keep = mask.sum(axis=1) > 0
    out = pd.DataFrame({
        "series_index": index, "n": mask.sum(axis=1),
        "num": ((r * d) * mask).sum(axis=1), "den": ((d * d) * mask).sum(axis=1),
        "sse_a": (((yn - pa) ** 2) * mask).sum(axis=1),
        "sse_b": (((yn - pb) ** 2) * mask).sum(axis=1)})
    return out[keep].reset_index(drop=True)


def add_targets(table: pd.DataFrame) -> pd.DataFrame:
    """g* from the closed form, q from the frozen Gate-v2 regret definition."""
    table = table.copy()
    table["mse_a"] = table["sse_a"] / table["n"]
    table["mse_b"] = table["sse_b"] / table["n"]
    table["regret"] = table["mse_a"] - table["mse_b"]
    flat = table["den"].to_numpy() <= FLAT_THRESHOLD
    raw = np.divide(table["num"].to_numpy(),
                    np.where(flat, 1.0, table["den"].to_numpy()))
    table["g_star"] = np.where(flat, 0.5, np.clip(raw, 0.0, 1.0))
    table["flat"] = flat
    table["mse_convex"] = np.where(
        flat, table["mse_a"],
        (table["sse_a"] - 2 * table["g_star"] * table["num"]
         + table["g_star"] ** 2 * table["den"]) / table["n"])
    table["mse_hard"] = np.minimum(table["sse_a"], table["sse_b"]) / table["n"]
    return table


def legacy_table(name: str, a: str, b: str) -> pd.DataFrame:
    """M5/Favorita: reuse the stored expert OOF predictions, no retraining."""
    frame = pd.read_parquet(DIVERSITY_OUT / f"oof_experts_{name}.parquet")
    frame = frame[frame["target_mask"] > 0].copy()
    frame["series_id"] = frame["series_id"].astype(str)
    data = screen.load_dataset(name)
    cfg = screen.config_for(name)
    scale = pd.Series(train_scale({"y": data["y"], "z": data["z"]}, cfg),
                      index=pd.Index(data["series_id"]).astype(str))
    s = np.maximum(scale.loc[frame["series_id"]].to_numpy(np.float64), 1e-9)
    frame["yn"] = frame["y_observed"].to_numpy(np.float64) / s
    frame["na"] = frame[a].to_numpy(np.float64) / s
    frame["nb"] = frame[b].to_numpy(np.float64) / s
    keys = ["fold", "series_id", "origin"]
    grouped = frame.groupby(keys)
    table = pd.DataFrame({
        "n": grouped.size(),
        "num": grouped.apply(lambda d: float(((d["yn"] - d["na"]) * (d["nb"] - d["na"])).sum()),
                             include_groups=False),
        "den": grouped.apply(lambda d: float(((d["nb"] - d["na"]) ** 2).sum()),
                             include_groups=False),
        "sse_a": grouped.apply(lambda d: float(((d["yn"] - d["na"]) ** 2).sum()),
                               include_groups=False),
        "sse_b": grouped.apply(lambda d: float(((d["yn"] - d["nb"]) ** 2).sum()),
                               include_groups=False),
        "xs_a_mean": grouped["na"].mean(), "xs_b_mean": grouped["nb"].mean(),
        "xs_abs_disagreement": grouped.apply(
            lambda d: float((d["nb"] - d["na"]).abs().mean()), include_groups=False),
        "xs_max_disagreement": grouped.apply(
            lambda d: float((d["nb"] - d["na"]).abs().max()), include_groups=False),
        "xs_signed_disagreement": grouped.apply(
            lambda d: float((d["nb"] - d["na"]).mean()), include_groups=False),
    }).reset_index()
    structure = F.structure_table(data, table[["series_id", "origin"]])
    table = pd.concat([table, structure.drop(columns=["series_id", "origin"])], axis=1)
    return F.add_missing_flags(add_targets(table), F.STRUCTURE_COLUMNS)


def external_table(name: str, a: str, b: str, device) -> pd.DataFrame:
    """FreshRetailNet/UCI: generate OOF folds inside train with the frozen experts."""
    data = MB.load_grid(name)
    cfg = MB.config_for(data)
    n_series = data["y"].shape[0]
    pieces = []
    for k, fold_cfg in enumerate(MB.fold_configs(cfg)):
        split = MB.build_split(data, fold_cfg, MB.STRIDE)
        preds = {e: MB.expert_predictions(MB.ID_TO_KEY.get(e, e), fold_cfg, split, device, data)
                 for e in (a, b)}
        scale = split.scale
        axis = np.repeat(np.arange(n_series), split.test.n_origins)
        s = np.maximum(scale[axis].astype(np.float64), 1e-9)[:, None]
        mask = split.test.target_mask.astype(np.float64)
        yn = split.test.target.astype(np.float64) / s
        pa, pb = preds[a].astype(np.float64) / s, preds[b].astype(np.float64) / s
        table = quadratic_from_arrays(pa, pb, yn, mask, axis)
        keep = mask.sum(axis=1) > 0
        frame = MB.gate_frame(data, fold_cfg, split, scale, n_series)[keep].reset_index(drop=True)
        state = MB.pair_state(preds, a, b, np.maximum(scale[axis], 1e-9))[keep].reset_index(drop=True)
        block = pd.concat([table, frame.drop(columns=["series_id", "origin"]), state], axis=1)
        block["series_id"] = frame["series_id"].to_numpy()
        block["origin"] = frame["origin"].to_numpy()
        block["fold"] = k
        pieces.append(block)
        print(f"   [{name}] fold {k}: {len(block):,} origins", flush=True)
    table = pd.concat(pieces, ignore_index=True)
    return F.add_missing_flags(add_targets(table), F.STRUCTURE_COLUMNS)


def cmd_build(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = _pair()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {"built_at_utc": _utc(), "expert_pair": [a, b],
                "test_used": False, "existing_test_scored": False,
                "reused_without_retraining": list(LEGACY), "datasets": {}}
    for name in args.datasets:
        print(f"assembling {name} ...", flush=True)
        table = (legacy_table(name, a, b) if name in LEGACY
                 else external_table(name, a, b, device))
        table.to_parquet(OUT / f"oof_{name}.parquet", index=False)
        manifest["datasets"][name] = {
            "n_origins": int(len(table)), "n_folds": int(table["fold"].nunique()),
            "n_series": int(table["series_id"].nunique()) if "series_id" in table
                        else int(table["series_index"].nunique()),
            "flat_share": float(table["flat"].mean()),
            "source": "reused Gate-v2 OOF" if name in LEGACY else "generated in train region"}
        print(f"  {name}: {len(table):,} origins, {table['fold'].nunique()} folds", flush=True)
    manifest["git_commit"] = cli._git_commit()
    (OUT / "gate_v3_audit.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({n: b["n_origins"] for n, b in manifest["datasets"].items()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("Gate-v3 OOF assembly")
    sub = parser.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("--datasets", nargs="*", default=list(DATASETS))
    b.set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
