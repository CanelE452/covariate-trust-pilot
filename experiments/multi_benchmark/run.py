"""Score every frozen method on the admitted external datasets, in one pass.

Both datasets are scored inside a single command so no result from one can
inform the handling of the other.  The spec hash is checked first; if the frozen
configuration is not present, nothing runs.

The method is the one the earlier work selected, read from its artifact, and it
is refit here rather than transferred: weights come from each dataset's own
train window, the definition does not change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..expert_diversity.oof_experts import classical_predictions
from ..external_validity_screen import classical_benchmark as CB
from ..external_validity_screen import cli, prereg, rule_replication as rr, screen
from ..om_factorization_killtest import models as km_models
from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from ..structure_gate import features as F, gate as G, gate_v2 as V2
from ..structure_gate.convex_oracle import FLAT_THRESHOLD
from ..unified_temporal_27_v3.config import ExperimentConfig
from ..unified_temporal_27_v3.training import make_windows, train_scale
from .audit import HORIZON, LOOKBACK, MIN_SPAN, OUT, STRIDE, TEST_ORIGINS
from .build import PROCESSED

DRAWS = prereg.BOOTSTRAP["draws"]
SEED = prereg.BOOTSTRAP["seed"]
MODEL_SEED = prereg.MODELS["canonical_model_seed"]
N_OOF_FOLDS = 3


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_grid(name: str) -> dict:
    """The built grid, in load_dataset's dict shape plus the masks it does not carry."""
    blob = np.load(PROCESSED / f"{name}_grid.npz", allow_pickle=False)
    y = blob["y"].astype(np.float32)
    observed = blob["observed_mask"]
    return {"name": name, "y": y, "z": (y > 0).astype(np.float32),
            "series_id": blob["series_id"].astype(str),
            "observed_mask": observed,
            "available_from": np.zeros(len(y), dtype=int),
            "first_positive": np.array([int(np.argmax(r > 0)) if (r > 0).any() else len(r)
                                        for r in y])}


def config_for(data: dict) -> ExperimentConfig:
    length = data["y"].shape[1]
    test_block = STRIDE * (TEST_ORIGINS - 1) + HORIZON
    val_end = length - test_block
    return ExperimentConfig(length=length, lookback=LOOKBACK, horizon=HORIZON,
                            period=HORIZON, train_end=val_end - HORIZON, val_end=val_end)


def build_split(data: dict, cfg, stride: int):
    """screen.build_split, with the observed mask standing in for availability."""
    arrays = {"y": data["y"], "z": data["z"]}
    scale = train_scale(arrays, cfg)
    train_o = km_train.dense_origins(0, cfg.train_end, cfg.horizon,
                                     cfg.lookback)[::prereg.SPLITS["train_origin_stride"]]
    val_o = km_train.dense_origins(cfg.train_end, cfg.val_end, cfg.horizon, cfg.lookback)
    test_o = km_train.stride_origins(cfg.val_end, cfg.length, cfg.horizon, stride)
    windows = []
    for origins, lo, hi in ((train_o, 0, cfg.train_end), (val_o, cfg.train_end, cfg.val_end),
                            (test_o, cfg.val_end, cfg.length)):
        block = make_windows(arrays, origins, lo, hi, cfg, scale)
        keep = data["observed_mask"][:, None, :][:, :, :0]  # placeholder, replaced below
        step = np.arange(cfg.horizon)[None, :]
        absolute = np.asarray(block.origins)[:, None] + step
        allowed = data["observed_mask"][:, absolute]          # (N, W, H)
        mask = block.target_mask.reshape(len(data["y"]), len(block.origins), cfg.horizon)
        windows.append(replace(block, target_mask=(mask & allowed).reshape(
            len(data["y"]) * len(block.origins), cfg.horizon)))
    y_train = data["y"][:, :cfg.train_end]
    positives = y_train[data["z"][:, :cfg.train_end] > 0]
    variance = float(positives.var(ddof=1)) if positives.size > 1 else 1.0
    return km_train.Split(windows[0], windows[1], windows[2], scale, max(variance, 1e-6))


def fold_configs(cfg) -> list:
    block = cfg.horizon * (1 + TEST_ORIGINS)
    return [replace(cfg, train_end=cfg.train_end - block * (N_OOF_FOLDS - k),
                    val_end=cfg.train_end - block * (N_OOF_FOLDS - k) + cfg.horizon,
                    length=cfg.train_end - block * (N_OOF_FOLDS - k) + block)
            for k in range(N_OOF_FOLDS)]


def expert_predictions(key: str, cfg, split, device, data) -> np.ndarray:
    """One expert's mean prediction on the split's test windows."""
    if key in km_models.BUILDERS:
        fit = rr.train_frozen(key, cfg, MODEL_SEED, device, split)
        return fit["predictions"]["mean_prediction"]
    prediction, _ = classical_predictions(key, split, cfg)
    return prediction


def gate_frame(data, cfg, split, scale, n_series) -> pd.DataFrame:
    """As-of-origin structure features for the split's test origins."""
    windows = split.test
    axis = np.repeat(np.arange(n_series), windows.n_origins)
    origins = np.tile(np.asarray(windows.origins), n_series)
    pairs = pd.DataFrame({"series_id": data["series_id"][axis], "origin": origins})
    structure = F.structure_table(data, pairs)
    frame = pd.concat([pairs.reset_index(drop=True),
                       structure.drop(columns=["series_id", "origin"])], axis=1)
    return F.add_missing_flags(frame, F.STRUCTURE_COLUMNS)


def pair_state(predictions: dict, a: str, b: str, scale_row: np.ndarray) -> pd.DataFrame:
    na = predictions[a].astype(np.float64) / scale_row[:, None]
    nb = predictions[b].astype(np.float64) / scale_row[:, None]
    gap = nb - na
    return pd.DataFrame({"xs_a_mean": na.mean(1), "xs_b_mean": nb.mean(1),
                         "xs_abs_disagreement": np.abs(gap).mean(1),
                         "xs_max_disagreement": np.abs(gap).max(1),
                         "xs_signed_disagreement": gap.mean(1)})


def quadratic(split, predictions, a, b, scale, n_series) -> pd.DataFrame:
    """Per-origin coefficients of the mixture's squared error, on the train scale."""
    windows = split.test
    mask = windows.target_mask.astype(np.float64)
    target = windows.target.astype(np.float64)
    axis = np.repeat(np.arange(n_series), windows.n_origins)
    s = np.maximum(scale[axis].astype(np.float64), 1e-9)[:, None]
    pa = predictions[a].astype(np.float64) / s
    pb = predictions[b].astype(np.float64) / s
    yn = target / s
    d, r = pb - pa, yn - pa
    keep = mask.sum(axis=1) > 0
    out = pd.DataFrame({"n": mask.sum(axis=1),
                        "num": ((r * d) * mask).sum(axis=1),
                        "den": ((d * d) * mask).sum(axis=1),
                        "sse_a": (((yn - pa) ** 2) * mask).sum(axis=1),
                        "sse_b": (((yn - pb) ** 2) * mask).sum(axis=1)})
    return out[keep].reset_index(drop=True)


def convex(out: pd.DataFrame):
    flat = out["den"].to_numpy() <= FLAT_THRESHOLD
    g = np.where(flat, 0.5,
                 np.clip(np.divide(out["num"].to_numpy(),
                                   np.where(flat, 1.0, out["den"].to_numpy())), 0, 1))
    sse = np.where(flat, out["sse_a"],
                   out["sse_a"] - 2 * g * out["num"] + g ** 2 * out["den"])
    n = out["n"].to_numpy()
    return g, sse / n, np.minimum(out["sse_a"], out["sse_b"]).to_numpy() / n


def mix(out: pd.DataFrame, g) -> np.ndarray:
    return ((out["sse_a"].to_numpy() - 2 * g * out["num"].to_numpy()
             + g ** 2 * out["den"].to_numpy()) / out["n"].to_numpy())


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


ID_TO_KEY = {"dlinear_point": prereg.MODELS["point"],
             "dlinear_hurdle": prereg.MODELS["hurdle"],
             "dlinear_point_plain": "M0_point_mse",
             "dlinear_hurdle_ztnb": "M2_hurdle_ztnb"}


def score_dataset(name: str, spec: dict, device):
    proposed = spec["proposed_method"]
    a, b = proposed["expert_pair"]
    ref_a, ref_b = proposed["reference_pair"]
    classical = list(spec["expert_pool"]["classical"])
    wanted = sorted(set([a, b, ref_a, ref_b] + list(ID_TO_KEY) + classical))

    data = load_grid(name)
    cfg = config_for(data)
    n_series = data["y"].shape[0]
    print("[%s] %d series, length %d, train_end %d, val_end %d"
          % (name, n_series, cfg.length, cfg.train_end, cfg.val_end), flush=True)

    blocks = []
    for k, fold_cfg in enumerate(fold_configs(cfg)):
        split = build_split(data, fold_cfg, STRIDE)
        preds = {e: expert_predictions(ID_TO_KEY.get(e, e), fold_cfg, split, device, data)
                 for e in (a, b)}
        scale = split.scale
        axis = np.repeat(np.arange(n_series), split.test.n_origins)
        keep = split.test.target_mask.sum(axis=1) > 0
        table = quadratic(split, preds, a, b, scale, n_series)
        frame = gate_frame(data, fold_cfg, split, scale, n_series)[keep].reset_index(drop=True)
        state = pair_state(preds, a, b, np.maximum(scale[axis], 1e-9))[keep].reset_index(drop=True)
        block = pd.concat([table, frame.drop(columns=["series_id", "origin"]), state], axis=1)
        block["fold"] = k
        block["mse_a"] = block["sse_a"] / block["n"]
        block["mse_b"] = block["sse_b"] / block["n"]
        block["regret"] = block["mse_a"] - block["mse_b"]
        blocks.append(block)
        print("   OOF fold %d: %d origins" % (k, len(block)), flush=True)
    oof = pd.concat(blocks, ignore_index=True)

    columns = F.feature_columns(proposed["gate"]["variant"], list(oof.columns))
    medians = F.fit_imputer(oof, F.STRUCTURE_COLUMNS)
    filled = F.apply_imputer(oof, medians)
    x = filled[columns].to_numpy(np.float64)
    centre = x.mean(0)
    spread = np.where(x.std(0) > 0, x.std(0), 1.0)
    gate_model = V2.fit(proposed["gate"]["loss"], ((x - centre) / spread).astype(np.float32),
                        oof, proposed["gate"]["architecture"], device)
    alphas = np.arange(0, 1.0001, 0.05)
    alpha = float(min(alphas, key=lambda w: mix(oof, np.full(len(oof), w)).mean()))
    print("[%s] gate fitted on %d OOF origins, global alpha %.2f" % (name, len(oof), alpha),
          flush=True)

    split = build_split(data, cfg, STRIDE)
    scale = split.scale
    preds = {}
    for expert in wanted:
        preds[expert] = expert_predictions(ID_TO_KEY.get(expert, expert), cfg, split,
                                           device, data)
        print("   scored %s" % expert, flush=True)
    axis = np.repeat(np.arange(n_series), split.test.n_origins)
    keep = split.test.target_mask.sum(axis=1) > 0
    scale_row = np.maximum(scale[axis], 1e-9)

    frame = gate_frame(data, cfg, split, scale, n_series)[keep].reset_index(drop=True)
    state = pair_state(preds, a, b, scale_row)[keep].reset_index(drop=True)
    gate_input = F.apply_imputer(pd.concat([frame, state], axis=1), medians)
    xt = ((gate_input[columns].to_numpy(np.float64) - centre) / spread).astype(np.float32)
    g = G.gate_weights(gate_model, xt, device).astype(np.float64)

    table = quadratic(split, preds, a, b, scale, n_series)
    ref_table = quadratic(split, preds, ref_a, ref_b, scale, n_series)
    _, mse_convex, mse_hard = convex(table)
    _, ref_convex, _ = convex(ref_table)

    mask = split.test.target_mask.astype(np.float64)
    target = split.test.target.astype(np.float64)
    methods = {}
    for expert in wanted:
        e = ((preds[expert].astype(np.float64) - target) / scale_row[:, None]) ** 2
        methods[expert] = ((e * mask).sum(1) / np.maximum(mask.sum(1), 1))[keep]
    methods["proposed_gate"] = mix(table, g)
    methods["proposed_equal"] = mix(table, np.full(len(table), 0.5))
    methods["proposed_alpha"] = mix(table, np.full(len(table), alpha))
    ref_alpha = float(min(alphas,
                          key=lambda w: mix(ref_table, np.full(len(ref_table), w)).mean()))
    methods["reference_equal"] = mix(ref_table, np.full(len(ref_table), 0.5))
    methods["reference_alpha"] = mix(ref_table, np.full(len(ref_table), ref_alpha))

    index = np.repeat(np.arange(n_series), split.test.n_origins)[keep]
    per_series = pd.DataFrame(methods)
    per_series["series_index"] = index
    per_series = per_series.groupby("series_index").mean()
    per_series["oracle_convex"] = pd.Series(mse_convex, index=index).groupby(level=0).mean()
    per_series["oracle_hard"] = pd.Series(mse_hard, index=index).groupby(level=0).mean()
    per_series["reference_oracle_convex"] = pd.Series(ref_convex, index=index).groupby(level=0).mean()

    means = {c: float(per_series[c].mean()) for c in per_series.columns}
    static = ["proposed_equal", "proposed_alpha", "reference_equal", "reference_alpha"]
    singles = [m for m in ID_TO_KEY]
    best_static = min(static + singles, key=lambda m: means[m])
    best_classical = min(classical, key=lambda m: means[m])
    best_single = min(singles, key=lambda m: means[m])
    targets = {best_static, best_classical, best_single, "proposed_equal",
               "proposed_alpha", "reference_alpha"}
    comparisons = {"proposed_vs_" + m: boot(per_series, "proposed_gate", m) for m in targets}
    denominator = means[best_static] - means["oracle_convex"]
    recovery = ((means[best_static] - means["proposed_gate"]) / denominator
                if denominator > 0 else "CONVEX_GAP_UNAVAILABLE")

    m = split.test.target_mask.astype(bool)
    resid = {}
    for label, (p, q) in (("proposed", (a, b)), ("reference", (ref_a, ref_b))):
        ep = ((preds[p].astype(np.float64) - target) / scale_row[:, None])[m]
        eq = ((preds[q].astype(np.float64) - target) / scale_row[:, None])[m]
        resid[label] = float(np.corrcoef(ep, eq)[0, 1])

    return {
        "n_series": int(per_series.shape[0]), "n_origins": int(len(table)),
        "config": {"length": cfg.length, "train_end": cfg.train_end, "val_end": cfg.val_end,
                   "lookback": cfg.lookback, "horizon": cfg.horizon,
                   "test_origins": split.test.origins.tolist()},
        "global_alpha": alpha, "reference_alpha": ref_alpha,
        "means": means, "best_static": best_static, "best_classical": best_classical,
        "best_single_neural": best_single,
        "paired_bootstrap": comparisons, "convex_oracle_recovery": recovery,
        "diversity": {"proposed_residual_correlation": resid["proposed"],
                      "reference_residual_correlation": resid["reference"],
                      "proposed_convex_ceiling": means["oracle_convex"],
                      "reference_convex_ceiling": means["reference_oracle_convex"]},
        "gate_weight": {"mean": float(g.mean()), "sd": float(g.std()),
                        "extreme_share": float(((g < 0.05) | (g > 0.95)).mean())},
        "per_series_path": "external_per_series_%s.csv" % name,
    }, per_series


def cmd_run(args) -> None:
    spec = json.loads((OUT / "benchmark_spec_v1.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = _utc()
    if not (spec["frozen_at_utc"] < started):
        raise screen.ScreenFailure("spec was not frozen before scoring")
    report = {"analysis": "external benchmark, all admitted datasets in one locked pass",
              "spec_sha256": spec["spec_sha256"],
              "spec_frozen_at_utc": spec["frozen_at_utc"],
              "first_external_test_scoring_utc": started,
              "pooled_across_datasets": False, "datasets": {}}
    for name in args.datasets:
        block, per_series = score_dataset(name, spec, device)
        per_series.to_csv(OUT / block["per_series_path"])
        report["datasets"][name] = block
        print("[%s] best_static=%s proposed=%.6f static=%.6f recovery=%s"
              % (name, block["best_static"], block["means"]["proposed_gate"],
                 block["means"][block["best_static"]], block["convex_oracle_recovery"]),
              flush=True)
    report["finished_at_utc"] = _utc()
    report["git_commit"] = cli._git_commit()
    (OUT / "external_benchmark.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({n: {"best_static": b["best_static"],
                          "vs_best_static": b["paired_bootstrap"][
                              "proposed_vs_" + b["best_static"]]}
                      for n, b in report["datasets"].items()}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser("external benchmark scoring")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="*", default=["freshretailnet", "uci"])
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
