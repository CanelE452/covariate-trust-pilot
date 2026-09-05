"""Stage 8/9: train the two clean models on both datasets, then run the sanity gate.

Model A: TQNet, the shared cross-channel forecaster (official config).
Model B: DLinear with individual heads, the channel-independent control.

Per-epoch snapshots are kept so Track G can pick its early / middle checkpoints
by a fixed fraction of the realised schedule rather than by inspecting results.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import psutil
import torch

import paths
import attempts as A
from contract import DATASETS, CLEAN_SEEDS, tqnet_config, dlinear_config
import data as D
import engine as E

STAGE = "clean_baselines"
SHARED_WEAK_RATIO = 1.30   # TQNet worse than DLinear by >= 30% -> SHARED_BASELINE_WEAK


def sanity(dataset: str, cfg, state, hist) -> dict:
    """Blocking conditions, evaluated before any track uses the checkpoint."""
    m = E.test_metrics(cfg, dataset, state, per_channel=True)
    model = E.build_model(cfg).to(E.DEVICE)
    model.load_state_dict(state)
    model.eval()
    x, y, c = D.window_batch(dataset, "test", list(range(0, 512, 64)), E.DEVICE)
    with torch.no_grad():
        out = E.forward(model, cfg, x, c)[:, -cfg.pred_len:, :]
    pred = out.cpu().numpy()
    flags = []
    if not np.isfinite(m["mse"]):
        flags.append("NAN_METRIC")
    if pred.shape != (x.shape[0], cfg.pred_len, cfg.enc_in):
        flags.append("WRONG_OUTPUT_SHAPE")
    if float(pred.std(axis=1).mean()) < 1e-6:
        flags.append("CONSTANT_PREDICTION")
    losses = [h["train_loss"] for h in hist]
    if any((not np.isfinite(v)) or v > 1e4 for v in losses):
        flags.append("EXPLODING_LOSS")
    return {"metrics": m, "flags": flags,
            "pred_shape": list(pred.shape),
            "pred_temporal_std": float(pred.std(axis=1).mean())}


def main():
    done = A.completed(STAGE)
    if done:
        print(f"[resume] {STAGE} already completed at {done}")
        return

    att = A.new_attempt(STAGE)
    mon = subprocess.Popen([sys.executable, str(paths.EXP / "common" / "resmon.py"),
                            str(psutil.Process().pid), str(att)])
    out = {"attempt": str(att), "runs": {}, "gate": {}}
    ck_dir = att / "checkpoints"
    ck_dir.mkdir(exist_ok=True)
    t_start = time.time()
    try:
        for dsname in DATASETS:
            for mk, fn in [("TQNet", tqnet_config), ("DLinear", dlinear_config)]:
                for seed in CLEAN_SEEDS:
                    cfg = fn(dsname, seed)
                    key = f"{dsname}/{mk}/seed{seed}"
                    print(f"[train] {key}", flush=True)
                    r = E.train_model(cfg, dsname, capture_all_epochs=True,
                                      log=lambda s: print(s, flush=True))
                    # Fixed rule, independent of any result: early = 25% and
                    # middle = 50% of the realised schedule; best = best val MSE.
                    n = r.epochs_run
                    sel = {"early": max(1, int(np.ceil(0.25 * n))),
                           "middle": max(1, int(np.ceil(0.50 * n))),
                           "best": r.best_epoch}
                    tag_files = {}
                    for tag, ep in sel.items():
                        st = r.checkpoints["best"] if tag == "best" else r.checkpoints[f"epoch{ep:03d}"]
                        f = ck_dir / f"{dsname}_{mk}_s{seed}_{tag}.pt"
                        torch.save(st, f)
                        tag_files[tag] = str(f.relative_to(paths.ROOT))
                    s = sanity(dsname, cfg, r.checkpoints["best"], r.history)
                    out["runs"][key] = {
                        "config": cfg.as_dict(),
                        "best_val_mse": r.best_val_mse, "best_epoch": r.best_epoch,
                        "epochs_run": r.epochs_run, "wall_s": round(r.wall_s, 1),
                        "history": r.history,
                        "checkpoint_epochs": sel, "checkpoint_files": tag_files,
                        "test_mse": s["metrics"]["mse"], "test_mae": s["metrics"]["mae"],
                        "test_mse_per_channel": s["metrics"]["mse_per_channel"],
                        "sanity_flags": s["flags"], "pred_shape": s["pred_shape"],
                        "pred_temporal_std": s["pred_temporal_std"],
                    }
                    print(f"  -> val {r.best_val_mse:.5f} test_mse {s['metrics']['mse']:.5f} "
                          f"test_mae {s['metrics']['mae']:.5f} flags {s['flags']}", flush=True)
                    A.write_json(att / "clean_baselines.json", out)
                    del r
                    torch.cuda.empty_cache()

        for dsname in DATASETS:
            tq = np.mean([out["runs"][f"{dsname}/TQNet/seed{s}"]["test_mse"] for s in CLEAN_SEEDS])
            dl = np.mean([out["runs"][f"{dsname}/DLinear/seed{s}"]["test_mse"] for s in CLEAN_SEEDS])
            flags = sorted({f for k, v in out["runs"].items() if k.startswith(dsname)
                            for f in v["sanity_flags"]})
            status = "BLOCKED" if flags else "OK"
            weak = bool(tq > dl * SHARED_WEAK_RATIO)
            out["gate"][dsname] = {
                "tqnet_test_mse": float(tq), "dlinear_test_mse": float(dl),
                "tqnet_over_dlinear": float(tq / dl),
                "status": status, "blocking_flags": flags,
                "shared_baseline_weak": weak,
                "note": ("SHARED_BASELINE_WEAK: TQNet is at least 30% worse than DLinear on clean "
                         "test MSE, so Track X may not reach METHOD_GO on this dataset"
                         if weak else None),
            }
            print(f"[gate] {dsname} TQNet {tq:.5f} DLinear {dl:.5f} ratio {tq/dl:.3f} "
                  f"{status} weak={weak}", flush=True)

        out["total_wall_s"] = round(time.time() - t_start, 1)
        peak = att / "resource_peak.json"
        out["resource_peak"] = json.loads(peak.read_text()) if peak.exists() else None
        A.write_json(att / "clean_baselines.json", out)
        A.write_json(paths.RESULTS / "clean_baselines.json",
                     {k: v for k, v in out.items() if k != "runs"} |
                     {"runs": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                               for k, v in out["runs"].items()}})
        A.finish(att, {"stage": STAGE, "ok": True,
                       "blocked": [d for d, g in out["gate"].items() if g["status"] == "BLOCKED"]})
        print("DONE", out["total_wall_s"], "s", flush=True)
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
