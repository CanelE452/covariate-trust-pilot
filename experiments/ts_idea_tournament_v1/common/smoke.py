"""Runtime smoke: 200 train windows / 200 eval windows per model+dataset.

Measures 1-epoch wall, inference wall, GPU peak, RSS, checkpoint size, and
projects the full-study GPU budget so the runtime tier can be frozen BEFORE any
scientific result is seen.
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
from contract import DATASETS, tqnet_config, dlinear_config, CLEAN_SEEDS
import data as D
import engine as E

paths.ensure_dirs()
STAGE = "smoke"


def main():
    done = A.completed(STAGE)
    if done:
        print(f"[resume] smoke already completed at {done}")
        print(json.dumps(json.loads((done / 'smoke.json').read_text()), indent=1)[:2000])
        return

    att = A.new_attempt(STAGE)
    mon = subprocess.Popen([sys.executable, str(paths.EXP / "common" / "resmon.py"),
                            str(psutil.Process().pid), str(att)])
    out = {"attempt": str(att), "env": A.environment(), "models": {}}
    try:
        for dsname in DATASETS:
            for mk, mkfn in [("TQNet", tqnet_config), ("DLinear", dlinear_config)]:
                cfg = mkfn(dsname, CLEAN_SEEDS[0])
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                t0 = time.time()
                r = E.train_model(cfg, dsname, max_train_windows=200,
                                  max_eval_windows=200, epochs=1)
                ep_wall = time.time() - t0
                t1 = time.time()
                m = E.test_metrics(cfg, dsname, r.checkpoints["best"],
                                   per_channel=False, max_eval_windows=200)
                inf_wall = time.time() - t1
                ck = att / f"ck_{dsname}_{mk}.pt"
                torch.save(r.checkpoints["best"], ck)
                n_params = sum(v.numel() for v in r.checkpoints["best"].values())
                gpu_peak = (torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available() else None)
                n_train = len(D.get_dataset(dsname, "train"))
                key = f"{dsname}/{mk}"
                out["models"][key] = {
                    "epoch_wall_200w_s": round(ep_wall, 2),
                    "infer_wall_200w_s": round(inf_wall, 2),
                    "gpu_peak_gb": round(gpu_peak, 3) if gpu_peak else None,
                    "rss_gb": round(psutil.Process().memory_info().rss / 1e9, 3),
                    "ckpt_bytes": ck.stat().st_size,
                    "n_params": int(n_params),
                    "smoke_val_mse": round(r.best_val_mse, 5),
                    "smoke_test_mse": round(m["mse"], 5),
                    "n_train_windows_full": n_train,
                    "batch_size": cfg.batch_size,
                }
                print(key, json.dumps(out["models"][key]))
                del r
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Projection: scale the measured 200-window epoch to the full train split.
        proj = {}
        for key, v in out["models"].items():
            per_epoch_full = v["epoch_wall_200w_s"] * v["n_train_windows_full"] / 200.0
            proj[key] = {"per_epoch_full_s": round(per_epoch_full, 1),
                         "full_fit_30ep_h": round(per_epoch_full * 30 / 3600, 2)}
        out["projection_per_fit"] = proj
        out["peak"] = json.loads((att / "resource_peak.json").read_text()) \
            if (att / "resource_peak.json").exists() else None
        A.write_json(att / "smoke.json", out)
        A.finish(att, {"stage": STAGE, "ok": True})
        print("\n=== PROJECTION PER FIT ===")
        print(json.dumps(proj, indent=1))
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
