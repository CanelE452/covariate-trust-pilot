"""Track G, stage 17: ERM vs PCGrad vs norm-balanced ERM vs probe-gated PCGrad.

Every arm shares the model, optimiser base learning rate, training windows,
epoch budget, early-stopping rule and seed. Only the shared-parameter update
rule differs. The probe-gated arm's extra forward/backward cost is reported
separately and never mixed into the accuracy comparison.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import psutil
import torch

import paths
import attempts as A
from contract import DATASETS, CLEAN_SEEDS, tqnet_config
import data as D
import engine as E
import rules as RU
from run_diagnostic import tasks_for

STAGE = "track_g_intervention"
ARMS = ["erm", "pcgrad", "norm_balanced", "probe_gated"]


def macro_mse(per_channel: list, tasks: list) -> float:
    """Macro MSE over the task variables: the mean of per-variable MSE."""
    return float(np.mean([per_channel[i] for i in tasks]))


def main():
    done = A.completed(STAGE)
    if done:
        print(f"[resume] {STAGE} already completed at {done}")
        return
    tier = json.loads((paths.RESULTS / "runtime_tier.json").read_text())
    n_seeds = tier["tier_settings"]["intervention_seeds"]
    seeds = CLEAN_SEEDS[:n_seeds]

    att = A.new_attempt(STAGE)
    mon = subprocess.Popen([sys.executable, str(paths.EXP / "common" / "resmon.py"),
                            str(psutil.Process().pid), str(att)])
    log = lambda s: print(s, flush=True)
    out = {"attempt": str(att), "arms": ARMS, "seeds": seeds, "runs": {}}
    ck = att / "checkpoints"
    ck.mkdir(exist_ok=True)
    t0 = time.time()
    try:
        for dsname in DATASETS:
            tasks = tasks_for(dsname)
            for arm in ARMS:
                for seed in seeds:
                    cfg = tqnet_config(dsname, seed)
                    key = f"{dsname}/{arm}/seed{seed}"
                    rule = RU.GradRule(arm, dsname, tasks, seed)
                    log(f"[train] {key}")
                    t1 = time.time()
                    r = E.train_model(cfg, dsname, grad_rule=rule,
                                      log=lambda s: print(s, flush=True))
                    wall = time.time() - t1
                    m = E.test_metrics(cfg, dsname, r.checkpoints["best"], per_channel=True)
                    f = ck / f"{dsname}_{arm}_s{seed}.pt"
                    torch.save(r.checkpoints["best"], f)
                    out["runs"][key] = {
                        "arm": arm, "dataset": dsname, "seed": seed, "tasks": tasks,
                        "best_val_mse": r.best_val_mse, "best_epoch": r.best_epoch,
                        "epochs_run": r.epochs_run, "wall_s": round(wall, 1),
                        "test_mse": m["mse"], "test_mae": m["mae"],
                        "test_mse_per_channel": m["mse_per_channel"],
                        "macro_mse": macro_mse(m["mse_per_channel"], tasks),
                        "extra_probe_forward": rule.extra_forward,
                        "extra_probe_backward": rule.extra_backward,
                        "checkpoint_file": str(f.relative_to(paths.ROOT)),
                        "history": r.history,
                    }
                    log(f"  -> macro_mse {out['runs'][key]['macro_mse']:.6f} "
                        f"test_mse {m['mse']:.6f} val {r.best_val_mse:.6f} "
                        f"wall {wall:.0f}s")
                    A.write_json(att / "intervention.json", out)
                    del r
                    torch.cuda.empty_cache()
        out["wall_s"] = round(time.time() - t0, 1)
        A.write_json(att / "intervention.json", out)
        A.finish(att, {"stage": STAGE, "ok": True})
        log(f"DONE {out['wall_s']}s")
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
