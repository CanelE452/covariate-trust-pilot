"""Track V, stage 27: validation stability audit.

Auxiliary only. A model that looked best on one validation window may not be
best in the next period; this measures how much that costs. Track V never
promotes a method and never changes a G, X or F verdict.

The adaptive rolling window rule comes from the official eliselyhan/ARW code,
imported read-only from the vendor clone.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import psutil
import torch

import paths
import attempts as A
import evalgeom
from contract import DATASETS, CLEAN_SEEDS, PRED_LEN, tqnet_config, dlinear_config
import data as D
import engine as E

STAGE = "track_v"
ARW_TIMEBOX_MINUTES = 45


def load_arw():
    """Import the official ARW module from the vendor clone, unmodified."""
    f = paths.VENDOR / "ARW" / "code-synthetic-data" / "ARW.py"
    if not f.exists():
        return None
    spec = importlib.util.spec_from_file_location("official_arw", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def kendall_tau(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    num = 0
    den = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = np.sign(a[i] - a[j])
            sb = np.sign(b[i] - b[j])
            if sa != 0 and sb != 0:
                num += sa * sb
                den += 1
    return float(num / den) if den else float("nan")


@torch.no_grad()
def origin_losses(model, cfg, dataset, flag):
    """Mean MSE at each of the eight origins of a split."""
    model.eval()
    out = []
    for o in evalgeom.origins(dataset, flag):
        x, y, c = D.window_batch(dataset, flag, o["window_starts"], E.DEVICE)
        p = E.forward(model, cfg, x, c)[:, -PRED_LEN:, :]
        out.append(float(((p - y[:, -PRED_LEN:, :]) ** 2).mean()))
    return np.array(out)


def collect_candidates(dsname):
    """Only models this study actually produced are eligible."""
    cands = {}
    clean = json.loads((paths.RESULTS / "clean_baselines.json").read_text())
    for mk, fn in [("TQNet", tqnet_config), ("DLinear", dlinear_config)]:
        k = f"{dsname}/{mk}/seed{CLEAN_SEEDS[0]}"
        if k in clean["runs"]:
            cands[f"clean_{mk}"] = (fn(dsname, CLEAN_SEEDS[0]),
                                    paths.ROOT / clean["runs"][k]["checkpoint_files"]["best"])
    gi = sorted((paths.RUNS / "track_g_intervention").glob("attempt_*/intervention.json"))
    if gi:
        d = json.loads(gi[-1].read_text())
        for key, r in d["runs"].items():
            if r["dataset"] == dsname and r["seed"] == CLEAN_SEEDS[0]:
                cands[f"TQNet_{r['arm']}"] = (tqnet_config(dsname, r["seed"]),
                                              paths.ROOT / r["checkpoint_file"])
    xm = sorted((paths.RUNS / "track_x_mitigations").glob("attempt_*/mitigations.json"))
    if xm:
        f = xm[-1].parent / f"{dsname}_dropout_s{CLEAN_SEEDS[0]}.pt"
        if f.exists():
            cands["TQNet_channel_dropout"] = (tqnet_config(dsname, CLEAN_SEEDS[0]), f)
    return cands


def main():
    done = A.completed(STAGE)
    if done:
        print(f"[resume] {STAGE} already completed at {done}")
        return
    att = A.new_attempt(STAGE)
    mon = subprocess.Popen([sys.executable, str(paths.EXP / "common" / "resmon.py"),
                            str(psutil.Process().pid), str(att)])
    log = lambda s: print(s, flush=True)
    t_arw0 = time.time()
    arw = load_arw()
    arw_status = "OFFICIAL_ARW" if arw is not None else "ARW_NOT_AVAILABLE"
    out = {"attempt": str(att), "arw_status": arw_status,
           "arw_source": "eliselyhan/ARW code-synthetic-data/ARW.py, imported unmodified",
           "role": "AUXILIARY_AUDIT_NEVER_PROMOTES_A_METHOD", "datasets": {}}
    t0 = time.time()
    try:
        for dsname in DATASETS:
            cands = collect_candidates(dsname)
            if len(cands) < 2:
                out["datasets"][dsname] = {"status": "BLOCKED_TOO_FEW_CANDIDATES",
                                           "n_candidates": len(cands)}
                continue
            names = sorted(cands)
            valL, tstL = {}, {}
            for n in names:
                cfg, f = cands[n]
                m = E.build_model(cfg).to(E.DEVICE)
                m.load_state_dict(torch.load(f, map_location=E.DEVICE))
                valL[n] = origin_losses(m, cfg, dsname, "val")
                tstL[n] = origin_losses(m, cfg, dsname, "test")
                del m
                torch.cuda.empty_cache()
            V = np.stack([valL[n] for n in names])     # [M, 8]
            T = np.stack([tstL[n] for n in names])
            test_mean = T.mean(axis=1)
            best_test = int(np.argmin(test_mean))
            rules = {}
            for label, sel in [("latest_1", lambda v: v[:, -1:]),
                               ("last_2", lambda v: v[:, -2:]),
                               ("last_4", lambda v: v[:, -4:]),
                               ("full_8", lambda v: v)]:
                pick = int(np.argmin(sel(V).mean(axis=1)))
                rules[label] = {"selected": names[pick],
                                "test_regret": float((test_mean[pick] - test_mean[best_test])
                                                     / test_mean[best_test])}
            if arw is not None and (time.time() - t_arw0) / 60 < ARW_TIMEBOX_MINUTES:
                # One period per validation origin; ARWME picks the window size.
                B_arr = np.ones(V.shape[1], dtype=int)
                khat = []
                for i, n in enumerate(names):
                    k, _ = arw.ARWME(U=V[i], B_arr=B_arr, delta=0.1, M=float(V.max()))
                    khat.append(int(k))
                k_use = int(np.median(khat))
                pick = int(np.argmin(V[:, -k_use:].mean(axis=1)))
                rules["arw"] = {"selected": names[pick], "window": k_use,
                                "per_model_window": khat,
                                "test_regret": float((test_mean[pick] - test_mean[best_test])
                                                     / test_mean[best_test])}
            # Rank agreement and flip rate, per origin
            taus = [kendall_tau(V[:, o], T[:, o]) for o in range(V.shape[1])]
            flips = []
            for o in range(V.shape[1] - 1):
                r1 = np.argsort(V[:, o])
                r2 = np.argsort(V[:, o + 1])
                flips.append(float(np.mean(r1 != r2)))
            worst = {label: float(np.max([(T[names.index(r["selected"]), o]
                                           - T[best_test, o]) / T[best_test, o]
                                          for o in range(T.shape[1])]))
                     for label, r in rules.items()}
            out["datasets"][dsname] = {
                "candidates": names, "n_candidates": len(names),
                "val_origin_loss": {n: valL[n].tolist() for n in names},
                "test_origin_loss": {n: tstL[n].tolist() for n in names},
                "best_by_test": names[best_test],
                "selection_rules": rules,
                "kendall_tau_val_vs_test_per_origin": taus,
                "kendall_tau_mean": float(np.nanmean(taus)),
                "rank_flip_rate_between_adjacent_val_origins": float(np.mean(flips)),
                "selection_consistency": float(len({r["selected"] for r in rules.values()}) == 1),
                "worst_period_regret": worst,
            }
            log(f"[V] {dsname} " + " ".join(
                f"{k}={v['test_regret']:+.4f}" for k, v in rules.items()))
        out["wall_s"] = round(time.time() - t0, 1)
        A.write_json(att / "validation_stability.json", out)
        A.finish(att, {"stage": STAGE, "ok": True})
        log(f"DONE {out['wall_s']}s")
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
