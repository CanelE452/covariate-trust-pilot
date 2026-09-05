"""Track F, stages 20-25: severity calibration, modified training set,
five existing selectors plus the coherence-aware intervention, retraining and
evaluation on both a clean and a shifted test set, and negative controls.
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
import torch.nn as nn

import paths
import attempts as A
from contract import DATASETS, CLEAN_SEEDS, SEQ_LEN, PRED_LEN, dlinear_config
import data as D
import engine as E
import evalgeom
import filters as SEL
import windows as W

STAGE = "track_f_selection"
METHODS = ["no_filter", "random_removal", "high_loss_removal", "rho_loss",
           "adarho", "coherence_aware"]
SHIFT_TEST_SEED = 2026090641          # independent of the training shift seed
REF_HOLDOUT_FRACTION = 0.25


# --------------------------------------------------------------------------- #
# Array-level training (the modified training set is an explicit sample list)
# --------------------------------------------------------------------------- #

def train_on_arrays(cfg, X, Y, Xv, Yv, seed, epochs=None, log=None):
    E.set_seed(seed)
    model = E.build_model(cfg).to(E.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    crit = nn.MSELoss()
    n_ep = epochs or cfg.train_epochs
    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()
    rng = np.random.RandomState(seed)
    for ep in range(1, n_ep + 1):
        model.train()
        order = rng.permutation(len(Xt))
        for i in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            x = Xt[idx].to(E.DEVICE)
            y = Yt[idx].to(E.DEVICE)
            cyc = torch.zeros(len(x), dtype=torch.int32, device=E.DEVICE)
            opt.zero_grad()
            out = E.forward(model, cfg, x, cyc)[:, -PRED_LEN:, :]
            loss = crit(out, y)
            loss.backward()
            opt.step()
        v = float(np.mean(SEL.window_losses(model, cfg, Xv, Yv)))
        lr = E.adjust_lr(opt, ep + 1, cfg)
        if log:
            log(f"    ep{ep:02d} val {v:.5f}")
        if v < best:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val": best, "best_epoch": best_ep, "epochs_run": ep}


def val_arrays(dataset: str):
    ds = D.get_dataset(dataset, "val")
    dx, dy = np.asarray(ds.data_x, np.float32), np.asarray(ds.data_y, np.float32)
    starts = np.arange(0, len(dx) - (SEQ_LEN + PRED_LEN), 24)
    X = np.stack([dx[s:s + SEQ_LEN] for s in starts])
    Y = np.stack([dy[s + SEQ_LEN:s + SEQ_LEN + PRED_LEN] for s in starts])
    return X, Y


def test_arrays(dataset: str, shifted: bool, sev_shift: float):
    """Clean test windows, or the same windows with the training shift family
    applied under an independent seed and independently drawn affected windows.
    The test targets are never inspected when choosing what to transform.
    """
    ds = D.get_dataset(dataset, "test")
    dx, dy = np.asarray(ds.data_x, np.float32), np.asarray(ds.data_y, np.float32)
    starts = np.arange(0, len(dx) - (SEQ_LEN + PRED_LEN), 24)
    X = np.stack([dx[s:s + SEQ_LEN] for s in starts])
    Y = np.stack([dy[s + SEQ_LEN:s + SEQ_LEN + PRED_LEN] for s in starts])
    if not shifted:
        return X, Y, np.zeros(len(X), dtype=bool)
    iqr = W.train_iqr(dataset)
    rng = np.random.RandomState(SHIFT_TEST_SEED)
    hit = rng.rand(len(X)) < W.SHIFT_FRACTION
    Xo, Yo = X.copy(), Y.copy()
    for i in np.flatnonzero(hit):
        kind = W.SHIFT_KINDS[rng.randint(len(W.SHIFT_KINDS))]
        Xo[i], Yo[i] = W.apply_shift(X[i], Y[i], sev_shift, iqr, rng, kind)
    return Xo, Yo, hit


# --------------------------------------------------------------------------- #
# Severity calibration (train-only)
# --------------------------------------------------------------------------- #

def calibrate_severity(dataset, cfg, clean_state, calib_starts, log):
    """Pick the (corruption, shift) severity pair whose median training loss
    gap is smallest, subject to both classes raising the median loss by at
    least 25% over clean. Uses a held-out train subset only; no test data.
    """
    model = E.build_model(cfg).to(E.DEVICE)
    model.load_state_dict(clean_state)
    model.eval()
    dx, dy = W.build_arrays(dataset)
    iqr = W.train_iqr(dataset)

    def med_loss(starts, transform, sev):
        rng = np.random.RandomState(999)
        Xs, Ys = [], []
        for s in starts:
            if s + SEQ_LEN + PRED_LEN > len(dx):
                continue
            x, y = W.make_window(dx, dy, s)
            if transform is not None:
                kinds = W.CORRUPTION_KINDS if transform is W.apply_corruption else W.SHIFT_KINDS
                x, y = transform(x, y, sev, iqr, rng, kinds[rng.randint(len(kinds))])
            Xs.append(x)
            Ys.append(y)
        return float(np.median(SEL.window_losses(model, cfg, np.stack(Xs), np.stack(Ys))))

    clean_med = med_loss(calib_starts["CLEAN"], None, 0.0)
    table = {"clean_median_loss": clean_med, "corruption": {}, "shift": {}}
    for sev in W.SEVERITY_CANDIDATES:
        table["corruption"][sev] = med_loss(calib_starts["CORRUPTION"], W.apply_corruption, sev)
        table["shift"][sev] = med_loss(calib_starts["LEGITIMATE_SHIFT"], W.apply_shift, sev)
    floor = clean_med * 1.25
    ok = [(sc, ss) for sc in W.SEVERITY_CANDIDATES for ss in W.SEVERITY_CANDIDATES
          if table["corruption"][sc] >= floor and table["shift"][ss] >= floor]
    weak = False
    if ok:
        sc, ss = min(ok, key=lambda p: abs(table["corruption"][p[0]] - table["shift"][p[1]]))
    else:
        weak = True
        sc, ss = min(((a, b) for a in W.SEVERITY_CANDIDATES for b in W.SEVERITY_CANDIDATES),
                     key=lambda p: abs(table["corruption"][p[0]] - table["shift"][p[1]]))
    del model
    torch.cuda.empty_cache()
    log(f"  [calib] clean_med {clean_med:.4f} floor {floor:.4f} -> "
        f"sev_corrupt {sc} ({table['corruption'][sc]:.4f}) "
        f"sev_shift {ss} ({table['shift'][ss]:.4f}) weak={weak}")
    return {"table": table, "sev_corrupt": sc, "sev_shift": ss,
            "LOSS_MATCHING_WEAK": weak, "floor": floor}


# --------------------------------------------------------------------------- #

def removal_rates(cls, keep) -> dict:
    out = {}
    for c in ("CORRUPTION", "LEGITIMATE_SHIFT", "CLEAN"):
        m = cls == c
        out[f"{c.lower()}_removal_rate"] = float((~keep[m]).mean()) if m.any() else float("nan")
    out["shift_retention"] = 1.0 - out["legitimate_shift_removal_rate"]
    out["overall_removal_rate"] = float((~keep).mean())
    return out


def run_dataset(dsname, clean, log, att):
    cfg = dlinear_config(dsname, CLEAN_SEEDS[0])
    clean_state = torch.load(
        paths.ROOT / clean["runs"][f"{dsname}/DLinear/seed{CLEAN_SEEDS[0]}"]["checkpoint_files"]["best"],
        map_location=E.DEVICE)

    classes = W.assign_classes(dsname)
    split = W.calibration_split(dsname, classes)
    calib_starts = {c: split[c]["calib"] for c in split}
    study_starts = {c: split[c]["study"] for c in split}
    assert not (set(calib_starts["CLEAN"]) & set(study_starts["CLEAN"]))
    allsets = [set(v) for v in classes.values()]
    assert not (allsets[0] & allsets[1]) and not (allsets[0] & allsets[2]) and not (allsets[1] & allsets[2])

    cal = calibrate_severity(dsname, cfg, clean_state, calib_starts, log)
    X, Y, cls, kinds, starts = W.materialise(dsname, study_starts,
                                             cal["sev_corrupt"], cal["sev_shift"],
                                             seed=W.CLASS_SEED + 7)
    log(f"  [set] {dsname} n={len(X)} "
        f"corr={int((cls=='CORRUPTION').sum())} shift={int((cls=='LEGITIMATE_SHIFT').sum())} "
        f"clean={int((cls=='CLEAN').sum())}")

    # Scores
    m0 = E.build_model(cfg).to(E.DEVICE)
    m0.load_state_dict(clean_state)
    loss_init = SEL.window_losses(m0, cfg, X, Y)

    # RHO reference / irreducible-loss model: trained only on a disjoint
    # temporal holdout block inside the training interval.
    cut = int(np.quantile(starts, REF_HOLDOUT_FRACTION))
    ref_mask = starts <= cut
    tgt_mask = starts > cut + SEQ_LEN + PRED_LEN
    Xv, Yv = val_arrays(dsname)
    log(f"  [rho] reference block n={int(ref_mask.sum())} target block n={int(tgt_mask.sum())} "
        f"time-disjoint={bool(starts[ref_mask].max() + SEQ_LEN + PRED_LEN < starts[tgt_mask].min())}")
    ref_model, ref_info = train_on_arrays(cfg, X[ref_mask], Y[ref_mask], Xv, Yv,
                                          seed=CLEAN_SEEDS[0] + 5, epochs=15)
    loss_ref = SEL.window_losses(ref_model, cfg, X, Y)

    # AdaRho, paper-faithful local implementation
    at = E.build_model(cfg).to(E.DEVICE)
    at.load_state_dict(clean_state)
    ar = E.build_model(cfg).to(E.DEVICE)
    ar.load_state_dict(clean_state)
    adarho_freq = SEL.adarho_scores(at, ar, cfg, X, Y, seed=CLEAN_SEEDS[0] + 11)
    del at, ar
    torch.cuda.empty_cache()

    coh = SEL.coherence_scores(X, Y, W.train_iqr(dsname))

    n = len(X)
    keeps = {
        "no_filter": SEL.sel_no_filter(n),
        "random_removal": SEL.sel_random(n, seed=CLEAN_SEEDS[0] + 3),
        "high_loss_removal": SEL.sel_high_loss(loss_init),
        "rho_loss": SEL.sel_rho(loss_init, loss_ref),
        "adarho": SEL.sel_adarho(adarho_freq),
        "coherence_aware": SEL.sel_coherence_aware(loss_init, loss_ref, coh),
    }
    budget = {k: int((~v).sum()) for k, v in keeps.items()}
    log(f"  [budget] {budget}")

    # Negative controls, scored but not retrained
    rngc = np.random.RandomState(CLEAN_SEEDS[0] + 17)
    controls = {
        "class_identity_shuffle": removal_rates(rngc.permutation(cls), keeps["coherence_aware"]),
        "coherence_row_shuffle": removal_rates(
            cls, SEL.sel_coherence_aware(loss_init, loss_ref, coh[rngc.permutation(n)])),
        "random_filter": removal_rates(cls, SEL.sel_random(n, seed=CLEAN_SEEDS[0] + 23)),
    }

    diag = {m: removal_rates(cls, keeps[m]) for m in METHODS}
    for m in METHODS:
        log(f"  [diag] {m:18s} corr_rm {diag[m]['corruption_removal_rate']:.3f} "
            f"shift_rm {diag[m]['legitimate_shift_removal_rate']:.3f} "
            f"clean_rm {diag[m]['clean_removal_rate']:.3f} "
            f"n_removed {budget[m]}")

    # Retrain and evaluate
    Xc, Yc, _ = test_arrays(dsname, False, cal["sev_shift"])
    Xs, Ys, hit = test_arrays(dsname, True, cal["sev_shift"])
    tier = json.loads((paths.RESULTS / "runtime_tier.json").read_text())
    seeds = CLEAN_SEEDS[:tier["tier_settings"]["intervention_seeds"]]
    retrain = {}
    for m in METHODS:
        for seed in seeds:
            k = keeps[m]
            cfg_s = dlinear_config(dsname, seed)
            mdl, info = train_on_arrays(cfg_s, X[k], Y[k], Xv, Yv, seed=seed)
            lc = SEL.window_losses(mdl, cfg_s, Xc, Yc)
            ls = SEL.window_losses(mdl, cfg_s, Xs, Ys)
            retrain[f"{m}/seed{seed}"] = {
                "method": m, "seed": seed, "n_train": int(k.sum()), **info,
                "clean_test_mse": float(lc.mean()),
                "shifted_test_mse": float(ls.mean()),
                "shifted_test_mse_on_shifted_windows": float(ls[hit].mean()),
                "per_window_clean": None,
            }
            np.savez_compressed(att / f"windowloss_{dsname}_{m}_s{seed}.npz",
                                clean=lc.astype(np.float32), shifted=ls.astype(np.float32),
                                hit=hit)
            log(f"  [retrain] {dsname} {m:18s} s{seed} clean {lc.mean():.5f} "
                f"shifted {ls.mean():.5f} shifted@hit {ls[hit].mean():.5f}")
            del mdl
            torch.cuda.empty_cache()

    np.savez_compressed(att / f"scores_{dsname}.npz", loss_init=loss_init,
                        loss_ref=loss_ref, adarho=adarho_freq, coherence=coh,
                        cls=cls, kinds=kinds, starts=starts,
                        **{f"keep_{m}": keeps[m] for m in METHODS})
    del m0, ref_model
    torch.cuda.empty_cache()
    return {
        "calibration": cal, "n_windows": int(n),
        "class_counts": {c: int((cls == c).sum()) for c in np.unique(cls)},
        "rho_reference": {**ref_info, "n_reference": int(ref_mask.sum()),
                          "time_disjoint": bool(starts[ref_mask].max() + SEQ_LEN + PRED_LEN
                                                < starts[tgt_mask].min())},
        "removal_budget": budget,
        "selection_diagnostic": diag,
        "controls": controls,
        "retraining": retrain,
        "n_test_windows": int(len(Xc)),
        "n_shifted_test_windows": int(hit.sum()),
    }


def main():
    done = A.completed(STAGE)
    if done:
        print(f"[resume] {STAGE} already completed at {done}")
        return
    clean = json.loads((paths.RESULTS / "clean_baselines.json").read_text())
    att = A.new_attempt(STAGE)
    mon = subprocess.Popen([sys.executable, str(paths.EXP / "common" / "resmon.py"),
                            str(psutil.Process().pid), str(att)])
    log = lambda s: print(s, flush=True)
    out = {"attempt": str(att), "methods": METHODS, "datasets": {}}
    t0 = time.time()
    try:
        for dsname in DATASETS:
            log(f"[track F] {dsname}")
            out["datasets"][dsname] = run_dataset(dsname, clean, log, att)
            A.write_json(att / "selection.json", out)
        out["wall_s"] = round(time.time() - t0, 1)
        A.write_json(att / "selection.json", out)
        A.finish(att, {"stage": STAGE, "ok": True})
        log(f"DONE {out['wall_s']}s")
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
