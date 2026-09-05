"""Track X, stages 11-13: robust clipping, channel-dropout training, and the
detected-quarantine intervention.

No oracle corruption mask is ever used. The quarantine detector is calibrated on
CLEAN VALIDATION windows to a 5% channel-window false-positive rate; the
threshold is then frozen for test.
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
import bootstrap as B
from contract import DATASETS, CLEAN_SEEDS, SEQ_LEN, PRED_LEN, tqnet_config
import data as D
import engine as E
import corruption as X
import spec as S
from phenomenon import channel_losses, spliced_input, test_offset

STAGE = "track_x_mitigations"
CLIP_K = 5.0
DROPOUT_RATE = 0.10
FP_TARGET = 0.05
QUARANTINE_SEED = 2026090651


# --------------------------------------------------------------------------- #
# Baseline 1: robust clipping (train statistics, no retraining)
# --------------------------------------------------------------------------- #

def clip_inputs(x: torch.Tensor, med, iqr) -> torch.Tensor:
    lo = torch.as_tensor(med - CLIP_K * iqr, device=x.device, dtype=x.dtype)
    hi = torch.as_tensor(med + CLIP_K * iqr, device=x.device, dtype=x.dtype)
    return torch.max(torch.min(x, hi.view(1, 1, -1)), lo.view(1, 1, -1))


# --------------------------------------------------------------------------- #
# Baseline 2: channel-dropout training
# --------------------------------------------------------------------------- #

def channel_dropout_transform(med):
    m = np.asarray(med, dtype=np.float32)

    def fn(bx, by, cyc, rng):
        B_, T_, C_ = bx.shape
        drop = torch.from_numpy(
            (rng.rand(B_, C_) < DROPOUT_RATE)).to(bx.device)
        if drop.any():
            rep = torch.from_numpy(m).to(bx.device).view(1, 1, -1).expand(B_, T_, C_)
            bx = torch.where(drop.unsqueeze(1), rep, bx)
        return bx, by
    return fn


# --------------------------------------------------------------------------- #
# Intervention: detected quarantine
# --------------------------------------------------------------------------- #

def anomaly_scores(x: np.ndarray, med, iqr, std) -> np.ndarray:
    """Per (window, channel) anomaly score from three robust components.

    A_spike: max robust |z| over the lookback.
    A_shift: |median(last 24) - median(previous 24)| / IQR_train.
    A_stuck: 1 - std(last 24) / std_train.
    """
    med = np.asarray(med, np.float32)
    iqr = np.asarray(iqr, np.float32) + 1e-6
    std = np.asarray(std, np.float32) + 1e-6
    z = np.abs(x - med[None, None, :]) / iqr[None, None, :]
    a_spike = z.max(axis=1)
    a_shift = np.abs(np.median(x[:, -24:, :], axis=1)
                     - np.median(x[:, -48:-24, :], axis=1)) / iqr[None, :]
    a_stuck = 1.0 - x[:, -24:, :].std(axis=1) / std[None, :]
    return np.stack([a_spike / 6.0, a_shift / 3.0, np.clip(a_stuck, 0, None)], axis=0).max(axis=0)


def calibrate_threshold(dataset: str, med, iqr, std) -> dict:
    """Threshold giving a 5% channel-window false-positive rate on CLEAN
    validation windows. Test data is never used.
    """
    ds = D.get_dataset(dataset, "val")
    dx = np.asarray(ds.data_x, np.float32)
    rng = np.random.RandomState(QUARANTINE_SEED)
    starts = rng.choice(len(dx) - SEQ_LEN - PRED_LEN, size=min(2000, len(dx) // 4),
                        replace=False)
    Xw = np.stack([dx[s:s + SEQ_LEN] for s in starts])
    sc = anomaly_scores(Xw, med, iqr, std)
    thr = float(np.quantile(sc.ravel(), 1.0 - FP_TARGET))
    return {"threshold": thr, "achieved_fp_rate": float((sc > thr).mean()),
            "n_calibration_windows": int(len(starts)), "split": "validation (clean)"}


def quarantine_inputs(x: torch.Tensor, med, iqr, std, thr) -> tuple:
    xn = x.detach().cpu().numpy()
    sc = anomaly_scores(xn, med, iqr, std)
    flag = sc > thr
    out = x.clone()
    if flag.any():
        rep = torch.as_tensor(np.asarray(med, np.float32), device=x.device).view(1, 1, -1)
        mask = torch.from_numpy(flag).to(x.device).unsqueeze(1)
        out = torch.where(mask, rep.expand_as(x), out)
    return out, flag


# --------------------------------------------------------------------------- #

def evaluate_arm(model, cfg, dsname, spec, scaler, transform, log, tag):
    """Return clean MSE, direct damage and off-diagonal spillover for one arm."""
    off = test_offset(dsname)
    C = cfg.enc_in
    origins = spec["evaluation_origins"]
    chans = spec["corrupted_channels"]
    clean_cache, clean_total = {}, []
    for o in origins:
        x, y, c = D.window_batch(dsname, "test", o["window_starts"], E.DEVICE)
        xt = transform(x)[0] if transform else x
        L = channel_losses(model, cfg, xt, y, c)
        clean_cache[o["origin"]] = (L, x, y, c)
        clean_total.append(float(L.mean()))
    recs = []
    flagged = []
    for sev in spec["severities"]:
        for fam in spec["corruption_families"]:
            cser = X.scaled_corrupted_series(dsname, sev, fam, scaler)
            for j in chans:
                for o in origins:
                    L0, x, y, c = clean_cache[o["origin"]]
                    starts = [off + s for s in o["window_starts"]]
                    xc = spliced_input(x, cser, starts, j, E.DEVICE)
                    if transform:
                        xc, fl = transform(xc)
                        if fl is not None:
                            flagged.append(float(fl[:, j].mean()))
                    Ld = channel_losses(model, cfg, xc, y, c)
                    rel = (Ld - L0) / (L0 + 1e-8)
                    recs.append({"family": fam, "severity": sev, "j": int(j),
                                 "origin": o["origin"], "rel": rel.tolist()})
    offd, diag, byo = [], [], {}
    for r in recs:
        rel = np.array(r["rel"])
        j = r["j"]
        o = [rel[i] for i in range(C) if i != j]
        offd.extend(o)
        diag.append(rel[j])
        byo.setdefault(r["origin"], []).extend(o)
    res = {"clean_mse": float(np.mean(clean_total)),
           "median_offdiag_spillover": float(np.median(offd)),
           "mean_offdiag_spillover": float(np.mean(offd)),
           "p90_offdiag_spillover": float(np.percentile(offd, 90)),
           "direct_damage_median": float(np.median(diag)),
           "detector_flag_rate_on_corrupted_channel": (float(np.mean(flagged))
                                                       if flagged else None),
           "n_offdiag_samples": int(len(offd))}
    log(f"    [{tag}] clean {res['clean_mse']:.5f} offdiag_med "
        f"{res['median_offdiag_spillover']:+.4f} direct {res['direct_damage_median']:+.4f}")
    return res, byo


def main():
    done = A.completed(STAGE)
    if done:
        print(f"[resume] {STAGE} already completed at {done}")
        return
    clean = json.loads((paths.RESULTS / "clean_baselines.json").read_text())
    tier = json.loads((paths.RESULTS / "runtime_tier.json").read_text())
    seeds = CLEAN_SEEDS[:tier["tier_settings"]["intervention_seeds"]]

    att = A.new_attempt(STAGE)
    mon = subprocess.Popen([sys.executable, str(paths.EXP / "common" / "resmon.py"),
                            str(psutil.Process().pid), str(att)])
    log = lambda s: print(s, flush=True)
    out = {"attempt": str(att), "datasets": {}}
    t0 = time.time()
    try:
        for dsname in DATASETS:
            log(f"[track X mitigations] {dsname}")
            spec = S.build(dsname)
            med = np.array(spec["train_robust_stats"]["median"], np.float32)
            iqr = np.array(spec["train_robust_stats"]["iqr"], np.float32)
            std = np.array(spec["train_robust_stats"]["std"], np.float32)
            scaler = D.get_dataset(dsname, "train").scaler
            cfg = tqnet_config(dsname, CLEAN_SEEDS[0])
            base_state = torch.load(
                paths.ROOT / clean["runs"][f"{dsname}/TQNet/seed{CLEAN_SEEDS[0]}"]["checkpoint_files"]["best"],
                map_location=E.DEVICE)
            model = E.build_model(cfg).to(E.DEVICE)
            model.load_state_dict(base_state)
            model.eval()

            arms, byo = {}, {}
            arms["clean_model"], byo["clean_model"] = evaluate_arm(
                model, cfg, dsname, spec, scaler, None, log, "clean model")
            base_clean_mse = arms["clean_model"]["clean_mse"]
            base_spill = arms["clean_model"]["median_offdiag_spillover"]
            base_direct = arms["clean_model"]["direct_damage_median"]

            arms["clipping"], byo["clipping"] = evaluate_arm(
                model, cfg, dsname, spec, scaler,
                lambda x: (clip_inputs(x, med, iqr), None), log, "clipping")
            del model
            torch.cuda.empty_cache()

            # Channel-dropout retraining, same budget, rate frozen at 0.10
            drop_states = {}
            for seed in seeds:
                cfg_s = tqnet_config(dsname, seed)
                log(f"    [train] channel dropout seed{seed}")
                r = E.train_model(cfg_s, dsname,
                                  batch_transform=channel_dropout_transform(med))
                drop_states[seed] = r.checkpoints["best"]
                torch.save(r.checkpoints["best"], att / f"{dsname}_dropout_s{seed}.pt")
                del r
                torch.cuda.empty_cache()
            dmodel = E.build_model(tqnet_config(dsname, seeds[0])).to(E.DEVICE)
            dmodel.load_state_dict(drop_states[seeds[0]])
            dmodel.eval()
            arms["channel_dropout"], byo["channel_dropout"] = evaluate_arm(
                dmodel, tqnet_config(dsname, seeds[0]), dsname, spec, scaler, None,
                log, "channel dropout")

            cal = calibrate_threshold(dsname, med, iqr, std)
            log(f"    [quarantine] threshold {cal['threshold']:.4f} "
                f"(clean-val FP {cal['achieved_fp_rate']:.3f})")
            arms["quarantine"], byo["quarantine"] = evaluate_arm(
                dmodel, tqnet_config(dsname, seeds[0]), dsname, spec, scaler,
                lambda x: quarantine_inputs(x, med, iqr, std, cal["threshold"]),
                log, "dropout + quarantine")
            del dmodel
            torch.cuda.empty_cache()

            for k, v in arms.items():
                v["clean_mse_degradation"] = (v["clean_mse"] - base_clean_mse) / base_clean_mse
                v["direct_damage_degradation"] = v["direct_damage_median"] - base_direct
                v["spillover_reduction_vs_clean_model"] = (
                    (base_spill - v["median_offdiag_spillover"]) / (abs(base_spill) + 1e-12))

            best_simple = max(("clipping", "channel_dropout"),
                              key=lambda k: arms[k]["spillover_reduction_vs_clean_model"])
            qb = B.paired_difference_bootstrap(byo["quarantine"], byo[best_simple],
                                               relative=True, block=2)
            out["datasets"][dsname] = {
                "baseline_spillover": base_spill,
                "baseline_clean_mse": base_clean_mse,
                "baseline_direct_damage": base_direct,
                "quarantine_calibration": cal,
                "best_simple_for_bootstrap": best_simple,
                "quarantine_bootstrap": qb,
                "arms": arms,
                "dropout_rate": DROPOUT_RATE, "clip_k": CLIP_K,
                "oracle_mask_used": False,
            }
            A.write_json(att / "mitigations.json", out)
        out["wall_s"] = round(time.time() - t0, 1)
        A.write_json(att / "mitigations.json", out)
        A.finish(att, {"stage": STAGE, "ok": True})
        log(f"DONE {out['wall_s']}s")
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
