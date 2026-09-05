"""Track X, stage 10: does corrupting one input channel damage the OTHER
channels' forecasts through a shared multivariate model?

For every (dataset, family, severity, corrupted channel j, origin) we evaluate
the clean checkpoint on inputs whose channel j alone carries the official
TSRBench corruption, and record the per-channel loss inflation
S[j -> i] = (L_i(corrupted j) - L_i(clean)) / L_i(clean).
The diagonal is direct damage; the off-diagonal is spillover.
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
from contract import DATASETS, CLEAN_SEEDS, PRED_LEN, SEQ_LEN, tqnet_config, dlinear_config
import data as D
import engine as E
import corruption as X
import spec as S

STAGE = "track_x_phenomenon"


def channel_losses(model, cfg, x, y, cyc) -> np.ndarray:
    """Per-output-channel MSE over a batch of windows."""
    with torch.no_grad():
        out = E.forward(model, cfg, x, cyc)[:, -PRED_LEN:, :]
        d = (out - y[:, -PRED_LEN:, :]) ** 2
        return d.mean(dim=(0, 1)).double().cpu().numpy()


def spliced_input(x_clean: torch.Tensor, corr_series: np.ndarray, starts, j: int,
                  device) -> torch.Tensor:
    """Replace channel j of every lookback window with the corrupted series."""
    x = x_clean.clone()
    rows = np.stack([corr_series[s:s + SEQ_LEN, j] for s in starts])
    x[:, :, j] = torch.from_numpy(rows).float().to(device)
    return x


def test_offset(dataset: str) -> int:
    """Row offset of the test split inside the raw CSV, so that a test window
    start maps to the right rows of the corrupted full series.
    """
    n = D.get_dataset(dataset, "test").data_x.shape[0]
    total = json.loads((paths.RESULTS / "data_manifest.json").read_text())[dataset]["row_count"]
    if dataset == "ETTm1":
        return 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - SEQ_LEN
    num_train = int(total * 0.7)
    num_test = int(total * 0.2)
    return total - num_test - SEQ_LEN


def run_model(dsname, model_key, cfg_fn, ckpt_files, spec, log):
    """Return the raw per-(family, severity, j, origin) channel-loss table."""
    cfg = cfg_fn(dsname, CLEAN_SEEDS[0])
    scaler = D.get_dataset(dsname, "train").scaler
    off = test_offset(dsname)
    C = cfg.enc_in
    origins = spec["evaluation_origins"]
    chans = spec["corrupted_channels"]
    recs = []
    for seed, ck in ckpt_files.items():
        cfg = cfg_fn(dsname, seed)
        model = E.build_model(cfg).to(E.DEVICE)
        model.load_state_dict(torch.load(ck, map_location=E.DEVICE))
        model.eval()
        clean_by_origin = {}
        for o in origins:
            x, y, c = D.window_batch(dsname, "test", o["window_starts"], E.DEVICE)
            clean_by_origin[o["origin"]] = (channel_losses(model, cfg, x, y, c), x, y, c)
        for sev in spec["severities"]:
            for fam in spec["corruption_families"]:
                cser = X.scaled_corrupted_series(dsname, sev, fam, scaler)
                for j in chans:
                    for o in origins:
                        Lc, x, y, c = clean_by_origin[o["origin"]]
                        starts = [off + s for s in o["window_starts"]]
                        xc = spliced_input(x, cser, starts, j, E.DEVICE)
                        Ld = channel_losses(model, cfg, xc, y, c)
                        rel = (Ld - Lc) / (Lc + 1e-8)
                        recs.append({"model": model_key, "seed": seed, "family": fam,
                                     "severity": sev, "j": int(j), "origin": o["origin"],
                                     "rel": rel.tolist()})
                log(f"    {dsname} {model_key} s{seed} sev{sev} {fam} done")
        del model
        torch.cuda.empty_cache()
    return recs


def summarise(recs, chans, C) -> dict:
    """Aggregate the raw table into the quantities the phenomenon gate reads."""
    out = {}
    fam_sev = sorted({(r["family"], r["severity"]) for r in recs})
    for fam, sev in fam_sev:
        sub = [r for r in recs if r["family"] == fam and r["severity"] == sev]
        offd, diag = [], []
        per_source = {int(j): [] for j in chans}
        worst, frac_gt1 = [], []
        for r in sub:
            rel = np.array(r["rel"])
            j = r["j"]
            o = [rel[i] for i in range(C) if i != j]
            offd.extend(o)
            diag.append(rel[j])
            per_source[j].extend(o)
            worst.append(max(o))
            frac_gt1.append(float(np.mean([v > 0.01 for v in o])))
        offd = np.array(offd)
        share = {j: float(np.sum(np.clip(v, 0, None))) for j, v in per_source.items()}
        tot = sum(share.values()) + 1e-12
        out[f"{fam}_sev{sev}"] = {
            "family": fam, "severity": sev,
            "direct_damage_median": float(np.median(diag)),
            "median_offdiag_spillover": float(np.median(offd)),
            "mean_offdiag_spillover": float(np.mean(offd)),
            "p90_offdiag_spillover": float(np.percentile(offd, 90)),
            "fraction_normal_channels_damage_gt_1pct": float(np.mean(frac_gt1)),
            "worst_normal_channel_damage": float(np.max(worst)),
            "max_single_source_channel_share": float(max(share.values()) / tot),
            "n_offdiag_samples": int(offd.size),
        }
    return out


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
    out = {"attempt": str(att), "datasets": {}}
    t0 = time.time()
    try:
        for dsname in DATASETS:
            spec = S.build(dsname)
            A.write_json(paths.RESULTS / "track_x" / "pre_analysis_spec.json"
                         if dsname == "ETTm1" else
                         paths.RESULTS / "track_x" / f"pre_analysis_spec_{dsname}.json", spec)
            splice = {f"{fam}_sev{sev}": X.corruption_matches_official(dsname, sev, fam)
                      for sev in spec["severities"] for fam in spec["corruption_families"]}
            C = DATASETS[dsname].enc_in
            entry = {"spec_channels": spec["corrupted_channels"],
                     "splice_verification": splice, "raw": {}, "summary": {}}
            for mk, fn in [("TQNet", tqnet_config), ("DLinear", dlinear_config)]:
                cks = {s: paths.ROOT / clean["runs"][f"{dsname}/{mk}/seed{s}"]["checkpoint_files"]["best"]
                       for s in CLEAN_SEEDS}
                recs = run_model(dsname, mk, fn, cks, spec, log)
                entry["raw"][mk] = recs
                entry["summary"][mk] = summarise(recs, spec["corrupted_channels"], C)
            entry["tqnet_over_dlinear_spillover"] = {
                k: (entry["summary"]["TQNet"][k]["median_offdiag_spillover"]
                    / (abs(entry["summary"]["DLinear"][k]["median_offdiag_spillover"]) + 1e-8))
                for k in entry["summary"]["TQNet"]}
            out["datasets"][dsname] = entry
            A.write_json(att / "phenomenon_raw.json", out)
            for k, v in entry["summary"]["TQNet"].items():
                log(f"  [{dsname}] {k}: direct {v['direct_damage_median']:+.4f} "
                    f"offdiag_med {v['median_offdiag_spillover']:+.4f} "
                    f"p90 {v['p90_offdiag_spillover']:+.4f} "
                    f"ratio_vs_DLinear {entry['tqnet_over_dlinear_spillover'][k]:.2f}")
        out["wall_s"] = round(time.time() - t0, 1)
        A.write_json(att / "phenomenon_raw.json", out)
        A.finish(att, {"stage": STAGE, "ok": True})
        log(f"DONE {out['wall_s']}s")
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
