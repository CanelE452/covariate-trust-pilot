"""Track G, stages 15/16/18: gradient diagnostic, exact virtual-step harm, controls.

For each clean checkpoint (early / middle / best) we build train-only temporal
probe pairs, compute the same-batch conflict cosine and the cross-probe harmful
affinity, then validate a sample of task pairs with an exact virtual update.
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
import grads as G

STAGE = "track_g_diagnostic"
HARM_THRESHOLD = 0.005
N_EXACT_TOP = 64
N_EXACT_RANDOM = 64
PROBE_SEED = 2026090621
CKPT_TAGS = ["early", "middle", "best"]


def weather_task_subset(n_take: int = 12) -> list:
    """Track G uses the same 12-variable subset rule as the pre-analysis spec:
    sort channels by train variance and take evenly spaced sorted positions so
    that low, medium and high variance variables are all represented.
    Test data is never consulted.
    """
    x, _, _ = D.as_arrays("Weather", "train")
    order = np.argsort(x.var(axis=0))
    pos = np.linspace(0, len(order) - 1, n_take).round().astype(int)
    return sorted(int(order[p]) for p in np.unique(pos))


def tasks_for(dataset: str) -> list:
    if dataset == "ETTm1":
        return list(range(DATASETS[dataset].enc_in))
    return weather_task_subset(12)


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision, computed directly to avoid a sklearn dependency."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    o = np.argsort(-scores)
    lab = labels[o]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / lab.sum())


def run_checkpoint(dataset, seed, tag, ckpt_file, n_pairs, log, npz_path):
    cfg = tqnet_config(dataset, seed)
    model = E.build_model(cfg).to(E.DEVICE)
    state = torch.load(ckpt_file, map_location=E.DEVICE)
    model.load_state_dict(state)
    model.eval()
    base_state = {k: v.detach().to(E.DEVICE) for k, v in model.state_dict().items()}
    params = G.shared_param_list(model, cfg.enc_in)
    tasks = tasks_for(dataset)
    T = len(tasks)
    pairs = G.probe_pairs(dataset, n_pairs, batch=64, seed=PROBE_SEED + seed)
    assert all(G.blocks_disjoint(p) for p in pairs), "probe blocks overlap"

    rows = []
    cos_all, aff_all = [], []
    rng = np.random.RandomState(PROBE_SEED)
    for pi, pr in enumerate(pairs):
        xb, yb, cb = D.window_batch(dataset, "train", pr["train_starts"], E.DEVICE)
        xp, yp, cp = D.window_batch(dataset, "train", pr["probe_starts"], E.DEVICE)
        Gb = G.all_task_grads(model, cfg, xb, yb, cb, tasks, params)
        Gp = G.all_task_grads(model, cfg, xp, yp, cp, tasks, params)
        Cij = G.cosine_matrix(Gb)                               # same-batch conflict
        Gbn = Gb / (Gb.norm(dim=1, keepdim=True) + 1e-12)
        Gpn = Gp / (Gp.norm(dim=1, keepdim=True) + 1e-12)
        Aff = -(Gpn @ Gbn.T).detach().cpu().numpy()             # A[i, j] = -cos(g_i_probe, g_j_train)
        step = G.erm_step_norm(model, cfg, xb, yb, cb, params)
        for a in range(T):
            for b in range(T):
                if a == b:
                    continue
                rows.append({"pair": pi, "i": tasks[a], "j": tasks[b],
                             "cos_same_batch": float(Cij[a, b]),
                             "cross_probe_affinity": float(Aff[a, b]),
                             "step_norm": step, "ia": a, "jb": b})
        cos_all.append(Cij)
        aff_all.append(Aff)
        del Gb, Gp
        torch.cuda.empty_cache()

    # Exact virtual-step validation: the 64 pairs the affinity flags as most
    # harmful, plus 64 pairs drawn at random. Selection uses only train/probe
    # diagnostics, never a test outcome.
    idx_sorted = sorted(range(len(rows)), key=lambda k: -rows[k]["cross_probe_affinity"])
    top = idx_sorted[:min(N_EXACT_TOP, len(rows))]
    rest = [k for k in range(len(rows)) if k not in set(top)]
    rand = list(rng.choice(rest, size=min(N_EXACT_RANDOM, len(rest)), replace=False)) if rest else []
    to_check = sorted(set(top) | set(int(k) for k in rand))

    cache = {}
    for k in to_check:
        r = rows[k]
        pr = pairs[r["pair"]]
        if r["pair"] not in cache:
            cache.clear()
            cache[r["pair"]] = (D.window_batch(dataset, "train", pr["train_starts"], E.DEVICE),
                                D.window_batch(dataset, "train", pr["probe_starts"], E.DEVICE))
        (xb, yb, cb), (xp, yp, cp) = cache[r["pair"]]
        h = G.exact_harm(model, cfg, base_state, params, xb, yb, cb, xp, yp, cp,
                         j=r["j"], i=r["i"], step_norm=r["step_norm"])
        r["exact_harm"] = float(h)
        r["harm_label"] = bool(h > HARM_THRESHOLD)
        r["selection"] = "top_affinity" if k in set(top) else "random"

    checked = [r for r in rows if "exact_harm" in r]
    lab = np.array([r["harm_label"] for r in checked], dtype=float)
    cos_s = np.array([-r["cos_same_batch"] for r in checked])   # more negative cos -> higher score
    aff_s = np.array([r["cross_probe_affinity"] for r in checked])
    cos_detect = np.array([r["cos_same_batch"] < 0 for r in checked], dtype=float)

    fp = float(((cos_detect == 1) & (lab == 0)).sum() / max(1, (cos_detect == 1).sum()))
    fn = float(((cos_detect == 0) & (lab == 1)).sum() / max(1, (lab == 1).sum()))
    out = {
        "dataset": dataset, "seed": seed, "checkpoint_tag": tag,
        "n_tasks": T, "tasks": tasks, "n_pairs": len(pairs),
        "n_task_pairs_scored": len(rows), "n_exact_checked": len(checked),
        "exact_harm_rate": float(lab.mean()),
        "exact_harm_rate_random_subset": float(
            np.mean([r["harm_label"] for r in checked if r["selection"] == "random"])
            if any(r["selection"] == "random" for r in checked) else float("nan")),
        "exact_harm_rate_top_affinity": float(
            np.mean([r["harm_label"] for r in checked if r["selection"] == "top_affinity"])),
        "same_batch_cosine_detector": {"false_positive_rate": fp, "false_negative_rate": fn,
                                       "flag_rate": float(cos_detect.mean())},
        "auprc_same_batch_cosine": auprc(cos_s, lab),
        "auprc_cross_probe_affinity": auprc(aff_s, lab),
        "mean_cos_same_batch": float(np.mean([r["cos_same_batch"] for r in rows])),
        "frac_negative_cosine": float(np.mean([r["cos_same_batch"] < 0 for r in rows])),
        "rows": checked,
        "cos_quantiles": [float(q) for q in np.quantile(
            [r["cos_same_batch"] for r in rows], [0.05, 0.25, 0.5, 0.75, 0.95])],
        "affinity_quantiles": [float(q) for q in np.quantile(
            [r["cross_probe_affinity"] for r in rows], [0.05, 0.25, 0.5, 0.75, 0.95])],
    }
    out["auprc_gain_cross_probe_over_cosine"] = (out["auprc_cross_probe_affinity"]
                                                 - out["auprc_same_batch_cosine"])
    # Negative controls
    perm = rng.permutation(len(checked))
    out["control_random_task_pairing_auprc"] = auprc(aff_s[perm], lab)
    out["control_gradient_sign_randomised_auprc"] = auprc(
        aff_s * rng.choice([-1.0, 1.0], size=len(aff_s)), lab)
    np.savez_compressed(
        npz_path,
        cos=np.array([r["cos_same_batch"] for r in rows], dtype=np.float32),
        aff=np.array([r["cross_probe_affinity"] for r in rows], dtype=np.float32),
        pair=np.array([r["pair"] for r in rows], dtype=np.int32),
        i=np.array([r["i"] for r in rows], dtype=np.int32),
        j=np.array([r["j"] for r in rows], dtype=np.int32))
    log(f"    {dataset} s{seed} {tag}: harm {out['exact_harm_rate']:.3f} "
        f"AUPRC cos {out['auprc_same_batch_cosine']:.3f} probe "
        f"{out['auprc_cross_probe_affinity']:.3f}")
    del model
    torch.cuda.empty_cache()
    return out


def main():
    done = A.completed(STAGE)
    if done:
        print(f"[resume] {STAGE} already completed at {done}")
        return
    tier = json.loads((paths.RESULTS / "runtime_tier.json").read_text())["tier"]
    n_pairs = {"FULL": 32, "COMPACT": 24, "MINIMAL-COMPLETE": 16}[tier]
    clean = json.loads((paths.RESULTS / "clean_baselines.json").read_text())

    att = A.new_attempt(STAGE)
    mon = subprocess.Popen([sys.executable, str(paths.EXP / "common" / "resmon.py"),
                            str(psutil.Process().pid), str(att)])
    log = lambda s: print(s, flush=True)
    out = {"attempt": str(att), "tier": tier, "n_probe_pairs": n_pairs,
           "harm_threshold": HARM_THRESHOLD, "checkpoints": {}}
    try:
        m0 = E.build_model(tqnet_config("ETTm1", CLEAN_SEEDS[0]))
        out["shared_parameter_audit"] = {
            "ETTm1": G.audit_shared_parameters(m0, DATASETS["ETTm1"].enc_in)}
        m1 = E.build_model(tqnet_config("Weather", CLEAN_SEEDS[0]))
        out["shared_parameter_audit"]["Weather"] = G.audit_shared_parameters(
            m1, DATASETS["Weather"].enc_in)
        del m0, m1

        t0 = time.time()
        for dsname in DATASETS:
            for seed in CLEAN_SEEDS:
                run = clean["runs"][f"{dsname}/TQNet/seed{seed}"]
                for tag in CKPT_TAGS:
                    f = paths.ROOT / run["checkpoint_files"][tag]
                    key = f"{dsname}/seed{seed}/{tag}"
                    log(f"[diag] {key}")
                    out["checkpoints"][key] = run_checkpoint(
                        dsname, seed, tag, f, n_pairs, log,
                        att / f"scores_{dsname}_s{seed}_{tag}.npz")
                    A.write_json(att / "gradient_diagnostic.json", out)
        out["wall_s"] = round(time.time() - t0, 1)
        A.write_json(att / "gradient_diagnostic.json", out)
        A.finish(att, {"stage": STAGE, "ok": True})
        log(f"DONE {out['wall_s']}s")
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
