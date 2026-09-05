"""Origin-cluster / moving-block bootstrap for time-dependent evaluation.

An iid row bootstrap is never used: ETTm1 and Weather windows are strongly
serially dependent. Resampling happens at the level of evaluation origins (or
probe blocks), and every channel result belonging to an origin travels with it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import paths
from contract import BOOTSTRAP_SEED


def n_resamples() -> int:
    return json.loads((paths.RESULTS / "runtime_tier.json").read_text())["tier_settings"]["bootstrap"]


def cluster_bootstrap(values_by_cluster: dict, stat=np.median, n: int | None = None,
                      seed: int = BOOTSTRAP_SEED, block: int = 1) -> dict:
    """Resample whole clusters with replacement and recompute the statistic.

    values_by_cluster maps a cluster key (an origin, a probe block) to the list
    of values observed inside it. block > 1 draws contiguous runs of clusters,
    which is the moving-block form for clusters ordered in time.
    """
    keys = sorted(values_by_cluster)
    K = len(keys)
    if K == 0:
        return {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_resamples": 0, "n_clusters": 0}
    n = n or n_resamples()
    rng = np.random.RandomState(seed)
    flat = np.concatenate([np.asarray(values_by_cluster[k], dtype=float) for k in keys])
    point = float(stat(flat))
    draws = np.empty(n, dtype=float)
    n_blocks = int(np.ceil(K / block))
    for b in range(n):
        picked = []
        for _ in range(n_blocks):
            s = rng.randint(0, K)
            picked.extend(keys[(s + o) % K] for o in range(block))
        vals = np.concatenate([np.asarray(values_by_cluster[k], dtype=float)
                               for k in picked[:K]])
        draws[b] = stat(vals)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci_lower": float(lo), "ci_upper": float(hi),
            "n_resamples": int(n), "n_clusters": int(K), "block": int(block),
            "seed": int(seed)}


def paired_difference_bootstrap(a_by_cluster: dict, b_by_cluster: dict,
                                relative: bool = True, n: int | None = None,
                                seed: int = BOOTSTRAP_SEED, block: int = 1) -> dict:
    """Bootstrap the paired improvement of arm a over arm b, clustering origins.

    relative=True reports (mean_b - mean_a) / mean_b, the fractional gain of a.
    """
    keys = sorted(set(a_by_cluster) & set(b_by_cluster))
    K = len(keys)
    if K == 0:
        return {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_resamples": 0, "n_clusters": 0}
    n = n or n_resamples()
    rng = np.random.RandomState(seed)

    def stat(ks):
        A = np.mean([np.mean(a_by_cluster[k]) for k in ks])
        B = np.mean([np.mean(b_by_cluster[k]) for k in ks])
        return (B - A) / B if relative else (B - A)

    point = float(stat(keys))
    draws = np.empty(n, dtype=float)
    n_blocks = int(np.ceil(K / block))
    for i in range(n):
        picked = []
        for _ in range(n_blocks):
            s = rng.randint(0, K)
            picked.extend(keys[(s + o) % K] for o in range(block))
        draws[i] = stat(picked[:K])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci_lower": float(lo), "ci_upper": float(hi),
            "n_resamples": int(n), "n_clusters": int(K), "block": int(block),
            "seed": int(seed), "relative": bool(relative)}
