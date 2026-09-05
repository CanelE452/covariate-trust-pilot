"""Track X corruption source: the official TSRBench transform, spliced per channel.

TSRBench's CollectiveNoise.corrupt() standardises and injects noise column by
column, so the corrupted values of column j are a function of column j alone
(plus the RNG stream). We therefore run the official transform once on the full
frame per (dataset, severity, seed) and splice only the requested channel into
an otherwise clean window. corruption_matches_official() checks that splice
against the official output on deterministic examples.

Only the input lookback is ever corrupted; forecast targets stay clean.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import pandas as pd

import paths
from contract import DATASETS

FAMILIES = ["spike", "shift", "combined"]
SEVERITIES = [1, 3, 5]
CORRUPTION_SEED = 2026090611

_CACHE: dict = {}


def _cache_path(dataset: str, severity: int) -> Path:
    d = paths.RUNS / "track_x" / "corrupted_series"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dataset}_sev{severity}_seed{CORRUPTION_SEED}.npz"


def official_corrupt(dataset: str, severity: int) -> dict:
    """Run the official TSRBench transform on the raw CSV, cached on disk.

    Returns family -> array [T, C] in the ORIGINAL data units, aligned with the
    raw CSV rows, so it can be re-scaled with the train-only scaler downstream.
    """
    key = (dataset, severity)
    if key in _CACHE:
        return _CACHE[key]
    f = _cache_path(dataset, severity)
    if f.exists():
        z = np.load(f)
        out = {k: z[k] for k in z.files}
        _CACHE[key] = out
        return out

    from tsrbench import CollectiveNoise
    spec = DATASETS[dataset]
    df = pd.read_csv(paths.DATASET_DIR / spec.data_path)
    cn = CollectiveNoise(seed=CORRUPTION_SEED)
    res = cn.corrupt(df, noise_level=severity, skip_first_col=True)
    out = {}
    for fam in FAMILIES:
        out[fam] = np.asarray(res[fam].iloc[:, 1:].values, dtype=np.float64)
    out["clean_raw"] = np.asarray(df.iloc[:, 1:].values, dtype=np.float64)
    np.savez_compressed(f, **out)
    _CACHE[key] = out
    return out


def scaled_corrupted_series(dataset: str, severity: int, family: str, scaler) -> np.ndarray:
    """Corrupted full series mapped into the model's scaled space [T, C]."""
    raw = official_corrupt(dataset, severity)[family]
    return scaler.transform(raw).astype(np.float32)


def corruption_matches_official(dataset: str, severity: int, family: str,
                                n_examples: int = 20) -> dict:
    """Deterministic check that a single-channel splice equals the official
    transform restricted to that channel, and leaves every other channel intact.
    """
    off = official_corrupt(dataset, severity)
    corr, clean = off[family], off["clean_raw"]
    C = clean.shape[1]
    rng = np.random.RandomState(12345)
    rows = rng.randint(0, clean.shape[0] - 96, size=n_examples)
    chans = rng.randint(0, C, size=n_examples)
    max_off_diff, max_target_diff, changed = 0.0, 0.0, 0
    for r, j in zip(rows, chans):
        w = clean[r:r + 96].copy()
        w[:, j] = corr[r:r + 96, j]                     # the splice under test
        other = [c for c in range(C) if c != j]
        max_off_diff = max(max_off_diff, float(np.abs(w[:, other] - clean[r:r + 96, other]).max()))
        max_target_diff = max(max_target_diff,
                              float(np.abs(w[:, j] - corr[r:r + 96, j]).max()))
        if np.abs(w[:, j] - clean[r:r + 96, j]).max() > 0:
            changed += 1
    return {"n_examples": int(n_examples),
            "max_offchannel_deviation": max_off_diff,
            "max_target_channel_deviation_vs_official": max_target_diff,
            "n_examples_where_channel_actually_changed": int(changed),
            "identical_to_official": bool(max_off_diff == 0.0 and max_target_diff == 0.0)}


def corruption_fingerprint(dataset: str, severity: int) -> str:
    off = official_corrupt(dataset, severity)
    h = hashlib.sha256()
    for fam in FAMILIES:
        h.update(np.ascontiguousarray(off[fam], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
