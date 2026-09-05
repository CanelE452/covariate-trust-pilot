"""Shared evaluation geometry: non-overlapping origins over a split.

This is infrastructure, not a track result. Each track calls it independently;
no track reads another track's artifacts through it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import data as D

N_ORIGINS = 8
ORIGIN_BATCH = 64


def origins(dataset: str, flag: str = "test", n: int = N_ORIGINS,
            batch: int = ORIGIN_BATCH) -> list:
    """n non-overlapping origins spread evenly across the split."""
    total = len(D.get_dataset(dataset, flag))
    usable = total - batch
    starts = np.linspace(0, usable, n).round().astype(int)
    out = [{"origin": int(k), "start": int(s),
            "window_starts": list(range(int(s), int(s) + batch))}
           for k, s in enumerate(starts)]
    for a, b in zip(out, out[1:]):
        assert a["start"] + batch <= b["start"], "origins overlap"
    return out
