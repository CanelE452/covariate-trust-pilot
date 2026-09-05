"""Track F window construction: CLEAN / CORRUPTION / LEGITIMATE_SHIFT.

CORRUPTION damages the supervised target in a way the input history cannot
explain. LEGITIMATE_SHIFT moves input and target together, so the regime change
is already visible in the lookback. Both are applied to explicit window sets
that never overlap, and both use severities calibrated on a train-only subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np

from contract import SEQ_LEN, PRED_LEN
import data as D

CLASS_SEED = 2026090631
CORRUPT_FRACTION = 0.10
SHIFT_FRACTION = 0.10
SHIFT_CHANNEL_FRACTION = 0.30
SHIFT_INPUT_TAIL = 48
CALIB_FRACTION = 0.10          # train-only calibration subset
WINDOW_STRIDE = 12             # window starts; each window carries exactly one class
SEVERITY_CANDIDATES = [0.5, 1.0, 2.0, 4.0]


def assign_classes(dataset: str, seed: int = CLASS_SEED) -> dict:
    """Partition non-overlapping training windows into the three classes.

    Every window carries exactly one class, so the three classes are mutually
    exclusive. Windows are materialised as independent samples, so a transform
    applied to one window never leaks into another.
    """
    n = len(D.get_dataset(dataset, "train"))
    starts = np.arange(0, n - (SEQ_LEN + PRED_LEN), WINDOW_STRIDE)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(starts))
    n_c = int(round(CORRUPT_FRACTION * len(starts)))
    n_s = int(round(SHIFT_FRACTION * len(starts)))
    idx_c = set(int(i) for i in perm[:n_c])
    idx_s = set(int(i) for i in perm[n_c:n_c + n_s])
    out = {"CORRUPTION": [], "LEGITIMATE_SHIFT": [], "CLEAN": []}
    for k, s in enumerate(starts):
        cls = "CORRUPTION" if k in idx_c else ("LEGITIMATE_SHIFT" if k in idx_s else "CLEAN")
        out[cls].append(int(s))
    return out


def calibration_split(dataset: str, classes: dict, seed: int = CLASS_SEED + 1) -> dict:
    """A disjoint train-only subset used only for severity calibration."""
    rng = np.random.RandomState(seed)
    out = {}
    for cls, starts in classes.items():
        idx = rng.permutation(len(starts))
        k = max(1, int(round(CALIB_FRACTION * len(starts))))
        out[cls] = {"calib": [starts[i] for i in idx[:k]],
                    "study": [starts[i] for i in idx[k:]]}
    return out


def build_arrays(dataset: str):
    ds = D.get_dataset(dataset, "train")
    return np.asarray(ds.data_x, dtype=np.float32), np.asarray(ds.data_y, dtype=np.float32)


def train_iqr(dataset: str) -> np.ndarray:
    x, _ = build_arrays(dataset)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    return (q75 - q25).astype(np.float32)


def make_window(dx, dy, s: int):
    return dx[s:s + SEQ_LEN].copy(), dy[s + SEQ_LEN:s + SEQ_LEN + PRED_LEN].copy()


def apply_corruption(x, y, sev: float, iqr, rng, kind: str):
    """Label error only: the input history is left untouched.

    isolated_spike puts a few large excursions on random target steps;
    block_noise perturbs a contiguous target block. Neither is explainable from
    the lookback, which is what makes them corruption rather than regime change.
    """
    y = y.copy()
    C = y.shape[1]
    chans = rng.choice(C, size=max(1, int(round(0.3 * C))), replace=False)
    if kind == "isolated_spike":
        for c in chans:
            pos = rng.choice(PRED_LEN, size=4, replace=False)
            y[pos, c] += rng.choice([-1.0, 1.0], size=4) * sev * iqr[c]
    else:
        b = int(rng.randint(0, PRED_LEN - 24))
        for c in chans:
            y[b:b + 24, c] += rng.randn(24).astype(np.float32) * sev * iqr[c]
    return x, y


def apply_shift(x, y, sev: float, iqr, rng, kind: str):
    """Coherent regime change: the last 48 input steps and the whole target move
    the same way, so the lookback already carries evidence of the new regime.

    Both kinds ramp the change in over the last 48 input steps, which is what
    makes the change visible to a window statistic that compares the input tail
    with the input head. A change applied as a flat offset to the input tail AND
    the target would cancel out of the tail-to-target comparison and could never
    be recognised as coherent, so neither kind uses that form.

    persistent_level: ramps to d by the forecast origin and holds there.
    coherent_trend:   ramps to d by the origin and keeps going to 2d.
    """
    x, y = x.copy(), y.copy()
    C = x.shape[1]
    # ceil, not round: the spec asks for at least 30% of channels to carry the
    # shift, and rounding down would put a window below the 30% coherence bar by
    # construction (7 * 0.3 -> 2 of 7 = 28.6%).
    chans = rng.choice(C, size=max(1, int(np.ceil(SHIFT_CHANNEL_FRACTION * C))), replace=False)
    ramp_in = np.linspace(0.0, 1.0, SHIFT_INPUT_TAIL, dtype=np.float32)
    for c in chans:
        sign = float(rng.choice([-1.0, 1.0]))
        d = sign * sev * iqr[c]
        x[-SHIFT_INPUT_TAIL:, c] += ramp_in * d
        if kind == "persistent_level":
            y[:, c] += d
        else:  # coherent_trend
            y[:, c] += d * (1.0 + np.linspace(0.0, 1.0, PRED_LEN, dtype=np.float32))
    return x, y


CORRUPTION_KINDS = ["isolated_spike", "block_noise"]
SHIFT_KINDS = ["persistent_level", "coherent_trend"]


def materialise(dataset: str, starts_by_class: dict, sev_corrupt: float,
                sev_shift: float, seed: int):
    """Build the modified training set as explicit (x, y, class, kind) records."""
    dx, dy = build_arrays(dataset)
    iqr = train_iqr(dataset)
    rng = np.random.RandomState(seed)
    X, Y, cls_ids, kinds, starts = [], [], [], [], []
    for cls, ss in starts_by_class.items():
        for s in ss:
            if s + SEQ_LEN + PRED_LEN > len(dx):
                continue
            x, y = make_window(dx, dy, s)
            kind = "none"
            if cls == "CORRUPTION":
                kind = CORRUPTION_KINDS[rng.randint(len(CORRUPTION_KINDS))]
                x, y = apply_corruption(x, y, sev_corrupt, iqr, rng, kind)
            elif cls == "LEGITIMATE_SHIFT":
                kind = SHIFT_KINDS[rng.randint(len(SHIFT_KINDS))]
                x, y = apply_shift(x, y, sev_shift, iqr, rng, kind)
            X.append(x)
            Y.append(y)
            cls_ids.append(cls)
            kinds.append(kind)
            starts.append(s)
    return (np.stack(X), np.stack(Y), np.array(cls_ids), np.array(kinds),
            np.array(starts))
