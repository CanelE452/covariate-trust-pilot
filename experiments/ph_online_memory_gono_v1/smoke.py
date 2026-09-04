"""Construction helpers for the preregistered 200-series smoke run."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


def stratified_smoke_ids(
    descriptors: pd.DataFrame, *, n: int, seed: int
) -> list[str]:
    """Sample as evenly as possible from 4x4 train-only quantile strata."""

    required = ("series_id", "zero_ratio_train", "train_scale")
    if not isinstance(descriptors, pd.DataFrame):
        raise TypeError("descriptors must be a pandas DataFrame")
    missing = [name for name in required if name not in descriptors.columns]
    if missing:
        raise ValueError(f"descriptors are missing columns: {missing}")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral):
        raise TypeError("n must be an integer")
    sample_size = int(n)
    if sample_size < 1 or sample_size > len(descriptors):
        raise ValueError("n must be between one and the population size")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")

    frame = descriptors.loc[:, required].copy()
    if frame["series_id"].isna().any() or frame["series_id"].duplicated().any():
        raise ValueError("series_id values must be present and unique")
    frame["series_id"] = frame["series_id"].astype(str)
    zero_ratio = pd.to_numeric(frame["zero_ratio_train"], errors="raise")
    train_scale = pd.to_numeric(frame["train_scale"], errors="raise")
    if not bool(np.isfinite(zero_ratio).all() and np.isfinite(train_scale).all()):
        raise ValueError("stratification values must be finite")
    if bool((train_scale <= 0.0).any()):
        raise ValueError("train_scale must be positive")

    # Sorting makes ties reproducible independently of loader row order.  Rank
    # tie-breaking retains exactly four quantile bands even for discrete zero
    # ratios, which is common in demand data.
    frame = frame.sort_values("series_id", kind="mergesort").reset_index(drop=True)
    zero_rank = frame["zero_ratio_train"].rank(method="first")
    log_scale_rank = np.log(frame["train_scale"]).rank(method="first")
    frame["zero_band"] = pd.qcut(zero_rank, 4, labels=False)
    frame["scale_band"] = pd.qcut(log_scale_rank, 4, labels=False)

    groups = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in frame.groupby(
            ["zero_band", "scale_band"], sort=True, observed=True
        )
    ]
    if not groups:
        raise ValueError("no nonempty stratum was constructed")

    rng = np.random.default_rng(int(seed))
    shuffled = [rng.permutation(indices) for indices in groups]
    selected: list[int] = []
    positions = np.zeros(len(shuffled), dtype=np.int64)

    # Round-robin allocation is exactly balanced while a stratum has capacity;
    # sparse cells naturally hand their unused quota to the remaining cells.
    while len(selected) < sample_size:
        progressed = False
        for group_index, indices in enumerate(shuffled):
            position = int(positions[group_index])
            if position >= len(indices):
                continue
            selected.append(int(indices[position]))
            positions[group_index] += 1
            progressed = True
            if len(selected) == sample_size:
                break
        if not progressed:
            raise AssertionError("stratified sampler exhausted before reaching n")

    return frame.iloc[selected]["series_id"].tolist()
