"""Rebuild the Favorita full pool from the raw competition file.

``prepare_favorita.py`` was never migrated with the SCREEN, so the pool it drew
the Stage A 1200 from does not exist in this repository — which is what forced
the earlier Favorita transfer to run inside the Stage A sample and come back
LOW_SUPPORT.  This reconstructs the pool from ``data/train.csv`` instead of
guessing at the preprocessing, and every step that could be guessed is checked
against something already recorded.

The target transform was recovered by testing candidates against the 1200
series the repository already holds: clip at zero, round, and treat a missing
(item, store, date) row as a zero.  All 1200 x 1688 = 2,025,600 cells match, so
the rule is not an assumption.  ``verify_target_transform`` re-runs that check.

The eligibility count is the second check.  The derivation note records
"eligible 56,918 -> 1,200" for Favorita; if this rebuild lands anywhere else,
the selection rule was not reproduced and nothing downstream should be trusted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import prereg, screen

OUT = screen.OUT / "favorita_full_pool"
POOL = screen.REPO / "data" / "processed" / "favorita_full_pool.parquet"
RAW = screen.REPO / "data" / "train.csv"

#: From docs/m5_favorita_data_derivation.md section 4. Not recomputed.
FIRST_DAY_MAX = 90
MIN_POSITIVE_TRAIN = 2
MIN_TRAIN_LENGTH = 184
DOCUMENTED_ELIGIBLE = 56_918
CHUNK = 5_000_000


def transform(unit_sales: np.ndarray) -> np.ndarray:
    """The recovered rule: negatives (returns) become zero, then round."""
    return np.rint(np.clip(unit_sales, 0.0, None))


def scan_raw() -> pd.DataFrame:
    """One pass over train.csv, accumulating a dense per-series day vector.

    train.csv omits zero-sales rows entirely, so the dense grid has to be
    materialised rather than pivoted; a missing row is a real zero, not a gap.
    """
    reference = pd.read_parquet(screen.DATASETS["favorita"]["parquet"],
                                columns=["timestamp"])
    start = reference["timestamp"].min()
    end = reference["timestamp"].max()
    length = (end - start).days + 1
    if length != prereg.SPLITS["favorita"]["length"]:
        raise screen.ScreenFailure(
            f"calendar length {length} != frozen {prereg.SPLITS['favorita']['length']}")

    totals: dict[int, np.ndarray] = {}
    first_seen: dict[int, int] = {}
    rows = 0
    reader = pd.read_csv(RAW, usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
                         dtype={"store_nbr": "int16", "item_nbr": "int32",
                                "unit_sales": "float32"},
                         parse_dates=["date"], chunksize=CHUNK)
    for i, chunk in enumerate(reader):
        day = (chunk["date"] - start).dt.days.to_numpy()
        inside = (day >= 0) & (day < length)
        chunk, day = chunk[inside], day[inside]
        key = (chunk["item_nbr"].to_numpy(np.int64) << 8) | chunk["store_nbr"].to_numpy(np.int64)
        value = transform(chunk["unit_sales"].to_numpy(np.float64))
        order = np.argsort(key, kind="stable")
        key, day, value = key[order], day[order], value[order]
        edges = np.flatnonzero(np.diff(key)) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, key.size]):
            k = int(key[lo])
            vec = totals.get(k)
            if vec is None:
                vec = np.zeros(length, dtype=np.float32)
                totals[k] = vec
            np.add.at(vec, day[lo:hi], value[lo:hi])
            d = int(day[lo:hi].min())
            if k not in first_seen or d < first_seen[k]:
                first_seen[k] = d
        rows += len(chunk)
        if i % 5 == 0:
            print(f"  chunk {i}: {rows:,} rows, {len(totals):,} series", flush=True)

    keys = np.fromiter(totals, dtype=np.int64, count=len(totals))
    keys.sort()
    frame = pd.DataFrame({
        "series_id": [f"{k >> 8}_{k & 0xFF}" for k in keys],
        "first_day": [first_seen[int(k)] for k in keys]})
    matrix = np.stack([totals[int(k)] for k in keys])
    return frame, matrix, start


def verify_target_transform(frame: pd.DataFrame, matrix: np.ndarray) -> dict:
    """The rebuilt values must equal the stored 1200 series exactly."""
    stored = pd.read_parquet(screen.DATASETS["favorita"]["parquet"])
    wide = stored.pivot_table(index="series_id", columns="timestamp",
                              values="target").sort_index()
    index = pd.Index(frame["series_id"]).get_indexer(wide.index)
    if (index < 0).any():
        missing = wide.index[index < 0].tolist()[:5]
        raise screen.ScreenFailure(
            f"TARGET_TRANSFORM_NOT_REPRODUCED: {int((index < 0).sum())} stored series "
            f"absent from the rebuild, e.g. {missing}")
    rebuilt = matrix[index]
    reference = wide.to_numpy(np.float64)
    mismatches = int((rebuilt != reference).sum())
    return {"n_series_checked": int(len(wide)), "n_cells": int(reference.size),
            "n_mismatched_cells": mismatches, "reproduced": mismatches == 0,
            "rule": "clip(unit_sales, 0, None) -> rint; absent row = 0"}


def eligibility(frame: pd.DataFrame, matrix: np.ndarray) -> dict:
    """first_day <= 90, then the same SBC eligibility M5 used."""
    cfg = screen.config_for("favorita")
    train = matrix[:, :cfg.train_end]
    n_positive = (train > 0).sum(axis=1)
    full_life = frame["first_day"].to_numpy() <= FIRST_DAY_MAX
    long_enough = train.shape[1] >= MIN_TRAIN_LENGTH
    enough_events = n_positive >= MIN_POSITIVE_TRAIN
    eligible = full_life & enough_events & bool(long_enough)
    return {"n_series_in_raw": int(len(frame)),
            "n_full_life": int(full_life.sum()),
            "n_eligible": int(eligible.sum()),
            "documented_eligible": DOCUMENTED_ELIGIBLE,
            "matches_documented": int(eligible.sum()) == DOCUMENTED_ELIGIBLE,
            "first_day_max": FIRST_DAY_MAX,
            "min_positive_train": MIN_POSITIVE_TRAIN,
            "min_train_length": MIN_TRAIN_LENGTH,
            "mask": eligible}


def cmd_build(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("scanning train.csv ...", flush=True)
    frame, matrix, start = scan_raw()
    print(f"raw series: {len(frame):,}", flush=True)

    target_check = verify_target_transform(frame, matrix)
    print(json.dumps(target_check, indent=2))
    if not target_check["reproduced"]:
        raise screen.ScreenFailure(
            f"TARGET_TRANSFORM_NOT_REPRODUCED: {target_check['n_mismatched_cells']} cells differ")

    check = eligibility(frame, matrix)
    mask = check.pop("mask")
    print(json.dumps(check, indent=2))
    report = {"built_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(RAW.relative_to(screen.REPO)),
              "target_transform_check": target_check,
              "eligibility_check": check,
              "calendar_start": str(start.date()),
              "length": int(matrix.shape[1])}
    (OUT / "full_pool_manifest.json").write_text(json.dumps(report, indent=2, default=str))

    if not check["matches_documented"]:
        raise screen.ScreenFailure(
            f"ELIGIBILITY_NOT_REPRODUCED: {check['n_eligible']} != "
            f"{DOCUMENTED_ELIGIBLE} recorded in the derivation note")

    keep = np.flatnonzero(mask)
    POOL.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix[keep].astype(np.int32),
                 index=pd.Index(frame["series_id"].to_numpy()[keep], name="series_id"),
                 columns=pd.date_range(start, periods=matrix.shape[1], freq="D")
                 ).to_parquet(POOL)
    print(f"wrote {POOL.relative_to(screen.REPO)}  shape={matrix[keep].shape}")


def main() -> None:
    parser = argparse.ArgumentParser("Favorita full pool rebuild")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
