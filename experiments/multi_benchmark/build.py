"""Turn the admitted external datasets into the shape the existing pipeline reads.

`screen.load_dataset` returns a dict of dense arrays, and everything downstream --
`build_split`, the expert trainers, the classical baselines, the gate -- speaks
that dict.  Building these datasets into the same shape means the frozen method
runs on them unmodified, which is the whole point of a method-transfer test.

Series are aligned on a common calendar window so the split is a date boundary
for every series at once.  A series that does not cover that window is dropped
here, on coverage, not on anything the forecasts say.

FreshRetailNet carries a stockout counter, so a zero on a day with stockout
hours is censored rather than observed; that day's mask is cleared exactly the
way M5's pre-availability days were.  UCI carries no such signal at all, and its
mask stays open with the ambiguity recorded rather than papered over.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..external_validity_screen import cli, screen
from .audit import DATA, MIN_POSITIVE_TRAIN, MIN_SPAN, OUT

PROCESSED = DATA / "processed"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_grid(long: pd.DataFrame, value: str, mask_source: pd.Series | None = None) -> dict:
    """Dense series x date arrays over the window every kept series covers."""
    long = long.copy()
    long["date"] = pd.to_datetime(long["date"])
    calendar = pd.date_range(long["date"].min(), long["date"].max(), freq="D")
    wide = long.pivot_table(index="series_id", columns="date", values=value,
                            aggfunc="sum").reindex(columns=calendar)
    # A day is observed when it falls inside the series' active life, not only when
    # a record exists for it. Transaction files carry a row only on days with a sale,
    # so keying "observed" off record presence would discard every quiet day -- which
    # is precisely the zero the forecast has to predict. The frozen spec already says
    # such days are filled with zero and reported as availability-unknown.
    present = wide.notna()
    positions = np.arange(len(calendar))
    first = pd.Series(np.where(present.any(axis=1),
                               present.values.argmax(axis=1), len(calendar)),
                      index=wide.index)
    last = pd.Series(np.where(present.any(axis=1),
                              len(calendar) - 1 - present.values[:, ::-1].argmax(axis=1), -1),
                     index=wide.index)
    observed = pd.DataFrame(
        (positions[None, :] >= first.to_numpy()[:, None])
        & (positions[None, :] <= last.to_numpy()[:, None]),
        index=wide.index, columns=calendar)
    # A series is kept when it is observed across the trailing MIN_SPAN window,
    # so one date boundary splits every kept series the same way.
    covered = observed.iloc[:, -MIN_SPAN:].mean(axis=1) >= 0.95
    wide = wide.loc[covered]
    observed = observed.loc[covered]
    values = wide.fillna(0.0).to_numpy(np.float32)
    mask = observed.to_numpy()
    result = {"series_id": wide.index.to_numpy().astype(str), "calendar": calendar,
              "y": values, "observed_mask": mask}
    if mask_source is not None:
        stock = long.pivot_table(index="series_id", columns="date", values=mask_source.name,
                                 aggfunc="max").reindex(columns=calendar).loc[covered]
        result["stockout_mask"] = (stock.fillna(0).to_numpy() > 0)
    return result


def build_freshretailnet() -> dict:
    frame = pd.read_parquet(DATA / "freshretailnet" / "train.parquet",
                            columns=["store_id", "product_id", "dt", "sale_amount",
                                     "stock_hour6_22_cnt"])
    frame["series_id"] = frame["store_id"].astype(str) + "_" + frame["product_id"].astype(str)
    frame = frame.rename(columns={"dt": "date"})
    grid = to_grid(frame, "sale_amount", frame["stock_hour6_22_cnt"])
    # Stockout days with no sale are censored demand: drop them from the loss,
    # exactly as M5's pre-availability days were dropped.
    censored = grid["stockout_mask"] & (grid["y"] <= 0)
    grid["observed_mask"] = grid["observed_mask"] & ~censored
    grid["censored_share"] = float(censored.mean())
    return grid


def build_uci(construction: str) -> dict:
    sheets = pd.read_excel(DATA / "online_retail_ii" / "online_retail_II.xlsx", sheet_name=None)
    frame = pd.concat(sheets.values(), ignore_index=True)
    frame.columns = [c.strip() for c in frame.columns]
    cancelled = frame["Invoice"].astype(str).str.startswith("C")
    frame = frame[~cancelled & (frame["Quantity"] > 0)].copy()
    frame["date"] = pd.to_datetime(frame["InvoiceDate"]).dt.normalize()
    keys = ["StockCode", "Country"] if "Country" in construction else ["StockCode"]
    frame["series_id"] = frame[keys].astype(str).agg("|".join, axis=1)
    return to_grid(frame[["series_id", "date", "Quantity"]], "Quantity")


def finalise(name: str, grid: dict) -> dict:
    """Apply eligibility, then write the arrays the loader will read back."""
    train_end = grid["y"].shape[1] - (MIN_SPAN - 96 - 28)   # keep the frozen tail geometry
    positive = ((grid["y"] > 0) & grid["observed_mask"])[:, :train_end].sum(axis=1)
    keep = positive >= MIN_POSITIVE_TRAIN
    for key in ("y", "observed_mask", "stockout_mask"):
        if key in grid:
            grid[key] = grid[key][keep]
    grid["series_id"] = grid["series_id"][keep]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / f"{name}_grid.npz",
                        y=grid["y"], observed_mask=grid["observed_mask"],
                        stockout_mask=grid.get("stockout_mask",
                                               np.ones_like(grid["observed_mask"])),
                        series_id=grid["series_id"],
                        calendar=np.array([str(d.date()) for d in grid["calendar"]]))
    return {"n_series": int(len(grid["series_id"])),
            "n_days": int(grid["y"].shape[1]),
            "calendar": [str(grid["calendar"][0].date()), str(grid["calendar"][-1].date())],
            "eligible_rule": f"n_positive over the train window >= {MIN_POSITIVE_TRAIN}",
            "observed_share": float(grid["observed_mask"].mean()),
            "zero_share_observed": float(
                ((grid["y"] <= 0) & grid["observed_mask"]).sum() / grid["observed_mask"].sum()),
            "censored_share": grid.get("censored_share"),
            "has_stockout_signal": "stockout_mask" in grid}


def cmd_run(_args) -> None:
    spec = json.loads((OUT / "benchmark_spec_v1.json").read_text())
    report = {"built_at_utc": _utc(), "spec_sha256": spec["spec_sha256"],
              "test_outcome_computed": False, "datasets": {}}
    print("building freshretailnet ...", flush=True)
    report["datasets"]["freshretailnet"] = finalise("freshretailnet", build_freshretailnet())
    print(json.dumps(report["datasets"]["freshretailnet"], indent=2))
    construction = spec["datasets"]["uci_online_retail_ii"]["series_key"]
    print(f"building uci ({construction}) ...", flush=True)
    report["datasets"]["uci"] = finalise("uci", build_uci(construction))
    report["datasets"]["uci"]["series_key"] = construction
    print(json.dumps(report["datasets"]["uci"], indent=2))
    report["git_commit"] = cli._git_commit()
    (OUT / "grid_manifest.json").write_text(json.dumps(report, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser("build external benchmark grids")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("run").set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
