"""Outcome-blind audit of the three prospective external datasets.

Nothing here forecasts anything.  The point is to decide, before any test
target is touched, whether each dataset can carry the protocol M5 and Favorita
were run under -- lookback 96, horizon 28, a validation block and a strided test
block -- and whether its zeros mean "no demand" or "not on the shelf".

The three datasets are audited together and the decision is taken for all of
them at once.  Auditing one, seeing its result, and then adjusting the next is
exactly the sequential tuning the design forbids.

Where a field's meaning is not certain from the file alone it is recorded as an
ambiguity rather than resolved by assumption; the mapping table carries the
evidence for every semantic role it assigns.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..external_validity_screen import cli, prereg, screen

OUT = screen.OUT.parent / "multi_benchmark"
DATA = screen.REPO / "data"

#: The protocol M5 and Favorita ran under. A dataset that cannot supply this
#: much history is reported as unsupported rather than given a shorter window.
LOOKBACK = prereg.SPLITS["m5"]["lookback"]
HORIZON = prereg.SPLITS["m5"]["horizon"]
TEST_ORIGINS = 3
STRIDE = prereg.SPLITS["m5"]["test_origin_stride"]
#: train + validation(1 horizon) + test block(stride * (origins-1) + horizon)
MIN_SPAN = LOOKBACK + HORIZON + HORIZON + STRIDE * (TEST_ORIGINS - 1) + HORIZON
MIN_POSITIVE_TRAIN = prereg.ELIGIBILITY["primary_threshold"]

#: Operational admission thresholds, fixed before any dataset was scored.
FULL_BENCHMARK_MIN = 1000
LIMITED_BENCHMARK_MIN = 300


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def describe_grid(long: pd.DataFrame, key: str, date: str, value: str,
                  extra_masks: dict | None = None) -> dict:
    """Dense daily grid statistics without materialising every series array."""
    frame = long[[key, date, value]].copy()
    frame[date] = pd.to_datetime(frame[date])
    span_start, span_end = frame[date].min(), frame[date].max()
    calendar_days = (span_end - span_start).days + 1
    grouped = frame.groupby(key)
    first, last = grouped[date].min(), grouped[date].max()
    observed = grouped.size()
    positive = frame[frame[value] > 0].groupby(key).size().reindex(first.index).fillna(0)
    life = (last - first).dt.days + 1
    return {"n_series_raw": int(len(first)),
            "calendar_days": int(calendar_days),
            "span": [str(span_start.date()), str(span_end.date())],
            "observed_rows_per_series": {"median": float(observed.median()),
                                         "p10": float(observed.quantile(.1)),
                                         "p90": float(observed.quantile(.9))},
            "series_life_days": {"median": float(life.median()),
                                 "p10": float(life.quantile(.1)),
                                 "p90": float(life.quantile(.9)),
                                 "max": float(life.max())},
            "n_positive_per_series": {"median": float(positive.median()),
                                      "p10": float(positive.quantile(.1)),
                                      "p90": float(positive.quantile(.9))},
            "series_with_full_span": int((life >= calendar_days * 0.98).sum()),
            "series_meeting_min_span": int((life >= MIN_SPAN).sum()),
            "series_meeting_min_positive": int((positive >= MIN_POSITIVE_TRAIN).sum()),
            "series_meeting_both": int(((life >= MIN_SPAN)
                                        & (positive >= MIN_POSITIVE_TRAIN)).sum()),
            "extra": extra_masks or {}}


def audit_freshretailnet() -> dict:
    train = pd.read_parquet(DATA / "freshretailnet" / "train.parquet",
                            columns=["store_id", "product_id", "dt", "sale_amount",
                                     "stock_hour6_22_cnt", "discount", "holiday_flag"])
    evaluation = pd.read_parquet(DATA / "freshretailnet" / "eval.parquet",
                                 columns=["store_id", "product_id", "dt", "sale_amount"])
    train["series_id"] = train["store_id"].astype(str) + "_" + train["product_id"].astype(str)
    evaluation["series_id"] = (evaluation["store_id"].astype(str) + "_"
                               + evaluation["product_id"].astype(str))
    grid = describe_grid(train, "series_id", "dt", "sale_amount")
    duplicates = int(train.duplicated(["series_id", "dt"]).sum())
    stock = train["stock_hour6_22_cnt"]
    return {
        "role": "PRIMARY_EXTERNAL_CONFIRMATION",
        "series_key": "store_id x product_id",
        "series_key_evidence": "both identifiers present as integer columns",
        "n_rows": int(len(train)),
        "eval_rows": int(len(evaluation)),
        "eval_labelled": bool(evaluation["sale_amount"].notna().all()),
        "eval_span": [str(pd.to_datetime(evaluation['dt']).min().date()),
                      str(pd.to_datetime(evaluation['dt']).max().date())],
        "eval_horizon_days": int(pd.to_datetime(evaluation["dt"]).nunique()),
        "duplicate_series_date": duplicates,
        "value_field": "sale_amount",
        "value_is_integer": bool((train["sale_amount"] % 1 == 0).all()),
        "value_min": float(train["sale_amount"].min()),
        "value_max": float(train["sale_amount"].max()),
        "negative_values": int((train["sale_amount"] < 0).sum()),
        "zero_rows_share": float((train["sale_amount"] <= 0).mean()),
        "availability_field": "stock_hour6_22_cnt",
        "availability_evidence": (
            "verified against hours_stock_status: stock_hour6_22_cnt equals the number "
            "of hours in [6,22) with status 1 (match rate 1.0000), and status-1 hours "
            "carry 12x lower sales than status-0 hours (0.0085 vs 0.1022), so 1 marks "
            "a stockout hour and the counter is stockout hours, not in-stock hours"),
        "stockout_hours_mean": float(stock.mean()),
        "full_day_stockout_row_share": float((stock >= 16).mean()),
        "no_stockout_row_share": float((stock == 0).mean()),
        "grid": grid,
        "ambiguities": [
            "sale_amount is not integer (max %.1f); it is a weight or value measure, "
            "so count-based intermittency descriptors are approximations"
            % float(train["sale_amount"].max()),
            "the eval block is 7 days, shorter than the frozen horizon of %d" % HORIZON,
            "a zero sale on a day with stockout hours is censored demand, not observed "
            "zero demand; the two are kept apart by the mask rather than merged"],
    }


def audit_retail_stocks() -> dict:
    sales = pd.read_csv(DATA / "retail_transactions_stocks" / "retail_sales_ml_apl.csv",
                        usecols=["Transaction Date", "Is Return", "Product No", "Store",
                                 "Qty Sold", "Sales Type"])
    inventory = pd.read_csv(DATA / "retail_transactions_stocks" / "retail_inventory_ml_apl.csv",
                            usecols=["Start Date", "End Date", "Stock Status", "Product No"],
                            nrows=2_000_000)
    sales["date"] = pd.to_datetime(sales["Transaction Date"])
    sales["series_id"] = sales["Store"].astype(str) + "|" + sales["Product No"].astype(str)
    returns = sales[sales["Is Return"] == 1]
    demand = sales[sales["Is Return"] == 0]
    daily = demand.groupby(["series_id", "date"], as_index=False)["Qty Sold"].sum()
    grid = describe_grid(daily, "series_id", "date", "Qty Sold")
    inventory_dates = pd.to_datetime(inventory["Start Date"], errors="coerce")
    store_in_inventory = "Store" in pd.read_csv(
        DATA / "retail_transactions_stocks" / "retail_inventory_ml_apl.csv", nrows=1).columns
    return {
        "role": "SECONDARY_EXTERNAL_CONFIRMATION",
        "series_key": "Store x Product No",
        "series_key_evidence": "both columns present in the sales file",
        "n_rows": int(len(sales)),
        "n_stores": int(sales["Store"].nunique()),
        "n_products": int(sales["Product No"].nunique()),
        "span": [str(sales["date"].min().date()), str(sales["date"].max().date())],
        "calendar_days": int((sales["date"].max() - sales["date"].min()).days + 1),
        "return_rows": int(len(returns)),
        "return_share": float(len(returns) / len(sales)),
        "negative_qty_rows": int((sales["Qty Sold"] < 0).sum()),
        "value_field": "Qty Sold, returns (Is Return == 1) excluded from demand",
        "grid": grid,
        "inventory_rows_sampled": int(len(inventory)),
        "inventory_has_store": bool(store_in_inventory),
        "inventory_span": [str(inventory_dates.min().date()), str(inventory_dates.max().date())],
        "inventory_stock_status_values": sorted(
            inventory["Stock Status"].dropna().unique().tolist())[:10],
        "ambiguities": [
            "inventory rows are Start Date / End Date intervals, not daily snapshots, "
            "so an on-hand quantity per day is not directly available",
            "inventory file has no Store column" if not store_in_inventory
            else "inventory carries Store, join is possible",
            "product descriptions are templated (Scholar Footwear, Femme Footwear ...), "
            "which is consistent with a synthetic or anonymised catalogue"],
    }


def audit_uci() -> dict:
    frame = pd.read_excel(DATA / "online_retail_ii" / "online_retail_II.xlsx",
                          sheet_name=None)
    sheets = {name: block for name, block in frame.items()}
    combined = pd.concat(sheets.values(), ignore_index=True)
    combined.columns = [c.strip() for c in combined.columns]
    invoice = combined["Invoice"].astype(str)
    cancellations = invoice.str.startswith("C")
    combined["date"] = pd.to_datetime(combined["InvoiceDate"]).dt.normalize()
    clean = combined[~cancellations & (combined["Quantity"] > 0)].copy()
    variants = {}
    for label, key in (("StockCode x Country", ["StockCode", "Country"]),
                       ("StockCode", ["StockCode"])):
        block = clean.copy()
        block["series_id"] = block[key].astype(str).agg("|".join, axis=1)
        daily = block.groupby(["series_id", "date"], as_index=False)["Quantity"].sum()
        variants[label] = describe_grid(daily, "series_id", "date", "Quantity")
    return {
        "role": "EXTERNAL_ROBUSTNESS",
        "sheets": list(sheets),
        "n_rows": int(len(combined)),
        "span": [str(combined["date"].min().date()), str(combined["date"].max().date())],
        "calendar_days": int((combined["date"].max() - combined["date"].min()).days + 1),
        "cancellation_rows": int(cancellations.sum()),
        "negative_quantity_rows": int((combined["Quantity"] < 0).sum()),
        "zero_quantity_rows": int((combined["Quantity"] == 0).sum()),
        "n_stockcodes": int(combined["StockCode"].nunique()),
        "n_countries": int(combined["Country"].nunique()),
        "store_identifier": None,
        "availability_field": None,
        "availability_status": "AVAILABILITY_UNKNOWN",
        "series_construction_candidates": variants,
        "ambiguities": [
            "transaction-only: a date with no invoice line cannot be distinguished "
            "between zero demand and the product not being offered",
            "cancellations are identified by an Invoice starting with C and are removed "
            "along with non-positive quantities; they are never counted as demand",
            "there is no store or location field, so no per-store series exist"],
    }


def admission(eligible: int) -> str:
    if eligible >= FULL_BENCHMARK_MIN:
        return "FULL_BENCHMARK_ELIGIBLE"
    if eligible >= LIMITED_BENCHMARK_MIN:
        return "LIMITED_BENCHMARK"
    return "INSUFFICIENT_SUPPORT"


def cmd_run(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"analysis": "outcome-blind structural audit of the prospective datasets",
              "test_outcome_computed": False, "audited_at_utc": _utc(),
              "protocol": {"lookback": LOOKBACK, "horizon": HORIZON,
                           "test_origins": TEST_ORIGINS, "stride": STRIDE,
                           "min_span_days": MIN_SPAN,
                           "min_positive_train": MIN_POSITIVE_TRAIN},
              "admission_thresholds": {"full": FULL_BENCHMARK_MIN,
                                       "limited": LIMITED_BENCHMARK_MIN},
              "datasets": {}}
    for name, fn in (("freshretailnet", audit_freshretailnet),
                     ("retail_stocks", audit_retail_stocks),
                     ("uci_online_retail_ii", audit_uci)):
        print(f"auditing {name} ...", flush=True)
        block = fn()
        if "grid" in block:
            block["admission"] = admission(block["grid"]["series_meeting_both"])
            block["standard_protocol_supported"] = bool(
                block["grid"]["series_meeting_min_span"] > 0)
        else:
            best = max(v["series_meeting_both"]
                       for v in block["series_construction_candidates"].values())
            block["admission"] = admission(best)
            block["standard_protocol_supported"] = bool(
                max(v["series_meeting_min_span"]
                    for v in block["series_construction_candidates"].values()) > 0)
        report["datasets"][name] = block
        print(f"  {name}: {block['admission']}  "
              f"standard_protocol={block['standard_protocol_supported']}", flush=True)
    report["git_commit"] = cli._git_commit()
    (OUT / "dataset_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print("wrote dataset_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser("multi-dataset structural audit")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("run").set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
