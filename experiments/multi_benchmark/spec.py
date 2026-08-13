"""Freeze every choice for all three prospective datasets at once, before scoring.

The reason this is one file and one command is the failure it is designed to
prevent: auditing FreshRetailNet, looking at its numbers, and then choosing the
UCI construction in light of them.  Splits, cleaning, eligibility, the method
and the metrics are written together and hashed, and the scoring commands refuse
to run without that hash.

The proposed method is read out of the existing artifacts rather than restated,
so what gets frozen here is whatever the expert-diversity screen and Gate-v2
actually selected.  Weights are refit per dataset; the method definition is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from ..expert_diversity.oof_experts import CLASSICAL, NEURAL, REFERENCE_PAIR
from ..external_validity_screen import cli, prereg, screen
from ..om_factorization_killtest import prereg as km_prereg
from ..structure_gate import gate_v2 as V2
from .audit import (FULL_BENCHMARK_MIN, HORIZON, LIMITED_BENCHMARK_MIN, LOOKBACK,
                    MIN_POSITIVE_TRAIN, MIN_SPAN, OUT, STRIDE, TEST_ORIGINS)

SPEC = OUT / "benchmark_spec_v1.json"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def proposed_method() -> dict:
    """Read the frozen method out of the artifacts, not out of a description."""
    expert_set = json.loads(
        (screen.OUT.parent / "expert_diversity" / "expert_set_spec.json").read_text())
    gate = json.loads(V2.SPEC.read_text())
    return {
        "expert_pair": expert_set["top_pair"],
        "expert_pair_source": "results/expert_diversity/expert_set_spec.json",
        "expert_selection_formula": expert_set["selection_formula"],
        "use_triple": expert_set["use_triple"],
        "reference_pair": list(REFERENCE_PAIR),
        "gate": gate["selected"],
        "gate_source": "results/structure_gate/gate_v2_spec.json",
        "gate_cv_scheme": gate["cv_scheme"],
        "gate_hyperparameters": {"epochs": V2.EPOCHS, "lr": V2.LR, "seed": V2.SEED,
                                 "mlp_hidden": None},
        "mixing_rule": "y = (1 - g) * expert_a + g * expert_b, one g per forecast origin",
        "transfer": "METHOD TRANSFER, not WEIGHT TRANSFER: weights are refit per dataset",
    }


def dataset_plan(audit: dict) -> dict:
    """Roles, series construction, cleaning and splits for all five datasets."""
    fresh = audit["datasets"]["freshretailnet"]
    stocks = audit["datasets"]["retail_stocks"]
    uci = audit["datasets"]["uci_online_retail_ii"]

    # UCI: rule A unless it is structurally too thin, decided on eligibility counts only.
    candidates = uci["series_construction_candidates"]
    primary = "StockCode x Country"
    uci_construction = (primary
                        if candidates[primary]["series_meeting_both"] >= FULL_BENCHMARK_MIN
                        else "StockCode")
    return {
        "m5": {"role": "DEVELOPMENT_BENCHMARK",
               "reuse_existing_results": True,
               "reason": "test outcomes already used during development"},
        "favorita": {"role": "DEVELOPMENT_BENCHMARK",
                     "reuse_existing_results": True,
                     "reason": "test outcomes already used during development"},
        "freshretailnet": {
            "role": "PRIMARY_EXTERNAL_CONFIRMATION",
            "admission": fresh["admission"],
            "series_key": "store_id x product_id",
            "value_field": "sale_amount",
            "cleaning": "none; sale_amount is already a daily aggregate and is never negative",
            "availability": {
                "field": "stock_hour6_22_cnt",
                "semantics": "number of stockout hours in [6,22); verified against "
                             "hours_stock_status with a match rate of 1.0000",
                "policy": "a day with any stockout hour is marked stockout_mask=1 and "
                          "its zero is treated as censored, never as observed zero demand"},
            "official_eval_used": False,
            "official_eval_reason": "the shipped eval block is 7 days, shorter than the "
                                    "frozen horizon of %d; a temporal holdout is cut from "
                                    "train instead" % HORIZON,
            "split": "temporal, taken from the end of the train file"},
        "retail_stocks": {
            "role": "SECONDARY_EXTERNAL_CONFIRMATION",
            "admission": stocks["admission"],
            "excluded_from_scoring": stocks["admission"] == "INSUFFICIENT_SUPPORT",
            "exclusion_reason": ("only %d series carry both the %d-day span and %d positive "
                                 "train observations the protocol needs; median series life "
                                 "is %.0f days. Excluded on structure, before any forecast."
                                 % (stocks["grid"]["series_meeting_both"], MIN_SPAN,
                                    MIN_POSITIVE_TRAIN,
                                    stocks["grid"]["series_life_days"]["median"])),
            "series_key": "Store x Product No",
            "cleaning": "rows with Is Return == 1 are dropped; returns are never demand"},
        "uci_online_retail_ii": {
            "role": "EXTERNAL_ROBUSTNESS",
            "admission": uci["admission"],
            "series_key": uci_construction,
            "series_key_rule": ("StockCode x Country when it yields at least %d eligible "
                                "series, otherwise StockCode; decided on eligibility counts "
                                "alone" % FULL_BENCHMARK_MIN),
            "eligible_under_chosen_key": candidates[uci_construction]["series_meeting_both"],
            "value_field": "Quantity",
            "cleaning": ("invoices beginning with C are cancellations and are dropped; "
                         "non-positive quantities are dropped; neither is counted as demand"),
            "availability": {"field": None, "status": "AVAILABILITY_UNKNOWN",
                             "policy": "days without an invoice line are filled with zero to "
                                       "form the grid, but are reported as unknown rather "
                                       "than observed zero demand"},
            "split": "temporal, taken from the end of the observed span"},
    }


def cmd_freeze(_args) -> None:
    if SPEC.exists():
        raise SystemExit(f"{SPEC} already frozen; refusing to overwrite")
    audit = json.loads((OUT / "dataset_audit.json").read_text())
    spec = {
        "name": "BENCHMARK_SPEC_V1",
        "frozen_at_utc": _utc(),
        "frozen_before_any_external_test_scoring": True,
        "sequential_tuning_forbidden": ("all three prospective datasets are configured in "
                                        "this single freeze; no dataset's outcome may "
                                        "inform another's configuration"),
        "protocol": {"lookback": LOOKBACK, "horizon": HORIZON, "test_origins": TEST_ORIGINS,
                     "stride": STRIDE, "min_span_days": MIN_SPAN,
                     "min_positive_train": MIN_POSITIVE_TRAIN,
                     "split_rule": "date boundaries only; never a random split",
                     "validation": "one horizon, used for early stopping, global alpha "
                                   "and gate OOF support"},
        "eligibility": {"n_positive_train_min": MIN_POSITIVE_TRAIN,
                        "span_min_days": MIN_SPAN,
                        "train_scale_positive": True},
        "admission_thresholds": {"full": FULL_BENCHMARK_MIN, "limited": LIMITED_BENCHMARK_MIN},
        "datasets": dataset_plan(audit),
        "proposed_method": proposed_method(),
        "expert_pool": {"neural": list(NEURAL), "classical": list(CLASSICAL)},
        "baselines": {
            "neural": ["dlinear_point", "dlinear_hurdle", "dlinear_point_plain",
                       "dlinear_hurdle_ztnb"],
            "classical": list(CLASSICAL),
            "static": ["reference 50:50", "reference global alpha",
                       "diverse-pair equal mixture", "diverse-pair global alpha"],
            "adaptive": ["reference Point/Hurdle Gate-v2", "proposed diverse gate"],
            "oracle": ["hard per-origin", "convex per-origin"],
            "oracle_excluded_from_deployable_ranking": True},
        "training": dict(km_prereg.TRAINING),
        "normalization": "per-series mean over the train window (train_scale), train only",
        "global_alpha": "fitted on validation/OOF only, frozen before test scoring",
        "gate_fitting": "expanding-time OOF inside train; no gate hyperparameter reselection",
        "metrics": {"primary": ["overall RMSE", "overall MAE", "mean per-series RMSE",
                                "median per-series RMSE"],
                    "additional": ["point win %", "hurdle win %", "mean method rank"],
                    "bootstrap": dict(prereg.BOOTSTRAP)},
        "hypotheses": {
            "RQ1": "the proposed adaptive gate beats the best single expert and the best "
                   "static mixture on unseen datasets",
            "RQ2": "the diverse pair raises the convex-oracle ceiling over the reference pair",
            "RQ3": "the gate recovers part of that raised ceiling",
            "RQ4": "results differ between datasets with and without availability signal",
            "RQ5": "where the method sits against classical intermittent baselines"},
        "exclusion_criteria": ("a dataset is excluded only on structural grounds decided "
                               "before scoring; never because its result is unfavourable"),
        "failure_handling": ("a method that fails on a dataset is reported as failed, with "
                             "the reason, and is not removed from the table"),
        "git_commit": cli._git_commit(),
    }
    payload = json.dumps(spec, indent=2, default=str, sort_keys=True)
    spec["spec_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    print(json.dumps({"spec_sha256": spec["spec_sha256"],
                      "frozen_at_utc": spec["frozen_at_utc"],
                      "proposed_pair": spec["proposed_method"]["expert_pair"],
                      "gate": spec["proposed_method"]["gate"],
                      "roles": {k: v["role"] for k, v in spec["datasets"].items()},
                      "excluded": [k for k, v in spec["datasets"].items()
                                   if v.get("excluded_from_scoring")]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("benchmark spec freeze")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
