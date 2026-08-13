"""Independent M5 confirmation of H2, on series Stage A never saw.

Stage A tested H2 inside the 1200-series stratified sample and got n = 39
candidates, which left the interval far too wide to decide anything.  The
post-hoc diagnostic then sized the untouched full pool at 714 candidates and
5,142 controls and called the expansion HIGH_VALUE.  This module runs that
expansion as a confirmation rather than a rescue: the cutoffs come from the
frozen Stage A record, the Stage A series are removed outright, and the
specification is written before a single prediction exists.

The H2 condition is the synthetic study's Point-favourable corner —
high ADI, weak occurrence signal, persistent magnitude — where a factorised
hurdle is expected to lose to a plain point forecast.

    population   descriptors -> eligibility -> Stage A removal -> cutoffs
                 -> 1:2 matching -> SMD gate -> frozen spec
    train        one Point and one Hurdle on the matched population
    analyse      delta_relative, candidate vs control, series bootstrap

`population` refuses to overwrite a spec and `train` refuses to run without
one, so the freeze can never end up downstream of a prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from ..unified_temporal_27_v3.training import train_scale
from . import cli, posthoc, prereg, screen

OUT = screen.OUT / "h2_confirmatory"
SPEC = OUT / "confirmatory_spec.json"

#: Matching uses train-only structure. rho_magnitude is deliberately absent:
#: it is the variable that defines candidate versus control, so balancing on
#: it would erase the contrast under test.
MATCH_VARIABLES = ("log_ADI", "CV2_positive", "log_train_scale", "abs_rho_interval")
MAX_CONTROLS_PER_CANDIDATE = 2
SMD_TARGET, SMD_LIMIT = 0.10, 0.20
MIN_CANDIDATES = 100


def _sha_of(values) -> str:
    joined = "\n".join(sorted(str(v) for v in values))
    return hashlib.sha256(joined.encode()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------- population --


def m5_full() -> dict:
    """The untouched 30,490-series M5 file in load_dataset's shape.

    load_dataset reads data/processed/series.parquet, which holds only the
    1200 series Stage A drew, so it cannot serve an independent sample.
    """
    wide = pd.read_csv(screen.REPO / "data" / "sales_train_evaluation.csv")
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    y = wide[day_cols].to_numpy(dtype=np.float32)
    available_from = screen.m5_availability(wide[["item_id", "store_id"]])
    first_positive = np.array([int(np.argmax(r > 0)) if (r > 0).any() else len(r)
                               for r in y])
    return {"name": "m5", "y": y, "z": (y > 0).astype(np.float32),
            "series_id": wide["id"].to_numpy().astype(str),
            "available_from": np.asarray(available_from, dtype=int),
            "first_positive": first_positive}


def frozen_cutoffs_from_artifact() -> dict:
    """Read the cutoffs Stage A used; never recompute them on the full pool."""
    source = posthoc.OUT / "posthoc_diagnostic.json"
    stored = json.loads(source.read_text())["datasets"]["m5"]["frozen_cutoffs_recovered"]
    missing = [k for k in ("HIGH_ADI_min", "LOW_OCC_max", "MAG_PERSISTENT_min")
               if k not in stored]
    if missing:
        raise screen.ScreenFailure(f"FROZEN_H2_CUTOFF_NOT_REPRODUCIBLE: {missing}")
    return {k: float(stored[k]) for k in
            ("HIGH_ADI_min", "LOW_OCC_max", "MAG_PERSISTENT_min")} | {
        "source": str(source.relative_to(screen.REPO)),
        "recomputed_on_full_pool": False}


def descriptor_frame(data: dict, cfg) -> pd.DataFrame:
    """Train-only descriptors for every series, availability-aware like Stage A."""
    rows = [screen.describe_series(data["y"][i], int(data["available_from"][i]),
                                   cfg.train_end)
            for i in range(len(data["series_id"]))]
    table = pd.DataFrame(rows)
    table["series_id"] = data["series_id"]
    table["train_scale"] = train_scale({"y": data["y"], "z": data["z"]}, cfg)
    return table


def build_population(table: pd.DataFrame, stage_a_ids: set, cutoffs: dict) -> dict:
    threshold = prereg.ELIGIBILITY["primary_threshold"]
    eligible = table[table["n_positive_train"] >= threshold].copy()
    independent = eligible[~eligible["series_id"].isin(stage_a_ids)].copy()

    adi = independent["ADI_train"].to_numpy(float)
    occ = independent["rho_interval_abs_train"].to_numpy(float)
    mag = independent["rho_magnitude_train"].to_numpy(float)
    high = adi >= cutoffs["HIGH_ADI_min"]
    low_occ = occ <= cutoffs["LOW_OCC_max"]
    persistent = mag >= cutoffs["MAG_PERSISTENT_min"]

    independent["group"] = "unused"
    independent.loc[high & low_occ & persistent, "group"] = "candidate"
    independent.loc[high & low_occ & ~persistent, "group"] = "control_pool"

    overlap = set(independent["series_id"]) & stage_a_ids
    return {"threshold": threshold,
            "n_total_source": int(len(table)),
            "n_descriptor_eligible": int(len(eligible)),
            "n_after_stage_a_removal": int(len(independent)),
            "n_stage_a_removed": int(len(eligible) - len(independent)),
            "n_candidate": int((independent["group"] == "candidate").sum()),
            "n_control_pool": int((independent["group"] == "control_pool").sum()),
            "stage_a_overlap": len(overlap),
            "n_control_pool_nan_rho_magnitude": int(
                ((independent["group"] == "control_pool")
                 & ~np.isfinite(independent["rho_magnitude_train"])).sum()),
            "frame": independent}


# --------------------------------------------------------------- matching --


def matching_design(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "series_id": frame["series_id"].to_numpy(),
        "group": frame["group"].to_numpy(),
        "log_ADI": np.log(frame["ADI_train"].to_numpy(float)),
        "CV2_positive": frame["CV2_positive_train"].to_numpy(float),
        "log_train_scale": np.log(frame["train_scale"].to_numpy(float)),
        "abs_rho_interval": frame["rho_interval_abs_train"].to_numpy(float),
        "rho_magnitude": frame["rho_magnitude_train"].to_numpy(float)})
    return out[np.isfinite(out[list(MATCH_VARIABLES)].to_numpy(float)).all(axis=1)]


def smd(candidate: np.ndarray, control: np.ndarray) -> float:
    """Standardised mean difference with the pooled within-group variance."""
    pooled = np.sqrt((candidate.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((candidate.mean() - control.mean()) / pooled)


def match(design: pd.DataFrame, seed: int) -> dict:
    """Greedy 1:k nearest neighbour on standardised train-only covariates.

    Without replacement, so no control is counted twice, and in a seeded
    random candidate order because greedy matching is order dependent.
    """
    cand = design[design["group"] == "candidate"].reset_index(drop=True)
    ctrl = design[design["group"] == "control_pool"].reset_index(drop=True)
    columns = list(MATCH_VARIABLES)
    everything = design[columns].to_numpy(float)
    centre, spread = everything.mean(axis=0), everything.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    zc = (cand[columns].to_numpy(float) - centre) / spread
    zk = (ctrl[columns].to_numpy(float) - centre) / spread

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(zc))
    taken = np.zeros(len(zk), dtype=bool)
    picks: dict[int, list[int]] = {}
    for ci in order:
        distance = np.sqrt(((zk - zc[ci]) ** 2).sum(axis=1))
        distance[taken] = np.inf
        chosen = []
        for _ in range(MAX_CONTROLS_PER_CANDIDATE):
            j = int(np.argmin(distance))
            if not np.isfinite(distance[j]):
                break
            chosen.append(j)
            taken[j] = True
            distance[j] = np.inf
        picks[int(ci)] = chosen

    matched_ctrl = ctrl.iloc[sorted({j for v in picks.values() for j in v})]
    balance = {v: smd(cand[v].to_numpy(float), matched_ctrl[v].to_numpy(float))
               for v in MATCH_VARIABLES}
    worst = max(abs(v) for v in balance.values())
    status = ("SMD_PASS" if worst <= SMD_TARGET else
              "SMD_WARN" if worst <= SMD_LIMIT else "SMD_FAIL")
    return {"candidate_ids": cand["series_id"].tolist(),
            "control_ids": matched_ctrl["series_id"].tolist(),
            "n_candidate": int(len(cand)), "n_control_matched": int(len(matched_ctrl)),
            "n_control_pool": int(len(ctrl)),
            "ratio": float(len(matched_ctrl) / max(len(cand), 1)),
            "balance_smd": balance, "worst_abs_smd": float(worst),
            "status": status, "seed": seed,
            "method": (f"greedy 1:{MAX_CONTROLS_PER_CANDIDATE} nearest neighbour, "
                       "standardised Euclidean, without replacement"),
            "variables": list(MATCH_VARIABLES),
            "excluded_variable": "rho_magnitude_train defines the contrast"}


# ------------------------------------------------------------------ freeze --


def cmd_population(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if SPEC.exists():
        raise SystemExit(f"{SPEC} already frozen; refusing to overwrite")

    cfg = screen.config_for("m5")
    cutoffs = frozen_cutoffs_from_artifact()
    stage_a = set(pd.read_csv(screen.OUT / "per_series_metrics.csv")
                  .query("dataset == 'm5'")["series_id"])

    print("loading full M5 ...")
    data = m5_full()
    print(f"describing {len(data['series_id'])} series (train split only) ...")
    table = descriptor_frame(data, cfg)
    population = build_population(table, stage_a, cutoffs)

    if population["stage_a_overlap"] != 0:
        raise screen.ScreenFailure(
            f"STAGE_A_OVERLAP={population['stage_a_overlap']}; must be 0")
    if population["n_candidate"] < MIN_CANDIDATES:
        raise screen.ScreenFailure(
            f"CANDIDATE_POOL_TOO_SMALL: {population['n_candidate']} < {MIN_CANDIDATES}")

    design = matching_design(population["frame"])
    matched = match(design, prereg.BOOTSTRAP["seed"])
    if matched["status"] == "SMD_FAIL":
        (OUT / "matching_failed.json").write_text(json.dumps(matched, indent=2, default=str))
        raise screen.ScreenFailure(
            f"MATCHING_SMD_FAIL: worst |SMD| = {matched['worst_abs_smd']:.4f} > {SMD_LIMIT}")

    if set(matched["candidate_ids"]) & stage_a or set(matched["control_ids"]) & stage_a:
        raise screen.ScreenFailure("STAGE_A_OVERLAP in the matched population")

    spec = {
        "study": "external-validity H2 confirmatory, M5 only, independent series",
        "frozen_at_utc": _utc(),
        "frozen_before_any_prediction": True,
        "frozen_h2_cutoffs": cutoffs,
        "population": {k: v for k, v in population.items() if k != "frame"},
        "stage_a_excluded_id_sha256": _sha_of(stage_a),
        "n_stage_a_excluded": len(stage_a),
        "candidate_id_sha256": _sha_of(matched["candidate_ids"]),
        "control_id_sha256": _sha_of(matched["control_ids"]),
        "matching": {k: v for k, v in matched.items()
                     if k not in ("candidate_ids", "control_ids")},
        "models": {"point": prereg.MODELS["point"], "hurdle": prereg.MODELS["hurdle"],
                   "model_seed": prereg.MODELS["canonical_model_seed"],
                   "source": "unchanged from Stage A"},
        "training": dict(km_prereg.TRAINING),
        "split": dict(prereg.SPLITS["m5"]),
        "training_population": ("one Point and one Hurdle trained on candidate and "
                                "matched control together, exactly as Stage A trained "
                                "one model over its whole 1200-series sample"),
        "primary_metric": ("delta_relative = (RMSE_Point - RMSE_Hurdle) / RMSE_Point "
                           "per series on realized y; negative favours Point"),
        "primary_contrast": "mean(delta_relative | candidate) - mean(delta_relative | control)",
        "expected_sign": "negative",
        "bootstrap": dict(prereg.BOOTSTRAP),
        "verdict_rule": {
            "STRONG_CONFIRMATION": ("H2_effect < 0 and its CI excludes 0 and candidate "
                                    "Point win rate > control and candidate mean "
                                    "delta_relative < 0"),
            "PARTIAL_CONFIRMATION": "H2_effect < 0 but the CI covers 0 or the candidate mean is positive",
            "NOT_CONFIRMED": "H2_effect >= 0"},
        "git_commit": cli._git_commit(),
    }
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    pd.DataFrame({"series_id": matched["candidate_ids"], "group": "candidate"}).to_csv(
        OUT / "candidate_ids.csv", index=False)
    pd.DataFrame({"series_id": matched["control_ids"], "group": "control"}).to_csv(
        OUT / "control_ids.csv", index=False)
    design.to_csv(OUT / "matching_design.csv", index=False)
    print(json.dumps({k: spec["population"][k] for k in
                      ("n_total_source", "n_descriptor_eligible",
                       "n_after_stage_a_removal", "n_candidate", "n_control_pool",
                       "stage_a_overlap")}, indent=2))
    print(json.dumps({"matched": matched["n_control_matched"],
                      "ratio": matched["ratio"], "status": matched["status"],
                      "balance_smd": matched["balance_smd"]}, indent=2))
    print(f"froze {SPEC}")


# ---------------------------------------------------------------- training --


def cmd_train(_args) -> None:
    if not SPEC.exists():
        raise SystemExit("run `population` before generating any prediction")
    spec = json.loads(SPEC.read_text())
    started = _utc()
    if not (spec["frozen_at_utc"] < started):
        raise screen.ScreenFailure("spec timestamp is not before training start")

    cand = pd.read_csv(OUT / "candidate_ids.csv")["series_id"].astype(str).tolist()
    ctrl = pd.read_csv(OUT / "control_ids.csv")["series_id"].astype(str).tolist()
    if _sha_of(cand) != spec["candidate_id_sha256"] or _sha_of(ctrl) != spec["control_id_sha256"]:
        raise screen.ScreenFailure("candidate/control ids do not match the frozen spec")

    wanted = cand + ctrl
    group = np.array(["candidate"] * len(cand) + ["control"] * len(ctrl))
    cfg = screen.config_for("m5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading full M5 ...")
    full = m5_full()
    index = pd.Index(full["series_id"]).get_indexer(wanted)
    if (index < 0).any():
        raise screen.ScreenFailure("a frozen id is absent from the source file")
    data = {"name": "m5", "y": full["y"][index], "z": full["z"][index],
            "series_id": np.asarray(wanted), "available_from": full["available_from"][index],
            "first_positive": full["first_positive"][index]}
    n_series = len(wanted)
    split = screen.build_split(data, cfg, prereg.SPLITS["m5"]["test_origin_stride"])

    per_model, raw = {}, {}
    for role, key in (("point", spec["models"]["point"]), ("hurdle", spec["models"]["hurdle"])):
        print(f"training {role} ({key}) on {device}, {n_series} series ...")
        fit = cli.train_on_split(key, cfg, spec["models"]["model_seed"], device, split)
        per_model[role] = screen.test_metrics(fit["predictions"], split.test, n_series)
        per_model[role]["_n_parameters"] = fit["n_parameters"]
        per_model[role]["_train_seconds"] = fit["train_seconds"]
        raw[role] = fit["predictions"]

    metrics = pd.DataFrame({
        "series_id": wanted, "group": group,
        "rmse_point": per_model["point"]["rmse_realized"],
        "rmse_hurdle": per_model["hurdle"]["rmse_realized"],
        "mae_point": per_model["point"]["mae_realized"],
        "mae_hurdle": per_model["hurdle"]["mae_realized"]})
    metrics["delta_raw"] = metrics["rmse_point"] - metrics["rmse_hurdle"]
    degenerate = metrics["rmse_point"] <= posthoc.RELATIVE_FLOOR
    metrics["delta_relative"] = np.where(degenerate, np.nan,
                                         metrics["delta_raw"] / metrics["rmse_point"].where(~degenerate, 1.0))
    metrics.to_csv(OUT / "per_series_metrics.csv", index=False)

    save_raw_predictions(raw, split, data, group, cfg)

    manifest = {"trained_at_utc": started, "finished_at_utc": _utc(),
                "spec_frozen_at_utc": spec["frozen_at_utc"],
                "n_series": n_series, "device": str(device), "torch": torch.__version__,
                "platform": platform.platform(),
                "test_origins": split.test.origins.tolist(),
                "n_parameters": {r: int(m["_n_parameters"]) for r, m in per_model.items()},
                "train_seconds": {r: float(m["_train_seconds"]) for r, m in per_model.items()},
                "n_degenerate_rmse_point": int(degenerate.sum()),
                "git_commit": cli._git_commit()}
    (OUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("n_series", "device", "train_seconds")},
                     indent=2, default=str))


def save_raw_predictions(raw: dict, split, data: dict, group: np.ndarray, cfg) -> None:
    """Persist what predict() already returned. Nothing is recomputed here.

    km_train.predict emits observation-level arrays of shape
    (n_series * n_origins, horizon); this only reshapes and labels them, so the
    stored numbers are the forward pass verbatim.
    """
    windows = split.test
    n_series = len(data["series_id"])
    n_origins = windows.n_origins
    series_axis = np.repeat(np.arange(n_series), n_origins)
    origin_axis = np.tile(np.asarray(windows.origins), n_series)
    horizon = windows.target.shape[1]

    frame = pd.DataFrame({
        "series_id": np.repeat(data["series_id"][series_axis], horizon),
        "group": np.repeat(group[series_axis], horizon),
        "origin": np.repeat(origin_axis, horizon),
        "step": np.tile(np.arange(horizon), len(series_axis)),
        "y_observed": windows.target.reshape(-1),
        "occurrence": windows.occurrence.reshape(-1),
        "target_mask": windows.target_mask.reshape(-1),
        "point_mean_prediction": raw["point"]["mean_prediction"].reshape(-1),
        "hurdle_mean_prediction": raw["hurdle"]["mean_prediction"].reshape(-1),
        "hurdle_p_prediction": raw["hurdle"]["p_prediction"].reshape(-1),
        "hurdle_mu_prediction": raw["hurdle"]["mu_prediction"].reshape(-1)})
    frame.to_parquet(OUT / "raw_predictions.parquet", index=False)
    print(f"wrote raw_predictions.parquet  rows={len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser("H2 confirmatory (M5, independent series)")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("population"); p.set_defaults(func=cmd_population)
    t = sub.add_parser("train"); t.set_defaults(func=cmd_train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
