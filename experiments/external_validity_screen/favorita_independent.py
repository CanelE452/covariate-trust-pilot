"""The M5-frozen H2 rule on Favorita series Stage A never saw.

This is the cross-dataset external replication the design asked for, and it is
now runnable: ``favorita_full_pool`` rebuilt the 56,918-series pool from the raw
competition file and verified the rebuild twice — the target transform against
all 2,025,600 cells of the stored Stage A sample, and the eligibility count
against the 56,918 the derivation note records.

The earlier ``favorita_transfer`` run stays where it is.  It applied the same
cutoffs inside the Stage A sample because that was the only Favorita data
present, returned LOW_SUPPORT on 18 candidates, and is superseded rather than
corrected — a reader comparing the two should be able to see both.

What transfers between datasets is the rule, not the model.  The Point and
Hurdle weights are Favorita's own, trained on Favorita's Stage A 1200, and the
independent series appear in no training split.  Only the three numeric cutoffs
come from M5, and they are read from the frozen artifact rather than recomputed
on Favorita, whose own quantiles differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from ..unified_temporal_27_v3.training import train_scale
from . import cli, confirmatory_h2 as conf, favorita_transfer as ft
from . import posthoc, prereg, rule_replication as rr, screen

OUT = screen.OUT / "favorita_independent"
SPEC = OUT / "independent_spec.json"
POOL = screen.REPO / "data" / "processed" / "favorita_full_pool.parquet"
DATASET = "favorita"
MIN_CANDIDATES = 100


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_of(values) -> str:
    return hashlib.sha256("\n".join(sorted(str(v) for v in values)).encode()).hexdigest()


def load_pool() -> dict:
    """The rebuilt full pool in load_dataset's shape.

    Favorita's leading-zero policy is `raw`, so available_from is zero for every
    series exactly as ``screen.load_dataset`` sets it; the `first_day <= 90`
    filter is what makes that safe, and it was applied when the pool was built.
    """
    frame = pd.read_parquet(POOL)
    y = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or (y < 0).any():
        raise screen.ScreenFailure("full pool has non-finite or negative demand")
    first_positive = np.array([int(np.argmax(r > 0)) if (r > 0).any() else len(r) for r in y])
    return {"name": DATASET, "y": y, "z": (y > 0).astype(np.float32),
            "series_id": frame.index.to_numpy().astype(str),
            "available_from": np.zeros(len(y), dtype=int),
            "first_positive": first_positive}


def cmd_freeze(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if SPEC.exists():
        raise SystemExit(f"{SPEC} already frozen; refusing to overwrite")
    if not ft.REPRODUCTION.exists():
        raise SystemExit("run `favorita_transfer reproduce` first; the models come from it")
    if json.loads(ft.REPRODUCTION.read_text())["verdict"] != "FAVORITA_STAGE_A_REPRODUCED":
        raise SystemExit("Favorita Stage A models were not reproduced")

    manifest = json.loads((screen.OUT / "favorita_full_pool" / "full_pool_manifest.json").read_text())
    if not manifest["target_transform_check"]["reproduced"]:
        raise screen.ScreenFailure("full pool target transform was not reproduced")
    if not manifest["eligibility_check"]["matches_documented"]:
        raise screen.ScreenFailure("full pool eligibility count does not match the derivation note")

    cfg = screen.config_for(DATASET)
    cutoffs = conf.frozen_cutoffs_from_artifact()
    data = load_pool()
    print(f"describing {len(data['series_id'])} series (train split only) ...")
    rows = [screen.describe_series(data["y"][i], 0, cfg.train_end)
            for i in range(len(data["series_id"]))]
    table = pd.DataFrame(rows)
    table["series_id"] = data["series_id"]
    table["train_scale"] = train_scale({"y": data["y"], "z": data["z"]}, cfg)

    stage_a = set(pd.read_parquet(screen.DATASETS[DATASET]["parquet"])["series_id"].astype(str))
    eligible = table[table["n_positive_train"] >= prereg.ELIGIBILITY["primary_threshold"]]
    independent = eligible[~eligible["series_id"].isin(stage_a)].copy()

    adi = independent["ADI_train"].to_numpy(float)
    occ = independent["rho_interval_abs_train"].to_numpy(float)
    mag = independent["rho_magnitude_train"].to_numpy(float)
    high = adi >= cutoffs["HIGH_ADI_min"]
    low_occ = occ <= cutoffs["LOW_OCC_max"]
    persistent = mag >= cutoffs["MAG_PERSISTENT_min"]
    independent["group"] = "unused"
    independent.loc[high & low_occ & persistent, "group"] = "candidate"
    independent.loc[high & low_occ & ~persistent, "group"] = "control"

    overlap = len(set(independent["series_id"]) & stage_a)
    if overlap:
        raise screen.ScreenFailure(f"STAGE_A_OVERLAP={overlap}")
    candidate = sorted(independent.loc[independent["group"] == "candidate", "series_id"])
    control = sorted(independent.loc[independent["group"] == "control", "series_id"])

    spec = {
        "study": "H2 frozen-rule external replication on unseen Favorita series",
        "frozen_at_utc": _utc(),
        "frozen_before_any_independent_prediction": True,
        "frozen_h2_cutoffs": cutoffs,
        "cutoffs_refitted_to_favorita": False,
        "favorita_own_quantiles_for_reference_only": {
            "ADI_median": float(np.nanmedian(adi)),
            "abs_rho_interval_lower_tertile": float(np.nanquantile(occ, 1 / 3)),
            "rho_magnitude_upper_tertile": float(np.nanquantile(mag, 2 / 3))},
        "full_pool_manifest": manifest,
        "population": {"full_pool": int(len(table)),
                       "descriptor_eligible": int(len(eligible)),
                       "after_stage_a_removal": int(len(independent)),
                       "stage_a_removed": int(len(eligible) - len(independent)),
                       "candidate": len(candidate), "control": len(control),
                       "stage_a_overlap": overlap},
        "sampling_rule": "none; every candidate and every control is evaluated",
        "candidate_id_sha256": _sha_of(candidate), "control_id_sha256": _sha_of(control),
        "stage_a_excluded_id_sha256": _sha_of(stage_a),
        "models": {"provenance": "Favorita Stage A 1200, reproduced and frozen by "
                                 "favorita_transfer reproduce",
                   "point": prereg.MODELS["point"], "hurdle": prereg.MODELS["hurdle"],
                   "model_seed": prereg.MODELS["canonical_model_seed"],
                   "independent_series_in_training": False,
                   "model_transferred_across_datasets": False},
        "training": dict(km_prereg.TRAINING), "split": dict(prereg.SPLITS[DATASET]),
        "primary": {"population": "unmatched candidate vs unmatched control",
                    "metric": "delta_relative = (RMSE_Point - RMSE_Hurdle) / RMSE_Point",
                    "expected_sign": "negative", "matching_applied": False},
        "bootstrap": dict(prereg.BOOTSTRAP),
        "pooled_with_m5": False,
        "supersedes": "favorita_transfer (Stage A sample only, LOW_SUPPORT), kept for comparison",
        "git_commit": cli._git_commit()}
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    independent[independent["group"] != "unused"].to_csv(OUT / "independent_population.csv",
                                                         index=False)
    print(json.dumps(spec["population"], indent=2))
    print(f"froze {SPEC}")


def cmd_primary(_args) -> None:
    if not SPEC.exists():
        raise SystemExit("run `freeze` first")
    spec = json.loads(SPEC.read_text())
    started = _utc()
    if not (spec["frozen_at_utc"] < started):
        raise screen.ScreenFailure("spec timestamp is not before scoring")

    population = pd.read_csv(OUT / "independent_population.csv")
    population["series_id"] = population["series_id"].astype(str)
    candidate = sorted(population.loc[population["group"] == "candidate", "series_id"])
    control = sorted(population.loc[population["group"] == "control", "series_id"])
    if (_sha_of(candidate) != spec["candidate_id_sha256"]
            or _sha_of(control) != spec["control_id_sha256"]):
        raise screen.ScreenFailure("population ids do not match the frozen spec")

    wanted = candidate + control
    group = np.array(["candidate"] * len(candidate) + ["control"] * len(control))
    cfg = screen.config_for(DATASET)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pool = load_pool()
    index = pd.Index(pool["series_id"]).get_indexer(wanted)
    if (index < 0).any():
        raise screen.ScreenFailure("a frozen id is absent from the rebuilt pool")
    data = {"name": DATASET, "y": pool["y"][index], "z": pool["z"][index],
            "series_id": np.asarray(wanted),
            "available_from": pool["available_from"][index],
            "first_positive": pool["first_positive"][index]}
    stage_a = set(pd.read_parquet(screen.DATASETS[DATASET]["parquet"])["series_id"].astype(str))
    if set(wanted) & stage_a:
        raise screen.ScreenFailure("STAGE_A_OVERLAP in the scored population")

    windows = rr.test_windows_only(data, cfg, prereg.SPLITS[DATASET]["test_origin_stride"])
    n_series = len(wanted)
    raw, metrics = {}, {}
    for role in ("point", "hurdle"):
        payload = torch.load(ft.MODELS / f"{role}.pt", map_location=device, weights_only=False)
        if payload["trained_on"] != "stage_a_favorita_1200":
            raise screen.ScreenFailure(f"unexpected training population {payload['trained_on']}")
        from ..om_factorization_killtest import models
        model = models.BUILDERS[payload["builder_key"]](payload["lookback"],
                                                        payload["horizon"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        print(f"scoring {role} on {n_series} unseen Favorita series ...")
        raw[role] = km_train.predict(model, windows, device)
        metrics[role] = screen.test_metrics(raw[role], windows, n_series)

    frame = pd.DataFrame({"series_id": wanted, "group": group,
                          "rmse_point": metrics["point"]["rmse_realized"],
                          "rmse_hurdle": metrics["hurdle"]["rmse_realized"],
                          "mae_point": metrics["point"]["mae_realized"],
                          "mae_hurdle": metrics["hurdle"]["mae_realized"]})
    frame["delta_raw"] = frame["rmse_point"] - frame["rmse_hurdle"]
    degenerate = frame["rmse_point"] <= posthoc.RELATIVE_FLOOR
    frame["delta_relative"] = np.where(
        degenerate, np.nan, frame["delta_raw"] / frame["rmse_point"].where(~degenerate, 1.0))
    frame.to_csv(OUT / "independent_per_series.csv", index=False)
    rr._save_raw(raw, windows, data, group, OUT / "independent_raw_predictions.parquet")

    is_cand = (frame["group"] == "candidate").to_numpy()
    dr = frame["delta_relative"].to_numpy(float)
    dw = frame["delta_raw"].to_numpy(float)
    cand = rr.summarise(dr[is_cand], dw[is_cand])
    ctrl = rr.summarise(dr[~is_cand], dw[~is_cand])
    effect = cand["mean_delta_relative"] - ctrl["mean_delta_relative"]

    c, k = dr[is_cand], dr[~is_cand]
    c, k = c[np.isfinite(c)], k[np.isfinite(k)]
    draws, seed = prereg.BOOTSTRAP["draws"], prereg.BOOTSTRAP["seed"]
    rng = np.random.default_rng(seed)
    effect_ci = rr._boot_ci(lambda r: (c[r.integers(0, c.size, c.size)].mean()
                                       - k[r.integers(0, k.size, k.size)].mean()), rng, draws)
    rng = np.random.default_rng(seed)
    win_ci = rr._boot_ci(lambda r: ((c[r.integers(0, c.size, c.size)] < 0).mean()
                                    - (k[r.integers(0, k.size, k.size)] < 0).mean()), rng, draws)
    rng = np.random.default_rng(seed)
    cand_ci = rr._boot_ci(lambda r: c[r.integers(0, c.size, c.size)].mean(), rng, draws)
    rng = np.random.default_rng(seed)
    ctrl_ci = rr._boot_ci(lambda r: k[r.integers(0, k.size, k.size)].mean(), rng, draws)

    excludes = bool(effect_ci[0] > 0 or effect_ci[1] < 0)
    if c.size < MIN_CANDIDATES:
        verdict = "FAVORITA_RULE_LOW_SUPPORT"
    elif effect >= 0:
        verdict = "FAVORITA_RULE_NOT_TRANSFERRED"
    elif excludes and cand["point_win_rate"] > ctrl["point_win_rate"] and cand["mean_delta_relative"] < 0:
        verdict = "FAVORITA_RULE_STRONG_TRANSFER"
    else:
        verdict = "FAVORITA_RULE_PARTIAL_TRANSFER"

    report = {"verdict": verdict, "analysis": "PRIMARY, unmatched, M5-frozen rule, unseen Favorita",
              "matching_applied": False, "pooled_with_m5": False,
              "candidate": cand | {"mean_ci": cand_ci},
              "control": ctrl | {"mean_ci": ctrl_ci},
              "H2_FAVORITA_rule_effect": float(effect), "effect_ci": effect_ci,
              "effect_ci_excludes_zero": excludes,
              "point_win_rate_difference": float(cand["point_win_rate"] - ctrl["point_win_rate"]),
              "point_win_rate_difference_ci": win_ci,
              "population": spec["population"], "bootstrap": dict(prereg.BOOTSTRAP),
              "n_degenerate_rmse_point": int(degenerate.sum()),
              "scored_at_utc": started, "spec_frozen_at_utc": spec["frozen_at_utc"],
              "device": str(device), "git_commit": cli._git_commit()}
    (OUT / "primary_result.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k2: report[k2] for k2 in
                      ("verdict", "H2_FAVORITA_rule_effect", "effect_ci",
                       "point_win_rate_difference")}, indent=2))


def cmd_secondary(_args) -> None:
    if not (OUT / "primary_result.json").exists():
        raise SystemExit("run `primary` first")
    per_series = pd.read_csv(OUT / "independent_per_series.csv")
    per_series["series_id"] = per_series["series_id"].astype(str)
    population = pd.read_csv(OUT / "independent_population.csv")
    population["series_id"] = population["series_id"].astype(str)
    frame = per_series.merge(population, on="series_id", suffixes=("", "_pop"))
    frame = frame[np.isfinite(frame["delta_relative"])]
    frame["log_ADI"] = np.log(frame["ADI_train"])
    frame["CV2_positive"] = frame["CV2_positive_train"]
    frame["log_train_scale"] = np.log(frame["train_scale"])
    frame["abs_rho_interval"] = frame["rho_interval_abs_train"]
    variables = list(rr.MATCH_VARIABLES)
    frame = frame[np.isfinite(frame[variables].to_numpy(float)).all(axis=1)]

    design = frame[variables].to_numpy(float)
    treated = (frame["group"] == "candidate").to_numpy().astype(int)
    y = frame["delta_relative"].to_numpy(float)
    e, w = rr.propensity_weights(design, treated)
    unweighted = {v: conf.smd(frame.loc[treated == 1, v].to_numpy(float),
                              frame.loc[treated == 0, v].to_numpy(float)) for v in variables}
    balance = {v: rr.weighted_smd(frame[v].to_numpy(float), w, treated) for v in variables}
    worst = max(abs(v) for v in balance.values())
    ess = {g: float(w[treated == t].sum() ** 2 / (w[treated == t] ** 2).sum())
           for g, t in (("candidate", 1), ("control", 0))}

    report = {"analysis": "SECONDARY -- OVERLAP_ADJUSTED_ASSOCIATION (Favorita, independent)",
              "not_a_causal_effect": True, "separate_from_primary": True,
              "n_used": int(len(frame)), "unweighted_smd": unweighted,
              "weighted_smd": balance, "worst_abs_weighted_smd": float(worst),
              "effective_sample_size": ess,
              "propensity": {"candidate_median": float(np.median(e[treated == 1])),
                             "control_median": float(np.median(e[treated == 0]))}}
    if worst > rr.WEIGHTED_SMD_LIMIT:
        report["verdict"] = "OVERLAP_DIAGNOSTIC_INCONCLUSIVE"
    else:
        adjusted = (np.average(y[treated == 1], weights=w[treated == 1])
                    - np.average(y[treated == 0], weights=w[treated == 0]))
        rng = np.random.default_rng(prereg.BOOTSTRAP["seed"])

        def draw(r):
            idx = r.integers(0, len(frame), len(frame))
            t2 = treated[idx]
            if t2.sum() < 30 or (1 - t2).sum() < 30:
                return np.nan
            try:
                _, w2 = rr.propensity_weights(design[idx], t2)
            except Exception:
                return np.nan
            y2 = y[idx]
            return (np.average(y2[t2 == 1], weights=w2[t2 == 1])
                    - np.average(y2[t2 == 0], weights=w2[t2 == 0]))

        ci = rr._boot_ci(draw, rng, prereg.BOOTSTRAP["draws"])
        report["verdict"] = "OVERLAP_ADJUSTED_ASSOCIATION_COMPUTED"
        report["overlap_adjusted_association"] = float(adjusted)
        report["overlap_adjusted_association_ci"] = ci
        report["ci_excludes_zero"] = bool(ci[0] > 0 or ci[1] < 0)
    (OUT / "secondary_overlap.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in report if k in
                      ("verdict", "worst_abs_weighted_smd", "overlap_adjusted_association",
                       "overlap_adjusted_association_ci")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("Favorita independent rule replication")
    sub = parser.add_subparsers(required=True)
    for name, func in (("freeze", cmd_freeze), ("primary", cmd_primary),
                       ("secondary", cmd_secondary)):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
