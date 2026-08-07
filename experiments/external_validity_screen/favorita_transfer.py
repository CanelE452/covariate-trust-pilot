"""The M5-frozen H2 rule applied to Favorita, numbers unchanged.

What the design asked for is the rule transferred to Favorita's full pool with
the Stage A 1200 removed.  That cannot be run here: the repository holds only
``data/processed/favorita_series.parquet``, which *is* the Stage A 1200, and no
raw Favorita source.  The post-hoc diagnostic already flagged this as
"Favorita full pool 미물질화".  Materialising it needs the original
Corporación Favorita competition file, and the Store Sales competition is not a
substitute — that one is aggregated to 33 families per store while every
identifier here is item_nbr_store_nbr.

So this runs the reduced form and labels it as such.  The rule keeps M5's
numeric cutoffs exactly; what it loses is independence from Stage A's sample,
because those are the only Favorita series that exist locally.  It is a
rule-transfer check, not the external replication the design specified, and the
verdict carries LOW_SUPPORT on top: the M5 cutoffs select 18 candidates here.

Nothing about the rule is refitted to Favorita.  Recomputing HIGH_ADI from
Favorita's own median, or LOW_OCC from its own tertile, would make the result
unfalsifiable, and Favorita's tertiles differ (0.0088 against M5's 0.0074), so
the difference is not cosmetic.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from . import cli, confirmatory_h2 as conf, posthoc, prereg
from . import rule_replication as rr, screen

OUT = screen.OUT / "favorita_transfer"
MODELS = OUT / "models"
REPRODUCTION = OUT / "favorita_stage_a_reproduction.json"

#: Fixed before the reproduction ran, same shape as the M5 one.
REPRODUCTION_TOLERANCE = dict(rr.REPRODUCTION_TOLERANCE)

MIN_CANDIDATES = 100
DATASET = "favorita"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cmd_reproduce(_args) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    stored = json.loads((screen.OUT / "stage_a_results.json").read_text())
    reference = stored["manifest"][DATASET]["overall"]
    reference_series = pd.read_csv(screen.OUT / "per_series_metrics.csv") \
        .query("dataset == @DATASET").set_index("series_id")["delta_rmse"]

    cfg = screen.config_for(DATASET)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = prereg.MODELS["canonical_model_seed"]

    print(f"loading Stage A's {DATASET} series ...")
    data = screen.load_dataset(DATASET)
    split = screen.build_split(data, cfg, prereg.SPLITS[DATASET]["test_origin_stride"])
    n_series = data["y"].shape[0]

    metrics = {}
    for role, key in (("point", prereg.MODELS["point"]),
                      ("hurdle", prereg.MODELS["hurdle"])):
        print(f"retraining {role} ({key}) on {device} ...")
        fit = rr.train_frozen(key, cfg, seed, device, split)
        metrics[role] = screen.test_metrics(fit["predictions"], split.test, n_series)
        torch.save({"state_dict": fit["model"].state_dict(), "builder_key": key,
                    "lookback": cfg.lookback, "horizon": cfg.horizon,
                    "model_seed": seed, "trained_on": "stage_a_favorita_1200",
                    "training": dict(km_prereg.TRAINING)}, MODELS / f"{role}.pt")

    delta = metrics["point"]["rmse_realized"] - metrics["hurdle"]["rmse_realized"]
    fresh = {"rmse_point": float(np.mean(metrics["point"]["rmse_realized"])),
             "rmse_hurdle": float(np.mean(metrics["hurdle"]["rmse_realized"])),
             "mae_point": float(np.mean(metrics["point"]["mae_realized"])),
             "mae_hurdle": float(np.mean(metrics["hurdle"]["mae_realized"])),
             "mean_delta": float(np.mean(delta)),
             "point_win_pct": float(np.mean(delta < 0) * 100)}

    from scipy import stats
    aligned = reference_series.loc[data["series_id"]].to_numpy(float)
    rank = float(stats.spearmanr(aligned, delta).statistic)
    checks = {
        "rmse_point_relative": abs(fresh["rmse_point"] - reference["rmse_point"]) / reference["rmse_point"],
        "rmse_hurdle_relative": abs(fresh["rmse_hurdle"] - reference["rmse_hurdle"]) / reference["rmse_hurdle"],
        "point_win_pct_absolute": abs(fresh["point_win_pct"] - reference["point_win_pct"]),
        "per_series_delta_spearman": rank}
    passed = (checks["rmse_point_relative"] <= REPRODUCTION_TOLERANCE["mean_rmse_relative"]
              and checks["rmse_hurdle_relative"] <= REPRODUCTION_TOLERANCE["mean_rmse_relative"]
              and checks["point_win_pct_absolute"] <= REPRODUCTION_TOLERANCE["point_win_pct_absolute"]
              and rank >= REPRODUCTION_TOLERANCE["per_series_delta_spearman_min"])

    report = {"verdict": "FAVORITA_STAGE_A_REPRODUCED" if passed else "FAVORITA_STAGE_A_NOT_REPRODUCED",
              "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
              "stage_a_reference": reference, "reproduction": fresh,
              "n_series": int(n_series), "device": str(device),
              "torch": torch.__version__, "platform": platform.platform(),
              "reproduced_at_utc": _utc(), "git_commit": cli._git_commit()}
    REPRODUCTION.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": report["verdict"], "checks": checks}, indent=2))
    if not passed:
        raise SystemExit("FAVORITA_STAGE_A_NOT_REPRODUCED")


def cmd_transfer(_args) -> None:
    if not REPRODUCTION.exists():
        raise SystemExit("run `reproduce` first")
    if json.loads(REPRODUCTION.read_text())["verdict"] != "FAVORITA_STAGE_A_REPRODUCED":
        raise SystemExit("Favorita Stage A was not reproduced")

    cutoffs = conf.frozen_cutoffs_from_artifact()
    series = pd.read_csv(screen.OUT / "per_series_metrics.csv")
    block = series[series["dataset"] == DATASET].copy()
    pool = block[(block["n_positive_train"] >= posthoc.THRESHOLD)
                 & np.isfinite(block["delta_rmse"])].reset_index(drop=True)

    adi = pool["ADI_train"].to_numpy(float)
    occ = pool["rho_interval_abs_train"].to_numpy(float)
    mag = pool["rho_magnitude_train"].to_numpy(float)
    high = adi >= cutoffs["HIGH_ADI_min"]
    low_occ = occ <= cutoffs["LOW_OCC_max"]
    persistent = mag >= cutoffs["MAG_PERSISTENT_min"]
    pool["group"] = "unused"
    pool.loc[high & low_occ & persistent, "group"] = "candidate"
    pool.loc[high & low_occ & ~persistent, "group"] = "control"

    degenerate = pool["rmse_point"] <= posthoc.RELATIVE_FLOOR
    pool["delta_raw"] = pool["delta_rmse"]
    pool["delta_relative"] = np.where(
        degenerate, np.nan, pool["delta_raw"] / pool["rmse_point"].where(~degenerate, 1.0))

    cand = pool[pool["group"] == "candidate"]
    ctrl = pool[pool["group"] == "control"]
    c = cand["delta_relative"].to_numpy(float)
    k = ctrl["delta_relative"].to_numpy(float)
    c, k = c[np.isfinite(c)], k[np.isfinite(k)]
    effect = float(c.mean() - k.mean()) if c.size and k.size else float("nan")

    draws, seed = prereg.BOOTSTRAP["draws"], prereg.BOOTSTRAP["seed"]
    rng = np.random.default_rng(seed)
    effect_ci = rr._boot_ci(lambda r: (c[r.integers(0, c.size, c.size)].mean()
                                       - k[r.integers(0, k.size, k.size)].mean()), rng, draws)
    rng = np.random.default_rng(seed)
    cand_ci = rr._boot_ci(lambda r: c[r.integers(0, c.size, c.size)].mean(), rng, draws)
    rng = np.random.default_rng(seed)
    ctrl_ci = rr._boot_ci(lambda r: k[r.integers(0, k.size, k.size)].mean(), rng, draws)

    excludes_zero = bool(effect_ci[0] > 0 or effect_ci[1] < 0)
    cand_win, ctrl_win = float((c < 0).mean()), float((k < 0).mean())
    if c.size < MIN_CANDIDATES:
        verdict = "FAVORITA_RULE_LOW_SUPPORT"
    elif effect >= 0:
        verdict = "FAVORITA_RULE_NOT_TRANSFERRED"
    elif excludes_zero and cand_win > ctrl_win and c.mean() < 0:
        verdict = "FAVORITA_RULE_STRONG_TRANSFER"
    else:
        verdict = "FAVORITA_RULE_PARTIAL_TRANSFER"

    report = {
        "verdict": verdict,
        "design_deviation": {
            "specified": "M5-frozen rule on the Favorita FULL pool with Stage A 1200 removed",
            "executed": "M5-frozen rule on the Favorita Stage A 1200, the only Favorita "
                        "series present in the repository",
            "reason": "no raw Favorita source; data/processed/favorita_series.parquet "
                      "contains exactly the Stage A sample",
            "independence_from_stage_a_sample": False,
            "cutoffs_refitted_to_favorita": False,
            "not_the_specified_external_replication": True},
        "frozen_cutoffs_used": cutoffs,
        "favorita_own_quantiles_for_reference_only": {
            "ADI_median": float(np.nanmedian(adi)),
            "abs_rho_interval_lower_tertile": float(np.nanquantile(occ, 1 / 3)),
            "rho_magnitude_upper_tertile": float(np.nanquantile(mag, 2 / 3)),
            "note": "reported to show the transfer is not vacuous; never used as a cutoff"},
        "population": {"pool": int(len(pool)),
                       "high_adi": int(high.sum()), "low_occ": int(low_occ.sum()),
                       "mag_persistent": int(persistent.sum()),
                       "candidate": int(len(cand)), "control": int(len(ctrl)),
                       "min_candidates_for_conclusion": MIN_CANDIDATES},
        "candidate": {"n": int(c.size), "mean_delta_relative": float(c.mean()),
                      "median_delta_relative": float(np.median(c)),
                      "mean_ci": cand_ci, "point_win_rate": cand_win,
                      "mean_delta_raw": float(cand["delta_raw"].mean())},
        "control": {"n": int(k.size), "mean_delta_relative": float(k.mean()),
                    "median_delta_relative": float(np.median(k)),
                    "mean_ci": ctrl_ci, "point_win_rate": ctrl_win,
                    "mean_delta_raw": float(ctrl["delta_raw"].mean())},
        "H2_FAVORITA_rule_effect": effect, "effect_ci": effect_ci,
        "effect_ci_excludes_zero": excludes_zero,
        "point_win_rate_difference": cand_win - ctrl_win,
        "bootstrap": dict(prereg.BOOTSTRAP),
        "pooled_with_m5": False,
        "transferred_at_utc": _utc(), "git_commit": cli._git_commit()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "primary_transfer.json").write_text(json.dumps(report, indent=2, default=str))
    pool[pool["group"] != "unused"].to_csv(OUT / "transfer_population.csv", index=False)
    print(json.dumps({"verdict": verdict, "candidate": int(c.size), "control": int(k.size),
                      "effect": effect, "effect_ci": effect_ci}, indent=2))


def cmd_secondary(_args) -> None:
    if not (OUT / "primary_transfer.json").exists():
        raise SystemExit("run `transfer` first")
    pool = pd.read_csv(OUT / "transfer_population.csv")
    data = screen.load_dataset(DATASET)
    cfg = screen.config_for(DATASET)
    from ..unified_temporal_27_v3.training import train_scale
    scale = pd.Series(train_scale({"y": data["y"], "z": data["z"]}, cfg),
                      index=pd.Index(data["series_id"]).astype(str))

    pool["series_id"] = pool["series_id"].astype(str)
    pool["log_ADI"] = np.log(pool["ADI_train"])
    pool["CV2_positive"] = pool["CV2_positive_train"]
    pool["log_train_scale"] = np.log(scale.loc[pool["series_id"]].to_numpy(float))
    pool["abs_rho_interval"] = pool["rho_interval_abs_train"]
    variables = list(rr.MATCH_VARIABLES)
    frame = pool[np.isfinite(pool[variables + ["delta_relative"]].to_numpy(float)).all(axis=1)]

    treated = (frame["group"] == "candidate").to_numpy().astype(int)
    design = frame[variables].to_numpy(float)
    y = frame["delta_relative"].to_numpy(float)
    unweighted = {v: conf.smd(frame.loc[treated == 1, v].to_numpy(float),
                              frame.loc[treated == 0, v].to_numpy(float)) for v in variables}

    report = {"analysis": "SECONDARY -- OVERLAP_ADJUSTED_ASSOCIATION (Favorita)",
              "not_a_causal_effect": True, "separate_from_primary": True,
              "n_used": int(len(frame)), "n_candidate": int(treated.sum()),
              "n_control": int((1 - treated).sum()),
              "unweighted_smd": unweighted}
    if treated.sum() < 30:
        report["verdict"] = "OVERLAP_DIAGNOSTIC_INCONCLUSIVE"
        report["reason"] = (f"only {int(treated.sum())} candidates; a propensity model on "
                            f"{len(variables)} covariates is not identifiable here")
    else:
        e, w = rr.propensity_weights(design, treated)
        balance = {v: rr.weighted_smd(frame[v].to_numpy(float), w, treated) for v in variables}
        worst = max(abs(v) for v in balance.values())
        report["weighted_smd"] = balance
        report["worst_abs_weighted_smd"] = float(worst)
        report["effective_sample_size"] = {
            g: float(w[treated == t].sum() ** 2 / (w[treated == t] ** 2).sum())
            for g, t in (("candidate", 1), ("control", 0))}
        if worst > rr.WEIGHTED_SMD_LIMIT:
            report["verdict"] = "OVERLAP_DIAGNOSTIC_INCONCLUSIVE"
        else:
            adjusted = (np.average(y[treated == 1], weights=w[treated == 1])
                        - np.average(y[treated == 0], weights=w[treated == 0]))
            report["verdict"] = "OVERLAP_ADJUSTED_ASSOCIATION_COMPUTED"
            report["overlap_adjusted_association"] = float(adjusted)
    (OUT / "secondary_overlap.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in report
                      if k in ("verdict", "reason", "n_candidate", "n_control",
                               "worst_abs_weighted_smd")}, indent=2))


def cmd_occurrence(_args) -> None:
    """Observation-level Hurdle diagnostic on the transfer population."""
    pool = pd.read_csv(OUT / "transfer_population.csv")
    pool["series_id"] = pool["series_id"].astype(str)
    cfg = screen.config_for(DATASET)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = screen.load_dataset(DATASET)
    ids = pd.Index(data["series_id"]).astype(str)
    keep = ids.get_indexer(pool["series_id"])
    subset = {"name": DATASET, "y": data["y"][keep], "z": data["z"][keep],
              "series_id": np.asarray(pool["series_id"]),
              "available_from": data["available_from"][keep],
              "first_positive": data["first_positive"][keep]}
    windows = rr.test_windows_only(subset, cfg, prereg.SPLITS[DATASET]["test_origin_stride"])

    raw = {}
    for role in ("point", "hurdle"):
        payload = torch.load(MODELS / f"{role}.pt", map_location=device, weights_only=False)
        from ..om_factorization_killtest import models
        model = models.BUILDERS[payload["builder_key"]](payload["lookback"],
                                                        payload["horizon"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        raw[role] = km_train.predict(model, windows, device)
    group = pool["group"].to_numpy()
    rr._save_raw(raw, windows, subset, group, OUT / "transfer_raw_predictions.parquet")

    frame = pd.read_parquet(OUT / "transfer_raw_predictions.parquet")
    frame = frame[frame["target_mask"] > 0]
    train_rate = pd.Series(data["z"][keep][:, :cfg.train_end].mean(axis=1),
                           index=pool["series_id"].to_numpy())
    frame["p_const"] = train_rate.loc[frame["series_id"]].to_numpy()
    out = {}
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
    for g in ("candidate", "control"):
        block = frame[frame["group"] == g]
        occ = block["occurrence"].to_numpy(float)
        p = np.clip(block["hurdle_p_prediction"].to_numpy(float), 1e-7, 1 - 1e-7)
        pc = np.clip(block["p_const"].to_numpy(float), 1e-7, 1 - 1e-7)
        brier, brier_c = float(((p - occ) ** 2).mean()), float(((pc - occ) ** 2).mean())
        both = len(np.unique(occ)) > 1
        out[g] = {"n_observations": int(len(block)), "prevalence": float(occ.mean()),
                  "brier_hurdle": brier, "brier_constant": brier_c,
                  "brier_skill_score": float(1 - brier / brier_c) if brier_c > 0 else float("nan"),
                  "roc_auc": float(roc_auc_score(occ, p)) if both else float("nan"),
                  "pr_auc": float(average_precision_score(occ, p)) if both else float("nan"),
                  "log_loss_hurdle": float(log_loss(occ, p, labels=[0, 1])) if both else float("nan"),
                  "log_loss_constant": float(log_loss(occ, pc, labels=[0, 1])) if both else float("nan"),
                  "mean_p_hat_on_zero": float(p[occ == 0].mean()) if (occ == 0).any() else float("nan"),
                  "mean_p_hat_on_positive": float(p[occ == 1].mean()) if (occ == 1).any() else float("nan")}
        out[g]["p_hat_separation"] = out[g]["mean_p_hat_on_positive"] - out[g]["mean_p_hat_on_zero"]
    (OUT / "occurrence_diagnostic.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser("Favorita rule transfer")
    sub = parser.add_subparsers(required=True)
    for name, func in (("reproduce", cmd_reproduce), ("transfer", cmd_transfer),
                       ("secondary", cmd_secondary), ("occurrence", cmd_occurrence)):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
