"""Does the frozen H2 rule pick Point-favourable series it has never seen?

The confirmatory attempt before this one stopped at a matching gate, because
rho_magnitude — the descriptor that defines a candidate — turns out to be
correlated +0.575 with series scale in real M5.  Matching that away answers a
different question from the one the rule was written to answer, so the work is
split in two:

    primary     the frozen rule applied to unseen series, unmatched.  Does the
                condition select series where Point does relatively better?
    secondary   the same contrast inside a covariate-overlap population.  Is
                anything left once scale, ADI, CV2 and occurrence dependence
                are balanced?

Primary is external validation of a selection rule and is reported first and
alone.  Secondary is a mechanism diagnostic and never renames itself a causal
effect.

Stage A saved no checkpoint — cli.train_on_split keeps its best state in
memory and the only torch.save in the tree belongs to a different study — so
the models are retrained on Stage A's own 1200 series and checked against the
stored Stage A metrics before they are allowed to score anything.  The
independent series are never in a training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ..om_factorization_killtest import prereg as km_prereg
from ..om_factorization_killtest import train as km_train
from ..unified_temporal_27_v3.training import make_windows, train_scale
from . import cli, confirmatory_h2 as conf, posthoc, prereg, screen

OUT = screen.OUT / "rule_replication"
MODELS = OUT / "models"
SPEC = OUT / "replication_spec.json"
REPRODUCTION = OUT / "stage_a_reproduction.json"

#: Fixed before the reproduction ran. Stage A trained on Linux with torch
#: 2.5.1+cu121; this machine is Windows with 2.13.0+cu126, so GPU kernel
#: selection and reduction order differ and bit-equality is not available.
#: The rank agreement is the binding check: H1 and H2 are statements about how
#: series are ordered by delta, so a model that reorders them is not the same
#: model however close its aggregate RMSE lands.
REPRODUCTION_TOLERANCE = {
    "mean_rmse_relative": 0.02,
    "point_win_pct_absolute": 5.0,
    "per_series_delta_spearman_min": 0.80,
    "fixed_before_running": True,
}

MATCH_VARIABLES = conf.MATCH_VARIABLES
WEIGHTED_SMD_LIMIT = 0.10


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_of(values) -> str:
    return hashlib.sha256("\n".join(sorted(str(v) for v in values)).encode()).hexdigest()


# ------------------------------------------------- Stage A reproduction ----


def train_frozen(key: str, cfg, seed: int, device, split) -> dict:
    """cli.train_on_split verbatim, returning the fitted model as well.

    cli.train_on_split discards the model once it has predicted, and scoring
    unseen series needs the weights, so the loop is copied rather than
    imported.  Nothing in it is altered: same seeding order, optimizer, batch
    size, epoch budget, patience and checkpoint criterion.  The guarantee that
    the copy did not drift is empirical — reproduce() compares this model's
    Stage A metrics against the stored Stage A artifact.
    """
    import copy, time
    from ..om_factorization_killtest import models
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = models.BUILDERS[key](cfg.lookback, cfg.horizon).to(device)
    loss_fn = models.LOSSES[key]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=km_prereg.TRAINING["learning_rate"],
                                 weight_decay=km_prereg.TRAINING["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = km_train._loader(split.train, km_prereg.TRAINING["batch_size"],
                                    True, generator)
    val_loader = km_train._loader(split.validation, 1024, False)
    variance = torch.tensor(split.positive_variance, device=device)
    best, best_state, bad = float("inf"), None, 0
    started = time.time()
    for _ in range(km_prereg.TRAINING["max_epochs"]):
        model.train()
        for history, target, occurrence, mask, scale in train_loader:
            history, target = history.to(device), target.to(device)
            occurrence, mask = occurrence.to(device), mask.to(device)
            scale = scale.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(history, scale), target, occurrence, mask, variance)
            loss.backward()
            optimizer.step()
        current = km_train._validation_mean_mse(model, val_loader, device)
        if current < best - 1e-9:
            best, bad = current, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= km_prereg.TRAINING["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model,
            "predictions": km_train.predict(model, split.test, device),
            "n_parameters": models.count_parameters(model),
            "train_seconds": time.time() - started}


def cmd_reproduce(_args) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    stored = json.loads((screen.OUT / "stage_a_results.json").read_text())
    reference = stored["manifest"]["m5"]["overall"]
    reference_series = pd.read_csv(screen.OUT / "per_series_metrics.csv") \
        .query("dataset == 'm5'").set_index("series_id")["delta_rmse"]

    cfg = screen.config_for("m5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = prereg.MODELS["canonical_model_seed"]

    print("loading Stage A's 1200 series ...")
    data = screen.load_dataset("m5")
    split = screen.build_split(data, cfg, prereg.SPLITS["m5"]["test_origin_stride"])
    n_series = data["y"].shape[0]

    per_model = {}
    for role, key in (("point", prereg.MODELS["point"]),
                      ("hurdle", prereg.MODELS["hurdle"])):
        print(f"retraining {role} ({key}) on {device} ...")
        fit = train_frozen(key, cfg, seed, device, split)
        per_model[role] = {"metrics": screen.test_metrics(fit["predictions"],
                                                          split.test, n_series),
                           "fit": fit}
        torch.save({"state_dict": fit["model"].state_dict(), "builder_key": key,
                    "lookback": cfg.lookback, "horizon": cfg.horizon,
                    "model_seed": seed, "trained_on": "stage_a_m5_1200",
                    "training": dict(km_prereg.TRAINING)},
                   MODELS / f"{role}.pt")

    delta = (per_model["point"]["metrics"]["rmse_realized"]
             - per_model["hurdle"]["metrics"]["rmse_realized"])
    fresh = {"rmse_point": float(np.mean(per_model["point"]["metrics"]["rmse_realized"])),
             "rmse_hurdle": float(np.mean(per_model["hurdle"]["metrics"]["rmse_realized"])),
             "mae_point": float(np.mean(per_model["point"]["metrics"]["mae_realized"])),
             "mae_hurdle": float(np.mean(per_model["hurdle"]["metrics"]["mae_realized"])),
             "mean_delta": float(np.mean(delta)),
             "median_delta": float(np.median(delta)),
             "point_win_pct": float(np.mean(delta < 0) * 100)}

    from scipy import stats
    aligned = reference_series.loc[data["series_id"]].to_numpy(float)
    rank = float(stats.spearmanr(aligned, delta).statistic)

    checks = {
        "rmse_point_relative": abs(fresh["rmse_point"] - reference["rmse_point"]) / reference["rmse_point"],
        "rmse_hurdle_relative": abs(fresh["rmse_hurdle"] - reference["rmse_hurdle"]) / reference["rmse_hurdle"],
        "point_win_pct_absolute": abs(fresh["point_win_pct"] - reference["point_win_pct"]),
        "per_series_delta_spearman": rank,
    }
    passed = (checks["rmse_point_relative"] <= REPRODUCTION_TOLERANCE["mean_rmse_relative"]
              and checks["rmse_hurdle_relative"] <= REPRODUCTION_TOLERANCE["mean_rmse_relative"]
              and checks["point_win_pct_absolute"] <= REPRODUCTION_TOLERANCE["point_win_pct_absolute"]
              and rank >= REPRODUCTION_TOLERANCE["per_series_delta_spearman_min"])

    report = {"verdict": "STAGE_A_REPRODUCED" if passed else "STAGE_A_NOT_REPRODUCED",
              "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
              "stage_a_reference": reference, "reproduction": fresh,
              "n_series": int(n_series), "device": str(device),
              "torch": torch.__version__, "platform": platform.platform(),
              "stage_a_torch": stored.get("torch"), "stage_a_platform": stored.get("platform"),
              "reproduced_at_utc": _utc(), "git_commit": cli._git_commit()}
    REPRODUCTION.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": report["verdict"], "checks": checks}, indent=2))
    if not passed:
        raise SystemExit("STAGE_A_NOT_REPRODUCED; the frozen models may not be used")


# ------------------------------------------------------------- freezing ----


def cmd_freeze(_args) -> None:
    if not REPRODUCTION.exists():
        raise SystemExit("run `reproduce` first")
    if json.loads(REPRODUCTION.read_text())["verdict"] != "STAGE_A_REPRODUCED":
        raise SystemExit("Stage A was not reproduced; refusing to freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    if SPEC.exists():
        raise SystemExit(f"{SPEC} already frozen; refusing to overwrite")

    cfg = screen.config_for("m5")
    cutoffs = conf.frozen_cutoffs_from_artifact()
    stage_a = set(pd.read_csv(screen.OUT / "per_series_metrics.csv")
                  .query("dataset == 'm5'")["series_id"].astype(str))

    print("loading full M5 ...")
    data = conf.m5_full()
    print(f"describing {len(data['series_id'])} series (train split only) ...")
    table = conf.descriptor_frame(data, cfg)
    population = conf.build_population(table, stage_a, cutoffs)

    audit = {"candidate": 675, "control_pool": 5018, "stage_a_overlap": 0}
    got = {"candidate": population["n_candidate"],
           "control_pool": population["n_control_pool"],
           "stage_a_overlap": population["stage_a_overlap"]}
    if got != audit:
        raise screen.ScreenFailure(f"POPULATION_AUDIT_MISMATCH: expected {audit}, got {got}")

    frame = population["frame"]
    candidate = sorted(frame.loc[frame["group"] == "candidate", "series_id"].astype(str))
    control = sorted(frame.loc[frame["group"] == "control_pool", "series_id"].astype(str))

    spec = {
        "study": "H2 frozen-rule external validation on unseen M5 series",
        "frozen_at_utc": _utc(),
        "frozen_before_any_independent_prediction": True,
        "frozen_h2_cutoffs": cutoffs,
        "population": {k: v for k, v in population.items() if k != "frame"},
        "population_audit": {"expected": audit, "observed": got, "match": True},
        "sampling_rule": "none; every candidate and every control is evaluated",
        "candidate_id_sha256": _sha_of(candidate), "n_candidate": len(candidate),
        "control_id_sha256": _sha_of(control), "n_control": len(control),
        "stage_a_excluded_id_sha256": _sha_of(stage_a), "n_stage_a_excluded": len(stage_a),
        "models": {"provenance": "retrained on Stage A's 1200 series, verified against "
                                 "stage_a_results.json, then frozen",
                   "point": prereg.MODELS["point"], "hurdle": prereg.MODELS["hurdle"],
                   "model_seed": prereg.MODELS["canonical_model_seed"],
                   "independent_series_in_training": False},
        "training": dict(km_prereg.TRAINING), "split": dict(prereg.SPLITS["m5"]),
        "primary": {
            "population": "unmatched candidate vs unmatched control",
            "metric": "delta_relative = (RMSE_Point - RMSE_Hurdle) / RMSE_Point",
            "contrast": "mean(delta_relative | candidate) - mean(delta_relative | control)",
            "expected_sign": "negative",
            "matching_applied": False},
        "secondary": {
            "name": "OVERLAP_ADJUSTED_ASSOCIATION",
            "method": "logistic propensity on train-only covariates, overlap (ATO) weights",
            "covariates": list(MATCH_VARIABLES),
            "excluded_covariate": "rho_magnitude_train defines the contrast",
            "not_a_causal_effect": True,
            "balance_requirement": f"all weighted |SMD| <= {WEIGHTED_SMD_LIMIT}"},
        "bootstrap": dict(prereg.BOOTSTRAP),
        "git_commit": cli._git_commit(),
    }
    SPEC.write_text(json.dumps(spec, indent=2, default=str))
    frame[frame["group"] != "unused"].to_csv(OUT / "independent_population.csv", index=False)
    print(json.dumps({"population": got, "n_candidate": len(candidate),
                      "n_control": len(control)}, indent=2))
    print(f"froze {SPEC}")


# -------------------------------------------------------------- scoring ----


def test_windows_only(data: dict, cfg, stride: int):
    """screen.build_split's test branch, without building train or validation.

    build_split never masks the test targets and asserts every series is
    available before val_end, so the same guard is kept here; the train and
    validation windows it also builds are pure cost for a population that is
    only ever scored.
    """
    arrays = {"y": data["y"], "z": data["z"]}
    if data["available_from"].max() >= cfg.val_end:
        raise screen.ScreenFailure("a series becomes available inside the "
                                   "evaluation window")
    scale = train_scale(arrays, cfg)
    origins = km_train.stride_origins(cfg.val_end, cfg.length, cfg.horizon, stride)
    return make_windows(arrays, origins, cfg.val_end, cfg.length, cfg, scale)


def load_frozen(role: str, device):
    from ..om_factorization_killtest import models
    payload = torch.load(MODELS / f"{role}.pt", map_location=device, weights_only=False)
    model = models.BUILDERS[payload["builder_key"]](payload["lookback"],
                                                    payload["horizon"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _boot_ci(fn, rng, draws: int):
    values = np.array([fn(rng) for _ in range(draws)], dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 100:
        return (float("nan"), float("nan"))
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def summarise(delta_relative: np.ndarray, delta_raw: np.ndarray) -> dict:
    finite = delta_relative[np.isfinite(delta_relative)]
    return {"n": int(delta_relative.size), "n_finite": int(finite.size),
            "mean_delta_relative": float(finite.mean()),
            "median_delta_relative": float(np.median(finite)),
            "mean_delta_raw": float(np.nanmean(delta_raw)),
            "median_delta_raw": float(np.nanmedian(delta_raw)),
            # delta = RMSE_Point - RMSE_Hurdle; negative means Point won.
            "point_win_rate": float((finite < 0).mean()),
            "hurdle_win_rate": float((finite > 0).mean())}


def cmd_primary(_args) -> None:
    if not SPEC.exists():
        raise SystemExit("run `freeze` before scoring the independent series")
    spec = json.loads(SPEC.read_text())
    started = _utc()
    if not (spec["frozen_at_utc"] < started):
        raise screen.ScreenFailure("spec timestamp is not before scoring")

    population = pd.read_csv(OUT / "independent_population.csv")
    population["series_id"] = population["series_id"].astype(str)
    candidate = sorted(population.loc[population["group"] == "candidate", "series_id"])
    control = sorted(population.loc[population["group"] == "control_pool", "series_id"])
    if _sha_of(candidate) != spec["candidate_id_sha256"] or _sha_of(control) != spec["control_id_sha256"]:
        raise screen.ScreenFailure("population ids do not match the frozen spec")

    wanted = candidate + control
    group = np.array(["candidate"] * len(candidate) + ["control"] * len(control))
    cfg = screen.config_for("m5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading full M5 ...")
    full = conf.m5_full()
    index = pd.Index(full["series_id"]).get_indexer(wanted)
    if (index < 0).any():
        raise screen.ScreenFailure("a frozen id is absent from the source file")
    data = {"name": "m5", "y": full["y"][index], "z": full["z"][index],
            "series_id": np.asarray(wanted),
            "available_from": full["available_from"][index],
            "first_positive": full["first_positive"][index]}
    stage_a = set(pd.read_csv(screen.OUT / "per_series_metrics.csv")
                  .query("dataset == 'm5'")["series_id"].astype(str))
    if set(wanted) & stage_a:
        raise screen.ScreenFailure("STAGE_A_OVERLAP in the scored population")

    windows = test_windows_only(data, cfg, prereg.SPLITS["m5"]["test_origin_stride"])
    n_series = len(wanted)

    raw, metrics = {}, {}
    for role in ("point", "hurdle"):
        model, payload = load_frozen(role, device)
        if payload["trained_on"] != "stage_a_m5_1200":
            raise screen.ScreenFailure("frozen model was not trained on Stage A only")
        print(f"scoring {role} on {n_series} unseen series ...")
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

    _save_raw(raw, windows, data, group, OUT / "independent_raw_predictions.parquet")

    is_cand = (frame["group"] == "candidate").to_numpy()
    dr = frame["delta_relative"].to_numpy(float)
    dw = frame["delta_raw"].to_numpy(float)
    cand_summary = summarise(dr[is_cand], dw[is_cand])
    ctrl_summary = summarise(dr[~is_cand], dw[~is_cand])
    effect = cand_summary["mean_delta_relative"] - ctrl_summary["mean_delta_relative"]

    draws, seed = prereg.BOOTSTRAP["draws"], prereg.BOOTSTRAP["seed"]
    c_vals, k_vals = dr[is_cand], dr[~is_cand]
    c_vals, k_vals = c_vals[np.isfinite(c_vals)], k_vals[np.isfinite(k_vals)]

    def draw_effect(rng):
        a = c_vals[rng.integers(0, c_vals.size, c_vals.size)]
        b = k_vals[rng.integers(0, k_vals.size, k_vals.size)]
        return a.mean() - b.mean()

    def draw_winrate(rng):
        a = c_vals[rng.integers(0, c_vals.size, c_vals.size)]
        b = k_vals[rng.integers(0, k_vals.size, k_vals.size)]
        return (a < 0).mean() - (b < 0).mean()

    rng = np.random.default_rng(seed)
    effect_ci = _boot_ci(draw_effect, rng, draws)
    rng = np.random.default_rng(seed)
    cand_ci = _boot_ci(lambda r: c_vals[r.integers(0, c_vals.size, c_vals.size)].mean(), rng, draws)
    rng = np.random.default_rng(seed)
    ctrl_ci = _boot_ci(lambda r: k_vals[r.integers(0, k_vals.size, k_vals.size)].mean(), rng, draws)
    rng = np.random.default_rng(seed)
    winrate_ci = _boot_ci(draw_winrate, rng, draws)

    excludes_zero = (effect_ci[0] > 0) or (effect_ci[1] < 0)
    if effect >= 0:
        verdict = "H2_RULE_NOT_REPLICATED"
    elif (excludes_zero and cand_summary["point_win_rate"] > ctrl_summary["point_win_rate"]
          and cand_summary["mean_delta_relative"] < 0):
        verdict = "H2_RULE_STRONG_REPLICATION"
    else:
        verdict = "H2_RULE_PARTIAL_REPLICATION"

    report = {"verdict": verdict, "analysis": "PRIMARY, unmatched, frozen rule",
              "matching_applied": False,
              "candidate": cand_summary | {"mean_ci": cand_ci},
              "control": ctrl_summary | {"mean_ci": ctrl_ci},
              "H2_rule_effect": float(effect), "H2_rule_effect_ci": effect_ci,
              "effect_ci_excludes_zero": bool(excludes_zero),
              "point_win_rate_difference": float(cand_summary["point_win_rate"]
                                                 - ctrl_summary["point_win_rate"]),
              "point_win_rate_difference_ci": winrate_ci,
              "bootstrap": dict(prereg.BOOTSTRAP),
              "n_degenerate_rmse_point": int(degenerate.sum()),
              "scored_at_utc": started, "finished_at_utc": _utc(),
              "spec_frozen_at_utc": spec["frozen_at_utc"],
              "device": str(device), "torch": torch.__version__,
              "git_commit": cli._git_commit()}
    (OUT / "primary_result.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in
                      ("verdict", "H2_rule_effect", "H2_rule_effect_ci")}, indent=2))


def _save_raw(raw: dict, windows, data: dict, group: np.ndarray, path) -> None:
    """Persist predict()'s output verbatim; nothing is recomputed."""
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
    frame.to_parquet(path, index=False)
    print(f"wrote {path.name}  rows={len(frame)}")


# ------------------------------------------------------------ secondary ----


def propensity_weights(design: np.ndarray, treated: np.ndarray):
    """Logistic propensity and overlap (ATO) weights.

    Overlap weights are w = 1 - e for the treated and w = e for the controls.
    They put the most weight where the two groups actually coexist, which is
    the honest response to a population whose top scale decile has 68
    candidates and 4 controls.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(design)
    model = LogisticRegression(penalty=None, max_iter=2000).fit(x, treated)
    e = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    return e, np.where(treated == 1, 1.0 - e, e)


def weighted_smd(values, weights, treated) -> float:
    t, c = treated == 1, treated == 0
    wt, wc = weights[t], weights[c]
    mt = np.average(values[t], weights=wt)
    mc = np.average(values[c], weights=wc)
    pooled = np.sqrt((values[t].var(ddof=1) + values[c].var(ddof=1)) / 2.0)
    return float((mt - mc) / pooled) if pooled > 0 else float("nan")


def cmd_secondary(_args) -> None:
    if not (OUT / "primary_result.json").exists():
        raise SystemExit("run `primary` first; secondary never precedes it")
    spec = json.loads(SPEC.read_text())
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
    frame = frame[np.isfinite(frame[list(MATCH_VARIABLES)].to_numpy(float)).all(axis=1)]

    design = frame[list(MATCH_VARIABLES)].to_numpy(float)
    treated = (frame["group"] == "candidate").to_numpy().astype(int)
    y = frame["delta_relative"].to_numpy(float)
    e, w = propensity_weights(design, treated)

    unweighted = {v: conf.smd(frame.loc[treated == 1, v].to_numpy(float),
                              frame.loc[treated == 0, v].to_numpy(float))
                  for v in MATCH_VARIABLES}
    balance = {v: weighted_smd(frame[v].to_numpy(float), w, treated) for v in MATCH_VARIABLES}
    worst = max(abs(v) for v in balance.values())
    ess = {g: float(w[treated == t].sum() ** 2 / (w[treated == t] ** 2).sum())
           for g, t in (("candidate", 1), ("control", 0))}

    report = {"analysis": "SECONDARY -- OVERLAP_ADJUSTED_ASSOCIATION",
              "not_a_causal_effect": True,
              "separate_from_primary": True,
              "n_used": int(len(frame)),
              "propensity": {"min": float(e.min()), "p25": float(np.quantile(e, .25)),
                             "median": float(np.median(e)), "p75": float(np.quantile(e, .75)),
                             "max": float(e.max()),
                             "candidate_median": float(np.median(e[treated == 1])),
                             "control_median": float(np.median(e[treated == 0])),
                             "control_above_candidate_p05": float(
                                 (e[treated == 0] > np.quantile(e[treated == 1], .05)).mean())},
              "effective_sample_size": ess,
              "unweighted_smd": unweighted,
              "weighted_smd": balance, "worst_abs_weighted_smd": float(worst),
              "balance_limit": WEIGHTED_SMD_LIMIT}

    if worst > WEIGHTED_SMD_LIMIT:
        report["verdict"] = "OVERLAP_DIAGNOSTIC_INCONCLUSIVE"
        report["reason"] = f"worst weighted |SMD| {worst:.4f} > {WEIGHTED_SMD_LIMIT}"
    else:
        adjusted = (np.average(y[treated == 1], weights=w[treated == 1])
                    - np.average(y[treated == 0], weights=w[treated == 0]))
        rng = np.random.default_rng(prereg.BOOTSTRAP["seed"])

        def draw(rng):
            idx = rng.integers(0, len(frame), len(frame))
            t2 = treated[idx]
            if t2.sum() < 30 or (1 - t2).sum() < 30:
                return np.nan
            try:
                _, w2 = propensity_weights(design[idx], t2)
            except Exception:
                return np.nan
            y2 = y[idx]
            return (np.average(y2[t2 == 1], weights=w2[t2 == 1])
                    - np.average(y2[t2 == 0], weights=w2[t2 == 0]))

        ci = _boot_ci(draw, rng, prereg.BOOTSTRAP["draws"])
        report["verdict"] = "OVERLAP_ADJUSTED_ASSOCIATION_COMPUTED"
        report["overlap_adjusted_association"] = float(adjusted)
        report["overlap_adjusted_association_ci"] = ci
        report["ci_excludes_zero"] = bool(ci[0] > 0 or ci[1] < 0)
        report["bootstrap"] = dict(prereg.BOOTSTRAP) | {
            "note": "propensity refitted inside every draw"}

    (OUT / "secondary_overlap.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in report if k in
                      ("verdict", "worst_abs_weighted_smd",
                       "overlap_adjusted_association",
                       "overlap_adjusted_association_ci")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("H2 frozen-rule external validation")
    sub = parser.add_subparsers(required=True)
    for name, func in (("reproduce", cmd_reproduce), ("freeze", cmd_freeze),
                       ("primary", cmd_primary), ("secondary", cmd_secondary)):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
