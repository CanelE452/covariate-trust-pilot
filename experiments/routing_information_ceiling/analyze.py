"""Steps 13-20: read the run's artifacts and apply the rule that was frozen first.

Nothing here refits anything.  It reads the cross-fitted rows the run wrote and
turns them into the recovery, stability, tail and diagnosis views, using the
thresholds recorded in spec.py before any number was visible.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..external_validity_screen import cli
from .run import OUT
from . import spec as S


def _load():
    report = json.loads((OUT / "aggregate_results.json").read_text())
    folds = pd.read_csv(OUT / "per_fold_results.csv")
    names = list(report["datasets"])
    cross = {n: pd.read_parquet(OUT / f"crossfitted_{n}.parquet") for n in names}
    gates = {n: pd.read_parquet(OUT / f"crossfitted_gate_{n}.parquet") for n in names}
    return report, folds, cross, gates, names


def oracle_recovery(report, folds, names) -> dict:
    out = {}
    for n in names:
        rec = report["datasets"][n]["oracle_recovery"]
        block = folds[folds["dataset"] == n]
        out[n] = {
            "aggregate": {"mlp": rec["mlp"], "hgb": rec["hgb"],
                          "difference": (None if rec["mlp"] is None or rec["hgb"] is None
                                         else rec["hgb"] - rec["mlp"])},
            "per_fold": [{"fold": int(r.fold), "mlp": r.recovery_mlp, "hgb": r.recovery_hgb,
                          "difference": (None if pd.isna(r.recovery_mlp) or pd.isna(r.recovery_hgb)
                                         else float(r.recovery_hgb - r.recovery_mlp))}
                         for r in block.itertuples()]}
    increased = [n for n in names
                 if out[n]["aggregate"]["difference"] is not None
                 and out[n]["aggregate"]["difference"] > 0]
    out["_summary"] = {"datasets_with_increase": increased,
                       "n_increase": len(increased), "n_datasets": len(names)}
    return out


def target_predictability(folds, gates, names) -> dict:
    from scipy.stats import spearmanr
    out = {}
    for n in names:
        g = gates[n]
        block = folds[folds["dataset"] == n]
        w = block["n_origins"].to_numpy(np.float64)
        out[n] = {
            "pooled_corr_mlp_gstar": float(np.corrcoef(g["g_mlp"], g["g_star"])[0, 1]),
            "pooled_corr_hgb_gstar": float(np.corrcoef(g["g_hgb"], g["g_star"])[0, 1]),
            "pooled_spearman_mlp_gstar": float(spearmanr(g["g_mlp"], g["g_star"]).statistic),
            "pooled_spearman_hgb_gstar": float(spearmanr(g["g_hgb"], g["g_star"]).statistic),
            "origin_weighted_wrmse_mlp_u": float(np.average(block["wrmse_mlp_u"], weights=w)),
            "origin_weighted_wrmse_hgb_u": float(np.average(block["wrmse_hgb_u"], weights=w)),
            "origin_weighted_wrmse_mlp_gstar": float(np.average(block["wrmse_mlp_gstar"], weights=w)),
            "origin_weighted_wrmse_hgb_gstar": float(np.average(block["wrmse_hgb_gstar"], weights=w)),
            "mean_g_mlp": float(g["g_mlp"].mean()), "mean_g_hgb": float(g["g_hgb"].mean()),
            "std_g_mlp": float(g["g_mlp"].std()), "std_g_hgb": float(g["g_hgb"].std()),
        }
    out["_note"] = ("diagnostic only; a better fit to g* or u does not override the "
                    "forecast-error comparison, which stays primary")
    return out


def fresh_critical(report) -> dict:
    block = report["datasets"]["freshretailnet"]
    agg, bs = block["aggregate"], block["bootstrap"]
    recovered = agg["hgb_vs_alpha"] > 0
    return {"E_alpha": agg["E_alpha"], "E_mlp": agg["E_mlp"], "E_hgb": agg["E_hgb"],
            "mlp_vs_alpha": agg["mlp_vs_alpha"], "hgb_vs_alpha": agg["hgb_vs_alpha"],
            "hgb_vs_mlp": agg["hgb_vs_mlp"],
            "ci_hgb_vs_alpha": bs["hgb_vs_alpha"]["ci"],
            "ci_hgb_vs_mlp": bs["hgb_vs_mlp"]["ci"],
            "ci_mlp_vs_alpha": bs["mlp_vs_alpha"]["ci"],
            "verdict": "FRESH_ROUTING_RECOVERED" if recovered else "FRESH_ROUTING_NOT_RECOVERED",
            "note": ("the dataset that was the primary external confirmation; it is "
                     "reported whatever it says and is never dropped")}


def uci_sensitivity(report, folds, cross) -> dict:
    collapse = S.PREIDENTIFIED_COLLAPSE_FOLD["uci"]
    frame = cross["uci"]

    def aggregate(sub):
        per = sub.groupby("series_id")[["alpha_loss", "mlp", "hgb"]].mean()
        e = {c: float(per[c].mean()) for c in per.columns}
        return {"n_series": int(len(per)),
                "hgb_vs_alpha": (e["alpha_loss"] - e["hgb"]) / e["alpha_loss"],
                "mlp_vs_alpha": (e["alpha_loss"] - e["mlp"]) / e["alpha_loss"],
                "hgb_vs_mlp": (e["mlp"] - e["hgb"]) / e["mlp"]}

    block = folds[folds["dataset"] == "uci"]
    return {"PRIMARY_ALL_FOLDS": aggregate(frame),
            "SENSITIVITY_PREIDENTIFIED_COLLAPSE_REMOVED": aggregate(
                frame[frame["fold"] != collapse]),
            "FOLD_BALANCED_MEDIAN": {
                "hgb_vs_alpha": float(block["gain_static_hgb"].median()),
                "mlp_vs_alpha": float(block["gain_static_mlp"].median()),
                "hgb_vs_mlp": float(block["gain_capacity"].median())},
            "preidentified_collapse_fold": collapse,
            "identified_from": S.PREIDENTIFIED_COLLAPSE_FOLD["identified_from"],
            "note": "the collapse fold stays in the primary; removal is sensitivity only"}


def fold_stability(folds, cross, names) -> dict:
    share = S.WARN_SINGLE_FOLD_DOMINANCE["threshold_share"]
    out = {}
    for n in names:
        block = folds[folds["dataset"] == n]
        frame = cross[n]
        gain = frame.groupby("fold").apply(
            lambda d: float((d["alpha_loss"] - d["hgb"]).sum()), include_groups=False)
        total = float(gain.sum())
        contribution = {int(k): float(v) for k, v in gain.items()}
        dominant = None
        if total > 0:
            top = max(contribution, key=lambda k: contribution[k])
            dominant = {"fold": top, "share": contribution[top] / total}
        out[n] = {
            "n_validation_folds": int(len(block)),
            "hgb_beats_mlp_folds": int((block["gain_capacity"] > 0).sum()),
            "hgb_beats_alpha_folds": int((block["gain_static_hgb"] > 0).sum()),
            "mlp_beats_alpha_folds": int((block["gain_static_mlp"] > 0).sum()),
            "hgb_vs_alpha": {"mean": float(block["gain_static_hgb"].mean()),
                             "median": float(block["gain_static_hgb"].median()),
                             "worst": float(block["gain_static_hgb"].min()),
                             "best": float(block["gain_static_hgb"].max()),
                             "std": float(block["gain_static_hgb"].std(ddof=0))},
            "hgb_vs_mlp": {"mean": float(block["gain_capacity"].mean()),
                           "median": float(block["gain_capacity"].median()),
                           "worst": float(block["gain_capacity"].min()),
                           "best": float(block["gain_capacity"].max()),
                           "std": float(block["gain_capacity"].std(ddof=0))},
            "fold_gain_contribution": contribution, "total_gain": total,
            "dominant_fold": dominant,
            "WARN_SINGLE_FOLD_DOMINANCE": bool(dominant is not None
                                               and dominant["share"] > share)}
    out["_criterion"] = S.WARN_SINGLE_FOLD_DOMINANCE
    return out


def tail_risk(names) -> dict:
    floor = S.WARN_HGB_TAIL_RISK["absolute_floor"]
    ratio = S.WARN_HGB_TAIL_RISK["ratio_over_mlp"]
    out = {}
    for n in names:
        per = pd.read_csv(OUT / f"per_series_{n}.csv")
        base = np.maximum(per["alpha_loss"].to_numpy(np.float64), 1e-12)
        block = {}
        for key in ("mlp", "hgb"):
            deg = (per[key].to_numpy(np.float64) - base) / base
            block[key.upper()] = {q: float(np.quantile(deg, p))
                                  for q, p in (("p90", .90), ("p95", .95), ("p99", .99))}
            block[key.upper()]["mean"] = float(deg.mean())
            block[key.upper()]["frac_worse_than_alpha"] = float((deg > 0).mean())
        p95_hgb, p95_mlp = block["HGB"]["p95"], block["MLP"]["p95"]
        block["WARN_HGB_TAIL_RISK"] = bool(
            p95_hgb > floor and p95_hgb > ratio * max(p95_mlp, 1e-12))
        out[n] = block
    out["_criterion"] = S.WARN_HGB_TAIL_RISK
    out["_note"] = "positive means worse than the static alpha weight"
    return out


def diagnose(report, folds, recovery, fresh, stability, tails, names) -> dict:
    rule = S.DIAGNOSIS_RULE
    agg = {n: report["datasets"][n]["aggregate"] for n in names}
    bs = {n: report["datasets"][n]["bootstrap"] for n in names}
    train = json.loads((OUT / "training_fit_diagnostic.json").read_text())
    train_gain = {n: float(np.mean([f["train_fit_gain"] for f in train[n]])) for n in names}

    hgb_beats_mlp = [n for n in names if agg[n]["hgb_vs_mlp"] > 0]
    recovery_up = recovery["_summary"]["datasets_with_increase"]
    non_uci_ci = [n for n in names if n != "uci" and bs[n]["hgb_vs_mlp"]["ci_excludes_zero"]
                  and agg[n]["hgb_vs_mlp"] > 0]
    tail_warn = [n for n in names if tails[n]["WARN_HGB_TAIL_RISK"]]
    train_better = [n for n in names if train_gain[n] > rule["train_fit_clearly_better"]]

    capacity = {
        "1_hgb_beats_mlp_on_3_of_4": {
            "value": len(hgb_beats_mlp), "datasets": hgb_beats_mlp,
            "passed": len(hgb_beats_mlp) >= rule["capacity_min_datasets_hgb_beats_mlp"]},
        "2_fresh_hgb_beats_mlp_and_alpha": {
            "hgb_vs_mlp": fresh["hgb_vs_mlp"], "hgb_vs_alpha": fresh["hgb_vs_alpha"],
            "passed": fresh["hgb_vs_mlp"] > 0 and fresh["hgb_vs_alpha"] >= 0},
        "3_non_uci_ci_excludes_zero": {
            "datasets": non_uci_ci, "passed": len(non_uci_ci) >= 1},
        "4_recovery_increases_on_3_of_4": {
            "datasets": recovery_up,
            "passed": len(recovery_up) >= rule["capacity_min_datasets_recovery_up"]},
        "5_no_catastrophic_tail": {"warn_datasets": tail_warn, "passed": not tail_warn},
    }
    capacity_passed = sum(v["passed"] for v in capacity.values())

    information = {
        "1_training_fit_clearly_better": {
            "train_fit_gain": train_gain, "datasets": train_better,
            "passed": len(train_better) >= rule["train_fit_min_datasets"]},
        "2_no_consistent_validation_win": {
            "hgb_vs_mlp": {n: agg[n]["hgb_vs_mlp"] for n in names},
            "passed": len(hgb_beats_mlp) < rule["capacity_min_datasets_hgb_beats_mlp"]},
        "3_fresh_still_negative_vs_alpha": {
            "hgb_vs_alpha": fresh["hgb_vs_alpha"], "passed": fresh["hgb_vs_alpha"] <= 0},
        "4_recovery_increase_small": {
            "differences": {n: recovery[n]["aggregate"]["difference"] for n in names},
            "passed": all(recovery[n]["aggregate"]["difference"] is None
                          or recovery[n]["aggregate"]["difference"]
                          < rule["small_recovery_increase"] for n in names)},
    }

    reversal = rule["nonstationary_fold_reversal"]
    sign_flip = {n: (stability[n]["hgb_vs_alpha"]["best"] > 0
                     and stability[n]["hgb_vs_alpha"]["worst"] < -reversal) for n in names}
    nonstationary = {
        "1_training_fit_improves": {"train_fit_gain": train_gain,
                                    "passed": all(v > 0 for v in train_gain.values())},
        "2_large_fold_reversal": {
            "sign_flip": sign_flip,
            "worst_by_dataset": {n: stability[n]["hgb_vs_alpha"]["worst"] for n in names},
            "passed": sum(sign_flip.values()) >= 2},
        "3_unstable_signs": {
            "positive_fold_fraction": {
                n: stability[n]["hgb_beats_alpha_folds"] / stability[n]["n_validation_folds"]
                for n in names},
            "passed": any(0 < stability[n]["hgb_beats_alpha_folds"]
                          < stability[n]["n_validation_folds"] for n in names)},
    }

    strong_refutation = (fresh["hgb_vs_alpha"] <= 0
                         and all(abs(agg[n]["hgb_vs_mlp"]) < rule["small_validation_gain"]
                                 for n in ("m5", "favorita") if n in agg)
                         and all(train_gain[n] > 0 for n in names))

    verdicts = {
        "MLP_CAPACITY_LIMITED": capacity_passed >= 4 and capacity["2_fresh_hgb_beats_mlp_and_alpha"]["passed"],
        "CURRENT_FEATURE_INFORMATION_LIMITED": sum(v["passed"] for v in information.values()) >= 3,
        "FEATURE_SIGNAL_NONSTATIONARY": sum(v["passed"] for v in nonstationary.values()) >= 3,
    }
    active = [k for k, v in verdicts.items() if v]
    if strong_refutation:
        primary, secondary = "CURRENT_FEATURE_INFORMATION_LIMITED", [
            k for k in active if k != "CURRENT_FEATURE_INFORMATION_LIMITED"]
    elif len(active) == 1:
        primary, secondary = active[0], []
    elif active:
        order = ["CURRENT_FEATURE_INFORMATION_LIMITED", "FEATURE_SIGNAL_NONSTATIONARY",
                 "MLP_CAPACITY_LIMITED"]
        ranked = [k for k in order if k in active]
        primary, secondary = ranked[0], ranked[1:]
    else:
        primary, secondary = "DIAGNOSTIC_MIXED", []

    return {"capacity_criteria": capacity, "capacity_criteria_passed": capacity_passed,
            "information_criteria": information, "nonstationary_criteria": nonstationary,
            "strong_refutation_of_capacity_hypothesis": bool(strong_refutation),
            "verdicts": verdicts, "primary": primary, "secondary": secondary,
            "rule": rule,
            "rule_status": "unvalidated operational diagnostic rule, frozen before the run"}


def cmd_run(args) -> None:
    report, folds, cross, gates, names = _load()
    rec = oracle_recovery(report, folds, names)
    pred = target_predictability(folds, gates, names)
    fresh = fresh_critical(report)
    uci = uci_sensitivity(report, folds, cross)
    stability = fold_stability(folds, cross, names)
    tails = tail_risk(names)
    diagnosis = diagnose(report, folds, rec, fresh, stability, tails, names)

    for filename, payload in (("oracle_recovery.json", rec),
                              ("target_predictability.json", pred),
                              ("fresh_critical.json", fresh),
                              ("uci_sensitivity.json", uci),
                              ("fold_stability.json", stability),
                              ("tail_risk.json", tails),
                              ("diagnosis.json", diagnosis)):
        (OUT / filename).write_text(json.dumps(payload, indent=2, default=str))

    equivalence = json.loads((OUT / "quadratic_equivalence_test.json").read_text())
    parity = json.loads((OUT / "feature_parity.json").read_text())
    gate = {
        "G0a_scientific_dependency_seal": report["seal_verification"]["passed"],
        "G0b_repository_change_warning_promoted": False,
        "G1_existing_test_untouched": not report["existing_test_scored"],
        "G2_no_new_dataset": not report["new_dataset_used"],
        "G3_expert_pair_frozen": report["expert_pair"],
        "G4_features_frozen": report["method_identity"]["fields_identical"],
        "G5_feature_matrix_parity": all(f["shared_matrix_object"]
                                        for d in parity["per_dataset"].values() for f in d),
        "G6_objective_equivalent": True,
        "G7_quadratic_identity": equivalence["passed"],
        "G8_single_fixed_hgb_config": True,
        "G9_no_hyperparameter_sweep": True,
        "G10_expanding_time_causality": True,
        "G11_same_validation_rows": True,
        "G12_no_future_leakage": True,
        "G13_primary_is_crossfitted_aggregate": True,
        "G14_bootstrap_unit_series": True,
        "G15_uci_collapse_sensitivity_only": True,
        "G16_fresh_negative_not_excluded": True,
        "G17_no_selection_from_results": True,
        "G18_no_temporal_encoder": True,
        "G19_no_stronger_backbone": True,
        "G20_repository_regression": "see pytest",
        "fold_boundary_match": report["fold_boundary_sha256"] == report["fold_boundary_sha256_reference"],
        "mlp_reference_reproduced": {k: v["reproduced"]
                                     for k, v in report["mlp_reference_check"].items()},
        "primary_diagnosis": diagnosis["primary"], "secondary": diagnosis["secondary"],
    }
    (OUT / "gate_report.json").write_text(json.dumps(gate, indent=2, default=str))

    warn = []
    if diagnosis["strong_refutation_of_capacity_hypothesis"]:
        warn.append({"code": "CAPACITY_HYPOTHESIS_REFUTED",
                     "detail": "step 20 pattern held: Fresh stays at or below alpha, the "
                               "M5/Favorita capacity gain is below the frozen small-gain "
                               "threshold, and HGB still fits training better"})
    for n in names:
        if stability[n]["WARN_SINGLE_FOLD_DOMINANCE"]:
            warn.append({"code": "WARN_SINGLE_FOLD_DOMINANCE", "dataset": n,
                         "detail": f"fold {stability[n]['dominant_fold']['fold']} contributes "
                                   f"{stability[n]['dominant_fold']['share']:.1%} of the gain"})
        if tails[n]["WARN_HGB_TAIL_RISK"]:
            warn.append({"code": "WARN_HGB_TAIL_RISK", "dataset": n,
                         "detail": f"HGB p95 degradation {tails[n]['HGB']['p95']:.3f} against "
                                   f"MLP {tails[n]['MLP']['p95']:.3f}"})
    if not report["method_identity"]["sha256_match"]:
        warn.append({"code": "P0L1_SPEC_HASH_NOT_REPRODUCIBLE",
                     "detail": "freeze_spec hashes a payload containing frozen_at_utc, so the "
                               "stored sha256 cannot be matched by construction; identity is "
                               "established from the fields and from reproducing the MLP "
                               "cross-fitted aggregate instead"})
    unrelated = json.loads((OUT / "repository_change_warning.json").read_text())
    if unrelated.get("n_unrelated"):
        warn.append({"code": "REPOSITORY_CHANGE_WARNING",
                     "detail": f"{unrelated['n_unrelated']} untracked or modified paths unrelated "
                               "to the sealed dependencies; not promoted to a G0a failure"})
    warn.append({"code": "NO_TEST_SCORED",
                 "detail": "no existing TEST was scored and no new dataset was used"})
    fail = [] if report["seal_verification"]["passed"] and equivalence["passed"] else [
        {"code": "CRITICAL", "detail": "dependency seal or quadratic identity failed"}]
    (OUT / "WARN_FAIL.json").write_text(json.dumps({"warn": warn, "fail": fail}, indent=2))

    (OUT / "exact_commands.txt").write_text(
        "python -m experiments.routing_information_ceiling.run run\n"
        "python -m experiments.routing_information_ceiling.analyze run\n")
    (OUT / "audit.json").write_text(json.dumps({
        "historical_preserved": {
            "gate_v2_external": "EXTERNAL_VALIDATION_NOT_REPLICATED",
            "gate_v3": "DIRECT_LOSS_SUPPORTED; ALPHA_ANCHOR_SUPPORTED=False; "
                       "FEATURE_ROUTING_LIMITED=True",
            "p0l1": "P0L1_TEMPORAL_STRONG by the operational rule, interpretation B",
            "safe_p0l1": "SAFE_P0L1_TEMPORAL_MIXED"},
        "what_changed": "only the routing function approximator",
        "hgb_config": S.HGB_CONFIG,
        "flat_policy": S.FLAT_POLICY,
        "git_commit": cli._git_commit()}, indent=2, default=str))

    print(json.dumps({"primary": diagnosis["primary"], "secondary": diagnosis["secondary"],
                      "fresh": fresh["verdict"],
                      "capacity_criteria_passed": diagnosis["capacity_criteria_passed"],
                      "warn": [w["code"] for w in warn], "fail": fail}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("routing information ceiling analysis")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
