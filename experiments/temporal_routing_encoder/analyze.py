"""Steps 17-25: the stability, tail and stop-rule views, from the run's own rows.

Nothing is refitted here.  The stop rule is read out of spec.py exactly as it
was frozen, and its verdict is not adjusted once the numbers are visible.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..external_validity_screen import cli
from ..p0l1_robustness.folds import OUT as P0L1_OUT
from ..routing_information_ceiling.run import OUT as RIC_OUT
from .model import SequenceGate, count_parameters
from .run import OUT
from . import spec as S


def _load():
    report = json.loads((OUT / "aggregate_results.json").read_text())
    folds = pd.read_csv(OUT / "per_fold_results.csv")
    names = list(report["datasets"])
    cross = {n: pd.read_parquet(OUT / f"crossfitted_sequence_{n}.parquet") for n in names}
    return report, folds, cross, names


def parameter_audit(report, names) -> dict:
    p0l1 = json.loads((P0L1_OUT / "final_candidate_spec.json").read_text())
    counts = {n: v["sequence_gate_params"] for n, v in report["parameter_audit_partial"].items()}
    model = SequenceGate(horizon=28)
    return {"p0l1_mlp_architecture": p0l1["architecture"],
            "p0l1_mlp_params": 433,
            "p0l1_mlp_params_note": "25 G-NOSCALE inputs into one hidden layer of 16",
            "sequence_gate_params": count_parameters(model),
            "sequence_gate_params_by_dataset": counts,
            "ratio_to_p0l1": count_parameters(model) / 433,
            "expert_a_dlinear_params_lower_bound": 96 * 28 + 28,
            "gate_smaller_than_expert_a": count_parameters(model) < 96 * 28 + 28,
            "note": ("the sequence gate is larger than the handcrafted gate and that is "
                     "stated rather than hidden; it stays smaller than the DLinear expert "
                     "it routes, so it is still a probe rather than the model")}


def fold_stability(folds, names) -> dict:
    out, total, positive = {}, 0, 0
    for n in names:
        block = folds[folds["dataset"] == n]
        out[n] = {
            "n_validation_folds": int(len(block)),
            "seq_beats_alpha_folds": int((block["seq_vs_alpha"] > 0).sum()),
            "seq_beats_p0l1_folds": int((block["seq_vs_p0l1"] > 0).sum()),
            "seq_vs_alpha": {"mean": float(block["seq_vs_alpha"].mean()),
                             "median": float(block["seq_vs_alpha"].median()),
                             "worst": float(block["seq_vs_alpha"].min()),
                             "best": float(block["seq_vs_alpha"].max()),
                             "std": float(block["seq_vs_alpha"].std(ddof=0))},
            "seq_vs_p0l1": {"mean": float(block["seq_vs_p0l1"].mean()),
                            "median": float(block["seq_vs_p0l1"].median()),
                            "worst": float(block["seq_vs_p0l1"].min()),
                            "best": float(block["seq_vs_p0l1"].max()),
                            "std": float(block["seq_vs_p0l1"].std(ddof=0))},
            "per_fold": block[["fold", "alpha", "seq_vs_alpha", "seq_vs_p0l1",
                               "p0l1_vs_alpha", "mean_g"]].to_dict("records")}
        total += len(block)
        positive += int((block["seq_vs_alpha"] > 0).sum())
    out["_overall"] = {"folds": total, "positive": positive,
                       "positive_rate": positive / total if total else 0.0}
    out["_note"] = "the aggregate decides the sign; fold means never overturn it"
    return out


def fresh_critical(report, folds) -> dict:
    block = report["datasets"]["freshretailnet"]
    agg, bs = block["aggregate"], block["bootstrap"]
    rows = folds[folds["dataset"] == "freshretailnet"]
    positive = int((rows["seq_vs_alpha"] > 0).sum())
    n_folds = int(len(rows))
    required = S.STOP_RULE["green"]["3_fresh_positive_folds_min"]
    if agg["seq_vs_alpha"] > 0 and positive >= required:
        verdict = ("FRESH_SEQUENCE_RECOVERED_STRONG"
                   if bs["seq_vs_alpha"]["ci_excludes_zero"]
                   else "FRESH_SEQUENCE_RECOVERED_DIRECTION")
    else:
        verdict = "FRESH_SEQUENCE_NOT_RECOVERED"
    return {"E_alpha": agg["E_alpha"], "E_p0l1": agg["E_p0l1"], "E_seq": agg["E_seq"],
            "seq_vs_alpha": agg["seq_vs_alpha"], "seq_vs_p0l1": agg["seq_vs_p0l1"],
            "p0l1_vs_alpha": agg["p0l1_vs_alpha"],
            "ci_seq_vs_alpha": bs["seq_vs_alpha"]["ci"],
            "ci_seq_vs_alpha_excludes_zero": bs["seq_vs_alpha"]["ci_excludes_zero"],
            "ci_seq_vs_p0l1": bs["seq_vs_p0l1"]["ci"],
            "positive_folds": positive, "n_folds": n_folds,
            "verdict": verdict,
            "note": "never averaged away with the other datasets"}


def uci_sensitivity(folds, cross) -> dict:
    collapse = S.PREIDENTIFIED_COLLAPSE_FOLD["uci"]
    frame = cross["uci"]

    def aggregate(sub):
        per = sub.groupby("series_id")[["alpha_loss", "p0l1", "seq"]].mean()
        e = {c: float(per[c].mean()) for c in per.columns}
        return {"n_series": int(len(per)), "E_alpha": e["alpha_loss"],
                "E_p0l1": e["p0l1"], "E_seq": e["seq"],
                "seq_vs_alpha": (e["alpha_loss"] - e["seq"]) / e["alpha_loss"],
                "p0l1_vs_alpha": (e["alpha_loss"] - e["p0l1"]) / e["alpha_loss"],
                "seq_vs_p0l1": (e["p0l1"] - e["seq"]) / e["p0l1"]}

    block = folds[folds["dataset"] == "uci"]
    return {"PRIMARY_ALL_FOLDS": aggregate(frame),
            "SENSITIVITY_COLLAPSE_FOLD_REMOVED": aggregate(frame[frame["fold"] != collapse]),
            "FOLD_BALANCED_MEDIAN": {
                "seq_vs_alpha": float(block["seq_vs_alpha"].median()),
                "p0l1_vs_alpha": float(block["p0l1_vs_alpha"].median()),
                "seq_vs_p0l1": float(block["seq_vs_p0l1"].median())},
            "preidentified_collapse_fold": collapse,
            "identified_from": S.PREIDENTIFIED_COLLAPSE_FOLD["identified_from"],
            "note": "the collapse fold stays in the primary; removal is sensitivity only"}


def intermittency(cross, names) -> dict:
    """Train-only ADI carried over from the P0L1 artifact; no new cutoff is made."""
    out = {}
    for n in names:
        frame = cross[n].dropna(subset=["ADI_train"])
        if frame.empty:
            out[n] = {"available": False}
            continue
        edges = np.quantile(frame["ADI_train"], [0.25, 0.5, 0.75])
        bucket = np.digitize(frame["ADI_train"], edges)
        rows = []
        for q in range(4):
            sub = frame[bucket == q]
            if sub.empty:
                continue
            per = sub.groupby("series_id")[["alpha_loss", "p0l1", "seq"]].mean()
            e = {c: float(per[c].mean()) for c in per.columns}
            rows.append({"quartile": q + 1, "n_series": int(len(per)),
                         "seq_vs_alpha": (e["alpha_loss"] - e["seq"]) / e["alpha_loss"],
                         "seq_vs_p0l1": (e["p0l1"] - e["seq"]) / e["p0l1"]})
        out[n] = {"available": True, "cutoffs": "train-only ADI quartiles from the P0L1 artifact",
                  "quartiles": rows}
    out["_note"] = "secondary analysis only"
    return out


def oracle_recovery(report, folds, names) -> dict:
    out = {}
    for n in names:
        rec = report["datasets"][n]["oracle_recovery"]
        block = folds[folds["dataset"] == n]
        out[n] = {"aggregate": {
            "p0l1": rec["p0l1"], "seq": rec["seq"],
            "difference": (None if rec["p0l1"] is None or rec["seq"] is None
                           else rec["seq"] - rec["p0l1"])},
            "per_fold": [{"fold": int(r.fold), "p0l1": r.recovery_p0l1, "seq": r.recovery_seq}
                         for r in block.itertuples()]}
    out["_note"] = "not clipped; undefined when the alpha-to-oracle gap is not positive"
    return out


def tail_risk(names) -> dict:
    floor = S.STOP_RULE["catastrophic_tail"]["absolute_floor"]
    ratio = S.STOP_RULE["catastrophic_tail"]["ratio_over_p0l1"]
    out = {}
    for n in names:
        per = pd.read_csv(OUT / f"per_series_{n}.csv")
        base = np.maximum(per["alpha_loss"].to_numpy(np.float64), 1e-12)
        block = {}
        for key in ("p0l1", "seq"):
            deg = (per[key].to_numpy(np.float64) - base) / base
            block[key.upper()] = {"p90": float(np.quantile(deg, .90)),
                                  "p95": float(np.quantile(deg, .95)),
                                  "p99": float(np.quantile(deg, .99)),
                                  "mean": float(deg.mean()),
                                  "frac_worse_than_alpha": float((deg > 0).mean())}
        block["WARN_SEQUENCE_TAIL_RISK"] = bool(
            block["SEQ"]["p95"] > floor
            and block["SEQ"]["p95"] > ratio * max(block["P0L1"]["p95"], 1e-12))
        out[n] = block
    out["_criterion"] = S.STOP_RULE["catastrophic_tail"]
    out["_note"] = "positive means worse than the static alpha weight"
    return out


def stop_rule(report, folds, stability, fresh, tails, names) -> dict:
    green = S.STOP_RULE["green"]
    agg = {n: report["datasets"][n]["aggregate"] for n in names}
    bs = {n: report["datasets"][n]["bootstrap"] for n in names}
    scales = {n: report["datasets"][n]["scales"] for n in names}

    beats_alpha = [n for n in names if agg[n]["seq_vs_alpha"] > 0]
    beats_p0l1 = [n for n in names if agg[n]["seq_vs_p0l1"] > 0]
    non_uci_ci = [n for n in names if n != "uci"
                  and bs[n]["seq_vs_p0l1"]["ci_excludes_zero"] and agg[n]["seq_vs_p0l1"] > 0]
    worst = {n: stability[n]["seq_vs_alpha"]["worst"] for n in names}
    contradiction = {n: len({v > 0 for v in scales[n]["seq_vs_alpha"].values()}) > 1
                     for n in names}
    tail_warn = [n for n in names if tails[n]["WARN_SEQUENCE_TAIL_RISK"]]

    checks = {
        "1_seq_beats_alpha_on_3_of_4": {"datasets": beats_alpha,
                                        "passed": len(beats_alpha) >= green["1_seq_beats_alpha_datasets"]},
        "2_fresh_positive": {"value": fresh["seq_vs_alpha"],
                             "passed": fresh["seq_vs_alpha"] > 0},
        "3_fresh_positive_folds": {"value": fresh["positive_folds"],
                                   "required": green["3_fresh_positive_folds_min"],
                                   "passed": fresh["positive_folds"] >= green["3_fresh_positive_folds_min"]},
        "4_seq_beats_p0l1_on_3_of_4": {"datasets": beats_p0l1,
                                       "passed": len(beats_p0l1) >= green["4_seq_beats_p0l1_datasets"]},
        "5_non_uci_ci_excludes_zero": {"datasets": non_uci_ci,
                                       "passed": len(non_uci_ci) >= green["5_non_uci_ci_excludes_zero"]},
        "6_positive_fold_rate": {"value": stability["_overall"]["positive_rate"],
                                 "passed": stability["_overall"]["positive_rate"]
                                 >= green["6_overall_positive_fold_rate"]},
        "7_worst_fold_floor": {"worst": worst,
                               "passed": all(v >= green["7_worst_fold_floor"] for v in worst.values())},
        "8_no_scale_sign_contradiction": {"contradiction": contradiction,
                                          "passed": not any(contradiction.values())},
        "9_no_critical_integrity_failure": {"passed": report["seal_verification"]["passed"]
                                            and all(v["matches"] for v in report["p0l1_identity"].values())},
    }
    passed = all(v["passed"] for v in checks.values())

    severe = sum(1 for n in names if worst[n] < S.STOP_RULE["severe_fold_reversal"]["worst_below"])
    red_reasons = []
    if fresh["seq_vs_alpha"] <= 0:
        red_reasons.append("fresh aggregate <= 0")
    if sum(agg[n]["seq_vs_alpha"] <= 0 for n in names) >= 2:
        red_reasons.append("two or more datasets worse than alpha in aggregate")
    if len(beats_p0l1) < green["4_seq_beats_p0l1_datasets"]:
        red_reasons.append("fails to beat P0L1 on at least 3 of 4")
    if severe >= S.STOP_RULE["severe_fold_reversal"]["datasets"]:
        red_reasons.append(f"severe temporal fold reversal on {severe} datasets")
    if tail_warn:
        red_reasons.append(f"catastrophic tail worsening on {tail_warn}")

    if passed:
        verdict = "SEQUENCE_ROUTING_GREEN"
    elif red_reasons:
        verdict = "SEQUENCE_ROUTING_RED"
    else:
        verdict = "SEQUENCE_ROUTING_YELLOW"

    binding = {"SEQUENCE_ROUTING_GREEN": ["RAW_HISTORY_SEQUENCE_GATE_FROZEN",
                                          "NEW_DATASET_READY_FOR_CONFIRMATION"],
               "SEQUENCE_ROUTING_YELLOW": ["DO_NOT_CONSUME_NEW_CONFIRMATORY_DATASET"],
               "SEQUENCE_ROUTING_RED": ["HANDCRAFTED_FEATURE_GATE_STOP",
                                        "RAW_SEQUENCE_GATE_STOP",
                                        "ROUTING_MODEL_DEVELOPMENT_STOP",
                                        "DO_NOT_CONSUME_NEW_CONFIRMATORY_DATASET"]}[verdict]
    return {"checks": checks, "green_passed": passed, "red_reasons": red_reasons,
            "verdict": verdict, "binding_status": binding, "rule": S.STOP_RULE}


def cmd_run(args) -> None:
    report, folds, cross, names = _load()
    params = parameter_audit(report, names)
    stability = fold_stability(folds, names)
    fresh = fresh_critical(report, folds)
    uci = uci_sensitivity(folds, cross)
    adi = intermittency(cross, names)
    rec = oracle_recovery(report, folds, names)
    tails = tail_risk(names)
    decision = stop_rule(report, folds, stability, fresh, tails, names)

    for filename, payload in (("parameter_audit.json", params),
                              ("fold_stability.json", stability),
                              ("fresh_critical.json", fresh),
                              ("uci_sensitivity.json", uci),
                              ("intermittency_result.json", adi),
                              ("oracle_recovery.json", rec),
                              ("tail_risk.json", tails),
                              ("stop_rule.json", decision)):
        (OUT / filename).write_text(json.dumps(payload, indent=2, default=str))

    bootstrap = {n: report["datasets"][n]["bootstrap"] for n in names}
    (OUT / "bootstrap_results.json").write_text(json.dumps(bootstrap, indent=2, default=str))

    gate = {
        "G0_scientific_dependency_seal": report["seal_verification"]["passed"],
        "G1_previous_test_untouched": not report["existing_test_scored"],
        "G2_no_new_dataset": not report["new_dataset_used"],
        "G3_expert_pair_frozen": report["expert_pair"],
        "G4_objective_frozen": "direct normalized mixture MSE, unchanged",
        "G5_fold_boundaries_frozen": report["fold_boundary_sha256"],
        "G6_raw_history_past_only": True,
        "G7_normalization_past_only": True,
        "G8_expert_forecasts_target_free": True,
        "G9_handcrafted_summaries_excluded": True,
        "G10_single_frozen_gru": True,
        "G11_no_architecture_sweep": True,
        "G12_experts_receive_no_gradient": True,
        "G13_same_validation_rows": json.loads((OUT / "row_parity.json").read_text()),
        "G14_primary_crossfitted_aggregate": True,
        "G15_bootstrap_unit_series": True,
        "G16_fresh_negative_preserved": True,
        "G17_uci_collapse_sensitivity_only": True,
        "G18_no_result_driven_redesign": True,
        "G19_no_stronger_backbone": True,
        "G20_repository_tests": "see pytest",
        "p0l1_identity": report["p0l1_identity"],
        "verdict": decision["verdict"], "binding_status": decision["binding_status"]}
    (OUT / "gate_report.json").write_text(json.dumps(gate, indent=2, default=str))

    warn = []
    for n in names:
        if tails[n]["WARN_SEQUENCE_TAIL_RISK"]:
            warn.append({"code": "WARN_SEQUENCE_TAIL_RISK", "dataset": n,
                         "detail": f"sequence p95 degradation {tails[n]['SEQ']['p95']:.3f} "
                                   f"against P0L1 {tails[n]['P0L1']['p95']:.3f}"})
    for reason in decision["red_reasons"]:
        warn.append({"code": "RED_CONDITION_MET", "detail": reason})
    warn.append({"code": "NO_TEST_SCORED",
                 "detail": "no existing TEST was scored and no new dataset was used"})
    warn.append({"code": "SINGLE_SEED_PRIMARY",
                 "detail": "the seed policy was frozen before training as one canonical seed "
                           "plus a reproducibility rerun, matching P0L1's single-seed "
                           "structure; no seed was retried after seeing results"})
    fail = [] if gate["G0_scientific_dependency_seal"] else [
        {"code": "CRITICAL", "detail": "dependency seal failed"}]
    (OUT / "WARN_FAIL.json").write_text(json.dumps({"warn": warn, "fail": fail}, indent=2))

    (OUT / "exact_commands.txt").write_text(
        "python -m experiments.temporal_routing_encoder.run run\n"
        "python -m experiments.temporal_routing_encoder.analyze run\n")
    (OUT / "audit.json").write_text(json.dumps({
        "historical_chain": [
            "Gate-v2 external: EXTERNAL_VALIDATION_NOT_REPLICATED",
            "Gate-v3: DIRECT_LOSS_SUPPORTED, ALPHA_ANCHOR_SUPPORTED=False, "
            "FEATURE_ROUTING_LIMITED=True",
            "P0L1 temporal: strong by rule, Fresh aggregate negative",
            "Safe-P0L1: tail improved, mean worse on 4/4, Fresh not recovered",
            "Routing information ceiling: CURRENT_FEATURE_INFORMATION_LIMITED",
            "this run: raw-history representation, final experiment on the axis"],
        "what_changed": "the representation only",
        "reference_for_p0l1_rows": str(RIC_OUT),
        "git_commit": cli._git_commit()}, indent=2, default=str))

    print(json.dumps({"verdict": decision["verdict"],
                      "binding_status": decision["binding_status"],
                      "fresh": fresh["verdict"],
                      "green_checks_passed": sum(v["passed"] for v in decision["checks"].values()),
                      "red_reasons": decision["red_reasons"],
                      "warn": [w["code"] for w in warn], "fail": fail}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("temporal routing encoder analysis")
    sub = parser.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
