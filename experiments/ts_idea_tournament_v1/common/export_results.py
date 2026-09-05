"""Export the compact per-track result files listed in the study contract.

The heavy raw payload stays under runs/ (never committed); results/ carries the
compact summaries plus every pre-analysis spec.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import paths
import attempts as A

R = paths.RESULTS
sys.path.insert(0, str(paths.EXP / "track_x"))
sys.path.insert(0, str(paths.EXP / "track_g"))
sys.path.insert(0, str(paths.EXP / "track_f"))


def latest(stage, fname):
    fs = sorted((paths.RUNS / stage).glob(f"attempt_*/{fname}"))
    fs = [f for f in fs if (f.parent / "completion.json").exists()]
    return json.loads(fs[-1].read_text()) if fs else None


def export_x():
    phen = latest("track_x_phenomenon", "phenomenon_raw.json")
    meth = latest("track_x_mitigations", "mitigations.json")
    d = R / "track_x"
    import spec as S
    A.write_json(d / "pre_analysis_spec.json",
                 {ds: S.build(ds) for ds in ("ETTm1", "Weather")})
    old = d / "pre_analysis_spec_Weather.json"
    if old.exists():
        old.unlink()
    if phen:
        A.write_json(d / "phenomenon.json", {
            "attempt": phen["attempt"],
            "datasets": {ds: {"spec_channels": e["spec_channels"],
                              "splice_verification": e["splice_verification"],
                              "summary": e["summary"],
                              "tqnet_over_dlinear_spillover": e["tqnet_over_dlinear_spillover"]}
                         for ds, e in phen["datasets"].items()},
            "note": ("Per-origin raw tables live under runs/ and are not committed; "
                     "every number here is reproducible from them."),
        })
    if meth:
        A.write_json(d / "methods.json", meth)


def export_g():
    diag = latest("track_g_diagnostic", "gradient_diagnostic.json")
    inter = latest("track_g_intervention", "intervention.json")
    d = R / "track_g"
    import run_diagnostic as RD
    A.write_json(d / "pre_analysis_spec.json", {
        "tasks": {ds: RD.tasks_for(ds) for ds in ("ETTm1", "Weather")},
        "task_definition": "one output variable is one task; loss is MSE over batch and all 96 steps",
        "checkpoint_tags": RD.CKPT_TAGS,
        "checkpoint_rule": ("early = 25% and middle = 50% of the realised schedule, "
                            "best = best validation MSE; never chosen by looking at a result"),
        "harm_threshold": RD.HARM_THRESHOLD,
        "n_exact_top_affinity": RD.N_EXACT_TOP, "n_exact_random": RD.N_EXACT_RANDOM,
        "probe_seed": RD.PROBE_SEED,
        "probe_rule": ("optimisation and probe blocks are disjoint temporal blocks of the "
                       "TRAIN split, separated by at least one full window"),
        "shared_parameter_rule": (diag or {}).get("shared_parameter_audit", {})
                                 .get("ETTm1", {}).get("rule"),
    })
    if diag:
        A.write_json(d / "gradient_diagnostic.json", {
            "attempt": diag["attempt"], "tier": diag["tier"],
            "n_probe_pairs": diag["n_probe_pairs"], "harm_threshold": diag["harm_threshold"],
            "shared_parameter_audit": diag["shared_parameter_audit"],
            "checkpoints": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                            for k, v in diag["checkpoints"].items()},
            "note": "Per-pair rows live under runs/ and are not committed.",
        })
    if inter:
        A.write_json(d / "intervention.json", {
            "attempt": inter["attempt"], "arms": inter["arms"], "seeds": inter["seeds"],
            "runs": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                     for k, v in inter["runs"].items()},
            "wall_s": inter.get("wall_s"),
        })


def export_f():
    sel = latest("track_f_selection", "selection.json")
    d = R / "track_f"
    import windows as W
    import filters as SEL
    import run_selection as RS
    A.write_json(d / "pre_analysis_spec.json", {
        "model": "DLinear individual heads (fast model for Track F)",
        "class_fractions": {"CLEAN": 1 - W.CORRUPT_FRACTION - W.SHIFT_FRACTION,
                            "CORRUPTION": W.CORRUPT_FRACTION,
                            "LEGITIMATE_SHIFT": W.SHIFT_FRACTION},
        "class_seed": W.CLASS_SEED, "window_stride": W.WINDOW_STRIDE,
        "corruption_kinds": W.CORRUPTION_KINDS, "shift_kinds": W.SHIFT_KINDS,
        "shift_channel_fraction": W.SHIFT_CHANNEL_FRACTION,
        "shift_input_tail": W.SHIFT_INPUT_TAIL,
        "severity_candidates": W.SEVERITY_CANDIDATES,
        "severity_rule": ("smallest median training-loss gap between the two classes, "
                          "subject to both raising the median loss by at least 25% over "
                          "clean; calibrated on a held-out TRAIN subset only"),
        "removal_budget": SEL.REMOVAL_BUDGET,
        "methods": RS.METHODS,
        "coherence_rule": {"A_min_iqr": SEL.COHERENCE_A_MIN, "B_min_iqr": SEL.COHERENCE_B_MIN,
                           "channel_fraction": SEL.COHERENCE_CHANNEL_FRACTION},
        "rho_reference_rule": "trained only on a disjoint temporal holdout block of the training interval",
        "shifted_test_seed": RS.SHIFT_TEST_SEED,
        "adarho": {"n_B": 256, "n_b": 128, "n_r": 64, "reference_lr_divisor": 10.0,
                   "status": "PAPER_FAITHFUL_LOCAL_ADARHO"},
    })
    if sel:
        A.write_json(d / "selection_diagnostic.json", {
            "attempt": sel["attempt"], "methods": sel["methods"],
            "datasets": {ds: {"calibration": e["calibration"], "n_windows": e["n_windows"],
                              "class_counts": e["class_counts"],
                              "rho_reference": e["rho_reference"],
                              "removal_budget": e["removal_budget"],
                              "selection_diagnostic": e["selection_diagnostic"],
                              "controls": e["controls"]}
                         for ds, e in sel["datasets"].items()},
        })
        A.write_json(d / "retraining.json", {
            "attempt": sel["attempt"],
            "datasets": {ds: {"retraining": e["retraining"],
                              "n_test_windows": e["n_test_windows"],
                              "n_shifted_test_windows": e["n_shifted_test_windows"]}
                         for ds, e in sel["datasets"].items()},
        })


if __name__ == "__main__":
    export_x()
    export_g()
    export_f()
    for f in sorted(R.rglob("*.json")):
        print(f"{f.relative_to(R)}  {f.stat().st_size / 1024:.1f} KB")
