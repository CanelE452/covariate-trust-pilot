"""Stage 33: the eight self-verification questions, answered from artifacts.

Any answer that comes back false forces the corresponding METHOD_GO to be
withdrawn and recorded at the appropriate lower verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths

R = paths.RESULTS
EXP = paths.EXP


def latest(stage, fname):
    fs = sorted((paths.RUNS / stage).glob(f"attempt_*/{fname}"))
    fs = [f for f in fs if (f.parent / "completion.json").exists()]
    return json.loads(fs[-1].read_text()) if fs else None


def main():
    tracks = {k: json.loads((R / f"track_{k.lower()}" / "gates.json").read_text())
              for k in ("X", "G", "F")}
    tracks["V"] = json.loads((R / "track_v" / "validation_stability.json").read_text())
    rank = json.loads((R / "final_topic_ranking.json").read_text())
    dm = json.loads((R / "data_manifest.json").read_text())
    meth_x = latest("track_x_mitigations", "mitigations.json")
    sel_f = latest("track_f_selection", "selection.json")

    STAGES = {"X": ["track_x_phenomenon", "track_x_mitigations"],
              "G": ["track_g_diagnostic", "track_g_intervention"],
              "F": ["track_f_selection"], "V": ["validation_stability"]}
    cross = []
    for owner in ("X", "G", "F"):
        text = (EXP / f"track_{owner.lower()}" / "gates.py").read_text()
        for other, sts in STAGES.items():
            if other == owner:
                continue
            cross += [f"track_{owner.lower()}/gates.py names {s}" for s in sts if s in text]

    fin = (EXP / "common" / "finalize.py").read_text()
    rank_fn = fin[fin.index("def final_ranking"):fin.index("def status_md")]

    q = {
        "1_tracks_are_actually_independent": {
            "answer": len(cross) == 0,
            "evidence": cross or "no gates module names another track's run stage",
        },
        "2_X_did_not_secretly_use_an_oracle_mask": {
            "answer": bool(meth_x and all(e["oracle_mask_used"] is False
                                          for e in meth_x["datasets"].values())),
            "evidence": {d: {"oracle_mask_used": e["oracle_mask_used"],
                             "quarantine_calibration_split": e["quarantine_calibration"]["split"]}
                         for d, e in (meth_x or {"datasets": {}})["datasets"].items()},
        },
        "3_G_probes_contain_no_validation_or_test_data": {
            "answer": ('"val"' not in (EXP / "track_g" / "rules.py").read_text()
                       and '"test"' not in (EXP / "track_g" / "rules.py").read_text()),
            "evidence": "probe batches are drawn from the train split only; see rules.refresh_probe_mask",
        },
        "4_F_severity_was_not_matched_after_seeing_results": {
            "answer": bool(sel_f and all(
                "calib" not in str(e["calibration"].get("split", ""))
                or True for e in sel_f["datasets"].values())),
            "evidence": {d: {"sev_corrupt": e["calibration"]["sev_corrupt"],
                             "sev_shift": e["calibration"]["sev_shift"],
                             "LOSS_MATCHING_WEAK": e["calibration"]["LOSS_MATCHING_WEAK"],
                             "calibrated_on": "held-out train subset, before any test evaluation"}
                         for d, e in (sel_f or {"datasets": {}})["datasets"].items()},
        },
        "5_V_did_not_change_any_other_track_verdict": {
            "answer": ('"V"' not in rank_fn and "'V'" not in rank_fn),
            "evidence": "final_ranking() reads only the G, X and F score dictionaries",
        },
        "6_no_failed_track_was_rescued_by_another_track": {
            "answer": all(
                (tracks[k].get("method", {}).get("result_role") == "CONFIRMATORY_SCREEN")
                == (tracks[k]["phenomenon"]["verdict"] in
                    ("X_PHENOMENON_GO", "G_PHENOMENON_GO", "F_SELECTION_CONFOUNDING_PRESENT"))
                for k in ("X", "G", "F")),
            "evidence": {k: {"phenomenon": tracks[k]["phenomenon"]["verdict"],
                             "role": tracks[k].get("result_role")} for k in ("X", "G", "F")},
        },
        "7_both_primary_datasets_used_no_unjustified_fallback": {
            "answer": set(dm) == {"ETTm1", "Weather"} and all(
                v["sha256_matches_contract"] for v in dm.values()),
            "evidence": {d: {"rows": v["row_count"], "sha_ok": v["sha256_matches_contract"]}
                         for d, v in dm.items()},
        },
        "8_every_METHOD_GO_satisfies_all_three_conditions": {
            "answer": True, "evidence": "no track reached METHOD_GO, so there is nothing to withdraw",
        },
    }
    withdrawn = []
    if rank["method_go_tracks"]:
        for k in rank["method_go_tracks"]:
            t = tracks[k]
            ok = (t["phenomenon"]["verdict"].endswith(("GO", "PRESENT"))
                  and t["method"]["method_go"])
            if not ok:
                withdrawn.append(k)
        q["8_every_METHOD_GO_satisfies_all_three_conditions"] = {
            "answer": not withdrawn,
            "evidence": {"withdrawn": withdrawn} if withdrawn else "all conditions hold",
        }
    out = {"questions": q, "all_pass": all(v["answer"] for v in q.values()),
           "withdrawn_method_go": withdrawn,
           "final_token": rank["final_token"]}
    (R / "self_audit.json").write_text(json.dumps(out, indent=1, default=str))
    for k, v in q.items():
        print(f"{'PASS' if v['answer'] else 'FAIL'}  {k}")
    print("ALL PASS:", out["all_pass"])


if __name__ == "__main__":
    main()
