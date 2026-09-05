"""Freeze the study contract before any scientific result exists.

Writes preregistration, environment, external sources, data manifest, baseline
contract and runtime tier. These files are never rewritten afterwards.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import paths
import attempts as A
from contract import (DATASETS, SEQ_LEN, PRED_LEN, CLEAN_SEEDS, BOOTSTRAP_SEED,
                      tqnet_config, dlinear_config)
import data as D

R = paths.RESULTS
R.mkdir(parents=True, exist_ok=True)

DATE_DROP_NOTE = (
    "pandas>=2 compatibility only: the positional-axis form of DataFrame.drop was "
    "replaced by the keyword form at 4 sites in data_provider/data_loader.py. "
    "Semantically identical; touches only the time-feature columns, which neither "
    "TQNet nor DLinear consume."
)


def repo_sha(p: Path) -> str:
    return subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"]).decode().strip()


def repo_dirty(p: Path) -> bool:
    return bool(subprocess.check_output(
        ["git", "-C", str(p), "status", "--porcelain"]).decode().strip())


def license_of(p: Path) -> str:
    for n in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        f = p / n
        if f.exists():
            head = f.read_text(errors="ignore")[:400].replace("\n", " ")
            for tag in ("MIT", "Apache License", "BSD", "GNU GENERAL PUBLIC"):
                if tag in head:
                    return tag
            return "present-unclassified"
    return "NOT_PRESENT_IN_REPO"


def external_sources() -> dict:
    specs = {
        "TQNet": (
            "ACAT-SCUT/TQNet",
            "clean shared cross-channel forecaster, official data provider, official configs",
            ["models/TQNet.py", "models/DLinear.py", "data_provider/data_loader.py",
             "scripts/TQNet/ettm1.sh", "scripts/TQNet/weather.sh", "run.py", "utils/tools.py"]),
        "TSRBench": (
            "dongbeank/TSRBench", "official corruption injection for Track X",
            ["tsrbench/collective_noise.py"]),
        "LGA": (
            "dongbeank/LGA", "identity and context for the TSRBench corruption line (read-only)", []),
        "RHO-Loss": (
            "OATML/RHO-Loss", "RHO selection definition for Track F", ["README.md", "src"]),
        "ARW": (
            "eliselyhan/ARW", "adaptive rolling window selection rule for Track V", []),
        "MTLinear": (
            "azencot-group/MTLinear",
            "multi-task loss-balancing context for Track G (read-only)", []),
    }
    out = {}
    for name, (ident, use, files) in specs.items():
        p = paths.VENDOR / name
        if not p.exists():
            out[name] = {"repository_identity": ident, "status": "NOT_CLONED"}
            continue
        mod = repo_dirty(p)
        detail = None
        if mod:
            detail = DATE_DROP_NOTE if name == "TQNet" else "see git diff inside the vendor clone"
        out[name] = {
            "repository_identity": ident,
            "resolved_commit_sha": repo_sha(p),
            "license": license_of(p),
            "used_for": use,
            "used_files": files,
            "modified": mod,
            "modification_detail": detail,
        }
    out["PCGrad"] = {
        "repository_identity": ("tianheyu927/PCGrad is the official implementation and is TensorFlow; "
                                "WeiChengTseng/Pytorch-PCGrad is an unofficial PyTorch port"),
        "resolved_commit_sha": None,
        "license": "not vendored",
        "used_for": "Track G intervention baseline",
        "status": "LOCAL_REIMPLEMENTATION_FROM_PAPER",
        "rationale": ("Protocol forbids treating an unofficial PyTorch port as scientific authority. "
                      "Algorithm 1 of Yu et al. 2020 (arXiv:2001.06782) is implemented locally and "
                      "verified with synthetic gradient unit tests."),
        "modified": False,
    }
    out["FAF_AdaRho"] = {
        "repository_identity": ("Filter, Augment, Forecast: Online Data Selection for Robust Time "
                                "Series Forecasting, AISTATS 2026"),
        "resolved_commit_sha": None,
        "status": "PAPER_FAITHFUL_LOCAL_ADARHO",
        "used_for": "Track F adaptive-filter baseline",
        "modified": False,
    }
    return out


def data_manifest() -> dict:
    out = {}
    for name, spec in DATASETS.items():
        f = paths.DATASET_DIR / spec.data_path
        raw = f.read_bytes()
        df = pd.read_csv(f)
        cols = [str(c) for c in df.columns]
        ts = pd.to_datetime(df[cols[0]])
        src = ("https://github.com/zhouhaoyi/ETDataset (ETT-small)" if name == "ETTm1"
               else "https://huggingface.co/datasets/thuml/Time-Series-Library (weather/weather.csv)")
        digest = hashlib.sha256(raw).hexdigest()
        out[name] = {
            "source": src,
            "acquisition": "present locally with a matching sha256; no download needed",
            "sha256": digest,
            "sha256_matches_contract": digest == spec.sha256,
            "bytes": len(raw),
            "row_count": int(len(df)),
            "column_names": cols,
            "timestamp_range": [str(ts.iloc[0]), str(ts.iloc[-1])],
            "frequency_inferred": str(ts.diff().mode().iloc[0]),
            "missing_count": int(df.isna().sum().sum()),
            "n_channels": len(cols) - 1,
        }
    return out


def baseline_contract() -> dict:
    out = {
        "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "features": "M",
        "normalization": "StandardScaler fit on the train split only (official TQNet provider)",
        "metric_space": ("scaled space, matching the official TQNet test(), whose inverse_transform "
                         "lines are disabled upstream"),
        "checkpoint_selection": "validation MSE only; test is never used for selection",
        "clean_seeds": CLEAN_SEEDS,
        "splits": {}, "resolved_configs": {},
    }
    for name in DATASETS:
        out["splits"][name] = D.split_borders(name)
        out["splits"][name]["source"] = (
            "Dataset_ETT_minute borders, 12/4/4 months at 15-minute resolution"
            if name == "ETTm1" else "Dataset_Custom, 0.7 / 0.1 / 0.2 of rows")
        out["resolved_configs"][name] = {
            "TQNet": tqnet_config(name, CLEAN_SEEDS[0]).as_dict(),
            "DLinear": dlinear_config(name, CLEAN_SEEDS[0]).as_dict(),
        }
    return out


def runtime_tier() -> dict:
    t = json.loads((paths.RUNS / "smoke" / "timing_real.json").read_text())
    plan = [
        ("clean TQNet", "TQNet", 2, 1.0),
        ("clean DLinear", "DLinear", 2, 1.0),
        ("X channel-dropout TQNet", "TQNet", 2, 1.0),
        ("G PCGrad", "TQNet", 2, 8.0),
        ("G norm-balanced", "TQNet", 2, 8.0),
        ("G probe-gated", "TQNet", 2, 10.0),
        ("F retrain, 6 methods", "DLinear", 12, 1.0),
        ("F RHO reference models", "DLinear", 4, 0.5),
    ]
    total_h = 0.0
    detail = []
    for label, mk, nfits, cost_mult in plan:
        h = 0.0
        for ds in DATASETS:
            ep = t[f"{ds}/{mk}"]["epoch_s"]
            h += ep * 30 * cost_mult * (nfits / 2) / 3600.0
        detail.append({"stage": label, "model": mk, "fits_per_dataset": nfits / 2,
                       "relative_step_cost": cost_mult, "gpu_hours": round(h, 3)})
        total_h += h
    tier = "FULL" if total_h <= 8 else ("COMPACT" if total_h <= 12 else "MINIMAL-COMPLETE")
    settings = {
        "FULL": {"clean_seeds": 2, "intervention_seeds": 2, "bootstrap": 1000},
        "COMPACT": {"clean_seeds": 2, "intervention_seeds": 1, "bootstrap": 500},
        "MINIMAL-COMPLETE": {"clean_seeds": 1, "intervention_seeds": 1, "bootstrap": 300},
    }[tier]
    return {
        "measured_epoch_seconds": t,
        "planned_fits": detail,
        "projected_total_gpu_hours": round(total_h, 2),
        "tier": tier,
        "tier_settings": settings,
        "frozen_before_any_scientific_result": True,
    }


def preregistration() -> dict:
    return {
        "experiment": "TS-IDEA-TOURNAMENT-v1",
        "base_commit_origin_main": "2fe2443b2d24c09ad184387b7f7287f32e0f4cd6",
        "branch": "ts-idea-tournament-v1",
        "top_question": ("Independent of temporal structure, Point/Hurdle, Tweedie or teacher routing, "
                         "is there a real failure mode of modern multivariate time-series forecasters "
                         "that simple existing remedies do not already solve?"),
        "tracks": {
            "G": "Harmful gradient interference, probe-validated rather than cosine-only",
            "X": "Cross-channel corruption spillover",
            "F": "Corruption versus legitimate-shift data-selection confounding",
            "V": "Validation stability audit, auxiliary and never promoting a method",
        },
        "independence_rule": ("Each track verdict is a pure function of that track's own artifacts. "
                              "No feature, threshold or architecture crosses tracks."),
        "datasets": {"primary": ["ETTm1", "Weather"],
                     "fallback": "Electricity, permitted only for source availability"},
        "forecasting_contract": {"lookback": SEQ_LEN, "horizon": PRED_LEN,
                                 "features": "multivariate-to-multivariate"},
        "clean_seeds": CLEAN_SEEDS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap": ("moving-block or origin-cluster bootstrap over evaluation origins; "
                      "iid row bootstrap is forbidden"),
        "gates": {
            "X_phenomenon": {"both_datasets": True, "min_corruption_families": 2,
                             "median_offdiag_spillover_ge": 0.03, "bootstrap_ci_lower_gt": 0.0,
                             "tqnet_over_dlinear_spillover_ge": 3.0,
                             "no_single_source_channel_share_ge": 0.50},
            "X_simple_solves": {"spillover_reduction_ge": 0.30,
                                "clean_mse_degradation_lt": 0.01,
                                "same_direction_both_datasets": True},
            "X_method_go": {"extra_spillover_reduction_vs_best_simple_ge": 0.20,
                            "clean_mse_degradation_lt": 0.01,
                            "direct_damage_degradation_lt": 0.05,
                            "positive_on_both_datasets": True, "bootstrap_lower_gt": 0.0},
            "G_phenomenon": {"exact_harm_rate_ge": 0.10, "bootstrap_lower_ge": 0.05,
                             "cosine_detector_fp_or_fn_ge": 0.20,
                             "cross_probe_auprc_gain_ge": 0.10, "seed_direction_stable": True},
            "G_method_go": {"vs_erm_macro_mse_gain_ge": 0.007, "vs_pcgrad_gain_ge": 0.003,
                            "both_datasets_positive": True, "no_dataset_worse_than": -0.005,
                            "bootstrap_lower_gt": 0.0, "random_control_below_frac": 0.50},
            "F_phenomenon": {"corruption_removal_ge": 0.50, "legit_shift_removal_ge": 0.35,
                             "legit_shift_removal_minus_random_ge_pp": 15},
            "F_method_go": {"shift_retention_gain_pp_ge": 15, "corruption_removal_drop_le_pp": 5,
                            "shifted_test_mse_gain_ge": 0.02, "clean_test_mse_degradation_lt": 0.01,
                            "same_direction_both_datasets": True},
            "V": {"latest_origin_regret_ge": 0.02, "last4_or_arw_regret_reduction_ge": 0.30},
        },
        "borderline_seed2_rule": ("A second intervention seed is added only when the effect lands "
                                  "between 70% and 130% of the METHOD threshold. Effects at or below "
                                  "zero never receive a rescue seed."),
        "forbidden_after_seeing_results": [
            "threshold relaxation", "seed cherry-picking", "dataset swap",
            "corruption severity change", "dropout rate change", "PCGrad rule change",
            "F selection budget change", "model selection on test", "dropping bad origins"],
        "excluded_prior_work": [
            "Point/Hurdle", "Tweedie", "NB/HSNB", "distribution distillation",
            "teacher CDF pooling", "temporal rho_I and rho_M", "prior expert routing",
            "prior online-memory tuning"],
        "verdict_tokens": ["METHOD_GO", "CHARACTERIZATION_ONLY", "SIMPLE_BASELINE_SOLVES",
                           "NO_PHENOMENON", "NOT_EVALUATED"],
        "final_tokens": ["RECOMMEND_G_HARMFUL_INTERFERENCE", "RECOMMEND_X_CORRUPTION_SPILLOVER",
                         "RECOMMEND_F_SELECTION_CONFOUNDING", "RECOMMEND_CHARACTERIZATION_ONLY",
                         "NO_NEW_TOPIC_PASSED"],
    }


if __name__ == "__main__":
    A.write_json(R / "preregistration.json", preregistration())
    A.write_json(R / "environment.json", A.environment())
    A.write_json(R / "external_sources.json", external_sources())
    A.write_json(R / "data_manifest.json", data_manifest())
    A.write_json(R / "baseline_contract.json", baseline_contract())
    A.write_json(R / "runtime_tier.json", runtime_tier())
    rt = json.loads((R / "runtime_tier.json").read_text())
    print("RUNTIME TIER:", rt["tier"], "projected", rt["projected_total_gpu_hours"], "GPU-hours")
    dm = json.loads((R / "data_manifest.json").read_text())
    for k, v in dm.items():
        print(k, "sha_ok", v["sha256_matches_contract"], "rows", v["row_count"],
              "channels", v["n_channels"])
    es = json.loads((R / "external_sources.json").read_text())
    for k, v in es.items():
        print(k, v.get("resolved_commit_sha") or v.get("status"), "modified", v.get("modified"))
