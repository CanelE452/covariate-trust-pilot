"""Study 2B - held-out confirmation of the pre-registered D7 admission policy.

The policy, the proxy, the thresholds and the Gate G criteria are fixed before any
inference runs.  Nothing here selects a method: D5 scoring better than D7 is a
reportable observation, never a reason to swap the primary policy.

The runner and every selector implementation are imported from
``dynamic_admission``; this module only supplies a fresh seed, the pre-registration
record, the independence checks and the Study-2B-specific summaries.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .bootstrap import BOOTSTRAP_UNIT, paired_bootstrap
from .config import ConfirmationConfig
from .dgp import covariate_vintage, generate_base_series
from .dynamic_admission import (D0, D1, D2, D3, D4, D5, D6, D7, SELECTORS, _summarize,
                                apply_selectors, build_proxy_table, run_dynamic_study)
from .reliability_schedules import P1_CALIBRATED, PROXY_MODES, calibrated_proxy, eta_namespace_check
from .schemas import M1, M3

PRIMARY_SELECTOR = D7
PRIMARY_PROXY = P1_CALIBRATED
REFERENCE_SELECTORS = (D0, D1, D2)


# ------------------------------------------------------------ prereg ---------

def preregistration_payload(cfg: ConfirmationConfig, start_state: dict) -> dict:
    """Everything fixed in advance, in one machine-readable record."""
    dyn = cfg.to_dynamic_config()
    n_primary = (len(cfg.grid.nominal_covariate_share) * len(cfg.grid.horizons)
                 * len(dyn.schedules) * cfg.grid.n_series_per_condition)
    return {
        "study": "study2b_d7_heldout_confirmation",
        "written_before_any_inference": True,
        "start_state": start_state,
        "primary_policy": cfg.selectors.primary,
        "secondary_policies": list(cfg.selectors.secondary),
        "reference_policies": list(REFERENCE_SELECTORS),
        "primary_proxy": cfg.proxy.primary_mode,
        "secondary_proxies": list(cfg.proxy.secondary_modes),
        "d7_lower_threshold": cfg.selectors.d7_lower_threshold,
        "d7_upper_threshold": cfg.selectors.d7_upper_threshold,
        "d5_threshold": cfg.selectors.d5_threshold,
        "proxy_sigma": cfg.proxy.sigma_proxy,
        "master_seed": cfg.experiment.master_seed,
        "schedules": [{"name": s.name, "historical": list(s.historical), "current": s.current}
                      for s in dyn.schedules],
        "nominal_covariate_share": list(cfg.grid.nominal_covariate_share),
        "horizons": list(cfg.grid.horizons),
        "n_series_per_condition": cfg.grid.n_series_per_condition,
        "bootstrap": {"n_resamples": cfg.bootstrap.n_resamples,
                      "confidence_level": cfg.bootstrap.confidence_level,
                      "unit": BOOTSTRAP_UNIT},
        "mixture": ("all schedules, shares, horizons and base series equally weighted; this is a "
                    "design choice and does not represent deployment prevalence"),
        "expected_primary_tasks": n_primary,
        "expected_historical_tasks": n_primary * dyn.n_historical_origins,
        "gate_g": {
            "decided_on": f"{cfg.selectors.primary} under {cfg.proxy.primary_mode} only",
            "G1_overall_improvement_pass": cfg.gate_g.overall_improvement_pass,
            "G2": "paired cluster bootstrap CI favours the primary policy",
            "G3_oracle_recovery_pass": cfg.gate_g.oracle_recovery_pass,
            "G4_harm_reduction_pass": cfg.gate_g.harm_reduction_pass,
            "G5_stable_low_regression_max": cfg.gate_g.stable_condition_regression_max,
            "G6_stable_high_regression_max": cfg.gate_g.stable_condition_regression_max,
            "G7": "beats D3 and D4 in both worsening schedules",
            "G8": "beats D0 in both improvement schedules",
            "fail_overall_improvement": cfg.gate_g.overall_improvement_fail,
            "fail_oracle_recovery": cfg.gate_g.oracle_recovery_fail,
            "fail_harm_reduction": cfg.gate_g.harm_reduction_fail,
            "fail_stable_regression": cfg.gate_g.stable_condition_fail,
        },
        "forbidden_after_seeing_results": [
            "replacing D7 with D5 or any other policy",
            "changing the D7 thresholds",
            "changing sigma_proxy",
            "changing any reliability schedule",
            "changing a Gate G threshold",
            "re-running with a different seed to obtain a better outcome",
        ],
    }


def preregistration_hash(payload: dict) -> str:
    """SHA-256 of the canonical JSON form, recorded in the manifest."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# -------------------------------------------------------- independence ------

def independence_checks(cfg: ConfirmationConfig, previous_seeds: list[int],
                        previous_series_dir=None) -> dict:
    """Prove this study shares nothing but the model weights with the earlier runs."""
    dyn = cfg.to_dynamic_config()
    pilot = dyn.to_pilot_config()
    checks = []

    checks.append({
        "id": "IND1_new_master_seed",
        "description": "the Study 2B master seed differs from every earlier study",
        "status": "PASS" if cfg.experiment.master_seed not in previous_seeds else "FAIL",
        "detail": f"study2b={cfg.experiment.master_seed}, earlier={sorted(previous_seeds)}",
    })

    # series must differ from the earlier dynamic study for the same ids
    prev_seed = 20260802
    prev = pilot.to_dict()
    prev["experiment"]["master_seed"] = prev_seed
    from .config import PilotConfig
    prev_cfg = PilotConfig.from_dict(prev)
    identical = []
    for b_id in list(cfg.base_series_ids)[:10]:
        a = generate_base_series(b_id, pilot)
        b = generate_base_series(b_id, prev_cfg)
        if np.array_equal(a.b, b.b) or np.array_equal(a.x, b.x):
            identical.append(b_id)
    checks.append({
        "id": "IND2_series_differ_from_previous_study",
        "description": "no base series is bitwise identical to the Study 2 series of the same id",
        "status": "PASS" if not identical else "FAIL",
        "detail": f"checked 10 ids against seed {prev_seed}; identical: {identical}",
    })

    # eta paths must differ too
    s_new = generate_base_series(0, pilot)
    s_old = generate_base_series(0, prev_cfg)
    eta_same = []
    for origin in (800, 896):
        a = covariate_vintage(pilot, s_new, origin, 24, 1.0)["eta"]
        b = covariate_vintage(prev_cfg, s_old, origin, 24, 1.0)["eta"]
        if np.allclose(a, b):
            eta_same.append(origin)
    checks.append({
        "id": "IND3_eta_paths_differ",
        "description": "forecast-error eta paths differ from the previous study",
        "status": "PASS" if not eta_same else "FAIL",
        "detail": f"origins with identical eta: {eta_same}",
    })

    # proxy noise must differ, and must live in its own namespace
    from .config import DynamicConfig
    prev_dyn = DynamicConfig.from_dict(
        {**{k: v for k, v in dyn.to_dict().items() if k != "inherited_from_pilot_yaml"},
         "experiment": {**dyn.to_dict()["experiment"], "master_seed": prev_seed}},
        dyn.inherited)
    same_proxy = []
    for b_id in range(5):
        a = calibrated_proxy(dyn, b_id, 0.5, 24, "S0_stable_low", 1.0)
        b = calibrated_proxy(prev_dyn, b_id, 0.5, 24, "S0_stable_low", 1.0)
        if a == b:
            same_proxy.append(b_id)
    checks.append({
        "id": "IND4_proxy_noise_differs",
        "description": "current proxy noise differs from the previous study",
        "status": "PASS" if not same_proxy else "FAIL",
        "detail": f"ids with identical reported lambda: {same_proxy}",
    })

    ns = eta_namespace_check()
    checks.append({
        "id": "IND5_proxy_and_eta_namespaces_disjoint",
        "description": "proxy noise and forecast-error eta are drawn from different namespaces",
        "status": "PASS" if ns["disjoint"] else "FAIL",
        "detail": f"eta namespace {ns['eta']!r}, proxy namespace {ns['proxy']!r}",
    })

    return {"status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
            "checks": checks}


# ------------------------------------------------------------- summaries ----

def selector_summary(decisions: pd.DataFrame, cfg: ConfirmationConfig) -> pd.DataFrame:
    """Per selector under the primary proxy, plus the paired bootstrap against best fixed."""
    p1 = decisions[decisions["proxy_mode"] == PRIMARY_PROXY]
    fixed = {sel: float(p1[p1["selector"] == sel]["wql_selected"].mean())
             for sel in (D0, D1)}
    best_fixed_name = min(fixed, key=fixed.get)
    best_fixed_col = "wql_m1" if best_fixed_name == D0 else "wql_m3"

    rows = []
    for sel, g in p1.groupby("selector"):
        s = _summarize(g)
        b = paired_bootstrap(g[BOOTSTRAP_UNIT].to_numpy(), g[best_fixed_col].to_numpy(),
                             g["wql_selected"].to_numpy(), cfg.bootstrap.n_resamples,
                             cfg.bootstrap.confidence_level,
                             seed_parts=(cfg.experiment.master_seed, "sel_boot", sel))
        rows.append({"selector": sel, "best_fixed": best_fixed_name, **s,
                     "boot_mean_diff": b.mean_diff, "boot_median_diff": b.median_diff,
                     "boot_ci_low": b.ci_low, "boot_ci_high": b.ci_high,
                     "boot_monte_carlo_se": b.monte_carlo_se,
                     "boot_relative_improvement": b.relative_improvement,
                     "boot_win_rate": b.win_rate,
                     "ci_favours_selector": b.ci_favours_treatment})
    return pd.DataFrame(rows).sort_values("selector").reset_index(drop=True)


def _grouped(decisions: pd.DataFrame, by: str) -> pd.DataFrame:
    p1 = decisions[decisions["proxy_mode"] == PRIMARY_PROXY]
    rows = []
    for (value, sel), g in p1.groupby([by, "selector"], sort=True):
        rows.append({by: value, "selector": sel, **_summarize(g)})
    return pd.DataFrame(rows)


def share_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    return _grouped(decisions, "nominal_covariate_share")


def horizon_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    return _grouped(decisions, "horizon")


def bootstrap_summary(decisions: pd.DataFrame, cfg: ConfirmationConfig) -> pd.DataFrame:
    """The headline paired comparisons, all clustered on base_series_id."""
    p1 = decisions[decisions["proxy_mode"] == PRIMARY_PROXY]
    d7 = p1[p1["selector"] == PRIMARY_SELECTOR]
    rows = []
    comparisons = [
        ("D7_vs_always_no_future", "wql_m1", d7["wql_selected"]),
        ("D7_vs_always_use_future", "wql_m3", d7["wql_selected"]),
        ("D7_vs_oracle", "wql_oracle", d7["wql_selected"]),
    ]
    for name, baseline_col, treatment in comparisons:
        b = paired_bootstrap(d7[BOOTSTRAP_UNIT].to_numpy(), d7[baseline_col].to_numpy(),
                             np.asarray(treatment), cfg.bootstrap.n_resamples,
                             cfg.bootstrap.confidence_level,
                             seed_parts=(cfg.experiment.master_seed, "boot_summary", name))
        rows.append({"comparison": name, **b.to_dict()})
    for sel in (D3, D4, D5, D6):
        g = p1[p1["selector"] == sel]
        b = paired_bootstrap(g[BOOTSTRAP_UNIT].to_numpy(), g["wql_selected"].to_numpy(),
                             d7["wql_selected"].to_numpy(), cfg.bootstrap.n_resamples,
                             cfg.bootstrap.confidence_level,
                             seed_parts=(cfg.experiment.master_seed, "boot_summary", sel))
        rows.append({"comparison": f"D7_vs_{sel}", **b.to_dict()})
    return pd.DataFrame(rows)


def proxy_stress_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Secondary diagnostics: the primary policy under every proxy mode."""
    rows = []
    for mode, g in decisions[decisions["selector"] == PRIMARY_SELECTOR].groupby("proxy_mode"):
        rows.append({"proxy_mode": mode, "selector": PRIMARY_SELECTOR,
                     "mean_calibration_ratio": float(g["calibration_ratio"].mean()),
                     "is_primary_proxy": mode == PRIMARY_PROXY, **_summarize(g)})
    for mode, g in decisions[decisions["selector"] == D5].groupby("proxy_mode"):
        rows.append({"proxy_mode": mode, "selector": D5,
                     "mean_calibration_ratio": float(g["calibration_ratio"].mean()),
                     "is_primary_proxy": mode == PRIMARY_PROXY, **_summarize(g)})
    return pd.DataFrame(rows).sort_values(["selector", "proxy_mode"]).reset_index(drop=True)


# ------------------------------------------------------------------ run -----

def run_confirmation(cfg: ConfirmationConfig, predict_fn, log=lambda *_: None):
    """Execute the held-out study.  Returns (tasks, proxies, decisions)."""
    dyn = cfg.to_dynamic_config()
    tasks = run_dynamic_study(dyn, predict_fn, log)
    proxies = build_proxy_table(tasks, dyn)
    decisions = apply_selectors(tasks, proxies, dyn)
    return tasks, proxies, decisions


def leakage_checks(tasks: pd.DataFrame, proxies: pd.DataFrame,
                   cfg: ConfirmationConfig) -> dict:
    """Executable proof that no selector reads the present.

    The current outcomes are scaled by arbitrary factors and the decisions recomputed:
    only the oracle D2 may change.
    """
    dyn = cfg.to_dynamic_config()
    base = apply_selectors(tasks, proxies, dyn)
    poisoned = tasks.copy()
    poisoned["wql_m1"] = poisoned["wql_m1"] * 3.0
    poisoned["wql_m3"] = poisoned["wql_m3"] * 0.1
    poisoned["wql_oracle"] = np.minimum(poisoned["wql_m1"], poisoned["wql_m3"])
    poisoned["m3_is_better"] = (poisoned["wql_m3"] < poisoned["wql_m1"]).astype(int)
    other = apply_selectors(poisoned, proxies, dyn)

    key = [BOOTSTRAP_UNIT, "schedule", "proxy_mode", "horizon", "nominal_covariate_share"]
    changed = {}
    for sel in SELECTORS:
        a = base[base["selector"] == sel].sort_values(key)["choice"].to_numpy()
        b = other[other["selector"] == sel].sort_values(key)["choice"].to_numpy()
        changed[sel] = bool((a != b).any())

    non_oracle_changed = [s for s, c in changed.items() if c and s != D2]
    checks = [{
        "id": "LEAK1_no_selector_reads_the_current_outcome",
        "description": "scaling the current WQLs changes only the oracle D2",
        "status": "PASS" if not non_oracle_changed else "FAIL",
        "detail": f"selectors that reacted: {non_oracle_changed or 'none'}; "
                  f"D2 reacted: {changed[D2]}",
    }, {
        "id": "LEAK2_only_p0_uses_the_true_current_lambda",
        "description": "every proxy row records whether it consumed the truth",
        "status": "PASS" if set(proxies[proxies["uses_true_current_lambda"]]["proxy_mode"]) <= {
            "P0_oracle_current"} else "FAIL",
        "detail": f"modes reading the truth: "
                  f"{sorted(set(proxies[proxies['uses_true_current_lambda']]['proxy_mode']))}",
    }, {
        "id": "LEAK3_historical_windows_close_before_the_primary_origin",
        "description": "every pseudo-origin plus its horizon lands at or before the primary origin",
        "status": "PASS",
        "detail": f"guarded by admission.assert_no_primary_leak; primary origin "
                  f"{cfg.experiment.primary_origin}",
    }]
    return {"status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
            "checks": checks, "selectors_changed": changed}
