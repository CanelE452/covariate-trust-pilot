"""Gate F, Go/No-Go and existing-gate regression checks (follow-up tests 34-36)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covariate_trust.dynamic_admission import (D0, D1, D3, D5, D7, apply_selectors,
                                               build_proxy_table)
from covariate_trust.followup_gates import gate_f, go_no_go
from covariate_trust.reliability_schedules import P1_CALIBRATED
from covariate_trust.schemas import M1, M3
from test_dynamic_admission import fake_tasks


def _decisions(cfg, tasks: pd.DataFrame) -> pd.DataFrame:
    return apply_selectors(tasks, build_proxy_table(tasks, cfg), cfg)


def test_f34_gate_f_pass_fixture(small_dynamic_cfg):
    """Test 34a: on data where the current lambda determines the winner, Gate F can PASS."""
    cfg = small_dynamic_cfg
    tasks = fake_tasks(cfg, n_series=30)
    out = gate_f(_decisions(cfg, tasks), cfg)
    assert out["status"] in {"PASS", "INCONCLUSIVE"}, out["checks"]
    assert out["primary_proxy"] == P1_CALIBRATED
    assert out["primary_selector"] in {D5, D7}
    assert out["checks"]["relative_improvement"] > 0


def test_f34b_gate_f_fail_fixture(small_dynamic_cfg):
    """Test 34b: when M3 is uniformly bad, no proxy selector can beat always-no-future."""
    cfg = small_dynamic_cfg
    tasks = fake_tasks(cfg, n_series=20)
    tasks["wql_m3"] = tasks["wql_m1"] * 1.5           # M3 always worse
    tasks["wql_oracle"] = np.minimum(tasks["wql_m1"], tasks["wql_m3"])
    tasks["m3_is_better"] = 0
    out = gate_f(_decisions(cfg, tasks), cfg)
    assert out["status"] == "FAIL", out["checks"]


def test_f34c_gate_f_reports_every_proxy_diagnostic(small_dynamic_cfg):
    cfg = small_dynamic_cfg
    out = gate_f(_decisions(cfg, fake_tasks(cfg, n_series=12)), cfg)
    assert set(out["proxy_diagnostics"]) == {"P0_oracle_current", "P2_overconfident",
                                             "P3_underconfident", "P4_stale_history"}
    assert set(out["per_selector_calibrated_proxy"]) >= {D0, D1, D3, D5, D7}
    # P0 must never be the gate's primary evidence
    assert out["primary_proxy"] == P1_CALIBRATED


def test_f34d_gate_f_conditions_are_reported_per_schedule(small_dynamic_cfg):
    cfg = small_dynamic_cfg
    out = gate_f(_decisions(cfg, fake_tasks(cfg, n_series=12)), cfg)
    checks = out["checks"]
    assert set(checks["worsening_beats_history_only"]) == {"S2_sudden_worsening",
                                                           "S4_gradual_worsening"}
    assert set(checks["improvement_condition_relative_regression"]) == {
        "S3_sudden_improvement", "S5_gradual_improvement"}
    assert "stable_low_relative_regression" in checks
    assert "stable_high_relative_regression" in checks


def test_f34e_go_no_go_verdicts():
    pas, fail, inc = {"status": "PASS"}, {"status": "FAIL"}, {"status": "INCONCLUSIVE"}
    assert go_no_go(pas, pas, pas, True, True)["verdict"] == "FINAL GO"
    assert go_no_go(pas, fail, pas, True, True)["verdict"] == "NO-GO METHOD"
    assert go_no_go(fail, None, pas, True, True)["verdict"] == "NO-GO PHENOMENON"
    assert go_no_go(inc, None, pas, True, True)["verdict"] == "CONDITIONAL GO"
    assert go_no_go(pas, inc, pas, True, True)["verdict"] == "CONDITIONAL GO"
    assert go_no_go(pas, pas, pas, True, False)["verdict"] == "CONDITIONAL GO"
    assert go_no_go(None, None, None, True, True)["verdict"] == "BLOCKED"


def test_f35_existing_gate_d_is_untouched(pilot_cfg):
    """Test 35: the existing Gate D implementation and its inputs are unchanged."""
    from covariate_trust.admission import A1, A2, PSEUDO_ORIGINS, gate_d, SELECTORS as D_SELECTORS
    assert PSEUDO_ORIGINS == {24: [800, 824, 848, 872], 96: [512, 608, 704, 800]}
    assert D_SELECTORS == (A1, A2)
    assert callable(gate_d)
    # Gate D lives in admission.py; Gate F lives in followup_gates.py - separate functions
    from covariate_trust import followup_gates
    assert not hasattr(followup_gates, "gate_d")


def test_f36_existing_gate_constants_are_unchanged(pilot_cfg):
    """Test 36: coarse-pilot thresholds were not retuned by the follow-up."""
    from covariate_trust.gates import (GATE_A_PRIMARY_SHARES, HIGH_NOISE_MIN_LAMBDA,
                                       LOW_NOISE_MAX_LAMBDA, NEGATIVE_CONTROL_SHARE)
    assert GATE_A_PRIMARY_SHARES == (0.50, 0.75)
    assert NEGATIVE_CONTROL_SHARE == 0.0
    assert LOW_NOISE_MAX_LAMBDA == 0.5
    assert HIGH_NOISE_MIN_LAMBDA == 1.5
    g = pilot_cfg.gates
    assert (g.clean_gain_pass, g.clean_gain_fail) == (0.05, 0.01)
    assert (g.oracle_headroom_pass, g.oracle_headroom_fail) == (0.03, 0.01)
    assert (g.harm_relative_threshold, g.high_noise_harm_rate) == (0.05, 0.20)


def test_f36b_followup_configs_inherit_the_pilot_dgp(boundary_cfg, dynamic_cfg, pilot_cfg):
    """Both follow-up studies must run the coarse pilot's generating equations."""
    assert boundary_cfg.to_pilot_config().dgp == pilot_cfg.dgp
    assert dynamic_cfg.to_pilot_config().dgp == pilot_cfg.dgp
    assert boundary_cfg.to_pilot_config().gates == pilot_cfg.gates
    assert dynamic_cfg.to_pilot_config().gates == pilot_cfg.gates
    assert boundary_cfg.to_pilot_config().model.cross_learning is False
    assert dynamic_cfg.to_pilot_config().model.cross_learning is False
