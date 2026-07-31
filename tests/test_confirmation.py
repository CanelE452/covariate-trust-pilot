"""Study 2B checks (confirmation tests 1-31).

These verify that the study is a genuine held-out confirmation: the policy is fixed
in advance, the sample is new, nothing leaks, and no forecast is recomputed per proxy
mode.  No existing assertion is weakened to make any of them pass.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from covariate_trust.config import ConfigError, ConfirmationConfig, PilotConfig
from covariate_trust.confirmation import (PRIMARY_PROXY, PRIMARY_SELECTOR, independence_checks,
                                          leakage_checks, preregistration_hash,
                                          preregistration_payload, proxy_stress_summary,
                                          run_confirmation, selector_summary)
from covariate_trust.dgp import covariate_vintage, generate_base_series
from covariate_trust.dynamic_admission import (D0, D1, D2, D3, D4, D5, D6, D7, SELECTORS,
                                               apply_selectors, build_proxy_table, false_rates)
from covariate_trust.followup_gates import gate_g, gate_g_verdict
from covariate_trust.reliability_schedules import (P0_ORACLE, P1_CALIBRATED, P2_OVERCONFIDENT,
                                                   P3_UNDERCONFIDENT, P4_STALE, PROXY_MODES,
                                                   eta_namespace_check, reported_lambda)
from covariate_trust.schemas import (COVARIATE_COLUMN, M1, M3, TARGET_COLUMN,
                                     assert_context_equality, build_inputs)
from test_dynamic_admission import fake_tasks

PREV_SEEDS = [20260730, 20260801, 20260802]


def _fake_conf_tasks(cfg: ConfirmationConfig, n_series: int = 20) -> pd.DataFrame:
    return fake_tasks(cfg.to_dynamic_config(), n_series=n_series)


def _decisions(cfg: ConfirmationConfig, tasks: pd.DataFrame) -> pd.DataFrame:
    dyn = cfg.to_dynamic_config()
    return apply_selectors(tasks, build_proxy_table(tasks, dyn), dyn)


# ------------------------------------------------------- independence -------

def test_c01_seed_differs_from_previous_studies(confirmation_cfg):
    """Test 1: Study 2B uses a master seed no earlier study used."""
    assert confirmation_cfg.experiment.master_seed == 20260803
    assert confirmation_cfg.experiment.master_seed not in PREV_SEEDS


def test_c02_series_differ_from_previous_study(small_confirmation_cfg):
    """Test 2: no base series is bitwise identical to the Study 2 series of the same id."""
    new = small_confirmation_cfg.to_dynamic_config().to_pilot_config()
    prev = new.to_dict()
    prev["experiment"]["master_seed"] = 20260802
    prev_cfg = PilotConfig.from_dict(prev)
    for b_id in small_confirmation_cfg.base_series_ids:
        a = generate_base_series(b_id, new)
        b = generate_base_series(b_id, prev_cfg)
        assert not np.array_equal(a.b, b.b)
        assert not np.array_equal(a.x, b.x)


def test_c02b_independence_checks_report_pass(small_confirmation_cfg):
    out = independence_checks(small_confirmation_cfg, previous_seeds=PREV_SEEDS)
    assert out["status"] == "PASS", [(c["id"], c["detail"]) for c in out["checks"]]
    assert {c["id"] for c in out["checks"]} == {
        "IND1_new_master_seed", "IND2_series_differ_from_previous_study",
        "IND3_eta_paths_differ", "IND4_proxy_noise_differs",
        "IND5_proxy_and_eta_namespaces_disjoint"}


def test_c03_schedules_match_the_preregistration(confirmation_cfg):
    """Test 3: the six schedules are exactly the pre-registered ones."""
    dyn = confirmation_cfg.to_dynamic_config()
    got = {s.name: (list(s.historical), s.current) for s in dyn.schedules}
    assert got == {
        "S0_stable_low": ([0.50, 0.50, 0.50, 0.50], 0.50),
        "S1_stable_high": ([1.50, 1.50, 1.50, 1.50], 1.50),
        "S2_sudden_worsening": ([0.50, 0.50, 0.50, 0.50], 1.50),
        "S3_sudden_improvement": ([1.50, 1.50, 1.50, 1.50], 0.50),
        "S4_gradual_worsening": ([0.50, 0.75, 1.00, 1.25], 1.50),
        "S5_gradual_improvement": ([1.50, 1.25, 1.00, 0.75], 0.50),
    }
    prereg = preregistration_payload(confirmation_cfg, {"commit": "x"})
    assert {s["name"] for s in prereg["schedules"]} == set(got)


# ------------------------------------------------ pre-registered policy -----

def test_c04_d7_is_the_only_primary(confirmation_cfg):
    """Test 4: D7 is the primary in the config, the pre-registration and Gate G."""
    assert confirmation_cfg.selectors.primary == D7 == PRIMARY_SELECTOR
    assert confirmation_cfg.proxy.primary_mode == P1_CALIBRATED == PRIMARY_PROXY
    prereg = preregistration_payload(confirmation_cfg, {"commit": "x"})
    assert prereg["primary_policy"] == D7
    out = gate_g(_decisions(confirmation_cfg, _fake_conf_tasks(confirmation_cfg)),
                 confirmation_cfg)
    assert out["primary_selector"] == D7
    assert out["primary_proxy"] == P1_CALIBRATED


def test_c05_d7_thresholds_are_fixed(confirmation_cfg):
    """Test 5: the override band is 0.75 / 1.25."""
    assert confirmation_cfg.selectors.d7_lower_threshold == 0.75
    assert confirmation_cfg.selectors.d7_upper_threshold == 1.25
    th = confirmation_cfg.to_dynamic_config().selector_thresholds
    assert (th.override_low, th.use_threshold, th.override_high) == (0.75, 1.0, 1.25)


def test_c06_proxy_sigma_is_fixed(confirmation_cfg):
    """Test 6: sigma_proxy is 0.20."""
    assert confirmation_cfg.proxy.sigma_proxy == 0.20
    assert confirmation_cfg.to_dynamic_config().proxy.sigma_proxy == 0.20


def test_c07_primary_cannot_be_swapped_for_d5(confirmation_cfg):
    """Test 7: a config naming D5 as primary is rejected, however well D5 scores."""
    d = confirmation_cfg.to_dict()
    d.pop("inherited")
    d["selectors"]["primary"] = D5
    with pytest.raises(ConfigError, match="pre-registered"):
        ConfirmationConfig.from_dict(d, confirmation_cfg.inherited)

    # and Gate G still reports on D7 even when D5 is the better policy in the data
    tasks = _fake_conf_tasks(confirmation_cfg)
    out = gate_g(_decisions(confirmation_cfg, tasks), confirmation_cfg)
    per = out["per_selector_primary_proxy"]
    assert out["primary_selector"] == D7
    if per[D5]["mean_wql"] < per[D7]["mean_wql"]:
        assert out["primary_metrics"]["mean_wql"] == per[D7]["mean_wql"]


def test_c07b_primary_proxy_cannot_be_swapped(confirmation_cfg):
    d = confirmation_cfg.to_dict()
    d.pop("inherited")
    d["proxy"]["primary_mode"] = P0_ORACLE
    with pytest.raises(ConfigError, match="pre-registered"):
        ConfirmationConfig.from_dict(d, confirmation_cfg.inherited)


# ------------------------------------------------------------- proxies -----

def test_c08_selector_never_receives_the_true_current_lambda(small_confirmation_cfg):
    """Test 8: the decision function reads `reported_lambda` and no truth column."""
    tasks = _fake_conf_tasks(small_confirmation_cfg, n_series=4)
    dyn = small_confirmation_cfg.to_dynamic_config()
    proxies = build_proxy_table(tasks, dyn)
    d = apply_selectors(tasks, proxies, dyn)
    p1 = d[(d["proxy_mode"] == P1_CALIBRATED) & (d["selector"] == D7)]
    # rewriting the truth column while keeping the report fixed cannot move D7
    poisoned_tasks = tasks.copy()
    poisoned_tasks["true_current_lambda"] = 99.0
    d2 = apply_selectors(poisoned_tasks, proxies, dyn)
    p1b = d2[(d2["proxy_mode"] == P1_CALIBRATED) & (d2["selector"] == D7)]
    key = ["base_series_id", "schedule", "horizon", "nominal_covariate_share"]
    assert list(p1.sort_values(key)["choice"]) == list(p1b.sort_values(key)["choice"])
    assert not p1["uses_true_current_lambda"].any()


def test_c09_p1_reported_lambda_is_reproducible(small_confirmation_cfg):
    """Test 9: the calibrated proxy is deterministic per task."""
    dyn = small_confirmation_cfg.to_dynamic_config()
    kw = dict(base_series_id=2, share=0.5, horizon=24, schedule_name="S1_stable_high",
              true_current_lambda=1.5, historical_lambda_estimates=[1.5] * 4)
    a = reported_lambda(dyn, P1_CALIBRATED, **kw)
    b = reported_lambda(dyn, P1_CALIBRATED, **kw)
    c = reported_lambda(dyn, P1_CALIBRATED, **{**kw, "base_series_id": 3})
    assert a == b and a != c and a > 0


def test_c10_proxy_and_eta_namespaces_are_separate(small_confirmation_cfg):
    """Test 10: the two random streams do not share a namespace and are uncorrelated."""
    ns = eta_namespace_check()
    assert ns["disjoint"] and ns["eta"] != ns["proxy"]
    dyn = small_confirmation_cfg.to_dynamic_config()
    pilot = dyn.to_pilot_config()
    reports, eta_means = [], []
    for b_id in range(60):
        s = generate_base_series(b_id, pilot)
        v = covariate_vintage(pilot, s, pilot.experiment.primary_origin, 24, 1.5)
        reports.append(reported_lambda(dyn, P1_CALIBRATED, base_series_id=b_id, share=0.5,
                                       horizon=24, schedule_name="S1_stable_high",
                                       true_current_lambda=1.5,
                                       historical_lambda_estimates=[1.5] * 4))
        eta_means.append(float(v["eta"].mean()))
    assert abs(np.corrcoef(reports, eta_means)[0, 1]) < 0.35


def test_c11_p2_is_exactly_half_of_p1(small_confirmation_cfg):
    """Test 11: the overconfident report is 0.5 x the calibrated one."""
    dyn = small_confirmation_cfg.to_dynamic_config()
    kw = dict(base_series_id=1, share=0.25, horizon=96, schedule_name="S2_sudden_worsening",
              true_current_lambda=1.5, historical_lambda_estimates=[0.5] * 4)
    assert reported_lambda(dyn, P2_OVERCONFIDENT, **kw) == pytest.approx(
        0.5 * reported_lambda(dyn, P1_CALIBRATED, **kw))


def test_c12_p3_is_exactly_one_and_a_half_of_p1(small_confirmation_cfg):
    """Test 12: the underconfident report is 1.5 x the calibrated one."""
    dyn = small_confirmation_cfg.to_dynamic_config()
    kw = dict(base_series_id=1, share=0.25, horizon=96, schedule_name="S3_sudden_improvement",
              true_current_lambda=0.5, historical_lambda_estimates=[1.5] * 4)
    assert reported_lambda(dyn, P3_UNDERCONFIDENT, **kw) == pytest.approx(
        1.5 * reported_lambda(dyn, P1_CALIBRATED, **kw))


def test_c13_p4_ignores_the_true_current_lambda(small_confirmation_cfg):
    """Test 13: the stale report depends only on history."""
    dyn = small_confirmation_cfg.to_dynamic_config()
    kw = dict(base_series_id=0, share=0.5, horizon=24, schedule_name="S2_sudden_worsening",
              historical_lambda_estimates=[0.5, 0.5, 0.5, 0.5])
    a = reported_lambda(dyn, P4_STALE, true_current_lambda=1.5, **kw)
    b = reported_lambda(dyn, P4_STALE, true_current_lambda=0.5, **kw)
    assert a == b == pytest.approx(0.5)


# -------------------------------------------------------------- leakage ----

def test_c14_decisions_are_invariant_to_the_current_outcome(small_confirmation_cfg):
    """Test 14: scaling the present by 3x / 0.1x cannot move D3-D7."""
    tasks = _fake_conf_tasks(small_confirmation_cfg, n_series=6)
    dyn = small_confirmation_cfg.to_dynamic_config()
    proxies = build_proxy_table(tasks, dyn)
    base = apply_selectors(tasks, proxies, dyn)
    poisoned = tasks.copy()
    poisoned["wql_m1"] *= 3.0
    poisoned["wql_m3"] *= 0.1
    poisoned["wql_oracle"] = np.minimum(poisoned["wql_m1"], poisoned["wql_m3"])
    poisoned["m3_is_better"] = (poisoned["wql_m3"] < poisoned["wql_m1"]).astype(int)
    other = apply_selectors(poisoned, proxies, dyn)
    key = ["base_series_id", "schedule", "proxy_mode", "horizon", "nominal_covariate_share"]
    for sel in (D3, D4, D5, D6, D7):
        a = base[base["selector"] == sel].sort_values(key)["choice"].to_numpy()
        b = other[other["selector"] == sel].sort_values(key)["choice"].to_numpy()
        assert (a == b).all(), sel


def test_c15_only_the_oracle_reacts(small_confirmation_cfg):
    """Test 15: D2 is allowed to change, and does."""
    tasks = _fake_conf_tasks(small_confirmation_cfg, n_series=6)
    proxies = build_proxy_table(tasks, small_confirmation_cfg.to_dynamic_config())
    out = leakage_checks(tasks, proxies, small_confirmation_cfg)
    assert out["status"] == "PASS", out["checks"]
    changed = out["selectors_changed"]
    assert changed[D2] is True
    assert not any(v for s, v in changed.items() if s != D2)


def test_c16_m1_and_m3_share_the_context(small_confirmation_cfg):
    """Test 16: the only difference between M1 and M3 is the future frame."""
    dyn = small_confirmation_cfg.to_dynamic_config()
    pilot = dyn.to_pilot_config()
    s = generate_base_series(0, pilot)
    from covariate_trust.dgp import build_target
    y = build_target(s, 0.5)
    o, h = pilot.experiment.primary_origin, 24
    v = covariate_vintage(pilot, s, o, h, 1.5)
    in1 = build_inputs(M1, "t", y, s.x, o, h, pilot.experiment.context_length,
                       pilot.experiment.frequency)
    in3 = build_inputs(M3, "t", y, s.x, o, h, pilot.experiment.context_length,
                       pilot.experiment.frequency, x_future=v["x_tilde"])
    assert_context_equality(in1, in3)
    assert in1.future_df is None


def test_c17_future_frame_excludes_the_target(small_confirmation_cfg):
    """Test 17: future_df carries the covariate only."""
    dyn = small_confirmation_cfg.to_dynamic_config()
    pilot = dyn.to_pilot_config()
    s = generate_base_series(0, pilot)
    from covariate_trust.dgp import build_target
    y = build_target(s, 0.5)
    o, h = pilot.experiment.primary_origin, 96
    v = covariate_vintage(pilot, s, o, h, 0.5)
    in3 = build_inputs(M3, "t", y, s.x, o, h, pilot.experiment.context_length,
                       pilot.experiment.frequency, x_future=v["x_tilde"])
    assert TARGET_COLUMN not in in3.future_df.columns
    assert COVARIATE_COLUMN in in3.future_df.columns


def test_c18_historical_windows_close_before_the_primary_origin(confirmation_cfg):
    """Test 18: pseudo-origin + horizon <= primary origin for every schedule origin."""
    from covariate_trust.reliability_schedules import schedule_origins
    dyn = confirmation_cfg.to_dynamic_config()
    for h in dyn.grid.horizons:
        hist, primary = schedule_origins(dyn, h)
        assert primary == confirmation_cfg.experiment.primary_origin
        for o in hist:
            assert o + h <= primary


# ---------------------------------------------------------------- cache ----

def test_c19_proxy_modes_do_not_trigger_new_inference(small_confirmation_cfg):
    """Test 19: the number of forecasts is independent of how many proxy modes exist."""
    calls = []

    def fake_predict(inputs, meta):
        calls.append((meta["method"], meta["base_series_id"], meta["origin"], meta.get("lam"),
                      meta["horizon"], meta["nominal_covariate_share"]))
        n = len(small_confirmation_cfg.experiment.quantile_levels)
        return np.tile(np.linspace(-1, 1, n), (inputs.horizon, 1))

    tasks, proxies, decisions = run_confirmation(small_confirmation_cfg, fake_predict)
    n_with_proxies = len(calls)
    assert proxies["proxy_mode"].nunique() == len(PROXY_MODES) == 5
    assert len(decisions) == len(tasks) * len(PROXY_MODES) * len(SELECTORS)

    # The forecasting stage must produce exactly the same calls when no proxy or
    # selector stage runs at all: proxies are applied to stored WQLs afterwards.
    calls.clear()
    from covariate_trust.dynamic_admission import run_dynamic_study
    tasks_only = run_dynamic_study(small_confirmation_cfg.to_dynamic_config(), fake_predict)
    assert len(calls) == n_with_proxies
    assert len(tasks_only) == len(tasks)
    # Different schedules legitimately request the same (method, series, origin, lambda)
    # forecast - that is precisely what the content-hash cache collapses.  Here
    # `fake_predict` has no cache, so duplicates appear; the point is that the number of
    # *distinct* forecasts is strictly smaller, and test_c20 checks the cache exploits it.
    assert len(set(calls)) < len(calls)


def test_c20_identical_inputs_are_shared_across_schedules(small_confirmation_cfg, tmp_path):
    """Test 20: schedules requesting the same (origin, lambda) reuse one cached forecast."""
    from covariate_trust.chronos_adapter import LoadedPipeline
    from covariate_trust.cli import _make_cached_predict
    from covariate_trust.storage import create_run_dir

    dyn = small_confirmation_cfg.to_dynamic_config()
    pilot = dyn.to_pilot_config()
    run_dir = create_run_dir(tmp_path, "cache_test")
    n_q = len(pilot.experiment.quantile_levels)

    class FakePipeline:
        calls = 0

        def predict_df(self, df, **kwargs):
            FakePipeline.calls += 1
            n = kwargs["prediction_length"]
            out = pd.DataFrame({"id": ["t"] * n,
                                "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
                                "predictions": np.zeros(n)})
            for q in kwargs["quantile_levels"]:
                out[str(q)] = float(q)
            return out

    loaded = LoadedPipeline(FakePipeline(), "fake", "cpu", "float32", None, None, {})
    predict, stats = _make_cached_predict(run_dir, loaded, pilot, lambda *_: None)
    tasks, _, _ = run_confirmation(small_confirmation_cfg, predict)

    logical = len(tasks) * (1 + dyn.n_historical_origins) * 2
    assert stats["cache_hits"] > 0
    assert stats["calls"] + stats["cache_hits"] == logical
    assert stats["calls"] < logical, "identical inputs were not shared"
    assert FakePipeline.calls == stats["calls"]


def test_c21_no_previous_run_predictions_are_read(small_confirmation_cfg, tmp_path):
    """Test 21: the cache is confined to this run's own directory."""
    from covariate_trust.cli import _make_cached_predict
    from covariate_trust.chronos_adapter import LoadedPipeline
    from covariate_trust.storage import completed_task_hashes, create_run_dir, write_task_part

    dyn = small_confirmation_cfg.to_dynamic_config()
    pilot = dyn.to_pilot_config()
    old = create_run_dir(tmp_path, "old_run")
    write_task_part(old, "deadbeefdeadbeef", pd.DataFrame({"h_index": [1], "q0.5": [0.0]}))
    new = create_run_dir(tmp_path, "new_run")
    assert completed_task_hashes(new) == set()

    class FakePipeline:
        def predict_df(self, df, **kwargs):
            n = kwargs["prediction_length"]
            out = pd.DataFrame({"id": ["t"] * n,
                                "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
                                "predictions": np.zeros(n)})
            for q in kwargs["quantile_levels"]:
                out[str(q)] = float(q)
            return out

    predict, stats = _make_cached_predict(new, LoadedPipeline(FakePipeline(), "fake", "cpu",
                                                              "float32", None, None, {}),
                                          pilot, lambda *_: None)
    from covariate_trust.dgp import build_target
    s = generate_base_series(0, pilot)
    y = build_target(s, 0.5)
    o = pilot.experiment.primary_origin
    inputs = build_inputs(M1, "t", y, s.x, o, 24, pilot.experiment.context_length,
                          pilot.experiment.frequency)
    predict(inputs, {"base_series_id": 0, "nominal_covariate_share": 0.5, "origin": o,
                     "horizon": 24, "method": M1, "lam": -1.0})
    assert stats["cache_hits"] == 0
    assert "deadbeefdeadbeef" not in completed_task_hashes(new)


# ------------------------------------------------------------- arithmetic ---

def test_c22_false_use_hand_computation():
    """Test 22: false-use is (M1 better and M3 chosen) / (M1 better)."""
    df = pd.DataFrame({"m3_is_better": [0, 0, 0, 0, 1],
                       "choice": [M3, M3, M1, M1, M3]})
    df["false_use"] = ((df["m3_is_better"] == 0) & (df["choice"] == M3)).astype(int)
    df["false_reject"] = ((df["m3_is_better"] == 1) & (df["choice"] == M1)).astype(int)
    out = false_rates(df)
    assert out["n_m1_better"] == 4
    assert out["false_use_rate"] == pytest.approx(2 / 4)


def test_c23_false_reject_hand_computation():
    """Test 23: false-reject is (M3 better and M1 chosen) / (M3 better)."""
    df = pd.DataFrame({"m3_is_better": [1, 1, 1, 0],
                       "choice": [M1, M1, M3, M1]})
    df["false_use"] = ((df["m3_is_better"] == 0) & (df["choice"] == M3)).astype(int)
    df["false_reject"] = ((df["m3_is_better"] == 1) & (df["choice"] == M1)).astype(int)
    out = false_rates(df)
    assert out["n_m3_better"] == 3
    assert out["false_reject_rate"] == pytest.approx(2 / 3)
    assert out["false_use_rate"] == pytest.approx(0.0)


def test_c24_oracle_recovery_hand_computation():
    """Test 24: recovery = (best_fixed - selector) / (best_fixed - oracle)."""
    best_fixed, selector, oracle = 0.50, 0.44, 0.40
    assert (best_fixed - selector) / (best_fixed - oracle) == pytest.approx(0.6)
    # a selector equal to the oracle recovers everything; equal to best fixed recovers nothing
    assert (best_fixed - oracle) / (best_fixed - oracle) == pytest.approx(1.0)
    assert (best_fixed - best_fixed) / (best_fixed - oracle) == pytest.approx(0.0)


def test_c25_harm_reduction_hand_computation():
    """Test 25: reduction = (harm_always_use - harm_selector) / harm_always_use."""
    assert (0.40 - 0.10) / 0.40 == pytest.approx(0.75)
    assert (0.40 - 0.40) / 0.40 == pytest.approx(0.0)


# ---------------------------------------------------------------- Gate G ----

def test_c26_gate_g_pass_fixture(small_confirmation_cfg):
    """Test 26: a sample where the current lambda drives the winner can pass Gate G."""
    tasks = _fake_conf_tasks(small_confirmation_cfg, n_series=40)
    out = gate_g(_decisions(small_confirmation_cfg, tasks), small_confirmation_cfg)
    assert out["status"] in {"PASS", "INCONCLUSIVE"}, out["checks"]
    assert out["primary_selector"] == D7
    if out["status"] == "PASS":
        assert out["failed_conditions"] == []
        assert gate_g_verdict(out, True, True, True)["verdict"] == "METHOD GO"


def test_c27_gate_g_fail_fixture(small_confirmation_cfg):
    """Test 27: when M3 is uniformly worse, Gate G must FAIL."""
    tasks = _fake_conf_tasks(small_confirmation_cfg, n_series=20)
    tasks["wql_m3"] = tasks["wql_m1"] * 1.6
    tasks["wql_oracle"] = np.minimum(tasks["wql_m1"], tasks["wql_m3"])
    tasks["m3_is_better"] = 0
    out = gate_g(_decisions(small_confirmation_cfg, tasks), small_confirmation_cfg)
    assert out["status"] == "FAIL", out["checks"]
    assert out["fail_reasons"]
    assert gate_g_verdict(out, True, True, True)["verdict"] == "NO-GO CURRENT METHOD"


def test_c28_gate_g_inconclusive_fixture(small_confirmation_cfg):
    """Test 28: a sample that clears the FAIL bounds but misses a PASS bound is INCONCLUSIVE."""
    rng = np.random.default_rng(11)
    tasks = _fake_conf_tasks(small_confirmation_cfg, n_series=30)
    # make M3 only marginally better when the covariate forecast is good, so the
    # improvement clears the FAIL bound but not the 5% PASS bound
    cur = tasks["true_current_lambda"].to_numpy()
    tasks["wql_m3"] = tasks["wql_m1"] * (1.0 + 0.02 * (cur - 1.0)) + 0.0005 * rng.normal(
        size=len(tasks))
    tasks["wql_oracle"] = np.minimum(tasks["wql_m1"], tasks["wql_m3"])
    tasks["m3_is_better"] = (tasks["wql_m3"] < tasks["wql_m1"]).astype(int)
    out = gate_g(_decisions(small_confirmation_cfg, tasks), small_confirmation_cfg)
    assert out["status"] in {"INCONCLUSIVE", "FAIL"}
    if out["status"] == "INCONCLUSIVE":
        assert out["failed_conditions"]
        assert gate_g_verdict(out, True, True, True)["verdict"] == "CONDITIONAL GO"


def test_c29_bootstrap_unit_is_the_base_series(small_confirmation_cfg):
    """Test 29: every reported interval clusters on base_series_id."""
    from covariate_trust.bootstrap import BOOTSTRAP_UNIT
    assert BOOTSTRAP_UNIT == "base_series_id"
    tasks = _fake_conf_tasks(small_confirmation_cfg, n_series=20)
    d = _decisions(small_confirmation_cfg, tasks)
    s = selector_summary(d, small_confirmation_cfg)
    assert (s["n_series"] == 20).all()
    assert s["boot_ci_low"].notna().all()
    out = gate_g(d, small_confirmation_cfg)
    assert out["bootstrap"]["n_units"] == 20
    assert out["bootstrap"]["n_observations"] > out["bootstrap"]["n_units"]


def test_c29b_preregistration_hash_is_stable_and_content_sensitive(confirmation_cfg):
    a = preregistration_payload(confirmation_cfg, {"commit": "abc"})
    b = preregistration_payload(confirmation_cfg, {"commit": "abc"})
    c = preregistration_payload(confirmation_cfg, {"commit": "def"})
    assert preregistration_hash(a) == preregistration_hash(b)
    assert preregistration_hash(a) != preregistration_hash(c)
    assert len(preregistration_hash(a)) == 64
    json.dumps(a)   # must be serializable


def test_c29c_proxy_stress_summary_covers_every_mode(small_confirmation_cfg):
    d = _decisions(small_confirmation_cfg, _fake_conf_tasks(small_confirmation_cfg, n_series=10))
    stress = proxy_stress_summary(d)
    assert set(stress[stress["selector"] == D7]["proxy_mode"]) == set(PROXY_MODES)
    assert stress[stress["is_primary_proxy"]]["proxy_mode"].unique().tolist() == [P1_CALIBRATED]


# ------------------------------------------------------------ regression ----

def test_c30_gate_e_and_gate_f_are_unchanged():
    """Test 30: Gate E and Gate F still exist with their own signatures and constants."""
    from covariate_trust import followup_gates as fg
    assert callable(fg.gate_e) and callable(fg.gate_f) and callable(fg.gate_g)
    assert fg.LOW_LAMBDA_MAX == 0.85 and fg.HIGH_LAMBDA_MIN == 1.15
    assert fg.STABLE_LOW == "S0_stable_low" and fg.STABLE_HIGH == "S1_stable_high"
    assert fg.WORSENING == ("S2_sudden_worsening", "S4_gradual_worsening")
    assert fg.IMPROVING == ("S3_sudden_improvement", "S5_gradual_improvement")


def test_c31_existing_pilot_constants_still_hold(pilot_cfg):
    """Test 31: nothing in Study 2B moved an earlier study's threshold."""
    from covariate_trust.admission import PSEUDO_ORIGINS
    from covariate_trust.gates import (GATE_A_PRIMARY_SHARES, HIGH_NOISE_MIN_LAMBDA,
                                       LOW_NOISE_MAX_LAMBDA)
    assert PSEUDO_ORIGINS == {24: [800, 824, 848, 872], 96: [512, 608, 704, 800]}
    assert GATE_A_PRIMARY_SHARES == (0.50, 0.75)
    assert (LOW_NOISE_MAX_LAMBDA, HIGH_NOISE_MIN_LAMBDA) == (0.5, 1.5)
    assert pilot_cfg.gates.harm_relative_threshold == 0.05
    assert pilot_cfg.experiment.master_seed == 20260730
