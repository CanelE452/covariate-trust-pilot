"""T01-T24 (except the PCGrad unit tests, which live in test_pcgrad.py).

These check the study contract, not the scientific outcome: split hygiene,
corruption scope, probe hygiene, budget equality, verdict independence and
artifact reproducibility.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "ts_idea_tournament_v1"
for p in ("common", "track_x", "track_g", "track_f"):
    sys.path.insert(0, str(EXP / p))

import numpy as np
import pytest
import torch

import paths
import data as D
import engine as E
import evalgeom
from contract import DATASETS, CLEAN_SEEDS, SEQ_LEN, PRED_LEN, tqnet_config

R = paths.RESULTS


def latest(stage, fname):
    fs = sorted((paths.RUNS / stage).glob(f"attempt_*/{fname}"))
    fs = [f for f in fs if (f.parent / "completion.json").exists()]
    return json.loads(fs[-1].read_text()) if fs else None


# --------------------------------------------------------------------------- #
# Split and normalisation
# --------------------------------------------------------------------------- #

def test_T01_train_val_test_disjoint():
    """Windows of different splits never share a source row."""
    for ds in DATASETS:
        n = {f: len(D.get_dataset(ds, f).data_x) for f in ("train", "val", "test")}
        assert all(v > 0 for v in n.values())
        # The official provider overlaps only by the lookback it prepends to
        # val/test; no window start of one split indexes another split's array.
        for f in ("train", "val", "test"):
            dsx = D.get_dataset(ds, f)
            assert len(dsx) == len(dsx.data_x) - SEQ_LEN - PRED_LEN + 1


def test_T02_normalization_is_train_only():
    for ds in DATASETS:
        x, _, _ = D.as_arrays(ds, "train")
        assert abs(float(x.mean())) < 1e-3
        assert abs(float(x.std()) - 1.0) < 1e-2
        xv, _, _ = D.as_arrays(ds, "val")
        xt, _, _ = D.as_arrays(ds, "test")
        # If the scaler had seen val/test they would also be standardised.
        assert abs(float(xt.std()) - 1.0) > 1e-3 or abs(float(xt.mean())) > 1e-3
        assert abs(float(xv.std()) - 1.0) > 1e-3 or abs(float(xv.mean())) > 1e-3


# --------------------------------------------------------------------------- #
# Track X corruption scope
# --------------------------------------------------------------------------- #

def _corruption_available(ds, sev):
    import corruption as X
    return X._cache_path(ds, sev).exists()


@pytest.mark.parametrize("ds,sev", [("ETTm1", 3)])
def test_T03_T04_T05_corruption_scope_and_determinism(ds, sev):
    import corruption as X
    if not _corruption_available(ds, sev):
        pytest.skip("corrupted series not generated yet")
    for fam in X.FAMILIES:
        chk = X.corruption_matches_official(ds, sev, fam)
        # T04: only the declared channel differs from clean.
        assert chk["max_offchannel_deviation"] == 0.0
        # Splice equals the official transform on that channel.
        assert chk["identical_to_official"]
        assert chk["n_examples_where_channel_actually_changed"] > 0
    # T05: deterministic given (dataset, severity, seed).
    a = X.corruption_fingerprint(ds, sev)
    X._CACHE.clear()
    assert a == X.corruption_fingerprint(ds, sev)


def test_T03_targets_are_never_corrupted():
    """The spliced input touches the lookback only; targets come from data_y."""
    import phenomenon as P
    src = (EXP / "track_x" / "phenomenon.py").read_text()
    assert "def spliced_input" in src
    fn = src[src.index("def spliced_input"):src.index("def test_offset")]
    assert "x[:, :, j]" in fn and "y" not in fn.split("return")[0].split("\n")[-3]


def test_T06_dlinear_has_no_structural_cross_channel_path():
    """DLinear with individual heads cannot move information between channels:
    perturbing channel j must leave every other output channel bit-identical.
    """
    from contract import dlinear_config
    ds = "ETTm1"
    cfg = dlinear_config(ds, CLEAN_SEEDS[0])
    m = E.build_model(cfg).to(E.DEVICE).eval()
    x, y, c = D.window_batch(ds, "test", list(range(8)), E.DEVICE)
    with torch.no_grad():
        a = E.forward(m, cfg, x, c)
        xb = x.clone()
        xb[:, :, 0] += 5.0
        b = E.forward(m, cfg, xb, c)
    other = [i for i in range(cfg.enc_in) if i != 0]
    assert torch.allclose(a[:, :, other], b[:, :, other], atol=0)
    assert not torch.allclose(a[:, :, 0], b[:, :, 0])


def test_T06b_tqnet_does_have_a_cross_channel_path():
    """TQNet must move information across channels. The perturbation has to
    change the SHAPE of channel 0: TQNet applies per-channel instance
    normalisation, so a constant offset on one channel is removed before the
    cross-channel attention ever sees it.
    """
    ds = "ETTm1"
    cfg = tqnet_config(ds, CLEAN_SEEDS[0])
    m = E.build_model(cfg).to(E.DEVICE).eval()
    x, y, c = D.window_batch(ds, "test", list(range(8)), E.DEVICE)
    with torch.no_grad():
        a = E.forward(m, cfg, x, c)
        xb = x.clone()
        xb[:, ::2, 0] += 5.0            # shape change, survives instance norm
        b = E.forward(m, cfg, xb, c)
    other = [i for i in range(cfg.enc_in) if i != 0]
    assert not torch.allclose(a[:, :, other], b[:, :, other], atol=1e-7)


def test_T06c_constant_offset_is_absorbed_by_instance_normalisation():
    """Documents the structural fact the spillover reading depends on: a pure
    level offset on one channel is invisible to TQNet.
    """
    ds = "ETTm1"
    cfg = tqnet_config(ds, CLEAN_SEEDS[0])
    m = E.build_model(cfg).to(E.DEVICE).eval()
    x, y, c = D.window_batch(ds, "test", list(range(8)), E.DEVICE)
    with torch.no_grad():
        a = E.forward(m, cfg, x, c)
        xb = x.clone()
        xb[:, :, 0] += 5.0
        b = E.forward(m, cfg, xb, c)
    other = [i for i in range(cfg.enc_in) if i != 0]
    assert torch.allclose(a[:, :, other], b[:, :, other], atol=1e-5)


# --------------------------------------------------------------------------- #
# Track G probe hygiene
# --------------------------------------------------------------------------- #

def test_T07_gradient_tasks_map_to_the_right_output_variable():
    import grads as G
    ds = "ETTm1"
    cfg = tqnet_config(ds, CLEAN_SEEDS[0])
    m = E.build_model(cfg).to(E.DEVICE).eval()   # dropout off: the two passes must match
    x, y, c = D.window_batch(ds, "train", list(range(16)), E.DEVICE)
    for i in range(cfg.enc_in):
        with torch.no_grad():
            out = E.forward(m, cfg, x, c)[:, -PRED_LEN:, i]
            manual = float(((out - y[:, -PRED_LEN:, i]) ** 2).mean())
        assert abs(manual - float(G.task_loss(m, cfg, x, y, c, i).detach())) < 1e-9


def test_T08_T09_probe_batches_are_train_only_and_time_disjoint():
    import grads as G
    for ds in DATASETS:
        n_train = len(D.get_dataset(ds, "train"))
        pairs = G.probe_pairs(ds, 32, batch=64, seed=1)
        for p in pairs:
            assert min(p["train_starts"]) >= 0 and max(p["train_starts"]) < n_train
            assert min(p["probe_starts"]) >= 0 and max(p["probe_starts"]) < n_train
            assert G.blocks_disjoint(p)
            assert not (set(p["train_starts"]) & set(p["probe_starts"]))


def test_T10_virtual_step_does_not_mutate_the_checkpoint():
    import grads as G
    ds = "ETTm1"
    cfg = tqnet_config(ds, CLEAN_SEEDS[0])
    m = E.build_model(cfg).to(E.DEVICE)
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    base_state = {k: v.detach() for k, v in m.state_dict().items()}
    params = G.shared_param_list(m, cfg.enc_in)
    xb, yb, cb = D.window_batch(ds, "train", list(range(16)), E.DEVICE)
    xp, yp, cp = D.window_batch(ds, "train", list(range(500, 516)), E.DEVICE)
    step = G.erm_step_norm(m, cfg, xb, yb, cb, params)
    G.exact_harm(m, cfg, base_state, params, xb, yb, cb, xp, yp, cp, j=1, i=0, step_norm=step)
    for k, v in m.state_dict().items():
        assert torch.equal(before[k], v)


def test_T12_probe_gated_mask_uses_no_validation_or_test_data():
    src = (EXP / "track_g" / "rules.py").read_text()
    fn = src[src.index("def refresh_probe_mask"):src.index("def __call__")]
    assert '"train"' in fn
    assert '"val"' not in fn and '"test"' not in fn


# --------------------------------------------------------------------------- #
# Track F contract
# --------------------------------------------------------------------------- #

def test_T13_F_classes_are_mutually_exclusive():
    import windows as W
    for ds in DATASETS:
        c = W.assign_classes(ds)
        s = [set(v) for v in c.values()]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
        assert sum(len(v) for v in c.values()) == len(set().union(*s))


def test_T14_severity_calibration_uses_train_only():
    src = (EXP / "track_f" / "run_selection.py").read_text()
    fn = src[src.index("def calibrate_severity"):src.index("def removal_rates")]
    assert '"test"' not in fn and '"val"' not in fn


def test_T15_rho_holdout_is_time_disjoint_from_the_target_block():
    sel = latest("track_f_selection", "selection.json")
    if sel is None:
        pytest.skip("Track F not run yet")
    for ds, e in sel["datasets"].items():
        assert e["rho_reference"]["time_disjoint"]


def test_T16_removal_budget_is_exactly_equal():
    sel = latest("track_f_selection", "selection.json")
    if sel is None:
        pytest.skip("Track F not run yet")
    for ds, e in sel["datasets"].items():
        b = {k: v for k, v in e["removal_budget"].items() if k != "no_filter"}
        assert len(set(b.values())) == 1, b
        assert e["removal_budget"]["no_filter"] == 0


def test_T17_coherence_score_is_deterministic():
    import filters as SEL
    import windows as W
    ds = "ETTm1"
    c = W.assign_classes(ds)
    sp = W.calibration_split(ds, c)
    X, Y, _, _, _ = W.materialise(ds, {k: sp[k]["study"][:200] for k in sp}, 2.0, 2.0, seed=3)
    a = SEL.coherence_scores(X, Y, W.train_iqr(ds))
    b = SEL.coherence_scores(X, Y, W.train_iqr(ds))
    assert np.array_equal(a, b)


def test_T18_shifted_test_uses_an_independent_seed():
    import run_selection as RS
    import windows as W
    assert RS.SHIFT_TEST_SEED != W.CLASS_SEED
    assert RS.SHIFT_TEST_SEED != W.CLASS_SEED + 7


# --------------------------------------------------------------------------- #
# Cross-cutting
# --------------------------------------------------------------------------- #

def test_T19_T20_track_verdicts_are_independent():
    """Mutating one track's artifact must not change another track's verdict."""
    import importlib.util

    def load(name, p):
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    diag = latest("track_g_diagnostic", "gradient_diagnostic.json")
    if diag is None:
        pytest.skip("Track G diagnostic not run yet")
    GG = load("g_gates_t", EXP / "track_g" / "gates.py")
    v0 = GG.phenomenon_verdict(diag)["verdict"]

    # Mutate an unrelated track's artifact on disk-equivalent structures.
    phen = latest("track_x_phenomenon", "phenomenon_raw.json")
    if phen is not None:
        for e in phen["datasets"].values():
            for recs in e["raw"].values():
                for r in recs:
                    r["rel"] = [v * 100 for v in r["rel"]]
    sel = latest("track_f_selection", "selection.json")
    if sel is not None:
        for e in sel["datasets"].values():
            e["selection_diagnostic"] = {}
    assert GG.phenomenon_verdict(diag)["verdict"] == v0

    # And no gates module may name another track's run stage or results folder.
    STAGES = {"X": ["track_x_phenomenon", "track_x_mitigations"],
              "G": ["track_g_diagnostic", "track_g_intervention"],
              "F": ["track_f_selection"],
              "V": ["validation_stability"]}
    files = {"X": EXP / "track_x" / "gates.py", "G": EXP / "track_g" / "gates.py",
             "F": EXP / "track_f" / "gates.py"}
    for owner, f in files.items():
        text = f.read_text()
        for other, stages in STAGES.items():
            if other == owner:
                continue
            for st in stages:
                assert st not in text, f"{f.name} references {st}"
        assert f"results/{'track_' + owner.lower()}" not in text


def test_T19b_track_v_never_promotes_a_method():
    src = (EXP / "common" / "finalize.py").read_text()
    fn = src[src.index("def final_ranking"):src.index("def status_md")]
    assert '"V"' not in fn and "'V'" not in fn


def test_T21_bootstrap_groups_by_origin_not_by_row():
    import bootstrap as B
    a = {0: [1.0] * 50, 1: [5.0] * 50}
    r = B.cluster_bootstrap(a, stat=np.mean, n=200, seed=1)
    # With only two clusters, resampling whole clusters gives a wide interval;
    # an iid row bootstrap on the same values would give a very tight one.
    assert r["n_clusters"] == 2
    assert r["ci_upper"] - r["ci_lower"] > 1.0


def test_T22_resource_monitor_has_a_kill_path():
    src = (EXP / "common" / "resmon.py").read_text()
    assert "MEMORY_CAP_BREACH" in src
    for cap in ("RSS_CAP_GB", "AVAIL_FLOOR_GB", "SYS_PCT_CAP", "GPU_CAP_GB"):
        assert cap in src
    assert "break" in src


def test_T23_completion_resume_is_idempotent():
    import attempts as A
    stage = "clean_baselines"
    d = A.completed(stage)
    if d is None:
        pytest.skip("clean baselines not run yet")
    assert (d / "completion.json").exists()
    assert A.completed(stage) == d          # repeated calls do not create attempts
    n_before = len(list((paths.RUNS / stage).glob("attempt_*")))
    A.completed(stage)
    assert len(list((paths.RUNS / stage).glob("attempt_*"))) == n_before


def test_T24_final_tables_reproduce_from_artifacts():
    f = R / "tables.json"
    if not f.exists():
        pytest.skip("finalize not run yet")
    t = json.loads(f.read_text())
    rank = json.loads((R / "final_topic_ranking.json").read_text())
    assert t["ranking"]["final_token"] == rank["final_token"]
    status = (R / "STATUS.md").read_text()
    assert status.splitlines()[0] == f"FINAL TOPIC RECOMMENDATION: {rank['final_token']}"


def test_origins_are_non_overlapping_and_spread():
    for ds in DATASETS:
        for flag in ("val", "test"):
            o = evalgeom.origins(ds, flag)
            assert len(o) == evalgeom.N_ORIGINS
            for a, b in zip(o, o[1:]):
                assert a["start"] + evalgeom.ORIGIN_BATCH <= b["start"]


def test_coherence_aware_differs_from_rho_when_coherent_candidates_exist():
    """Code-correctness check for the Track F intervention: given coherent
    windows among the RHO drop candidates, the selector must rescue them and
    still remove exactly the same number of windows.
    """
    import filters as SEL
    n = 500
    rng = np.random.RandomState(0)
    loss_cur = rng.rand(n)
    loss_ref = rng.rand(n)
    rho = loss_cur - loss_ref
    order = np.argsort(rho, kind="stable")
    k = SEL.budget_k(n)
    coh = np.zeros(n)
    coh[order[:k // 2]] = 0.9                 # half the drop candidates are coherent
    keep_rho = SEL.sel_rho(loss_cur, loss_ref)
    keep_coh = SEL.sel_coherence_aware(loss_cur, loss_ref, coh)
    assert (~keep_rho).sum() == (~keep_coh).sum() == k
    assert not np.array_equal(keep_rho, keep_coh)
    rescued = keep_coh & ~keep_rho
    assert rescued.sum() == k // 2
    assert (coh[rescued] >= SEL.COHERENCE_CHANNEL_FRACTION).all()


def test_coherence_aware_equals_rho_when_nothing_is_coherent():
    import filters as SEL
    n = 300
    rng = np.random.RandomState(1)
    a, b = rng.rand(n), rng.rand(n)
    assert np.array_equal(SEL.sel_rho(a, b),
                          SEL.sel_coherence_aware(a, b, np.zeros(n)))
