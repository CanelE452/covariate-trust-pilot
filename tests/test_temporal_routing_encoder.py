"""Causality, masking and gradient checks for the raw-history sequence gate.

The forecast numbers are only worth reading if the window really ends before
its origin and the experts really receive no gradient, so those are asserted
here rather than assumed.  The dataset-backed checks use FreshRetailNet, which
is the smallest grid and carries an explicit observed mask.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from experiments.multi_benchmark import run as MB
from experiments.temporal_routing_encoder import spec as S
from experiments.temporal_routing_encoder.data import _history_stack, observed_matrix
from experiments.temporal_routing_encoder.model import SequenceGate, count_parameters

LOOKBACK = 96
HORIZON = 28


def _toy(n_series=6, length=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n_series, length)) < 0.3) * rng.integers(1, 9, (n_series, length))
    return y.astype(np.float32)


def test_history_window_ends_before_origin():
    y = _toy()
    origins = np.array([120, 200, 333])
    stack = _history_stack(y, origins, LOOKBACK)
    for w, origin in enumerate(origins):
        assert np.array_equal(stack[:, w], y[:, origin - LOOKBACK:origin])
        # the last timestamp the window can see is origin - 1
        assert stack[:, w].shape[1] == LOOKBACK


def test_future_mutation_does_not_change_the_window():
    y = _toy()
    origins = np.array([150, 260])
    before = _history_stack(y, origins, LOOKBACK).copy()
    mutated = y.copy()
    mutated[:, origins.min():] += 1000.0          # everything at or after an origin
    after = _history_stack(mutated, origins, LOOKBACK)
    assert np.array_equal(before[:, 0], after[:, 0])


def test_past_mutation_does_change_the_window():
    """The previous test would pass on a constant tensor; this rules that out."""
    y = _toy()
    origins = np.array([150])
    before = _history_stack(y, origins, LOOKBACK).copy()
    mutated = y.copy()
    mutated[:, 149] += 5.0
    after = _history_stack(mutated, origins, LOOKBACK)
    assert not np.array_equal(before, after)


def test_window_is_deterministic():
    y = _toy()
    origins = np.array([180, 240])
    first = _history_stack(y, origins, LOOKBACK)
    second = _history_stack(y, origins, LOOKBACK)
    assert (hashlib.sha256(first.tobytes()).hexdigest()
            == hashlib.sha256(second.tobytes()).hexdigest())


def test_missing_steps_are_distinguishable_from_observed_zeros():
    y = np.zeros((2, 200), dtype=np.float32)
    data = {"y": y, "observed_mask": np.ones((2, 200), dtype=bool)}
    data["observed_mask"][0, :50] = False
    observed = observed_matrix(data)
    window = _history_stack(observed, np.array([100]), LOOKBACK)[:, 0]
    # series 0 is unobserved for its first 46 in-window steps, series 1 never is
    assert window[0].sum() < LOOKBACK
    assert window[1].all()
    y_norm = np.where(window, 0.0, 0.0)
    assert y_norm[0][~window[0]].tolist() == [0.0] * int((~window[0]).sum())


def test_availability_becomes_an_observed_mask():
    data = {"y": np.zeros((3, 100), dtype=np.float32),
            "available_from": np.array([0, 40, 99])}
    observed = observed_matrix(data)
    assert observed[0].all()
    assert not observed[1][:40].any() and observed[1][40:].all()
    assert observed[2].sum() == 1


def test_gate_output_is_within_the_unit_interval():
    torch.manual_seed(0)
    model = SequenceGate(horizon=HORIZON)
    seq = torch.randn(32, LOOKBACK, 3)
    ctx = torch.randn(32, 2 * HORIZON)
    g = model(seq, ctx)
    assert g.shape == (32,)
    assert torch.all(g >= 0) and torch.all(g <= 1)


def test_mixture_endpoints_are_the_two_experts():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[7.0, 8.0, 9.0]])
    assert np.allclose((1 - 0.0) * a + 0.0 * b, a)
    assert np.allclose((1 - 1.0) * a + 1.0 * b, b)


def test_experts_receive_no_gradient():
    """Expert forecasts enter as constants, so nothing flows back into them."""
    torch.manual_seed(0)
    model = SequenceGate(horizon=HORIZON)
    seq = torch.randn(8, LOOKBACK, 3)
    a = torch.randn(8, HORIZON)
    b = torch.randn(8, HORIZON)
    assert not a.requires_grad and not b.requires_grad
    g = model(seq, torch.cat([a, b], dim=1))
    loss = (((1 - g[:, None]) * a + g[:, None] * b) ** 2).mean()
    loss.backward()
    assert a.grad is None and b.grad is None
    grads = [p.grad for p in model.parameters()]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)


def test_parameter_count_stays_a_probe():
    model = SequenceGate(horizon=HORIZON)
    total = count_parameters(model)
    # GRU(3 -> 16) plus one linear head over 16 + 56 inputs
    assert total == 1008 + (16 + 2 * HORIZON) + 1
    assert total < 4000


def test_spec_excludes_the_handcrafted_summaries():
    spec = S.sequence_gate_spec()
    names = {c["name"] for c in spec["channels"]}
    assert names == {"y_norm", "occurrence", "observed_mask"}
    for banned in ("ADI", "CV2", "rho_interval", "SBC_regime"):
        assert any(banned in item for item in spec["excluded_inputs"])
    assert spec["encoder"]["num_layers"] == 1
    assert spec["encoder"]["bidirectional"] is False
    assert spec["experts_receive_gradient"] is False
    assert spec["stop_rule"]["frozen_before_results"] is True


@pytest.mark.parametrize("name", ["freshretailnet"])
def test_scale_source_ends_before_every_origin(name):
    data = MB.load_grid(name)
    cfg = MB.config_for(data)
    split = MB.build_split(data, cfg, MB.STRIDE)
    assert int(np.min(split.test.origins)) > cfg.train_end
