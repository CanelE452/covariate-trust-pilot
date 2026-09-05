"""T11: PCGrad unit tests on synthetic gradients (orthogonal, aligned,
opposite, three-task), checked against the closed form of Algorithm 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "ts_idea_tournament_v1" / "track_g"))
sys.path.insert(0, str(ROOT / "experiments" / "ts_idea_tournament_v1" / "common"))

import numpy as np
import torch

from rules import pcgrad_project, norm_balanced


def rng(seed=0):
    return np.random.RandomState(seed)


def test_orthogonal_gradients_are_untouched():
    g = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = pcgrad_project(g, rng())
    assert torch.allclose(out, g.mean(0), atol=1e-6)


def test_aligned_gradients_are_untouched():
    g = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    out = pcgrad_project(g, rng())
    assert torch.allclose(out, g.mean(0), atol=1e-6)


def test_opposite_gradients_cancel_to_zero():
    # g1 = -g2 exactly: each projection removes the whole component.
    g = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    out = pcgrad_project(g, rng())
    assert torch.allclose(out, torch.zeros(2), atol=1e-6)


def test_partial_conflict_matches_closed_form():
    g = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])   # dot = 0 -> no conflict
    assert torch.allclose(pcgrad_project(g, rng()), g.mean(0), atol=1e-6)

    g = torch.tensor([[1.0, 1.0], [-2.0, 1.0]])   # dot = -1 -> both project
    g0 = g[0] - (g[0].dot(g[1]) / g[1].dot(g[1])) * g[1]
    g1 = g[1] - (g[1].dot(g[0]) / g[0].dot(g[0])) * g[0]
    expected = torch.stack([g0, g1]).mean(0)
    assert torch.allclose(pcgrad_project(g, rng()), expected, atol=1e-6)


def test_three_task_projection_removes_all_negative_dots():
    torch.manual_seed(0)
    g = torch.tensor([[1.0, 0.0, 0.0], [-0.8, 0.6, 0.0], [-0.5, -0.5, 0.7]])
    out = pcgrad_project(g, rng(3))
    # The averaged update must not be a plain mean when conflicts exist.
    assert not torch.allclose(out, g.mean(0), atol=1e-6)
    # And it must stay finite and of the right shape.
    assert out.shape == (3,) and torch.isfinite(out).all()


def test_mask_disables_projection():
    g = torch.tensor([[1.0, 1.0], [-2.0, 1.0]])
    mask = torch.zeros(2, 2, dtype=torch.bool)
    assert torch.allclose(pcgrad_project(g, rng(), mask=mask), g.mean(0), atol=1e-6)
    full = torch.ones(2, 2, dtype=torch.bool)
    assert torch.allclose(pcgrad_project(g, rng(), mask=full),
                          pcgrad_project(g, rng()), atol=1e-6)


def test_norm_balanced_preserves_direction_and_equalises_norm():
    g = torch.tensor([[3.0, 4.0], [0.1, 0.0], [0.0, 2.0]])
    n = g.norm(dim=1)
    med = n.median()
    scaled = g * (med / n).unsqueeze(1)
    assert torch.allclose(norm_balanced(g), scaled.mean(0), atol=1e-6)
    for a, b in zip(g, scaled):
        cos = torch.dot(a, b) / (a.norm() * b.norm())
        assert cos > 1 - 1e-6           # direction unchanged
        assert abs(float(b.norm() - med)) < 1e-5
