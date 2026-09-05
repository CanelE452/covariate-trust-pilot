"""Track G core: shared-parameter ownership audit, per-task gradients,
train-only temporal probes, and exact virtual-update harm.

A "task" is one output variable. Gradients are compared only on parameters that
every channel's forward path uses identically; parameters carrying an explicit
per-channel axis are excluded and the audit is written to an artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

import engine as E
from contract import PRED_LEN, SEQ_LEN
import data as D


# --------------------------------------------------------------------------- #
# Shared-parameter ownership audit
# --------------------------------------------------------------------------- #

def audit_shared_parameters(model: nn.Module, enc_in: int) -> dict:
    """Classify parameters as shared or per-channel by inspecting shapes.

    In the official TQNet, temporalQuery has shape (cycle_len, enc_in): its
    second axis indexes channels, so it is NOT a shared parameter. Every other
    parameter (the channel aggregator, the input projection, the MLP and the
    output projection) is applied identically to every channel.
    """
    shared, private, detail = [], [], {}
    for name, p in model.named_parameters():
        per_channel = (enc_in in tuple(p.shape)) and name.endswith("temporalQuery")
        (private if per_channel else shared).append(name)
        detail[name] = {"shape": list(p.shape), "numel": int(p.numel()),
                        "class": "per_channel" if per_channel else "shared"}
    return {"shared_parameters": shared, "per_channel_parameters": private,
            "detail": detail,
            "n_shared_scalars": int(sum(detail[n]["numel"] for n in shared)),
            "rule": ("A parameter is per-channel when it carries an explicit enc_in axis "
                     "that indexes output variables (TQNet temporalQuery). All other "
                     "parameters take part identically in every channel's forward path.")}


def shared_param_list(model: nn.Module, enc_in: int):
    names = set(audit_shared_parameters(model, enc_in)["shared_parameters"])
    return [(n, p) for n, p in model.named_parameters() if n in names]


# --------------------------------------------------------------------------- #
# Per-task loss and gradient
# --------------------------------------------------------------------------- #

def task_loss(model, cfg, x, y, cyc, i: int):
    """MSE across batch and all 96 forecast steps, for output channel i only."""
    out = E.forward(model, cfg, x, cyc)[:, -PRED_LEN:, i]
    return ((out - y[:, -PRED_LEN:, i]) ** 2).mean()


def task_grad(model, cfg, x, y, cyc, i: int, params) -> torch.Tensor:
    """Flat gradient of task i over the shared parameters."""
    loss = task_loss(model, cfg, x, y, cyc, i)
    gs = torch.autograd.grad(loss, [p for _, p in params], retain_graph=False,
                             allow_unused=True)
    flat = [torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for (_, p), g in zip(params, gs)]
    return torch.cat(flat)


def all_task_grads(model, cfg, x, y, cyc, tasks, params) -> torch.Tensor:
    return torch.stack([task_grad(model, cfg, x, y, cyc, i, params) for i in tasks])


def cosine_matrix(G: torch.Tensor) -> np.ndarray:
    Gn = G / (G.norm(dim=1, keepdim=True) + 1e-12)
    return (Gn @ Gn.T).detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Train-only temporal probe pairs
# --------------------------------------------------------------------------- #

def probe_pairs(dataset: str, n_pairs: int, batch: int, seed: int, gap: int = None):
    """Non-overlapping optimisation / probe blocks, both inside the train split.

    Each pair is two disjoint temporal blocks of the train split separated by at
    least one full window so that no sample can appear on both sides.
    """
    n = len(D.get_dataset(dataset, "train"))
    span = batch                        # contiguous window starts per block
    if gap is None:
        gap = SEQ_LEN + PRED_LEN        # no shared timestep between blocks
    rng = np.random.RandomState(seed)
    pairs = []
    lo, hi = 0, n - (2 * span + gap)
    for k in range(n_pairs):
        a = int(rng.randint(lo, max(lo + 1, hi)))
        b = a + span + gap
        pairs.append({"train_starts": list(range(a, a + span)),
                      "probe_starts": list(range(b, b + span)),
                      "train_block": [a, a + span - 1],
                      "probe_block": [b, b + span - 1]})
    return pairs


def blocks_disjoint(pair) -> bool:
    """A pair is valid when no timestep is read by both blocks."""
    a0, a1 = pair["train_block"]
    b0, b1 = pair["probe_block"]
    a_end = a1 + SEQ_LEN + PRED_LEN     # last timestep the train block reads
    b_end = b1 + SEQ_LEN + PRED_LEN
    return (a_end < b0) or (b_end < a0)


# --------------------------------------------------------------------------- #
# Exact virtual update
# --------------------------------------------------------------------------- #

def erm_step_norm(model, cfg, x, y, cyc, params) -> float:
    """Norm of one ordinary ERM gradient step over the shared parameters."""
    out = E.forward(model, cfg, x, cyc)[:, -PRED_LEN:, :]
    loss = ((out - y[:, -PRED_LEN:, :]) ** 2).mean()
    gs = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    flat = torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                      for (_, p), g in zip(params, gs)])
    return float(flat.norm())


@torch.no_grad()
def virtual_probe_loss(model, cfg, base_state, delta_flat, params, x, y, cyc, i):
    """Loss of task i under theta + delta, without ever mutating the model."""
    shapes = [(n, p.shape, p.numel()) for n, p in params]
    over = {}
    off = 0
    for n, shp, k in shapes:
        over[n] = base_state[n] + delta_flat[off:off + k].view(shp)
        off += k
    full = dict(base_state)
    full.update(over)
    out = functional_call(model, full, (x, cyc) if cfg.model in E.NEEDS_CYCLE else (x,))
    out = out[:, -PRED_LEN:, i]
    return float(((out - y[:, -PRED_LEN:, i]) ** 2).mean())


def exact_harm(model, cfg, base_state, params, xb, yb, cb, xp, yp, cp,
               j: int, i: int, step_norm: float) -> float:
    """Relative change of probe loss for variable i after one normalised step
    along task j's gradient. H > 0.005 counts as harm.
    """
    gj = task_grad(model, cfg, xb, yb, cb, j, params)
    delta = -step_norm * gj / (gj.norm() + 1e-12)
    with torch.no_grad():
        base = virtual_probe_loss(model, cfg, base_state, torch.zeros_like(delta),
                                  params, xp, yp, cp, i)
        moved = virtual_probe_loss(model, cfg, base_state, delta, params, xp, yp, cp, i)
    return (moved - base) / (base + 1e-12)
