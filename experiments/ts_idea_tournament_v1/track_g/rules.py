"""Track G gradient rules: ERM, PCGrad, norm-balanced ERM, probe-gated PCGrad.

PCGrad follows Algorithm 1 of Yu et al. 2020 (arXiv:2001.06782) directly; the
official release is TensorFlow, so no unofficial PyTorch port is used as
authority. The implementation is verified by synthetic gradient unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import torch

import engine as E
from contract import PRED_LEN
import data as D
import grads as G

PROBE_REFRESH_EVERY = 8       # optimizer steps between probe-mask refreshes
PROBE_BATCH = 64


def pcgrad_project(grads: torch.Tensor, order_rng: np.random.RandomState,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
    """Project conflicting gradients, then average.

    grads: [T, P]. mask (if given) is a [T, T] boolean gate; g_i is only
    projected off g_j when mask[i, j] is True, on top of the usual
    g_i . g_j < 0 conflict test.
    """
    T = grads.shape[0]
    out = grads.clone()
    for i in range(T):
        order = order_rng.permutation(T)
        for j in order:
            if int(j) == i:
                continue
            if mask is not None and not bool(mask[i, j]):
                continue
            gj = grads[j]
            dot = torch.dot(out[i], gj)
            if dot < 0:
                out[i] = out[i] - dot / (gj.dot(gj) + 1e-12) * gj
    return out.mean(dim=0)


def norm_balanced(grads: torch.Tensor) -> torch.Tensor:
    """Rescale every task gradient to the batch's median task-gradient norm.
    Directions are untouched; only magnitudes are balanced, then averaged.
    """
    n = grads.norm(dim=1, keepdim=True)
    med = n.median()
    return (grads * (med / (n + 1e-12))).mean(dim=0)


def assign_flat_grad(params, flat: torch.Tensor) -> None:
    off = 0
    for _, p in params:
        k = p.numel()
        p.grad = flat[off:off + k].view_as(p).clone()
        off += k


class GradRule:
    """Callable passed to engine.train_model as grad_rule."""

    def __init__(self, kind: str, dataset: str, tasks: list, seed: int):
        self.kind = kind
        self.dataset = dataset
        self.tasks = tasks
        self.order_rng = np.random.RandomState(seed)
        self.probe_rng = np.random.RandomState(seed + 1)
        self.mask = None
        self.extra_forward = 0
        self.extra_backward = 0
        n = len(D.get_dataset(dataset, "train"))
        self.n_train = n
        self._params = None

    def params(self, model, cfg):
        if self._params is None:
            self._params = G.shared_param_list(model, cfg.enc_in)
            self._pset = {id(p) for _, p in self._params}
        return self._params

    def refresh_probe_mask(self, model, cfg, params):
        """Recompute the harmful cross-probe affinity gate from a train-only
        probe batch. No validation or test data is ever touched.
        """
        s = int(self.probe_rng.randint(0, self.n_train - PROBE_BATCH))
        xp, yp, cp = D.window_batch(self.dataset, "train",
                                    list(range(s, s + PROBE_BATCH)), E.DEVICE)
        Gp = G.all_task_grads(model, cfg, xp, yp, cp, self.tasks, params)
        self.extra_forward += len(self.tasks)
        self.extra_backward += len(self.tasks)
        self._probe = Gp / (Gp.norm(dim=1, keepdim=True) + 1e-12)
        self._probe_start = s

    def __call__(self, model, cfg, bx, by, cyc, crit, step):
        params = self.params(model, cfg)
        if self.kind == "erm":
            out = E.forward(model, cfg, bx, cyc)[:, -PRED_LEN:, :]
            loss = crit(out, by[:, -PRED_LEN:, :])
            loss.backward()
            return loss

        Gb = G.all_task_grads(model, cfg, bx, by, cyc, self.tasks, params)
        mask = None
        if self.kind == "probe_gated":
            if step % PROBE_REFRESH_EVERY == 0 or self.mask is None:
                self.refresh_probe_mask(model, cfg, params)
            Gbn = Gb / (Gb.norm(dim=1, keepdim=True) + 1e-12)
            aff = -(self._probe @ Gbn.T)          # aff[i, j] harmful when > 0
            mask = aff > 0
            self.mask = mask

        if self.kind == "pcgrad":
            flat = pcgrad_project(Gb, self.order_rng)
        elif self.kind == "norm_balanced":
            flat = norm_balanced(Gb)
        elif self.kind == "probe_gated":
            flat = pcgrad_project(Gb, self.order_rng, mask=mask)
        else:
            raise ValueError(self.kind)

        # Tasks cover only the selected output variables; the remaining
        # parameters still take the plain ERM gradient so the comparison stays
        # a comparison of the shared-parameter update rule alone.
        out = E.forward(model, cfg, bx, cyc)[:, -PRED_LEN:, :]
        loss = crit(out, by[:, -PRED_LEN:, :])
        loss.backward()
        assign_flat_grad(params, flat)
        return loss.detach()
