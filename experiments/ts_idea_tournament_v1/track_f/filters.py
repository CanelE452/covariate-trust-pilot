"""Track F selection rules, all sharing an exactly equal 20% removal budget.

no_filter, random_removal, high_loss_removal, rho_loss, adarho, coherence_aware.
The coherence-aware rule is the only new intervention: it starts from the RHO
ranking and rescues windows whose input and target move together.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import torch

import engine as E
from contract import SEQ_LEN, PRED_LEN

REMOVAL_BUDGET = 0.20
COHERENCE_A_MIN = 1.0          # |A| >= 1 train IQR
COHERENCE_B_MIN = 1.0          # |B| >= 1 train IQR
COHERENCE_CHANNEL_FRACTION = 0.30


@torch.no_grad()
def window_losses(model, cfg, X, Y, batch=512) -> np.ndarray:
    """Per-window MSE of a fixed model over an explicit window array."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).float().to(E.DEVICE)
        y = torch.from_numpy(Y[i:i + batch]).float().to(E.DEVICE)
        cyc = torch.zeros(len(x), dtype=torch.int32, device=E.DEVICE)
        p = E.forward(model, cfg, x, cyc)[:, -PRED_LEN:, :]
        out.append(((p - y) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def budget_k(n: int) -> int:
    return int(round(REMOVAL_BUDGET * n))


def remove_lowest(score: np.ndarray, k: int) -> np.ndarray:
    """Boolean keep-mask removing exactly the k lowest scores."""
    order = np.argsort(score, kind="stable")
    keep = np.ones(len(score), dtype=bool)
    keep[order[:k]] = False
    return keep


def remove_highest(score: np.ndarray, k: int) -> np.ndarray:
    return remove_lowest(-score, k)


def sel_no_filter(n, **kw):
    return np.ones(n, dtype=bool)


def sel_random(n, seed, **kw):
    rng = np.random.RandomState(seed)
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(n, size=budget_k(n), replace=False)] = False
    return keep


def sel_high_loss(loss, **kw):
    """Remove the top 20% by training loss under the clean initial checkpoint."""
    return remove_highest(loss, budget_k(len(loss)))


def sel_rho(loss_current, loss_reference, **kw):
    """RHO-LOSS reducible loss: current model loss minus irreducible-loss model.

    RHO-LOSS selects the TOP reducible points to train on. Under an equal
    budget, selecting the top 80% is identical to removing the bottom 20%, so
    the removal form used here is equivalent to the original selection rule.
    """
    rho = loss_current - loss_reference
    return remove_lowest(rho, budget_k(len(rho)))


def coherence_scores(X, Y, iqr) -> np.ndarray:
    """Fraction of channels whose input tail and target move the same way.

    A = median(input[-48:]) - median(input[-96:-48])
    B = median(target[:48]) - median(input[-48:])
    A channel is coherent when both are at least one train IQR and share a sign.
    """
    a = np.median(X[:, -48:, :], axis=1) - np.median(X[:, -96:-48, :], axis=1)
    b = np.median(Y[:, :48, :], axis=1) - np.median(X[:, -48:, :], axis=1)
    a = a / (iqr[None, :] + 1e-8)
    b = b / (iqr[None, :] + 1e-8)
    coh = (np.abs(a) >= COHERENCE_A_MIN) & (np.abs(b) >= COHERENCE_B_MIN) & (np.sign(a) == np.sign(b))
    return coh.mean(axis=1)


def sel_coherence_aware(loss_current, loss_reference, coh, **kw):
    """RHO ranking, but coherent-shift candidates are preserved.

    Windows the RHO ranking would drop are rescued when at least 30% of their
    channels look like a coherent shift; the budget is held at exactly 20% by
    dropping the next-lowest non-coherent RHO windows instead.
    """
    rho = loss_current - loss_reference
    n = len(rho)
    k = budget_k(n)
    order = list(np.argsort(rho, kind="stable"))
    candidate = set(order[:k])
    coherent = coh >= COHERENCE_CHANNEL_FRACTION
    drop = [i for i in order[:k] if not coherent[i]]
    for i in order[k:]:
        if len(drop) >= k:
            break
        if not coherent[i]:
            drop.append(i)
    keep = np.ones(n, dtype=bool)
    keep[np.array(drop[:k], dtype=int)] = False
    return keep


def adarho_scores(model_target, model_ref, cfg, X, Y, seed, n_B=256, n_b=128, n_r=64,
                  lr_target=None, lr_ref_div=10.0, epochs=1):
    """AdaRho (FAF, AISTATS 2026) Algorithm 1, implemented from the paper.

    Repeatedly draw a candidate batch of n_B windows, keep the n_b with the
    highest reducible loss for the target update, and refresh the reference
    model on n_r held-out candidates at a tenth of the target learning rate.
    The per-window selection frequency is returned as the score, so the
    equal-budget removal rule can be applied on top of it.
    """
    import torch.nn as nn
    rng = np.random.RandomState(seed)
    n = len(X)
    seen = np.zeros(n, dtype=np.int64)
    chosen = np.zeros(n, dtype=np.int64)
    opt_t = torch.optim.Adam(model_target.parameters(), lr=lr_target or cfg.learning_rate)
    opt_r = torch.optim.Adam(model_ref.parameters(), lr=(lr_target or cfg.learning_rate) / lr_ref_div)
    crit = nn.MSELoss()
    steps = int(epochs * n / n_B)
    for _ in range(max(1, steps)):
        idx = rng.choice(n, size=min(n_B, n), replace=False)
        x = torch.from_numpy(X[idx]).float().to(E.DEVICE)
        y = torch.from_numpy(Y[idx]).float().to(E.DEVICE)
        cyc = torch.zeros(len(x), dtype=torch.int32, device=E.DEVICE)
        with torch.no_grad():
            lt = ((E.forward(model_target, cfg, x, cyc)[:, -PRED_LEN:, :] - y) ** 2).mean(dim=(1, 2))
            lr_ = ((E.forward(model_ref, cfg, x, cyc)[:, -PRED_LEN:, :] - y) ** 2).mean(dim=(1, 2))
        red = (lt - lr_).cpu().numpy()
        seen[idx] += 1
        take = np.argsort(-red)[:n_b]
        chosen[idx[take]] += 1
        opt_t.zero_grad()
        loss = crit(E.forward(model_target, cfg, x[take], cyc[:len(take)])[:, -PRED_LEN:, :], y[take])
        loss.backward()
        opt_t.step()
        rest = np.setdiff1d(np.arange(len(idx)), take)[:n_r]
        if len(rest):
            opt_r.zero_grad()
            lr_loss = crit(E.forward(model_ref, cfg, x[rest], cyc[:len(rest)])[:, -PRED_LEN:, :], y[rest])
            lr_loss.backward()
            opt_r.step()
    return chosen / np.maximum(seen, 1)


def sel_adarho(adarho_freq, **kw):
    """Remove the 20% least often selected by AdaRho."""
    return remove_lowest(adarho_freq, budget_k(len(adarho_freq)))
