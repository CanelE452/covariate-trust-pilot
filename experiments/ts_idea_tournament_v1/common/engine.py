"""Training / evaluation engine faithful to TQNet exp/exp_main.py.

Adam + MSE, lradj=type3, EarlyStopping on validation loss, checkpoint chosen by
validation MSE only. Metrics are computed in the scaled space, exactly as the
official test() does (its inverse_transform lines are commented out upstream).
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

import paths
from contract import ModelConfig, SEQ_LEN, PRED_LEN
import data as D

paths.add_vendor_to_path()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEEDS_CYCLE = {"TQNet"}


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.model == "TQNet":
        from models.TQNet import Model
    elif cfg.model == "DLinear":
        from models.DLinear import Model
    else:
        raise ValueError(cfg.model)

    class _NS:
        pass
    ns = _NS()
    for k, v in cfg.as_dict().items():
        setattr(ns, k, v)
    return Model(ns)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def forward(model: nn.Module, cfg: ModelConfig, x, cyc):
    if cfg.model in NEEDS_CYCLE:
        return model(x, cyc)
    return model(x)


def adjust_lr(optim, epoch: int, cfg: ModelConfig) -> float:
    # lradj == 'type3', the run.py default
    lr = cfg.learning_rate if epoch < 3 else cfg.learning_rate * (0.8 ** (epoch - 3))
    for g in optim.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, cfg, loader, per_channel: bool = False):
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    n = 0
    C = cfg.enc_in
    se_c = np.zeros(C, dtype=np.float64)
    cnt_c = 0
    for batch in loader:
        bx, by = batch[0].float().to(DEVICE), batch[1].float().to(DEVICE)
        cyc = batch[4].int().to(DEVICE)
        out = forward(model, cfg, bx, cyc)[:, -PRED_LEN:, :]
        tgt = by[:, -PRED_LEN:, :]
        d = out - tgt
        se_sum += float((d ** 2).sum())
        ae_sum += float(d.abs().sum())
        n += d.numel()
        if per_channel:
            se_c += (d ** 2).sum(dim=(0, 1)).double().cpu().numpy()
            cnt_c += d.shape[0] * d.shape[1]
    res = {"mse": se_sum / n, "mae": ae_sum / n, "n_elem": n}
    if per_channel:
        res["mse_per_channel"] = (se_c / cnt_c).tolist()
    return res


@dataclass
class TrainResult:
    best_val_mse: float
    best_epoch: int
    epochs_run: int
    history: list
    wall_s: float
    checkpoints: dict  # tag -> state_dict (cpu)


def train_model(cfg: ModelConfig, dataset: str,
                max_train_windows: Optional[int] = None,
                max_eval_windows: Optional[int] = None,
                epochs: Optional[int] = None,
                num_workers: int = 0,
                batch_transform: Optional[Callable] = None,
                grad_rule: Optional[Callable] = None,
                capture_all_epochs: bool = False,
                log: Optional[Callable] = None) -> TrainResult:
    """Train one model.

    batch_transform(bx, by, cyc, rng) -> (bx, by) lets a track modify inputs
    (e.g. channel dropout). grad_rule(model, bx, by, cyc, criterion, step) -> loss
    replaces the plain backward pass (e.g. PCGrad). capture_fracs snapshots the
    shared weights at fixed fractions of the training schedule.
    """
    set_seed(cfg.random_seed)
    model = build_model(cfg).to(DEVICE)
    n_epochs = epochs if epochs is not None else cfg.train_epochs

    tr = D.get_dataset(dataset, "train")
    va = D.get_dataset(dataset, "val")
    if max_train_windows is not None:
        tr = torch.utils.data.Subset(tr, list(range(min(max_train_windows, len(tr)))))
    if max_eval_windows is not None:
        va = torch.utils.data.Subset(va, list(range(min(max_eval_windows, len(va)))))
    tl = torch.utils.data.DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                                     num_workers=num_workers, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    crit = nn.MSELoss()
    best = math.inf
    best_state = None
    best_epoch = -1
    bad = 0
    hist = []
    caps = {}
    rng = np.random.RandomState(cfg.random_seed)
    t0 = time.time()
    step = 0

    for ep in range(1, n_epochs + 1):
        model.train()
        losses = []
        for batch in tl:
            bx, by = batch[0].float().to(DEVICE), batch[1].float().to(DEVICE)
            cyc = batch[4].int().to(DEVICE)
            if batch_transform is not None:
                bx, by = batch_transform(bx, by, cyc, rng)
            optim.zero_grad()
            if grad_rule is not None:
                loss = grad_rule(model, cfg, bx, by, cyc, crit, step)
            else:
                out = forward(model, cfg, bx, cyc)[:, -PRED_LEN:, :]
                loss = crit(out, by[:, -PRED_LEN:, :])
                loss.backward()
            optim.step()
            losses.append(float(loss))
            step += 1
        v = evaluate(model, cfg, vl)["mse"]
        lr = adjust_lr(optim, ep + 1, cfg)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_mse": v, "next_lr": lr})
        if log:
            log(f"  ep{ep:02d} train {np.mean(losses):.5f} val {v:.5f}")
        if capture_all_epochs:
            caps[f"epoch{ep:03d}"] = {k: t.detach().cpu().clone()
                                      for k, t in model.state_dict().items()}
        if v < best:
            best, best_epoch, bad = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    caps["best"] = best_state
    return TrainResult(best_val_mse=best, best_epoch=best_epoch, epochs_run=ep,
                       history=hist, wall_s=time.time() - t0, checkpoints=caps)


def test_metrics(cfg: ModelConfig, dataset: str, state_dict, per_channel=True,
                 max_eval_windows: Optional[int] = None, num_workers: int = 0):
    model = build_model(cfg).to(DEVICE)
    model.load_state_dict(state_dict)
    te = D.get_dataset(dataset, "test")
    if max_eval_windows is not None:
        te = torch.utils.data.Subset(te, list(range(min(max_eval_windows, len(te)))))
    tl = torch.utils.data.DataLoader(te, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=num_workers)
    return evaluate(model, cfg, tl, per_channel=per_channel)
