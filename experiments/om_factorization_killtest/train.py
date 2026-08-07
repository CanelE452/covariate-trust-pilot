"""One trainer shared by every DLinear variant.

``experiments/unified_temporal_27_v3/training.py`` is not reused as a whole:
it carries gap loss, lambda selection and an MSE gate that this kill test is
forbidden to touch.  Its windowing helpers ARE reused, so the input tensors,
lookback, horizon, split and train-only scale are byte-identical to the
existing pipeline.

Everything that could differ between models is fixed here: same windows, same
optimizer, same batch size, same epoch budget, same early-stopping rule, same
checkpoint criterion (validation MSE of the FINAL MEAN prediction), same data
and model seeds.  Only the head set and the loss change.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..unified_temporal_27_v3.config import ExperimentConfig
from ..unified_temporal_27_v3.training import make_windows, train_scale
from . import models, prereg


def experiment_config() -> ExperimentConfig:
    d = prereg.DATA
    return ExperimentConfig(
        length=d["length"], lookback=d["lookback"], horizon=d["horizon"],
        period=d["period"], train_end=d["train_end"], val_end=d["val_end"],
    )


def dense_origins(start: int, end: int, horizon: int, lookback: int) -> np.ndarray:
    """Every origin whose full horizon stays inside the split."""
    lo = max(start, lookback)
    return np.arange(lo, end - horizon + 1, dtype=np.int32)


def stride_origins(start: int, end: int, horizon: int, stride: int) -> np.ndarray:
    return np.arange(start, end - horizon + 1, stride, dtype=np.int32)


@dataclass
class Split:
    train: object
    validation: object
    test: object
    scale: np.ndarray
    positive_variance: float


def build_splits(block: dict[str, np.ndarray], cfg: ExperimentConfig) -> Split:
    data = {"y": block["y"], "z": block["z"]}
    scale = train_scale(data, cfg)
    train_o = dense_origins(0, cfg.train_end, cfg.horizon, cfg.lookback)
    val_o = dense_origins(cfg.train_end, cfg.val_end, cfg.horizon, cfg.lookback)
    test_o = stride_origins(cfg.val_end, cfg.length, cfg.horizon,
                            prereg.DATA["test_origin_stride"])

    train_w = make_windows(data, train_o, 0, cfg.train_end, cfg, scale)
    val_w = make_windows(data, val_o, cfg.train_end, cfg.val_end, cfg, scale)
    test_w = make_windows(data, test_o, cfg.val_end, cfg.length, cfg, scale)

    # Positive variance for the factorized loss: TRAIN observations only.
    y_train = block["y"][:, :cfg.train_end]
    z_train = block["z"][:, :cfg.train_end]
    positives = y_train[z_train > 0]
    positive_variance = float(positives.var(ddof=1)) if positives.size > 1 else 1.0
    return Split(train_w, val_w, test_w, scale, max(positive_variance, 1e-6))


def _loader(windows, batch_size: int, shuffle: bool, generator=None) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(windows.history), torch.from_numpy(windows.target),
        torch.from_numpy(windows.occurrence),
        torch.from_numpy(windows.target_mask.astype(np.float32)),
        torch.from_numpy(windows.scale),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, drop_last=False)


@torch.no_grad()
def _validation_mean_mse(model, loader, device) -> float:
    """The single checkpoint criterion for every DLinear variant."""
    model.eval()
    total, count = 0.0, 0.0
    for history, target, _occ, mask, scale in loader:
        history, target = history.to(device), target.to(device)
        mask, scale = mask.to(device), scale.to(device)
        mean = model(history, scale)["mean_prediction"]
        total += float((((mean - target) ** 2) * mask).sum())
        count += float(mask.sum())
    return total / max(count, 1.0)


@torch.no_grad()
def predict(model, windows, device) -> dict[str, np.ndarray]:
    model.eval()
    loader = _loader(windows, 1024, shuffle=False)
    keys = ("mean_prediction", "p_prediction", "mu_prediction",
            "nb_mean", "dispersion")
    chunks: dict[str, list] = {k: [] for k in keys}
    for history, _t, _o, _m, scale in loader:
        out = model(history.to(device), scale.to(device))
        for key in keys:
            if key not in out:
                continue
            value = out[key]
            if value.ndim == 0:
                value = value.expand(history.shape[0], model.horizon)
            chunks[key].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in chunks.items() if v}


def train_one(model_name: str, block: dict[str, np.ndarray], cfg: ExperimentConfig,
              model_seed: int, device: torch.device) -> dict:
    splits = build_splits(block, cfg)
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    model = models.BUILDERS[model_name](cfg.lookback, cfg.horizon).to(device)
    loss_fn = models.LOSSES[model_name]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=prereg.TRAINING["learning_rate"],
                                 weight_decay=prereg.TRAINING["weight_decay"])
    generator = torch.Generator().manual_seed(model_seed)
    train_loader = _loader(splits.train, prereg.TRAINING["batch_size"], True, generator)
    val_loader = _loader(splits.validation, 1024, False)
    positive_variance = torch.tensor(splits.positive_variance, device=device)

    best = float("inf")
    best_state, best_epoch, bad = None, -1, 0
    started = time.time()
    for epoch in range(prereg.TRAINING["max_epochs"]):
        model.train()
        for history, target, occurrence, mask, scale in train_loader:
            history, target = history.to(device), target.to(device)
            occurrence, mask = occurrence.to(device), mask.to(device)
            scale = scale.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(history, scale)
            loss = loss_fn(out, target, occurrence, mask, positive_variance)
            loss.backward()
            optimizer.step()
        current = _validation_mean_mse(model, val_loader, device)
        if current < best - 1e-9:
            best, best_epoch, bad = current, epoch, 0
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= prereg.TRAINING["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    seconds = time.time() - started

    return {
        "model": model,
        "splits": splits,
        "predictions": predict(model, splits.test, device),
        "best_epoch": best_epoch,
        "best_validation_mean_mse": best,
        "train_seconds": seconds,
        "n_parameters": models.count_parameters(model),
    }
