"""Adapter for training canonical models on a caller-supplied split."""

from __future__ import annotations

import threading
from unittest import mock

import torch

from experiments.om_factorization_killtest import train as km_train
from experiments.unified_temporal_27_v3.config import ExperimentConfig


_TRAIN_ONE_LOCK = threading.Lock()


def train_one_on_split(
    model_name: str,
    split: km_train.Split,
    cfg: ExperimentConfig,
    model_seed: int,
    device: torch.device,
) -> dict:
    """Return the canonical ``train_one`` result unchanged.

    The supplied split is installed by temporarily replacing the canonical
    module's global ``build_splits`` binding. Calls through this adapter are
    serialized, but other canonical training must still run sequentially or
    in an isolated process because that global substitution is not thread-safe.
    """
    if not isinstance(split, km_train.Split):
        raise TypeError("split must be an instance of km_train.Split")

    with _TRAIN_ONE_LOCK:
        with mock.patch.object(
            km_train, "build_splits", autospec=True, return_value=split
        ):
            return km_train.train_one(model_name, {}, cfg, model_seed, device)
