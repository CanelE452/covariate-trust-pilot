"""Frozen forecasting contract: datasets, official split, official TQNet/DLinear configs.

Every hyperparameter here is read off the official TQNet scripts
(runs/vendor/TQNet/scripts/TQNet/{ettm1,weather}.sh) and run.py defaults.
No new hyperparameter search is performed anywhere in this study.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List

SEQ_LEN = 96
PRED_LEN = 96
LABEL_LEN = 0
FEATURES = "M"

CLEAN_SEEDS = [2026090601, 2026090602]
BOOTSTRAP_SEED = 2026090699


@dataclass
class DatasetSpec:
    name: str
    data_path: str
    data_name: str            # TQNet data_factory key
    enc_in: int
    cycle: int
    tqnet_batch_size: int
    tqnet_lr: float
    freq: str
    sha256: str


DATASETS: Dict[str, DatasetSpec] = {
    "ETTm1": DatasetSpec(
        name="ETTm1", data_path="ETTm1.csv", data_name="ETTm1",
        enc_in=7, cycle=96, tqnet_batch_size=256, tqnet_lr=0.001, freq="t",
        sha256="6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
    ),
    "Weather": DatasetSpec(
        name="Weather", data_path="weather.csv", data_name="custom",
        enc_in=21, cycle=144, tqnet_batch_size=64, tqnet_lr=0.001, freq="t",
        sha256="34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63",
    ),
}


@dataclass
class ModelConfig:
    """Mirrors the argparse namespace the official model classes consume."""
    model: str
    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    label_len: int = LABEL_LEN
    enc_in: int = 7
    cycle: int = 96
    d_model: int = 512            # run.py default
    dropout: float = 0.5          # ettm1.sh / weather.sh
    model_type: str = "mlp"       # run.py default
    use_revin: int = 1            # run.py default
    individual: int = 0           # run.py default
    features: str = FEATURES
    learning_rate: float = 0.001
    batch_size: int = 256
    train_epochs: int = 30
    patience: int = 5
    lradj: str = "type3"          # run.py default
    random_seed: int = 2026090601

    def as_dict(self) -> dict:
        return asdict(self)


def tqnet_config(dataset: str, seed: int) -> ModelConfig:
    d = DATASETS[dataset]
    return ModelConfig(
        model="TQNet", enc_in=d.enc_in, cycle=d.cycle,
        learning_rate=d.tqnet_lr, batch_size=d.tqnet_batch_size,
        dropout=0.5, random_seed=seed,
    )


def dlinear_config(dataset: str, seed: int) -> ModelConfig:
    """DLinear channel-independent control.

    The TQNet ablation script only ships DLinear configs for electricity/PEMS.
    We keep DLinear's own official knobs (individual=1 for channel independence)
    and reuse the same dataset-level optimisation budget as the shared model so
    that the two clean models differ only in cross-channel structure.
    """
    d = DATASETS[dataset]
    return ModelConfig(
        model="DLinear", enc_in=d.enc_in, cycle=d.cycle,
        learning_rate=d.tqnet_lr, batch_size=d.tqnet_batch_size,
        dropout=0.5, individual=1, random_seed=seed,
    )
