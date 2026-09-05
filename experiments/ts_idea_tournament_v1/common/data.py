"""Dataset access through the OFFICIAL TQNet data provider (read-only import).

We never re-derive split indices: Dataset_ETT_minute / Dataset_Custom from
runs/vendor/TQNet/data_provider/data_loader.py own that contract, including the
train-only StandardScaler fit.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

import paths  # noqa: E402
from contract import DATASETS, SEQ_LEN, PRED_LEN, LABEL_LEN, FEATURES

paths.add_vendor_to_path()

from data_provider.data_loader import Dataset_ETT_minute, Dataset_Custom  # noqa: E402

_LOADER = {"ETTm1": Dataset_ETT_minute, "custom": Dataset_Custom}
_CACHE: dict = {}


def get_dataset(dataset: str, flag: str):
    """flag in {train, val, test}. Cached because the CSV parse dominates."""
    key = (dataset, flag)
    if key in _CACHE:
        return _CACHE[key]
    spec = DATASETS[dataset]
    cls = _LOADER[spec.data_name]
    ds = cls(
        root_path=str(paths.DATASET_DIR),
        data_path=spec.data_path,
        flag=flag,
        size=[SEQ_LEN, LABEL_LEN, PRED_LEN],
        features=FEATURES,
        target="OT",
        scale=True,
        timeenc=0,
        freq=spec.freq,
        cycle=spec.cycle,
    )
    _CACHE[key] = ds
    return ds


def split_borders(dataset: str) -> dict:
    """Resolve the actual split index ranges the official loader used."""
    out = {}
    for flag in ("train", "val", "test"):
        ds = get_dataset(dataset, flag)
        out[flag] = {
            "n_rows": int(len(ds.data_x)),
            "n_windows": int(len(ds)),
            "first_stamp": str(ds.data_stamp[0].tolist()) if hasattr(ds.data_stamp, "tolist") else None,
        }
    return out


def make_loader(dataset: str, flag: str, batch_size: int, shuffle: bool,
                num_workers: int = 0, drop_last: bool = False) -> DataLoader:
    ds = get_dataset(dataset, flag)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last)


def as_arrays(dataset: str, flag: str):
    """Return (data_x [T,C] scaled, cycle_index [T], scaler) for window surgery."""
    ds = get_dataset(dataset, flag)
    return np.asarray(ds.data_x, dtype=np.float32), np.asarray(ds.cycle_index), ds.scaler


def window_batch(dataset: str, flag: str, starts, device="cpu"):
    """Build (x, y, cycle_index) for explicit window start indices."""
    ds = get_dataset(dataset, flag)
    dx = np.asarray(ds.data_x, dtype=np.float32)
    dy = np.asarray(ds.data_y, dtype=np.float32)
    xs, ys, cs = [], [], []
    for s in starts:
        e = s + SEQ_LEN
        xs.append(dx[s:e])
        ys.append(dy[e:e + PRED_LEN])
        cs.append(ds.cycle_index[e])
    x = torch.from_numpy(np.stack(xs)).to(device)
    y = torch.from_numpy(np.stack(ys)).to(device)
    c = torch.as_tensor(np.stack(cs)).int().to(device)
    return x, y, c
