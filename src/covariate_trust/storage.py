"""Run directories, atomic writes and resume bookkeeping.

Rules enforced here:
  * a run directory is never reused or overwritten;
  * every artifact is written to ``<name>.tmp`` first and renamed on success;
  * ``diagnostic`` is the only resumable command, keyed by task hash.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

RUN_SUBDIRS = ("generated", "predictions", "tables", "figures", "logs", "reports")


def project_root() -> Path:
    """Repository root (``src/covariate_trust/storage.py`` -> two levels up)."""
    return Path(__file__).resolve().parents[2]


def new_run_id(kind: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{kind}"


def create_run_dir(root: Path, kind: str, now: datetime | None = None) -> Path:
    """Create a fresh run directory.  Never overwrites an existing one."""
    runs = Path(root) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    base = new_run_id(kind, now)
    candidate = runs / base
    suffix = 1
    while candidate.exists():
        candidate = runs / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    for sub in RUN_SUBDIRS:
        (candidate / sub).mkdir()
    (candidate / "predictions" / "parts").mkdir()
    return candidate


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(Path(path), text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def atomic_write_csv(path: Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def atomic_savefig(path: Path, fig) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    # the ".tmp" suffix hides the real extension, so the format must be explicit
    fig.savefig(tmp, dpi=150, bbox_inches="tight", format=path.suffix.lstrip(".") or "png")
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------
# resume support (diagnostic only)
# ----------------------------------------------------------------------------

def parts_dir(run_dir: Path) -> Path:
    d = Path(run_dir) / "predictions" / "parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def completed_task_hashes(run_dir: Path) -> set[str]:
    return {p.stem for p in parts_dir(run_dir).glob("*.parquet")}


def write_task_part(run_dir: Path, task_hash: str, df: pd.DataFrame) -> Path:
    return atomic_write_parquet(parts_dir(run_dir) / f"{task_hash}.parquet", df)


def read_all_parts(run_dir: Path) -> pd.DataFrame:
    files = sorted(parts_dir(run_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
