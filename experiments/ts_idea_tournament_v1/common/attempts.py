"""Append-only attempt directories with a terminal completion.json."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import paths


def stage_dir(stage: str) -> Path:
    d = paths.RUNS / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def completed(stage: str) -> Path | None:
    """Return the newest attempt dir that carries completion.json, else None."""
    d = stage_dir(stage)
    done = sorted(p for p in d.glob("attempt_*") if (p / "completion.json").exists())
    return done[-1] if done else None


def new_attempt(stage: str) -> Path:
    d = stage_dir(stage)
    existing = sorted(p.name for p in d.glob("attempt_*"))
    n = int(existing[-1].split("_")[1]) + 1 if existing else 1
    a = d / f"attempt_{n:04d}"
    a.mkdir(parents=True, exist_ok=False)
    (a / "started.json").write_text(json.dumps(
        {"stage": stage, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "pid": os.getpid()}, indent=1))
    return a


def finish(attempt: Path, payload: dict) -> None:
    """completion.json is written LAST, after every other artifact is flushed."""
    for f in attempt.glob("*.json"):
        pass
    (attempt / "completion.json").write_text(json.dumps(
        {"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload},
        indent=1, default=str))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def environment() -> dict:
    import numpy, pandas, torch
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        if torch.cuda.is_available() else None,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
