"""Shared paths for TS-IDEA-TOURNAMENT-v1. Single source of truth."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "ts_idea_tournament_v1"
RESULTS = ROOT / "results" / "ts_idea_tournament_v1"
RUNS = ROOT / "runs" / "ts_idea_tournament_v1"
VENDOR = ROOT / "runs" / "vendor"
DATASET_DIR = ROOT / "runs" / "dataset"

TQNET = VENDOR / "TQNet"


def add_vendor_to_path() -> None:
    """Make the official TQNet package importable without modifying it."""
    p = str(TQNET)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_dirs() -> None:
    for d in (RESULTS, RUNS):
        d.mkdir(parents=True, exist_ok=True)


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
