"""Pre-generate and cache the official TSRBench corrupted series.

CPU-bound (SPOT / EVT calibration), so the combinations are farmed out to
separate processes and cached on disk before any GPU evaluation starts.

Usage: python gen_corruption.py <dataset> <severity>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import corruption as X


def main():
    dataset, severity = sys.argv[1], int(sys.argv[2])
    t = time.time()
    off = X.official_corrupt(dataset, severity)
    checks = {fam: X.corruption_matches_official(dataset, severity, fam) for fam in X.FAMILIES}
    print(json.dumps({
        "dataset": dataset, "severity": severity,
        "wall_s": round(time.time() - t, 1),
        "shape": list(off["spike"].shape),
        "fingerprint": X.corruption_fingerprint(dataset, severity),
        "splice_checks": checks,
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()
