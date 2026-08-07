"""How many expanding-time validation blocks does each train region actually hold?

The factorial that selected P0L1 averaged over two validation folds per dataset,
which is thin enough that a single bad alpha fit can carry a dataset -- UCI's
first fold did exactly that.  This cuts the train region into as many blocks as
it can hold under the frozen lookback and horizon, and freezes the boundaries
before a single error is computed.

Nothing about the protocol moves to make the fold count come out nicer.  A
dataset that cannot supply three usable validation blocks is reported as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from ..external_validity_screen import cli, prereg, screen
from ..multi_benchmark import run as MB
from ..structure_gate.oof import ORIGINS_PER_FOLD

OUT = screen.OUT.parent / "gate_p0l1_robustness"
DATASETS = ("m5", "favorita", "freshretailnet", "uci")
#: Target number of expanding validation blocks. A dataset supplies fewer only
#: when its train region is physically too short; the protocol never shrinks.
TARGET_FOLDS = 5
MIN_FOLDS = 3


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def geometry(name: str) -> dict:
    """Frozen lookback, horizon, stride and the train boundary for one dataset."""
    if name in ("m5", "favorita"):
        cfg = screen.config_for(name)
        stride = prereg.SPLITS[name]["test_origin_stride"]
    else:
        data = MB.load_grid(name)
        cfg = MB.config_for(data)
        stride = MB.STRIDE
    return {"lookback": cfg.lookback, "horizon": cfg.horizon, "stride": stride,
            "train_end": cfg.train_end, "length": cfg.length}


def plan(name: str) -> dict:
    """Tile validation blocks backwards from train_end, as many as fit."""
    g = geometry(name)
    block = g["horizon"] * (1 + ORIGINS_PER_FOLD)      # validation horizon + strided origins
    earliest_cutoff = g["lookback"] + g["horizon"]      # a fold needs history to train on
    usable = (g["train_end"] - earliest_cutoff) // block
    n_folds = int(min(TARGET_FOLDS, max(usable, 0)))
    cutoffs = [g["train_end"] - block * (n_folds - k) for k in range(n_folds)]
    folds = []
    for k, cutoff in enumerate(cutoffs):
        folds.append({"fold": k, "train_start": 0, "train_end": int(cutoff),
                      "validation_start": int(cutoff + g["horizon"]),
                      "validation_end": int(cutoff + block),
                      "origins": [int(cutoff + g["horizon"] + g["stride"] * i)
                                  for i in range(ORIGINS_PER_FOLD)]})
    return {"geometry": g, "block_days": int(block), "max_usable_folds": int(usable),
            "n_folds": n_folds, "folds": folds,
            "n_validation_folds": max(n_folds - 1, 0),
            "warn_limited": bool(n_folds - 1 < MIN_FOLDS)}


def cmd_freeze(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"frozen_at_utc": _utc(), "target_folds": TARGET_FOLDS,
                "min_validation_folds": MIN_FOLDS,
                "rule": ("expanding time; a gate is fitted only on folds strictly earlier "
                         "than the one it is scored on, so fold 0 never validates"),
                "protocol_unchanged": True, "datasets": {}}
    for name in args.datasets:
        block = plan(name)
        manifest["datasets"][name] = block
        print(f"[{name}] train_end {block['geometry']['train_end']}  block {block['block_days']}d  "
              f"folds {block['n_folds']} (validation {block['n_validation_folds']})"
              f"{'  WARN_LIMITED_TEMPORAL_FOLDS' if block['warn_limited'] else ''}")
        for f in block["folds"]:
            print(f"    fold {f['fold']}: train [0,{f['train_end']})  "
                  f"validation [{f['validation_start']},{f['validation_end']})  "
                  f"origins {f['origins']}")
    payload = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    manifest["fold_boundary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    manifest["git_commit"] = cli._git_commit()
    (OUT / "expanded_fold_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"fold_boundary_sha256": manifest["fold_boundary_sha256"],
                      "validation_folds": {n: b["n_validation_folds"]
                                           for n, b in manifest["datasets"].items()}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("expanded temporal fold plan")
    sub = parser.add_subparsers(required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--datasets", nargs="*", default=list(DATASETS))
    f.set_defaults(func=cmd_freeze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
