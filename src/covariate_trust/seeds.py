"""Stable hierarchical seeds.

Python's built-in ``hash()`` is randomized per process (PYTHONHASHSEED) and must
never be used to derive seeds here.  Everything is derived from SHA-256 of a
canonical string, so seeds are reproducible across processes and machines.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

_MASK64 = (1 << 64) - 1


def stable_seed(*parts: Any) -> int:
    """Deterministic 64-bit seed derived from ``parts`` via SHA-256."""
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & _MASK64


def make_rng(*parts: Any) -> np.random.Generator:
    """NumPy Generator seeded by :func:`stable_seed` of ``parts``."""
    return np.random.default_rng(np.random.SeedSequence(stable_seed(*parts)))


def stable_hash(*parts: Any, length: int = 16) -> str:
    """Short hex digest used for task keys and content hashes."""
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def array_hash(arr: np.ndarray, length: int = 16) -> str:
    """Content hash of a numeric array (bit-exact, shape and dtype aware)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()[:length]


def seed_hierarchy(master_seed: int, base_series_ids: list[int], horizons: list[int],
                   origins: list[int]) -> dict:
    """Full, explicit record of every seed used by the pilot.

    Recorded in the manifest so a run can be reproduced without re-reading code.
    """
    hierarchy: dict[str, Any] = {
        "master_seed": master_seed,
        "scheme": "sha256(repr-joined parts) -> first 8 bytes -> uint64",
        "namespaces": {
            "series_params": "(master, 'series_params', base_series_id)",
            "innovations_base": "(master, 'innov_base', base_series_id)",
            "innovations_covariate": "(master, 'innov_cov', base_series_id)",
            "eta_path": "(master, 'eta', base_series_id, origin, horizon)",
        },
        "series": {},
    }
    for b in base_series_ids:
        entry = {
            "series_params": stable_seed(master_seed, "series_params", b),
            "innov_base": stable_seed(master_seed, "innov_base", b),
            "innov_cov": stable_seed(master_seed, "innov_cov", b),
            "eta": {
                f"origin{o}_h{h}": stable_seed(master_seed, "eta", b, o, h)
                for o in origins
                for h in horizons
            },
        }
        hierarchy["series"][str(b)] = entry
    return hierarchy
