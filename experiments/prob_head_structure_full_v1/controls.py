"""Deterministic negative controls and the frozen signal-identification rule.

Every control is keyed by the frozen seed plus its declared scope, so a control result
is reproducible from its scope alone. Controls never touch prediction keys or outer
labels: fit controls disturb training rows only, score controls replace evaluation
scores without refitting.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Mapping, Sequence

import numpy as np

CONTROL_SEED = 2026090551
RECOVERY_FAIL_AT = 0.5
LABEL_SYMMETRY_TOLERANCE = 1e-12

FIFTY_PERCENT_RULE_CONTROLS = (
    "A teacher identity shuffle",
    "A teacher quantile shuffle",
    "B regret label shuffle",
    "B temporal feature row shuffle",
    "C time shuffle",
    "C scale-only feature",
    "C random sensor score",
    "C no-change synthetic sequence",
)

DIAGNOSTIC_ONLY_CONTROLS = (
    "A single-teacher soft target",
    "B missing indicator removal diagnostic",
    "C teacher name permutation",
)


class IdentificationFailure(ValueError):
    """A control comparison cannot identify a signal because the real effect is nonpositive."""


def _scope_seed(scope: Sequence[Any]) -> int:
    payload = "|".join(str(item) for item in ("prob_head_structure_full_v1", CONTROL_SEED, *scope))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def control_rng(*scope: Any) -> np.random.Generator:
    """A generator bound to the frozen seed and this control's exact scope."""
    return np.random.default_rng(_scope_seed(scope))


def _row_permutation(rows: int, scope: Sequence[Any]) -> np.ndarray:
    return control_rng(*scope, "rows", rows).permutation(rows)


def teacher_identity_shuffle(
    *,
    p_zero: np.ndarray,
    quantiles: np.ndarray,
    predictive_mean: np.ndarray,
    scope: Sequence[Any],
) -> dict[str, np.ndarray]:
    """Permute the three complete teacher identities jointly on each training row."""
    zero = np.asarray(p_zero, dtype=np.float64)
    values = np.asarray(quantiles, dtype=np.float64)
    mean = np.asarray(predictive_mean, dtype=np.float64)
    if zero.ndim != 2 or mean.shape != zero.shape or values.shape[:2] != zero.shape:
        raise ValueError("teacher channels must share their row-by-head shape")
    generator = control_rng(*scope, "teacher_identity")
    heads = zero.shape[1]
    out_zero, out_mean = np.empty_like(zero), np.empty_like(mean)
    out_values = np.empty_like(values)
    for row in range(zero.shape[0]):
        order = generator.permutation(heads)
        out_zero[row] = zero[row, order]
        out_mean[row] = mean[row, order]
        out_values[row] = values[row, order]
    return {"p_zero": out_zero, "quantiles": out_values, "predictive_mean": out_mean}


def teacher_quantile_shuffle(
    *, quantiles: np.ndarray, p_zero: np.ndarray, scope: Sequence[Any]
) -> dict[str, np.ndarray]:
    """Permute whole monotone quantile vectors across rows within each head; p0 untouched."""
    values = np.asarray(quantiles, dtype=np.float64)
    zero = np.asarray(p_zero, dtype=np.float64)
    if values.ndim != 3 or values.shape[:2] != zero.shape:
        raise ValueError("quantiles must be [row,head,q] and agree with p_zero")
    shuffled = np.empty_like(values)
    for head in range(values.shape[1]):
        order = _row_permutation(values.shape[0], (*scope, "teacher_quantile", head))
        shuffled[:, head, :] = values[order, head, :]
    return {"quantiles": shuffled, "p_zero": zero.copy()}


def regret_label_shuffle(regret: np.ndarray, *, scope: Sequence[Any]) -> np.ndarray:
    """Permute the complete three-regret vector across training rows."""
    values = np.asarray(regret, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("regret must be a row-by-head matrix")
    return values[_row_permutation(values.shape[0], (*scope, "regret_label"))]


def feature_row_shuffle(features: np.ndarray, *, scope: Sequence[Any]) -> np.ndarray:
    """Permute the complete extended-feature plus indicator row across training rows."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be a row-by-column matrix")
    return values[_row_permutation(values.shape[0], (*scope, "feature_row"))]


def time_shuffle(
    rows: np.ndarray, *, series_ids: Sequence[Any], scope: Sequence[Any]
) -> np.ndarray:
    """Permute complete disagreement rows across origins inside one series only."""
    values = np.asarray(rows, dtype=np.float64)
    identifiers = np.asarray(series_ids)
    if values.ndim != 2 or identifiers.shape[0] != values.shape[0]:
        raise ValueError("time shuffle needs one series id per disagreement row")
    shuffled = values.copy()
    for series in sorted({str(item) for item in identifiers}):
        positions = np.flatnonzero(identifiers.astype(str) == series)
        order = _row_permutation(positions.size, (*scope, "time_shuffle", series))
        shuffled[positions] = values[positions][order]
    return shuffled


def teacher_name_permutation(
    *,
    p_zero: np.ndarray,
    quantiles: np.ndarray,
    predictive_mean: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    """Label-symmetry integrity control: renaming teachers must not move any component."""
    from .sensor import disagreement_components

    baseline = disagreement_components(
        p_zero=p_zero, quantiles=quantiles, predictive_mean=predictive_mean, scale=scale
    )
    heads = np.asarray(p_zero).shape[0]
    permutation = list(range(1, heads)) + [0]
    permuted = disagreement_components(
        p_zero=np.asarray(p_zero)[permutation],
        quantiles=np.asarray(quantiles)[permutation],
        predictive_mean=np.asarray(predictive_mean)[permutation],
        scale=scale,
    )
    difference = max(abs(baseline[name] - permuted[name]) for name in baseline)
    return {
        "permutation": permutation,
        "max_absolute_difference": float(difference),
        "invariant": bool(difference <= LABEL_SYMMETRY_TOLERANCE),
        "tolerance": LABEL_SYMMETRY_TOLERANCE,
    }


def random_sensor_scores(row_keys: Sequence[Sequence[Any]]) -> np.ndarray:
    """Deterministic U[0,1) drawn once per frozen row key, independent of row order."""
    scores = np.empty(len(row_keys), dtype=np.float64)
    for position, key in enumerate(row_keys):
        payload = "|".join(str(item) for item in ("random_sensor_score", CONTROL_SEED, *key))
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        scores[position] = int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)
    return scores


def scale_only_features(train_scale: Sequence[float] | np.ndarray) -> np.ndarray:
    """The C scale-only control keeps the train RMS column and nothing else."""
    values = np.asarray(train_scale, dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("train scale must be finite and positive")
    return values


def recovery_ratio(*, real_effect: float, control_effect: float) -> dict[str, Any]:
    """Frozen identification rule: a control recovering at least half the real effect fails."""
    real = float(real_effect)
    if not np.isfinite(real) or real <= 0.0:
        raise IdentificationFailure(
            "RETRIEVAL_SIGNAL_NOT_IDENTIFIED: a nonpositive real effect cannot be identified"
        )
    recovery = float(control_effect) / real
    return {
        "real_effect": real,
        "control_effect": float(control_effect),
        "recovery": recovery,
        "fail_at": RECOVERY_FAIL_AT,
        "passed": bool(recovery < RECOVERY_FAIL_AT),
    }


__all__ = [
    "CONTROL_SEED",
    "DIAGNOSTIC_ONLY_CONTROLS",
    "FIFTY_PERCENT_RULE_CONTROLS",
    "IdentificationFailure",
    "LABEL_SYMMETRY_TOLERANCE",
    "RECOVERY_FAIL_AT",
    "control_rng",
    "feature_row_shuffle",
    "random_sensor_scores",
    "recovery_ratio",
    "regret_label_shuffle",
    "scale_only_features",
    "teacher_identity_shuffle",
    "teacher_name_permutation",
    "teacher_quantile_shuffle",
    "time_shuffle",
]
