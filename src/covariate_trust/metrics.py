"""Forecast accuracy metrics.

Primary metric is WQL (weighted quantile loss).  Quantiles are aggregated
through the pinball loss; they are never averaged as plain numbers.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-8


def pinball(y: np.ndarray, qhat: np.ndarray, q: float) -> np.ndarray:
    """q * max(y - qhat, 0) + (1 - q) * max(qhat - y, 0)."""
    diff = y - qhat
    return q * np.maximum(diff, 0.0) + (1.0 - q) * np.maximum(-diff, 0.0)


def wql(y: np.ndarray, q_pred: np.ndarray, quantile_levels: list[float]) -> float:
    """WQL = 2 * sum(pinball) / (n_quantiles * (sum|y| + eps)).

    ``q_pred`` has shape (horizon, n_quantiles) aligned with ``quantile_levels``.
    """
    y = np.asarray(y, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    if q_pred.shape != (len(y), len(quantile_levels)):
        raise ValueError(f"q_pred shape {q_pred.shape} != {(len(y), len(quantile_levels))}")
    total = 0.0
    for j, q in enumerate(quantile_levels):
        total += float(pinball(y, q_pred[:, j], q).sum())
    denom = len(quantile_levels) * (float(np.abs(y).sum()) + EPSILON)
    return 2.0 * total / denom


def median_index(quantile_levels: list[float]) -> int:
    arr = np.asarray(quantile_levels, dtype=float)
    idx = int(np.argmin(np.abs(arr - 0.5)))
    if abs(arr[idx] - 0.5) > 1e-9:
        raise ValueError("quantile_levels must contain 0.5 for the median-based metrics")
    return idx


def nmae(y: np.ndarray, q_pred: np.ndarray, quantile_levels: list[float]) -> float:
    """Normalized MAE of the median forecast."""
    med = q_pred[:, median_index(quantile_levels)]
    return float(np.abs(y - med).sum() / (np.abs(y).sum() + EPSILON))


def mse(y: np.ndarray, q_pred: np.ndarray, quantile_levels: list[float]) -> float:
    """MSE of the median forecast."""
    med = q_pred[:, median_index(quantile_levels)]
    return float(np.mean((y - med) ** 2))


def quantile_crossing_rate(q_pred: np.ndarray) -> float:
    """Fraction of adjacent quantile pairs that are out of order."""
    q_pred = np.asarray(q_pred, dtype=float)
    if q_pred.shape[1] < 2:
        return 0.0
    bad = (np.diff(q_pred, axis=1) < 0).sum()
    return float(bad) / float(q_pred.shape[0] * (q_pred.shape[1] - 1))


def relative_delta(wql_baseline: float, wql_treatment: float) -> float:
    """(WQL_treatment - WQL_baseline) / WQL_baseline.  Negative = treatment better."""
    return (wql_treatment - wql_baseline) / (wql_baseline + EPSILON)


def is_harm(wql_baseline: float, wql_treatment: float, threshold: float) -> bool:
    """Harm = treatment worse than baseline by more than ``threshold`` in relative terms."""
    return bool(wql_treatment > (1.0 + threshold) * wql_baseline)


def future_value(wql_m1: float, wql_m3: float) -> float:
    """V_future = WQL(M1) - WQL(M3).  Positive = future covariate helped."""
    return wql_m1 - wql_m3
