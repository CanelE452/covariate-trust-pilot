"""Structure-conditioned teacher routing fitted only on strictly earlier inner origins.

Fold ``k`` sees origins ``1..k-1`` and nothing else: its temperature is chosen on
still-earlier out-of-fold origins, its imputer and scaler are fitted on its own fit rows,
and no heldout weight is ever recomputed once a later fold has been fitted.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .temporal_features import fold_median_imputer

TEMPERATURE_GRID = (0.25, 0.5, 1.0)
FOLD_TWO_TEMPERATURE = 0.25
INNER_ORIGIN_TARGET = 8
INNER_ORIGIN_MINIMUM = 4
LOGISTIC_MAX_ITERATIONS = 1000


class RoutingBranchBlocked(RuntimeError):
    """The B branch cannot run because its inner-origin geometry is insufficient."""


class InsufficientRoutingVariation(ValueError):
    """A routing statistic is undefined because a compared vector is constant."""


def head_regret(losses: np.ndarray) -> np.ndarray:
    """R_m = sCRPS_m - min_j sCRPS_j on every row."""
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("losses must be a two-dimensional row-by-head matrix")
    if not np.isfinite(values).all():
        raise ValueError("head losses must be finite")
    return values - values.min(axis=1, keepdims=True)


def soft_routing_target(regret: np.ndarray, *, temperature: float) -> np.ndarray:
    """z_m = softmax(-R_m / T) computed with the log-sum-exp shift."""
    values = np.asarray(regret, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("regret must be a two-dimensional row-by-head matrix")
    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scaled = -values / float(temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    total = weights.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0) or not np.isfinite(total).all():
        raise ValueError("the soft routing target underflowed to an empty distribution")
    return weights / total


def select_inner_origins(*, lookback: int, horizon: int, model_train_end: int) -> tuple[int, ...]:
    """Last eight non-overlapping valid origins, else eight evenly spaced unique origins."""
    lowest, highest = int(lookback), int(model_train_end) - int(horizon)
    if highest < lowest:
        raise RoutingBranchBlocked(
            "B_INNER_ORIGIN_GEOMETRY_BLOCKED: no valid inner origin exists"
        )
    stride = int(horizon)
    spaced = [highest - offset * stride for offset in range(INNER_ORIGIN_TARGET)]
    if min(spaced) >= lowest:
        return tuple(sorted(spaced))

    positions = np.linspace(lowest, highest, INNER_ORIGIN_TARGET)
    origins = sorted({int(round(float(position))) for position in positions})
    if len(origins) < INNER_ORIGIN_MINIMUM:
        raise RoutingBranchBlocked(
            "B_INNER_ORIGIN_GEOMETRY_BLOCKED: fewer than four unique inner origins"
        )
    return tuple(origins)


def select_temperature(
    *, fold_index: int, prior_origin_scores: Mapping[float, float] | None
) -> dict[str, Any]:
    """Fold two has no prior heldout origin, so it takes the fixed tie-first temperature."""
    if int(fold_index) <= 2 or not prior_origin_scores:
        return {"temperature": FOLD_TWO_TEMPERATURE, "rule": "k2_fixed_tie_first", "scores": {}}
    scores = {float(key): float(value) for key, value in prior_origin_scores.items()}
    if set(scores) != set(TEMPERATURE_GRID):
        raise ValueError("temperature selection requires one score per frozen grid temperature")
    best = min(scores.items(), key=lambda item: (item[1], item[0]))
    return {
        "temperature": best[0],
        "rule": "expanding_oof_expected_regret",
        "scores": scores,
    }


def _fit_soft_router(features: np.ndarray, targets: np.ndarray) -> LogisticRegression:
    """Fit multinomial logistic regression by class-row expansion with soft sample weights."""
    rows, heads = targets.shape
    expanded_features = np.repeat(features, heads, axis=0)
    expanded_classes = np.tile(np.arange(heads), rows)
    sample_weight = targets.reshape(-1)
    model = LogisticRegression(
        max_iter=LOGISTIC_MAX_ITERATIONS, multi_class="multinomial", solver="lbfgs"
    )
    model.fit(expanded_features, expanded_classes, sample_weight=sample_weight)
    return model


def _predict_weights(model: LogisticRegression, features: np.ndarray, heads: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    full = np.zeros((features.shape[0], heads), dtype=np.float64)
    for position, label in enumerate(model.classes_):
        full[:, int(label)] = probabilities[:, position]
    total = full.sum(axis=1, keepdims=True)
    return full / np.where(total > 0.0, total, 1.0)


def _expected_regret(weights: np.ndarray, regret: np.ndarray) -> float:
    return float(np.mean(np.sum(weights * regret, axis=1)))


def _stack(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(record[key], dtype=np.float64) for record in records], axis=0)


def expanding_crossfit_weights(
    panel: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    secondary: bool = False,
) -> dict[str, Any]:
    """Produce heldout routing weights for origins two onward without any backfill."""
    records = list(panel)
    if len(records) < 2:
        raise RoutingBranchBlocked(
            "B_INNER_ORIGIN_GEOMETRY_BLOCKED: expanding cross-fit needs at least two origins"
        )
    names = [str(name) for name in feature_names]
    heads = int(np.asarray(records[0]["losses"]).shape[1])
    folds: list[dict[str, Any]] = []

    for position in range(1, len(records)):
        fit_records = records[:position]
        held = records[position]
        fit_features = _stack(fit_records, "features")
        fit_regret = head_regret(_stack(fit_records, "losses"))
        imputer = fold_median_imputer(fit_features, names)
        fit_matrix = imputer["transform"](fit_features)
        held_matrix = imputer["transform"](np.asarray(held["features"], dtype=np.float64))

        prior_scores: dict[float, float] | None = None
        if position >= 2:
            prior_scores = {}
            for temperature in TEMPERATURE_GRID:
                out_of_fold: list[float] = []
                for inner in range(1, position):
                    inner_fit = records[:inner]
                    inner_features = _stack(inner_fit, "features")
                    inner_regret = head_regret(_stack(inner_fit, "losses"))
                    inner_imputer = fold_median_imputer(inner_features, names)
                    inner_model = _fit_soft_router(
                        inner_imputer["transform"](inner_features),
                        soft_routing_target(inner_regret, temperature=temperature),
                    )
                    evaluated = records[inner]
                    predicted = _predict_weights(
                        inner_model,
                        inner_imputer["transform"](np.asarray(evaluated["features"], dtype=np.float64)),
                        heads,
                    )
                    out_of_fold.append(
                        _expected_regret(predicted, head_regret(np.asarray(evaluated["losses"])))
                    )
                prior_scores[float(temperature)] = float(np.mean(out_of_fold))

        chosen = select_temperature(fold_index=position + 1, prior_origin_scores=prior_scores)
        targets = soft_routing_target(fit_regret, temperature=chosen["temperature"])
        model = _fit_soft_router(fit_matrix, targets)
        weights = _predict_weights(model, held_matrix, heads)

        fold: dict[str, Any] = {
            "origin": int(held["origin"]),
            "fit_origins": [int(record["origin"]) for record in fit_records],
            "temperature": float(chosen["temperature"]),
            "temperature_rule": chosen["rule"],
            "temperature_scores": chosen["scores"],
            "weights": weights,
            "regret": head_regret(np.asarray(held["losses"], dtype=np.float64)),
            "all_missing_features": imputer["all_missing_features"],
            "imputation_record": imputer["record"],
        }
        if secondary:
            fold["secondary_weights"] = _secondary_weights(fit_matrix, targets, held_matrix)
        folds.append(fold)

    return {
        "folds": folds,
        "feature_names": names,
        "heads": heads,
        "backfill": "forbidden",
    }


def _secondary_weights(
    fit_matrix: np.ndarray, targets: np.ndarray, held_matrix: np.ndarray
) -> np.ndarray:
    """HistGradientBoosting is reported beside the primary router and never promoted."""
    columns = []
    for head in range(targets.shape[1]):
        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(fit_matrix, targets[:, head])
        columns.append(np.clip(model.predict(held_matrix), 0.0, None))
    stacked = np.stack(columns, axis=1)
    total = stacked.sum(axis=1, keepdims=True)
    return np.where(total > 0.0, stacked / np.where(total > 0.0, total, 1.0), 1.0 / targets.shape[1])


def regret_spearman(weights: np.ndarray, regret: np.ndarray) -> float:
    """Spearman between flattened predicted weights and flattened negative true regret."""
    predicted = np.asarray(weights, dtype=np.float64).reshape(-1)
    truth = -np.asarray(regret, dtype=np.float64).reshape(-1)
    if predicted.shape != truth.shape:
        raise ValueError("weights and regret must share their row-by-head shape")
    if predicted.size < 2 or np.ptp(predicted) == 0.0 or np.ptp(truth) == 0.0:
        raise InsufficientRoutingVariation(
            "INSUFFICIENT_VARIATION: a constant vector cannot be rank correlated"
        )
    statistic = spearmanr(predicted, truth).statistic
    if not np.isfinite(statistic):
        raise InsufficientRoutingVariation("INSUFFICIENT_VARIATION: the rank correlation is undefined")
    return float(statistic)


__all__ = [
    "FOLD_TWO_TEMPERATURE",
    "INNER_ORIGIN_MINIMUM",
    "INNER_ORIGIN_TARGET",
    "TEMPERATURE_GRID",
    "InsufficientRoutingVariation",
    "RoutingBranchBlocked",
    "expanding_crossfit_weights",
    "head_regret",
    "regret_spearman",
    "select_inner_origins",
    "select_temperature",
    "soft_routing_target",
]
