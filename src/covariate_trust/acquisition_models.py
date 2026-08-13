"""Study 4 ex-ante value predictors.

Four candidates, all fitted on the training split only and selected on one
number: validation K=1 portfolio WQL.  The selected model is then frozen for
validation K=2, the retrospective test and the fresh confirmation.

The two-part model is kept honest: if either sign class is too small to fit, it
reports failure instead of silently degrading into one of the other candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CANDIDATES = ("ridge", "huber", "hist_gradient_boosting", "two_part_expected_gain")
MIN_CLASS_SAMPLES = 50
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
HGB_GRID = ({"max_depth": 3, "learning_rate": 0.05}, {"max_depth": 6, "learning_rate": 0.1})


class ModelFitError(RuntimeError):
    pass


def _preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [c for c in feature_columns if c == "zone"]
    numeric = [c for c in feature_columns if c != "zone"]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def _make_regressor(name: str, params: dict | None = None) -> Any:
    params = params or {}
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0), random_state=None)
    if name == "huber":
        return HuberRegressor(alpha=params.get("alpha", 1e-4), max_iter=500)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=200,
            random_state=0,
        )
    raise ModelFitError(f"unknown regressor: {name}")


@dataclass
class TwoPartExpectedGain:
    """P(V>0)·E[V|V>0] − P(V≤0)·E[−V|V≤0]."""

    feature_columns: list[str]
    classifier: Any = None
    positive_model: Any = None
    negative_model: Any = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TwoPartExpectedGain":
        y = np.asarray(y, dtype=float)
        positive = y > 0
        n_pos, n_neg = int(positive.sum()), int((~positive).sum())
        if n_pos < MIN_CLASS_SAMPLES or n_neg < MIN_CLASS_SAMPLES:
            raise ModelFitError(
                f"two_part_expected_gain needs >= {MIN_CLASS_SAMPLES} samples per sign class, "
                f"got positive={n_pos}, negative={n_neg}"
            )
        self.classifier = Pipeline(
            [("prep", _preprocessor(self.feature_columns)),
             ("clf", LogisticRegression(max_iter=1000))]
        ).fit(X, positive.astype(int))
        self.positive_model = Pipeline(
            [("prep", _preprocessor(self.feature_columns)), ("reg", Ridge(alpha=1.0))]
        ).fit(X[positive], y[positive])
        self.negative_model = Pipeline(
            [("prep", _preprocessor(self.feature_columns)), ("reg", Ridge(alpha=1.0))]
        ).fit(X[~positive], -y[~positive])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_pos = self.classifier.predict_proba(X)[:, 1]
        gain = self.positive_model.predict(X)
        loss = self.negative_model.predict(X)
        return p_pos * gain - (1.0 - p_pos) * loss


@dataclass
class FittedModel:
    name: str
    params: dict
    estimator: Any

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)


def fit_candidate(
    name: str, X: pd.DataFrame, y: np.ndarray, feature_columns: list[str], params: dict | None = None
) -> FittedModel:
    params = params or {}
    if name == "two_part_expected_gain":
        return FittedModel(name, params, TwoPartExpectedGain(feature_columns).fit(X, y))
    pipeline = Pipeline(
        [("prep", _preprocessor(feature_columns)), ("reg", _make_regressor(name, params))]
    ).fit(X, np.asarray(y, dtype=float))
    return FittedModel(name, params, pipeline)


def candidate_grid(name: str) -> list[dict]:
    if name == "ridge":
        return [{"alpha": a} for a in RIDGE_ALPHAS]
    if name == "hist_gradient_boosting":
        return [dict(p) for p in HGB_GRID]
    return [{}]


@dataclass
class SelectionResult:
    selected: FittedModel | None
    table: pd.DataFrame
    failures: dict[str, str] = field(default_factory=dict)


def select_value_model(
    candidates: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_columns: list[str],
    score_fn,
) -> SelectionResult:
    """Fit every candidate/grid point and keep the one ``score_fn`` likes best.

    ``score_fn(fitted) -> float`` must be computed on the VALIDATION split only
    (portfolio WQL at K=1); lower is better.
    """
    rows: list[dict] = []
    failures: dict[str, str] = {}
    best: tuple[float, FittedModel] | None = None
    for name in candidates:
        for params in candidate_grid(name):
            try:
                fitted = fit_candidate(name, X_train, y_train, feature_columns, params)
                score = float(score_fn(fitted))
            except (ModelFitError, ValueError) as exc:
                failures[f"{name}:{params}"] = str(exc)
                rows.append({"candidate": name, "params": str(params), "validation_wql_k1": np.nan,
                             "status": f"FAILED: {exc}"})
                continue
            rows.append({"candidate": name, "params": str(params),
                         "validation_wql_k1": score, "status": "ok"})
            if best is None or score < best[0]:
                best = (score, fitted)
    table = pd.DataFrame(rows).sort_values("validation_wql_k1", na_position="last")
    return SelectionResult(best[1] if best else None, table.reset_index(drop=True), failures)
