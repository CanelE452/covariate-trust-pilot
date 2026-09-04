"""Read-only reproduction of the frozen three-origin Point/Hurdle panel."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .gates import spearman_diagnostic
from .policies import exponential_hurdle_weight


__all__ = ["reproduce_three_origin"]


_DATASETS = ("m5", "favorita")
_HORIZON = 28
_ORIGINS_PER_SERIES = 3
_PRACTICAL_GAIN_THRESHOLD = 2.0
_CONVEX_HURDLE_WEIGHT_GRID = tuple(index / 20.0 for index in range(21))
_BOOTSTRAP_SEED = 20260904
_BOOTSTRAP_DRAWS = 2_000
_STATIC_CONVEX_ATOL = 5e-6
_BOOTSTRAP_POINT_ESTIMATE_ATOL = 5e-6

_STATIC_CONVEX_EXPECTED = {
    "favorita": {
        "loss_50_50": 2.4211301803588867,
        "best_static_hurdle_weight": 0.85,
        "best_static_loss": 2.407510995864868,
        "series_oracle_loss": 2.355938,
        "origin_oracle_loss": 2.3281784,
    },
    "m5": {
        "loss_50_50": 1.631527066230774,
        "best_static_hurdle_weight": 0.6,
        "best_static_loss": 1.6311854124069214,
        "series_oracle_loss": 1.6096307,
        "origin_oracle_loss": 1.5923551,
    },
}
_BOOTSTRAP_POINT_ESTIMATE_EXPECTED = {
    "m5": 0.5612785090289383,
    "favorita": -0.47483706629762645,
}

_RAW_SHA256 = {
    "m5": "a2810787033baac622e6558a53da526f0cb9b2e80d09cf2c4dbe699d2f207f6f",
    "favorita": "0df546bb479f70b8667c515c421ffe2a4f91dada9f1d50c0ee0300a776c22822",
}
_REFERENCE_SHA256 = {
    "condition_discovery": (
        "0917e4cc69948c72bf9c33afcccf969a8dc8bb2c74271bb00777cba1d5da532f"
    ),
    "recoverability": (
        "4c21015897e140de30177d7b8c97093c8ccadbf3ed00545ef8070e6206485e41"
    ),
}
_REFERENCE_RELATIVE_PATHS = {
    "condition_discovery": Path("pointhurdle_condition_discovery")
    / "paired_panel.parquet",
    "recoverability": Path("pointhurdle_recoverability")
    / "multi_origin_paired_panel.parquet",
}

_RAW_COLUMNS = (
    "series_id",
    "group",
    "origin",
    "step",
    "y_observed",
    "occurrence",
    "target_mask",
    "point_mean_prediction",
    "hurdle_mean_prediction",
    "hurdle_p_prediction",
    "hurdle_mu_prediction",
)
_RAW_NUMERIC_COLUMNS = (
    "origin",
    "step",
    "y_observed",
    "occurrence",
    "point_mean_prediction",
    "hurdle_mean_prediction",
    "hurdle_p_prediction",
    "hurdle_mu_prediction",
)
_RAW_ROWS = {"m5": 478_212, "favorita": 454_020}
_PANEL_ROWS = {"m5": 17_079, "favorita": 16_215}
_SERIES = {"m5": 5_693, "favorita": 5_405}
_ORIGINS = {
    "m5": (1857, 1885, 1913),
    "favorita": (1604, 1632, 1660),
}
_STEPS = tuple(range(_HORIZON))

_CONDITION_COLUMNS = (
    "series_id",
    "group",
    "outer_origin_id",
    "n_steps",
    "mse_point",
    "mse_hurdle",
    "n_positive_test",
    "mean_y",
    "loss_point",
    "loss_hurdle",
    "dataset_id",
    "paired_valid",
    "delta_rmse",
    "G",
    "winner",
)
_CONDITION_KEYS = ("dataset_id", "series_id", "outer_origin_id")
_CONDITION_FLOAT_COLUMNS = (
    "mse_point",
    "mse_hurdle",
    "n_positive_test",
    "mean_y",
    "loss_point",
    "loss_hurdle",
    "delta_rmse",
    "G",
)
_CONDITION_EXACT_COLUMNS = ("group", "n_steps", "paired_valid", "winner")
_CONDITION_ATOL = 1e-7

_RECOVERABILITY_COLUMNS = (
    "dataset_id",
    "series_id",
    "origin_id",
    "loss_point",
    "loss_hurdle",
    "n_steps",
    "G",
    "winner",
)
_RECOVERABILITY_KEYS = ("dataset_id", "series_id", "origin_id")
_RECOVERABILITY_GAIN_ATOL = 5e-5

_WINNER_COUNTS = {"point": 11_269, "neutral": 7_101, "hurdle": 14_924}
_WINNER_RATES = {
    "point": 33.846939388478404,
    "neutral": 21.32816723734006,
    "hurdle": 44.824893374181535,
}
# Phase 0 recorded five decimal places.  The earlier recoverability audit fixed
# the tolerance for its two-decimal reported win shares at 0.05 percentage point.
_AUDIT_LABEL_RATES = {
    "point": 33.84694,
    "neutral": 21.32817,
    "hurdle": 44.82489,
}
_REPORTED_LABEL_RATES = {"point": 33.85, "neutral": 21.33, "hurdle": 44.82}
_AUDIT_LABEL_RATE_ATOL = 5e-6
_REPORTED_LABEL_RATE_ATOL = 5e-2


def _convex_loss_grid(steps: pd.DataFrame) -> pd.DataFrame:
    """Return per-series-origin RMSE for every static Hurdle weight."""
    required = (
        "dataset_id",
        "series_id",
        "origin",
        "y_observed",
        "point_mean_prediction",
        "hurdle_mean_prediction",
    )
    missing = [column for column in required if column not in steps.columns]
    if missing:
        raise ValueError(f"convex step frame is missing columns: {missing}")
    if steps.empty:
        raise ValueError("convex step frame must not be empty")
    if bool(steps.loc[:, required[:3]].isna().any(axis=None)):
        raise ValueError("convex step keys must not be missing")

    numeric = {}
    for column in required[2:]:
        try:
            values = pd.to_numeric(steps[column], errors="raise").to_numpy()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"convex column {column!r} must be numeric") from exc
        if not bool(np.isfinite(np.asarray(values, dtype=np.float64)).all()):
            raise ValueError(f"convex column {column!r} must be finite")
        numeric[column] = values

    keys = [steps["dataset_id"], steps["series_id"], steps["origin"]]
    losses = {}
    for hurdle_weight in _CONVEX_HURDLE_WEIGHT_GRID:
        prediction = (
            (1.0 - hurdle_weight) * numeric["point_mean_prediction"]
            + hurdle_weight * numeric["hurdle_mean_prediction"]
        )
        squared_error = pd.Series(
            (numeric["y_observed"] - prediction) ** 2,
            index=steps.index,
        )
        losses[hurdle_weight] = np.sqrt(squared_error.groupby(keys).mean())
    result = pd.DataFrame(losses)
    result.index.names = ["dataset_id", "series_id", "origin_id"]
    values = result.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValueError("static convex grid produced invalid RMSE values")
    return result


def _stable_exponential_hurdle_weight(
    point_loss: np.ndarray,
    hurdle_loss: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Apply the audited stable softmax and verify its Hurdle-weight sign."""
    point = np.asarray(point_loss, dtype=np.float64)
    hurdle = np.asarray(hurdle_loss, dtype=np.float64)
    weights = np.asarray(
        exponential_hurdle_weight(point, hurdle, eta), dtype=np.float64
    )
    if weights.shape != point.shape:
        raise ValueError("exponential weighting returned the wrong shape")
    if not bool(np.isfinite(weights).all()):
        raise ValueError("exponential weighting produced non-finite weights")
    if bool((weights < 0.0).any() or (weights > 1.0).any()):
        raise ValueError("exponential weights must fall inside [0, 1]")
    if float(eta) > 0.0:
        tolerance = 8.0 * np.finfo(np.float64).eps
        wrong_sign = (
            ((hurdle < point) & (weights < 0.5 - tolerance))
            | ((hurdle > point) & (weights > 0.5 + tolerance))
            | ((hurdle == point) & (weights != 0.5))
        )
        if bool(wrong_sign.any()):
            raise ValueError("exponential Hurdle weight has the wrong loss sign")
    return weights


def _discounted_hurdle_weights(
    point_history: np.ndarray,
    hurdle_history: np.ndarray,
    *,
    eta: float,
    half_life: int,
) -> np.ndarray:
    """Weight Hurdle from discounted past losses ordered oldest to newest."""
    point = np.asarray(point_history, dtype=np.float64)
    hurdle = np.asarray(hurdle_history, dtype=np.float64)
    if point.ndim != 2 or hurdle.ndim != 2:
        raise ValueError("discounted loss histories must be two-dimensional")
    if point.shape != hurdle.shape:
        raise ValueError("discounted loss histories must have identical shapes")
    if point.size == 0 or point.shape[1] == 0:
        raise ValueError("discounted loss histories must not be empty")
    if not bool(np.isfinite(point).all() and np.isfinite(hurdle).all()):
        raise ValueError("discounted loss histories must all be finite")
    if bool((point < 0.0).any() or (hurdle < 0.0).any()):
        raise ValueError("discounted loss histories must be nonnegative")
    if (
        isinstance(half_life, (bool, np.bool_))
        or not isinstance(half_life, (int, np.integer))
        or int(half_life) < 1
    ):
        raise ValueError("half_life must be a positive integer")

    gamma = 0.5 ** (1.0 / int(half_life))
    ages = np.arange(point.shape[1], 0, -1, dtype=np.float64)
    discounts = np.power(gamma, ages)
    discounted_point = point @ discounts
    discounted_hurdle = hurdle @ discounts
    return _stable_exponential_hurdle_weight(
        discounted_point,
        discounted_hurdle,
        eta,
    )


def _series_cluster_resample_indices(
    series_ids: np.ndarray, sampled_series_ids: np.ndarray
) -> np.ndarray:
    """Expand sampled IDs into intact three-origin row clusters."""
    source = np.asarray(series_ids, dtype=object)
    sampled = np.asarray(sampled_series_ids, dtype=object)
    if source.ndim != 1 or sampled.ndim != 1:
        raise ValueError("series ID inputs must be one-dimensional")
    if source.size == 0 or sampled.size == 0:
        raise ValueError("series ID inputs must not be empty")
    if bool(pd.isna(source).any() or pd.isna(sampled).any()):
        raise ValueError("series IDs must not be missing")

    rows_by_series: dict[object, list[int]] = {}
    try:
        for row_index, series_id in enumerate(source):
            rows_by_series.setdefault(series_id, []).append(row_index)
    except TypeError as exc:
        raise TypeError("series IDs must be hashable") from exc
    if any(
        len(indices) != _ORIGINS_PER_SERIES
        for indices in rows_by_series.values()
    ):
        raise ValueError("every reproduction series must contain exactly three origins")

    expanded = []
    for series_id in sampled:
        if series_id not in rows_by_series:
            raise ValueError(f"sampled unknown series ID: {series_id!r}")
        expanded.extend(rows_by_series[series_id])
    return np.asarray(expanded, dtype=np.int64)


def _series_cluster_bootstrap_relative_improvement(
    series_ids: np.ndarray,
    baseline_loss: np.ndarray,
    candidate_loss: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap relative improvement while carrying all three origins."""
    source = np.asarray(series_ids, dtype=object)
    baseline = np.asarray(baseline_loss, dtype=np.float64)
    candidate = np.asarray(candidate_loss, dtype=np.float64)
    if source.ndim != 1 or baseline.ndim != 1 or candidate.ndim != 1:
        raise ValueError("bootstrap inputs must be one-dimensional")
    if not (source.shape == baseline.shape == candidate.shape):
        raise ValueError("bootstrap inputs must have identical lengths")
    if not bool(np.isfinite(baseline).all() and np.isfinite(candidate).all()):
        raise ValueError("bootstrap losses must all be finite")
    if bool((baseline < 0.0).any() or (candidate < 0.0).any()):
        raise ValueError("bootstrap losses must be nonnegative")
    if isinstance(draws, (bool, np.bool_)) or int(draws) != draws or draws < 1:
        raise ValueError("bootstrap draws must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or int(seed) != seed or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")

    positions: dict[object, list[int]] = {}
    try:
        for row_index, series_id in enumerate(source.tolist()):
            positions.setdefault(series_id, []).append(row_index)
    except TypeError as exc:
        raise TypeError("series IDs must be hashable") from exc
    unique_series = tuple(positions)
    if any(
        len(indices) != _ORIGINS_PER_SERIES
        for indices in positions.values()
    ):
        raise ValueError("every bootstrap series must contain exactly three origins")
    baseline_sums = np.asarray(
        [baseline[positions[series_id]].sum() for series_id in unique_series]
    )
    candidate_sums = np.asarray(
        [candidate[positions[series_id]].sum() for series_id in unique_series]
    )
    if bool((baseline_sums <= 0.0).any()):
        raise ValueError("every bootstrap series needs positive baseline loss")

    rng = np.random.default_rng(int(seed))
    result = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        selected = rng.integers(0, len(unique_series), size=len(unique_series))
        baseline_total = float(baseline_sums[selected].sum())
        candidate_total = float(candidate_sums[selected].sum())
        result[draw] = 100.0 * (baseline_total - candidate_total) / baseline_total
    if not bool(np.isfinite(result).all()):
        raise ValueError("series-cluster bootstrap produced non-finite values")
    return result


def _constant_spearman_diagnostic(outcome: np.ndarray) -> dict[str, object]:
    """Exercise the explicit constant-predictor degeneracy path."""
    values = np.asarray(outcome, dtype=np.float64)
    return spearman_diagnostic(np.ones(values.shape, dtype=np.float64), values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_unique(series: pd.Series) -> tuple:
    """Return unique values in their audited first-seen order."""
    return tuple(pd.unique(series).tolist())


def _require_columns(frame: pd.DataFrame, expected: tuple[str, ...], label: str) -> None:
    actual = tuple(frame.columns)
    if actual != expected:
        raise ValueError(
            f"{label} schema mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_raw_frame(dataset: str, frame: pd.DataFrame) -> None:
    _require_columns(frame, _RAW_COLUMNS, f"{dataset} raw")

    if len(frame) != _RAW_ROWS[dataset]:
        raise ValueError(
            f"{dataset} raw row count mismatch: expected {_RAW_ROWS[dataset]}, "
            f"got {len(frame)}"
        )
    if not pd.api.types.is_integer_dtype(frame["origin"].dtype):
        raise ValueError(f"{dataset} raw schema requires integer origin values")
    if not pd.api.types.is_integer_dtype(frame["step"].dtype):
        raise ValueError(f"{dataset} raw schema requires integer step values")
    if not pd.api.types.is_bool_dtype(frame["target_mask"].dtype):
        raise ValueError(f"{dataset} raw mask schema must be boolean")
    for column in _RAW_NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column].dtype):
            raise ValueError(
                f"{dataset} raw schema requires numeric column {column!r}"
            )

    if frame[["series_id", "group"]].isna().to_numpy().any():
        raise ValueError(f"{dataset} raw key/group values contain nulls")
    if bool(frame["target_mask"].isna().any()):
        raise ValueError(f"{dataset} raw target mask contains null values")
    if frame.loc[:, _RAW_NUMERIC_COLUMNS].isna().to_numpy().any():
        raise ValueError(
            f"{dataset} raw numeric values must all be finite (no NaN/Inf)"
        )
    numeric = frame.loc[:, _RAW_NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{dataset} raw numeric values must all be finite (no NaN/Inf)")

    if not bool(frame["target_mask"].to_numpy(dtype=bool).all()):
        raise ValueError(f"{dataset} raw target mask must be true for every scored step")

    actual_origins = _ordered_unique(frame["origin"])
    if actual_origins != _ORIGINS[dataset]:
        raise ValueError(
            f"{dataset} raw origins mismatch: expected {_ORIGINS[dataset]!r}, "
            f"got {actual_origins!r}"
        )
    actual_steps = _ordered_unique(frame["step"])
    if actual_steps != _STEPS:
        raise ValueError(
            f"{dataset} raw step order/values mismatch: expected 0..{_HORIZON - 1}"
        )

    key_columns = ["series_id", "origin", "step"]
    if bool(frame.duplicated(key_columns, keep=False).any()):
        raise ValueError(
            f"{dataset} raw prediction key is duplicate; pairing is not one-to-one"
        )

    series_count = int(frame["series_id"].nunique(dropna=False))
    if series_count != _SERIES[dataset]:
        raise ValueError(
            f"{dataset} raw series/key count mismatch: expected {_SERIES[dataset]}, "
            f"got {series_count}"
        )
    groups_per_series = frame.groupby("series_id", sort=False)["group"].nunique(
        dropna=False
    )
    if not bool(groups_per_series.eq(1).all()):
        raise ValueError(f"{dataset} raw group is not constant within series key")

    origins_per_series = frame.groupby("series_id", sort=False)["origin"].nunique(
        dropna=False
    )
    if not bool(origins_per_series.eq(len(_ORIGINS[dataset])).all()):
        raise ValueError(
            f"{dataset} raw origin pairing is incomplete for at least one series"
        )

    cells = frame.groupby(["series_id", "origin"], sort=False)["step"].agg(
        ["size", "nunique", "min", "max"]
    )
    expected_cells = _SERIES[dataset] * len(_ORIGINS[dataset])
    complete = (
        len(cells) == expected_cells
        and bool(cells["size"].eq(_HORIZON).all())
        and bool(cells["nunique"].eq(_HORIZON).all())
        and bool(cells["min"].eq(0).all())
        and bool(cells["max"].eq(_HORIZON - 1).all())
    )
    if not complete:
        raise ValueError(f"{dataset} raw step pairing is incomplete")

    observed = frame["y_observed"].to_numpy()
    occurrence = frame["occurrence"].to_numpy()
    if not np.array_equal(occurrence, observed > 0):
        raise ValueError(
            f"{dataset} raw occurrence values are inconsistent with y_observed > 0"
        )

    hurdle_mean = frame["hurdle_mean_prediction"].to_numpy()
    factorized = (
        frame["hurdle_p_prediction"].to_numpy()
        * frame["hurdle_mu_prediction"].to_numpy()
    )
    if not np.array_equal(hurdle_mean, factorized):
        raise ValueError(
            f"{dataset} raw hurdle factorization is broken: mean != p * mu"
        )


def _build_condition_panel(raw_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for dataset in _DATASETS:
        masked = raw_frames[dataset].loc[
            raw_frames[dataset]["target_mask"]
        ].copy()
        masked["se_point"] = (
            masked["y_observed"] - masked["point_mean_prediction"]
        ) ** 2
        masked["se_hurdle"] = (
            masked["y_observed"] - masked["hurdle_mean_prediction"]
        ) ** 2
        grouped = masked.groupby(
            ["series_id", "group", "origin"], as_index=False, sort=False
        ).agg(
            n_steps=("se_point", "size"),
            mse_point=("se_point", "mean"),
            mse_hurdle=("se_hurdle", "mean"),
            n_positive_test=("occurrence", "sum"),
            mean_y=("y_observed", "mean"),
        )
        grouped["loss_point"] = np.sqrt(grouped["mse_point"])
        grouped["loss_hurdle"] = np.sqrt(grouped["mse_hurdle"])
        grouped["dataset_id"] = dataset
        grouped = grouped.rename(columns={"origin": "outer_origin_id"})
        pieces.append(grouped)

    panel = pd.concat(pieces, ignore_index=True)
    panel["paired_valid"] = (
        panel["loss_point"].notna()
        & panel["loss_hurdle"].notna()
        & panel["n_steps"].gt(0)
    )
    panel["delta_rmse"] = panel["loss_point"] - panel["loss_hurdle"]
    panel["G"] = 100.0 * (
        1.0 - panel["loss_hurdle"] / panel["loss_point"]
    )
    panel["winner"] = np.where(
        panel["G"] > _PRACTICAL_GAIN_THRESHOLD,
        "hurdle",
        np.where(
            panel["G"] < -_PRACTICAL_GAIN_THRESHOLD, "point", "neutral"
        ),
    )

    if len(panel) != sum(_PANEL_ROWS.values()):
        raise ValueError(
            f"rebuilt panel row count mismatch: expected {sum(_PANEL_ROWS.values())}, "
            f"got {len(panel)}"
        )
    dataset_rows = panel["dataset_id"].value_counts(sort=False).to_dict()
    if dataset_rows != _PANEL_ROWS:
        raise ValueError(
            f"rebuilt panel dataset row counts mismatch: {dataset_rows!r}"
        )
    if bool(panel.duplicated(list(_CONDITION_KEYS), keep=False).any()):
        raise ValueError("rebuilt panel contains duplicate series-origin keys")
    if not bool(panel["paired_valid"].all()):
        raise ValueError("rebuilt panel has incomplete Point/Hurdle pairing")
    panel_numeric = panel.loc[:, _CONDITION_FLOAT_COLUMNS].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(panel_numeric).all():
        raise ValueError("rebuilt panel contains non-finite RMSE or gain values")
    return panel


def _align_to_reference(
    computed: pd.DataFrame,
    reference: pd.DataFrame,
    keys: tuple[str, ...],
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bool(computed.duplicated(list(keys), keep=False).any()):
        raise ValueError(f"computed {label} panel has duplicate keys")
    if bool(reference.duplicated(list(keys), keep=False).any()):
        raise ValueError(f"frozen {label} panel has duplicate keys")

    computed_indexed = computed.set_index(list(keys), drop=True)
    reference_indexed = reference.set_index(list(keys), drop=True)
    missing = reference_indexed.index.difference(computed_indexed.index)
    extra = computed_indexed.index.difference(reference_indexed.index)
    if len(missing) or len(extra):
        raise ValueError(
            f"{label} panel key mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    return computed_indexed.reindex(reference_indexed.index), reference_indexed


def _compare_condition_reference(
    panel: pd.DataFrame, reference: pd.DataFrame
) -> float:
    _require_columns(reference, _CONDITION_COLUMNS, "condition-discovery reference")
    if len(reference) != sum(_PANEL_ROWS.values()):
        raise ValueError("condition-discovery reference row count mismatch")

    computed, frozen = _align_to_reference(
        panel, reference, _CONDITION_KEYS, "condition-discovery"
    )
    for column in _CONDITION_EXACT_COLUMNS:
        if not np.array_equal(
            computed[column].to_numpy(), frozen[column].to_numpy()
        ):
            raise ValueError(
                f"condition-discovery reference mismatch in exact column {column!r}"
            )

    maximum = 0.0
    for column in _CONDITION_FLOAT_COLUMNS:
        left = computed[column].to_numpy(dtype=np.float64)
        right = frozen[column].to_numpy(dtype=np.float64)
        if not np.isfinite(right).all():
            raise ValueError(
                f"condition-discovery reference has non-finite {column!r} values"
            )
        difference = np.abs(left - right)
        column_maximum = float(difference.max(initial=0.0))
        maximum = max(maximum, column_maximum)
        if column_maximum > _CONDITION_ATOL:
            raise ValueError(
                "condition-discovery panel differs in "
                f"{column!r} by {column_maximum:.12g}, above {_CONDITION_ATOL:g}"
            )
    return maximum


def _compare_recoverability_reference(
    panel: pd.DataFrame, reference: pd.DataFrame
) -> float:
    _require_columns(reference, _RECOVERABILITY_COLUMNS, "recoverability reference")
    if len(reference) != sum(_PANEL_ROWS.values()):
        raise ValueError("recoverability reference row count mismatch")

    computed = panel[
        [
            "dataset_id",
            "series_id",
            "outer_origin_id",
            "loss_point",
            "loss_hurdle",
            "n_steps",
            "G",
            "winner",
        ]
    ].rename(columns={"outer_origin_id": "origin_id"})
    computed, frozen = _align_to_reference(
        computed, reference, _RECOVERABILITY_KEYS, "recoverability"
    )

    for column in ("n_steps", "winner"):
        if not np.array_equal(computed[column].to_numpy(), frozen[column].to_numpy()):
            raise ValueError(
                f"recoverability reference mismatch in exact column {column!r}"
            )
    frozen_numeric = frozen.loc[:, ["loss_point", "loss_hurdle", "G"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(frozen_numeric).all():
        raise ValueError("recoverability reference has non-finite loss/gain values")
    frozen_gain = frozen_numeric[:, 2]
    difference = np.abs(
        computed["G"].to_numpy(dtype=np.float64) - frozen_gain
    )
    maximum = float(difference.max(initial=0.0))
    if maximum > _RECOVERABILITY_GAIN_ATOL:
        raise ValueError(
            "recoverability panel gain differs by "
            f"{maximum:.12g}, above {_RECOVERABILITY_GAIN_ATOL:g}"
        )
    return maximum


def _winner_aggregates(panel: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    counts = {
        winner: int(panel["winner"].eq(winner).sum())
        for winner in ("point", "neutral", "hurdle")
    }
    if counts != _WINNER_COUNTS:
        raise ValueError(
            f"winner aggregate count mismatch: expected {_WINNER_COUNTS!r}, got {counts!r}"
        )

    rates = {
        winner: float(panel["winner"].eq(winner).mean() * 100.0)
        for winner in ("point", "neutral", "hurdle")
    }
    if any(abs(rates[key] - _WINNER_RATES[key]) > 1e-12 for key in rates):
        raise ValueError(
            f"winner aggregate rate mismatch: expected {_WINNER_RATES!r}, got {rates!r}"
        )
    if any(
        abs(rates[key] - _AUDIT_LABEL_RATES[key]) > _AUDIT_LABEL_RATE_ATOL
        for key in rates
    ):
        raise ValueError("winner rates exceed the Phase-0 audit's five-decimal tolerance")
    if any(
        abs(rates[key] - _REPORTED_LABEL_RATES[key])
        > _REPORTED_LABEL_RATE_ATOL
        for key in rates
    ):
        raise ValueError("winner rates exceed the reported two-decimal aggregate tolerance")
    return counts, rates


def _static_convex_diagnostic(
    raw_frames: Mapping[str, pd.DataFrame], panel: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame]:
    """Recompute the frozen static-convex ladder from step predictions."""
    step_columns = [
        "series_id",
        "origin",
        "y_observed",
        "point_mean_prediction",
        "hurdle_mean_prediction",
    ]
    pieces = []
    for dataset in _DATASETS:
        raw = raw_frames[dataset]
        steps = raw.loc[raw["target_mask"], step_columns].copy()
        steps.insert(0, "dataset_id", dataset)
        pieces.append(steps)
    loss_grid = _convex_loss_grid(pd.concat(pieces, ignore_index=True))
    if len(loss_grid) != sum(_PANEL_ROWS.values()):
        raise ValueError("static convex grid has the wrong series-origin row count")
    if tuple(loss_grid.columns) != _CONVEX_HURDLE_WEIGHT_GRID:
        raise ValueError("static convex grid does not contain exactly 0:.05:1")

    indexed_panel = (
        panel.rename(columns={"outer_origin_id": "origin_id"})
        .set_index(list(_RECOVERABILITY_KEYS), verify_integrity=True)
        .reindex(loss_grid.index)
    )
    if bool(indexed_panel[["loss_point", "loss_hurdle"]].isna().any(axis=None)):
        raise ValueError("static convex grid cannot align to the rebuilt panel")
    endpoint_differences = np.column_stack(
        [
            np.abs(
                loss_grid[0.0].to_numpy(dtype=np.float64)
                - indexed_panel["loss_point"].to_numpy(dtype=np.float64)
            ),
            np.abs(
                loss_grid[1.0].to_numpy(dtype=np.float64)
                - indexed_panel["loss_hurdle"].to_numpy(dtype=np.float64)
            ),
        ]
    )
    endpoint_maximum = float(endpoint_differences.max(initial=0.0))
    if endpoint_maximum > _CONDITION_ATOL:
        raise ValueError(
            "static convex grid endpoints differ from Point/Hurdle RMSE by "
            f"{endpoint_maximum:.12g}"
        )

    grid_values = loss_grid.to_numpy(dtype=np.float64)
    endpoint_upper = np.maximum(
        loss_grid[0.0].to_numpy(dtype=np.float64),
        loss_grid[1.0].to_numpy(dtype=np.float64),
    )
    range_tolerance = (
        64.0
        * np.finfo(np.float32).eps
        * np.maximum(1.0, endpoint_upper)
    )
    if bool(
        (grid_values < 0.0).any()
        or (grid_values > endpoint_upper[:, None] + range_tolerance[:, None]).any()
    ):
        raise ValueError("static convex loss falls outside its endpoint RMSE range")

    summaries: dict[str, dict[str, float]] = {}
    for dataset in _DATASETS:
        curves = loss_grid.xs(dataset, level="dataset_id")
        mean_curve = curves.mean(axis=0)
        best_weight = float(mean_curve.idxmin())
        series_curve = curves.groupby(level="series_id", sort=False).mean()
        best_by_series = series_curve.idxmin(axis=1)
        row_weights = best_by_series.reindex(
            curves.index.get_level_values("series_id")
        ).to_numpy()
        column_indices = curves.columns.get_indexer(row_weights)
        if bool((column_indices < 0).any()):
            raise ValueError("series convex oracle selected a weight outside the grid")
        series_selected = curves.to_numpy()[
            np.arange(len(curves)), column_indices
        ]
        summary = {
            "loss_50_50": float(mean_curve.loc[0.5]),
            "best_static_hurdle_weight": best_weight,
            "best_static_loss": float(mean_curve.loc[best_weight]),
            "series_oracle_loss": float(np.mean(series_selected)),
            "origin_oracle_loss": float(curves.min(axis=1).mean()),
            "grid_mean_loss_min": float(mean_curve.min()),
            "grid_mean_loss_max": float(mean_curve.max()),
        }
        values = np.asarray(tuple(summary.values()), dtype=np.float64)
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"{dataset} static convex summary is non-finite")
        hierarchy = (
            summary["origin_oracle_loss"],
            summary["series_oracle_loss"],
            summary["best_static_loss"],
            summary["loss_50_50"],
        )
        if any(
            left > right + _STATIC_CONVEX_ATOL
            for left, right in zip(hierarchy, hierarchy[1:])
        ):
            raise ValueError(
                f"{dataset} static convex reported quantities violate oracle bounds"
            )

        expected = _STATIC_CONVEX_EXPECTED[dataset]
        for quantity, expected_value in expected.items():
            difference = abs(summary[quantity] - expected_value)
            if difference > _STATIC_CONVEX_ATOL:
                raise ValueError(
                    f"{dataset} static convex {quantity} differs by "
                    f"{difference:.12g}, above {_STATIC_CONVEX_ATOL:g}"
                )
        summaries[dataset] = summary

    return (
        {
            "hurdle_weight_grid": list(_CONVEX_HURDLE_WEIGHT_GRID),
            "datasets": summaries,
            "endpoint_max_abs_diff": endpoint_maximum,
            "convex_range_valid": True,
            "reported_quantity_ordering_valid": True,
        },
        loss_grid,
    )


def _frozen_discounted_weight_diagnostic(
    panel: pd.DataFrame,
) -> dict[str, object]:
    """Stress the new discounted-weight primitive on the frozen panel."""
    extreme_weights = _stable_exponential_hurdle_weight(
        np.array([1_000.0, 2_000.0, 1_000_000.0]),
        np.array([2_000.0, 1_000.0, 1_000_000.0]),
        eta=1_000_000.0,
    )
    if not np.array_equal(extreme_weights, np.array([0.0, 1.0, 0.5])):
        raise ValueError("stable exponential weighting failed its underflow guard")

    eta = 32.0
    half_life = 1
    gamma = 0.5 ** (1.0 / half_life)
    all_weights = []
    sign_consistent = True
    for dataset in _DATASETS:
        subset = panel.loc[
            panel["dataset_id"].eq(dataset),
            ["series_id", "outer_origin_id", "loss_point", "loss_hurdle"],
        ]
        series_order = _ordered_unique(subset["series_id"])
        indexed = subset.set_index(
            ["series_id", "outer_origin_id"], verify_integrity=True
        )
        point = (
            indexed["loss_point"]
            .unstack("outer_origin_id")
            .reindex(index=series_order, columns=_ORIGINS[dataset])
            .to_numpy(dtype=np.float64)
        )
        hurdle = (
            indexed["loss_hurdle"]
            .unstack("outer_origin_id")
            .reindex(index=series_order, columns=_ORIGINS[dataset])
            .to_numpy(dtype=np.float64)
        )
        point = np.square(point)
        hurdle = np.square(hurdle)
        if not bool(np.isfinite(point).all() and np.isfinite(hurdle).all()):
            raise ValueError("frozen discounted-weight histories are non-finite")

        for history_length in range(1, _ORIGINS_PER_SERIES):
            point_history = point[:, :history_length]
            hurdle_history = hurdle[:, :history_length]
            weights = _discounted_hurdle_weights(
                point_history,
                hurdle_history,
                eta=eta,
                half_life=half_life,
            )
            ages = np.arange(history_length, 0, -1, dtype=np.float64)
            discounts = np.power(gamma, ages)
            discounted_point = point_history @ discounts
            discounted_hurdle = hurdle_history @ discounts
            tolerance = 8.0 * np.finfo(np.float64).eps
            wrong_sign = (
                ((discounted_hurdle < discounted_point) & (weights < 0.5 - tolerance))
                | ((discounted_hurdle > discounted_point) & (weights > 0.5 + tolerance))
                | ((discounted_hurdle == discounted_point) & (weights != 0.5))
            )
            sign_consistent = sign_consistent and not bool(wrong_sign.any())
            all_weights.append(weights)

    frozen_weights = np.concatenate(all_weights)
    expected_count = sum(_SERIES.values()) * (_ORIGINS_PER_SERIES - 1)
    if frozen_weights.size != expected_count:
        raise ValueError(
            "frozen discounted-weight count mismatch: "
            f"expected {expected_count}, got {frozen_weights.size}"
        )
    if not bool(np.isfinite(frozen_weights).all()):
        raise ValueError("frozen discounted weights contain NaN or Inf")
    if bool((frozen_weights < 0.0).any() or (frozen_weights > 1.0).any()):
        raise ValueError("frozen discounted weights fall outside [0, 1]")
    if not sign_consistent:
        raise ValueError("frozen discounted weights have the wrong loss sign")

    return {
        "scope": (
            "numerical/sign diagnostic on squared frozen RMSE histories; "
            "not a reproduction of historical undiscounted D3 or six-origin B4"
        ),
        "eta": eta,
        "half_life": half_life,
        "extreme_hurdle_weights": extreme_weights.tolist(),
        "extreme_underflow_guarded": True,
        "frozen_discounted_weight_count": int(frozen_weights.size),
        "frozen_weights_finite": True,
        "frozen_weight_sign_consistent": True,
        "frozen_hurdle_weight_min": float(frozen_weights.min()),
        "frozen_hurdle_weight_max": float(frozen_weights.max()),
    }


def _bootstrap_diagnostic(
    panel: pd.DataFrame, loss_grid: pd.DataFrame
) -> dict[str, object]:
    """Recompute P3 point estimates and bootstrap intact series clusters."""
    datasets: dict[str, dict[str, object]] = {}
    all_deterministic = True
    all_finite = True
    for target_dataset in _DATASETS:
        source_dataset = next(
            dataset for dataset in _DATASETS if dataset != target_dataset
        )
        source = panel.loc[panel["dataset_id"].eq(source_dataset)]
        source_means = {
            "point": float(source["loss_point"].mean()),
            "hurdle": float(source["loss_hurdle"].mean()),
        }
        baseline_arm = min(
            ("point", "hurdle"), key=lambda arm: source_means[arm]
        )
        if baseline_arm != "hurdle":
            raise ValueError(
                f"{source_dataset} no longer selects the frozen Hurdle baseline"
            )

        target_grid = loss_grid.xs(target_dataset, level="dataset_id")
        target = (
            panel.loc[panel["dataset_id"].eq(target_dataset)]
            .rename(columns={"outer_origin_id": "origin_id"})
            .set_index(["series_id", "origin_id"], verify_integrity=True)
            .reindex(target_grid.index)
        )
        baseline = target[f"loss_{baseline_arm}"].to_numpy()
        candidate = target_grid[0.5].to_numpy()
        if not bool(np.isfinite(baseline).all() and np.isfinite(candidate).all()):
            raise ValueError("bootstrap comparison contains NaN or Inf")
        baseline_mean = float(np.mean(baseline))
        candidate_mean = float(np.mean(candidate))
        if baseline_mean <= 0.0:
            raise ValueError("bootstrap comparison requires positive baseline loss")
        point_estimate = 100.0 * (
            baseline_mean - candidate_mean
        ) / baseline_mean
        expected = _BOOTSTRAP_POINT_ESTIMATE_EXPECTED[target_dataset]
        difference = abs(point_estimate - expected)
        if difference > _BOOTSTRAP_POINT_ESTIMATE_ATOL:
            raise ValueError(
                f"{target_dataset} 50:50 point estimate differs by "
                f"{difference:.12g}, above {_BOOTSTRAP_POINT_ESTIMATE_ATOL:g}"
            )

        series_ids = target_grid.index.get_level_values("series_id").to_numpy()
        first = _series_cluster_bootstrap_relative_improvement(
            series_ids,
            baseline,
            candidate,
            draws=_BOOTSTRAP_DRAWS,
            seed=_BOOTSTRAP_SEED,
        )
        second = _series_cluster_bootstrap_relative_improvement(
            series_ids,
            baseline,
            candidate,
            draws=_BOOTSTRAP_DRAWS,
            seed=_BOOTSTRAP_SEED,
        )
        deterministic = bool(np.array_equal(first, second))
        finite = bool(np.isfinite(first).all())
        if not deterministic:
            raise ValueError("series-cluster bootstrap is not deterministic")
        if not finite:
            raise ValueError("series-cluster bootstrap produced NaN or Inf")
        ci_low, ci_high = np.percentile(first, [2.5, 97.5])
        all_deterministic = all_deterministic and deterministic
        all_finite = all_finite and finite
        datasets[target_dataset] = {
            "source_dataset": source_dataset,
            "source_baseline_choice": baseline_arm,
            "point_estimate": point_estimate,
            "mean": float(first.mean()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    return {
        "seed": _BOOTSTRAP_SEED,
        "draws": _BOOTSTRAP_DRAWS,
        "deterministic": all_deterministic,
        "finite": all_finite,
        "reference_scope": (
            "point estimates reproduce the frozen P3 artifact; intervals are "
            "new seed-20260904 diagnostics because the historical intervals used "
            "seed 20260813"
        ),
        "datasets": datasets,
    }


def _hash_inputs(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in paths.items()}


def reproduce_three_origin(
    raw_paths: dict[str, Path], reference_root: Path
) -> dict:
    """Read, validate, recompute, and return a report without writing."""
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != set(_DATASETS):
        actual = (
            tuple(raw_paths)
            if isinstance(raw_paths, Mapping)
            else type(raw_paths).__name__
        )
        raise ValueError(
            f"raw_paths must contain exactly the dataset keys {_DATASETS!r}; got {actual!r}"
        )

    ordered_raw_paths = {dataset: Path(raw_paths[dataset]) for dataset in _DATASETS}
    reference_root = Path(reference_root)
    reference_paths = {
        name: reference_root / relative
        for name, relative in _REFERENCE_RELATIVE_PATHS.items()
    }
    frozen_paths = {
        **{f"raw:{name}": path for name, path in ordered_raw_paths.items()},
        **{f"reference:{name}": path for name, path in reference_paths.items()},
    }
    before = _hash_inputs(frozen_paths)

    try:
        for dataset in _DATASETS:
            actual = before[f"raw:{dataset}"]
            if actual != _RAW_SHA256[dataset]:
                raise ValueError(
                    f"{dataset} raw SHA-256 hash mismatch: "
                    f"expected {_RAW_SHA256[dataset]}, got {actual}"
                )
        for name in _REFERENCE_RELATIVE_PATHS:
            actual = before[f"reference:{name}"]
            if actual != _REFERENCE_SHA256[name]:
                raise ValueError(
                    f"{name} reference SHA-256 hash mismatch: "
                    f"expected {_REFERENCE_SHA256[name]}, got {actual}"
                )

        raw_frames = {}
        for dataset in _DATASETS:
            frame = pd.read_parquet(ordered_raw_paths[dataset])
            _validate_raw_frame(dataset, frame)
            raw_frames[dataset] = frame

        panel = _build_condition_panel(raw_frames)
        condition_reference = pd.read_parquet(
            reference_paths["condition_discovery"]
        )
        recoverability_reference = pd.read_parquet(
            reference_paths["recoverability"]
        )
        condition_maximum = _compare_condition_reference(
            panel, condition_reference
        )
        recoverability_maximum = _compare_recoverability_reference(
            panel, recoverability_reference
        )
        winner_counts, winner_rates = _winner_aggregates(panel)
        static_convex, convex_loss_grid = _static_convex_diagnostic(
            raw_frames, panel
        )
        exponential_weighting = _frozen_discounted_weight_diagnostic(panel)
        series_cluster_bootstrap = _bootstrap_diagnostic(
            panel, convex_loss_grid
        )
        constant_spearman = _constant_spearman_diagnostic(
            panel["G"].to_numpy(dtype=np.float64)
        )
        expected_constant_spearman = {
            "status": "DEGENERATE",
            "rho": None,
            "pvalue": None,
            "n": sum(_PANEL_ROWS.values()),
        }
        if constant_spearman != expected_constant_spearman:
            raise ValueError(
                "constant-predictor Spearman was not reported as DEGENERATE"
            )

        report = {
            "status": "PASS",
            "input_sha256": {
                dataset: before[f"raw:{dataset}"] for dataset in _DATASETS
            },
            "reference_sha256": {
                name: before[f"reference:{name}"]
                for name in _REFERENCE_RELATIVE_PATHS
            },
            "raw_rows": dict(_RAW_ROWS),
            "panel_rows": int(len(panel)),
            "dataset_rows": dict(_PANEL_ROWS),
            "series": dict(_SERIES),
            "origins": {
                dataset: list(_ORIGINS[dataset]) for dataset in _DATASETS
            },
            "panel_max_abs_diff": condition_maximum,
            "recoverability_panel_max_abs_gain_diff": recoverability_maximum,
            "winner_counts": winner_counts,
            "winner_rates_percent": winner_rates,
            "static_convex": static_convex,
            "exponential_weighting": exponential_weighting,
            "series_cluster_bootstrap": series_cluster_bootstrap,
            "constant_spearman": constant_spearman,
            "checks": {
                "schema": True,
                "row_counts": True,
                "keys_unique": True,
                "pairing": True,
                "masks": True,
                "finiteness": True,
                "occurrence_identity": True,
                "hurdle_factorization": True,
                "condition_discovery_reference": True,
                "recoverability_reference": True,
                "aggregate_tolerances": True,
                "static_convex_grid": True,
                "exponential_weighting": True,
                "series_cluster_bootstrap": True,
                "constant_spearman_degenerate": True,
            },
        }
    finally:
        after = _hash_inputs(frozen_paths)
        if after != before:
            changed = [
                name
                for name in frozen_paths
                if after.get(name) != before.get(name)
            ]
            raise ValueError(
                "frozen input hash changed during read-only reproduction: "
                + ", ".join(changed)
            )

    return report
