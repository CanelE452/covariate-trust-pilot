"""Pure split-boundary helpers for the online-memory pilot."""

import hashlib
from numbers import Integral

import numpy as np
import pandas as pd

from experiments.external_validity_screen import (
    confirmatory_h2,
    favorita_independent,
    screen,
)
from experiments.om_factorization_killtest import train as km_train
from experiments.unified_temporal_27_v3.config import ExperimentConfig
from experiments.unified_temporal_27_v3.training import make_windows, train_scale


_DATASET_LAYOUTS = {
    "m5": {
        "length": 1941,
        "train_end": 1717,
        "val_end": 1745,
        "stage_a": "series.parquet",
        "descriptor_variant": "availability_aware",
    },
    "favorita": {
        "length": 1688,
        "train_end": 1464,
        "val_end": 1492,
        "stage_a": "favorita_series.parquet",
        "descriptor_variant": "raw",
    },
}


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def evaluation_origins(
    series_length: int, horizon: int, count: int
) -> np.ndarray:
    """Return the final ``count`` non-overlapping forecast origins.

    The final forecast ends exactly at ``series_length``.  Earlier origins are
    spaced one horizon apart, so no phantom tail origin can be constructed.
    """

    length = _integer("series_length", series_length, minimum=1)
    forecast_horizon = _integer("horizon", horizon, minimum=1)
    origin_count = _integer("count", count, minimum=1)
    first_origin = length - forecast_horizon * origin_count
    if first_origin < 0:
        raise ValueError("series_length is too short for the requested origins")

    origins = first_origin + forecast_horizon * np.arange(
        origin_count, dtype=np.int64
    )
    if int(origins[-1]) + forecast_horizon != length:
        raise AssertionError("final evaluation origin does not end at series length")
    return origins


def eligible_series_mask(
    values: np.ndarray, model_train_end: int, min_positive: int
) -> np.ndarray:
    """Mark rows with enough strictly positive model-training observations."""

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional series-by-time array")

    train_end = _integer("model_train_end", model_train_end, minimum=1)
    required_positive = _integer("min_positive", min_positive, minimum=0)
    if train_end > array.shape[1]:
        raise ValueError("model_train_end exceeds the available series length")

    # Slice before conversion and validation: future values are deliberately not
    # read by this eligibility decision.
    train_values = np.asarray(array[:, :train_end], dtype=np.float64)
    if not bool(np.isfinite(train_values).all()):
        raise ValueError("model-training values must all be finite")
    return np.count_nonzero(train_values > 0.0, axis=1) >= required_positive


def dataset_config(name: str, n_series: int) -> ExperimentConfig:
    """Build the frozen dataset-specific configuration for this experiment."""

    if name not in _DATASET_LAYOUTS:
        raise ValueError(f"unknown dataset: {name}")
    count = _integer("n_series", n_series, minimum=1)
    layout = _DATASET_LAYOUTS[name]
    return ExperimentConfig(
        n_series=count,
        length=layout["length"],
        lookback=96,
        horizon=28,
        period=28,
        train_end=layout["train_end"],
        val_end=layout["val_end"],
    )


def _source_dataset(name: str) -> dict:
    if name == "m5":
        return confirmatory_h2.m5_full()
    if name == "favorita":
        return favorita_independent.load_pool()
    raise ValueError(f"unknown dataset: {name}")


def _validate_source(data: dict, cfg: ExperimentConfig) -> dict[str, np.ndarray]:
    y = np.asarray(data["y"], dtype=np.float32)
    z = np.asarray(data["z"], dtype=np.float32)
    series_id = np.asarray(data["series_id"]).astype(str)
    available_from = np.asarray(data["available_from"])
    expected_shape = (cfg.n_series, cfg.length)

    if y.shape != expected_shape or z.shape != expected_shape:
        raise ValueError(
            f"source length or row count mismatch: expected {expected_shape}, "
            f"got y={y.shape}, z={z.shape}"
        )
    if series_id.shape != (cfg.n_series,):
        raise ValueError("series_id must contain one ID per source row")
    if np.unique(series_id).size != cfg.n_series:
        raise ValueError("source series_id values must be unique")
    if available_from.shape != (cfg.n_series,):
        raise ValueError("available_from must contain one value per source row")
    if not np.issubdtype(available_from.dtype, np.integer):
        if not np.equal(available_from, np.floor(available_from)).all():
            raise ValueError("available_from must be integer-valued")
    available_from = available_from.astype(np.int64, copy=False)
    if bool((available_from < 0).any()) or bool(
        (available_from >= cfg.length).any()
    ):
        raise ValueError("available_from must fall inside the native series length")
    if not bool(np.isfinite(y).all()) or bool((y < 0).any()):
        raise ValueError("source y must be finite and nonnegative")
    if not bool(np.isfinite(z).all()) or not np.array_equal(z, (y > 0)):
        raise ValueError("source z must align with positive y values")

    first_positive = np.asarray(
        data.get(
            "first_positive",
            [int(np.argmax(row > 0)) if (row > 0).any() else cfg.length for row in y],
        ),
        dtype=np.int64,
    )
    if first_positive.shape != (cfg.n_series,):
        raise ValueError("first_positive must contain one value per source row")
    return {
        "series_id": series_id,
        "y": y,
        "z": z,
        "available_from": available_from,
        "first_positive": first_positive,
    }


def _id_sha256(values: np.ndarray) -> str:
    ordered = sorted(str(value) for value in values)
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _distribution_summary(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"cannot summarize non-finite {values.name}")
    quantiles = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "max": float(array.max()),
    }


def load_independent_population(name: str, min_positive: int = 20) -> dict:
    """Load, re-qualify, and audit the independent natural population."""

    if name not in _DATASET_LAYOUTS:
        raise ValueError(f"unknown dataset: {name}")
    threshold = _integer("min_positive", min_positive, minimum=0)
    source = _source_dataset(name)
    source_count = int(np.asarray(source["series_id"]).size)
    source_cfg = dataset_config(name, n_series=source_count)
    arrays = _validate_source(source, source_cfg)
    layout = _DATASET_LAYOUTS[name]

    if layout["descriptor_variant"] == "availability_aware":
        starts = arrays["available_from"]
    else:
        starts = np.zeros(source_count, dtype=np.int64)
    descriptor_rows = [
        screen.describe_series(arrays["y"][i], int(starts[i]), source_cfg.train_end)
        for i in range(source_count)
    ]
    descriptors = pd.DataFrame(descriptor_rows)
    descriptors["series_id"] = arrays["series_id"]
    descriptors["segment_start"] = starts
    descriptors["descriptor_variant"] = layout["descriptor_variant"]
    descriptors["train_scale"] = train_scale(
        {"y": arrays["y"], "z": arrays["z"]}, source_cfg
    )

    eligible = descriptors["n_positive_train"].to_numpy(dtype=np.int64) >= threshold
    stage_path = screen.REPO / "data" / "processed" / layout["stage_a"]
    stage_frame = pd.read_parquet(stage_path, columns=["series_id"])
    if "series_id" not in stage_frame:
        raise ValueError(f"Stage-A source has no series_id column: {stage_path}")
    stage_ids = set(stage_frame["series_id"].astype(str))
    is_stage_a = np.fromiter(
        (series_id in stage_ids for series_id in arrays["series_id"]),
        dtype=bool,
        count=source_count,
    )
    selected = eligible & ~is_stage_a
    if bool((arrays["available_from"][selected] >= source_cfg.val_end).any()):
        raise ValueError(
            "selected available_from values must all precede val_end"
        )
    selected_indices = np.flatnonzero(selected)
    selected_indices = selected_indices[
        np.argsort(arrays["series_id"][selected_indices], kind="stable")
    ]

    independent_data = {
        "name": name,
        "series_id": arrays["series_id"][selected_indices].copy(),
        "y": arrays["y"][selected_indices].copy(),
        "z": arrays["z"][selected_indices].copy(),
        "available_from": arrays["available_from"][selected_indices].copy(),
        "first_positive": arrays["first_positive"][selected_indices].copy(),
    }
    independent_descriptors = descriptors.iloc[selected_indices].reset_index(drop=True)
    independent_count = int(selected_indices.size)
    if independent_count == 0:
        raise ValueError("independent eligible population is empty")

    eligible_before = int(eligible.sum())
    stage_excluded = int((eligible & is_stage_a).sum())
    sensitivity: dict[str, dict[str, int | str]] = {}
    positive_counts = descriptors["n_positive_train"].to_numpy(dtype=np.int64)
    for sensitivity_threshold in (15, 20, 30):
        sensitivity_eligible = positive_counts >= sensitivity_threshold
        sensitivity_selected = sensitivity_eligible & ~is_stage_a
        sensitivity_ids = arrays["series_id"][sensitivity_selected]
        sensitivity[str(sensitivity_threshold)] = {
            "min_positive_train": sensitivity_threshold,
            "eligible_before_stage_a": int(sensitivity_eligible.sum()),
            "stage_a_excluded": int((sensitivity_eligible & is_stage_a).sum()),
            "eligible_independent": int(sensitivity_selected.sum()),
            "independent_id_sha256": _id_sha256(sensitivity_ids),
        }
    manifest = {
        "dataset": name,
        "descriptor_variant": layout["descriptor_variant"],
        "model_train_end": source_cfg.train_end,
        "min_positive": threshold,
        "full_total": source_count,
        "eligible_before_stage_a": eligible_before,
        "stage_a_excluded": stage_excluded,
        "eligible_independent": independent_count,
        "ineligible": int(source_count - eligible_before),
        "exclusion_reasons": {
            f"n_positive_train_below_{threshold}": int(source_count - eligible_before),
            "stage_a_membership": stage_excluded,
        },
        "independent_id_sha256": _id_sha256(independent_data["series_id"]),
        "eligible_before_stage_a_id_sha256": _id_sha256(
            arrays["series_id"][eligible]
        ),
        "stage_a_excluded_id_sha256": _id_sha256(
            arrays["series_id"][eligible & is_stage_a]
        ),
        "eligibility_sensitivity_no_training": sensitivity,
        "distributions": {
            column: _distribution_summary(independent_descriptors[column])
            for column in (
                "n_positive_train",
                "zero_ratio_train",
                "train_scale",
            )
        },
    }
    cfg = dataset_config(name, n_series=independent_count)
    return {
        "data": independent_data,
        "descriptors": independent_descriptors,
        "manifest": manifest,
        "cfg": cfg,
    }


def build_external_split(
    data: dict[str, np.ndarray],
    cfg: object,
    *,
    train_origin_stride: int,
    forecast_origins: np.ndarray,
):
    """Build the frozen external split without changing the canonical trainer.

    Training origins are the canonical dense origins thinned by the frozen
    stride.  Validation remains dense.  Test origins are supplied explicitly
    so the warm-up and evaluation schedule cannot acquire a phantom origin.
    M5 pre-availability targets are masked only in training and validation.
    """

    stride = _integer("train_origin_stride", train_origin_stride, minimum=1)
    arrays = {
        "y": np.asarray(data["y"], dtype=np.float32),
        "z": np.asarray(data["z"], dtype=np.float32),
    }
    if arrays["y"].ndim != 2 or arrays["y"].shape != arrays["z"].shape:
        raise ValueError("y and z must have identical series-by-time shapes")
    if arrays["y"].shape[1] != int(cfg.length):
        raise ValueError("data length does not match the experiment config")

    available = np.asarray(data["available_from"])
    if available.shape != (arrays["y"].shape[0],):
        raise ValueError("available_from must contain one value per series")
    if not np.issubdtype(available.dtype, np.integer):
        if not np.equal(available, np.floor(available)).all():
            raise ValueError("available_from must be integer-valued")
    available = available.astype(np.int64, copy=False)
    if bool((available < 0).any()):
        raise ValueError("available_from must be nonnegative")
    if int(available.max(initial=0)) >= int(cfg.val_end):
        raise ValueError("a series becomes available inside the evaluation window")

    raw_origins = np.asarray(forecast_origins)
    if raw_origins.ndim != 1 or raw_origins.size == 0:
        raise ValueError("forecast_origins must be a nonempty vector")
    if not np.issubdtype(raw_origins.dtype, np.number):
        raise ValueError("forecast_origins must be numeric integers")
    numeric_origins = raw_origins.astype(np.float64)
    if not bool(np.isfinite(numeric_origins).all()) or not bool(
        np.equal(numeric_origins, np.floor(numeric_origins)).all()
    ):
        raise ValueError("forecast_origins must be finite integer values")
    integer_origins = numeric_origins.astype(np.int64)
    if bool((integer_origins < int(cfg.val_end)).any()):
        raise ValueError("forecast origin precedes the evaluation split")
    if bool((integer_origins + int(cfg.horizon) > int(cfg.length)).any()):
        raise ValueError("forecast origin leaves a partial tail horizon")
    if bool(
        (integer_origins < np.iinfo(np.int32).min).any()
        or (integer_origins > np.iinfo(np.int32).max).any()
    ):
        raise ValueError("forecast origin exceeds int32 range")
    test_origins = integer_origins.astype(np.int32)
    if np.unique(test_origins).size != test_origins.size:
        raise ValueError("forecast_origins must be unique")
    if bool(np.any(np.diff(test_origins) <= 0)):
        raise ValueError("forecast_origins must be strictly increasing")

    scale = train_scale(arrays, cfg)
    train_origins = km_train.dense_origins(
        0, cfg.train_end, cfg.horizon, cfg.lookback
    )[::stride]
    validation_origins = km_train.dense_origins(
        cfg.train_end, cfg.val_end, cfg.horizon, cfg.lookback
    )
    train_windows = make_windows(
        arrays, train_origins, 0, cfg.train_end, cfg, scale
    )
    validation_windows = make_windows(
        arrays,
        validation_origins,
        cfg.train_end,
        cfg.val_end,
        cfg,
        scale,
    )
    test_windows = make_windows(
        arrays, test_origins, cfg.val_end, cfg.length, cfg, scale
    )

    if int(available.max(initial=0)) > 0:
        train_windows = screen._mask_before_availability(
            train_windows, available, cfg
        )
        validation_windows = screen._mask_before_availability(
            validation_windows, available, cfg
        )

    train_values = arrays["y"][:, : cfg.train_end]
    train_occurrence = arrays["z"][:, : cfg.train_end]
    positives = train_values[train_occurrence > 0.0]
    positive_variance = (
        float(positives.var(ddof=1)) if positives.size > 1 else 1.0
    )
    return km_train.Split(
        train=train_windows,
        validation=validation_windows,
        test=test_windows,
        scale=scale,
        positive_variance=max(positive_variance, 1e-6),
    )
