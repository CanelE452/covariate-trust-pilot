"""Orchestration for the preregistered, non-scientific M5 smoke run.

This module deliberately performs no artifact writes.  Its return value keeps
the paired prediction and loss frames in memory and exposes a JSON-safe report
that a separate, exclusive artifact writer may persist after the run.
"""

from __future__ import annotations

import io
import time
from collections.abc import Mapping
from dataclasses import fields, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.om_factorization_killtest import train as km_train

from .bridge import build_policy_cases
from .data import build_external_split
from .evaluation import evaluate_b4_cases, evaluate_m1_cases
from .pipeline import (
    build_prediction_frame,
    normalized_loss_frame,
    project_full_seed0_runtime,
    project_retrieval_runtime,
)
from .smoke import stratified_smoke_ids
from .trainer import train_one_on_split


POINT_MODEL = "M0PM_point_mse_param_matched"
HURDLE_MODEL = "M1_factorized_mean"
SMOKE_SERIES = 200
SMOKE_SEED = 20260904
MODEL_SEED = 0
TRAIN_ORIGIN_STRIDE = 7
SMOKE_ORIGINS = np.array([1745, 1773], dtype=np.int32)
FULL_FORECAST_ORIGINS = 7
RUNTIME_GATE_GPU_HOURS = 6.0
FALLBACK_SERIES_PER_DATASET = 2_000
RETRIEVAL_K = 32
SMOKE_B4_ETA = 0.5
SMOKE_B4_HALF_LIFE = 1
SMOKE_M1_LAMBDA_MAX = 0.25
FULL_EVALUATION_ORIGINS = 6
RETRIEVAL_PASSES_PER_DATASET = 7


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (Integral, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return int(value)
    if isinstance(value, (Real, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("report values must be finite")
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _positive_counts(full_series_counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(full_series_counts, Mapping):
        raise TypeError("full_series_counts must be a mapping")
    if set(full_series_counts) != {"m5", "favorita"}:
        raise ValueError("full_series_counts must have exactly m5 and favorita")
    result: dict[str, int] = {}
    for dataset, count in full_series_counts.items():
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
            raise TypeError(f"full_series_counts[{dataset}] must be an integer")
        result[dataset] = int(count)
        if result[dataset] <= 0:
            raise ValueError("full-series counts must be positive")
    return result


def _validate_population(population: Mapping[str, object]) -> tuple[dict, pd.DataFrame, object, Mapping]:
    if not isinstance(population, Mapping):
        raise TypeError("population must be a mapping")
    missing = [
        key for key in ("data", "descriptors", "cfg", "manifest")
        if key not in population
    ]
    if missing:
        raise ValueError(f"population is missing keys: {missing}")
    data = population["data"]
    descriptors = population["descriptors"]
    cfg = population["cfg"]
    manifest = population["manifest"]
    if not isinstance(data, Mapping):
        raise TypeError("population data must be a mapping")
    if not isinstance(descriptors, pd.DataFrame):
        raise TypeError("population descriptors must be a DataFrame")
    if not isinstance(manifest, Mapping):
        raise TypeError("population manifest must be a mapping")
    for key in ("name", "series_id", "y", "z", "available_from"):
        if key not in data:
            raise ValueError(f"population data is missing {key}")
    if str(data["name"]).lower() != "m5":
        raise ValueError("the preregistered smoke dataset is M5")
    expected_cfg = {
        "length": 1941,
        "lookback": 96,
        "horizon": 28,
        "train_end": 1717,
        "val_end": 1745,
    }
    for name, expected in expected_cfg.items():
        if int(getattr(cfg, name)) != expected:
            raise ValueError(f"M5 smoke cfg {name} must equal {expected}")
    return dict(data), descriptors, cfg, manifest


def _subset_data(data: Mapping[str, object], selected_ids: list[str]) -> dict[str, object]:
    all_ids = np.asarray(data["series_id"]).astype(str)
    if all_ids.ndim != 1 or len(set(all_ids)) != len(all_ids):
        raise ValueError("population series_id must be a unique vector")
    positions = {series_id: index for index, series_id in enumerate(all_ids)}
    if len(selected_ids) != SMOKE_SERIES or len(set(selected_ids)) != SMOKE_SERIES:
        raise ValueError("the smoke sample must contain 200 unique series")
    unknown = [series_id for series_id in selected_ids if series_id not in positions]
    if unknown:
        raise ValueError("smoke sampler returned an unknown series_id")
    indices = np.asarray([positions[series_id] for series_id in selected_ids], dtype=np.int64)

    result: dict[str, object] = {}
    for key, value in data.items():
        if key == "series_id":
            result[key] = all_ids[indices]
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(all_ids):
            result[key] = value[indices]
        elif isinstance(value, pd.Series) and len(value) == len(all_ids):
            result[key] = value.iloc[indices].to_numpy()
        else:
            result[key] = value
    return result


def _single_origin_windows(windows: object, origin_index: int) -> object:
    n_series = int(windows.n_series)
    n_origins = int(windows.n_origins)
    if origin_index < 0 or origin_index >= n_origins:
        raise IndexError("origin_index is outside the window array")
    changes: dict[str, object] = {}
    per_row = {
        "history",
        "target",
        "occurrence",
        "target_mask",
        "gap",
        "gap_event_observed",
        "gap_censor_lower",
        "scale",
    }
    for field in fields(windows):
        name = field.name
        value = getattr(windows, name)
        if name in per_row:
            array = np.asarray(value)
            expected_rows = n_series * n_origins
            if array.shape[0] != expected_rows:
                raise ValueError(f"{name} does not align with series x origins")
            reshaped = array.reshape((n_series, n_origins) + array.shape[1:])
            changes[name] = reshaped[:, origin_index].copy()
    origin = int(np.asarray(windows.origins)[origin_index])
    valid_length = int(np.asarray(windows.valid_lengths)[origin_index])
    changes.update(
        origins=np.array([origin], dtype=np.int32),
        valid_lengths=np.array([valid_length], dtype=np.int32),
        split_start=origin,
        split_end=origin + valid_length,
    )
    return replace(windows, **changes)


def _combine_origin_predictions(
    predictions: list[Mapping[str, np.ndarray]], n_series: int
) -> dict[str, np.ndarray]:
    if not predictions:
        raise ValueError("at least one origin prediction is required")
    keys = set(predictions[0])
    if not keys or any(set(item) != keys for item in predictions[1:]):
        raise ValueError("prediction heads changed across origins")
    result: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        arrays = [np.asarray(item[key]) for item in predictions]
        if any(array.ndim < 1 or array.shape[0] != n_series for array in arrays):
            raise ValueError(f"prediction head {key} does not align with series")
        if any(array.shape[1:] != arrays[0].shape[1:] for array in arrays[1:]):
            raise ValueError(f"prediction head {key} changed shape across origins")
        result[key] = np.stack(arrays, axis=1).reshape(
            (n_series * len(arrays),) + arrays[0].shape[1:]
        )
    return result


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(function, device: torch.device):
    _sync_cuda(device)
    started = time.perf_counter()
    result = function()
    _sync_cuda(device)
    elapsed = float(time.perf_counter() - started)
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("wall-clock measurement must be finite and positive")
    return result, elapsed


def _train_origin_counts(m5_count: int) -> dict[str, int]:
    favorita = km_train.dense_origins(0, 1464, 28, 96)[::TRAIN_ORIGIN_STRIDE]
    return {"m5": int(m5_count), "favorita": int(len(favorita))}


def _storage_projection(
    actual_bytes: int,
    *,
    reference_series: int,
    reference_origins: int,
    series_counts: Mapping[str, int],
) -> dict[str, object]:
    unit = actual_bytes / float(reference_series * reference_origins)
    by_dataset = {
        dataset: int(round(unit * count * FULL_FORECAST_ORIGINS))
        for dataset, count in sorted(series_counts.items())
    }
    return {
        "estimated_bytes_per_series_origin": float(unit),
        "by_dataset": by_dataset,
        "total_bytes": int(sum(by_dataset.values())),
    }


def run_m5_smoke(
    population: Mapping[str, object],
    device: torch.device,
    full_series_counts: Mapping[str, int],
) -> dict[str, object]:
    """Run the 200-series M5 timing smoke and return in-memory artifacts.

    Point and Hurdle are fitted and inferred strictly sequentially with model
    seed zero.  The two-origin smoke is a runtime/integration check only: origin
    1745 becomes resolved memory at the first evaluation query, origin 1773.
    """

    data, descriptors, cfg, manifest = _validate_population(population)
    counts = _positive_counts(full_series_counts)
    torch_device = torch.device(device)
    selected_ids = stratified_smoke_ids(
        descriptors, n=SMOKE_SERIES, seed=SMOKE_SEED
    )
    selected_data = _subset_data(data, selected_ids)
    split = build_external_split(
        selected_data,
        cfg,
        train_origin_stride=TRAIN_ORIGIN_STRIDE,
        forecast_origins=SMOKE_ORIGINS.copy(),
    )
    np.testing.assert_array_equal(np.asarray(split.test.origins), SMOKE_ORIGINS)
    if int(split.test.n_series) != SMOKE_SERIES:
        raise ValueError("smoke split does not contain exactly 200 series")

    arm_specs = (("point", POINT_MODEL), ("hurdle", HURDLE_MODEL))
    combined_predictions: dict[str, dict[str, np.ndarray]] = {}
    training_report: dict[str, dict[str, object]] = {}
    inference_report: dict[str, dict[str, object]] = {}
    peak_by_arm: dict[str, int] = {}

    for arm, model_name in arm_specs:
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
        trained, train_wall = _timed_call(
            lambda model_name=model_name: train_one_on_split(
                model_name, split, cfg, MODEL_SEED, torch_device
            ),
            torch_device,
        )
        model = trained["model"]
        origin_predictions: list[Mapping[str, np.ndarray]] = []
        origin_seconds: list[float] = []
        for origin_index in range(split.test.n_origins):
            origin_windows = _single_origin_windows(split.test, origin_index)
            prediction, seconds = _timed_call(
                lambda windows=origin_windows: km_train.predict(
                    model, windows, torch_device
                ),
                torch_device,
            )
            origin_predictions.append(prediction)
            origin_seconds.append(seconds)
        combined_predictions[arm] = _combine_origin_predictions(
            origin_predictions, SMOKE_SERIES
        )
        canonical_seconds = float(trained["train_seconds"])
        if not np.isfinite(canonical_seconds) or canonical_seconds < 0.0:
            raise ValueError("canonical train_seconds must be finite and nonnegative")
        n_parameters = int(trained["n_parameters"])
        if n_parameters != 7056:
            raise ValueError(
                f"{model_name} parameter count drifted: expected 7056, "
                f"got {n_parameters}"
            )
        training_report[arm] = {
            "model_id": model_name,
            "model_seed": MODEL_SEED,
            "wall_seconds": train_wall,
            "canonical_seconds": canonical_seconds,
            "best_epoch": int(trained["best_epoch"]),
            "best_validation_mean_mse": float(
                trained["best_validation_mean_mse"]
            ),
            "n_parameters": n_parameters,
        }
        inference_report[arm] = {
            "origin_count": len(origin_seconds),
            "seconds_by_origin": {
                str(int(origin)): float(seconds)
                for origin, seconds in zip(
                    split.test.origins, origin_seconds, strict=True
                )
            },
            "mean_seconds_per_origin": float(np.mean(origin_seconds)),
        }
        if torch_device.type == "cuda":
            peak_by_arm[arm] = int(torch.cuda.max_memory_allocated(torch_device))
        del model, trained

    predictions = build_prediction_frame(
        selected_data,
        split.test,
        combined_predictions["point"],
        combined_predictions["hurdle"],
    )
    losses = normalized_loss_frame(
        predictions,
        selected_data,
        model_train_end=int(cfg.train_end),
        horizon=int(cfg.horizon),
    )

    cases = build_policy_cases(
        predictions,
        selected_data,
        split.test,
        model_train_end=int(cfg.train_end),
        horizon=int(cfg.horizon),
        lookback=int(cfg.lookback),
    )
    b4_result = evaluate_b4_cases(
        cases,
        warmup_origin=int(SMOKE_ORIGINS[0]),
        evaluation_origins=[int(SMOKE_ORIGINS[1])],
        horizon=int(cfg.horizon),
        eta=SMOKE_B4_ETA,
        half_life=SMOKE_B4_HALF_LIFE,
    )
    m1_result, retrieval_wall_seconds = _timed_call(
        lambda: evaluate_m1_cases(
            cases,
            warmup_origin=int(SMOKE_ORIGINS[0]),
            evaluation_origins=[int(SMOKE_ORIGINS[1])],
            horizon=int(cfg.horizon),
            lookback=int(cfg.lookback),
            eta=SMOKE_B4_ETA,
            half_life=SMOKE_B4_HALF_LIFE,
            k=RETRIEVAL_K,
            lambda_max=SMOKE_M1_LAMBDA_MAX,
        ),
        torch.device("cpu"),
    )
    expected_keys = set(np.asarray(selected_data["series_id"]).astype(str))
    for name, evaluated, weight_column in (
        ("B4", b4_result, "b4_hurdle_weight"),
        ("M1", m1_result, "m1_hurdle_weight"),
    ):
        if len(evaluated) != SMOKE_SERIES:
            raise ValueError(f"{name} smoke must score every selected series once")
        if evaluated.duplicated(["dataset_id", "series_id", "origin"]).any():
            raise ValueError(f"{name} smoke contains duplicate query cases")
        if set(evaluated["series_id"].astype(str)) != expected_keys:
            raise ValueError(f"{name} smoke query coverage is incomplete")
        weights = evaluated[weight_column].to_numpy(dtype=np.float64)
        if not bool(np.isfinite(weights).all() and ((0.0 <= weights) & (weights <= 1.0)).all()):
            raise ValueError(f"{name} smoke produced an invalid convex weight")
    neighbor_counts = m1_result["neighbor_count"].to_numpy(dtype=np.int64)
    if bool((neighbor_counts <= 0).any()):
        raise ValueError("retrieval returned an empty neighbor set")
    same_series_neighbor_count = sum(
        int(str(row.series_id) in set(map(str, row.neighbor_series_ids)))
        for row in m1_result.itertuples(index=False)
    )
    if same_series_neighbor_count:
        raise AssertionError("same-series memory case survived retrieval exclusion")
    resolved_origins = sorted(
        {
            int(origin)
            for origins in m1_result["resolved_origins"]
            for origin in origins
        }
    )
    if resolved_origins != [int(SMOKE_ORIGINS[0])]:
        raise AssertionError("unresolved evaluation losses leaked into smoke memory")
    constant_flags = np.vstack(
        [
            np.asarray(flags, dtype=bool)
            for flags in m1_result["constant_continuous_features"]
        ]
    )

    buffer = io.BytesIO()
    predictions.to_parquet(buffer, index=False)
    actual_parquet_bytes = len(buffer.getvalue())
    if actual_parquet_bytes <= 0:
        raise RuntimeError("prediction parquet serialization produced no bytes")

    smoke_train_seconds = {
        arm: float(training_report[arm]["canonical_seconds"])
        for arm, _ in arm_specs
    }
    smoke_inference_seconds = {
        arm: float(inference_report[arm]["mean_seconds_per_origin"])
        for arm, _ in arm_specs
    }
    train_origin_counts = _train_origin_counts(int(split.train.n_origins))
    forecast_origin_counts = {
        "m5": FULL_FORECAST_ORIGINS,
        "favorita": FULL_FORECAST_ORIGINS,
    }
    full_runtime = project_full_seed0_runtime(
        smoke_train_seconds=smoke_train_seconds,
        smoke_inference_seconds_per_origin=smoke_inference_seconds,
        smoke_n_series=SMOKE_SERIES,
        full_series=counts,
        train_origins=train_origin_counts,
        forecast_origins=forecast_origin_counts,
    )
    fallback_counts = {
        "m5": FALLBACK_SERIES_PER_DATASET,
        "favorita": FALLBACK_SERIES_PER_DATASET,
    }
    fallback_runtime = project_full_seed0_runtime(
        smoke_train_seconds=smoke_train_seconds,
        smoke_inference_seconds_per_origin=smoke_inference_seconds,
        smoke_n_series=SMOKE_SERIES,
        full_series=fallback_counts,
        train_origins=train_origin_counts,
        forecast_origins=forecast_origin_counts,
    )
    full_retrieval_runtime = project_retrieval_runtime(
        smoke_seconds=retrieval_wall_seconds,
        smoke_n_series=SMOKE_SERIES,
        full_series=counts,
        evaluation_origins=FULL_EVALUATION_ORIGINS,
        retrieval_passes_per_dataset=RETRIEVAL_PASSES_PER_DATASET,
    )
    fallback_retrieval_runtime = project_retrieval_runtime(
        smoke_seconds=retrieval_wall_seconds,
        smoke_n_series=SMOKE_SERIES,
        full_series=fallback_counts,
        evaluation_origins=FULL_EVALUATION_ORIGINS,
        retrieval_passes_per_dataset=RETRIEVAL_PASSES_PER_DATASET,
    )
    full_storage = _storage_projection(
        actual_parquet_bytes,
        reference_series=SMOKE_SERIES,
        reference_origins=len(SMOKE_ORIGINS),
        series_counts=counts,
    )
    fallback_storage = _storage_projection(
        actual_parquet_bytes,
        reference_series=SMOKE_SERIES,
        reference_origins=len(SMOKE_ORIGINS),
        series_counts=fallback_counts,
    )
    exceeded = float(full_runtime["gpu_hours"]) > RUNTIME_GATE_GPU_HOURS
    report = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "M5_200_SERIES_SMOKE",
        "scientific_result": False,
        "dataset": "m5",
        "device": str(torch_device),
        "selected_series_count": SMOKE_SERIES,
        "selected_series_ids": selected_ids,
        "sample_seed": SMOKE_SEED,
        "model_seed": MODEL_SEED,
        "forecast_origins": SMOKE_ORIGINS.tolist(),
        "train_origin_stride": TRAIN_ORIGIN_STRIDE,
        "train_origin_counts": train_origin_counts,
        "population_manifest": _json_safe(manifest),
        "training": training_report,
        "inference": inference_report,
        "cuda_peak_memory_bytes": int(max(peak_by_arm.values(), default=0)),
        "cuda_peak_memory_by_arm_bytes": peak_by_arm,
        "retrieval_check": {
            "memory_origins": resolved_origins,
            "query_origin": int(SMOKE_ORIGINS[1]),
            "memory_case_count": int(
                (cases["origin"] == int(SMOKE_ORIGINS[0])).sum()
            )
            if "origin" in cases
            else SMOKE_SERIES,
            "query_count": SMOKE_SERIES,
            "k": RETRIEVAL_K,
            "min_neighbor_count": int(min(neighbor_counts)),
            "max_neighbor_count": int(max(neighbor_counts)),
            "queries_with_same_series_neighbor": same_series_neighbor_count,
            "b4_updated_query_count": int(len(b4_result)),
            "m1_updated_query_count": int(len(m1_result)),
            "measured_wall_seconds": retrieval_wall_seconds,
            "pipeline_validation_hyperparameters": {
                "eta": SMOKE_B4_ETA,
                "half_life_origins": SMOKE_B4_HALF_LIFE,
                "lambda_max": SMOKE_M1_LAMBDA_MAX,
                "scientific_selection": False,
            },
            "constant_continuous_features_seen": constant_flags.any(axis=0).tolist(),
        },
        "serialization": {
            "format": "parquet",
            "actual_series": SMOKE_SERIES,
            "actual_origins": len(SMOKE_ORIGINS),
            "actual_rows": int(len(predictions)),
            "actual_parquet_bytes": actual_parquet_bytes,
            "full_prediction_storage": full_storage,
            "prediction_storage_2000_per_dataset": fallback_storage,
        },
        "runtime_projection_full_seed0": full_runtime,
        "runtime_projection_2000_per_dataset": fallback_runtime,
        "retrieval_runtime_projection_full_seed0": full_retrieval_runtime,
        "retrieval_runtime_projection_2000_per_dataset": fallback_retrieval_runtime,
        "runtime_projection_basis": {
            "training_seconds": "canonical train_seconds",
            "reason": (
                "outer wall_seconds includes the canonical function's own "
                "two-origin prediction; inference is projected separately"
            ),
            "inference_seconds": "separate synchronized predict calls per origin",
            "retrieval_seconds": (
                "measured exact first-query-origin B4+M1 policy evaluation; "
                "reported separately as CPU wall-time because the preregistered "
                "six-hour gate is explicitly GPU runtime"
            ),
        },
        "runtime_gate": {
            "threshold_gpu_hours": RUNTIME_GATE_GPU_HOURS,
            "projected_gpu_hours": float(full_runtime["gpu_hours"]),
            "exceeded": exceeded,
            "action": "STOP_FOR_APPROVAL" if exceeded else "CONTINUE_FULL_SEED0",
        },
    }
    safe_report = _json_safe(report)
    return {"report": safe_report, "predictions": predictions, "losses": losses}


__all__ = ["run_m5_smoke"]
