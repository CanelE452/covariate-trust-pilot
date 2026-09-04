"""Data adapter contracts for PROB-HEAD-STRUCTURE-FULL-v1."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .integrity import verify_source_manifest


INTEGER_TOLERANCE = 1e-6
RMS_EPSILON = 1e-8
REAL_SAMPLING_SEED = 2026090521
REAL_SERIES_PER_RUNTIME_TIER = {
    "FULL": 4_000,
    "COMPACT": 2_000,
    "MINIMAL-COMPLETE": 1_000,
}
COUNT_LIKELIHOOD_INDEX_AMBIGUITY_TOKEN = (
    "COUNT_LIKELIHOOD_INDEX_AMBIGUITY_HARD_STOP"
)
REAL_DATASET_PRIORITY = ("m5", "auto", "carparts", "raf", "online_retail")
M5_CANONICAL_ADAPTER_FUNCTION = (
    "experiments.prob_head_structure_full_v1.data._m5_panel_from_verified_sources"
)
M5_UPSTREAM_REUSED_SOURCE = (
    "experiments.external_validity_screen.confirmatory_h2.m5_full; "
    "experiments.external_validity_screen.screen.m5_availability semantics "
    "reimplemented directly from verified calendar/sell_prices sources"
)
ONLINE_RETAIL_ADAPTER_FUNCTION = (
    "experiments.prob_head_structure_full_v1.data.aggregate_online_retail_tweediegp_frame"
)
ONLINE_RETAIL_UPSTREAM_REUSED_SOURCE = (
    "StefanoDamato/TweedieGP@2567d1322c8cc65f19df4f2d1774c610b167fb66 "
    "UCI-352 aggregation"
)
M5_EXPECTED_SOURCE_SHA256 = {
    "sales_train_evaluation.csv": "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0",
    "calendar.csv": "d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5",
    "sell_prices.csv": "9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef",
}
M5_STAGE_A_SNAPSHOT_NAME = "series.parquet"
M5_STAGE_A_EXPECTED_SHA256 = (
    "aa8d96ecd6ed6eaa91274087b4b90880b5da4ec3954962d94d57e37947f13aba"
)
M5_STAGE_A_EXPECTED_N_SERIES = 1_200
M5_STAGE_A_CANONICAL_ID_SHA256 = (
    "c73bb7506b56c76b7fbebdf67a20e0e08f48894976da7a05adfad749bba9334e"
)
M5_STAGE_A_ADAPTER_FUNCTION = (
    "experiments.prob_head_structure_full_v1.data.load_m5_stage_a_exclusion"
)
ONLINE_RETAIL_EXPECTED_SHA256 = (
    "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
)
CANONICAL_COUNT_PRIMARY_PANEL_CONTRACTS = {
    "m5": {
        "panel_shape": [29_290, 1_941],
        "panel_binding_sha256": "3844515e707de5fff22c1df62ba918f1cab2bf901eee5ff72d33b5cc2acbb2be",
        "ordered_series_id_sha256": "c4d8f4c4be936c875abdd6f67e1fd18f87b18caa717f6ac9cc1ec31e5535258e",
        "adapter_function": M5_CANONICAL_ADAPTER_FUNCTION,
    },
    "online_retail": {
        "panel_shape": [2_036, 374],
        "panel_binding_sha256": "ff41f533c572d6f2fd603dfdac7339374a05b2cd531760ec51b9edc9ec7f4348",
        "ordered_series_id_sha256": "96443093e11320948afd5bb57540ad48fdfb4549cb2b16a383ec83ab6ee4e2a0",
        "adapter_function": ONLINE_RETAIL_ADAPTER_FUNCTION,
    },
}
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SNAPSHOT_ROOT = (
    REPO_ROOT / "runs" / "prob_head_structure_full_v1" / "source_snapshots"
)
WINDOWS_REPARSE_POINT_ATTRIBUTE = 0x400


@dataclass(frozen=True)
class RealSplit:
    train: tuple[int, int]
    validation: tuple[int, int]
    warmup: tuple[int, int]
    origins: tuple[int, ...]
    horizon: int = 28
    lookback: int = 96


@dataclass(frozen=True)
class WindowRequest:
    """A role-bound, hash-sealed request for target-bearing history windows."""

    dataset_id: str
    role: str
    origins: tuple[int, ...]
    split: RealSplit
    panel_length: int
    dataset_audit_sha256: str
    sample_manifest_sha256: str
    sampled_panel_binding_sha256: str
    sampling_runtime_tier: str
    sampling_seed: int
    confirmatory_eligible: bool
    split_contract_sha256: str
    request_sha256: str


@dataclass(frozen=True)
class WindowBatch:
    history: np.ndarray
    target: np.ndarray
    occurrence: np.ndarray
    target_mask: np.ndarray
    scale: np.ndarray
    key_frame: pd.DataFrame
    origins: np.ndarray
    series_id: np.ndarray
    split_name: str
    dataset_audit_sha256: str
    sample_manifest_sha256: str
    sampled_panel_binding_sha256: str
    confirmatory_eligible: bool
    split_contract_sha256: str
    request_sha256: str


REAL_SPLITS = {
    "m5": RealSplit(
        train=(0, 1717),
        validation=(1717, 1745),
        warmup=(1745, 1773),
        origins=(1773, 1801, 1829, 1857, 1885, 1913),
    ),
    "online_retail": RealSplit(
        train=(0, 150),
        validation=(150, 178),
        warmup=(178, 206),
        origins=(206, 234, 262, 290, 318, 346),
    ),
}


_TRAIN_WINDOW_ROLES = frozenset(
    {
        "model_train",
        "teacher_train",
        "student_train",
        "inner_train",
        "inner_validation",
    }
)
_VALIDATION_WINDOW_ROLES = frozenset(
    {
        "validation",
        "teacher_validation",
        "student_validation",
        "pool_validation",
        "router_validation",
        "sensor_validation",
    }
)
_WARMUP_WINDOW_ROLES = frozenset({"warmup"})
_EVALUATION_WINDOW_ROLES = frozenset({"evaluation", "outer_evaluation"})
_WINDOW_ROLES = (
    _TRAIN_WINDOW_ROLES
    | _VALIDATION_WINDOW_ROLES
    | _WARMUP_WINDOW_ROLES
    | _EVALUATION_WINDOW_ROLES
)


ONLINE_RETAIL_EXPECTED = {
    "sheet_name": "Year 2010-2011",
    "raw_rows": 541_910,
    "date_start": "2010-12-01",
    "date_end": "2011-12-09",
    "n_series": 2_036,
    "n_dates": 374,
    "source": "TweedieGP UCI-352 Description x daily Quantity aggregation",
}


_ROUNDING_TRANSFORM_KEYS = {
    "transform",
    "transforms",
    "operation",
    "operations",
    "cast",
    "casts",
    "preprocessing",
    "target_transform",
    "target_transforms",
    "target_cast",
}
_ROUNDING_TRANSFORM_TOKENS = {
    "rint",
    "np.rint",
    "numpy.rint",
    "round",
    "rounded",
    "np.round",
    "numpy.round",
    "integer_cast",
    "cast_int",
    "astype(int)",
    "astype('int')",
    'astype("int")',
}


def _tokens_from_transform_text(text: str) -> set[str]:
    normalized = str(text).strip().lower()
    if normalized in _ROUNDING_TRANSFORM_TOKENS:
        return {normalized}
    tokens: set[str] = set()
    for token in _ROUNDING_TRANSFORM_TOKENS:
        pattern = rf"(?<![A-Za-z0-9_.]){re.escape(token)}(?![A-Za-z0-9_.])"
        if re.search(pattern, normalized):
            tokens.add(token)
    return tokens


def _rounding_transform_tokens(provenance: object, *, direct_text: bool = True) -> set[str]:
    if provenance is None:
        return set()
    if isinstance(provenance, Mapping):
        tokens: set[str] = set()
        for key, value in provenance.items():
            key_text = str(key).lower()
            if key_text in _ROUNDING_TRANSFORM_KEYS:
                tokens.update(_rounding_transform_tokens(value, direct_text=True))
            elif isinstance(value, Mapping):
                tokens.update(_rounding_transform_tokens(value, direct_text=False))
        return tokens
    if isinstance(provenance, (list, tuple, set, frozenset)):
        tokens = set()
        for value in provenance:
            tokens.update(_rounding_transform_tokens(value, direct_text=direct_text))
        return tokens
    if direct_text:
        return _tokens_from_transform_text(str(provenance))
    return set()


def audit_target_support(
    values: object,
    *,
    dataset_id: str,
    provenance: object | None = None,
    integer_tolerance: float = INTEGER_TOLERANCE,
) -> dict[str, object]:
    """Record count-primary support diagnostics without mutating or rounding values."""
    y = np.asarray(values, dtype=np.float64)
    flat = y.reshape(-1)
    finite = np.isfinite(flat)
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    nonfinite_count = int((~finite).sum())
    finite_values = flat[finite]
    negative_count = int((finite_values < 0.0).sum())
    rounded_values = np.rint(finite_values)
    integer_distance = np.abs(finite_values - rounded_values)
    noninteger = (
        integer_distance > float(integer_tolerance)
    )
    noninteger_count = int(noninteger.sum())
    exact_integer = finite_values == rounded_values
    within_tolerance = ~noninteger
    nonexact_within_tolerance = within_tolerance & ~exact_integer
    exact_integer_count = int(exact_integer.sum())
    nonexact_within_tolerance_count = int(nonexact_within_tolerance.sum())
    zero = finite_values == 0.0
    positive = finite_values > 0.0
    positive_values = finite_values[positive]
    rounding_tokens = _rounding_transform_tokens(provenance)
    rounded_provenance = bool(rounding_tokens)
    reasons = []
    if flat.size == 0:
        reasons.append("empty_count_support_scope")
    if nonfinite_count:
        reasons.append("nonfinite_target_values")
    if negative_count:
        reasons.append("negative_target_values")
    if noninteger_count:
        reasons.append("noninteger_count_values")
    if nonexact_within_tolerance_count:
        reasons.append(COUNT_LIKELIHOOD_INDEX_AMBIGUITY_TOKEN)
    if rounded_provenance:
        reasons.append("rounded_or_rint_provenance")
    valid_support = not (
        flat.size == 0 or nonfinite_count or negative_count or noninteger_count
    )
    count_likelihood_index_exact = nonexact_within_tolerance_count == 0
    count_primary_eligible = (
        valid_support and count_likelihood_index_exact and not rounded_provenance
    )
    positive_variance = (
        float(positive_values.var(ddof=1)) if positive_values.size > 1 else None
    )
    return {
        "dataset_id": str(dataset_id),
        "support": (
            "nonnegative_integer_count"
            if valid_support
            else ("empty_count_support_scope" if flat.size == 0 else "invalid_count_support")
        ),
        "status": "PASS" if count_primary_eligible else "COUNT_SUPPORT_FAIL",
        "integer_tolerance": float(integer_tolerance),
        "n_observations": int(flat.size),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_count": nonfinite_count,
        "negative_count": negative_count,
        "noninteger_count": noninteger_count,
        "tolerance_integer_count": int(within_tolerance.sum()),
        "exact_integer_count": exact_integer_count,
        "nonexact_within_tolerance_count": nonexact_within_tolerance_count,
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "zero_count": int(zero.sum()),
        "positive_count": int(positive.sum()),
        "zero_ratio": float(zero.sum() / flat.size) if flat.size else None,
        "positive_mean": float(positive_values.mean()) if positive_values.size else None,
        "positive_variance": positive_variance,
        "rounded_provenance_detected": bool(rounded_provenance),
        "provenance_transform_tokens": tuple(sorted(rounding_tokens)),
        "count_likelihood_index_exact": bool(count_likelihood_index_exact),
        "hard_stop_token": (
            COUNT_LIKELIHOOD_INDEX_AMBIGUITY_TOKEN
            if nonexact_within_tolerance_count
            else None
        ),
        "count_primary_eligible": bool(count_primary_eligible),
        "confirmatory_eligible": bool(count_primary_eligible),
        "ineligible_reason": ";".join(reasons) if reasons else None,
    }


def enforce_target_support(
    audit_or_values: object, *, dataset_id: str | None = None
) -> dict[str, object]:
    """Raise only after an audit row has recorded invalid support diagnostics."""
    audit = (
        dict(audit_or_values)
        if isinstance(audit_or_values, Mapping)
        else audit_target_support(audit_or_values, dataset_id=dataset_id or "dataset")
    )
    if not bool(audit.get("count_primary_eligible", False)):
        reason = audit.get("ineligible_reason") or "unknown_count_support_failure"
        raise ValueError(f"count-primary target support failed: {reason}")
    return audit


def _assert_float32_exact_roundtrip(values: object, *, label: str) -> None:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    if raw.size == 0:
        return
    roundtrip = raw.astype(np.float32).astype(np.float64)
    changed = raw != roundtrip
    if bool(np.any(changed)):
        raise ValueError(
            f"{label} values fail exact float32 round-trip for {int(changed.sum())} targets"
        )


def audit_model_train_target_support(
    values: object,
    *,
    split: RealSplit,
    dataset_id: str,
    available_from: object | None = None,
    series_ids: Sequence[object] | None = None,
    provenance: object | None = None,
    integer_tolerance: float = INTEGER_TOLERANCE,
) -> dict[str, object]:
    """Audit count support on model-train targets only."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("values must be a two-dimensional series-by-time array")
    train_start, train_end = int(split.train[0]), int(split.train[1])
    if train_start < 0 or train_end <= train_start or train_end > y.shape[1]:
        raise ValueError("model-train support slice is outside the panel")
    availability = validate_available_from(
        np.zeros(y.shape[0], dtype=np.int32) if available_from is None else available_from,
        n_series=y.shape[0],
        length=y.shape[1],
    )
    ids = (
        np.asarray([str(index) for index in range(y.shape[0])])
        if series_ids is None
        else _string_series_ids(series_ids)
    )
    if ids.shape != (y.shape[0],):
        raise ValueError("series_ids must contain one ID per series")
    included_segments = []
    excluded_ids = []
    for index in range(y.shape[0]):
        start = max(train_start, int(availability[index]))
        if start >= train_end:
            excluded_ids.append(str(ids[index]))
            continue
        included_segments.append(y[index, start:train_end])
    audited_values = (
        np.concatenate(included_segments)
        if included_segments
        else np.asarray([], dtype=np.float64)
    )
    audit = audit_target_support(
        audited_values,
        dataset_id=dataset_id,
        provenance=provenance,
        integer_tolerance=integer_tolerance,
    )
    audit.update(
        {
            "target_scope": "model_train",
            "time_start": train_start,
            "time_end": train_end,
            "full_time_length": int(y.shape[1]),
            "support_slice_rule": "half_open_available_train_interval_only",
            "availability_excluded_count": int(len(excluded_ids)),
            "availability_excluded_series_ids": tuple(excluded_ids),
        }
    )
    return audit


def _availability_model_train_values(
    values: object, *, split: RealSplit, available_from: object | None = None
) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("values must be a two-dimensional series-by-time array")
    train_start, train_end = int(split.train[0]), int(split.train[1])
    availability = validate_available_from(
        np.zeros(y.shape[0], dtype=np.int32) if available_from is None else available_from,
        n_series=y.shape[0],
        length=y.shape[1],
    )
    segments = [
        y[index, max(train_start, int(availability[index])) : train_end]
        for index in range(y.shape[0])
        if max(train_start, int(availability[index])) < train_end
    ]
    return np.concatenate(segments) if segments else np.asarray([], dtype=np.float64)


def train_rms_scale(
    values: object,
    *,
    train_end: int,
    available_from: object | None = None,
) -> np.ndarray:
    """Per-series RMS scale fit only on availability-observed train values."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("values must be a two-dimensional series-by-time array")
    end = int(train_end)
    if not 0 < end <= y.shape[1]:
        raise ValueError("train_end must fall within the time axis")
    availability = validate_available_from(
        np.zeros(y.shape[0], dtype=np.int32) if available_from is None else available_from,
        n_series=y.shape[0],
        length=y.shape[1],
    )
    scales = np.empty(y.shape[0], dtype=np.float64)
    for index, start in enumerate(availability):
        observed = y[index, int(start) : end]
        if observed.size == 0:
            raise ValueError("series has no availability-observed model-train values for scaling")
        if not np.isfinite(observed).all():
            raise ValueError("train values must be finite before scaling")
        scales[index] = np.sqrt(np.mean(observed * observed) + RMS_EPSILON)
    return scales.astype(np.float32)


def validate_disjoint_splits(splits: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, int]]:
    """Validate half-open temporal splits with no overlap."""
    normalized: dict[str, tuple[int, int]] = {}
    intervals = []
    for name, bounds in splits.items():
        if len(bounds) != 2:
            raise ValueError(f"{name} split must have start and end")
        start, end = int(bounds[0]), int(bounds[1])
        if start < 0 or end <= start:
            raise ValueError(f"{name} split has invalid bounds")
        normalized[str(name)] = (start, end)
        intervals.append((start, end, str(name)))
    for index, (start, end, name) in enumerate(intervals):
        for other_start, other_end, other_name in intervals[index + 1 :]:
            if start < other_end and other_start < end:
                raise ValueError(f"split overlap between {name} and {other_name}")
    return normalized


def validate_real_split(split: RealSplit, *, length: int) -> dict[str, object]:
    """Validate frozen real-data geometry including every origin+horizon key."""
    total_length = int(length)
    if total_length <= 0:
        raise ValueError("real split length must be positive")
    normalized = validate_disjoint_splits(
        {
            "train": split.train,
            "validation": split.validation,
            "warmup": split.warmup,
        }
    )
    if normalized["train"][0] != 0:
        raise ValueError("real split train interval must start at 0")
    if normalized["train"][1] != normalized["validation"][0]:
        raise ValueError("train and validation splits must be contiguous")
    if normalized["validation"][1] != normalized["warmup"][0]:
        raise ValueError("validation and warmup splits must be contiguous")
    if normalized["validation"][1] - normalized["validation"][0] != int(split.horizon):
        raise ValueError("validation split width must equal the forecast horizon")
    if normalized["warmup"][1] - normalized["warmup"][0] != int(split.horizon):
        raise ValueError("warmup split width must equal the forecast horizon")
    if normalized["warmup"][1] > total_length:
        raise ValueError("warmup split exceeds series length")
    origins = _integer_vector("origins", np.asarray(split.origins, dtype=np.int32))
    if origins.size != 6:
        raise ValueError("real split origins must contain exactly six origins")
    if np.any(np.diff(origins) <= 0):
        raise ValueError("origins must be strictly increasing")
    if np.any(np.diff(origins) != int(split.horizon)):
        raise ValueError("real split origins must be horizon-spaced")
    if np.any(origins < int(split.lookback)):
        raise ValueError("origin lacks sufficient lookback history")
    if int(origins[0]) != normalized["warmup"][1]:
        raise ValueError("warmup split end must equal the first evaluation origin")
    if np.any(origins < normalized["warmup"][1]):
        raise ValueError("origin starts before the warmup split ends")
    if np.any(origins + int(split.horizon) > total_length):
        raise ValueError("origin plus horizon exceeds series length")
    if int(origins[-1]) + int(split.horizon) != total_length:
        raise ValueError("final origin plus horizon must equal series length")
    return {
        "status": "PASS",
        "length": total_length,
        "train": normalized["train"],
        "validation": normalized["validation"],
        "warmup": normalized["warmup"],
        "origins": tuple(int(origin) for origin in origins.tolist()),
        "horizon": int(split.horizon),
        "lookback": int(split.lookback),
    }


def _split_contract_payload(
    *, dataset_id: str, split: RealSplit, panel_length: int
) -> dict[str, object]:
    return {
        "dataset_id": str(dataset_id),
        "panel_length": int(panel_length),
        "train": [int(value) for value in split.train],
        "validation": [int(value) for value in split.validation],
        "warmup": [int(value) for value in split.warmup],
        "origins": [int(value) for value in split.origins],
        "horizon": int(split.horizon),
        "lookback": int(split.lookback),
    }


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_request_payload(
    *,
    dataset_id: str,
    role: str,
    origins: Sequence[int],
    dataset_audit_sha256: str,
    sample_manifest_sha256: str,
    sampled_panel_binding_sha256: str,
    sampling_runtime_tier: str,
    sampling_seed: int,
    confirmatory_eligible: bool,
    split_contract_sha256: str,
) -> dict[str, object]:
    return {
        "dataset_id": str(dataset_id),
        "role": str(role),
        "origins": [int(origin) for origin in origins],
        "dataset_audit_sha256": str(dataset_audit_sha256),
        "sample_manifest_sha256": str(sample_manifest_sha256),
        "sampled_panel_binding_sha256": str(sampled_panel_binding_sha256),
        "sampling_runtime_tier": str(sampling_runtime_tier),
        "sampling_seed": int(sampling_seed),
        "confirmatory_eligible": bool(confirmatory_eligible),
        "split_contract_sha256": str(split_contract_sha256),
    }


def build_window_request(
    *,
    dataset_id: str,
    split: RealSplit,
    panel_length: int,
    role: str,
    origins: object,
    panel: Mapping[str, object],
    dataset_audit: Mapping[str, object],
    sample_manifest: Mapping[str, object],
) -> WindowRequest:
    """Bind requested forecast origins to one immutable split role.

    Training roles may use any increasing rolling-origin subset whose entire
    target horizon remains in ``model_train``.  Validation and warmup roles are
    fixed to their single frozen horizon; evaluation roles require the complete
    frozen outer-origin sequence.
    """
    normalized_dataset_id = str(dataset_id).strip()
    if not normalized_dataset_id:
        raise ValueError("dataset_id must be nonempty")
    if not isinstance(split, RealSplit):
        raise TypeError("split must be a RealSplit")
    total_length = int(panel_length)
    validate_real_split(split, length=total_length)
    verified_audit = _verify_sealed_dataset_audit(dataset_audit)
    if (
        verified_audit.get("audit_type") != "COUNT_PRIMARY_DATASET_AUDIT"
        or verified_audit.get("dataset_id") != normalized_dataset_id
        or verified_audit.get("panel_shape") is None
        or [int(value) for value in verified_audit["panel_shape"]][1]
        != total_length
    ):
        raise ValueError("window request requires the matching sealed count-primary dataset audit")
    verified_sample = verify_train_only_sample_manifest(
        panel,
        dataset_audit=verified_audit,
        sample_manifest=sample_manifest,
    )
    audit_split = verified_audit.get("split_validation")
    expected_split = validate_real_split(split, length=total_length)
    if not isinstance(audit_split, Mapping) or any(
        tuple(audit_split.get(name, ())) != tuple(expected_split[name])
        for name in ("train", "validation", "warmup", "origins")
    ) or any(
        int(audit_split.get(name, -1)) != int(expected_split[name])
        for name in ("length", "horizon", "lookback")
    ):
        raise ValueError("window split differs from the sealed dataset audit")
    frozen_split = REAL_SPLITS.get(normalized_dataset_id)
    if frozen_split is not None and split != frozen_split:
        raise ValueError(f"{normalized_dataset_id} split differs from the frozen contract")

    normalized_role = str(role)
    if normalized_role not in _WINDOW_ROLES:
        raise ValueError(f"unknown window role: {normalized_role}")
    integer_origins = _integer_vector("origins", origins)
    if np.any(np.diff(integer_origins) <= 0):
        raise ValueError("window origins must be strictly increasing")
    origin_tuple = tuple(int(origin) for origin in integer_origins.tolist())

    if normalized_role in _TRAIN_WINDOW_ROLES:
        permitted = split.train
        for origin in origin_tuple:
            if origin < max(int(permitted[0]), int(split.lookback)) or origin + int(
                split.horizon
            ) > int(permitted[1]):
                raise ValueError(
                    f"{normalized_role} target interval violation: origin {origin} "
                    "crosses the permitted target interval"
                )
    elif normalized_role in _VALIDATION_WINDOW_ROLES:
        expected = (int(split.validation[0]),)
        if origin_tuple != expected:
            raise ValueError(
                f"{normalized_role} target interval must equal the frozen validation horizon"
            )
    elif normalized_role in _WARMUP_WINDOW_ROLES:
        expected = (int(split.warmup[0]),)
        if origin_tuple != expected:
            raise ValueError("warmup target interval must equal the frozen warmup horizon")
    else:
        expected = tuple(int(origin) for origin in split.origins)
        if origin_tuple != expected:
            raise ValueError(
                f"{normalized_role} target interval must equal all frozen evaluation origins"
            )

    split_payload = _split_contract_payload(
        dataset_id=normalized_dataset_id,
        split=split,
        panel_length=total_length,
    )
    split_sha256 = _canonical_sha256(split_payload)
    request_sha256 = _canonical_sha256(
        _window_request_payload(
            dataset_id=normalized_dataset_id,
            role=normalized_role,
            origins=origin_tuple,
            dataset_audit_sha256=str(verified_audit["audit_sha256"]),
            sample_manifest_sha256=str(verified_sample["manifest_sha256"]),
            sampled_panel_binding_sha256=str(
                verified_sample["sampled_panel_binding_sha256"]
            ),
            sampling_runtime_tier=str(verified_sample["runtime_tier"]),
            sampling_seed=int(verified_sample["seed"]),
            confirmatory_eligible=bool(
                verified_audit.get("confirmatory_eligible") is True
                and verified_audit.get("canonical_source_attested") is True
            ),
            split_contract_sha256=split_sha256,
        )
    )
    return WindowRequest(
        dataset_id=normalized_dataset_id,
        role=normalized_role,
        origins=origin_tuple,
        split=split,
        panel_length=total_length,
        dataset_audit_sha256=str(verified_audit["audit_sha256"]),
        sample_manifest_sha256=str(verified_sample["manifest_sha256"]),
        sampled_panel_binding_sha256=str(
            verified_sample["sampled_panel_binding_sha256"]
        ),
        sampling_runtime_tier=str(verified_sample["runtime_tier"]),
        sampling_seed=int(verified_sample["seed"]),
        confirmatory_eligible=bool(
            verified_audit.get("confirmatory_eligible") is True
            and verified_audit.get("canonical_source_attested") is True
        ),
        split_contract_sha256=split_sha256,
        request_sha256=request_sha256,
    )


def _validate_window_request(
    request: WindowRequest,
    panel: Mapping[str, object],
    dataset_audit: Mapping[str, object],
) -> WindowRequest:
    if not isinstance(request, WindowRequest):
        raise TypeError("request must be a WindowRequest")
    expected_split_sha256 = _canonical_sha256(
        _split_contract_payload(
            dataset_id=request.dataset_id,
            split=request.split,
            panel_length=request.panel_length,
        )
    )
    expected_request_sha256 = _canonical_sha256(
        _window_request_payload(
            dataset_id=request.dataset_id,
            role=request.role,
            origins=request.origins,
            dataset_audit_sha256=request.dataset_audit_sha256,
            sample_manifest_sha256=request.sample_manifest_sha256,
            sampled_panel_binding_sha256=request.sampled_panel_binding_sha256,
            sampling_runtime_tier=request.sampling_runtime_tier,
            sampling_seed=request.sampling_seed,
            confirmatory_eligible=request.confirmatory_eligible,
            split_contract_sha256=expected_split_sha256,
        )
    )
    if request.split_contract_sha256 != expected_split_sha256:
        raise ValueError("window split contract hash mismatch")
    if request.request_sha256 != expected_request_sha256:
        raise ValueError("window request hash mismatch")
    verified_audit = _verify_sealed_dataset_audit(dataset_audit)
    if verified_audit.get("audit_sha256") != request.dataset_audit_sha256:
        raise ValueError("window request dataset audit hash mismatch")
    reconstructed_sample = seal_train_only_sample_manifest(
        panel,
        dataset_audit=verified_audit,
        runtime_tier=request.sampling_runtime_tier,
        seed=request.sampling_seed,
    )
    if reconstructed_sample["manifest_sha256"] != request.sample_manifest_sha256:
        raise ValueError("window request sample manifest hash mismatch")
    if (
        reconstructed_sample["sampled_panel_binding_sha256"]
        != request.sampled_panel_binding_sha256
    ):
        raise ValueError("window request sampled panel binding mismatch")
    rebuilt = build_window_request(
        dataset_id=request.dataset_id,
        split=request.split,
        panel_length=request.panel_length,
        role=request.role,
        origins=request.origins,
        panel=panel,
        dataset_audit=verified_audit,
        sample_manifest=reconstructed_sample,
    )
    if rebuilt != request:
        raise ValueError("window request does not match its canonical reconstruction")
    return request


def _string_series_ids(values: object) -> np.ndarray:
    ids = np.asarray(values).astype(str)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("series_ids must be a nonempty vector")
    if pd.Series(ids).duplicated().any():
        raise ValueError("duplicate series IDs are not allowed")
    return ids


def _integer_vector(name: str, values: object) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    numeric = raw.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integer values")
    return numeric.astype(np.int32)


def validate_available_from(
    available_from: object, *, n_series: int, length: int
) -> np.ndarray:
    """Validate M5-style first-available index, one value per series."""
    raw = np.asarray(available_from)
    if raw.shape != (int(n_series),):
        raise ValueError("available_from must contain one value per series")
    numeric = raw.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("available_from must contain finite integer indices")
    indices = numeric.astype(np.int32)
    if np.any(indices < 0) or np.any(indices > int(length)):
        raise ValueError("available_from values must satisfy 0 <= available_from <= T")
    return indices


def origin_key_frame(
    *,
    dataset_id: str,
    series_ids: Sequence[object],
    origins: object,
    horizon: int,
) -> pd.DataFrame:
    """Build the exact prediction keys shared by every probabilistic head."""
    ids = _string_series_ids(series_ids)
    integer_origins = _integer_vector("origins", origins)
    if pd.Series(integer_origins).duplicated().any():
        raise ValueError("duplicate origins are not allowed")
    forecast_horizon = int(horizon)
    if forecast_horizon <= 0:
        raise ValueError("horizon must be positive")
    rows = [
        {
            "dataset_id": str(dataset_id),
            "series_id": str(series_id),
            "origin": int(origin),
            "step": int(step),
        }
        for origin in integer_origins
        for series_id in ids
        for step in range(forecast_horizon)
    ]
    return pd.DataFrame(rows, columns=["dataset_id", "series_id", "origin", "step"])


def assert_same_prediction_keys(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Require byte-identical prediction-key order across heads."""
    key_columns = ["dataset_id", "series_id", "origin", "step"]
    if list(left.columns[:4]) != key_columns or list(right.columns[:4]) != key_columns:
        raise ValueError("prediction keys must start with dataset_id, series_id, origin, step")
    if left[key_columns].duplicated().any() or right[key_columns].duplicated().any():
        raise ValueError("duplicate prediction keys are not allowed")
    if len(left) != len(right) or not left[key_columns].equals(right[key_columns]):
        raise ValueError("prediction keys differ across heads")


def make_history_windows(
    panel: Mapping[str, object],
    *,
    request: WindowRequest,
    dataset_audit: Mapping[str, object],
) -> WindowBatch:
    """Build windows from a role-bound request whose integrity is rechecked."""
    request = _validate_window_request(request, panel, dataset_audit)
    verified_audit = _verify_sealed_dataset_audit(dataset_audit)
    raw_y_source = panel["raw_y"] if "raw_y" in panel else panel["y"]
    raw_y = np.asarray(raw_y_source)
    if raw_y.ndim != 2:
        raise ValueError("panel y must have shape (series, time)")
    if raw_y.shape[1] != int(request.panel_length):
        raise ValueError("panel length differs from the window split contract")
    panel_dataset_id = panel.get("name")
    if panel_dataset_id is not None and str(panel_dataset_id) != request.dataset_id:
        raise ValueError("panel dataset identity differs from the window request")
    y_numeric = raw_y.astype(np.float64)
    if not np.isfinite(y_numeric).all():
        raise ValueError("panel y contains nonfinite values before windowing")
    y = y_numeric.astype(np.float32)
    series_ids = _string_series_ids(panel.get("series_id", np.arange(y.shape[0])))
    if series_ids.shape != (y.shape[0],):
        raise ValueError("series_id must contain one ID per series")
    integer_origins = np.asarray(request.origins, dtype=np.int32)
    forecast_horizon = int(request.split.horizon)
    history_length = int(request.split.lookback)
    available_from = validate_available_from(
        panel.get("available_from", np.zeros(y.shape[0], dtype=np.int32)),
        n_series=y.shape[0],
        length=y.shape[1],
    )
    if [int(y.shape[0]), int(y.shape[1])] != [
        int(value) for value in verified_audit["panel_shape"]
    ]:
        raise ValueError("panel shape differs from the sealed dataset audit")
    actual_panel_binding = _panel_binding_sha256(y_numeric, available_from, series_ids)
    if actual_panel_binding != verified_audit.get("panel_binding_sha256"):
        raise ValueError("panel binding differs from the sealed dataset audit")
    sample_manifest = seal_train_only_sample_manifest(
        panel,
        dataset_audit=verified_audit,
        runtime_tier=request.sampling_runtime_tier,
        seed=request.sampling_seed,
    )
    if sample_manifest["manifest_sha256"] != request.sample_manifest_sha256:
        raise ValueError("window sample manifest differs from the sealed request")
    selected_positions = np.asarray(
        sample_manifest["selected_positions"], dtype=np.int64
    )
    y_numeric = y_numeric[selected_positions]
    y = y[selected_positions]
    available_from = available_from[selected_positions]
    series_ids = series_ids[selected_positions]
    if (
        _panel_binding_sha256(y_numeric, available_from, series_ids)
        != request.sampled_panel_binding_sha256
    ):
        raise ValueError("sampled panel binding differs from the sealed request")
    scale_array = train_rms_scale(
        y_numeric,
        train_end=int(request.split.train[1]),
        available_from=available_from,
    )

    history_rows = []
    target_rows = []
    raw_target_rows = []
    occurrence_rows = []
    mask_rows = []
    scale_rows = []
    for origin in integer_origins:
        history_rows.append(y[:, int(origin) - history_length : int(origin)])
        target = y[:, int(origin) : int(origin) + forecast_horizon]
        target_rows.append(target)
        raw_target_rows.append(
            y_numeric[:, int(origin) : int(origin) + forecast_horizon]
        )
        occurrence_rows.append((target > 0.0).astype(np.float32))
        target_times = np.arange(int(origin), int(origin) + forecast_horizon, dtype=np.int32)
        mask_rows.append(target_times[None, :] >= available_from[:, None])
        scale_rows.append(scale_array)
    history = np.concatenate(history_rows, axis=0).astype(np.float32)
    target = np.concatenate(target_rows, axis=0).astype(np.float32)
    raw_target = np.concatenate(raw_target_rows, axis=0).astype(np.float64)
    occurrence = np.concatenate(occurrence_rows, axis=0).astype(np.float32)
    target_mask = np.concatenate(mask_rows, axis=0).astype(bool)
    count_audit = audit_target_support(
        raw_target[target_mask], dataset_id=request.dataset_id
    )
    if not bool(count_audit["count_primary_eligible"]):
        raise ValueError(f"count target support failed: {count_audit['ineligible_reason']}")
    _assert_float32_exact_roundtrip(raw_target[target_mask], label="count target")
    repeated_scale = np.concatenate(scale_rows, axis=0).astype(np.float32)
    return WindowBatch(
        history=history,
        target=target,
        occurrence=occurrence,
        target_mask=target_mask,
        scale=repeated_scale,
        key_frame=origin_key_frame(
            dataset_id=request.dataset_id,
            series_ids=series_ids,
            origins=integer_origins,
            horizon=forecast_horizon,
        ),
        origins=integer_origins.copy(),
        series_id=series_ids.copy(),
        split_name=request.role,
        dataset_audit_sha256=request.dataset_audit_sha256,
        sample_manifest_sha256=request.sample_manifest_sha256,
        sampled_panel_binding_sha256=request.sampled_panel_binding_sha256,
        confirmatory_eligible=request.confirmatory_eligible,
        split_contract_sha256=request.split_contract_sha256,
        request_sha256=request.request_sha256,
    )


def _positive_cv2(values: np.ndarray) -> float:
    positives = values[values > 0.0]
    if positives.size < 2:
        return np.nan
    mean = float(positives.mean())
    if mean <= 0.0:
        return np.nan
    return float((positives.std(ddof=1) / mean) ** 2)


def train_only_descriptors(
    values: object, metadata: pd.DataFrame, *, train_end: int
) -> pd.DataFrame:
    """Compute sampling descriptors from the availability-aware train segment only."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if len(metadata) != y.shape[0]:
        raise ValueError("metadata row count must match values")
    if not {"dataset_id", "series_id"}.issubset(metadata.columns):
        raise ValueError("metadata must include dataset_id and series_id")
    end = int(train_end)
    if not 0 < end <= y.shape[1]:
        raise ValueError("train_end must fall within the time axis")
    train = y[:, :end]
    if not np.isfinite(train).all() or (train < 0.0).any():
        raise ValueError("train descriptors require finite nonnegative values")
    available_from = validate_available_from(
        metadata["available_from"].to_numpy()
        if "available_from" in metadata.columns
        else np.zeros(y.shape[0], dtype=np.int32),
        n_series=y.shape[0],
        length=y.shape[1],
    )
    rows = []
    for index, segment in enumerate(train):
        available_start = int(available_from[index])
        descriptor_eligible = available_start < end
        observed = segment[available_start:end] if descriptor_eligible else segment[:0]
        scale = (
            float(np.sqrt(np.mean(observed * observed) + RMS_EPSILON))
            if descriptor_eligible
            else np.nan
        )
        positives = observed[observed > 0.0]
        rows.append(
            {
                "dataset_id": str(metadata.iloc[index]["dataset_id"]),
                "series_id": str(metadata.iloc[index]["series_id"]),
                "available_from": available_start,
                "n_train": int(observed.size),
                "n_positive_train": int(positives.size),
                "zero_ratio_train": (
                    float(np.mean(observed <= 0.0)) if observed.size else np.nan
                ),
                "ADI_train": (
                    float(observed.size / positives.size) if positives.size else np.nan
                ),
                "positive_CV2_train": _positive_cv2(observed),
                "log_train_scale": float(np.log(scale)) if descriptor_eligible else np.nan,
                "descriptor_eligible": bool(descriptor_eligible),
                "descriptor_ineligible_reason": (
                    None
                    if descriptor_eligible
                    else "available_from_at_or_after_train_end"
                ),
            }
        )
    return pd.DataFrame(rows)


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "big")


_SAMPLING_DESCRIPTOR_COLUMNS = (
    "zero_ratio_train",
    "ADI_train",
    "positive_CV2_train",
    "log_train_scale",
)
_SAMPLING_BIN_COLUMNS = tuple(
    f"{column}_bin" for column in _SAMPLING_DESCRIPTOR_COLUMNS
)


def _ordered_string_sha256(values: Sequence[object]) -> str:
    return hashlib.sha256(
        json.dumps(
            [str(value) for value in values],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_record_sha256(
    record: Mapping[str, object], *, excluded_key: str | None = None
) -> str:
    payload = {
        str(key): value
        for key, value in record.items()
        if excluded_key is None or str(key) != excluded_key
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("sealed record must contain finite JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sample_manifest_hash(record: Mapping[str, object]) -> str:
    return _finite_record_sha256(record, excluded_key="manifest_sha256")


def _eligible_descriptor_strata(descriptors: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset_id",
        "series_id",
        *_SAMPLING_DESCRIPTOR_COLUMNS,
    }
    if not required.issubset(descriptors.columns):
        missing = sorted(required - set(descriptors.columns))
        raise ValueError(f"descriptors are missing columns: {missing}")
    frame = descriptors.copy()
    frame["dataset_id"] = frame["dataset_id"].astype(str)
    frame["series_id"] = frame["series_id"].astype(str)
    if frame[["dataset_id", "series_id"]].duplicated().any():
        raise ValueError("duplicate descriptor series keys are not allowed")
    if "descriptor_eligible" in frame.columns:
        frame = frame[frame["descriptor_eligible"].astype(bool)].copy()
    if frame.empty:
        raise ValueError("no eligible descriptor rows remain after availability filtering")
    frame = frame.sort_values(
        ["dataset_id", "series_id"], kind="mergesort"
    ).reset_index(drop=True)
    blocks: list[pd.DataFrame] = []
    for _, group in frame.groupby("dataset_id", sort=True):
        block = group.copy()
        for column in _SAMPLING_DESCRIPTOR_COLUMNS:
            block[f"{column}_bin"] = _quantile_bins(
                block[column], block["series_id"]
            )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True).sort_values(
        ["dataset_id", "series_id"], kind="mergesort"
    ).reset_index(drop=True)


def _quantile_bins(values: pd.Series, ids: pd.Series | None = None, q: int = 4) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().nunique() <= 1:
        return pd.Series(np.zeros(len(values), dtype=np.int16), index=values.index)
    sort_ids = (
        pd.Series(np.arange(len(values)).astype(str), index=values.index)
        if ids is None
        else ids.astype(str)
    )
    ordered = pd.DataFrame(
        {"value": numeric, "series_id": sort_ids}, index=values.index
    ).sort_values(["value", "series_id"], kind="mergesort")
    ranked = pd.Series(np.nan, index=values.index, dtype=float)
    finite_index = ordered[ordered["value"].notna()].index
    ranked.loc[finite_index] = np.arange(1, len(finite_index) + 1, dtype=float)
    bins = pd.qcut(ranked, q=min(q, int(numeric.notna().sum())), labels=False, duplicates="drop")
    return bins.fillna(-1).astype(np.int16)


def _dataset_allocations(counts: pd.Series, n: int) -> dict[object, int]:
    total = int(counts.sum())
    requested = int(n)
    if requested < 1 or requested > total:
        raise ValueError("sample size must be between 1 and the population size")
    raw = counts.astype(float) * requested / total
    base = np.floor(raw).astype(int)
    for dataset in counts.index:
        if counts[dataset] > 0 and base[dataset] == 0 and requested >= len(counts):
            base[dataset] = 1
    while int(base.sum()) > requested:
        candidates = [item for item in counts.index if base[item] > 0]
        victim = min(candidates, key=lambda item: (raw[item] - base[item], str(item)))
        base[victim] -= 1
    remainder = requested - int(base.sum())
    order = sorted(counts.index, key=lambda item: (-(raw[item] - base[item]), str(item)))
    for dataset in order[:remainder]:
        base[dataset] += 1
    return {key: int(value) for key, value in base.items()}


def stratified_train_descriptor_sample(
    descriptors: pd.DataFrame,
    *,
    n: int | None = None,
    n_per_dataset: int | None = None,
    seed: int = REAL_SAMPLING_SEED,
) -> pd.DataFrame:
    """Deterministically sample within dataset-specific descriptor quantile bins."""
    frame = _eligible_descriptor_strata(descriptors)
    sampled_indices: list[int] = []
    counts = frame.groupby("dataset_id", sort=True).size()
    if n_per_dataset is not None:
        requested = int(n_per_dataset)
        if requested <= 0:
            raise ValueError("n_per_dataset must be positive")
        allocations = {dataset_id: min(requested, int(count)) for dataset_id, count in counts.items()}
    else:
        if n is None:
            raise ValueError("n_per_dataset is required unless sampling one dataset with n")
        if len(counts) != 1:
            raise ValueError("n_per_dataset is required when descriptors contain multiple datasets")
        requested = int(n)
        if requested <= 0:
            raise ValueError("n must be positive")
        only_dataset = counts.index[0]
        allocations = {only_dataset: min(requested, int(counts.iloc[0]))}
    for dataset_id, take in allocations.items():
        group = frame[frame["dataset_id"] == str(dataset_id)].copy()
        if int(take) == len(group):
            sampled_indices.extend(int(index) for index in group.index.to_numpy())
            continue
        strata = group.groupby(
            list(_SAMPLING_BIN_COLUMNS),
            sort=True,
        )
        stratum_sizes = strata.size()
        stratum_take = _dataset_allocations(stratum_sizes, min(take, len(group)))
        chosen: list[int] = []
        for key, count in stratum_take.items():
            if count <= 0:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            block = strata.get_group(key).sort_values("series_id", kind="mergesort")
            rng = np.random.default_rng(_stable_seed(int(seed), "stratum", dataset_id, key))
            order = rng.permutation(block.index.to_numpy())
            chosen.extend(int(item) for item in order[:count])
        if len(chosen) < take:
            remaining = np.array(
                [idx for idx in group.index.to_numpy() if int(idx) not in set(chosen)]
            )
            fallback_rng = np.random.default_rng(_stable_seed(int(seed), "dataset", dataset_id, "fallback"))
            chosen.extend(int(item) for item in fallback_rng.permutation(remaining)[: take - len(chosen)])
        sampled_indices.extend(chosen[:take])
    return frame.loc[sampled_indices].reset_index(drop=True)


def _json_descriptor_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return numeric
    return str(value)


def _descriptor_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    columns = (
        "dataset_id",
        "series_id",
        "available_from",
        "n_train",
        "n_positive_train",
        *_SAMPLING_DESCRIPTOR_COLUMNS,
        "descriptor_eligible",
        "descriptor_ineligible_reason",
    )
    return [
        {column: _json_descriptor_value(row[column]) for column in columns}
        for _, row in frame.sort_values(
            ["dataset_id", "series_id"], kind="mergesort"
        ).iterrows()
    ]


def _stratum_record(row: pd.Series) -> dict[str, object]:
    return {
        "zero_ratio_train_bin": int(row["zero_ratio_train_bin"]),
        "ADI_train_bin": int(row["ADI_train_bin"]),
        "positive_CV2_train_bin": int(row["positive_CV2_train_bin"]),
        "log_train_scale_bin": int(row["log_train_scale_bin"]),
    }


def seal_train_only_sample_manifest(
    panel: Mapping[str, object],
    *,
    dataset_audit: Mapping[str, object],
    runtime_tier: str,
    seed: int = REAL_SAMPLING_SEED,
) -> dict[str, object]:
    """Recompute and seal the preregistered real-series sample from train only."""
    verified_audit = _verify_sealed_dataset_audit(dataset_audit)
    if verified_audit.get("audit_type") != "COUNT_PRIMARY_DATASET_AUDIT":
        raise ValueError("sampling requires a sealed count-primary dataset audit")
    tier = str(runtime_tier)
    if tier not in REAL_SERIES_PER_RUNTIME_TIER:
        raise ValueError("unknown preregistered runtime tier")
    if isinstance(seed, bool) or int(seed) != REAL_SAMPLING_SEED:
        raise ValueError("real sampling seed differs from the preregistered seed")
    if not isinstance(panel, Mapping):
        raise TypeError("panel must be a mapping")
    dataset_id = str(panel.get("name", panel.get("dataset_id", "")))
    if dataset_id != str(verified_audit.get("dataset_id")):
        raise ValueError("sample panel dataset identity differs from its audit")
    raw_source = panel.get("raw_y", panel.get("y"))
    if raw_source is None:
        raise ValueError("sample panel requires its target array")
    values = np.asarray(raw_source, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("sample panel target must be two-dimensional")
    series_ids = _string_series_ids(
        panel.get("series_id", np.arange(values.shape[0]))
    )
    if series_ids.shape != (values.shape[0],):
        raise ValueError("sample panel requires one series ID per row")
    available_from = validate_available_from(
        panel.get("available_from", np.zeros(values.shape[0], dtype=np.int32)),
        n_series=values.shape[0],
        length=values.shape[1],
    )
    if [int(values.shape[0]), int(values.shape[1])] != [
        int(value) for value in verified_audit["panel_shape"]
    ]:
        raise ValueError("sample panel shape differs from its dataset audit")
    full_binding = _panel_binding_sha256(values, available_from, series_ids)
    if full_binding != verified_audit.get("panel_binding_sha256"):
        raise ValueError("sample panel binding differs from its dataset audit")
    split_audit = verified_audit.get("split_validation")
    if not isinstance(split_audit, Mapping):
        raise ValueError("sample dataset split audit is missing")
    train_interval = tuple(int(value) for value in split_audit.get("train", ()))
    if len(train_interval) != 2 or train_interval[0] != 0:
        raise ValueError("sample dataset train interval is malformed")
    train_end = int(train_interval[1])
    metadata = pd.DataFrame(
        {
            "dataset_id": np.repeat(dataset_id, values.shape[0]),
            "series_id": series_ids,
            "available_from": available_from,
        }
    )
    descriptors = train_only_descriptors(values, metadata, train_end=train_end)
    stratified = _eligible_descriptor_strata(descriptors)
    requested_n = int(REAL_SERIES_PER_RUNTIME_TIER[tier])
    selected = stratified_train_descriptor_sample(
        descriptors,
        n=requested_n,
        seed=REAL_SAMPLING_SEED,
    )
    selected_ids = selected["series_id"].astype(str).tolist()
    positions_by_id = {str(value): index for index, value in enumerate(series_ids)}
    selected_positions = [int(positions_by_id[value]) for value in selected_ids]
    position_array = np.asarray(selected_positions, dtype=np.int64)
    selected_strata = [
        {"series_id": str(row["series_id"]), **_stratum_record(row)}
        for _, row in selected.iterrows()
    ]
    stratum_population = (
        stratified.groupby(list(_SAMPLING_BIN_COLUMNS), sort=True)
        .size()
        .rename("population_n")
        .reset_index()
    )
    stratum_selected = (
        selected.groupby(list(_SAMPLING_BIN_COLUMNS), sort=True)
        .size()
        .rename("selected_n")
        .reset_index()
    )
    allocation = stratum_population.merge(
        stratum_selected, on=list(_SAMPLING_BIN_COLUMNS), how="left"
    ).fillna({"selected_n": 0})
    allocation_records = [
        {
            **_stratum_record(row),
            "population_n": int(row["population_n"]),
            "selected_n": int(row["selected_n"]),
        }
        for _, row in allocation.iterrows()
    ]
    descriptor_rows = _descriptor_records(descriptors)
    record: dict[str, object] = {
        "manifest_type": "TRAIN_ONLY_STRATIFIED_SAMPLE_MANIFEST",
        "dataset_id": dataset_id,
        "runtime_tier": tier,
        "seed": REAL_SAMPLING_SEED,
        "requested_n": requested_n,
        "actual_n": len(selected_ids),
        "sampling_unit": "series_id",
        "descriptor_scope": "model_train_only_availability_aware",
        "descriptor_names": list(_SAMPLING_DESCRIPTOR_COLUMNS),
        "stratification": "within_dataset_joint_quartile_proportional_largest_remainder",
        "train_end": train_end,
        "dataset_audit_sha256": str(verified_audit["audit_sha256"]),
        "source_manifest_aggregate_sha256": str(
            verified_audit["source_manifest_aggregate_sha256"]
        ),
        "full_panel_shape": [int(values.shape[0]), int(values.shape[1])],
        "full_panel_binding_sha256": full_binding,
        "descriptor_table_sha256": _finite_record_sha256(
            {"rows": descriptor_rows}
        ),
        "eligible_pool_n": int(len(stratified)),
        "eligible_pool_series_id_sha256": _ordered_string_sha256(
            stratified["series_id"].astype(str).tolist()
        ),
        "selected_positions": selected_positions,
        "selected_series_ids": selected_ids,
        "selected_series_id_sha256": _ordered_string_sha256(selected_ids),
        "selected_strata": selected_strata,
        "stratum_allocation": allocation_records,
        "sampled_panel_shape": [len(selected_ids), int(values.shape[1])],
        "sampled_panel_binding_sha256": _panel_binding_sha256(
            values[position_array],
            available_from[position_array],
            series_ids[position_array],
        ),
    }
    record["manifest_sha256"] = _sample_manifest_hash(record)
    return record


def verify_train_only_sample_manifest(
    panel: Mapping[str, object],
    *,
    dataset_audit: Mapping[str, object],
    sample_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Reject any sample not equal to the canonical train-only reconstruction."""
    if not isinstance(sample_manifest, Mapping):
        raise TypeError("sample_manifest must be a mapping")
    supplied = dict(sample_manifest)
    digest = supplied.get("manifest_sha256")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or digest != _sample_manifest_hash(supplied)
    ):
        raise ValueError("sample manifest hash mismatch")
    reconstructed = seal_train_only_sample_manifest(
        panel,
        dataset_audit=dataset_audit,
        runtime_tier=str(supplied.get("runtime_tier", "")),
        seed=int(supplied.get("seed", -1)),
    )
    if supplied != reconstructed:
        raise ValueError("sample manifest differs from its canonical train-only reconstruction")
    return reconstructed


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_hash(path: Path | str, expected_sha256: str) -> dict[str, object]:
    source = Path(path)
    actual = file_sha256(source)
    expected = str(expected_sha256).lower()
    if actual.lower() != expected:
        raise ValueError("source hash mismatch")
    return {
        "path": str(source),
        "size_bytes": int(source.stat().st_size),
        "sha256": actual,
        "expected_sha256": expected,
        "status": "PASS",
    }


def _source_provenance_record(
    path: Path | str,
    *,
    expected_sha256: str,
    source_identity: str,
    adapter_function: str,
) -> dict[str, object]:
    guard = verify_source_hash(path, expected_sha256)
    return {
        **guard,
        "source_identity": source_identity,
        "adapter_function": adapter_function,
    }


def _require_frozen_sha256(
    provided: str | None, frozen: str, *, label: str
) -> str:
    if provided is None:
        raise ValueError(f"expected_sha256 is required for {label}")
    normalized = str(provided).lower()
    if normalized != str(frozen).lower():
        raise ValueError(f"expected_sha256 for {label} must equal the frozen literal")
    return normalized


def _require_frozen_source_hashes(
    provided: Mapping[str, str] | None, frozen: Mapping[str, str], *, label: str
) -> dict[str, str]:
    if provided is None:
        raise ValueError(f"expected_source_hashes is required for {label}")
    required = set(frozen)
    provided_keys = set(provided)
    missing = sorted(required - provided_keys)
    extra = sorted(provided_keys - required)
    if missing or extra:
        raise ValueError(
            f"expected_source_hashes for {label} must match frozen source keys; "
            f"missing={missing}, extra={extra}"
        )
    normalized = {key: str(value).lower() for key, value in provided.items()}
    expected = {key: str(value).lower() for key, value in frozen.items()}
    if normalized != expected:
        raise ValueError(f"expected_source_hashes for {label} must equal frozen literals")
    return normalized


def _resolve_source_snapshot_file(
    root_or_file: Path | str, *, dataset_id: str, filename: str
) -> Path:
    path = Path(root_or_file)
    if path.name == filename:
        return path
    direct = path / filename
    if direct.exists():
        return direct
    return path / dataset_id / filename


def _path_key(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        stat_result = path.stat(follow_symlinks=False)
    except (FileNotFoundError, OSError):
        return path.is_symlink()
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return bool(path.is_symlink() or attributes & WINDOWS_REPARSE_POINT_ATTRIBUTE)


def _reject_reparse_or_symlink_components(path: Path) -> None:
    checks = [path, *path.parents]
    for candidate in checks:
        if candidate.exists() or candidate.is_symlink():
            if _is_reparse_or_symlink(candidate):
                raise ValueError(
                    "production source_snapshots paths must not contain a symlink or reparse point"
                )


def _production_snapshot_dataset_dir(
    root_or_file: Path | str, *, dataset_id: str, filename: str | None = None
) -> Path:
    path = Path(root_or_file)
    path_name = path.name.casefold()
    dataset_name = dataset_id.casefold()
    if filename is not None and path_name == filename.casefold():
        candidate = path.parent
    elif path_name == dataset_name:
        candidate = path
    elif path_name == "source_snapshots":
        candidate = path / dataset_id
    else:
        candidate = path
    expected = SOURCE_SNAPSHOT_ROOT / dataset_id
    _reject_reparse_or_symlink_components(candidate)
    if _path_key(candidate) != _path_key(expected):
        raise ValueError(
            "production source path must resolve to "
            f"runs/prob_head_structure_full_v1/source_snapshots/{dataset_id}"
        )
    return candidate


def _production_snapshot_file(
    root_or_file: Path | str, *, dataset_id: str, filename: str
) -> Path:
    path = Path(root_or_file)
    dataset_dir = _production_snapshot_dataset_dir(
        path, dataset_id=dataset_id, filename=filename
    )
    source = path if path.name.casefold() == filename.casefold() else dataset_dir / filename
    _reject_reparse_or_symlink_components(source)
    return source


def _m5_source_paths_from_snapshot_layout(source_dir: Path | str) -> dict[str, Path]:
    dataset_dir = _production_snapshot_dataset_dir(source_dir, dataset_id="m5")
    return {
        name: _production_snapshot_file(dataset_dir, dataset_id="m5", filename=name)
        for name in sorted(M5_EXPECTED_SOURCE_SHA256)
    }


def _canonical_series_id_hash(series_ids: Sequence[object]) -> str:
    joined = "\n".join(sorted(str(series_id) for series_id in series_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_m5_stage_a_exclusion(
    source_dir: Path | str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Load the frozen Stage-A M5 ID exclusion set from a hash-verified snapshot."""
    frozen_sha256 = _require_frozen_sha256(
        expected_sha256,
        M5_STAGE_A_EXPECTED_SHA256,
        label="the production M5 Stage-A exclusion loader",
    )
    source = _production_snapshot_file(
        source_dir, dataset_id="m5", filename=M5_STAGE_A_SNAPSHOT_NAME
    )
    source_record = _source_provenance_record(
        source,
        expected_sha256=frozen_sha256,
        source_identity="M5 prior Stage-A sampled series snapshot",
        adapter_function=M5_STAGE_A_ADAPTER_FUNCTION,
    )
    frame = pd.read_parquet(source, columns=["series_id"])
    ids = frame["series_id"].dropna().astype(str).drop_duplicates().tolist()
    ids = sorted(ids)
    canonical_hash = _canonical_series_id_hash(ids)
    if len(ids) != M5_STAGE_A_EXPECTED_N_SERIES:
        raise ValueError("M5 Stage-A exclusion must contain exactly 1200 unique IDs")
    if canonical_hash != M5_STAGE_A_CANONICAL_ID_SHA256:
        raise ValueError("M5 Stage-A exclusion canonical ID hash mismatch")
    return {
        "status": "PASS",
        "series_ids": tuple(ids),
        "n_series": int(len(ids)),
        "canonical_id_sha256": canonical_hash,
        "source": source_record,
        "source_snapshot": source_record,
    }


def _m5_stage_a_exclusion_for_test(series_ids: Sequence[object]) -> dict[str, object]:
    ids = tuple(sorted(str(series_id) for series_id in series_ids))
    return {
        "status": "PASS",
        "series_ids": ids,
        "n_series": int(len(ids)),
        "canonical_id_sha256": _canonical_series_id_hash(ids),
        "source": {
            "path": "memory://m5-stage-a",
            "size_bytes": 0,
            "sha256": "0" * 64,
            "expected_sha256": "0" * 64,
            "source_identity": "unit-test M5 Stage-A exclusion",
            "adapter_function": "tests._m5_stage_a_exclusion_for_test",
            "status": "PASS",
        },
        "test_only": True,
    }


def _id_set(values: set[object] | Sequence[object] | object | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        return {str(values)}
    return {str(value) for value in values}  # type: ignore[union-attr]


def _validate_m5_stage_a_exclusion(
    stage_a_exclusion: Mapping[str, object] | None, *, allow_test_stage_a: bool
) -> tuple[set[str], dict[str, object]]:
    if stage_a_exclusion is None:
        raise ValueError("stage_a_exclusion is required for M5 loading")
    if not isinstance(stage_a_exclusion, Mapping):
        raise ValueError("stage_a_exclusion must be a verified mapping")
    if stage_a_exclusion.get("status") != "PASS":
        raise ValueError("stage_a_exclusion must have PASS status")
    ids = _id_set(stage_a_exclusion.get("series_ids"))
    if not ids:
        raise ValueError("stage_a_exclusion must include series_ids")
    n_series = int(stage_a_exclusion.get("n_series", -1))
    canonical_hash = _canonical_series_id_hash(tuple(ids))
    if n_series != len(ids) or canonical_hash != stage_a_exclusion.get("canonical_id_sha256"):
        raise ValueError("stage_a_exclusion ID count or canonical hash mismatch")
    if not allow_test_stage_a:
        source = stage_a_exclusion.get("source")
        if (
            n_series != M5_STAGE_A_EXPECTED_N_SERIES
            or canonical_hash != M5_STAGE_A_CANONICAL_ID_SHA256
            or not isinstance(source, Mapping)
            or str(source.get("sha256", "")).lower() != M5_STAGE_A_EXPECTED_SHA256
        ):
            raise ValueError("stage_a_exclusion must be the frozen verified M5 Stage-A snapshot")
    source_obj = stage_a_exclusion.get("source")
    return ids, {
        "status": "PASS",
        "n_series": n_series,
        "canonical_id_sha256": canonical_hash,
        "source": dict(source_obj) if isinstance(source_obj, Mapping) else {},
        "test_only": bool(stage_a_exclusion.get("test_only", False)),
    }


def exclude_stage_a_series(
    panel: Mapping[str, object], prior_stage_a_ids: set[object] | Sequence[object]
) -> dict[str, object]:
    """Remove known Stage-A IDs and sort survivors for deterministic provenance."""
    y = np.asarray(panel["y"])
    series_ids = np.asarray(panel["series_id"]).astype(str)
    if y.shape[0] != series_ids.size:
        raise ValueError("panel y and series_id disagree on series count")
    excluded = _id_set(prior_stage_a_ids)
    keep_mask = np.asarray([series_id not in excluded for series_id in series_ids], dtype=bool)
    kept_positions = np.flatnonzero(keep_mask)
    order = kept_positions[np.argsort(series_ids[kept_positions], kind="mergesort")]
    filtered: dict[str, object] = {}
    for key, value in panel.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == series_ids.size:
            filtered[key] = array[order]
        else:
            filtered[key] = value
    filtered["series_id"] = series_ids[order]
    filtered["excluded_stage_a_count"] = int((~keep_mask).sum())
    filtered["excluded_stage_a_ids"] = tuple(sorted(excluded))
    return filtered


def _m5_availability_from_verified_paths(
    meta: pd.DataFrame, *, calendar_path: Path | str, sell_prices_path: Path | str
) -> np.ndarray:
    """M5 availability: first day of the first priced week for each store/item."""
    required = {"store_id", "item_id"}
    if not required.issubset(meta.columns):
        missing = sorted(required - set(meta.columns))
        raise ValueError(f"M5 availability metadata is missing columns: {missing}")
    calendar = pd.read_csv(calendar_path, usecols=["d", "wm_yr_wk"])
    calendar["day_idx"] = calendar["d"].astype(str).str.slice(2).astype(int) - 1
    week_start = calendar.groupby("wm_yr_wk")["day_idx"].min()
    prices = pd.read_csv(
        sell_prices_path, usecols=["store_id", "item_id", "wm_yr_wk"]
    )
    first_week = prices.groupby(["store_id", "item_id"])["wm_yr_wk"].min()
    out: list[int] = []
    for _, row in meta.iterrows():
        key = (row["store_id"], row["item_id"])
        if key not in first_week.index:
            raise ValueError(
                f"no sell_prices row for {key}; M5 availability cannot be defined"
            )
        week = first_week[key]
        if week not in week_start.index:
            raise ValueError(f"calendar has no day for M5 wm_yr_wk={week!r}")
        out.append(int(week_start[week]))
    return np.asarray(out, dtype=np.int32)


def _m5_panel_from_verified_sources(source_paths: Mapping[str, Path]) -> dict[str, object]:
    """Canonical M5 full-population adapter from verified snapshot files."""

    wide = pd.read_csv(source_paths["sales_train_evaluation.csv"])
    day_cols = [column for column in wide.columns if column.startswith("d_")]
    if not day_cols:
        raise ValueError("M5 source has no d_* daily target columns")
    required_meta = {"id", "item_id", "store_id"}
    if not required_meta.issubset(wide.columns):
        missing = sorted(required_meta - set(wide.columns))
        raise ValueError(f"M5 source is missing columns: {missing}")
    y = wide[day_cols].to_numpy()
    available_from = _m5_availability_from_verified_paths(
        wide[["item_id", "store_id"]],
        calendar_path=source_paths["calendar.csv"],
        sell_prices_path=source_paths["sell_prices.csv"],
    )
    y_numeric = np.asarray(y, dtype=np.float64)
    first_positive = np.array(
        [int(np.argmax(row > 0.0)) if (row > 0.0).any() else len(row) for row in y_numeric],
        dtype=np.int32,
    )
    return {
        "name": "m5",
        "y": y,
        "z": (y_numeric > 0.0).astype(np.float32),
        "series_id": wide["id"].to_numpy().astype(str),
        "available_from": np.asarray(available_from, dtype=np.int32),
        "first_positive": first_positive,
    }


def _finalize_m5_panel(
    panel: Mapping[str, object],
    *,
    stage_a_exclusion: Mapping[str, object] | None,
    source_provenance: Sequence[Mapping[str, object]],
    adapter_function: str,
    allow_test_stage_a: bool = False,
) -> dict[str, object]:
    if not isinstance(panel, Mapping):
        raise ValueError("M5 loader must return a mapping")
    working = dict(panel)
    excluded_ids, stage_a_record = _validate_m5_stage_a_exclusion(
        stage_a_exclusion, allow_test_stage_a=allow_test_stage_a
    )
    working = exclude_stage_a_series(working, excluded_ids)
    raw_y = np.asarray(working["y"])
    if raw_y.ndim != 2:
        raise ValueError("M5 panel must have shape (series, time)")
    y_numeric = raw_y.astype(np.float64)
    y = y_numeric.astype(np.float32)
    series_id = _string_series_ids(working["series_id"])
    if y.ndim != 2 or y.shape[0] != series_id.size:
        raise ValueError("M5 panel must have shape (series, time) and aligned IDs")
    z = np.asarray(working.get("z", (y > 0.0).astype(np.float32)), dtype=np.float32)
    if z.shape != y.shape:
        raise ValueError("M5 occurrence panel must match target shape")
    available_from = validate_available_from(
        working.get("available_from", np.zeros(y.shape[0], dtype=np.int32)),
        n_series=y.shape[0],
        length=y.shape[1],
    )
    split_validation = validate_real_split(REAL_SPLITS["m5"], length=y.shape[1])
    support_audit = audit_model_train_target_support(
        y_numeric,
        split=REAL_SPLITS["m5"],
        dataset_id="m5",
        available_from=available_from,
        series_ids=series_id,
    )
    enforce_target_support(support_audit)
    _assert_float32_exact_roundtrip(
        _availability_model_train_values(
            y_numeric, split=REAL_SPLITS["m5"], available_from=available_from
        ),
        label="m5 model_train target",
    )
    provenance = {
        "dataset_id": "m5",
        "sources": [dict(source) for source in source_provenance],
        "adapter_function": adapter_function,
        "local_adapter_function": adapter_function,
        "upstream_reused_source": M5_UPSTREAM_REUSED_SOURCE,
        "stage_a_exclusion": stage_a_record,
        "stage_a_overlap_count": int(len(set(series_id.tolist()) & excluded_ids)),
        "support_audit": support_audit,
        "split_validation": split_validation,
        "source_snapshots_layout": "runs/prob_head_structure_full_v1/source_snapshots/m5/{sales_train_evaluation.csv,calendar.csv,sell_prices.csv,series.parquet}",
    }
    if provenance["stage_a_overlap_count"] != 0:
        raise ValueError("M5 Stage-A exclusion overlap remains after filtering")
    return {
        **working,
        "name": "m5",
        "y": y,
        "z": z,
        "series_id": series_id,
        "available_from": available_from,
        "split": REAL_SPLITS["m5"],
        "split_validation": split_validation,
        "support_audit": support_audit,
        "provenance": provenance,
        "source_adapter": adapter_function,
        "existing_pipeline_reused": True,
    }


def load_m5_via_existing_pipeline(
    source_dir: Path | str,
    *,
    expected_source_hashes: Mapping[str, str] | None = None,
    stage_a_exclusion: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load M5 from hash-verified raw sources using the canonical repo adapter."""
    frozen_hashes = _require_frozen_source_hashes(
        expected_source_hashes, M5_EXPECTED_SOURCE_SHA256, label="the production M5 loader"
    )
    source_paths = _m5_source_paths_from_snapshot_layout(source_dir)
    _validate_m5_stage_a_exclusion(stage_a_exclusion, allow_test_stage_a=False)
    source_records = [
        _source_provenance_record(
            source_paths[name],
            expected_sha256=frozen_hashes[name],
            source_identity=f"M5 {name}",
            adapter_function=M5_CANONICAL_ADAPTER_FUNCTION,
        )
        for name in sorted(M5_EXPECTED_SOURCE_SHA256)
    ]
    panel = _m5_panel_from_verified_sources(source_paths)
    return _finalize_m5_panel(
        panel,
        stage_a_exclusion=stage_a_exclusion,
        source_provenance=source_records,
        adapter_function=M5_CANONICAL_ADAPTER_FUNCTION,
    )


def _load_m5_via_existing_pipeline_for_test(
    *,
    loader: Callable[[], Mapping[str, object]],
    expected_loader_identity: str,
    source_provenance: Sequence[Mapping[str, object]],
    stage_a_exclusion: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Private test seam; production must use hash-verified source files."""
    if not expected_loader_identity:
        raise ValueError("expected_loader_identity is required for test M5 loader")
    if not source_provenance:
        raise ValueError("source_provenance is required for test M5 loader")
    panel = loader()
    result = _finalize_m5_panel(
        panel,
        stage_a_exclusion=stage_a_exclusion,
        source_provenance=source_provenance,
        adapter_function="experiments.prob_head_structure_full_v1.data._load_m5_via_existing_pipeline_for_test",
        allow_test_stage_a=True,
    )
    result["provenance"]["expected_loader_identity"] = expected_loader_identity
    return result


def aggregate_online_retail_tweediegp_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Reproduce TweedieGP's UCI-352 Description-level daily aggregation."""
    required = {"Description", "InvoiceDate", "Quantity"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"OnlineRetail frame is missing columns: {missing}")
    raw = frame.loc[:, ["Description", "InvoiceDate", "Quantity"]].copy()
    raw["Description"] = raw["Description"].astype(str)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"]).dt.normalize()
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="raise")
    items = sorted(raw["Description"].unique().tolist())
    sentinel = "add stock to allocate online orders"
    if sentinel in items:
        items = items[: items.index(sentinel)]
    start = raw["InvoiceDate"].min()
    end = raw["InvoiceDate"].max()
    grouped = raw.groupby(["Description", "InvoiceDate"], group_keys=False)["Quantity"].sum()
    calendar = pd.date_range(start, end, freq="D")
    available_items = set(grouped.index.get_level_values(0))
    series_ids: list[str] = []
    values: list[np.ndarray] = []
    for item in items:
        if item not in available_items:
            continue
        series = grouped.loc[item]
        if bool((series < 0).any()):
            continue
        if int((series.index.min() - start).days) >= 200:
            continue
        series_ids.append(item)
        values.append(series.reindex(calendar, fill_value=0).to_numpy(dtype=np.float64))
    raw_y = (
        np.vstack(values).astype(np.float64)
        if values
        else np.zeros((0, len(calendar)), dtype=np.float64)
    )
    y = raw_y.astype(np.float32)
    return {
        "series_id": pd.Series(series_ids, dtype=str),
        "calendar": calendar,
        "y": y,
        "raw_y": raw_y,
        "z": (y > 0.0).astype(np.float32),
        "date_start": str(start.date()),
        "date_end": str(end.date()),
        "raw_rows": int(len(frame)),
        "construction": ONLINE_RETAIL_EXPECTED["source"],
    }


def load_online_retail_tweediegp(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    enforce_frozen_shape: bool = True,
) -> dict[str, object]:
    """Load the local OnlineRetail II workbook through the frozen TweedieGP adapter."""
    frozen_sha256 = _require_frozen_sha256(
        expected_sha256,
        ONLINE_RETAIL_EXPECTED_SHA256,
        label="the production OnlineRetail loader",
    )
    source = _production_snapshot_file(
        path, dataset_id="online_retail", filename="online_retail_II.xlsx"
    )
    source_record = _source_provenance_record(
        source,
        expected_sha256=frozen_sha256,
        source_identity="Online Retail II workbook, sheet Year 2010-2011",
        adapter_function=ONLINE_RETAIL_ADAPTER_FUNCTION,
    )
    frame = pd.read_excel(
        source,
        sheet_name=ONLINE_RETAIL_EXPECTED["sheet_name"],
        usecols=["Description", "InvoiceDate", "Quantity"],
    )
    grid = aggregate_online_retail_tweediegp_frame(frame)
    if grid["raw_rows"] != ONLINE_RETAIL_EXPECTED["raw_rows"]:
        raise ValueError("OnlineRetail raw row count does not match frozen expectation")
    if [grid["date_start"], grid["date_end"]] != [
        ONLINE_RETAIL_EXPECTED["date_start"],
        ONLINE_RETAIL_EXPECTED["date_end"],
    ]:
        raise ValueError("OnlineRetail date range does not match frozen expectation")
    if enforce_frozen_shape and tuple(grid["y"].shape) != (
        ONLINE_RETAIL_EXPECTED["n_series"],
        ONLINE_RETAIL_EXPECTED["n_dates"],
    ):
        raise ValueError("OnlineRetail dense grid shape does not match frozen expectation")
    support_audit = audit_model_train_target_support(
        grid["raw_y"], split=REAL_SPLITS["online_retail"], dataset_id="online_retail"
    )
    enforce_target_support(support_audit)
    _assert_float32_exact_roundtrip(
        _availability_model_train_values(
            grid["raw_y"], split=REAL_SPLITS["online_retail"]
        ),
        label="online_retail model_train target",
    )
    split_validation = validate_real_split(
        REAL_SPLITS["online_retail"], length=grid["y"].shape[1]
    )
    provenance = {
        "dataset_id": "online_retail",
        "sources": [source_record],
        "adapter_function": source_record["adapter_function"],
        "local_adapter_function": ONLINE_RETAIL_ADAPTER_FUNCTION,
        "upstream_reused_source": ONLINE_RETAIL_UPSTREAM_REUSED_SOURCE,
        "raw_rows": int(grid["raw_rows"]),
        "date_start": grid["date_start"],
        "date_end": grid["date_end"],
        "n_series": int(grid["y"].shape[0]),
        "n_dates": int(grid["y"].shape[1]),
        "support_audit": support_audit,
        "split_validation": split_validation,
        "source_snapshots_layout": "runs/prob_head_structure_full_v1/source_snapshots/online_retail/online_retail_II.xlsx",
    }
    return {
        "name": "online_retail",
        "split": REAL_SPLITS["online_retail"],
        "split_validation": split_validation,
        "support_audit": support_audit,
        "provenance": provenance,
        "available_from": np.zeros(grid["y"].shape[0], dtype=np.int32),
        **grid,
    }


def audit_fixed_length_dataset(
    *, dataset_id: str, length: int, lookback: int, horizon: int
) -> dict[str, object]:
    """Seal a geometry-only audit that can never establish count eligibility."""
    required = lookback + horizon + horizon + 6 * horizon
    supported = int(length) >= required
    record: dict[str, object] = {
        "audit_type": "GEOMETRY_ONLY_AUDIT",
        "dataset_id": str(dataset_id),
        "length": int(length),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "required_length": int(required),
        "geometry_status": "PASS" if supported else "INSUFFICIENT_LENGTH",
        "status": "SUPPORT_SOURCE_AUDIT_REQUIRED" if supported else "INSUFFICIENT_LENGTH",
        "count_primary_eligible": False,
        "confirmatory_eligible": False,
    }
    record["audit_sha256"] = _dataset_audit_hash(record)
    return record


def _dataset_audit_hash(record: Mapping[str, object]) -> str:
    unhashed = {str(key): value for key, value in record.items() if key != "audit_sha256"}
    try:
        encoded = json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset audit must contain finite JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _panel_binding_sha256(
    values: np.ndarray,
    available_from: np.ndarray,
    series_ids: np.ndarray,
) -> str:
    numeric = np.ascontiguousarray(values, dtype=np.float64)
    availability = np.ascontiguousarray(available_from, dtype=np.int32)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "shape": list(numeric.shape),
                "dtype": "float64",
                "available_from_shape": list(availability.shape),
                "series_ids": [str(value) for value in series_ids],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    digest.update(memoryview(numeric).cast("B"))
    digest.update(memoryview(availability).cast("B"))
    return digest.hexdigest()


def seal_count_primary_dataset_audit(
    dataset: Mapping[str, object],
    *,
    source_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Recompute and bind support, split, panel, and physical source evidence."""
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping")
    dataset_id = str(dataset.get("name", dataset.get("dataset_id", "")))
    if not dataset_id:
        raise ValueError("dataset audit requires a nonempty dataset_id")
    panel_values = dataset.get("raw_y", dataset.get("y"))
    if panel_values is None:
        raise ValueError("count-primary audit requires the actual target panel")
    values = np.asarray(panel_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("count-primary audit target panel must be two-dimensional")
    split_contract = dataset.get("split")
    if not isinstance(split_contract, RealSplit):
        raise ValueError("count-primary audit requires its RealSplit contract")
    frozen_split = REAL_SPLITS.get(dataset_id)
    if frozen_split is not None and split_contract != frozen_split:
        raise ValueError("dataset split differs from the frozen canonical split")
    split = validate_real_split(split_contract, length=values.shape[1])
    availability = validate_available_from(
        dataset.get("available_from", np.zeros(values.shape[0], dtype=np.int32)),
        n_series=values.shape[0],
        length=values.shape[1],
    )
    series_ids = _string_series_ids(
        dataset.get("series_id", np.arange(values.shape[0]))
    )
    if series_ids.shape != (values.shape[0],):
        raise ValueError("count-primary audit requires one series ID per panel row")
    provenance = dataset.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("count-primary audit requires source provenance")
    if provenance.get("dataset_id") != dataset_id:
        raise ValueError("source provenance dataset_id does not match")
    support = audit_model_train_target_support(
        values,
        split=split_contract,
        dataset_id=dataset_id,
        available_from=availability,
        series_ids=series_ids,
        provenance=provenance,
    )
    enforce_target_support(support)
    _assert_float32_exact_roundtrip(
        _availability_model_train_values(
            values, split=split_contract, available_from=availability
        ),
        label=f"{dataset_id} model_train target",
    )
    provided_support = dataset.get("support_audit")
    if isinstance(provided_support, Mapping) and dict(provided_support) != support:
        raise ValueError("stored support audit differs from the recomputed target audit")
    provided_split = dataset.get("split_validation")
    if isinstance(provided_split, Mapping) and dict(provided_split) != split:
        raise ValueError("stored split audit differs from the recomputed split audit")
    if not isinstance(source_manifest, Mapping):
        raise ValueError("count-primary audit requires a frozen source manifest")
    root_identity = source_manifest.get("repository_root_identity")
    if not isinstance(root_identity, Mapping) or not isinstance(root_identity.get("resolved_path"), str):
        raise ValueError("source manifest root identity is malformed")
    source_root = Path(str(root_identity["resolved_path"]))
    verify_source_manifest(source_root, source_manifest)
    manifest_files = {
        str(row["path"]): dict(row)
        for row in source_manifest.get("files", [])
        if isinstance(row, Mapping) and "path" in row
    }
    sources = provenance.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("source provenance requires at least one hashed source")
    canonical_sources: list[dict[str, object]] = []
    seen_sources: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("source provenance record is malformed")
        source_path = Path(str(source.get("path", ""))).resolve()
        if not source_path.is_relative_to(source_root.resolve()):
            raise ValueError("dataset source is outside the frozen source manifest root")
        relative = source_path.relative_to(source_root.resolve()).as_posix()
        if relative in seen_sources or relative not in manifest_files:
            raise ValueError("dataset source is duplicated or absent from the frozen source manifest")
        frozen_source = manifest_files[relative]
        actual = file_sha256(source_path).lower()
        expected = str(source.get("expected_sha256", "")).lower()
        if (
            source.get("status") != "PASS"
            or re.fullmatch(r"[0-9a-f]{64}", actual) is None
            or actual != expected
            or actual != str(source.get("sha256", "")).lower()
            or actual != str(frozen_source.get("sha256", "")).lower()
            or source_path.stat().st_size != int(frozen_source.get("bytes", -1))
        ):
            raise ValueError("source provenance hash audit is not eligible")
        canonical_sources.append(
            {
                "path": relative,
                "bytes": int(source_path.stat().st_size),
                "sha256": actual,
            }
        )
        seen_sources.add(relative)
    canonical_sources.sort(key=lambda row: str(row["path"]))
    panel_shape = [int(values.shape[0]), int(values.shape[1])]
    panel_binding_sha256 = _panel_binding_sha256(
        values, availability, series_ids
    )
    ordered_series_id_sha256 = _ordered_string_sha256(series_ids.tolist())
    canonical_contract = CANONICAL_COUNT_PRIMARY_PANEL_CONTRACTS.get(dataset_id)
    canonical_source_attested = canonical_contract is not None
    if canonical_contract is not None:
        if (
            panel_shape != canonical_contract["panel_shape"]
            or panel_binding_sha256
            != canonical_contract["panel_binding_sha256"]
            or ordered_series_id_sha256
            != canonical_contract["ordered_series_id_sha256"]
            or provenance.get("adapter_function")
            != canonical_contract["adapter_function"]
        ):
            raise ValueError(
                "panel is not the frozen canonical full-pool derivation of its source"
            )
    record: dict[str, object] = {
        "audit_type": "COUNT_PRIMARY_DATASET_AUDIT",
        "dataset_id": dataset_id,
        "status": "PASS",
        "geometry_status": "PASS",
        "count_primary_eligible": True,
        "confirmatory_eligible": bool(canonical_source_attested),
        "support_audit": dict(support),
        "split_validation": dict(split),
        "source_records": canonical_sources,
        "source_manifest_aggregate_sha256": str(source_manifest.get("aggregate_sha256", "")),
        "panel_shape": panel_shape,
        "panel_binding_sha256": panel_binding_sha256,
        "ordered_series_id_sha256": ordered_series_id_sha256,
        "canonical_source_attested": canonical_source_attested,
    }
    record["audit_sha256"] = _dataset_audit_hash(record)
    return record


def _verify_sealed_dataset_audit(raw: Mapping[str, object]) -> dict[str, object]:
    row = dict(raw)
    digest = row.get("audit_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("sealed dataset audit hash is missing")
    if digest != _dataset_audit_hash(row):
        raise ValueError("sealed dataset audit hash mismatch")
    audit_type = row.get("audit_type")
    if audit_type == "GEOMETRY_ONLY_AUDIT":
        required = {
            "audit_type", "dataset_id", "length", "lookback", "horizon",
            "required_length", "geometry_status", "status",
            "count_primary_eligible", "confirmatory_eligible", "audit_sha256",
        }
        if set(row) != required or row.get("count_primary_eligible") is not False or row.get("confirmatory_eligible") is not False:
            raise ValueError("sealed geometry dataset audit is malformed")
        if row.get("status") not in {"INSUFFICIENT_LENGTH", "SUPPORT_SOURCE_AUDIT_REQUIRED"}:
            raise ValueError("sealed geometry dataset audit status is invalid")
        return row
    if audit_type != "COUNT_PRIMARY_DATASET_AUDIT":
        raise ValueError("sealed dataset audit type is invalid")
    required = {
        "audit_type", "dataset_id", "status", "geometry_status",
        "count_primary_eligible", "confirmatory_eligible", "support_audit",
        "split_validation", "source_records", "source_manifest_aggregate_sha256",
        "panel_shape", "panel_binding_sha256", "ordered_series_id_sha256",
        "canonical_source_attested", "audit_sha256",
    }
    if set(row) != required:
        raise ValueError("sealed count-primary dataset audit is malformed")
    support = row.get("support_audit")
    split = row.get("split_validation")
    sources = row.get("source_records")
    if (
        row.get("status") != "PASS"
        or row.get("geometry_status") != "PASS"
        or row.get("count_primary_eligible") is not True
        or row.get("confirmatory_eligible")
        is not bool(row.get("canonical_source_attested") is True)
        or not isinstance(support, Mapping)
        or support.get("dataset_id") != row.get("dataset_id")
        or support.get("status") != "PASS"
        or support.get("target_scope") != "model_train"
        or support.get("count_primary_eligible") is not True
        or support.get("count_likelihood_index_exact") is not True
        or not isinstance(split, Mapping)
        or split.get("status") != "PASS"
        or not isinstance(sources, list)
        or not sources
        or not all(
            isinstance(source, Mapping)
            and re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) is not None
            for source in sources
        )
        or re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_manifest_aggregate_sha256", ""))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(row.get("panel_binding_sha256", ""))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(row.get("ordered_series_id_sha256", ""))) is None
        or not isinstance(row.get("panel_shape"), list)
        or len(row["panel_shape"]) != 2
    ):
        raise ValueError("sealed count-primary dataset audit is not canonical")
    canonical_contract = CANONICAL_COUNT_PRIMARY_PANEL_CONTRACTS.get(
        str(row.get("dataset_id"))
    )
    if canonical_contract is not None:
        if (
            row.get("canonical_source_attested") is not True
            or row.get("confirmatory_eligible") is not True
            or row.get("panel_shape") != canonical_contract["panel_shape"]
            or row.get("panel_binding_sha256")
            != canonical_contract["panel_binding_sha256"]
            or row.get("ordered_series_id_sha256")
            != canonical_contract["ordered_series_id_sha256"]
        ):
            raise ValueError(
                "sealed production dataset audit lacks canonical source attestation"
            )
    elif (
        row.get("canonical_source_attested") is not False
        or row.get("confirmatory_eligible") is not False
    ):
        raise ValueError("unattested dataset audit cannot be confirmatory")
    return row


def audit_favorita_count_eligibility(
    values: object, *, provenance: object
) -> dict[str, object]:
    """Outcome-blind guard for the preregistered rounded Favorita artifact."""
    return audit_target_support(values, dataset_id="favorita", provenance=provenance)


def _real_selection_manifest_hash(record: Mapping[str, object]) -> str:
    return _dataset_audit_hash(
        {
            key: value
            for key, value in record.items()
            if key != "selection_manifest_sha256"
        }
    )


def _seal_real_selection_record(record: dict[str, object]) -> dict[str, object]:
    sealed = dict(record)
    sealed["selection_manifest_sha256"] = _real_selection_manifest_hash(sealed)
    return sealed


def select_real_datasets(audits: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Choose M5 plus the first eligible non-M5 datasets in frozen priority order."""
    max_non_m5 = 2
    by_id: dict[str, dict[str, object]] = {}
    for raw in audits:
        if not isinstance(raw, Mapping):
            raise ValueError("sealed dataset audit must be a mapping")
        dataset_id = str(raw.get("dataset_id", ""))
        if dataset_id in by_id:
            raise ValueError(f"duplicate audit row for dataset_id={dataset_id}")
        by_id[dataset_id] = dict(raw)
    if set(by_id) != set(REAL_DATASET_PRIORITY):
        raise ValueError("complete canonical audit payload is required for every priority dataset")
    verified = {
        dataset_id: _verify_sealed_dataset_audit(by_id[dataset_id])
        for dataset_id in REAL_DATASET_PRIORITY
    }

    def eligible(dataset_id: str) -> bool:
        row = verified.get(dataset_id)
        return bool(
            row
            and row.get("audit_type") == "COUNT_PRIMARY_DATASET_AUDIT"
            and row.get("status") == "PASS"
            and row.get("count_primary_eligible") is True
            and row.get("confirmatory_eligible") is True
            and row.get("canonical_source_attested") is True
        )

    selected: list[dict[str, object]] = []
    if not eligible("m5"):
        ordered_audits = [verified[dataset_id] for dataset_id in REAL_DATASET_PRIORITY]
        return _seal_real_selection_record({
            "manifest_type": "COUNT_PRIMARY_DATASET_SELECTION_MANIFEST",
            "status": "REAL_CROSS_DATASET_EVIDENCE_LIMITED",
            "selected_dataset_ids": [],
            "selected": [],
            "audits": ordered_audits,
            "audit_manifest_sha256": _dataset_audit_hash({"audits": ordered_audits}),
            "requested_non_m5": max_non_m5,
            "eligible_non_m5_count": 0,
            "priority": REAL_DATASET_PRIORITY,
        })
    selected.append(verified["m5"])
    non_m5_passes: list[dict[str, object]] = []
    for dataset_id in REAL_DATASET_PRIORITY:
        if dataset_id == "m5":
            continue
        if eligible(dataset_id):
            non_m5_passes.append(verified[dataset_id])
        if len(non_m5_passes) >= max_non_m5:
            break
    selected.extend(non_m5_passes)
    limited = len(selected) < 2
    ordered_audits = [verified[dataset_id] for dataset_id in REAL_DATASET_PRIORITY]
    return _seal_real_selection_record({
        "manifest_type": "COUNT_PRIMARY_DATASET_SELECTION_MANIFEST",
        "status": "REAL_CROSS_DATASET_EVIDENCE_LIMITED" if limited else "PASS",
        "selected_dataset_ids": [str(row["dataset_id"]) for row in selected],
        "selected": selected,
        "audits": ordered_audits,
        "audit_manifest_sha256": _dataset_audit_hash({"audits": ordered_audits}),
        "requested_non_m5": max_non_m5,
        "eligible_non_m5_count": len(non_m5_passes),
        "priority": REAL_DATASET_PRIORITY,
    })


def verify_real_dataset_selection_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Recompute the fixed-priority selection and reject caller-declared scope."""
    if not isinstance(manifest, Mapping):
        raise TypeError("dataset selection manifest must be a mapping")
    supplied = dict(manifest)
    digest = supplied.get("selection_manifest_sha256")
    if (
        supplied.get("manifest_type")
        != "COUNT_PRIMARY_DATASET_SELECTION_MANIFEST"
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or digest != _real_selection_manifest_hash(supplied)
    ):
        raise ValueError("dataset selection manifest hash or schema mismatch")
    audits = supplied.get("audits")
    if not isinstance(audits, list):
        raise ValueError("dataset selection manifest lacks the complete audit list")
    reconstructed = select_real_datasets(audits)
    if supplied != reconstructed:
        raise ValueError(
            "dataset selection manifest differs from the fixed-priority reconstruction"
        )
    return reconstructed
