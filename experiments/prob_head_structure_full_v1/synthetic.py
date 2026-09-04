"""Synthetic DGP helpers for PROB-HEAD-STRUCTURE-FULL-v1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .runtime import verify_runtime_tier_decision


RECOVERED_SOURCE_SHA256 = (
    "0ad1b5607dd75bfe0c825fef778e55b17bf6667d16849a30beab295a66fd3e71"
)
RECOVERED_SOURCE_COMMIT = "03043927c1ea89e8569ab4e9d0bfe0b6d041a47f"
DATASET_VERSION = "prob-head-structure-full-v1-synthetic-1.0.0"

LAMBDA_LOW = 5.0
LAMBDA_HIGH = 15.0
SUPPORTED_D = (4, 8)
RHO_VALUES = (-0.8, 0.0, 0.8)
SERIES_LENGTH = 576
KNOWN_CHANGE_POINT = 288
KNOWN_CHANGE_SHIFT_TYPES = (
    "rho_I_positive",
    "rho_I_negative",
    "rho_M_positive",
    "rho_M_negative",
    "rho_I_and_rho_M",
    "no_change",
)
BURN_IN_EVENTS = 500
MAX_EVENT_MARGIN = 8

_GENERATED_BLOCK_ARRAY_FIELDS = (
    "base_series_index",
    "base_innovation_id",
    "y",
    "z",
    "positive_magnitude",
    "interval_states",
    "magnitude_states",
    "gaps",
    "event_times",
    "poisson_innovations",
    "train_positive_count",
)


@dataclass(frozen=True)
class SyntheticSplit:
    train: tuple[int, int] = (0, 380)
    validation: tuple[int, int] = (380, 408)
    warmup: tuple[int, int] = (408, 436)
    origins: tuple[int, ...] = (436, 464, 492, 520, 548)
    horizon: int = 28
    lookback: int = 96


@dataclass(frozen=True)
class KnownChangeGeometry:
    length: int = SERIES_LENGTH
    changepoint: int = KNOWN_CHANGE_POINT
    pre_change: tuple[int, int] = (0, KNOWN_CHANGE_POINT)
    post_change: tuple[int, int] = (KNOWN_CHANGE_POINT, SERIES_LENGTH)
    origins: tuple[int, ...] = tuple(range(120, 541, 28))
    horizon: int = 28
    lookback: int = 96


SYNTHETIC_SPLIT = SyntheticSplit()
CSYNTHETIC_GEOMETRY = KnownChangeGeometry()


def _rng(*parts: object) -> np.random.Generator:
    payload = "|".join(str(part) for part in parts)
    seed = int.from_bytes(
        hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest(), "big"
    )
    return np.random.default_rng(np.random.SeedSequence(seed))


def _base_identity_sha256(base: Mapping[str, object]) -> str:
    n_series = int(base["n_series"])
    base_id = str(base["base_id"])
    raw_indices = np.asarray(base.get("base_series_index"))
    if raw_indices.shape != (n_series,):
        raise ValueError("base innovation identities require one series index per series")
    try:
        numeric_indices = raw_indices.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("base innovation identities require integer series indices") from error
    if (
        not np.isfinite(numeric_indices).all()
        or not np.equal(numeric_indices, np.floor(numeric_indices)).all()
    ):
        raise ValueError("base innovation identities require integer series indices")
    indices = numeric_indices.astype(np.int64)
    if not np.array_equal(indices, np.arange(n_series, dtype=np.int64)):
        raise ValueError("base innovation identities require canonical ordered series indices")

    raw_ids = np.asarray(base.get("base_innovation_id"))
    if raw_ids.shape != (n_series,):
        raise ValueError("base innovation identities require one ID per series")
    ids = raw_ids.astype(str)
    expected_ids = np.asarray(
        [f"{base_id}_series{index:06d}" for index in range(n_series)]
    )
    if len(set(ids.tolist())) != n_series or not np.array_equal(ids, expected_ids):
        raise ValueError("base innovation identities must be unique and canonically ordered")

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(indices.astype("<i8")).tobytes())
    for identity in ids.tolist():
        encoded = identity.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _base_stream_fingerprint(base: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for key in ("d", "data_seed", "n_series", "length", "n_events"):
        digest.update(str(key).encode("utf-8"))
        digest.update(str(base[key]).encode("utf-8"))
    for key in (
        "interval_initial_uniforms",
        "interval_transition_uniforms",
        "magnitude_initial_uniforms",
        "magnitude_transition_uniforms",
        "poisson_low",
        "poisson_high",
    ):
        array = np.ascontiguousarray(np.asarray(base[key]))
        digest.update(str(key).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    digest.update(_base_identity_sha256(base).encode("ascii"))
    return digest.hexdigest()


def _generated_block_sha256(block: Mapping[str, object]) -> str:
    """Hash every raw field used by the specialization DGP audit."""
    digest = hashlib.sha256()
    for key in (
        "d",
        "data_seed",
        "base_id",
        "base_identity_sha256",
        "base_stream_fingerprint",
        "n_series",
        "series_length",
        "base_n_events",
        "rho_interval",
        "rho_magnitude",
        "rho_I",
        "rho_M",
    ):
        if key not in block:
            raise ValueError(f"generated block is missing raw field {key!r}")
        digest.update(key.encode("utf-8"))
        digest.update(repr(block[key]).encode("utf-8"))
    for key in _GENERATED_BLOCK_ARRAY_FIELDS:
        if key not in block:
            raise ValueError(f"generated block is missing raw array {key!r}")
        array = np.asarray(block[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
        if array.dtype.kind in {"U", "S", "O"}:
            for value in array.astype(str).reshape(-1).tolist():
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    slices = block.get("series_event_slices")
    if slices is None:
        raise ValueError("generated block is missing raw series_event_slices")
    slice_array = np.asarray(slices, dtype="<i8")
    digest.update(b"series_event_slices")
    digest.update(repr(tuple(int(value) for value in slice_array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(slice_array).tobytes())
    return digest.hexdigest()


def _validate_d(d: int) -> int:
    value = int(d)
    if value not in SUPPORTED_D:
        raise ValueError(f"d must be one of {SUPPORTED_D}, got {d!r}")
    return value


def _validate_rho(rho: float) -> float:
    value = float(rho)
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"rho must lie in [-1, 1], got {rho!r}")
    return value


def build_common_base(
    *, d: int, n_series: int, seed: int, length: int = SERIES_LENGTH
) -> dict[str, object]:
    """Build rho-independent base streams for one ``d`` and data seed."""
    gap_center = _validate_d(d)
    series_count = int(n_series)
    retained_length = int(length)
    if series_count <= 0:
        raise ValueError("n_series must be positive")
    if retained_length != SERIES_LENGTH:
        raise ValueError(f"synthetic length is frozen at {SERIES_LENGTH}")
    short_gap = gap_center - 1
    n_events = BURN_IN_EVENTS + retained_length // short_gap + MAX_EVENT_MARGIN
    shape = (series_count, n_events)
    base_id = f"synthetic_d{gap_center}_seed{int(seed)}"
    base = {
        "d": gap_center,
        "data_seed": int(seed),
        "base_id": base_id,
        "base_series_index": np.arange(series_count, dtype=np.int32),
        "base_innovation_id": np.asarray(
            [f"{base_id}_series{index:06d}" for index in range(series_count)]
        ),
        "n_series": series_count,
        "length": retained_length,
        "n_events": n_events,
        "interval_initial_uniforms": _rng(
            DATASET_VERSION, "interval_initial", gap_center, seed
        ).random(series_count),
        "interval_transition_uniforms": _rng(
            DATASET_VERSION, "interval_transition", gap_center, seed
        ).random(shape),
        "magnitude_initial_uniforms": _rng(
            DATASET_VERSION, "magnitude_initial", gap_center, seed
        ).random(series_count),
        "magnitude_transition_uniforms": _rng(
            DATASET_VERSION, "magnitude_transition", gap_center, seed
        ).random(shape),
        "poisson_low": _rng(DATASET_VERSION, "poisson_low", gap_center, seed).poisson(
            LAMBDA_LOW - 1.0, size=shape
        ),
        "poisson_high": _rng(
            DATASET_VERSION, "poisson_high", gap_center, seed
        ).poisson(LAMBDA_HIGH - 1.0, size=shape),
    }
    base["base_identity_sha256"] = _base_identity_sha256(base)
    base["base_stream_fingerprint"] = _base_stream_fingerprint(base)
    return base


def _states_from_uniforms(
    initial_uniforms: np.ndarray, transition_uniforms: np.ndarray, rho: float
) -> np.ndarray:
    stay = (1.0 + _validate_rho(rho)) / 2.0
    initial = np.where(np.asarray(initial_uniforms) < 0.5, 1, -1).astype(np.int8)
    transitions = np.asarray(transition_uniforms)
    states = np.empty(transitions.shape, dtype=np.int8)
    states[:, 0] = initial
    flips = transitions >= stay
    for event in range(1, transitions.shape[1]):
        states[:, event] = np.where(flips[:, event], -states[:, event - 1], states[:, event - 1])
    return states


def _states_from_piecewise_uniforms(
    initial_uniforms: np.ndarray,
    transition_uniforms: np.ndarray,
    rho_before: float,
    rho_after: float,
    changepoint_event: int,
) -> np.ndarray:
    transitions = np.asarray(transition_uniforms)
    states = np.empty(transitions.shape, dtype=np.int8)
    states[:, 0] = np.where(np.asarray(initial_uniforms) < 0.5, 1, -1).astype(np.int8)
    stay_before = (1.0 + _validate_rho(rho_before)) / 2.0
    stay_after = (1.0 + _validate_rho(rho_after)) / 2.0
    for event in range(1, transitions.shape[1]):
        stay = stay_before if event < changepoint_event else stay_after
        flip = transitions[:, event] >= stay
        states[:, event] = np.where(flip, -states[:, event - 1], states[:, event - 1])
    return states


def _retained_block_from_states(
    base: Mapping[str, np.ndarray | int],
    interval_states: np.ndarray,
    magnitude_states: np.ndarray,
    *,
    event_times: np.ndarray | None = None,
) -> dict[str, object]:
    stored_identity_sha256 = str(base.get("base_identity_sha256", ""))
    actual_identity_sha256 = _base_identity_sha256(base)
    if (
        not stored_identity_sha256
        or stored_identity_sha256 != actual_identity_sha256
    ):
        raise ValueError("base innovation identities do not match their frozen hash")
    stored_fingerprint = str(base.get("base_stream_fingerprint", ""))
    actual_fingerprint = _base_stream_fingerprint(base)
    if not stored_fingerprint or stored_fingerprint != actual_fingerprint:
        raise ValueError("synthetic common-base fingerprint mismatch")
    d = _validate_d(int(base["d"]))
    n_series = int(base["n_series"])
    length = int(base["length"])
    gaps = np.where(interval_states < 0, d - 1, d + 1).astype(np.int16)
    if event_times is None:
        absolute_event_times = np.cumsum(gaps, axis=1).astype(np.int32)
    else:
        absolute_event_times = np.asarray(event_times, dtype=np.int32)
        if absolute_event_times.shape != gaps.shape:
            raise ValueError("event_times must match the state arrays")
    poisson_low = np.asarray(base["poisson_low"], dtype=np.int16)
    poisson_high = np.asarray(base["poisson_high"], dtype=np.int16)
    magnitudes = 1 + np.where(magnitude_states < 0, poisson_low, poisson_high)

    y = np.zeros((n_series, length), dtype=np.float32)
    z = np.zeros((n_series, length), dtype=np.float32)
    positive = np.full((n_series, length), np.nan, dtype=np.float32)
    kept_interval_states: list[np.ndarray] = []
    kept_magnitude_states: list[np.ndarray] = []
    kept_gaps: list[np.ndarray] = []
    kept_times: list[np.ndarray] = []
    kept_absolute_indices: list[np.ndarray] = []
    slices: list[tuple[int, int]] = []
    offset = 0
    for series in range(n_series):
        times = absolute_event_times[series]
        burn_in_time = int(times[BURN_IN_EVENTS - 1])
        inside = (times >= burn_in_time) & (times < burn_in_time + length)
        index = times[inside] - burn_in_time
        values = magnitudes[series, inside].astype(np.float32)
        y[series, index] = values
        z[series, index] = 1.0
        positive[series, index] = values
        series_interval = interval_states[series, inside].astype(np.int8)
        series_magnitude = magnitude_states[series, inside].astype(np.int8)
        series_gaps = gaps[series, inside].astype(np.int16)
        series_times = index.astype(np.int32)
        series_absolute = np.flatnonzero(inside).astype(np.int32)
        kept_interval_states.append(series_interval)
        kept_magnitude_states.append(series_magnitude)
        kept_gaps.append(series_gaps)
        kept_times.append(series_times)
        kept_absolute_indices.append(series_absolute)
        slices.append((offset, offset + int(series_interval.size)))
        offset += int(series_interval.size)

    return {
        "d": d,
        "data_seed": int(base["data_seed"]),
        "base_id": str(base["base_id"]),
        "base_series_index": np.asarray(
            base.get("base_series_index", np.arange(n_series)), dtype=np.int32
        ).copy(),
        "base_innovation_id": np.asarray(
            base.get(
                "base_innovation_id",
                [f"{base['base_id']}_series{index:06d}" for index in range(n_series)],
            )
        ).astype(str),
        "base_identity_sha256": actual_identity_sha256,
        "base_stream_fingerprint": str(
            stored_fingerprint
        ),
        "n_series": n_series,
        "series_length": length,
        "base_n_events": int(base["n_events"]),
        "split": SYNTHETIC_SPLIT,
        "y": y,
        "z": z,
        "positive_magnitude": positive,
        "interval_states": np.concatenate(kept_interval_states).astype(np.int8),
        "magnitude_states": np.concatenate(kept_magnitude_states).astype(np.int8),
        "gaps": np.concatenate(kept_gaps).astype(np.int16),
        "event_times": np.concatenate(kept_times).astype(np.int32),
        "series_event_slices": tuple(slices),
        "series_interval_states": tuple(kept_interval_states),
        "series_magnitude_states": tuple(kept_magnitude_states),
        "series_event_times": tuple(kept_times),
        "series_absolute_event_indices": tuple(kept_absolute_indices),
        "poisson_innovations": np.stack([poisson_low, poisson_high], axis=0),
        "train_positive_count": z[:, : SYNTHETIC_SPLIT.train[1]].sum(axis=1).astype(np.int32),
    }


def transform_common_base(
    base: Mapping[str, np.ndarray | int],
    *,
    rho_interval: float,
    rho_magnitude: float,
) -> dict[str, object]:
    """Apply rho-specific Markov transitions to a rho-independent base."""
    d = _validate_d(int(base["d"]))
    n_series = int(base["n_series"])
    length = int(base["length"])
    interval_states = _states_from_uniforms(
        np.asarray(base["interval_initial_uniforms"]),
        np.asarray(base["interval_transition_uniforms"]),
        rho_interval,
    )
    magnitude_states = _states_from_uniforms(
        np.asarray(base["magnitude_initial_uniforms"]),
        np.asarray(base["magnitude_transition_uniforms"]),
        rho_magnitude,
    )
    block = _retained_block_from_states(base, interval_states, magnitude_states)
    result = {
        **block,
        "rho_interval": float(rho_interval),
        "rho_magnitude": float(rho_magnitude),
        "rho_I": float(rho_interval),
        "rho_M": float(rho_magnitude),
        "generator_argument_adapter": {
            "rho_interval": "rho_I",
            "rho_magnitude": "rho_M",
        },
    }
    result["generated_block_sha256"] = _generated_block_sha256(result)
    return result


def _balance_targets(d: int) -> dict[str, float]:
    expected_mean = 0.5 * LAMBDA_LOW + 0.5 * LAMBDA_HIGH
    low_variance = LAMBDA_LOW - 1.0
    high_variance = LAMBDA_HIGH - 1.0
    expected_variance = 0.5 * (
        low_variance + (LAMBDA_LOW - expected_mean) ** 2
    ) + 0.5 * (high_variance + (LAMBDA_HIGH - expected_mean) ** 2)
    return {
        "zero_rate": 1.0 - 1.0 / float(d),
        "positive_mean": expected_mean,
        "positive_variance": expected_variance,
    }


def _empirical_rho(states: np.ndarray) -> float:
    x = np.asarray(states, dtype=np.float64)
    x = x[x != 0]
    if x.size < 3:
        return np.nan
    return float(np.mean(x[:-1] * x[1:]))


def _nullable_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _mean_series_rho(block: Mapping[str, object], key: str) -> float:
    states = np.asarray(block[key], dtype=np.int8)
    if states.ndim != 1:
        raise ValueError(f"{key} must be a flat raw state vector")
    slices = block.get("series_event_slices")
    n_series = int(block.get("n_series", -1))
    if slices is None or len(slices) != n_series:  # type: ignore[arg-type]
        raise ValueError(f"{key} requires one raw event slice per series")
    canonical_end = 0
    values: list[float] = []
    for raw_slice in slices:  # type: ignore[union-attr]
        if len(raw_slice) != 2:
            raise ValueError("series_event_slices entries must contain start and end")
        start, end = (int(raw_slice[0]), int(raw_slice[1]))
        if start != canonical_end or end < start or end > states.size:
            raise ValueError("series_event_slices must be contiguous and canonical")
        values.append(_empirical_rho(states[start:end]))
        canonical_end = end
    if canonical_end != states.size:
        raise ValueError("series_event_slices must cover the complete state vector")
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def _verify_generated_block_against_source(
    block: Mapping[str, object],
    *,
    base_cache: dict[tuple[int, int, int], dict[str, object]],
) -> None:
    """Rebuild the deterministic source cell and compare its raw content seal."""
    if "y" not in block or "zero_rate" in block:
        raise ValueError("DGP audit requires raw generated blocks, not summary rows")
    stored_sha256 = block.get("generated_block_sha256")
    if not isinstance(stored_sha256, str) or len(stored_sha256) != 64:
        raise ValueError("raw generated block is missing its content SHA256")
    if stored_sha256 != _generated_block_sha256(block):
        raise ValueError("raw generated block content SHA256 mismatch")
    d = _validate_d(int(block["d"]))
    seed = int(block["data_seed"])
    n_series = int(block["n_series"])
    cache_key = (d, seed, n_series)
    base = base_cache.get(cache_key)
    if base is None:
        base = build_common_base(d=d, n_series=n_series, seed=seed)
        base_cache[cache_key] = base
    expected = transform_common_base(
        base,
        rho_interval=float(block["rho_interval"]),
        rho_magnitude=float(block["rho_magnitude"]),
    )
    if stored_sha256 != expected["generated_block_sha256"]:
        raise ValueError(
            "raw generated block does not match the deterministic verified source"
        )


def summarize_generated_block(
    block: Mapping[str, object], *, cell_id: str | None = None
) -> dict[str, object]:
    """Compute the DGP balance/audit statistics from a generated block."""
    base_identity_sha256 = _base_identity_sha256(block)
    stored_identity_sha256 = str(block.get("base_identity_sha256", ""))
    if stored_identity_sha256 and stored_identity_sha256 != base_identity_sha256:
        raise ValueError("base innovation identities do not match their frozen hash")
    y = np.asarray(block["y"], dtype=np.float64)
    n_series = int(block.get("n_series", -1))
    series_length = int(block.get("series_length", -1))
    if y.shape != (n_series, series_length):
        raise ValueError("raw generated y must match n_series and series_length")
    z = np.asarray(block.get("z", (y > 0.0).astype(np.float32)), dtype=np.float64)
    positive_panel = np.asarray(
        block.get("positive_magnitude", np.where(y > 0.0, y, np.nan)),
        dtype=np.float64,
    )
    finite_y = np.isfinite(y)
    nan_count = int(np.isnan(y).sum())
    inf_count = int(np.isinf(y).sum())
    negative_count = int((y[finite_y] < 0.0).sum())
    noninteger_count = int(
        (np.abs(y[finite_y] - np.rint(y[finite_y])) > 1e-6).sum()
    )
    y_support_valid = not (nan_count or inf_count or negative_count or noninteger_count)
    z_shape_valid = z.shape == y.shape
    z_binary_valid = z_shape_valid and bool(np.isin(z, [0.0, 1.0]).all())
    z_consistent = (
        z_binary_valid
        and bool(np.array_equal(z.astype(bool), np.asarray(y > 0.0, dtype=bool)))
    )
    positive_shape_valid = positive_panel.shape == y.shape
    if positive_shape_valid:
        positive_mask = (y > 0.0) & finite_y
        zero_mask = (y == 0.0) & finite_y
        positive_matches_y = bool(
            np.allclose(positive_panel[positive_mask], y[positive_mask])
        )
        zeros_are_missing = bool(np.isnan(positive_panel[zero_mask]).all())
    else:
        positive_matches_y = False
        zeros_are_missing = False
    positive_magnitude_consistent = (
        positive_shape_valid and positive_matches_y and zeros_are_missing
    )
    positives = positive_panel[np.isfinite(positive_panel)]
    positive_count = int(positives.size)
    positive_variance = float(positives.var(ddof=1)) if positive_count > 1 else None
    positive_mean = float(positives.mean()) if positive_count else None
    histogram_values, histogram_counts = (
        np.unique(positives.astype(np.int64), return_counts=True)
        if positive_count
        else (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    )
    n_observations = int(y.size)
    zero_count = int(((y == 0.0) & finite_y).sum())
    zero_rate = float(zero_count / n_observations) if n_observations else None
    realized_rho_interval = _nullable_float(_mean_series_rho(block, "interval_states"))
    realized_rho_magnitude = _nullable_float(_mean_series_rho(block, "magnitude_states"))
    positive_stats_valid = positive_count > 0
    positive_variance_valid = positive_count > 1
    cv2_valid = (
        positive_variance is not None
        and positive_mean is not None
        and positive_mean > 0.0
    )
    return {
        "cell_id": str(
            cell_id
            if cell_id is not None
            else f"d{int(block['d'])}_rhoI{float(block.get('rho_interval', np.nan))}_rhoM{float(block.get('rho_magnitude', np.nan))}"
        ),
        "d": int(block["d"]),
        "data_seed": int(block["data_seed"]) if "data_seed" in block else None,
        "base_id": str(block["base_id"]) if "base_id" in block else None,
        "base_stream_fingerprint": (
            str(block["base_stream_fingerprint"])
            if "base_stream_fingerprint" in block
            else None
        ),
        "base_identity_sha256": base_identity_sha256,
        "generated_block_sha256": block.get("generated_block_sha256"),
        "n_series": int(block.get("n_series", y.shape[0] if y.ndim == 2 else 0)),
        "series_length": int(
            block.get("series_length", y.shape[1] if y.ndim == 2 else y.size)
        ),
        "rho_interval": float(block.get("rho_interval", np.nan)),
        "rho_magnitude": float(block.get("rho_magnitude", np.nan)),
        "rho_I": float(block.get("rho_I", block.get("rho_interval", np.nan))),
        "rho_M": float(block.get("rho_M", block.get("rho_magnitude", np.nan))),
        "n_observations": n_observations,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "negative_count": negative_count,
        "noninteger_count": noninteger_count,
        "zero_count": zero_count,
        "positive_count": positive_count,
        "zero_rate": zero_rate,
        "positive_mean": positive_mean,
        "positive_variance": positive_variance,
        "positive_histogram": {
            int(value): int(count)
            for value, count in zip(histogram_values.tolist(), histogram_counts.tolist())
        },
        "ADI": float(n_observations / positive_count) if positive_count else None,
        "CV2": (
            float(positive_variance / (positive_mean * positive_mean))
            if cv2_valid
            else None
        ),
        "realized_rho_interval": realized_rho_interval,
        "realized_rho_magnitude": realized_rho_magnitude,
        "zero_rate_valid": zero_rate is not None,
        "positive_stats_valid": positive_stats_valid,
        "positive_variance_valid": positive_variance_valid,
        "ADI_valid": positive_stats_valid,
        "CV2_valid": cv2_valid,
        "realized_rho_interval_valid": realized_rho_interval is not None,
        "realized_rho_magnitude_valid": realized_rho_magnitude is not None,
        "y_support_valid": bool(y_support_valid),
        "z_consistent": bool(z_consistent),
        "positive_magnitude_consistent": bool(positive_magnitude_consistent),
    }


def _metric_float(row: Mapping[str, object], key: str) -> float | None:
    return _nullable_float(row.get(key))


def _metric_range(values: list[float | None], *, relative: bool) -> tuple[float | None, bool]:
    finite = [value for value in values if value is not None]
    if len(finite) != len(values) or not finite:
        return None, False
    span = float(max(finite) - min(finite))
    if not relative:
        return span, True
    denominator = float(np.mean(np.abs(finite)))
    if denominator <= 0.0:
        return None, False
    return float(span / denominator), True


def _rho_key(value: object) -> float:
    number = _nullable_float(value)
    if number is None:
        raise ValueError("DGP grid rows must include finite rho values")
    return round(number, 6)


def _validate_complete_dgp_grid(
    rows: list[Mapping[str, object]],
    *,
    expected_data_seeds: Sequence[object],
    expected_n_series: int,
) -> None:
    if not rows:
        raise ValueError("DGP audit requires a nonempty exact 18-cell grid")
    expected_series_count = int(expected_n_series)
    if expected_series_count <= 0:
        raise ValueError("expected_n_series must be positive")
    expected = {
        (round(float(rho_i), 6), round(float(rho_m), 6))
        for rho_i in RHO_VALUES
        for rho_m in RHO_VALUES
    }
    seen: set[tuple[object, int, float, float]] = set()
    by_seed_d: dict[tuple[object, int], list[Mapping[str, object]]] = {}
    for row in rows:
        if (
            row.get("data_seed") is None
            or row.get("base_id") is None
            or row.get("base_stream_fingerprint") is None
            or row.get("base_identity_sha256") is None
            or row.get("generated_block_sha256") is None
        ):
            raise ValueError(
                "DGP audit rows must preserve data_seed, base_id, common-base fingerprint, "
                "base innovation identity hash, and raw generated-block hash"
            )
        if int(row.get("n_series", -1)) != expected_series_count:
            raise ValueError(
                f"DGP audit row n_series must equal expected_n_series={expected_series_count}"
            )
        if int(row.get("series_length", -1)) != SERIES_LENGTH:
            raise ValueError(f"DGP audit row series_length must equal {SERIES_LENGTH}")
        if int(row.get("n_observations", -1)) != expected_series_count * SERIES_LENGTH:
            raise ValueError(
                "DGP audit row n_observations must equal expected_n_series * series_length"
            )
        seed = row["data_seed"]
        d = int(row["d"])
        cell = (_rho_key(row["rho_interval"]), _rho_key(row["rho_magnitude"]))
        duplicate_key = (seed, d, *cell)
        if duplicate_key in seen:
            raise ValueError(f"duplicate DGP audit cell: {duplicate_key}")
        seen.add(duplicate_key)
        by_seed_d.setdefault((seed, d), []).append(row)
    present_seeds = {seed for seed, _ in by_seed_d}
    required_seeds = set(expected_data_seeds)
    if not required_seeds:
        raise ValueError("expected_data_seeds must be explicit and nonempty")
    missing_seeds = sorted(required_seeds - present_seeds, key=str)
    extra_seeds = sorted(present_seeds - required_seeds, key=str)
    if missing_seeds:
        raise ValueError(f"missing DGP data_seed values: {missing_seeds}")
    if extra_seeds:
        raise ValueError(f"unexpected DGP data_seed values: {extra_seeds}")

    required_d = set(SUPPORTED_D)
    for seed in sorted(required_seeds, key=str):
        present_d = {d for row_seed, d in by_seed_d if row_seed == seed}
        if present_d != required_d:
            missing_d = sorted(required_d - present_d)
            extra_d = sorted(present_d - required_d)
            raise ValueError(
                f"DGP audit requires an exact 18-cell grid per seed with d={SUPPORTED_D}; "
                f"seed={seed}, missing_d={missing_d}, extra_d={extra_d}"
            )
        for d in SUPPORTED_D:
            group = by_seed_d[(seed, d)]
            _validate_d(d)
            _validate_dgp_base_group(seed, d, group, expected)


def _validate_dgp_base_group(
    seed: object,
    d: int,
    group: list[Mapping[str, object]],
    expected: set[tuple[float, float]],
) -> None:
    cells = {
        (_rho_key(row["rho_interval"]), _rho_key(row["rho_magnitude"]))
        for row in group
    }
    if cells != expected:
        missing = sorted(expected - cells)
        extra = sorted(cells - expected)
        raise ValueError(
            f"DGP audit requires an exact 9-cell grid per seed and d; "
            f"seed={seed}, d={d}, missing={missing}, extra={extra}"
        )
    fingerprints = {str(row["base_stream_fingerprint"]) for row in group}
    if len(fingerprints) != 1:
        raise ValueError(
            f"DGP audit rows for seed={seed}, d={d} must share one common-base fingerprint"
        )
    identity_hashes = {str(row["base_identity_sha256"]) for row in group}
    if len(identity_hashes) != 1:
        raise ValueError(
            f"DGP audit rows for seed={seed}, d={d} must share one base innovation identity hash"
        )
    base_ids = {str(row["base_id"]) for row in group}
    n_series = {int(row["n_series"]) for row in group}
    series_lengths = {int(row["series_length"]) for row in group}
    if len(base_ids) != 1 or len(n_series) != 1 or len(series_lengths) != 1:
        raise ValueError(
            f"DGP audit rows for seed={seed}, d={d} must share base_id and series dimensions"
        )


def audit_dgp_balance(
    rows: list[Mapping[str, object]],
    *,
    runtime_decision: Mapping[str, object],
    preregistration_path: Path,
) -> dict[str, object]:
    """Audit fixed marginal balance without deleting failed cells."""
    tolerances = {
        "zero_rate_abs": 0.02,
        "positive_mean_relative": 0.05,
        "positive_variance_relative": 0.10,
    }
    base_cache: dict[tuple[int, int, int], dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    for raw in rows:
        _verify_generated_block_against_source(raw, base_cache=base_cache)
        summaries.append(summarize_generated_block(raw))
    tier = verify_runtime_tier_decision(
        runtime_decision, preregistration_path=Path(preregistration_path)
    )
    tier_contract = tier["tier_contract"]
    if not isinstance(tier_contract, Mapping):
        raise ValueError("runtime-tier decision lacks a synthetic DGP contract")
    expected_data_seeds = tuple(tier_contract["synthetic_data_seeds"])
    expected_n_series = int(tier_contract["synthetic_series_per_cell"])
    _validate_complete_dgp_grid(
        summaries,
        expected_data_seeds=expected_data_seeds,
        expected_n_series=expected_n_series,
    )
    group_keys = sorted(
        {
            (int(row["d"]), row["data_seed"], str(row["base_stream_fingerprint"]))
            for row in summaries
        },
        key=lambda item: (item[0], str(item[1]), item[2]),
    )
    group_diagnostics: dict[tuple[int, object, str], dict[str, object]] = {}
    for key in group_keys:
        d, seed, fingerprint = key
        group = [
            row
            for row in summaries
            if int(row["d"]) == d
            and row["data_seed"] == seed
            and str(row["base_stream_fingerprint"]) == fingerprint
        ]
        zero_range, zero_range_valid = _metric_range(
            [_metric_float(row, "zero_rate") for row in group], relative=False
        )
        mean_range, mean_range_valid = _metric_range(
            [_metric_float(row, "positive_mean") for row in group], relative=True
        )
        variance_range, variance_range_valid = _metric_range(
            [_metric_float(row, "positive_variance") for row in group], relative=True
        )
        group_diagnostics[key] = {
            "balance_group_d": d,
            "balance_group_data_seed": seed,
            "balance_group_base_id": str(group[0]["base_id"]),
            "balance_group_base_stream_fingerprint": fingerprint,
            "balance_group_n_series": int(group[0]["n_series"]),
            "balance_group_series_length": int(group[0]["series_length"]),
            "zero_rate_pairwise_abs_range": zero_range,
            "positive_mean_pairwise_relative_range": mean_range,
            "positive_variance_pairwise_relative_range": variance_range,
            "pairwise_balance_metrics_valid": bool(
                zero_range_valid and mean_range_valid and variance_range_valid
            ),
        }

    audited = []
    for row in summaries:
        d = _validate_d(int(row["d"]))
        target = _balance_targets(d)
        zero_value = _metric_float(row, "zero_rate")
        mean_value = _metric_float(row, "positive_mean")
        variance_value = _metric_float(row, "positive_variance")
        zero_error = (
            abs(zero_value - target["zero_rate"]) if zero_value is not None else None
        )
        mean_rel = (
            abs(mean_value - target["positive_mean"]) / target["positive_mean"]
            if mean_value is not None
            else None
        )
        variance_rel = (
            abs(variance_value - target["positive_variance"]) / target["positive_variance"]
            if variance_value is not None
            else None
        )
        group_key = (d, row["data_seed"], str(row["base_stream_fingerprint"]))
        group = group_diagnostics[group_key]
        zero_pairwise = _metric_float(group, "zero_rate_pairwise_abs_range")
        mean_pairwise = _metric_float(group, "positive_mean_pairwise_relative_range")
        variance_pairwise = _metric_float(group, "positive_variance_pairwise_relative_range")
        pairwise_passed = (
            bool(group["pairwise_balance_metrics_valid"])
            and zero_pairwise is not None
            and mean_pairwise is not None
            and variance_pairwise is not None
            and zero_pairwise <= tolerances["zero_rate_abs"]
            and mean_pairwise <= tolerances["positive_mean_relative"]
            and variance_pairwise <= tolerances["positive_variance_relative"]
        )
        required_valid_flags = (
            "zero_rate_valid",
            "positive_stats_valid",
            "positive_variance_valid",
            "ADI_valid",
            "CV2_valid",
            "realized_rho_interval_valid",
            "realized_rho_magnitude_valid",
            "y_support_valid",
            "z_consistent",
            "positive_magnitude_consistent",
        )
        row_valid = all(bool(row.get(flag, False)) for flag in required_valid_flags)
        passed = (
            row_valid
            and pairwise_passed
        )
        audited.append(
            {
                **row,
                "target_zero_rate": target["zero_rate"],
                "target_positive_mean": target["positive_mean"],
                "target_positive_variance": target["positive_variance"],
                "zero_rate_abs_error": zero_error,
                "positive_mean_relative_error": mean_rel,
                "positive_variance_relative_error": variance_rel,
                "balance_metrics_valid": bool(
                    zero_error is not None
                    and mean_rel is not None
                    and variance_rel is not None
                ),
                "row_generation_valid": bool(row_valid),
                "theoretical_deviation_diagnostic_only": True,
                **group,
                "status": "PASS" if passed else "DGP_BALANCE_FAIL",
                "confirmatory_eligible": bool(passed),
            }
        )
    return {
        "runtime_decision_sha256": tier["runtime_decision_sha256"],
        "runtime_tier": tier["runtime_tier"],
        "preregistration_payload_sha256": tier["preregistration_payload_sha256"],
        "expected_data_seeds": list(expected_data_seeds),
        "expected_n_series": expected_n_series,
        "tolerances": tolerances,
        "rows": audited,
        "groups": {str(key): value for key, value in group_diagnostics.items()},
        "confirmatory_eligible": all(row["confirmatory_eligible"] for row in audited),
    }


def _known_change_rhos(shift_type: str) -> tuple[float, float, float, float]:
    if shift_type == "rho_I_positive":
        return 0.0, 0.8, 0.0, 0.0
    if shift_type == "rho_I_negative":
        return 0.0, -0.8, 0.0, 0.0
    if shift_type == "rho_M_positive":
        return 0.0, 0.0, 0.0, 0.8
    if shift_type == "rho_M_negative":
        return 0.0, 0.0, 0.0, -0.8
    if shift_type == "rho_I_and_rho_M":
        return 0.0, 0.8, 0.0, 0.8
    if shift_type == "no_change":
        return 0.0, 0.0, 0.0, 0.0
    raise ValueError(f"unknown known-change shift_type: {shift_type!r}")


def _known_change_block_sha256(block: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for key in (
        "d",
        "data_seed",
        "base_id",
        "base_identity_sha256",
        "base_stream_fingerprint",
        "n_series",
        "series_length",
        "base_n_events",
        "shift_type",
        "shift_token",
        "changepoint",
        "calendar_changepoint",
        "pre_rho_interval",
        "post_rho_interval",
        "pre_rho_magnitude",
        "post_rho_magnitude",
        "transition_kernel_convention",
    ):
        if key not in block:
            raise ValueError(f"known-change block is missing raw field {key!r}")
        digest.update(key.encode("utf-8"))
        digest.update(repr(block[key]).encode("utf-8"))
    for key in _GENERATED_BLOCK_ARRAY_FIELDS:
        if key not in block:
            raise ValueError(f"known-change block is missing raw array {key!r}")
        array = np.asarray(block[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
        if array.dtype.kind in {"U", "S", "O"}:
            for value in array.astype(str).reshape(-1).tolist():
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    for key in (
        "series_event_slices",
        "first_post_event_index_by_series",
        "state_continuity_checks",
    ):
        if key not in block:
            raise ValueError(f"known-change block is missing raw sequence {key!r}")
        array = np.asarray(block[key], dtype="<i8")
        digest.update(key.encode("utf-8"))
        digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _transition(previous: np.ndarray, uniforms: np.ndarray, stay: np.ndarray | float) -> np.ndarray:
    flip = np.asarray(uniforms) >= stay
    return np.where(flip, -previous, previous).astype(np.int8)


def _calendar_piecewise_interval_states(
    base: Mapping[str, np.ndarray | int], rho_before: float, rho_after: float
) -> tuple[np.ndarray, np.ndarray]:
    d = _validate_d(int(base["d"]))
    transitions = np.asarray(base["interval_transition_uniforms"])
    n_series, n_events = transitions.shape
    states = np.empty((n_series, n_events), dtype=np.int8)
    gaps = np.empty((n_series, n_events), dtype=np.int16)
    times = np.empty((n_series, n_events), dtype=np.int32)
    states[:, 0] = np.where(np.asarray(base["interval_initial_uniforms"]) < 0.5, 1, -1).astype(np.int8)
    gaps[:, 0] = np.where(states[:, 0] < 0, d - 1, d + 1).astype(np.int16)
    times[:, 0] = gaps[:, 0]
    stay_before = (1.0 + _validate_rho(rho_before)) / 2.0
    stay_after = (1.0 + _validate_rho(rho_after)) / 2.0
    for event in range(1, n_events):
        if event < BURN_IN_EVENTS:
            stay: np.ndarray | float = stay_before
        else:
            burn_in_time = times[:, BURN_IN_EVENTS - 1]
            previous_retained_time = times[:, event - 1] - burn_in_time
            stay = np.where(previous_retained_time >= KNOWN_CHANGE_POINT, stay_after, stay_before)
        states[:, event] = _transition(states[:, event - 1], transitions[:, event], stay)
        gaps[:, event] = np.where(states[:, event] < 0, d - 1, d + 1).astype(np.int16)
        times[:, event] = times[:, event - 1] + gaps[:, event]
    return states, times


def _calendar_piecewise_magnitude_states(
    base: Mapping[str, np.ndarray | int],
    event_times: np.ndarray,
    rho_before: float,
    rho_after: float,
) -> np.ndarray:
    transitions = np.asarray(base["magnitude_transition_uniforms"])
    states = np.empty(transitions.shape, dtype=np.int8)
    states[:, 0] = np.where(np.asarray(base["magnitude_initial_uniforms"]) < 0.5, 1, -1).astype(np.int8)
    burn_in_time = event_times[:, BURN_IN_EVENTS - 1]
    stay_before = (1.0 + _validate_rho(rho_before)) / 2.0
    stay_after = (1.0 + _validate_rho(rho_after)) / 2.0
    for event in range(1, transitions.shape[1]):
        if event < BURN_IN_EVENTS:
            stay: np.ndarray | float = stay_before
        else:
            previous_retained_time = event_times[:, event - 1] - burn_in_time
            stay = np.where(previous_retained_time >= KNOWN_CHANGE_POINT, stay_after, stay_before)
        states[:, event] = _transition(states[:, event - 1], transitions[:, event], stay)
    return states


def _first_post_continuity_checks(
    block: Mapping[str, object],
    base: Mapping[str, np.ndarray | int],
    event_times: np.ndarray,
    interval_states: np.ndarray,
    magnitude_states: np.ndarray,
    *,
    pre_i: float,
    post_i: float,
    pre_m: float,
    post_m: float,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    first_post_indices: list[int] = []
    checks: list[bool] = []
    interval_uniforms = np.asarray(base["interval_transition_uniforms"])
    magnitude_uniforms = np.asarray(base["magnitude_transition_uniforms"])
    for series, retained_times in enumerate(block["series_event_times"]):  # type: ignore[index]
        retained = np.asarray(retained_times, dtype=np.int32)
        candidates = np.flatnonzero(retained >= KNOWN_CHANGE_POINT)
        if candidates.size == 0:
            first_post_indices.append(-1)
            checks.append(False)
            continue
        first_post = int(candidates[0])
        first_post_indices.append(first_post)
        if first_post == 0:
            checks.append(False)
            continue
        absolute_index = int(block["series_absolute_event_indices"][series][first_post])  # type: ignore[index]
        previous_retained_time = int(retained[first_post - 1])
        interval_stay = (1.0 + (post_i if previous_retained_time >= KNOWN_CHANGE_POINT else pre_i)) / 2.0
        magnitude_stay = (1.0 + (post_m if previous_retained_time >= KNOWN_CHANGE_POINT else pre_m)) / 2.0
        expected_i = _transition(
            np.asarray([interval_states[series, absolute_index - 1]], dtype=np.int8),
            np.asarray([interval_uniforms[series, absolute_index]]),
            interval_stay,
        )[0]
        expected_m = _transition(
            np.asarray([magnitude_states[series, absolute_index - 1]], dtype=np.int8),
            np.asarray([magnitude_uniforms[series, absolute_index]]),
            magnitude_stay,
        )[0]
        checks.append(
            bool(
                interval_states[series, absolute_index] == expected_i
                and magnitude_states[series, absolute_index] == expected_m
            )
        )
    return tuple(first_post_indices), tuple(checks)


def build_known_change_block(
    *, d: int, shift_type: str, n_series: int, seed: int
) -> dict[str, object]:
    """Build a known-change block with a calendar changepoint and no state reinit."""
    pre_i, post_i, pre_m, post_m = _known_change_rhos(shift_type)
    base = build_common_base(d=d, n_series=n_series, seed=seed)
    interval_states, event_times = _calendar_piecewise_interval_states(base, pre_i, post_i)
    magnitude_states = _calendar_piecewise_magnitude_states(
        base, event_times, pre_m, post_m
    )
    block = _retained_block_from_states(
        base,
        interval_states,
        magnitude_states,
        event_times=event_times,
    )
    first_post_indices, continuity_checks = _first_post_continuity_checks(
        block,
        base,
        event_times,
        interval_states,
        magnitude_states,
        pre_i=pre_i,
        post_i=post_i,
        pre_m=pre_m,
        post_m=post_m,
    )
    result = {
        **block,
        "d": int(base["d"]),
        "shift_type": shift_type,
        "shift_token": shift_type,
        "changepoint": KNOWN_CHANGE_POINT,
        "calendar_changepoint": KNOWN_CHANGE_POINT,
        "split": CSYNTHETIC_GEOMETRY,
        "geometry": CSYNTHETIC_GEOMETRY,
        "state_continuity": bool(all(continuity_checks)),
        "state_continuity_checks": continuity_checks,
        "first_post_event_index_by_series": first_post_indices,
        "pre_rho_interval": pre_i,
        "post_rho_interval": post_i,
        "pre_rho_magnitude": pre_m,
        "post_rho_magnitude": post_m,
        "transition_kernel_convention": "previous_event_calendar_time",
        "transition_kernel_note": (
            "transition kernel is selected by the previous retained event calendar "
            "time; therefore the first event at/after cp may use pre kernel when "
            "the previous event was before cp"
        ),
        "pre_panel": {
            "y": block["y"][:, :KNOWN_CHANGE_POINT].copy(),
            "z": block["z"][:, :KNOWN_CHANGE_POINT].copy(),
        },
        "post_panel": {
            "y": block["y"][:, KNOWN_CHANGE_POINT:].copy(),
            "z": block["z"][:, KNOWN_CHANGE_POINT:].copy(),
        },
        "marginal_law": {
            "states": [-1, 1],
            "state_plus_probability": 0.5,
            "lambda_levels": [LAMBDA_LOW, LAMBDA_HIGH],
        },
    }
    result["known_change_block_sha256"] = _known_change_block_sha256(result)
    return result


def audit_known_change_panel(
    blocks: Sequence[Mapping[str, object]],
    *,
    runtime_decision: Mapping[str, object],
    preregistration_path: Path,
) -> dict[str, object]:
    """Verify the complete known-change panel before sensor evaluation."""
    tier = verify_runtime_tier_decision(
        runtime_decision, preregistration_path=Path(preregistration_path)
    )
    tier_contract = tier.get("tier_contract")
    if not isinstance(tier_contract, Mapping):
        raise ValueError("runtime-tier decision lacks a C-SYN panel contract")
    expected_seeds = tuple(int(value) for value in tier_contract["synthetic_data_seeds"])
    expected_n_series = int(tier_contract["synthetic_series_per_cell"])
    expected_keys = {
        (seed, d, shift_type)
        for seed in expected_seeds
        for d in SUPPORTED_D
        for shift_type in KNOWN_CHANGE_SHIFT_TYPES
    }
    if not blocks:
        raise ValueError("C-SYN audit requires the complete known-change panel")
    observed_keys: list[tuple[int, int, str]] = []
    rows: list[dict[str, object]] = []
    group_lineage: dict[tuple[int, int], set[tuple[str, str, str]]] = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("C-SYN panel entries must be raw generated blocks")
        stored_sha256 = block.get("known_change_block_sha256")
        if not isinstance(stored_sha256, str) or len(stored_sha256) != 64:
            raise ValueError("C-SYN raw block is missing its content SHA256")
        if stored_sha256 != _known_change_block_sha256(block):
            raise ValueError("C-SYN raw block content SHA256 mismatch")
        seed = int(block["data_seed"])
        d = _validate_d(int(block["d"]))
        shift_type = str(block["shift_type"])
        if shift_type not in KNOWN_CHANGE_SHIFT_TYPES:
            raise ValueError(f"unknown C-SYN shift type: {shift_type}")
        key = (seed, d, shift_type)
        observed_keys.append(key)
        n_series = int(block["n_series"])
        if n_series != expected_n_series:
            raise ValueError(
                "C-SYN n_series does not match the sealed runtime-tier decision"
            )
        expected = build_known_change_block(
            d=d, shift_type=shift_type, n_series=n_series, seed=seed
        )
        if stored_sha256 != expected["known_change_block_sha256"]:
            raise ValueError(
                "C-SYN raw block does not match the deterministic verified source"
            )
        y = np.asarray(block["y"])
        finite = np.isfinite(y)
        support_valid = bool(
            y.shape == (n_series, SERIES_LENGTH)
            and finite.all()
            and np.all(y >= 0.0)
            and np.all(np.abs(y - np.rint(y)) <= 1e-6)
        )
        geometry_valid = bool(
            int(block["series_length"]) == SERIES_LENGTH
            and int(block["changepoint"]) == KNOWN_CHANGE_POINT
            and int(block["calendar_changepoint"]) == KNOWN_CHANGE_POINT
            and block.get("split") == CSYNTHETIC_GEOMETRY
            and block.get("geometry") == CSYNTHETIC_GEOMETRY
            and block.get("state_continuity") is True
            and all(bool(value) for value in block["state_continuity_checks"])
        )
        pre_i, post_i, pre_m, post_m = _known_change_rhos(shift_type)
        shift_valid = bool(
            float(block["pre_rho_interval"]) == pre_i
            and float(block["post_rho_interval"]) == post_i
            and float(block["pre_rho_magnitude"]) == pre_m
            and float(block["post_rho_magnitude"]) == post_m
            and str(block["shift_token"]) == shift_type
        )
        no_change_valid = bool(
            shift_type != "no_change"
            or (pre_i == post_i == pre_m == post_m == 0.0)
        )
        if not (support_valid and geometry_valid and shift_valid and no_change_valid):
            raise ValueError("C-SYN raw block violates support, geometry, or shift contract")
        group_lineage.setdefault((seed, d), set()).add(
            (
                str(block["base_id"]),
                str(block["base_stream_fingerprint"]),
                str(block["base_identity_sha256"]),
            )
        )
        pre = y[:, :KNOWN_CHANGE_POINT]
        post = y[:, KNOWN_CHANGE_POINT:]
        rows.append(
            {
                "data_seed": seed,
                "d": d,
                "shift_type": shift_type,
                "n_series": n_series,
                "series_length": SERIES_LENGTH,
                "changepoint": KNOWN_CHANGE_POINT,
                "base_id": str(block["base_id"]),
                "base_stream_fingerprint": str(block["base_stream_fingerprint"]),
                "base_identity_sha256": str(block["base_identity_sha256"]),
                "known_change_block_sha256": stored_sha256,
                "pre_zero_rate": float(np.mean(pre == 0.0)),
                "post_zero_rate": float(np.mean(post == 0.0)),
                "pre_positive_mean": (
                    float(pre[pre > 0.0].mean()) if np.any(pre > 0.0) else None
                ),
                "post_positive_mean": (
                    float(post[post > 0.0].mean()) if np.any(post > 0.0) else None
                ),
                "support_valid": support_valid,
                "geometry_valid": geometry_valid,
                "shift_valid": shift_valid,
                "no_change_control_valid": no_change_valid,
                "status": "PASS",
            }
        )
    if len(set(observed_keys)) != len(observed_keys):
        raise ValueError("duplicate C-SYN panel cell")
    observed = set(observed_keys)
    if observed != expected_keys:
        missing = sorted(expected_keys - observed)
        extra = sorted(observed - expected_keys)
        raise ValueError(
            f"C-SYN audit requires the exact tier-bound panel; missing={missing}, extra={extra}"
        )
    if any(len(lineages) != 1 for lineages in group_lineage.values()):
        raise ValueError(
            "C-SYN shifts for each data_seed and d must share one common-base identity"
        )
    return {
        "runtime_decision_sha256": tier["runtime_decision_sha256"],
        "runtime_tier": tier["runtime_tier"],
        "preregistration_payload_sha256": tier["preregistration_payload_sha256"],
        "expected_data_seeds": list(expected_seeds),
        "expected_n_series": expected_n_series,
        "expected_cell_count": len(expected_keys),
        "rows": sorted(
            rows,
            key=lambda row: (
                int(row["data_seed"]),
                int(row["d"]),
                KNOWN_CHANGE_SHIFT_TYPES.index(str(row["shift_type"])),
            ),
        ),
        "status": "PASS",
        "confirmatory_eligible": True,
    }
