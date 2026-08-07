"""Forecast-origin conditional generation targets used by v3 evaluation.

Group A stores pointwise generator diagnostics in ``p_true``/``mu_true``.
Those arrays are useful for inspecting the realised rearrangement, but the
non-IID rows are not valid multi-step forecast targets: after the forecast
origin they incorporate states realised in the future.  This module builds a
history-only Markov reference instead.

The convention used throughout v3 is that an origin ``o`` is the first target
index.  The available history is therefore ``y[:o]`` and horizon offset 1
refers to ``y[o]``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import DEFAULT_CONFIG, ExperimentConfig


def _scalar(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(key)
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got {value.shape}")
    item = value.reshape(()).item()
    return item.decode() if isinstance(item, bytes) else item


def markov_transition(stationary_one: float, rho: float) -> np.ndarray:
    """Two-state transition matrix with stationary P(state=1) and ACF rho."""

    if not (0.0 < stationary_one < 1.0):
        raise ValueError("stationary_one must be in (0,1)")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0,1)")
    p01 = stationary_one * (1.0 - rho)
    p11 = stationary_one + rho * (1.0 - stationary_one)
    return np.asarray(
        ((1.0 - p01, p01), (1.0 - p11, p11)), dtype=np.float64
    )


def origin_target_mask(
    origins: np.ndarray,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Mask targets outside T or crossing a chronological split."""

    origins = np.asarray(origins, dtype=np.int32)
    if origins.ndim != 1:
        raise ValueError("origins must be one-dimensional")
    target_time = origins[:, None] + np.arange(config.horizon)[None, :]
    in_bounds = target_time < config.length
    split = np.zeros(config.length, dtype=np.int8)
    split[config.train_end:config.val_end] = 1
    split[config.val_end:] = 2
    clipped = np.minimum(target_time, config.length - 1)
    return (in_bounds & (split[clipped] == split[origins, None])).astype(np.uint8)


def _split_start(origin: int, config: ExperimentConfig) -> int:
    for start, end in config.split_bounds:
        if start <= origin < end:
            return start
    raise ValueError(f"origin {origin} is outside configured splits")


def _joint_transition(
    occurrence_transition: np.ndarray,
    magnitude_transition: np.ndarray,
) -> np.ndarray:
    """Calendar-time transition for (occurrence, last positive-size state).

    The magnitude state in the Group-A generator advances only on a positive
    occurrence.  State order is ``(z,s)=(0,0),(0,1),(1,0),(1,1)``.
    """

    result = np.zeros((4, 4), dtype=np.float64)
    for previous_z in (0, 1):
        for previous_s in (0, 1):
            source = 2 * previous_z + previous_s
            # No occurrence: the most recent positive-size state persists.
            result[source, previous_s] = occurrence_transition[previous_z, 0]
            # Occurrence: advance the event-indexed magnitude state.
            for next_s in (0, 1):
                destination = 2 + next_s
                result[source, destination] = (
                    occurrence_transition[previous_z, 1]
                    * magnitude_transition[previous_s, next_s]
                )
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("joint transition rows do not sum to one")
    return result


def _last_observed_magnitude_state(
    y: np.ndarray,
    series: int,
    split_start: int,
    origin: int,
    threshold: float,
) -> int | None:
    """Infer the current low/high state using positive values in history only."""

    positive_times = np.flatnonzero(y[series, split_start:origin] > 0.0)
    if not positive_times.size:
        return None
    last_time = split_start + int(positive_times[-1])
    return int(y[series, last_time] > threshold)


def group_a_conditional_target(
    data: Mapping[str, Any],
    origins: np.ndarray,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Group-A history-only E[y_future] and its split-safe mask.

    IID controls have the flat target ``p0*mu0``.  A non-IID occurrence-only
    row uses ``e_z^T P_z^h[:,1] * mu0``.  Magnitude rows use the calendar-time
    joint transition of occurrence and the event-indexed low/high magnitude
    state.  At a split's first origin no state from the preceding split is
    carried over; the stationary distribution is used.  Otherwise occurrence
    state and the most recent observed positive-size state are read only from
    ``y[:o]``/``z[:o]``.

    No value at index ``o`` or later is read by this function.
    """

    if str(_scalar(data, "group")) != "A":
        raise ValueError("group_a_conditional_target accepts Group A only")
    y = np.asarray(data["y"], dtype=np.float64)
    z = np.asarray(data["z"], dtype=np.uint8)
    expected_shape = (config.n_series, config.length)
    if y.shape != expected_shape or z.shape != expected_shape:
        raise ValueError(f"y/z must have shape {expected_shape}")
    origins = np.asarray(origins, dtype=np.int32)
    if origins.ndim != 1 or origins.size == 0:
        raise ValueError("origins must be a nonempty vector")
    if np.any(origins < config.lookback) or np.any(origins >= config.length):
        raise ValueError("origin is outside T or lacks configured lookback")

    variant = str(_scalar(data, "variant"))
    structure = str(_scalar(data, "structure"))
    strength = str(_scalar(data, "strength"))
    if variant not in {"iid", "non_iid"}:
        raise ValueError(f"unexpected Group-A variant {variant!r}")
    if structure not in {"occurrence", "magnitude", "both"}:
        raise ValueError(f"unexpected Group-A structure {structure!r}")
    if strength not in {"weak", "strong"}:
        raise ValueError(f"unexpected Group-A strength {strength!r}")

    p0 = float(config.a_occurrence_rate)
    mu0 = float(config.a_positive_mean)
    output = np.full(
        (config.n_series, origins.size, config.horizon),
        p0 * mu0,
        dtype=np.float64,
    )
    if variant == "iid":
        return output.astype(np.float32), origin_target_mask(origins, config)

    rho = config.a_weak_rho if strength == "weak" else config.a_strong_rho
    occurrence_rho = rho if structure in {"occurrence", "both"} else 0.0
    occurrence_transition = markov_transition(p0, occurrence_rho)

    if structure == "occurrence":
        for origin_index, origin_value in enumerate(origins):
            origin = int(origin_value)
            # Vectorise across series for non-boundary origins.
            if origin != _split_start(origin, config):
                distribution = np.zeros((config.n_series, 2), dtype=np.float64)
                distribution[np.arange(config.n_series), z[:, origin - 1]] = 1.0
                for horizon_index in range(config.horizon):
                    distribution = distribution @ occurrence_transition
                    output[:, origin_index, horizon_index] = (
                        distribution[:, 1] * mu0
                    )
            else:
                output[:, origin_index, :] = p0 * mu0
        return output.astype(np.float32), origin_target_mask(origins, config)

    high_probability = 1.0 - float(config.a_magnitude_high_quantile)
    magnitude_transition = markov_transition(high_probability, rho)
    joint_transition = _joint_transition(
        occurrence_transition, magnitude_transition
    )
    threshold = float(_scalar(data, "magnitude_threshold"))
    magnitude_means = np.asarray(
        (
            float(_scalar(data, "magnitude_low_mean")),
            float(_scalar(data, "magnitude_high_mean")),
        ),
        dtype=np.float64,
    )
    stationary_joint = np.asarray(
        (
            (1.0 - p0) * (1.0 - high_probability),
            (1.0 - p0) * high_probability,
            p0 * (1.0 - high_probability),
            p0 * high_probability,
        ),
        dtype=np.float64,
    )

    for origin_index, origin_value in enumerate(origins):
        origin = int(origin_value)
        split_start = _split_start(origin, config)
        distribution = np.zeros((config.n_series, 4), dtype=np.float64)
        split_boundary = origin == split_start
        for series in range(config.n_series):
            magnitude_state = _last_observed_magnitude_state(
                y, series, split_start, origin, threshold
            )
            if split_boundary:
                distribution[series] = stationary_joint
            elif magnitude_state is None:
                # Occurrence at o-1 is observed even when no positive size has
                # yet revealed the low/high state in this split.
                occurrence_state = int(z[series, origin - 1])
                distribution[series, 2 * occurrence_state] = 1.0 - high_probability
                distribution[series, 2 * occurrence_state + 1] = high_probability
            else:
                occurrence_state = int(z[series, origin - 1])
                distribution[series, 2 * occurrence_state + magnitude_state] = 1.0

        for horizon_index in range(config.horizon):
            # At a split start the first target has the stationary joint prior.
            # At later origins, advance one step from the observed occurrence
            # state and either the observed or stationary magnitude state.
            if horizon_index == 0 and not split_boundary:
                distribution = distribution @ joint_transition
            elif horizon_index > 0:
                distribution = distribution @ joint_transition
            output[:, origin_index, horizon_index] = (
                distribution[:, 2] * magnitude_means[0]
                + distribution[:, 3] * magnitude_means[1]
            )

    return output.astype(np.float32), origin_target_mask(origins, config)


def flatten_nonoverlap_target(
    target: np.ndarray,
    mask: np.ndarray,
    origins: np.ndarray,
    split_start: int,
    split_end: int,
) -> np.ndarray:
    """Flatten blocked origin/horizon targets into one chronological split."""

    target = np.asarray(target)
    mask = np.asarray(mask, dtype=bool)
    origins = np.asarray(origins, dtype=np.int32)
    if target.ndim != 3 or mask.shape not in {
        target.shape, target.shape[1:]
    }:
        raise ValueError("target/mask shapes are incompatible")
    broadcast_mask = np.broadcast_to(mask, target.shape)
    out = np.empty((target.shape[0], split_end - split_start), dtype=target.dtype)
    covered = np.zeros(split_end - split_start, dtype=bool)
    for window, origin_value in enumerate(origins):
        origin = int(origin_value)
        valid = int(broadcast_mask[0, window].sum())
        if valid and not np.all(broadcast_mask[:, window, :valid]):
            raise ValueError("target mask differs across series")
        destination = origin - split_start
        if destination < 0 or destination + valid > out.shape[1]:
            raise ValueError("origin target lies outside requested split")
        out[:, destination:destination + valid] = target[:, window, :valid]
        covered[destination:destination + valid] = True
    if not covered.all():
        raise ValueError("origins do not cover the complete split")
    return out


__all__ = [
    "markov_transition",
    "origin_target_mask",
    "group_a_conditional_target",
    "flatten_nonoverlap_target",
]
