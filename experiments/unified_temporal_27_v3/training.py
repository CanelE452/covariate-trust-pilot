"""Seed-0 CUDA training, validation-only selection, and v3 evaluation.

Official stage functions require a passed 27-row generation report.  The MSE
gate and lambda selection inspect validation predictions only; test targets are
used only after a method/configuration is fixed.  Importing this module does
not generate data, start a fit, or write an artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rcoi.seed import seed_everything

from .config import (
    DATA_DIR,
    DEFAULT_CONFIG,
    LOGS_DIR,
    RESULTS_DIR,
    ROOT,
    VALIDATION_PATH,
    ExperimentConfig,
)
from .conditional_targets import group_a_conditional_target
from .model import UnifiedDLinear, count_parameters
from .scenarios import SCENARIOS, ScenarioSpec, build_scenarios


METHODS = ("mse", "gamma_hurdle", "gamma_gap")
RESULTS_PATH = ROOT / "results_seed0.csv"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
MSE_GATE_PATH = ROOT / "mse_gate_27_seed0.csv"
LAMBDA_SELECTION_PATH = ROOT / "lambda_selection.csv"

GATE_SENTINELS = (
    "A04_occurrence_strong_non_iid",
    "A08_magnitude_strong_non_iid",
    "A12_both_strong_non_iid",
    "B07_season_strong_jitter0",
    "C01_regular_season_spike",
)
LAMBDA_SELECTION_IDS = tuple(
    spec.scenario_id
    for spec in SCENARIOS
    if (spec.group == "A" and spec.variant == "non_iid") or spec.group == "C"
)


@dataclass(frozen=True)
class WindowArrays:
    """Padded fixed-horizon windows and a validity mask for one split."""

    history: np.ndarray
    target: np.ndarray
    occurrence: np.ndarray
    target_mask: np.ndarray
    gap: np.ndarray
    gap_event_observed: np.ndarray
    gap_censor_lower: np.ndarray
    scale: np.ndarray
    origins: np.ndarray
    valid_lengths: np.ndarray
    n_series: int
    split_start: int
    split_end: int

    @property
    def n_origins(self) -> int:
        return int(self.origins.size)


def train_origins(config: ExperimentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Dense train origins whose full H targets remain in train."""

    return np.arange(
        config.lookback,
        config.train_end - config.horizon + 1,
        dtype=np.int32,
    )


def nonoverlap_origins(start: int, end: int, horizon: int) -> np.ndarray:
    """Blocked origins including the final partial horizon."""

    if not (0 <= start < end) or horizon <= 0:
        raise ValueError("expected 0 <= start < end and horizon > 0")
    return np.arange(start, end, horizon, dtype=np.int32)


def train_scale(
    data: Mapping[str, np.ndarray],
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Per-series mean scale using train observations only."""

    y = np.asarray(data["y"], dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < config.train_end:
        raise ValueError("y does not contain the configured train interval")
    return np.maximum(y[:, :config.train_end].mean(axis=1), 1.0).astype(np.float32)


def _window_gap_details(
    occurrence: np.ndarray,
    mask: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gap sentinel, event indicator, and right-censor lower bound.

    ``gap`` retains the historical H+1 sentinel for storage compatibility.
    It is an exact distance only when ``event_observed`` is true.  Otherwise
    the observation says only that the distance is at least
    ``valid_length + 1``.  The latter matters for the final partial block,
    where fewer than H future values are observable inside the split.
    """

    if occurrence.shape != mask.shape or occurrence.ndim != 3:
        raise ValueError("occurrence and mask must have equal (N,W,H) shapes")
    if horizon <= 0 or occurrence.shape[-1] != horizon:
        raise ValueError("the final window dimension must equal horizon")
    valid_lengths = mask.sum(axis=-1)
    if np.any(valid_lengths <= 0):
        raise ValueError("every gap window must contain at least one valid target")
    prefix = np.arange(horizon)[None, None, :] < valid_lengths[..., None]
    if not np.array_equal(mask, prefix):
        raise ValueError("target masks must be contiguous prefixes")
    hit = (occurrence > 0.5) & mask
    any_hit = hit.any(axis=-1)
    first = hit.argmax(axis=-1) + 1
    gap = np.where(any_hit, first, horizon + 1).astype(np.float32)
    censor_lower = np.where(any_hit, first, valid_lengths + 1).astype(np.float32)
    return gap, any_hit.astype(bool), censor_lower


def _window_gap(occurrence: np.ndarray, mask: np.ndarray, horizon: int) -> np.ndarray:
    """Backward-compatible H+1-sentinel view of :func:`_window_gap_details`."""

    return _window_gap_details(occurrence, mask, horizon)[0]


def make_windows(
    data: Mapping[str, np.ndarray],
    origins: np.ndarray,
    split_start: int,
    split_end: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
    scale: np.ndarray | None = None,
) -> WindowArrays:
    """Build windows whose forward input ends strictly before each origin."""

    y = np.asarray(data["y"], dtype=np.float32)
    z = np.asarray(data["z"], dtype=np.float32)
    if y.ndim != 2 or y.shape != z.shape:
        raise ValueError("y and z must have identical (N,T) shapes")
    n_series, length = y.shape
    origins = np.asarray(origins, dtype=np.int32)
    if origins.ndim != 1 or origins.size == 0:
        raise ValueError("origins must be a nonempty vector")
    if not (0 <= split_start < split_end <= length):
        raise ValueError("invalid split bounds")
    if np.any(origins < max(split_start, config.lookback)) or np.any(origins >= split_end):
        raise ValueError("origin is outside its split or lacks lookback")
    if scale is None:
        scale = train_scale(data, config)
    scale = np.asarray(scale, dtype=np.float32)
    if scale.shape != (n_series,) or np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("scale must be finite positive shape (N,)")

    n_origins = origins.size
    history = np.empty((n_series, n_origins, config.lookback), dtype=np.float32)
    target = np.zeros((n_series, n_origins, config.horizon), dtype=np.float32)
    occurrence = np.zeros_like(target)
    mask = np.zeros_like(target, dtype=bool)
    valid_lengths = np.minimum(config.horizon, split_end - origins).astype(np.int32)
    for window, (origin, valid) in enumerate(zip(origins, valid_lengths, strict=True)):
        history[:, window] = y[:, origin - config.lookback:origin]
        target[:, window, :valid] = y[:, origin:origin + valid]
        occurrence[:, window, :valid] = z[:, origin:origin + valid]
        mask[:, window, :valid] = True
    gap, gap_event_observed, gap_censor_lower = _window_gap_details(
        occurrence, mask, config.horizon
    )
    if "d_true" in data:
        stored = np.asarray(data["d_true"], dtype=np.float32)[:, origins]
        if stored.shape != gap.shape or not np.array_equal(stored, gap):
            raise ValueError("stored d_true violates the target-start origin contract")
    return WindowArrays(
        history=history.reshape(n_series * n_origins, config.lookback),
        target=target.reshape(n_series * n_origins, config.horizon),
        occurrence=occurrence.reshape(n_series * n_origins, config.horizon),
        target_mask=mask.reshape(n_series * n_origins, config.horizon),
        gap=gap.reshape(n_series * n_origins),
        gap_event_observed=gap_event_observed.reshape(n_series * n_origins),
        gap_censor_lower=gap_censor_lower.reshape(n_series * n_origins),
        scale=np.repeat(scale, n_origins).astype(np.float32),
        origins=origins.copy(),
        valid_lengths=valid_lengths,
        n_series=n_series,
        split_start=int(split_start),
        split_end=int(split_end),
    )


def _tensor_dataset(windows: WindowArrays) -> TensorDataset:
    # Only history and train-only scale are passed to model.forward.  Every
    # future quantity below is a loss/evaluation label.
    return TensorDataset(
        torch.from_numpy(windows.history),
        torch.from_numpy(windows.target),
        torch.from_numpy(windows.occurrence),
        torch.from_numpy(windows.target_mask),
        torch.from_numpy(windows.gap),
        torch.from_numpy(windows.gap_event_observed),
        torch.from_numpy(windows.gap_censor_lower),
        torch.from_numpy(windows.scale),
    )


def estimate_phi_train_positive(
    data: Mapping[str, np.ndarray],
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> float:
    """Pooled positive CV² from unique train observations only."""

    y = np.asarray(data["y"], dtype=np.float64)[:, :config.train_end]
    z = np.asarray(data["z"], dtype=np.float64)[:, :config.train_end]
    positive = y[z > 0.5]
    if positive.size < 2 or np.any(positive <= 0.0) or not np.isfinite(positive).all():
        raise ValueError("at least two finite positive train values are required")
    mean = float(positive.mean())
    phi = float(positive.var(ddof=0) / mean**2)
    if not math.isfinite(phi) or phi <= 0.0:
        raise ValueError(f"invalid train-only phi: {phi}")
    return phi


def _initial_statistics(
    data: Mapping[str, np.ndarray],
    scale: np.ndarray,
    train_windows_: WindowArrays,
    config: ExperimentConfig,
) -> dict[str, float]:
    y = np.asarray(data["y"], dtype=np.float64)[:, :config.train_end]
    z = np.asarray(data["z"], dtype=np.float64)[:, :config.train_end]
    normalized = y / np.asarray(scale, dtype=np.float64)[:, None]
    return {
        "phi": estimate_phi_train_positive(data, config),
        "train_positive_mean": float(y[z > 0.5].mean()),
        "initial_normalized_positive_mean": float(normalized[z > 0.5].mean()),
        "initial_occurrence_rate": float(z.mean()),
        "initial_gap_mean": float(train_windows_.gap.mean()),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("masked tensors differ in shape")
    weight = mask.to(values.dtype)
    if weight.sum().item() <= 0:
        raise ValueError("masked reduction has no valid values")
    return (values * weight).sum() / weight.sum()


def gamma_nll(
    mu_prediction: torch.Tensor,
    target: torch.Tensor,
    positive: torch.Tensor,
    dispersion: torch.Tensor,
) -> torch.Tensor:
    """Negative log GammaPDF under mean/CV² parameterization."""

    if not (mu_prediction.shape == target.shape == positive.shape):
        raise ValueError("Gamma loss tensors must have identical shapes")
    if not positive.any():
        return mu_prediction.sum() * 0.0
    mu, y = mu_prediction[positive], target[positive]
    if torch.any(mu <= 0.0) or torch.any(y <= 0.0):
        raise ValueError("Gamma loss requires positive mu and y")
    phi = dispersion.to(dtype=mu.dtype, device=mu.device).clamp_min(1e-6)
    shape = 1.0 / phi
    return (
        torch.lgamma(shape)
        + shape * torch.log(phi * mu)
        + y / (phi * mu)
        - (shape - 1.0) * torch.log(y)
    ).mean()


def scalar_gap_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    event_observed: torch.Tensor | None = None,
    censor_lower: torch.Tensor | None = None,
) -> torch.Tensor:
    """Censor-aware log-Huber loss for one gap prediction per origin.

    Observed next occurrences use the symmetric residual.  A right-censored
    origin contributes only when the prediction is shorter than the known
    lower bound; predictions at or above that bound receive zero loss.  The
    optional arguments preserve the old all-events-exact calling convention.
    """

    if prediction.shape != target.shape:
        raise ValueError("gap prediction and target shapes differ")
    if torch.any(prediction < 1.0) or torch.any(target < 1.0):
        raise ValueError("gap prediction and target must be at least one")
    if event_observed is None:
        event_observed = torch.ones_like(target, dtype=torch.bool)
    if censor_lower is None:
        censor_lower = target
    if event_observed.shape != target.shape or censor_lower.shape != target.shape:
        raise ValueError("gap censoring tensors differ in shape")
    event_observed = event_observed.to(dtype=torch.bool, device=prediction.device)
    censor_lower = censor_lower.to(dtype=prediction.dtype, device=prediction.device)
    if torch.any(censor_lower < 1.0):
        raise ValueError("gap censor lower bounds must be at least one")
    exact_residual = torch.log1p(prediction) - torch.log1p(target)
    censored_shortfall = torch.relu(
        torch.log1p(censor_lower) - torch.log1p(prediction)
    )
    residual = torch.where(event_observed, exact_residual, censored_shortfall)
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="mean")


def objective(
    outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    occurrence: torch.Tensor,
    target_mask: torch.Tensor,
    gap: torch.Tensor,
    method: str,
    config: ExperimentConfig = DEFAULT_CONFIG,
    *,
    gap_event_observed: torch.Tensor | None = None,
    gap_censor_lower: torch.Tensor | None = None,
    group: str = "",
) -> tuple[torch.Tensor, dict[str, float]]:
    if method not in METHODS:
        raise ValueError(method)
    if method == "mse":
        mse = _masked_mean((outputs["point_prediction"] - target).square(), target_mask)
        return mse, {
            "mse_loss": float(mse.detach().cpu()),
            "occurrence_loss": np.nan,
            "gamma_loss": np.nan,
            "gap_loss": np.nan,
        }
    group_b = str(group) == "B"
    if group_b:
        # Group B is continuous seasonality-only data.  There is no hurdle or
        # next-occurrence task: p is conceptually fixed to one and both BCE and
        # Gap are excluded from the gradient path.
        occurrence_loss = outputs["occurrence_logits"].sum() * 0.0
    else:
        occurrence_element = F.binary_cross_entropy_with_logits(
            outputs["occurrence_logits"], occurrence, reduction="none"
        )
        occurrence_loss = _masked_mean(occurrence_element, target_mask)
    positive = (occurrence > 0.5) & target_mask
    magnitude_loss = gamma_nll(
        outputs["mu_prediction"], target, positive, outputs["dispersion"]
    )
    if group_b:
        gap_loss = outputs["gap_prediction"].sum() * 0.0
        total = config.lambda_mag * magnitude_loss
    else:
        gap_loss = scalar_gap_loss(
            outputs["gap_prediction"], gap,
            gap_event_observed, gap_censor_lower,
        )
        total = occurrence_loss + config.lambda_mag * magnitude_loss
    if method == "gamma_gap" and not group_b:
        total = total + config.lambda_gap * gap_loss
    return total, {
        "mse_loss": np.nan,
        "occurrence_loss": (
            np.nan if group_b else float(occurrence_loss.detach().cpu())
        ),
        "gamma_loss": float(magnitude_loss.detach().cpu()),
        "gap_loss": np.nan if group_b else float(gap_loss.detach().cpu()),
    }


def _point(
    outputs: Mapping[str, torch.Tensor], method: str, group: str = ""
) -> torch.Tensor:
    if method == "mse":
        return outputs["point_prediction"]
    if str(group) == "B":
        return outputs["mu_prediction"]
    return outputs["demand_prediction"]


def _evaluate_windows(
    model: UnifiedDLinear,
    windows: WindowArrays,
    method: str,
    device: torch.device,
    batch_size: int,
    group: str = "",
) -> dict[str, float]:
    loader = DataLoader(_tensor_dataset(windows), batch_size=batch_size, shuffle=False)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (
            history, target, _occurrence, mask, _gap,
            _gap_event, _gap_censor_lower, scale,
        ) in loader:
            outputs = model(history.to(device), scale.to(device))
            predictions.append(_point(outputs, method, group).cpu())
            targets.append(target)
            masks.append(mask)
    prediction = torch.cat(predictions).double()
    target = torch.cat(targets).double()
    mask = torch.cat(masks).bool()
    return {"raw_mse": float(_masked_mean((prediction - target).square(), mask))}


def _train_one_unchecked(
    data: Mapping[str, np.ndarray],
    method: str,
    device: torch.device,
    config: ExperimentConfig,
) -> tuple[UnifiedDLinear, pd.DataFrame, dict[str, float | int]]:
    """Fit one model; stage-level validation gates are enforced by callers."""

    if method not in METHODS or config.train_seed != 0:
        raise ValueError("v3 accepts a known method and train_seed=0 only")
    seed_everything(0, deterministic=True)
    group = str(_scalar(data, "group", ""))
    scale = train_scale(data, config)
    train_windows_ = make_windows(
        data, train_origins(config), 0, config.train_end, config, scale
    )
    validation_windows = make_windows(
        data,
        nonoverlap_origins(config.train_end, config.val_end, config.horizon),
        config.train_end,
        config.val_end,
        config,
        scale,
    )
    stats = _initial_statistics(data, scale, train_windows_, config)
    model = UnifiedDLinear(
        config.lookback,
        config.horizon,
        config.moving_average_kernel,
        dispersion=stats["phi"],
        initial_normalized_positive_mean=stats["initial_normalized_positive_mean"],
        initial_occurrence_rate=stats["initial_occurrence_rate"],
        initial_gap_mean=stats["initial_gap_mean"],
        epsilon=max(config.epsilon, 1e-8),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(
        _tensor_dataset(train_windows_),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
        num_workers=0,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    best_mse = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    logs: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(config.max_epochs):
        model.train()
        pieces: dict[str, list[float]] = {
            key: [] for key in (
                "train_loss", "mse_loss", "occurrence_loss", "gamma_loss", "gap_loss"
            )
        }
        for (
            history, target, occurrence, mask, gap,
            gap_event_observed, gap_censor_lower, scale_batch,
        ) in loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            occurrence = occurrence.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            gap = gap.to(device, non_blocking=True)
            gap_event_observed = gap_event_observed.to(device, non_blocking=True)
            gap_censor_lower = gap_censor_lower.to(device, non_blocking=True)
            scale_batch = scale_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(history, scale_batch)
            loss, components = objective(
                outputs, target, occurrence, mask, gap, method, config,
                gap_event_observed=gap_event_observed,
                gap_censor_lower=gap_censor_lower,
                group=group,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {method} loss")
            loss.backward()
            optimizer.step()
            pieces["train_loss"].append(float(loss.detach().cpu()))
            for key, value in components.items():
                pieces[key].append(value)
        validation = _evaluate_windows(
            model, validation_windows, method, device, config.batch_size, group
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "method": method,
            "train_loss": float(np.mean(pieces["train_loss"])),
            "validation_raw_mse": validation["raw_mse"],
            "phi": stats["phi"],
        }
        for key in ("mse_loss", "occurrence_loss", "gamma_loss", "gap_loss"):
            values = np.asarray(pieces[key], dtype=np.float64)
            row[key] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        logs.append(row)
        if validation["raw_mse"] < best_mse - 1e-6:
            best_mse = validation["raw_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("fit produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    info: dict[str, float | int] = {
        "best_epoch": best_epoch,
        "epochs_ran": len(logs),
        "training_wall_seconds": time.time() - started,
        "phi": stats["phi"],
        "phi_train_positive_mean": stats["train_positive_mean"],
        "train_occurrence_rate": stats["initial_occurrence_rate"],
        "train_gap_mean": stats["initial_gap_mean"],
        "best_validation_raw_mse": best_mse,
        "parameter_count": count_parameters(model),
    }
    return model, pd.DataFrame(logs), info


def _restore_full_split(values: np.ndarray, windows: WindowArrays) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[:2] != (windows.n_series, windows.n_origins):
        raise ValueError("horizon values have an incompatible shape")
    output = np.empty(
        (windows.n_series, windows.split_end - windows.split_start), dtype=values.dtype
    )
    covered = np.zeros(windows.split_end - windows.split_start, dtype=bool)
    for window, (origin, valid) in enumerate(
        zip(windows.origins, windows.valid_lengths, strict=True)
    ):
        start = int(origin - windows.split_start)
        output[:, start:start + valid] = values[:, window, :valid]
        covered[start:start + valid] = True
    if not covered.all():
        raise RuntimeError("nonoverlap origins did not cover the complete split")
    return output


def predict_split(
    model: UnifiedDLinear,
    data: Mapping[str, np.ndarray],
    method: str,
    split_start: int,
    split_end: int,
    device: torch.device,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, np.ndarray]:
    if method not in METHODS:
        raise ValueError(method)
    scale = train_scale(data, config)
    group = str(_scalar(data, "group", ""))
    origins = nonoverlap_origins(split_start, split_end, config.horizon)
    windows = make_windows(data, origins, split_start, split_end, config, scale)
    loader = DataLoader(_tensor_dataset(windows), batch_size=config.batch_size, shuffle=False)
    point, probability, magnitude, gap = [], [], [], []
    model.eval()
    with torch.no_grad():
        for (
            history, _target, _occurrence, _mask, _gap,
            _gap_event, _gap_censor_lower, scale_batch,
        ) in loader:
            # Future labels are deliberately discarded before forward.
            outputs = model(history.to(device), scale_batch.to(device))
            point.append(_point(outputs, method, group).cpu().numpy())
            if group == "B" and method != "mse":
                probability.append(torch.ones_like(outputs["p_prediction"]).cpu().numpy())
            else:
                probability.append(outputs["p_prediction"].cpu().numpy())
            magnitude.append(outputs["mu_prediction"].cpu().numpy())
            gap.append(outputs["gap_prediction"].cpu().numpy())
    n, n_origins, horizon = windows.n_series, windows.n_origins, config.horizon

    def horizon_array(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks, axis=0).reshape(n, n_origins, horizon)

    prediction_horizon = horizon_array(point)
    probability_horizon = horizon_array(probability)
    magnitude_horizon = horizon_array(magnitude)
    mask = windows.target_mask.reshape(n, n_origins, horizon)[0]
    return {
        "prediction": _restore_full_split(prediction_horizon, windows),
        "p_prediction": _restore_full_split(probability_horizon, windows),
        "mu_prediction": _restore_full_split(magnitude_horizon, windows),
        "prediction_horizon": prediction_horizon,
        "p_prediction_horizon": probability_horizon,
        "mu_prediction_horizon": magnitude_horizon,
        "gap_prediction": np.concatenate(gap).reshape(n, n_origins),
        "gap_true": windows.gap.reshape(n, n_origins),
        "gap_event_observed": windows.gap_event_observed.reshape(n, n_origins),
        "gap_censor_lower": windows.gap_censor_lower.reshape(n, n_origins),
        "origins": origins,
        "horizon_mask": mask,
        "valid_lengths": windows.valid_lengths,
        "train_scale": scale,
        "split_start": np.int32(split_start),
        "split_end": np.int32(split_end),
    }


def _scalar(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    item = value.reshape(()).item()
    return item.decode() if isinstance(item, bytes) else item


def scenario_subset(spec: ScenarioSpec) -> str:
    if spec.group == "A":
        return "A_iid" if spec.variant == "iid" else "A_non_iid"
    return spec.group


def _pattern_windows(
    data: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    origins = np.asarray(outputs["origins"], dtype=np.int32)
    prediction = np.asarray(outputs["prediction_horizon"], dtype=np.float64)
    base_mask = np.asarray(outputs["horizon_mask"], dtype=bool)
    n, n_origins, horizon = prediction.shape
    if base_mask.shape != (n_origins, horizon):
        raise ValueError("prediction horizon mask shape mismatch")

    if all(key in data for key in (
        "generation_target", "generation_target_origins", "generation_target_mask"
    )):
        stored_origins = np.asarray(data["generation_target_origins"], dtype=np.int32)
        lookup = {int(origin): index for index, origin in enumerate(stored_origins)}
        try:
            indices = np.asarray([lookup[int(origin)] for origin in origins], dtype=np.int32)
        except KeyError as error:
            raise ValueError(f"origin-aware generation target misses {error.args[0]}") from error
        pattern = np.asarray(data["generation_target"], dtype=np.float64)[:, indices, :]
        stored_mask = np.asarray(data["generation_target_mask"], dtype=bool)
        if stored_mask.ndim == 2:
            stored_mask = stored_mask[indices][None, :, :]
        elif stored_mask.ndim == 3:
            stored_mask = stored_mask[:, indices, :]
        else:
            raise ValueError("generation_target_mask must have 2 or 3 dimensions")
        mask = np.broadcast_to(base_mask[None], (n, n_origins, horizon)) & np.broadcast_to(
            stored_mask, (n, n_origins, horizon)
        )
        source = "origin_aware_conditional_generation_target"
    else:
        group = str(_scalar(data, "group", ""))
        if group != "A":
            raise ValueError(
                f"Group {group!r} is missing its origin-aware generation target"
            )
        pattern, stored_mask = group_a_conditional_target(data, origins, config)
        if pattern.shape != prediction.shape:
            raise ValueError("computed Group-A target shape differs from predictions")
        mask = np.broadcast_to(base_mask[None], prediction.shape) & np.broadcast_to(
            np.asarray(stored_mask, dtype=bool)[None], prediction.shape
        )
        source = "origin_history_conditional_markov_target"
    observed = np.zeros_like(prediction)
    y = np.asarray(data["y"], dtype=np.float64)
    for window, origin in enumerate(origins):
        valid = int(base_mask[window].sum())
        observed[:, window, :valid] = y[:, origin:origin + valid]
    return pattern, observed, mask, source


def _rms_masked(error: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.square(error) * mask) / np.sum(mask)))


def select_representative_series(pattern_gap_per_series: np.ndarray) -> tuple[int, float]:
    values = np.asarray(pattern_gap_per_series, dtype=np.float64)
    indices = np.flatnonzero(np.isfinite(values))
    if not indices.size:
        raise ValueError("no finite series Pattern Gap")
    median = float(np.median(values[indices]))
    selected = indices[np.argmin(np.abs(values[indices] - median))]
    return int(selected), median


def compute_split_metrics(
    data: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    method: str,
    phi: float,
    split_start: int,
    split_end: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Metrics using origin-aware patterns when the generator stored them."""

    prediction = np.asarray(outputs["prediction"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)[:, split_start:split_end]
    z = np.asarray(data["z"], dtype=np.float64)[:, split_start:split_end]
    if prediction.shape != y.shape:
        raise ValueError("restored prediction and truth differ in shape")
    train_mean = np.asarray(data["y"], dtype=np.float64)[:, :config.train_end].mean(
        axis=1, keepdims=True
    )
    raw_mse = float(np.mean((prediction - y) ** 2))
    train_mean_mse = float(np.mean((train_mean - y) ** 2))

    pattern, observed_horizon, pattern_mask, source = _pattern_windows(data, outputs, config)
    prediction_horizon = np.asarray(outputs["prediction_horizon"], dtype=np.float64)
    pattern_rmse = _rms_masked(prediction_horizon - pattern, pattern_mask)
    noise_rmse = _rms_masked(observed_horizon - pattern, pattern_mask)
    flat = np.broadcast_to(train_mean[:, None, :], pattern.shape)
    flat_pattern_rmse = _rms_masked(flat - pattern, pattern_mask)
    per_series_count = pattern_mask.sum(axis=(1, 2))
    per_series_pattern = np.sqrt(
        (np.square(prediction_horizon - pattern) * pattern_mask).sum(axis=(1, 2))
        / per_series_count
    )
    per_series_noise = np.sqrt(
        (np.square(observed_horizon - pattern) * pattern_mask).sum(axis=(1, 2))
        / per_series_count
    )
    per_series_gap = per_series_pattern / (per_series_noise + config.epsilon)
    representative, median = select_representative_series(per_series_gap)
    group = str(_scalar(data, "group", ""))
    train_y = np.asarray(data["y"], dtype=np.float64)[:, :config.train_end]
    train_z = np.asarray(data["z"], dtype=np.float64)[:, :config.train_end]
    train_positive = train_y[train_z > 0.5]
    spike_threshold = (
        float(np.quantile(train_positive, 0.90)) if train_positive.size else np.nan
    )
    spike_mask = (z > 0.5) & (y >= spike_threshold) if group != "B" else np.zeros_like(
        z, dtype=bool
    )
    spike_count = int(spike_mask.sum())
    spike_shortfall = np.maximum(y - prediction, 0.0)
    result: dict[str, Any] = {
        "raw_mse": raw_mse,
        "train_mean_mse": train_mean_mse,
        "raw_mse_gain_vs_train_mean": 1.0 - raw_mse / (train_mean_mse + config.epsilon),
        "pattern_rmse": pattern_rmse,
        "pattern_noise_rmse": noise_rmse,
        "pattern_gap": pattern_rmse / (noise_rmse + config.epsilon),
        "pattern_gap_median": median,
        "pattern_gap_per_series": per_series_gap.astype(np.float32),
        "representative_series_index": representative,
        "flat_pattern_rmse": flat_pattern_rmse,
        "structure_gain_vs_flat": 1.0 - pattern_rmse / (
            flat_pattern_rmse + config.epsilon
        ),
        "tracks_pattern_vs_flat": bool(pattern_rmse < flat_pattern_rmse),
        "pattern_target_source": source,
        "brier_score": np.nan,
        "brier_score_diagnostic": np.nan,
        "brier_included_in_group_comparison": group != "B",
        "positive_magnitude_mae": np.nan,
        "gamma_nll": np.nan,
        "gap_mae": np.nan,
        "gap_mae_diagnostic": np.nan,
        "gap_log_error": np.nan,
        "gap_censor_violation_rate": np.nan,
        "gap_included_in_group_comparison": group != "B",
        "gap_supervised": method == "gamma_gap" and group != "B",
        "top_spike_threshold_train_q90": spike_threshold if group != "B" else np.nan,
        "top_spike_count": spike_count if group != "B" else 0,
        "top_spike_underprediction_mae": (
            float(spike_shortfall[spike_mask].mean()) if spike_count else np.nan
        ),
        "top_spike_underprediction_rate": (
            float(np.mean(prediction[spike_mask] < y[spike_mask]))
            if spike_count else np.nan
        ),
        "top_spike_bias": (
            float(np.mean(prediction[spike_mask] - y[spike_mask]))
            if spike_count else np.nan
        ),
        "top_spike_magnitude_underprediction_mae": np.nan,
        "top_spike_magnitude_underprediction_rate": np.nan,
    }
    if method == "mse":
        return result
    probability = np.asarray(outputs["p_prediction"], dtype=np.float64)
    magnitude = np.asarray(outputs["mu_prediction"], dtype=np.float64)
    positive = z > 0.5
    brier = float(np.mean((probability - z) ** 2)) if group != "B" else np.nan
    gap_prediction = np.asarray(outputs["gap_prediction"], dtype=np.float64)
    gap_true = np.asarray(outputs["gap_true"], dtype=np.float64)
    gap_event = np.asarray(
        outputs.get("gap_event_observed", gap_true <= config.horizon), dtype=bool
    )
    gap_censor_lower = np.asarray(
        outputs.get("gap_censor_lower", gap_true), dtype=np.float64
    )
    if not (
        gap_prediction.shape == gap_true.shape == gap_event.shape == gap_censor_lower.shape
    ):
        raise ValueError("gap metric tensors differ in shape")
    if group != "B":
        gap_mae = (
            float(np.mean(np.abs(gap_prediction[gap_event] - gap_true[gap_event])))
            if gap_event.any() else np.nan
        )
        gap_exact_log = np.abs(np.log1p(gap_prediction) - np.log1p(gap_true))
        gap_censored_log = np.maximum(
            np.log1p(gap_censor_lower) - np.log1p(gap_prediction), 0.0
        )
        gap_log_error = float(np.mean(np.where(gap_event, gap_exact_log, gap_censored_log)))
        censored = ~gap_event
        gap_censor_violation = (
            float(np.mean(gap_prediction[censored] < gap_censor_lower[censored]))
            if censored.any() else np.nan
        )
    else:
        gap_mae = np.nan
        gap_log_error = np.nan
        gap_censor_violation = np.nan
    magnitude_shortfall = np.maximum(y - magnitude, 0.0)
    result.update({
        "brier_score": brier,
        "brier_score_diagnostic": brier,
        "positive_magnitude_mae": float(np.mean(np.abs(magnitude[positive] - y[positive]))),
        "gamma_nll": float(gamma_nll(
            torch.from_numpy(magnitude),
            torch.from_numpy(y),
            torch.from_numpy(positive),
            torch.tensor(phi, dtype=torch.float64),
        )),
        "gap_mae": gap_mae,
        "gap_mae_diagnostic": gap_mae,
        "gap_log_error": gap_log_error,
        "gap_censor_violation_rate": gap_censor_violation,
        "top_spike_magnitude_underprediction_mae": (
            float(magnitude_shortfall[spike_mask].mean()) if spike_count else np.nan
        ),
        "top_spike_magnitude_underprediction_rate": (
            float(np.mean(magnitude[spike_mask] < y[spike_mask]))
            if spike_count else np.nan
        ),
    })
    return result


def compute_test_metrics(
    data: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    method: str,
    phi: float,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    return compute_split_metrics(
        data, outputs, method, phi, config.val_end, config.length, config
    )


def _parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "pass", "passed", "ok", "valid"}
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_current_validation_artifacts(
    report: pd.DataFrame,
    scenarios: tuple[ScenarioSpec, ...],
    data_dir: Path,
) -> None:
    """Bind a passed report to the current manifest and exact NPZ bytes."""

    hash_columns = {
        "npz_sha256", "manifest_npz_sha256", "manifest_file_sha256",
    }
    missing_columns = sorted(hash_columns - set(report.columns))
    if missing_columns:
        raise RuntimeError(
            "validation report lacks stale-report hash columns: "
            f"{missing_columns}"
        )

    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise RuntimeError("validation report is stale: current manifest.csv is missing")
    current_manifest_sha = _sha256_file(manifest_path)
    reported_manifest_sha = (
        report["manifest_file_sha256"].astype(str).str.strip().str.lower()
    )
    if not bool(reported_manifest_sha.eq(current_manifest_sha).all()):
        raise RuntimeError(
            "validation report is stale: manifest.csv SHA256 differs from the "
            "validated manifest"
        )

    try:
        manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise RuntimeError(f"current manifest.csv cannot be read: {exc}") from exc
    required_manifest = {"scenario_id", "filename", "sha256"}
    if not required_manifest.issubset(manifest.columns):
        missing = sorted(required_manifest - set(manifest.columns))
        raise RuntimeError(f"current manifest.csv lacks columns: {missing}")

    expected_ids = [spec.scenario_id for spec in scenarios]
    expected_files = [f"{scenario_id}.npz" for scenario_id in expected_ids]
    if (
        len(manifest) != len(scenarios)
        or manifest["scenario_id"].tolist() != expected_ids
        or manifest["filename"].tolist() != expected_files
    ):
        raise RuntimeError(
            "validation report is stale: current manifest rows/order/filenames "
            "differ from the canonical catalogue"
        )

    stale: list[str] = []
    for index, spec in enumerate(scenarios):
        npz_path = data_dir / expected_files[index]
        if not npz_path.is_file():
            stale.append(f"{spec.scenario_id}:missing NPZ")
            continue
        actual_sha = _sha256_file(npz_path)
        report_npz_sha = str(report.iloc[index]["npz_sha256"]).strip().lower()
        report_manifest_npz_sha = str(
            report.iloc[index]["manifest_npz_sha256"]
        ).strip().lower()
        current_manifest_npz_sha = str(
            manifest.iloc[index]["sha256"]
        ).strip().lower()
        if not (
            actual_sha == report_npz_sha
            == report_manifest_npz_sha
            == current_manifest_npz_sha
        ):
            stale.append(spec.scenario_id)
    if stale:
        preview = stale[:5]
        suffix = "" if len(stale) <= 5 else f" (+{len(stale) - 5} more)"
        raise RuntimeError(
            "validation report is stale: NPZ SHA256 differs for "
            f"{preview}{suffix}"
        )


def require_validation(
    path: Path | str = VALIDATION_PATH,
    config: ExperimentConfig = DEFAULT_CONFIG,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    path = Path(path)
    data_dir = Path(DATA_DIR if data_dir is None else data_dir)
    if not path.is_file():
        raise RuntimeError("validation_27.csv is missing")
    report = pd.read_csv(path)
    pass_column = next(
        (key for key in ("passed", "all_passed", "valid", "status") if key in report),
        None,
    )
    if pass_column is None or "scenario_id" not in report:
        raise RuntimeError("validation report lacks scenario_id/pass status")
    scenarios = build_scenarios(config)
    expected = [spec.scenario_id for spec in scenarios]
    if len(report) != 27 or report["scenario_id"].astype(str).tolist() != expected:
        raise RuntimeError("validation report must have the canonical 27-row order")
    if not bool(_parse_bool(report[pass_column]).all()):
        failed = report.loc[~_parse_bool(report[pass_column]), "scenario_id"].tolist()
        raise RuntimeError(f"generation validation failed: {failed}")
    _require_current_validation_artifacts(report, scenarios, data_dir)
    return report


def resolve_cuda_device(requested: str = "cuda") -> torch.device:
    device = torch.device(requested)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official v3 stage runs require an available CUDA device")
    return device


def load_dataset(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _validate_dataset(data: Mapping[str, np.ndarray], spec: ScenarioSpec,
                      config: ExperimentConfig) -> None:
    for key in ("y", "z", "p_true", "mu_true", "m_true", "d_true"):
        if np.asarray(data[key]).shape != (config.n_series, config.length):
            raise ValueError(f"{spec.scenario_id}:{key} shape mismatch")
    if str(_scalar(data, "scenario_id")) != spec.scenario_id:
        raise ValueError(f"dataset ID mismatch for {spec.scenario_id}")


def _save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)


def _save_log(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(dict(payload), handle)


def load_results(path: Path | str | None = None) -> pd.DataFrame:
    path = RESULTS_PATH if path is None else Path(path)
    return pd.read_csv(path) if path.is_file() and path.stat().st_size else pd.DataFrame()


def _append_result(row: Mapping[str, Any], path: Path | None = None) -> None:
    path = RESULTS_PATH if path is None else Path(path)
    frame = pd.DataFrame([dict(row)])
    existing = load_results(path)
    if not existing.empty:
        duplicate = existing[
            existing["scenario_id"].astype(str).eq(str(row["scenario_id"]))
            & existing["method"].astype(str).eq(str(row["method"]))
            & existing["train_seed"].eq(0)
        ]
        if len(duplicate):
            raise FileExistsError(f"duplicate result {row['scenario_id']}/{row['method']}")
        if list(existing.columns) != list(frame.columns):
            raise RuntimeError("results_seed0.csv schema mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "x"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, header=not path.stat().st_size)


def _prediction_payload(
    data: Mapping[str, np.ndarray], outputs: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any], spec: ScenarioSpec, method: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    payload = dict(outputs)
    for key in ("y", "z", "p_true", "mu_true", "m_true"):
        payload[key] = np.asarray(data[key])[:, config.val_end:config.length]
    payload.update({
        "scenario_id": np.str_(spec.scenario_id),
        "group": np.str_(spec.group),
        "method": np.str_(method),
        "train_seed": np.int16(0),
        "phi": np.float64(metrics["phi"]),
        "pattern_gap": np.float64(metrics["pattern_gap"]),
        "pattern_gap_per_series": np.asarray(metrics["pattern_gap_per_series"]),
        "representative_series_index": np.int16(metrics["representative_series_index"]),
    })
    pattern, _observed, mask, source = _pattern_windows(data, outputs, config)
    start = int(np.asarray(outputs["split_start"]).reshape(()))
    end = int(np.asarray(outputs["split_end"]).reshape(()))
    origins = np.asarray(outputs["origins"], dtype=np.int32)
    flat = np.empty((config.n_series, end - start), dtype=np.float32)
    flat_mask = np.zeros((config.n_series, end - start), dtype=np.uint8)
    covered = np.zeros(end - start, dtype=bool)
    for window, origin in enumerate(origins):
        offsets = np.flatnonzero(mask[0, window])
        if offsets.size and not np.array_equal(offsets, np.arange(offsets.size)):
            raise ValueError("generation target mask must be a contiguous prefix")
        valid = offsets.size
        destination = int(origin - start)
        flat[:, destination:destination + valid] = pattern[:, window, :valid]
        flat_mask[:, destination:destination + valid] = 1
        covered[destination:destination + valid] = True
    if not covered.all():
        raise RuntimeError("origin-aware generation target did not cover full split")
    payload.update({
        "generation_target": flat,
        "generation_target_flat": flat.copy(),
        "generation_target_mask": flat_mask,
        "generation_target_horizon": pattern.astype(np.float32),
        "generation_target_horizon_mask": mask.astype(np.uint8),
        "generation_target_origins": origins,
        "generation_target_source": np.str_(source),
    })
    return payload


def _run_one_and_publish(
    spec: ScenarioSpec, method: str, device: torch.device,
    config: ExperimentConfig, data_dir: Path = DATA_DIR,
) -> dict[str, Any]:
    data = load_dataset(data_dir / f"{spec.scenario_id}.npz")
    _validate_dataset(data, spec, config)
    stem = f"{spec.scenario_id}_{method}_seed0"
    prediction_path = PREDICTIONS_DIR / f"{stem}.npz"
    checkpoint_path = CHECKPOINTS_DIR / f"{stem}.pt"
    log_path = LOGS_DIR / f"{stem}.csv"
    existing_artifacts = [path for path in (prediction_path, checkpoint_path, log_path) if path.exists()]
    if existing_artifacts:
        raise FileExistsError(f"refusing existing artifacts: {existing_artifacts}")
    model, log, info = _train_one_unchecked(data, method, device, config)
    validation_output = predict_split(
        model, data, method, config.train_end, config.val_end, device, config
    )
    validation_metrics = compute_split_metrics(
        data, validation_output, method, float(info["phi"]),
        config.train_end, config.val_end, config,
    )
    test_output = predict_split(
        model, data, method, config.val_end, config.length, device, config
    )
    test_metrics = compute_test_metrics(data, test_output, method, float(info["phi"]), config)
    metrics = dict(test_metrics) | {"phi": float(info["phi"])}
    payload = _prediction_payload(data, test_output, metrics, spec, method, config)
    log.insert(0, "scenario_id", spec.scenario_id)
    log.insert(1, "group", spec.group)
    _save_npz(prediction_path, payload)
    _save_log(log_path, log)
    _save_checkpoint(checkpoint_path, {
        "schema_version": "unified_temporal_27_v3",
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scenario_id": spec.scenario_id,
        "method": method,
        "train_seed": 0,
        "config": config.to_dict(),
        "training_info": info,
        "phi_source": "pooled_unique_train_positive_cv2",
    })
    scalar_test = {
        key: value for key, value in test_metrics.items()
        if key != "pattern_gap_per_series"
    }
    row: dict[str, Any] = {
        "scenario_id": spec.scenario_id,
        "scenario_index": spec.index,
        "group": spec.group,
        "subset": scenario_subset(spec),
        "design": spec.design,
        "structure": spec.structure,
        "strength": spec.strength,
        "variant": spec.variant,
        "pair": spec.pair if spec.pair is not None else -1,
        "method": method,
        "data_seed": int(_scalar(data, "data_seed", config.data_seed)),
        "train_seed": 0,
        "device": str(device),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **info,
        "validation_pattern_gap": validation_metrics["pattern_gap"],
        "validation_structure_gain_vs_flat": validation_metrics["structure_gain_vs_flat"],
        **scalar_test,
        "positive_mae": scalar_test["positive_magnitude_mae"],
        "lambda_mag": config.lambda_mag if method != "mse" else np.nan,
        "lambda_gap": config.lambda_gap if method == "gamma_gap" else 0.0,
        "lambda_source": "validation_A_nonIID_plus_C" if method != "mse" else "not_applicable",
        "evaluation_protocol": "blocked_nonoverlap_final_partial_observed_history",
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "log_path": str(log_path.relative_to(ROOT)),
    }
    _append_result(row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def evaluate_mse_gate(
    results: pd.DataFrame,
    output_path: Path | str = MSE_GATE_PATH,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Write a 27-row, validation-only MSE fitting gate.

    Pass rule: positive validation structure gain for at least two of the three
    strong A non-IID sentinels, plus B07 and C01, and at least four of all five.
    """

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    scenarios = build_scenarios(config)
    mse = results[results["method"].astype(str).eq("mse")]
    expected = [spec.scenario_id for spec in scenarios]
    if mse["scenario_id"].astype(str).tolist() != expected:
        raise RuntimeError("MSE gate requires canonical complete 27 MSE rows")
    gain = dict(zip(
        mse["scenario_id"].astype(str),
        mse["validation_structure_gain_vs_flat"].astype(float),
    ))
    passes = {scenario_id: gain[scenario_id] > 0.0 for scenario_id in GATE_SENTINELS}
    a_pass = sum(passes[key] for key in GATE_SENTINELS[:3])
    total_pass = sum(passes.values())
    overall = a_pass >= 2 and passes[GATE_SENTINELS[3]] and passes[GATE_SENTINELS[4]] \
        and total_pass >= 4
    rows = []
    for spec in scenarios:
        sentinel = spec.scenario_id in passes
        rows.append({
            "scenario_id": spec.scenario_id,
            "group": spec.group,
            "validation_structure_gain_vs_flat": gain[spec.scenario_id],
            "gate_sentinel": sentinel,
            "sentinel_pass_gain_gt_zero": passes.get(spec.scenario_id, np.nan),
            "overall_gate_pass": overall,
            "gate_split": "validation",
            "gate_rule": "A_strong>=2of3 AND B07 AND C01 AND total>=4of5",
        })
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    return frame


def require_mse_gate(path: Path | str = MSE_GATE_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("mse_gate_27_seed0.csv is missing; Gamma stages are blocked")
    frame = pd.read_csv(path)
    if len(frame) != 27 or "overall_gate_pass" not in frame:
        raise RuntimeError("MSE gate has an invalid schema")
    if not bool(_parse_bool(frame["overall_gate_pass"]).all()):
        raise RuntimeError("MSE fitting gate failed; Gamma stages are blocked")
    return frame


def run_mse_stage(
    requested_device: str = "cuda",
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    require_validation(config=config)
    device = resolve_cuda_device(requested_device)
    existing = load_results()
    for spec in build_scenarios(config):
        if not existing.empty and (
            existing["scenario_id"].astype(str).eq(spec.scenario_id)
            & existing["method"].astype(str).eq("mse")
        ).any():
            continue
        _run_one_and_publish(spec, "mse", device, config)
        existing = load_results()
    return evaluate_mse_gate(load_results(), config=config)


def select_lambdas(
    requested_device: str = "cuda",
    config: ExperimentConfig = DEFAULT_CONFIG,
    output_path: Path | str = LAMBDA_SELECTION_PATH,
) -> pd.DataFrame:
    """Select one shared lambda_mag then lambda_gap on validation only."""

    require_validation(config=config)
    require_mse_gate()
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    device = resolve_cuda_device(requested_device)
    specs = {spec.scenario_id: spec for spec in build_scenarios(config)}
    rows: list[dict[str, Any]] = []

    def evaluate_candidate(method: str, candidate_config: ExperimentConfig) -> dict[str, float]:
        pattern_gaps, raw_mses, gap_maes = [], [], []
        for scenario_id in LAMBDA_SELECTION_IDS:
            data = load_dataset(DATA_DIR / f"{scenario_id}.npz")
            _validate_dataset(data, specs[scenario_id], candidate_config)
            model, _log, info = _train_one_unchecked(data, method, device, candidate_config)
            output = predict_split(
                model, data, method, candidate_config.train_end,
                candidate_config.val_end, device, candidate_config,
            )
            metrics = compute_split_metrics(
                data, output, method, float(info["phi"]), candidate_config.train_end,
                candidate_config.val_end, candidate_config,
            )
            pattern_gaps.append(float(metrics["pattern_gap"]))
            raw_mses.append(float(metrics["raw_mse"]))
            if np.isfinite(metrics["gap_mae"]):
                gap_maes.append(float(metrics["gap_mae"]))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return {
            "aggregate_validation_pattern_gap": float(np.mean(pattern_gaps)),
            "aggregate_validation_raw_mse": float(np.mean(raw_mses)),
            "aggregate_validation_gap_mae": (
                float(np.mean(gap_maes)) if gap_maes else np.nan
            ),
        }

    for lambda_mag in config.lambda_mag_candidates:
        candidate = replace(config, lambda_mag=float(lambda_mag))
        metrics = evaluate_candidate("gamma_hurdle", candidate)
        rows.append({
            "selection_stage": "magnitude",
            "method": "gamma_hurdle",
            "lambda_mag": float(lambda_mag),
            "lambda_gap": 0.0,
            **metrics,
        })
    mag_rows = [row for row in rows if row["selection_stage"] == "magnitude"]
    selected_mag = min(
        mag_rows, key=lambda row: (row["aggregate_validation_pattern_gap"], row["lambda_mag"])
    )["lambda_mag"]
    for lambda_gap in config.lambda_gap_candidates:
        candidate = replace(
            config, lambda_mag=float(selected_mag), lambda_gap=float(lambda_gap)
        )
        metrics = evaluate_candidate("gamma_gap", candidate)
        rows.append({
            "selection_stage": "gap",
            "method": "gamma_gap",
            "lambda_mag": float(selected_mag),
            "lambda_gap": float(lambda_gap),
            **metrics,
        })
    gap_rows = [row for row in rows if row["selection_stage"] == "gap"]
    selected_gap = min(
        gap_rows, key=lambda row: (row["aggregate_validation_pattern_gap"], row["lambda_gap"])
    )["lambda_gap"]
    for row in rows:
        row.update({
            "selected": (
                row["lambda_mag"] == selected_mag
                and ((row["selection_stage"] == "magnitude" and row["lambda_gap"] == 0.0)
                     or (row["selection_stage"] == "gap" and row["lambda_gap"] == selected_gap))
            ),
            "scenario_count": len(LAMBDA_SELECTION_IDS),
            "selection_scenarios": "|".join(LAMBDA_SELECTION_IDS),
            "selection_split": "validation",
            "test_accessed": False,
            "train_seed": 0,
        })
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    return frame


def selected_lambdas(path: Path | str = LAMBDA_SELECTION_PATH) -> tuple[float, float]:
    frame = pd.read_csv(path)
    selected = frame[_parse_bool(frame["selected"])]
    magnitude = selected[selected["selection_stage"].eq("magnitude")]
    gap = selected[selected["selection_stage"].eq("gap")]
    if len(magnitude) != 1 or len(gap) != 1 or bool(
        _parse_bool(frame["test_accessed"]).any()
    ):
        raise RuntimeError("lambda_selection.csv is invalid or test-informed")
    lambda_mag = float(magnitude.iloc[0]["lambda_mag"])
    if not np.isclose(lambda_mag, float(gap.iloc[0]["lambda_mag"])):
        raise RuntimeError("Gamma and Gamma+Gap do not share lambda_mag")
    return lambda_mag, float(gap.iloc[0]["lambda_gap"])


def run_loss_stage(
    method: str,
    requested_device: str = "cuda",
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if method not in {"gamma_hurdle", "gamma_gap"}:
        raise ValueError("loss stage must be gamma_hurdle or gamma_gap")
    require_validation(config=config)
    require_mse_gate()
    lambda_mag, lambda_gap = selected_lambdas()
    fair_config = replace(config, lambda_mag=lambda_mag, lambda_gap=lambda_gap)
    device = resolve_cuda_device(requested_device)
    existing = load_results()
    if method == "gamma_gap":
        if existing.empty or not {"scenario_id", "method"}.issubset(existing.columns):
            raise RuntimeError(
                "gamma_gap is blocked until all 27 gamma_hurdle rows are complete"
            )
        hurdle = existing[existing["method"].astype(str).eq("gamma_hurdle")]
        expected = [spec.scenario_id for spec in build_scenarios(fair_config)]
        if hurdle["scenario_id"].astype(str).tolist() != expected:
            raise RuntimeError(
                "gamma_gap is blocked until all 27 gamma_hurdle rows are complete"
            )
    for spec in build_scenarios(fair_config):
        if not existing.empty and (
            existing["scenario_id"].astype(str).eq(spec.scenario_id)
            & existing["method"].astype(str).eq(method)
        ).any():
            continue
        _run_one_and_publish(spec, method, device, fair_config)
        existing = load_results()
    return load_results()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gated v3 seed-0 CUDA stages")
    parser.add_argument(
        "stage",
        choices=("mse", "select-lambda", "gamma-hurdle", "gamma-gap"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.stage == "mse":
        frame = run_mse_stage(args.device)
    elif args.stage == "select-lambda":
        frame = select_lambdas(args.device)
    elif args.stage == "gamma-hurdle":
        frame = run_loss_stage("gamma_hurdle", args.device)
    else:
        frame = run_loss_stage("gamma_gap", args.device)
    print(frame.tail().to_string(index=False))


__all__ = [
    "METHODS", "RESULTS_PATH", "MSE_GATE_PATH", "LAMBDA_SELECTION_PATH",
    "GATE_SENTINELS", "LAMBDA_SELECTION_IDS", "WindowArrays",
    "train_origins", "nonoverlap_origins", "train_scale", "make_windows",
    "estimate_phi_train_positive", "gamma_nll", "scalar_gap_loss", "objective",
    "predict_split", "compute_split_metrics", "compute_test_metrics",
    "select_representative_series", "require_validation", "resolve_cuda_device",
    "load_dataset", "load_results", "scenario_subset", "evaluate_mse_gate", "require_mse_gate",
    "run_mse_stage", "select_lambdas", "selected_lambdas", "run_loss_stage",
]


if __name__ == "__main__":
    main()
