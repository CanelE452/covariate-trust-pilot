"""Frozen teacher training loop: full-NLL objective, sCRPS checkpointing, append-only attempts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .evaluation import CRPS_QUANTILE_GRID, approximate_crps
from .integrity import publish_completion_marker, reserve_or_resume_attempt
from .models import ProbabilisticDLinear, build_teacher, count_parameters

MAXIMUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
SCHEDULED_VALIDATION_EPOCHS = tuple(range(2, MAXIMUM_EPOCHS + 1, 2))
EFFECTIVE_BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0


class NumericalBranchBlocked(RuntimeError):
    """A head produced a nonfinite loss; the branch is blocked without retuning."""


class OutOfMemoryBranchBlocked(RuntimeError):
    """A head hit a second out-of-memory failure after its single frozen retry."""


@dataclass(frozen=True)
class TrainingWindows:
    """Target-bearing windows already scaled by their train-only series scale."""

    history: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    scale: np.ndarray
    retained_rows: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        history = np.asarray(self.history, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        mask = np.asarray(self.target_mask).astype(bool)
        scale = np.asarray(self.scale, dtype=np.float64)
        if history.ndim != 2 or target.ndim != 2 or mask.shape != target.shape:
            raise ValueError("history and target must be two-dimensional with a matching mask")
        if history.shape[0] != target.shape[0] or scale.shape != (history.shape[0],):
            raise ValueError("history, target and scale must agree on the row count")
        if not np.isfinite(history).all() or not np.isfinite(scale).all():
            raise ValueError("history and scale must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("train-only scale must be positive")
        if np.any(target[mask] < 0.0) or not np.isfinite(target[mask]).all():
            raise ValueError("valid targets must be finite and nonnegative")
        rows = self.retained_rows
        rows = np.arange(history.shape[0]) if rows is None else np.asarray(rows, dtype=np.int64)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "target_mask", mask)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "retained_rows", rows)

    @property
    def row_count(self) -> int:
        return int(self.history.shape[0])

    @property
    def valid_target_count(self) -> int:
        return int(self.target_mask.sum())

    def without_fully_masked_rows(self) -> "TrainingWindows":
        """Drop rows whose horizon is entirely masked, preserving the original order."""
        keep = np.flatnonzero(self.target_mask.any(axis=1))
        return TrainingWindows(
            history=self.history[keep],
            target=self.target[keep],
            target_mask=self.target_mask[keep],
            scale=self.scale[keep],
            retained_rows=self.retained_rows[keep],
        )

    def select(self, rows: np.ndarray) -> "TrainingWindows":
        index = np.asarray(rows, dtype=np.int64)
        return TrainingWindows(
            history=self.history[index],
            target=self.target[index],
            target_mask=self.target_mask[index],
            scale=self.scale[index],
            retained_rows=self.retained_rows[index],
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Every training numeric is frozen; only identity fields vary between fits."""

    head_name: str
    lookback: int
    horizon: int
    seed: int
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    microbatch_size: int = EFFECTIVE_BATCH_SIZE
    gradient_accumulation: int = 1
    maximum_epochs: int = MAXIMUM_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.microbatch_size * self.gradient_accumulation != self.effective_batch_size:
            raise ValueError("microbatch size times accumulation must equal the effective batch")
        if self.maximum_epochs < 1 or self.patience < 1:
            raise ValueError("maximum epochs and patience must be positive")

    def scheduled_epochs(self) -> tuple[int, ...]:
        return tuple(epoch for epoch in SCHEDULED_VALIDATION_EPOCHS if epoch <= self.maximum_epochs)

    def identity(self) -> dict[str, Any]:
        return {
            "head": self.head_name,
            "seed": int(self.seed),
            "lookback": int(self.lookback),
            "horizon": int(self.horizon),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "effective_batch_size": int(self.effective_batch_size),
            "microbatch_size": int(self.microbatch_size),
            "gradient_accumulation": int(self.gradient_accumulation),
            "maximum_epochs": int(self.maximum_epochs),
            "patience": int(self.patience),
        }


@dataclass
class TrainingResult:
    model: ProbabilisticDLinear
    config: TrainingConfig
    best_epoch: int | None
    best_score: float | None
    best_state: dict[str, torch.Tensor]
    scores: list[tuple[int, float]]
    stopped_epoch: int
    stop_reason: str
    checks_evaluated: int
    microbatch_size: int
    gradient_accumulation: int
    oom_retries: int
    parameter_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "head": self.config.head_name,
            "seed": int(self.config.seed),
            "best_epoch": self.best_epoch,
            "best_score": None if self.best_score is None else float(self.best_score),
            "scores": [[int(epoch), float(score)] for epoch, score in self.scores],
            "stopped_epoch": int(self.stopped_epoch),
            "stop_reason": self.stop_reason,
            "checks_evaluated": int(self.checks_evaluated),
            "microbatch_size": int(self.microbatch_size),
            "gradient_accumulation": int(self.gradient_accumulation),
            "effective_batch_size": int(self.microbatch_size * self.gradient_accumulation),
            "oom_retries": int(self.oom_retries),
            "parameter_count": int(self.parameter_count),
            "configuration": self.config.identity(),
        }


def _tensors(windows: TrainingWindows, device: str) -> tuple[torch.Tensor, ...]:
    history = torch.as_tensor(windows.history, dtype=torch.float32, device=device)
    target = torch.as_tensor(windows.target, dtype=torch.float32, device=device)
    mask = torch.as_tensor(windows.target_mask, dtype=torch.bool, device=device)
    scale = torch.as_tensor(windows.scale, dtype=torch.float32, device=device)
    return history, target, mask, scale


def teacher_nll_objective(
    model: ProbabilisticDLinear,
    windows: TrainingWindows,
    *,
    denominator: int | None = None,
) -> torch.Tensor:
    """Sum the head's own full negative log likelihood over valid cells, divided by their count."""
    valid = windows.valid_target_count
    if valid == 0:
        raise ValueError("the objective requires at least one valid target cell")
    total = valid if denominator is None else int(denominator)
    if total <= 0:
        raise ValueError("the objective denominator must be positive")
    history, target, mask, scale = _tensors(windows, model.trend.weight.device.type)
    distribution = model(history, scale)["distribution"]
    log_prob = distribution.log_prob(target)
    masked = torch.where(mask, -log_prob, torch.zeros_like(log_prob))
    return masked.sum() / float(total)


def validation_scaled_crps(
    model: ProbabilisticDLinear,
    windows: TrainingWindows,
    *,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> float:
    """Score a checkpoint with deterministic native head quantiles on the frozen grid."""
    if tuple(float(value) for value in quantile_grid) != CRPS_QUANTILE_GRID:
        raise ValueError("the validation checkpoint quantile grid is frozen")
    device = model.trend.weight.device.type
    history, _, _, scale = _tensors(windows, device)
    with torch.no_grad():
        distribution = model(history, scale)["distribution"]
        probabilities = torch.as_tensor(
            np.asarray(quantile_grid, dtype=np.float64), dtype=history.dtype, device=device
        )
        quantiles = distribution.quantile(probabilities)
    values = np.moveaxis(quantiles.detach().cpu().numpy().astype(np.float64), 0, -1)
    crps = approximate_crps(windows.target, values, quantile_grid)
    scaled = crps / windows.scale[:, None]
    mask = windows.target_mask
    counts = mask.sum(axis=1)
    if not np.any(counts > 0):
        raise ValueError("validation scoring requires at least one valid row")
    per_row = np.where(counts > 0, np.sum(np.where(mask, scaled, 0.0), axis=1) / np.maximum(counts, 1), np.nan)
    return float(np.nanmean(per_row))


def _microbatch_bounds(row_count: int, microbatch_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + microbatch_size, row_count))
        for start in range(0, row_count, microbatch_size)
    ]


def _run_epoch(
    model: ProbabilisticDLinear,
    optimizer: torch.optim.Optimizer,
    windows: TrainingWindows,
    order: np.ndarray,
    *,
    microbatch_size: int,
    gradient_accumulation: int,
    epoch: int,
    inject_nan_at_step: int | None,
    oom_budget: list[int],
) -> None:
    model.train()
    effective = microbatch_size * gradient_accumulation
    step = 0
    for batch_start in range(0, order.size, effective):
        rows = order[batch_start : batch_start + effective]
        batch = windows.select(rows)
        denominator = batch.valid_target_count
        if denominator == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        for start, stop in _microbatch_bounds(rows.size, microbatch_size):
            micro = batch.select(np.arange(start, stop))
            if micro.valid_target_count == 0:
                continue
            if oom_budget[0] > 0:
                oom_budget[0] -= 1
                raise torch.cuda.OutOfMemoryError("injected out-of-memory failure")
            loss = teacher_nll_objective(model, micro, denominator=denominator)
            if inject_nan_at_step is not None and step == inject_nan_at_step:
                loss = loss * torch.tensor(float("nan"), device=loss.device)
            if not bool(torch.isfinite(loss)):
                raise NumericalBranchBlocked(
                    f"NUMERICAL_BRANCH_BLOCKED: nonfinite loss at epoch {epoch} step {step}"
                )
            loss.backward()
            step += 1
        optimizer.step()


def _fit(
    config: TrainingConfig,
    train_windows: TrainingWindows,
    validation_windows: TrainingWindows,
    *,
    score_sequence: Sequence[float] | None,
    inject_nan_at_step: int | None,
    oom_budget: list[int],
) -> TrainingResult:
    torch.manual_seed(config.seed)
    model = build_teacher(config.head_name, lookback=config.lookback, horizon=config.horizon)
    model.to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = np.random.default_rng(config.seed)
    scheduled = config.scheduled_epochs()
    scored = list(score_sequence) if score_sequence is not None else None

    best_epoch: int | None = None
    best_score: float | None = None
    best_state: dict[str, torch.Tensor] = {}
    scores: list[tuple[int, float]] = []
    without_improvement = 0
    stop_reason = "max_epochs"
    stopped_epoch = 0

    def evaluate(epoch: int) -> bool:
        nonlocal best_epoch, best_score, best_state, without_improvement, stop_reason
        if scored is not None:
            if len(scores) >= len(scored):
                return False
            score = float(scored[len(scores)])
        else:
            score = validation_scaled_crps(model, validation_windows)
        scores.append((epoch, score))
        if best_score is None or score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            without_improvement = 0
            return False
        without_improvement += 1
        if without_improvement >= config.patience:
            stop_reason = "patience"
            return True
        return False

    for epoch in range(1, config.maximum_epochs + 1):
        order = generator.permutation(train_windows.row_count)
        _run_epoch(
            model,
            optimizer,
            train_windows,
            order,
            microbatch_size=config.microbatch_size,
            gradient_accumulation=config.gradient_accumulation,
            epoch=epoch,
            inject_nan_at_step=inject_nan_at_step,
            oom_budget=oom_budget,
        )
        stopped_epoch = epoch
        if epoch in scheduled and evaluate(epoch):
            break
    else:
        if stopped_epoch and stopped_epoch not in [epoch for epoch, _ in scores]:
            evaluate(stopped_epoch)

    if best_state:
        model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        config=config,
        best_epoch=best_epoch,
        best_score=best_score,
        best_state=best_state,
        scores=scores,
        stopped_epoch=stopped_epoch,
        stop_reason=stop_reason,
        checks_evaluated=len(scores),
        microbatch_size=config.microbatch_size,
        gradient_accumulation=config.gradient_accumulation,
        oom_retries=0,
        parameter_count=count_parameters(model),
    )


def train_teacher(
    config: TrainingConfig,
    train_windows: TrainingWindows,
    validation_windows: TrainingWindows,
    *,
    score_sequence: Sequence[float] | None = None,
    _inject_nan_at_step: int | None = None,
    _inject_oom_times: int = 0,
) -> TrainingResult:
    """Fit one teacher, allowing exactly one frozen out-of-memory restart.

    ``score_sequence`` replaces the computed validation score so the frozen patience,
    tie and restore contracts can be exercised deterministically; production fits leave
    it unset and score with :func:`validation_scaled_crps`.
    """
    train_windows = train_windows.without_fully_masked_rows()
    if train_windows.row_count == 0:
        raise ValueError("training requires at least one row with a valid target")
    budget = [int(_inject_oom_times)]
    try:
        return _fit(
            config,
            train_windows,
            validation_windows,
            score_sequence=score_sequence,
            inject_nan_at_step=_inject_nan_at_step,
            oom_budget=budget,
        )
    except torch.cuda.OutOfMemoryError:
        pass

    if config.microbatch_size % 2 != 0:
        raise OutOfMemoryBranchBlocked("OOM_MODEL_BRANCH_BLOCKED: microbatch cannot be halved")
    retry = replace(
        config,
        microbatch_size=config.microbatch_size // 2,
        gradient_accumulation=config.gradient_accumulation * 2,
    )
    try:
        result = _fit(
            retry,
            train_windows,
            validation_windows,
            score_sequence=score_sequence,
            inject_nan_at_step=_inject_nan_at_step,
            oom_budget=budget,
        )
    except torch.cuda.OutOfMemoryError as error:
        raise OutOfMemoryBranchBlocked(
            "OOM_MODEL_BRANCH_BLOCKED: the single frozen retry also ran out of memory"
        ) from error
    result.oom_retries = 1
    return result


def run_training_attempt(
    runs_root: Path,
    stage: str,
    config: TrainingConfig,
    train_windows: TrainingWindows,
    validation_windows: TrainingWindows,
    *,
    score_sequence: Sequence[float] | None = None,
    identity_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit inside an append-only attempt and publish its completion marker last."""
    attempt, resumed = reserve_or_resume_attempt(Path(runs_root), stage)
    if resumed:
        summary = json.loads((attempt / "training_summary.json").read_text(encoding="utf-8"))
        return {"attempt": str(attempt), "resumed": True, "summary": summary, "result": None}

    result = train_teacher(
        config, train_windows, validation_windows, score_sequence=score_sequence
    )
    summary = result.summary()
    summary["identity_bindings"] = dict(identity_bindings or {})
    summary_path = attempt / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = attempt / "checkpoint.pt"
    torch.save({"state_dict": result.best_state or result.model.state_dict()}, checkpoint_path)
    publish_completion_marker(
        attempt,
        {
            "stage": stage,
            "head": config.head_name,
            "seed": int(config.seed),
            "best_epoch": result.best_epoch,
            "stop_reason": result.stop_reason,
        },
        [summary_path, checkpoint_path],
    )
    return {"attempt": str(attempt), "resumed": False, "summary": summary, "result": result}


__all__ = [
    "EARLY_STOPPING_PATIENCE",
    "EFFECTIVE_BATCH_SIZE",
    "MAXIMUM_EPOCHS",
    "SCHEDULED_VALIDATION_EPOCHS",
    "NumericalBranchBlocked",
    "OutOfMemoryBranchBlocked",
    "TrainingConfig",
    "TrainingResult",
    "TrainingWindows",
    "run_training_attempt",
    "teacher_nll_objective",
    "train_teacher",
    "validation_scaled_crps",
]
