"""Masked hard and common-distribution-space student loss reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .student import QUANTILE_GRID


@dataclass(frozen=True)
class StudentLossSums:
    hard_zero_numerator: torch.Tensor
    hard_quantile_numerator: torch.Tensor
    soft_zero_numerator: torch.Tensor
    soft_quantile_numerator: torch.Tensor
    valid_cell_count: int
    quantile_count: int
    has_soft_targets: bool


def _student_shapes(
    p0: torch.Tensor,
    quantiles: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p0 = torch.as_tensor(p0)
    quantiles = torch.as_tensor(
        quantiles, dtype=p0.dtype, device=p0.device
    )
    target = torch.as_tensor(target, dtype=p0.dtype, device=p0.device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=p0.device)
    scale = torch.as_tensor(scale, dtype=p0.dtype, device=p0.device)
    if p0.ndim != 2 or target.shape != p0.shape or mask.shape != p0.shape:
        raise ValueError("p0, target, and target_mask must share [batch,horizon]")
    if quantiles.shape != p0.shape + (QUANTILE_GRID.numel(),):
        raise ValueError("student quantiles must have [batch,horizon,21] shape")
    if scale.ndim == 2 and scale.shape[1] == 1:
        scale = scale[:, 0]
    if scale.shape != (p0.shape[0],):
        raise ValueError("scale must have one train-only value per batch row")
    for name, values in (
        ("p0", p0),
        ("student quantiles", quantiles),
        ("scale", scale),
    ):
        if not torch.is_floating_point(values) or not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} must contain finite floating values")
    if bool((p0 <= 0).any()) or bool((p0 >= 1).any()):
        raise ValueError("p0 must lie strictly inside (0,1)")
    if bool((quantiles < 0).any()) or bool(
        (quantiles[..., 1:] < quantiles[..., :-1]).any()
    ):
        raise ValueError("student quantiles must be nonnegative and monotone")
    if bool((scale <= 0).any()):
        raise ValueError("scale must contain positive train-only values")
    valid_target = target[mask]
    if not bool(torch.isfinite(valid_target).all()):
        raise ValueError("valid target cells must be finite")
    if bool((valid_target < 0).any()) or bool(
        (valid_target != torch.floor(valid_target)).any()
    ):
        raise ValueError("valid target cells must be nonnegative exact counts")
    if int(mask.sum().item()) <= 0:
        raise ValueError("all-masked batches are forbidden")
    return p0, quantiles, target, mask, scale


def _teacher_tensors(
    *,
    p0_student: torch.Tensor,
    teacher_p0: torch.Tensor,
    teacher_quantiles: torch.Tensor,
    teacher_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p0 = torch.as_tensor(
        teacher_p0, dtype=p0_student.dtype, device=p0_student.device
    )
    quantiles = torch.as_tensor(
        teacher_quantiles, dtype=p0_student.dtype, device=p0_student.device
    )
    if p0.ndim == 2:
        if p0.shape != p0_student.shape or quantiles.shape != p0.shape + (
            QUANTILE_GRID.numel(),
        ):
            raise ValueError("single teacher targets have incompatible shapes")
        p0 = p0.unsqueeze(-1)
        quantiles = quantiles.unsqueeze(-2)
    elif p0.ndim != 3 or quantiles.shape != p0.shape + (
        QUANTILE_GRID.numel(),
    ):
        raise ValueError("routed teacher targets require [batch,horizon,teacher,...]")
    if p0.shape[:2] != p0_student.shape:
        raise ValueError("teacher targets differ from student batch/horizon")
    if (
        not bool(torch.isfinite(p0).all())
        or bool((p0 < 0).any())
        or bool((p0 > 1).any())
        or not bool(torch.isfinite(quantiles).all())
        or bool((quantiles < 0).any())
        or bool((quantiles[..., 1:] < quantiles[..., :-1]).any())
    ):
        raise ValueError("teacher distribution targets are invalid")

    teacher_count = int(p0.shape[-1])
    if teacher_weights is None:
        if teacher_count != 1:
            raise ValueError("routed teachers require explicit simplex weights")
        weights = torch.ones_like(p0)
    else:
        weights = torch.as_tensor(
            teacher_weights, dtype=p0.dtype, device=p0.device
        )
        if weights.shape == (p0.shape[0], teacher_count):
            weights = weights[:, None, :].expand_as(p0)
        elif weights.shape != p0.shape:
            raise ValueError("teacher weights must be [batch,teacher] or [batch,horizon,teacher]")
        tolerance = max(
            1e-12,
            4.0 * torch.finfo(weights.dtype).eps * max(1, teacher_count),
        )
        if (
            not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
            or bool((torch.abs(weights.sum(dim=-1) - 1.0) > tolerance).any())
        ):
            raise ValueError("teacher weights must be nonnegative simplex weights")
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return p0, quantiles, weights


def student_loss_sums(
    *,
    p0_student: torch.Tensor,
    quantiles_student: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    scale: torch.Tensor,
    teacher_p0: torch.Tensor | None = None,
    teacher_quantiles: torch.Tensor | None = None,
    teacher_weights: torch.Tensor | None = None,
) -> StudentLossSums:
    """Return unreduced numerators so microbatch accumulation is exact."""
    p0, quantiles, target, mask, scale = _student_shapes(
        p0_student, quantiles_student, target, target_mask, scale
    )
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    zero_target = (safe_target == 0).to(p0.dtype)
    mask_value = mask.to(p0.dtype)
    hard_zero = F.binary_cross_entropy(p0, zero_target, reduction="none")
    grid = QUANTILE_GRID.to(dtype=p0.dtype, device=p0.device)
    error = safe_target.unsqueeze(-1) - quantiles
    hard_quantile = torch.maximum(grid * error, (grid - 1.0) * error)
    hard_zero_numerator = torch.sum(hard_zero * mask_value)
    hard_quantile_numerator = torch.sum(
        hard_quantile * mask_value.unsqueeze(-1)
    )

    has_p0 = teacher_p0 is not None
    has_quantiles = teacher_quantiles is not None
    if has_p0 != has_quantiles:
        raise ValueError("teacher p0 and quantiles must be supplied together")
    if not has_p0:
        if teacher_weights is not None:
            raise ValueError("teacher weights require teacher targets")
        soft_zero_numerator = hard_zero_numerator.new_zeros(())
        soft_quantile_numerator = hard_quantile_numerator.new_zeros(())
    else:
        teacher_p0_tensor, teacher_q_tensor, weights = _teacher_tensors(
            p0_student=p0,
            teacher_p0=teacher_p0,  # type: ignore[arg-type]
            teacher_quantiles=teacher_quantiles,  # type: ignore[arg-type]
            teacher_weights=teacher_weights,
        )
        soft_zero = -(
            teacher_p0_tensor * torch.log(p0.unsqueeze(-1))
            + (1.0 - teacher_p0_tensor) * torch.log1p(-p0.unsqueeze(-1))
        )
        scaled_student = quantiles / scale[:, None, None]
        scaled_teacher = teacher_q_tensor / scale[:, None, None, None]
        soft_quantile = F.huber_loss(
            scaled_student.unsqueeze(-2).expand_as(scaled_teacher),
            scaled_teacher,
            reduction="none",
            delta=1.0,
        )
        soft_zero_numerator = torch.sum(
            soft_zero * weights * mask_value.unsqueeze(-1)
        )
        soft_quantile_numerator = torch.sum(
            soft_quantile
            * weights.unsqueeze(-1)
            * mask_value.unsqueeze(-1).unsqueeze(-1)
        )

    return StudentLossSums(
        hard_zero_numerator=hard_zero_numerator,
        hard_quantile_numerator=hard_quantile_numerator,
        soft_zero_numerator=soft_zero_numerator,
        soft_quantile_numerator=soft_quantile_numerator,
        valid_cell_count=int(mask.sum().item()),
        quantile_count=int(quantiles.shape[-1]),
        has_soft_targets=has_p0,
    )


def add_student_loss_sums(parts: Iterable[StudentLossSums]) -> StudentLossSums:
    items = list(parts)
    if not items:
        raise ValueError("at least one loss-sum part is required")
    first = items[0]
    if any(
        item.quantile_count != first.quantile_count
        or item.has_soft_targets != first.has_soft_targets
        for item in items
    ):
        raise ValueError("loss-sum parts use incompatible contracts")
    return StudentLossSums(
        hard_zero_numerator=sum(
            (item.hard_zero_numerator for item in items),
            first.hard_zero_numerator.new_zeros(()),
        ),
        hard_quantile_numerator=sum(
            (item.hard_quantile_numerator for item in items),
            first.hard_quantile_numerator.new_zeros(()),
        ),
        soft_zero_numerator=sum(
            (item.soft_zero_numerator for item in items),
            first.soft_zero_numerator.new_zeros(()),
        ),
        soft_quantile_numerator=sum(
            (item.soft_quantile_numerator for item in items),
            first.soft_quantile_numerator.new_zeros(()),
        ),
        valid_cell_count=sum(item.valid_cell_count for item in items),
        quantile_count=first.quantile_count,
        has_soft_targets=first.has_soft_targets,
    )


def student_loss_from_sums(
    sums: StudentLossSums, *, lambda_soft: float
) -> dict[str, torch.Tensor]:
    weight = float(lambda_soft)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("lambda_soft must lie in [0,1]")
    if sums.valid_cell_count <= 0:
        raise ValueError("all-masked loss sums are forbidden")
    if weight > 0.0 and not sums.has_soft_targets:
        raise ValueError("positive lambda_soft requires teacher targets")
    cells = float(sums.valid_cell_count)
    quantile_cells = cells * float(sums.quantile_count)
    hard_zero = sums.hard_zero_numerator / cells
    hard_quantile = sums.hard_quantile_numerator / quantile_cells
    soft_zero = sums.soft_zero_numerator / cells
    soft_quantile = sums.soft_quantile_numerator / quantile_cells
    hard = hard_zero + hard_quantile
    soft = soft_zero + soft_quantile
    return {
        "loss": (1.0 - weight) * hard + weight * soft,
        "hard": hard,
        "soft": soft,
        "hard_zero": hard_zero,
        "hard_quantile": hard_quantile,
        "soft_zero": soft_zero,
        "soft_quantile": soft_quantile,
    }


__all__ = [
    "StudentLossSums",
    "add_student_loss_sums",
    "student_loss_from_sums",
    "student_loss_sums",
]
