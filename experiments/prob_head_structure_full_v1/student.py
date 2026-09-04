"""Frozen lightweight monotone-quantile student architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rcoi.models.decomposition import MovingAverage

from experiments.unified_temporal_27_v3.model import _effective_kernel

from .models import EPSILON, MOVING_AVERAGE_KERNEL, _scale_column


QUANTILE_GRID = torch.tensor(
    [
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.99,
    ],
    dtype=torch.float64,
)
STUDENT_HIDDEN_WIDTH = 16
STUDENT_OUTPUT_WIDTH = 22


def _validate_quantile_tensor(quantiles: torch.Tensor) -> None:
    if quantiles.ndim < 1 or quantiles.shape[-1] != QUANTILE_GRID.numel():
        raise ValueError("quantiles must end with the frozen 21-point grid")
    if not torch.is_floating_point(quantiles) or not bool(torch.isfinite(quantiles).all()):
        raise ValueError("quantiles must be finite floating-point values")
    if bool((quantiles < 0).any()):
        raise ValueError("quantiles must be nonnegative")
    if bool((quantiles[..., 1:] < quantiles[..., :-1]).any()):
        raise ValueError("quantiles must be monotone")


def postprocess_student_quantiles(
    p0: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Enforce the frozen zero-atom biconditional on the common grid."""
    _validate_quantile_tensor(quantiles)
    probability_zero = torch.as_tensor(
        p0, dtype=quantiles.dtype, device=quantiles.device
    )
    if probability_zero.shape != quantiles.shape[:-1]:
        raise ValueError("p0 shape must equal the quantile leading shape")
    if (
        not bool(torch.isfinite(probability_zero).all())
        or bool((probability_zero <= 0).any())
        or bool((probability_zero >= 1).any())
    ):
        raise ValueError("p0 must contain finite probabilities strictly inside (0,1)")
    if bool((quantiles <= 0).any()):
        raise ValueError("preprocessed quantiles must be strictly positive")

    grid = QUANTILE_GRID.to(dtype=quantiles.dtype, device=quantiles.device)
    grid = grid.reshape((1,) * probability_zero.ndim + (grid.numel(),))
    expected_zero = grid <= probability_zero.unsqueeze(-1)
    result = torch.where(expected_zero, torch.zeros_like(quantiles), quantiles)
    if bool((result[..., 1:] < result[..., :-1]).any()):
        raise RuntimeError("student quantile postprocessing violated monotonicity")
    if not torch.equal(result == 0, expected_zero.expand_as(result)):
        raise RuntimeError("student zero-mass/quantile biconditional failed")
    return result


def quantile_integral_mean(quantiles: torch.Tensor) -> torch.Tensor:
    """Integrate quantiles with endpoint holds on [0,.01] and [.99,1]."""
    _validate_quantile_tensor(quantiles)
    grid = QUANTILE_GRID.to(dtype=quantiles.dtype, device=quantiles.device)
    endpoint_grid = torch.cat(
        [grid.new_tensor([0.0]), grid, grid.new_tensor([1.0])], dim=0
    )
    endpoint_quantiles = torch.cat(
        [quantiles[..., :1], quantiles, quantiles[..., -1:]], dim=-1
    )
    return torch.trapezoid(endpoint_quantiles, endpoint_grid, dim=-1)


class MonotoneQuantileStudent(nn.Module):
    """Small DLinear student with a shared per-horizon scalar MLP."""

    def __init__(self, lookback: int, horizon: int) -> None:
        super().__init__()
        if int(lookback) < 3 or int(horizon) <= 0:
            raise ValueError("lookback must be at least three and horizon must be positive")
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.moving_average_kernel = _effective_kernel(
            self.lookback, MOVING_AVERAGE_KERNEL
        )
        self.decomp = MovingAverage(self.moving_average_kernel)
        self.trend = nn.Linear(self.lookback, self.horizon)
        self.season = nn.Linear(self.lookback, self.horizon)
        self.output_head = nn.Sequential(
            nn.Linear(1, STUDENT_HIDDEN_WIDTH),
            nn.SiLU(),
            nn.Linear(STUDENT_HIDDEN_WIDTH, STUDENT_OUTPUT_WIDTH),
        )

    def forward(
        self, history: torch.Tensor, input_scale: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if history.ndim != 2 or history.shape[1] != self.lookback:
            raise ValueError(
                f"history must have shape [batch,{self.lookback}], got {tuple(history.shape)}"
            )
        if not torch.is_floating_point(history) or not bool(torch.isfinite(history).all()):
            raise ValueError("history must be finite and floating point")
        scale = _scale_column(input_scale, history)
        normalized = history / scale
        trend_component = self.decomp(normalized)
        latent = self.trend(trend_component) + self.season(
            normalized - trend_component
        )
        raw = self.output_head(latent.unsqueeze(-1))
        p0 = EPSILON + (1.0 - 2.0 * EPSILON) * torch.sigmoid(raw[..., 0])
        base = F.softplus(raw[..., 1:2]) + EPSILON
        increments = F.softplus(raw[..., 2:]) + EPSILON
        normalized_quantiles = torch.cat(
            [base, base + torch.cumsum(increments, dim=-1)], dim=-1
        )
        quantiles = normalized_quantiles * scale.unsqueeze(-1)
        evaluation_quantiles = postprocess_student_quantiles(p0, quantiles)
        return {
            "p0": p0,
            "quantiles": quantiles,
            "normalized_quantiles": normalized_quantiles,
            "evaluation_quantiles": evaluation_quantiles,
            "mean": quantile_integral_mean(evaluation_quantiles),
            "latent": latent,
            "raw_parameters": raw,
        }


def build_student(*, lookback: int, horizon: int) -> MonotoneQuantileStudent:
    return MonotoneQuantileStudent(lookback=lookback, horizon=horizon)


__all__ = [
    "MonotoneQuantileStudent",
    "QUANTILE_GRID",
    "STUDENT_HIDDEN_WIDTH",
    "STUDENT_OUTPUT_WIDTH",
    "build_student",
    "postprocess_student_quantiles",
    "quantile_integral_mean",
]
