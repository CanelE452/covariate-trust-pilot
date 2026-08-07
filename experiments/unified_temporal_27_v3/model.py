"""One scale-equivariant DLinear backbone shared by all v3 objectives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from rcoi.models.decomposition import MovingAverage


EPSILON = 1e-6


def _effective_kernel(lookback: int, requested: int) -> int:
    if lookback < 3:
        raise ValueError("lookback must be at least 3")
    kernel = min(int(requested), lookback if lookback % 2 else lookback - 1)
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 1:
        raise ValueError("moving-average kernel must be positive")
    return kernel


def inverse_softplus(value: float) -> float:
    value = max(float(value), EPSILON)
    return value if value > 20.0 else math.log(math.expm1(value))


class UnifiedDLinear(nn.Module):
    """DLinear with point, occurrence, conditional-mean, and gap heads.

    Every method instantiates this exact architecture.  MSE uses
    ``point_prediction``; both Gamma methods use
    ``demand_prediction = p_prediction * mu_prediction``.  The model accepts
    only observed history and a train-only scale.  Targets, future occurrence,
    generation patterns, and gap labels are not forward inputs.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        moving_average_kernel: int = 25,
        *,
        dispersion: float = 0.1,
        initial_normalized_positive_mean: float = 1.0,
        initial_occurrence_rate: float = 0.25,
        initial_gap_mean: float | None = None,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()
        if lookback <= 0 or horizon <= 0 or dispersion <= 0.0 or epsilon <= 0.0:
            raise ValueError("lookback/horizon/dispersion/epsilon must be positive")
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.epsilon = float(epsilon)
        kernel = _effective_kernel(self.lookback, moving_average_kernel)

        self.decomp = MovingAverage(kernel)
        self.trend = nn.Linear(self.lookback, self.horizon)
        self.season = nn.Linear(self.lookback, self.horizon)
        self.occurrence_head = nn.Linear(self.horizon, self.horizon)
        self.magnitude_head = nn.Linear(self.horizon, self.horizon)
        self.gap_head = nn.Linear(self.horizon, 1)
        self.register_buffer(
            "dispersion", torch.tensor(float(dispersion), dtype=torch.float32)
        )

        occurrence_rate = min(max(float(initial_occurrence_rate), 1e-4), 1.0 - 1e-4)
        if initial_gap_mean is None:
            initial_gap_mean = 0.5 * (self.horizon + 2.0)
        initial_gap_excess = max(float(initial_gap_mean) - 1.0, EPSILON)
        with torch.no_grad():
            self.occurrence_head.bias.fill_(
                math.log(occurrence_rate / (1.0 - occurrence_rate))
            )
            self.magnitude_head.bias.fill_(
                inverse_softplus(initial_normalized_positive_mean)
            )
            self.gap_head.bias.fill_(inverse_softplus(initial_gap_excess))

    def forward(
        self,
        history: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history.ndim != 2 or history.shape[1] != self.lookback:
            raise ValueError(
                f"history must have shape (batch,{self.lookback}), got {history.shape}"
            )
        if input_scale.ndim == 2 and input_scale.shape[1] == 1:
            input_scale = input_scale[:, 0]
        if input_scale.ndim != 1 or input_scale.shape[0] != history.shape[0]:
            raise ValueError("input_scale must have shape (batch,) or (batch,1)")

        scale = input_scale.to(history.dtype).unsqueeze(1).clamp_min(self.epsilon)
        normalized = history / scale
        trend = self.decomp(normalized)
        residual = normalized - trend
        latent = self.trend(trend) + self.season(residual)

        point_prediction = latent * scale
        occurrence_logits = self.occurrence_head(latent)
        p_prediction = torch.sigmoid(occurrence_logits)
        raw_mu = self.magnitude_head(latent)
        mu_prediction = F.softplus(raw_mu) * scale + self.epsilon
        raw_gap = self.gap_head(latent).squeeze(-1)
        # Do not cap the distance at H+1.  A no-hit label is right-censored:
        # the true next-occurrence distance can be arbitrarily larger than the
        # observed horizon, and predictions above the censoring bound must not
        # be penalised as if H+1 were an exact target.
        gap_prediction = 1.0 + F.softplus(raw_gap)
        return {
            "latent": latent,
            "point_prediction": point_prediction,
            "occurrence_logits": occurrence_logits,
            "p_prediction": p_prediction,
            "raw_mu": raw_mu,
            "mu_prediction": mu_prediction,
            "raw_gap": raw_gap,
            "gap_prediction": gap_prediction,
            "demand_prediction": p_prediction * mu_prediction,
            "dispersion": self.dispersion,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["EPSILON", "UnifiedDLinear", "count_parameters", "inverse_softplus"]
