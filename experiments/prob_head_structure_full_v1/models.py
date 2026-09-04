"""Parameter-matched DLinear teachers for the three frozen count families."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rcoi.models.decomposition import MovingAverage

from experiments.unified_temporal_27_v3.model import _effective_kernel, inverse_softplus

from .distributions import (
    NegativeBinomialDistribution,
    ShiftedHurdleNegativeBinomialDistribution,
    TweedieDistribution,
)


EPSILON = 1e-6
MOVING_AVERAGE_KERNEL = 25
HEAD_NAMES = ("NB", "HSNB", "TWEEDIE_FULL")
ADAPTER_WIDTHS = {"NB": 19, "HSNB": 14, "TWEEDIE_FULL": 29}
OUTPUT_MULTIPLICITIES = {"NB": 2, "HSNB": 3, "TWEEDIE_FULL": 1}


def _scale_column(
    input_scale: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    scale = torch.as_tensor(input_scale, dtype=history.dtype, device=history.device)
    if scale.ndim == 2 and scale.shape[1] == 1:
        scale = scale[:, 0]
    if scale.ndim != 1 or scale.shape[0] != history.shape[0]:
        raise ValueError("input_scale must have shape [batch] or [batch,1]")
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("input_scale must contain finite positive train-only scales")
    return scale[:, None]


class ProbabilisticDLinear(nn.Module):
    """The frozen common trunk with only its predictive family head changed."""

    def __init__(self, head_name: str, lookback: int, horizon: int) -> None:
        super().__init__()
        if head_name not in HEAD_NAMES:
            raise ValueError(f"unknown head_name {head_name!r}")
        if lookback < 3 or horizon <= 0:
            raise ValueError("lookback must be at least three and horizon must be positive")

        self.head_name = head_name
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.moving_average_kernel = _effective_kernel(
            self.lookback, MOVING_AVERAGE_KERNEL
        )
        self.decomp = MovingAverage(self.moving_average_kernel)
        self.trend = nn.Linear(self.lookback, self.horizon)
        self.season = nn.Linear(self.lookback, self.horizon)

        width = ADAPTER_WIDTHS[head_name]
        multiplicity = OUTPUT_MULTIPLICITIES[head_name]
        self.adapter = nn.Sequential(
            nn.Linear(self.horizon, width),
            nn.SiLU(),
            nn.Linear(width, self.horizon * multiplicity),
        )
        if head_name == "TWEEDIE_FULL":
            self.raw_phi = nn.Parameter(
                torch.tensor(inverse_softplus(1.0 - EPSILON), dtype=torch.float32)
            )
            self.raw_p = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _latent(self, history: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        normalized = history / scale
        trend = self.decomp(normalized)
        return self.trend(trend) + self.season(normalized - trend)

    def forward(
        self,
        history: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor | object]:
        if history.ndim != 2 or history.shape[1] != self.lookback:
            raise ValueError(
                f"history must have shape [batch,{self.lookback}], got {tuple(history.shape)}"
            )
        if not torch.is_floating_point(history) or not bool(torch.isfinite(history).all()):
            raise ValueError("history must be finite and floating point")
        scale = _scale_column(input_scale, history)
        latent = self._latent(history, scale)
        raw = self.adapter(latent).reshape(
            history.shape[0], self.horizon, OUTPUT_MULTIPLICITIES[self.head_name]
        )
        normalized_mu = F.softplus(raw[..., 0]) + EPSILON
        mu = normalized_mu * scale

        if self.head_name == "NB":
            r = F.softplus(raw[..., 1]) + EPSILON
            distribution = NegativeBinomialDistribution(mu, r)
            return {
                "distribution": distribution,
                "latent": latent,
                "raw_parameters": raw,
                "normalized_mu": normalized_mu,
                "mu": mu,
                "r": r,
            }

        if self.head_name == "HSNB":
            pi = EPSILON + (1.0 - 2.0 * EPSILON) * torch.sigmoid(raw[..., 0])
            normalized_mu = F.softplus(raw[..., 1]) + EPSILON
            mu = normalized_mu * scale
            r = F.softplus(raw[..., 2]) + EPSILON
            distribution = ShiftedHurdleNegativeBinomialDistribution(pi, mu, r)
            return {
                "distribution": distribution,
                "latent": latent,
                "raw_parameters": raw,
                "normalized_mu": normalized_mu,
                "mu": mu,
                "pi": pi,
                "r": r,
            }

        phi_scalar = F.softplus(self.raw_phi) + EPSILON
        p_scalar = 1.05 + 0.90 * torch.sigmoid(self.raw_p)
        phi = phi_scalar.to(mu.dtype).expand_as(mu)
        p = p_scalar.to(mu.dtype).expand_as(mu)
        distribution = TweedieDistribution(mu, phi, p)
        return {
            "distribution": distribution,
            "latent": latent,
            "raw_parameters": raw,
            "normalized_mu": normalized_mu,
            "mu": mu,
            "phi": phi,
            "p": p,
        }


def build_teacher(head_name: str, *, lookback: int, horizon: int) -> ProbabilisticDLinear:
    return ProbabilisticDLinear(head_name, lookback, horizon)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = [
    "ADAPTER_WIDTHS",
    "EPSILON",
    "HEAD_NAMES",
    "MOVING_AVERAGE_KERNEL",
    "OUTPUT_MULTIPLICITIES",
    "ProbabilisticDLinear",
    "build_teacher",
    "count_parameters",
]
