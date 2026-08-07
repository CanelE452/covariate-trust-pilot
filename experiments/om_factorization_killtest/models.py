"""Model variants sharing one DLinear backbone, plus the two baselines.

Every DLinear variant reuses the SAME building blocks as
``experiments/unified_temporal_27_v3/model.py``: the same moving-average
decomposition, the same trend/season linear maps, the same scale-equivariant
normalisation.  Only the output heads and the loss differ.

Parameter counts (lookback 96, horizon 24):

    M0    trend + season                                       4656
    M1    trend + season + occurrence head + magnitude head    5856
    M2    M1 + one scalar log-dispersion                       5857
    M0PM  trend + season + two point heads                     5856

M1 carries 25.8% more parameters than M0, which is above the 5% rule, so the
parameter-matched point control M0PM is included.  It stays a pure point
predictor with an MSE loss and no extra input or supervision; its two stacked
linear heads exist only to match the count.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..unified_temporal_27_v3.model import EPSILON, _effective_kernel, inverse_softplus
from rcoi.models.decomposition import MovingAverage

MOVING_AVERAGE_KERNEL = 25


class _Backbone(nn.Module):
    """Shared trunk: identical across every DLinear variant."""

    def __init__(self, lookback: int, horizon: int) -> None:
        super().__init__()
        self.lookback, self.horizon = int(lookback), int(horizon)
        self.decomp = MovingAverage(_effective_kernel(lookback, MOVING_AVERAGE_KERNEL))
        self.trend = nn.Linear(self.lookback, self.horizon)
        self.season = nn.Linear(self.lookback, self.horizon)

    def latent(self, history: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        normalized = history / scale
        trend = self.decomp(normalized)
        return self.trend(trend) + self.season(normalized - trend)


def _scale_column(input_scale: torch.Tensor) -> torch.Tensor:
    if input_scale.ndim == 2 and input_scale.shape[1] == 1:
        input_scale = input_scale[:, 0]
    return input_scale.unsqueeze(1).clamp_min(EPSILON)


class PointDLinear(_Backbone):
    """M0: one y_hat per horizon step, trained with MSE."""

    variant = "point"

    def forward(self, history, input_scale):
        scale = _scale_column(input_scale).to(history.dtype)
        latent = self.latent(history, scale)
        mean = latent * scale
        return {"mean_prediction": mean, "point_prediction": mean}


class PointDLinearParamMatched(_Backbone):
    """M0PM: same output and loss as M0, parameter count matched to M1."""

    variant = "point"

    def __init__(self, lookback: int, horizon: int) -> None:
        super().__init__(lookback, horizon)
        self.point_head_a = nn.Linear(self.horizon, self.horizon)
        self.point_head_b = nn.Linear(self.horizon, self.horizon)
        with torch.no_grad():
            for head in (self.point_head_a, self.point_head_b):
                head.weight.copy_(torch.eye(self.horizon))
                head.bias.zero_()

    def forward(self, history, input_scale):
        scale = _scale_column(input_scale).to(history.dtype)
        latent = self.point_head_b(self.point_head_a(self.latent(history, scale)))
        mean = latent * scale
        return {"mean_prediction": mean, "point_prediction": mean}


class FactorizedDLinear(_Backbone):
    """M1: occurrence logit and positive magnitude mean from the same latent."""

    variant = "factorized"

    def __init__(self, lookback: int, horizon: int, *,
                 initial_occurrence_rate: float = 0.25,
                 initial_normalized_positive_mean: float = 1.0) -> None:
        super().__init__(lookback, horizon)
        self.occurrence_head = nn.Linear(self.horizon, self.horizon)
        self.magnitude_head = nn.Linear(self.horizon, self.horizon)
        rate = min(max(float(initial_occurrence_rate), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.occurrence_head.bias.fill_(math.log(rate / (1.0 - rate)))
            self.magnitude_head.bias.fill_(
                inverse_softplus(initial_normalized_positive_mean))

    def forward(self, history, input_scale):
        scale = _scale_column(input_scale).to(history.dtype)
        latent = self.latent(history, scale)
        logits = self.occurrence_head(latent)
        p = torch.sigmoid(logits)
        mu = F.softplus(self.magnitude_head(latent)) * scale + EPSILON
        return {"occurrence_logits": logits, "p_prediction": p,
                "mu_prediction": mu, "mean_prediction": p * mu}


class HurdleZTNBDLinear(FactorizedDLinear):
    """M2: same two heads plus a learnable ZTNB dispersion.

    The magnitude head emits the UNTRUNCATED NB mean.  The zero-truncated mean
    that the forecast needs follows in closed form,

        P(0) = (r / (r + m_nb))**r,      mu_ztnb = m_nb / (1 - P(0))

    whereas going the other way has no closed form.  Emitting m_nb therefore
    keeps both the likelihood and the mean exact; emitting the truncated mean
    would force an inner solve inside every training step.
    """

    variant = "hurdle_ztnb"

    def __init__(self, lookback: int, horizon: int, *, initial_dispersion: float = 2.0,
                 **kwargs) -> None:
        super().__init__(lookback, horizon, **kwargs)
        self.log_dispersion = nn.Parameter(
            torch.tensor(math.log(float(initial_dispersion)), dtype=torch.float32))

    def forward(self, history, input_scale):
        scale = _scale_column(input_scale).to(history.dtype)
        latent = self.latent(history, scale)
        logits = self.occurrence_head(latent)
        p = torch.sigmoid(logits)
        nb_mean = F.softplus(self.magnitude_head(latent)) * scale + EPSILON
        r = F.softplus(self.log_dispersion) + EPSILON
        log_zero = r * (torch.log(r) - torch.log(r + nb_mean))
        survival = (-torch.expm1(log_zero)).clamp_min(1e-7)
        mu = nb_mean / survival
        return {"occurrence_logits": logits, "p_prediction": p,
                "nb_mean": nb_mean, "mu_prediction": mu,
                "mean_prediction": p * mu, "dispersion": r}


BUILDERS = {
    "M0_point_mse": PointDLinear,
    "M0PM_point_mse_param_matched": PointDLinearParamMatched,
    "M1_factorized_mean": FactorizedDLinear,
    "M2_hurdle_ztnb": HurdleZTNBDLinear,
}


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def point_loss(outputs, target, occurrence, mask, positive_variance):
    err = (outputs["mean_prediction"] - target) ** 2
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def factorized_loss(outputs, target, occurrence, mask, positive_variance):
    """mean BCE over all masked steps + mean normalised MSE over positives only."""
    bce = F.binary_cross_entropy_with_logits(
        outputs["occurrence_logits"], occurrence, reduction="none")
    occurrence_term = (bce * mask).sum() / mask.sum().clamp_min(1.0)

    positive = mask * occurrence
    n_positive = positive.sum()
    if n_positive < 1.0:
        # No positive event in this batch: skip the magnitude term rather than
        # dividing by zero.  lambda is fixed at 1 and never tuned per cell.
        return occurrence_term
    magnitude_err = (target - outputs["mu_prediction"]) ** 2
    magnitude_term = (magnitude_err * positive).sum() / (n_positive * positive_variance)
    return occurrence_term + magnitude_term


def _ztnb_nll_terms(value, nb_mean, dispersion):
    """-log P(M = value) for the zero-truncated NB with NB mean ``nb_mean``."""
    r = dispersion
    m = nb_mean.clamp_min(1e-6)
    log_prob = torch.lgamma(value + r) - torch.lgamma(r) - torch.lgamma(value + 1.0)
    log_prob = log_prob + r * (torch.log(r) - torch.log(r + m))
    log_prob = log_prob + value * (torch.log(m) - torch.log(r + m))
    log_zero = r * (torch.log(r) - torch.log(r + m))
    log_survival = torch.log((-torch.expm1(log_zero)).clamp_min(1e-7))
    return -(log_prob - log_survival)


def hurdle_ztnb_loss(outputs, target, occurrence, mask, positive_variance):
    bce = F.binary_cross_entropy_with_logits(
        outputs["occurrence_logits"], occurrence, reduction="none")
    occurrence_term = (bce * mask).sum() / mask.sum().clamp_min(1.0)

    positive = mask * occurrence
    n_positive = positive.sum()
    if n_positive < 1.0:
        return occurrence_term
    nll = _ztnb_nll_terms(target.clamp_min(1.0), outputs["nb_mean"],
                          outputs["dispersion"])
    return occurrence_term + (nll * positive).sum() / n_positive


LOSSES = {
    "M0_point_mse": point_loss,
    "M0PM_point_mse_param_matched": point_loss,
    "M1_factorized_mean": factorized_loss,
    "M2_hurdle_ztnb": hurdle_ztnb_loss,
}


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def seasonal_naive(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    """B0: repeat the value one period earlier."""
    history = np.asarray(history, dtype=np.float64)
    out = np.empty((history.shape[0], horizon))
    for h in range(horizon):
        out[:, h] = history[:, -period + (h % period)]
    return out


def tsb_state(series: np.ndarray, alpha: float, beta: float) -> tuple[float, float]:
    """Teunter-Syntetos-Babai recursion over one observed history."""
    series = np.asarray(series, dtype=np.float64)
    nonzero = series[series > 0]
    p = float((series > 0).mean()) if series.size else 0.0
    m = float(nonzero.mean()) if nonzero.size else 0.0
    for value in series:
        if value > 0:
            p += alpha * (1.0 - p)
            m += beta * (value - m)
        else:
            p += alpha * (0.0 - p)
    return p, m


def tsb_forecast(history: np.ndarray, horizon: int, alpha: float,
                 beta: float) -> tuple[np.ndarray, np.ndarray]:
    """B1: flat p_hat and mu_hat over the horizon; mean = p_hat * mu_hat."""
    history = np.asarray(history, dtype=np.float64)
    p = np.empty(history.shape[0])
    m = np.empty(history.shape[0])
    for i, row in enumerate(history):
        p[i], m[i] = tsb_state(row, alpha, beta)
    return (np.repeat(p[:, None], horizon, axis=1),
            np.repeat(m[:, None], horizon, axis=1))
