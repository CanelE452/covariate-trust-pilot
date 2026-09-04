"""Predictive distribution heads sharing a ``[batch, horizon]`` interface."""

from __future__ import annotations

import math
from itertools import product

import torch
from scipy import special as scipy_special

from .vendor.tweediegp.tweedie import compound_parameters, full_cdf, full_log_prob


CDF_ROUNDOFF_TOLERANCE = 1e-8


def _checked_cdf_probability(value: torch.Tensor, *, family: str) -> torch.Tensor:
    """Clamp only numerical roundoff; expose material probability defects."""
    if (
        not bool(torch.isfinite(value).all())
        or bool((value < -CDF_ROUNDOFF_TOLERANCE).any())
        or bool((value > 1.0 + CDF_ROUNDOFF_TOLERANCE).any())
    ):
        raise RuntimeError(f"{family} CDF probability error exceeds roundoff tolerance")
    return value.clamp(0.0, 1.0)


def _parameters(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    tensors = tuple(torch.as_tensor(value) for value in values)
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("distribution parameters must have [batch, horizon] shape")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("distribution parameters must share [batch, horizon] shape")
    if not all(torch.is_floating_point(value) for value in tensors):
        raise ValueError("distribution parameters must use a floating dtype")
    if len({value.dtype for value in tensors}) != 1:
        raise ValueError("distribution parameters must share one dtype")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("distribution parameters must share one device")
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("distribution parameters must be finite")
    return tensors


def _value(value: torch.Tensor | float, parameter: torch.Tensor) -> torch.Tensor:
    candidate = torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)
    try:
        torch.broadcast_shapes(candidate.shape, parameter.shape)
    except RuntimeError as error:
        raise ValueError("value must broadcast to [batch, horizon]") from error
    return candidate


def _cdf_value(value: torch.Tensor | float, parameter: torch.Tensor) -> torch.Tensor:
    candidate = torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)
    # A one-dimensional CDF argument is the common support/query axis, not a
    # horizon-shaped value.  Per-case values must use [batch,horizon] (or a
    # normally broadcastable higher-rank form).
    if candidate.ndim == 1:
        candidate = candidate.reshape((-1,) + (1,) * parameter.ndim)
    try:
        candidate, _ = torch.broadcast_tensors(candidate, parameter)
    except RuntimeError as error:
        raise ValueError(
            "CDF query must be scalar, one-dimensional support, or broadcast to [batch,horizon]"
        ) from error
    if bool(torch.isnan(candidate).any()):
        raise ValueError("CDF query must contain finite values or infinite endpoints")
    return candidate


def _quantile_bisect(distribution: object, q: torch.Tensor, *, tolerance: float = 1e-5) -> torch.Tensor:
    probs = torch.as_tensor(q, dtype=distribution.mu.dtype, device=distribution.mu.device)
    if (
        probs.ndim != 1
        or not bool(torch.isfinite(probs).all())
        or bool((probs < 0).any())
        or bool((probs > 1).any())
    ):
        raise ValueError("quantile probabilities must be finite, one-dimensional, and in [0, 1]")
    target = probs.reshape((-1,) + (1,) * distribution.mu.ndim)
    search_target = torch.where(target == 1, torch.zeros_like(target), target)
    low = torch.zeros_like(target + distribution.mu)
    variance = distribution.variance()
    high = (distribution.mean() + 12 * torch.sqrt(variance) + 1).expand_as(low).clone()
    for _ in range(64):
        need_more = distribution.cdf(high) < search_target
        if not bool(need_more.any()):
            break
        high = torch.where(need_more, high * 2, high)
    else:
        raise RuntimeError("failed to bracket requested distribution quantile")
    for _ in range(48):
        middle = (low + high) / 2
        low = torch.where(distribution.cdf(middle) < search_target, middle, low)
        high = torch.where(distribution.cdf(middle) >= search_target, middle, high)
        if float((high - low).max()) <= tolerance:
            break
    zero = torch.zeros_like(high)
    infinity = torch.full_like(high, torch.inf)
    atom = distribution.p_zero().expand_as(high)
    return torch.where(target == 1, infinity, torch.where(target <= atom, zero, high))


def _discrete_quantile(distribution: object, q: torch.Tensor) -> torch.Tensor:
    """Return ``inf{k in N_0: F(k) >= q}`` for an unbounded count support."""
    probs = torch.as_tensor(q, dtype=distribution.mu.dtype, device=distribution.mu.device)
    if (
        probs.ndim != 1
        or not bool(torch.isfinite(probs).all())
        or bool((probs < 0).any())
        or bool((probs > 1).any())
    ):
        raise ValueError("quantile probabilities must be finite, one-dimensional, and in [0, 1]")
    target = probs.reshape((-1,) + (1,) * distribution.mu.ndim)
    zero = torch.zeros_like(target + distribution.mu)
    infinity = torch.full_like(zero, torch.inf)
    atom = distribution.p_zero().expand_as(zero)
    active = (target > atom) & (target < 1)
    search_target = torch.where(active, target, torch.zeros_like(target))
    high = torch.ceil(distribution.mean() + 12 * torch.sqrt(distribution.variance()) + 1).to(torch.long).expand_as(zero).clone()
    for _ in range(64):
        need_more = active & (distribution.cdf(high) < search_target)
        if not bool(need_more.any()):
            break
        high = torch.where(need_more, high * 2 + 1, high)
    else:
        raise RuntimeError("failed to bracket requested discrete distribution quantile")
    low = torch.zeros_like(high)
    for _ in range(64):
        unfinished = active & (low < high)
        if not bool(unfinished.any()):
            break
        middle = torch.div(low + high, 2, rounding_mode="floor")
        below = distribution.cdf(middle) < search_target
        low = torch.where(unfinished & below, middle + 1, low)
        high = torch.where(unfinished & ~below, middle, high)
    else:
        raise RuntimeError("discrete distribution quantile bisection did not converge")
    return torch.where(target == 1, infinity, torch.where(target <= atom, zero, low.to(distribution.mu.dtype)))


def simplex_grid(dimensions: int, *, step: float = 0.1, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Enumerate every nonnegative ``step``-simplex weight vector deterministically."""
    reciprocal = round(1.0 / step)
    if dimensions < 2 or step <= 0 or not math.isclose(reciprocal * step, 1.0, abs_tol=1e-12):
        raise ValueError("dimensions must be at least two and step must divide one")
    states = [state for state in product(range(reciprocal + 1), repeat=dimensions) if sum(state) == reciprocal]
    return torch.tensor(states, dtype=dtype) / reciprocal


def pooled_cdf_quantile(cdfs: torch.Tensor, support: torch.Tensor, q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Invert a finite-support weighted CDF mixture without model-specific state."""
    cdfs = torch.as_tensor(cdfs)
    support = torch.as_tensor(support, dtype=cdfs.dtype, device=cdfs.device)
    weights = torch.as_tensor(weights, dtype=cdfs.dtype, device=cdfs.device)
    probabilities = torch.as_tensor(q, dtype=cdfs.dtype, device=cdfs.device)
    if cdfs.ndim < 2 or support.ndim != 1 or cdfs.shape[1] != support.numel():
        raise ValueError("cdfs must be [models, support, ...] with matching one-dimensional support")
    if not torch.is_floating_point(cdfs):
        raise ValueError("cdfs must use a floating dtype")
    weight_tolerance = max(
        1e-12,
        4.0 * torch.finfo(cdfs.dtype).eps * max(1, int(weights.numel())),
    )
    weight_sum = weights.to(torch.float64).sum()
    if (
        weights.ndim != 1
        or weights.numel() != cdfs.shape[0]
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or abs(float(weight_sum) - 1.0) > weight_tolerance
    ):
        raise ValueError("weights must be nonnegative simplex weights")
    if (
        probabilities.ndim != 1
        or not bool(torch.isfinite(probabilities).all())
        or bool((probabilities < 0).any())
        or bool((probabilities > 1).any())
    ):
        raise ValueError("q must contain finite, one-dimensional probabilities in [0, 1]")
    expected_support = torch.arange(
        support.numel(), dtype=support.dtype, device=support.device
    )
    if not bool(torch.isfinite(support).all()) or not torch.equal(
        support, expected_support
    ):
        raise ValueError("support must be the complete count support 0,1,...,K")
    if not bool(torch.isfinite(cdfs).all()) or bool((cdfs < 0).any()) or bool((cdfs > 1).any()) or bool((cdfs[:, 1:] < cdfs[:, :-1]).any()):
        raise ValueError("each CDF must be finite, in [0, 1], and monotone")
    mass_tolerance = max(1e-12, 8.0 * torch.finfo(cdfs.dtype).eps)
    if bool((torch.abs(cdfs[:, -1] - 1.0) > mass_tolerance).any()):
        raise ValueError("each terminal CDF mass must equal one within roundoff tolerance")
    normalized_weights = weights / weights.sum()
    mixture = torch.sum(normalized_weights.reshape((-1,) + (1,) * (cdfs.ndim - 1)) * cdfs, dim=0)
    if bool((mixture[1:] < mixture[:-1]).any()):
        raise ValueError("CDF mixture must be monotone on support")
    target = probabilities.reshape((-1,) + (1,) * (mixture.ndim - 1))
    if bool((mixture[-1].unsqueeze(0) < target).any()):
        raise ValueError("terminal mixture CDF mass does not reach every requested q")
    hits = mixture.unsqueeze(0) >= target.unsqueeze(1)
    indices = torch.argmax(hits.to(torch.int64), dim=1)
    return support[indices]


class NegativeBinomialDistribution:
    def __init__(self, mu: torch.Tensor, r: torch.Tensor):
        self.mu, self.r = _parameters(mu, r)
        if bool((self.mu <= 0).any()) or bool((self.r <= 0).any()):
            raise ValueError("NB requires mu > 0 and r > 0")

    def log_prob(self, y: torch.Tensor | float) -> torch.Tensor:
        y = _value(y, self.mu)
        valid = (y >= 0) & (y == torch.floor(y))
        safe = torch.where(valid, y, torch.zeros_like(y))
        result = (
            torch.lgamma(safe + self.r) - torch.lgamma(self.r) - torch.lgamma(safe + 1)
            + self.r * torch.log(self.r / (self.r + self.mu))
            + safe * torch.log(self.mu / (self.r + self.mu))
        )
        return torch.where(valid, result, torch.full_like(result, -torch.inf))

    def mean(self) -> torch.Tensor:
        return self.mu

    def variance(self) -> torch.Tensor:
        return self.mu + self.mu.square() / self.r

    def p_zero(self) -> torch.Tensor:
        return torch.exp(self.log_prob(torch.zeros_like(self.mu)))

    def cdf(self, y: torch.Tensor | float) -> torch.Tensor:
        y = _cdf_value(y, self.mu)
        positive_infinity = torch.isposinf(y)
        negative_infinity = torch.isneginf(y)
        if bool((torch.isfinite(y) & (y > 200_000)).any()):
            raise RuntimeError("exact NB CDF summation exceeded the 200000-count resource guard")
        # Evaluate the regularized incomplete-beta identity directly.  This
        # avoids materializing a [support,query,batch,horizon] PMF tensor during
        # repeated pooled-CDF inversion.
        finite_endpoint_safe = torch.where(positive_infinity | negative_infinity, torch.zeros_like(y), y)
        upper = torch.floor(finite_endpoint_safe).to(torch.long)
        nonnegative = upper >= 0
        safe_upper = torch.clamp_min(upper, 0)
        broadcast_y, broadcast_mu, broadcast_r = torch.broadcast_tensors(
            y, self.mu, self.r
        )
        del broadcast_y
        success_probability = broadcast_r / (broadcast_r + broadcast_mu)
        result_numpy = scipy_special.betainc(
            broadcast_r.detach().to(torch.float64).cpu().numpy(),
            (safe_upper + 1).detach().to(torch.float64).cpu().numpy(),
            success_probability.detach().to(torch.float64).cpu().numpy(),
        )
        result64 = torch.as_tensor(
            result_numpy, dtype=torch.float64, device=self.mu.device
        ) * nonnegative.to(torch.float64)
        result = _checked_cdf_probability(result64, family="NB").to(self.mu.dtype)
        return torch.where(positive_infinity, torch.ones_like(result), torch.where(negative_infinity, torch.zeros_like(result), result))

    def quantile(self, q: torch.Tensor) -> torch.Tensor:
        return _discrete_quantile(self, q)

    def sample(self, n: int, seed: int) -> torch.Tensor:
        if n < 1:
            raise ValueError("n must be positive")
        generator = torch.Generator(device=self.mu.device).manual_seed(seed)
        shape = (n,) + tuple(self.mu.shape)
        r = self.r.expand(shape)
        rate = self.mu.expand(shape) / r
        latent = torch._standard_gamma(r, generator=generator) * rate
        return torch.poisson(latent, generator=generator)


class ShiftedHurdleNegativeBinomialDistribution:
    def __init__(self, pi: torch.Tensor, mu: torch.Tensor, r: torch.Tensor):
        self.pi, self.mu, self.r = _parameters(pi, mu, r)
        if bool((self.pi <= 0).any()) or bool((self.pi >= 1).any()):
            raise ValueError("HSNB requires 0 < pi < 1")
        if bool((self.mu <= 0).any()) or bool((self.r <= 0).any()):
            raise ValueError("HSNB requires mu > 0 and r > 0")
        self._tail = NegativeBinomialDistribution(self.mu, self.r)

    def log_prob(self, y: torch.Tensor | float) -> torch.Tensor:
        y = _value(y, self.mu)
        zeros = y == 0
        positive = (y >= 1) & (y == torch.floor(y))
        tail = torch.log(self.pi) + self._tail.log_prob(y - 1)
        result = torch.where(zeros, torch.log1p(-self.pi), torch.full_like(tail, -torch.inf))
        return torch.where(positive, tail, result)

    def mean(self) -> torch.Tensor:
        return self.pi * (1 + self.mu)

    def variance(self) -> torch.Tensor:
        tail_mean = 1 + self.mu
        tail_variance = self._tail.variance()
        return self.pi * tail_variance + self.pi * (1 - self.pi) * tail_mean.square()

    def p_zero(self) -> torch.Tensor:
        return 1 - self.pi

    def cdf(self, y: torch.Tensor | float) -> torch.Tensor:
        y = _cdf_value(y, self.mu)
        pi64 = self.pi.to(torch.float64)
        positive_result = (1.0 - pi64) + pi64 * self._tail.cdf(y - 1).to(torch.float64)
        result = torch.where(
            y < 0,
            torch.zeros_like(y, dtype=torch.float64),
            torch.where(y < 1, 1.0 - pi64, positive_result),
        )
        return _checked_cdf_probability(result, family="HSNB").to(self.mu.dtype)

    def quantile(self, q: torch.Tensor) -> torch.Tensor:
        return _discrete_quantile(self, q)

    def sample(self, n: int, seed: int) -> torch.Tensor:
        if n < 1:
            raise ValueError("n must be positive")
        generator = torch.Generator(device=self.mu.device).manual_seed(seed)
        active = torch.rand((n,) + tuple(self.mu.shape), dtype=self.mu.dtype, device=self.mu.device, generator=generator) < self.pi
        r = self.r.expand_as(active)
        latent = torch._standard_gamma(r, generator=generator) * (self.mu.expand_as(active) / r)
        return active.to(self.mu.dtype) * (1 + torch.poisson(latent, generator=generator))


class TweedieDistribution:
    def __init__(self, mu: torch.Tensor, phi: torch.Tensor, p: torch.Tensor):
        self.mu, self.phi, self.p = _parameters(mu, phi, p)
        if bool((self.mu <= 0).any()) or bool((self.phi <= 0).any()):
            raise ValueError("Tweedie requires mu > 0 and phi > 0")
        if bool((self.p < 1.05).any()) or bool((self.p > 1.95).any()):
            raise ValueError("full Tweedie contract requires 1.05 <= p <= 1.95")

    def log_prob(self, y: torch.Tensor | float) -> torch.Tensor:
        return full_log_prob(_value(y, self.mu), self.mu, self.phi, self.p)

    def mean(self) -> torch.Tensor:
        return self.mu

    def variance(self) -> torch.Tensor:
        return self.phi * self.mu.pow(self.p)

    def p_zero(self) -> torch.Tensor:
        rate, _, _ = compound_parameters(self.mu, self.phi, self.p)
        return torch.exp(-rate)

    def cdf(self, y: torch.Tensor | float) -> torch.Tensor:
        return full_cdf(_cdf_value(y, self.mu), self.mu, self.phi, self.p)

    def quantile(self, q: torch.Tensor) -> torch.Tensor:
        return _quantile_bisect(self, q)

    def sample(self, n: int, seed: int) -> torch.Tensor:
        if n < 1:
            raise ValueError("n must be positive")
        generator = torch.Generator(device=self.mu.device).manual_seed(seed)
        rate, concentration, scale = compound_parameters(self.mu, self.phi, self.p)
        counts = torch.poisson(rate.expand((n,) + tuple(rate.shape)), generator=generator)
        shapes = counts * concentration.expand_as(counts)
        safe_shapes = torch.where(counts > 0, shapes, torch.ones_like(shapes))
        gamma_sum = torch._standard_gamma(safe_shapes, generator=generator) * scale.expand_as(counts)
        return torch.where(counts > 0, gamma_sum, torch.zeros_like(counts))
