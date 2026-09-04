"""Apache-2.0 adaptation of TweedieGP@2567d1322c8cc65f19df4f2d1774c610b167fb66.

The original ``Tweedie`` derives compound-Poisson--Gamma parameters from
``mu, phi, rho``.  This minimal derivative retains that normalized density but
uses direct log-mixture summation with helpers on the input device. Float32
inputs accumulate in float64 internally and recast at the public boundary. It
intentionally contains no FixedDispersionTweedie/deviance path.
"""

from __future__ import annotations

import math

import torch


def compound_parameters(mu: torch.Tensor, phi: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Poisson rate, Gamma concentration and Gamma scale for 1<p<2."""
    rate = mu.pow(2 - p) / (phi * (2 - p))
    concentration = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu.pow(p - 1)
    return rate, concentration, scale


def _positive_log_density(y: torch.Tensor, rate: torch.Tensor, concentration: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Normalized density, summing until a log-concave tail bound converges."""
    # Callers evaluate a mixed zero/positive tensor.  Never form log(0) in the
    # masked branch: ``where`` does not reliably protect backwards from NaNs.
    y = torch.where(y > 0, y, torch.ones_like(y))
    running = torch.full_like(y, -torch.inf)
    peak = torch.full_like(y, -torch.inf)
    chunk_size, maximum_terms, log_tolerance = 128, 200_000, 35.0
    for start in range(1, maximum_terms + 1, chunk_size):
        stop = min(start + chunk_size, maximum_terms + 1)
        j = torch.arange(start, stop, dtype=y.dtype, device=y.device)
        expanded = j.reshape((j.numel(),) + (1,) * y.ndim)
        shape = expanded * concentration
        log_poisson = expanded * torch.log(rate) - rate - torch.lgamma(expanded + 1)
        log_gamma = (shape - 1) * torch.log(y) - y / scale - torch.lgamma(shape) - shape * torch.log(scale)
        terms = log_poisson + log_gamma
        running = torch.logaddexp(running, torch.logsumexp(terms, dim=0))
        peak = torch.maximum(peak, torch.max(terms, dim=0).values)
        if j.numel() >= 2:
            decline = terms[-2] - terms[-1]
            # The terms are log-concave in j.  After the mode, their remaining
            # geometric envelope is exp(last)/(1-exp(-decline)).
            valid_decline = decline > 0
            log_tail = terms[-1] - torch.log1p(-torch.exp(-torch.clamp_min(decline, 1e-30)))
            if bool((valid_decline & (log_tail <= peak - log_tolerance)).all()):
                return running
    raise RuntimeError("Tweedie density series exceeded the 200000-term resource guard")


def full_log_prob(y: torch.Tensor, mu: torch.Tensor, phi: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Full Tweedie log probability, including its exact atom at zero."""
    output_dtype = y.dtype
    work_dtype = torch.float64 if y.dtype in (torch.float16, torch.bfloat16, torch.float32) else y.dtype
    y, mu, phi, p = (value.to(work_dtype) for value in torch.broadcast_tensors(y, mu, phi, p))
    rate, concentration, scale = compound_parameters(mu, phi, p)
    result = torch.full_like(y, -torch.inf)
    zeros = y == 0
    positive = y > 0
    result = torch.where(zeros, -rate, result)
    if bool(positive.any()):
        result = torch.where(positive, _positive_log_density(y, rate, concentration, scale), result)
    return result.to(output_dtype)


def full_cdf(y: torch.Tensor, mu: torch.Tensor, phi: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Compound-Poisson--Gamma CDF, evaluated without SciPy."""
    output_dtype = y.dtype
    work_dtype = torch.float64 if y.dtype in (torch.float16, torch.bfloat16, torch.float32) else y.dtype
    y, mu, phi, p = (value.to(work_dtype) for value in torch.broadcast_tensors(y, mu, phi, p))
    safe_y = torch.where(y > 0, y, torch.ones_like(y))
    rate, concentration, scale = compound_parameters(mu, phi, p)
    result = torch.zeros_like(y)
    nonnegative = y >= 0
    result = torch.where(nonnegative, torch.exp(-rate), result)
    positive = y > 0
    if bool(positive.any()):
        mixture = torch.zeros_like(y)
        chunk_size, maximum_terms, log_tolerance = 128, 200_000, math.log(1e-14)
        for start in range(1, maximum_terms + 1, chunk_size):
            stop = min(start + chunk_size, maximum_terms + 1)
            j = torch.arange(start, stop, dtype=y.dtype, device=y.device)
            expanded = j.reshape((j.numel(),) + (1,) * y.ndim)
            shape = expanded * concentration
            log_weight = expanded * torch.log(rate) - rate - torch.lgamma(expanded + 1)
            mixture = mixture + torch.sum(torch.exp(log_weight) * torch.special.gammainc(shape, safe_y / scale), dim=0)
            next_ratio = rate / (j[-1] + 1)
            log_next = log_weight[-1] + torch.log(next_ratio)
            log_tail = log_next - torch.log1p(-torch.clamp_max(next_ratio, 1 - 1e-15))
            if bool(((next_ratio < 1) & (log_tail <= log_tolerance)).all()):
                break
        else:
            raise RuntimeError("Tweedie CDF series exceeded the 200000-term resource guard")
        result = torch.where(positive, torch.exp(-rate) + mixture, result)
    tolerance = 1e-10 if output_dtype == torch.float64 else 1e-5
    if bool(((result < -tolerance) | (result > 1 + tolerance)).any()):
        raise RuntimeError("Tweedie CDF escaped [0, 1] beyond numerical tolerance")
    return result.clamp(0, 1).to(output_dtype)
