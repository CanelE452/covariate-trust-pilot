"""Pure scientific-gate diagnostics."""

from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy.stats import spearmanr


def _effectively_constant(values: np.ndarray) -> bool:
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = 8.0 * np.finfo(np.float64).eps * scale
    return float(np.ptp(values)) <= tolerance


def spearman_diagnostic(
    predictor: np.ndarray, outcome: np.ndarray
) -> dict[str, object]:
    """Compute SciPy Spearman correlation after an explicit degeneracy guard."""

    predictor_values = np.asarray(predictor, dtype=np.float64)
    outcome_values = np.asarray(outcome, dtype=np.float64)
    if predictor_values.ndim != 1 or outcome_values.ndim != 1:
        raise ValueError("predictor and outcome must be one-dimensional")
    if predictor_values.shape != outcome_values.shape:
        raise ValueError("predictor and outcome must have identical lengths")
    if predictor_values.size < 3:
        raise ValueError("Spearman diagnostic requires at least three observations")
    if not bool(
        np.isfinite(predictor_values).all()
        and np.isfinite(outcome_values).all()
    ):
        raise ValueError("predictor and outcome must all be finite")

    if _effectively_constant(predictor_values) or _effectively_constant(
        outcome_values
    ):
        return {
            "status": "DEGENERATE",
            "rho": None,
            "pvalue": None,
            "n": int(predictor_values.size),
        }

    result = spearmanr(predictor_values, outcome_values)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    if not np.isfinite(rho):
        return {
            "status": "DEGENERATE",
            "rho": None,
            "pvalue": None,
            "n": int(predictor_values.size),
        }
    return {
        "status": "OK",
        "rho": rho,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
        "n": int(predictor_values.size),
    }


def _finite_loss(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def family_matched_oracle_capture(
    policy_losses: Mapping[str, float],
    policy_families: Mapping[str, str],
    oracle_ladders: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Score each policy against the denominator for its own policy family."""

    if not isinstance(policy_losses, Mapping):
        raise TypeError("policy_losses must be a mapping")
    if not isinstance(policy_families, Mapping):
        raise TypeError("policy_families must be a mapping")
    if not isinstance(oracle_ladders, Mapping):
        raise TypeError("oracle_ladders must be a mapping")
    if set(policy_losses) != set(policy_families):
        raise ValueError("policy_losses and policy_families must have identical keys")

    captures: dict[str, float] = {}
    for policy_name, policy_loss_value in policy_losses.items():
        family = policy_families[policy_name]
        if family not in oracle_ladders:
            raise ValueError(
                f"missing oracle ladder for policy family {family!r}"
            )
        ladder = oracle_ladders[family]
        if not isinstance(ladder, Mapping):
            raise TypeError(f"oracle ladder {family!r} must be a mapping")
        missing = {
            "baseline_loss",
            "oracle_loss",
        }.difference(ladder)
        if missing:
            raise ValueError(
                f"oracle ladder {family!r} is missing fields: {sorted(missing)}"
            )

        policy_loss = _finite_loss(
            f"policy loss for {policy_name!r}", policy_loss_value
        )
        baseline_loss = _finite_loss(
            f"baseline loss for {family!r}", ladder["baseline_loss"]
        )
        oracle_loss = _finite_loss(
            f"oracle loss for {family!r}", ladder["oracle_loss"]
        )
        denominator = baseline_loss - oracle_loss
        if denominator <= 0.0:
            raise ValueError(
                f"oracle ladder {family!r} must have positive loss headroom"
            )
        captures[policy_name] = (baseline_loss - policy_loss) / denominator
    return captures
