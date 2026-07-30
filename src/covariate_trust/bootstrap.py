"""Monte-Carlo uncertainty for paired comparisons.

Every comparison in this study is paired: the same ``base_series_id`` (and the
same processes, phases and eta path) is used under both conditions.  The
bootstrap unit is therefore ``base_series_id``, resampled as a cluster so that
all of a series' tasks move together.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .seeds import make_rng

BOOTSTRAP_UNIT = "base_series_id"


@dataclass(frozen=True)
class PairedResult:
    n_units: int
    n_observations: int
    mean_baseline: float
    mean_treatment: float
    mean_diff: float            # baseline - treatment  (positive = treatment better)
    median_diff: float
    sd_diff: float
    monte_carlo_se: float
    ci_low: float
    ci_high: float
    relative_improvement: float  # (baseline - treatment) / baseline
    rel_ci_low: float
    rel_ci_high: float
    win_rate: float              # fraction of observations where treatment is better
    confidence_level: float
    n_resamples: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def ci_excludes_zero(self) -> bool:
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    @property
    def ci_favours_treatment(self) -> bool:
        return self.ci_low > 0.0

    @property
    def ci_favours_baseline(self) -> bool:
        return self.ci_high < 0.0


def paired_bootstrap(units: np.ndarray, baseline: np.ndarray, treatment: np.ndarray,
                     n_resamples: int, confidence_level: float,
                     seed_parts: tuple = ("bootstrap",)) -> PairedResult:
    """Cluster bootstrap over ``units`` for the paired difference baseline - treatment."""
    units = np.asarray(units)
    baseline = np.asarray(baseline, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if not (len(units) == len(baseline) == len(treatment)):
        raise ValueError("units, baseline and treatment must have equal length")
    if len(units) == 0:
        raise ValueError("empty comparison")

    diff = baseline - treatment
    uniq, inverse = np.unique(units, return_inverse=True)
    n_units = len(uniq)

    counts = np.bincount(inverse, minlength=n_units).astype(float)
    sum_diff = np.bincount(inverse, weights=diff, minlength=n_units)
    sum_base = np.bincount(inverse, weights=baseline, minlength=n_units)
    sum_treat = np.bincount(inverse, weights=treatment, minlength=n_units)

    rng = make_rng(*seed_parts, n_resamples, n_units)
    idx = rng.integers(0, n_units, size=(n_resamples, n_units))
    boot_counts = counts[idx].sum(axis=1)
    boot_diff = sum_diff[idx].sum(axis=1) / boot_counts
    boot_base = sum_base[idx].sum(axis=1) / boot_counts
    boot_treat = sum_treat[idx].sum(axis=1) / boot_counts
    boot_rel = (boot_base - boot_treat) / boot_base

    alpha = 1.0 - confidence_level
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    unit_means = sum_diff / counts
    return PairedResult(
        n_units=n_units,
        n_observations=len(diff),
        mean_baseline=float(baseline.mean()),
        mean_treatment=float(treatment.mean()),
        mean_diff=float(diff.mean()),
        median_diff=float(np.median(diff)),
        sd_diff=float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        monte_carlo_se=float(unit_means.std(ddof=1) / np.sqrt(n_units)) if n_units > 1 else float("nan"),
        ci_low=float(np.percentile(boot_diff, lo)),
        ci_high=float(np.percentile(boot_diff, hi)),
        relative_improvement=float((baseline.mean() - treatment.mean()) / baseline.mean()),
        rel_ci_low=float(np.percentile(boot_rel, lo)),
        rel_ci_high=float(np.percentile(boot_rel, hi)),
        win_rate=float((treatment < baseline).mean()),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )
