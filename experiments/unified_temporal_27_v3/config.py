"""Configuration shared by the self-contained v3 generators and trainers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"
VALIDATION_PATH = ROOT / "validation_27.csv"


@dataclass(frozen=True)
class ExperimentConfig:
    """One explicit source of truth for v3.

    Gamma variables use ``shape=1/cv2`` and ``scale=cv2*mean``.  Therefore
    their variance is ``cv2*mean**2`` and ``cv2`` is the positive-demand
    squared coefficient of variation.
    """

    # Common data and chronological split.
    n_series: int = 300
    length: int = 512
    lookback: int = 96
    horizon: int = 24
    period: int = 24
    train_end: int = 358
    val_end: int = 409
    data_seed: int = 42
    train_seed: int = 0
    epsilon: float = 1e-8

    # Group A: no seasonal parameter is used by generator_a.
    a_occurrence_rate: float = 0.25
    a_positive_mean: float = 10.0
    a_positive_cv2: float = 0.49
    a_weak_rho: float = 0.30
    a_strong_rho: float = 0.70
    a_magnitude_high_quantile: float = 0.50

    # Group B: seasonality only, no hurdle/spike process.
    b_baseline: float = 20.0
    b_noise_std: float = 1.5
    b_weak_amplitude: float = 2.0
    b_medium_amplitude: float = 5.0
    b_strong_amplitude: float = 8.0
    b_jitter_weak: float = 0.15
    b_jitter_strong: float = 0.35

    # Group C defaults.
    c_occurrence_rate: float = 0.25
    c_positive_mean: float = 10.0
    c_positive_cv2: float = 0.49
    c_weak_amplitude: float = 0.25
    c_strong_amplitude: float = 0.70
    c_regime_switch: int = 256

    # Shared training defaults; the trainer may expose these as CLI options.
    moving_average_kernel: int = 25
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 30
    patience: int = 4
    lambda_mag: float = 1.0
    lambda_gap: float = 1.0
    # Small validation-only candidate sets.  Gamma+Gap reuses the selected
    # magnitude weight, so its only additional choice is lambda_gap.
    lambda_mag_candidates: tuple[float, ...] = (0.5, 1.0)
    lambda_gap_candidates: tuple[float, ...] = (0.25, 0.5)

    def __post_init__(self) -> None:
        if self.n_series <= 0 or self.length <= 0:
            raise ValueError("n_series and length must be positive")
        if not (0 < self.train_end < self.val_end < self.length):
            raise ValueError("expected 0 < train_end < val_end < length")
        if self.lookback <= 0 or self.lookback > self.train_end:
            raise ValueError("lookback must be positive and fit in train")
        if self.horizon <= 0 or self.period <= 0:
            raise ValueError("horizon and period must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not (0.0 < self.a_occurrence_rate < 1.0):
            raise ValueError("a_occurrence_rate must be in (0,1)")
        if self.a_positive_mean <= 0.0 or self.a_positive_cv2 <= 0.0:
            raise ValueError("Group-A Gamma mean and CV^2 must be positive")
        if not (0.0 <= self.a_weak_rho < self.a_strong_rho < 1.0):
            raise ValueError("expected 0 <= weak rho < strong rho < 1")
        if not (0.0 < self.a_magnitude_high_quantile < 1.0):
            raise ValueError("magnitude split quantile must be in (0,1)")
        if self.b_baseline <= self.b_strong_amplitude + 4.0 * self.b_noise_std:
            raise ValueError("Group-B baseline is too small for positive observations")
        if not (0 < self.c_regime_switch < self.length):
            raise ValueError("c_regime_switch must be inside the series")
        if self.moving_average_kernel <= 0 or self.moving_average_kernel % 2 == 0:
            raise ValueError("moving_average_kernel must be a positive odd integer")
        if not self.lambda_mag_candidates or not self.lambda_gap_candidates:
            raise ValueError("lambda candidate sets cannot be empty")
        if any(value <= 0.0 for value in self.lambda_mag_candidates):
            raise ValueError("all magnitude lambda candidates must be positive")
        if any(value <= 0.0 for value in self.lambda_gap_candidates):
            raise ValueError("all gap lambda candidates must be positive")

    @property
    def split_bounds(self) -> tuple[tuple[int, int], ...]:
        """Half-open train, validation and test intervals."""

        return (
            (0, self.train_end),
            (self.train_end, self.val_end),
            (self.val_end, self.length),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["split_bounds"] = [list(bounds) for bounds in self.split_bounds]
        return result


DEFAULT_CONFIG = ExperimentConfig()


__all__ = [
    "ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "LOGS_DIR",
    "VALIDATION_PATH",
    "ExperimentConfig",
    "DEFAULT_CONFIG",
]
