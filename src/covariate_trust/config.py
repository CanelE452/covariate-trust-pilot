"""Typed configuration with strict validation.

Invalid values raise immediately; nothing is silently coerced or clipped.
Unknown keys are rejected so a typo can never be ignored.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for any invalid or unknown configuration entry."""


def _strict(cls, data: dict, where: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)} (known: {sorted(known)})")
    required = {f.name for f in fields(cls)
                if f.default is MISSING and f.default_factory is MISSING}
    missing = required - set(data)
    if missing:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}")
    return cls(**data)


# ---------------------------------------------------------------- study 0 ----

@dataclass(frozen=True)
class Study0Experiment:
    name: str
    seed: int
    n_repetitions: int

    def __post_init__(self):
        if self.n_repetitions < 1000:
            raise ConfigError("study0.experiment.n_repetitions must be >= 1000")


@dataclass(frozen=True)
class Study0DGP:
    prior_mean: float
    prior_variance: float
    beta: float
    target_noise_std: float

    def __post_init__(self):
        if self.prior_variance <= 0:
            raise ConfigError("study0.dgp.prior_variance must be > 0")
        if self.target_noise_std < 0:
            raise ConfigError("study0.dgp.target_noise_std must be >= 0")


@dataclass(frozen=True)
class Study0Grid:
    lambda_values: list[float]

    def __post_init__(self):
        if not self.lambda_values:
            raise ConfigError("study0.grid.lambda_values must not be empty")
        if any(l < 0 for l in self.lambda_values):
            raise ConfigError("study0.grid.lambda_values must all be >= 0")


@dataclass(frozen=True)
class Study0Config:
    experiment: Study0Experiment
    dgp: Study0DGP
    grid: Study0Grid

    @staticmethod
    def from_dict(d: dict) -> "Study0Config":
        unknown = set(d) - {"experiment", "dgp", "grid"}
        if unknown:
            raise ConfigError(f"study0: unknown top-level keys {sorted(unknown)}")
        return Study0Config(
            experiment=_strict(Study0Experiment, d["experiment"], "study0.experiment"),
            dgp=_strict(Study0DGP, d["dgp"], "study0.dgp"),
            grid=_strict(Study0Grid, d["grid"], "study0.grid"),
        )

    @staticmethod
    def load(path: str | Path) -> "Study0Config":
        with open(path, "r", encoding="utf-8") as fh:
            return Study0Config.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- pilot ----

@dataclass(frozen=True)
class PilotExperiment:
    name: str
    master_seed: int
    series_length: int
    standardization_end: int
    primary_origin: int
    context_length: int
    frequency: str
    quantile_levels: list[float]

    def __post_init__(self):
        if self.series_length < 32:
            raise ConfigError("experiment.series_length too small")
        if not 0 < self.standardization_end <= self.series_length:
            raise ConfigError("experiment.standardization_end out of range")
        if not 0 < self.primary_origin < self.series_length:
            raise ConfigError("experiment.primary_origin out of range")
        if self.standardization_end > self.primary_origin:
            raise ConfigError(
                "experiment.standardization_end must be <= primary_origin "
                "(standardization may not use forecast-window information)"
            )
        if not 0 < self.context_length <= self.primary_origin:
            raise ConfigError("experiment.context_length must be in (0, primary_origin]")
        q = self.quantile_levels
        if not q:
            raise ConfigError("experiment.quantile_levels must not be empty")
        if any(not 0.0 < v < 1.0 for v in q):
            raise ConfigError("experiment.quantile_levels must lie strictly inside (0, 1)")
        if any(b <= a for a, b in zip(q, q[1:])):
            raise ConfigError("experiment.quantile_levels must be strictly increasing")


@dataclass(frozen=True)
class PilotGrid:
    nominal_covariate_share: list[float]
    lambda_values: list[float]
    horizons: list[int]
    n_series_per_cell: int

    def __post_init__(self):
        if any(not 0.0 <= r < 1.0 for r in self.nominal_covariate_share):
            raise ConfigError("grid.nominal_covariate_share must lie in [0, 1)")
        if any(l < 0 for l in self.lambda_values):
            raise ConfigError("grid.lambda_values must all be >= 0")
        if any(h < 1 for h in self.horizons):
            raise ConfigError("grid.horizons must all be >= 1")
        if self.n_series_per_cell < 2:
            raise ConfigError("grid.n_series_per_cell must be >= 2")
        for name, values in (("nominal_covariate_share", self.nominal_covariate_share),
                             ("lambda_values", self.lambda_values),
                             ("horizons", self.horizons)):
            if len(set(values)) != len(values):
                raise ConfigError(f"grid.{name} contains duplicates")


@dataclass(frozen=True)
class PilotDGP:
    base_ar: float
    covariate_ar: float
    base_periods: list[int]
    covariate_periods: list[int]
    ar_innovation_std: float
    common_random_numbers: bool

    def __post_init__(self):
        for name, rho in (("base_ar", self.base_ar), ("covariate_ar", self.covariate_ar)):
            if not -1.0 < rho < 1.0:
                raise ConfigError(f"dgp.{name} must lie in (-1, 1) for stationarity")
        if self.ar_innovation_std <= 0:
            raise ConfigError("dgp.ar_innovation_std must be > 0")
        for name, periods in (("base_periods", self.base_periods),
                              ("covariate_periods", self.covariate_periods)):
            if len(periods) != 2:
                raise ConfigError(f"dgp.{name} must contain exactly 2 periods")
            if any(p < 2 for p in periods):
                raise ConfigError(f"dgp.{name} entries must be >= 2")
        if not self.common_random_numbers:
            raise ConfigError(
                "dgp.common_random_numbers=false is not supported in this pilot: "
                "the paired estimand assumes shared processes across grid cells"
            )


@dataclass(frozen=True)
class PilotModel:
    model_id: str
    package_version: str
    frozen: bool
    cross_learning: bool
    device: str
    allow_cpu_smoke: bool
    allow_cpu_diagnostic: bool
    attention_implementation: str

    def __post_init__(self):
        if not self.frozen:
            raise ConfigError("model.frozen=false is out of scope (no fine-tuning in this pilot)")
        if self.cross_learning:
            raise ConfigError("model.cross_learning=true is forbidden in this pilot")
        if self.device not in {"cuda", "cpu"}:
            raise ConfigError("model.device must be 'cuda' or 'cpu'")
        if self.allow_cpu_diagnostic:
            raise ConfigError("model.allow_cpu_diagnostic=true is forbidden in this pilot")


@dataclass(frozen=True)
class PilotBootstrap:
    n_resamples: int
    confidence_level: float

    def __post_init__(self):
        if self.n_resamples < 200:
            raise ConfigError("bootstrap.n_resamples must be >= 200")
        if not 0.5 < self.confidence_level < 1.0:
            raise ConfigError("bootstrap.confidence_level must lie in (0.5, 1)")


@dataclass(frozen=True)
class PilotGates:
    clean_gain_pass: float
    clean_gain_fail: float
    oracle_headroom_pass: float
    oracle_headroom_fail: float
    harm_relative_threshold: float
    high_noise_harm_rate: float
    # thresholds that make otherwise verbal gate wording executable
    dose_response_curve_fraction: float = 0.70
    negative_control_ratio: float = 0.50
    degenerate_policy_share: float = 0.95
    admission_pass_improvement: float = 0.02
    admission_fail_improvement: float = 0.0
    admission_recovery_pass: float = 0.30
    admission_recovery_fail: float = 0.10
    admission_harm_reduction_pass: float = 0.25
    admission_low_noise_regression_max: float = 0.01

    def __post_init__(self):
        if self.clean_gain_fail >= self.clean_gain_pass:
            raise ConfigError("gates.clean_gain_fail must be < gates.clean_gain_pass")
        if self.oracle_headroom_fail >= self.oracle_headroom_pass:
            raise ConfigError("gates.oracle_headroom_fail must be < gates.oracle_headroom_pass")
        if not 0 < self.harm_relative_threshold < 1:
            raise ConfigError("gates.harm_relative_threshold must lie in (0, 1)")
        if not 0 < self.high_noise_harm_rate <= 1:
            raise ConfigError("gates.high_noise_harm_rate must lie in (0, 1]")
        if not 0 < self.dose_response_curve_fraction <= 1:
            raise ConfigError("gates.dose_response_curve_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class PilotConfig:
    experiment: PilotExperiment
    grid: PilotGrid
    dgp: PilotDGP
    model: PilotModel
    bootstrap: PilotBootstrap
    gates: PilotGates
    source_path: str = field(default="", compare=False)

    def __post_init__(self):
        end = self.experiment.primary_origin + max(self.grid.horizons)
        if end > self.experiment.series_length:
            raise ConfigError(
                f"primary_origin + max(horizon) = {end} exceeds series_length "
                f"{self.experiment.series_length}"
            )

    @staticmethod
    def from_dict(d: dict, source_path: str = "") -> "PilotConfig":
        expected = {"experiment", "grid", "dgp", "model", "bootstrap", "gates"}
        unknown = set(d) - expected
        if unknown:
            raise ConfigError(f"pilot: unknown top-level keys {sorted(unknown)}")
        missing = expected - set(d)
        if missing:
            raise ConfigError(f"pilot: missing top-level keys {sorted(missing)}")
        return PilotConfig(
            experiment=_strict(PilotExperiment, d["experiment"], "experiment"),
            grid=_strict(PilotGrid, d["grid"], "grid"),
            dgp=_strict(PilotDGP, d["dgp"], "dgp"),
            model=_strict(PilotModel, d["model"], "model"),
            bootstrap=_strict(PilotBootstrap, d["bootstrap"], "bootstrap"),
            gates=_strict(PilotGates, d["gates"], "gates"),
            source_path=source_path,
        )

    @staticmethod
    def load(path: str | Path) -> "PilotConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return PilotConfig.from_dict(yaml.safe_load(fh), source_path=str(Path(path).resolve()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source_path", None)
        return d

    def resolved_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @property
    def base_series_ids(self) -> list[int]:
        return list(range(self.grid.n_series_per_cell))


def dump_yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
