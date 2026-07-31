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


# ============================================================================
# Follow-up studies (Study 1B and Study 2).
#
# These are additive: PilotConfig and Study0Config above are untouched, so the
# coarse pilot stays reproducible.  Both follow-up configs inherit the `dgp` and
# `gates` blocks from configs/pilot.yaml rather than restating them, which is what
# guarantees the generating equations are literally the ones the coarse pilot used.
# ============================================================================

INHERITED_BLOCKS = ("dgp", "gates")
INHERIT_FROM = "pilot.yaml"


def _inherited_blocks(config_path: str | Path) -> dict:
    src = Path(config_path).parent / INHERIT_FROM
    if not src.exists():
        raise ConfigError(
            f"follow-up configs inherit {list(INHERITED_BLOCKS)} from {src}, which is missing")
    with open(src, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    missing = [b for b in INHERITED_BLOCKS if b not in base]
    if missing:
        raise ConfigError(f"{src} has no {missing} block(s) to inherit")
    return {b: base[b] for b in INHERITED_BLOCKS}


@dataclass(frozen=True)
class FollowupModel:
    model_id: str
    frozen: bool
    cross_learning: bool
    device: str
    attention_implementation: str

    def __post_init__(self):
        if not self.frozen:
            raise ConfigError("model.frozen=false is out of scope (no fine-tuning)")
        if self.cross_learning:
            raise ConfigError("model.cross_learning=true is forbidden")
        if self.device not in {"cuda", "cpu"}:
            raise ConfigError("model.device must be 'cuda' or 'cpu'")

    def to_pilot_model(self) -> PilotModel:
        return PilotModel(
            model_id=self.model_id, package_version="2.3.1", frozen=True,
            cross_learning=False, device=self.device, allow_cpu_smoke=True,
            allow_cpu_diagnostic=False,
            attention_implementation=self.attention_implementation)


@dataclass(frozen=True)
class GateEConfig:
    min_finite_crossings: int
    min_narrow_crossings: int
    max_ci_width: float
    fail_max_finite_crossings: int
    fail_min_non_decreasing_curves: int

    def __post_init__(self):
        if self.max_ci_width <= 0:
            raise ConfigError("gate_e.max_ci_width must be > 0")
        if self.fail_max_finite_crossings >= self.min_finite_crossings:
            raise ConfigError("gate_e.fail_max_finite_crossings must be < min_finite_crossings")


@dataclass(frozen=True)
class BaselineConfig:
    ridge_parameter: float
    arx_seasonal_periods: list[int]

    def __post_init__(self):
        if not 0 < self.ridge_parameter < 1:
            raise ConfigError("baselines.ridge_parameter must lie in (0, 1)")
        if not self.arx_seasonal_periods:
            raise ConfigError("baselines.arx_seasonal_periods must not be empty")


@dataclass(frozen=True)
class BoundaryConfig:
    experiment: PilotExperiment
    grid: PilotGrid
    model: FollowupModel
    bootstrap: PilotBootstrap
    gate_e: GateEConfig
    baselines: BaselineConfig
    inherited: dict = field(default_factory=dict, compare=False)
    source_path: str = field(default="", compare=False)

    def __post_init__(self):
        end = self.experiment.primary_origin + max(self.grid.horizons)
        if end > self.experiment.series_length:
            raise ConfigError(
                f"primary_origin + max(horizon) = {end} exceeds series_length")
        if len(self.grid.nominal_covariate_share) * len(self.grid.horizons) < 2:
            raise ConfigError("grid must contain at least two (share, horizon) curves")

    @staticmethod
    def from_dict(d: dict, inherited: dict, source_path: str = "") -> "BoundaryConfig":
        expected = {"experiment", "grid", "model", "bootstrap", "gate_e", "baselines"}
        unknown, missing = set(d) - expected, expected - set(d)
        if unknown:
            raise ConfigError(f"boundary: unknown top-level keys {sorted(unknown)}")
        if missing:
            raise ConfigError(f"boundary: missing top-level keys {sorted(missing)}")
        return BoundaryConfig(
            experiment=_strict(PilotExperiment, d["experiment"], "experiment"),
            grid=_strict(PilotGrid, d["grid"], "grid"),
            model=_strict(FollowupModel, d["model"], "model"),
            bootstrap=_strict(PilotBootstrap, d["bootstrap"], "bootstrap"),
            gate_e=_strict(GateEConfig, d["gate_e"], "gate_e"),
            baselines=_strict(BaselineConfig, d["baselines"], "baselines"),
            inherited=inherited, source_path=source_path)

    @staticmethod
    def load(path: str | Path) -> "BoundaryConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return BoundaryConfig.from_dict(data, _inherited_blocks(path), str(Path(path).resolve()))

    def to_pilot_config(self) -> PilotConfig:
        """The PilotConfig the existing DGP / schema code path consumes."""
        return PilotConfig.from_dict({
            "experiment": asdict(self.experiment),
            "grid": asdict(self.grid),
            "dgp": self.inherited["dgp"],
            "model": asdict(self.model.to_pilot_model()),
            "bootstrap": asdict(self.bootstrap),
            "gates": self.inherited["gates"],
        })

    def to_dict(self) -> dict:
        d = {"experiment": asdict(self.experiment), "grid": asdict(self.grid),
             "model": asdict(self.model), "bootstrap": asdict(self.bootstrap),
             "gate_e": asdict(self.gate_e), "baselines": asdict(self.baselines),
             "inherited_from_pilot_yaml": self.inherited}
        return d

    def resolved_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @property
    def base_series_ids(self) -> list[int]:
        return list(range(self.grid.n_series_per_cell))


# ------------------------------------------------------------------ Study 2 ---

@dataclass(frozen=True)
class DynamicGrid:
    nominal_covariate_share: list[float]
    horizons: list[int]
    n_series_per_condition: int

    def __post_init__(self):
        if any(not 0.0 <= r < 1.0 for r in self.nominal_covariate_share):
            raise ConfigError("grid.nominal_covariate_share must lie in [0, 1)")
        if any(h < 1 for h in self.horizons):
            raise ConfigError("grid.horizons must all be >= 1")
        if self.n_series_per_condition < 2:
            raise ConfigError("grid.n_series_per_condition must be >= 2")


@dataclass(frozen=True)
class ReliabilitySchedule:
    name: str
    historical: list[float]
    current: float

    def __post_init__(self):
        if not self.name:
            raise ConfigError("schedule.name must not be empty")
        if any(l < 0 for l in self.historical) or self.current < 0:
            raise ConfigError(f"schedule {self.name}: lambdas must be >= 0")
        if not self.historical:
            raise ConfigError(f"schedule {self.name}: historical must not be empty")


@dataclass(frozen=True)
class ProxyConfig:
    sigma_proxy: float
    overconfident_multiplier: float
    underconfident_multiplier: float

    def __post_init__(self):
        if self.sigma_proxy <= 0:
            raise ConfigError("proxy.sigma_proxy must be > 0")
        if not 0 < self.overconfident_multiplier < 1:
            raise ConfigError("proxy.overconfident_multiplier must lie in (0, 1)")
        if self.underconfident_multiplier <= 1:
            raise ConfigError("proxy.underconfident_multiplier must be > 1")


@dataclass(frozen=True)
class SelectorThresholds:
    use_threshold: float
    override_low: float
    override_high: float

    def __post_init__(self):
        if not self.override_low < self.use_threshold < self.override_high:
            raise ConfigError(
                "selector_thresholds must satisfy override_low < use_threshold < override_high")


@dataclass(frozen=True)
class GateFConfig:
    pass_improvement: float
    pass_recovery: float
    pass_harm_reduction: float
    stable_regression_max: float
    improvement_condition_regression_max: float
    fail_recovery: float

    def __post_init__(self):
        if self.fail_recovery >= self.pass_recovery:
            raise ConfigError("gate_f.fail_recovery must be < gate_f.pass_recovery")


@dataclass(frozen=True)
class DynamicConfig:
    experiment: PilotExperiment
    grid: DynamicGrid
    model: FollowupModel
    bootstrap: PilotBootstrap
    schedules: list[ReliabilitySchedule]
    proxy: ProxyConfig
    selector_thresholds: SelectorThresholds
    gate_f: GateFConfig
    inherited: dict = field(default_factory=dict, compare=False)
    source_path: str = field(default="", compare=False)

    def __post_init__(self):
        end = self.experiment.primary_origin + max(self.grid.horizons)
        if end > self.experiment.series_length:
            raise ConfigError(f"primary_origin + max(horizon) = {end} exceeds series_length")
        names = [s.name for s in self.schedules]
        if len(set(names)) != len(names):
            raise ConfigError("schedule names must be unique")
        lengths = {len(s.historical) for s in self.schedules}
        if len(lengths) != 1:
            raise ConfigError(f"all schedules must have the same number of historical "
                              f"lambdas, got {sorted(lengths)}")

    @property
    def n_historical_origins(self) -> int:
        return len(self.schedules[0].historical)

    @property
    def all_lambdas(self) -> list[float]:
        vals = {float(s.current) for s in self.schedules}
        vals |= {float(l) for s in self.schedules for l in s.historical}
        return sorted(vals)

    @staticmethod
    def from_dict(d: dict, inherited: dict, source_path: str = "") -> "DynamicConfig":
        expected = {"experiment", "grid", "model", "bootstrap", "schedules", "proxy",
                    "selector_thresholds", "gate_f"}
        unknown, missing = set(d) - expected, expected - set(d)
        if unknown:
            raise ConfigError(f"dynamic: unknown top-level keys {sorted(unknown)}")
        if missing:
            raise ConfigError(f"dynamic: missing top-level keys {sorted(missing)}")
        if not isinstance(d["schedules"], list) or not d["schedules"]:
            raise ConfigError("dynamic.schedules must be a non-empty list")
        return DynamicConfig(
            experiment=_strict(PilotExperiment, d["experiment"], "experiment"),
            grid=_strict(DynamicGrid, d["grid"], "grid"),
            model=_strict(FollowupModel, d["model"], "model"),
            bootstrap=_strict(PilotBootstrap, d["bootstrap"], "bootstrap"),
            schedules=[_strict(ReliabilitySchedule, s, f"schedules[{i}]")
                       for i, s in enumerate(d["schedules"])],
            proxy=_strict(ProxyConfig, d["proxy"], "proxy"),
            selector_thresholds=_strict(SelectorThresholds, d["selector_thresholds"],
                                        "selector_thresholds"),
            gate_f=_strict(GateFConfig, d["gate_f"], "gate_f"),
            inherited=inherited, source_path=source_path)

    @staticmethod
    def load(path: str | Path) -> "DynamicConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return DynamicConfig.from_dict(data, _inherited_blocks(path), str(Path(path).resolve()))

    def to_pilot_config(self) -> PilotConfig:
        return PilotConfig.from_dict({
            "experiment": asdict(self.experiment),
            "grid": {"nominal_covariate_share": list(self.grid.nominal_covariate_share),
                     "lambda_values": self.all_lambdas,
                     "horizons": list(self.grid.horizons),
                     "n_series_per_cell": self.grid.n_series_per_condition},
            "dgp": self.inherited["dgp"],
            "model": asdict(self.model.to_pilot_model()),
            "bootstrap": asdict(self.bootstrap),
            "gates": self.inherited["gates"],
        })

    def to_dict(self) -> dict:
        return {"experiment": asdict(self.experiment), "grid": asdict(self.grid),
                "model": asdict(self.model), "bootstrap": asdict(self.bootstrap),
                "schedules": [asdict(s) for s in self.schedules],
                "proxy": asdict(self.proxy),
                "selector_thresholds": asdict(self.selector_thresholds),
                "gate_f": asdict(self.gate_f),
                "inherited_from_pilot_yaml": self.inherited}

    def resolved_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @property
    def base_series_ids(self) -> list[int]:
        return list(range(self.grid.n_series_per_condition))


# ============================================================================
# Study 2B - held-out confirmation of the pre-registered D7 policy.
#
# Additive again: PilotConfig, Study0Config, BoundaryConfig and DynamicConfig are
# untouched.  This config inherits the six reliability schedules from
# configs/study2_dynamic_reliability.yaml and the `gates` block from
# configs/pilot.yaml, then builds a real DynamicConfig internally, so the schedule
# definitions and the selector implementations are literally the ones Study 2 used.
# ============================================================================

CONFIRMATION_INHERIT_FROM = "study2_dynamic_reliability.yaml"
REQUIRED_PRIMARY_SELECTOR = "D7_hybrid_override"
REQUIRED_PRIMARY_PROXY = "P1_calibrated_noisy"


def _confirmation_inherited(config_path: str | Path) -> dict:
    """Schedules and proxy multipliers from Study 2, gates from the pilot."""
    src = Path(config_path).parent / CONFIRMATION_INHERIT_FROM
    if not src.exists():
        raise ConfigError(f"Study 2B inherits its schedules from {src}, which is missing")
    with open(src, "r", encoding="utf-8") as fh:
        study2 = yaml.safe_load(fh)
    for block in ("schedules", "proxy", "gate_f"):
        if block not in study2:
            raise ConfigError(f"{src} has no {block} block to inherit")
    pilot = _inherited_blocks(config_path)
    return {
        "schedules": study2["schedules"],
        "proxy_multipliers": {
            "overconfident_multiplier": study2["proxy"]["overconfident_multiplier"],
            "underconfident_multiplier": study2["proxy"]["underconfident_multiplier"],
        },
        "gate_f": study2["gate_f"],
        "gates": pilot["gates"],
        "source": str(src),
    }


@dataclass(frozen=True)
class ConfirmationModel:
    model_id: str
    frozen: bool
    cross_learning: bool
    device: str
    allow_cpu_diagnostic: bool
    attention_implementation: str

    def __post_init__(self):
        if not self.frozen:
            raise ConfigError("model.frozen=false is out of scope (no fine-tuning)")
        if self.cross_learning:
            raise ConfigError("model.cross_learning=true is forbidden")
        if self.device not in {"cuda", "cpu"}:
            raise ConfigError("model.device must be 'cuda' or 'cpu'")
        if self.allow_cpu_diagnostic:
            raise ConfigError("model.allow_cpu_diagnostic=true is forbidden")

    def to_followup_model(self) -> FollowupModel:
        return FollowupModel(model_id=self.model_id, frozen=True, cross_learning=False,
                             device=self.device,
                             attention_implementation=self.attention_implementation)


@dataclass(frozen=True)
class ConfirmationProxy:
    primary_mode: str
    sigma_proxy: float
    secondary_modes: list[str]

    def __post_init__(self):
        if self.primary_mode != REQUIRED_PRIMARY_PROXY:
            raise ConfigError(
                f"proxy.primary_mode is pre-registered as {REQUIRED_PRIMARY_PROXY!r} and may "
                f"not be changed (got {self.primary_mode!r})")
        if self.sigma_proxy <= 0:
            raise ConfigError("proxy.sigma_proxy must be > 0")
        if self.primary_mode in self.secondary_modes:
            raise ConfigError("the primary proxy must not also be listed as secondary")
        if len(set(self.secondary_modes)) != len(self.secondary_modes):
            raise ConfigError("proxy.secondary_modes contains duplicates")


@dataclass(frozen=True)
class ConfirmationSelectors:
    primary: str
    secondary: list[str]
    d7_lower_threshold: float
    d7_upper_threshold: float
    d5_threshold: float

    def __post_init__(self):
        if self.primary != REQUIRED_PRIMARY_SELECTOR:
            raise ConfigError(
                f"selectors.primary is pre-registered as {REQUIRED_PRIMARY_SELECTOR!r} and may "
                f"not be changed (got {self.primary!r}); a better-scoring secondary policy is "
                f"still not allowed to take its place")
        if self.primary in self.secondary:
            raise ConfigError("the primary policy must not also be listed as secondary")
        if not self.d7_lower_threshold < self.d5_threshold < self.d7_upper_threshold:
            raise ConfigError("thresholds must satisfy d7_lower < d5 < d7_upper")


@dataclass(frozen=True)
class GateGConfig:
    overall_improvement_pass: float
    overall_improvement_fail: float
    oracle_recovery_pass: float
    oracle_recovery_fail: float
    harm_reduction_pass: float
    harm_reduction_fail: float
    stable_condition_regression_max: float
    stable_condition_fail: float

    def __post_init__(self):
        pairs = (("overall_improvement", self.overall_improvement_fail,
                  self.overall_improvement_pass),
                 ("oracle_recovery", self.oracle_recovery_fail, self.oracle_recovery_pass),
                 ("harm_reduction", self.harm_reduction_fail, self.harm_reduction_pass),
                 ("stable_condition", self.stable_condition_regression_max,
                  self.stable_condition_fail))
        for name, weaker, stronger in pairs:
            if weaker >= stronger:
                raise ConfigError(f"gate_g.{name}: the FAIL bound must be looser than the PASS bound")


@dataclass(frozen=True)
class ConfirmationConfig:
    experiment: PilotExperiment
    grid: DynamicGrid
    dgp: PilotDGP
    model: ConfirmationModel
    proxy: ConfirmationProxy
    selectors: ConfirmationSelectors
    bootstrap: PilotBootstrap
    gate_g: GateGConfig
    inherited: dict = field(default_factory=dict, compare=False)
    source_path: str = field(default="", compare=False)

    def __post_init__(self):
        end = self.experiment.primary_origin + max(self.grid.horizons)
        if end > self.experiment.series_length:
            raise ConfigError(f"primary_origin + max(horizon) = {end} exceeds series_length")

    @staticmethod
    def from_dict(d: dict, inherited: dict, source_path: str = "") -> "ConfirmationConfig":
        expected = {"experiment", "grid", "dgp", "model", "proxy", "selectors", "bootstrap",
                    "gate_g"}
        unknown, missing = set(d) - expected, expected - set(d)
        if unknown:
            raise ConfigError(f"confirmation: unknown top-level keys {sorted(unknown)}")
        if missing:
            raise ConfigError(f"confirmation: missing top-level keys {sorted(missing)}")
        return ConfirmationConfig(
            experiment=_strict(PilotExperiment, d["experiment"], "experiment"),
            grid=_strict(DynamicGrid, d["grid"], "grid"),
            dgp=_strict(PilotDGP, d["dgp"], "dgp"),
            model=_strict(ConfirmationModel, d["model"], "model"),
            proxy=_strict(ConfirmationProxy, d["proxy"], "proxy"),
            selectors=_strict(ConfirmationSelectors, d["selectors"], "selectors"),
            bootstrap=_strict(PilotBootstrap, d["bootstrap"], "bootstrap"),
            gate_g=_strict(GateGConfig, d["gate_g"], "gate_g"),
            inherited=inherited, source_path=source_path)

    @staticmethod
    def load(path: str | Path) -> "ConfirmationConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return ConfirmationConfig.from_dict(data, _confirmation_inherited(path),
                                            str(Path(path).resolve()))

    def to_dynamic_config(self) -> DynamicConfig:
        """The DynamicConfig that Study 2's runner and selector code consume.

        Built from the inherited schedules and this file's thresholds, so the selector
        implementations exercised here are the same functions Study 2 used.
        """
        return DynamicConfig.from_dict({
            "experiment": asdict(self.experiment),
            "grid": asdict(self.grid),
            "model": asdict(self.model.to_followup_model()),
            "bootstrap": asdict(self.bootstrap),
            "schedules": self.inherited["schedules"],
            "proxy": {"sigma_proxy": self.proxy.sigma_proxy,
                      **self.inherited["proxy_multipliers"]},
            "selector_thresholds": {"use_threshold": self.selectors.d5_threshold,
                                    "override_low": self.selectors.d7_lower_threshold,
                                    "override_high": self.selectors.d7_upper_threshold},
            "gate_f": self.inherited["gate_f"],
        }, {"dgp": asdict(self.dgp), "gates": self.inherited["gates"]})

    def to_dict(self) -> dict:
        return {"experiment": asdict(self.experiment), "grid": asdict(self.grid),
                "dgp": asdict(self.dgp), "model": asdict(self.model),
                "proxy": asdict(self.proxy), "selectors": asdict(self.selectors),
                "bootstrap": asdict(self.bootstrap), "gate_g": asdict(self.gate_g),
                "inherited": self.inherited}

    def resolved_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @property
    def base_series_ids(self) -> list[int]:
        return list(range(self.grid.n_series_per_condition))


# ============================================================================
# Study 3 - real forecast-vintage external validation (NYISO + ECMWF).
# Additive; every earlier config class is untouched.
# ============================================================================

@dataclass(frozen=True)
class ExternalExperiment:
    name: str
    master_seed: int
    timezone_internal: str
    decision_origin_hour_utc: int
    context_length: int
    prediction_length: int
    quantile_levels: list[float]

    def __post_init__(self):
        if self.timezone_internal != "UTC":
            raise ConfigError("experiment.timezone_internal must be UTC")
        if not 0 <= self.decision_origin_hour_utc <= 23:
            raise ConfigError("experiment.decision_origin_hour_utc must be an hour of day")
        if self.context_length < 24 or self.prediction_length < 1:
            raise ConfigError("experiment context/prediction length out of range")
        q = self.quantile_levels
        if any(not 0.0 < v < 1.0 for v in q) or any(b <= a for a, b in zip(q, q[1:])):
            raise ConfigError("experiment.quantile_levels must be increasing inside (0, 1)")


@dataclass(frozen=True)
class ExternalPeriods:
    requested_start: str
    requested_end: str
    proxy_train_end: str
    proxy_validation_end: str
    heldout_test_start: str
    heldout_test_end: str

    def __post_init__(self):
        order = [self.requested_start, self.proxy_train_end, self.proxy_validation_end,
                 self.heldout_test_start, self.heldout_test_end]
        stamps = [__import__("pandas").Timestamp(x) for x in order]
        if any(b < a for a, b in zip(stamps, stamps[1:])):
            raise ConfigError("periods must be chronologically ordered")
        if __import__("pandas").Timestamp(self.heldout_test_end) > __import__("pandas").Timestamp(
                self.requested_end):
            raise ConfigError("heldout_test_end must lie inside the requested window")


@dataclass(frozen=True)
class ExternalZone:
    source_name: str
    canonical_name: str
    latitude: float
    longitude: float

    def __post_init__(self):
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ConfigError(f"zone {self.canonical_name}: coordinates out of range")


@dataclass(frozen=True)
class NyisoConfig:
    primary_index_url: str
    fallback_index_url: str
    target_frequency: str
    aggregation: str
    minimum_zone_count: int
    zones: list[ExternalZone]

    def __post_init__(self):
        if len(self.zones) < self.minimum_zone_count:
            raise ConfigError("fewer configured zones than minimum_zone_count")
        names = [z.canonical_name for z in self.zones]
        if len(set(names)) != len(names):
            raise ConfigError("duplicate canonical zone names")


@dataclass(frozen=True)
class WeatherConfig:
    forecast_endpoint: str
    verification_endpoint: str
    model: str
    variable: str
    primary_run_hour_utc: int
    revision_run_hour_utc: int
    decision_delay_hours: int
    request_timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float
    minimum_forecast_coverage: float
    # Secondary diagnostic only: the first 00 UTC run on IFS Cycle 50r1.  It never
    # enters a Gate H or Gate I criterion or any weighting.
    model_cycle_50r1_first_00z_run: str = "2026-05-13"

    def __post_init__(self):
        if not 0 <= self.primary_run_hour_utc <= 23 or not 0 <= self.revision_run_hour_utc <= 23:
            raise ConfigError("weather run hours must be hours of day")
        if self.decision_delay_hours < 1:
            raise ConfigError("weather.decision_delay_hours must be >= 1: a run is not usable "
                              "at its own initialisation time")
        if not 0 < self.minimum_forecast_coverage <= 1:
            raise ConfigError("weather.minimum_forecast_coverage must lie in (0, 1]")


@dataclass(frozen=True)
class ExternalModel:
    model_id: str
    frozen: bool
    cross_learning: bool
    device: str
    attention_implementation: str
    allow_cpu_full_run: bool

    def __post_init__(self):
        if not self.frozen:
            raise ConfigError("model.frozen=false is out of scope")
        if self.cross_learning:
            raise ConfigError("model.cross_learning=true is forbidden")
        if self.device not in {"cuda", "cpu"}:
            raise ConfigError("model.device must be 'cuda' or 'cpu'")


@dataclass(frozen=True)
class ExternalProxy:
    calibration_method: str
    revision_weight: float
    recent_error_weight: float
    recent_window_origins: int
    lower_threshold: float
    upper_threshold: float
    d5_threshold: float

    def __post_init__(self):
        if self.calibration_method != "isotonic":
            raise ConfigError("proxy.calibration_method must be 'isotonic' in this study")
        if abs(self.revision_weight + self.recent_error_weight - 1.0) > 1e-9:
            raise ConfigError("proxy weights must sum to 1")
        if not self.lower_threshold < self.d5_threshold < self.upper_threshold:
            raise ConfigError("proxy thresholds must satisfy lower < d5 < upper")
        if (self.lower_threshold, self.upper_threshold) != (0.75, 1.25):
            raise ConfigError(
                "the D7 override band is fixed at 0.75 / 1.25 by the synthetic studies and "
                "may not be retuned on real data")
        if self.recent_window_origins < 2:
            raise ConfigError("proxy.recent_window_origins must be >= 2")


@dataclass(frozen=True)
class HistoricalUtilityConfig:
    window_origins: int
    minimum_origins: int

    def __post_init__(self):
        if self.minimum_origins > self.window_origins:
            raise ConfigError("historical_utility.minimum_origins exceeds window_origins")


@dataclass(frozen=True)
class ExternalBootstrap:
    n_resamples: int
    confidence_level: float
    cluster: str

    def __post_init__(self):
        if self.n_resamples < 200:
            raise ConfigError("bootstrap.n_resamples must be >= 200")
        if not 0.5 < self.confidence_level < 1.0:
            raise ConfigError("bootstrap.confidence_level must lie in (0.5, 1)")
        if self.cluster != "calendar_week":
            raise ConfigError("bootstrap.cluster must be 'calendar_week': adjacent daily "
                              "origins are not independent")


@dataclass(frozen=True)
class GateHConfig:
    minimum_test_origins_per_zone: int
    oracle_gain_pass: float
    oracle_gain_fail: float
    minimum_m3_win_rate: float
    maximum_m3_win_rate: float
    oracle_headroom_pass: float
    proxy_spearman_pass: float

    def __post_init__(self):
        if self.oracle_gain_fail >= self.oracle_gain_pass:
            raise ConfigError("gate_h.oracle_gain_fail must be < oracle_gain_pass")
        if not 0 < self.minimum_m3_win_rate < self.maximum_m3_win_rate < 1:
            raise ConfigError("gate_h m3 win-rate window must satisfy 0 < min < max < 1")


@dataclass(frozen=True)
class GateIConfig:
    improvement_pass: float
    improvement_fail: float
    oracle_recovery_pass: float
    oracle_recovery_fail: float
    harm_reduction_pass: float
    maximum_subset_regression: float

    def __post_init__(self):
        if self.improvement_fail >= self.improvement_pass:
            raise ConfigError("gate_i.improvement_fail must be < improvement_pass")
        if self.oracle_recovery_fail >= self.oracle_recovery_pass:
            raise ConfigError("gate_i.oracle_recovery_fail must be < oracle_recovery_pass")


@dataclass(frozen=True)
class ExternalConfig:
    experiment: ExternalExperiment
    periods: ExternalPeriods
    nyiso: NyisoConfig
    weather: WeatherConfig
    model: ExternalModel
    proxy: ExternalProxy
    historical_utility: HistoricalUtilityConfig
    bootstrap: ExternalBootstrap
    gate_h: GateHConfig
    gate_i: GateIConfig
    source_path: str = field(default="", compare=False)

    def __post_init__(self):
        if self.weather.decision_delay_hours != self.experiment.decision_origin_hour_utc:
            raise ConfigError(
                "weather.decision_delay_hours must equal experiment.decision_origin_hour_utc: "
                "the decision origin is the run hour plus the publication delay")

    @staticmethod
    def from_dict(d: dict, source_path: str = "") -> "ExternalConfig":
        expected = {"experiment", "periods", "nyiso", "weather", "model", "proxy",
                    "historical_utility", "bootstrap", "gate_h", "gate_i"}
        unknown, missing = set(d) - expected, expected - set(d)
        if unknown:
            raise ConfigError(f"external: unknown top-level keys {sorted(unknown)}")
        if missing:
            raise ConfigError(f"external: missing top-level keys {sorted(missing)}")
        ny = dict(d["nyiso"])
        zones = [_strict(ExternalZone, z, f"nyiso.zones[{i}]") for i, z in enumerate(ny["zones"])]
        ny["zones"] = zones
        return ExternalConfig(
            experiment=_strict(ExternalExperiment, d["experiment"], "experiment"),
            periods=_strict(ExternalPeriods, d["periods"], "periods"),
            nyiso=_strict(NyisoConfig, ny, "nyiso"),
            weather=_strict(WeatherConfig, d["weather"], "weather"),
            model=_strict(ExternalModel, d["model"], "model"),
            proxy=_strict(ExternalProxy, d["proxy"], "proxy"),
            historical_utility=_strict(HistoricalUtilityConfig, d["historical_utility"],
                                       "historical_utility"),
            bootstrap=_strict(ExternalBootstrap, d["bootstrap"], "bootstrap"),
            gate_h=_strict(GateHConfig, d["gate_h"], "gate_h"),
            gate_i=_strict(GateIConfig, d["gate_i"], "gate_i"),
            source_path=source_path)

    @staticmethod
    def load(path: str | Path) -> "ExternalConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return ExternalConfig.from_dict(yaml.safe_load(fh), str(Path(path).resolve()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source_path", None)
        return d

    def resolved_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @property
    def zone_map(self) -> dict:
        return {z.source_name: z.canonical_name for z in self.nyiso.zones}
