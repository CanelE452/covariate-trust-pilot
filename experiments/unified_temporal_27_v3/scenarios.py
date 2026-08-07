"""Canonical, declarative catalogue for the v3 12 + 9 + 6 design."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import DEFAULT_CONFIG, ExperimentConfig


@dataclass(frozen=True)
class ScenarioSpec:
    """Settings that identify one generated dataset without running it."""

    index: int
    scenario_id: str
    group: str
    design: str
    structure: str
    strength: str
    variant: str
    p0: float
    mu0: float
    phi: float
    rho_occ: float = 0.0
    rho_mag: float = 0.0
    jitter: float = 0.0
    pair: int | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_scenarios(config: ExperimentConfig = DEFAULT_CONFIG) -> tuple[ScenarioSpec, ...]:
    """Return the exact 27 IDs in presentation order."""

    rows: list[ScenarioSpec] = []
    a_design = (
        (1, "occurrence", "weak", config.a_weak_rho, 0.0),
        (2, "occurrence", "strong", config.a_strong_rho, 0.0),
        (3, "magnitude", "weak", 0.0, config.a_weak_rho),
        (4, "magnitude", "strong", 0.0, config.a_strong_rho),
        (5, "both", "weak", config.a_weak_rho, config.a_weak_rho),
        (6, "both", "strong", config.a_strong_rho, config.a_strong_rho),
    )
    index = 1
    for pair, structure, strength, rho_occ, rho_mag in a_design:
        for variant in ("iid", "non_iid"):
            rows.append(ScenarioSpec(
                index=index,
                scenario_id=f"A{index:02d}_{structure}_{strength}_{variant}",
                group="A",
                design="structured_first_random_order_matched_pair",
                structure=structure,
                strength=strength,
                variant=variant,
                pair=pair,
                p0=config.a_occurrence_rate,
                mu0=config.a_positive_mean,
                phi=config.a_positive_cv2,
                rho_occ=rho_occ if variant == "non_iid" else 0.0,
                rho_mag=rho_mag if variant == "non_iid" else 0.0,
                notes=(
                    "split-local uniform random-order control of the structured pool"
                    if variant == "iid" else
                    "unconditioned stationary nonperiodic Markov occurrence/event-state path"
                ),
                extra={
                    "seasonality": False,
                    "periodic_component": False,
                    "matching_unit": "series_by_chronological_split",
                    "generation_order": "structured_first",
                    "control_semantics": "split_local_uniform_random_order",
                    "magnitude_state_quantile": config.a_magnitude_high_quantile,
                },
            ))
            index += 1

    b_index = 1
    for strength, amplitude in (
        ("weak", config.b_weak_amplitude),
        ("medium", config.b_medium_amplitude),
        ("strong", config.b_strong_amplitude),
    ):
        for label, jitter in (
            ("0", 0.0),
            ("015", config.b_jitter_weak),
            ("035", config.b_jitter_strong),
        ):
            rows.append(ScenarioSpec(
                index=12 + b_index,
                scenario_id=f"B{b_index:02d}_season_{strength}_jitter{label}",
                group="B",
                design="seasonality_only",
                structure="seasonality",
                strength=strength,
                variant=f"jitter{label}",
                p0=1.0,
                mu0=config.b_baseline,
                phi=0.0,
                jitter=jitter,
                notes="positive additive seasonal signal with Gaussian noise; no spike",
                extra={
                    "baseline": config.b_baseline,
                    "amplitude": amplitude,
                    "noise_std": config.b_noise_std,
                },
            ))
            b_index += 1

    common_c = {
        "p0": config.c_occurrence_rate,
        "mu0": config.c_positive_mean,
        "phi": config.c_positive_cv2,
    }
    rows.extend((
        ScenarioSpec(
            index=22, scenario_id="C01_regular_season_spike", group="C",
            design="spike_plus_seasonality", structure="both", strength="strong",
            variant="regular", **common_c,
            notes="regular periodic occurrence and magnitude with intermittent spikes",
            extra={"occurrence_amplitude": config.c_strong_amplitude,
                   "magnitude_amplitude": config.c_strong_amplitude},
        ),
        ScenarioSpec(
            index=23, scenario_id="C02_jitter_season_spike", group="C",
            design="spike_plus_seasonality", structure="both", strength="strong",
            variant="jitter035", jitter=config.b_jitter_strong, **common_c,
            notes="strong phase jitter with intermittent spikes",
            extra={"occurrence_amplitude": config.c_strong_amplitude,
                   "magnitude_amplitude": config.c_strong_amplitude},
        ),
        ScenarioSpec(
            index=24, scenario_id="C03_weak_season_erratic", group="C",
            design="spike_plus_seasonality", structure="both", strength="weak",
            variant="high_cv2", p0=config.c_occurrence_rate,
            mu0=config.c_positive_mean, phi=1.20,
            notes="weak seasonality and high positive CV^2",
            extra={"occurrence_amplitude": config.c_weak_amplitude,
                   "magnitude_amplitude": config.c_weak_amplitude},
        ),
        ScenarioSpec(
            index=25, scenario_id="C04_population_mix", group="C",
            design="population_mix", structure="population_mix", strength="strong",
            variant="half_occurrence_half_magnitude", **common_c,
            notes="stored 50/50 occurrence-driven and magnitude-driven membership",
            extra={"occurrence_amplitude": config.c_strong_amplitude,
                   "magnitude_amplitude": config.c_strong_amplitude,
                   "population_fraction": 0.5},
        ),
        ScenarioSpec(
            index=26, scenario_id="C05_regime_switch", group="C",
            design="regime_switch", structure="regime_switch", strength="strong",
            variant=f"switch_at_{config.c_regime_switch}", **common_c,
            notes="stored boundary and parameters for both regimes",
            extra={"regime_switch": config.c_regime_switch,
                   "before_structure": "occurrence_seasonality",
                   "before_occurrence_amplitude": config.c_strong_amplitude,
                   "before_magnitude_amplitude": 0.0,
                   "after_structure": "magnitude_seasonality",
                   "after_occurrence_amplitude": 0.0,
                   "after_magnitude_amplitude": config.c_strong_amplitude},
        ),
        ScenarioSpec(
            index=27, scenario_id="C06_lumpy_seasonal", group="C",
            design="spike_plus_seasonality", structure="both", strength="strong",
            variant="low_p_high_cv2", p0=0.12,
            mu0=config.c_positive_mean, phi=1.20,
            notes="low occurrence rate, high positive CV^2 and clear seasonality",
            extra={"occurrence_amplitude": config.c_strong_amplitude,
                   "magnitude_amplitude": config.c_strong_amplitude},
        ),
    ))

    scenarios = tuple(rows)
    validate_catalogue(scenarios)
    return scenarios


def validate_catalogue(scenarios: tuple[ScenarioSpec, ...]) -> None:
    ids = [spec.scenario_id for spec in scenarios]
    if len(scenarios) != 27 or len(set(ids)) != 27:
        raise ValueError("catalogue must have exactly 27 unique scenario IDs")
    counts = {group: sum(spec.group == group for spec in scenarios) for group in "ABC"}
    if counts != {"A": 12, "B": 9, "C": 6}:
        raise ValueError(f"invalid group counts: {counts}")
    if [spec.index for spec in scenarios] != list(range(1, 28)):
        raise ValueError("scenario indices must be consecutive 1..27")
    for spec in scenarios:
        if spec.group == "B":
            if spec.p0 != 1.0 or spec.phi != 0.0:
                raise ValueError(f"Group B must have p0=1 and no Gamma phi: {spec.scenario_id}")
        elif not (0.0 < spec.p0 < 1.0 and spec.mu0 > 0.0 and spec.phi > 0.0):
            raise ValueError(f"invalid hurdle parameters: {spec.scenario_id}")
        if spec.group == "A" and (
            spec.jitter != 0.0
            or spec.extra.get("seasonality") is not False
            or spec.extra.get("periodic_component") is not False
        ):
            raise ValueError(f"Group A cannot contain periodic settings: {spec.scenario_id}")


SCENARIOS = build_scenarios()
SCENARIO_BY_ID = {spec.scenario_id: spec for spec in SCENARIOS}
GROUP_A = tuple(spec for spec in SCENARIOS if spec.group == "A")
GROUP_B = tuple(spec for spec in SCENARIOS if spec.group == "B")
GROUP_C = tuple(spec for spec in SCENARIOS if spec.group == "C")


__all__ = [
    "ScenarioSpec",
    "build_scenarios",
    "validate_catalogue",
    "SCENARIOS",
    "SCENARIO_BY_ID",
    "GROUP_A",
    "GROUP_B",
    "GROUP_C",
]
