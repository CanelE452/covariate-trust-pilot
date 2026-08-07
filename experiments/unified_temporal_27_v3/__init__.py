"""Unified temporal 27-scenario experiment, version 3.

This package is intentionally independent from every earlier experiment.
Importing it never generates data, trains a model, or writes an artifact.
"""

from .config import DEFAULT_CONFIG, ExperimentConfig
from .scenarios import GROUP_A, GROUP_B, GROUP_C, SCENARIOS, ScenarioSpec

__all__ = [
    "DEFAULT_CONFIG",
    "ExperimentConfig",
    "ScenarioSpec",
    "SCENARIOS",
    "GROUP_A",
    "GROUP_B",
    "GROUP_C",
]
