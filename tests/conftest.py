from __future__ import annotations

from pathlib import Path

import pytest

from covariate_trust.config import PilotConfig, Study0Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pilot_cfg() -> PilotConfig:
    return PilotConfig.load(ROOT / "configs" / "pilot.yaml")


@pytest.fixture(scope="session")
def study0_cfg() -> Study0Config:
    return Study0Config.load(ROOT / "configs" / "study0.yaml")


@pytest.fixture(scope="session")
def small_cfg() -> PilotConfig:
    """Same structure as the pilot config but only a few series (fast tests)."""
    d = PilotConfig.load(ROOT / "configs" / "pilot.yaml").to_dict()
    d["grid"]["n_series_per_cell"] = 4
    return PilotConfig.from_dict(d)


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    return ROOT
