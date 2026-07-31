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


# ---------------------------------------------------------------- follow-up ---

@pytest.fixture(scope="session")
def boundary_cfg():
    from covariate_trust.config import BoundaryConfig
    return BoundaryConfig.load(ROOT / "configs" / "study1b_boundary.yaml")


@pytest.fixture(scope="session")
def dynamic_cfg():
    from covariate_trust.config import DynamicConfig
    return DynamicConfig.load(ROOT / "configs" / "study2_dynamic_reliability.yaml")


@pytest.fixture(scope="session")
def small_boundary_cfg(boundary_cfg):
    """Same structure, few series - keeps the tests fast."""
    from covariate_trust.config import BoundaryConfig
    d = boundary_cfg.to_dict()
    d.pop("inherited_from_pilot_yaml")
    d["grid"]["n_series_per_cell"] = 4
    return BoundaryConfig.from_dict(d, boundary_cfg.inherited)


@pytest.fixture(scope="session")
def small_dynamic_cfg(dynamic_cfg):
    from covariate_trust.config import DynamicConfig
    d = dynamic_cfg.to_dict()
    d.pop("inherited_from_pilot_yaml")
    d["grid"]["n_series_per_condition"] = 4
    return DynamicConfig.from_dict(d, dynamic_cfg.inherited)


@pytest.fixture(scope="session")
def confirmation_cfg():
    from covariate_trust.config import ConfirmationConfig
    return ConfirmationConfig.load(ROOT / "configs" / "study2b_d7_confirmation.yaml")


@pytest.fixture(scope="session")
def small_confirmation_cfg(confirmation_cfg):
    from covariate_trust.config import ConfirmationConfig
    d = confirmation_cfg.to_dict()
    d.pop("inherited")
    d["grid"]["n_series_per_condition"] = 4
    return ConfirmationConfig.from_dict(d, confirmation_cfg.inherited)


# ------------------------------------------------------------------ study 3 ---

@pytest.fixture(scope="session")
def external_cfg():
    from covariate_trust.config import ExternalConfig
    return ExternalConfig.load(ROOT / "configs" / "study3_real_vintage.yaml")


@pytest.fixture(scope="session")
def processed_dir() -> Path:
    return ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def real_panel(external_cfg, processed_dir):
    """Assembled real origins, or skip if the download has not been run."""
    import pandas as pd
    import pytest as _pytest
    from covariate_trust.real_vintage import assemble_origins
    need = ["load_hourly.parquet", "weather_verification.parquet",
            "weather_runs_v2_07utc.parquet"]
    if not all((processed_dir / n).exists() for n in need):
        _pytest.skip("external data has not been downloaded yet")
    load = pd.read_parquet(processed_dir / "load_hourly.parquet")
    ver = pd.read_parquet(processed_dir / "weather_verification.parquet")
    runs = pd.read_parquet(processed_dir / "weather_runs_v2_07utc.parquet")
    zones = sorted(load["zone"].unique())[:2]
    cutoff = pd.Timestamp(external_cfg.periods.requested_start) + pd.Timedelta(days=60)
    panel, _ = assemble_origins(load[load["zone"].isin(zones)],
                                ver[ver["zone"].isin(zones)],
                                runs[(runs["zone"].isin(zones)) & (runs["origin_utc"] <= cutoff)],
                                external_cfg)
    if panel.empty:
        _pytest.skip("no usable real origins in the sampled window")
    return panel
