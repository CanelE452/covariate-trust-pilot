"""Chronos API and inference-output checks (tests 21-27).

Tests 21-22 inspect the installed package and the adapter directly.  Tests 23-27
concern actual model output, so they read the most recent smoke run instead of
loading the model again; they skip when no smoke run exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from covariate_trust.chronos_adapter import (CROSS_LEARNING_ARG, ChronosEnvironmentError,
                                             predict_df_signature, resolve_quantile_columns,
                                             supports_cross_learning)
from covariate_trust.dgp import build_target, generate_base_series
from covariate_trust.schemas import M1, build_inputs

chronos = pytest.importorskip("chronos", reason="chronos-forecasting is not installed")


def _latest_smoke(root: Path) -> dict | None:
    runs = sorted((root / "runs").glob("*_smoke*"))
    for run in reversed(runs):
        report = run / "tables" / "smoke_report.json"
        if report.exists():
            data = json.loads(report.read_text())
            if data.get("checks"):
                return data
    return None


@pytest.fixture(scope="module")
def smoke_report(project_root_path):
    data = _latest_smoke(project_root_path)
    if data is None:
        pytest.skip("no smoke run available yet")
    return data


def test_21_installed_api_supports_cross_learning():
    """Test 21: the installed predict_df exposes a cross_learning argument."""
    sig = predict_df_signature()
    assert CROSS_LEARNING_ARG in sig.parameters, str(sig)
    assert supports_cross_learning() is True
    assert sig.parameters[CROSS_LEARNING_ARG].default is False


def test_22_adapter_always_passes_cross_learning_false(small_cfg):
    """Test 22: every adapter call sends cross_learning=False explicitly."""
    from covariate_trust.chronos_adapter import LoadedPipeline, predict_task

    captured = {}

    class FakePipeline:
        def predict_df(self, df, **kwargs):
            captured.update(kwargs)
            n = kwargs["prediction_length"]
            out = pd.DataFrame({"id": ["t"] * n,
                                "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
                                "predictions": np.zeros(n)})
            for q in kwargs["quantile_levels"]:
                out[str(q)] = float(q)
            return out

    s = generate_base_series(0, small_cfg)
    y = build_target(s, 0.5)
    inputs = build_inputs(M1, "t", y, s.x, small_cfg.experiment.primary_origin, 24,
                          small_cfg.experiment.context_length, small_cfg.experiment.frequency)
    loaded = LoadedPipeline(FakePipeline(), "fake", "cpu", "float32", None, None, {})
    q, _ = predict_task(loaded, inputs, small_cfg.experiment.quantile_levels,
                        small_cfg.experiment.context_length, small_cfg.experiment.frequency)
    assert captured["cross_learning"] is False
    assert captured["prediction_length"] == 24
    assert captured["context_length"] == small_cfg.experiment.context_length
    assert q.shape == (24, len(small_cfg.experiment.quantile_levels))


def test_22b_adapter_rejects_missing_quantile_columns():
    frame = pd.DataFrame({"predictions": [1.0], "0.5": [1.0]})
    with pytest.raises(ChronosEnvironmentError):
        resolve_quantile_columns(frame, [0.1, 0.5, 0.9])
    assert resolve_quantile_columns(frame, [0.5]) == ["0.5"]


def test_23_prediction_horizon_length(smoke_report):
    """Test 23: the model returns exactly prediction_length rows."""
    assert smoke_report["checks"]["prediction_length"]["all_match"] is True
    assert smoke_report["checks"]["prediction_length"]["expected"] == 24


def test_24_quantile_columns_are_present(smoke_report):
    """Test 24: every requested quantile level comes back as a column."""
    q = smoke_report["checks"]["quantile_columns"]
    assert len(q["resolved"]) == len(q["requested"])
    assert smoke_report["checks"]["predictions_column_present"] is True


def test_25_no_nan_or_inf_in_predictions(smoke_report):
    """Test 25: predictions are finite."""
    assert smoke_report["checks"]["no_nan_or_inf"] is True


def test_26_repeated_inference_is_deterministic(smoke_report):
    """Test 26: three identical calls agree to within the accepted tolerance."""
    det = smoke_report["checks"]["determinism"]
    assert det["status"] in {"EXACT", "WITHIN_TOLERANCE"}, det
    assert det["max_abs_diff"] <= det["accepted_tolerance"]


def test_27_batch_equivalence_probe_recorded_and_batching_disabled(smoke_report):
    """Test 27: batching is either proven equivalent or (as here) simply not used."""
    batch = smoke_report["checks"]["batch_equivalence"]
    assert batch["used_in_diagnostic"] is False
    if "max_abs_diff" in batch:
        assert batch["equivalent"] == (batch["max_abs_diff"] <= batch["tolerance"])


def test_27b_quantile_crossing_is_reported(smoke_report):
    assert 0.0 <= smoke_report["checks"]["quantile_crossing"]["max_rate"] <= 1.0
