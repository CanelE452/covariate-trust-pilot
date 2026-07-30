"""Thin adapter around the installed Chronos-2 pipeline.

Everything about the API here is checked against the *installed* signature via
``inspect``; nothing is assumed from documentation or memory.  ``cross_learning``
is always passed explicitly as ``False``.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import ID_COLUMN, TARGET_COLUMN, TIMESTAMP_COLUMN, ModelInputs
from .seeds import array_hash, stable_hash

CROSS_LEARNING_ARG = "cross_learning"


class ChronosEnvironmentError(RuntimeError):
    """Raised when the installed Chronos API cannot support this study."""


def set_hf_home(root: Path) -> str:
    """Point the HuggingFace cache inside the project before any HF import."""
    hf_home = str(Path(root) / ".cache" / "huggingface")
    os.environ["HF_HOME"] = hf_home
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    return hf_home


def predict_df_signature() -> inspect.Signature:
    from chronos import Chronos2Pipeline
    return inspect.signature(Chronos2Pipeline.predict_df)


def supports_cross_learning() -> bool:
    return CROSS_LEARNING_ARG in predict_df_signature().parameters


def api_report() -> dict:
    """Facts about the installed API, recorded in audit.json."""
    from chronos import Chronos2Pipeline
    sig = predict_df_signature()
    return {
        "predict_df_signature": str(sig),
        "predict_df_parameters": list(sig.parameters),
        "cross_learning_supported": CROSS_LEARNING_ARG in sig.parameters,
        "cross_learning_default": (
            sig.parameters[CROSS_LEARNING_ARG].default
            if CROSS_LEARNING_ARG in sig.parameters else None),
        "predict_signature": str(inspect.signature(Chronos2Pipeline.predict)),
        "from_pretrained_signature": str(inspect.signature(Chronos2Pipeline.from_pretrained)),
        "pipeline_methods": [m for m in dir(Chronos2Pipeline) if not m.startswith("_")],
    }


@dataclass
class LoadedPipeline:
    pipeline: Any
    model_id: str
    device: str
    dtype: str
    attention_implementation: str | None
    revision: str | None
    load_kwargs: dict


def load_pipeline(model_id: str, device: str, attention_implementation: str | None = None,
                  revision: str | None = None) -> LoadedPipeline:
    """Load Chronos-2, recording exactly which kwargs the install accepted."""
    from chronos import Chronos2Pipeline

    if not supports_cross_learning():
        raise ChronosEnvironmentError(
            "installed Chronos2Pipeline.predict_df has no 'cross_learning' argument; "
            "this study requires it and will not run"
        )

    base: dict[str, Any] = {"device_map": device}
    if revision:
        base["revision"] = revision

    attempts: list[dict] = []
    if attention_implementation:
        attempts.append({**base, "attn_implementation": attention_implementation})
    attempts.append(dict(base))

    last_error: Exception | None = None
    attempts_note: dict = attempts[-1]
    for kwargs in attempts:
        try:
            pipe = Chronos2Pipeline.from_pretrained(model_id, **kwargs)
            used_attn = kwargs.get("attn_implementation")
            dtype = str(getattr(getattr(pipe, "model", None), "dtype", "unknown"))
            return LoadedPipeline(pipe, model_id, device, dtype, used_attn, revision, kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded and retried without the kwarg
            last_error = exc
            attempts_note = kwargs
    raise ChronosEnvironmentError(
        f"could not load {model_id} (last kwargs {attempts_note}): {last_error}"
    ) from last_error


def resolve_quantile_columns(pred: pd.DataFrame, quantile_levels: list[float]) -> list[str]:
    """Map requested quantile levels onto the returned column names."""
    numeric: dict[float, str] = {}
    for col in pred.columns:
        try:
            numeric[round(float(col), 6)] = col
        except (TypeError, ValueError):
            continue
    missing = [q for q in quantile_levels if round(float(q), 6) not in numeric]
    if missing:
        raise ChronosEnvironmentError(
            f"prediction frame is missing quantile columns {missing}; columns = {list(pred.columns)}"
        )
    return [numeric[round(float(q), 6)] for q in quantile_levels]


def predict_task(loaded: LoadedPipeline, inputs: ModelInputs, quantile_levels: list[float],
                 context_length: int, freq: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Run one task and return (quantile matrix of shape (H, Q), raw frame)."""
    pred = loaded.pipeline.predict_df(
        inputs.context_df,
        future_df=inputs.future_df,
        id_column=ID_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        target=TARGET_COLUMN,
        prediction_length=inputs.horizon,
        quantile_levels=list(quantile_levels),
        context_length=context_length,
        cross_learning=False,
        freq=freq,
    )
    if len(pred) != inputs.horizon:
        raise ChronosEnvironmentError(
            f"expected {inputs.horizon} predicted rows, got {len(pred)}")
    cols = resolve_quantile_columns(pred, quantile_levels)
    q = pred[cols].to_numpy(dtype=float)
    if not np.isfinite(q).all():
        raise ChronosEnvironmentError("prediction contains NaN or inf")
    return q, pred


def task_hash(model_id: str, revision: str | None, base_series_id: int, share: float,
              origin: int, horizon: int, method: str, lam: float | None,
              inputs: ModelInputs) -> str:
    """Task key including content hashes of both dataframes."""
    ctx_hash = array_hash(inputs.context_df.drop(columns=[ID_COLUMN, TIMESTAMP_COLUMN]).to_numpy(dtype=float))
    if inputs.future_df is None:
        fut_hash = "none"
    else:
        fut_hash = array_hash(
            inputs.future_df.drop(columns=[ID_COLUMN, TIMESTAMP_COLUMN]).to_numpy(dtype=float))
    return stable_hash(model_id, revision or "default", base_series_id, round(float(share), 6),
                       origin, horizon, method, "na" if lam is None else round(float(lam), 6),
                       ctx_hash, fut_hash)


def context_and_future_hashes(inputs: ModelInputs) -> tuple[str, str]:
    ctx = array_hash(inputs.context_df.drop(columns=[ID_COLUMN, TIMESTAMP_COLUMN]).to_numpy(dtype=float))
    fut = ("none" if inputs.future_df is None else
           array_hash(inputs.future_df.drop(columns=[ID_COLUMN, TIMESTAMP_COLUMN]).to_numpy(dtype=float)))
    return ctx, fut
