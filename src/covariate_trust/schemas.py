"""Chronos-2 input construction and the fairness assertions between methods.

Method definitions
------------------
M0 target_only                  context: id, timestamp, target
M1 past_covariate_only          context: id, timestamp, target, x           future: none
M2 oracle_future_covariate      context: as M1                              future: id, timestamp, x (true)
M3 forecasted_future_covariate  context: as M1                              future: id, timestamp, x_tilde

``future_df`` never contains the target column.  M1 and M3 must see byte-identical
context; M2 and lambda=0 M3 must see byte-identical future covariates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

M0 = "M0_target_only"
M1 = "M1_past_covariate_only"
M2 = "M2_oracle_future_covariate"
M3 = "M3_forecasted_future_covariate"
METHODS = (M0, M1, M2, M3)

ID_COLUMN = "id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target"
COVARIATE_COLUMN = "x"

SERIES_START = pd.Timestamp("2020-01-01 00:00:00")


class SchemaError(AssertionError):
    """Raised when a fairness or schema invariant is violated."""


@dataclass(frozen=True)
class ModelInputs:
    method: str
    context_df: pd.DataFrame
    future_df: pd.DataFrame | None
    item_id: str
    origin: int
    horizon: int


def timestamps(index: np.ndarray | range, freq: str) -> pd.DatetimeIndex:
    idx = np.asarray(list(index), dtype=int)
    return pd.DatetimeIndex(SERIES_START + pd.to_timedelta(idx * pd.Timedelta(1, unit=freq)))


def build_inputs(method: str, item_id: str, y: np.ndarray, x: np.ndarray,
                 origin: int, horizon: int, context_length: int, freq: str,
                 x_future: np.ndarray | None = None) -> ModelInputs:
    """Build the (context_df, future_df) pair for one task."""
    if method not in METHODS:
        raise SchemaError(f"unknown method {method!r}")
    start = origin - context_length
    if start < 0:
        raise SchemaError(f"context window starts before the series ({start})")

    ctx_idx = np.arange(start, origin)
    ctx = pd.DataFrame({
        ID_COLUMN: item_id,
        TIMESTAMP_COLUMN: timestamps(ctx_idx, freq),
        TARGET_COLUMN: np.asarray(y[start:origin], dtype=float),
    })
    if method != M0:
        ctx[COVARIATE_COLUMN] = np.asarray(x[start:origin], dtype=float)

    future = None
    if method in (M2, M3):
        if x_future is None:
            raise SchemaError(f"{method} requires a future covariate path")
        if len(x_future) != horizon:
            raise SchemaError(f"future covariate length {len(x_future)} != horizon {horizon}")
        fut_idx = np.arange(origin, origin + horizon)
        future = pd.DataFrame({
            ID_COLUMN: item_id,
            TIMESTAMP_COLUMN: timestamps(fut_idx, freq),
            COVARIATE_COLUMN: np.asarray(x_future, dtype=float),
        })
    elif x_future is not None:
        raise SchemaError(f"{method} must not be given a future covariate path")

    inputs = ModelInputs(method, ctx, future, item_id, origin, horizon)
    assert_task_invariants(inputs)
    return inputs


# ---------------------------------------------------------------- checks ----

def assert_task_invariants(inputs: ModelInputs) -> None:
    ctx, fut, method = inputs.context_df, inputs.future_df, inputs.method

    expected_ctx = [ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN] + ([] if method == M0 else [COVARIATE_COLUMN])
    if list(ctx.columns) != expected_ctx:
        raise SchemaError(f"{method}: context columns {list(ctx.columns)} != {expected_ctx}")
    if method == M0 and COVARIATE_COLUMN in ctx.columns:
        raise SchemaError("M0 must not contain a covariate column")
    if method in (M0, M1) and fut is not None:
        raise SchemaError(f"{method} must not receive a future_df")
    if not np.isfinite(ctx[TARGET_COLUMN].to_numpy()).all():
        raise SchemaError(f"{method}: non-finite target in context")

    assert_timestamp_continuity(ctx[TIMESTAMP_COLUMN])

    if fut is not None:
        if TARGET_COLUMN in fut.columns:
            raise SchemaError(f"{method}: future_df must never contain the target column")
        if list(fut.columns) != [ID_COLUMN, TIMESTAMP_COLUMN, COVARIATE_COLUMN]:
            raise SchemaError(f"{method}: future_df columns {list(fut.columns)} unexpected")
        if len(fut) != inputs.horizon:
            raise SchemaError(f"{method}: future_df has {len(fut)} rows, expected {inputs.horizon}")
        assert_timestamp_continuity(fut[TIMESTAMP_COLUMN])
        step = ctx[TIMESTAMP_COLUMN].iloc[1] - ctx[TIMESTAMP_COLUMN].iloc[0]
        if fut[TIMESTAMP_COLUMN].iloc[0] - ctx[TIMESTAMP_COLUMN].iloc[-1] != step:
            raise SchemaError(f"{method}: future_df does not start one step after the context")


def assert_timestamp_continuity(ts: pd.Series) -> None:
    idx = pd.DatetimeIndex(ts)
    if len(idx) < 2:
        return
    deltas = np.diff(idx.asi8)
    if not (deltas == deltas[0]).all():
        raise SchemaError("timestamps are not regularly spaced")
    if deltas[0] <= 0:
        raise SchemaError("timestamps are not strictly increasing")


def assert_context_equality(a: ModelInputs, b: ModelInputs) -> None:
    """M1 and M3 must see exactly the same context."""
    if not a.context_df.equals(b.context_df):
        raise SchemaError(f"{a.method} and {b.method} contexts differ")


def assert_future_equality(a: ModelInputs, b: ModelInputs) -> None:
    """M2 and lambda=0 M3 must see exactly the same future covariates."""
    if a.future_df is None or b.future_df is None:
        raise SchemaError("both inputs must have a future_df to compare")
    if not a.future_df.equals(b.future_df):
        raise SchemaError(f"{a.method} and {b.method} future_df differ")


def assert_no_future_target_leak(fut: pd.DataFrame | None) -> None:
    if fut is None:
        return
    if TARGET_COLUMN in fut.columns:
        raise SchemaError("future target leakage detected")
