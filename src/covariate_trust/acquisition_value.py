"""Study 4 premium-value labels.

    V_wql(i,t) = L_base_wql(i,t) - L_premium_wql(i,t)
    V_q90(i,t) = L_base_q90(i,t) - L_premium_q90(i,t)

Positive means the premium (weather-conditioned) forecast helped at that
zone/origin.  WQL comes straight from the Study 3 task table; the q90 pinball is
not stored there, so it is recomputed from the Study 3 prediction cache and the
realised load using the *same* normalisation as ``metrics.wql`` restricted to a
single quantile level.

These labels use the realised future load and are therefore never features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .acquisition_features import KEY_COLUMNS, ORIGIN_COLUMN, ZONE_COLUMN
from .metrics import EPSILON, pinball


class ValueError_(ValueError):
    pass


def single_quantile_loss(y: np.ndarray, qhat: np.ndarray, level: float) -> float:
    """Same normalisation as ``metrics.wql`` with one quantile level."""
    y = np.asarray(y, dtype=float)
    qhat = np.asarray(qhat, dtype=float)
    total = float(pinball(y, qhat, level).sum())
    return 2.0 * total / (float(np.abs(y).sum()) + EPSILON)


def attach_realized_load(
    predictions: pd.DataFrame, load_hourly: pd.DataFrame, timestamp_column: str = "timestamp_utc"
) -> pd.DataFrame:
    """Join the realised load onto each (origin, horizon) prediction row."""
    work = predictions.copy()
    work["valid_time_utc"] = pd.to_datetime(work[ORIGIN_COLUMN]) + pd.to_timedelta(
        work["h_index"].astype(int), unit="h"
    )
    load = load_hourly.rename(columns={timestamp_column: "valid_time_utc"})
    merged = work.merge(
        load[[ZONE_COLUMN, "valid_time_utc", "load_mw"]],
        on=[ZONE_COLUMN, "valid_time_utc"],
        how="left",
    )
    return merged


def compute_q90_losses(
    predictions: pd.DataFrame,
    load_hourly: pd.DataFrame,
    base_method: str,
    premium_method: str,
    level: float = 0.9,
    horizon: int = 24,
) -> pd.DataFrame:
    """Per (zone, origin) normalised q90 pinball for the base and premium method."""
    wanted = predictions[predictions["method"].isin([base_method, premium_method])].copy()
    # The Study 3 prediction cache stores the origin as a string; every other
    # table uses datetime64, so normalise here rather than at each merge.
    wanted[ORIGIN_COLUMN] = pd.to_datetime(wanted[ORIGIN_COLUMN])
    merged = attach_realized_load(wanted, load_hourly)
    column = f"q{level}"
    if column not in merged.columns:
        raise ValueError_(f"prediction cache has no {column} column")

    rows: list[dict] = []
    for (zone, origin, method), group in merged.groupby(
        [ZONE_COLUMN, ORIGIN_COLUMN, "method"], sort=False
    ):
        if len(group) != horizon or group["load_mw"].isna().any():
            continue  # incomplete task: dropped, and counted by the caller
        loss = single_quantile_loss(group["load_mw"].to_numpy(), group[column].to_numpy(), level)
        rows.append({ZONE_COLUMN: zone, ORIGIN_COLUMN: origin, "method": method, "q90": loss})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError_("no complete tasks when computing the q90 pinball losses")
    wide = frame.pivot_table(
        index=list(KEY_COLUMNS), columns="method", values="q90", aggfunc="first"
    ).reset_index()
    wide = wide.rename(columns={base_method: "q90_m1", premium_method: "q90_m3"})
    return wide.dropna(subset=["q90_m1", "q90_m3"])


def build_value_labels(task_metrics: pd.DataFrame, q90_losses: pd.DataFrame) -> pd.DataFrame:
    """Assemble the base/premium losses and both premium-value labels."""
    needed = {"wql_m1", "wql_m3"}
    missing = needed - set(task_metrics.columns)
    if missing:
        raise ValueError_(f"task metrics are missing {sorted(missing)}")
    base = task_metrics[list(KEY_COLUMNS) + ["wql_m1", "wql_m3", "nmae_m1", "nmae_m3"]].copy()
    frame = base.merge(q90_losses, on=list(KEY_COLUMNS), how="inner")
    frame["v_wql"] = frame["wql_m1"] - frame["wql_m3"]
    frame["v_q90"] = frame["q90_m1"] - frame["q90_m3"]
    frame["premium_positive"] = (frame["v_wql"] > 0).astype(int)
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def value_distribution_summary(labels: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    """Skew and tail description of the premium value, overall or per group."""

    def _describe(frame: pd.DataFrame) -> dict:
        v = frame["v_wql"].to_numpy(dtype=float)
        return {
            "n": int(v.size),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "positive_rate": float((v > 0).mean()),
            "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
            "min": float(v.min()),
            "max": float(v.max()),
            "skew": float(pd.Series(v).skew()),
            "mean_positive": float(v[v > 0].mean()) if (v > 0).any() else float("nan"),
            "mean_negative": float(v[v <= 0].mean()) if (v <= 0).any() else float("nan"),
        }

    if group is None:
        return pd.DataFrame([{"group": "all", **_describe(labels)}])
    rows = [{"group": str(key), **_describe(sub)} for key, sub in labels.groupby(group)]
    return pd.DataFrame(rows)
