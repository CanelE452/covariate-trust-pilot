"""Study 4 selector features, all available at the 07 UTC decision origin.

The premium forecast (M3) is *defined* as the output you only get by spending a
slot, so the current M3 quantiles must never reach the selector.  This module
therefore builds the feature frame from an explicit allow-list and refuses to
emit any column outside it; ``assert_no_forbidden_columns`` is the guard the
tests exercise.

Nothing here touches Study 3 code or artifacts — the Study 3 tables are read
only, and every derived quantity is written into the Study 4 run directory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ZONE_COLUMN = "zone"
ORIGIN_COLUMN = "origin_utc"
KEY_COLUMNS = (ZONE_COLUMN, ORIGIN_COLUMN)

#: Substrings that must never appear in a selector feature name.  These cover
#: the current premium output, the current realised outcome and the ex-post gain.
FORBIDDEN_SUBSTRINGS = (
    "m3",
    "m2",
    "realized_weather_error_ratio",
    "wql_m1",
    "v_wql",
    "v_q90",
    "v_future",
    "v_oracle",
    "oracle",
    "premium",
    "load_mw_future",
    "target",
)

#: Explicitly allowed exceptions to the substring rule (names that contain a
#: forbidden substring but are legitimate past-only features).
ALLOWED_EXCEPTIONS = ("recent_base_wql_28",)


class FeatureLeakageError(ValueError):
    pass


def assert_no_forbidden_columns(frame: pd.DataFrame, allowed: list[str]) -> None:
    """Every non-key column must be in the allow-list and look past-only."""
    extra = [c for c in frame.columns if c not in set(allowed) | set(KEY_COLUMNS) | {"split"}]
    if extra:
        raise FeatureLeakageError(f"feature frame carries columns outside the allow-list: {extra}")
    for column in frame.columns:
        if column in KEY_COLUMNS or column in ALLOWED_EXCEPTIONS or column == "split":
            continue
        low = column.lower()
        hit = [s for s in FORBIDDEN_SUBSTRINGS if s in low]
        if hit:
            raise FeatureLeakageError(f"feature '{column}' matches forbidden pattern {hit}")


# --------------------------------------------------------------------------
# Individual feature blocks
# --------------------------------------------------------------------------


def calendar_features(origins: pd.Series) -> pd.DataFrame:
    ts = pd.DatetimeIndex(origins)
    month = ts.month.to_numpy(dtype=float)
    dow = ts.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "day_of_week_sin": np.sin(2 * np.pi * dow / 7.0),
            "day_of_week_cos": np.cos(2 * np.pi * dow / 7.0),
        }
    )


def recent_load_features(
    load_hourly: pd.DataFrame, keys: pd.DataFrame, timestamp_column: str = "timestamp_utc"
) -> pd.DataFrame:
    """Load statistics over the 24h and 168h strictly before each origin."""
    out = {name: np.full(len(keys), np.nan) for name in (
        "recent_load_mean_24", "recent_load_std_24", "recent_load_mean_168",
        "recent_load_std_168", "recent_load_trend_24", "recent_load_max_24",
        "recent_load_acf24",
    )}
    for zone, group in keys.groupby(ZONE_COLUMN, sort=False):
        series = (
            load_hourly[load_hourly[ZONE_COLUMN] == zone]
            .sort_values(timestamp_column)
            .set_index(timestamp_column)["load_mw"]
        )
        if series.empty:
            continue
        values = series.to_numpy(dtype=float)
        index = series.index
        for row_pos, origin in zip(group.index, group[ORIGIN_COLUMN]):
            # strictly before the origin
            end = int(np.searchsorted(index.values, np.datetime64(origin), side="left"))
            if end <= 0:
                continue
            w24 = values[max(0, end - 24) : end]
            w168 = values[max(0, end - 168) : end]
            if w24.size:
                out["recent_load_mean_24"][row_pos] = w24.mean()
                out["recent_load_std_24"][row_pos] = w24.std(ddof=1) if w24.size > 1 else 0.0
                out["recent_load_max_24"][row_pos] = w24.max()
                if w24.size >= 4:
                    x = np.arange(w24.size, dtype=float)
                    out["recent_load_trend_24"][row_pos] = np.polyfit(x, w24, 1)[0]
            if w168.size:
                out["recent_load_mean_168"][row_pos] = w168.mean()
                out["recent_load_std_168"][row_pos] = w168.std(ddof=1) if w168.size > 1 else 0.0
            if w168.size >= 48:
                a, b = w168[:-24], w168[24:]
                if a.std() > 1e-9 and b.std() > 1e-9:
                    out["recent_load_acf24"][row_pos] = float(np.corrcoef(a, b)[0, 1])
    return pd.DataFrame(out, index=keys.index)


def base_forecast_features(
    predictions: pd.DataFrame, keys: pd.DataFrame, base_method: str
) -> pd.DataFrame:
    """Width and level of the *base* (M1) forecast — never the premium one."""
    base = predictions[predictions["method"] == base_method]
    if base.empty:
        raise FeatureLeakageError(f"no rows for base method {base_method} in the prediction cache")
    work = base[[ZONE_COLUMN, ORIGIN_COLUMN, "q0.1", "q0.5", "q0.9"]].copy()
    work[ORIGIN_COLUMN] = pd.to_datetime(work[ORIGIN_COLUMN])
    work["width"] = work["q0.9"] - work["q0.1"]
    grouped = work.groupby([ZONE_COLUMN, ORIGIN_COLUMN], as_index=False).agg(
        base_interval_width_mean=("width", "mean"),
        base_interval_width_max=("width", "max"),
        base_forecast_level_mean=("q0.5", "mean"),
        base_forecast_peak=("q0.5", "max"),
    )
    return keys.merge(grouped, on=list(KEY_COLUMNS), how="left").drop(columns=list(KEY_COLUMNS))


def weather_features(weather_runs: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """00Z primary run statistics and the 00Z-vs-previous-12Z revision."""
    primary = weather_runs[weather_runs["run_kind"] == "primary"]
    revision = weather_runs[weather_runs["run_kind"] != "primary"]

    prim = primary.groupby([ZONE_COLUMN, ORIGIN_COLUMN], as_index=False).agg(
        primary_weather_mean=("temperature_forecast", "mean"),
        primary_weather_min=("temperature_forecast", "min"),
        primary_weather_max=("temperature_forecast", "max"),
    )
    prim["primary_weather_range"] = prim["primary_weather_max"] - prim["primary_weather_min"]

    merged = primary.merge(
        revision[[ZONE_COLUMN, ORIGIN_COLUMN, "valid_time_utc", "temperature_forecast"]],
        on=[ZONE_COLUMN, ORIGIN_COLUMN, "valid_time_utc"],
        how="inner",
        suffixes=("_primary", "_revision"),
    )
    merged["delta"] = (
        merged["temperature_forecast_primary"] - merged["temperature_forecast_revision"]
    )
    rev = merged.groupby([ZONE_COLUMN, ORIGIN_COLUMN], as_index=False).agg(
        revision_mean=("delta", "mean"),
        revision_bias=("delta", lambda s: float(np.mean(s))),
        revision_rms=("delta", lambda s: float(np.sqrt(np.mean(np.square(s))))),
    )
    out = keys.merge(prim, on=list(KEY_COLUMNS), how="left")
    out = out.merge(rev, on=list(KEY_COLUMNS), how="left")
    return out.drop(columns=list(KEY_COLUMNS))


def rolling_base_loss_features(
    losses: pd.DataFrame, keys: pd.DataFrame, window_days: int, minimum_days: int
) -> pd.DataFrame:
    """Mean base loss over the previous ``window_days`` *completed* origins.

    ``shift(1)`` is what makes this past-only: the current origin's own base
    loss is not yet known at 07 UTC.
    """
    work = losses.sort_values([ZONE_COLUMN, ORIGIN_COLUMN]).copy()
    for source, target in (("wql_m1", "recent_base_wql_28"), ("q90_m1", "recent_base_q90_loss_28")):
        rolled = (
            work.groupby(ZONE_COLUMN)[source]
            .apply(lambda s: s.shift(1).rolling(window_days, min_periods=minimum_days).mean())
            .reset_index(level=0, drop=True)
        )
        work[target] = rolled
    keep = list(KEY_COLUMNS) + ["recent_base_wql_28", "recent_base_q90_loss_28"]
    return keys.merge(work[keep], on=list(KEY_COLUMNS), how="left").drop(columns=list(KEY_COLUMNS))


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@dataclass
class FeatureSources:
    task_losses: pd.DataFrame       # zone, origin_utc, wql_m1, q90_m1 (+ premium, dropped here)
    predictions: pd.DataFrame
    load_hourly: pd.DataFrame
    weather_runs: pd.DataFrame
    reliability: pd.DataFrame       # zone, origin_utc, reported_reliability_ratio
    base_method: str


def build_selector_features(
    sources: FeatureSources, allowed: list[str], window_days: int, minimum_days: int
) -> pd.DataFrame:
    keys = (
        sources.task_losses[list(KEY_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    frame = keys.copy()
    frame["zone"] = keys[ZONE_COLUMN]
    frame = pd.concat([frame, calendar_features(keys[ORIGIN_COLUMN])], axis=1)
    frame = pd.concat([frame, recent_load_features(sources.load_hourly, keys)], axis=1)
    frame = pd.concat(
        [frame, base_forecast_features(sources.predictions, keys, sources.base_method)], axis=1
    )
    frame = pd.concat([frame, weather_features(sources.weather_runs, keys)], axis=1)
    frame = pd.concat(
        [
            frame,
            rolling_base_loss_features(
                sources.task_losses, keys, window_days, minimum_days
            ),
        ],
        axis=1,
    )
    frame = frame.merge(
        sources.reliability[list(KEY_COLUMNS) + ["reported_reliability_ratio"]],
        on=list(KEY_COLUMNS),
        how="left",
    )
    # ``zone`` is both a key and an allowed feature, so it must appear once.
    ordered = list(KEY_COLUMNS) + [
        c for c in allowed if c in frame.columns and c not in KEY_COLUMNS
    ]
    frame = frame.loc[:, ~frame.columns.duplicated()][ordered]
    assert_no_forbidden_columns(frame, allowed)
    missing = [c for c in allowed if c not in frame.columns]
    if missing:
        raise FeatureLeakageError(f"allowed features were not produced: {missing}")
    return frame


def fit_missing_value_fallback(
    frame: pd.DataFrame, train_mask: np.ndarray, feature_columns: list[str]
) -> pd.DataFrame:
    """Train-only median per (zone, month); global train median as the last resort."""
    work = frame.copy()
    work["_month"] = pd.DatetimeIndex(work[ORIGIN_COLUMN]).month
    train = work.loc[train_mask]
    numeric = [c for c in feature_columns if c != "zone"]
    by_cell = train.groupby([ZONE_COLUMN, "_month"])[numeric].median()
    global_median = train[numeric].median()
    filled = work.copy()
    for column in numeric:
        if not filled[column].isna().any():
            continue
        keys = list(zip(filled[ZONE_COLUMN], filled["_month"]))
        cell_values = pd.Series(
            [by_cell[column].get(k, np.nan) for k in keys], index=filled.index
        )
        filled[column] = filled[column].fillna(cell_values).fillna(global_median[column])
    return filled.drop(columns="_month")
