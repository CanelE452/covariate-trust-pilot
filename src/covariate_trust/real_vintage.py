"""Study 3 - the real forecast-vintage backtest itself.

One origin is one (zone, day) pair at 07:00 UTC.  ECMWF disseminates the 00 UTC HRES
hourly steps roughly between 05:45 and 06:12 UTC, so 07:00 is the first hour at which
the complete 24-hour path can be assumed in hand.  At that moment the decision maker
has: load up to 06:00 UTC, verified temperature up to 06:00 UTC, the 00Z run issued
seven hours earlier, and the previous day's 12Z run.  Nothing else.

M0  load history only                                          (sanity baseline)
M1  load + verified temperature history, future calendar
M2  M1 + *verified* future temperature      (oracle information bound, not a method)
M3  M1 + the 00Z ECMWF *forecast* of future temperature

M1, M2 and M3 receive an identical future calendar block, so the only thing that
differs between M1 and M3 is the forecasted temperature column.  M1 vs M3 is the
primary comparison; M2 shows what the covariate would be worth if it were perfect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import is_harm, mse, nmae, quantile_crossing_rate, wql
from .schemas import (COVARIATE_COLUMN, ID_COLUMN, M0, M1, M2, M3, ModelInputs,
                      SchemaError, TARGET_COLUMN, TIMESTAMP_COLUMN)
from .weather_proxy import (EPSILON, SEASONAL_LAG_DIAGNOSTIC, SEASONAL_LAG_HOURS,
                            origin_weather_errors)

LOCAL_TZ = "America/New_York"
CALENDAR_COLUMNS = ("local_hour_sin", "local_hour_cos", "day_of_week_sin",
                    "day_of_week_cos", "is_weekend")
TEMPERATURE_COLUMN = COVARIATE_COLUMN          # "x", the temperature covariate

D0 = "D0_always_no_future"
D1 = "D1_always_use_future"
D2 = "D2_oracle"
D3 = "D3_historical_utility"
D5 = "D5_current_proxy"
D7 = "D7_hybrid_override"
SELECTORS = (D0, D1, D2, D3, D5, D7)
PRIMARY_SELECTOR = D7
FIXED_POLICIES = (D0, D1)

HARM_THRESHOLD = 0.05


# --------------------------------------------------------------- inputs -----

def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar covariates on America/New_York local time.

    These are known for any future timestamp and are given to M1, M2 and M3 alike.
    Without them, only M3 would carry a future frame, and part of an M1-to-M3
    difference could come from the model simply seeing the forecast window's
    time-of-day and day-of-week pattern rather than from the weather content.

    Timestamps handed to the model stay in UTC; only the feature values are derived
    from the DST-aware local clock.
    """
    idx = pd.DatetimeIndex(index)
    local = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).tz_convert(LOCAL_TZ)
    h = local.hour.to_numpy(dtype=float)
    dow = local.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame({
        "local_hour_sin": np.sin(2 * np.pi * h / 24.0),
        "local_hour_cos": np.cos(2 * np.pi * h / 24.0),
        "day_of_week_sin": np.sin(2 * np.pi * dow / 7.0),
        "day_of_week_cos": np.cos(2 * np.pi * dow / 7.0),
        "is_weekend": (dow >= 5).astype(float),
    })


def build_real_inputs(method: str, item_id: str, ctx_index: pd.DatetimeIndex,
                      load: np.ndarray, temperature: np.ndarray,
                      fut_index: pd.DatetimeIndex | None = None,
                      x_future: np.ndarray | None = None) -> ModelInputs:
    """Build one Chronos task from real, timestamp-indexed data.

    context  M0: target only.  M1/M2/M3: target + verified temperature + calendar.
    future   M0: none.  M1: calendar only.  M2: calendar + verified temperature.
             M3: calendar + forecasted temperature.
    """
    ctx = pd.DataFrame({
        ID_COLUMN: item_id,
        TIMESTAMP_COLUMN: pd.DatetimeIndex(ctx_index),
        TARGET_COLUMN: np.asarray(load, dtype=float),
    })
    if method != M0:
        ctx[TEMPERATURE_COLUMN] = np.asarray(temperature, dtype=float)
        cal = calendar_features(ctx_index)
        for c in CALENDAR_COLUMNS:
            ctx[c] = cal[c].to_numpy()

    future = None
    if method == M0:
        # M0 still needs the horizon length (Chronos' prediction_length) but gets no
        # future frame at all; fut_index is read for its length only.
        if x_future is not None:
            raise SchemaError("M0 must not be given a future covariate path")
        if fut_index is None:
            raise SchemaError("M0 needs a future index to know the horizon length")
    else:
        if fut_index is None:
            raise SchemaError(f"{method} requires a future index for the calendar block")
        future = pd.DataFrame({
            ID_COLUMN: item_id,
            TIMESTAMP_COLUMN: pd.DatetimeIndex(fut_index),
        })
        fcal = calendar_features(fut_index)
        for c in CALENDAR_COLUMNS:
            future[c] = fcal[c].to_numpy()
        if method in (M2, M3):
            if x_future is None:
                raise SchemaError(f"{method} requires a future temperature path")
            if len(x_future) != len(fut_index):
                raise SchemaError(f"{method}: future temperature length mismatch")
            future[TEMPERATURE_COLUMN] = np.asarray(x_future, dtype=float)
        elif x_future is not None:
            raise SchemaError("M1 must not be given a future temperature path")

    inputs = ModelInputs(method, ctx, future, item_id, int(len(ctx)),
                         0 if fut_index is None else int(len(fut_index)))
    assert_real_task_invariants(inputs)
    return inputs


def assert_real_task_invariants(inputs: ModelInputs) -> None:
    """Schema rules for the real-data tasks (the synthetic ones keep their own)."""
    ctx, fut, method = inputs.context_df, inputs.future_df, inputs.method
    base = [ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN]
    expected_ctx = base if method == M0 else base + [TEMPERATURE_COLUMN] + list(CALENDAR_COLUMNS)
    if list(ctx.columns) != expected_ctx:
        raise SchemaError(f"{method}: context columns {list(ctx.columns)} != {expected_ctx}")
    if not np.isfinite(ctx[TARGET_COLUMN].to_numpy()).all():
        raise SchemaError(f"{method}: non-finite target in context")
    ts = pd.DatetimeIndex(ctx[TIMESTAMP_COLUMN])
    if not ts.is_monotonic_increasing or not ts.is_unique:
        raise SchemaError(f"{method}: context timestamps not strictly increasing")
    if len(ts) > 1 and len(set(np.diff(ts.asi8))) != 1:
        raise SchemaError(f"{method}: context timestamps not regularly spaced")

    if method == M0:
        if fut is not None:
            raise SchemaError("M0 must not receive a future frame")
        return
    if fut is None:
        raise SchemaError(f"{method}: every non-M0 method carries a future calendar block")
    if TARGET_COLUMN in fut.columns:
        raise SchemaError(f"{method}: future frame must never contain the target")
    expected_fut = [ID_COLUMN, TIMESTAMP_COLUMN] + list(CALENDAR_COLUMNS)
    if method in (M2, M3):
        expected_fut = expected_fut + [TEMPERATURE_COLUMN]
    if list(fut.columns) != expected_fut:
        raise SchemaError(f"{method}: future columns {list(fut.columns)} != {expected_fut}")
    if method == M1 and TEMPERATURE_COLUMN in fut.columns:
        raise SchemaError("M1 must not carry a future temperature column")
    fts = pd.DatetimeIndex(fut[TIMESTAMP_COLUMN])
    step = ts[1] - ts[0]
    if fts[0] - ts[-1] != step:
        raise SchemaError(f"{method}: future frame does not start one step after the context")


def assert_fair_comparison(in1: ModelInputs, in2: ModelInputs, in3: ModelInputs) -> None:
    """M1, M2 and M3 may differ only in the future temperature column."""
    if not in1.context_df.equals(in2.context_df) or not in1.context_df.equals(in3.context_df):
        raise SchemaError("M1/M2/M3 contexts are not identical")
    t1 = pd.DatetimeIndex(in1.future_df[TIMESTAMP_COLUMN])
    for other in (in2, in3):
        if not t1.equals(pd.DatetimeIndex(other.future_df[TIMESTAMP_COLUMN])):
            raise SchemaError("future timestamps differ across methods")
    cal1 = in1.future_df[list(CALENDAR_COLUMNS)]
    for other in (in2, in3):
        if not cal1.equals(other.future_df[list(CALENDAR_COLUMNS)]):
            raise SchemaError("future calendar blocks differ across methods")
    if TEMPERATURE_COLUMN in in1.future_df.columns:
        raise SchemaError("M1 carries a future temperature column")
    for other in (in2, in3):
        if TEMPERATURE_COLUMN not in other.future_df.columns:
            raise SchemaError(f"{other.method} lacks its future temperature column")
    only_temp = (in2.future_df.drop(columns=[TEMPERATURE_COLUMN])
                 .equals(in3.future_df.drop(columns=[TEMPERATURE_COLUMN])))
    if not only_temp:
        raise SchemaError("M2 and M3 differ in something other than the temperature column")


def assemble_origins(load: pd.DataFrame, verification: pd.DataFrame, runs: pd.DataFrame,
                     cfg, log=lambda *_: None) -> tuple[pd.DataFrame, dict]:
    """Every (zone, origin) that has a complete context, horizon, forecast and revision."""
    exp = cfg.experiment
    H, C = exp.prediction_length, exp.context_length
    primary = runs[runs["run_kind"] == "primary"]
    revision = runs[runs["run_kind"] == "revision"]

    rows, rejected = [], {"context_incomplete": 0, "target_incomplete": 0,
                          "verification_incomplete": 0, "forecast_incomplete": 0,
                          "revision_incomplete": 0}
    for zone, lz in load.groupby("zone"):
        lz = lz.sort_values("timestamp_utc").set_index("timestamp_utc")["load_mw"]
        vz = (verification[verification["zone"] == zone]
              .sort_values("valid_time_utc").set_index("valid_time_utc")["temperature_verified"])
        pz = primary[primary["zone"] == zone]
        rz = revision[revision["zone"] == zone]
        rz_by_origin = {o: g.sort_values("valid_time_utc") for o, g in rz.groupby("origin_utc")}

        for origin, g in pz.groupby("origin_utc"):
            origin = pd.Timestamp(origin)
            fut_index = pd.date_range(origin, periods=H, freq="h")
            ctx_index = pd.date_range(end=origin - pd.Timedelta(hours=1), periods=C, freq="h")

            y_ctx = lz.reindex(ctx_index).to_numpy(dtype=float)
            x_ctx = vz.reindex(ctx_index).to_numpy(dtype=float)
            y_fut = lz.reindex(fut_index).to_numpy(dtype=float)
            x_fut_true = vz.reindex(fut_index).to_numpy(dtype=float)
            g = g.sort_values("valid_time_utc")
            x_fut_fc = (g.set_index("valid_time_utc")["temperature_forecast"]
                        .reindex(fut_index).to_numpy(dtype=float))
            rg = rz_by_origin.get(origin)
            x_fut_rev = (rg.set_index("valid_time_utc")["temperature_forecast"]
                         .reindex(fut_index).to_numpy(dtype=float)
                         if rg is not None else np.full(H, np.nan))

            if not np.isfinite(y_ctx).all() or not np.isfinite(x_ctx).all():
                rejected["context_incomplete"] += 1
                continue
            if not np.isfinite(y_fut).all():
                rejected["target_incomplete"] += 1
                continue
            if not np.isfinite(x_fut_true).all():
                rejected["verification_incomplete"] += 1
                continue
            if not np.isfinite(x_fut_fc).all():
                rejected["forecast_incomplete"] += 1
                continue
            if not np.isfinite(x_fut_rev).all():
                rejected["revision_incomplete"] += 1
                continue

            naive168 = vz.reindex(fut_index - pd.Timedelta(hours=SEASONAL_LAG_HOURS)).to_numpy(float)
            naive24 = vz.reindex(fut_index - pd.Timedelta(hours=SEASONAL_LAG_DIAGNOSTIC)).to_numpy(float)
            errs = origin_weather_errors(x_fut_fc, x_fut_rev, x_fut_true, naive168, naive24)

            from .weather_proxy import model_cycle_label
            rows.append({
                "zone": zone, "origin_utc": origin,
                "weather_model_cycle": model_cycle_label(
                    origin, cfg.weather.model_cycle_50r1_first_00z_run),
                "iso_year": int(origin.isocalendar().year),
                "iso_week": int(origin.isocalendar().week),
                "calendar_week": f"{origin.isocalendar().year}-W{origin.isocalendar().week:02d}",
                "month": int(origin.month),
                "season": _season(origin),
                "ctx_start": ctx_index[0], "ctx_end": ctx_index[-1],
                "fut_start": fut_index[0], "fut_end": fut_index[-1],
                "mean_load_mw": float(np.mean(y_fut)),
                "mean_temp_verified": float(np.mean(x_fut_true)),
                "max_temp_verified": float(np.max(x_fut_true)),
                "min_temp_verified": float(np.min(x_fut_true)),
                **errs,
                "_y_ctx": y_ctx, "_x_ctx": x_ctx, "_y_fut": y_fut,
                "_x_fut_true": x_fut_true, "_x_fut_fc": x_fut_fc,
                "_ctx_index": ctx_index, "_fut_index": fut_index,
            })
    panel = pd.DataFrame(rows)
    report = {"n_origins": int(len(panel)), "rejected": rejected,
              "zones": sorted(panel["zone"].unique().tolist()) if len(panel) else []}
    log(f"assembled {len(panel)} usable (zone, origin) cells; rejected {rejected}")
    return panel, report


def _season(ts: pd.Timestamp) -> str:
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[ts.month]


# ------------------------------------------------------------ forecasting ---

def run_forecasts(panel: pd.DataFrame, cfg, predict_fn, log=lambda *_: None) -> pd.DataFrame:
    """Score M0/M1/M2/M3 at every origin.  ``predict_fn(inputs, meta) -> (H, Q)``."""
    q_levels = cfg.experiment.quantile_levels
    out, n = [], 0
    for _, r in panel.iterrows():
        item = f"{r['zone']}_{pd.Timestamp(r['origin_utc']).strftime('%Y%m%d')}"
        ctx_index, fut_index = r["_ctx_index"], r["_fut_index"]
        y_true = r["_y_fut"]

        in0 = build_real_inputs(M0, item, ctx_index, r["_y_ctx"], r["_x_ctx"], fut_index)
        in1 = build_real_inputs(M1, item, ctx_index, r["_y_ctx"], r["_x_ctx"], fut_index)
        in2 = build_real_inputs(M2, item, ctx_index, r["_y_ctx"], r["_x_ctx"],
                                fut_index, r["_x_fut_true"])
        in3 = build_real_inputs(M3, item, ctx_index, r["_y_ctx"], r["_x_ctx"],
                                fut_index, r["_x_fut_fc"])
        assert_fair_comparison(in1, in2, in3)

        meta = {"zone": r["zone"], "origin_utc": str(r["origin_utc"])}
        q0 = predict_fn(in0, {**meta, "method": M0})
        q1 = predict_fn(in1, {**meta, "method": M1})
        q2 = predict_fn(in2, {**meta, "method": M2})
        q3 = predict_fn(in3, {**meta, "method": M3})
        w0, w1, w2, w3 = (wql(y_true, q, q_levels) for q in (q0, q1, q2, q3))

        out.append({
            "zone": r["zone"], "origin_utc": r["origin_utc"],
            "calendar_week": r["calendar_week"], "season": r["season"],
            "weather_model_cycle": r.get("weather_model_cycle", "unknown"),
            "wql_m0": w0, "wql_m1": w1, "wql_m2": w2, "wql_m3": w3,
            "nmae_m1": nmae(y_true, q1, q_levels), "nmae_m3": nmae(y_true, q3, q_levels),
            "mse_m1": mse(y_true, q1, q_levels), "mse_m3": mse(y_true, q3, q_levels),
            "crossing_m3": quantile_crossing_rate(q3),
            "wql_oracle": min(w1, w3),
            "m3_is_better": int(w3 < w1),
            "harm_m3": int(is_harm(w1, w3, HARM_THRESHOLD)),
            "v_future": w1 - w3,
            "v_oracle": w1 - w2,
            "realized_weather_error_ratio": r["realized_weather_error_ratio"],
            "mean_load_mw": r["mean_load_mw"],
            "max_temp_verified": r["max_temp_verified"],
            "min_temp_verified": r["min_temp_verified"],
        })
        n += 1
        if n % 200 == 0:
            log(f"  forecasts {n}/{len(panel)} origins")
    return pd.DataFrame(out)


# -------------------------------------------------------------- selectors ---

def apply_selectors(tasks: pd.DataFrame, features: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach every policy's choice.  Only D2 may look at the current outcome."""
    keys = ["zone", "origin_utc"]
    need = ["reported_reliability_ratio", "hist_wql_m1", "hist_wql_m3", "n_hist_origins", "realized_weather_error_ratio",
            "recent_realized_ratio"]
    df = tasks.merge(features[keys + [c for c in need if c in features.columns]],
                     on=keys, how="left", suffixes=("", "_feat"))

    p = cfg.proxy
    hu = cfg.historical_utility
    enough_history = df["n_hist_origins"] >= hu.minimum_origins
    hist_supports_m3 = enough_history & (df["hist_wql_m3"] < df["hist_wql_m1"])
    proxy_ok = df["reported_reliability_ratio"] < p.d5_threshold

    choice = {
        D0: np.full(len(df), M1),
        D1: np.full(len(df), M3),
        D2: np.where(df["wql_m3"] < df["wql_m1"], M3, M1),
        D3: np.where(hist_supports_m3, M3, M1),
        D5: np.where(proxy_ok.fillna(False), M3, M1),
        D7: np.where(df["reported_reliability_ratio"] < p.lower_threshold, M3,
                     np.where(df["reported_reliability_ratio"] > p.upper_threshold, M1,
                              np.where(hist_supports_m3, M3, M1))),
    }
    frames = []
    for sel in SELECTORS:
        d = df.copy()
        d["selector"] = sel
        d["choice"] = choice[sel]
        d["wql_selected"] = np.where(d["choice"] == M3, d["wql_m3"], d["wql_m1"])
        d["nmae_selected"] = np.where(d["choice"] == M3, d["nmae_m3"], d["nmae_m1"])
        d["regret"] = d["wql_selected"] - d["wql_oracle"]
        d["chose_m3"] = (d["choice"] == M3).astype(int)
        d["harm"] = [int(is_harm(b, s, HARM_THRESHOLD))
                     for s, b in zip(d["wql_selected"], d["wql_m1"])]
        d["false_use"] = ((d["m3_is_better"] == 0) & (d["choice"] == M3)).astype(int)
        d["false_reject"] = ((d["m3_is_better"] == 1) & (d["choice"] == M1)).astype(int)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def historical_utility_features(tasks: pd.DataFrame, cfg) -> pd.DataFrame:
    """Rolling mean WQL of M1 and M3 over the previous completed origins only."""
    w = cfg.historical_utility.window_origins
    out = []
    for zone, g in tasks.sort_values(["zone", "origin_utc"]).groupby("zone", sort=False):
        g = g.copy()
        g["hist_wql_m1"] = g["wql_m1"].shift(1).rolling(w, min_periods=1).mean()
        g["hist_wql_m3"] = g["wql_m3"].shift(1).rolling(w, min_periods=1).mean()
        g["n_hist_origins"] = np.arange(len(g))
        out.append(g[["zone", "origin_utc", "hist_wql_m1", "hist_wql_m3", "n_hist_origins"]])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------- summaries --

def false_rates(d: pd.DataFrame) -> dict:
    n1 = int((d["m3_is_better"] == 0).sum())
    n3 = int((d["m3_is_better"] == 1).sum())
    return {"false_use_rate": float(d["false_use"].sum() / n1) if n1 else float("nan"),
            "false_reject_rate": float(d["false_reject"].sum() / n3) if n3 else float("nan"),
            "n_m1_better": n1, "n_m3_better": n3}


def summarize(g: pd.DataFrame) -> dict:
    m1, m3 = float(g["wql_m1"].mean()), float(g["wql_m3"].mean())
    best_fixed = min(m1, m3)
    oracle = float(g["wql_oracle"].mean())
    sel = float(g["wql_selected"].mean())
    gap = best_fixed - oracle
    return {
        "n_origins": int(len(g)), "n_zones": int(g["zone"].nunique()),
        "n_weeks": int(g["calendar_week"].nunique()),
        "mean_wql": sel, "median_nmae": float(g["nmae_selected"].median()),
        "mean_regret": float(g["regret"].mean()),
        "m3_choice_rate": float(g["chose_m3"].mean()),
        "harm_rate": float(g["harm"].mean()),
        "mean_wql_m1": m1, "mean_wql_m3": m3, "mean_wql_oracle": oracle,
        "best_fixed_mean": best_fixed,
        "relative_improvement_over_best_fixed": (best_fixed - sel) / best_fixed,
        "oracle_gap_recovery": (best_fixed - sel) / gap if gap > 0 else float("nan"),
        "win_rate_vs_best_fixed": float((g["wql_selected"] < (
            g["wql_m1"] if m1 <= m3 else g["wql_m3"])).mean()),
        **false_rates(g),
    }


def selector_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"selector": sel, **summarize(g)}
                         for sel, g in decisions.groupby("selector", sort=True)])


def grouped_summary(decisions: pd.DataFrame, by: str) -> pd.DataFrame:
    return pd.DataFrame([{by: k, "selector": sel, **summarize(g)}
                         for (k, sel), g in decisions.groupby([by, "selector"], sort=True)])


def week_cluster_bootstrap(decisions: pd.DataFrame, baseline_col: str, treatment_col: str,
                           cfg, seed_parts: tuple) -> dict:
    """Cluster bootstrap over ISO calendar weeks: adjacent daily origins are not independent."""
    from .seeds import make_rng
    d = decisions
    weeks = d["calendar_week"].to_numpy()
    base = d[baseline_col].to_numpy(dtype=float)
    treat = d[treatment_col].to_numpy(dtype=float)
    uniq, inverse = np.unique(weeks, return_inverse=True)
    n_w = len(uniq)
    counts = np.bincount(inverse, minlength=n_w).astype(float)
    s_base = np.bincount(inverse, weights=base, minlength=n_w)
    s_treat = np.bincount(inverse, weights=treat, minlength=n_w)

    rng = make_rng(*seed_parts, cfg.bootstrap.n_resamples, n_w)
    idx = rng.integers(0, n_w, size=(cfg.bootstrap.n_resamples, n_w))
    bc = counts[idx].sum(axis=1)
    bb = s_base[idx].sum(axis=1) / bc
    bt = s_treat[idx].sum(axis=1) / bc
    diff = bb - bt
    rel = diff / bb
    alpha = 1 - cfg.bootstrap.confidence_level
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    unit_mean = (s_base - s_treat) / counts
    return {
        "baseline": baseline_col, "treatment": treatment_col,
        "mean_baseline": float(base.mean()), "mean_treatment": float(treat.mean()),
        "mean_diff": float(base.mean() - treat.mean()),
        "median_diff": float(np.median(base - treat)),
        "relative_improvement": float((base.mean() - treat.mean()) / base.mean()),
        "ci_low": float(np.percentile(diff, lo)), "ci_high": float(np.percentile(diff, hi)),
        "rel_ci_low": float(np.percentile(rel, lo)), "rel_ci_high": float(np.percentile(rel, hi)),
        "monte_carlo_se": float(unit_mean.std(ddof=1) / np.sqrt(n_w)) if n_w > 1 else float("nan"),
        "n_weeks": int(n_w), "n_origins": int(len(d)), "n_zones": int(d["zone"].nunique()),
        "win_rate": float((treat < base).mean()),
        "ci_favours_treatment": bool(np.percentile(diff, lo) > 0),
        "ci_favours_baseline": bool(np.percentile(diff, hi) < 0),
    }


def reliability_shift_events(features: pd.DataFrame, cfg) -> pd.DataFrame:
    """Post-hoc event labels from realized_weather_error_ratio.  Never used in any decision."""
    lo, hi = cfg.proxy.lower_threshold, cfg.proxy.upper_threshold
    f = features.copy()
    f["worsening_event"] = ((f["recent_realized_ratio"] < lo) & (f["realized_weather_error_ratio"] > hi)).astype(int)
    f["improvement_event"] = ((f["recent_realized_ratio"] > hi) & (f["realized_weather_error_ratio"] < lo)).astype(int)
    return f[["zone", "origin_utc", "worsening_event", "improvement_event",
              "recent_realized_ratio", "realized_weather_error_ratio"]]
