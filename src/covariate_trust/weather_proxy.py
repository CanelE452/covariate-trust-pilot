"""Real covariate-forecast reliability: the evaluation ratio and the decision-time proxy.

Naming matters here.  The synthetic studies had a ``lambda`` defined as a multiple of a
known AR(1) conditional standard deviation.  Nothing on real data reproduces that
construction, so the real-data quantity carries a different name and is never claimed to
be the same object.

``realized_weather_error_ratio``
    RMSE(primary ECMWF forecast, verification) / RMSE(168-hour seasonal-naive, verification)
    over the 24 valid hours of one origin.  Both terms need the *future* verification
    series, so it is knowable only after the fact.  It is used for scoring, as the
    calibration target on the training period, and for the oracle diagnostic - never as
    an input to a held-out decision.

``reported_reliability_ratio``
    what a decision maker could actually have computed at the decision origin, from
    (a) the run-to-run revision between the 00Z primary and the previous-day 12Z run,
    scaled by a *past* baseline error level, and (b) the mean realized ratio of recent
    **completed** origins.  The current origin's own realized ratio never enters.

The D7 band 0.75 / 1.25 was fixed on the synthetic scale and is applied unchanged to
this analogous real-world ratio.  That is a transfer test, not a claim that the two
quantities are mathematically identical or that 1.0 is a theoretical WQL boundary.

The 0.70 / 0.30 weighting and the isotonic calibrator are fixed on the training period
and frozen thereafter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-9
SEASONAL_LAG_HOURS = 168        # primary denominator
SEASONAL_LAG_DIAGNOSTIC = 24    # reported as a secondary diagnostic only


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def origin_weather_errors(primary: np.ndarray, revision: np.ndarray,
                          verification: np.ndarray, naive168: np.ndarray,
                          naive24: np.ndarray) -> dict:
    """All per-origin weather quantities, computed once."""
    e_external = rmse(primary, verification)
    e_baseline = rmse(naive168, verification)
    return {
        "e_external_rmse": e_external,
        "e_baseline_rmse_168": e_baseline,
        "e_baseline_rmse_24": rmse(naive24, verification),
        "realized_weather_error_ratio": e_external / (e_baseline + EPSILON) if np.isfinite(e_baseline) else np.nan,
        "revision_rms": rmse(primary, revision),
        "n_valid_forecast_hours": int(np.isfinite(primary).sum()),
        "n_valid_verification_hours": int(np.isfinite(verification).sum()),
        "n_valid_revision_hours": int(np.isfinite(revision).sum()),
    }


def model_cycle_label(run_or_origin, first_50r1_00z: str) -> str:
    """Secondary diagnostic label only; never used in a Gate H or Gate I criterion."""
    return ("post_50r1" if pd.Timestamp(run_or_origin).normalize()
            >= pd.Timestamp(first_50r1_00z) else "pre_50r1")


def add_decision_time_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Attach the two proxy inputs, using only strictly earlier origins.

    ``shift(1)`` before every rolling window is what keeps the current origin's own
    outcome out of its own decision.
    """
    out = []
    for zone, g in df.sort_values(["zone", "origin_utc"]).groupby("zone", sort=False):
        g = g.copy()
        past_baseline = g["e_baseline_rmse_168"].shift(1).rolling(window, min_periods=2).mean()
        g["past_baseline_scale"] = past_baseline
        g["revision_ratio"] = g["revision_rms"] / (past_baseline + EPSILON)
        g["recent_realized_ratio"] = (g["realized_weather_error_ratio"].shift(1)
                              .rolling(window, min_periods=2).mean())
        g["n_past_origins"] = np.arange(len(g))
        out.append(g)
    return pd.concat(out, ignore_index=True)


def raw_proxy_score(revision_ratio, recent_realized_ratio, revision_weight: float,
                    recent_weight: float):
    """0.70 * revision_ratio + 0.30 * recent_realized_ratio, weights fixed by config."""
    return revision_weight * np.asarray(revision_ratio, dtype=float) + \
        recent_weight * np.asarray(recent_realized_ratio, dtype=float)


class IsotonicCalibrator:
    """Monotonic raw_proxy -> realized-ratio mapping, fitted once on the training period.

    Implemented with scipy's pool-adjacent-violators so the study does not gain a new
    dependency; ``fitted_`` guards against accidental refitting.
    """

    def __init__(self):
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.fitted_ = False
        self.n_train_ = 0

    def fit(self, raw: np.ndarray, target: np.ndarray) -> "IsotonicCalibrator":
        if self.fitted_:
            raise RuntimeError("the calibrator is frozen after fitting; refitting is forbidden")
        raw = np.asarray(raw, dtype=float)
        target = np.asarray(target, dtype=float)
        mask = np.isfinite(raw) & np.isfinite(target)
        if mask.sum() < 10:
            raise ValueError(f"not enough training points to calibrate: {int(mask.sum())}")
        raw, target = raw[mask], target[mask]
        order = np.argsort(raw, kind="mergesort")
        raw, target = raw[order], target[order]

        from scipy.optimize import isotonic_regression
        fit = isotonic_regression(target, increasing=True)
        self.x_, self.y_ = raw, np.asarray(fit.x, dtype=float)
        self.fitted_ = True
        self.n_train_ = int(len(raw))
        return self

    def predict(self, raw) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("calibrator has not been fitted")
        raw = np.atleast_1d(np.asarray(raw, dtype=float))
        out = np.interp(raw, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])
        out[~np.isfinite(raw)] = np.nan
        return out

    def to_dict(self) -> dict:
        return {"fitted": self.fitted_, "n_train": self.n_train_,
                "x_min": float(self.x_.min()) if self.fitted_ else None,
                "x_max": float(self.x_.max()) if self.fitted_ else None,
                "y_min": float(self.y_.min()) if self.fitted_ else None,
                "y_max": float(self.y_.max()) if self.fitted_ else None,
                "n_knots": int(len(np.unique(self.y_))) if self.fitted_ else 0}


def calibration_diagnostics(reported: np.ndarray, realized: np.ndarray) -> dict:
    """Slope, MAE and Spearman - reported for validation and (after the fact) for test."""
    from scipy import stats
    reported = np.asarray(reported, dtype=float)
    realized = np.asarray(realized, dtype=float)
    mask = np.isfinite(reported) & np.isfinite(realized)
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "spearman": np.nan, "mae": np.nan, "slope": np.nan,
                "top_quartile_mean": np.nan, "bottom_quartile_mean": np.nan,
                "quartile_ratio": np.nan}
    r, t = reported[mask], realized[mask]
    slope = float(np.polyfit(r, t, 1)[0]) if len(np.unique(r)) > 1 else np.nan
    q_lo, q_hi = np.quantile(r, [0.25, 0.75])
    top = t[r >= q_hi]
    bot = t[r <= q_lo]
    return {
        "n": int(mask.sum()),
        "spearman": float(stats.spearmanr(r, t).statistic),
        "mae": float(np.mean(np.abs(r - t))),
        "slope": slope,
        "top_quartile_mean": float(top.mean()) if len(top) else np.nan,
        "bottom_quartile_mean": float(bot.mean()) if len(bot) else np.nan,
        "quartile_ratio": float(top.mean() / bot.mean()) if len(bot) and bot.mean() > 0 else np.nan,
    }


def split_periods(df: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    """Strict chronological split.  The test period is never touched by any fit."""
    p = cfg.periods
    t = pd.to_datetime(df["origin_utc"])
    return {
        "train": df[t <= pd.Timestamp(p.proxy_train_end) + pd.Timedelta(days=1)],
        "validation": df[(t > pd.Timestamp(p.proxy_train_end) + pd.Timedelta(days=1))
                         & (t <= pd.Timestamp(p.proxy_validation_end) + pd.Timedelta(days=1))],
        "test": df[(t >= pd.Timestamp(p.heldout_test_start))
                   & (t <= pd.Timestamp(p.heldout_test_end) + pd.Timedelta(days=1))],
    }


def coverage_status(splits: dict[str, pd.DataFrame], cfg) -> dict:
    need = cfg.gate_h.minimum_test_origins_per_zone
    per_zone = splits["test"].groupby("zone")["origin_utc"].nunique().to_dict() if len(
        splits["test"]) else {}
    ok_zones = [z for z, n in per_zone.items() if n >= need]
    if len(splits["train"]) < 10:
        status = "BLOCKED_COVERAGE"
    elif len(ok_zones) >= cfg.nyiso.minimum_zone_count:
        status = "PASS_COVERAGE"
    elif ok_zones:
        status = "PARTIAL_COVERAGE"
    else:
        status = "BLOCKED_COVERAGE"
    return {
        "status": status,
        "n_train_rows": int(len(splits["train"])),
        "n_validation_rows": int(len(splits["validation"])),
        "n_test_rows": int(len(splits["test"])),
        "test_origins_per_zone": {k: int(v) for k, v in per_zone.items()},
        "zones_meeting_minimum": ok_zones,
        "minimum_test_origins_per_zone": need,
    }
