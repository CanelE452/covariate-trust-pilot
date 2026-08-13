"""Gate inputs, every one of them computed from history strictly before its origin.

The static descriptors the H1/H2 work used were computed once on the whole train
split.  Copying those onto an OOF origin a year earlier would hand the gate
information from the future, so nothing here reuses them.  Each origin gets its
own descriptors, recomputed on `y[start:origin]` by the repository's own
`screen.describe_series`, which is the same function that produced the frozen
descriptors -- only the window moves.

Two feature groups, matching the design:

    structure     what the series looked like up to this origin
    expert state  what Point and Hurdle predicted for this origin

Expert-state features read `km_train.predict`'s returned fields only.  They
describe the forecasts, never their error, so they carry no target information.

Undefined descriptors stay NaN here.  They are filled later from the gate
TRAIN split's median and always paired with an explicit missing flag; filling
with zero would tell the gate that an undefined autocorrelation is a zero one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..external_validity_screen import screen

#: Recent-window length for the short-memory features. One horizon, so the gate
#: sees the same span it is about to forecast.
RECENT = 28

STRUCTURE_COLUMNS = [
    "f_log_ADI", "f_CV2_positive", "f_rho_interval", "f_abs_rho_interval",
    "f_rho_magnitude", "f_zero_ratio", "f_occurrence_acf1",
    "f_age_since_positive", "f_recent_occurrence_rate",
    "f_recent_positive_mean", "f_log_history_scale", "f_log_n_positive",
]
#: Anything carrying the series' size. G-NOSCALE drops these to test whether the
#: gate is routing on structure or just on how big the series is.
SCALE_COLUMNS = ["f_log_history_scale", "f_recent_positive_mean"]

EXPERT_PREFIX = "xs_"


def as_of_origin(y_row: np.ndarray, start: int, origin: int) -> dict:
    """Descriptors from y[start:origin] only.

    `origin` is the first forecast day, so the slice stops one day before it and
    the target window is never touched.
    """
    entry = screen.describe_series(y_row, start, origin)
    segment = y_row[start:origin]
    events = np.flatnonzero(segment > 0)
    recent = segment[-RECENT:] if segment.size >= RECENT else segment
    recent_positive = recent[recent > 0]
    scale = float(segment.mean()) if segment.size else np.nan

    out = {
        "f_log_ADI": np.log(entry["ADI_train"]) if np.isfinite(entry["ADI_train"]) else np.nan,
        "f_CV2_positive": entry["CV2_positive_train"],
        "f_rho_interval": entry["rho_interval_train"],
        "f_abs_rho_interval": entry["rho_interval_abs_train"],
        "f_rho_magnitude": entry["rho_magnitude_train"],
        "f_zero_ratio": entry["zero_ratio_train"],
        "f_occurrence_acf1": entry["occurrence_binary_acf1_train"],
        # Days since the last positive demand; a full-length window means the
        # series has never sold, which is information rather than a missing value.
        "f_age_since_positive": float(segment.size - events[-1]) if events.size else float(segment.size),
        "f_recent_occurrence_rate": float((recent > 0).mean()) if recent.size else np.nan,
        "f_recent_positive_mean": float(recent_positive.mean()) if recent_positive.size else 0.0,
        "f_log_history_scale": np.log(max(scale, 1e-6)) if np.isfinite(scale) else np.nan,
        "f_log_n_positive": np.log1p(entry["n_positive_train"]),
    }
    return out


def structure_table(data: dict, origins_by_row: pd.DataFrame) -> pd.DataFrame:
    """One row per (series_id, origin), descriptors as of that origin.

    `origins_by_row` supplies the pairs to compute; the same function serves the
    OOF blocks and the test block, so a feature can never be defined one way for
    training and another for evaluation.
    """
    index = pd.Index(data["series_id"]).astype(str)
    lookup = {sid: i for i, sid in enumerate(index)}
    rows = []
    for sid, origin in zip(origins_by_row["series_id"].astype(str),
                           origins_by_row["origin"].astype(int)):
        i = lookup[sid]
        rows.append(as_of_origin(data["y"][i], int(data["available_from"][i]), origin))
    frame = pd.DataFrame(rows)
    frame.insert(0, "origin", origins_by_row["origin"].to_numpy())
    frame.insert(0, "series_id", origins_by_row["series_id"].to_numpy())
    return frame


def add_missing_flags(frame: pd.DataFrame, columns) -> pd.DataFrame:
    for column in columns:
        frame[f"{column}__missing"] = (~np.isfinite(frame[column].to_numpy(float))).astype(float)
    return frame


def fit_imputer(frame: pd.DataFrame, columns) -> dict:
    """Medians from the gate TRAIN rows only; reused verbatim everywhere else."""
    return {c: float(np.nanmedian(frame[c].to_numpy(float))) for c in columns}


def apply_imputer(frame: pd.DataFrame, medians: dict) -> pd.DataFrame:
    frame = frame.copy()
    for column, value in medians.items():
        values = frame[column].to_numpy(float)
        frame[column] = np.where(np.isfinite(values), values, value)
    return frame


def feature_columns(variant: str, available: list[str]) -> list[str]:
    """The three pre-registered variants, resolved against what actually exists."""
    structure = [c for c in available
                 if c in STRUCTURE_COLUMNS or c.endswith("__missing")]
    expert = [c for c in available if c.startswith(EXPERT_PREFIX)]
    if variant == "G-STRUCT":
        return structure
    if variant == "G-FULL":
        return structure + expert
    if variant == "G-NOSCALE":
        dropped = set(SCALE_COLUMNS) | {f"{c}__missing" for c in SCALE_COLUMNS}
        return [c for c in structure + expert if c not in dropped]
    raise ValueError(variant)
