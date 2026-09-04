"""Pure series-cluster bootstrap helpers."""

import numpy as np
import pandas as pd


ORIGINS_PER_SERIES = 6


def series_cluster_resample_indices(
    series_ids: np.ndarray, sampled_series_ids: np.ndarray
) -> np.ndarray:
    """Expand a sampled series draw into complete six-origin row clusters."""

    source_ids = np.asarray(series_ids, dtype=object)
    sampled_ids = np.asarray(sampled_series_ids, dtype=object)
    if source_ids.ndim != 1 or sampled_ids.ndim != 1:
        raise ValueError("series ID inputs must be one-dimensional")
    if source_ids.size == 0:
        raise ValueError("series_ids must not be empty")
    if sampled_ids.size == 0:
        raise ValueError("sampled_series_ids must not be empty")
    if bool(pd.isna(source_ids).any() or pd.isna(sampled_ids).any()):
        raise ValueError("series IDs must not be missing")

    rows_by_series: dict[object, list[int]] = {}
    try:
        for row_index, series_id in enumerate(source_ids):
            rows_by_series.setdefault(series_id, []).append(row_index)
    except TypeError as exc:
        raise TypeError("series IDs must be hashable") from exc

    incomplete = {
        series_id: len(indices)
        for series_id, indices in rows_by_series.items()
        if len(indices) != ORIGINS_PER_SERIES
    }
    if incomplete:
        raise ValueError(
            "every bootstrap series must contain exactly six origin rows; "
            f"found {incomplete}"
        )

    expanded: list[int] = []
    for sampled_id in sampled_ids:
        if sampled_id not in rows_by_series:
            raise ValueError(f"sampled unknown series ID: {sampled_id!r}")
        expanded.extend(rows_by_series[sampled_id])
    return np.asarray(expanded, dtype=np.int64)
