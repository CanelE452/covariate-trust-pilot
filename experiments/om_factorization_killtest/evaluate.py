"""Series-level metrics and paired bootstrap.

The aggregation unit is a SERIES, never a window row: windows from one series
overlap heavily and are not independent draws.  Comparisons between models are
paired on the same series and the same realization path.
"""

from __future__ import annotations

import numpy as np

from . import models, prereg


def truth_windows(block: dict[str, np.ndarray], windows) -> dict[str, np.ndarray]:
    """Line the stored truth up with the flattened (series, origin) window rows."""
    origins = windows.origins
    horizon = windows.target.shape[1]
    n_series = windows.n_series
    out: dict[str, np.ndarray] = {}
    for key in ("p_true", "mu_true", "mean_true"):
        values = np.asarray(block[key], dtype=np.float64)
        stacked = np.empty((n_series, origins.size, horizon))
        for w, origin in enumerate(origins):
            stacked[:, w] = values[:, origin:origin + horizon]
        out[key] = stacked.reshape(n_series * origins.size, horizon)
    out["series_index"] = np.repeat(np.arange(n_series), origins.size)
    return out


def _per_series(values: np.ndarray, mask: np.ndarray, series_index: np.ndarray,
                n_series: int) -> np.ndarray:
    """Masked sum of `values` grouped by series."""
    weighted = (values * mask).sum(axis=1)
    return np.bincount(series_index, weights=weighted, minlength=n_series)


def series_metrics(prediction: dict[str, np.ndarray], windows,
                   truth: dict[str, np.ndarray], dispersion=None) -> dict[str, np.ndarray]:
    """Per-series metric vectors; every one aggregates over that series' rows."""
    mask = windows.target_mask.astype(np.float64)
    target = windows.target.astype(np.float64)
    occurrence = windows.occurrence.astype(np.float64)
    series_index = truth["series_index"]
    n_series = windows.n_series
    counts = np.bincount(series_index, weights=mask.sum(axis=1), minlength=n_series)
    counts = np.maximum(counts, 1.0)

    mean_hat = prediction["mean_prediction"].astype(np.float64)
    out = {
        "rmse_mean_truth": np.sqrt(
            _per_series((mean_hat - truth["mean_true"]) ** 2, mask, series_index,
                        n_series) / counts),
        "realized_y_mse": _per_series((mean_hat - target) ** 2, mask, series_index,
                                      n_series) / counts,
        "realized_y_mae": _per_series(np.abs(mean_hat - target), mask, series_index,
                                      n_series) / counts,
    }

    if "p_prediction" in prediction:
        p_hat = np.clip(prediction["p_prediction"].astype(np.float64), 1e-7, 1 - 1e-7)
        mu_hat = prediction["mu_prediction"].astype(np.float64)
        out["occurrence_brier"] = _per_series((p_hat - occurrence) ** 2, mask,
                                              series_index, n_series) / counts
        out["rmse_p_truth"] = np.sqrt(
            _per_series((p_hat - truth["p_true"]) ** 2, mask, series_index,
                        n_series) / counts)
        out["rmse_mu_truth"] = np.sqrt(
            _per_series((mu_hat - truth["mu_true"]) ** 2, mask, series_index,
                        n_series) / counts)
        positive = mask * occurrence
        pos_counts = np.maximum(
            np.bincount(series_index, weights=positive.sum(axis=1),
                        minlength=n_series), 1.0)
        out["positive_only_magnitude_mse"] = _per_series(
            (mu_hat - target) ** 2, positive, series_index, n_series) / pos_counts

        if dispersion is not None:
            import torch
            r = torch.tensor(float(dispersion))
            nb_mean = torch.tensor(prediction.get("nb_mean", mu_hat))
            nll = models._ztnb_nll_terms(
                torch.tensor(np.maximum(target, 1.0)), nb_mean, r).numpy()
            out["ztnb_nll"] = _per_series(nll, positive, series_index,
                                          n_series) / pos_counts
    else:
        for key in ("occurrence_brier", "rmse_p_truth", "rmse_mu_truth",
                    "positive_only_magnitude_mse"):
            out[key] = np.full(n_series, np.nan)
    return out


# --------------------------------------------------------------------------
# Baselines evaluated on exactly the same windows
# --------------------------------------------------------------------------


def seasonal_naive_prediction(windows, period: int) -> dict[str, np.ndarray]:
    return {"mean_prediction": models.seasonal_naive(
        windows.history.astype(np.float64), windows.target.shape[1], period)}


def tsb_prediction(windows, alpha: float, beta: float) -> dict[str, np.ndarray]:
    p, mu = models.tsb_forecast(windows.history.astype(np.float64),
                                windows.target.shape[1], alpha, beta)
    return {"mean_prediction": p * mu, "p_prediction": p, "mu_prediction": mu}


def select_tsb(validation_windows) -> tuple[float, float, float]:
    """Grid search on the VALIDATION split only; never refit on test."""
    mask = validation_windows.target_mask.astype(np.float64)
    target = validation_windows.target.astype(np.float64)
    best = (None, None, float("inf"))
    for alpha in prereg.TSB_GRID["alpha"]:
        for beta in prereg.TSB_GRID["beta"]:
            pred = tsb_prediction(validation_windows, alpha, beta)
            mse = float((((pred["mean_prediction"] - target) ** 2) * mask).sum()
                        / max(mask.sum(), 1.0))
            if mse < best[2]:
                best = (alpha, beta, mse)
    return best


# --------------------------------------------------------------------------
# Paired bootstrap over series
# --------------------------------------------------------------------------


def paired_bootstrap_gain(model_values: np.ndarray, reference_values: np.ndarray,
                          draws: int = prereg.BOOTSTRAP_DRAWS,
                          level: float = prereg.BOOTSTRAP_LEVEL,
                          seed: int = prereg.BOOTSTRAP_SEED) -> dict:
    """Relative gain 1 - model/reference, resampling SERIES with replacement."""
    model_values = np.asarray(model_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    keep = np.isfinite(model_values) & np.isfinite(reference_values)
    model_values, reference_values = model_values[keep], reference_values[keep]
    if model_values.size == 0:
        return {"gain": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_series": 0}

    point = 1.0 - float(model_values.mean() / reference_values.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, model_values.size, size=(draws, model_values.size))
    boot = 1.0 - (model_values[idx].mean(axis=1) / reference_values[idx].mean(axis=1))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return {"gain": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_series": int(model_values.size),
            "ci_improving": bool(lo > 0.0), "ci_worsening": bool(hi < 0.0)}
