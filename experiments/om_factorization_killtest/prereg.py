"""Frozen preregistration for the occurrence-magnitude factorization kill test.

Everything here is fixed BEFORE any model is trained.  ``prereg_hash()`` is a
digest of this record; it is written into every artifact so a later reader can
confirm the criteria were not edited after the numbers appeared.
"""

from __future__ import annotations

import hashlib
import json

RESEARCH_QUESTION = (
    "In intermittent demand, does supervising occurrence probability p and "
    "positive magnitude mean mu separately recover the true conditional mean "
    "p*mu better than predicting y directly with MSE on the same DLinear "
    "backbone -- and does any such gain survive a TSB baseline?"
)

CELLS = ("C1_adi2_magOFF", "C2_adi2_magON", "C3_adi6_magOFF", "C4_adi6_magON")

MODELS = ("B0_seasonal_naive", "B1_TSB", "M0_point_mse",
          "M0PM_point_mse_param_matched", "M1_factorized_mean", "M2_hurdle_ztnb")

PRIMARY_METRIC = "rmse_mean_truth"
PRIMARY_METRIC_DEFINITION = (
    "RMSE(mean_hat, mean_true) over all test (origin, h) points of a series, "
    "where mean_true = p_true * mu_true; point models use mean_hat = y_hat and "
    "factorized models and TSB use mean_hat = p_hat * mu_hat"
)
SECONDARY_METRICS = (
    "realized_y_mse", "realized_y_mae", "occurrence_brier", "rmse_p_truth",
    "rmse_mu_truth", "positive_only_magnitude_mse", "ztnb_nll", "train_seconds",
    "n_parameters",
)

BOOTSTRAP_UNIT = "series"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_LEVEL = 0.95
BOOTSTRAP_PAIRED = True
BOOTSTRAP_SEED = 20260802

DATA = {
    "n_series_per_cell_per_seed": 40,
    "data_seeds": (0, 1),
    "model_seeds": (0, 1),
    "length": 512,
    "lookback": 96,
    "horizon": 24,
    "train_end": 358,
    "val_end": 409,
    "test_origin_stride": 24,
    "period": 24,
}

TRAINING = {
    "max_epochs": 30,
    "patience": 5,
    "batch_size": 256,
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "checkpoint_rule": "validation MSE of the final mean prediction, identical "
                       "for every DLinear variant",
    "lambda_factorized": 1.0,
    "hyperparameter_rule": "no per-cell tuning; one setting for every cell and "
                           "every model; TSB alpha/beta chosen on a fixed grid "
                           "using the validation split only",
}

TSB_GRID = {"alpha": (0.05, 0.1, 0.2, 0.3), "beta": (0.05, 0.1, 0.2, 0.3)}

DGP = {
    "note": "exploratory kill-test DGP, not the paper DGP",
    "target_eta_occ": 0.15,
    "target_eta_size": 0.30,
    "target_positive_mean": 10.0,
    "target_positive_cv2": 0.70,
    "occurrence": "logit(p_t) = alpha + A_occ sin(2 pi t / 24), no IID cell",
    "magnitude_off": "mu_t = 10",
    "magnitude_on": "log(mu_t) = beta + A_mag sin(2 pi t / 24 + pi/2)",
    "phase": "fixed, no hidden phase, regime or OOD shift",
}

DATA_GATES = {
    "D0": "every model consumes the same dataset content hash",
    "D1": "occurrence varies in time in every cell",
    "D2": "mu_true is constant in the magnitude-OFF cells",
    "D3": "mu_true is periodic in the magnitude-ON cells",
    "D4": "long-run positive mean and CV^2 agree between OFF and ON within tolerance",
    "D5": "realized ADI matches the configured 2 and 6",
    "D6": "mean_true equals p_true * mu_true exactly",
    "D7": "no test realization entered any parameter calibration",
    "D8": "mean_true beats a constant unconditional-mean predictor by >= 5% "
          "in realized MSE",
}

KILL_GATES = {
    "K1": {
        "name": "factorization headroom",
        "rule": "M1 beats M0 by >= 3% in at least 2 of 4 cells, at least one of "
                "those cells has a paired CI on the improving side, and the "
                "pooled gain is >= 2%",
        "min_cells_3pct": 2,
        "min_cells_ci": 1,
        "min_pooled_gain": 0.02,
        "cell_gain_threshold": 0.03,
    },
    "K2": {
        "name": "strong simple baseline",
        "rule": "M1 or M2 beats TSB by >= 2% in at least one cell without a "
                "paired CI indicating serious deterioration",
        "cell_gain_threshold": 0.02,
    },
    "K3": {"name": "decomposition interpretation",
           "labels": ("DECOMPOSITION_SIGNAL", "DISTRIBUTION_ALIGNMENT_ONLY",
                      "MECHANISM_ONLY_NOT_METHOD", "NO_FACTORIZATION_HEADROOM")},
    "K4": {"name": "realized-value damage",
           "rule": "WARN if realized Y MSE worsens by more than 2% pooled",
           "threshold": 0.02},
}

VERDICTS = ("GO_PILOT", "WARN_MECHANISM_ONLY", "NO_GO", "BLOCKED")

PARAMETER_MATCH_RULE = (
    "if a factorized model carries more than 5% more trainable parameters than "
    "M0, a parameter-matched point control M0-PM is added: same parameter "
    "count, single y_hat output, MSE loss, no extra input or supervision"
)

STOP_RULE = (
    "if this smoke shows no signal: no scenario expansion, no gap/period loss, "
    "no Pilot, no Full, no threshold change, no extra model search"
)


def record() -> dict:
    return {
        "research_question": RESEARCH_QUESTION,
        "cells": list(CELLS),
        "models": list(MODELS),
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_definition": PRIMARY_METRIC_DEFINITION,
        "secondary_metrics": list(SECONDARY_METRICS),
        "bootstrap": {"unit": BOOTSTRAP_UNIT, "draws": BOOTSTRAP_DRAWS,
                      "level": BOOTSTRAP_LEVEL, "paired": BOOTSTRAP_PAIRED,
                      "seed": BOOTSTRAP_SEED},
        "data": {k: list(v) if isinstance(v, tuple) else v for k, v in DATA.items()},
        "training": TRAINING,
        "tsb_grid": {k: list(v) for k, v in TSB_GRID.items()},
        "dgp": DGP,
        "data_gates": DATA_GATES,
        "kill_gates": KILL_GATES,
        "verdicts": list(VERDICTS),
        "parameter_match_rule": PARAMETER_MATCH_RULE,
        "stop_rule": STOP_RULE,
    }


def canonical_json() -> str:
    return json.dumps(record(), indent=2, sort_keys=True)


def prereg_hash() -> str:
    return hashlib.sha256(canonical_json().encode("utf-8")).hexdigest()
