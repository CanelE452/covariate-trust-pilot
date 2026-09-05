"""Track X pre-analysis spec: corrupted channels, evaluation origins, thresholds.

Everything here is a function of train statistics and split geometry only. It is
frozen to results/ts_idea_tournament_v1/track_x/pre_analysis_spec.json before the
first spillover number is computed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np

from contract import DATASETS, SEQ_LEN, PRED_LEN
import data as D
import evalgeom
from corruption import FAMILIES, SEVERITIES, CORRUPTION_SEED

N_ORIGINS = evalgeom.N_ORIGINS
ORIGIN_BATCH = evalgeom.ORIGIN_BATCH
WEATHER_N_CHANNELS = 12


def corrupted_channels(dataset: str) -> list:
    """ETTm1: every input channel. Weather: 12 channels chosen by train variance
    rank, evenly spaced over the sorted positions so low, medium and high
    variance channels are all represented. Test data is never consulted.
    """
    C = DATASETS[dataset].enc_in
    if dataset == "ETTm1":
        return list(range(C))
    x, _, _ = D.as_arrays(dataset, "train")
    order = np.argsort(x.var(axis=0))
    pos = np.linspace(0, C - 1, WEATHER_N_CHANNELS).round().astype(int)
    return sorted(int(order[p]) for p in np.unique(pos))


def evaluation_origins(dataset: str) -> list:
    """Eight non-overlapping origins spread evenly over the test split."""
    return evalgeom.origins(dataset, "test")


def train_robust_stats(dataset: str) -> dict:
    """Per-channel median and IQR of the scaled train split. Used by the
    clipping baseline, the quarantine detector and the coherence score.
    """
    x, _, _ = D.as_arrays(dataset, "train")
    med = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr = q75 - q25
    return {"median": med.tolist(), "iqr": iqr.tolist(), "std": x.std(axis=0).tolist()}


def build(dataset: str) -> dict:
    return {
        "dataset": dataset,
        "n_channels": DATASETS[dataset].enc_in,
        "corrupted_channels": corrupted_channels(dataset),
        "channel_selection_rule": (
            "ETTm1 uses every input channel. Weather sorts channels by train variance and "
            "takes 12 evenly spaced sorted positions, so low, medium and high variance "
            "channels are all covered. No test information is used."),
        "corruption_families": FAMILIES,
        "severities": SEVERITIES,
        "corruption_seed": CORRUPTION_SEED,
        "corruption_source": "official TSRBench CollectiveNoise.corrupt(), spliced per channel",
        "target_policy": "forecast targets are never corrupted; only the input lookback is",
        "n_origins": N_ORIGINS,
        "origin_batch": ORIGIN_BATCH,
        "evaluation_origins": evaluation_origins(dataset),
        "train_robust_stats": train_robust_stats(dataset),
        "clipping_rule": "clip scaled inputs to median_train +/- 5 * IQR_train",
        "channel_dropout_rate": 0.10,
        "quarantine_target_false_positive_rate": 0.05,
        "seq_len": SEQ_LEN, "pred_len": PRED_LEN,
    }
