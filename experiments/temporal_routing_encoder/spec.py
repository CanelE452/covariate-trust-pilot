"""The one architecture, the one seed policy, and the stop rule -- all before any error.

This is the last representation experiment on the gate axis, so the thresholds
that decide whether the axis continues are written down here and are not touched
after the numbers exist.
"""

from __future__ import annotations

import hashlib
import json

from ..structure_gate import gate_v2 as V2

# The GRU is a probe, not a proposal.  One architecture, no sweep of any kind.
ENCODER = {
    "type": "GRU",
    "input_size": 3,
    "hidden_size": 16,
    "num_layers": 1,
    "batch_first": True,
    "dropout": 0,
    "bidirectional": False,
}

HEAD = {
    "context": "concat(h_t, A_norm, B_norm)",
    "layers": "single Linear(hidden + 2*horizon, 1) then sigmoid",
    "hidden_mlp": None,
    "attention": False,
    "layer_norm": False,
}

CHANNELS = [
    {"name": "y_norm", "definition": "y_tau / scale_t, zeroed where unobserved"},
    {"name": "occurrence", "definition": "1[y_tau > 0] and observed, zeroed where unobserved"},
    {"name": "observed_mask", "definition": "1 where the step is observable at the origin"},
]

EXCLUDED_INPUTS = [
    "ADI", "CV2", "rho_interval", "rho_magnitude", "zero_ratio",
    "age_since_last_event", "recent_occurrence_rate", "train_scale_as_feature",
    "SBC_regime", "every G-NOSCALE descriptor", "mean(A)", "mean(B)",
    "disagreement summaries", "any hand-engineered forecast statistic",
]

# P0L1's optimiser, unchanged.  The only forced modification is memory tiling:
# gradients are accumulated over fixed chunks and the step is taken once per
# epoch, which is arithmetically the same update as one full-batch step.
TRAINING = {
    "optimizer": "Adam",
    "lr": V2.LR,
    "epochs": V2.EPOCHS,
    "seed": V2.SEED,
    "batching": "full batch, accumulated in chunks of 8192 rows for memory only",
    "chunk_rows": 8192,
    "patience": None,
    "search": "none",
}

# Declared before the first training run.  P0L1 is single-seed, so the primary
# stays single-seed for fairness and the second run exists only to show the
# pipeline is deterministic, not to average anything away.
SEED_POLICY = {
    "mode": "canonical_seed_primary_plus_reproducibility_rerun",
    "primary_seed": V2.SEED,
    "reproducibility_rerun": True,
    "additional_diagnostic_seeds": [],
    "post_hoc_seed_retry": "forbidden",
}

STOP_RULE = {
    "green": {
        "1_seq_beats_alpha_datasets": 3,
        "2_fresh_seq_vs_alpha_positive": True,
        "3_fresh_positive_folds_min": 2,
        "3_fresh_folds_total": 3,
        "4_seq_beats_p0l1_datasets": 3,
        "5_non_uci_ci_excludes_zero": 1,
        "6_overall_positive_fold_rate": 0.80,
        "7_worst_fold_floor": -0.02,
        "8_no_scale_sign_contradiction": True,
        "9_no_critical_integrity_failure": True,
    },
    "red_any_of": [
        "fresh aggregate <= 0",
        "two or more datasets worse than alpha in aggregate",
        "fails to beat P0L1 on at least 3 of 4",
        "severe temporal fold reversal",
        "catastrophic tail worsening",
    ],
    "severe_fold_reversal": {"datasets": 2, "worst_below": -0.02},
    "catastrophic_tail": {"absolute_floor": 0.05, "ratio_over_p0l1": 1.5},
    "frozen_before_results": True,
}

PREIDENTIFIED_COLLAPSE_FOLD = {
    "uci": 2,
    "identified_from": "results/gate_p0l1_robustness (largest E_alpha); recorded before this run",
}


def sequence_gate_spec() -> dict:
    spec = {
        "name": "RAW_HISTORY_SEQUENCE_GATE_V1",
        "role": "final representation experiment on the gate axis",
        "encoder": ENCODER,
        "head": HEAD,
        "channels": CHANNELS,
        "excluded_inputs": EXCLUDED_INPUTS,
        "expert_context": "raw A_norm and B_norm vectors of length horizon, no summaries",
        "mixture": "y = (1 - g) * A + g * B",
        "loss": "direct normalized mixture MSE, identical to P0L1",
        "training": TRAINING,
        "seed_policy": SEED_POLICY,
        "experts_receive_gradient": False,
        "folds": "P0L1 expanded temporal fold manifest, unchanged",
        "test_used": False,
        "new_dataset_used": False,
        "stop_rule": STOP_RULE,
    }
    payload = json.dumps(spec, indent=2, sort_keys=True, default=str)
    spec["spec_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return spec
