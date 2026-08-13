"""Everything that has to be fixed before an error is computed.

The diagnostic swaps one thing -- the function that maps gate features to g --
and holds the experts, the features, the folds and the objective at whatever the
P0L1 artifacts already say they are.  So the only genuinely new choices are the
boosting configuration and the warning thresholds, and both are written down
here rather than being decided once the numbers are visible.
"""

from __future__ import annotations

import hashlib
import json

from ..structure_gate.convex_oracle import FLAT_THRESHOLD

# The repository has no existing HistGradientBoosting configuration to inherit,
# so this one is stated as an operational choice: enough capacity to be clearly
# more flexible than a 529-parameter MLP, not a tuned optimum.  Nothing here is
# searched, and it does not change after the results are read.
HGB_CONFIG = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}

HGB_RATIONALE = (
    "single fixed diagnostic configuration; no grid, no random search, no "
    "comparison against other learners; chosen for flexibility relative to the "
    "frozen small MLP, not for optimality"
)

# den <= FLAT_THRESHOLD means the two experts predict the same thing on that
# origin, so the routing target u = num/den is not identified.  Those rows are
# dropped from fitting and kept in evaluation, where any g is nearly harmless.
FLAT_POLICY = "excluded_from_fitting_kept_in_evaluation"

# Frozen before the run.
WARN_SINGLE_FOLD_DOMINANCE = {
    "definition": (
        "per dataset, contribution of fold k is the summed origin-level gain "
        "sum_j (E_alpha_j - E_HGB_j) over that fold's validation origins; the "
        "dataset total is the sum over folds.  WARN when the total is positive "
        "and the largest single fold contributes more than the threshold share."
    ),
    "threshold_share": 0.70,
}

WARN_HGB_TAIL_RISK = {
    "definition": (
        "per-series degradation against alpha, positive meaning worse than "
        "alpha.  WARN when the HGB p95 degradation is both above the absolute "
        "floor and worse than the MLP p95 by more than the ratio."
    ),
    "absolute_floor": 0.05,
    "ratio_over_mlp": 1.5,
}

# Thresholds for the step 19/20 operational rule, also frozen before the run.
DIAGNOSIS_RULE = {
    "capacity_min_datasets_hgb_beats_mlp": 3,
    "capacity_min_datasets_recovery_up": 3,
    "train_fit_clearly_better": 0.10,
    "train_fit_min_datasets": 3,
    "small_validation_gain": 0.005,
    "small_recovery_increase": 0.05,
    "nonstationary_fold_reversal": 0.02,
}

PREIDENTIFIED_COLLAPSE_FOLD = {
    "uci": 2,
    "identified_from": "results/gate_p0l1_robustness (largest E_alpha); recorded before this run",
}


def diagnostic_spec() -> dict:
    spec = {
        "name": "ROUTING_INFORMATION_CEILING_DIAGNOSTIC_V1",
        "purpose": (
            "separate H_CAPACITY from H_INFORMATION by changing only the routing "
            "function approximator"
        ),
        "m0": "frozen P0L1 small MLP (hidden 16, neutral sigmoid, direct mixture MSE)",
        "m1": "sklearn HistGradientBoostingRegressor, single fixed config",
        "m1_is_not_a_proposed_model": True,
        "hgb_config": HGB_CONFIG,
        "hgb_rationale": HGB_RATIONALE,
        "objective": (
            "direct normalized mixture MSE, expressed exactly as weighted least "
            "squares: target u = num/den, sample_weight = den/n, prediction "
            "clipped to [0,1] afterwards rather than the target being clipped"
        ),
        "flat_threshold": FLAT_THRESHOLD,
        "flat_policy": FLAT_POLICY,
        "warn_single_fold_dominance": WARN_SINGLE_FOLD_DOMINANCE,
        "warn_hgb_tail_risk": WARN_HGB_TAIL_RISK,
        "diagnosis_rule": DIAGNOSIS_RULE,
        "preidentified_collapse_fold": PREIDENTIFIED_COLLAPSE_FOLD,
        "test_used": False,
        "new_dataset_used": False,
    }
    payload = json.dumps(spec, indent=2, sort_keys=True, default=str)
    spec["spec_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return spec
