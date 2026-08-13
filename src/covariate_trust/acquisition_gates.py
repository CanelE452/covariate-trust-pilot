"""Study 4 decision gates BA0-BA5.

Pure functions over an evidence dictionary so every verdict has a fixture test.
These gates are independent of Study 3's Gate H / Gate I, which are neither
re-evaluated nor modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_EVALUABLE = "NOT_EVALUABLE_LOW_COUNT"
NOT_RUN = "NOT_RUN"

MAX_ZONE_CONCENTRATION = 0.80
BA1_FAIL_ORACLE_VS_RANDOM = 0.005
BA1_FAIL_SINGLE_ZONE_RECOVERY = 0.95
BA2_MAX_WQL_DIFFERENCE = 0.01
BA3_FAIL_RECOVERY = 0.15
BA4_MAX_OVERLAP = 0.95
BA5_MIN_RECOVERY = 0.20


@dataclass
class Criterion:
    cid: str
    description: str
    value: Any
    threshold: Any
    passed: bool


@dataclass
class GateResult:
    name: str
    status: str
    decision: str
    criteria: list[Criterion] = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "decision": self.decision,
            "criteria": [asdict(c) for c in self.criteria],
            "fail_reasons": self.fail_reasons,
            "evidence": self.evidence,
        }


def _degradation(rel: float) -> float:
    return max(0.0, -float(rel))


# --------------------------------------------------------------------------


def evaluate_ba0(evidence: dict[str, Any]) -> GateResult:
    checks = evidence["checks"]
    counts = evidence["portfolio_days"]
    minimums = evidence["minimum_days"]

    criteria = [
        Criterion("BA0.1", "four zones in every retained portfolio",
                  checks["four_zone_portfolios"], True, bool(checks["four_zone_portfolios"])),
        Criterion("BA0.2", "M1/M3 context and future-calendar equality guards present",
                  checks["fair_comparison_guard"], True, bool(checks["fair_comparison_guard"])),
        Criterion("BA0.3", "no future target or realised outcome in the selector features",
                  checks["no_future_leakage"], True, bool(checks["no_future_leakage"])),
        Criterion("BA0.4", "current premium output absent from the feature frame",
                  checks["no_current_premium_feature"], True,
                  bool(checks["no_current_premium_feature"])),
        Criterion("BA0.5", "chronological train/validation/test separation",
                  checks["chronological_split"], True, bool(checks["chronological_split"])),
        Criterion("BA0.6", "complete portfolio days per split",
                  counts, minimums,
                  all(counts.get(k, 0) >= v for k, v in minimums.items())),
        Criterion("BA0.7", "cross_learning is False in the reused Study 3 run",
                  checks["cross_learning_false"], True, bool(checks["cross_learning_false"])),
        Criterion("BA0.8", "prior Study 3 artifact hashes unchanged",
                  checks["study3_hashes_unchanged"], True,
                  bool(checks["study3_hashes_unchanged"])),
    ]
    failed = [c.cid for c in criteria if not c.passed]
    if failed:
        return GateResult("BA0", FAIL, "INVALID_PILOT", criteria,
                          [f"integrity check failed: {failed}"], evidence)
    return GateResult("BA0", PASS, "PILOT_INTEGRITY_CONFIRMED", criteria, [], evidence)


def evaluate_ba1(evidence: dict[str, Any], thresholds: dict[str, float]) -> GateResult:
    per_k = evidence["per_k"]
    vs_no_premium = {k: float(v["oracle_vs_no_premium"]) for k, v in per_k.items()}
    vs_random = {k: float(v["oracle_vs_random"]) for k, v in per_k.items()}
    ci_ok = {k: bool(v["ci_favours_oracle"]) for k, v in per_k.items()}
    concentration = {k: float(v["max_zone_share"]) for k, v in per_k.items()}
    positive_rate = float(evidence["premium_positive_rate"])
    single_zone_recovery = {k: float(v["best_fixed_zone_recovery"]) for k, v in per_k.items()}

    t_np = thresholds["ba1_oracle_vs_no_premium"]
    t_rand = thresholds["ba1_oracle_vs_random"]

    criteria = [
        Criterion("BA1.1", "oracle beats NO_PREMIUM on portfolio WQL (every K)",
                  vs_no_premium, t_np, all(v >= t_np for v in vs_no_premium.values())),
        Criterion("BA1.2", "oracle beats RANDOM_K on portfolio WQL (every K)",
                  vs_random, t_rand, all(v >= t_rand for v in vs_random.values())),
        Criterion("BA1.3", "week-cluster CI favours the oracle (every K)",
                  ci_ok, True, all(ci_ok.values())),
        Criterion("BA1.4", "oracle does not keep picking one zone",
                  concentration, MAX_ZONE_CONCENTRATION,
                  all(v < MAX_ZONE_CONCENTRATION for v in concentration.values())),
        Criterion("BA1.5", "both premium-positive and premium-negative tasks exist",
                  positive_rate, (0.05, 0.95), 0.05 < positive_rate < 0.95),
    ]

    reasons: list[str] = []
    if any(v <= BA1_FAIL_ORACLE_VS_RANDOM for v in vs_random.values()):
        reasons.append(
            f"oracle vs random improvement {vs_random} <= {BA1_FAIL_ORACLE_VS_RANDOM}: "
            "there is no allocation headroom to exploit"
        )
    if any(v >= BA1_FAIL_SINGLE_ZONE_RECOVERY for v in single_zone_recovery.values()):
        reasons.append(
            f"a fixed single-zone policy already recovers {single_zone_recovery} of the "
            "oracle gain, so the problem is not a selection problem"
        )
    if not 0.05 < positive_rate < 0.95:
        reasons.append(f"premium is almost always {'better' if positive_rate > 0.95 else 'worse'}")

    if reasons:
        return GateResult("BA1", FAIL, "BUDGETED_ACQUISITION_NO_GO", criteria, reasons, evidence)
    if all(c.passed for c in criteria):
        return GateResult("BA1", PASS, "BUDGET_HEADROOM_CONFIRMED", criteria, [], evidence)
    return GateResult("BA1", INCONCLUSIVE, "INCONCLUSIVE", criteria, [], evidence)


def evaluate_ba2(evidence: dict[str, Any], thresholds: dict[str, float]) -> GateResult:
    recovery = float(evidence["best_heuristic_recovery"])
    wql_difference = float(evidence["heuristic_minus_value_wql_rel"])
    threshold = thresholds["ba2_simple_heuristic_oracle_recovery_sufficient"]

    criteria = [
        Criterion("BA2.1", "best simple heuristic recovers most of the oracle headroom",
                  recovery, threshold, recovery >= threshold),
        Criterion("BA2.2", "value predictor adds less than 1% WQL over the heuristic",
                  wql_difference, BA2_MAX_WQL_DIFFERENCE,
                  abs(wql_difference) < BA2_MAX_WQL_DIFFERENCE),
    ]
    if all(c.passed for c in criteria):
        return GateResult(
            "BA2", PASS, "SIMPLE_HEURISTIC_SUFFICIENT / NEW_VALUE_MODEL_NO_GO",
            criteria,
            ["a simple ordering rule already captures the available headroom"],
            evidence,
        )
    return GateResult("BA2", FAIL, "SIMPLE_HEURISTIC_INSUFFICIENT / PROCEED_TO_BA3",
                      criteria, [], evidence)


def evaluate_ba3(evidence: dict[str, Any], thresholds: dict[str, float]) -> GateResult:
    per_k = evidence["per_k"]
    vs_random = {k: float(v["value_vs_random"]) for k, v in per_k.items()}
    vs_no_premium = {k: float(v["value_vs_no_premium"]) for k, v in per_k.items()}
    recovery = {k: float(v["oracle_recovery"]) for k, v in per_k.items()}
    ci_ok = {k: bool(v["ci_favours_value"]) for k, v in per_k.items()}
    concentration = {k: float(v["max_zone_share"]) for k, v in per_k.items()}
    zones_improved = int(evidence["zones_improved"])

    t_rand = thresholds["ba3_value_vs_random"]
    t_rec = thresholds["ba3_value_oracle_recovery"]

    criteria = [
        Criterion("BA3.1", "beats RANDOM_K on portfolio WQL (every K)",
                  vs_random, t_rand, all(v >= t_rand for v in vs_random.values())),
        Criterion("BA3.2", "beats NO_PREMIUM on portfolio WQL (every K)",
                  vs_no_premium, 0.0, all(v > 0.0 for v in vs_no_premium.values())),
        Criterion("BA3.3", "recovers enough of the oracle headroom (every K)",
                  recovery, t_rec, all(v >= t_rec for v in recovery.values())),
        Criterion("BA3.4", "week-cluster CI favours the value predictor (every K)",
                  ci_ok, True, all(ci_ok.values())),
        Criterion("BA3.5", "improves in at least three zones", zones_improved, 3,
                  zones_improved >= 3),
        Criterion("BA3.6", "selection is not concentrated on a single zone",
                  concentration, MAX_ZONE_CONCENTRATION,
                  all(v < MAX_ZONE_CONCENTRATION for v in concentration.values())),
    ]

    reasons: list[str] = []
    if any(v <= 0.0 for v in vs_random.values()):
        reasons.append(f"no improvement over random: {vs_random}")
    if any(v <= BA3_FAIL_RECOVERY for v in recovery.values()):
        reasons.append(f"oracle recovery {recovery} <= {BA3_FAIL_RECOVERY}")
    if not any(ci_ok.values()):
        reasons.append("the week-cluster CI does not favour the value predictor at any K")

    if reasons:
        return GateResult("BA3", FAIL,
                          "VALUE_NOT_PREDICTABLE / BUDGET_PROBLEM_EXISTS_METHOD_NO_GO",
                          criteria, reasons, evidence)
    if all(c.passed for c in criteria):
        return GateResult("BA3", PASS, "FORECAST_VALUE_ROUTING_GO", criteria, [], evidence)
    return GateResult("BA3", INCONCLUSIVE, "INCONCLUSIVE", criteria, [], evidence)


def evaluate_ba4(evidence: dict[str, Any], thresholds: dict[str, float]) -> GateResult:
    q90_gain = float(evidence["q90_selector_vs_wql_selector_on_q90"])
    ci_ok = bool(evidence["ci_favours_q90_selector"])
    overlap = float(evidence["selection_overlap"])
    wql_degradation = _degradation(float(evidence["q90_selector_vs_wql_selector_on_wql"]))
    threshold = thresholds["ba4_q90_value_vs_wql_selector"]

    criteria = [
        Criterion("BA4.1", "q90-value selector improves the q90 objective", q90_gain,
                  threshold, q90_gain >= threshold),
        Criterion("BA4.2", "week-cluster CI favours the q90-value selector", ci_ok, True, ci_ok),
        Criterion("BA4.3", "the two selectors do not choose the same zones", overlap,
                  BA4_MAX_OVERLAP, overlap < BA4_MAX_OVERLAP),
        Criterion("BA4.4", "WQL guard: q90 selector does not cost more than 1% WQL",
                  wql_degradation, 0.01, wql_degradation <= 0.01),
    ]

    reasons: list[str] = []
    if q90_gain <= 0.0:
        reasons.append(f"q90 improvement {q90_gain:.4f} <= 0")
    if overlap >= BA4_MAX_OVERLAP:
        reasons.append(f"selection overlap {overlap:.3f}: the two selectors are effectively identical")
    if wql_degradation > 0.01:
        reasons.append(f"WQL guard violated: {wql_degradation:.4f} > 0.01")

    if reasons:
        return GateResult("BA4", FAIL, "DECISION_SPECIFIC_VALUE_NOT_ESTABLISHED",
                          criteria, reasons, evidence)
    if all(c.passed for c in criteria):
        return GateResult("BA4", PASS, "DECISION_VALUE_ACQUISITION_CANDIDATE",
                          criteria, [], evidence)
    return GateResult("BA4", INCONCLUSIVE, "INCONCLUSIVE", criteria, [], evidence)


def evaluate_ba5(evidence: dict[str, Any]) -> GateResult:
    n_days = int(evidence["n_portfolio_days"])
    minimum = int(evidence["minimum_days"])
    if n_days < minimum:
        return GateResult(
            "BA5", NOT_EVALUABLE,
            f"NOT_EVALUABLE_LOW_COUNT ({n_days} < {minimum} complete portfolio days)",
            [], [], evidence,
        )
    value_vs_random = float(evidence["value_vs_random"])
    recovery = float(evidence["oracle_recovery"])
    q90_vs_random = float(evidence["q90_selector_vs_random_on_q90"])
    budget_ok = bool(evidence["budget_respected"])

    criteria = [
        Criterion("BA5.1", "value selector beats random in direction", value_vs_random, 0.0,
                  value_vs_random > 0.0),
        Criterion("BA5.2", "oracle recovery at least 20%", recovery, BA5_MIN_RECOVERY,
                  recovery >= BA5_MIN_RECOVERY),
        Criterion("BA5.3", "q90 selector beats random on the q90 objective", q90_vs_random,
                  0.0, q90_vs_random > 0.0),
        Criterion("BA5.4", "no budget violation", budget_ok, True, budget_ok),
    ]
    if all(c.passed for c in criteria):
        return GateResult("BA5", PASS, "SMALL_FRESH_CONFIRMATION_SUPPORT", criteria, [], evidence)
    return GateResult("BA5", FAIL, "FRESH_CONFIRMATION_NOT_SUPPORTED", criteria, [], evidence)


def final_status(
    ba0: GateResult | None, ba1: GateResult | None, ba2: GateResult | None,
    ba3: GateResult | None, ba4: GateResult | None, ba5: GateResult | None,
) -> str:
    if ba0 is None:
        return NOT_RUN
    if ba0.status == FAIL:
        return "INVALID_PILOT"
    if ba1 is None:
        return "BA1_NOT_RUN"
    if ba1.status == FAIL:
        return "BUDGETED_ACQUISITION_NO_GO"
    if ba1.status == INCONCLUSIVE:
        return "INCONCLUSIVE (BA1)"
    if ba2 is not None and ba2.status == PASS:
        return "SIMPLE_RULE_OPERATIONAL_RESULT / NEW_VALUE_MODEL_NO_GO"
    if ba3 is None:
        return "BA3_NOT_RUN"
    if ba3.status == FAIL:
        return "VALUE_NOT_PREDICTABLE / BUDGET_PROBLEM_EXISTS_METHOD_NO_GO"
    if ba3.status == INCONCLUSIVE:
        return "INCONCLUSIVE (BA3)"
    base = "FORECAST_VALUE_ROUTING_GO"
    if ba4 is None or ba4.status != PASS:
        base += " / DECISION_SPECIFIC_VALUE_NOT_ESTABLISHED"
    else:
        base = "DECISION_VALUE_ACQUISITION_CANDIDATE"
    if ba5 is not None and ba5.status == PASS:
        base += " / SMALL_FRESH_CONFIRMATION_SUPPORT"
    return base
