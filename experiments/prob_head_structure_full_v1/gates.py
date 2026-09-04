"""Pure, serialization-ready reducers for every preregistered scientific gate."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_mapping(values: Mapping[str, float], name: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return {str(key): _finite(value, f"{name}.{key}") for key, value in values.items()}


_HEADS = {"NB", "HSNB", "TWEEDIE_FULL"}
_PAIR_KEYS = {"NB|HSNB", "NB|TWEEDIE_FULL", "HSNB|TWEEDIE_FULL"}
_DATASET_PRIORITY = ("m5", "auto", "carparts", "raf", "online_retail")
_DATASET_DISPLAY = {
    "m5": "M5",
    "auto": "Auto",
    "carparts": "Carparts",
    "raf": "RAF",
    "online_retail": "OnlineRetail",
}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    """Recompute the data-layer canonical digest without trusting stored hashes."""
    payload = {str(key): deepcopy(item) for key, item in value.items() if key != "audit_sha256"}
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("sealed dataset audit is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _verify_selection_dataset_audit(
    raw: Mapping[str, Any], *, expected_dataset_id: str
) -> dict[str, Any]:
    """Validate the complete sealed row emitted by ``data.select_real_datasets``."""
    row = deepcopy(dict(raw))
    if row.get("dataset_id") != expected_dataset_id:
        raise ValueError("complete five-row sealed audit has an invalid dataset order")
    if not _is_sha256(row.get("audit_sha256")) or row["audit_sha256"] != _canonical_sha256(row):
        raise ValueError("sealed audit SHA256 mismatch; audit payload SHA256 mismatch")
    audit_type = row.get("audit_type")
    if audit_type == "GEOMETRY_ONLY_AUDIT":
        required = {
            "audit_type", "dataset_id", "length", "lookback", "horizon",
            "required_length", "geometry_status", "status",
            "count_primary_eligible", "confirmatory_eligible", "audit_sha256",
        }
        integer_fields = ("length", "lookback", "horizon", "required_length")
        if (
            set(row) != required
            or any(isinstance(row.get(key), bool) or not isinstance(row.get(key), int) for key in integer_fields)
            or min(int(row[key]) for key in integer_fields) < 0
            or row["required_length"] != row["lookback"] + 8 * row["horizon"]
            or row.get("status") not in {"INSUFFICIENT_LENGTH", "SUPPORT_SOURCE_AUDIT_REQUIRED"}
            or row.get("count_primary_eligible") is not False
            or row.get("confirmatory_eligible") is not False
        ):
            raise ValueError("sealed geometry dataset audit is malformed")
        expected_geometry = (
            "PASS" if row["length"] >= row["required_length"] else "INSUFFICIENT_LENGTH"
        )
        if row.get("geometry_status") != expected_geometry:
            raise ValueError("sealed geometry dataset audit is internally inconsistent")
        return row
    required = {
        "audit_type", "dataset_id", "status", "geometry_status",
        "count_primary_eligible", "confirmatory_eligible", "support_audit",
        "split_validation", "source_records", "source_manifest_aggregate_sha256",
        "panel_shape", "panel_binding_sha256", "ordered_series_id_sha256",
        "canonical_source_attested", "audit_sha256",
    }
    support = row.get("support_audit")
    split = row.get("split_validation")
    sources = row.get("source_records")
    panel_shape = row.get("panel_shape")
    if (
        audit_type != "COUNT_PRIMARY_DATASET_AUDIT"
        or set(row) != required
        or row.get("status") != "PASS"
        or row.get("geometry_status") != "PASS"
        or row.get("count_primary_eligible") is not True
        or row.get("confirmatory_eligible") is not True
        or not isinstance(support, Mapping)
        or support.get("dataset_id") != expected_dataset_id
        or support.get("status") != "PASS"
        or support.get("target_scope") != "model_train"
        or support.get("count_primary_eligible") is not True
        or support.get("count_likelihood_index_exact") is not True
        or not isinstance(split, Mapping)
        or split.get("status") != "PASS"
        or not isinstance(sources, list)
        or not sources
        or not all(
            isinstance(source, Mapping)
            and set(source) == {"path", "bytes", "sha256"}
            and isinstance(source.get("path"), str)
            and bool(source.get("path"))
            and isinstance(source.get("bytes"), int)
            and not isinstance(source.get("bytes"), bool)
            and int(source["bytes"]) >= 0
            and _is_sha256(source.get("sha256"))
            for source in sources
        )
        or not _is_sha256(row.get("source_manifest_aggregate_sha256"))
        or not _is_sha256(row.get("panel_binding_sha256"))
        or not _is_sha256(row.get("ordered_series_id_sha256"))
        or row.get("canonical_source_attested") is not True
        or not isinstance(panel_shape, list)
        or len(panel_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in panel_shape)
    ):
        raise ValueError("sealed count-primary dataset audit is not canonical")
    return row


@dataclass(frozen=True, init=False)
class FrozenPrimaryDatasetManifest:
    """Audit-bound count-primary selection made before any outcome is observed.

    ``eligible_datasets_in_priority_order`` is the ordered list that passed the
    support/source audit.  The selected datasets must be M5 plus the first one
    or two eligible non-M5 datasets, exactly as required by the frozen policy.
    """

    selected_datasets: tuple[str, ...]
    eligible_datasets_in_priority_order: tuple[str, ...]
    canonical_audit_payload_json: str
    audit_sha256: str

    @staticmethod
    def _validate_policy(
        selected: tuple[str, ...], eligible: tuple[str, ...]
    ) -> None:
        def valid_order(items: tuple[str, ...]) -> bool:
            return (
                bool(items)
                and len(set(items)) == len(items)
                and all(item in _DATASET_PRIORITY for item in items)
                and tuple(sorted(items, key=_DATASET_PRIORITY.index)) == items
            )

        if not valid_order(eligible) or eligible[0] != "m5" or len(eligible) < 2:
            raise ValueError(
                "primary dataset manifest eligible identities must be a unique "
                "M5-first subsequence of the frozen priority with at least two datasets"
            )
        expected_selected = eligible[: min(3, len(eligible))]
        if len(selected) not in {2, 3} or selected != expected_selected:
            raise ValueError(
                "primary dataset manifest selection must be M5 plus up to the first "
                "two audited eligible non-M5 datasets"
            )

    @classmethod
    def from_audit_payload(
        cls,
        *,
        selected_datasets: Sequence[str],
        eligible_datasets_in_priority_order: Sequence[str],
        audit_payload: Mapping[str, Any],
        expected_sha256: str | None = None,
    ) -> "FrozenPrimaryDatasetManifest":
        selected = tuple(str(item) for item in selected_datasets)
        eligible = tuple(str(item) for item in eligible_datasets_in_priority_order)
        cls._validate_policy(selected, eligible)
        if not isinstance(audit_payload, Mapping) or not audit_payload:
            raise ValueError("primary dataset manifest requires a non-empty audit payload")
        binding = {
            "schema": "prob_head_structure_full_v1.primary_dataset_manifest.v1",
            "selected_dataset_ids": list(selected),
            "eligible_dataset_ids_in_priority_order": list(eligible),
            "audit_payload": deepcopy(dict(audit_payload)),
        }
        try:
            canonical = json.dumps(
                binding,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("primary dataset audit payload is not canonical JSON") from error
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if expected_sha256 is not None and str(expected_sha256).lower() != digest:
            raise ValueError("primary dataset audit payload SHA256 mismatch")
        instance = object.__new__(cls)
        object.__setattr__(instance, "selected_datasets", selected)
        object.__setattr__(instance, "eligible_datasets_in_priority_order", eligible)
        object.__setattr__(instance, "canonical_audit_payload_json", canonical)
        object.__setattr__(instance, "audit_sha256", digest)
        return instance

    @classmethod
    def from_selection_audit(
        cls,
        selection_audit: Mapping[str, Any],
        *,
        expected_sha256: str | None = None,
    ) -> "FrozenPrimaryDatasetManifest":
        if not isinstance(selection_audit, Mapping):
            raise ValueError("primary dataset selection audit must be a mapping")
        required_fields = {
            "manifest_type",
            "status",
            "selected_dataset_ids",
            "selected",
            "audits",
            "audit_manifest_sha256",
            "requested_non_m5",
            "eligible_non_m5_count",
            "priority",
            "selection_manifest_sha256",
        }
        priority = tuple(str(item) for item in selection_audit.get("priority", ()))
        audits = selection_audit.get("audits")
        if (
            set(selection_audit) != required_fields
            or selection_audit.get("manifest_type")
            != "COUNT_PRIMARY_DATASET_SELECTION_MANIFEST"
            or priority != _DATASET_PRIORITY
            or not isinstance(audits, Sequence)
            or isinstance(audits, (str, bytes))
            or len(audits) != len(_DATASET_PRIORITY)
            or not all(isinstance(row, Mapping) for row in audits)
        ):
            raise ValueError("primary dataset selection requires a complete five-row sealed audit")
        verified = tuple(
            _verify_selection_dataset_audit(row, expected_dataset_id=dataset_id)
            for dataset_id, row in zip(_DATASET_PRIORITY, audits, strict=True)
        )
        audit_manifest = _canonical_sha256({"audits": list(verified)})
        if (
            not _is_sha256(selection_audit.get("audit_manifest_sha256"))
            or selection_audit.get("audit_manifest_sha256") != audit_manifest
        ):
            raise ValueError("primary dataset audit manifest SHA256 mismatch")
        eligible = tuple(
            str(row["dataset_id"])
            for row in verified
            if row.get("audit_type") == "COUNT_PRIMARY_DATASET_AUDIT"
            and row.get("status") == "PASS"
            and row.get("count_primary_eligible") is True
            and row.get("confirmatory_eligible") is True
        )
        if not eligible or eligible[0] != "m5":
            raise ValueError("primary dataset selection audit is not a frozen PASS artifact")
        expected_selected = eligible[: min(3, len(eligible))]
        selected = tuple(str(item) for item in selection_audit.get("selected_dataset_ids", ()))
        selected_rows = selection_audit.get("selected")
        rows_are_valid = (
            isinstance(selected_rows, Sequence)
            and not isinstance(selected_rows, (str, bytes))
            and [deepcopy(dict(row)) for row in selected_rows]
            == [verified[_DATASET_PRIORITY.index(dataset_id)] for dataset_id in expected_selected]
        )
        if (
            selection_audit.get("status") != "PASS"
            or selection_audit.get("requested_non_m5") != 2
            or selection_audit.get("eligible_non_m5_count") != min(2, len(eligible) - 1)
            or selected != expected_selected
            or not rows_are_valid
        ):
            raise ValueError(
                "primary dataset selection audit is not a frozen PASS artifact and is "
                "not derived audited eligibility; "
                "audit payload SHA256 mismatch"
            )
        return cls.from_audit_payload(
            selected_datasets=expected_selected,
            eligible_datasets_in_priority_order=eligible,
            audit_payload=selection_audit,
            expected_sha256=expected_sha256,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrozenPrimaryDatasetManifest":
        if not isinstance(record, Mapping):
            raise ValueError("primary dataset manifest record must be a mapping")
        result = cls.from_audit_payload(
            selected_datasets=record.get("selected_dataset_ids", ()),
            eligible_datasets_in_priority_order=record.get(
                "eligible_dataset_ids_in_priority_order", ()
            ),
            audit_payload=record.get("audit_payload", {}),
            expected_sha256=str(record.get("audit_sha256", "")),
        )
        if deepcopy(dict(record)) != result.as_dict():
            raise ValueError("primary dataset manifest record is not canonical")
        return result

    def as_dict(self) -> dict[str, Any]:
        binding = json.loads(self.canonical_audit_payload_json)
        return {
            "selected_dataset_ids": list(self.selected_datasets),
            "eligible_dataset_ids_in_priority_order": list(
                self.eligible_datasets_in_priority_order
            ),
            "selected_display_names": [
                _DATASET_DISPLAY[item] for item in self.selected_datasets
            ],
            "eligible_display_names_in_priority_order": [
                _DATASET_DISPLAY[item] for item in self.eligible_datasets_in_priority_order
            ],
            "audit_payload": binding["audit_payload"],
            "audit_sha256": self.audit_sha256,
        }


def _primary_manifest(
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> tuple[str, ...]:
    if not isinstance(primary_datasets, FrozenPrimaryDatasetManifest):
        raise ValueError(
            "frozen primary dataset manifest object with audited identity is required"
        )
    return primary_datasets.selected_datasets


def _require_primary_mapping(
    values: Mapping[str, Any],
    *,
    manifest: tuple[str, ...],
    name: str,
) -> dict[str, Any]:
    keys = tuple(str(key) for key in values)
    if keys != manifest:
        raise ValueError(
            f"{name} must match the exact primary dataset manifest order {list(manifest)}"
        )
    return {str(key): value for key, value in values.items()}


def _validate_undefined_reasons(
    undefined_datasets: Sequence[str],
    reasons: Mapping[str, str],
    *,
    allowed_tokens: set[str],
    name: str,
) -> dict[str, str]:
    expected = tuple(str(dataset) for dataset in undefined_datasets)
    actual = tuple(str(dataset) for dataset in reasons)
    if actual != expected:
        raise ValueError(
            f"{name} undefined reason keys must exactly match undefined datasets {list(expected)}"
        )
    normalized = {str(dataset): str(reason) for dataset, reason in reasons.items()}
    if any(reason not in allowed_tokens for reason in normalized.values()):
        raise ValueError(
            f"{name} undefined reason must be one of {sorted(allowed_tokens)}"
        )
    return normalized


def _optional_finite_primary_mapping(
    values: Mapping[str, float | None],
    *,
    manifest: tuple[str, ...],
    reasons: Mapping[str, str],
    allowed_tokens: set[str],
    name: str,
) -> tuple[dict[str, float | None], dict[str, str]]:
    raw = _require_primary_mapping(values, manifest=manifest, name=name)
    normalized: dict[str, float | None] = {}
    undefined: list[str] = []
    for dataset, value in raw.items():
        if value is None:
            normalized[dataset] = None
            undefined.append(dataset)
        else:
            normalized[dataset] = _finite(value, f"{name}.{dataset}")
    normalized_reasons = _validate_undefined_reasons(
        undefined,
        reasons,
        allowed_tokens=allowed_tokens,
        name=name,
    )
    return normalized, normalized_reasons


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    verdict: str
    failure_label: str | None
    observations: Mapping[str, Any]
    criteria: Mapping[str, Any]
    upstream_required_gates: tuple[str, ...] = ()
    upstream_gate_status: Mapping[str, str] = MappingProxyType({})
    confirmatory_eligible: bool = True
    scientific_role: str = "CONFIRMATORY"

    def __post_init__(self) -> None:
        if not self.gate or not self.verdict:
            raise ValueError("gate and verdict must be non-empty")
        if self.passed != (self.failure_label is None):
            raise ValueError("failure label must be absent exactly when a gate passes")
        object.__setattr__(self, "observations", _freeze(self.observations))
        object.__setattr__(self, "criteria", _freeze(self.criteria))
        object.__setattr__(self, "upstream_gate_status", _freeze(self.upstream_gate_status))
        if self.confirmatory_eligible and self.scientific_role != "CONFIRMATORY":
            raise ValueError("confirmatory gate result has a diagnostic role")
        if not self.confirmatory_eligible and not self.scientific_role.startswith(
            "DIAGNOSTIC_CONTINUATION_AFTER_"
        ):
            raise ValueError("ineligible result must retain its diagnostic lineage")

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @classmethod
    def pass_result(
        cls,
        gate: str,
        pass_label: str,
        observations: Mapping[str, Any],
        criteria: Mapping[str, Any],
    ) -> "GateResult":
        return cls(gate, True, pass_label, None, observations, criteria)

    @classmethod
    def fail_result(
        cls,
        gate: str,
        fail_label: str,
        observations: Mapping[str, Any],
        criteria: Mapping[str, Any],
    ) -> "GateResult":
        return cls(gate, False, fail_label, fail_label, observations, criteria)

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch": str(self.observations.get("branch", self.gate)),
            "gate": self.gate,
            "status": self.status,
            "verdict": self.verdict,
            "failure_label": self.failure_label,
            "observations": _thaw(self.observations),
            "criteria": _thaw(self.criteria),
            "upstream_required_gates": list(self.upstream_required_gates),
            "upstream_gate_status": _thaw(self.upstream_gate_status),
            "confirmatory_eligible": self.confirmatory_eligible,
            "scientific_role": self.scientific_role,
        }


def _result(
    gate: str,
    passed: bool,
    pass_label: str,
    fail_label: str,
    observations: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> GateResult:
    if passed:
        return GateResult.pass_result(gate, pass_label, observations, criteria)
    return GateResult.fail_result(gate, fail_label, observations, criteria)


def attach_lineage(
    result: GateResult,
    *,
    branch: str,
    upstream: Sequence[GateResult],
) -> GateResult:
    """Return a new result with immutable upstream status and earliest failure."""

    statuses: dict[str, str] = {}

    def add_status(gate: str, status: str) -> None:
        if status not in {"PASS", "FAIL"}:
            raise ValueError(f"invalid immutable upstream status for {gate}: {status}")
        prior = statuses.get(gate)
        if prior is not None and prior != status:
            raise ValueError(f"conflicting immutable upstream status for {gate}")
        if prior is None:
            statuses[gate] = status

    # Reattaching lineage must be a union, never a replacement.  This preserves
    # the first failed gate even when a diagnostic result passes downstream.
    for gate, status in result.upstream_gate_status.items():
        add_status(str(gate), str(status))
    for item in upstream:
        for gate, status in item.upstream_gate_status.items():
            add_status(str(gate), str(status))
        add_status(item.gate, item.status)
    failed = [gate for gate, status in statuses.items() if status == "FAIL"]
    inherited_ineligible_roles = [
        value.scientific_role
        for value in (result, *upstream)
        if not value.confirmatory_eligible
    ]
    eligible = not failed and not inherited_ineligible_roles
    if eligible:
        role = "CONFIRMATORY"
    elif failed:
        role = f"DIAGNOSTIC_CONTINUATION_AFTER_{failed[0]}"
    else:
        role = inherited_ineligible_roles[0]
    # `branch` is intentionally recorded as observation metadata without changing
    # the gate identity or any upstream object.
    observations = _thaw(result.observations)
    observations.setdefault("branch", branch)
    return replace(
        result,
        observations=observations,
        upstream_required_gates=tuple(statuses),
        upstream_gate_status=statuses,
        confirmatory_eligible=eligible,
        scientific_role=role,
    )


def gate_s1(
    *,
    exact_best_cell_counts: Mapping[str, int],
    practical_winner_shares: Mapping[str, float],
    total_cells: int,
) -> GateResult:
    if int(total_cells) != 18:
        raise ValueError("S1 confirmatory grid must contain exactly 18 cells")
    counts = {str(head): int(count) for head, count in exact_best_cell_counts.items()}
    shares = _finite_mapping(practical_winner_shares, "practical_winner_shares")
    if set(counts) != _HEADS or set(shares) != _HEADS or any(count < 0 for count in counts.values()):
        raise ValueError("S1 head identities/counts are inconsistent")
    criterion_a_heads = sorted(head for head, count in counts.items() if count >= 3)
    criterion_b_heads = sorted(head for head, share in shares.items() if share >= 0.15)
    criteria = {
        "at_least_two_heads_best_in_at_least_3_cells": len(criterion_a_heads) >= 2,
        "at_least_two_heads_practical_share_at_least_15pct": len(criterion_b_heads) >= 2,
    }
    return _result(
        "S1",
        any(criteria.values()),
        "HEAD_SPECIALIZATION_GO",
        "HEAD_SPECIALIZATION_NO_GO",
        {
            "total_cells": total_cells,
            "exact_best_cell_counts": counts,
            "practical_winner_shares": shares,
            "criterion_a_heads": criterion_a_heads,
            "criterion_b_heads": criterion_b_heads,
        },
        criteria,
    )


def gate_s2(*, cell_oracle_gain: float, series_origin_oracle_gain: float) -> GateResult:
    cell = _finite(cell_oracle_gain, "cell_oracle_gain")
    series = _finite(series_origin_oracle_gain, "series_origin_oracle_gain")
    criteria = {
        "cell_oracle_gain_at_least_2pct": cell >= 0.02,
        "series_origin_oracle_gain_at_least_3pct": series >= 0.03,
    }
    return _result(
        "S2",
        all(criteria.values()),
        "HEAD_COMPLEMENTARITY_GO",
        "HEAD_COMPLEMENTARITY_TOO_SMALL",
        {"cell_oracle_gain": cell, "series_origin_oracle_gain": series},
        criteria,
    )


_S3_PAIRS = {"NB_vs_HSNB", "NB_vs_TWEEDIE_FULL", "HSNB_vs_TWEEDIE_FULL"}
_S3_FACTORS = {
    "d",
    "rho_I_L",
    "rho_I_Q",
    "rho_M_L",
    "rho_M_Q",
    "d*rho_I_L",
    "d*rho_I_Q",
    "d*rho_M_L",
    "d*rho_M_Q",
    "rho_I_L*rho_M_L",
    "rho_I_L*rho_M_Q",
    "rho_I_Q*rho_M_L",
    "rho_I_Q*rho_M_Q",
    "d*rho_I_L*rho_M_L",
    "d*rho_I_L*rho_M_Q",
    "d*rho_I_Q*rho_M_L",
    "d*rho_I_Q*rho_M_Q",
}


def gate_s3(contrasts: Sequence[Mapping[str, Any]]) -> GateResult:
    rows: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, str]] = set()
    for raw in contrasts:
        pair = str(raw["pair"])
        contrast = str(raw["contrast"])
        if pair not in _S3_PAIRS or contrast not in _S3_FACTORS:
            raise ValueError(f"unregistered S3 contrast: {pair}/{contrast}")
        key = (pair, contrast)
        if key in observed_keys:
            raise ValueError(f"duplicate S3 contrast: {pair}/{contrast}")
        observed_keys.add(key)
        effect = _finite(raw["effect"], "effect")
        lower = _finite(raw["ci_lower"], "ci_lower")
        upper = _finite(raw["ci_upper"], "ci_upper")
        if lower > upper:
            raise ValueError("S3 confidence interval is reversed")
        excludes_zero = lower > 0.0 or upper < 0.0
        passes = abs(effect) >= 2.0 and excludes_zero
        rows.append(
            {
                "pair": pair,
                "contrast": contrast,
                "effect": effect,
                "ci_lower": lower,
                "ci_upper": upper,
                "absolute_effect_at_least_2pp": abs(effect) >= 2.0,
                "ci_excludes_zero": excludes_zero,
                "passes": passes,
            }
        )
    expected_keys = {(pair, factor) for pair in _S3_PAIRS for factor in _S3_FACTORS}
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            "S3 requires exactly the unique 3 pairs x 17 contrasts; "
            f"missing={missing}, extra={extra}"
        )
    passing = sum(bool(row["passes"]) for row in rows)
    criteria = {"at_least_one_registered_temporal_contrast_passes": passing >= 1}
    return _result(
        "S3",
        all(criteria.values()),
        "TEMPORAL_STRUCTURE_EFFECT_GO",
        "TEMPORAL_STRUCTURE_EFFECT_NO_GO",
        {"contrasts": rows, "passing_contrast_count": passing},
        criteria,
    )


def gate_r1(
    head_quality: Mapping[str, Mapping[str, Any]],
    *,
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    manifest = _primary_manifest(primary_datasets)
    if set(map(str, head_quality)) != _HEADS:
        raise ValueError(f"R1 requires exactly the heads {sorted(_HEADS)}")
    observations: dict[str, Any] = {}
    qualifying: list[str] = []
    for head, values in head_quality.items():
        raw_gaps = _require_primary_mapping(
            values["relative_scrps_gap"],
            manifest=manifest,
            name=f"{head}.relative_scrps_gap",
        )
        gaps = _finite_mapping(raw_gaps, f"{head}.relative_scrps_gap")
        if any(gap < 0.0 for gap in gaps.values()):
            raise ValueError("relative sCRPS gaps cannot be negative")
        within_five = sum(gap <= 0.05 for gap in gaps.values())
        never_over_ten = all(gap <= 0.10 for gap in gaps.values())
        metric_best = bool(values["zero_brier_best"]) or bool(values["tail_sql_best"])
        qualifies = within_five >= 2 and never_over_ten and metric_best
        observations[str(head)] = {
            "relative_scrps_gap": gaps,
            "datasets_within_5pct": within_five,
            "never_over_10pct": never_over_ten,
            "best_zero_or_tail": metric_best,
            "qualifies": qualifies,
        }
        if qualifies:
            qualifying.append(str(head))
    criteria = {"at_least_two_quality_heads": len(qualifying) >= 2}
    return _result(
        "R1",
        all(criteria.values()),
        "REAL_TEACHER_QUALITY_GO",
        "REAL_TEACHER_QUALITY_NO_GO",
        {
            "primary_dataset_manifest": primary_datasets.as_dict(),
            "heads": observations,
            "qualifying_heads": qualifying,
        },
        criteria,
    )


def gate_r2(
    *,
    practical_winner_shares: Mapping[str, float],
    macro_oracle_gain: float,
    dataset_oracle_gains: Mapping[str, float],
    pairwise_correlation_ci: Mapping[str, Mapping[str, Any]],
    dataset_best_heads: Mapping[str, str],
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    manifest = _primary_manifest(primary_datasets)
    shares = _finite_mapping(practical_winner_shares, "practical_winner_shares")
    if set(shares) != _HEADS:
        raise ValueError(f"R2 practical winner shares require exactly {sorted(_HEADS)}")
    gains = _finite_mapping(
        _require_primary_mapping(
            dataset_oracle_gains,
            manifest=manifest,
            name="dataset_oracle_gains",
        ),
        "dataset_oracle_gains",
    )
    macro = _finite(macro_oracle_gain, "macro_oracle_gain")
    if set(map(str, pairwise_correlation_ci)) != _PAIR_KEYS:
        raise ValueError(f"R2 correlation pairs must be exactly {sorted(_PAIR_KEYS)}")
    correlation_rows: dict[str, dict[str, Any]] = {}
    for pair, raw in pairwise_correlation_ci.items():
        if "upper" not in raw or "degenerate_resample_present" not in raw:
            raise ValueError(f"R2 correlation summary for {pair} is incomplete")
        correlation_rows[str(pair)] = {
            "upper": _finite(raw["upper"], f"{pair}.upper"),
            "degenerate_resample_present": bool(raw["degenerate_resample_present"]),
        }
    best_heads = _require_primary_mapping(
        dataset_best_heads,
        manifest=manifest,
        name="dataset_best_heads",
    )
    if any(str(head) not in _HEADS for head in best_heads.values()):
        raise ValueError("R2 dataset best-head identity is not a registered head")
    best_heads = {dataset: str(head) for dataset, head in best_heads.items()}
    distinct_heads = sorted(set(best_heads.values()))
    criteria = {
        "at_least_two_practical_shares_at_least_15pct": sum(
            share >= 0.15 for share in shares.values()
        )
        >= 2,
        "macro_oracle_gain_at_least_2pct": macro >= 0.02,
        "at_least_two_datasets_oracle_gain_at_least_1pct": sum(
            gain >= 0.01 for gain in gains.values()
        )
        >= 2,
        "all_three_pairwise_correlation_upper95_below_0p99": all(
            row["upper"] < 0.99 and not row["degenerate_resample_present"]
            for row in correlation_rows.values()
        ),
        "at_least_two_dataset_level_best_head_identities": len(distinct_heads) >= 2,
    }
    return _result(
        "R2",
        all(criteria.values()),
        "REAL_HEAD_COMPLEMENTARITY_GO",
        "REAL_HEAD_COMPLEMENTARITY_NO_GO",
        {
            "practical_winner_shares": shares,
            "macro_oracle_gain": macro,
            "dataset_oracle_gains": gains,
            "primary_dataset_manifest": primary_datasets.as_dict(),
            "pairwise_correlation_ci": correlation_rows,
            "dataset_best_heads": best_heads,
            "distinct_dataset_best_heads": distinct_heads,
        },
        criteria,
    )


def gate_r3(
    *,
    macro_scrps_improvement: float,
    dataset_scrps_improvements: Mapping[str, float],
    tail_sql_improvement: float,
    q50_deterioration: float,
    zero_brier_deterioration: float,
    nrmse_deterioration: float,
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    manifest = _primary_manifest(primary_datasets)
    datasets = _finite_mapping(
        _require_primary_mapping(
            dataset_scrps_improvements,
            manifest=manifest,
            name="dataset_scrps_improvements",
        ),
        "dataset_scrps_improvements",
    )
    observations = {
        "macro_scrps_improvement": _finite(macro_scrps_improvement, "macro_scrps_improvement"),
        "primary_dataset_manifest": primary_datasets.as_dict(),
        "dataset_scrps_improvements": datasets,
        "mean_q95_q99_sql_improvement": _finite(tail_sql_improvement, "tail_sql_improvement"),
        "q50_deterioration": _finite(q50_deterioration, "q50_deterioration"),
        "zero_brier_deterioration": _finite(zero_brier_deterioration, "zero_brier_deterioration"),
        "nrmse_deterioration": _finite(nrmse_deterioration, "nrmse_deterioration"),
    }
    criteria = {
        "macro_scrps_improvement_at_least_1pct": observations["macro_scrps_improvement"] >= 0.01,
        "every_dataset_improvement_at_least_minus_0p5pct": all(value >= -0.005 for value in datasets.values()),
        "tail_sql_improvement_at_least_2pct": observations["mean_q95_q99_sql_improvement"] >= 0.02,
        "q50_deterioration_below_1pct": observations["q50_deterioration"] < 0.01,
        "zero_brier_deterioration_below_1pct": observations["zero_brier_deterioration"] < 0.01,
        "nrmse_deterioration_below_1pct": observations["nrmse_deterioration"] < 0.01,
    }
    return _result("R3", all(criteria.values()), "REAL_DISTRIBUTION_POOL_GO", "REAL_DISTRIBUTION_POOL_NO_GO", observations, criteria)


def gate_a1(
    *,
    scrps_improvement: float,
    tail_sql_improvement: float,
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    _primary_manifest(primary_datasets)
    observations = {
        "primary_dataset_manifest": primary_datasets.as_dict(),
        "scrps_improvement": _finite(scrps_improvement, "scrps_improvement"),
        "mean_q95_q99_sql_improvement": _finite(tail_sql_improvement, "tail_sql_improvement"),
    }
    criteria = {
        "scrps_improvement_at_least_1pct": observations["scrps_improvement"] >= 0.01,
        "tail_sql_improvement_at_least_2pct": observations["mean_q95_q99_sql_improvement"] >= 0.02,
    }
    return _result("A1", all(criteria.values()), "A_TEACHER_POOL_GO", "A_TEACHER_POOL_NO_GO", observations, criteria)


def gate_a2(
    *,
    best_single_loss: float,
    distilled_loss: float,
    pool_loss: float,
    improvement_vs_a0: float,
    macro_ci_lower: float,
    dataset_effects: Mapping[str, float],
    dataset_ci_lowers: Mapping[str, float],
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    manifest = _primary_manifest(primary_datasets)
    single = _finite(best_single_loss, "best_single_loss")
    student = _finite(distilled_loss, "distilled_loss")
    pool = _finite(pool_loss, "pool_loss")
    denominator = single - pool
    recovery = None if denominator <= 0.0 else (single - student) / denominator
    effects = _finite_mapping(
        _require_primary_mapping(
            dataset_effects,
            manifest=manifest,
            name="dataset_effects",
        ),
        "dataset_effects",
    )
    lowers = _finite_mapping(
        _require_primary_mapping(
            dataset_ci_lowers,
            manifest=manifest,
            name="dataset_ci_lowers",
        ),
        "dataset_ci_lowers",
    )
    observations = {
        "primary_dataset_manifest": primary_datasets.as_dict(),
        "best_single_loss": single,
        "distilled_loss": student,
        "teacher_pool_loss": pool,
        "recovery": recovery,
        "improvement_vs_a0": _finite(improvement_vs_a0, "improvement_vs_a0"),
        "macro_ci_lower": _finite(macro_ci_lower, "macro_ci_lower"),
        "dataset_effects": effects,
        "dataset_ci_lowers": lowers,
    }
    criteria = {
        "teacher_pool_has_positive_headroom": denominator > 0.0,
        "recovery_at_least_50pct": recovery is not None and recovery >= 0.50,
        "selected_a3_or_a4_improves_a0_at_least_0p5pct": observations["improvement_vs_a0"] >= 0.005,
        "macro_bootstrap_ci_lower_strictly_above_zero": observations["macro_ci_lower"] > 0.0,
        "at_least_two_positive_dataset_effects": sum(value > 0.0 for value in effects.values()) >= 2,
        "at_least_one_dataset_ci_lower_strictly_above_zero": sum(value > 0.0 for value in lowers.values()) >= 1,
    }
    return _result("A2", all(criteria.values()), "A_DISTILLATION_RECOVERY_GO", "A_DISTILLATION_RECOVERY_NO_GO", observations, criteria)


def gate_a3(
    deteriorations: Mapping[str, float],
    *,
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    _primary_manifest(primary_datasets)
    values = _finite_mapping(deteriorations, "deteriorations")
    required = {"zero_brier", "q50", "q99", "NRMSE", "coverage_90", "coverage_95"}
    if set(values) != required:
        raise ValueError(f"A3 deterioration metrics must be exactly {sorted(required)}")
    criteria = {
        f"{metric}_deterioration_at_most_1pct": value <= 0.01
        for metric, value in values.items()
    }
    return _result(
        "A3",
        all(criteria.values()),
        "A_DISTRIBUTION_TRADEOFF_GO",
        "A_DISTRIBUTION_TRADEOFF_NO_GO",
        {
            "primary_dataset_manifest": primary_datasets.as_dict(),
            "deteriorations": values,
        },
        criteria,
    )


def gate_a4(
    *,
    student_parameters: int,
    smallest_teacher_parameters: int,
    latency_by_device_batch: Mapping[str, Mapping[str, Mapping[str, float]]],
    cuda_peak_memory_by_batch: Mapping[str, Mapping[str, float]] | None,
    cuda_available: bool,
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    _primary_manifest(primary_datasets)
    if smallest_teacher_parameters <= 0 or student_parameters < 0:
        raise ValueError("parameter counts are invalid")
    required_batches = {"1", "256"}
    expected_devices = {"cpu", "cuda"} if cuda_available else {"cpu"}
    if set(latency_by_device_batch) != expected_devices:
        raise ValueError(
            f"A4 latency devices must be exactly {sorted(expected_devices)}"
        )
    latency: dict[str, dict[str, dict[str, float]]] = {}
    criteria: dict[str, bool] = {
        "student_parameters_positive": student_parameters > 0,
        "student_parameters_at_most_1p5x_smallest_teacher": student_parameters
        <= 1.5 * smallest_teacher_parameters,
        "cuda_available_for_preregistered_gpu_checks": bool(cuda_available),
    }
    for device in sorted(expected_devices):
        raw_batches = latency_by_device_batch[device]
        if set(raw_batches) != required_batches:
            raise ValueError(f"A4 {device} latency requires batch 1 and 256")
        latency[device] = {}
        for batch in sorted(required_batches):
            values = {
                str(key): _finite(value, f"latency.{device}.{batch}.{key}")
                for key, value in raw_batches[batch].items()
            }
            if set(values) != {"student", "single", "pool"}:
                raise ValueError("A4 latency fields are incomplete")
            if any(value <= 0.0 for value in values.values()):
                raise ValueError("A4 runtimes must be positive")
            latency[device][batch] = values
            criteria[
                f"student_latency_at_most_1p3x_single_{device}_batch_{batch}"
            ] = values["student"] <= 1.3 * values["single"]
            criteria[
                f"student_at_least_2x_faster_than_pool_{device}_batch_{batch}"
            ] = values["pool"] / values["student"] >= 2.0

    memory: dict[str, dict[str, float]] | None = None
    if cuda_available:
        if cuda_peak_memory_by_batch is None or set(cuda_peak_memory_by_batch) != required_batches:
            raise ValueError("A4 CUDA peak memory requires batch 1 and 256")
        memory = {}
        for batch in sorted(required_batches):
            values = {
                str(key): _finite(value, f"cuda_memory.{batch}.{key}")
                for key, value in cuda_peak_memory_by_batch[batch].items()
            }
            if set(values) != {"student", "pool"}:
                raise ValueError("A4 CUDA memory fields are incomplete")
            if any(value <= 0.0 for value in values.values()):
                raise ValueError("A4 CUDA memory values must be positive")
            memory[batch] = values
            criteria[
                f"student_cuda_memory_at_most_half_pool_batch_{batch}"
            ] = values["student"] <= 0.5 * values["pool"]
    elif cuda_peak_memory_by_batch is not None:
        raise ValueError("A4 unavailable CUDA must not contain fabricated memory measurements")

    return _result(
        "A4",
        all(criteria.values()),
        "A_COMPRESSION_VALUE_GO",
        "A_COMPRESSION_VALUE_NO_GO",
        {
            "primary_dataset_manifest": primary_datasets.as_dict(),
            "student_parameters": student_parameters,
            "smallest_teacher_parameters": smallest_teacher_parameters,
            "cuda_status": "AVAILABLE" if cuda_available else "UNAVAILABLE_EXPLICIT_GATE_FAIL",
            "latency_by_device_batch": latency,
            "cuda_peak_memory_by_batch": memory,
        },
        criteria,
    )


def gate_b1(
    *,
    regret_spearman_by_dataset: Mapping[str, float | None],
    undefined_reasons_by_dataset: Mapping[str, str],
    extended_minus_baseline: float,
    real_increment: float,
    shuffled_increment: float,
    cross_dataset_effects: Mapping[str, float],
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    manifest = _primary_manifest(primary_datasets)
    spearman, undefined_reasons = _optional_finite_primary_mapping(
        regret_spearman_by_dataset,
        manifest=manifest,
        reasons=undefined_reasons_by_dataset,
        allowed_tokens={"INSUFFICIENT_VARIATION"},
        name="regret_spearman_by_dataset",
    )
    effects = _finite_mapping(
        _require_primary_mapping(
            cross_dataset_effects,
            manifest=manifest,
            name="cross_dataset_effects",
        ),
        "cross_dataset_effects",
    )
    extended = _finite(extended_minus_baseline, "extended_minus_baseline")
    real = _finite(real_increment, "real_increment")
    shuffled = _finite(shuffled_increment, "shuffled_increment")
    criteria = {
        "at_least_two_datasets_spearman_at_least_0p20": sum(
            value is not None and value >= 0.20 for value in spearman.values()
        )
        >= 2,
        "extended_over_baseline_at_least_0p08": extended >= 0.08,
        "shuffle_increment_at_most_25pct_of_real": real > 0.0 and shuffled <= 0.25 * real,
        "cross_dataset_effect_sign_maintained": all(value > 0.0 for value in effects.values()),
    }
    return _result(
        "B1",
        all(criteria.values()),
        "STRUCTURE_REGRET_PREDICTABILITY_GO",
        "STRUCTURE_REGRET_PREDICTABILITY_NO_GO",
        {
            "primary_dataset_manifest": primary_datasets.as_dict(),
            "regret_spearman_by_dataset": spearman,
            "undefined_reasons_by_dataset": undefined_reasons,
            "extended_minus_baseline": extended,
            "real_increment": real,
            "shuffled_increment": shuffled,
            "cross_dataset_effects": effects,
        },
        criteria,
    )


def gate_b2(
    *,
    macro_scrps_improvement: float,
    dataset_scrps_improvements: Mapping[str, float],
    macro_ci_lower: float,
    improvement_over_b1: float,
    q99_deterioration: float,
    zero_brier_deterioration: float,
    worst_origin_improvement: float,
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    manifest = _primary_manifest(primary_datasets)
    datasets = _finite_mapping(
        _require_primary_mapping(
            dataset_scrps_improvements,
            manifest=manifest,
            name="dataset_scrps_improvements",
        ),
        "dataset_scrps_improvements",
    )
    observations = {
        "primary_dataset_manifest": primary_datasets.as_dict(),
        "macro_scrps_improvement": _finite(macro_scrps_improvement, "macro_scrps_improvement"),
        "dataset_scrps_improvements": datasets,
        "macro_ci_lower": _finite(macro_ci_lower, "macro_ci_lower"),
        "improvement_over_b1": _finite(improvement_over_b1, "improvement_over_b1"),
        "q99_deterioration": _finite(q99_deterioration, "q99_deterioration"),
        "zero_brier_deterioration": _finite(
            zero_brier_deterioration, "zero_brier_deterioration"
        ),
        "worst_origin_improvement": _finite(worst_origin_improvement, "worst_origin_improvement"),
    }
    criteria = {
        "macro_improvement_at_least_0p5pct": observations["macro_scrps_improvement"] >= 0.005,
        "every_dataset_improvement_at_least_minus_0p25pct": all(
            value >= -0.0025 for value in datasets.values()
        ),
        "macro_ci_lower_strictly_above_zero": observations["macro_ci_lower"] > 0.0,
        "at_least_two_positive_datasets": sum(value > 0.0 for value in datasets.values()) >= 2,
        "improvement_over_b1_at_least_0p2pct": observations["improvement_over_b1"] >= 0.002,
        "q99_deterioration_below_1pct": observations["q99_deterioration"] < 0.01,
        "zero_brier_deterioration_below_1pct": observations["zero_brier_deterioration"] < 0.01,
        "worst_origin_improvement_at_least_minus_0p5pct": observations[
            "worst_origin_improvement"
        ]
        >= -0.005,
    }
    return _result(
        "B2",
        all(criteria.values()),
        "STRUCTURE_CONDITIONED_DISTILLATION_EFFECT_GO",
        "STRUCTURE_CONDITIONED_DISTILLATION_EFFECT_NO_GO",
        observations,
        criteria,
    )


def gate_c1(
    dataset_metrics: Mapping[str, Mapping[str, float | None]],
    *,
    undefined_reasons_by_dataset: Mapping[str, str],
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    manifest = _primary_manifest(primary_datasets)
    dataset_metrics = _require_primary_mapping(
        dataset_metrics,
        manifest=manifest,
        name="dataset_metrics",
    )
    rows: dict[str, Any] = {}
    qualifying: list[str] = []
    required = {
        "auroc",
        "auprc",
        "c2_minus_c0_auprc",
        "c2_minus_c3_auprc",
        "c2_brier",
        "c0_brier",
    }
    for dataset, raw in dataset_metrics.items():
        if set(raw) != required:
            raise ValueError(f"C1 fields for {dataset} must be exactly {sorted(required)}")
        values = {
            key: None if value is None else _finite(value, f"{dataset}.{key}")
            for key, value in raw.items()
        }
        undefined = any(value is None for value in values.values())
        checks = (
            {
                "auroc_at_least_0p70": False,
                "auprc_at_least_0p35": False,
                "c2_over_c0_auprc_at_least_0p05": False,
                "c2_over_c3_auprc_at_least_0p02": False,
                "brier_not_worse_than_c0": False,
            }
            if undefined
            else {
                "auroc_at_least_0p70": values["auroc"] >= 0.70,
                "auprc_at_least_0p35": values["auprc"] >= 0.35,
                "c2_over_c0_auprc_at_least_0p05": values["c2_minus_c0_auprc"] >= 0.05,
                "c2_over_c3_auprc_at_least_0p02": values["c2_minus_c3_auprc"] >= 0.02,
                "brier_not_worse_than_c0": values["c2_brier"] <= values["c0_brier"],
            }
        )
        rows[str(dataset)] = {
            "metrics": values,
            "criteria": checks,
            "qualifies": all(checks.values()),
        }
        if all(checks.values()):
            qualifying.append(str(dataset))
    undefined_datasets = [
        dataset
        for dataset in manifest
        if any(value is None for value in rows[dataset]["metrics"].values())
    ]
    undefined_reasons = _validate_undefined_reasons(
        undefined_datasets,
        undefined_reasons_by_dataset,
        allowed_tokens={"SINGLE_CLASS", "EMPTY_FAILURE_SET"},
        name="C1 dataset_metrics",
    )
    for dataset, reason in undefined_reasons.items():
        rows[dataset]["undefined_reason"] = reason
    criteria = {
        "at_least_two_real_datasets_meet_all_detection_thresholds": len(qualifying)
        >= 2
    }
    return _result(
        "C1",
        all(criteria.values()),
        "DISAGREEMENT_FAILURE_DETECTION_GO",
        "DISAGREEMENT_FAILURE_DETECTION_NO_GO",
        {
            "primary_dataset_manifest": primary_datasets.as_dict(),
            "datasets": rows,
            "qualifying_datasets": qualifying,
            "undefined_reasons_by_dataset": undefined_reasons,
        },
        criteria,
    )


def gate_c2(
    *,
    auprc: float,
    false_alarm_rate: float,
    median_delay_horizons: float,
    no_change_false_positive: float,
    component_separation: bool,
) -> GateResult:
    observations = {
        "change_vs_no_change_auprc": _finite(auprc, "auprc"),
        "false_alarm_rate": _finite(false_alarm_rate, "false_alarm_rate"),
        "median_delay_horizons": _finite(median_delay_horizons, "median_delay_horizons"),
        "no_change_false_positive": _finite(
            no_change_false_positive, "no_change_false_positive"
        ),
        "component_separation": bool(component_separation),
    }
    criteria = {
        "auprc_at_least_0p70": observations["change_vs_no_change_auprc"] >= 0.70,
        "false_alarm_rate_at_most_10pct": observations["false_alarm_rate"] <= 0.10,
        "median_delay_at_most_one_horizon": observations["median_delay_horizons"] <= 1.0,
        "no_change_false_positive_at_most_10pct": observations["no_change_false_positive"] <= 0.10,
        "component_wise_interval_magnitude_separation": observations["component_separation"],
    }
    return _result("C2", all(criteria.values()), "DISAGREEMENT_CHANGE_SENSOR_GO", "DISAGREEMENT_CHANGE_SENSOR_NO_GO", observations, criteria)


def gate_c3(
    *,
    worst_decile_scrps_improvement: float,
    coverage_error_reductions: Mapping[str, float],
    mean_scrps_deterioration: float,
    selective_coverage: float,
    false_alarm_dataset_deteriorations: Mapping[str, float | None],
    undefined_reasons_by_dataset: Mapping[str, str],
    primary_datasets: FrozenPrimaryDatasetManifest,
) -> GateResult:
    manifest = _primary_manifest(primary_datasets)
    coverage = _finite_mapping(coverage_error_reductions, "coverage_error_reductions")
    if set(coverage) != {"90", "95"}:
        raise ValueError("C3 requires both 90 and 95 coverage reductions")
    false_alarm, undefined_reasons = _optional_finite_primary_mapping(
        false_alarm_dataset_deteriorations,
        manifest=manifest,
        reasons=undefined_reasons_by_dataset,
        allowed_tokens={"EMPTY_FAILURE_SET"},
        name="false_alarm_dataset_deteriorations",
    )
    observations = {
        "primary_dataset_manifest": primary_datasets.as_dict(),
        "worst_decile_scrps_improvement": _finite(worst_decile_scrps_improvement, "worst_decile_scrps_improvement"),
        "coverage_error_reductions": coverage,
        "mean_scrps_deterioration": _finite(mean_scrps_deterioration, "mean_scrps_deterioration"),
        "selective_coverage": _finite(selective_coverage, "selective_coverage"),
        "false_alarm_dataset_deteriorations": false_alarm,
        "undefined_reasons_by_dataset": undefined_reasons,
    }
    criteria = {
        "worst_decile_scrps_improvement_at_least_10pct": observations["worst_decile_scrps_improvement"] >= 0.10,
        "both_coverage_error_reductions_at_least_15pct": all(value >= 0.15 for value in coverage.values()),
        "mean_scrps_deterioration_below_0p5pct": observations["mean_scrps_deterioration"] < 0.005,
        "selective_coverage_at_least_80pct": observations["selective_coverage"] >= 0.80,
        "every_false_alarm_dataset_deterioration_below_0p5pct": all(
            value is not None and value < 0.005 for value in false_alarm.values()
        ),
    }
    return _result("C3", all(criteria.values()), "DISAGREEMENT_ACTION_VALUE_GO", "DISAGREEMENT_ACTION_VALUE_NO_GO", observations, criteria)


_CONTROL_REGISTRY: dict[str, dict[str, tuple[str, str]]] = {
    "A": {
        "teacher_identity_shuffle": ("effect_destroying", "branch_real_effect"),
        "teacher_quantile_shuffle": ("effect_destroying", "branch_real_effect"),
        "single_teacher": ("diagnostic", "branch_real_effect"),
    },
    "B": {
        "regret_label_shuffle": ("effect_destroying", "branch_real_effect"),
        "temporal_feature_row_shuffle": ("effect_destroying", "branch_real_effect"),
        "remove_missing_indicators": ("diagnostic", "branch_real_effect"),
    },
    "C": {
        "time_shuffle": ("effect_destroying", "target1_auprc_gain"),
        "teacher_name_permutation": ("invariance", "exact_invariance_zero"),
        "scale_only": ("effect_destroying", "target1_auprc_gain"),
        "random_score": ("effect_destroying", "target1_auprc_gain"),
        "no_change": ("effect_destroying", "synthetic_change_signal"),
    },
}

_CONTROL_REFERENCE_MANIFEST = {
    "A": {"branch_real_effect"},
    "B": {"branch_real_effect"},
    "C": {"target1_auprc_gain", "synthetic_change_signal"},
}


def gate_negative_control(
    *,
    branch: str,
    primary_datasets: FrozenPrimaryDatasetManifest,
    reference_effects: Mapping[str, float],
    control_effects: Mapping[str, float],
    invariance_differences: Mapping[str, float],
) -> GateResult:
    _primary_manifest(primary_datasets)
    branch = str(branch)
    if branch not in _CONTROL_REGISTRY:
        raise ValueError(f"negative-control branch must be one of {sorted(_CONTROL_REGISTRY)}")
    controls = _finite_mapping(control_effects, "control_effects")
    registry = _CONTROL_REGISTRY[branch]
    if set(controls) != set(registry):
        raise ValueError(
            f"control effects must match the fixed registry for branch {branch}: {sorted(registry)}"
        )
    references = _finite_mapping(reference_effects, "reference_effects")
    if set(references) != _CONTROL_REFERENCE_MANIFEST[branch]:
        raise ValueError(
            f"reference effects must match the fixed registry for branch {branch}: "
            f"{sorted(_CONTROL_REFERENCE_MANIFEST[branch])}"
        )
    roles = {name: values[0] for name, values in registry.items()}
    expected_invariance = {name for name, role in roles.items() if role == "invariance"}
    differences = _finite_mapping(invariance_differences, "invariance_differences") if invariance_differences else {}
    if set(differences) != expected_invariance:
        raise ValueError(
            f"invariance differences must match the fixed registry for branch {branch}: "
            f"{sorted(expected_invariance)}"
        )
    eligible = {name: value for name, value in controls.items() if roles[name] == "effect_destroying"}
    if not eligible:
        raise ValueError("fixed control registry must include an effect-destroying control")
    reference_valid = all(value > 0.0 for value in references.values())
    recoveries: dict[str, float | None] = {}
    control_references: dict[str, dict[str, Any]] = {}
    for name, value in controls.items():
        role, reference_name = registry[name]
        reference_value = 0.0 if role == "invariance" else references[reference_name]
        control_references[name] = {
            "name": reference_name,
            "value": reference_value,
        }
        if role == "effect_destroying":
            recoveries[name] = (
                value / reference_value if reference_value > 0.0 else None
            )
    below = reference_valid and all(
        value is not None and value < 0.50 for value in recoveries.values()
    )
    invariant = all(abs(value) <= 1e-12 for value in differences.values())
    criteria = {
        "all_control_reference_effects_positive": reference_valid,
        "all_controls_below_half": below,
        "invariance_controls_equal_within_1e_minus_12": invariant,
    }
    return _result(
        f"CONTROL_{branch}",
        all(criteria.values()),
        "SIGNAL_IDENTIFIED_GO",
        "SIGNAL_IDENTIFICATION_FAILURE",
        {
            "branch": branch,
            "primary_dataset_manifest": primary_datasets.as_dict(),
            "reference_effects": references,
            "control_effects": controls,
            "control_roles": dict(roles),
            "control_references": control_references,
            "identification_eligible_controls": sorted(eligible),
            "recovery_fractions": recoveries,
            "invariance_differences": differences,
            "invariance_tolerance": 1e-12,
        },
        criteria,
    )


def _all_pass(
    gates: Mapping[str, GateResult],
    required: Sequence[str],
    *,
    real_manifest_gates: Sequence[str] = (),
    additional_real_manifest_results: Sequence[GateResult] = (),
) -> bool:
    missing = set(required).difference(gates)
    if missing:
        raise ValueError(f"missing required gates: {sorted(missing)}")
    for name in required:
        if gates[name].gate != name:
            raise ValueError(f"gate ledger key {name} contains result for {gates[name].gate}")
    if not set(real_manifest_gates).issubset(required):
        raise ValueError("real manifest gates must be a subset of required gates")
    manifests: list[FrozenPrimaryDatasetManifest] = []
    manifest_results = [gates[name] for name in real_manifest_gates]
    manifest_results.extend(additional_real_manifest_results)
    for result in manifest_results:
        record = result.observations.get("primary_dataset_manifest")
        if record is None:
            return False
        try:
            manifests.append(FrozenPrimaryDatasetManifest.from_record(_thaw(record)))
        except (TypeError, ValueError):
            return False
    if manifests and any(
        manifest.audit_sha256 != manifests[0].audit_sha256
        for manifest in manifests[1:]
    ):
        raise ValueError(
            "all real-data gates must carry an identical frozen primary dataset manifest"
        )
    return all(
        gates[name].passed and gates[name].confirmatory_eligible for name in required
    )


def _control_passes(control: GateResult, *, branch: str) -> bool:
    if control.gate != f"CONTROL_{branch}":
        raise ValueError(
            f"identification control must be the CONTROL_{branch} gate result"
        )
    if str(control.observations.get("branch", "")) != branch:
        raise ValueError(f"identification control branch must be {branch}")
    return control.passed and control.confirmatory_eligible


def combine_head_verdict(gates: Mapping[str, GateResult]) -> str:
    passed = _all_pass(gates, ("DGP_BALANCE", "S1", "S2", "S3"))
    return "HEAD_SPECIALIZATION_GO" if passed else "HEAD_SPECIALIZATION_NO_GO"


def combine_real_verdict(
    gates: Mapping[str, GateResult],
    *,
    tweedie_valid: bool,
) -> str:
    passed = bool(tweedie_valid) and _all_pass(
        gates,
        ("R1", "R2", "R3"),
        real_manifest_gates=("R1", "R2", "R3"),
    )
    return "REAL_DISTRIBUTION_POOL_GO" if passed else "REAL_DISTRIBUTION_POOL_NO_GO"


def combine_a_verdict(
    gates: Mapping[str, GateResult],
    *,
    tweedie_valid: bool,
    identification_control: GateResult,
) -> str:
    scientific_pass = _all_pass(
        gates,
        ("R1", "R2", "R3", "A1", "A2", "A3", "A4"),
        real_manifest_gates=("R1", "R2", "R3", "A1", "A2", "A3", "A4"),
        additional_real_manifest_results=(identification_control,),
    )
    passed = bool(tweedie_valid) and _control_passes(
        identification_control, branch="A"
    ) and scientific_pass
    return "DISTRIBUTION_SPACE_DISTILLATION_GO" if passed else "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"


def combine_b_verdict(
    gates: Mapping[str, GateResult],
    *,
    tweedie_valid: bool,
    identification_control: GateResult,
) -> str:
    scientific_pass = _all_pass(
        gates,
        ("R1", "R2", "B1", "B2"),
        real_manifest_gates=("R1", "R2", "B1", "B2"),
        additional_real_manifest_results=(identification_control,),
    )
    passed = bool(tweedie_valid) and _control_passes(
        identification_control, branch="B"
    ) and scientific_pass
    return "STRUCTURE_CONDITIONED_ROUTING_GO" if passed else "STRUCTURE_CONDITIONED_ROUTING_NO_GO"


def combine_c_verdict(
    gates: Mapping[str, GateResult],
    *,
    tweedie_valid: bool,
    identification_control: GateResult,
) -> str:
    scientific_pass = _all_pass(
        gates,
        ("R1", "C1", "C2", "C3"),
        real_manifest_gates=("R1", "C1", "C3"),
        additional_real_manifest_results=(identification_control,),
    )
    passed = bool(tweedie_valid) and _control_passes(
        identification_control, branch="C"
    ) and scientific_pass
    return "DISAGREEMENT_SENSOR_GO" if passed else "DISAGREEMENT_SENSOR_NO_GO"


def final_recommendation(
    *,
    integrity_blocked: bool = False,
    a_go: bool = False,
    a_confirmatory: bool = False,
    c_go: bool = False,
    c_confirmatory: bool = False,
    b_go: bool = False,
    b_confirmatory: bool = False,
    synthetic_temporal_effect: bool = False,
    real_head_specialization: bool = False,
) -> str:
    """Apply the fixed A -> C -> B -> characterization -> no-go priority."""

    if integrity_blocked:
        return "INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT"
    if a_go and a_confirmatory:
        return "RECOMMEND_A_DISTRIBUTION_DISTILLATION"
    if c_go and c_confirmatory:
        return "RECOMMEND_C_DISAGREEMENT_SENSOR"
    if b_go and b_confirmatory:
        return "RECOMMEND_B_STRUCTURE_CONDITIONED_ROUTING"
    if synthetic_temporal_effect or real_head_specialization:
        return "RECOMMEND_CHARACTERIZATION_ONLY"
    return "ALL_NEW_METHOD_BRANCHES_NO_GO"
