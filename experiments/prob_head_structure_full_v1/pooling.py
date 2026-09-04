"""Validation-only distribution-space pooling utilities."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation import (
    CRPS_QUANTILE_GRID,
    EVALUATION_QUANTILE_GRID,
    PredictionIntegrityError,
    ScientificForecastProvenance,
    SealedEvaluationTarget,
    _seal_scientific_forecast_artifact,
    approximate_crps,
    coverage_quantiles_from_common_grid,
    midpoint_cell_widths,
    pinball_loss,
    quantile_column,
    quantile_implied_mean,
)
from .integrity import BranchEligibility


POOL_PENALTIES = (0.0, 0.01, 0.1, 1.0)
HEAD_ORDER = ("NB", "HSNB", "TWEEDIE_FULL")
POOLED_FLAT_CASE_CHUNK = 448
POOLED_Q_CHUNK = 1


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_component(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_component(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        array = None
    if array is not None:
        if array.dtype.kind not in "biuf":
            raise PredictionIntegrityError(
                "teacher prediction component must be a numeric array"
            )
        numeric = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise PredictionIntegrityError(
                "teacher prediction component contains NaN/Inf"
            )
        return {"shape": list(numeric.shape), "values": numeric.tolist()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise PredictionIntegrityError(
                "teacher prediction component contains NaN/Inf"
            )
        return numeric
    raise PredictionIntegrityError("teacher prediction component type is unsupported")


def _component_sha256(value: Any) -> str:
    """Hash numeric forecast components without constructing Python value lists."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key, nested in sorted(item.items(), key=lambda pair: str(pair[0])):
                encoded_key = str(key).encode("utf-8")
                digest.update(len(encoded_key).to_bytes(8, "big"))
                digest.update(encoded_key)
                update(nested)
            digest.update(b"}")
            return
        if hasattr(item, "detach"):
            item = item.detach()
        if hasattr(item, "cpu"):
            item = item.cpu()
        if hasattr(item, "numpy"):
            item = item.numpy()
        if isinstance(item, (np.ndarray, list, tuple)):
            array = np.asarray(item)
            if array.dtype.kind not in "biuf":
                raise PredictionIntegrityError(
                    "teacher prediction component must be a numeric array"
                )
            numeric = np.ascontiguousarray(array, dtype="<f8")
            if not np.all(np.isfinite(numeric)):
                raise PredictionIntegrityError(
                    "teacher prediction component contains NaN/Inf"
                )
            metadata = json.dumps(
                {"dtype": "<f8", "shape": list(numeric.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(b"array")
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            byte_view = memoryview(numeric).cast("B")
            for start in range(0, len(byte_view), 1 << 20):
                digest.update(byte_view[start : start + (1 << 20)])
            return
        if isinstance(item, (bool, np.bool_)):
            digest.update(b"true" if bool(item) else b"false")
            return
        if isinstance(item, (int, float, np.integer, np.floating)):
            numeric = float(item)
            if not np.isfinite(numeric):
                raise PredictionIntegrityError(
                    "teacher prediction component contains NaN/Inf"
                )
            digest.update(b"number")
            digest.update(np.asarray([numeric], dtype="<f8").tobytes())
            return
        raise PredictionIntegrityError("teacher prediction component type is unsupported")

    digest.update(b"prob_head_structure_full_v1.component.v2\0")
    update(value)
    return digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> tuple[str, str]:
    canonical = json.dumps(
        deepcopy(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class SealedValidationArtifact:
    """Validation-only target and teacher prediction identity for pool selection."""

    target_artifact: SealedEvaluationTarget
    canonical_payload_json: str
    artifact_sha256: str

    @classmethod
    def seal(
        cls,
        *,
        target_artifact: SealedEvaluationTarget,
        head_order: Sequence[str],
        teacher_predictions: Mapping[str, Mapping[str, Any]],
        teacher_case_keys: Mapping[str, Sequence[Sequence[Any]]],
        teacher_model_seeds: Mapping[str, Sequence[int]],
        validation_group_ids: Sequence[Any],
        teacher_quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
        sample_manifest_sha256: str,
        source_manifest_sha256: str,
        preregistration_sha256: str,
        dataset_manifest_sha256: str,
    ) -> "SealedValidationArtifact":
        if not isinstance(target_artifact, SealedEvaluationTarget):
            raise PredictionIntegrityError(
                "validation selection requires a sealed evaluation target"
            )
        if tuple(head_order) != HEAD_ORDER:
            raise PredictionIntegrityError(
                f"validation artifact requires canonical head order {HEAD_ORDER}"
            )
        if (
            tuple(teacher_predictions) != HEAD_ORDER
            or tuple(teacher_case_keys) != HEAD_ORDER
            or tuple(teacher_model_seeds) != HEAD_ORDER
        ):
            raise PredictionIntegrityError(
                "teacher predictions and seeds must use the canonical head order"
            )
        component_hashes: dict[str, dict[str, str]] = {}
        seed_rows: dict[str, list[int]] = {}
        for head in HEAD_ORDER:
            components = teacher_predictions[head]
            if not isinstance(components, Mapping) or not components:
                raise PredictionIntegrityError(
                    f"validation artifact has no prediction components for {head}"
                )
            component_hashes[head] = {
                str(name): _component_sha256(value)
                for name, value in sorted(components.items(), key=lambda item: str(item[0]))
            }
            seeds = tuple(teacher_model_seeds[head])
            if (
                not seeds
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
                or len(set(seeds)) != len(seeds)
            ):
                raise PredictionIntegrityError(
                    "teacher model seeds must be nonempty unique integer lists"
                )
            seed_rows[head] = [int(seed) for seed in seeds]
        if any(seed_rows[head] != seed_rows[HEAD_ORDER[0]] for head in HEAD_ORDER[1:]):
            raise PredictionIntegrityError(
                "every teacher must use the identical ordered model-seed manifest"
            )
        hashes = {
            "sample_manifest_sha256": sample_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "preregistration_sha256": preregistration_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
        }
        if not all(_valid_sha256(value) for value in hashes.values()):
            raise PredictionIntegrityError(
                "validation artifact provenance hashes must be lowercase SHA256"
            )
        target_record = target_artifact.as_dict()
        target_payload = target_record["payload"]
        if (
            target_payload["sample_manifest_sha256"] != sample_manifest_sha256
            or
            target_payload["source_manifest_sha256"] != source_manifest_sha256
            or target_payload["preregistration_sha256"] != preregistration_sha256
            or target_payload["dataset_manifest_sha256"] != dataset_manifest_sha256
        ):
            raise PredictionIntegrityError(
                "validation artifact provenance differs from its target artifact"
            )
        rows = target_payload["rows"]
        if not all(row["target_mask"] is True for row in rows):
            raise PredictionIntegrityError(
                "validation selector artifact requires exact complete valid case rows"
            )
        case_keys = [
            [row["dataset_id"], row["series_id"], row["origin"], row["step"]]
            for row in rows
        ]
        for head in HEAD_ORDER:
            if [list(row) for row in teacher_case_keys[head]] != case_keys:
                raise PredictionIntegrityError(
                    f"{head} forecast keys/order differ from the sealed validation target"
                )
        if tuple(map(float, teacher_quantile_grid)) != CRPS_QUANTILE_GRID:
            raise PredictionIntegrityError("teacher quantile grid is not the frozen common grid")
        _, case_key_sha = _canonical_digest({"case_keys": case_keys})
        _, target_sha = _canonical_digest(
            {"y": [row["y"] for row in rows], "scale": [row["scale"] for row in rows]}
        )
        groups = list(validation_group_ids)
        if len(groups) != len(case_keys) or any(
            value is None
            or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
            for value in groups
        ):
            raise PredictionIntegrityError(
                "validation grouping must provide one finite sealed ID per case"
            )
        try:
            set(groups)
        except TypeError as error:
            raise PredictionIntegrityError("validation grouping IDs must be hashable") from error
        payload = {
            "schema": "prob_head_structure_full_v1.validation_selection.v1",
            "head_order": list(HEAD_ORDER),
            "case_key_columns": ["dataset_id", "series_id", "origin", "step"],
            "case_keys": case_keys,
            "case_key_sha256": case_key_sha,
            "target_scale_sha256": target_sha,
            "validation_group_ids": deepcopy(groups),
            "validation_group_sha256": _canonical_digest(
                {"validation_group_ids": groups}
            )[1],
            "split_name": target_artifact.split_name,
            "split_bounds": list(target_artifact.split_bounds),
            "target_artifact_sha256": target_artifact.artifact_sha256,
            "teacher_prediction_component_sha256": component_hashes,
            "teacher_model_seeds": seed_rows,
            "teacher_case_key_sha256": {head: case_key_sha for head in HEAD_ORDER},
            "teacher_quantile_grid": list(CRPS_QUANTILE_GRID),
            **hashes,
        }
        canonical, digest = _canonical_digest(payload)
        instance = object.__new__(cls)
        object.__setattr__(instance, "target_artifact", target_artifact)
        object.__setattr__(instance, "canonical_payload_json", canonical)
        object.__setattr__(instance, "artifact_sha256", digest)
        return instance

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        window_batch: Any,
        window_request: Any,
        panel: Mapping[str, Any],
        dataset_audit: Mapping[str, Any],
        source_manifest: Mapping[str, Any],
        sample_manifest: Mapping[str, Any],
    ) -> "SealedValidationArtifact":
        if not isinstance(record, Mapping) or set(record) != {
            "payload", "artifact_sha256", "target_artifact"
        }:
            raise PredictionIntegrityError("validation artifact record is malformed")
        target = SealedEvaluationTarget.from_record(
            record["target_artifact"],
            window_batch=window_batch,
            window_request=window_request,
            panel=panel,
            dataset_audit=dataset_audit,
            source_manifest=source_manifest,
            sample_manifest=sample_manifest,
        )
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise PredictionIntegrityError("validation artifact payload is malformed")
        # Rebuild the immutable shell only after validating both nested and outer digests.
        canonical, digest = _canonical_digest(payload)
        if digest != record.get("artifact_sha256"):
            raise PredictionIntegrityError("validation artifact digest mismatch")
        if (
            payload.get("target_artifact_sha256") != target.artifact_sha256
            or payload.get("head_order") != list(HEAD_ORDER)
            or payload.get("split_name") != "validation"
        ):
            raise PredictionIntegrityError("validation artifact binding is inconsistent")
        instance = object.__new__(cls)
        object.__setattr__(instance, "target_artifact", target)
        object.__setattr__(instance, "canonical_payload_json", canonical)
        object.__setattr__(instance, "artifact_sha256", digest)
        if instance.as_dict() != deepcopy(dict(record)):
            raise PredictionIntegrityError("validation artifact record is not canonical")
        return instance

    def verify_teacher_components(
        self, components: Mapping[str, Mapping[str, Any]]
    ) -> None:
        if tuple(components) != HEAD_ORDER:
            raise PredictionIntegrityError(
                "teacher component input has a head-axis permutation"
            )
        frozen = json.loads(self.canonical_payload_json)[
            "teacher_prediction_component_sha256"
        ]
        for head in HEAD_ORDER:
            if not isinstance(components[head], Mapping) or not components[head]:
                raise PredictionIntegrityError("teacher prediction component is missing")
            for name, value in components[head].items():
                if name not in frozen[head] or _component_sha256(value) != frozen[head][name]:
                    raise PredictionIntegrityError(
                        f"teacher prediction component hash mismatch for {head}.{name}"
                    )

    def verify_teacher_component(self, head: str, name: str, value: Any) -> None:
        if head not in HEAD_ORDER:
            raise PredictionIntegrityError("teacher component has an unregistered head")
        frozen = json.loads(self.canonical_payload_json)[
            "teacher_prediction_component_sha256"
        ][head]
        if set(frozen).difference({str(name)}) == set(frozen):
            raise PredictionIntegrityError(
                f"teacher prediction component is not sealed for {head}.{name}"
            )
        if frozen[str(name)] != _component_sha256(value):
            raise PredictionIntegrityError(
                f"teacher prediction component hash mismatch for {head}.{name}"
            )

    @property
    def model_seeds(self) -> tuple[int, ...]:
        payload = json.loads(self.canonical_payload_json)
        return tuple(int(seed) for seed in payload["teacher_model_seeds"][HEAD_ORDER[0]])

    def require_validation_split(self) -> None:
        payload = json.loads(self.canonical_payload_json)
        if self.target_artifact.split_name != "validation" or payload.get(
            "split_name"
        ) != "validation":
            raise PredictionIntegrityError(
                "confirmatory pool selection requires a sealed validation split artifact"
            )

    @property
    def validation_group_ids(self) -> tuple[Any, ...]:
        return tuple(json.loads(self.canonical_payload_json)["validation_group_ids"])

    def verify_target_arrays(
        self,
        *,
        validation_y: np.ndarray,
        validation_scale: np.ndarray,
        validation_case_keys: Sequence[Sequence[Any]],
    ) -> None:
        payload = json.loads(self.canonical_payload_json)
        y = np.asarray(validation_y, dtype=np.float64)
        scale = np.asarray(validation_scale, dtype=np.float64)
        if y.ndim != 1 or scale.shape != y.shape:
            raise PredictionIntegrityError("validation target arrays have invalid shapes")
        if (
            not np.all(np.isfinite(y))
            or np.any(y < 0.0)
            or not np.array_equal(y, np.rint(y))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise PredictionIntegrityError(
                "validation target/scale contains invalid support or NaN/Inf"
            )
        keys = [list(item) for item in validation_case_keys]
        if keys != payload["case_keys"]:
            raise PredictionIntegrityError(
                "validation case keys/order do not match the sealed artifact"
            )
        _, target_sha = _canonical_digest({"y": y.tolist(), "scale": scale.tolist()})
        if target_sha != payload["target_scale_sha256"]:
            raise PredictionIntegrityError(
                "validation target/scale do not match the sealed artifact"
            )

    def verify_target_y(
        self,
        *,
        validation_y: np.ndarray,
        validation_case_keys: Sequence[Sequence[Any]],
    ) -> None:
        payload = json.loads(self.canonical_payload_json)
        y = np.asarray(validation_y, dtype=np.float64)
        keys = [list(item) for item in validation_case_keys]
        frozen_y = np.asarray(
            [row["y"] for row in self.target_artifact.as_dict()["payload"]["rows"]],
            dtype=np.float64,
        )
        if y.shape != frozen_y.shape or not np.array_equal(y, frozen_y):
            raise PredictionIntegrityError(
                "validation target does not match the sealed artifact"
            )
        if keys != payload["case_keys"]:
            raise PredictionIntegrityError(
                "validation case keys/order do not match the sealed artifact"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "payload": json.loads(self.canonical_payload_json),
            "artifact_sha256": self.artifact_sha256,
            "target_artifact": self.target_artifact.as_dict(),
        }


def _seal_selection(
    selection: Mapping[str, Any], artifact: SealedValidationArtifact
) -> dict[str, Any]:
    payload = {
        **deepcopy(dict(selection)),
        "validation_artifact_sha256": artifact.artifact_sha256,
    }
    _, digest = _canonical_digest(payload)
    payload["selection_sha256"] = digest
    return payload


def _verify_selection_binding(
    selection: Mapping[str, Any], artifact: SealedValidationArtifact
) -> dict[str, Any]:
    if not isinstance(artifact, SealedValidationArtifact) or not isinstance(selection, Mapping):
        raise PredictionIntegrityError("a sealed pool selection and validation artifact are required")
    payload = deepcopy(dict(selection))
    digest = payload.pop("selection_sha256", None)
    _, actual = _canonical_digest(payload)
    if (
        not _valid_sha256(digest)
        or digest != actual
        or payload.get("validation_artifact_sha256") != artifact.artifact_sha256
    ):
        raise PredictionIntegrityError("pool selection binding or digest mismatch")
    return payload


def simplex_grid(step: float = 0.1, components: int = 3) -> np.ndarray:
    """Return the frozen lexicographic simplex grid (66 states for 3 x .1)."""

    if components != 3 or not np.isclose(step, 0.1):
        raise ValueError("PROB-HEAD-STRUCTURE-FULL-v1 freezes a 3-component 0.1 simplex")
    states: list[tuple[float, float, float]] = []
    for first in range(11):
        for second in range(11 - first):
            third = 10 - first - second
            states.append((first / 10.0, second / 10.0, third / 10.0))
    return np.asarray(states, dtype=np.float64)


def validate_simplex(weights: np.ndarray, *, components: int = 3) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (components,) or not np.all(np.isfinite(values)):
        raise PredictionIntegrityError(f"pool weights must be a finite {components}-component simplex")
    if (
        np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise PredictionIntegrityError("pool weights violate the probability simplex")
    return values


def _validate_cdfs(cdfs: np.ndarray) -> np.ndarray:
    values = np.asarray(cdfs, dtype=np.float64)
    if values.ndim < 3:
        raise PredictionIntegrityError("teacher CDFs must have shape [teacher, case, support]")
    if values.shape[0] != 3:
        raise PredictionIntegrityError("exactly three distribution teachers are required")
    if not np.all(np.isfinite(values)):
        raise PredictionIntegrityError("CDF contains NaN/Inf")
    if np.any((values < -1e-12) | (values > 1.0 + 1e-12)):
        raise PredictionIntegrityError("CDF is outside [0, 1]")
    if np.any(np.diff(values, axis=-1) < -1e-12):
        raise PredictionIntegrityError("CDF monotonicity violation")
    return np.clip(values, 0.0, 1.0)


def linear_cdf_pool(cdfs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool CDF values, never raw family parameters."""

    values = _validate_cdfs(cdfs)
    simplex = validate_simplex(weights, components=values.shape[0])
    pooled = np.tensordot(simplex, values, axes=(0, 0))
    if np.any(np.diff(pooled, axis=-1) < -1e-12):
        raise PredictionIntegrityError("pooled CDF monotonicity violation")
    return np.clip(pooled, 0.0, 1.0)


def mixture_functionals(
    *,
    teacher_p_zero: np.ndarray,
    teacher_means: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the linear-mixture zero mass and mean."""

    p_zero = np.asarray(teacher_p_zero, dtype=np.float64)
    means = np.asarray(teacher_means, dtype=np.float64)
    if p_zero.shape != means.shape or p_zero.ndim < 2 or p_zero.shape[0] != 3:
        raise PredictionIntegrityError("teacher p0 and mean shapes must match [3, ...]")
    if not np.all(np.isfinite(p_zero)) or np.any((p_zero < 0.0) | (p_zero > 1.0)):
        raise PredictionIntegrityError("teacher zero probability is invalid")
    if not np.all(np.isfinite(means)) or np.any(means < 0.0):
        raise PredictionIntegrityError("teacher predictive mean is invalid")
    simplex = validate_simplex(weights)
    return {
        "p_zero": np.tensordot(simplex, p_zero, axes=(0, 0)),
        "mean": np.tensordot(simplex, means, axes=(0, 0)),
    }


def invert_cdf(
    support: np.ndarray,
    cdf: np.ndarray,
    probabilities: Sequence[float],
    *,
    exact_discrete_support: bool = False,
) -> np.ndarray:
    """Compute a left inverse only on an explicitly complete discrete support."""

    if not exact_discrete_support:
        raise PredictionIntegrityError(
            "support-table inversion is allowed only for verified exact discrete support"
        )

    values = np.asarray(support, dtype=np.float64)
    cumulative = np.asarray(cdf, dtype=np.float64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if cumulative.ndim == 1:
        cumulative = cumulative[None, :]
    if cumulative.ndim != 2:
        raise PredictionIntegrityError("CDF inversion expects [case, support]")
    if values.ndim == 1:
        values = np.broadcast_to(values, cumulative.shape)
    if values.shape != cumulative.shape:
        raise PredictionIntegrityError("support and CDF shapes differ")
    expected_support = np.arange(values.shape[1], dtype=np.float64)[None, :]
    if (
        not np.all(np.isfinite(values))
        or not np.array_equal(values, np.broadcast_to(expected_support, values.shape))
    ):
        raise PredictionIntegrityError(
            "exact discrete CDF support must be contiguous integer 0..K with unit spacing"
        )
    if not np.all(np.isfinite(cumulative)) or np.any((cumulative < -1e-12) | (cumulative > 1.0 + 1e-12)):
        raise PredictionIntegrityError("CDF is invalid")
    if np.any(np.diff(cumulative, axis=1) < -1e-12):
        raise PredictionIntegrityError("CDF is not monotone")
    if probabilities_array.ndim != 1 or np.any((probabilities_array <= 0.0) | (probabilities_array >= 1.0)):
        raise ValueError("inverse probabilities must lie strictly inside (0, 1)")
    result = np.empty((cumulative.shape[0], probabilities_array.size), dtype=np.float64)
    for probability_index, probability in enumerate(probabilities_array):
        reached = cumulative >= probability
        if np.any(~np.any(reached, axis=1)):
            raise PredictionIntegrityError(
                f"CDF support does not reach inverse probability {probability:g}"
            )
        indices = np.argmax(reached, axis=1)
        result[:, probability_index] = np.take_along_axis(
            values, indices[:, None], axis=1
        )[:, 0]
    return result


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True, init=False)
class BoundCDFArtifact:
    """Native distribution CDF bound to one sealed head/seed forecast component."""

    head: str
    model_seed: int
    validation_artifact_sha256: str
    case_count: int
    case_shape: tuple[int, ...]
    mean_values: np.ndarray
    p_zero_values: np.ndarray
    _evaluate: Callable[..., np.ndarray]
    _evaluate_grid: Callable[..., np.ndarray]

    def __call__(
        self, values: np.ndarray, *, case_indices: np.ndarray | None = None
    ) -> np.ndarray:
        return self._evaluate(values, case_indices=case_indices)

    def evaluate_grid(
        self, values: np.ndarray, *, case_indices: np.ndarray
    ) -> np.ndarray:
        return self._evaluate_grid(values, case_indices=case_indices)


def cdf_callable_for_distribution(
    distribution: Any,
    *,
    head: str,
    model_seed: int,
    validation_artifact: SealedValidationArtifact,
) -> tuple[BoundCDFArtifact, np.ndarray, tuple[int, ...]]:
    """Adapt the common distribution interface to the flattened pool inverter.

    The native distribution keeps its ``[batch, horizon]`` case shape.  The
    returned callable accepts the inverter's flat case vector, restores that
    native shape for ``distribution.cdf``, and flattens the result again.
    """

    if not isinstance(validation_artifact, SealedValidationArtifact):
        raise PredictionIntegrityError("distribution CDF adapter requires a sealed artifact")
    normalized_head = str(head)
    if normalized_head not in HEAD_ORDER:
        raise PredictionIntegrityError("distribution CDF adapter has an unregistered head")
    seed = int(model_seed)
    if seed not in validation_artifact.model_seeds:
        raise PredictionIntegrityError("distribution CDF adapter model seed is not sealed")
    from .distributions import (
        NegativeBinomialDistribution,
        ShiftedHurdleNegativeBinomialDistribution,
        TweedieDistribution,
    )

    contracts = {
        "NB": (NegativeBinomialDistribution, ("mu", "r")),
        "HSNB": (ShiftedHurdleNegativeBinomialDistribution, ("pi", "mu", "r")),
        "TWEEDIE_FULL": (TweedieDistribution, ("mu", "phi", "p")),
    }
    expected_type, expected_parameters = contracts[normalized_head]
    if type(distribution) is not expected_type:
        raise PredictionIntegrityError(
            f"{normalized_head} CDF adapter requires native {expected_type.__name__}"
        )
    if not callable(getattr(distribution, "mean", None)) or not callable(
        getattr(distribution, "cdf", None)
    ):
        raise ValueError("distribution adapter requires callable mean() and cdf()")
    mean = _as_numpy(distribution.mean())
    if mean.ndim == 0 or not np.all(np.isfinite(mean)) or np.any(mean < 0.0):
        raise PredictionIntegrityError("distribution adapter found an invalid predictive mean")
    case_shape = tuple(int(value) for value in mean.shape)
    case_count = int(mean.size)
    p_zero = _as_numpy(distribution.p_zero())
    if (
        p_zero.shape != mean.shape
        or not np.all(np.isfinite(p_zero))
        or np.any((p_zero < 0.0) | (p_zero > 1.0))
    ):
        raise PredictionIntegrityError(
            "distribution adapter found an invalid zero probability"
        )
    component_name = f"cdf_parameters_seed_{seed}"
    validation_artifact.verify_teacher_component(
        normalized_head,
        component_name,
        {
            name: _as_numpy(getattr(distribution, name))
            for name in expected_parameters
        },
    )

    def evaluate(
        flat_values: np.ndarray, *, case_indices: np.ndarray | None = None
    ) -> np.ndarray:
        values = np.asarray(flat_values, dtype=np.float64)
        indices = (
            np.arange(case_count, dtype=np.int64)
            if case_indices is None
            else np.asarray(case_indices, dtype=np.int64)
        )
        if (
            indices.ndim != 1
            or values.shape != (indices.size,)
            or np.any(indices < 0)
            or np.any(indices >= case_count)
            or len(set(indices.tolist())) != indices.size
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise PredictionIntegrityError("flattened distribution CDF input is invalid")
        if indices.size == case_count and np.array_equal(
            indices, np.arange(case_count, dtype=np.int64)
        ):
            result = _as_numpy(distribution.cdf(values.reshape(case_shape)))
            if result.shape != case_shape:
                raise PredictionIntegrityError(
                    "distribution CDF output changed the native case shape"
                )
            return result.reshape(-1)
        if callable(getattr(distribution, "subset", None)):
            subset = distribution.subset(indices.tolist())
        else:
            sliced = [
                getattr(distribution, name)
                .reshape(-1)[indices.tolist()]
                .reshape(-1, 1)
                for name in expected_parameters
            ]
            subset = type(distribution)(*sliced)
        result = _as_numpy(subset.cdf(values.reshape(-1, 1)))
        if result.shape != (values.size, 1):
            raise PredictionIntegrityError(
                "chunked distribution CDF output changed the case shape"
            )
        return result.reshape(-1)

    def evaluate_grid(
        grid_values: np.ndarray, *, case_indices: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(grid_values, dtype=np.float64)
        indices = np.asarray(case_indices, dtype=np.int64)
        if (
            values.ndim != 2
            or values.shape[1] != indices.size
            or np.any(indices < 0)
            or np.any(indices >= case_count)
            or len(set(indices.tolist())) != indices.size
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise PredictionIntegrityError("gridded distribution CDF input is invalid")
        sliced = [
            getattr(distribution, name)
            .reshape(-1)[indices.tolist()]
            .reshape(-1, 1)
            for name in expected_parameters
        ]
        subset = type(distribution)(*sliced)
        result = _as_numpy(subset.cdf(values[..., None]))
        if result.shape != values.shape + (1,):
            raise PredictionIntegrityError(
                "gridded distribution CDF output changed the state/case shape"
            )
        return result[..., 0]

    bound = object.__new__(BoundCDFArtifact)
    object.__setattr__(bound, "head", normalized_head)
    object.__setattr__(bound, "model_seed", seed)
    object.__setattr__(
        bound, "validation_artifact_sha256", validation_artifact.artifact_sha256
    )
    object.__setattr__(bound, "case_count", case_count)
    object.__setattr__(bound, "case_shape", case_shape)
    mean_values = mean.reshape(-1).copy()
    mean_values.flags.writeable = False
    p_zero_values = p_zero.reshape(-1).copy()
    p_zero_values.flags.writeable = False
    object.__setattr__(bound, "mean_values", mean_values)
    object.__setattr__(bound, "p_zero_values", p_zero_values)
    object.__setattr__(bound, "_evaluate", evaluate)
    object.__setattr__(bound, "_evaluate_grid", evaluate_grid)
    return bound, mean.reshape(-1), case_shape


def _bound_cdf_seed_grid(
    values: Sequence[Sequence[BoundCDFArtifact]],
    artifact: SealedValidationArtifact,
    *,
    case_count: int,
) -> tuple[tuple[BoundCDFArtifact, ...], ...]:
    rows = tuple(tuple(row) for row in values)
    if len(rows) != len(artifact.model_seeds):
        raise PredictionIntegrityError("CDF adapter seed axis differs from the sealed manifest")
    for seed_index, (seed, row) in enumerate(zip(artifact.model_seeds, rows, strict=True)):
        if len(row) != len(HEAD_ORDER):
            raise PredictionIntegrityError("CDF adapter head axis must contain exactly three heads")
        for head, adapter in zip(HEAD_ORDER, row, strict=True):
            if (
                not isinstance(adapter, BoundCDFArtifact)
                or adapter.head != head
                or adapter.model_seed != seed
                or adapter.validation_artifact_sha256 != artifact.artifact_sha256
                or adapter.case_count != int(case_count)
            ):
                raise PredictionIntegrityError(
                    f"CDF adapter grid is not bound at seed={seed_index}, head={head}"
                )
    return rows


def invert_pooled_cdf(
    *,
    cdf_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
    weights: np.ndarray,
    probabilities: Sequence[float],
    case_count: int,
    initial_upper: np.ndarray | float,
    x_tolerance: float = 1e-6,
    max_iterations: int = 128,
    max_expansions: int = 128,
) -> dict[str, Any]:
    """Invert a true linear CDF pool by bracket expansion and bisection.

    ``cdf_functions`` is designed to receive bound ``distribution.cdf`` methods:
    each callable takes one nonnegative value per forecast case and returns the
    corresponding one-dimensional CDF vector.  P1 calls this function with
    :func:`equal_pool_weights`; P2 calls it for each of the 66 simplex states.
    """

    if len(cdf_functions) != 3 or not all(callable(function) for function in cdf_functions):
        raise ValueError("pooled CDF inversion requires exactly three callable teacher CDFs")
    simplex = validate_simplex(weights)
    q = np.asarray(probabilities, dtype=np.float64)
    if q.ndim != 1 or q.size == 0 or np.any((q <= 0.0) | (q >= 1.0)) or np.any(np.diff(q) <= 0.0):
        raise ValueError("inverse probabilities must be finite, increasing, and inside (0, 1)")
    if not np.all(np.isfinite(q)) or int(case_count) <= 0:
        raise ValueError("case count and inverse probabilities are invalid")
    if not np.isfinite(x_tolerance) or x_tolerance <= 0.0:
        raise ValueError("x_tolerance must be finite and positive")
    if x_tolerance != 1e-6 or max_iterations != 128 or max_expansions != 128:
        raise ValueError(
            "pooled CDF inversion is frozen at tolerance=1e-6 and 128 iteration/expansion caps"
        )
    upper = np.asarray(initial_upper, dtype=np.float64)
    if upper.ndim == 0:
        upper = np.full(int(case_count), float(upper), dtype=np.float64)
    if upper.shape != (int(case_count),) or not np.all(np.isfinite(upper)) or np.any(upper < 0.0):
        raise PredictionIntegrityError("initial CDF inversion upper bracket is invalid")
    # Frozen bracket rule: caller supplies max teacher mean or initial q99;
    # numerical inversion never starts below one.
    upper = np.maximum(upper, 1.0)
    initial_upper_used = upper.copy()

    def evaluate(points: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        for function in cdf_functions:
            result = np.asarray(function(points.copy()), dtype=np.float64)
            if result.shape != (int(case_count),):
                raise PredictionIntegrityError(
                    "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF callable returned the wrong case shape"
                )
            if not np.all(np.isfinite(result)) or np.any((result < -1e-12) | (result > 1.0 + 1e-12)):
                raise PredictionIntegrityError(
                    "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF callable returned invalid probabilities"
                )
            values.append(np.clip(result, 0.0, 1.0))
        return np.tensordot(simplex, np.stack(values), axes=(0, 0))

    zero = np.zeros(int(case_count), dtype=np.float64)
    cdf_zero = evaluate(zero)
    cdf_upper = evaluate(upper)
    if np.any(cdf_upper + 1e-12 < cdf_zero):
        raise PredictionIntegrityError(
            "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF callable is nonmonotone"
        )
    expansions = 0
    target_max = float(q[-1])
    while np.any(cdf_upper < target_max):
        mask = cdf_upper < target_max
        previous = cdf_upper.copy()
        upper[mask] *= 2.0
        if not np.all(np.isfinite(upper)):
            raise PredictionIntegrityError("CDF bracket expansion overflowed")
        cdf_upper = evaluate(upper)
        if np.any(cdf_upper + 1e-12 < previous):
            raise PredictionIntegrityError(
                "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF callable decreased during expansion"
            )
        expansions += 1
        if expansions >= max_expansions and np.any(cdf_upper < target_max):
            raise PredictionIntegrityError("CDF_INVERSION_NONBRACKETING_HARD_STOP")

    quantiles = np.zeros((int(case_count), q.size), dtype=np.float64)
    error_bounds = np.zeros_like(quantiles)
    iterations_used = 0
    for probability_index, probability in enumerate(q):
        active = cdf_zero < probability
        low = zero.copy()
        high = upper.copy()
        cdf_low = cdf_zero.copy()
        cdf_high = cdf_upper.copy()
        iterations = 0
        tolerance = np.maximum(x_tolerance, x_tolerance * np.abs(high))
        while np.any(active & ((high - low) > tolerance)):
            midpoint = (low + high) / 2.0
            cdf_midpoint = evaluate(midpoint)
            if np.any(cdf_midpoint + 1e-12 < cdf_low) or np.any(cdf_midpoint - 1e-12 > cdf_high):
                raise PredictionIntegrityError(
                    "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF callable violated monotonicity"
                )
            move_high = active & (cdf_midpoint >= probability)
            move_low = active & ~move_high
            high[move_high] = midpoint[move_high]
            cdf_high[move_high] = cdf_midpoint[move_high]
            low[move_low] = midpoint[move_low]
            cdf_low[move_low] = cdf_midpoint[move_low]
            tolerance = np.maximum(x_tolerance, x_tolerance * np.abs(high))
            iterations += 1
            if iterations >= max_iterations and np.any(
                active & ((high - low) > tolerance)
            ):
                raise PredictionIntegrityError("CDF bisection did not reach the frozen error tolerance")
        quantiles[active, probability_index] = high[active]
        error_bounds[active, probability_index] = high[active] - low[active]
        iterations_used = max(iterations_used, iterations)
    if np.any(np.diff(quantiles, axis=1) < -x_tolerance):
        raise PredictionIntegrityError("inverted pooled quantiles are nonmonotone")
    return {
        "quantiles": quantiles.tolist(),
        "max_x_error_bound": float(np.max(error_bounds)),
        "x_tolerance": float(x_tolerance),
        "error_rule": "width<=max(abs_tolerance,abs_tolerance*upper)",
        "iterations": iterations_used,
        "bracket_expansions": expansions,
        "max_iterations": int(max_iterations),
        "max_expansions": int(max_expansions),
        "lower_bracket": 0.0,
        "initial_upper_used": initial_upper_used.tolist(),
        "initial_upper_source": "max(1,caller_max_teacher_mean_or_initial_q99)",
        "method": "bracket_expansion_then_bisection_true_linear_CDF_pool",
    }


def diagnostic_two_head_cdf_pool(
    *,
    cdf_functions: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    source_weights: Mapping[str, float],
    probabilities: Sequence[float],
    case_count: int,
    initial_upper: np.ndarray | float,
) -> dict[str, Any]:
    """Apply an explicitly non-confirmatory NB/HSNB pool after Tweedie blocks.

    The frozen three-head weights are renormalized over the two surviving
    teachers.  A zero-weight duplicate callable is passed to the verified
    three-component numerical inverter; it has no effect on the pooled CDF.
    """

    if tuple(map(str, cdf_functions)) != ("NB", "HSNB"):
        raise ValueError("two-head diagnostic CDF functions must be ordered NB, HSNB")
    if tuple(map(str, source_weights)) != HEAD_ORDER:
        raise ValueError(f"diagnostic source weights must be ordered {HEAD_ORDER}")
    frozen_weights = validate_simplex(
        np.asarray([source_weights[head] for head in HEAD_ORDER], dtype=np.float64)
    )
    surviving_mass = float(frozen_weights[:2].sum())
    if surviving_mass <= 0.0:
        raise PredictionIntegrityError(
            "Tweedie-blocked diagnostic weights have no surviving NB/HSNB mass"
        )
    surviving = frozen_weights[:2] / surviving_mass
    padded_weights = np.asarray([surviving[0], surviving[1], 0.0], dtype=np.float64)
    nb = cdf_functions["NB"]
    hsnb = cdf_functions["HSNB"]
    inversion = invert_pooled_cdf(
        cdf_functions=(nb, hsnb, nb),
        weights=padded_weights,
        probabilities=probabilities,
        case_count=case_count,
        initial_upper=initial_upper,
    )
    return {
        **inversion,
        "pool": "P2_DIAGNOSTIC_NB_HSNB",
        "weights": {"NB": float(surviving[0]), "HSNB": float(surviving[1])},
        "source_three_head_weights": {
            head: float(frozen_weights[index]) for index, head in enumerate(HEAD_ORDER)
        },
        "renormalization_denominator": surviving_mass,
        "excluded_head": "TWEEDIE_FULL",
        "upstream_required_gates": ["TWEEDIE_VALID"],
        "upstream_gate_status": {"TWEEDIE_VALID": "FAIL"},
        "confirmatory_eligible": False,
        "scientific_role": "DIAGNOSTIC_CONTINUATION_AFTER_TWEEDIE_BRANCH_BLOCKED_HARD",
    }


def _invert_bound_cdf_states(
    *,
    cdf_artifacts: Sequence[BoundCDFArtifact],
    states: np.ndarray,
    probability: float,
    case_indices: np.ndarray,
    initial_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize one-q inversion over all 66 states for one seed/case chunk."""

    simplex_states = np.asarray(states, dtype=np.float64)
    indices = np.asarray(case_indices, dtype=np.int64)
    upper_base = np.asarray(initial_upper, dtype=np.float64)
    if (
        len(cdf_artifacts) != 3
        or simplex_states.shape != (66, 3)
        or indices.ndim != 1
        or upper_base.shape != (indices.size,)
        or not np.isfinite(probability)
        or not 0.0 < probability < 1.0
    ):
        raise PredictionIntegrityError("vectorized P2 inversion inputs are invalid")
    upper = np.broadcast_to(np.maximum(upper_base, 1.0), (66, indices.size)).copy()
    zero = np.zeros_like(upper)

    def pooled(points: np.ndarray) -> np.ndarray:
        by_head = np.stack(
            [
                artifact.evaluate_grid(points, case_indices=indices)
                for artifact in cdf_artifacts
            ],
            axis=0,
        )
        if (
            not np.all(np.isfinite(by_head))
            or np.any((by_head < -1e-12) | (by_head > 1.0 + 1e-12))
        ):
            raise PredictionIntegrityError(
                "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF grid is invalid"
            )
        return np.einsum("wh,hwc->wc", simplex_states, np.clip(by_head, 0.0, 1.0))

    cdf_zero = pooled(zero)
    cdf_upper = pooled(upper)
    if np.any(cdf_upper + 1e-12 < cdf_zero):
        raise PredictionIntegrityError(
            "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF callable is nonmonotone"
        )
    for expansion in range(128):
        needs = cdf_upper < probability
        if not np.any(needs):
            break
        previous = cdf_upper.copy()
        upper[needs] *= 2.0
        if not np.all(np.isfinite(upper)):
            raise PredictionIntegrityError("CDF bracket expansion overflowed")
        cdf_upper = pooled(upper)
        if np.any(cdf_upper + 1e-12 < previous):
            raise PredictionIntegrityError(
                "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF callable decreased during expansion"
            )
    else:
        raise PredictionIntegrityError("CDF_INVERSION_NONBRACKETING_HARD_STOP")

    active = cdf_zero < probability
    low = zero.copy()
    high = upper.copy()
    cdf_low = cdf_zero.copy()
    cdf_high = cdf_upper.copy()
    for iteration in range(128):
        tolerance = np.maximum(1e-6, 1e-6 * np.abs(high))
        unfinished = active & ((high - low) > tolerance)
        if not np.any(unfinished):
            break
        middle = (low + high) / 2.0
        cdf_middle = pooled(middle)
        if np.any(cdf_middle + 1e-12 < cdf_low) or np.any(
            cdf_middle - 1e-12 > cdf_high
        ):
            raise PredictionIntegrityError(
                "INVALID_PREDICTIVE_DISTRIBUTION_HARD_STOP: CDF callable violated monotonicity"
            )
        move_high = unfinished & (cdf_middle >= probability)
        move_low = unfinished & ~move_high
        high[move_high] = middle[move_high]
        cdf_high[move_high] = cdf_middle[move_high]
        low[move_low] = middle[move_low]
        cdf_low[move_low] = cdf_middle[move_low]
    else:
        raise PredictionIntegrityError("CDF bisection did not reach the frozen error tolerance")
    result = np.where(active, high, 0.0)
    return result, np.where(active, high - low, 0.0)


def select_best_single_teacher(
    *,
    validation_teacher_quantiles: np.ndarray,
    validation_teacher_p_zero: np.ndarray,
    validation_y: np.ndarray,
    validation_scale: np.ndarray,
    validation_case_keys: Sequence[Sequence[Any]],
    validation_artifact: SealedValidationArtifact,
) -> dict[str, Any]:
    """Recompute deployment P0 on validation, seed-macro sCRPS and tail loss."""

    if not isinstance(validation_artifact, SealedValidationArtifact):
        raise PredictionIntegrityError("P0 requires a sealed validation artifact")
    validation_artifact.require_validation_split()
    y = np.asarray(validation_y, dtype=np.float64)
    scale = np.asarray(validation_scale, dtype=np.float64)
    validation_artifact.verify_target_arrays(
        validation_y=y,
        validation_scale=scale,
        validation_case_keys=validation_case_keys,
    )
    quantiles = np.asarray(validation_teacher_quantiles, dtype=np.float64)
    p_zero = np.asarray(validation_teacher_p_zero, dtype=np.float64)
    seed_count = len(validation_artifact.model_seeds)
    if quantiles.shape != (seed_count, 3, y.size, len(CRPS_QUANTILE_GRID)):
        raise PredictionIntegrityError(
            "P0 teacher quantiles must have explicit [seed,3,case,21] shape"
        )
    if p_zero.shape != (seed_count, 3, y.size):
        raise PredictionIntegrityError(
            "P0 teacher p0 must have explicit [seed,3,case] shape"
        )
    if (
        not np.all(np.isfinite(quantiles))
        or np.any(quantiles < 0.0)
        or np.any(np.diff(quantiles, axis=-1) < 0.0)
        or not np.all(np.isfinite(p_zero))
        or np.any((p_zero < 0.0) | (p_zero > 1.0))
    ):
        raise PredictionIntegrityError("P0 teacher forecasts are invalid")
    validation_artifact.verify_teacher_components(
        {
            head: {
                "quantiles": quantiles[:, index],
                "p_zero": p_zero[:, index],
            }
            for index, head in enumerate(HEAD_ORDER)
        }
    )
    q = np.asarray(CRPS_QUANTILE_GRID, dtype=np.float64)
    tail_positions = [int(np.flatnonzero(q == level)[0]) for level in (0.95, 0.99)]
    rows: list[tuple[str, float, float]] = []
    for head_index, head in enumerate(HEAD_ORDER):
        seed_scrps = []
        seed_tail = []
        for seed_index in range(seed_count):
            values = quantiles[seed_index, head_index]
            seed_scrps.append(float(np.mean(approximate_crps(y, values, q) / scale)))
            seed_tail.append(
                float(
                    np.mean(
                        [
                            np.mean(
                                2.0
                                * pinball_loss(y, values[:, position], q[position])
                                / scale
                            )
                            for position in tail_positions
                        ]
                    )
                )
            )
        rows.append((head, float(np.mean(seed_scrps)), float(np.mean(seed_tail))))
    best_index = min(range(len(rows)), key=lambda index: (rows[index][1], rows[index][2], index))
    return _seal_selection({
        "pool": "P0",
        "teacher": rows[best_index][0],
        "validation_sCRPS": rows[best_index][1],
        "validation_tail_sQL": rows[best_index][2],
        "tie_break_order": ["validation_sCRPS", "mean_q95_q99_sQL", *HEAD_ORDER],
        "selection_source": "validation_only",
        "semantic_label": "validation_selected_P0",
        "model_seed_axis": list(validation_artifact.model_seeds),
        "seed_aggregation": "equal_mean_of_per_seed_scalar_scores",
        "cross_seed_distribution_averaging": False,
    }, validation_artifact)


def equal_pool_weights() -> np.ndarray:
    return np.full(3, 1.0 / 3.0, dtype=np.float64)


def select_global_cdf_pool(
    *,
    validation_y: np.ndarray,
    validation_scale: np.ndarray,
    validation_case_keys: Sequence[Sequence[Any]],
    validation_artifact: SealedValidationArtifact,
    validation_cdfs: np.ndarray | None = None,
    support: np.ndarray | None = None,
    validation_cdf_functions: Sequence[Sequence[BoundCDFArtifact]] | None = None,
    initial_upper: np.ndarray | float | None = None,
    discrete_support_exact: bool = False,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
    validation_group_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Select P2 from bound native CDFs, scoring seeds independently.

    Finite CDF tables are deliberately excluded from confirmatory selection:
    NB, HSNB, and Tweedie all have unbounded support (and Tweedie is continuous).
    ``invert_cdf`` remains available solely for isolated diagnostic/unit fixtures.
    """

    if tuple(map(float, quantile_grid)) != CRPS_QUANTILE_GRID:
        raise ValueError("P2 quantile grid is frozen")
    if not isinstance(validation_artifact, SealedValidationArtifact):
        raise PredictionIntegrityError("P2 requires a sealed validation artifact")
    validation_artifact.require_validation_split()
    y = np.asarray(validation_y, dtype=np.float64)
    scale = np.asarray(validation_scale, dtype=np.float64)
    validation_artifact.verify_target_arrays(
        validation_y=y,
        validation_scale=scale,
        validation_case_keys=validation_case_keys,
    )
    if (
        y.ndim != 1
        or scale.shape != y.shape
        or not np.all(np.isfinite(y))
        or np.any(y < 0.0)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise PredictionIntegrityError("validation CDF, target, and scale shapes are inconsistent")
    if validation_cdfs is not None or support is not None or discrete_support_exact:
        raise PredictionIntegrityError(
            "confirmatory P2 forbids finite support tables; use bound native CDF artifacts"
        )
    if validation_cdf_functions is None or initial_upper is None:
        raise ValueError("P2 requires bound native CDF artifacts and an initial upper bracket")
    bound_functions = _bound_cdf_seed_grid(
        validation_cdf_functions,
        validation_artifact,
        case_count=y.size,
    )
    upper_values = np.asarray(initial_upper, dtype=np.float64)
    seed_count = len(validation_artifact.model_seeds)
    if upper_values.ndim == 0:
        upper_values = np.full((seed_count, y.size), float(upper_values))
    elif upper_values.shape == (y.size,) and seed_count == 1:
        upper_values = upper_values[None, :]
    if (
        upper_values.shape != (seed_count, y.size)
        or not np.all(np.isfinite(upper_values))
        or np.any(upper_values < 0.0)
    ):
        raise PredictionIntegrityError(
            "P2 initial upper bracket must have explicit [seed, case] shape"
        )
    frozen_groups = np.asarray(validation_artifact.validation_group_ids, dtype=object)
    if validation_group_ids is not None and list(validation_group_ids) != list(
        validation_artifact.validation_group_ids
    ):
        raise PredictionIntegrityError(
            "validation group ids differ from the sealed selection artifact"
        )
    groups = frozen_groups
    if groups.shape != y.shape:
        raise PredictionIntegrityError("validation group ids must have one value per case")
    if groups.size == 0 or any(
        value is None
        or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        for value in groups.tolist()
    ):
        raise PredictionIntegrityError("validation group ids must be finite and nonempty")
    try:
        set(groups.tolist())
    except TypeError as error:
        raise PredictionIntegrityError("validation group ids must be hashable") from error

    def equal_group_mean(case_values: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    np.mean(case_values[groups == group])
                    for group in sorted(dict.fromkeys(groups.tolist()), key=lambda value: str(value))
                ]
            )
        )

    states = simplex_grid()
    q_array = np.asarray(quantile_grid, dtype=np.float64)
    q_weights = midpoint_cell_widths(q_array)
    ordered_groups = sorted(
        dict.fromkeys(groups.tolist()), key=lambda value: str(value)
    )
    group_counts = np.asarray(
        [np.sum(groups == group) for group in ordered_groups], dtype=np.float64
    )
    seed_state_scores = np.zeros((seed_count, len(states)), dtype=np.float64)
    seed_state_tail = np.zeros_like(seed_state_scores)
    inversion_error_bounds = np.zeros(len(states), dtype=np.float64)
    for seed_index, seed_functions in enumerate(bound_functions):
        group_crps_sums = np.zeros((len(states), len(ordered_groups)), dtype=np.float64)
        group_tail_sums = np.zeros_like(group_crps_sums)
        for start in range(0, y.size, POOLED_FLAT_CASE_CHUNK):
            stop = min(start + POOLED_FLAT_CASE_CHUNK, y.size)
            indices = np.arange(start, stop, dtype=np.int64)
            local_groups = groups[start:stop]
            for q_index, probability in enumerate(q_array):
                quantile_states, error_bounds = _invert_bound_cdf_states(
                    cdf_artifacts=seed_functions,
                    states=states,
                    probability=float(probability),
                    case_indices=indices,
                    initial_upper=upper_values[seed_index, start:stop],
                )
                inversion_error_bounds = np.maximum(
                    inversion_error_bounds, np.max(error_bounds, axis=1)
                )
                error = y[None, start:stop] - quantile_states
                scaled_pinball = np.maximum(
                    probability * error, (probability - 1.0) * error
                ) / scale[None, start:stop]
                for group_index, group in enumerate(ordered_groups):
                    local_mask = local_groups == group
                    group_crps_sums[:, group_index] += np.sum(
                        2.0 * q_weights[q_index] * scaled_pinball[:, local_mask],
                        axis=1,
                    )
                    if probability in (0.95, 0.99):
                        group_tail_sums[:, group_index] += np.sum(
                            scaled_pinball[:, local_mask], axis=1
                        )
        seed_state_scores[seed_index] = np.mean(
            group_crps_sums / group_counts[None, :], axis=1
        )
        # mean(q95,q99 of 2*pinball) == sum of the two pinball losses.
        seed_state_tail[seed_index] = np.mean(
            group_tail_sums / group_counts[None, :], axis=1
        )
    scores = np.mean(seed_state_scores, axis=0)
    tail_scores = np.mean(seed_state_tail, axis=0)
    best_index = min(
        range(len(states)),
        key=lambda index: (scores[index], tail_scores[index], tuple(states[index])),
    )
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(tail_scores)):
        raise PredictionIntegrityError("P2 candidate scores contain NaN/Inf")
    return _seal_selection({
        "pool": "P2",
        "weights": states[best_index].tolist(),
        "validation_sCRPS": scores[best_index],
        "validation_tail_sQL": tail_scores[best_index],
        "candidate_count": len(states),
        "grid_step": 0.1,
        "grid_index": best_index,
        "selection_source": "validation_only",
        "validation_aggregation": "equal_group_macro",
        "validation_group_sha256": json.loads(
            validation_artifact.canonical_payload_json
        )["validation_group_sha256"],
        "inversion_method": "bracket_expansion_bisection_bound_native_cdf",
        "max_x_error_bound": float(inversion_error_bounds[best_index]),
        "pooled_flat_case_chunk": POOLED_FLAT_CASE_CHUNK,
        "pooled_q_chunk": POOLED_Q_CHUNK,
        "model_seed_axis": list(validation_artifact.model_seeds),
        "seed_aggregation": "equal_mean_of_per_seed_scalar_scores",
        "cross_seed_distribution_averaging": False,
        "case_key_sha256": json.loads(validation_artifact.canonical_payload_json)[
            "case_key_sha256"
        ],
    }, validation_artifact)


def _seal_pool_forecast_rows(
    *,
    quantiles: np.ndarray,
    p_zero: np.ndarray,
    mean: np.ndarray,
    target_artifact: SealedEvaluationTarget,
    branch_eligibility: BranchEligibility,
    provenance: ScientificForecastProvenance,
    source_kind: str,
    source_binding: Mapping[str, Any],
    model_seeds: Sequence[int],
) -> tuple[tuple[pd.DataFrame, ...], tuple[Any, ...]]:
    values = np.asarray(quantiles, dtype=np.float64)
    zero = np.asarray(p_zero, dtype=np.float64)
    means = np.asarray(mean, dtype=np.float64)
    seeds = tuple(int(seed) for seed in model_seeds)
    if (
        values.shape != (len(seeds), target_artifact.row_count, len(CRPS_QUANTILE_GRID))
        or zero.shape != values.shape[:-1]
        or means.shape != values.shape[:-1]
    ):
        raise PredictionIntegrityError("pooled scientific forecast shapes are inconsistent")
    target_rows = target_artifact.as_dict()["payload"]["rows"]
    frames: list[pd.DataFrame] = []
    artifacts: list[Any] = []
    for seed_index, seed in enumerate(seeds):
        frame = pd.DataFrame(target_rows)
        frame["p_zero"] = zero[seed_index]
        frame["mean"] = means[seed_index]
        for q_index, probability in enumerate(CRPS_QUANTILE_GRID):
            frame[quantile_column(probability)] = values[seed_index, :, q_index]
        derived = coverage_quantiles_from_common_grid(
            values[seed_index],
            CRPS_QUANTILE_GRID,
            p_zero=zero[seed_index],
        )
        frame[quantile_column(0.025)] = derived["q025"]
        frame[quantile_column(0.975)] = derived["q975"]
        frame = frame[
            [
                "dataset_id",
                "series_id",
                "origin",
                "step",
                "y",
                "scale",
                "target_mask",
                "p_zero",
                "mean",
                *(quantile_column(q) for q in EVALUATION_QUANTILE_GRID),
            ]
        ]
        binding = {**dict(source_binding), "model_seed": seed}
        artifact = _seal_scientific_forecast_artifact(
            frame=frame,
            provenance=provenance,
            target_artifact=target_artifact,
            branch_eligibility=branch_eligibility,
            source_kind=source_kind,
            source_binding=binding,
        )
        frames.append(frame)
        artifacts.append(artifact)
    return tuple(frames), tuple(artifacts)


def apply_global_cdf_pool(
    *,
    selection: Mapping[str, Any],
    validation_artifact: SealedValidationArtifact,
    prediction_artifact: SealedValidationArtifact,
    cdf_functions: Sequence[Sequence[BoundCDFArtifact]],
    probabilities: Sequence[float],
    case_count: int,
    initial_upper: np.ndarray | float,
    branch_eligibility: BranchEligibility,
) -> dict[str, Any]:
    """Apply only a digest-verified validation-selected P2 weight vector."""

    frozen = _verify_selection_binding(selection, validation_artifact)
    if frozen.get("pool") != "P2":
        raise PredictionIntegrityError("P2 application requires a sealed P2 selection")
    if (
        not isinstance(branch_eligibility, BranchEligibility)
        or not branch_eligibility.confirmatory_eligible
        or branch_eligibility.role != "CONFIRMATORY"
    ):
        raise PredictionIntegrityError(
            "P2 confirmatory application requires eligible branch lineage"
        )
    if prediction_artifact.model_seeds != validation_artifact.model_seeds:
        raise PredictionIntegrityError(
            "P2 application seed manifest differs from validation selection"
        )
    bound_functions = _bound_cdf_seed_grid(
        cdf_functions,
        prediction_artifact,
        case_count=case_count,
    )
    simplex = validate_simplex(
        np.asarray(frozen.get("weights"), dtype=np.float64)
    )
    q = np.asarray(probabilities, dtype=np.float64)
    if (
        q.ndim != 1
        or q.size == 0
        or not np.all(np.isfinite(q))
        or np.any((q <= 0.0) | (q >= 1.0))
        or np.any(np.diff(q) <= 0.0)
    ):
        raise PredictionIntegrityError(
            "P2 application probabilities must be finite, unique, and strictly increasing"
        )
    if tuple(map(float, q)) != CRPS_QUANTILE_GRID:
        raise PredictionIntegrityError(
            "P2 confirmatory application requires the frozen common quantile grid"
        )
    upper_values = np.asarray(initial_upper, dtype=np.float64)
    seed_count = len(bound_functions)
    if upper_values.ndim == 0:
        upper_values = np.full((seed_count, int(case_count)), float(upper_values))
    elif upper_values.shape == (int(case_count),) and seed_count == 1:
        upper_values = upper_values[None, :]
    if upper_values.shape != (seed_count, int(case_count)):
        raise PredictionIntegrityError(
            "P2 application upper bracket must have explicit [seed, case] shape"
        )
    quantiles = np.empty((seed_count, int(case_count), q.size), dtype=np.float64)
    max_error = 0.0
    max_iterations = 0
    max_expansions = 0
    for seed_index, seed_functions in enumerate(bound_functions):
        for start in range(0, int(case_count), POOLED_FLAT_CASE_CHUNK):
            stop = min(start + POOLED_FLAT_CASE_CHUNK, int(case_count))
            indices = np.arange(start, stop, dtype=np.int64)
            chunk_functions = tuple(
                (
                    lambda values, _function=function, _indices=indices: _function(
                        values, case_indices=_indices
                    )
                )
                for function in seed_functions
            )
            for q_start in range(0, q.size, POOLED_Q_CHUNK):
                q_stop = min(q_start + POOLED_Q_CHUNK, q.size)
                inversion = invert_pooled_cdf(
                    cdf_functions=chunk_functions,
                    weights=simplex,
                    probabilities=q[q_start:q_stop],
                    case_count=stop - start,
                    initial_upper=upper_values[seed_index, start:stop],
                )
                quantiles[seed_index, start:stop, q_start:q_stop] = np.asarray(
                    inversion["quantiles"], dtype=np.float64
                )
                max_error = max(max_error, float(inversion["max_x_error_bound"]))
                max_iterations = max(max_iterations, int(inversion["iterations"]))
                max_expansions = max(
                    max_expansions, int(inversion["bracket_expansions"])
                )
    if np.any(np.diff(quantiles, axis=-1) < -1e-6):
        raise PredictionIntegrityError("P2 application produced quantile crossing")
    teacher_means = np.asarray(
        [[item.mean_values for item in row] for row in bound_functions],
        dtype=np.float64,
    )
    teacher_pzero = np.asarray(
        [[item.p_zero_values for item in row] for row in bound_functions],
        dtype=np.float64,
    )
    means = np.einsum("h,shc->sc", simplex, teacher_means)
    p_zero = np.einsum("h,shc->sc", simplex, teacher_pzero)
    forecast_frames, forecast_artifacts = _seal_pool_forecast_rows(
        quantiles=quantiles,
        p_zero=p_zero,
        mean=means,
        target_artifact=prediction_artifact.target_artifact,
        branch_eligibility=branch_eligibility,
        provenance=ScientificForecastProvenance(
            quantile_source="native_exact_or_numerical_inverse",
            mean_source="analytical_predictive_mean",
        ),
        source_kind="P2",
        source_binding={
            "selection_sha256": str(selection["selection_sha256"]),
            "validation_artifact_sha256": validation_artifact.artifact_sha256,
            "prediction_artifact_sha256": prediction_artifact.artifact_sha256,
        },
        model_seeds=prediction_artifact.model_seeds,
    )
    result = {
        "quantiles": quantiles,
        "p_zero": p_zero,
        "mean": means,
        "max_x_error_bound": max_error,
        "iterations": max_iterations,
        "bracket_expansions": max_expansions,
        "method": "chunked_bracket_expansion_then_bisection_true_linear_CDF_pool",
        "pooled_flat_case_chunk": POOLED_FLAT_CASE_CHUNK,
        "pooled_q_chunk": POOLED_Q_CHUNK,
        "model_seed_axis": list(validation_artifact.model_seeds),
        "cross_seed_distribution_averaging": False,
        "forecast_frames": forecast_frames,
        "scientific_forecast_artifacts": forecast_artifacts,
    }
    return {
        **result,
        "validation_artifact_sha256": validation_artifact.artifact_sha256,
        "prediction_artifact_sha256": prediction_artifact.artifact_sha256,
        "selection_sha256": str(selection["selection_sha256"]),
        "selection_source": "validation_only",
    }


def select_primary_pool(
    *,
    p2_selection: Mapping[str, Any],
    p3_selection: Mapping[str, Any],
    validation_artifact: SealedValidationArtifact,
) -> dict[str, Any]:
    """Freeze R3 primary pool by sCRPS, tail sQL, then simpler P2."""

    validation_artifact.require_validation_split()
    p2_payload = _verify_selection_binding(p2_selection, validation_artifact)
    p3_payload = _verify_selection_binding(p3_selection, validation_artifact)
    if p2_payload.get("pool") != "P2" or p3_payload.get("pool") != "P3":
        raise PredictionIntegrityError("primary pool requires sealed P2 and P3 selections")
    p2 = float(p2_payload.get("validation_sCRPS", np.nan))
    p3 = float(p3_payload.get("validation_sCRPS", np.nan))
    p2_tail = float(p2_payload.get("validation_tail_sQL", np.nan))
    p3_tail = float(p3_payload.get("validation_tail_sQL", np.nan))
    if not all(np.isfinite(value) for value in (p2, p3, p2_tail, p3_tail)):
        raise PredictionIntegrityError("primary pool validation score contains NaN/Inf")
    primary = min(
        ("P2", "P3"),
        key=lambda name: (
            p2 if name == "P2" else p3,
            p2_tail if name == "P2" else p3_tail,
            0 if name == "P2" else 1,
        ),
    )
    return _seal_selection({
        "primary_pool": primary,
        "validation_sCRPS": p2 if primary == "P2" else p3,
        "validation_tail_sQL": p2_tail if primary == "P2" else p3_tail,
        "candidate_scores": {
            "P2": {"sCRPS": p2, "tail_sQL": p2_tail},
            "P3": {"sCRPS": p3, "tail_sQL": p3_tail},
        },
        "tie_break_order": ["validation_sCRPS", "mean_q95_q99_sQL", "P2", "P3"],
        "selection_source": "validation_only",
        "candidate_selection_sha256": {
            "P2": str(p2_selection["selection_sha256"]),
            "P3": str(p3_selection["selection_sha256"]),
        },
    }, validation_artifact)


def select_pzero_pool(
    *,
    validation_teacher_p_zero: np.ndarray,
    validation_y: np.ndarray,
    validation_case_keys: Sequence[Sequence[Any]],
    validation_artifact: SealedValidationArtifact,
) -> dict[str, Any]:
    """Select P3's separate zero-mass simplex using validation Brier only."""

    probabilities = np.asarray(validation_teacher_p_zero, dtype=np.float64)
    y = np.asarray(validation_y, dtype=np.float64)
    if not isinstance(validation_artifact, SealedValidationArtifact):
        raise PredictionIntegrityError("P3 p0 selection requires a sealed validation artifact")
    validation_artifact.require_validation_split()
    if y.ndim != 1:
        raise PredictionIntegrityError("p0 validation target must be one-dimensional")
    validation_artifact.verify_target_y(
        validation_y=y, validation_case_keys=validation_case_keys
    )
    seed_count = len(validation_artifact.model_seeds)
    if probabilities.shape != (seed_count, 3, y.size):
        raise PredictionIntegrityError(
            "teacher p0 must have explicit shape [seed, 3, validation_case]"
        )
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise PredictionIntegrityError("p0 validation target is invalid")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise PredictionIntegrityError("teacher p0 contains an invalid probability")
    validation_artifact.verify_teacher_components(
        {
            head: {"p_zero": probabilities[:, index]}
            for index, head in enumerate(HEAD_ORDER)
        }
    )
    target = (y == 0.0).astype(np.float64)
    states = simplex_grid()
    scores = []
    for state in states:
        per_seed = [
            float(
                np.mean(
                    (
                        np.tensordot(state, probabilities[seed_index], axes=(0, 0))
                        - target
                    )
                    ** 2
                )
            )
            for seed_index in range(seed_count)
        ]
        scores.append(float(np.mean(per_seed)))
    best_index = min(
        range(len(states)), key=lambda index: (scores[index], tuple(states[index]))
    )
    if not np.all(np.isfinite(scores)):
        raise PredictionIntegrityError("P3 p0 candidate scores contain NaN/Inf")
    return _seal_selection({
        "pool": "P3_PZERO",
        "weights": states[best_index].tolist(),
        "validation_zero_brier": scores[best_index],
        "candidate_count": len(states),
        "selection_source": "validation_only",
        "model_seed_axis": list(validation_artifact.model_seeds),
        "seed_aggregation": "equal_mean_of_per_seed_scalar_scores",
        "cross_seed_probability_averaging": False,
        "tie_break": "lexicographic_(w_NB,w_HSNB,w_TWEEDIE_FULL)",
    }, validation_artifact)


def cumulative_max_projection(quantiles: np.ndarray) -> np.ndarray:
    values = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("quantiles must have shape [case, quantile]")
    return np.maximum.accumulate(values, axis=1)


def postprocess_quantiles_with_pzero(
    *,
    quantiles: np.ndarray,
    p_zero: np.ndarray,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> dict[str, Any]:
    """Apply cummax -> q<=p0 zeroing -> cummax, in that fixed order."""

    values = np.asarray(quantiles, dtype=np.float64)
    probabilities = np.asarray(p_zero, dtype=np.float64)
    q = np.asarray(quantile_grid, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != q.size
        or probabilities.shape != values.shape[:-1]
    ):
        raise PredictionIntegrityError("p0-aware quantile postprocess shapes are inconsistent")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise PredictionIntegrityError("p0-aware quantiles are invalid")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise PredictionIntegrityError("p0-aware zero probabilities are invalid")
    flat_values = values.reshape(-1, q.size)
    flat_probabilities = probabilities.reshape(-1)
    monotone_first = cumulative_max_projection(flat_values)
    zero_mask = q[None, :] <= flat_probabilities[:, None]
    zero_adjustment = zero_mask & (monotone_first != 0.0)
    zeroed = monotone_first.copy()
    zeroed[zero_mask] = 0.0
    final = cumulative_max_projection(zeroed).reshape(values.shape)
    return {
        "quantiles": final,
        "pre_crossing_rate": _crossing_rate(flat_values),
        "post_crossing_rate": _crossing_rate(final.reshape(-1, q.size)),
        "zero_adjustment_count": int(zero_adjustment.sum()),
        "zero_adjustment_rate": float(zero_adjustment.mean()),
        "order": ["cumulative_max", "zero_where_q_le_p0", "cumulative_max"],
    }


P3_ZERO_QUANTILE_TOLERANCE = 1e-12
P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD = (
    "P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD"
)


def _require_p3_grid_coherence(
    quantiles: np.ndarray,
    p_zero: np.ndarray,
    quantile_grid: np.ndarray,
) -> None:
    quantiles = np.asarray(quantiles, dtype=np.float64)
    p_zero = np.asarray(p_zero, dtype=np.float64)
    quantile_grid = np.asarray(quantile_grid, dtype=np.float64)
    if quantiles.shape[:-1] != p_zero.shape or quantiles.shape[-1] != quantile_grid.size:
        raise PredictionIntegrityError(P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD)
    zero_quantile = np.isclose(
        quantiles, 0.0, rtol=0.0, atol=P3_ZERO_QUANTILE_TOLERANCE
    )
    below_or_at_mass = quantile_grid.reshape(
        (1,) * p_zero.ndim + (quantile_grid.size,)
    ) <= p_zero[..., None]
    if not np.array_equal(zero_quantile, below_or_at_mass):
        raise PredictionIntegrityError(P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD)
    derived = coverage_quantiles_from_common_grid(
        quantiles,
        quantile_grid,
        p_zero=p_zero,
    )
    for probability, name in ((0.025, "q025"), (0.975, "q975")):
        derived_zero = np.isclose(
            np.asarray(derived[name], dtype=np.float64),
            0.0,
            rtol=0.0,
            atol=P3_ZERO_QUANTILE_TOLERANCE,
        )
        if not np.array_equal(derived_zero, probability <= p_zero):
            raise PredictionIntegrityError(P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD)


def apply_quantile_specific_pool(
    *,
    teacher_quantiles: np.ndarray,
    teacher_p_zero: np.ndarray,
    selection: Mapping[str, Any],
    validation_artifact: SealedValidationArtifact,
    prediction_artifact: SealedValidationArtifact,
    branch_eligibility: BranchEligibility,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
) -> dict[str, Any]:
    """Apply a frozen P3 path to any split, then perform the frozen projection."""

    frozen = _verify_selection_binding(selection, validation_artifact)
    if frozen.get("pool") != "P3":
        raise PredictionIntegrityError("P3 application requires a sealed P3 selection")
    if (
        not isinstance(branch_eligibility, BranchEligibility)
        or not branch_eligibility.confirmatory_eligible
        or branch_eligibility.role != "CONFIRMATORY"
    ):
        raise PredictionIntegrityError(
            "P3 confirmatory application requires eligible branch lineage"
        )
    predictions = np.asarray(teacher_quantiles, dtype=np.float64)
    weights = np.asarray(frozen.get("weights_by_quantile"), dtype=np.float64)
    q = np.asarray(quantile_grid, dtype=np.float64)
    seed_count = len(prediction_artifact.model_seeds)
    if prediction_artifact.model_seeds != validation_artifact.model_seeds:
        raise PredictionIntegrityError(
            "P3 application seed manifest differs from validation selection"
        )
    if predictions.ndim != 4 or predictions.shape[:2] != (seed_count, 3):
        raise PredictionIntegrityError(
            "teacher quantiles must have explicit shape [seed, 3, case, quantile]"
        )
    if tuple(map(float, q)) != CRPS_QUANTILE_GRID:
        raise ValueError("P3 outer quantile grid is frozen")
    if weights.shape != (predictions.shape[3], 3) or predictions.shape[3] != q.size:
        raise PredictionIntegrityError("P3 weight path shape does not match teacher quantiles")
    for row in weights:
        validate_simplex(row)
    if not np.all(np.isfinite(predictions)) or np.any(predictions < 0.0):
        raise PredictionIntegrityError("teacher quantiles are invalid")
    if np.any(np.diff(predictions, axis=3) < 0.0):
        raise PredictionIntegrityError("a teacher quantile function crosses")
    pzero_values = np.asarray(teacher_p_zero, dtype=np.float64)
    if pzero_values.shape != (seed_count, 3, predictions.shape[2]):
        raise PredictionIntegrityError(
            "outer P3 teacher p0 must have explicit shape [seed, 3, case]"
        )
    if not np.all(np.isfinite(pzero_values)) or np.any(
        (pzero_values < 0.0) | (pzero_values > 1.0)
    ):
        raise PredictionIntegrityError("outer P3 teacher p0 is invalid")
    pzero_pool = frozen.get("p_zero_pool")
    if not isinstance(pzero_pool, Mapping):
        raise PredictionIntegrityError("sealed P3 selection lacks its p0 pool")
    _verify_selection_binding(pzero_pool, validation_artifact)
    pzero_simplex = validate_simplex(
        np.asarray(pzero_pool.get("weights"), dtype=np.float64)
    )
    prediction_artifact.verify_teacher_components(
        {
            head: {
                "quantiles": predictions[:, index],
                "p_zero": pzero_values[:, index],
            }
            for index, head in enumerate(HEAD_ORDER)
        }
    )
    pooled_pzero = np.einsum("h,shc->sc", pzero_simplex, pzero_values)
    raw = np.empty((seed_count, predictions.shape[2], q.size), dtype=np.float64)
    projected = np.empty_like(raw)
    zero_adjustment_count = 0
    pre_crossing_weighted = 0.0
    post_crossing_weighted = 0.0
    entry_count = 0
    for start in range(0, predictions.shape[2], POOLED_FLAT_CASE_CHUNK):
        stop = min(start + POOLED_FLAT_CASE_CHUNK, predictions.shape[2])
        raw_chunk = np.einsum(
            "qh,shcq->scq", weights, predictions[:, :, start:stop, :]
        )
        chunk = postprocess_quantiles_with_pzero(
            quantiles=raw_chunk,
            p_zero=pooled_pzero[:, start:stop],
            quantile_grid=q,
        )
        raw[:, start:stop] = raw_chunk
        projected[:, start:stop] = np.asarray(chunk["quantiles"], dtype=np.float64)
        chunk_entries = raw_chunk.size
        entry_count += chunk_entries
        zero_adjustment_count += int(chunk["zero_adjustment_count"])
        pre_crossing_weighted += float(chunk["pre_crossing_rate"]) * chunk_entries
        post_crossing_weighted += float(chunk["post_crossing_rate"]) * chunk_entries
    _require_p3_grid_coherence(
        projected, pooled_pzero, q
    )
    implied_mean = quantile_implied_mean(projected, q)
    forecast_frames, forecast_artifacts = _seal_pool_forecast_rows(
        quantiles=projected,
        p_zero=pooled_pzero,
        mean=implied_mean,
        target_artifact=prediction_artifact.target_artifact,
        branch_eligibility=branch_eligibility,
        provenance=ScientificForecastProvenance(
            quantile_source="monotone_piecewise_common_grid",
            mean_source="quantile_integral_endpoint_hold",
        ),
        source_kind="P3",
        source_binding={
            "selection_sha256": str(selection["selection_sha256"]),
            "validation_artifact_sha256": validation_artifact.artifact_sha256,
            "prediction_artifact_sha256": prediction_artifact.artifact_sha256,
        },
        model_seeds=prediction_artifact.model_seeds,
    )
    return {
        "quantiles": projected,
        "p_zero": pooled_pzero,
        "raw_quantiles": raw,
        "mean": implied_mean,
        "forecast_frames": forecast_frames,
        "scientific_forecast_artifacts": forecast_artifacts,
        "pre_crossing_rate": pre_crossing_weighted / max(entry_count, 1),
        "post_crossing_rate": post_crossing_weighted / max(entry_count, 1),
        "zero_adjustment_count": zero_adjustment_count,
        "zero_adjustment_rate": zero_adjustment_count / max(entry_count, 1),
        "order": ["cumulative_max", "zero_where_q_le_p0", "cumulative_max"],
        "p_zero_weights": pzero_simplex.tolist(),
        "validation_artifact_sha256": validation_artifact.artifact_sha256,
        "prediction_artifact_sha256": prediction_artifact.artifact_sha256,
        "selection_sha256": str(selection["selection_sha256"]),
        "model_seed_axis": list(validation_artifact.model_seeds),
        "cross_seed_quantile_averaging": False,
        "pooled_flat_case_chunk": POOLED_FLAT_CASE_CHUNK,
        "pooled_q_chunk": POOLED_Q_CHUNK,
    }


def _crossing_rate(quantiles: np.ndarray) -> float:
    if quantiles.shape[1] < 2:
        return 0.0
    return float(np.mean(quantiles[:, 1:] < quantiles[:, :-1]))


def _path_for_penalty(
    per_level_losses: np.ndarray,
    states: np.ndarray,
    penalty: float,
) -> tuple[tuple[int, ...], float]:
    if per_level_losses.ndim != 2 or per_level_losses.shape[0] == 0:
        raise ValueError("P3 per-level losses must be a non-empty matrix")
    level_count = per_level_losses.shape[0]
    transition_count = max(level_count - 1, 1)
    # Each entry is (penalized cost, complete lexicographic path).  Explicit paths
    # make the preregistered lexicographic tie-break independent of ndarray order.
    previous: list[tuple[float, tuple[int, ...]]] = [
        (float(per_level_losses[0, state]) / level_count, (state,))
        for state in range(len(states))
    ]
    for level in range(1, per_level_losses.shape[0]):
        current: list[tuple[float, tuple[int, ...]]] = []
        for state_index, state in enumerate(states):
            candidates: list[tuple[float, tuple[int, ...]]] = []
            for previous_index, (previous_cost, previous_path) in enumerate(previous):
                smoothness = float(np.sum((state - states[previous_index]) ** 2))
                candidates.append(
                    (
                        previous_cost
                        + float(per_level_losses[level, state_index]) / level_count
                        + penalty * smoothness / transition_count,
                        previous_path + (state_index,),
                    )
                )
            minimum = min(value[0] for value in candidates)
            tied = [value for value in candidates if value[0] == minimum]
            current.append(min(tied, key=lambda value: value[1]))
        previous = current
    minimum = min(value[0] for value in previous)
    tied = [value for value in previous if value[0] == minimum]
    cost, path = min(tied, key=lambda value: value[1])
    return path, cost


def select_quantile_specific_pool(
    *,
    teacher_quantiles: np.ndarray,
    validation_y: np.ndarray,
    validation_scale: np.ndarray,
    validation_case_keys: Sequence[Sequence[Any]],
    validation_artifact: SealedValidationArtifact,
    quantile_grid: Sequence[float] = CRPS_QUANTILE_GRID,
    penalties: Sequence[float] = POOL_PENALTIES,
    validation_teacher_p_zero: np.ndarray,
) -> dict[str, Any]:
    """Fit the P3 quantile-weight path on validation pinball loss."""

    predictions = np.asarray(teacher_quantiles, dtype=np.float64)
    y = np.asarray(validation_y, dtype=np.float64)
    scale = np.asarray(validation_scale, dtype=np.float64)
    q = np.asarray(quantile_grid, dtype=np.float64)
    if not isinstance(validation_artifact, SealedValidationArtifact):
        raise PredictionIntegrityError("P3 requires a sealed validation artifact")
    validation_artifact.require_validation_split()
    validation_artifact.verify_target_arrays(
        validation_y=y,
        validation_scale=scale,
        validation_case_keys=validation_case_keys,
    )
    if tuple(map(float, q)) != CRPS_QUANTILE_GRID:
        raise ValueError("P3 quantile grid is frozen")
    seed_count = len(validation_artifact.model_seeds)
    if predictions.shape != (seed_count, 3, y.size, q.size):
        raise PredictionIntegrityError(
            "teacher quantiles must have explicit shape [seed, 3, case, quantile]"
        )
    if (
        y.ndim != 1
        or scale.shape != y.shape
        or not np.all(np.isfinite(y))
        or np.any(y < 0.0)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise PredictionIntegrityError("validation target and scale shapes are inconsistent")
    if not np.all(np.isfinite(predictions)) or np.any(predictions < 0.0):
        raise PredictionIntegrityError("teacher quantiles are invalid")
    if np.any(np.diff(predictions, axis=-1) < 0.0):
        raise PredictionIntegrityError("a teacher quantile function crosses")
    teacher_pzero = np.asarray(validation_teacher_p_zero, dtype=np.float64)
    if teacher_pzero.shape != (seed_count, 3, y.size):
        raise PredictionIntegrityError(
            "teacher p0 must have explicit shape [seed, 3, validation_case]"
        )
    validation_artifact.verify_teacher_components(
        {
            head: {
                "quantiles": predictions[:, index],
                "p_zero": teacher_pzero[:, index],
            }
            for index, head in enumerate(HEAD_ORDER)
        }
    )
    frozen_penalties = tuple(float(value) for value in penalties)
    if frozen_penalties != POOL_PENALTIES:
        raise ValueError(f"P3 penalty grid is frozen at {POOL_PENALTIES}")

    states = simplex_grid()
    pzero = select_pzero_pool(
        validation_teacher_p_zero=validation_teacher_p_zero,
        validation_y=y,
        validation_case_keys=validation_case_keys,
        validation_artifact=validation_artifact,
    )
    validation_pzero = np.einsum(
        "h,shc->sc",
        np.asarray(pzero["weights"], dtype=np.float64),
        teacher_pzero,
    )
    per_level_losses = np.empty((q.size, len(states)), dtype=np.float64)
    for level, probability in enumerate(q):
        loss_sums = np.zeros((seed_count, len(states)), dtype=np.float64)
        for start in range(0, y.size, POOLED_FLAT_CASE_CHUNK):
            stop = min(start + POOLED_FLAT_CASE_CHUNK, y.size)
            pooled = np.einsum(
                "wh,shc->swc",
                states,
                predictions[:, :, start:stop, level],
            )
            error = y[None, None, start:stop] - pooled
            losses = np.maximum(probability * error, (probability - 1.0) * error)
            loss_sums += np.sum(losses / scale[None, None, start:stop], axis=2)
        per_level_losses[level] = np.mean(loss_sums / y.size, axis=0)

    candidates: list[dict[str, Any]] = []
    infeasible_penalties: list[float] = []
    for penalty in frozen_penalties:
        path, penalized_loss = _path_for_penalty(per_level_losses, states, penalty)
        quantiles = np.stack(
            [
                np.einsum(
                    "h,shc->sc",
                    states[state_index],
                    predictions[:, :, :, level],
                )
                for level, state_index in enumerate(path)
            ],
            axis=-1,
        )
        processed = postprocess_quantiles_with_pzero(
            quantiles=quantiles,
            p_zero=validation_pzero,
            quantile_grid=q,
        )
        candidate_quantiles = np.asarray(processed["quantiles"], dtype=np.float64)
        try:
            _require_p3_grid_coherence(candidate_quantiles, validation_pzero, q)
        except PredictionIntegrityError:
            infeasible_penalties.append(penalty)
            continue
        validation_loss = float(
            np.mean(
                [
                    np.mean(
                        approximate_crps(y, candidate_quantiles[seed_index], q)
                        / scale
                    )
                    for seed_index in range(seed_count)
                ]
            )
        )
        tail_positions = [
            index
            for index, probability in enumerate(q)
            if np.isclose(probability, 0.95, rtol=0.0, atol=1e-12)
            or np.isclose(probability, 0.99, rtol=0.0, atol=1e-12)
        ]
        tail_loss = (
            float(
                np.mean(
                    [
                        2.0
                        * np.mean(
                            [
                                np.mean(
                                    pinball_loss(
                                        y,
                                        candidate_quantiles[seed_index, :, level],
                                        q[level],
                                    )
                                    / scale
                                )
                                for seed_index in range(seed_count)
                            ]
                        )
                        for level in tail_positions
                    ]
                )
            )
            if len(tail_positions) == 2
            else None
        )
        candidates.append(
            {
                "penalty": penalty,
                "path": path,
                "penalized_loss": penalized_loss,
                "validation_scrps": validation_loss,
                "validation_tail_sql": tail_loss,
                "quantiles": quantiles,
                "processed": processed,
            }
        )
    if not candidates:
        raise PredictionIntegrityError(P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD)
    if not all(
        np.isfinite(candidate["validation_scrps"])
        and candidate["validation_tail_sql"] is not None
        and np.isfinite(candidate["validation_tail_sql"])
        and np.isfinite(candidate["penalized_loss"])
        for candidate in candidates
    ):
        raise PredictionIntegrityError("P3 candidate scores contain NaN/Inf")
    best_loss = min(candidate["validation_scrps"] for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if candidate["validation_scrps"] == best_loss
    ]
    selected = min(
        tied,
        key=lambda candidate: (
            candidate["validation_tail_sql"]
            if candidate["validation_tail_sql"] is not None
            else 0.0,
            candidate["penalty"],
            candidate["path"],
        ),
    )
    processed = selected["processed"]
    raw = np.asarray(selected["quantiles"], dtype=np.float64)
    projected = np.asarray(processed["quantiles"], dtype=np.float64)
    return _seal_selection({
        "pool": "P3",
        "weights_by_quantile": [states[index].tolist() for index in selected["path"]],
        "path_indices": list(selected["path"]),
        "penalty": selected["penalty"],
        "path_selection_validation_sCRPS": selected["validation_scrps"],
        "path_selection_validation_tail_sQL": selected["validation_tail_sql"],
        "validation_sCRPS": selected["validation_scrps"],
        "validation_tail_sQL": selected["validation_tail_sql"],
        "penalized_objective": selected["penalized_loss"],
        "pre_crossing_rate": processed["pre_crossing_rate"],
        "post_crossing_rate": processed["post_crossing_rate"],
        "zero_adjustment_count": processed["zero_adjustment_count"],
        "zero_adjustment_rate": processed["zero_adjustment_rate"],
        "postprocess_order": processed["order"],
        "raw_quantiles_sha256": _component_sha256(raw),
        "projected_quantiles_sha256": _component_sha256(projected),
        "candidate_penalties": list(frozen_penalties),
        "infeasible_penalties": infeasible_penalties,
        "state_count": len(states),
        "selection_source": "validation_only",
        "model_seed_axis": list(validation_artifact.model_seeds),
        "seed_aggregation": "equal_mean_of_per_seed_scalar_scores",
        "cross_seed_quantile_averaging": False,
        "pooled_flat_case_chunk": POOLED_FLAT_CASE_CHUNK,
        "pooled_q_chunk": POOLED_Q_CHUNK,
        "p_zero_pool": pzero,
        "validation_p_zero_sha256": _component_sha256(validation_pzero),
        "mean_source": "quantile_integral_endpoint_hold",
        "distribution_coherence": "quantile_plus_separately_pooled_p0_not_single_CDF",
        "grid_p0_quantile_consistency": "q_le_p0_iff_Qq_zero",
        "grid_zero_tolerance": P3_ZERO_QUANTILE_TOLERANCE,
    }, validation_artifact)
