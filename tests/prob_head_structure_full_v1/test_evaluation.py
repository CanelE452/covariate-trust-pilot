from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.prob_head_structure_full_v1.data as data_module
import experiments.prob_head_structure_full_v1.evaluation as evaluation_module
from experiments.prob_head_structure_full_v1 import integrity
from experiments.prob_head_structure_full_v1.data import (
    RealSplit,
    build_window_request,
    file_sha256,
    make_history_windows,
    seal_count_primary_dataset_audit,
    seal_train_only_sample_manifest,
)

from experiments.prob_head_structure_full_v1.bootstrap import (
    bootstrap_draws_for_tier,
    factorial_temporal_contrasts,
    paired_cluster_bootstrap,
    pairwise_loss_correlation_bootstrap,
    resample_intact_clusters,
)
from experiments.prob_head_structure_full_v1.evaluation import (
    CRPS_QUANTILE_GRID,
    EVALUATION_QUANTILE_GRID,
    DiagnosticEvaluationResult,
    PredictionIntegrityError,
    ScientificForecastProvenance,
    ScientificForecastArtifact,
    SealedEvaluationTarget,
    approximate_crps,
    coverage_quantiles_from_common_grid,
    diagnostic_empirical_cdf,
    diagnostic_empirical_quantiles,
    evaluate_diagnostic_prediction_frame,
    evaluate_native_diagnostic_prediction_frame,
    evaluate_prediction_frame,
    exact_prediction_join,
    midpoint_cell_widths,
    pairwise_relative_scrps_gaps,
    quantile_column,
    quantile_implied_mean,
    relative_loss_improvement,
    summarize_oracle_ladder,
    summarize_two_head_diagnostic_oracle_ladder,
    summarize_practical_winners,
    validate_exact_prediction_keys,
    _seal_scientific_forecast_artifact,
)
from experiments.prob_head_structure_full_v1.integrity import BranchEligibility, GateStatus
from experiments.prob_head_structure_full_v1.distributions import (
    NegativeBinomialDistribution,
    ShiftedHurdleNegativeBinomialDistribution,
    TweedieDistribution,
)
from experiments.prob_head_structure_full_v1.pooling import (
    HEAD_ORDER,
    SealedValidationArtifact,
    _path_for_penalty,
    cumulative_max_projection,
    cdf_callable_for_distribution,
    diagnostic_two_head_cdf_pool,
    apply_quantile_specific_pool,
    apply_global_cdf_pool,
    invert_cdf,
    invert_pooled_cdf,
    linear_cdf_pool,
    select_global_cdf_pool,
    select_best_single_teacher,
    select_pzero_pool,
    select_primary_pool,
    select_quantile_specific_pool,
    simplex_grid,
    _component_sha256,
)


KEYS = ["dataset_id", "series_id", "origin", "step"]
EVAL_PROVENANCE = {
    "provenance": ScientificForecastProvenance(
        quantile_source="native_exact_or_numerical_inverse",
        mean_source="analytical_predictive_mean",
    ),
}


def _prediction_frame() -> pd.DataFrame:
    rows = []
    for series_id, y in [("a", 0.0), ("b", 2.0)]:
        row = {
            "dataset_id": "auto",
            "series_id": series_id,
            "origin": 1,
            "step": 0,
            "y": y,
        "scale": 2.0,
            "target_mask": True,
            "p_zero": 1.0 if y == 0 else 0.0,
            "mean": y,
        }
        for q in EVALUATION_QUANTILE_GRID:
            row[quantile_column(q)] = y
        rows.append(row)
    return pd.DataFrame(rows)


def _sealed_target(
    frame: pd.DataFrame | None = None, *, split_name: str = "validation"
) -> SealedEvaluationTarget:
    target_frame = _prediction_frame() if frame is None else frame
    if target_frame["series_id"].duplicated().any():
        raise ValueError("test target helper expects one validation row per series")
    series_ids = target_frame["series_id"].astype(str).to_numpy()
    horizon = 1
    split = RealSplit(
        train=(0, 1),
        validation=(1, 2),
        warmup=(2, 3),
        origins=(3, 4, 5, 6, 7, 8),
        horizon=horizon,
        lookback=1,
    )
    panel_y = np.zeros((len(target_frame), 9), dtype=np.float64)
    panel_y[:, 0] = np.rint(target_frame["scale"].to_numpy(dtype=np.float64))
    panel_y[:, 1] = target_frame["y"].to_numpy(dtype=np.float64)
    for origin in split.origins:
        panel_y[:, origin] = target_frame["y"].to_numpy(dtype=np.float64)
    panel = {
        "name": "auto",
        "y": panel_y,
        "split": split,
        "available_from": np.zeros(len(target_frame), dtype=np.int32),
        "series_id": series_ids,
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "auto.bin"
        source.write_bytes(b"auto")
        digest = file_sha256(source)
        source_manifest = integrity.build_source_manifest(
            root,
            {source.name: digest},
            repository_root_identity="test_source_root",
        )
        panel["provenance"] = {
            "dataset_id": "auto",
            "sources": [
                {
                    "path": source.resolve().as_posix(),
                    "size_bytes": source.stat().st_size,
                    "sha256": digest,
                    "expected_sha256": digest,
                    "status": "PASS",
                }
            ],
        }
        audit = seal_count_primary_dataset_audit(
            panel, source_manifest=source_manifest
        )
        # Production sealing deliberately leaves noncanonical toy datasets
        # ineligible.  This test-only record exercises downstream cryptographic
        # lineage without weakening the data-layer production factory.
        audit = dict(audit)
        audit["canonical_source_attested"] = True
        audit["confirmatory_eligible"] = True
        audit["audit_sha256"] = data_module._dataset_audit_hash(audit)
        original_verifier = data_module._verify_sealed_dataset_audit
        original_sealer = evaluation_module.seal_count_primary_dataset_audit
        data_module._verify_sealed_dataset_audit = lambda raw: deepcopy(dict(raw))
        evaluation_module.seal_count_primary_dataset_audit = (
            lambda _panel, *, source_manifest: deepcopy(audit)
        )
        try:
            sample_manifest = seal_train_only_sample_manifest(
                panel,
                dataset_audit=audit,
                runtime_tier="MINIMAL-COMPLETE",
            )
            request = build_window_request(
                dataset_id="auto",
                split=split,
                panel_length=panel_y.shape[1],
                role=split_name,
                origins=(split.validation[0],)
                if split_name == "validation"
                else split.origins,
                panel=panel,
                dataset_audit=audit,
                sample_manifest=sample_manifest,
            )
            batch = make_history_windows(
                panel, request=request, dataset_audit=audit
            )
            return SealedEvaluationTarget.seal(
                window_batch=batch,
                window_request=request,
                panel=panel,
                dataset_audit=audit,
                source_manifest=source_manifest,
                sample_manifest=sample_manifest,
                preregistration_sha256="3" * 64,
                dataset_manifest_sha256="4" * 64,
            )
        finally:
            evaluation_module.seal_count_primary_dataset_audit = original_sealer
            data_module._verify_sealed_dataset_audit = original_verifier


def _confirmatory_lineage() -> BranchEligibility:
    return BranchEligibility.begin("R1", [])


def _diagnostic_lineage() -> BranchEligibility:
    return BranchEligibility.begin(
        "R2", [GateStatus.scientific_failure("R1")]
    )


def _synthetic_lineage() -> BranchEligibility:
    return BranchEligibility.begin(
        "S2",
        [GateStatus.passed("DGP_BALANCE"), GateStatus.passed("S1")],
    )


def _scientific_evaluate(frame: pd.DataFrame):
    target = _sealed_target(frame)
    lineage = _confirmatory_lineage()
    return evaluate_prediction_frame(
        frame,
        forecast_artifact=_test_forecast_artifact(
            frame, target=target, lineage=lineage
        ),
        target_artifact=target,
        branch_eligibility=lineage,
    )


def _test_forecast_artifact(
    frame: pd.DataFrame,
    *,
    target: SealedEvaluationTarget,
    lineage: BranchEligibility,
    provenance: ScientificForecastProvenance = EVAL_PROVENANCE["provenance"],
) -> ScientificForecastArtifact:
    return _seal_scientific_forecast_artifact(
        frame=frame,
        provenance=provenance,
        target_artifact=target,
        branch_eligibility=lineage,
        source_kind="NATIVE_TEACHER",
        source_binding={"verified_test_adapter_sha256": "a" * 64},
    )


def _validation_artifact(
    teacher_components: dict[str, dict[str, object]],
    frame: pd.DataFrame | None = None,
    *,
    model_seeds: tuple[int, ...] = (2026090511,),
    split_name: str = "validation",
) -> SealedValidationArtifact:
    target_frame = _prediction_frame() if frame is None else frame
    target_artifact = _sealed_target(target_frame, split_name=split_name)
    target_payload = target_artifact.as_dict()["payload"]
    sealed_case_keys = [
        [row[column] for column in KEYS] for row in target_payload["rows"]
    ]
    return SealedValidationArtifact.seal(
        target_artifact=target_artifact,
        head_order=HEAD_ORDER,
        teacher_predictions=teacher_components,
        teacher_case_keys={head: sealed_case_keys for head in HEAD_ORDER},
        teacher_model_seeds={
            head: list(model_seeds) for head in HEAD_ORDER
        },
        validation_group_ids=[row[1] for row in sealed_case_keys],
        sample_manifest_sha256=target_payload["sample_manifest_sha256"],
        source_manifest_sha256=target_payload["source_manifest_sha256"],
        preregistration_sha256=target_payload["preregistration_sha256"],
        dataset_manifest_sha256=target_payload["dataset_manifest_sha256"],
    )


def _native_distribution_grid(
    case_count: int,
    *,
    model_seeds: tuple[int, ...] = (2026090511,),
) -> tuple[dict[str, dict[str, object]], list[list[object]]]:
    components = {head: {} for head in HEAD_ORDER}
    rows: list[list[object]] = []
    for seed_index, seed in enumerate(model_seeds):
        mu = torch.full((case_count, 1), 1.0 + seed_index, dtype=torch.float64)
        distributions = [
            NegativeBinomialDistribution(mu, torch.full_like(mu, 2.0)),
            ShiftedHurdleNegativeBinomialDistribution(
                torch.full_like(mu, 0.5), mu, torch.full_like(mu, 2.0)
            ),
            TweedieDistribution(
                mu, torch.full_like(mu, 1.0), torch.full_like(mu, 1.5)
            ),
        ]
        rows.append(distributions)
        for head, distribution in zip(HEAD_ORDER, distributions, strict=True):
            parameter_names = {
                "NB": ("mu", "r"),
                "HSNB": ("pi", "mu", "r"),
                "TWEEDIE_FULL": ("mu", "phi", "p"),
            }[head]
            components[head][f"cdf_parameters_seed_{seed}"] = {
                name: getattr(distribution, name).detach().cpu().numpy()
                for name in parameter_names
            }
    return components, rows


def _bound_distribution_grid(
    distributions: list[list[object]],
    artifact: SealedValidationArtifact,
) -> tuple[list[list[object]], np.ndarray]:
    bound: list[list[object]] = []
    means: list[list[np.ndarray]] = []
    for seed, row in zip(artifact.model_seeds, distributions, strict=True):
        bound_row = []
        mean_row = []
        for head, distribution in zip(HEAD_ORDER, row, strict=True):
            function, mean, _ = cdf_callable_for_distribution(
                distribution,
                head=head,
                model_seed=seed,
                validation_artifact=artifact,
            )
            bound_row.append(function)
            mean_row.append(mean)
        bound.append(bound_row)
        means.append(mean_row)
    return bound, np.max(np.asarray(means), axis=1)


def _pool_target_frame(y: np.ndarray, scale: np.ndarray) -> pd.DataFrame:
    frame = _prediction_frame().iloc[: len(y)].copy().reset_index(drop=True)
    if len(frame) != len(y):
        rows = []
        for index, (target, divisor) in enumerate(zip(y, scale, strict=True)):
            row = _prediction_frame().iloc[0].to_dict()
            row.update(series_id=f"pool-{index}", y=float(target), scale=float(divisor))
            rows.append(row)
        frame = pd.DataFrame(rows)
    frame["y"] = np.asarray(y, dtype=np.float64)
    frame["scale"] = np.asarray(scale, dtype=np.float64)
    frame["target_mask"] = True
    return frame


def _case_keys(frame: pd.DataFrame) -> list[list[object]]:
    return frame[KEYS].values.tolist()


def _sealed_case_keys(artifact: SealedValidationArtifact) -> list[list[object]]:
    return artifact.as_dict()["payload"]["case_keys"]


def test_validation_selector_artifact_is_content_bound_and_validation_only() -> None:
    q = np.asarray(CRPS_QUANTILE_GRID)
    quantiles = np.ones((1, 3, 2, q.size), dtype=np.float64)
    p_zero = np.full((1, 3, 2), 0.2, dtype=np.float64)
    components = {
        head: {
            "quantiles": quantiles[:, index],
            "p_zero": p_zero[:, index],
        }
        for index, head in enumerate(HEAD_ORDER)
    }
    artifact = _validation_artifact(components)
    selected = select_best_single_teacher(
        validation_teacher_quantiles=quantiles,
        validation_teacher_p_zero=p_zero,
        validation_y=np.array([0.0, 2.0]),
        validation_scale=np.full(2, 2.0),
        validation_case_keys=_sealed_case_keys(artifact),
        validation_artifact=artifact,
    )
    assert selected["validation_artifact_sha256"] == artifact.artifact_sha256
    assert selected["selection_sha256"]

    altered = quantiles.copy()
    altered[:, 0] = 0.9
    with pytest.raises(PredictionIntegrityError, match="prediction component"):
        select_best_single_teacher(
            validation_teacher_quantiles=altered,
            validation_teacher_p_zero=p_zero,
            validation_y=np.array([0.0, 2.0]),
            validation_scale=np.full(2, 2.0),
            validation_case_keys=_sealed_case_keys(artifact),
            validation_artifact=artifact,
        )
    with pytest.raises(TypeError):
        select_best_single_teacher(
            validation_teacher_quantiles=quantiles,
            validation_teacher_p_zero=p_zero,
            validation_y=np.array([0.0, 2.0]),
            validation_scale=np.full(2, 2.0),
            validation_case_keys=_sealed_case_keys(artifact),
        )

    with pytest.raises(PredictionIntegrityError, match="canonical head order"):
        SealedValidationArtifact.seal(
            target_artifact=_sealed_target(split_name="validation"),
            head_order=("HSNB", "NB", "TWEEDIE_FULL"),
            teacher_predictions={
                "HSNB": components["HSNB"],
                "NB": components["NB"],
                "TWEEDIE_FULL": components["TWEEDIE_FULL"],
            },
            teacher_case_keys={
                "HSNB": _case_keys(_prediction_frame()),
                "NB": _case_keys(_prediction_frame()),
                "TWEEDIE_FULL": _case_keys(_prediction_frame()),
            },
            teacher_model_seeds={
                "HSNB": [2026090511],
                "NB": [2026090511],
                "TWEEDIE_FULL": [2026090511],
            },
            validation_group_ids=["a", "b"],
            sample_manifest_sha256="5" * 64,
            source_manifest_sha256="2" * 64,
            preregistration_sha256="3" * 64,
            dataset_manifest_sha256="4" * 64,
        )


def test_sealed_evaluation_target_binds_keys_targets_scale_mask_and_lineage() -> None:
    frame = _prediction_frame()
    artifact = _sealed_target(frame)
    confirmatory = _confirmatory_lineage()
    forecast = _test_forecast_artifact(
        frame, target=artifact, lineage=confirmatory
    )
    result = evaluate_prediction_frame(
        frame,
        forecast_artifact=forecast,
        target_artifact=artifact,
        branch_eligibility=confirmatory,
    )
    assert result.validity["target_artifact_sha256"] == artifact.artifact_sha256

    for column, replacement in (("y", 1.0), ("scale", 3.0), ("target_mask", False)):
        tampered = frame.copy()
        tampered.loc[0, column] = replacement
        with pytest.raises(PredictionIntegrityError, match="target artifact"):
            evaluate_prediction_frame(
                tampered,
                forecast_artifact=forecast,
                target_artifact=artifact,
                branch_eligibility=confirmatory,
            )

    fractional = frame.copy()
    fractional.loc[0, "y"] = 0.5
    with pytest.raises((PredictionIntegrityError, ValueError), match="count.*support"):
        _sealed_target(fractional)
    with pytest.raises(PredictionIntegrityError, match="eligible branch lineage"):
        evaluate_prediction_frame(
            frame,
            forecast_artifact=forecast,
            target_artifact=artifact,
            branch_eligibility=_diagnostic_lineage(),
        )

    diagnostic_lineage = _diagnostic_lineage()
    diagnostic = evaluate_native_diagnostic_prediction_frame(
        frame,
        forecast_artifact=_test_forecast_artifact(
            frame, target=artifact, lineage=diagnostic_lineage
        ),
        target_artifact=artifact,
        branch_eligibility=diagnostic_lineage,
    )
    assert diagnostic.validity["confirmatory_eligible"] is False
    assert diagnostic.validity["scientific_role"].startswith(
        "DIAGNOSTIC_CONTINUATION_AFTER_"
    )


def test_exact_prediction_keys_reject_duplicates_missing_extra_and_order() -> None:
    reference = _prediction_frame()
    target_artifact = _sealed_target(reference)
    validate_exact_prediction_keys(
        {"NB": reference, "HSNB": reference.copy()},
        target_artifact=target_artifact,
    )
    joined = exact_prediction_join(
        {"NB": reference, "HSNB": reference.copy()},
        target_artifact=target_artifact,
    )
    assert joined[KEYS].equals(reference[KEYS])
    assert "mean__NB" in joined and "mean__HSNB" in joined

    duplicate = pd.concat([reference, reference.iloc[[0]]], ignore_index=True)
    with pytest.raises(PredictionIntegrityError, match="duplicate"):
        validate_exact_prediction_keys(
            {"NB": reference, "HSNB": duplicate}, target_artifact=target_artifact
        )

    with pytest.raises(PredictionIntegrityError, match="row count"):
        validate_exact_prediction_keys(
            {"NB": reference, "HSNB": reference.iloc[:1]},
            target_artifact=target_artifact,
        )

    extra = reference.copy()
    extra.loc[len(extra)] = extra.iloc[-1]
    extra.loc[len(extra) - 1, "series_id"] = "c"
    with pytest.raises(PredictionIntegrityError, match="row count"):
        validate_exact_prediction_keys(
            {"NB": reference, "HSNB": extra}, target_artifact=target_artifact
        )

    reordered = reference.iloc[::-1].reset_index(drop=True)
    with pytest.raises(PredictionIntegrityError, match="order"):
        validate_exact_prediction_keys(
            {"NB": reference, "HSNB": reordered}, target_artifact=target_artifact
        )

    for column, replacement in (("y", 1.0), ("scale", 99.0), ("target_mask", False)):
        altered = reference.copy()
        altered.loc[0, column] = replacement
        with pytest.raises(PredictionIntegrityError, match="target artifact"):
            validate_exact_prediction_keys(
                {"NB": reference, "HSNB": altered},
                target_artifact=target_artifact,
            )


def test_midpoint_weighted_crps_matches_hand_calculation() -> None:
    q = np.asarray(CRPS_QUANTILE_GRID)
    weights = midpoint_cell_widths(q)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx((q[0] + q[1]) / 2.0)
    assert weights[-1] == pytest.approx(1.0 - (q[-2] + q[-1]) / 2.0)

    y = np.array([0.0])
    quantiles = np.ones((1, len(q)))
    expected = 2.0 * np.sum(weights * (1.0 - q))
    assert approximate_crps(y, quantiles, q)[0] == pytest.approx(expected)
    assert relative_loss_improvement(2.0, 1.5) == pytest.approx(0.25)
    with pytest.raises(PredictionIntegrityError, match="positive baseline"):
        relative_loss_improvement(0.0, 0.0)


def test_evaluation_retains_all_aggregation_levels_and_hand_metrics() -> None:
    result = _scientific_evaluate(_prediction_frame())
    assert set(result.levels) == {"step", "series_origin", "series", "dataset"}
    dataset = result.levels["dataset"].iloc[0]
    assert dataset["sCRPS"] == pytest.approx(0.0)
    assert dataset["zero_brier"] == pytest.approx(0.0)
    assert dataset["NRMSE"] == pytest.approx(0.0)
    assert dataset["NMAE"] == pytest.approx(0.0)
    assert dataset["coverage_error_50"] == pytest.approx(0.5)
    assert dataset["interval_width_95"] == pytest.approx(0.0)
    assert result.to_serializable()["dataset"][0]["dataset_id"] == "auto"
    assert result.validity["quantile_source"] == "native_exact_or_numerical_inverse"
    assert result.validity["confirmatory_eligible"] is True
    json.dumps(result.to_serializable(), allow_nan=False)

def test_scientific_evaluator_rejects_empirical_sample_provenance() -> None:
    student = ScientificForecastProvenance(
        quantile_source="monotone_piecewise_common_grid",
        mean_source="quantile_integral_endpoint_hold",
    )
    frame = _prediction_frame()
    target = _sealed_target(frame)
    lineage = _confirmatory_lineage()
    forecast = _test_forecast_artifact(
        frame, target=target, lineage=lineage, provenance=student
    )
    assert evaluate_prediction_frame(
        frame,
        forecast_artifact=forecast,
        target_artifact=target,
        branch_eligibility=lineage,
    ).validity["mean_source"] == "quantile_integral_endpoint_hold"
    with pytest.raises(PredictionIntegrityError, match="scientific provenance"):
        ScientificForecastProvenance(
            quantile_source="empirical_inverse_from_samples",
            mean_source="empirical_sample_mean",
        )
    with pytest.raises(TypeError):
        evaluate_prediction_frame(
            frame,
            provenance={
                "quantile_source": "empirical_inverse_from_samples",
                "mean_source": "empirical_sample_mean",
            },
            target_artifact=target,
            branch_eligibility=lineage,
        )

    swapped = frame.copy()
    swapped[["p_zero", "mean"]] = swapped[["p_zero", "mean"]].iloc[::-1].to_numpy()
    with pytest.raises(PredictionIntegrityError, match="content.*binding"):
        evaluate_prediction_frame(
            swapped,
            forecast_artifact=forecast,
            target_artifact=target,
            branch_eligibility=lineage,
        )

    diagnostic = evaluate_diagnostic_prediction_frame(
        _prediction_frame(),
        quantile_source="empirical_inverse_from_samples",
        mean_source="empirical_sample_mean",
        scientific_role="DIAGNOSTIC_CONTINUATION_AFTER_SAMPLE_STABILITY_CHECK",
    )
    assert isinstance(diagnostic, DiagnosticEvaluationResult)
    assert diagnostic.validity["confirmatory_eligible"] is False
    assert diagnostic.validity["scientific_role"] == "DIAGNOSTIC_CONTINUATION_AFTER_SAMPLE_STABILITY_CHECK"


def test_invalid_probability_crossing_nonfinite_and_negative_are_integrity_errors() -> None:
    bad = _prediction_frame()
    bad.loc[0, "p_zero"] = 1.1
    with pytest.raises(PredictionIntegrityError, match="probability"):
        _scientific_evaluate(bad)

    bad = _prediction_frame()
    bad.loc[0, quantile_column(0.50)] = 5.0
    bad.loc[0, quantile_column(0.55)] = 4.0
    with pytest.raises(PredictionIntegrityError, match="cross"):
        _scientific_evaluate(bad)

    bad = _prediction_frame()
    bad.loc[0, "mean"] = np.nan
    with pytest.raises(PredictionIntegrityError, match="NaN/Inf"):
        _scientific_evaluate(bad)

    bad = _prediction_frame()
    bad.loc[0, quantile_column(0.01)] = -0.1
    with pytest.raises(PredictionIntegrityError, match="negative"):
        _scientific_evaluate(bad)


def test_quantile_implied_mean_uses_endpoint_hold_and_trapezoids() -> None:
    grid = np.asarray(CRPS_QUANTILE_GRID)
    quantiles = np.stack([grid, np.full_like(grid, 3.0)])
    means = quantile_implied_mean(quantiles, grid)
    assert means.tolist() == pytest.approx([0.5, 3.0])

    coverage = coverage_quantiles_from_common_grid(
        quantiles, grid, p_zero=np.array([0.0, 0.0])
    )
    assert coverage["q025"].tolist() == pytest.approx([0.025, 3.0])
    assert coverage["q975"].tolist() == pytest.approx([0.975, 3.0])
    assert coverage["source"] == "monotone_piecewise_common_grid"

    plateau_quantiles = np.stack([np.where(grid <= 0.01, 0.0, 4.0)])
    plateau = coverage_quantiles_from_common_grid(
        plateau_quantiles, grid, p_zero=np.array([0.03])
    )
    assert plateau["q025"].tolist() == [0.0]


def test_scaled_width_and_normalized_mean_errors_follow_series_rms() -> None:
    frame = _prediction_frame().iloc[[0]].copy()
    frame["y"] = 1.0
    frame["scale"] = 2.0
    frame["mean"] = 3.0
    frame["p_zero"] = 0.0
    for probability in EVALUATION_QUANTILE_GRID:
        frame[quantile_column(probability)] = 4.0 * probability
    row = _scientific_evaluate(frame).levels["dataset"].iloc[0]
    assert row["NRMSE"] == pytest.approx(1.0)
    assert row["NMAE"] == pytest.approx(1.0)
    assert row["interval_width_50"] == pytest.approx(1.0)  # raw width 2 / scale 2


def test_simplex_cdf_pool_and_left_inverse_contracts() -> None:
    states = simplex_grid()
    assert len(states) == 66
    assert np.allclose(states.sum(axis=1), 1.0)

    cdfs = np.array(
        [
            [[0.2, 0.8, 1.0]],
            [[0.4, 0.9, 1.0]],
            [[0.0, 0.5, 1.0]],
        ]
    )
    pooled = linear_cdf_pool(cdfs, np.array([0.2, 0.3, 0.5]))
    assert pooled[0].tolist() == pytest.approx([0.16, 0.68, 1.0])
    values = np.array([0.0, 1.0, 2.0])
    assert invert_cdf(
        values,
        pooled,
        [0.10, 0.50, 0.90],
        exact_discrete_support=True,
    ).tolist() == [[0.0, 1.0, 2.0]]

    with pytest.raises(PredictionIntegrityError, match="simplex"):
        linear_cdf_pool(cdfs, np.array([0.2, 0.2, 0.2]))


def test_callable_pooled_cdf_inverse_expands_bracket_and_reports_error() -> None:
    def exponential(values: np.ndarray) -> np.ndarray:
        return 1.0 - np.exp(-values)

    result = invert_pooled_cdf(
        cdf_functions=[exponential, exponential, exponential],
        weights=np.array([0.2, 0.3, 0.5]),
        probabilities=[0.5, 0.9],
        case_count=2,
        initial_upper=np.array([0.05, 0.1]),
    )
    quantiles = np.asarray(result["quantiles"])
    assert quantiles[:, 0] == pytest.approx(np.log(2.0), abs=2e-6)
    assert quantiles[:, 1] == pytest.approx(np.log(10.0), abs=3e-6)
    assert result["max_x_error_bound"] <= 3e-6
    assert result["bracket_expansions"] > 0
    assert result["max_iterations"] == 128
    assert result["initial_upper_used"] == [1.0, 1.0]
    assert result["initial_upper_source"] == "max(1,caller_max_teacher_mean_or_initial_q99)"
    with pytest.raises(ValueError, match="frozen"):
        invert_pooled_cdf(
            cdf_functions=[exponential, exponential, exponential],
            weights=np.array([0.2, 0.3, 0.5]),
            probabilities=[0.5],
            case_count=1,
            initial_upper=1.0,
            x_tolerance=1e-7,
        )


def test_tweedie_blocked_two_head_pool_is_explicit_diagnostic_and_renormalized() -> None:
    exponential = lambda values: 1.0 - np.exp(-np.asarray(values, dtype=np.float64))
    result = diagnostic_two_head_cdf_pool(
        cdf_functions={"NB": exponential, "HSNB": exponential},
        source_weights={"NB": 0.2, "HSNB": 0.3, "TWEEDIE_FULL": 0.5},
        probabilities=(0.5,),
        case_count=1,
        initial_upper=0.1,
    )
    assert result["weights"] == pytest.approx({"NB": 0.4, "HSNB": 0.6})
    assert np.asarray(result["quantiles"])[0, 0] == pytest.approx(np.log(2.0), abs=2e-6)
    assert result["confirmatory_eligible"] is False
    assert result["scientific_role"] == "DIAGNOSTIC_CONTINUATION_AFTER_TWEEDIE_BRANCH_BLOCKED_HARD"
    assert result["pool"] == "P2_DIAGNOSTIC_NB_HSNB"


def test_distribution_cdf_adapter_preserves_native_case_shape_then_flattens() -> None:
    target = _pool_target_frame(np.zeros(2), np.ones(2))
    components, distributions = _native_distribution_grid(2)
    artifact = _validation_artifact(components, target)
    for head, distribution in zip(HEAD_ORDER, distributions[0], strict=True):
        function, flat_mean, case_shape = cdf_callable_for_distribution(
            distribution,
            head=head,
            model_seed=2026090511,
            validation_artifact=artifact,
        )
        assert case_shape == (2, 1)
        assert flat_mean.shape == (2,)
        full = function(np.array([0.0, 1.0]))
        partial = function(np.array([1.0]), case_indices=np.array([1]))
        assert partial.tolist() == pytest.approx(full[[1]].tolist())


def test_diagnostic_empirical_cdf_and_quantile_require_and_return_lineage() -> None:
    samples = np.array([[0.0, 0.0, 2.0, 4.0], [1.0, 3.0, 3.0, 9.0]])
    support = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 9.0])
    role = "DIAGNOSTIC_CONTINUATION_AFTER_SAMPLE_STABILITY_CHECK"
    cdf = diagnostic_empirical_cdf(samples, support, scientific_role=role)
    assert cdf["values"][0].tolist() == pytest.approx([0.5, 0.5, 0.75, 0.75, 1.0, 1.0])
    quantiles = diagnostic_empirical_quantiles(
        samples, [0.5, 0.75, 0.99], scientific_role=role
    )
    assert quantiles["values"].tolist() == [[0.0, 2.0, 4.0], [3.0, 3.0, 9.0]]
    for result in (cdf, quantiles):
        assert result["confirmatory_eligible"] is False
        assert result["scientific_role"] == role
    with pytest.raises(ValueError, match="diagnostic scientific_role"):
        diagnostic_empirical_cdf(samples, support, scientific_role="CONFIRMATORY")
    assert not hasattr(evaluation_module, "empirical_cdf")
    assert not hasattr(evaluation_module, "empirical_quantiles")


def test_p2_uses_bound_native_cdfs_and_frozen_grid_tiebreak(monkeypatch) -> None:
    def fast_cdf(self, values):
        tensor = torch.as_tensor(values, dtype=self.mu.dtype, device=self.mu.device)
        return 1.0 - torch.exp(-tensor)

    for distribution_type in (
        NegativeBinomialDistribution,
        ShiftedHurdleNegativeBinomialDistribution,
        TweedieDistribution,
    ):
        monkeypatch.setattr(distribution_type, "cdf", fast_cdf)
        monkeypatch.setattr(
            distribution_type, "p_zero", lambda self: torch.zeros_like(self.mu)
        )
        monkeypatch.setattr(distribution_type, "p_zero", lambda self: torch.zeros_like(self.mu))
    target_frame = _pool_target_frame(np.array([0.0, 2.0]), np.ones(2))
    components, distributions = _native_distribution_grid(2)
    artifact = _validation_artifact(components, target_frame)
    bound, initial_upper = _bound_distribution_grid(distributions, artifact)
    selected = select_global_cdf_pool(
        validation_y=np.array([0.0, 2.0]),
        validation_scale=np.ones(2),
        validation_case_keys=_sealed_case_keys(artifact),
        validation_artifact=artifact,
        validation_cdf_functions=bound,
        initial_upper=initial_upper,
    )
    assert selected["weights"] == [0.0, 0.0, 1.0]
    assert selected["candidate_count"] == 66
    assert "test_y" not in selected
    assert selected["validation_artifact_sha256"] == artifact.artifact_sha256

    with pytest.raises(PredictionIntegrityError, match="forbids finite support"):
        select_global_cdf_pool(
            validation_cdfs=np.ones((3, 2, 3)),
            support=np.arange(3.0),
            validation_y=np.array([0.0, 2.0]),
            validation_scale=np.ones(2),
            validation_case_keys=_sealed_case_keys(artifact),
            validation_artifact=artifact,
            discrete_support_exact=True,
        )

    reversed_frame = target_frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(PredictionIntegrityError, match="case keys/order"):
        select_global_cdf_pool(
            validation_y=np.array([2.0, 0.0]),
            validation_scale=np.ones(2),
            validation_case_keys=_case_keys(reversed_frame),
            validation_artifact=artifact,
            validation_cdf_functions=bound,
            initial_upper=initial_upper,
        )

    with pytest.raises(PredictionIntegrityError, match="sealed selection artifact"):
        select_global_cdf_pool(
            validation_y=np.array([0.0, 2.0]),
            validation_scale=np.ones(2),
            validation_case_keys=_sealed_case_keys(artifact),
            validation_artifact=artifact,
            validation_group_ids=[np.nan, "x"],
            validation_cdf_functions=bound,
            initial_upper=initial_upper,
        )


def test_p2_callable_selection_uses_frozen_case_and_quantile_chunks(monkeypatch) -> None:
    case_count = 449
    y = np.zeros(case_count, dtype=np.float64)
    scale = np.ones(case_count, dtype=np.float64)
    target_frame = _pool_target_frame(y, scale)
    components, distributions = _native_distribution_grid(case_count)
    artifact = _validation_artifact(components, target_frame)
    maximum = {head: 0 for head in HEAD_ORDER}
    calls = {head: 0 for head in HEAD_ORDER}

    def make_spy(head):
        def cdf(self, values):
            tensor = torch.as_tensor(values)
            case_size = int(tensor.shape[-2])
            assert case_size <= 448
            maximum[head] = max(maximum[head], case_size)
            calls[head] += 1
            tensor = tensor.to(dtype=self.mu.dtype, device=self.mu.device)
            return 1.0 - torch.exp(-tensor)
        return cdf

    for head, distribution_type in zip(
        HEAD_ORDER,
        (NegativeBinomialDistribution, ShiftedHurdleNegativeBinomialDistribution, TweedieDistribution),
        strict=True,
    ):
        monkeypatch.setattr(distribution_type, "cdf", make_spy(head))
    bound, initial_upper = _bound_distribution_grid(distributions, artifact)
    selected = select_global_cdf_pool(
        validation_y=y,
        validation_scale=scale,
        validation_case_keys=_sealed_case_keys(artifact),
        validation_artifact=artifact,
        validation_cdf_functions=bound,
        initial_upper=initial_upper,
    )
    assert selected["pooled_flat_case_chunk"] == 448
    assert selected["pooled_q_chunk"] == 1
    assert all(value == 448 for value in maximum.values())
    assert all(value < 2_000 for value in calls.values())


def test_validation_selected_p2_applies_to_distinct_sealed_outer_without_cross_split_hash_reuse(monkeypatch) -> None:
    def fast_cdf(self, values):
        tensor = torch.as_tensor(values, dtype=self.mu.dtype, device=self.mu.device)
        return 1.0 - torch.exp(-tensor)

    for distribution_type in (
        NegativeBinomialDistribution,
        ShiftedHurdleNegativeBinomialDistribution,
        TweedieDistribution,
    ):
        monkeypatch.setattr(distribution_type, "cdf", fast_cdf)
        monkeypatch.setattr(
            distribution_type, "p_zero", lambda self: torch.zeros_like(self.mu)
        )
    target = _pool_target_frame(np.array([0.0, 2.0]), np.ones(2))
    validation_components, validation_distributions = _native_distribution_grid(2)
    validation = _validation_artifact(validation_components, target)
    validation_bound, validation_upper = _bound_distribution_grid(
        validation_distributions, validation
    )
    selection = select_global_cdf_pool(
        validation_y=np.array([0.0, 2.0]),
        validation_scale=np.ones(2),
        validation_case_keys=_sealed_case_keys(validation),
        validation_artifact=validation,
        validation_cdf_functions=validation_bound,
        initial_upper=validation_upper,
    )

    outer_components, outer_distributions = _native_distribution_grid(12)
    outer = _validation_artifact(
        outer_components, target, split_name="outer_evaluation"
    )
    outer_bound, outer_upper = _bound_distribution_grid(outer_distributions, outer)
    applied = apply_global_cdf_pool(
        selection=selection,
        validation_artifact=validation,
        prediction_artifact=outer,
        cdf_functions=outer_bound,
        probabilities=CRPS_QUANTILE_GRID,
        case_count=12,
        initial_upper=outer_upper,
        branch_eligibility=_confirmatory_lineage(),
    )
    assert np.asarray(applied["quantiles"]).shape == (1, 12, 21)
    assert len(applied["scientific_forecast_artifacts"]) == 1
    assert applied["prediction_artifact_sha256"] == outer.artifact_sha256
    swapped = [[outer_bound[0][1], outer_bound[0][0], outer_bound[0][2]]]
    with pytest.raises(PredictionIntegrityError, match="head"):
        apply_global_cdf_pool(
            selection=selection,
            validation_artifact=validation,
            prediction_artifact=outer,
            cdf_functions=swapped,
            probabilities=CRPS_QUANTILE_GRID,
            case_count=12,
            initial_upper=outer_upper,
            branch_eligibility=_confirmatory_lineage(),
        )
    for invalid_q in ((0.9, 0.1), (0.5, 0.5), (0.5, np.nan)):
        with pytest.raises(PredictionIntegrityError, match="strictly increasing"):
            apply_global_cdf_pool(
                selection=selection,
                validation_artifact=validation,
                prediction_artifact=outer,
                cdf_functions=outer_bound,
                probabilities=invalid_q,
                case_count=12,
                initial_upper=outer_upper,
                branch_eligibility=_confirmatory_lineage(),
            )

    wrong_seed_components, _ = _native_distribution_grid(
        12, model_seeds=(2026090512,)
    )
    wrong_seed_outer = _validation_artifact(
        wrong_seed_components,
        target,
        split_name="outer_evaluation",
        model_seeds=(2026090512,),
    )
    with pytest.raises(PredictionIntegrityError, match="seed manifest"):
        apply_global_cdf_pool(
            selection=selection,
            validation_artifact=validation,
            prediction_artifact=wrong_seed_outer,
            cdf_functions=outer_bound,
            probabilities=CRPS_QUANTILE_GRID,
            case_count=12,
            initial_upper=outer_upper,
            branch_eligibility=_confirmatory_lineage(),
        )


def test_component_hash_streams_canonical_numeric_bytes_across_numpy_and_torch() -> None:
    values = np.arange(200_000, dtype=np.float64).reshape(1_000, 200)
    assert _component_sha256(values) == _component_sha256(torch.from_numpy(values))
    assert _component_sha256(values) != _component_sha256(values[:, ::-1])


def test_p0_and_p3_pzero_use_validation_only_tiebreaks() -> None:
    quantiles = np.ones((1, 3, 2, len(CRPS_QUANTILE_GRID)))
    pzero_values = np.full((1, 3, 2), 0.2)
    p0_artifact = _validation_artifact(
        {
            head: {
                "quantiles": quantiles[:, index],
                "p_zero": pzero_values[:, index],
            }
            for index, head in enumerate(HEAD_ORDER)
        }
    )
    p0 = select_best_single_teacher(
        validation_teacher_quantiles=quantiles,
        validation_teacher_p_zero=pzero_values,
        validation_y=np.array([0.0, 2.0]),
        validation_scale=np.full(2, 2.0),
        validation_case_keys=_sealed_case_keys(p0_artifact),
        validation_artifact=p0_artifact,
    )
    assert p0["teacher"] == "NB"

    pzero_values = np.array([[[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]])
    target_frame = _pool_target_frame(np.array([0.0, 1.0]), np.ones(2))
    pzero_artifact = _validation_artifact(
        {
            head: {"p_zero": pzero_values[:, index]}
            for index, head in enumerate(HEAD_ORDER)
        },
        target_frame,
    )
    pzero = select_pzero_pool(
        validation_teacher_p_zero=pzero_values,
        validation_y=np.array([0.0, 1.0]),
        validation_case_keys=_sealed_case_keys(pzero_artifact),
        validation_artifact=pzero_artifact,
    )
    assert pzero["weights"] == [1.0, 0.0, 0.0]
    assert pzero["selection_source"] == "validation_only"


def test_p0_and_p3_average_seed_scores_without_cross_seed_forecast_mixing() -> None:
    q_count = len(CRPS_QUANTILE_GRID)
    seeds = (2026090511, 2026090512)
    target = _pool_target_frame(np.array([10.0, 10.0]), np.ones(2))
    quantiles = np.empty((2, 3, 2, q_count), dtype=np.float64)
    quantiles[0, 0] = 0.1
    quantiles[1, 0] = 19.9
    quantiles[:, 1] = 9.0
    quantiles[:, 2] = 30.0
    pzero = np.zeros((2, 3, 2), dtype=np.float64)
    components = {
        head: {
            "quantiles": quantiles[:, index],
            "p_zero": pzero[:, index],
        }
        for index, head in enumerate(HEAD_ORDER)
    }
    artifact = _validation_artifact(
        components, target, model_seeds=seeds
    )
    p0 = select_best_single_teacher(
        validation_teacher_quantiles=quantiles,
        validation_teacher_p_zero=pzero,
        validation_y=np.array([10.0, 10.0]),
        validation_scale=np.ones(2),
        validation_case_keys=_sealed_case_keys(artifact),
        validation_artifact=artifact,
    )
    # Averaging NB forecasts across seeds would predict 10 exactly and win;
    # averaging scalar seed losses correctly selects HSNB instead.
    assert p0["teacher"] == "HSNB"
    p3 = select_quantile_specific_pool(
        teacher_quantiles=quantiles,
        validation_teacher_p_zero=pzero,
        validation_y=np.array([10.0, 10.0]),
        validation_scale=np.ones(2),
        validation_case_keys=_sealed_case_keys(artifact),
        validation_artifact=artifact,
    )
    assert p3["model_seed_axis"] == list(seeds)
    assert p3["cross_seed_quantile_averaging"] is False
    assert not all(weights == [1.0, 0.0, 0.0] for weights in p3["weights_by_quantile"])


def test_p3_dynamic_path_reports_crossing_and_post_projection(monkeypatch) -> None:
    q = np.asarray(CRPS_QUANTILE_GRID)
    teacher_quantiles = np.array(
        [
            np.stack([10.0 * q, 10.0 * q]),
            np.stack([np.full_like(q, 8.0), np.full_like(q, 8.0)]),
            np.stack([1.0 + 8.0 * q, 1.0 + 8.0 * q]),
        ]
    )[None, ...]
    teacher_pzero = np.array([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]])
    target_frame = _pool_target_frame(np.array([10.0, 10.0]), np.ones(2))
    components, distributions = _native_distribution_grid(2)
    for index, head in enumerate(HEAD_ORDER):
        components[head].update(
            quantiles=teacher_quantiles[:, index],
            p_zero=teacher_pzero[:, index],
        )
    artifact = _validation_artifact(
        components,
        target_frame,
    )
    result = select_quantile_specific_pool(
        teacher_quantiles=teacher_quantiles,
        validation_y=np.array([10.0, 10.0]),
        validation_scale=np.ones(2),
        validation_teacher_p_zero=teacher_pzero,
        validation_case_keys=_sealed_case_keys(artifact),
        validation_artifact=artifact,
    )
    assert result["penalty"] in [0.0, 0.01, 0.1, 1.0]
    assert result["pre_crossing_rate"] >= result["post_crossing_rate"]
    assert result["post_crossing_rate"] == 0.0
    assert result["p_zero_pool"]["selection_source"] == "validation_only"
    assert result["distribution_coherence"] == "quantile_plus_separately_pooled_p0_not_single_CDF"
    assert result["zero_adjustment_count"] > 0
    reapplied = apply_quantile_specific_pool(
        teacher_quantiles=teacher_quantiles,
        teacher_p_zero=teacher_pzero,
        selection=result,
        validation_artifact=artifact,
        prediction_artifact=artifact,
        branch_eligibility=_confirmatory_lineage(),
    )
    reapplied_q = np.asarray(reapplied["quantiles"])
    assert np.all(reapplied_q[:, 1:] >= reapplied_q[:, :-1])
    pooled_p0 = np.asarray(reapplied["p_zero"])
    for row, p0 in zip(reapplied_q.reshape(-1, reapplied_q.shape[-1]), pooled_p0.reshape(-1)):
        assert np.all(row[np.asarray(CRPS_QUANTILE_GRID) <= p0] == 0.0)
    assert reapplied["zero_adjustment_count"] > 0

    tampered = deepcopy(result)
    tampered["weights_by_quantile"][0] = [1.0, 0.0, 0.0]
    with pytest.raises(PredictionIntegrityError, match="selection binding"):
        apply_quantile_specific_pool(
            teacher_quantiles=teacher_quantiles,
            teacher_p_zero=teacher_pzero,
            selection=tampered,
            validation_artifact=artifact,
            prediction_artifact=artifact,
            branch_eligibility=_confirmatory_lineage(),
        )

    def point_ten(self, values):
        tensor = torch.as_tensor(values, dtype=self.mu.dtype, device=self.mu.device)
        return (tensor >= 10.0).to(self.mu.dtype)

    for distribution_type in (
        NegativeBinomialDistribution,
        ShiftedHurdleNegativeBinomialDistribution,
        TweedieDistribution,
    ):
        monkeypatch.setattr(distribution_type, "cdf", point_ten)
        monkeypatch.setattr(distribution_type, "p_zero", lambda self: torch.zeros_like(self.mu))
    bound, initial_upper = _bound_distribution_grid(distributions, artifact)
    p2 = select_global_cdf_pool(
        validation_y=np.array([10.0, 10.0]),
        validation_scale=np.ones(2),
        validation_case_keys=_sealed_case_keys(artifact),
        validation_artifact=artifact,
        validation_cdf_functions=bound,
        initial_upper=np.full_like(initial_upper, 10.0),
    )
    primary = select_primary_pool(
        p2_selection=p2,
        p3_selection=result,
        validation_artifact=artifact,
    )
    assert primary["primary_pool"] in {"P2", "P3"}
    assert primary["validation_artifact_sha256"] == artifact.artifact_sha256

    applied_p2 = apply_global_cdf_pool(
        selection=p2,
        validation_artifact=artifact,
        prediction_artifact=artifact,
        cdf_functions=bound,
        probabilities=CRPS_QUANTILE_GRID,
        case_count=2,
        initial_upper=np.full_like(initial_upper, 10.0),
        branch_eligibility=_confirmatory_lineage(),
    )
    assert np.asarray(applied_p2["quantiles"]) == pytest.approx(
        np.full((1, 2, len(CRPS_QUANTILE_GRID)), 10.0), abs=2e-5
    )

    with pytest.raises(PredictionIntegrityError, match="selection binding"):
        select_primary_pool(
            p2_selection={
                **result,
                "pool": "P2",
                "validation_sCRPS": result["validation_sCRPS"],
                "validation_tail_sQL": result["validation_tail_sQL"],
            },
            p3_selection=result,
            validation_artifact=artifact,
        )


def test_p3_penalized_objective_means_levels_and_adjacent_transitions() -> None:
    states = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    losses = np.array([[0.0, 10.0], [0.0, 10.0], [2.5, 0.0]])
    path, objective = _path_for_penalty(losses, states, penalty=1.0)
    # Constant state zero: mean_q loss=2.5/3 and mean transition penalty=0.
    # The unnormalised implementation instead chose the switching path at 2.0.
    assert path == (0, 0, 0)
    assert objective == pytest.approx(2.5 / 3.0)

    # Only mathematically exact ties use the lexicographic state/path rule.
    near_tie_path, _ = _path_for_penalty(
        np.array([[1.0 + 5e-13, 1.0]]), states, penalty=0.0
    )
    assert near_tie_path == (1,)


def test_p3_bidirectional_zero_mass_coherence_blocks_infeasible_distribution() -> None:
    q = np.asarray(CRPS_QUANTILE_GRID)
    quantiles = np.zeros((1, 3, 2, q.size), dtype=np.float64)
    pzero = np.full((1, 3, 2), 0.1, dtype=np.float64)
    target = _pool_target_frame(np.array([0.0, 0.0]), np.ones(2))
    artifact = _validation_artifact(
        {
            head: {"quantiles": quantiles[:, index], "p_zero": pzero[:, index]}
            for index, head in enumerate(HEAD_ORDER)
        },
        target,
    )
    with pytest.raises(
        PredictionIntegrityError,
        match="P3_INCOHERENT_DISTRIBUTION_BLOCKED_HARD",
    ):
        select_quantile_specific_pool(
            teacher_quantiles=quantiles,
            validation_y=np.zeros(2),
            validation_scale=np.ones(2),
            validation_case_keys=_sealed_case_keys(artifact),
            validation_artifact=artifact,
            validation_teacher_p_zero=pzero,
        )


def test_cluster_resampling_keeps_all_origins_and_methods_paired() -> None:
    rows = []
    for series in ["a", "b"]:
        for origin in [1, 2]:
            for method, value in [("base", 2.0), ("new", 1.0)]:
                rows.append({"dataset_id": "d", "series_id": series, "origin": origin, "method": method, "loss": value})
    frame = pd.DataFrame(rows)
    sampled = resample_intact_clusters(frame, {"d": ["b", "b"]})
    assert len(sampled) == 8
    assert sampled["series_id"].nunique() == 2  # bootstrap instance ids, not source ids
    assert set(sampled["source_cluster_id"]) == {"b"}
    for _, block in sampled.groupby("series_id"):
        assert set(block["origin"]) == {1, 2}
        assert set(block["method"]) == {"base", "new"}


def test_paired_cluster_bootstrap_is_deterministic_and_dataset_macro_weighted() -> None:
    rows = []
    # Equal dataset weighting: d1 has one cluster, d2 has three.
    for dataset, clusters, base, new in [("d1", ["a"], 10.0, 5.0), ("d2", ["b", "c", "d"], 10.0, 9.0)]:
        for series in clusters:
            for method, loss in [("base", base), ("new", new)]:
                rows.append({"dataset_id": dataset, "series_id": series, "origin": 1, "method": method, "loss": loss})
    frame = pd.DataFrame(rows)
    first = paired_cluster_bootstrap(frame, baseline="base", candidate="new", tier="MINIMAL-COMPLETE")
    second = paired_cluster_bootstrap(frame, baseline="base", candidate="new", tier="MINIMAL-COMPLETE")
    assert first == second
    assert first["point"] == pytest.approx((0.5 + 0.1) / 2.0)
    assert first["cluster_counts"] == {"d1": 1, "d2": 3}
    assert first["seed"] == 2026090531
    assert bootstrap_draws_for_tier("FULL") == 2000
    assert bootstrap_draws_for_tier("COMPACT") == 1000
    assert bootstrap_draws_for_tier("MINIMAL-COMPLETE") == 500


def test_pairwise_loss_correlation_bootstrap_returns_all_three_pairs() -> None:
    rows = []
    for index, series in enumerate(["a", "b", "c", "d"], start=1):
        for method, loss in [("NB", float(index)), ("HSNB", float(2 * index)), ("TWEEDIE_FULL", float(6 - index))]:
            rows.append({"dataset_id": "d", "series_id": series, "origin": 1, "method": method, "loss": loss})
    result = pairwise_loss_correlation_bootstrap(pd.DataFrame(rows), tier="MINIMAL-COMPLETE")
    assert len(result["pairs"]) == 3
    assert tuple(result["pairs"]) == (
        "NB|HSNB",
        "NB|TWEEDIE_FULL",
        "HSNB|TWEEDIE_FULL",
    )
    assert result["pairs"]["NB|HSNB"]["point"] == pytest.approx(1.0)
    assert result["pairs"]["NB|HSNB"]["upper"] <= 1.0 + 1e-12

    constant = pd.DataFrame(
        [
            {"dataset_id": "d", "series_id": series, "origin": 1, "method": method, "loss": loss}
            for series in ["a", "b"]
            for method, loss in [("NB", 1.0), ("HSNB", 2.0), ("TWEEDIE_FULL", 3.0)]
        ]
    )
    degenerate = pairwise_loss_correlation_bootstrap(constant, tier="MINIMAL-COMPLETE")
    assert all(row["point"] == 1.0 for row in degenerate["pairs"].values())
    assert all(row["degenerate_resample_present"] for row in degenerate["pairs"].values())


def test_pairwise_correlation_averages_seed_replicates_before_r2() -> None:
    rows = []
    amplitude = 4.25
    raw_nb: list[float] = []
    raw_hsnb: list[float] = []
    for step, center in enumerate((10.0, 11.0, 12.0)):
        for model_seed, sign in ((0, 1.0), (1, -1.0)):
            values = {
                "NB": center + sign * amplitude,
                "HSNB": center - sign * amplitude,
                "TWEEDIE_FULL": 2.0 * center + sign * amplitude,
            }
            raw_nb.append(values["NB"])
            raw_hsnb.append(values["HSNB"])
            for method, loss in values.items():
                rows.append(
                    {
                        "dataset_id": "d",
                        "series_id": "s",
                        "origin": 1,
                        "step": step,
                        "model_seed": model_seed,
                        "method": method,
                        "loss": loss,
                    }
                )
    assert np.corrcoef(raw_nb, raw_hsnb)[0, 1] < -0.90
    result = pairwise_loss_correlation_bootstrap(
        pd.DataFrame(rows), tier="MINIMAL-COMPLETE"
    )
    assert result["pairs"]["NB|HSNB"]["point"] == pytest.approx(1.0)
    assert result["replicate_aggregation"] == "mean_before_pairwise_correlation"
    assert result["replicate_columns"] == ["model_seed"]


def test_practical_winners_and_oracle_ladder_are_hand_calculable() -> None:
    rows = []
    losses = {
        (4, -0.8, "s1", 1): {"NB": 1.00, "HSNB": 1.005, "TWEEDIE_FULL": 1.20},
        (4, 0.8, "s2", 1): {"NB": 1.10, "HSNB": 1.00, "TWEEDIE_FULL": 1.30},
    }
    for (d, rho_i, series, origin), by_head in losses.items():
        for head, loss in by_head.items():
            rows.append({"dataset_id": "syn", "d": d, "rho_I": rho_i, "rho_M": 0.0, "series_id": series, "origin": origin, "head": head, "sCRPS": loss})
    frame = pd.DataFrame(rows)
    winners = summarize_practical_winners(frame)
    assert winners["exact_best_counts"] == {"NB": 1, "HSNB": 1, "TWEEDIE_FULL": 0}
    assert winners["practical_winner_shares"]["NB"] == pytest.approx(0.5)
    assert winners["practical_winner_shares"]["HSNB"] == pytest.approx(1.0)
    ladder = summarize_oracle_ladder(
        frame,
        branch_eligibility=_synthetic_lineage(),
        tweedie_valid=GateStatus.passed("TWEEDIE_VALID"),
    )
    assert ladder["best_global_head"] == "HSNB"
    assert ladder["series_origin_oracle_loss"] == pytest.approx(1.0)
    assert ladder["series_origin_oracle_gain"] > 0.0


def test_s2_oracle_ladder_uses_equal_cell_macro_not_row_weighting() -> None:
    rows = []
    specs = [(-0.8, "only", {"NB": 1.0, "HSNB": 2.0, "TWEEDIE_FULL": 3.0})]
    specs.extend(
        (0.8, f"many{index}", {"NB": 10.0, "HSNB": 9.0, "TWEEDIE_FULL": 11.0})
        for index in range(3)
    )
    for rho_i, series, losses in specs:
        for head, loss in losses.items():
            rows.append({"dataset_id": "syn", "d": 4, "rho_I": rho_i, "rho_M": 0.0, "series_id": series, "origin": 1, "head": head, "sCRPS": loss})
    ladder = summarize_oracle_ladder(
        pd.DataFrame(rows),
        branch_eligibility=_synthetic_lineage(),
        tweedie_valid=GateStatus.passed("TWEEDIE_VALID"),
    )
    assert ladder["best_global_head"] == "NB"  # equal-cell tie -> frozen NB order
    assert ladder["aggregation"] == "equal_cell_macro"


def test_s2_confirmatory_oracle_requires_three_heads_and_two_head_result_is_diagnostic() -> None:
    rows = []
    cell_index = 0
    for d in (4, 8):
        for rho_i in (-0.8, 0.0, 0.8):
            for rho_m in (-0.8, 0.0, 0.8):
                preferred = "NB" if cell_index % 2 == 0 else "HSNB"
                for head in ("NB", "HSNB"):
                    rows.append(
                        {
                            "dataset_id": "syn",
                            "d": d,
                            "rho_I": rho_i,
                            "rho_M": rho_m,
                            "series_id": f"s{cell_index}",
                            "origin": 1,
                            "head": head,
                            "sCRPS": 1.0 if head == preferred else 2.0,
                        }
                    )
                cell_index += 1
    frame = pd.DataFrame(rows)
    with pytest.raises(PredictionIntegrityError, match="exactly.*NB.*HSNB.*TWEEDIE_FULL"):
        summarize_oracle_ladder(
            frame,
            branch_eligibility=_synthetic_lineage(),
            tweedie_valid=GateStatus.passed("TWEEDIE_VALID"),
        )
    diagnostic = summarize_two_head_diagnostic_oracle_ladder(
        frame,
        tweedie_valid=GateStatus.hard_failure("TWEEDIE_VALID"),
    )
    assert diagnostic["cell_oracle_gain"] == pytest.approx(1.0 / 3.0)
    assert diagnostic["confirmatory_eligible"] is False
    assert diagnostic["scientific_role"] == "DIAGNOSTIC_CONTINUATION_AFTER_TWEEDIE_BRANCH_BLOCKED_HARD"
    assert diagnostic["head_order"] == ["NB", "HSNB"]


def test_practical_winner_shares_are_equal_dataset_macro() -> None:
    rows = []
    for dataset, series_ids, winner in [("small", ["s"], "NB"), ("large", list("abcdefghi"), "HSNB")]:
        for series in series_ids:
            for head in ["NB", "HSNB", "TWEEDIE_FULL"]:
                rows.append({"dataset_id": dataset, "series_id": series, "origin": 1, "head": head, "sCRPS": 1.0 if head == winner else 2.0})
    summary = summarize_practical_winners(pd.DataFrame(rows))
    assert summary["practical_winner_shares"]["NB"] == pytest.approx(0.5)
    assert summary["practical_winner_shares"]["HSNB"] == pytest.approx(0.5)
    assert summary["per_dataset"]["small"]["practical_winner_shares"]["NB"] == 1.0


def test_winner_and_oracle_average_exact_model_seed_replicates_first() -> None:
    rows = []
    # NB wins one seed spectacularly and loses the other; HSNB is the best mean.
    losses = {
        11: {"NB": 0.1, "HSNB": 1.0, "TWEEDIE_FULL": 3.0},
        12: {"NB": 2.1, "HSNB": 1.0, "TWEEDIE_FULL": 3.0},
    }
    for seed, by_head in losses.items():
        for head, loss in by_head.items():
            rows.append(
                {
                    "dataset_id": "syn",
                    "series_id": "s",
                    "origin": 1,
                    "d": 4,
                    "rho_I": 0.0,
                    "rho_M": 0.0,
                    "model_seed": seed,
                    "head": head,
                    "sCRPS": loss,
                }
            )
    frame = pd.DataFrame(rows)
    summary = summarize_practical_winners(frame, expected_model_seeds=(11, 12))
    assert summary["exact_best_counts"] == {"HSNB": 1, "NB": 0, "TWEEDIE_FULL": 0}
    ladder = summarize_oracle_ladder(
        frame,
        expected_model_seeds=(11, 12),
        branch_eligibility=_synthetic_lineage(),
        tweedie_valid=GateStatus.passed("TWEEDIE_VALID"),
    )
    assert ladder["best_global_head"] == "HSNB"
    with pytest.raises(PredictionIntegrityError, match="exact model-seed coverage"):
        summarize_practical_winners(
            frame.loc[~((frame["model_seed"] == 12) & (frame["head"] == "NB"))],
            expected_model_seeds=(11, 12),
        )
    with pytest.raises(PredictionIntegrityError, match="expected model-seed manifest"):
        summarize_practical_winners(frame)


def test_bootstrap_aggregates_each_series_before_equal_resampling() -> None:
    rows = []
    for origin in range(9):
        rows.extend([
            {"dataset_id": "d", "series_id": "many", "origin": origin, "method": "base", "loss": 10.0},
            {"dataset_id": "d", "series_id": "many", "origin": origin, "method": "new", "loss": 9.0},
        ])
    rows.extend([
        {"dataset_id": "d", "series_id": "one", "origin": 0, "method": "base", "loss": 10.0},
        {"dataset_id": "d", "series_id": "one", "origin": 0, "method": "new", "loss": 5.0},
    ])
    result = paired_cluster_bootstrap(pd.DataFrame(rows), baseline="base", candidate="new", tier="MINIMAL-COMPLETE")
    assert result["point"] == pytest.approx(0.30)  # equal series: mean candidate=(9+5)/2


def test_pool_selectors_reject_nonfinite_target_scale_and_support() -> None:
    target = _pool_target_frame(np.array([0.0]), np.array([1.0]))
    components, _ = _native_distribution_grid(1)
    artifact = _validation_artifact(components, target)
    cdfs = np.repeat(np.array([[[0.5, 1.0]]]), 3, axis=0)
    with pytest.raises(PredictionIntegrityError):
        select_global_cdf_pool(
            validation_cdfs=cdfs,
            support=np.array([0.0, np.nan]),
            validation_y=np.array([0.0]),
            validation_scale=np.array([1.0]),
            validation_case_keys=_sealed_case_keys(artifact),
            validation_artifact=artifact,
            discrete_support_exact=True,
        )
    with pytest.raises(PredictionIntegrityError, match="one-dimensional"):
        select_pzero_pool(
            validation_teacher_p_zero=np.zeros((1, 3, 1)),
            validation_y=np.array([[0.0]]),
            validation_case_keys=_sealed_case_keys(artifact),
            validation_artifact=artifact,
        )
    incomplete_cdfs = np.repeat(np.array([[[0.5, 0.995]]]), 3, axis=0)
    with pytest.raises(PredictionIntegrityError, match="forbids finite support"):
        select_global_cdf_pool(
            validation_cdfs=incomplete_cdfs,
            support=np.array([0.0, 1.0]),
            validation_y=np.array([0.0]),
            validation_scale=np.array([1.0]),
            validation_case_keys=_sealed_case_keys(artifact),
            validation_artifact=artifact,
            discrete_support_exact=True,
        )
    with pytest.raises(PredictionIntegrityError):
        select_global_cdf_pool(
            validation_cdfs=cdfs,
            support=np.array([0.0, 1.0]),
            validation_y=np.array([np.inf]),
            validation_scale=np.array([1.0]),
            validation_case_keys=_sealed_case_keys(artifact),
            validation_artifact=artifact,
            discrete_support_exact=True,
        )

    for bad_support in (np.array([0.0, 2.0]), np.array([0.0, 0.5, 1.0])):
        with pytest.raises(PredictionIntegrityError, match="contiguous integer"):
            invert_cdf(
                bad_support,
                np.ones((1, bad_support.size)),
                [0.5],
                exact_discrete_support=True,
            )


def test_factorial_temporal_contrasts_effect_code_endpoint_difference() -> None:
    rows = []
    for data_seed in (2026090501, 2026090502):
        for cluster in ["z1", "z2"]:
            for d in [4, 8]:
                for model_seed in (2026090511, 2026090512):
                    for origin in (436, 464):
                        for rho_i in [-0.8, 0.0, 0.8]:
                            for rho_m in [-0.8, 0.0, 0.8]:
                                # The frozen percentage-point gap has endpoint difference 2.0.
                                gap = -1.0 if d == 4 else 1.0
                                rows.append({"pair": "NB_vs_HSNB", "data_seed": data_seed, "base_innovation_id": cluster, "model_seed": model_seed, "origin": origin, "d": d, "rho_I": rho_i, "rho_M": rho_m, "relative_gap": gap})
    frame = pd.DataFrame(rows)
    result = factorial_temporal_contrasts(
        frame,
        tier="MINIMAL-COMPLETE",
        expected_data_seeds=(2026090501, 2026090502),
        expected_model_seeds=(2026090511, 2026090512),
    )
    d_row = next(row for row in result if row["contrast"] == "d")
    assert d_row["effect"] == pytest.approx(2.0)
    assert d_row["ci_lower"] == pytest.approx(2.0)
    assert d_row["ci_upper"] == pytest.approx(2.0)

    missing_zero_cell = frame.drop(
        frame.index[
            (frame["base_innovation_id"] == "z1")
            & (frame["d"] == 4)
            & (frame["model_seed"] == 2026090511)
            & (frame["origin"] == 436)
            & (frame["rho_I"] == 0.0)
            & (frame["rho_M"] == 0.0)
        ][0]
    )
    with pytest.raises(PredictionIntegrityError, match="exact 9-rho-cell grid"):
        factorial_temporal_contrasts(
            missing_zero_cell,
            tier="MINIMAL-COMPLETE",
            expected_data_seeds=(2026090501, 2026090502),
            expected_model_seeds=(2026090511, 2026090512),
        )
    with pytest.raises(PredictionIntegrityError, match="duplicate"):
        factorial_temporal_contrasts(
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            tier="MINIMAL-COMPLETE",
            expected_data_seeds=(2026090501, 2026090502),
            expected_model_seeds=(2026090511, 2026090512),
        )

    unequal_seed_coverage = frame.loc[
        ~(
            (frame["data_seed"] == 2026090501)
            & (frame["base_innovation_id"] == "z1")
            & (frame["d"] == 4)
            & (frame["model_seed"] == 2026090512)
        )
    ]
    with pytest.raises(PredictionIntegrityError, match="model-seed coverage"):
        factorial_temporal_contrasts(
            unequal_seed_coverage,
            tier="MINIMAL-COMPLETE",
            expected_data_seeds=(2026090501, 2026090502),
            expected_model_seeds=(2026090511, 2026090512),
        )

    quadratic = frame.copy()
    quadratic["relative_gap"] = np.where(
        quadratic["rho_I"] == 0.0, -2.0, 1.0
    )
    quadratic_result = factorial_temporal_contrasts(
        quadratic,
        tier="MINIMAL-COMPLETE",
        expected_data_seeds=(2026090501, 2026090502),
        expected_model_seeds=(2026090511, 2026090512),
    )
    by_name = {row["contrast"]: row for row in quadratic_result}
    assert len(by_name) == 17
    assert by_name["rho_I_Q"]["effect"] == pytest.approx(3.0)
    assert by_name["rho_I_L"]["effect"] == pytest.approx(0.0, abs=1e-12)


def test_s3_pairwise_gap_builder_uses_frozen_percentage_point_formula() -> None:
    frame = pd.DataFrame(
        [
            {"dataset_id": "syn", "series_id": "a", "origin": 1, "head": head, "sCRPS": loss}
            for head, loss in [("NB", 1.1), ("HSNB", 1.0), ("TWEEDIE_FULL", 2.0)]
        ]
    )
    gaps = pairwise_relative_scrps_gaps(frame)
    by_pair = dict(zip(gaps["pair"], gaps["relative_gap"]))
    assert by_pair["NB_vs_HSNB"] == pytest.approx(10.0)
    assert by_pair["NB_vs_TWEEDIE_FULL"] == pytest.approx(-45.0)


def test_s3_pairwise_gap_builder_keeps_paired_cell_identity() -> None:
    rows = []
    for d in (4, 8):
        for head, loss in (("NB", 1.1), ("HSNB", 1.0), ("TWEEDIE_FULL", 2.0)):
            rows.append(
                    {
                        "dataset_id": "syn",
                        "series_id": "shared_base",
                        "origin": 436,
                        "data_seed": 2026090501,
                        "model_seed": 2026090511,
                    "base_innovation_id": "shared_base",
                    "d": d,
                    "rho_I": 0.0,
                    "rho_M": 0.0,
                    "head": head,
                    "sCRPS": loss,
                }
            )
    gaps = pairwise_relative_scrps_gaps(pd.DataFrame(rows))
    assert len(gaps) == 6
    assert set(gaps["d"]) == {4, 8}
