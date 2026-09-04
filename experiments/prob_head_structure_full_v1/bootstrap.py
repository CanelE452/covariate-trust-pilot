"""Paired cluster bootstrap with frozen tier counts and seed."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation import PredictionIntegrityError


BOOTSTRAP_SEED = 2026090531
_TIER_DRAWS = {"FULL": 2000, "COMPACT": 1000, "MINIMAL-COMPLETE": 500}
_HEAD_ORDER = ("NB", "HSNB", "TWEEDIE_FULL")


def bootstrap_draws_for_tier(tier: str) -> int:
    try:
        return _TIER_DRAWS[str(tier).upper()]
    except KeyError as error:
        raise ValueError(f"unknown runtime tier: {tier}") from error


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def resample_intact_clusters(
    frame: pd.DataFrame,
    selections_by_dataset: Mapping[Any, Sequence[Any]],
    *,
    dataset_column: str = "dataset_id",
    cluster_column: str = "series_id",
) -> pd.DataFrame:
    """Materialize a specified bootstrap selection without splitting clusters."""

    if dataset_column not in frame or cluster_column not in frame:
        raise ValueError("dataset and cluster columns are required")
    blocks: list[pd.DataFrame] = []
    for dataset, selections in selections_by_dataset.items():
        dataset_frame = frame.loc[frame[dataset_column] == dataset]
        for draw_index, source_cluster in enumerate(selections):
            block = dataset_frame.loc[dataset_frame[cluster_column] == source_cluster].copy()
            if block.empty:
                raise PredictionIntegrityError(
                    f"selected cluster is absent: dataset={dataset!r}, cluster={source_cluster!r}"
                )
            block["source_cluster_id"] = block[cluster_column]
            block[cluster_column] = f"bootstrap_{draw_index:08d}"
            blocks.append(block)
    if not blocks:
        return frame.iloc[:0].copy()
    return pd.concat(blocks, ignore_index=True)


def _validate_paired_rows(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    dataset_column: str,
    cluster_column: str,
    method_column: str,
    metric_column: str,
    observation_columns: Sequence[str] | None,
) -> pd.DataFrame:
    required = {dataset_column, cluster_column, method_column, metric_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame is missing columns: {sorted(missing)}")
    selected = frame.loc[frame[method_column].isin([baseline, candidate])].copy()
    if set(selected[method_column].unique()) != {baseline, candidate}:
        raise PredictionIntegrityError("both paired methods must be present")
    if not np.all(np.isfinite(selected[metric_column].to_numpy(dtype=np.float64))):
        raise PredictionIntegrityError("bootstrap metric contains NaN/Inf")
    identity = [dataset_column, cluster_column]
    if observation_columns is None:
        candidates = ("dgp_seed", "model_seed", "d", "rho_I", "rho_M", "origin", "step")
        identity.extend(column for column in candidates if column in selected.columns)
    else:
        identity.extend(column for column in observation_columns if column not in identity)
    missing_identity = set(identity).difference(selected.columns)
    if missing_identity:
        raise ValueError(f"bootstrap observation columns are missing: {sorted(missing_identity)}")
    counts = selected.groupby(identity, dropna=False)[method_column].agg(
        lambda values: tuple(sorted(values))
    )
    expected = tuple(sorted((baseline, candidate)))
    if any(value != expected for value in counts):
        raise PredictionIntegrityError("bootstrap methods are not paired at every observation key")
    if selected.duplicated(identity + [method_column]).any():
        raise PredictionIntegrityError("duplicate method row at a paired observation key")
    return selected


def _dataset_relative_improvement(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    dataset_column: str,
    cluster_column: str,
    method_column: str,
    metric_column: str,
) -> tuple[dict[Any, float], float]:
    effects: dict[Any, float] = {}
    cluster_means = (
        frame.groupby(
            [dataset_column, cluster_column, method_column],
            sort=False,
            dropna=False,
        )[metric_column]
        .mean()
        .reset_index()
    )
    for dataset, block in cluster_means.groupby(
        dataset_column, sort=False, dropna=False
    ):
        means = block.groupby(method_column, sort=False)[metric_column].mean()
        baseline_value = float(means[baseline])
        candidate_value = float(means[candidate])
        if baseline_value <= 0.0:
            raise PredictionIntegrityError("relative bootstrap requires positive baseline loss")
        effects[_json_scalar(dataset)] = (baseline_value - candidate_value) / baseline_value
    return effects, float(np.mean(list(effects.values())))


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    tier: str,
    dataset_column: str = "dataset_id",
    cluster_column: str = "series_id",
    method_column: str = "method",
    metric_column: str = "loss",
    observation_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bootstrap a paired relative loss improvement with equal dataset weight."""

    selected = _validate_paired_rows(
        frame,
        baseline=baseline,
        candidate=candidate,
        dataset_column=dataset_column,
        cluster_column=cluster_column,
        method_column=method_column,
        metric_column=metric_column,
        observation_columns=observation_columns,
    )
    draws = bootstrap_draws_for_tier(tier)
    cluster_values: dict[Any, list[Any]] = {}
    cluster_summaries: dict[Any, dict[str, np.ndarray]] = {}
    for dataset in sorted(pd.unique(selected[dataset_column]), key=lambda value: str(value)):
        block = selected.loc[selected[dataset_column] == dataset]
        clusters = sorted(pd.unique(block[cluster_column]), key=lambda value: str(value))
        if not clusters:
            raise PredictionIntegrityError(f"dataset {dataset!r} has no bootstrap clusters")
        cluster_values[dataset] = clusters
        summary = (
            block.groupby([cluster_column, method_column], sort=False)[metric_column]
            .mean()
        )
        cluster_summaries[dataset] = {
            "baseline_mean": np.asarray(
                [summary.loc[(cluster, baseline)] for cluster in clusters],
                dtype=np.float64,
            ),
            "candidate_mean": np.asarray(
                [summary.loc[(cluster, candidate)] for cluster in clusters],
                dtype=np.float64,
            ),
        }

    dataset_effects, point = _dataset_relative_improvement(
        selected,
        baseline=baseline,
        candidate=candidate,
        dataset_column=dataset_column,
        cluster_column=cluster_column,
        method_column=method_column,
        metric_column=metric_column,
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(draws, dtype=np.float64)
    provenance_hasher = hashlib.sha256()
    for draw_index in range(draws):
        draw_effects: list[float] = []
        for dataset, clusters in cluster_values.items():
            indices = generator.integers(0, len(clusters), size=len(clusters))
            provenance_hasher.update(
                json.dumps(
                    [_json_scalar(dataset), draw_index, indices.tolist()],
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            summary = cluster_summaries[dataset]
            baseline_mean = float(np.mean(summary["baseline_mean"][indices]))
            candidate_mean = float(np.mean(summary["candidate_mean"][indices]))
            if baseline_mean <= 0.0:
                raise PredictionIntegrityError("relative bootstrap requires positive baseline loss")
            draw_effects.append((baseline_mean - candidate_mean) / baseline_mean)
        values[draw_index] = float(np.mean(draw_effects))
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "effect": "paired_relative_loss_improvement",
        "baseline": baseline,
        "candidate": candidate,
        "point": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "dataset_effects": dataset_effects,
        "cluster_counts": {
            _json_scalar(dataset): len(clusters) for dataset, clusters in cluster_values.items()
        },
        "draws": draws,
        "seed": BOOTSTRAP_SEED,
        "rng": "numpy.random.Generator(PCG64)",
        "sampling_sha256": provenance_hasher.hexdigest(),
        "paired": True,
        "dataset_macro": "equal_weight",
        "within_dataset_aggregation": "equal_cluster_mean_before_resampling",
    }


def _correlation_from_sufficient(statistics: np.ndarray) -> tuple[float, bool]:
    n, sum_x, sum_y, sum_x2, sum_y2, sum_xy = map(float, statistics)
    if n <= 0.0:
        raise PredictionIntegrityError("correlation sample is empty")
    centered_x2 = max(0.0, sum_x2 - sum_x * sum_x / n)
    centered_y2 = max(0.0, sum_y2 - sum_y * sum_y / n)
    denominator = np.sqrt(centered_x2 * centered_y2)
    if denominator <= np.finfo(np.float64).eps:
        # Undefined correlation provides no evidence that losses are clearly
        # below one, irrespective of whether the two constants differ.
        return 1.0, True
    numerator = sum_xy - sum_x * sum_y / n
    return float(np.clip(numerator / denominator, -1.0, 1.0)), False


def pairwise_loss_correlation_bootstrap(
    frame: pd.DataFrame,
    *,
    tier: str,
    dataset_column: str = "dataset_id",
    cluster_column: str = "series_id",
    method_column: str = "method",
    metric_column: str = "loss",
    observation_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Cluster-bootstrap seed-averaged Pearson loss correlations.

    Model/data-seed replicates are averaged at the registered observation key
    before any correlation is computed.  Seed identities therefore cannot act
    as extra observations or manufacture complementarity.
    """

    required = {dataset_column, cluster_column, method_column, metric_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"correlation frame is missing columns: {sorted(required - set(frame.columns))}")
    if not np.all(np.isfinite(frame[metric_column].to_numpy(dtype=np.float64))):
        raise PredictionIntegrityError("correlation loss contains NaN/Inf")
    methods_present = set(map(str, frame[method_column].unique()))
    if methods_present != set(_HEAD_ORDER):
        raise PredictionIntegrityError(
            f"R2 correlation requires exactly the frozen heads {list(_HEAD_ORDER)}"
        )
    replicate_columns = [
        column for column in ("data_seed", "dgp_seed", "model_seed") if column in frame.columns
    ]
    if observation_columns is None:
        registered_observations = [
            column
            for column in ("d", "rho_I", "rho_M", "origin", "step")
            if column in frame.columns
        ]
    else:
        registered_observations = [str(column) for column in observation_columns]
        if any(column in replicate_columns for column in registered_observations):
            raise ValueError("seed columns cannot remain in the R2 observation key")
        missing_observations = set(registered_observations).difference(frame.columns)
        if missing_observations:
            raise ValueError(
                f"correlation observation columns are missing: {sorted(missing_observations)}"
            )
    key_columns = list(
        dict.fromkeys([dataset_column, cluster_column, *registered_observations])
    )
    replicate_key = [*key_columns, *replicate_columns]
    if frame.duplicated(replicate_key + [method_column]).any():
        raise PredictionIntegrityError("duplicate loss at a seed-replicate observation key")
    paired_methods = frame.groupby(replicate_key, sort=False, dropna=False)[method_column].agg(
        lambda values: set(map(str, values))
    )
    if any(methods != set(_HEAD_ORDER) for methods in paired_methods):
        raise PredictionIntegrityError("teacher losses are not paired at every seed replicate")
    aggregated = (
        frame.groupby([*key_columns, method_column], sort=False, dropna=False)[metric_column]
        .mean()
        .reset_index()
    )
    wide = aggregated.pivot(index=key_columns, columns=method_column, values=metric_column)
    if wide.isna().any().any():
        raise PredictionIntegrityError("teacher losses are not paired for correlation")
    wide.columns = wide.columns.map(str)
    methods = list(_HEAD_ORDER)
    wide = wide.loc[:, methods]
    draws = bootstrap_draws_for_tier(tier)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cluster_values = {}
    for dataset in sorted(pd.unique(aggregated[dataset_column]), key=lambda value: str(value)):
        block = aggregated.loc[aggregated[dataset_column] == dataset]
        cluster_values[dataset] = sorted(
            pd.unique(block[cluster_column]), key=lambda value: str(value)
        )
    draw_indices: dict[Any, np.ndarray] = {}
    provenance_hasher = hashlib.sha256()
    for dataset, clusters in cluster_values.items():
        indices = rng.integers(0, len(clusters), size=(draws, len(clusters)))
        draw_indices[dataset] = indices
        provenance_hasher.update(str(_json_scalar(dataset)).encode("utf-8"))
        provenance_hasher.update(indices.tobytes())
    wide_frame = wide.reset_index()
    completeness = wide_frame.groupby(
        [dataset_column, cluster_column], sort=False, dropna=False
    ).size()
    for dataset in pd.unique(wide_frame[dataset_column]):
        dataset_counts = completeness.loc[dataset]
        if dataset_counts.nunique() != 1:
            raise PredictionIntegrityError(
                f"correlation clusters are incomplete in dataset {dataset!r}"
            )
    output: dict[str, Any] = {}
    for left_name, right_name in combinations(methods, 2):
        cluster_statistics: dict[Any, np.ndarray] = {}
        point_correlations: list[float] = []
        point_degenerate = False
        for dataset, clusters in cluster_values.items():
            dataset_wide = wide_frame.loc[wide_frame[dataset_column] == dataset]
            rows: list[list[float]] = []
            for cluster in clusters:
                block = dataset_wide.loc[dataset_wide[cluster_column] == cluster]
                left = block[left_name].to_numpy(dtype=np.float64)
                right = block[right_name].to_numpy(dtype=np.float64)
                rows.append(
                    [
                        float(len(left)),
                        float(left.sum()),
                        float(right.sum()),
                        float(np.dot(left, left)),
                        float(np.dot(right, right)),
                        float(np.dot(left, right)),
                    ]
                )
            statistics = np.asarray(rows, dtype=np.float64)
            cluster_statistics[dataset] = statistics
            correlation, degenerate = _correlation_from_sufficient(statistics.sum(axis=0))
            point_correlations.append(correlation)
            point_degenerate = point_degenerate or degenerate
        point = float(np.mean(point_correlations))
        samples = np.empty(draws, dtype=np.float64)
        any_degenerate = point_degenerate
        for draw_index in range(draws):
            correlations: list[float] = []
            for dataset in cluster_values:
                statistics = cluster_statistics[dataset][draw_indices[dataset][draw_index]].sum(axis=0)
                correlation, degenerate = _correlation_from_sufficient(statistics)
                correlations.append(correlation)
                any_degenerate = any_degenerate or degenerate
            samples[draw_index] = float(np.mean(correlations))
        lower, upper = np.percentile(samples, [2.5, 97.5])
        output[f"{left_name}|{right_name}"] = {
            "point": point,
            "lower": float(lower),
            "upper": float(upper),
            "degenerate_resample_present": any_degenerate,
        }
    return {
        "pairs": output,
        "draws": draws,
        "seed": BOOTSTRAP_SEED,
        "sampling_sha256": provenance_hasher.hexdigest(),
        "dataset_macro": "equal_weight",
        "replicate_aggregation": "mean_before_pairwise_correlation",
        "replicate_columns": replicate_columns,
        "observation_columns": registered_observations,
        "head_order": list(_HEAD_ORDER),
        "criterion": "maximum_pairwise_upper95_below_0.99",
    }


_FACTOR_TERMS = (
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
)

_FACTOR_RANGES = np.asarray(
    [2, 2, 3, 2, 3, 4, 6, 4, 6, 4, 6, 6, 9, 8, 12, 12, 18],
    dtype=np.float64,
)


def _factor_matrix(frame: pd.DataFrame) -> np.ndarray:
    d_raw = frame["d"].to_numpy(dtype=np.float64)
    rho_i_raw = frame["rho_I"].to_numpy(dtype=np.float64)
    rho_m_raw = frame["rho_M"].to_numpy(dtype=np.float64)
    if not set(np.unique(d_raw)).issubset({4.0, 8.0}):
        raise PredictionIntegrityError("S3 d levels must be 4 or 8")
    if not set(np.round(np.unique(rho_i_raw), 10)).issubset({-0.8, 0.0, 0.8}) or not set(
        np.round(np.unique(rho_m_raw), 10)
    ).issubset({-0.8, 0.0, 0.8}):
        raise PredictionIntegrityError("S3 rho levels must be -0.8, 0, or +0.8")
    d = np.where(d_raw == 4.0, -1.0, 1.0)
    rho_i_l = rho_i_raw / 0.8
    rho_m_l = rho_m_raw / 0.8
    rho_i_q = np.where(rho_i_raw == 0.0, -2.0, 1.0)
    rho_m_q = np.where(rho_m_raw == 0.0, -2.0, 1.0)
    return np.column_stack(
        [
            np.ones(len(frame)),
            d,
            rho_i_l,
            rho_i_q,
            rho_m_l,
            rho_m_q,
            d * rho_i_l,
            d * rho_i_q,
            d * rho_m_l,
            d * rho_m_q,
            rho_i_l * rho_m_l,
            rho_i_l * rho_m_q,
            rho_i_q * rho_m_l,
            rho_i_q * rho_m_q,
            d * rho_i_l * rho_m_l,
            d * rho_i_l * rho_m_q,
            d * rho_i_q * rho_m_l,
            d * rho_i_q * rho_m_q,
        ]
    )


def _factorial_effects(frame: pd.DataFrame, value_column: str) -> np.ndarray:
    design = _factor_matrix(frame)
    response = frame[value_column].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(response)):
        raise PredictionIntegrityError("S3 relative gap contains NaN/Inf")
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank != design.shape[1]:
        raise PredictionIntegrityError("S3 factorial design is rank deficient")
    # Report beta times the complete coding range product.  D/L ranges are 2;
    # the orthogonal three-level quadratic contrast (1,-2,1) has range 3.
    return coefficients[1:] * _FACTOR_RANGES


def factorial_temporal_contrasts(
    frame: pd.DataFrame,
    *,
    tier: str,
    pair_column: str = "pair",
    cluster_column: str = "base_innovation_id",
    value_column: str = "relative_gap",
    data_seed_column: str = "data_seed",
    expected_data_seeds: Sequence[Any],
    model_seed_column: str = "model_seed",
    expected_model_seeds: Sequence[Any],
    origin_column: str = "origin",
) -> list[dict[str, Any]]:
    """Estimate frozen S3 endpoint contrasts and base-innovation bootstrap CIs.

    The response must be the preregistered percentage-point relative loss gap
    ``100 * (sCRPS_head_a - sCRPS_head_b) / sCRPS_head_b`` for ordered pairs
    NB-vs-HSNB, NB-vs-TWEEDIE_FULL, and HSNB-vs-TWEEDIE_FULL.  d is coded
    4 -> -1 and 8 -> +1.  Each rho has an orthogonal linear code
    ``(-1,0,+1)`` and quadratic code ``(+1,-2,+1)``.  Each reported term is
    beta times the product of its code ranges (D/L=2, Q=3).
    """

    required = {
        pair_column,
        cluster_column,
        data_seed_column,
        model_seed_column,
        origin_column,
        value_column,
        "d",
        "rho_I",
        "rho_M",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"S3 frame is missing columns: {sorted(required - set(frame.columns))}")
    seeds = tuple(expected_data_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected S3 data seeds must be unique and non-empty")
    if set(pd.unique(frame[data_seed_column])) != set(seeds):
        raise PredictionIntegrityError("S3 rows do not match the expected data-seed manifest")
    model_seeds = tuple(expected_model_seeds)
    if not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("expected S3 model seeds must be unique and non-empty")
    if set(pd.unique(frame[model_seed_column])) != set(model_seeds):
        raise PredictionIntegrityError("S3 rows do not match the expected model-seed manifest")
    exact_cells = {
        (rho_i, rho_m)
        for rho_i in (-0.8, 0.0, 0.8)
        for rho_m in (-0.8, 0.0, 0.8)
    }
    cell_key = [
        pair_column,
        data_seed_column,
        cluster_column,
        "d",
        model_seed_column,
        origin_column,
        "rho_I",
        "rho_M",
    ]
    if frame.duplicated(cell_key).any():
        raise PredictionIntegrityError(
            "duplicate S3 row for a base-innovation/data-seed/cell identity"
        )
    for (pair, seed, d, cluster, model_seed, origin), block in frame.groupby(
        [pair_column, data_seed_column, "d", cluster_column, model_seed_column, origin_column],
        sort=False,
        dropna=False,
    ):
        observed_cells = {
            (float(row.rho_I), float(row.rho_M))
            for row in block.loc[:, ["rho_I", "rho_M"]].itertuples(index=False)
        }
        if observed_cells != exact_cells or len(block) != len(exact_cells):
            raise PredictionIntegrityError(
                "S3 requires an exact 9-rho-cell grid for every pair/data-seed/d/"
                "base-innovation/model-seed/origin stratum; "
                f"pair={pair!r}, seed={seed!r}, d={d!r}, cluster={cluster!r}, "
                f"model_seed={model_seed!r}, origin={origin!r}"
            )
    pair_seed_counts = frame.groupby(pair_column, sort=False, dropna=False)[data_seed_column].agg(
        lambda values: set(values)
    )
    if any(observed != set(seeds) for observed in pair_seed_counts):
        raise PredictionIntegrityError("every S3 pair must contain every expected data seed")
    coverage_columns = [pair_column, data_seed_column, "d", cluster_column]
    for identity, block in frame.groupby(coverage_columns, sort=False, dropna=False):
        observed_model_seeds = set(pd.unique(block[model_seed_column]))
        per_seed_origins = {
            seed: tuple(sorted(pd.unique(seed_block[origin_column]).tolist()))
            for seed, seed_block in block.groupby(model_seed_column, sort=False, dropna=False)
        }
        if observed_model_seeds != set(model_seeds) or len(set(per_seed_origins.values())) != 1:
            raise PredictionIntegrityError(
                f"S3 model-seed coverage differs within cluster {identity!r}"
            )
    draws = bootstrap_draws_for_tier(tier)
    output: list[dict[str, Any]] = []
    for pair, pair_frame in frame.groupby(pair_column, sort=False, dropna=False):
        pair_frame = pair_frame.copy()
        pair_frame["__bootstrap_cluster_key__"] = list(
            zip(pair_frame[data_seed_column], pair_frame["d"], pair_frame[cluster_column])
        )
        clusters = list(dict.fromkeys(pair_frame["__bootstrap_cluster_key__"].tolist()))
        if not clusters:
            raise PredictionIntegrityError(f"S3 pair {pair!r} has no base-innovation clusters")
        point = _factorial_effects(pair_frame, value_column)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = np.empty((draws, len(_FACTOR_TERMS)), dtype=np.float64)
        digest = hashlib.sha256()
        for draw_index in range(draws):
            blocks: list[pd.DataFrame] = []
            sampled_indices: list[int] = []
            instance = 0
            for stratum in [(seed, d) for seed in seeds for d in (4, 8)]:
                eligible_indices = [
                    index
                    for index, cluster in enumerate(clusters)
                    if cluster[:2] == stratum
                ]
                if not eligible_indices:
                    raise PredictionIntegrityError(
                        f"S3 bootstrap stratum has no cluster: {stratum!r}"
                    )
                indices = rng.choice(
                    eligible_indices, size=len(eligible_indices), replace=True
                )
                sampled_indices.extend(int(index) for index in indices)
                for index in indices:
                    block = pair_frame.loc[
                        pair_frame["__bootstrap_cluster_key__"] == clusters[int(index)]
                    ].copy()
                    block[cluster_column] = f"bootstrap_{instance:08d}"
                    blocks.append(block)
                    instance += 1
            sampled = pd.concat(blocks, ignore_index=True)
            samples[draw_index] = _factorial_effects(sampled, value_column)
            digest.update(np.asarray(sampled_indices, dtype=np.int64).tobytes())
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)
        for index, term in enumerate(_FACTOR_TERMS):
            output.append(
                {
                    "pair": str(pair),
                    "contrast": term,
                    "effect": float(point[index]),
                    "ci_lower": float(lower[index]),
                    "ci_upper": float(upper[index]),
                    "effect_unit": "percentage_point_relative_sCRPS_gap",
                    "response": "100*(sCRPS_head_a-sCRPS_head_b)/sCRPS_head_b",
                    "coding": {"d": {"4": -1, "8": 1}, "rho": {"-0.8": -1, "0": 0, "0.8": 1}},
                    "quadratic_coding": {"rho": {"-0.8": 1, "0": -2, "0.8": 1}},
                    "contrast_scale": "beta_times_product_of_code_ranges_D_L_2_Q_3",
                    "cluster_unit": [data_seed_column, "d", cluster_column],
                    "cluster_count": len(clusters),
                    "expected_data_seeds": [_json_scalar(seed) for seed in seeds],
                    "expected_model_seeds": [_json_scalar(seed) for seed in model_seeds],
                    "draws": draws,
                    "seed": BOOTSTRAP_SEED,
                    "sampling_sha256": digest.hexdigest(),
                }
            )
    return output
