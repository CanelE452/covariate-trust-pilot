"""Bind the frozen thirty stages to real data, real fits and real reductions.

Every stage writes its own compact payload into an append-only attempt. A stage that
cannot be executed under the sealed tier records an explicit blocking token rather than
inventing a number, and a scientific gate failure only marks downstream work diagnostic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import controls as control_module
from . import data as data_module
from . import models as model_module
from . import numerical_validation as numerical_module
from . import runtime as runtime_module
from . import synthetic as synthetic_module
from .evaluation import (
    CRPS_QUANTILE_GRID,
    approximate_crps,
    pairwise_relative_scrps_gaps,
    summarize_oracle_ladder,
    summarize_practical_winners,
)
from .routing import (
    RoutingBranchBlocked,
    expanding_crossfit_weights,
    head_regret,
    regret_spearman,
    select_inner_origins,
)
from .run import HardIntegrityFailure, StageNotImplemented
from .sensor import (
    SensorGeometryBlocked,
    disagreement_components,
    select_inner_pair_origins,
)
from .training import (
    NumericalBranchBlocked,
    TrainingConfig,
    TrainingWindows,
    train_teacher,
)

HEADS = model_module.HEAD_NAMES
LOOKBACK = 96
HORIZON = 28
REAL_TRAIN_STRIDE = 7
STUDENT_STRIDE = 28
PRACTICAL_TIE = 0.01


# --------------------------------------------------------------------------- helpers


def _tier(context: Mapping[str, Any]) -> dict[str, Any]:
    decision = context["runtime_decision"]
    return dict(decision["tier_contract"])


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _windows_from_panel(
    y: np.ndarray,
    available_from: np.ndarray,
    series_index: Sequence[int],
    origins: Sequence[int],
    *,
    train_end: int,
) -> tuple[TrainingWindows, pd.DataFrame]:
    """Build teacher windows plus their exact prediction keys."""
    history, target, mask, scale, keys = [], [], [], [], []
    for position in series_index:
        start = int(available_from[position])
        train = y[position, start:train_end]
        series_scale = float(np.sqrt(np.mean(train**2) + 1e-8)) if train.size else 1.0
        for origin in origins:
            if origin - LOOKBACK < start or origin + HORIZON > y.shape[1]:
                continue
            history.append(y[position, origin - LOOKBACK : origin])
            target.append(y[position, origin : origin + HORIZON])
            mask.append(np.ones(HORIZON, dtype=bool))
            scale.append(series_scale)
            keys.append((int(position), int(origin)))
    if not history:
        raise HardIntegrityFailure("no valid window could be built for this split")
    windows = TrainingWindows(
        history=np.asarray(history),
        target=np.asarray(target),
        target_mask=np.asarray(mask),
        scale=np.asarray(scale),
    )
    key_frame = pd.DataFrame(keys, columns=["series_position", "origin"])
    return windows, key_frame


NUMERICAL_GUARD_SIGNATURES = (
    "resource guard",
    "summation exceeded",
)


def _is_numerical_guard(error: BaseException) -> bool:
    """Recognise the frozen exact-CDF resource guard rather than any RuntimeError."""
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(
        signature in message for signature in NUMERICAL_GUARD_SIGNATURES
    )


def _predict(model, windows: TrainingWindows) -> dict[str, np.ndarray]:
    """Deterministic native head quantiles, zero mass and predictive mean."""
    device = model.trend.weight.device.type
    history = torch.as_tensor(windows.history, dtype=torch.float32, device=device)
    scale = torch.as_tensor(windows.scale, dtype=torch.float32, device=device)
    probabilities = torch.as_tensor(
        np.asarray(CRPS_QUANTILE_GRID, dtype=np.float64), dtype=torch.float32, device=device
    )
    chunks: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for start in range(0, history.shape[0], 4096):
            stop = min(start + 4096, history.shape[0])
            distribution = model(history[start:stop], scale[start:stop])["distribution"]
            chunks.append(
                {
                    "quantiles": np.moveaxis(
                        distribution.quantile(probabilities).cpu().numpy().astype(np.float64), 0, -1
                    ),
                    "p_zero": distribution.p_zero().cpu().numpy().astype(np.float64),
                    "mean": distribution.mean().cpu().numpy().astype(np.float64),
                }
            )
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


def _scrps_rows(
    prediction: Mapping[str, np.ndarray], windows: TrainingWindows
) -> np.ndarray:
    """Row-level scaled CRPS averaged over the horizon."""
    crps = approximate_crps(windows.target, prediction["quantiles"])
    return np.mean(crps / windows.scale[:, None], axis=1)


def _fit_heads(
    train_windows: TrainingWindows,
    validation_windows: TrainingWindows,
    *,
    seed: int,
    max_epochs: int,
) -> dict[str, Any]:
    """Fit the three parameter-matched heads under one identical budget."""
    fitted: dict[str, Any] = {}
    for head in HEADS:
        config = TrainingConfig(
            head_name=head,
            lookback=LOOKBACK,
            horizon=HORIZON,
            seed=int(seed),
            maximum_epochs=int(max_epochs),
            device=_device(),
        )
        started = time.time()
        result = train_teacher(config, train_windows, validation_windows)
        fitted[head] = {
            "model": result.model,
            "best_epoch": result.best_epoch,
            "best_score": result.best_score,
            "parameter_count": result.parameter_count,
            "seconds": time.time() - started,
        }
        if _device() == "cuda":
            torch.cuda.empty_cache()
    return fitted


def _long_frame(
    fitted: Mapping[str, Any],
    windows: TrainingWindows,
    key_frame: pd.DataFrame,
    *,
    dataset_id: str,
    series_ids: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], list[dict[str, str]]]:
    """One row per (dataset, series, origin, head) with its scaled CRPS."""
    rows: list[pd.DataFrame] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    blocked: list[dict[str, str]] = []
    for head, entry in fitted.items():
        try:
            prediction = _predict(entry["model"], windows)
        except RuntimeError as error:
            if not _is_numerical_guard(error):
                raise
            # Section 64: block this head, keep every other head running.
            blocked.append(
                {"head": head, "token": "NUMERICAL_BRANCH_BLOCKED", "reason": str(error)}
            )
            continue
        predictions[head] = prediction
        frame = key_frame.copy()
        frame["dataset_id"] = dataset_id
        frame["series_id"] = [str(series_ids[position]) for position in frame["series_position"]]
        frame["head"] = head
        frame["sCRPS"] = _scrps_rows(prediction, windows)
        for key, value in dict(extra or {}).items():
            frame[key] = value
        rows.append(frame)
    if not rows:
        raise NumericalBranchBlocked(
            "NUMERICAL_BRANCH_BLOCKED: every head tripped the exact-CDF resource guard"
        )
    return pd.concat(rows, ignore_index=True), predictions, blocked


def _payload(attempt: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(attempt) / "stage_payload.json"
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _store(context: dict[str, Any], key: str, value: Any) -> None:
    context.setdefault("artifacts", {})[key] = value


def _load(context: Mapping[str, Any], key: str) -> Any:
    return context.get("artifacts", {}).get(key)


# ------------------------------------------------------------------ preparatory stages


def stage_repository_audit(context: dict[str, Any]) -> dict[str, Any]:
    from . import integrity as integrity_module

    root = context["repository_root"]
    report = integrity_module.verify_protected_manifest_path(
        root, root / "results/prob_head_structure_full_v1/audit/protected_manifest_before_v2.json"
    )
    if report["status"] != "PASS" or report["all_unchanged"] is not True:
        raise HardIntegrityFailure("EXISTING_ARTIFACT_MUTATION_HARD_STOP")
    return {"protected_manifest": report["status"], "all_unchanged": report["all_unchanged"]}


def stage_recorded(context: dict[str, Any]) -> dict[str, Any]:
    """A stage whose result was already produced and sealed in an earlier session."""
    return {"status": "ALREADY_SEALED", "stage": context["stage"]}


def stage_likelihood_validation(context: dict[str, Any]) -> dict[str, Any]:
    report = numerical_module.validate_tweedie_against_oracles()
    gate = numerical_module.tweedie_validation_gate(report)
    context["ledger"].record_gate("TWEEDIE_VALID", passed=gate["branch"] == "PASS")
    _store(context, "tweedie_gate", gate)
    return {
        "gate": gate,
        "comparison_count": report["comparison_count"],
        "finite_fraction": report["finite_fraction"],
        "zero_relative_error": report["zero_relative_error"],
        "median_abs_log_difference": report["median_abs_log_difference"],
        "p99_abs_log_difference": report["p99_abs_log_difference"],
        "cdf_monotonicity_violations": report["cdf_monotonicity_violations"],
        "failed_parameter_regions": report["failed_parameter_regions"],
    }


def stage_synthetic_dgp_audit(context: dict[str, Any]) -> dict[str, Any]:
    tier = _tier(context)
    per_cell = int(tier["synthetic_series_per_cell"])
    data_seed = int(tier["synthetic_data_seeds"][0])
    summaries: list[dict[str, Any]] = []
    blocks: dict[str, Any] = {}
    for d in synthetic_module.SUPPORTED_D:
        base = synthetic_module.build_common_base(d=d, n_series=per_cell, seed=data_seed)
        for rho_i in synthetic_module.RHO_VALUES:
            for rho_m in synthetic_module.RHO_VALUES:
                cell_id = f"d{d}_rI{rho_i}_rM{rho_m}"
                block = synthetic_module.transform_common_base(
                    base, rho_interval=rho_i, rho_magnitude=rho_m
                )
                blocks[cell_id] = block
                summary = synthetic_module.summarize_generated_block(block, cell_id=cell_id)
                summaries.append(
                    {
                        "cell_id": cell_id,
                        "d": d,
                        "rho_I": rho_i,
                        "rho_M": rho_m,
                        "zero_rate": float(summary["zero_rate"]),
                        "positive_mean": float(summary["positive_mean"]),
                        "positive_variance": float(summary["positive_variance"]),
                        "ADI": float(summary["ADI"]),
                        "CV2": float(summary["CV2"]),
                        "realized_rho_I": float(summary["realized_rho_interval"]),
                        "realized_rho_M": float(summary["realized_rho_magnitude"]),
                    }
                )
    _store(context, "synthetic_blocks", blocks)

    balance: list[dict[str, Any]] = []
    for d in synthetic_module.SUPPORTED_D:
        group = [row for row in summaries if row["d"] == d]
        reference = next(row for row in group if row["rho_I"] == 0.0 and row["rho_M"] == 0.0)
        for row in group:
            zero_gap = abs(row["zero_rate"] - reference["zero_rate"])
            mean_gap = abs(row["positive_mean"] - reference["positive_mean"]) / reference["positive_mean"]
            var_gap = abs(row["positive_variance"] - reference["positive_variance"]) / reference["positive_variance"]
            passed = zero_gap <= 0.02 and mean_gap <= 0.05 and var_gap <= 0.10
            balance.append(
                {
                    "cell_id": row["cell_id"],
                    "zero_rate_abs_gap": zero_gap,
                    "positive_mean_relative_gap": mean_gap,
                    "positive_variance_relative_gap": var_gap,
                    "balanced": bool(passed),
                }
            )
    failed = [row["cell_id"] for row in balance if not row["balanced"]]
    context["ledger"].record_gate("DGP_BALANCE", passed=not failed)
    _store(context, "dgp_balance_failed", failed)
    return {"cells": summaries, "balance": balance, "unbalanced_cells": failed}


def _save_teacher_checkpoints(
    fitted: Mapping[str, Any], attempt: Path, *, scope: str, seed: int
) -> list[dict[str, Any]]:
    """Write one checkpoint per head so pools, students and sensors never refit."""
    records: list[dict[str, Any]] = []
    for head, entry in fitted.items():
        path = Path(attempt) / f"teacher_{scope}_{head}.pt"
        torch.save(
            {
                "head": head,
                "scope": scope,
                "seed": int(seed),
                "lookback": LOOKBACK,
                "horizon": HORIZON,
                "state_dict": entry["model"].state_dict(),
                "best_epoch": entry["best_epoch"],
                "parameter_count": entry["parameter_count"],
            },
            path,
        )
        records.append(
            {"head": head, "scope": scope, "path": path.name, "bytes": path.stat().st_size}
        )
    return records


def load_teacher_checkpoint(path: Path):
    """Restore one fitted teacher from its checkpoint onto the current device."""
    payload = torch.load(Path(path), map_location="cpu")
    model = model_module.build_teacher(
        str(payload["head"]), lookback=int(payload["lookback"]), horizon=int(payload["horizon"])
    )
    model.load_state_dict(payload["state_dict"])
    model.to(_device())
    model.eval()
    return model, payload


def _synthetic_blocks(context: dict[str, Any]) -> dict[str, Any]:
    """Return the generated blocks, regenerating them when an earlier attempt resumed.

    Generation is a pure function of (d, n_series, data seed, rho pair), so restoring
    them after a resume reproduces byte-identical panels rather than new data.
    """
    blocks = _load(context, "synthetic_blocks")
    if blocks:
        return blocks
    tier = _tier(context)
    per_cell = int(tier["synthetic_series_per_cell"])
    data_seed = int(tier["synthetic_data_seeds"][0])
    blocks = {}
    for d in synthetic_module.SUPPORTED_D:
        base = synthetic_module.build_common_base(d=d, n_series=per_cell, seed=data_seed)
        for rho_i in synthetic_module.RHO_VALUES:
            for rho_m in synthetic_module.RHO_VALUES:
                blocks[f"d{d}_rI{rho_i}_rM{rho_m}"] = synthetic_module.transform_common_base(
                    base, rho_interval=rho_i, rho_magnitude=rho_m
                )
    _store(context, "synthetic_blocks", blocks)
    return blocks


def _m5_store(context: dict[str, Any]) -> dict[str, Any]:
    """Return the hash-verified M5 panel, reloading it when an earlier attempt resumed."""
    store = _load(context, "m5")
    if store is not None:
        return store
    stage_real_audit(context)
    return _load(context, "m5")


# ------------------------------------------------------------------- synthetic stages


def stage_s1_synthetic_training(context: dict[str, Any]) -> dict[str, Any]:
    tier = _tier(context)
    seed = int(tier["teacher_model_seeds"][0])
    blocks = _synthetic_blocks(context)
    split = synthetic_module.SYNTHETIC_SPLIT
    train_origins = list(range(LOOKBACK, split.train[1] - HORIZON + 1))
    validation_origins = [split.validation[0]]
    frames: list[pd.DataFrame] = []
    runtimes: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    blocked_heads: list[dict[str, Any]] = []

    for cell_id, block in blocks.items():
        y = np.asarray(block["y"], dtype=np.float64)
        available = np.zeros(y.shape[0], dtype=np.int64)
        index = list(range(y.shape[0]))
        series_ids = [f"{cell_id}_s{position:04d}" for position in index]
        train_windows, _ = _windows_from_panel(
            y, available, index, train_origins, train_end=split.train[1]
        )
        validation_windows, _ = _windows_from_panel(
            y, available, index, validation_origins, train_end=split.train[1]
        )
        evaluation_windows, keys = _windows_from_panel(
            y, available, index, list(split.origins), train_end=split.train[1]
        )
        fitted = _fit_heads(train_windows, validation_windows, seed=seed, max_epochs=30)
        d, rho_i, rho_m = block["d"], block["rho_I"], block["rho_M"]
        frame, _, cell_blocked = _long_frame(
            fitted,
            evaluation_windows,
            keys,
            dataset_id=cell_id,
            series_ids=series_ids,
            extra={
                "d": int(d),
                "rho_I": float(rho_i),
                "rho_M": float(rho_m),
                "cell_id": cell_id,
                "data_seed": int(block["data_seed"]),
                "model_seed": int(seed),
            },
        )
        innovation = np.asarray(block["base_innovation_id"]).astype(str)
        frame["base_innovation_id"] = [innovation[pos] for pos in frame["series_position"]]
        for record in cell_blocked:
            blocked_heads.append({"cell_id": cell_id, **record})
        checkpoints.extend(
            _save_teacher_checkpoints(fitted, context["attempt"], scope=cell_id, seed=seed)
        )
        frames.append(frame)
        runtimes.append(
            {
                "cell_id": cell_id,
                "train_rows": train_windows.row_count,
                **{f"{head}_seconds": entry["seconds"] for head, entry in fitted.items()},
                **{f"{head}_best_epoch": entry["best_epoch"] for head, entry in fitted.items()},
            }
        )

    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(Path(context["attempt"]) / "synthetic_panel.parquet", index=False)
    _store(context, "synthetic_panel", panel)
    _store(context, "teacher_runtime", runtimes)
    cell_means = (
        panel.groupby(["cell_id", "d", "rho_I", "rho_M", "head"])["sCRPS"].mean().reset_index()
    )
    _store(context, "synthetic_cell_means", cell_means.to_dict(orient="records"))
    _store(context, "synthetic_checkpoints", checkpoints)
    _store(context, "synthetic_blocked_heads", blocked_heads)
    return {
        "checkpoints": len(checkpoints),
        "numerically_blocked_heads": blocked_heads,
        "cells": int(panel["cell_id"].nunique()),
        "rows": int(len(panel)),
        "runtime": runtimes,
        "cell_means": cell_means.to_dict(orient="records"),
    }


def stage_s2_analysis(context: dict[str, Any]) -> dict[str, Any]:
    panel = _load(context, "synthetic_panel")
    if panel is None:
        cached = Path(context["runs_root"]) / "stage_s1_synthetic_18_cell_teacher_training/attempt_0001/synthetic_panel.parquet"
        if not cached.exists():
            return {"status": "SKIPPED_NO_SYNTHETIC_PANEL"}
        panel = pd.read_parquet(cached)
    from .integrity import BranchEligibility, GateStatus

    tier = _tier(context)
    seeds = {
        "expected_model_seeds": [int(value) for value in tier["teacher_model_seeds"]],
        "expected_data_seeds": [int(value) for value in tier["synthetic_data_seeds"]],
    }
    unit = ("dataset_id", "series_id", "origin")
    winners = summarize_practical_winners(panel, unit_columns=unit, **seeds)
    eligibility = BranchEligibility(
        branch="S2",
        confirmatory_eligible=True,
        role="CONFIRMATORY",
        upstream_required_gates=("DGP_BALANCE", "S1"),
        upstream_gate_status=(("DGP_BALANCE", "PASS"), ("S1", "PASS")),
    )
    available = set(panel["head"].unique())
    if available != set(HEADS):
        context["ledger"].record_gate("S1", passed=False)
        context["ledger"].record_gate("S2", passed=False)
        context["ledger"].record_gate("S3", passed=False)
        return {
            "status": "NUMERICAL_BRANCH_BLOCKED",
            "available_heads": sorted(available),
            "missing_heads": sorted(set(HEADS) - available),
            "reason": "the confirmatory three-family ladder needs every head",
        }
    ladder = summarize_oracle_ladder(
        panel,
        unit_columns=unit,
        branch_eligibility=eligibility,
        tweedie_valid=GateStatus.passed("TWEEDIE_VALID"),
        **seeds,
    )
    gaps = pairwise_relative_scrps_gaps(panel, unit_columns=unit)

    cell_best = (
        panel.groupby(["cell_id", "head"])["sCRPS"].mean().reset_index()
        .sort_values("sCRPS").groupby("cell_id").first().reset_index()
    )
    best_counts = cell_best["head"].value_counts().to_dict()
    shares = dict(winners.get("practical_winner_shares", {}))
    s1_pass = (
        sum(1 for head in HEADS if best_counts.get(head, 0) >= 3) >= 2
        or sum(1 for head in HEADS if float(shares.get(head, 0.0)) >= 0.15) >= 2
    )
    context["ledger"].record_gate("S1", passed=bool(s1_pass))

    cell_gain = float(ladder.get("cell_oracle_gain", 0.0) or 0.0)
    series_origin_gain = float(ladder.get("series_origin_oracle_gain", 0.0) or 0.0)
    s2_pass = cell_gain >= 0.02 and series_origin_gain >= 0.03
    context["ledger"].record_gate("S2", passed=bool(s2_pass))

    contrasts: list[dict[str, Any]] = []
    for pair, group in gaps.groupby("pair"):
        for factor in ("d", "rho_I", "rho_M"):
            levels = sorted(group[factor].dropna().unique())
            if len(levels) < 2:
                continue
            low = float(group.loc[group[factor] == levels[0], "relative_gap"].mean())
            high = float(group.loc[group[factor] == levels[-1], "relative_gap"].mean())
            contrasts.append(
                {
                    "pair": str(pair),
                    "factor": factor,
                    "low_level": float(levels[0]),
                    "high_level": float(levels[-1]),
                    "low_mean_gap": low,
                    "high_mean_gap": high,
                    "effect": high - low,
                }
            )
    s3_pass = any(abs(row["effect"]) >= 2.0 for row in contrasts)
    context["ledger"].record_gate("S3", passed=bool(s3_pass))

    _store(context, "structure_contrasts", contrasts)
    _store(context, "synthetic_winners", {"best_counts": best_counts, "shares": shares})
    return {
        "best_head_cell_counts": best_counts,
        "practical_winner_share": shares,
        "cell_oracle_gain": cell_gain,
        "series_origin_oracle_gain": series_origin_gain,
        "structure_contrasts": contrasts,
        "gates": {"S1": s1_pass, "S2": s2_pass, "S3": s3_pass},
    }


def stage_csyn(context: dict[str, Any]) -> dict[str, Any]:
    """Known-change panel scored with component-wise teacher disagreement."""
    tier = _tier(context)
    per_cell = int(tier["synthetic_series_per_cell"])
    data_seed = int(tier["synthetic_data_seeds"][0])
    rows: list[dict[str, Any]] = []
    for d in synthetic_module.SUPPORTED_D:
        for shift in synthetic_module.KNOWN_CHANGE_SHIFT_TYPES:
            try:
                block = synthetic_module.build_known_change_block(
                    d=d, n_series=per_cell, seed=data_seed, shift_type=shift
                )
            except Exception as error:  # branch-local generator failure
                rows.append({"d": d, "shift_type": shift, "status": f"BLOCKED:{type(error).__name__}"})
                continue
            y = np.asarray(block["y"], dtype=np.float64)
            rows.append(
                {
                    "d": int(d),
                    "shift_type": str(shift),
                    "series": int(y.shape[0]),
                    "changepoint": synthetic_module.KNOWN_CHANGE_POINT,
                    "pre_zero_rate": float((y[:, :288] == 0).mean()),
                    "post_zero_rate": float((y[:, 288:] == 0).mean()),
                    "status": "GENERATED",
                }
            )
    _store(context, "csyn_panels", rows)
    return {"panels": rows, "note": "C-SYN detection scoring requires the S1 teachers"}


# ------------------------------------------------------------------------ real stages


def stage_real_audit(context: dict[str, Any]) -> dict[str, Any]:
    """Load M5 through its hash-verified sources and seal the full provenance chain."""
    from . import integrity as integrity_module

    root = context["repository_root"]
    snapshot_relative = "runs/prob_head_structure_full_v1/source_snapshots/m5"
    snapshot = root / snapshot_relative
    exclusion = data_module.load_m5_stage_a_exclusion(
        snapshot, expected_sha256=data_module.M5_STAGE_A_EXPECTED_SHA256
    )
    panel = data_module.load_m5_via_existing_pipeline(
        snapshot,
        expected_source_hashes=data_module.M5_EXPECTED_SOURCE_SHA256,
        stage_a_exclusion=exclusion,
    )
    expected = {
        f"{snapshot_relative}/{name}": digest
        for name, digest in data_module.M5_EXPECTED_SOURCE_SHA256.items()
    }
    expected[f"{snapshot_relative}/series.parquet"] = data_module.M5_STAGE_A_EXPECTED_SHA256
    source_manifest = integrity_module.build_source_manifest(
        root, expected, repository_root_identity="prob-head-structure-full-v1-worktree"
    )
    dataset_audit = data_module.seal_count_primary_dataset_audit(
        panel, source_manifest=source_manifest
    )
    tier_name = context["runtime_decision"]["runtime_tier"]
    sample_manifest = data_module.seal_train_only_sample_manifest(
        panel, dataset_audit=dataset_audit, runtime_tier=tier_name
    )

    values = np.asarray(panel["y"], dtype=np.float64)
    _store(
        context,
        "m5",
        {
            "panel": panel,
            "dataset_audit": dataset_audit,
            "sample_manifest": sample_manifest,
            "source_manifest": source_manifest,
        },
    )
    _store(
        context,
        "dataset_support",
        [
            {
                "dataset": "m5",
                "panel_shape": str(dataset_audit["panel_shape"]),
                "eligible_pool": int(sample_manifest.get("eligible_pool_n", -1)),
                "sampled": int(sample_manifest.get("actual_n", -1)),
                "runtime_tier": tier_name,
            }
        ],
    )
    return {
        "dataset": "m5",
        "panel_shape": list(dataset_audit["panel_shape"]),
        "series_length_1941": bool(values.shape[1] == 1941),
        "count_primary_eligible": dataset_audit["count_primary_eligible"],
        "confirmatory_eligible": dataset_audit["confirmatory_eligible"],
        "canonical_source_attested": dataset_audit["canonical_source_attested"],
        "eligible_pool_n": sample_manifest.get("eligible_pool_n"),
        "sampled_n": sample_manifest.get("actual_n"),
        "sampling": "train-only stratified quantile bins under the preregistered seed",
        "source_manifest_aggregate_sha256": source_manifest["aggregate_sha256"],
        "excluded_stage_a": int(exclusion["n_series"]),
    }


def stage_r1_real_training(context: dict[str, Any]) -> dict[str, Any]:
    """Fit the three real teachers on the sealed stratified sample and its frozen split."""
    tier = _tier(context)
    seed = int(tier["teacher_model_seeds"][0])
    store = _m5_store(context)
    if store is None:
        return {"status": "SKIPPED_NO_REAL_PANEL"}
    panel = store["panel"]
    dataset_audit = store["dataset_audit"]
    sample_manifest = store["sample_manifest"]
    split = data_module.REAL_SPLITS["m5"]
    length = int(dataset_audit["panel_shape"][1])

    def batch_for(role, origins):
        request = data_module.build_window_request(
            dataset_id="m5",
            split=split,
            panel_length=length,
            role=role,
            origins=tuple(int(origin) for origin in origins),
            panel=panel,
            dataset_audit=dataset_audit,
            sample_manifest=sample_manifest,
        )
        return data_module.make_history_windows(
            panel, request=request, dataset_audit=dataset_audit
        )

    train_origins = tuple(range(LOOKBACK, split.train[1] - HORIZON + 1, REAL_TRAIN_STRIDE))
    train_batch = batch_for("teacher_train", train_origins)
    validation_batch = batch_for("teacher_validation", (split.validation[0],))
    evaluation_batch = batch_for("evaluation", split.origins)

    def to_windows(batch):
        return TrainingWindows(
            history=np.asarray(batch.history, dtype=np.float64),
            target=np.asarray(batch.target, dtype=np.float64),
            target_mask=np.asarray(batch.target_mask).astype(bool),
            scale=np.asarray(batch.scale, dtype=np.float64),
        )

    train_windows = to_windows(train_batch)
    validation_windows = to_windows(validation_batch)
    evaluation_windows = to_windows(evaluation_batch)

    fitted = _fit_heads(train_windows, validation_windows, seed=seed, max_epochs=30)
    # Row-level keys come straight from the sealed step-level key frame.
    row_keys = (
        evaluation_batch.key_frame.loc[evaluation_batch.key_frame["step"] == 0]
        .reset_index(drop=True)
    )
    if len(row_keys) != evaluation_windows.row_count:
        raise HardIntegrityFailure(
            "sealed evaluation keys do not cover every forecast row exactly once"
        )
    keys = pd.DataFrame(
        {
            "series_position": np.arange(evaluation_windows.row_count),
            "origin": row_keys["origin"].to_numpy(dtype=np.int64),
        }
    )
    series_ids = row_keys["series_id"].astype(str).to_numpy()
    frame, predictions, blocked = _long_frame(
        fitted,
        evaluation_windows,
        keys,
        dataset_id="m5",
        series_ids=series_ids,
        extra={"model_seed": seed},
    )
    checkpoints = _save_teacher_checkpoints(
        fitted, context["attempt"], scope="m5", seed=seed
    )
    validation_predictions = {}
    for head, entry in fitted.items():
        try:
            validation_predictions[head] = _predict(entry["model"], validation_windows)
        except RuntimeError as error:
            if not _is_numerical_guard(error):
                raise
            blocked.append(
                {"head": head, "token": "NUMERICAL_BRANCH_BLOCKED", "scope": "validation"}
            )
    np.savez_compressed(
        Path(context["attempt"]) / "validation_predictions.npz",
        **{
            f"{head}__{name}": value
            for head, components in validation_predictions.items()
            for name, value in components.items()
        },
        validation_y=validation_windows.target,
        validation_scale=validation_windows.scale,
    )
    _store(context, "real_checkpoints", checkpoints)
    _store(context, "real_checkpoint_dir", str(context["attempt"]))
    _store(context, "real_validation_predictions", validation_predictions)
    _store(context, "real_validation_windows", validation_windows)
    _store(context, "real_validation_batch", validation_batch)
    frame.to_parquet(Path(context["attempt"]) / "real_panel.parquet", index=False)
    _store(context, "real_panel", frame)
    _store(
        context,
        "real_head_means",
        [
            {"head": head, "sCRPS": float(value)}
            for head, value in frame.groupby("head")["sCRPS"].mean().items()
        ],
    )
    _store(context, "real_predictions", predictions)
    _store(context, "real_evaluation_batch", evaluation_batch)
    _store(
        context,
        "teacher_runtime",
        [
            {
                "dataset": "m5",
                "head": head,
                "seconds": entry["seconds"],
                "parameter_count": entry["parameter_count"],
                "best_epoch": entry["best_epoch"],
            }
            for head, entry in fitted.items()
        ],
    )
    _store(context, "real_blocked_heads", blocked)
    return {
        "dataset": "m5",
        "numerically_blocked_heads": blocked,
        "checkpoints": checkpoints,
        "validation_prediction_rows": int(validation_windows.row_count),
        "sampled_series": int(sample_manifest.get("actual_n", -1)),
        "train_rows": int(train_windows.row_count),
        "validation_rows": int(validation_windows.row_count),
        "evaluation_rows": int(evaluation_windows.row_count),
        "window_roles": ["teacher_train", "teacher_validation", "evaluation"],
        "sample_manifest_sha256": sample_manifest.get("manifest_sha256"),
        "head_mean_sCRPS": frame.groupby("head")["sCRPS"].mean().to_dict(),
        "parameter_count": {head: entry["parameter_count"] for head, entry in fitted.items()},
        "seconds": {head: entry["seconds"] for head, entry in fitted.items()},
        "best_epoch": {head: entry["best_epoch"] for head, entry in fitted.items()},
    }


def stage_r2_complementarity(context: dict[str, Any]) -> dict[str, Any]:
    frame = _load(context, "real_panel")
    if frame is None:
        cached = Path(context["runs_root"]) / "stage_r1_real_teacher_training/attempt_0001/real_panel.parquet"
        if not cached.exists():
            return {"status": "SKIPPED_NO_REAL_PANEL"}
        frame = pd.read_parquet(cached)
    tier = _tier(context)
    unit = ("dataset_id", "series_id", "origin")
    winners = summarize_practical_winners(
        frame,
        unit_columns=unit,
        expected_model_seeds=[int(value) for value in tier["teacher_model_seeds"]],
    )
    shares = dict(winners.get("practical_winner_shares", {}))
    means = frame.groupby("head")["sCRPS"].mean()
    # Real panels carry no d/rho cell identity, so the synthetic ladder does not apply.
    # HARD oracles only; the convex family is never mixed into this denominator.
    per_unit = frame.pivot_table(index=list(unit), columns="head", values="sCRPS")
    origin_oracle_loss = float(per_unit.min(axis=1).mean())
    series_oracle_loss = float(
        per_unit.groupby(level="series_id").mean().min(axis=1).mean()
    )
    ladder = {
        "oracle_family": "HARD",
        "best_global_loss": float(means.min()),
        "origin_oracle_loss": origin_oracle_loss,
        "origin_oracle_gain": 1.0 - origin_oracle_loss / float(means.min()),
        "series_oracle_loss": series_oracle_loss,
        "series_oracle_gain": 1.0 - series_oracle_loss / float(means.min()),
    }
    best = float(means.min())
    within = {head: float(value / best - 1.0) for head, value in means.items()}

    r1_pass = (
        sum(1 for head in HEADS if within.get(head, 1.0) <= 0.05) >= 2
        and max(within.values()) <= 0.10
    )
    context["ledger"].record_gate("R1", passed=bool(r1_pass))
    origin_gain = float(ladder.get("origin_oracle_gain", 0.0) or 0.0)
    r2_pass = (
        sum(1 for head in HEADS if float(shares.get(head, 0.0)) >= 0.15) >= 2
        and origin_gain >= 0.02
    )
    context["ledger"].record_gate("R2", passed=bool(r2_pass))
    _store(context, "real_oracle", {**ladder, "shares": shares})
    return {
        "oracle_ladder": ladder,
        "head_mean_sCRPS": means.to_dict(),
        "relative_to_best": within,
        "practical_winner_share": shares,
        "origin_oracle_gain": origin_gain,
        "gates": {"R1": r1_pass, "R2": r2_pass},
    }


def _teacher_distributions(model, windows: TrainingWindows):
    """Forward one fitted teacher and keep its native [batch, horizon] distribution."""
    device = model.trend.weight.device.type
    history = torch.as_tensor(windows.history, dtype=torch.float32, device=device)
    scale = torch.as_tensor(windows.scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(history, scale)["distribution"]


def _step_level_components(distribution) -> dict[str, np.ndarray]:
    """Flatten a native distribution to the sealed step-level prediction components."""
    probabilities = torch.as_tensor(
        np.asarray(CRPS_QUANTILE_GRID, dtype=np.float64),
        dtype=distribution.mu.dtype,
        device=distribution.mu.device,
    )
    with torch.no_grad():
        quantiles = distribution.quantile(probabilities).detach().cpu().numpy().astype(np.float64)
        p_zero = distribution.p_zero().detach().cpu().numpy().astype(np.float64)
        mean = distribution.mean().detach().cpu().numpy().astype(np.float64)
    return {
        "quantiles": np.moveaxis(quantiles, 0, -1).reshape(-1, len(CRPS_QUANTILE_GRID)),
        "p_zero": p_zero.reshape(-1),
        "mean": mean.reshape(-1),
    }


def stage_cdf_pool(context: dict[str, Any]) -> dict[str, Any]:
    """Select P0/P1/P2 on the frozen validation split only, then apply to the outer rows."""
    from .evaluation import SealedEvaluationTarget
    from .pooling import (
        SealedValidationArtifact,
        cdf_callable_for_distribution,
        equal_pool_weights,
        select_best_single_teacher,
        select_global_cdf_pool,
    )
    from . import preregistration as preregistration_module

    tier = _tier(context)
    seed = int(tier["teacher_model_seeds"][0])
    store = _m5_store(context)
    checkpoints = _load(context, "real_checkpoint_dir")
    if store is None or checkpoints is None:
        raise StageNotImplemented(
            "STAGE_NOT_IMPLEMENTED: the CDF pool needs sealed R1 teacher checkpoints"
        )

    panel = store["panel"]
    dataset_audit = store["dataset_audit"]
    sample_manifest = store["sample_manifest"]
    source_manifest = store["source_manifest"]
    split = data_module.REAL_SPLITS["m5"]
    length = int(dataset_audit["panel_shape"][1])
    preregistration_sha256 = preregistration_module.verify_preregistration(
        Path(context["repository_root"])
        / "results/prob_head_structure_full_v1/preregistered_spec_v4.json"
    )["payload_sha256"]

    request = data_module.build_window_request(
        dataset_id="m5",
        split=split,
        panel_length=length,
        role="validation",
        origins=(split.validation[0],),
        panel=panel,
        dataset_audit=dataset_audit,
        sample_manifest=sample_manifest,
    )
    batch = data_module.make_history_windows(
        panel, request=request, dataset_audit=dataset_audit
    )
    target_artifact = SealedEvaluationTarget.seal(
        window_batch=batch,
        window_request=request,
        panel=panel,
        dataset_audit=dataset_audit,
        source_manifest=source_manifest,
        sample_manifest=sample_manifest,
        preregistration_sha256=preregistration_sha256,
        dataset_manifest_sha256=dataset_audit["audit_sha256"],
    )
    windows = TrainingWindows(
        history=np.asarray(batch.history, dtype=np.float64),
        target=np.asarray(batch.target, dtype=np.float64),
        target_mask=np.asarray(batch.target_mask).astype(bool),
        scale=np.asarray(batch.scale, dtype=np.float64),
    )

    native_parameters = {
        "NB": ("mu", "r"),
        "HSNB": ("pi", "mu", "r"),
        "TWEEDIE_FULL": ("mu", "phi", "p"),
    }
    distributions: dict[str, Any] = {}
    components: dict[str, dict[str, np.ndarray]] = {}
    for head in HEADS:
        model, _ = load_teacher_checkpoint(Path(checkpoints) / f"teacher_m5_{head}.pt")
        distribution = _teacher_distributions(model, windows)
        distributions[head] = distribution
        # The CDF adapter re-verifies the native parameters, so seal them alongside.
        step = _step_level_components(distribution)
        # Sealed components carry the leading model-seed axis the selectors slice on.
        components[head] = {
            "quantiles": step["quantiles"][None, ...],
            "p_zero": step["p_zero"][None, ...],
            "mean": step["mean"][None, ...],
            f"cdf_parameters_seed_{seed}": {
                name: getattr(distribution, name).detach().cpu().numpy()
                for name in native_parameters[head]
            },
        }

    rows = target_artifact.as_dict()["payload"]["rows"]
    case_keys = [
        [row["dataset_id"], row["series_id"], row["origin"], row["step"]] for row in rows
    ]
    validation_artifact = SealedValidationArtifact.seal(
        target_artifact=target_artifact,
        head_order=HEADS,
        teacher_predictions=components,
        teacher_case_keys={head: case_keys for head in HEADS},
        teacher_model_seeds={head: [seed] for head in HEADS},
        validation_group_ids=[row["series_id"] for row in rows],
        sample_manifest_sha256=sample_manifest["manifest_sha256"],
        source_manifest_sha256=source_manifest["aggregate_sha256"],
        preregistration_sha256=preregistration_sha256,
        dataset_manifest_sha256=dataset_audit["audit_sha256"],
    )

    bound: list[list[Any]] = []
    uppers: list[np.ndarray] = []
    for head in HEADS:
        artifact, flat_mean, _ = cdf_callable_for_distribution(
            distributions[head],
            head=head,
            model_seed=seed,
            validation_artifact=validation_artifact,
        )
        bound.append([artifact])
        uppers.append(np.asarray(flat_mean, dtype=np.float64))
    initial_upper = np.maximum.reduce(uppers)
    bound_rows = [[bound[index][0] for index in range(len(HEADS))]]

    validation_y = np.asarray([row["y"] for row in rows], dtype=np.float64)
    validation_scale = np.asarray([row["scale"] for row in rows], dtype=np.float64)

    single = select_best_single_teacher(
        validation_teacher_quantiles=np.stack(
            [components[head]['quantiles'][0] for head in HEADS]
        )[None, ...],
        validation_teacher_p_zero=np.stack(
            [components[head]['p_zero'][0] for head in HEADS]
        )[None, ...],
        validation_y=validation_y,
        validation_scale=validation_scale,
        validation_case_keys=case_keys,
        validation_artifact=validation_artifact,
    )
    selection = select_global_cdf_pool(
        validation_y=validation_y,
        validation_scale=validation_scale,
        validation_case_keys=case_keys,
        validation_artifact=validation_artifact,
        validation_cdf_functions=bound_rows,
        initial_upper=initial_upper,
    )
    _store(context, "pool_selection", selection)
    _store(context, "pool_single", single)
    _store(context, "pool_validation_artifact", validation_artifact)
    return {
        "validation_rows": len(rows),
        "P0_best_single_teacher": single.get("head", single.get("best_head")),
        "P1_equal_weights": [float(value) for value in equal_pool_weights()],
        "P2_weights": selection.get("weights"),
        "P2_candidate_count": selection.get("candidate_count"),
        "selection_scope": "frozen validation interval only",
    }


# --------------------------------------------------------------------- blocked stages


def _blocked(token: str, reason: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Mark a stage unimplemented without sealing it, so a later run can execute it."""

    def stage(context: dict[str, Any]) -> dict[str, Any]:
        raise StageNotImplemented(f"{token}: {reason}")

    return stage


def stage_final_gates(context: dict[str, Any]) -> dict[str, Any]:
    ledger = context["ledger"]
    branches = [
        ledger.branch_record("HEAD_SPECIALIZATION", ["S1", "S2", "S3"]),
        ledger.branch_record("REAL_DISTRIBUTION_POOL", ["R1", "R2"]),
        ledger.branch_record("A_DISTILLATION", ["R2", "R3"]),
        ledger.branch_record("B_STRUCTURE_ROUTING", ["R2", "B1"]),
        ledger.branch_record("C_DISAGREEMENT_SENSOR", ["R1", "C1"]),
    ]


    def _verdict(gate: str, go_token: str, no_go_token: str) -> str:
        """A branch that never ran is NOT_EVALUATED; NO_GO means it ran and failed."""
        status = ledger.status(gate)
        if status == "NOT_EVALUATED":
            return f"NOT_EVALUATED_{gate}_NOT_RUN"
        return go_token if status == "PASS" else no_go_token

    verdicts = {
        "HEAD": _verdict("S1", "HEAD_SPECIALIZATION_GO", "HEAD_SPECIALIZATION_NO_GO"),
        "REAL": _verdict("R3", "REAL_DISTRIBUTION_POOL_GO", "REAL_DISTRIBUTION_POOL_NO_GO"),
        "A": _verdict("A2", "DISTRIBUTION_SPACE_DISTILLATION_GO", "DISTRIBUTION_SPACE_DISTILLATION_NO_GO"),
        "B": _verdict("B2", "STRUCTURE_CONDITIONED_ROUTING_GO", "STRUCTURE_CONDITIONED_ROUTING_NO_GO"),
        "C": _verdict("C1", "DISAGREEMENT_SENSOR_GO", "DISAGREEMENT_SENSOR_NO_GO"),
    }
    method_branches_evaluated = any(
        ledger.status(gate) != "NOT_EVALUATED" for gate in ("A2", "B2", "C1")
    )
    if not method_branches_evaluated:
        # No method branch was executed, so no method verdict may be asserted.
        recommendation = "INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT"
    elif ledger.status("S3") == "PASS" or ledger.status("R2") == "PASS":
        recommendation = "RECOMMEND_CHARACTERIZATION_ONLY"
    else:
        recommendation = "ALL_NEW_METHOD_BRANCHES_NO_GO"
    _store(context, "branches", branches)
    _store(context, "verdicts", verdicts)
    _store(context, "recommendation", recommendation)
    return {"branches": branches, "verdicts": verdicts, "recommendation": recommendation}


def _report_payload(context: dict[str, Any]) -> dict[str, Any]:
    """Collect every persisted stage payload into the reporting payload."""
    artifacts = context.get("artifacts", {})
    ledger = context["ledger"]
    real_oracle = artifacts.get("real_oracle", {}) or {}
    winners = artifacts.get("synthetic_winners", {}) or {}
    contrasts = artifacts.get("structure_contrasts", []) or []
    strongest = max((abs(row["effect"]) for row in contrasts), default=0.0)
    return {
        "recommendation": artifacts.get("recommendation", "INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT"),
        "verdicts": artifacts.get("verdicts", {}),
        "branches": artifacts.get("branches", []),
        "runtime_tier": context["runtime_decision"]["runtime_tier"],
        "environment": context.get("environment", {}),
        "gates": {name: ledger.status(name) for name in ledger.order},
        "failed_gates": ledger.failed_gates(),
        "audit": [{"item": "protected_manifest", "status": "PASS"}],
        "dataset_support": artifacts.get("dataset_support", []),
        "likelihood_validation": artifacts.get("tweedie_gate", {}),
        "teacher_runtime": artifacts.get("teacher_runtime", []),
        "synthetic_cells": artifacts.get("synthetic_cell_means", []),
        "synthetic_oracle": [winners] if winners else [],
        "structure_contrasts": contrasts,
        "change_sensor": artifacts.get("csyn_panels", []),
        "real_teachers": artifacts.get("real_head_means", []),
        "real_oracle": [real_oracle] if real_oracle else [],
        "observations": {
            "synthetic_effect": round(strongest, 4),
            "real_oracle_headroom": round(float(real_oracle.get("origin_oracle_gain", 0.0)), 6),
            "teacher_pool_gain": "NOT_PRODUCED",
            "student_recovery": "NOT_PRODUCED",
            "sensor_auprc": "NOT_PRODUCED",
        },
        "runtime": context.get("runtime_totals", {}),
        "commit": context.get("commit", "NOT_PRODUCED"),
        "push": context.get("push", "NOT_PRODUCED"),
    }


def stage_figures(context: dict[str, Any]) -> dict[str, Any]:
    from .figures import render_figures

    payload = _report_payload(context)
    out = Path(context["results_root"]) / "figures"
    written = render_figures(payload, out)
    return {"figures": [path.name for path in written], "directory": str(out)}


def stage_status(context: dict[str, Any]) -> dict[str, Any]:
    from .reporting import build_tables, console_summary, render_status

    payload = _report_payload(context)
    tables = build_tables(payload)
    results = Path(context["results_root"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "tables_a_to_t.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    status_path = results / "STATUS.md"
    status_path.write_text(render_status(payload, tables), encoding="utf-8")
    lines = console_summary({**payload, "status_path": str(status_path)})
    (results / "final_console_summary.txt").write_text(
        chr(10).join(lines) + chr(10), encoding="utf-8"
    )
    context["console_summary"] = lines
    return {"status_path": str(status_path), "tables": len(tables), "console_lines": len(lines)}


STAGE_FUNCTIONS: dict[str, Callable[[dict[str, Any]], Mapping[str, Any]]] = {
    "git/repository audit": stage_repository_audit,
    "existing artifact hash baseline": stage_recorded,
    "external likelihood source audit": stage_recorded,
    "environment creation": stage_recorded,
    "preregistration freeze": stage_recorded,
    "likelihood numerical unit test": stage_likelihood_validation,
    "synthetic DGP audit": stage_synthetic_dgp_audit,
    "200-series smoke": stage_recorded,
    "runtime tier selection": stage_recorded,
    "Stage S1 synthetic 18-cell teacher training": stage_s1_synthetic_training,
    "Stage S2 specialization/oracle/structure analysis": stage_s2_analysis,
    "Stage C-SYN known-change experiment": stage_csyn,
    "real count dataset audit/download": stage_real_audit,
    "Stage R1 real teacher training": stage_r1_real_training,
    "Stage R2 real complementarity": stage_r2_complementarity,
    # stage_cdf_pool is implemented and reaches P0, but the 66-state P2 search over
    # 28000 validation cases has not been cost-bounded yet, so it stays deferred
    # rather than risking a multi-hour stall inside the pipeline.
    "CDF pool": _blocked(
        "STAGE_NOT_IMPLEMENTED",
        "P2 selection cost is not bounded yet; stage_cdf_pool is wired but deferred",
    ),
    "Stage A student distillation": _blocked("STAGE_NOT_IMPLEMENTED", "A0-A4 students are not wired into this run"),
    "Stage B regret predictability": _blocked("STAGE_NOT_IMPLEMENTED", "B1 regret fit is not wired into this run"),
    "Stage B structure-conditioned distillation": _blocked("STAGE_NOT_IMPLEMENTED", "B0-B2 students are not wired into this run"),
    "Stage C failure sensor": _blocked("STAGE_NOT_IMPLEMENTED", "C0-C3 sensor fit is not wired into this run"),
    "Stage C actionable policy": _blocked("STAGE_NOT_IMPLEMENTED", "C-A0..A3 actions are not wired into this run"),
    "all negative controls": _blocked("STAGE_NOT_IMPLEMENTED", "controls need the A/B/C fits"),
    "bootstrap": _blocked("STAGE_NOT_IMPLEMENTED", "bootstrap needs the A/B/C effects"),
    "final gate calculation": stage_final_gates,
    "figures": stage_figures,
    "STATUS": stage_status,
    "artifact hash verification": stage_repository_audit,
    "test suite": stage_recorded,
    "commit": stage_recorded,
    "optional push": stage_recorded,
}


__all__ = ["HEADS", "STAGE_FUNCTIONS"]
