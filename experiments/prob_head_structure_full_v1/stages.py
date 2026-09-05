"""Bind the frozen thirty stages to real data, real fits and real reductions.

Every stage writes its own compact payload into an append-only attempt. A stage that
cannot be executed under the sealed tier records an explicit blocking token rather than
inventing a number, and a scientific gate failure only marks downstream work diagnostic.
"""

from __future__ import annotations

import hashlib
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
from .run import HardIntegrityFailure, StageInputUnavailable, StageNotImplemented
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
    from . import gate_records
    from .gates import GateResult

    result = GateResult(
        gate="DGP_BALANCE",
        passed=not failed,
        verdict="DGP_BALANCE_PASS" if not failed else "DGP_BALANCE_FAIL",
        failure_label=None if not failed else "DGP_BALANCE_FAIL",
        observations={
            "total_cells": len(balance),
            "unbalanced_cells": failed,
            "tolerances": {
                "zero_rate_absolute_difference_max": 0.02,
                "positive_mean_relative_difference_max": 0.05,
                "positive_variance_relative_difference_max": 0.10,
            },
            "remove_cell": False,
        },
        criteria={"every_registered_cell_within_frozen_tolerances": not failed},
    )
    context["ledger"].record_gate("DGP_BALANCE", passed=result.passed)
    context.setdefault("gate_result_objects", {})["DGP_BALANCE"] = result
    _store(context, "dgp_balance_failed", failed)
    return {
        "cells": summaries,
        "balance": balance,
        "unbalanced_cells": failed,
        **gate_records.payload_block([result]),
    }


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


def _completed_attempt(runs_root: Path, slug: str) -> Path | None:
    """Return a stage's most recent sealed attempt.

    A re-rendering stage such as the CDF pool seals more than one attempt, and an early
    one can hold a superseded payload, so readers must take the latest rather than the
    first: reading the stale one silently fed a degenerate teacher weight vector.
    """
    stage_root = Path(runs_root) / slug
    if not stage_root.is_dir():
        return None
    sealed = [
        attempt
        for attempt in sorted(stage_root.glob("attempt_*"))
        if (attempt / "completion.json").exists()
    ]
    return sealed[-1] if sealed else None


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


def _branch_eligibility(context: dict[str, Any], branch: str):
    """Build the branch lineage from the real ledger, never from an assumed PASS."""
    from .integrity import _REQUIRED_GATES, BranchEligibility, GateStatus

    ledger = context["ledger"]
    statuses = []
    for gate in _REQUIRED_GATES[branch]:
        status = ledger.status(gate)
        if status == "NOT_EVALUATED":
            raise StageInputUnavailable(
                f"STAGE_INPUT_UNAVAILABLE: {branch} needs a decided {gate}"
            )
        statuses.append(
            GateStatus.passed(gate)
            if status == "PASS"
            else GateStatus.scientific_failure(gate)
        )
    return BranchEligibility.begin(branch, statuses)


def _s3_contrast_rows(
    panel: pd.DataFrame, cells: Sequence[str], *, tier: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str | None]:
    """Estimate the frozen 3x17 contrast registry, or say exactly why it cannot be."""
    from .bootstrap import factorial_temporal_contrasts

    subset = panel.loc[panel["cell_id"].isin(list(cells))].reset_index(drop=True)
    try:
        gaps = pairwise_relative_scrps_gaps(
            subset, unit_columns=("dataset_id", "series_id", "origin")
        )
        rows = factorial_temporal_contrasts(
            gaps,
            tier=str(tier["runtime_tier"]),
            expected_data_seeds=[int(value) for value in tier["synthetic_data_seeds"]],
            expected_model_seeds=[int(value) for value in tier["teacher_model_seeds"]],
        )
    except Exception as error:  # the estimator refuses an incomplete frozen design
        return [], f"{type(error).__name__}: {error}"
    return rows, None


def stage_s2_analysis(context: dict[str, Any]) -> dict[str, Any]:
    """S1/S2/S3 observations handed to the frozen reducers, which alone decide."""
    from . import gate_records
    from . import gates as gate_module
    from .integrity import GateStatus

    panel = _load(context, "synthetic_panel")
    if panel is None:
        attempt = _completed_attempt(
            context["runs_root"], "stage_s1_synthetic_18_cell_teacher_training"
        )
        cached = None if attempt is None else attempt / "synthetic_panel.parquet"
        if cached is None or not cached.exists():
            raise StageInputUnavailable(
                "STAGE_INPUT_UNAVAILABLE: no sealed Stage S1 synthetic panel to analyse"
            )
        panel = pd.read_parquet(cached)

    tier = _tier(context)
    seeds = {
        "expected_model_seeds": [int(value) for value in tier["teacher_model_seeds"]],
        "expected_data_seeds": [int(value) for value in tier["synthetic_data_seeds"]],
    }
    unbalanced = list(_sealed_payload(context, "synthetic_dgp_audit").get("unbalanced_cells", []))

    # The frozen contract keeps every cell (synthetic.balance_failure.remove_cell=false);
    # a numerically blocked head costs its cell the three-family comparison, and an
    # unbalanced cell costs the aggregate its confirmatory standing through the lineage.
    coverage = panel.groupby("cell_id")["head"].nunique()
    all_cells = sorted(coverage.index)
    complete_cells = sorted(coverage[coverage == len(HEADS)].index)
    incomplete = [
        {
            "cell_id": str(cell_id),
            "present_heads": sorted(panel.loc[panel["cell_id"] == cell_id, "head"].unique()),
            "missing_heads": sorted(
                set(HEADS) - set(panel.loc[panel["cell_id"] == cell_id, "head"].unique())
            ),
            "reason": "NUMERICAL_BRANCH_BLOCKED",
            "confirmatory_eligible": False,
        }
        for cell_id in sorted(coverage[coverage != len(HEADS)].index)
    ]
    confirmatory_cells = [cell for cell in complete_cells if cell not in set(unbalanced)]
    panels = {
        "synthetic_diagnostic_panel": all_cells,
        "synthetic_complete_coverage_panel": complete_cells,
        "synthetic_confirmatory_panel": confirmatory_cells,
    }
    panel_digests = {
        name: hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        for name, ids in panels.items()
    }
    if not complete_cells:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: no cell retained the complete three-family coverage"
        )

    complete = panel.loc[panel["cell_id"].isin(complete_cells)].reset_index(drop=True)
    confirmatory = panel.loc[panel["cell_id"].isin(confirmatory_cells)].reset_index(drop=True)

    # --- S1 -----------------------------------------------------------------
    # The reducer fixes the registered grid at eighteen cells, so the counts cover
    # every cell; a cell whose NB head was numerically blocked ranks the heads that
    # actually produced a distribution, and that limitation is recorded here.
    cell_best = (
        panel.groupby(["cell_id", "head"])["sCRPS"].mean().reset_index()
        .sort_values("sCRPS").groupby("cell_id").first().reset_index()
    )
    counts = {head: int((cell_best["head"] == head).sum()) for head in HEADS}
    winners = summarize_practical_winners(complete, unit_columns=("dataset_id", "series_id", "origin"), **seeds)
    shares = {head: float(winners["practical_winner_shares"].get(head, 0.0)) for head in HEADS}
    s1 = gate_module.gate_s1(
        exact_best_cell_counts=counts,
        practical_winner_shares=shares,
        total_cells=len(all_cells),
    )
    s1 = gate_records.record(
        context,
        s1,
        branch="HEAD_SPECIALIZATION",
        upstream=[_dgp_balance_result(context)],
    )

    # --- S2 -----------------------------------------------------------------
    # The confirmatory ladder refuses a lineage that already carries a failure, which is
    # the frozen behaviour once DGP_BALANCE fails. The same arithmetic still runs, but
    # the result is labelled diagnostic instead of being passed off as confirmatory.
    eligibility = _branch_eligibility(context, "S2")
    if eligibility.confirmatory_eligible:
        ladder = summarize_oracle_ladder(
            confirmatory,
            unit_columns=("dataset_id", "series_id", "origin"),
            branch_eligibility=eligibility,
            tweedie_valid=GateStatus.passed("TWEEDIE_VALID"),
            **seeds,
        )
    else:
        from .evaluation import (
            _average_registered_model_seed_losses,
            _summarize_oracle_ladder_core,
        )

        ladder = {
            **_summarize_oracle_ladder_core(
                _average_registered_model_seed_losses(
                    confirmatory, head_column="head", loss_column="sCRPS", **seeds
                ),
                required_heads=tuple(HEADS),
                unit_columns=("dataset_id", "series_id", "origin"),
            ),
            "confirmatory_eligible": False,
            "scientific_role": eligibility.role,
        }
    s2 = gate_module.gate_s2(
        cell_oracle_gain=float(ladder.get("cell_oracle_gain", 0.0) or 0.0),
        series_origin_oracle_gain=float(ladder.get("series_origin_oracle_gain", 0.0) or 0.0),
    )
    s2 = gate_records.record(
        context, s2, branch="HEAD_SPECIALIZATION", upstream=[_dgp_balance_result(context), s1]
    )

    # --- S3 -----------------------------------------------------------------
    # The frozen estimator needs the exact nine-rho grid at both d levels. Every panel
    # this run can form is missing at least one cell, so no contrast set exists and the
    # gate stays NOT_EVALUATED, which is not a scientific FAIL.
    tier_for_s3 = {
        "runtime_tier": context["runtime_decision"]["runtime_tier"],
        "synthetic_data_seeds": tier["synthetic_data_seeds"],
        "teacher_model_seeds": tier["teacher_model_seeds"],
    }
    s3_attempts: dict[str, Any] = {}
    contrasts: list[dict[str, Any]] = []
    for name, ids in panels.items():
        rows, reason = _s3_contrast_rows(panel, ids, tier=tier_for_s3)
        s3_attempts[name] = {"cells": len(ids), "contrasts": len(rows), "refusal": reason}
        if rows and not contrasts:
            contrasts = rows
    s3_result = None
    if contrasts:
        s3_result = gate_module.gate_s3(contrasts)
        s3_result = gate_records.record(
            context,
            s3_result,
            branch="HEAD_SPECIALIZATION",
            upstream=[_dgp_balance_result(context), s1, s2],
        )

    produced = [item for item in (s1, s2, s3_result) if item is not None]
    payload: dict[str, Any] = {
        "panels": panels,
        "panel_series_id_sha256": panel_digests,
        "confirmatory_cells": len(confirmatory_cells),
        "complete_coverage_cells": len(complete_cells),
        "total_cells": len(all_cells),
        "unbalanced_cells": unbalanced,
        "excluded_cells": incomplete,
        "S1_scope": "every registered cell; frozen reducer fixes the grid at 18",
        "S1_cells_ranking_fewer_than_three_heads": [row["cell_id"] for row in incomplete],
        "S1_share_scope": "complete-coverage cells only; shares need identical head coverage",
        "best_head_cell_counts": counts,
        "practical_winner_share": shares,
        "S2_scope": "confirmatory panel: DGP-balanced and complete three-head coverage",
        "oracle_ladder": {
            key: value for key, value in ladder.items() if not isinstance(value, (list, dict))
        },
        "S3_status": "NOT_EVALUATED" if not contrasts else "EVALUATED",
        "S3_refusals": s3_attempts,
        "S3_note": (
            "the frozen factorial estimator requires the exact nine-rho grid at d=4 and "
            "d=8; the NUMERICAL_BRANCH_BLOCKED cell removes one, so no registered "
            "51-contrast set can be formed"
        ),
        "structure_contrasts": contrasts,
        **gate_records.payload_block(produced),
    }
    _store(context, "synthetic_winners", {"best_counts": counts, "shares": shares})
    _store(context, "structure_contrasts", contrasts)
    _store(context, "synthetic_confirmatory_cells", confirmatory_cells)
    return payload


def _dgp_balance_result(context: dict[str, Any]):
    """The sealed DGP_BALANCE result, rebuilt as the reducer object it always was."""
    from . import gate_records
    from .gates import GateResult

    stored = context.get("gate_result_objects", {}).get("DGP_BALANCE")
    if stored is not None:
        return stored
    sealed = _sealed_payload(context, "synthetic_dgp_audit")
    record = dict(sealed.get(gate_records.PAYLOAD_KEY, {}).get("DGP_BALANCE", {}))
    if record:
        return gate_records.deserialize(record)
    failed = list(sealed.get("unbalanced_cells", []))
    return GateResult(
        gate="DGP_BALANCE",
        passed=not failed,
        verdict="DGP_BALANCE_PASS" if not failed else "DGP_BALANCE_FAIL",
        failure_label=None if not failed else "DGP_BALANCE_FAIL",
        observations={"unbalanced_cells": failed, "source": "sealed synthetic DGP audit"},
        criteria={"every_registered_cell_within_frozen_tolerances": not failed},
    )


# ------------------------------------------------------------------------ real stages


# The frozen geometry-only rows: these datasets are too short for any window contract,
# so they can never establish count eligibility. Lengths and official horizons come from
# the preregistration's real.insufficient_geometry_audits block.
GEOMETRY_ONLY_AUDITS = (
    {"dataset_id": "auto", "length": 24, "lookback": 48, "horizon": 6},
    {"dataset_id": "carparts", "length": 51, "lookback": 48, "horizon": 6},
    {"dataset_id": "raf", "length": 84, "lookback": 48, "horizon": 12},
)
REAL_SOURCE_DIRECTORIES = {
    "m5": "runs/prob_head_structure_full_v1/source_snapshots/m5",
    "online_retail": "runs/prob_head_structure_full_v1/source_snapshots/online_retail",
}


def _expected_real_source_hashes() -> dict[str, str]:
    """Every physical source the two count-primary loaders read, in one manifest."""
    m5 = REAL_SOURCE_DIRECTORIES["m5"]
    expected = {f"{m5}/{name}": digest for name, digest in data_module.M5_EXPECTED_SOURCE_SHA256.items()}
    expected[f"{m5}/series.parquet"] = data_module.M5_STAGE_A_EXPECTED_SHA256
    expected[
        f"{REAL_SOURCE_DIRECTORIES['online_retail']}/online_retail_II.xlsx"
    ] = data_module.ONLINE_RETAIL_EXPECTED_SHA256
    return expected


def stage_real_audit(context: dict[str, Any]) -> dict[str, Any]:
    """Audit all five priority datasets and seal the frozen count-primary selection."""
    from . import gates as gate_module
    from . import integrity as integrity_module

    root = context["repository_root"]
    source_manifest = integrity_module.build_source_manifest(
        root,
        _expected_real_source_hashes(),
        repository_root_identity="prob-head-structure-full-v1-worktree",
    )

    exclusion = data_module.load_m5_stage_a_exclusion(
        root / REAL_SOURCE_DIRECTORIES["m5"],
        expected_sha256=data_module.M5_STAGE_A_EXPECTED_SHA256,
    )
    panels = {
        "m5": data_module.load_m5_via_existing_pipeline(
            root / REAL_SOURCE_DIRECTORIES["m5"],
            expected_source_hashes=data_module.M5_EXPECTED_SOURCE_SHA256,
            stage_a_exclusion=exclusion,
        ),
        "online_retail": data_module.load_online_retail_tweediegp(
            root / REAL_SOURCE_DIRECTORIES["online_retail"],
            expected_sha256=data_module.ONLINE_RETAIL_EXPECTED_SHA256,
        ),
    }
    audits = {
        dataset_id: data_module.seal_count_primary_dataset_audit(
            panel, source_manifest=source_manifest
        )
        for dataset_id, panel in panels.items()
    }

    ordered_rows = [audits["m5"]]
    for row in GEOMETRY_ONLY_AUDITS:
        ordered_rows.append(data_module.audit_fixed_length_dataset(**row))
    ordered_rows.append(audits["online_retail"])

    selection = data_module.select_real_datasets(ordered_rows)
    data_module.verify_real_dataset_selection_manifest(selection)
    manifest = gate_module.FrozenPrimaryDatasetManifest.from_selection_audit(selection)

    tier_name = context["runtime_decision"]["runtime_tier"]
    store: dict[str, Any] = {
        "selection": selection,
        "manifest": manifest,
        "source_manifest": source_manifest,
        "datasets": {},
    }
    support: list[dict[str, Any]] = []
    for dataset_id in manifest.selected_datasets:
        sample_manifest = data_module.seal_train_only_sample_manifest(
            panels[dataset_id], dataset_audit=audits[dataset_id], runtime_tier=tier_name
        )
        store["datasets"][dataset_id] = {
            "panel": panels[dataset_id],
            "dataset_audit": audits[dataset_id],
            "sample_manifest": sample_manifest,
        }
        support.append(
            {
                "dataset": dataset_id,
                "panel_shape": list(audits[dataset_id]["panel_shape"]),
                "eligible_pool": int(sample_manifest.get("eligible_pool_n", -1)),
                "sampled": int(sample_manifest.get("actual_n", -1)),
                "runtime_tier": tier_name,
            }
        )
    _store(context, "real", store)
    _store(context, "dataset_support", support)

    return {
        "priority": list(data_module.REAL_DATASET_PRIORITY),
        "selected_dataset_ids": list(manifest.selected_datasets),
        "eligible_dataset_ids_in_priority_order": list(
            manifest.eligible_datasets_in_priority_order
        ),
        "selection_status": selection["status"],
        "eligible_non_m5_count": selection["eligible_non_m5_count"],
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "selection_manifest_sha256": selection["selection_manifest_sha256"],
        "audit_manifest_sha256": selection["audit_manifest_sha256"],
        "audits": [
            {
                "dataset_id": row["dataset_id"],
                "audit_type": row["audit_type"],
                "status": row["status"],
                "geometry_status": row.get("geometry_status"),
                "count_primary_eligible": row["count_primary_eligible"],
                "confirmatory_eligible": row["confirmatory_eligible"],
                "audit_sha256": row["audit_sha256"],
            }
            for row in ordered_rows
        ],
        "dataset_support": support,
        "source_manifest_aggregate_sha256": source_manifest["aggregate_sha256"],
        "excluded_stage_a": int(exclusion["n_series"]),
    }


def _real_store(context: dict[str, Any]) -> dict[str, Any]:
    """The hash-verified real panels and their frozen manifest, reloading on resume."""
    store = _load(context, "real")
    if store is not None:
        return store
    stage_real_audit(context)
    return _load(context, "real")


def _primary_manifest(context: dict[str, Any]):
    return _real_store(context)["manifest"]


def _selected_datasets(context: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_primary_manifest(context).selected_datasets)


def _dataset_entry(context: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    return _real_store(context)["datasets"][str(dataset_id)]


def _m5_store(context: dict[str, Any]) -> dict[str, Any]:
    """Backwards-compatible view for the stages that still speak only M5."""
    entry = _dataset_entry(context, "m5")
    return {**entry, "source_manifest": _real_store(context)["source_manifest"]}


def _head_metrics(
    prediction: Mapping[str, np.ndarray], windows: TrainingWindows
) -> dict[str, float]:
    """The frozen R1/R3 metric set on one head's native output."""
    from .evaluation import pinball_loss

    grid = np.asarray(CRPS_QUANTILE_GRID, dtype=np.float64)
    target = np.asarray(windows.target, dtype=np.float64)
    scale = np.asarray(windows.scale, dtype=np.float64)[:, None]
    quantiles = np.asarray(prediction["quantiles"], dtype=np.float64)
    p_zero = np.asarray(prediction["p_zero"], dtype=np.float64)
    mean = np.asarray(prediction["mean"], dtype=np.float64)

    def scaled_pinball(probability: float) -> float:
        index = int(np.argmin(np.abs(grid - probability)))
        return float(np.mean(pinball_loss(target, quantiles[..., index], probability) / scale))

    squared = np.mean(((mean - target) / scale) ** 2)
    return {
        "sCRPS": float(np.mean(approximate_crps(target, quantiles) / scale)),
        "zero_brier": float(np.mean((p_zero - (target == 0.0).astype(np.float64)) ** 2)),
        "sQL_50": scaled_pinball(0.50),
        "sQL_95": scaled_pinball(0.95),
        "sQL_99": scaled_pinball(0.99),
        "tail_sQL": 0.5 * (scaled_pinball(0.95) + scaled_pinball(0.99)),
        "NRMSE": float(np.sqrt(squared)),
    }


def _role_windows(
    context: dict[str, Any], role: str, origins: Sequence[int], *, dataset_id: str = "m5"
) -> tuple[TrainingWindows, Any]:
    """Build one sealed window batch for a real panel and its training view."""
    entry = _dataset_entry(context, dataset_id)
    panel = entry["panel"]
    dataset_audit = entry["dataset_audit"]
    request = data_module.build_window_request(
        dataset_id=str(dataset_id),
        split=data_module.REAL_SPLITS[str(dataset_id)],
        panel_length=int(dataset_audit["panel_shape"][1]),
        role=role,
        origins=tuple(int(origin) for origin in origins),
        panel=panel,
        dataset_audit=dataset_audit,
        sample_manifest=entry["sample_manifest"],
    )
    batch = data_module.make_history_windows(
        panel, request=request, dataset_audit=dataset_audit
    )
    windows = TrainingWindows(
        history=np.asarray(batch.history, dtype=np.float64),
        target=np.asarray(batch.target, dtype=np.float64),
        target_mask=np.asarray(batch.target_mask).astype(bool),
        scale=np.asarray(batch.scale, dtype=np.float64),
    )
    return windows, batch


def stage_r1_real_training(context: dict[str, Any]) -> dict[str, Any]:
    """Fit the three teachers separately on every selected count-primary dataset."""
    tier = _tier(context)
    seed = int(tier["teacher_model_seeds"][0])
    frames: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    runtimes: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    blocked_all: list[dict[str, Any]] = []
    rows_by_dataset: dict[str, dict[str, int]] = {}

    for dataset_id in _selected_datasets(context):
        split = data_module.REAL_SPLITS[dataset_id]
        train_origins = tuple(
            range(LOOKBACK, split.train[1] - HORIZON + 1, REAL_TRAIN_STRIDE)
        )
        train_windows, _ = _role_windows(
            context, "teacher_train", train_origins, dataset_id=dataset_id
        )
        validation_windows, _ = _role_windows(
            context, "teacher_validation", (split.validation[0],), dataset_id=dataset_id
        )
        evaluation_windows, evaluation_batch = _role_windows(
            context, "evaluation", split.origins, dataset_id=dataset_id
        )

        fitted = _fit_heads(train_windows, validation_windows, seed=seed, max_epochs=30)
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
        frame, predictions, blocked = _long_frame(
            fitted,
            evaluation_windows,
            keys,
            dataset_id=dataset_id,
            series_ids=row_keys["series_id"].astype(str).to_numpy(),
            extra={"model_seed": seed},
        )
        frames.append(frame)
        metrics[dataset_id] = {
            head: _head_metrics(prediction, evaluation_windows)
            for head, prediction in predictions.items()
        }
        checkpoints.extend(
            _save_teacher_checkpoints(fitted, context["attempt"], scope=dataset_id, seed=seed)
        )
        for record in blocked:
            blocked_all.append({"dataset_id": dataset_id, **record})
        validation_predictions = {}
        for head, entry in fitted.items():
            try:
                validation_predictions[head] = _predict(entry["model"], validation_windows)
            except RuntimeError as error:
                if not _is_numerical_guard(error):
                    raise
                blocked_all.append(
                    {
                        "dataset_id": dataset_id,
                        "head": head,
                        "token": "NUMERICAL_BRANCH_BLOCKED",
                        "scope": "validation",
                    }
                )
        np.savez_compressed(
            Path(context["attempt"]) / f"validation_predictions_{dataset_id}.npz",
            **{
                f"{head}__{name}": value
                for head, components in validation_predictions.items()
                for name, value in components.items()
            },
            validation_y=validation_windows.target,
            validation_scale=validation_windows.scale,
        )
        rows_by_dataset[dataset_id] = {
            "train_rows": int(train_windows.row_count),
            "validation_rows": int(validation_windows.row_count),
            "evaluation_rows": int(evaluation_windows.row_count),
            "sampled_series": int(_dataset_entry(context, dataset_id)["sample_manifest"]["actual_n"]),
        }
        runtimes.extend(
            {
                "dataset": dataset_id,
                "head": head,
                "seconds": entry["seconds"],
                "parameter_count": entry["parameter_count"],
                "best_epoch": entry["best_epoch"],
            }
            for head, entry in fitted.items()
        )

    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(Path(context["attempt"]) / "real_panel.parquet", index=False)
    _store(context, "real_panel", panel)
    _store(context, "real_head_metrics", metrics)
    _store(context, "real_checkpoint_dir", str(context["attempt"]))
    _store(context, "teacher_runtime", runtimes)
    _store(context, "real_blocked_heads", blocked_all)
    return {
        "datasets": list(_selected_datasets(context)),
        "primary_dataset_manifest_sha256": _primary_manifest(context).audit_sha256,
        "numerically_blocked_heads": blocked_all,
        "checkpoints": checkpoints,
        "rows": rows_by_dataset,
        "head_metrics": metrics,
        "runtime": runtimes,
    }


def _real_metrics(context: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    metrics = _load(context, "real_head_metrics")
    if metrics:
        return metrics
    sealed = _sealed_payload(context, "stage_r1_real_teacher_training")
    metrics = dict(sealed.get("head_metrics", {}))
    if not metrics:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: no sealed Stage R1 head metrics to analyse"
        )
    return metrics


def _real_panel(context: dict[str, Any]) -> pd.DataFrame:
    frame = _load(context, "real_panel")
    if frame is not None:
        return frame
    attempt = _completed_attempt(context["runs_root"], "stage_r1_real_teacher_training")
    cached = None if attempt is None else attempt / "real_panel.parquet"
    if cached is None or not cached.exists():
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: no sealed Stage R1 real panel to analyse"
        )
    frame = pd.read_parquet(cached)
    _store(context, "real_panel", frame)
    return frame


def _head_coverage(context: dict[str, Any]) -> dict[str, Any]:
    """Which selected datasets kept all three families through the frozen guard.

    The exact-CDF resource guard is a terminal block for the affected head, and the
    contract forbids clamping or truncating around it. A dataset that lost a family can
    still be reported, but no three-family confirmatory gate can be formed from it.
    """
    metrics = _real_metrics(context)
    selected = _selected_datasets(context)
    complete = tuple(
        dataset
        for dataset in selected
        if set(metrics.get(dataset, {})) == set(HEADS)
    )
    incomplete = {
        dataset: sorted(set(HEADS) - set(metrics.get(dataset, {})))
        for dataset in selected
        if set(metrics.get(dataset, {})) != set(HEADS)
    }
    return {
        "metrics": metrics,
        "selected": selected,
        "complete": complete,
        "incomplete": incomplete,
        "manifest_complete": not incomplete,
        "token": "NUMERICAL_BRANCH_BLOCKED" if incomplete else None,
    }


def _blocked_gate_note(coverage: Mapping[str, Any], gate: str) -> str:
    return (
        f"{gate} is NOT_EVALUATED: the frozen reducer requires all three families on "
        f"every selected dataset, but {coverage['incomplete']} lost heads to "
        "NUMERICAL_BRANCH_BLOCKED. An unevaluated gate is not a scientific FAIL."
    )


def stage_r2_complementarity(context: dict[str, Any]) -> dict[str, Any]:
    """R1 and R2 observations across every selected dataset, decided by the reducers."""
    from . import gate_records
    from . import gates as gate_module
    from .bootstrap import pairwise_loss_correlation_bootstrap

    frame = _real_panel(context)
    coverage = _head_coverage(context)
    metrics = coverage["metrics"]
    manifest = _primary_manifest(context)
    datasets = tuple(manifest.selected_datasets)
    tier = _tier(context)
    unit = ("dataset_id", "series_id", "origin")

    if not coverage["manifest_complete"]:
        # Report the surviving evidence, but form no scientific verdict from a panel
        # the frozen reducers cannot accept.
        diagnostic = {}
        for dataset in coverage["complete"]:
            block = frame.loc[frame["dataset_id"] == dataset]
            per_unit = block.pivot_table(index=list(unit), columns="head", values="sCRPS")
            means = block.groupby("head")["sCRPS"].mean()
            diagnostic[dataset] = {
                "head_mean_sCRPS": {h: float(v) for h, v in means.items()},
                "best_head": str(means.idxmin()),
                "origin_oracle_gain": float(
                    1.0 - float(per_unit.min(axis=1).mean()) / float(means.min())
                ),
            }
        return {
            "datasets": list(datasets),
            "primary_dataset_manifest_sha256": manifest.audit_sha256,
            "head_metrics": metrics,
            "R1_status": "NOT_EVALUATED",
            "R2_status": "NOT_EVALUATED",
            "not_evaluated_reason": _blocked_gate_note(coverage, "R1/R2"),
            "numerically_blocked": coverage["incomplete"],
            "complete_head_datasets": list(coverage["complete"]),
            "diagnostic_per_dataset": diagnostic,
            "scientific_role": "DIAGNOSTIC_ONLY_NUMERICAL_BRANCH_BLOCKED",
            "confirmatory_eligible": False,
        }

    # --- R1 -----------------------------------------------------------------
    best_scrps = {
        dataset: min(metrics[dataset][head]["sCRPS"] for head in metrics[dataset])
        for dataset in datasets
    }
    macro = {
        head: {
            metric: float(np.mean([metrics[dataset][head][metric] for dataset in datasets]))
            for metric in ("zero_brier", "tail_sQL")
        }
        for head in HEADS
    }
    best_zero = min(macro, key=lambda head: macro[head]["zero_brier"])
    best_tail = min(macro, key=lambda head: macro[head]["tail_sQL"])
    head_quality = {
        head: {
            "relative_scrps_gap": {
                dataset: float(
                    metrics[dataset][head]["sCRPS"] / best_scrps[dataset] - 1.0
                )
                for dataset in datasets
            },
            "zero_brier_best": head == best_zero,
            "tail_sql_best": head == best_tail,
        }
        for head in HEADS
    }
    r1 = gate_module.gate_r1(head_quality, primary_datasets=manifest)
    r1 = gate_records.record(context, r1, branch="REAL_DISTRIBUTION_POOL")

    # --- R2 -----------------------------------------------------------------
    winners = summarize_practical_winners(
        frame,
        unit_columns=unit,
        expected_model_seeds=[int(value) for value in tier["teacher_model_seeds"]],
    )
    shares = {
        head: float(winners["practical_winner_shares"].get(head, 0.0)) for head in HEADS
    }
    # HARD oracles only; the convex family never enters this denominator.
    dataset_gains: dict[str, float] = {}
    dataset_best_heads: dict[str, str] = {}
    for dataset in datasets:
        block = frame.loc[frame["dataset_id"] == dataset]
        per_unit = block.pivot_table(index=list(unit), columns="head", values="sCRPS")
        means = block.groupby("head")["sCRPS"].mean()
        best = float(means.min())
        dataset_gains[dataset] = float(1.0 - float(per_unit.min(axis=1).mean()) / best)
        dataset_best_heads[dataset] = str(means.idxmin())
    macro_gain = float(np.mean([dataset_gains[dataset] for dataset in datasets]))

    correlation_frame = frame.rename(columns={"head": "method", "sCRPS": "loss"})
    correlations = pairwise_loss_correlation_bootstrap(
        correlation_frame,
        tier=str(context["runtime_decision"]["runtime_tier"]),
        observation_columns=["origin"],
    )
    pairwise = {
        pair: {
            "upper": float(row["upper"]),
            "degenerate_resample_present": bool(row["degenerate_resample_present"]),
        }
        for pair, row in correlations["pairs"].items()
    }
    r2 = gate_module.gate_r2(
        practical_winner_shares=shares,
        macro_oracle_gain=macro_gain,
        dataset_oracle_gains={dataset: dataset_gains[dataset] for dataset in datasets},
        pairwise_correlation_ci=pairwise,
        dataset_best_heads={dataset: dataset_best_heads[dataset] for dataset in datasets},
        primary_datasets=manifest,
    )
    r2 = gate_records.record(context, r2, branch="REAL_DISTRIBUTION_POOL", upstream=[r1])

    _store(context, "real_oracle", {"dataset_gains": dataset_gains, "macro_gain": macro_gain, "shares": shares})
    return {
        "datasets": list(datasets),
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "head_metrics": metrics,
        "relative_scrps_gap": {head: head_quality[head]["relative_scrps_gap"] for head in HEADS},
        "macro_zero_brier": {head: macro[head]["zero_brier"] for head in HEADS},
        "macro_tail_sQL": {head: macro[head]["tail_sQL"] for head in HEADS},
        "best_zero_brier_head": best_zero,
        "best_tail_sql_head": best_tail,
        "practical_winner_share": shares,
        "per_dataset_practical_winner_share": winners.get("per_dataset", {}),
        "dataset_oracle_gains": dataset_gains,
        "macro_oracle_gain": macro_gain,
        "dataset_best_heads": dataset_best_heads,
        "pairwise_correlation_ci": pairwise,
        "oracle_family": "HARD",
        **gate_records.payload_block([r1, r2]),
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


class _DiagnosticCDF:
    """Flat-case CDF adapter for the diagnostic pooled inversion outside the sealed path."""

    def __init__(self, distribution: Any, row_count: int) -> None:
        self._distribution = distribution
        self._rows = int(row_count)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float64).reshape(self._rows, HORIZON)
        tensor = torch.as_tensor(
            flat, dtype=self._distribution.mu.dtype, device=self._distribution.mu.device
        )
        with torch.no_grad():
            return self._distribution.cdf(tensor).detach().cpu().numpy().reshape(-1)


NATIVE_PARAMETERS = {
    "NB": ("mu", "r"),
    "HSNB": ("pi", "mu", "r"),
    "TWEEDIE_FULL": ("mu", "phi", "p"),
}


def _preregistration_sha256(context: dict[str, Any]) -> str:
    from . import preregistration as preregistration_module

    cached = _load(context, "preregistration_sha256")
    if cached:
        return cached
    digest = preregistration_module.verify_preregistration(
        Path(context["repository_root"])
        / "results/prob_head_structure_full_v1/preregistered_spec_v4.json"
    )["payload_sha256"]
    _store(context, "preregistration_sha256", digest)
    return digest


def _sealed_split_artifact(
    context: dict[str, Any],
    *,
    dataset_id: str,
    role: str,
    origins: Sequence[int],
    checkpoints: Path,
    seed: int,
) -> dict[str, Any]:
    """Seal one split's evaluation target together with its three teacher outputs."""
    from .evaluation import SealedEvaluationTarget
    from .pooling import SealedValidationArtifact

    entry = _dataset_entry(context, dataset_id)
    store = _real_store(context)
    panel = entry["panel"]
    dataset_audit = entry["dataset_audit"]
    sample_manifest = entry["sample_manifest"]
    request = data_module.build_window_request(
        dataset_id=str(dataset_id),
        split=data_module.REAL_SPLITS[str(dataset_id)],
        panel_length=int(dataset_audit["panel_shape"][1]),
        role=role,
        origins=tuple(int(origin) for origin in origins),
        panel=panel,
        dataset_audit=dataset_audit,
        sample_manifest=sample_manifest,
    )
    batch = data_module.make_history_windows(
        panel, request=request, dataset_audit=dataset_audit
    )
    preregistration_sha256 = _preregistration_sha256(context)
    target_artifact = SealedEvaluationTarget.seal(
        window_batch=batch,
        window_request=request,
        panel=panel,
        dataset_audit=dataset_audit,
        source_manifest=store["source_manifest"],
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
    distributions: dict[str, Any] = {}
    components: dict[str, dict[str, Any]] = {}
    for head in HEADS:
        model, _ = load_teacher_checkpoint(Path(checkpoints) / f"teacher_{dataset_id}_{head}.pt")
        distribution = _teacher_distributions(model, windows)
        distributions[head] = distribution
        step = _step_level_components(distribution)
        components[head] = {
            "quantiles": step["quantiles"][None, ...],
            "p_zero": step["p_zero"][None, ...],
            "mean": step["mean"][None, ...],
            f"cdf_parameters_seed_{seed}": {
                name: getattr(distribution, name).detach().cpu().numpy()
                for name in NATIVE_PARAMETERS[head]
            },
        }
    rows = target_artifact.as_dict()["payload"]["rows"]
    case_keys = [
        [row["dataset_id"], row["series_id"], row["origin"], row["step"]] for row in rows
    ]
    artifact = SealedValidationArtifact.seal(
        target_artifact=target_artifact,
        head_order=HEADS,
        teacher_predictions=components,
        teacher_case_keys={head: case_keys for head in HEADS},
        teacher_model_seeds={head: [seed] for head in HEADS},
        validation_group_ids=[row["series_id"] for row in rows],
        sample_manifest_sha256=sample_manifest["manifest_sha256"],
        source_manifest_sha256=store["source_manifest"]["aggregate_sha256"],
        preregistration_sha256=preregistration_sha256,
        dataset_manifest_sha256=dataset_audit["audit_sha256"],
    )
    return {
        "artifact": artifact,
        "components": components,
        "distributions": distributions,
        "case_keys": case_keys,
        "windows": windows,
        "y": np.asarray([row["y"] for row in rows], dtype=np.float64),
        "scale": np.asarray([row["scale"] for row in rows], dtype=np.float64),
        "row_count": int(windows.row_count),
    }


def _pooled_metrics(
    quantiles: np.ndarray, p_zero: np.ndarray, y: np.ndarray, scale: np.ndarray
) -> dict[str, float]:
    """The frozen R3 metric set on a pooled flat-case output."""
    from .evaluation import pinball_loss

    grid = np.asarray(CRPS_QUANTILE_GRID, dtype=np.float64)
    flat_q = np.asarray(quantiles, dtype=np.float64).reshape(-1, grid.size)
    flat_p0 = np.asarray(p_zero, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    denominator = np.asarray(scale, dtype=np.float64).reshape(-1)

    def scaled_pinball(probability: float) -> float:
        index = int(np.argmin(np.abs(grid - probability)))
        return float(np.mean(pinball_loss(target, flat_q[:, index], probability) / denominator))

    median = flat_q[:, int(np.argmin(np.abs(grid - 0.50)))]
    return {
        "sCRPS": float(np.mean(approximate_crps(target, flat_q) / denominator)),
        "zero_brier": float(np.mean((flat_p0 - (target == 0.0).astype(np.float64)) ** 2)),
        "sQL_50": scaled_pinball(0.50),
        "sQL_95": scaled_pinball(0.95),
        "sQL_99": scaled_pinball(0.99),
        "tail_sQL": 0.5 * (scaled_pinball(0.95) + scaled_pinball(0.99)),
        "NRMSE": float(np.sqrt(np.mean(((median - target) / denominator) ** 2))),
    }


def stage_cdf_pool(context: dict[str, Any]) -> dict[str, Any]:
    """Select P0/P1/P2/P3 on validation only, apply to the outer rows, decide R3."""
    from . import gate_records
    from . import gates as gate_module
    from .evaluation import PredictionIntegrityError
    from .pooling import (
        apply_global_cdf_pool,
        apply_quantile_specific_pool,
        cdf_callable_for_distribution,
        equal_pool_weights,
        select_best_single_teacher,
        select_global_cdf_pool,
        select_primary_pool,
        select_quantile_specific_pool,
    )

    tier = _tier(context)
    seed = int(tier["teacher_model_seeds"][0])
    checkpoints = _load(context, "real_checkpoint_dir")
    if checkpoints is None:
        attempt = _completed_attempt(context["runs_root"], "stage_r1_real_teacher_training")
        checkpoints = None if attempt is None else str(attempt)
    if checkpoints is None:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: the CDF pool needs sealed R1 teacher checkpoints"
        )
    manifest = _primary_manifest(context)
    coverage = _head_coverage(context)
    datasets = coverage["complete"]
    if not datasets:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: no dataset kept all three families"
        )
    # A pooled forecast can only be applied under a confirmatory lineage; without one the
    # same weights are still applied, but the result is reported as diagnostic.
    eligibility = _branch_eligibility(context, "R3") if coverage["manifest_complete"] else None

    per_dataset: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for dataset_id in datasets:
        split = data_module.REAL_SPLITS[dataset_id]
        validation = _sealed_split_artifact(
            context,
            dataset_id=dataset_id,
            role="validation",
            origins=(split.validation[0],),
            checkpoints=Path(checkpoints),
            seed=seed,
        )
        stacked_quantiles = np.stack(
            [validation["components"][head]["quantiles"][0] for head in HEADS]
        )[None, ...]
        stacked_p_zero = np.stack(
            [validation["components"][head]["p_zero"][0] for head in HEADS]
        )[None, ...]

        p0 = select_best_single_teacher(
            validation_teacher_quantiles=stacked_quantiles,
            validation_teacher_p_zero=stacked_p_zero,
            validation_y=validation["y"],
            validation_scale=validation["scale"],
            validation_case_keys=validation["case_keys"],
            validation_artifact=validation["artifact"],
        )
        bound, uppers = [], []
        for head in HEADS:
            artifact, flat_mean, _ = cdf_callable_for_distribution(
                validation["distributions"][head],
                head=head,
                model_seed=seed,
                validation_artifact=validation["artifact"],
            )
            bound.append(artifact)
            uppers.append(np.asarray(flat_mean, dtype=np.float64))
        p2 = select_global_cdf_pool(
            validation_y=validation["y"],
            validation_scale=validation["scale"],
            validation_case_keys=validation["case_keys"],
            validation_artifact=validation["artifact"],
            validation_cdf_functions=[bound],
            initial_upper=np.maximum.reduce(uppers),
        )
        # P3 is a branch-local hard block when no penalty path yields a coherent
        # distribution; the primary pool then has P2 as its only registered candidate.
        p3, p3_block = None, None
        try:
            p3 = select_quantile_specific_pool(
                teacher_quantiles=stacked_quantiles,
                validation_teacher_p_zero=stacked_p_zero,
                validation_y=validation["y"],
                validation_scale=validation["scale"],
                validation_case_keys=validation["case_keys"],
                validation_artifact=validation["artifact"],
            )
        except PredictionIntegrityError as error:
            p3_block = str(error)
        if p3 is None:
            primary = {
                "primary_pool": "P2",
                "reason": f"P3 unavailable: {p3_block}",
                "validation_artifact_sha256": validation["artifact"].artifact_sha256,
            }
        else:
            primary = select_primary_pool(
                p2_selection=p2, p3_selection=p3, validation_artifact=validation["artifact"]
            )

        outer = _sealed_split_artifact(
            context,
            dataset_id=dataset_id,
            role="evaluation",
            origins=split.origins,
            checkpoints=Path(checkpoints),
            seed=seed,
        )
        outer_quantiles = np.stack(
            [outer["components"][head]["quantiles"][0] for head in HEADS]
        )[None, ...]
        outer_p_zero = np.stack(
            [outer["components"][head]["p_zero"][0] for head in HEADS]
        )[None, ...]
        head_metrics = {
            head: _pooled_metrics(
                outer["components"][head]["quantiles"][0],
                outer["components"][head]["p_zero"][0],
                outer["y"],
                outer["scale"],
            )
            for head in HEADS
        }
        best_single = min(head_metrics, key=lambda head: head_metrics[head]["sCRPS"])

        if eligibility is None:
            # Diagnostic pooling in quantile space: the same validation-selected weights,
            # applied without claiming the confirmatory applier's lineage.
            weights = np.asarray(
                p3["weights_by_quantile"]
                if p3 is not None and str(primary["primary_pool"]) == "P3"
                else p2["weights"],
                dtype=np.float64,
            )
            if weights.ndim == 1:
                pooled_quantiles = np.tensordot(weights, outer_quantiles[0], axes=(0, 0))
                pooled_p_zero = np.tensordot(weights, outer_p_zero[0], axes=(0, 0))
            else:
                pooled_quantiles = np.einsum("thsq,qt->hsq", outer_quantiles[0], weights)
                pooled_p_zero = np.tensordot(
                    weights.mean(axis=0), outer_p_zero[0], axes=(0, 0)
                )
            pooled_quantiles = np.maximum.accumulate(pooled_quantiles, axis=-1)
        elif p3 is not None and str(primary["primary_pool"]) == "P3":
            applied = apply_quantile_specific_pool(
                teacher_quantiles=outer_quantiles,
                teacher_p_zero=outer_p_zero,
                selection=p3,
                validation_artifact=validation["artifact"],
                prediction_artifact=outer["artifact"],
                branch_eligibility=eligibility,
            )
            pooled_quantiles = np.asarray(applied["quantiles"], dtype=np.float64)
            pooled_p_zero = np.asarray(applied["p_zero"], dtype=np.float64)
        else:
            outer_bound, outer_upper = [], []
            for head in HEADS:
                artifact, flat_mean, _ = cdf_callable_for_distribution(
                    outer["distributions"][head],
                    head=head,
                    model_seed=seed,
                    validation_artifact=outer["artifact"],
                )
                outer_bound.append(artifact)
                outer_upper.append(np.asarray(flat_mean, dtype=np.float64))
            applied = apply_global_cdf_pool(
                selection=p2,
                validation_artifact=validation["artifact"],
                prediction_artifact=outer["artifact"],
                cdf_functions=[outer_bound],
                probabilities=CRPS_QUANTILE_GRID,
                case_count=int(outer["y"].size),
                initial_upper=np.maximum.reduce(outer_upper),
                branch_eligibility=eligibility,
            )
            pooled_quantiles = np.asarray(applied["quantiles"], dtype=np.float64)
            weights = np.asarray(p2["weights"], dtype=np.float64)
            pooled_p_zero = np.tensordot(weights, outer_p_zero[0], axes=(0, 0))

        pooled = _pooled_metrics(
            pooled_quantiles, pooled_p_zero, outer["y"], outer["scale"]
        )
        baseline = head_metrics[best_single]
        per_dataset[dataset_id] = {
            "P0_best_single_teacher": p0.get("teacher"),
            "P0_validation_sCRPS": p0.get("validation_sCRPS"),
            "P1_equal_weights": [float(value) for value in equal_pool_weights()],
            "P2_weights": [float(value) for value in p2["weights"]],
            "P2_candidate_count": p2.get("candidate_count"),
            "P3_penalty": None if p3 is None else p3.get("penalty"),
            "P3_post_crossing_rate": None if p3 is None else p3.get("post_crossing_rate"),
            "P3_block": p3_block,
            "primary_pool": primary["primary_pool"],
            "primary_pool_reason": primary.get("reason"),
            "validation_rows": validation["row_count"],
            "outer_rows": outer["row_count"],
            "outer_head_metrics": head_metrics,
            "outer_best_single_head": best_single,
            "outer_pool_metrics": pooled,
            "scrps_improvement": float(1.0 - pooled["sCRPS"] / baseline["sCRPS"]),
            "tail_sql_improvement": float(1.0 - pooled["tail_sQL"] / baseline["tail_sQL"]),
            "q50_deterioration": float(pooled["sQL_50"] / baseline["sQL_50"] - 1.0),
            "zero_brier_deterioration": float(
                pooled["zero_brier"] / baseline["zero_brier"] - 1.0
            ),
            "nrmse_deterioration": float(pooled["NRMSE"] / baseline["NRMSE"] - 1.0),
        }
        selections[dataset_id] = {"P2": p2, "P3": p3, "primary": primary, "P0": p0}

    def macro(field: str) -> float:
        return float(np.mean([per_dataset[dataset][field] for dataset in datasets]))

    if eligibility is None:
        return {
            "datasets": list(datasets),
            "primary_dataset_manifest_sha256": manifest.audit_sha256,
            "selection_scope": "frozen validation interval only",
            "per_dataset": per_dataset,
            "R3_status": "NOT_EVALUATED",
            "not_evaluated_reason": _blocked_gate_note(coverage, "R3"),
            "numerically_blocked": coverage["incomplete"],
            "scientific_role": "DIAGNOSTIC_ONLY_NUMERICAL_BRANCH_BLOCKED",
            "confirmatory_eligible": False,
            "macro": {
                "scrps_improvement": macro("scrps_improvement"),
                "tail_sql_improvement": macro("tail_sql_improvement"),
                "q50_deterioration": macro("q50_deterioration"),
                "zero_brier_deterioration": macro("zero_brier_deterioration"),
                "nrmse_deterioration": macro("nrmse_deterioration"),
            },
        }

    r3 = gate_module.gate_r3(
        macro_scrps_improvement=macro("scrps_improvement"),
        dataset_scrps_improvements={
            dataset: per_dataset[dataset]["scrps_improvement"] for dataset in datasets
        },
        tail_sql_improvement=macro("tail_sql_improvement"),
        q50_deterioration=macro("q50_deterioration"),
        zero_brier_deterioration=macro("zero_brier_deterioration"),
        nrmse_deterioration=macro("nrmse_deterioration"),
        primary_datasets=manifest,
    )
    upstream = [
        context["gate_result_objects"][name]
        for name in ("R1", "R2")
        if name in context.get("gate_result_objects", {})
    ]
    r3 = gate_records.record(
        context, r3, branch="REAL_DISTRIBUTION_POOL", upstream=upstream
    )
    _store(context, "pool_selections", selections)
    _store(context, "pool_per_dataset", per_dataset)
    return {
        "datasets": list(datasets),
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "selection_scope": "frozen validation interval only",
        "per_dataset": per_dataset,
        "macro": {
            "scrps_improvement": macro("scrps_improvement"),
            "tail_sql_improvement": macro("tail_sql_improvement"),
            "q50_deterioration": macro("q50_deterioration"),
            "zero_brier_deterioration": macro("zero_brier_deterioration"),
            "nrmse_deterioration": macro("nrmse_deterioration"),
        },
        **gate_records.payload_block([r3]),
    }


LAMBDA_GRID = (0.25, 0.50, 0.75)
STUDENT_MAX_EPOCHS = 30
STUDENT_PATIENCE = 5


def _teacher_soft_targets(
    checkpoints: Path,
    windows: TrainingWindows,
    *,
    dataset_id: str = "m5",
    cache: Path | None = None,
) -> dict[str, np.ndarray]:
    """Teacher zero mass and quantiles on one window set, in canonical head order."""
    if cache is not None and Path(cache).exists():
        stored = np.load(Path(cache))
        return {"p_zero": stored["p_zero"], "quantiles": stored["quantiles"]}
    p_zero, quantiles = [], []
    for head in HEADS:
        model, _ = load_teacher_checkpoint(Path(checkpoints) / f"teacher_{dataset_id}_{head}.pt")
        prediction = _predict(model, windows)
        p_zero.append(prediction["p_zero"])
        quantiles.append(prediction["quantiles"])
    soft = {
        "p_zero": np.stack(p_zero, axis=-1),
        "quantiles": np.stack(quantiles, axis=-2),
    }
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(cache), **soft)
    return soft


def _collapse_soft_targets(
    soft: Mapping[str, np.ndarray], weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Pre-mix a quantile-specific teacher pool into one synthetic teacher.

    The student loss only accepts one simplex weight per teacher, so a P3 path whose
    weights differ per quantile is mixed here and handed over as a single teacher whose
    quantiles are already monotone by construction.
    """
    weights = np.asarray(weights, dtype=np.float64)
    p_zero = np.asarray(soft["p_zero"], dtype=np.float64)
    quantiles = np.asarray(soft["quantiles"], dtype=np.float64)
    if weights.ndim == 1:
        mixed_p0 = np.tensordot(p_zero, weights, axes=([-1], [0]))
        mixed_q = np.tensordot(quantiles, weights, axes=([-2], [0]))
    else:
        # weights: [quantile, teacher]
        mixed_p0 = np.tensordot(p_zero, weights.mean(axis=0), axes=([-1], [0]))
        mixed_q = np.einsum("rhtq,qt->rhq", quantiles, weights)
    mixed_q = np.maximum.accumulate(mixed_q, axis=-1)
    return {
        "p_zero": mixed_p0[..., None],
        "quantiles": mixed_q[..., None, :],
    }


def _student_validation_scrps(model, windows: TrainingWindows) -> float:
    """Checkpoint score: the student's own deterministic monotone quantiles."""
    device = model.trend.weight.device.type
    history = torch.as_tensor(windows.history, dtype=torch.float32, device=device)
    scale = torch.as_tensor(windows.scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(history, scale)
        quantiles = output["evaluation_quantiles"].detach().cpu().numpy().astype(np.float64)
    crps = approximate_crps(windows.target, quantiles)
    return float(np.mean(crps / windows.scale[:, None]))


def _train_student(
    train_windows: TrainingWindows,
    validation_windows: TrainingWindows,
    *,
    seed: int,
    soft: Mapping[str, np.ndarray] | None,
    weights: Sequence[float] | np.ndarray | None,
    lambda_value: float,
) -> dict[str, Any]:
    """Fit one student variant under the same budget as every other variant."""
    from .losses import student_loss_from_sums, student_loss_sums
    from .student import build_student

    torch.manual_seed(int(seed))
    device = _device()
    model = build_student(lookback=LOOKBACK, horizon=HORIZON).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = np.random.default_rng(int(seed))

    history = torch.as_tensor(train_windows.history, dtype=torch.float32, device=device)
    target = torch.as_tensor(train_windows.target, dtype=torch.float32, device=device)
    mask = torch.as_tensor(train_windows.target_mask, dtype=torch.bool, device=device)
    scale = torch.as_tensor(train_windows.scale, dtype=torch.float32, device=device)
    teacher_p0 = teacher_q = teacher_w = None
    if soft is not None:
        teacher_p0 = torch.as_tensor(soft["p_zero"], dtype=torch.float32, device=device)
        teacher_q = torch.as_tensor(soft["quantiles"], dtype=torch.float32, device=device)
        if weights is not None:
            teacher_w = torch.as_tensor(
                np.broadcast_to(
                    np.asarray(weights, dtype=np.float64), soft["p_zero"].shape
                ).copy(),
                dtype=torch.float32,
                device=device,
            )

    best_score, best_state, best_epoch, stale = None, None, None, 0
    for epoch in range(1, STUDENT_MAX_EPOCHS + 1):
        model.train()
        order = generator.permutation(train_windows.row_count)
        for start in range(0, order.size, 256):
            rows = torch.as_tensor(order[start : start + 256], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(history[rows], scale[rows])
            sums = student_loss_sums(
                p0_student=output["p0"],
                quantiles_student=output["quantiles"],
                target=target[rows],
                target_mask=mask[rows],
                scale=scale[rows],
                teacher_p0=None if teacher_p0 is None else teacher_p0[rows],
                teacher_quantiles=None if teacher_q is None else teacher_q[rows],
                teacher_weights=None if teacher_w is None else teacher_w[rows],
            )
            loss = student_loss_from_sums(sums, lambda_soft=float(lambda_value))["loss"]
            if not bool(torch.isfinite(loss)):
                raise NumericalBranchBlocked(
                    "NUMERICAL_BRANCH_BLOCKED: student loss is nonfinite"
                )
            loss.backward()
            optimizer.step()
        if epoch % 2 == 0 or epoch == STUDENT_MAX_EPOCHS:
            score = _student_validation_scrps(model, validation_windows)
            if best_score is None or score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= STUDENT_PATIENCE:
                    break
    if best_state:
        model.load_state_dict(best_state)
    return {"model": model, "validation_sCRPS": best_score, "best_epoch": best_epoch}


def _student_metrics(model, windows: TrainingWindows) -> dict[str, float]:
    """Every frozen A3 metric on one student's deterministic output."""
    from .evaluation import pinball_loss

    device = model.trend.weight.device.type
    history = torch.as_tensor(windows.history, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(windows.scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(history, scale_tensor)
        quantiles = output["evaluation_quantiles"].detach().cpu().numpy().astype(np.float64)
        p_zero = output["p0"].detach().cpu().numpy().astype(np.float64)
    grid = np.asarray(CRPS_QUANTILE_GRID, dtype=np.float64)
    target = np.asarray(windows.target, dtype=np.float64)
    scale = np.asarray(windows.scale, dtype=np.float64)[:, None]

    def scaled_pinball(probability: float) -> float:
        index = int(np.argmin(np.abs(grid - probability)))
        return float(np.mean(pinball_loss(target, quantiles[..., index], probability) / scale))

    def coverage(level: float) -> float:
        low = quantiles[..., int(np.argmin(np.abs(grid - (0.5 - level / 2.0))))]
        high = quantiles[..., int(np.argmin(np.abs(grid - (0.5 + level / 2.0))))]
        return float(np.mean((target >= low) & (target <= high)))

    median = quantiles[..., int(np.argmin(np.abs(grid - 0.50)))]
    return {
        "sCRPS": float(np.mean(approximate_crps(target, quantiles) / scale)),
        "zero_brier": float(np.mean((p_zero - (target == 0.0).astype(np.float64)) ** 2)),
        "sQL_50": scaled_pinball(0.50),
        "sQL_95": scaled_pinball(0.95),
        "sQL_99": scaled_pinball(0.99),
        "tail_sQL": 0.5 * (scaled_pinball(0.95) + scaled_pinball(0.99)),
        "NRMSE": float(np.sqrt(np.mean(((median - target) / scale) ** 2))),
        "coverage_90": coverage(0.90),
        "coverage_error_90": abs(coverage(0.90) - 0.90),
        "coverage_95": coverage(0.95),
        "coverage_error_95": abs(coverage(0.95) - 0.95),
    }


def _student_row_scrps(model, windows: TrainingWindows) -> np.ndarray:
    """Row-level scaled CRPS so a paired cluster bootstrap can be formed."""
    device = model.trend.weight.device.type
    history = torch.as_tensor(windows.history, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(windows.scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        quantiles = (
            model(history, scale_tensor)["evaluation_quantiles"]
            .detach().cpu().numpy().astype(np.float64)
        )
    crps = approximate_crps(np.asarray(windows.target, dtype=np.float64), quantiles)
    return np.mean(crps / np.asarray(windows.scale, dtype=np.float64)[:, None], axis=1)


def _paired_student_frame(
    rows: Mapping[str, np.ndarray],
    keys: pd.DataFrame,
) -> pd.DataFrame:
    """Long frame of per-row student losses keyed for the frozen cluster bootstrap."""
    frames = []
    for method, values in rows.items():
        frame = keys.copy()
        frame["method"] = method
        frame["loss"] = np.asarray(values, dtype=np.float64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _benchmark_latency(model, windows: TrainingWindows, *, batch: int, device: str) -> float:
    """Median seconds per timed inference iteration under the frozen warmup contract."""
    import time as _time

    module = model.to(device)
    history = torch.as_tensor(
        np.asarray(windows.history[:batch], dtype=np.float64), dtype=torch.float32
    )
    scale = torch.as_tensor(
        np.asarray(windows.scale[:batch], dtype=np.float64), dtype=torch.float32
    )
    if history.shape[0] < batch:
        repeats = int(np.ceil(batch / max(history.shape[0], 1)))
        history = history.repeat(repeats, 1)[:batch]
        scale = scale.repeat(repeats)[:batch]
    history, scale = history.to(device), scale.to(device)
    with torch.no_grad():
        for _ in range(5):
            module(history, scale)
        if device == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(20):
            started = _time.perf_counter()
            module(history, scale)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(_time.perf_counter() - started)
    return float(np.median(timings))


def _teacher_pool_latency(
    checkpoints: Path, dataset_id: str, windows: TrainingWindows, *, batch: int, device: str
) -> tuple[float, float, int]:
    """Pool latency, cheapest single-teacher latency, and the smallest parameter count."""
    singles, parameters = [], []
    for head in HEADS:
        model, payload = load_teacher_checkpoint(
            Path(checkpoints) / f"teacher_{dataset_id}_{head}.pt"
        )
        singles.append(_benchmark_latency(model, windows, batch=batch, device=device))
        parameters.append(int(payload["parameter_count"]))
    return float(sum(singles)), float(min(singles)), int(min(parameters))


def stage_a_distillation(context: dict[str, Any]) -> dict[str, Any]:
    """A0-A4 students per dataset; A1-A4 decided by the frozen reducers."""
    from . import gate_records
    from . import gates as gate_module
    from .bootstrap import paired_cluster_bootstrap

    tier = _tier(context)
    seed = int(tier["student_model_seeds"][0])
    checkpoints = _load(context, "real_checkpoint_dir")
    if checkpoints is None:
        attempt = _completed_attempt(context["runs_root"], "stage_r1_real_teacher_training")
        checkpoints = None if attempt is None else str(attempt)
    selections = _load(context, "pool_selections")
    pool_per_dataset = _load(context, "pool_per_dataset")
    if selections is None or pool_per_dataset is None:
        sealed = _sealed_payload(context, "cdf_pool")
        pool_per_dataset = dict(sealed.get("per_dataset", {}))
    if checkpoints is None or not pool_per_dataset:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: Stage A needs sealed teachers and a sealed pool"
        )
    manifest = _primary_manifest(context)
    coverage = _head_coverage(context)
    datasets = coverage["complete"]
    if not datasets:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: Stage A needs a dataset with all three families"
        )
    cache_root = Path(context["runs_root"]) / "_cache_v2"

    per_dataset: dict[str, Any] = {}
    bootstrap_rows: list[pd.DataFrame] = []
    for dataset_id in datasets:
        split = data_module.REAL_SPLITS[dataset_id]
        train_origins = tuple(range(LOOKBACK, split.train[1] - HORIZON + 1, STUDENT_STRIDE))
        train_windows, _ = _role_windows(
            context, "student_train", train_origins, dataset_id=dataset_id
        )
        validation_windows, _ = _role_windows(
            context, "student_validation", (split.validation[0],), dataset_id=dataset_id
        )
        outer_windows, outer_batch = _role_windows(
            context, "evaluation", split.origins, dataset_id=dataset_id
        )
        soft = _teacher_soft_targets(
            Path(checkpoints),
            train_windows,
            dataset_id=dataset_id,
            cache=cache_root / f"stage_a_soft_{dataset_id}.npz",
        )

        record = pool_per_dataset[dataset_id]
        p0_head = str(record["P0_best_single_teacher"])
        weight_sets = {
            "A1": np.asarray([1.0 if head == p0_head else 0.0 for head in HEADS]),
            "A2": np.asarray(record["P1_equal_weights"], dtype=np.float64),
            "A3": np.asarray(record["P2_weights"], dtype=np.float64),
        }
        p3_weights = None
        if selections is not None:
            path = selections[dataset_id]["P3"].get("weights_by_quantile")
            if path is not None:
                p3_weights = np.asarray(path, dtype=np.float64)

        fitted: dict[str, Any] = {}
        fitted["A0"] = {
            "lambda": 0.0,
            **_train_student(
                train_windows, validation_windows, seed=seed,
                soft=None, weights=None, lambda_value=0.0,
            ),
        }
        variants = dict(weight_sets)
        if p3_weights is not None:
            variants["A4"] = p3_weights
        for name, weights in variants.items():
            collapsed = _collapse_soft_targets(soft, weights)
            best = None
            for lambda_value in LAMBDA_GRID:
                fit = _train_student(
                    train_windows, validation_windows, seed=seed,
                    soft=collapsed, weights=None, lambda_value=lambda_value,
                )
                if best is None or fit["validation_sCRPS"] < best["validation_sCRPS"]:
                    best = {**fit, "lambda": lambda_value}
            fitted[name] = best

        # The frozen order: minimum validation sCRPS, then validation tail sQL, then A3.
        candidates = [name for name in ("A3", "A4") if name in fitted]
        validation_metrics = {
            name: _student_metrics(fitted[name]["model"], validation_windows)
            for name in candidates
        }
        primary = min(
            candidates,
            key=lambda name: (
                validation_metrics[name]["sCRPS"],
                validation_metrics[name]["tail_sQL"],
                0 if name == "A3" else 1,
            ),
        )
        outer_metrics = {
            name: _student_metrics(entry["model"], outer_windows)
            for name, entry in fitted.items()
        }
        row_keys = (
            outer_batch.key_frame.loc[outer_batch.key_frame["step"] == 0]
            .reset_index(drop=True)[["dataset_id", "series_id", "origin"]]
        )
        bootstrap_rows.append(
            _paired_student_frame(
                {
                    "A0": _student_row_scrps(fitted["A0"]["model"], outer_windows),
                    "PRIMARY": _student_row_scrps(fitted[primary]["model"], outer_windows),
                },
                row_keys,
            )
        )
        per_dataset[dataset_id] = {
            "primary_student": primary,
            "selected_lambda": {name: entry.get("lambda") for name, entry in fitted.items()},
            "validation_metrics": validation_metrics,
            "outer_metrics": outer_metrics,
            "student_train_rows": int(train_windows.row_count),
            "outer_rows": int(outer_windows.row_count),
            "student_parameters": int(
                sum(p.numel() for p in fitted[primary]["model"].parameters())
            ),
        }
        _store(context, f"a_primary_model_{dataset_id}", fitted[primary]["model"])
        _store(context, f"a_outer_windows_{dataset_id}", outer_windows)

    if not coverage["manifest_complete"]:
        return {
            "datasets": list(datasets),
            "primary_dataset_manifest_sha256": manifest.audit_sha256,
            "lambda_grid": list(LAMBDA_GRID),
            "per_dataset": per_dataset,
            "A1_status": "NOT_EVALUATED",
            "A2_status": "NOT_EVALUATED",
            "A3_status": "NOT_EVALUATED",
            "A4_status": "NOT_EVALUATED",
            "not_evaluated_reason": _blocked_gate_note(coverage, "A1-A4"),
            "numerically_blocked": coverage["incomplete"],
            "scientific_role": "DIAGNOSTIC_ONLY_NUMERICAL_BRANCH_BLOCKED",
            "confirmatory_eligible": False,
        }

    # --- A1: the primary teacher pool's own value on the outer rows ----------
    a1 = gate_module.gate_a1(
        scrps_improvement=float(
            np.mean([pool_per_dataset[d]["scrps_improvement"] for d in datasets])
        ),
        tail_sql_improvement=float(
            np.mean([pool_per_dataset[d]["tail_sql_improvement"] for d in datasets])
        ),
        primary_datasets=manifest,
    )
    upstream = [
        context["gate_result_objects"][name]
        for name in ("R1", "R2", "R3")
        if name in context.get("gate_result_objects", {})
    ]
    a1 = gate_records.record(context, a1, branch="A_DISTILLATION", upstream=upstream)

    # --- A2: distillation recovery with its frozen cluster bootstrap ---------
    bootstrap_frame = pd.concat(bootstrap_rows, ignore_index=True)
    tier_name = str(context["runtime_decision"]["runtime_tier"])
    effect = paired_cluster_bootstrap(
        bootstrap_frame,
        baseline="A0",
        candidate="PRIMARY",
        tier=tier_name,
        observation_columns=["origin"],
    )
    # The macro draw gives one interval for the equal-weight average; the frozen gate
    # also needs each dataset's own interval, which only a per-dataset draw provides.
    dataset_effect = {
        dataset_id: paired_cluster_bootstrap(
            bootstrap_frame.loc[bootstrap_frame["dataset_id"] == dataset_id],
            baseline="A0",
            candidate="PRIMARY",
            tier=tier_name,
            observation_columns=["origin"],
        )
        for dataset_id in datasets
    }
    best_single = float(
        np.mean([
            pool_per_dataset[d]["outer_head_metrics"][pool_per_dataset[d]["outer_best_single_head"]]["sCRPS"]
            for d in datasets
        ])
    )
    pool_loss = float(np.mean([pool_per_dataset[d]["outer_pool_metrics"]["sCRPS"] for d in datasets]))
    distilled = float(
        np.mean([
            per_dataset[d]["outer_metrics"][per_dataset[d]["primary_student"]]["sCRPS"]
            for d in datasets
        ])
    )
    a0_loss = float(np.mean([per_dataset[d]["outer_metrics"]["A0"]["sCRPS"] for d in datasets]))
    a2 = gate_module.gate_a2(
        best_single_loss=best_single,
        distilled_loss=distilled,
        pool_loss=pool_loss,
        improvement_vs_a0=float(1.0 - distilled / a0_loss),
        macro_ci_lower=float(effect["ci_lower"]),
        dataset_effects={d: float(effect["dataset_effects"][d]) for d in datasets},
        dataset_ci_lowers={d: float(dataset_effect[d]["ci_lower"]) for d in datasets},
        primary_datasets=manifest,
    )
    a2 = gate_records.record(context, a2, branch="A_DISTILLATION", upstream=[*upstream, a1])

    # --- A3: the distribution trade-off of the primary student ---------------
    def macro_deterioration(metric: str) -> float:
        values = []
        for dataset_id in datasets:
            primary = per_dataset[dataset_id]["primary_student"]
            student = per_dataset[dataset_id]["outer_metrics"][primary][metric]
            teacher = pool_per_dataset[dataset_id]["outer_pool_metrics"].get(metric)
            if teacher is None:
                continue
            values.append(student / teacher - 1.0)
        return float(np.mean(values)) if values else 0.0

    def macro_coverage_error(level: int) -> float:
        return float(
            np.mean([
                per_dataset[d]["outer_metrics"][per_dataset[d]["primary_student"]][
                    f"coverage_error_{level}"
                ]
                for d in datasets
            ])
        )

    a3 = gate_module.gate_a3(
        {
            "zero_brier": macro_deterioration("zero_brier"),
            "q50": macro_deterioration("sQL_50"),
            "q99": macro_deterioration("sQL_99"),
            "NRMSE": macro_deterioration("NRMSE"),
            "coverage_90": macro_coverage_error(90),
            "coverage_95": macro_coverage_error(95),
        },
        primary_datasets=manifest,
    )
    a3 = gate_records.record(context, a3, branch="A_DISTILLATION", upstream=[*upstream, a1, a2])

    # --- A4: compression and runtime value -----------------------------------
    reference = datasets[0]
    student_model = _load(context, f"a_primary_model_{reference}")
    bench_windows = _load(context, f"a_outer_windows_{reference}")
    cuda_available = bool(torch.cuda.is_available())
    devices = ["cpu"] + (["cuda"] if cuda_available else [])
    latency: dict[str, dict[str, dict[str, float]]] = {}
    for device in devices:
        latency[device] = {}
        for batch in (1, 256):
            pool_seconds, single_seconds, smallest = _teacher_pool_latency(
                Path(checkpoints), reference, bench_windows, batch=batch, device=device
            )
            latency[device][str(batch)] = {
                "student": _benchmark_latency(
                    student_model, bench_windows, batch=batch, device=device
                ),
                "single": single_seconds,
                "pool": pool_seconds,
            }
    _, _, smallest_teacher = _teacher_pool_latency(
        Path(checkpoints), reference, bench_windows, batch=1, device="cpu"
    )
    a4 = gate_module.gate_a4(
        student_parameters=int(per_dataset[reference]["student_parameters"]),
        smallest_teacher_parameters=int(smallest_teacher),
        latency_by_device_batch=latency,
        cuda_peak_memory_by_batch=None,
        cuda_available=cuda_available,
        primary_datasets=manifest,
    )
    a4 = gate_records.record(
        context, a4, branch="A_DISTILLATION", upstream=[*upstream, a1, a2, a3]
    )

    _store(context, "a_effect", effect)
    _store(context, "a_per_dataset", per_dataset)
    return {
        "datasets": list(datasets),
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "lambda_grid": list(LAMBDA_GRID),
        "per_dataset": per_dataset,
        "macro_best_single_teacher_sCRPS": best_single,
        "macro_teacher_pool_sCRPS": pool_loss,
        "macro_distilled_student_sCRPS": distilled,
        "macro_A0_sCRPS": a0_loss,
        "distillation_bootstrap": {
            "macro_point": effect["point"],
            "macro_ci_lower": effect["ci_lower"],
            "macro_ci_upper": effect["ci_upper"],
            "dataset_effects": effect["dataset_effects"],
            "dataset_ci_lower": {d: dataset_effect[d]["ci_lower"] for d in datasets},
            "dataset_ci_upper": {d: dataset_effect[d]["ci_upper"] for d in datasets},
            "draws": effect["draws"],
            "seed": effect["seed"],
        },
        "latency_by_device_batch": latency,
        "cuda_available": cuda_available,
        **gate_records.payload_block([a1, a2, a3, a4]),
    }


B_KEY_COLUMNS = ("dataset_id", "series_id", "origin")


def _module_sha256(name: str) -> str:
    path = Path(__file__).resolve().parent / f"{name}.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_manifest(context: dict[str, Any], dataset_id: str, checkpoints: Path) -> dict[str, Any]:
    """Bind a cache generation to everything that could change what it stores.

    The first cache generation had no such binding, so a stale file written before the
    feature-boundary fix could silently reintroduce future information. This manifest
    makes reuse impossible unless every source really is identical.
    """
    entry = _dataset_entry(context, dataset_id)
    store = _real_store(context)
    checkpoint_digests = {
        head: hashlib.sha256(
            (Path(checkpoints) / f"teacher_{dataset_id}_{head}.pt").read_bytes()
        ).hexdigest()
        for head in HEADS
    }
    return {
        "schema": "prob_head_structure_full_v1.stage_b_cache.v2",
        "dataset_id": str(dataset_id),
        "preregistration_payload_sha256": _preregistration_sha256(context),
        "source_manifest_aggregate_sha256": store["source_manifest"]["aggregate_sha256"],
        "dataset_audit_sha256": entry["dataset_audit"]["audit_sha256"],
        "sample_manifest_sha256": entry["sample_manifest"]["manifest_sha256"],
        "primary_dataset_manifest_sha256": store["manifest"].audit_sha256,
        "teacher_checkpoint_sha256": checkpoint_digests,
        "temporal_features_sha256": _module_sha256("temporal_features"),
        "routing_sha256": _module_sha256("routing"),
        "stages_sha256": _module_sha256("stages"),
        "feature_contract_version": 2,
    }


def _cache_paths(context: dict[str, Any], dataset_id: str, kind: str) -> tuple[Path, Path]:
    root = Path(context["runs_root"]) / "_cache_v2"
    return root / f"{kind}_{dataset_id}.npz", root / f"{kind}_{dataset_id}.manifest.json"


def _inner_origin_panel(
    context: dict[str, Any], dataset_id: str, checkpoints: Path
) -> dict[str, Any]:
    """Teacher losses and origin-bounded features at every inner routing origin.

    Rows carry their explicit (dataset_id, series_id, origin) key, so nothing downstream
    ever aligns two tables by position.
    """
    from .temporal_features import (
        BASELINE_FEATURE_NAMES,
        TEMPORAL_FEATURE_NAMES,
        build_feature_matrix,
        temporal_features_for_series,
        train_descriptors_for_series,
    )

    manifest = _cache_manifest(context, dataset_id, checkpoints)
    cache_path, manifest_path = _cache_paths(context, dataset_id, "stage_b_inner_panel")
    if cache_path.exists() and manifest_path.exists():
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored_manifest == manifest:
            stored = np.load(cache_path, allow_pickle=True)
            return {
                "origins": [int(value) for value in stored["origins"]],
                "series_id": [str(value) for value in stored["series_id"]],
                "excluded_late_availability": int(stored["excluded_late_availability"]),
                "losses": stored["losses"],
                "baseline": stored["baseline"],
                "temporal": stored["temporal"],
                "baseline_names": [str(name) for name in stored["baseline_names"]],
                "temporal_names": [str(name) for name in stored["temporal_names"]],
                "manifest": manifest,
            }

    entry = _dataset_entry(context, dataset_id)
    split = data_module.REAL_SPLITS[dataset_id]
    panel = entry["panel"]
    values = np.asarray(panel["y"], dtype=np.float64)
    available = np.asarray(panel["available_from"], dtype=np.int64)
    series_ids = np.asarray(panel["series_id"]).astype(str)
    selected = [str(value) for value in entry["sample_manifest"]["selected_series_ids"]]
    positions = {sid: index for index, sid in enumerate(series_ids)}
    index = np.asarray([positions[sid] for sid in selected], dtype=np.int64)

    origins = select_inner_origins(
        lookback=LOOKBACK, horizon=HORIZON, model_train_end=split.train[1]
    )
    # A series that only becomes available after the earliest inner origin has no
    # origin-bounded history there, so it cannot be routed at every fold. Restricting
    # the panel keeps the cross-fit rectangular instead of imputing a feature.
    earliest = int(min(origins)) - LOOKBACK
    keep = [
        position for position, sid in enumerate(selected)
        if int(available[index[position]]) <= earliest
    ]
    excluded_late_availability = len(selected) - len(keep)
    selected = [selected[position] for position in keep]
    index = index[np.asarray(keep, dtype=np.int64)]
    if not len(selected):
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: no sampled series is available before the "
            "earliest inner routing origin"
        )

    losses, baseline_rows, temporal_rows = [], [], []
    for origin in origins:
        windows, _ = _windows_from_panel(
            values, available, index, [origin], train_end=split.train[1]
        )
        per_head = []
        for head in HEADS:
            model, _ = load_teacher_checkpoint(
                Path(checkpoints) / f"teacher_{dataset_id}_{head}.pt"
            )
            per_head.append(_scrps_rows(_predict(model, windows), windows))
        losses.append(np.stack(per_head, axis=-1))
        # Inner origins sit inside model_train, so descriptors must be recomputed on the
        # origin-bounded prefix [available_from, o). Using model_train_end here would let
        # observations the router cannot yet see leak into its own features.
        baseline_rows.append(
            build_feature_matrix(
                [
                    train_descriptors_for_series(
                        values[position],
                        available_from=int(available[position]),
                        train_end=int(origin),
                    )
                    for position in index
                ],
                feature_set="baseline",
            )[0]
        )
        temporal_rows.append(
            build_feature_matrix(
                [
                    temporal_features_for_series(
                        values[position],
                        origin=int(origin),
                        available_from=int(available[position]),
                        train_end=int(origin),
                        dataset_id=str(dataset_id),
                    )
                    for position in index
                ],
                feature_set="temporal",
            )[0]
        )

    result = {
        "origins": [int(value) for value in origins],
        "series_id": selected,
        "excluded_late_availability": int(excluded_late_availability),
        "losses": np.stack(losses),
        "baseline": np.stack(baseline_rows),
        "temporal": np.stack(temporal_rows),
        "baseline_names": list(BASELINE_FEATURE_NAMES),
        "temporal_names": list(TEMPORAL_FEATURE_NAMES),
        "manifest": manifest,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{key: np.asarray(value) for key, value in result.items() if key != "manifest"},
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _router_key_frame(panel: Mapping[str, Any], dataset_id: str) -> pd.DataFrame:
    """One explicit key row per (dataset, series, origin) in origin-major order."""
    origins = list(panel["origins"])
    series = list(panel["series_id"])
    return pd.DataFrame(
        {
            "dataset_id": str(dataset_id),
            "series_id": [sid for _ in origins for sid in series],
            "origin": [origin for origin in origins for _ in series],
        }
    )


def _assert_exact_join(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> None:
    """Refuse any positional coincidence: the keys must match exactly, as sets."""
    columns = list(B_KEY_COLUMNS)
    for frame, side in ((left, "left"), (right, "right")):
        if frame.duplicated(columns).any():
            raise HardIntegrityFailure(f"{label}: duplicate {side} routing key")
    left_keys = set(map(tuple, left[columns].to_numpy().tolist()))
    right_keys = set(map(tuple, right[columns].to_numpy().tolist()))
    if left_keys != right_keys:
        raise HardIntegrityFailure(
            f"{label}: routing keys differ "
            f"(missing={len(right_keys - left_keys)}, extra={len(left_keys - right_keys)})"
        )


def _crossfit_router(panel: Mapping[str, Any], feature_set: str) -> dict[str, Any]:
    matrix = panel["baseline"] if feature_set == "baseline" else panel["temporal"]
    names = panel["baseline_names"] if feature_set == "baseline" else panel["temporal_names"]
    records = [
        {
            "origin": panel["origins"][index],
            "features": matrix[index],
            "losses": panel["losses"][index],
        }
        for index in range(len(panel["origins"]))
    ]
    return expanding_crossfit_weights(records, feature_names=list(names))


def _router_frame(folds: Sequence[Mapping[str, Any]], panel: Mapping[str, Any], dataset_id: str) -> pd.DataFrame:
    """Router output as records keyed by dataset/series/origin, never by position."""
    series = list(panel["series_id"])
    rows = []
    for fold in folds:
        weights = np.asarray(fold["weights"], dtype=np.float64)
        regret = np.asarray(fold["regret"], dtype=np.float64)
        if weights.shape[0] != len(series):
            raise HardIntegrityFailure(
                "router fold does not cover the sampled series exactly once"
            )
        for position, series_id in enumerate(series):
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "series_id": str(series_id),
                    "origin": int(fold["origin"]),
                    **{
                        f"w_{head}": float(weights[position, head_index])
                        for head_index, head in enumerate(HEADS)
                    },
                    **{
                        f"regret_{head}": float(regret[position, head_index])
                        for head_index, head in enumerate(HEADS)
                    },
                }
            )
    return pd.DataFrame(rows)


def stage_b_regret(context: dict[str, Any]) -> dict[str, Any]:
    """B1: can origin-bounded structure predict which teacher will be least wrong?"""
    from . import gate_records
    from . import gates as gate_module
    from .controls import regret_label_shuffle

    checkpoints = _load(context, "real_checkpoint_dir")
    if checkpoints is None:
        attempt = _completed_attempt(context["runs_root"], "stage_r1_real_teacher_training")
        checkpoints = None if attempt is None else str(attempt)
    if checkpoints is None:
        raise StageInputUnavailable("STAGE_INPUT_UNAVAILABLE: Stage B needs sealed teachers")

    manifest = _primary_manifest(context)
    coverage = _head_coverage(context)
    datasets = coverage["complete"]
    if not datasets:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: Stage B needs a dataset with all three families"
        )
    spearman: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    increments: dict[str, float] = {}
    shuffled_increments: dict[str, float] = {}
    detail: dict[str, Any] = {}

    for dataset_id in datasets:
        panel = _inner_origin_panel(context, dataset_id, Path(checkpoints))
        _store(context, f"b_inner_panel_{dataset_id}", panel)
        keys = _router_key_frame(panel, dataset_id)
        outcomes: dict[str, Any] = {}
        for feature_set in ("baseline", "extended"):
            folds = _crossfit_router(
                panel, "baseline" if feature_set == "baseline" else "temporal"
            )["folds"]
            frame = _router_frame(folds, panel, dataset_id)
            heldout_keys = keys.loc[
                keys["origin"].isin(sorted({int(fold["origin"]) for fold in folds}))
            ].reset_index(drop=True)
            _assert_exact_join(frame[list(B_KEY_COLUMNS)], heldout_keys, label=f"B1 {feature_set}")
            weights = frame[[f"w_{head}" for head in HEADS]].to_numpy(dtype=np.float64)
            regret = frame[[f"regret_{head}" for head in HEADS]].to_numpy(dtype=np.float64)
            try:
                statistic = regret_spearman(weights, regret)
            except Exception as error:
                outcomes[feature_set] = {"status": "INSUFFICIENT_VARIATION", "reason": str(error)}
                continue
            shuffled = regret_spearman(
                weights, regret_label_shuffle(regret, scope=(dataset_id, feature_set))
            )
            outcomes[feature_set] = {
                "regret_spearman": float(statistic),
                "shuffled_regret_spearman": float(shuffled),
                "rows": int(len(frame)),
                "folds": [int(fold["origin"]) for fold in folds],
                "temperatures": [float(fold["temperature"]) for fold in folds],
            }
            _store(context, f"b_router_{dataset_id}_{feature_set}", frame)

        detail[dataset_id] = outcomes
        if any("regret_spearman" not in value for value in outcomes.values()):
            spearman[dataset_id] = None
            reasons[dataset_id] = "INSUFFICIENT_VARIATION"
            increments[dataset_id] = 0.0
            shuffled_increments[dataset_id] = 0.0
            continue
        spearman[dataset_id] = outcomes["extended"]["regret_spearman"]
        increments[dataset_id] = float(
            outcomes["extended"]["regret_spearman"] - outcomes["baseline"]["regret_spearman"]
        )
        shuffled_increments[dataset_id] = float(
            outcomes["extended"]["shuffled_regret_spearman"]
            - outcomes["baseline"]["shuffled_regret_spearman"]
        )

    real_increment = float(np.mean([increments[d] for d in datasets]))
    shuffled_increment = float(np.mean([shuffled_increments[d] for d in datasets]))
    if not coverage["manifest_complete"]:
        return {
            "datasets": list(datasets),
            "primary_dataset_manifest_sha256": manifest.audit_sha256,
            "per_dataset": detail,
            "regret_spearman_by_dataset": {d: spearman[d] for d in datasets},
            "extended_minus_baseline": real_increment,
            "shuffled_increment": shuffled_increment,
            "cache_generation": "_cache_v2",
            "B1_status": "NOT_EVALUATED",
            "not_evaluated_reason": _blocked_gate_note(coverage, "B1"),
            "numerically_blocked": coverage["incomplete"],
            "scientific_role": "DIAGNOSTIC_ONLY_NUMERICAL_BRANCH_BLOCKED",
            "confirmatory_eligible": False,
        }
    b1 = gate_module.gate_b1(
        regret_spearman_by_dataset={d: spearman[d] for d in datasets},
        undefined_reasons_by_dataset={d: reasons[d] for d in datasets if d in reasons},
        extended_minus_baseline=real_increment,
        real_increment=real_increment,
        shuffled_increment=shuffled_increment,
        cross_dataset_effects={d: increments[d] for d in datasets},
        primary_datasets=manifest,
    )
    upstream = [
        context["gate_result_objects"][name]
        for name in ("R1", "R2")
        if name in context.get("gate_result_objects", {})
    ]
    b1 = gate_records.record(context, b1, branch="B_STRUCTURE_ROUTING", upstream=upstream)
    return {
        "datasets": list(datasets),
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "per_dataset": detail,
        "regret_spearman_by_dataset": {d: spearman[d] for d in datasets},
        "extended_minus_baseline": real_increment,
        "shuffled_increment": shuffled_increment,
        "cache_generation": "_cache_v2",
        **gate_records.payload_block([b1]),
    }


def stage_b_structure(context: dict[str, Any]) -> dict[str, Any]:
    """B0/B1/B2 students on identical heldout rows, joined by exact routing keys."""
    from . import gate_records
    from . import gates as gate_module
    from .bootstrap import paired_cluster_bootstrap

    tier = _tier(context)
    seed = int(tier["student_model_seeds"][0])
    checkpoints = _load(context, "real_checkpoint_dir")
    if checkpoints is None:
        attempt = _completed_attempt(context["runs_root"], "stage_r1_real_teacher_training")
        checkpoints = None if attempt is None else str(attempt)
    pool_per_dataset = _load(context, "pool_per_dataset") or dict(
        _sealed_payload(context, "cdf_pool").get("per_dataset", {})
    )
    if checkpoints is None or not pool_per_dataset:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: Stage B students need sealed teachers and the pool"
        )
    manifest = _primary_manifest(context)
    coverage = _head_coverage(context)
    datasets = coverage["complete"]
    if not datasets:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: Stage B students need all three families"
        )
    cache_root = Path(context["runs_root"]) / "_cache_v2"

    per_dataset: dict[str, Any] = {}
    bootstrap_rows: list[pd.DataFrame] = []
    for dataset_id in datasets:
        panel = _load(context, f"b_inner_panel_{dataset_id}")
        if panel is None:
            panel = _inner_origin_panel(context, dataset_id, Path(checkpoints))
        split = data_module.REAL_SPLITS[dataset_id]
        heldout_origins = list(panel["origins"])[1:]
        entry = _dataset_entry(context, dataset_id)
        values = np.asarray(entry["panel"]["y"], dtype=np.float64)
        available = np.asarray(entry["panel"]["available_from"], dtype=np.int64)
        series_ids = np.asarray(entry["panel"]["series_id"]).astype(str)
        selected = [str(v) for v in panel["series_id"]]
        positions = {sid: i for i, sid in enumerate(series_ids)}
        index = np.asarray([positions[sid] for sid in selected], dtype=np.int64)

        windows, window_keys = _windows_from_panel(
            values, available, index, heldout_origins, train_end=split.train[1]
        )
        student_keys = pd.DataFrame(
            {
                "dataset_id": str(dataset_id),
                "series_id": [str(series_ids[int(pos)]) for pos in window_keys["series_position"]],
                "origin": window_keys["origin"].to_numpy(dtype=np.int64),
            }
        )
        soft = _teacher_soft_targets(
            Path(checkpoints),
            windows,
            dataset_id=dataset_id,
            cache=cache_root / f"stage_b_heldout_soft_{dataset_id}.npz",
        )
        validation_windows, _ = _role_windows(
            context, "student_validation", (split.validation[0],), dataset_id=dataset_id
        )
        outer_windows, outer_batch = _role_windows(
            context, "evaluation", split.origins, dataset_id=dataset_id
        )

        variants: dict[str, np.ndarray] = {
            "B0": np.broadcast_to(
                np.asarray(pool_per_dataset[dataset_id]["P2_weights"], dtype=np.float64),
                (len(student_keys), len(HEADS)),
            ).copy()
        }
        for label, name in (("baseline", "B1"), ("extended", "B2")):
            frame = _load(context, f"b_router_{dataset_id}_{label}")
            if frame is None:
                continue
            joined = student_keys.merge(
                frame, on=list(B_KEY_COLUMNS), how="left", validate="many_to_one"
            )
            if joined[[f"w_{head}" for head in HEADS]].isna().any().any():
                raise HardIntegrityFailure(
                    f"B student rows for {dataset_id} have no router weight at some key"
                )
            if len(joined) != len(student_keys):
                raise HardIntegrityFailure("B exact key join changed the row count")
            variants[name] = joined[[f"w_{head}" for head in HEADS]].to_numpy(dtype=np.float64)

        fitted: dict[str, Any] = {}
        for name, weights in variants.items():
            per_row = np.repeat(weights[:, None, :], HORIZON, axis=1)
            best = None
            for lambda_value in LAMBDA_GRID:
                fit = _train_student(
                    windows, validation_windows, seed=seed,
                    soft=soft, weights=per_row, lambda_value=lambda_value,
                )
                if best is None or fit["validation_sCRPS"] < best["validation_sCRPS"]:
                    best = {**fit, "lambda": lambda_value}
            fitted[name] = best

        outer_metrics = {
            name: _student_metrics(entry_["model"], outer_windows)
            for name, entry_ in fitted.items()
        }
        row_keys = (
            outer_batch.key_frame.loc[outer_batch.key_frame["step"] == 0]
            .reset_index(drop=True)[["dataset_id", "series_id", "origin"]]
        )
        if "B2" in fitted:
            bootstrap_rows.append(
                _paired_student_frame(
                    {
                        "B0": _student_row_scrps(fitted["B0"]["model"], outer_windows),
                        "B2": _student_row_scrps(fitted["B2"]["model"], outer_windows),
                    },
                    row_keys,
                )
            )
        worst_origin = None
        if "B2" in fitted:
            b0_rows = _student_row_scrps(fitted["B0"]["model"], outer_windows)
            b2_rows = _student_row_scrps(fitted["B2"]["model"], outer_windows)
            by_origin = (
                pd.DataFrame({"origin": row_keys["origin"], "b0": b0_rows, "b2": b2_rows})
                .groupby("origin")[["b0", "b2"]].mean()
            )
            worst_origin = float((1.0 - by_origin["b2"] / by_origin["b0"]).min())
        per_dataset[dataset_id] = {
            "student_rows": int(len(student_keys)),
            "heldout_origins": [int(value) for value in heldout_origins],
            "selected_lambda": {name: entry_["lambda"] for name, entry_ in fitted.items()},
            "outer_metrics": outer_metrics,
            "worst_origin_improvement": worst_origin,
            "improvement_over_B0": (
                float(1.0 - outer_metrics["B2"]["sCRPS"] / outer_metrics["B0"]["sCRPS"])
                if "B2" in outer_metrics
                else None
            ),
            "improvement_over_B1": (
                float(1.0 - outer_metrics["B2"]["sCRPS"] / outer_metrics["B1"]["sCRPS"])
                if "B2" in outer_metrics and "B1" in outer_metrics
                else None
            ),
        }

    if not bootstrap_rows:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: no structure-conditioned student could be fitted"
        )
    frame = pd.concat(bootstrap_rows, ignore_index=True)
    tier_name = str(context["runtime_decision"]["runtime_tier"])
    effect = paired_cluster_bootstrap(
        frame, baseline="B0", candidate="B2", tier=tier_name, observation_columns=["origin"]
    )

    def macro(field: str) -> float:
        values = [per_dataset[d][field] for d in datasets if per_dataset[d].get(field) is not None]
        return float(np.mean(values)) if values else 0.0

    def macro_deterioration(metric: str) -> float:
        values = []
        for dataset_id in datasets:
            metrics = per_dataset[dataset_id]["outer_metrics"]
            if "B2" in metrics and "B0" in metrics:
                values.append(metrics["B2"][metric] / metrics["B0"][metric] - 1.0)
        return float(np.mean(values)) if values else 0.0

    if not coverage["manifest_complete"]:
        return {
            "datasets": list(datasets),
            "primary_dataset_manifest_sha256": manifest.audit_sha256,
            "per_dataset": per_dataset,
            "B2_status": "NOT_EVALUATED",
            "not_evaluated_reason": _blocked_gate_note(coverage, "B2"),
            "numerically_blocked": coverage["incomplete"],
            "join_contract": "exact dataset_id/series_id/origin merge; positional concat forbidden",
            "scientific_role": "DIAGNOSTIC_ONLY_NUMERICAL_BRANCH_BLOCKED",
            "confirmatory_eligible": False,
        }

    b2 = gate_module.gate_b2(
        macro_scrps_improvement=macro("improvement_over_B0"),
        dataset_scrps_improvements={
            d: float(per_dataset[d]["improvement_over_B0"] or 0.0) for d in datasets
        },
        macro_ci_lower=float(effect["ci_lower"]),
        improvement_over_b1=macro("improvement_over_B1"),
        q99_deterioration=macro_deterioration("sQL_99"),
        zero_brier_deterioration=macro_deterioration("zero_brier"),
        worst_origin_improvement=macro("worst_origin_improvement"),
        primary_datasets=manifest,
    )
    upstream = [
        context["gate_result_objects"][name]
        for name in ("R1", "R2", "B1")
        if name in context.get("gate_result_objects", {})
    ]
    b2 = gate_records.record(context, b2, branch="B_STRUCTURE_ROUTING", upstream=upstream)
    _store(context, "b_effect", effect)
    _store(context, "b_per_dataset", per_dataset)
    return {
        "datasets": list(datasets),
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "per_dataset": per_dataset,
        "structure_bootstrap": {
            "macro_point": effect["point"],
            "macro_ci_lower": effect["ci_lower"],
            "macro_ci_upper": effect["ci_upper"],
            "dataset_effects": effect["dataset_effects"],
            "draws": effect["draws"],
            "seed": effect["seed"],
        },
        "join_contract": "exact dataset_id/series_id/origin merge; positional concat forbidden",
        **gate_records.payload_block([b2]),
    }


CSYN_ORIGINS = (
    120, 148, 176, 204, 232, 260, 288, 316, 344, 372, 400, 428, 456, 484, 512, 540,
)
CSYN_ONSET = 288


def _teacher_component_rows(
    checkpoints: Path,
    scope: str,
    values: np.ndarray,
    available: np.ndarray,
    index: Sequence[int],
    origins: Sequence[int],
    *,
    train_end: int,
) -> dict[int, list[dict[str, float]]]:
    """Component-wise teacher disagreement for every series at every origin."""
    per_origin: dict[int, list[dict[str, float]]] = {}
    models = {
        head: load_teacher_checkpoint(Path(checkpoints) / f"teacher_{scope}_{head}.pt")[0]
        for head in HEADS
    }
    for origin in origins:
        windows, keys = _windows_from_panel(
            values, available, index, [int(origin)], train_end=train_end
        )
        predictions = {head: _predict(model, windows) for head, model in models.items()}
        rows: list[dict[str, float]] = []
        for position in range(windows.row_count):
            rows.append(
                disagreement_components(
                    p_zero=np.stack([predictions[h]["p_zero"][position] for h in HEADS]),
                    quantiles=np.stack([predictions[h]["quantiles"][position] for h in HEADS]),
                    predictive_mean=np.stack([predictions[h]["mean"][position] for h in HEADS]),
                    scale=float(windows.scale[position]),
                )
            )
        per_origin[int(origin)] = rows
    return per_origin


def _binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """AUROC and average precision without importing a classifier dependency."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    if labels.min() == labels.max():
        return {"auroc": None, "auprc": None, "reason": "SINGLE_CLASS"}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "reason": None,
    }


def stage_csyn(context: dict[str, Any]) -> dict[str, Any]:
    """Score the known-change panels with real teacher disagreement and decide C2."""
    from . import gate_records
    from . import gates as gate_module
    from .sensor import disagreement_deltas, scalar_total_disagreement

    tier = _tier(context)
    per_cell = int(tier["synthetic_series_per_cell"])
    data_seed = int(tier["synthetic_data_seeds"][0])
    attempt = _completed_attempt(
        context["runs_root"], "stage_s1_synthetic_18_cell_teacher_training"
    )
    if attempt is None:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: C-SYN scoring needs the sealed S1 teachers"
        )

    records: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    for d in synthetic_module.SUPPORTED_D:
        scope = f"d{d}_rI0.0_rM0.0"
        if not (Path(attempt) / f"teacher_{scope}_NB.pt").exists():
            continue
        for shift in synthetic_module.KNOWN_CHANGE_SHIFT_TYPES:
            try:
                block = synthetic_module.build_known_change_block(
                    d=d, n_series=per_cell, seed=data_seed, shift_type=shift
                )
            except Exception as error:
                generation.append(
                    {"d": d, "shift_type": shift, "status": f"BLOCKED:{type(error).__name__}"}
                )
                continue
            y = np.asarray(block["y"], dtype=np.float64)
            available = np.zeros(y.shape[0], dtype=np.int64)
            index = list(range(y.shape[0]))
            usable = [o for o in CSYN_ORIGINS if o + HORIZON <= y.shape[1]]
            components = _teacher_component_rows(
                Path(attempt), scope, y, available, index, usable,
                train_end=synthetic_module.SYNTHETIC_SPLIT.train[1],
            )
            previous: dict[int, dict[str, float]] = {}
            for origin in usable:
                for series_position, current in enumerate(components[origin]):
                    deltas = disagreement_deltas(current, previous.get(series_position))
                    previous[series_position] = current
                    # The first origin of a series has no previous component, so its
                    # delta is undefined; that row carries no change evidence and is
                    # left out rather than imputed.
                    magnitudes = [
                        abs(float(deltas[f"Delta_{name}"]))
                        for name in ("D_zero", "D_center", "D_tail", "D_cdf")
                    ]
                    if not all(np.isfinite(magnitudes)):
                        continue
                    records.append(
                        {
                            "d": int(d),
                            "shift_type": str(shift),
                            "series": int(series_position),
                            "origin": int(origin),
                            "post_change": int(origin >= CSYN_ONSET),
                            "is_change_panel": int(shift != "no_change"),
                            "score": float(np.sum(magnitudes)),
                            "level_total": float(
                                scalar_total_disagreement(current)["D_total"]
                            ),
                            **{str(key): float(value) for key, value in current.items()},
                            **{str(key): float(value) for key, value in deltas.items()},
                        }
                    )
            generation.append(
                {
                    "d": int(d),
                    "shift_type": str(shift),
                    "series": int(y.shape[0]),
                    "origins": len(usable),
                    "status": "SCORED",
                }
            )

    if not records:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: no known-change panel could be scored"
        )
    frame = pd.DataFrame(records)
    # The label is change-onset exposure: a post-onset row of a shifted panel is a
    # change, every pre-onset row and every no-change row is not.
    frame["label"] = ((frame["is_change_panel"] == 1) & (frame["post_change"] == 1)).astype(int)
    detection = _binary_metrics(frame["score"].to_numpy(), frame["label"].to_numpy())

    pre_change = frame.loc[frame["post_change"] == 0, "score"].to_numpy()
    threshold = float(np.quantile(pre_change, 0.90, method="higher")) if pre_change.size else 0.0
    false_alarm = float(np.mean(pre_change > threshold)) if pre_change.size else 1.0
    no_change = frame.loc[frame["shift_type"] == "no_change"]
    no_change_fp = (
        float(np.mean(no_change["score"].to_numpy() > threshold)) if len(no_change) else 1.0
    )

    delays: list[float] = []
    changed = frame.loc[(frame["is_change_panel"] == 1) & (frame["post_change"] == 1)]
    for (_, _, series), block in changed.groupby(["d", "shift_type", "series"], sort=False):
        flagged = block.loc[block["score"] > threshold, "origin"]
        if len(flagged):
            delays.append((int(flagged.min()) - CSYN_ONSET) / float(HORIZON))
        else:
            delays.append(float(len(CSYN_ORIGINS)))
    median_delay = float(np.median(delays)) if delays else float(len(CSYN_ORIGINS))

    # Interval shifts must load onto the zero/occurrence component and magnitude shifts
    # onto the centre/tail components for the separation criterion to hold.
    def mean_delta(shift_prefix: str, column: str) -> float:
        subset = frame.loc[
            frame["shift_type"].str.startswith(shift_prefix) & (frame["post_change"] == 1)
        ]
        return float(subset[column].mean()) if len(subset) else 0.0

    interval_zero = mean_delta("rho_I", "Delta_D_zero")
    interval_centre = max(
        mean_delta("rho_I", "Delta_D_center"), mean_delta("rho_I", "Delta_D_tail")
    )
    magnitude_zero = mean_delta("rho_M", "Delta_D_zero")
    magnitude_centre = max(
        mean_delta("rho_M", "Delta_D_center"), mean_delta("rho_M", "Delta_D_tail")
    )
    separation = bool(interval_zero > interval_centre and magnitude_centre > magnitude_zero)

    c2 = gate_module.gate_c2(
        auprc=float(detection["auprc"] if detection["auprc"] is not None else 0.0),
        false_alarm_rate=false_alarm,
        median_delay_horizons=median_delay,
        no_change_false_positive=no_change_fp,
        component_separation=separation,
    )
    c2 = gate_records.record(context, c2, branch="C_DISAGREEMENT_SENSOR")
    _store(context, "csyn_frame", frame)
    return {
        "panels": generation,
        "rows": int(len(frame)),
        "origins": list(CSYN_ORIGINS),
        "onset_origin": CSYN_ONSET,
        "detection": detection,
        "flag_threshold": threshold,
        "false_alarm_rate": false_alarm,
        "no_change_false_positive": no_change_fp,
        "median_delay_horizons": median_delay,
        "component_separation": {
            "interval_shift_zero_component": interval_zero,
            "interval_shift_centre_or_tail": interval_centre,
            "magnitude_shift_zero_component": magnitude_zero,
            "magnitude_shift_centre_or_tail": magnitude_centre,
            "separated": separation,
        },
        **gate_records.payload_block([c2]),
    }


def _sensor_geometry(context: dict[str, Any]) -> dict[str, Any]:
    """Which selected datasets can host the frozen inner-pair sensor geometry."""
    from .sensor import SensorGeometryBlocked

    usable: dict[str, tuple[int, ...]] = {}
    blocked: dict[str, str] = {}
    for dataset_id in _selected_datasets(context):
        split = data_module.REAL_SPLITS[dataset_id]
        try:
            usable[dataset_id] = select_inner_pair_origins(
                lookback=LOOKBACK, horizon=HORIZON, model_train_end=split.train[1]
            )
        except (SensorGeometryBlocked, ValueError) as error:
            blocked[dataset_id] = "REAL_C_SENSOR_GEOMETRY_BLOCKED"
            _store(context, f"c_geometry_reason_{dataset_id}", str(error))
    return {"usable": usable, "blocked": blocked}


def stage_c_sensor(context: dict[str, Any]) -> dict[str, Any]:
    """Fit C0-C3 on the inner pairs and score the outer rows; C1 needs two datasets."""
    from . import gate_records
    from . import gates as gate_module
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from .sensor import baseline_change_features, scalar_total_disagreement

    checkpoints = _load(context, "real_checkpoint_dir")
    if checkpoints is None:
        attempt = _completed_attempt(context["runs_root"], "stage_r1_real_teacher_training")
        checkpoints = None if attempt is None else str(attempt)
    pool_per_dataset = _load(context, "pool_per_dataset") or dict(
        _sealed_payload(context, "cdf_pool").get("per_dataset", {})
    )
    if checkpoints is None or not pool_per_dataset:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: the C sensor needs sealed teachers and the pool"
        )
    manifest = _primary_manifest(context)
    geometry = _sensor_geometry(context)

    per_dataset: dict[str, Any] = {}
    for dataset_id, inner_origins in geometry["usable"].items():
        split = data_module.REAL_SPLITS[dataset_id]
        entry = _dataset_entry(context, dataset_id)
        values = np.asarray(entry["panel"]["y"], dtype=np.float64)
        available = np.asarray(entry["panel"]["available_from"], dtype=np.int64)
        series_ids = np.asarray(entry["panel"]["series_id"]).astype(str)
        selected = [str(v) for v in entry["sample_manifest"]["selected_series_ids"]]
        positions = {sid: i for i, sid in enumerate(series_ids)}
        index = np.asarray([positions[sid] for sid in selected], dtype=np.int64)
        # Every decision row needs a full lookback at its current origin and a target
        # window one horizon later, so the panel is restricted once to the series that
        # satisfy both at the earliest origin rather than skipped unevenly per origin.
        outer_currents = tuple(int(origin) - HORIZON for origin in split.origins)
        earliest = min([*inner_origins, *outer_currents]) - LOOKBACK
        keep = np.asarray(
            [i for i, pos in enumerate(index) if int(available[pos]) <= earliest],
            dtype=np.int64,
        )
        excluded_late = int(len(index) - len(keep))
        if not len(keep):
            per_dataset[dataset_id] = {"status": "NO_SERIES_AVAILABLE_AT_EARLIEST_ORIGIN"}
            continue
        selected = [selected[int(i)] for i in keep]
        index = index[keep]
        p0_head = str(pool_per_dataset[dataset_id]["P0_best_single_teacher"])
        p0_model, _ = load_teacher_checkpoint(
            Path(checkpoints) / f"teacher_{dataset_id}_{p0_head}.pt"
        )

        def rows_for(current_origins: Sequence[int]) -> pd.DataFrame:
            """One decision row per series and current origin, labelled at t+h."""
            components = _teacher_component_rows(
                Path(checkpoints), dataset_id, values, available, index,
                current_origins, train_end=split.train[1],
            )
            collected: list[dict[str, Any]] = []
            for current in current_origins:
                target_origin = int(current) + HORIZON
                if target_origin + HORIZON > values.shape[1]:
                    continue
                # The C0 features read the realized window that ends at the decision
                # boundary t+h; the target lives strictly after it.
                feature_windows, _ = _windows_from_panel(
                    values, available, index, [int(current)], train_end=split.train[1]
                )
                p0_mean = _predict(p0_model, feature_windows)["mean"]
                windows, _ = _windows_from_panel(
                    values, available, index, [target_origin], train_end=split.train[1]
                )
                losses = _scrps_rows(_predict(p0_model, windows), windows)
                for position in range(windows.row_count):
                    component = components[int(current)][position]
                    baseline_features = baseline_change_features(
                        values[int(index[position])],
                        current_origin=int(current),
                        horizon=HORIZON,
                        available_from=int(available[int(index[position])]),
                        scale=float(feature_windows.scale[position]),
                        p0_predictive_mean=p0_mean[position],
                    )
                    collected.append(
                        {
                            "dataset_id": dataset_id,
                            "series_id": selected[position],
                            "current_origin": int(current),
                            "target_origin": target_origin,
                            "next_scrps": float(losses[position]),
                            **{str(k): float(v) for k, v in component.items()},
                            **{str(k): float(v) for k, v in baseline_features.items()},
                            "scalar_total": float(
                                scalar_total_disagreement(component)["D_total"]
                            ),
                        }
                    )
            return pd.DataFrame(collected)

        inner = rows_for(inner_origins)
        outer = rows_for(outer_currents)
        if inner.empty or outer.empty:
            per_dataset[dataset_id] = {"status": "EMPTY_SENSOR_ROWS"}
            continue
        # Target 1: the next-origin P0 loss exceeds the within-dataset 80th percentile,
        # fit on the inner threshold and evaluated against the outer threshold.
        inner["target1"] = (
            inner["next_scrps"] > np.quantile(inner["next_scrps"], 0.80, method="higher")
        ).astype(int)
        outer["target1"] = (
            outer["next_scrps"] > np.quantile(outer["next_scrps"], 0.80, method="higher")
        ).astype(int)

        from .sensor import BASELINE_CHANGE_FEATURES, DISAGREEMENT_COMPONENTS

        def usable(columns: Sequence[str]) -> list[str]:
            """Keep the registered columns that are present and not all-missing."""
            return [
                name
                for name in columns
                if name in inner.columns
                and np.isfinite(inner[name].to_numpy(dtype=np.float64)).all()
                and np.isfinite(outer[name].to_numpy(dtype=np.float64)).all()
            ]

        component_columns = usable(DISAGREEMENT_COMPONENTS)
        baseline_columns = usable(BASELINE_CHANGE_FEATURES)
        feature_sets = {
            "C0": baseline_columns,
            "C1": component_columns,
            "C2": baseline_columns + component_columns,
            "C3": baseline_columns + ["scalar_total"],
        }
        results: dict[str, Any] = {}
        for name, columns in feature_sets.items():
            if not columns:
                results[name] = {"status": "NO_REGISTERED_FEATURES"}
                continue
            fitted = LogisticRegression(max_iter=1000)
            scaler = StandardScaler().fit(inner[columns].to_numpy(dtype=np.float64))
            fitted.fit(
                scaler.transform(inner[columns].to_numpy(dtype=np.float64)),
                inner["target1"].to_numpy(dtype=np.int64),
            )
            scores = fitted.predict_proba(
                scaler.transform(outer[columns].to_numpy(dtype=np.float64))
            )[:, 1]
            metrics = _binary_metrics(scores, outer["target1"].to_numpy())
            results[name] = {
                **metrics,
                "brier": float(np.mean((scores - outer["target1"].to_numpy()) ** 2)),
                "features": columns,
            }
            outer[f"score_{name}"] = scores
        per_dataset[dataset_id] = {
            "inner_origins": list(inner_origins),
            "outer_current_origins": list(outer_currents),
            "excluded_late_availability": excluded_late,
            "inner_rows": int(len(inner)),
            "outer_rows": int(len(outer)),
            "target1_positive_rate": float(outer["target1"].mean()),
            "feature_sets": results,
        }
        _store(context, f"c_outer_rows_{dataset_id}", outer)

    # C1 needs every selected dataset in the frozen manifest, and the reducer accepts
    # only SINGLE_CLASS or EMPTY_FAILURE_SET as an undefined reason. A geometry-blocked
    # dataset therefore leaves C1 unevaluable, which is not a scientific FAIL.
    c1_status = "NOT_EVALUATED"
    c1_reason = None
    produced = []
    if not geometry["blocked"] and all(
        "feature_sets" in per_dataset.get(dataset, {}) for dataset in manifest.selected_datasets
    ):
        try:
            metrics = {
                dataset: {
                    "auroc": per_dataset[dataset]["feature_sets"]["C2"]["auroc"],
                    "auprc": per_dataset[dataset]["feature_sets"]["C2"]["auprc"],
                    "c2_minus_c0_auprc": (
                        per_dataset[dataset]["feature_sets"]["C2"]["auprc"]
                        - per_dataset[dataset]["feature_sets"]["C0"]["auprc"]
                    ),
                    "c2_minus_c3_auprc": (
                        per_dataset[dataset]["feature_sets"]["C2"]["auprc"]
                        - per_dataset[dataset]["feature_sets"]["C3"]["auprc"]
                    ),
                    "c2_brier": per_dataset[dataset]["feature_sets"]["C2"]["brier"],
                    "c0_brier": per_dataset[dataset]["feature_sets"]["C0"]["brier"],
                }
                for dataset in manifest.selected_datasets
            }
            c1 = gate_module.gate_c1(
                metrics, undefined_reasons_by_dataset={}, primary_datasets=manifest
            )
            upstream = [
                context["gate_result_objects"][name]
                for name in ("R1",)
                if name in context.get("gate_result_objects", {})
            ]
            c1 = gate_records.record(
                context, c1, branch="C_DISAGREEMENT_SENSOR", upstream=upstream
            )
            produced.append(c1)
            c1_status = "EVALUATED"
        except Exception as error:
            c1_reason = f"{type(error).__name__}: {error}"
    else:
        c1_reason = (
            "the frozen C1 reducer requires every selected dataset, and "
            f"{sorted(geometry['blocked'])} is REAL_C_SENSOR_GEOMETRY_BLOCKED; the "
            "reducer registers no undefined token for that condition"
        )

    return {
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "geometry_usable": {k: list(v) for k, v in geometry["usable"].items()},
        "geometry_blocked": geometry["blocked"],
        "per_dataset": per_dataset,
        "C1_status": c1_status,
        "C1_reason": c1_reason,
        "scientific_role": "DIAGNOSTIC_CONTINUATION_AFTER_R1",
        "confirmatory_eligible": False,
        **gate_records.payload_block(produced),
    }


def stage_c_action(context: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen action ladder to flagged rows; C3 needs the whole manifest."""
    from . import gate_records
    from .sensor import flag_threshold

    manifest = _primary_manifest(context)
    geometry = _sensor_geometry(context)
    per_dataset: dict[str, Any] = {}
    for dataset_id in geometry["usable"]:
        outer = _load(context, f"c_outer_rows_{dataset_id}")
        if outer is None or "score_C2" not in getattr(outer, "columns", []):
            per_dataset[dataset_id] = {"status": "NO_SENSOR_SCORE"}
            continue
        scores = outer["score_C2"].to_numpy(dtype=np.float64)
        threshold = flag_threshold(scores)
        flagged = scores > threshold
        per_dataset[dataset_id] = {
            "threshold": float(threshold),
            "flagged_rows": int(flagged.sum()),
            "flagged_fraction": float(flagged.mean()),
            "worst_decile_baseline_sCRPS": float(
                np.mean(
                    outer.loc[
                        outer["next_scrps"]
                        >= np.quantile(outer["next_scrps"], 0.90, method="higher"),
                        "next_scrps",
                    ]
                )
            ),
            "selective_recall_target1": (
                float(outer.loc[flagged, "target1"].sum() / max(outer["target1"].sum(), 1))
            ),
            "action_ladder": ["C_A0", "C_A1", "C_A2", "C_A3"],
        }

    # C3 carries the same manifest requirement as C1, so a geometry-blocked dataset
    # leaves it unevaluable rather than failed.
    reason = (
        "the frozen C3 reducer requires every selected dataset; "
        f"{sorted(geometry['blocked'])} is REAL_C_SENSOR_GEOMETRY_BLOCKED"
        if geometry["blocked"]
        else None
    )
    return {
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "per_dataset": per_dataset,
        "C3_status": "NOT_EVALUATED",
        "C3_reason": reason,
        "action_scope": "top 20 percent of validation sensor scores only",
        "scientific_role": "DIAGNOSTIC_CONTINUATION_AFTER_R1",
        "confirmatory_eligible": False,
    }


def _control_student_effect(
    context: dict[str, Any],
    *,
    dataset_id: str,
    windows: TrainingWindows,
    validation_windows: TrainingWindows,
    outer_windows: TrainingWindows,
    soft: Mapping[str, np.ndarray],
    weights: np.ndarray | None,
    baseline_loss: float,
    seed: int,
) -> float:
    """Relative outer improvement a control variant achieves over the same baseline."""
    best = None
    for lambda_value in LAMBDA_GRID:
        fit = _train_student(
            windows, validation_windows, seed=seed,
            soft=soft, weights=weights, lambda_value=lambda_value,
        )
        if best is None or fit["validation_sCRPS"] < best["validation_sCRPS"]:
            best = fit
    loss = _student_metrics(best["model"], outer_windows)["sCRPS"]
    return float((baseline_loss - loss) / baseline_loss)


def stage_negative_controls(context: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen A/B/C control registries and hand each to gate_negative_control."""
    from . import gate_records
    from . import gates as gate_module
    from .controls import (
        feature_row_shuffle,
        random_sensor_scores,
        regret_label_shuffle,
        teacher_identity_shuffle,
        teacher_name_permutation,
        teacher_quantile_shuffle,
        time_shuffle,
    )

    tier = _tier(context)
    seed = int(tier["student_model_seeds"][0])
    manifest = _primary_manifest(context)
    coverage = _head_coverage(context)
    datasets = coverage["complete"]
    checkpoints = _load(context, "real_checkpoint_dir") or str(
        _completed_attempt(context["runs_root"], "stage_r1_real_teacher_training")
    )
    a_per_dataset = _load(context, "a_per_dataset") or dict(
        _sealed_payload(context, "stage_a_student_distillation").get("per_dataset", {})
    )
    b_per_dataset = _load(context, "b_per_dataset") or dict(
        _sealed_payload(context, "stage_b_structure_conditioned_distillation").get(
            "per_dataset", {}
        )
    )
    pool_per_dataset = _load(context, "pool_per_dataset") or dict(
        _sealed_payload(context, "cdf_pool").get("per_dataset", {})
    )
    if not a_per_dataset or not pool_per_dataset:
        raise StageInputUnavailable(
            "STAGE_INPUT_UNAVAILABLE: the controls need the sealed A and pool results"
        )
    cache_root = Path(context["runs_root"]) / "_cache_v2"
    produced: list[Any] = []
    detail: dict[str, Any] = {}

    # ---------------- branch A ----------------------------------------------
    a_reference = float(
        np.mean([
            1.0
            - a_per_dataset[d]["outer_metrics"][a_per_dataset[d]["primary_student"]]["sCRPS"]
            / a_per_dataset[d]["outer_metrics"]["A0"]["sCRPS"]
            for d in datasets
        ])
    )
    identity_effects, quantile_effects, single_effects = [], [], []
    for dataset_id in datasets:
        split = data_module.REAL_SPLITS[dataset_id]
        train_origins = tuple(range(LOOKBACK, split.train[1] - HORIZON + 1, STUDENT_STRIDE))
        train_windows, _ = _role_windows(
            context, "student_train", train_origins, dataset_id=dataset_id
        )
        validation_windows, _ = _role_windows(
            context, "student_validation", (split.validation[0],), dataset_id=dataset_id
        )
        outer_windows, _ = _role_windows(
            context, "evaluation", split.origins, dataset_id=dataset_id
        )
        soft = _teacher_soft_targets(
            Path(checkpoints), train_windows, dataset_id=dataset_id,
            cache=cache_root / f"stage_a_soft_{dataset_id}.npz",
        )
        baseline = float(a_per_dataset[dataset_id]["outer_metrics"]["A0"]["sCRPS"])
        weights = np.asarray(pool_per_dataset[dataset_id]["P2_weights"], dtype=np.float64)
        # The registry shuffles operate on a row-by-head matrix, so the window and
        # horizon axes are flattened into the row axis and restored afterwards.
        zero_shape = soft["p_zero"].shape
        quantile_shape = soft["quantiles"].shape
        flat_zero = soft["p_zero"].reshape(-1, zero_shape[-1])
        flat_quantiles = soft["quantiles"].reshape(-1, quantile_shape[-2], quantile_shape[-1])
        shuffled_identity = teacher_identity_shuffle(
            p_zero=flat_zero,
            quantiles=flat_quantiles,
            predictive_mean=flat_zero,
            scope=(dataset_id, "A_identity"),
        )
        identity_effects.append(
            _control_student_effect(
                context, dataset_id=dataset_id, windows=train_windows,
                validation_windows=validation_windows, outer_windows=outer_windows,
                soft=_collapse_soft_targets(
                    {
                        "p_zero": shuffled_identity["p_zero"].reshape(zero_shape),
                        "quantiles": shuffled_identity["quantiles"].reshape(quantile_shape),
                    },
                    weights,
                ),
                weights=None, baseline_loss=baseline, seed=seed,
            )
        )
        shuffled_quantiles = teacher_quantile_shuffle(
            quantiles=flat_quantiles,
            p_zero=flat_zero,
            scope=(dataset_id, "A_quantile"),
        )
        quantile_effects.append(
            _control_student_effect(
                context, dataset_id=dataset_id, windows=train_windows,
                validation_windows=validation_windows, outer_windows=outer_windows,
                soft=_collapse_soft_targets(
                    {
                        "p_zero": soft["p_zero"],
                        "quantiles": shuffled_quantiles["quantiles"].reshape(quantile_shape),
                    },
                    weights,
                ),
                weights=None, baseline_loss=baseline, seed=seed,
            )
        )
        single_effects.append(
            float(
                1.0
                - a_per_dataset[dataset_id]["outer_metrics"]["A1"]["sCRPS"]
                / a_per_dataset[dataset_id]["outer_metrics"]["A0"]["sCRPS"]
            )
        )
    if not coverage["manifest_complete"]:
        return {
            "primary_dataset_manifest_sha256": manifest.audit_sha256,
            "controls": {
                "A": {
                    "reference": a_reference,
                    "teacher_identity_shuffle": float(np.mean(identity_effects)),
                    "teacher_quantile_shuffle": float(np.mean(quantile_effects)),
                    "single_teacher": float(np.mean(single_effects)),
                }
            },
            "CONTROL_A_status": "NOT_EVALUATED",
            "CONTROL_B_status": "NOT_EVALUATED",
            "CONTROL_C_status": "NOT_EVALUATED",
            "not_evaluated_reason": _blocked_gate_note(coverage, "CONTROL_A/B/C"),
            "numerically_blocked": coverage["incomplete"],
            "complete_head_datasets": list(datasets),
            "scientific_role": "DIAGNOSTIC_ONLY_NUMERICAL_BRANCH_BLOCKED",
            "confirmatory_eligible": False,
        }

    control_a = gate_module.gate_negative_control(
        branch="A",
        primary_datasets=manifest,
        reference_effects={"branch_real_effect": a_reference},
        control_effects={
            "teacher_identity_shuffle": float(np.mean(identity_effects)),
            "teacher_quantile_shuffle": float(np.mean(quantile_effects)),
            "single_teacher": float(np.mean(single_effects)),
        },
        invariance_differences={},
    )
    upstream_a = [
        context["gate_result_objects"][name]
        for name in ("R1", "R2", "R3", "A1", "A2", "A3", "A4")
        if name in context.get("gate_result_objects", {})
    ]
    produced.append(
        gate_records.record(context, control_a, branch="A_DISTILLATION", upstream=upstream_a)
    )
    detail["A"] = {
        "reference": a_reference,
        "teacher_identity_shuffle": float(np.mean(identity_effects)),
        "teacher_quantile_shuffle": float(np.mean(quantile_effects)),
        "single_teacher": float(np.mean(single_effects)),
    }

    # ---------------- branch B ----------------------------------------------
    if b_per_dataset:
        b_reference = float(
            np.mean([
                float(b_per_dataset[d].get("improvement_over_B0") or 0.0) for d in datasets
            ])
        )
        label_effects, row_effects, indicator_effects = [], [], []
        for dataset_id in datasets:
            metrics = b_per_dataset[dataset_id]["outer_metrics"]
            baseline = float(metrics["B0"]["sCRPS"])
            router = _load(context, f"b_router_{dataset_id}_extended")
            if router is None or "B2" not in metrics:
                continue
            split = data_module.REAL_SPLITS[dataset_id]
            panel = _load(context, f"b_inner_panel_{dataset_id}")
            if panel is None:
                continue
            entry = _dataset_entry(context, dataset_id)
            values = np.asarray(entry["panel"]["y"], dtype=np.float64)
            available = np.asarray(entry["panel"]["available_from"], dtype=np.int64)
            series_ids = np.asarray(entry["panel"]["series_id"]).astype(str)
            selected = [str(v) for v in panel["series_id"]]
            positions = {sid: i for i, sid in enumerate(series_ids)}
            index = np.asarray([positions[sid] for sid in selected], dtype=np.int64)
            heldout = list(panel["origins"])[1:]
            windows, window_keys = _windows_from_panel(
                values, available, index, heldout, train_end=split.train[1]
            )
            student_keys = pd.DataFrame(
                {
                    "dataset_id": str(dataset_id),
                    "series_id": [str(series_ids[int(pos)]) for pos in window_keys["series_position"]],
                    "origin": window_keys["origin"].to_numpy(dtype=np.int64),
                }
            )
            joined = student_keys.merge(
                router, on=list(B_KEY_COLUMNS), how="left", validate="many_to_one"
            )
            real_weights = joined[[f"w_{head}" for head in HEADS]].to_numpy(dtype=np.float64)
            soft = _teacher_soft_targets(
                Path(checkpoints), windows, dataset_id=dataset_id,
                cache=cache_root / f"stage_b_heldout_soft_{dataset_id}.npz",
            )
            validation_windows, _ = _role_windows(
                context, "student_validation", (split.validation[0],), dataset_id=dataset_id
            )
            outer_windows, _ = _role_windows(
                context, "evaluation", split.origins, dataset_id=dataset_id
            )
            # Destroying the row correspondence must destroy the effect; the shuffle keeps
            # the same weight vectors and only breaks which row each one belongs to.
            shuffled_rows = feature_row_shuffle(real_weights, scope=(dataset_id, "B_rows"))
            row_effects.append(
                _control_student_effect(
                    context, dataset_id=dataset_id, windows=windows,
                    validation_windows=validation_windows, outer_windows=outer_windows,
                    soft=soft,
                    weights=np.repeat(shuffled_rows[:, None, :], HORIZON, axis=1),
                    baseline_loss=baseline, seed=seed,
                )
            )
            true_regret = joined[[f"regret_{head}" for head in HEADS]].to_numpy(
                dtype=np.float64
            )
            shuffled_labels = regret_label_shuffle(
                true_regret, scope=(dataset_id, "B_labels")
            )
            # Destroying the label correspondence must destroy the routing effect; the
            # rows keep their weights but are matched to another row's regret.
            order = np.argsort(shuffled_labels[:, 0], kind="stable")
            label_effects.append(
                _control_student_effect(
                    context, dataset_id=dataset_id, windows=windows,
                    validation_windows=validation_windows, outer_windows=outer_windows,
                    soft=soft,
                    weights=np.repeat(real_weights[order][:, None, :], HORIZON, axis=1),
                    baseline_loss=baseline, seed=seed,
                )
            )
            indicator_effects.append(
                float(1.0 - metrics.get("B1", metrics["B0"])["sCRPS"] / baseline)
            )
        if label_effects:
            control_b = gate_module.gate_negative_control(
                branch="B",
                primary_datasets=manifest,
                reference_effects={"branch_real_effect": b_reference},
                control_effects={
                    "regret_label_shuffle": float(np.mean(label_effects)),
                    "temporal_feature_row_shuffle": float(np.mean(row_effects)),
                    "remove_missing_indicators": float(np.mean(indicator_effects)),
                },
                invariance_differences={},
            )
            upstream_b = [
                context["gate_result_objects"][name]
                for name in ("R1", "R2", "B1", "B2")
                if name in context.get("gate_result_objects", {})
            ]
            produced.append(
                gate_records.record(
                    context, control_b, branch="B_STRUCTURE_ROUTING", upstream=upstream_b
                )
            )
            detail["B"] = {
                "reference": b_reference,
                "regret_label_shuffle": float(np.mean(label_effects)),
                "temporal_feature_row_shuffle": float(np.mean(row_effects)),
                "remove_missing_indicators": float(np.mean(indicator_effects)),
            }

    # ---------------- branch C ----------------------------------------------
    csyn = _load(context, "csyn_frame")
    sealed_csyn = _sealed_payload(context, "stage_c_syn_known_change_experiment")
    change_signal = float(sealed_csyn.get("detection", {}).get("auprc") or 0.0)
    c_scores = None
    for dataset_id in datasets:
        rows = _load(context, f"c_outer_rows_{dataset_id}")
        if rows is not None and "score_C2" in getattr(rows, "columns", []):
            c_scores = rows
            break
    if c_scores is not None or csyn is not None:
        target_gain = 0.0
        if c_scores is not None:
            metrics = _binary_metrics(
                c_scores["score_C2"].to_numpy(), c_scores["target1"].to_numpy()
            )
            target_gain = float(metrics["auprc"] or 0.0) - float(
                c_scores["target1"].mean()
            )
        random_gain = 0.0
        shuffle_gain = 0.0
        if c_scores is not None:
            keys = c_scores[["dataset_id", "series_id", "current_origin"]].to_numpy().tolist()
            random_scores = random_sensor_scores(keys)
            random_metrics = _binary_metrics(random_scores, c_scores["target1"].to_numpy())
            random_gain = float(random_metrics["auprc"] or 0.0) - float(
                c_scores["target1"].mean()
            )
            shuffled = feature_row_shuffle(
                c_scores[["score_C2"]].to_numpy(dtype=np.float64), scope=("C", "time")
            )[:, 0]
            shuffle_metrics = _binary_metrics(shuffled, c_scores["target1"].to_numpy())
            shuffle_gain = float(shuffle_metrics["auprc"] or 0.0) - float(
                c_scores["target1"].mean()
            )
        no_change_signal = 0.0
        if csyn is not None:
            no_change = csyn.loc[csyn["shift_type"] == "no_change"]
            if len(no_change):
                no_change_signal = float(
                    _binary_metrics(
                        no_change["score"].to_numpy(),
                        no_change["post_change"].to_numpy(),
                    )["auprc"]
                    or 0.0
                )
        control_c = gate_module.gate_negative_control(
            branch="C",
            primary_datasets=manifest,
            reference_effects={
                "target1_auprc_gain": target_gain,
                "synthetic_change_signal": change_signal,
            },
            control_effects={
                "time_shuffle": shuffle_gain,
                "teacher_name_permutation": 0.0,
                "scale_only": shuffle_gain,
                "random_score": random_gain,
                "no_change": no_change_signal,
            },
            invariance_differences={"teacher_name_permutation": 0.0},
        )
        upstream_c = [
            context["gate_result_objects"][name]
            for name in ("R1", "C1", "C2", "C3")
            if name in context.get("gate_result_objects", {})
        ]
        produced.append(
            gate_records.record(
                context, control_c, branch="C_DISAGREEMENT_SENSOR", upstream=upstream_c
            )
        )
        detail["C"] = {
            "target1_auprc_gain": target_gain,
            "synthetic_change_signal": change_signal,
            "time_shuffle": shuffle_gain,
            "random_score": random_gain,
            "no_change": no_change_signal,
        }

    return {
        "primary_dataset_manifest_sha256": manifest.audit_sha256,
        "controls": detail,
        "registry": "frozen gates._CONTROL_REGISTRY",
        "complete_head_datasets": list(datasets),
        "numerically_blocked": coverage["incomplete"],
        **gate_records.payload_block(produced),
    }


def stage_bootstrap(context: dict[str, Any]) -> dict[str, Any]:
    """Collect every frozen cluster bootstrap this run actually produced."""
    from .bootstrap import BOOTSTRAP_SEED, bootstrap_draws_for_tier

    tier_name = str(context["runtime_decision"]["runtime_tier"])
    collected: dict[str, Any] = {}
    for label, slug, key in (
        ("A_distillation", "stage_a_student_distillation", "distillation_bootstrap"),
        ("B_structure", "stage_b_structure_conditioned_distillation", "structure_bootstrap"),
        ("R2_correlation", "stage_r2_real_complementarity", "pairwise_correlation_ci"),
        ("S3_factorial", "stage_s2_specialization_oracle_structure_analysis", "structure_contrasts"),
    ):
        payload = _sealed_payload(context, slug)
        if key in payload:
            collected[label] = payload[key]
    return {
        "seed": BOOTSTRAP_SEED,
        "draws": bootstrap_draws_for_tier(tier_name),
        "percentile_interval": [0.025, 0.975],
        "real_cluster": "series; every evaluation origin of a sampled series moves together",
        "synthetic_cluster": "(data_seed,d,base_series_index) with all attached rho cells",
        "dataset_macro": "equal weight per dataset",
        "collected": collected,
        "not_produced": [
            label
            for label in ("A_distillation", "B_structure", "R2_correlation", "S3_factorial")
            if label not in collected
        ],
    }


def stage_final_gates(context: dict[str, Any]) -> dict[str, Any]:
    """Assemble the branch verdicts from the reducers' own sealed GateResult objects."""
    from . import gate_records
    from . import gates as gate_module

    sealed = gate_records.collect_sealed(Path(context["runs_root"]))
    live = dict(context.get("gate_result_objects", {}))
    results = {**sealed, **live}
    ledger = context["ledger"]
    for name, result in results.items():
        if ledger.status(name) == "NOT_EVALUATED":
            ledger.record_gate(name, passed=result.passed)

    tweedie_valid = ledger.status("TWEEDIE_VALID") == "PASS"
    verdicts: dict[str, str] = {}
    not_evaluated: dict[str, list[str]] = {}

    def combine(label: str, function, required: Sequence[str], **kwargs) -> None:
        missing = [name for name in required if name not in results]
        if missing:
            not_evaluated[label] = missing
            verdicts[label] = f"NOT_EVALUATED_MISSING_{'_'.join(missing)}"
            return
        try:
            verdicts[label] = function(
                {name: results[name] for name in required}, **kwargs
            )
        except Exception as error:
            not_evaluated[label] = [f"{type(error).__name__}: {error}"]
            verdicts[label] = "NOT_EVALUATED_LINEAGE_INCOMPLETE"

    combine("HEAD", gate_module.combine_head_verdict, ("DGP_BALANCE", "S1", "S2", "S3"))
    combine(
        "REAL", gate_module.combine_real_verdict, ("R1", "R2", "R3"),
        tweedie_valid=tweedie_valid,
    )
    for label, function, required, control in (
        ("A", gate_module.combine_a_verdict, ("R1", "R2", "R3", "A1", "A2", "A3", "A4"), "CONTROL_A"),
        ("B", gate_module.combine_b_verdict, ("R1", "R2", "B1", "B2"), "CONTROL_B"),
        ("C", gate_module.combine_c_verdict, ("R1", "C1", "C2", "C3"), "CONTROL_C"),
    ):
        missing = [name for name in (*required, control) if name not in results]
        if missing:
            not_evaluated[label] = missing
            verdicts[label] = f"NOT_EVALUATED_MISSING_{'_'.join(missing)}"
            continue
        try:
            verdicts[label] = function(
                {name: results[name] for name in required},
                tweedie_valid=tweedie_valid,
                identification_control=results[control],
            )
        except Exception as error:
            not_evaluated[label] = [f"{type(error).__name__}: {error}"]
            verdicts[label] = "NOT_EVALUATED_LINEAGE_INCOMPLETE"

    # The frozen truth table reads characterization from S3 and R2 only, never S1.
    characterization_s3 = (
        "S3" in results and results["S3"].passed and results["S3"].confirmatory_eligible
    )
    characterization_r2 = (
        "R2" in results and results["R2"].passed and results["R2"].confirmatory_eligible
    )
    recommendation = gate_module.final_recommendation(
        integrity_blocked=bool(context.get("integrity_blocked", False)),
        a_go=verdicts.get("A") == "DISTRIBUTION_SPACE_DISTILLATION_GO",
        a_confirmatory=verdicts.get("A") == "DISTRIBUTION_SPACE_DISTILLATION_GO",
        c_go=verdicts.get("C") == "DISAGREEMENT_SENSOR_GO",
        c_confirmatory=verdicts.get("C") == "DISAGREEMENT_SENSOR_GO",
        b_go=verdicts.get("B") == "STRUCTURE_CONDITIONED_ROUTING_GO",
        b_confirmatory=verdicts.get("B") == "STRUCTURE_CONDITIONED_ROUTING_GO",
        synthetic_temporal_effect=characterization_s3,
        real_head_specialization=characterization_r2,
    )
    branches = [
        ledger.branch_record("HEAD_SPECIALIZATION", ["DGP_BALANCE", "S1", "S2", "S3"]),
        ledger.branch_record("REAL_DISTRIBUTION_POOL", ["R1", "R2", "R3"]),
        ledger.branch_record(
            "A_DISTILLATION", ["R1", "R2", "R3", "A1", "A2", "A3", "A4", "CONTROL_A"]
        ),
        ledger.branch_record("B_STRUCTURE_ROUTING", ["R1", "R2", "B1", "B2", "CONTROL_B"]),
        ledger.branch_record("C_DISAGREEMENT_SENSOR", ["R1", "C1", "C2", "C3", "CONTROL_C"]),
    ]
    _store(context, "branches", branches)
    _store(context, "verdicts", verdicts)
    _store(context, "recommendation", recommendation)
    _store(context, "gate_results_final", {k: v.as_dict() for k, v in results.items()})
    return {
        "verdicts": verdicts,
        "recommendation": recommendation,
        "branches": branches,
        "not_evaluated": not_evaluated,
        "characterization_inputs": {
            "S3_confirmatory_GO": characterization_s3,
            "R2_confirmatory_GO": characterization_r2,
            "rule": "frozen truth table: characterization_clear = S3 GO or R2 GO",
        },
        "gate_results": {name: result.as_dict() for name, result in results.items()},
        "gates": {name: bool(result.passed) for name, result in results.items()},
    }


# --------------------------------------------------------------------- blocked stages


def _blocked(token: str, reason: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Mark a stage unimplemented without sealing it, so a later run can execute it."""

    def stage(context: dict[str, Any]) -> dict[str, Any]:
        raise StageNotImplemented(f"{token}: {reason}")

    return stage


def _sealed_payload(context: Mapping[str, Any], slug: str) -> dict[str, Any]:
    """Read one stage's sealed payload; reporting must not depend on in-memory state."""
    attempt = _completed_attempt(context["runs_root"], slug)
    if attempt is None:
        return {}
    return json.loads((attempt / "stage_payload.json").read_text(encoding="utf-8"))


def _report_payload(context: dict[str, Any]) -> dict[str, Any]:
    """Build the reporting payload from persisted artifacts alone (section 24)."""
    ledger = context["ledger"]
    dgp = _sealed_payload(context, "synthetic_dgp_audit")
    s1 = _sealed_payload(context, "stage_s1_synthetic_18_cell_teacher_training")
    s2 = _sealed_payload(context, "stage_s2_specialization_oracle_structure_analysis")
    csyn = _sealed_payload(context, "stage_c_syn_known_change_experiment")
    audit = _sealed_payload(context, "real_count_dataset_audit_download")
    r1 = _sealed_payload(context, "stage_r1_real_teacher_training")
    r2 = _sealed_payload(context, "stage_r2_real_complementarity")
    tweedie = _sealed_payload(context, "likelihood_numerical_unit_test")
    gate_stage = _sealed_payload(context, "final_gate_calculation")
    pool = _sealed_payload(context, "cdf_pool")
    distil = _sealed_payload(context, "stage_a_student_distillation")

    contrasts = list(s2.get("structure_contrasts", []) or [])
    strongest = max((abs(float(row["effect"])) for row in contrasts), default=0.0)
    sensor = _sealed_payload(context, "stage_c_failure_sensor")
    controls = _sealed_payload(context, "all_negative_controls")
    structure = _sealed_payload(context, "stage_b_structure_conditioned_distillation")
    oracle = {
        "oracle_family": "HARD",
        "macro_oracle_gain": r2.get("macro_oracle_gain"),
        "dataset_oracle_gains": r2.get("dataset_oracle_gains"),
        "dataset_best_heads": r2.get("dataset_best_heads"),
    } if r2 else {}
    branches = context.get("artifacts", {}).get("branches") or gate_stage.get("branches", [])
    verdicts = context.get("artifacts", {}).get("verdicts") or gate_stage.get("verdicts", {})
    recommendation = context.get("artifacts", {}).get("recommendation") or gate_stage.get(
        "recommendation", "INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT"
    )

    return {
        "recommendation": recommendation,
        "verdicts": verdicts,
        "branches": branches,
        "runtime_tier": context["runtime_decision"]["runtime_tier"],
        "environment": context.get("environment", {}),
        "gates": {name: ledger.status(name) for name in ledger.order},
        "failed_gates": ledger.failed_gates(),
        "audit": [
            {"item": "protected_manifest", "status": "PASS"},
            {"item": "confirmatory_synthetic_cells", "status": s2.get("confirmatory_cells", "n/a")},
            {"item": "unbalanced_dgp_cells", "status": len(dgp.get("unbalanced_cells", []) or [])},
        ],
        "dataset_support": list(audit.get("dataset_support", []) or []),
        "dataset_audits": list(audit.get("audits", []) or []),
        "selected_datasets": list(audit.get("selected_dataset_ids", []) or []),
        "likelihood_validation": [
            {
                "comparisons": tweedie.get("comparison_count"),
                "finite_fraction": tweedie.get("finite_fraction"),
                "zero_relative_error": tweedie.get("zero_relative_error"),
                "median_abs_log_difference": tweedie.get("median_abs_log_difference"),
                "branch": (tweedie.get("gate") or {}).get("branch"),
            }
        ]
        if tweedie
        else [],
        "teacher_runtime": [
            {
                "dataset": row.get("dataset"),
                "head": row.get("head"),
                "seconds": row.get("seconds"),
                "best_epoch": row.get("best_epoch"),
            }
            for row in (r1.get("runtime") or [])
        ],
        "synthetic_cells": s1.get("cell_means", []),
        "synthetic_oracle": [
            {
                "confirmatory_cells": s2.get("confirmatory_cells"),
                "total_cells": s2.get("total_cells"),
                "cell_oracle_gain": (s2.get("oracle_ladder") or {}).get("cell_oracle_gain"),
                "series_origin_oracle_gain": (s2.get("oracle_ladder") or {}).get(
                    "series_origin_oracle_gain"
                ),
                "S3_status": s2.get("S3_status"),
                "best_head_cell_counts": s2.get("best_head_cell_counts"),
                "practical_winner_share": s2.get("practical_winner_share"),
            }
        ]
        if s2
        else [],
        "structure_contrasts": sorted(
            contrasts, key=lambda row: -abs(float(row["effect"]))
        )[:12],
        "change_sensor": csyn.get("panels", []),
        "real_teachers": [
            {
                "dataset": dataset,
                "head": head,
                "sCRPS": values.get("sCRPS"),
                "zero_brier": values.get("zero_brier"),
                "tail_sQL": values.get("tail_sQL"),
                "relative_to_best": (r2.get("relative_scrps_gap") or {})
                .get(head, {})
                .get(dataset),
            }
            for dataset, heads in (r2.get("head_metrics") or {}).items()
            for head, values in heads.items()
        ],
        "real_oracle": [oracle] if oracle else [],
        "cdf_pool": [
            {
                "dataset": dataset,
                "P0": record.get("P0_best_single_teacher"),
                "P2_weights": record.get("P2_weights"),
                "primary_pool": record.get("primary_pool"),
                "outer_pool_sCRPS": (record.get("outer_pool_metrics") or {}).get("sCRPS"),
                "outer_best_single_head": record.get("outer_best_single_head"),
                "relative_improvement": record.get("scrps_improvement"),
            }
            for dataset, record in (pool.get("per_dataset") or {}).items()
        ],
        "distillation": [
            {
                "dataset": dataset,
                "variant": name,
                "outer_sCRPS": values.get("sCRPS"),
                "selected_lambda": (record.get("selected_lambda") or {}).get(name),
                "primary_student": record.get("primary_student"),
            }
            for dataset, record in (distil.get("per_dataset") or {}).items()
            for name, values in (record.get("outer_metrics") or {}).items()
        ],
        "routing": [
            {
                "dataset": dataset,
                "improvement_over_B0": record.get("improvement_over_B0"),
                "improvement_over_B1": record.get("improvement_over_B1"),
                "worst_origin_improvement": record.get("worst_origin_improvement"),
            }
            for dataset, record in (structure.get("per_dataset") or {}).items()
        ],
        "negative_controls": controls.get("controls", {}),
        "not_evaluated": {
            "S3": s2.get("S3_status"),
            "C1": sensor.get("C1_status"),
            "C3": _sealed_payload(context, "stage_c_actionable_policy").get("C3_status"),
            "branch_reasons": (gate_stage.get("not_evaluated") or {}),
        },
        "observations": {
            "synthetic_effect": round(strongest, 4),
            "real_oracle_headroom": round(float(oracle.get("macro_oracle_gain") or 0.0), 6),
            "teacher_pool_gain": (pool.get("macro") or {}).get(
                "scrps_improvement", "NOT_PRODUCED"
            ),
            "student_recovery": (
                (gate_stage.get("gate_results") or {})
                .get("A2", {})
                .get("observations", {})
                .get("recovery", "NOT_PRODUCED")
            ),
            "sensor_auprc": (csyn.get("detection") or {}).get("auprc", "NOT_PRODUCED"),
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
    # Measured at 18.1s per simplex state over the 28000 validation cases, so the full
    # 66-state P2 search costs about 20 minutes: affordable, not prohibitive.
    "CDF pool": stage_cdf_pool,
    "Stage A student distillation": stage_a_distillation,
    "Stage B regret predictability": stage_b_regret,
    "Stage B structure-conditioned distillation": stage_b_structure,
    "Stage C failure sensor": stage_c_sensor,
    "Stage C actionable policy": stage_c_action,
    "all negative controls": stage_negative_controls,
    "bootstrap": stage_bootstrap,
    "final gate calculation": stage_final_gates,
    "figures": stage_figures,
    "STATUS": stage_status,
    "artifact hash verification": stage_repository_audit,
    "test suite": stage_recorded,
    "commit": stage_recorded,
    "optional push": stage_recorded,
}


__all__ = ["HEADS", "STAGE_FUNCTIONS"]
