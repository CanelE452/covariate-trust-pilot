"""Independent numerical validation for the full Tweedie predictive head.

Task 6 should call :func:`validate_tweedie_against_oracles` immediately after
the authoritative v4 preregistration freeze, persist the returned serializable payload, then
use :func:`tweedie_validation_gate` to carry a branch-local hard block forward.
This module deliberately performs no writes itself.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .distributions import TweedieDistribution


MU_GRID = (0.1, 0.5, 1.0, 5.0, 20.0)
PHI_GRID = (0.1, 0.5, 1.0, 2.0)
P_GRID = (1.1, 1.3, 1.5, 1.7, 1.9)
TOLERANCES = {
    "finite_fraction": 0.999,
    "zero_relative_error": 1e-5,
    "median_abs_log_difference": 1e-4,
    "p99_abs_log_difference": 1e-3,
}
EXPECTED_GRID_POINTS = len(MU_GRID) * len(PHI_GRID) * len(P_GRID)
EXPECTED_COMPARISON_COUNT = EXPECTED_GRID_POINTS * 6
EXPECTED_DISTRIBUTION_CHECKS = EXPECTED_GRID_POINTS * 2
TWEEDIE_REFERENCE_COMMIT = "f14a189d7cd80d41886041f44f40ae4db27d0067"
TWEEDIEGP_SOURCE_COMMIT = "2567d1322c8cc65f19df4f2d1774c610b167fb66"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_report_hash(report: dict[str, object]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Tweedie validation report must be finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _load_reference_factory():
    """Load the independent package lazily so its absence blocks only Tweedie."""
    return importlib.import_module("tweedie").tweedie


def _scalar_logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def compound_poisson_gamma_logpdf(y: torch.Tensor | float, mu: float, phi: float, p: float) -> torch.Tensor:
    """Independent scalar CP-Gamma density oracle with a residual tail bound.

    This intentionally does not call the vendored torch implementation.  Its
    scalar recurrence and log-concave residual criterion are independently
    structured, so a shared truncation defect cannot make both agree.
    """
    value = torch.as_tensor(y, dtype=torch.float64)
    if value.numel() != 1:
        raise ValueError("independent oracle accepts one scalar y at a time")
    y_scalar = float(value.item())
    if not (mu > 0 and phi > 0 and 1 < p < 2):
        raise ValueError("requires mu > 0, phi > 0, and 1 < p < 2")
    if y_scalar < 0:
        raise ValueError("oracle is defined for nonnegative values")
    rate = mu ** (2 - p) / (phi * (2 - p))
    if y_scalar == 0:
        return torch.tensor(-rate, dtype=torch.float64)
    alpha = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu ** (p - 1)
    log_y, log_rate, log_scale = math.log(y_scalar), math.log(rate), math.log(scale)
    running, peak, previous = -math.inf, -math.inf, None
    for j in range(1, 200_001):
        shape = j * alpha
        term = j * log_rate - rate - math.lgamma(j + 1)
        term += (shape - 1) * log_y - y_scalar / scale - math.lgamma(shape) - shape * log_scale
        running = _scalar_logaddexp(running, term)
        peak = max(peak, term)
        if previous is not None and previous > term:
            decline = previous - term
            log_tail = term - math.log1p(-math.exp(-decline))
            if log_tail <= peak - 35.0:
                return torch.tensor(running, dtype=torch.float64)
        previous = term
    raise RuntimeError("independent Tweedie oracle exceeded the 200000-term resource guard")


def _summary(
    differences: list[float | None],
    zero_relative_errors: list[float | None],
    finite: list[bool],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    indexed = [
        (index, float(value))
        for index, value in enumerate(differences)
        if value is not None and math.isfinite(float(value))
    ]
    values = np.asarray([value for _, value in indexed], dtype=float)
    finite_zero = [
        float(value)
        for value in zero_relative_errors
        if value is not None and math.isfinite(float(value))
    ]
    worst_index = max(indexed, key=lambda item: item[1])[0] if indexed else None
    return {
        "comparison_count": len(differences),
        "finite_fraction": float(np.mean(finite)) if finite else 0.0,
        "zero_relative_error": max(finite_zero) if finite_zero else None,
        "median_abs_log_difference": float(np.median(values)) if values.size else None,
        "p99_abs_log_difference": float(np.quantile(values, 0.99)) if values.size else None,
        "max_abs_log_difference": float(differences[worst_index]) if worst_index is not None else None,
        "worst_row": rows[worst_index] if worst_index is not None else None,
    }


def _passes(summary: dict[str, object]) -> bool:
    required = (
        "finite_fraction",
        "zero_relative_error",
        "median_abs_log_difference",
        "p99_abs_log_difference",
    )
    if any(summary.get(key) is None for key in required):
        return False
    return (
        float(summary["finite_fraction"]) >= TOLERANCES["finite_fraction"]
        and float(summary["zero_relative_error"]) <= TOLERANCES["zero_relative_error"]
        and float(summary["median_abs_log_difference"]) <= TOLERANCES["median_abs_log_difference"]
        and float(summary["p99_abs_log_difference"]) <= TOLERANCES["p99_abs_log_difference"]
    )


def _same_finite_number(left: object, right: object) -> bool:
    try:
        left_value = float(left)  # type: ignore[arg-type]
        right_value = float(right)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12)
    )


def _report_structure_is_complete(report: dict[str, object]) -> bool:
    rows = report.get("rows")
    if (
        report.get("grid_points") != EXPECTED_GRID_POINTS
        or report.get("comparison_count") != EXPECTED_COMPARISON_COUNT
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_COMPARISON_COUNT
    ):
        return False
    expected_keys = {
        (float(mu), float(phi), float(p), float(y))
        for mu in MU_GRID
        for phi in PHI_GRID
        for p in P_GRID
        for y in (0.0, 0.05 * mu, 0.25 * mu, mu, 2.0 * mu, 5.0 * mu)
    }
    observed_keys: list[tuple[float, float, float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            return False
        try:
            key = tuple(float(row[name]) for name in ("mu", "phi", "p", "y"))
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in key):
            return False
        observed_keys.append(key)  # type: ignore[arg-type]
    if len(set(observed_keys)) != EXPECTED_COMPARISON_COUNT:
        return False
    if set(observed_keys) != expected_keys:
        return False

    precision = report.get("precision")
    if not isinstance(precision, dict):
        return False
    oracle_columns = {
        ("float64", "reference"): ("log_prob_float64", "reference_log_prob"),
        ("float64", "compound_poisson_gamma"): (
            "log_prob_float64",
            "oracle_log_prob",
        ),
        ("float32", "reference"): (
            "log_prob_float32",
            "reference_log_prob_float32_parameters",
        ),
        ("float32", "compound_poisson_gamma"): (
            "log_prob_float32",
            "oracle_log_prob_float32_parameters",
        ),
    }
    for (precision_name, oracle_name), (ours_key, oracle_key) in oracle_columns.items():
        try:
            reported_summary = precision[precision_name][oracle_name]
        except (KeyError, TypeError):
            return False
        if not isinstance(reported_summary, dict):
            return False
        differences: list[float | None] = []
        zero_errors: list[float | None] = []
        finite_flags: list[bool] = []
        for row in rows:
            try:
                ours = float(row[ours_key])
                oracle = float(row[oracle_key])
            except (KeyError, TypeError, ValueError):
                ours = math.nan
                oracle = math.nan
            is_finite = math.isfinite(ours) and math.isfinite(oracle)
            finite_flags.append(is_finite)
            differences.append(abs(ours - oracle) if is_finite else None)
            if float(row["y"]) == 0.0:
                if is_finite and math.exp(oracle) > 0.0:
                    zero_errors.append(
                        abs(math.exp(ours) - math.exp(oracle)) / math.exp(oracle)
                    )
                else:
                    zero_errors.append(None)
        recomputed = _summary(differences, zero_errors, finite_flags, rows)
        if reported_summary.get("comparison_count") != EXPECTED_COMPARISON_COUNT:
            return False
        for name in (
            "finite_fraction",
            "zero_relative_error",
            "median_abs_log_difference",
            "p99_abs_log_difference",
            "max_abs_log_difference",
        ):
            if not _same_finite_number(reported_summary.get(name), recomputed.get(name)):
                return False

    checks = report.get("distribution_checks")
    reported_monotonicity = report.get("monotonicity")
    if (
        not isinstance(checks, list)
        or len(checks) != EXPECTED_DISTRIBUTION_CHECKS
        or not isinstance(reported_monotonicity, dict)
    ):
        return False
    expected_check_keys = {
        (float(mu), float(phi), float(p), precision_name)
        for mu in MU_GRID
        for phi in PHI_GRID
        for p in P_GRID
        for precision_name in ("float64", "float32")
    }
    observed_check_keys: list[tuple[float, float, float, str]] = []
    recomputed_monotonicity = {
        precision_name: {
            "cdf": 0,
            "quantile": 0,
            "cdf_failures": 0,
            "quantile_failures": 0,
            "zero_identity": 0,
            "zero_identity_failures": 0,
        }
        for precision_name in ("float64", "float32")
    }
    frozen_probabilities = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    for check in checks:
        if not isinstance(check, dict):
            return False
        try:
            precision_name = str(check["precision"])
            check_key = (
                float(check["mu"]),
                float(check["phi"]),
                float(check["p"]),
                precision_name,
            )
        except (KeyError, TypeError, ValueError):
            return False
        if precision_name not in recomputed_monotonicity:
            return False
        observed_check_keys.append(check_key)
        dtype = torch.float64 if precision_name == "float64" else torch.float32
        expected_queries = [
            float(torch.tensor(value, dtype=dtype).item())
            for value in (
                0.0,
                0.05 * check_key[0],
                0.25 * check_key[0],
                check_key[0],
                2.0 * check_key[0],
                5.0 * check_key[0],
            )
        ]
        if check.get("cdf_query_values") != expected_queries:
            return False
        cdf_error = check.get("cdf_error")
        cdf_values = check.get("cdf_values")
        if cdf_error is None:
            if not isinstance(cdf_values, list) or len(cdf_values) != 6:
                return False
            try:
                cdf_array = np.asarray(cdf_values, dtype=np.float64)
            except (TypeError, ValueError):
                return False
            if (
                not np.isfinite(cdf_array).all()
                or np.any(cdf_array < 0.0)
                or np.any(cdf_array > 1.0)
            ):
                return False
            recomputed_monotonicity[precision_name]["cdf"] += int(
                np.count_nonzero(cdf_array[1:] < cdf_array[:-1])
            )
        else:
            if not isinstance(cdf_error, str) or not cdf_error or cdf_values is not None:
                return False
            recomputed_monotonicity[precision_name]["cdf_failures"] += 1

        zero_identity_error = check.get("zero_identity_error")
        p_zero_value = check.get("p_zero_value")
        log_prob_zero_value = check.get("log_prob_zero_value")
        analytical_zero_mass = check.get("analytical_zero_mass")
        reported_zero_relative_error = check.get("zero_identity_max_relative_error")
        if zero_identity_error is None:
            try:
                identity_values = np.asarray(
                    [
                        cdf_values[0],  # type: ignore[index]
                        p_zero_value,
                        math.exp(float(log_prob_zero_value)),
                    ],
                    dtype=np.float64,
                )
                analytical = float(analytical_zero_mass)
            except (TypeError, ValueError, OverflowError, IndexError):
                return False
            if (
                not np.isfinite(identity_values).all()
                or not math.isfinite(analytical)
                or analytical < 0.0
                or analytical > 1.0
            ):
                return False
            denominator = max(analytical, float(torch.finfo(dtype).tiny))
            recomputed_zero_relative_error = float(
                np.max(np.abs(identity_values - analytical) / denominator)
            )
            if not _same_finite_number(
                reported_zero_relative_error, recomputed_zero_relative_error
            ):
                return False
            recomputed_monotonicity[precision_name]["zero_identity"] += int(
                recomputed_zero_relative_error > TOLERANCES["zero_relative_error"]
            )
        else:
            if (
                not isinstance(zero_identity_error, str)
                or not zero_identity_error
                or p_zero_value is not None
                or log_prob_zero_value is not None
                or analytical_zero_mass is not None
                or reported_zero_relative_error is not None
            ):
                return False
            recomputed_monotonicity[precision_name]["zero_identity_failures"] += 1

        if check.get("quantile_probabilities") != frozen_probabilities:
            return False
        quantile_error = check.get("quantile_error")
        quantile_values = check.get("quantile_values")
        infinity_indices = check.get("quantile_positive_infinity_indices")
        if quantile_error is None:
            if (
                not isinstance(quantile_values, list)
                or len(quantile_values) != len(frozen_probabilities)
                or infinity_indices != [len(frozen_probabilities) - 1]
                or quantile_values[-1] is not None
            ):
                return False
            try:
                finite_quantiles = np.asarray(quantile_values[:-1], dtype=np.float64)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(finite_quantiles).all() or np.any(finite_quantiles < 0.0):
                return False
            recomputed_monotonicity[precision_name]["quantile"] += int(
                np.count_nonzero(finite_quantiles[1:] < finite_quantiles[:-1])
            )
        else:
            if (
                not isinstance(quantile_error, str)
                or not quantile_error
                or quantile_values is not None
                or infinity_indices != []
            ):
                return False
            recomputed_monotonicity[precision_name]["quantile_failures"] += 1
    if (
        len(set(observed_check_keys)) != EXPECTED_DISTRIBUTION_CHECKS
        or set(observed_check_keys) != expected_check_keys
        or reported_monotonicity != recomputed_monotonicity
        or report.get("cdf_monotonicity_violations")
        != sum(values["cdf"] for values in recomputed_monotonicity.values())
        or report.get("quantile_monotonicity_violations")
        != sum(values["quantile"] for values in recomputed_monotonicity.values())
    ):
        return False

    reference_provenance = report.get("reference_provenance")
    training_provenance = report.get("training_likelihood_provenance")
    if (
        not isinstance(reference_provenance, dict)
        or reference_provenance.get("package") != "tweedie"
        or reference_provenance.get("version") != "0.0.9"
        or reference_provenance.get("source_commit") != TWEEDIE_REFERENCE_COMMIT
        or not isinstance(training_provenance, dict)
        or training_provenance.get("source_commit") != TWEEDIEGP_SOURCE_COMMIT
        or not all(
            isinstance(provenance.get(name), str)
            and len(str(provenance[name])) == 64
            for provenance in (reference_provenance, training_provenance)
            for name in ("file_sha256",)
        )
    ):
        return False
    return True


def _report_semantics_pass(report: dict[str, object]) -> bool:
    precision = report.get("precision")
    monotonicity = report.get("monotonicity")
    failures = report.get("failed_parameter_regions")
    if (
        not _report_structure_is_complete(report)
        or not isinstance(precision, dict)
        or not isinstance(monotonicity, dict)
        or not isinstance(failures, list)
    ):
        return False
    try:
        summaries_pass = all(
            _passes(precision[precision_name][oracle_name])
            for precision_name in ("float64", "float32")
            for oracle_name in ("reference", "compound_poisson_gamma")
        )
        monotone = all(
            int(monotonicity[precision_name][metric]) == 0
            for precision_name in ("float64", "float32")
            for metric in (
                "cdf",
                "quantile",
                "cdf_failures",
                "quantile_failures",
                "zero_identity",
                "zero_identity_failures",
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(summaries_pass and monotone and not failures)


def _report_passes(report: dict[str, object]) -> bool:
    digest = report.get("report_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != _validation_report_hash(report)
    ):
        return False
    return _report_semantics_pass(report)


def _validate_cdf_values(cdf: torch.Tensor) -> None:
    if not bool(torch.isfinite(cdf).all()):
        raise FloatingPointError("nonfinite CDF")
    if bool(((cdf < 0.0) | (cdf > 1.0)).any()):
        raise FloatingPointError("CDF values must lie in [0, 1]")


def tweedie_validation_gate(report: dict[str, object]) -> dict[str, object]:
    """Derive the branch-local status from metrics rather than caller labels."""
    passed = _report_passes(report)
    expected_branch = "PASS" if passed else "TWEEDIE_BRANCH_BLOCKED_HARD"
    expected_gate = {
        "branch": expected_branch,
        "confirmatory_eligible": passed,
        "hard_block": not passed,
    }
    if passed and (
        report.get("branch") != expected_branch
        or report.get("confirmatory_eligible") is not True
        or report.get("gate") != expected_gate
    ):
        passed = False
    branch = "PASS" if passed else "TWEEDIE_BRANCH_BLOCKED_HARD"
    return {
        "branch": branch,
        "confirmatory_eligible": passed,
        "hard_block": not passed,
    }


def validate_tweedie_against_oracles() -> dict[str, object]:
    """Evaluate all 600 frozen rows in float64 and execution float32.

    Both the PyPI reference and the structurally independent CP-Gamma oracle
    are separately gated against the training density.  No row is dropped and
    no density error is clamped.
    """
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    distribution_checks: list[dict[str, object]] = []
    try:
        reference_module = importlib.import_module("tweedie")
        reference_factory = _load_reference_factory()
        reference_path = Path(str(reference_module.__file__)).resolve()
        reference_provenance: dict[str, object] = {
            "package": "tweedie",
            "version": importlib.metadata.version("tweedie"),
            "source_commit": TWEEDIE_REFERENCE_COMMIT,
            "file_sha256": _file_sha256(reference_path),
        }
        reference_load_error = None
    except Exception as error:  # branch-local external reference failure
        reference_factory = None
        reference_load_error = f"{type(error).__name__}: {error}"
        reference_provenance = {
            "package": "tweedie",
            "version": None,
            "source_commit": TWEEDIE_REFERENCE_COMMIT,
            "file_sha256": None,
            "error": reference_load_error,
        }
    training_path = (
        Path(__file__).resolve().parent / "vendor" / "tweediegp" / "tweedie.py"
    )
    training_likelihood_provenance = {
        "repository": "https://github.com/StefanoDamato/TweedieGP",
        "source_commit": TWEEDIEGP_SOURCE_COMMIT,
        "file_sha256": _file_sha256(training_path),
        "license": "Apache-2.0",
    }
    differences: dict[str, dict[str, list[float | None]]] = {
        precision: {oracle: [] for oracle in ("reference", "compound_poisson_gamma")}
        for precision in ("float64", "float32")
    }
    zero_errors: dict[str, dict[str, list[float | None]]] = {
        precision: {oracle: [] for oracle in ("reference", "compound_poisson_gamma")}
        for precision in differences
    }
    finite = {precision: {oracle: [] for oracle in ("reference", "compound_poisson_gamma")} for precision in differences}
    monotonicity = {
        precision: {
            "cdf": 0,
            "quantile": 0,
            "cdf_failures": 0,
            "quantile_failures": 0,
            "zero_identity": 0,
            "zero_identity_failures": 0,
        }
        for precision in differences
    }
    for mu in MU_GRID:
        for phi in PHI_GRID:
            for p in P_GRID:
                ys = (0.0, 0.05 * mu, 0.25 * mu, mu, 2.0 * mu, 5.0 * mu)
                parameters: dict[str, TweedieDistribution | None] = {}
                for precision, dtype in (("float64", torch.float64), ("float32", torch.float32)):
                    cdf_query_values = [
                        float(torch.tensor(value, dtype=dtype).item()) for value in ys
                    ]
                    cdf_values: list[float] | None = None
                    cdf_error: str | None = None
                    p_zero_value: float | None = None
                    log_prob_zero_value: float | None = None
                    analytical_zero_mass: float | None = None
                    zero_identity_max_relative_error: float | None = None
                    zero_identity_error: str | None = None
                    quantile_probabilities = [
                        0.0,
                        0.01,
                        0.05,
                        0.1,
                        0.25,
                        0.5,
                        0.75,
                        0.9,
                        0.99,
                        1.0,
                    ]
                    quantile_values: list[float | None] | None = None
                    quantile_infinity_indices: list[int] = []
                    quantile_error: str | None = None
                    try:
                        dist = TweedieDistribution(
                            torch.tensor([[mu]], dtype=dtype),
                            torch.tensor([[phi]], dtype=dtype),
                            torch.tensor([[p]], dtype=dtype),
                        )
                    except Exception as error:
                        dist = None
                        failures.append(
                            {
                                "mu": mu, "phi": phi, "p": p, "y": None,
                                "precision": precision, "oracle": "training_distribution",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    parameters[precision] = dist
                    try:
                        if dist is None:
                            raise RuntimeError("training distribution construction failed")
                        cdf = dist.cdf(torch.tensor(ys, dtype=dtype).reshape(-1, 1, 1)).reshape(-1)
                        _validate_cdf_values(cdf)
                        monotonicity[precision]["cdf"] += int(torch.count_nonzero(cdf[1:] < cdf[:-1]).item())
                        cdf_values = [float(value) for value in cdf.detach().cpu().tolist()]
                    except Exception as error:
                        cdf_error = f"{type(error).__name__}: {error}"
                        monotonicity[precision]["cdf_failures"] += 1
                        failures.append(
                            {
                                "mu": mu, "phi": phi, "p": p, "y": None,
                                "precision": precision, "oracle": "cdf",
                                "error": cdf_error,
                            }
                        )
                    try:
                        if dist is None or cdf_values is None:
                            raise RuntimeError(
                                "training distribution or CDF zero value unavailable"
                            )
                        zero = torch.tensor([[0.0]], dtype=dtype)
                        p_zero_value = float(dist.p_zero().reshape(-1)[0].item())
                        log_prob_zero_value = float(dist.log_prob(zero).reshape(-1)[0].item())
                        effective_mu = float(torch.tensor(mu, dtype=dtype).item())
                        effective_phi = float(torch.tensor(phi, dtype=dtype).item())
                        effective_p = float(torch.tensor(p, dtype=dtype).item())
                        rate = effective_mu ** (2.0 - effective_p) / (
                            effective_phi * (2.0 - effective_p)
                        )
                        analytical_zero_mass = float(
                            torch.exp(torch.tensor(-rate, dtype=dtype)).item()
                        )
                        identity_values = np.asarray(
                            [
                                cdf_values[0],
                                p_zero_value,
                                math.exp(log_prob_zero_value),
                            ],
                            dtype=np.float64,
                        )
                        if (
                            not np.isfinite(identity_values).all()
                            or not math.isfinite(analytical_zero_mass)
                            or analytical_zero_mass < 0.0
                        ):
                            raise FloatingPointError("nonfinite Tweedie zero-mass identity")
                        denominator = max(
                            analytical_zero_mass, float(torch.finfo(dtype).tiny)
                        )
                        zero_identity_max_relative_error = float(
                            np.max(
                                np.abs(identity_values - analytical_zero_mass)
                                / denominator
                            )
                        )
                        monotonicity[precision]["zero_identity"] += int(
                            zero_identity_max_relative_error
                            > TOLERANCES["zero_relative_error"]
                        )
                    except Exception as error:
                        p_zero_value = None
                        log_prob_zero_value = None
                        analytical_zero_mass = None
                        zero_identity_max_relative_error = None
                        zero_identity_error = f"{type(error).__name__}: {error}"
                        monotonicity[precision]["zero_identity_failures"] += 1
                        failures.append(
                            {
                                "mu": mu,
                                "phi": phi,
                                "p": p,
                                "y": 0.0,
                                "precision": precision,
                                "oracle": "zero_mass_identity",
                                "error": zero_identity_error,
                            }
                        )
                    try:
                        if dist is None:
                            raise RuntimeError("training distribution construction failed")
                        qs = torch.tensor(quantile_probabilities, dtype=dtype)
                        quantiles = dist.quantile(qs).reshape(-1)
                        if (
                            bool(torch.isnan(quantiles).any())
                            or bool(torch.isneginf(quantiles).any())
                            or not bool(torch.isfinite(quantiles[:-1]).all())
                            or not bool(torch.isposinf(quantiles[-1]))
                            or bool((quantiles[:-1] < 0.0).any())
                        ):
                            raise FloatingPointError("invalid quantile endpoint or value")
                        monotonicity[precision]["quantile"] += int(torch.count_nonzero(quantiles[1:] < quantiles[:-1]).item())
                        quantile_values = [
                            float(value) for value in quantiles[:-1].detach().cpu().tolist()
                        ] + [None]
                        quantile_infinity_indices = [len(quantile_probabilities) - 1]
                    except Exception as error:
                        quantile_error = f"{type(error).__name__}: {error}"
                        monotonicity[precision]["quantile_failures"] += 1
                        failures.append(
                            {
                                "mu": mu, "phi": phi, "p": p, "y": None,
                                "precision": precision, "oracle": "quantile",
                                "error": quantile_error,
                            }
                        )
                    distribution_checks.append(
                        {
                            "mu": mu,
                            "phi": phi,
                            "p": p,
                            "precision": precision,
                            "cdf_query_values": cdf_query_values,
                            "cdf_values": cdf_values,
                            "cdf_error": cdf_error,
                            "p_zero_value": p_zero_value,
                            "log_prob_zero_value": log_prob_zero_value,
                            "analytical_zero_mass": analytical_zero_mass,
                            "zero_identity_max_relative_error": zero_identity_max_relative_error,
                            "zero_identity_error": zero_identity_error,
                            "quantile_probabilities": quantile_probabilities,
                            "quantile_values": quantile_values,
                            "quantile_positive_infinity_indices": quantile_infinity_indices,
                            "quantile_error": quantile_error,
                        }
                    )
                for y in ys:
                    expected_by_precision: dict[str, dict[str, float | None]] = {}
                    expected_errors: dict[str, dict[str, str | None]] = {}
                    for precision, dtype in (("float64", torch.float64), ("float32", torch.float32)):
                        effective_mu, effective_phi, effective_p = (
                            float(torch.tensor(value, dtype=dtype).item()) for value in (mu, phi, p)
                        )
                        effective_y = float(torch.tensor(y, dtype=dtype).item())
                        expected_by_precision[precision] = {}
                        expected_errors[precision] = {}
                        for oracle_name in ("reference", "compound_poisson_gamma"):
                            try:
                                if oracle_name == "reference":
                                    if reference_factory is None:
                                        raise ImportError(reference_load_error or "reference unavailable")
                                    value = float(
                                        np.asarray(
                                            reference_factory(
                                                p=effective_p, mu=effective_mu, phi=effective_phi
                                            ).logpdf(effective_y)
                                        ).item()
                                    )
                                else:
                                    value = float(
                                        compound_poisson_gamma_logpdf(
                                            effective_y,
                                            effective_mu,
                                            effective_phi,
                                            effective_p,
                                        ).item()
                                    )
                                if not math.isfinite(value):
                                    raise FloatingPointError("nonfinite oracle log density")
                                expected_by_precision[precision][oracle_name] = value
                                expected_errors[precision][oracle_name] = None
                            except Exception as error:
                                expected_by_precision[precision][oracle_name] = None
                                expected_errors[precision][oracle_name] = f"{type(error).__name__}: {error}"
                    row: dict[str, object] = {
                        "mu": mu, "phi": phi, "p": p, "y": y,
                        "mu_float32_execution": float(torch.tensor(mu, dtype=torch.float32).item()),
                        "phi_float32_execution": float(torch.tensor(phi, dtype=torch.float32).item()),
                        "p_float32_execution": float(torch.tensor(p, dtype=torch.float32).item()),
                        "y_float32_execution": float(torch.tensor(y, dtype=torch.float32).item()),
                        "reference_log_prob": expected_by_precision["float64"]["reference"],
                        "oracle_log_prob": expected_by_precision["float64"]["compound_poisson_gamma"],
                        "reference_log_prob_float32_parameters": expected_by_precision["float32"]["reference"],
                        "oracle_log_prob_float32_parameters": expected_by_precision["float32"]["compound_poisson_gamma"],
                    }
                    for precision, dtype in (("float64", torch.float64), ("float32", torch.float32)):
                        try:
                            distribution = parameters[precision]
                            if distribution is None:
                                raise RuntimeError("training distribution construction failed")
                            ours_value = float(
                                distribution.log_prob(torch.tensor([[y]], dtype=dtype)).item()
                            )
                            ours = ours_value if math.isfinite(ours_value) else None
                            ours_error = None if ours is not None else "nonfinite training log density"
                        except Exception as error:
                            ours = None
                            ours_error = f"{type(error).__name__}: {error}"
                        row[f"log_prob_{precision}"] = ours
                        for oracle_name, expected in expected_by_precision[precision].items():
                            is_finite = ours is not None and expected is not None
                            difference = abs(ours - expected) if is_finite else None
                            differences[precision][oracle_name].append(difference)
                            finite[precision][oracle_name].append(is_finite)
                            if y == 0.0:
                                if is_finite and math.exp(expected) > 0.0:
                                    zero_errors[precision][oracle_name].append(
                                        abs(math.exp(ours) - math.exp(expected)) / math.exp(expected)
                                    )
                                else:
                                    zero_errors[precision][oracle_name].append(None)
                            if not is_finite:
                                failures.append(
                                    {
                                        "mu": mu, "phi": phi, "p": p, "y": y,
                                        "precision": precision, "oracle": oracle_name,
                                        "error": ours_error or expected_errors[precision][oracle_name] or "nonfinite comparison",
                                    }
                                )
                    rows.append(row)
    precision: dict[str, dict[str, dict[str, object]]] = {}
    for precision_name in differences:
        precision[precision_name] = {}
        for oracle_name in differences[precision_name]:
            precision[precision_name][oracle_name] = _summary(
                differences[precision_name][oracle_name], zero_errors[precision_name][oracle_name], finite[precision_name][oracle_name], rows
            )
    reference64 = precision["float64"]["reference"]
    report: dict[str, object] = {
        "grid_points": len(MU_GRID) * len(PHI_GRID) * len(P_GRID),
        "comparison_count": len(rows),
        "tolerances": TOLERANCES,
        "precision": precision,
        "oracles": {
            "reference": {name: summaries["reference"] for name, summaries in precision.items()},
            "compound_poisson_gamma": {name: summaries["compound_poisson_gamma"] for name, summaries in precision.items()},
        },
        "reference_provenance": reference_provenance,
        "training_likelihood_provenance": training_likelihood_provenance,
        "distribution_checks": distribution_checks,
        "monotonicity": monotonicity,
        "finite_fraction": reference64["finite_fraction"],
        "zero_relative_error": reference64["zero_relative_error"],
        "median_abs_log_difference": reference64["median_abs_log_difference"],
        "p99_abs_log_difference": reference64["p99_abs_log_difference"],
        "cdf_monotonicity_violations": sum(values["cdf"] for values in monotonicity.values()),
        "quantile_monotonicity_violations": sum(values["quantile"] for values in monotonicity.values()),
        "reference_load_error": reference_load_error,
        "failed_parameter_regions": failures,
        "rows": rows,
    }
    passed = _report_semantics_pass(report)
    report["branch"] = "PASS" if passed else "TWEEDIE_BRANCH_BLOCKED_HARD"
    report["confirmatory_eligible"] = bool(passed)
    report["gate"] = {
        "branch": report["branch"],
        "confirmatory_eligible": bool(passed),
        "hard_block": not passed,
    }
    report["report_sha256"] = _validation_report_hash(report)
    return report
