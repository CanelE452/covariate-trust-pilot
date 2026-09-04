"""Immutable audit and execution contracts for PROB-HEAD-STRUCTURE-FULL-v1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


class ContractViolation(RuntimeError):
    """A hard integrity contract violation that must block its execution branch."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTEMPT_RE = re.compile(r"attempt_[0-9]{4}\Z")
_WINDOWS_REPARSE_POINT = 0x400


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_recursive_nonfinite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if isinstance(value, dict):
        for nested in value.values():
            _reject_recursive_nonfinite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_recursive_nonfinite_json(nested)


def _is_link_or_reparse(path: Path) -> bool:
    """Recognize POSIX links and Windows junction/reparse aliases without following them."""
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractViolation(f"cannot inspect path: {path}") from exc
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        _reject_recursive_nonfinite_json(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractViolation(f"{label} is unreadable or malformed JSON") from exc
    except ValueError as exc:
        detail = "duplicate JSON key" if "duplicate JSON key" in str(exc) else "nonfinite JSON number"
        raise ContractViolation(f"{label} contains {detail}") from exc
    if not isinstance(value, dict):
        raise ContractViolation(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    try:
        before = path.lstat()
        if _is_link_or_reparse(path) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ContractViolation("linked files are forbidden in manifests")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.lstat()
    except ContractViolation:
        raise
    except OSError as exc:
        raise ContractViolation(f"file could not be hashed: {path}") from exc
    if _is_link_or_reparse(path) or after.st_nlink != 1 or (
        before.st_ino,
        before.st_dev,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_ino,
        after.st_dev,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ContractViolation("file changed while hashing")
    return digest.hexdigest()


def _repository_relative_directory(repository_root: Path, raw: str) -> tuple[str, Path]:
    relative = Path(raw)
    repository = Path(repository_root).resolve()
    target = (repository / relative).resolve()
    if relative.is_absolute() or not relative.parts or not target.is_relative_to(repository):
        raise ValueError("protected directory must be repository-relative")
    normalized = target.relative_to(repository)
    if not normalized.parts or normalized.parts[0] != "results":
        raise ValueError("protected directory must be under results")
    raw_target = repository / relative
    cursor = repository
    for part in normalized.parts:
        cursor = cursor / part
        if _is_link_or_reparse(cursor):
            raise ContractViolation("linked protected directories are forbidden")
    if not target.is_dir():
        raise FileNotFoundError(target)
    return normalized.as_posix(), target


def _protected_tree_snapshot(repository: Path, target: Path) -> tuple[list[Path], dict[str, tuple[Any, ...]]]:
    entries = sorted(target.rglob("*"), key=lambda path: path.relative_to(repository).as_posix())
    if _is_link_or_reparse(target):
        raise ContractViolation("linked entries are forbidden in protected directories")
    target_metadata = target.lstat()
    if not stat.S_ISDIR(target_metadata.st_mode):
        raise ContractViolation("protected directory changed type while hashing")
    metadata: dict[str, tuple[Any, ...]] = {
        target.relative_to(repository).as_posix(): (
            "directory",
            target_metadata.st_ino,
            target_metadata.st_dev,
            target_metadata.st_size,
            target_metadata.st_mtime_ns,
            target_metadata.st_ctime_ns,
        )
    }
    for entry in entries:
        if _is_link_or_reparse(entry):
            raise ContractViolation("linked entries are forbidden in protected directories")
        observed = entry.lstat()
        if stat.S_ISREG(observed.st_mode):
            entry_type = "file"
            if observed.st_nlink != 1:
                raise ContractViolation("linked entries are forbidden in protected directories")
        elif stat.S_ISDIR(observed.st_mode):
            entry_type = "directory"
        else:
            raise ContractViolation("protected directories may contain only regular files and directories")
        metadata[entry.relative_to(repository).as_posix()] = (
            entry_type,
            observed.st_ino,
            observed.st_dev,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )
    return entries, metadata


def _directory_record(repository_root: Path, relative: str) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    _, target = _repository_relative_directory(repository, relative)
    entries, before_tree = _protected_tree_snapshot(repository, target)
    files = []
    for source in (path for path in entries if before_tree[path.relative_to(repository).as_posix()][0] == "file"):
        files.append(
            {
                "path": source.relative_to(repository).as_posix(),
                "bytes": source.lstat().st_size,
                "sha256": _sha256_file(source),
            }
        )
    _, after_tree = _protected_tree_snapshot(repository, target)
    if before_tree != after_tree:
        raise ContractViolation("protected tree changed while hashing")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def build_protected_manifest(
    repository_root: Path,
    protected_directories: Iterable[str],
    *,
    repository_root_identity: str = "repository_root",
) -> dict[str, Any]:
    """Record every protected file with portable paths and content hashes."""
    if not isinstance(repository_root_identity, str) or not repository_root_identity.strip():
        raise ValueError("repository root identity label must be non-empty")
    repository = Path(repository_root).resolve()
    entries = list(protected_directories)
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("protected directories must be a non-empty unique list")
    normalized = [_repository_relative_directory(repository, item)[0] for item in entries]
    if len(normalized) != len(set(normalized)):
        raise ValueError("protected directories must be unique after normalization")
    directories = {relative: _directory_record(repository, relative) for relative in normalized}
    root_identity = {"label": repository_root_identity, "resolved_path": repository.as_posix()}
    aggregate_bytes = _canonical_json_bytes({"repository_root_identity": root_identity, "directories": directories})
    return {
        "schema_version": 2,
        "repository_root_identity": root_identity,
        "protected_directories": normalized,
        "directories": directories,
        "aggregate_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
    }


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_file_records(records: object, *, prefix: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ContractViolation("manifest files must be a list")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise ContractViolation("manifest file record schema is invalid")
        path = raw["path"]
        byte_count = raw["bytes"]
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ContractViolation("manifest file path is invalid")
        if prefix is not None and not (path == prefix or path.startswith(prefix + "/")):
            raise ContractViolation("protected manifest file escapes its directory")
        if path in seen:
            raise ContractViolation("manifest file paths must be unique")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ContractViolation("manifest file byte count is invalid")
        if not _valid_sha256(raw["sha256"]):
            raise ContractViolation("manifest file SHA-256 is invalid")
        seen.add(path)
        validated.append(raw)
    if [item["path"] for item in validated] != sorted(seen):
        raise ContractViolation("manifest file records must be sorted")
    return validated


def _validate_protected_manifest_structure(baseline: Mapping[str, Any]) -> None:
    if not isinstance(baseline, dict) or set(baseline) != {
        "schema_version", "repository_root_identity", "protected_directories", "directories", "aggregate_sha256"
    }:
        raise ContractViolation("protected manifest schema is invalid")
    root_identity = baseline["repository_root_identity"]
    paths = baseline["protected_directories"]
    directories = baseline["directories"]
    if baseline["schema_version"] != 2 or isinstance(baseline["schema_version"], bool):
        raise ContractViolation("protected manifest schema version is invalid")
    if (
        not isinstance(root_identity, dict)
        or set(root_identity) != {"label", "resolved_path"}
        or not isinstance(root_identity["label"], str)
        or not root_identity["label"]
        or not isinstance(root_identity["resolved_path"], str)
        or not Path(root_identity["resolved_path"]).is_absolute()
    ):
        raise ContractViolation("protected manifest repository root identity is invalid")
    if not isinstance(paths, list) or not paths or not all(isinstance(item, str) for item in paths):
        raise ContractViolation("protected manifest directory list is invalid")
    if not isinstance(directories, dict):
        raise ContractViolation("protected manifest directory records do not match their list")
    if len(paths) != len(set(paths)) or paths != list(directories):
        raise ContractViolation("protected manifest directory records do not match their list")
    for path in paths:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "results" or relative.as_posix() != path:
            raise ContractViolation("protected manifest directory path is invalid")
    for path in paths:
        record = directories[path]
        if not isinstance(record, dict) or set(record) != {"file_count", "files", "aggregate_sha256"}:
            raise ContractViolation("protected manifest directory record schema is invalid")
        files = _validate_file_records(record["files"], prefix=path)
        count = record["file_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count != len(files):
            raise ContractViolation("protected manifest file count is invalid")
        digest = hashlib.sha256(_canonical_json_bytes_list(files)).hexdigest()
        if record["aggregate_sha256"] != digest:
            raise ContractViolation("protected manifest directory aggregate mismatch")
    aggregate = hashlib.sha256(_canonical_json_bytes({"repository_root_identity": root_identity, "directories": directories})).hexdigest()
    if baseline["aggregate_sha256"] != aggregate:
        raise ContractViolation("protected manifest aggregate mismatch")


def _validate_source_manifest_structure(baseline: Mapping[str, Any]) -> None:
    if not isinstance(baseline, dict) or set(baseline) != {
        "schema_version", "repository_root_identity", "files", "aggregate_sha256"
    }:
        raise ContractViolation("source manifest schema is invalid")
    if baseline["schema_version"] != 2 or isinstance(baseline["schema_version"], bool):
        raise ContractViolation("source manifest schema version is invalid")
    root_identity = baseline["repository_root_identity"]
    if (
        not isinstance(root_identity, dict)
        or set(root_identity) != {"label", "resolved_path"}
        or not isinstance(root_identity["label"], str)
        or not root_identity["label"].strip()
        or not isinstance(root_identity["resolved_path"], str)
        or not Path(root_identity["resolved_path"]).is_absolute()
    ):
        raise ContractViolation("source manifest repository root identity is invalid")
    files = _validate_file_records(baseline["files"])
    aggregate = hashlib.sha256(
        _canonical_json_bytes({"repository_root_identity": root_identity, "files": files})
    ).hexdigest()
    if baseline["aggregate_sha256"] != aggregate:
        raise ContractViolation("source manifest aggregate mismatch")


def _validate_original_source_record_structure(record: Mapping[str, Any]) -> None:
    if not isinstance(record, dict) or set(record) != {
        "schema_version", "record_type", "files", "aggregate_sha256"
    }:
        raise ContractViolation("original source record schema is invalid")
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool) or record["record_type"] != "ORIGINAL_SOURCE_PATHS":
        raise ContractViolation("original source record schema is invalid")
    files = record["files"]
    if not isinstance(files, list) or not files:
        raise ContractViolation("original source record files must be nonempty")
    seen: set[str] = set()
    for raw in files:
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise ContractViolation("original source record file schema is invalid")
        path = raw["path"]
        byte_count = raw["bytes"]
        if (
            not isinstance(path, str)
            or not path
            or not Path(path).is_absolute()
            or Path(path).resolve().as_posix() != path
            or path in seen
        ):
            raise ContractViolation("original source path is invalid")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0 or not _valid_sha256(raw["sha256"]):
            raise ContractViolation("original source file metadata is invalid")
        seen.add(path)
    if [item["path"] for item in files] != sorted(seen):
        raise ContractViolation("original source records must be sorted")
    aggregate = hashlib.sha256(
        _canonical_json_bytes({"record_type": record["record_type"], "files": files})
    ).hexdigest()
    if record["aggregate_sha256"] != aggregate:
        raise ContractViolation("original source record aggregate mismatch")


def verify_protected_manifest(repository_root: Path, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Compare a fresh manifest to the immutable before-fit baseline."""
    _validate_protected_manifest_structure(baseline)
    paths = baseline["protected_directories"]
    directories = baseline["directories"]
    expected_root = baseline["repository_root_identity"]
    actual_root = Path(repository_root).resolve().as_posix()
    if expected_root["resolved_path"] != actual_root:
        raise ContractViolation("protected manifest repository root mismatch")
    current = build_protected_manifest(
        repository_root,
        paths,
        repository_root_identity=expected_root["label"],
    )
    reports = {
        path: {
            "expected": directories[path],
            "current": current["directories"][path],
            "unchanged": directories[path] == current["directories"][path],
        }
        for path in paths
    }
    return {
        "status": "PASS" if all(item["unchanged"] for item in reports.values()) else "FAIL",
        "all_unchanged": all(item["unchanged"] for item in reports.values()),
        "directories": reports,
        "expected_aggregate_sha256": baseline.get("aggregate_sha256"),
        "current_aggregate_sha256": current["aggregate_sha256"],
    }


def freeze_protected_manifest(
    repository_root: Path,
    protected_directories: Iterable[str],
    destination: Path,
    *,
    repository_root_identity: str = "repository_root",
) -> dict[str, Any]:
    """Build and publish the before-fit baseline exactly once."""
    manifest = build_protected_manifest(
        repository_root,
        protected_directories,
        repository_root_identity=repository_root_identity,
    )
    _freeze_manifest(Path(destination), manifest)
    return manifest


def recover_manifest_companion(manifest_path: Path) -> dict[str, Any]:
    """Verify an existing companion; loss of the original binding is a hard stop."""
    source = Path(manifest_path)
    payload = _read_json_object(source, label="manifest")
    if "protected_directories" in payload:
        _validate_protected_manifest_structure(payload)
    elif payload.get("record_type") == "ORIGINAL_SOURCE_PATHS":
        _validate_original_source_record_structure(payload)
    elif "files" in payload:
        _validate_source_manifest_structure(payload)
    else:
        raise ContractViolation("manifest recovery requires a valid baseline object")
    companion = source.with_suffix(source.suffix + ".sha256.json")
    if not os.path.lexists(companion):
        raise ContractViolation("missing companion is a hard stop; recovery cannot establish a trust root")
    try:
        return _verify_manifest_companion(source)
    except ContractViolation as exc:
        raise ContractViolation("manifest companion mismatch") from exc


def _verify_manifest_companion(manifest_path: Path) -> dict[str, Any]:
    source = Path(manifest_path)
    _sha256_file(source)
    companion = source.with_suffix(source.suffix + ".sha256.json")
    if not os.path.lexists(companion):
        raise ContractViolation("manifest companion is missing")
    _sha256_file(companion)
    actual = _read_json_object(companion, label="manifest companion")
    expected = {
        "schema_version": 1,
        "algorithm": "SHA-256",
        "relative_filename": source.name,
        "sha256": _sha256_file(source),
    }
    if actual != expected:
        raise ContractViolation("manifest companion mismatch")
    return actual


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a reproducible child seed from explicit, NUL-separated identity parts."""
    if not isinstance(master_seed, int):
        raise TypeError("master seed must be an integer")
    encoded = "\0".join([str(master_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], byteorder="little", signed=False)


def audit_target_support(values: Iterable[float]) -> dict[str, float | int]:
    """Audit the frozen nonnegative integer target contract without mutating inputs."""
    observed = [float(value) for value in values]
    if not observed:
        raise ContractViolation("target support is empty")
    nan_count = sum(not math.isfinite(value) for value in observed)
    negative_count = sum(math.isfinite(value) and value < 0 for value in observed)
    noninteger_count = sum(
        math.isfinite(value) and abs(value - round(value)) > 1e-6 for value in observed
    )
    finite = [value for value in observed if math.isfinite(value)]
    positives = [value for value in finite if value > 0]
    mean = sum(positives) / len(positives) if positives else 0.0
    variance = sum((value - mean) ** 2 for value in positives) / len(positives) if positives else 0.0
    return {
        "total_count": len(observed),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "nan_count": nan_count,
        "negative_count": negative_count,
        "noninteger_count": noninteger_count,
        "zero_ratio": sum(value == 0 for value in observed) / len(observed),
        "positive_count": len(positives),
        "positive_mean": mean,
        "positive_variance": variance,
    }


def validate_count_target_support(audit: Mapping[str, float | int | None]) -> None:
    """Turn a recorded count-target audit into a hard stop only at the execution boundary."""
    required = {
        "total_count", "min", "max", "nan_count", "negative_count", "noninteger_count",
        "zero_ratio", "positive_count", "positive_mean", "positive_variance",
    }
    if not isinstance(audit, Mapping) or set(audit) != required:
        raise ContractViolation("target support audit schema is incomplete or has extra fields")
    count_fields = ("total_count", "nan_count", "negative_count", "noninteger_count", "positive_count")
    if any(isinstance(audit[field], bool) or not isinstance(audit[field], int) or audit[field] < 0 for field in count_fields):
        raise ContractViolation("target support audit counts are invalid")
    total_count = audit["total_count"]
    if total_count <= 0 or audit["positive_count"] > total_count:
        raise ContractViolation("target support audit must describe a nonempty sample")
    numeric_fields = ("min", "max", "zero_ratio", "positive_mean", "positive_variance")
    if any(isinstance(audit[field], bool) or not isinstance(audit[field], (int, float)) or not math.isfinite(float(audit[field])) for field in numeric_fields):
        raise ContractViolation("target support audit numeric fields are invalid")
    minimum = float(audit["min"])
    maximum = float(audit["max"])
    zero_ratio = float(audit["zero_ratio"])
    positive_mean = float(audit["positive_mean"])
    positive_variance = float(audit["positive_variance"])
    if minimum < 0 or maximum < minimum or not 0 <= zero_ratio <= 1 or positive_mean < 0 or positive_variance < 0:
        raise ContractViolation("target support audit values are internally inconsistent")
    for field, message in (
        ("nan_count", "NaN or nonfinite"),
        ("negative_count", "negative"),
        ("noninteger_count", "noninteger"),
    ):
        if audit[field] > 0:
            raise ContractViolation(f"target support contains {message} values")
    positive_count = audit["positive_count"]
    zero_count = total_count - positive_count
    if not math.isclose(zero_ratio, zero_count / total_count, rel_tol=0.0, abs_tol=1e-12):
        raise ContractViolation("target support zero ratio contradicts its counts")
    if abs(minimum - round(minimum)) > 1e-6 or abs(maximum - round(maximum)) > 1e-6:
        raise ContractViolation("target support extrema contradict integer support")
    if positive_count == 0:
        if maximum != 0 or positive_mean != 0 or positive_variance != 0:
            raise ContractViolation("all-zero target support summary is inconsistent")
    else:
        positive_minimum = minimum if zero_count == 0 else 1.0
        if maximum <= 0 or not positive_minimum <= positive_mean <= maximum:
            raise ContractViolation("positive target support summary is inconsistent")
        variance_upper_bound = (maximum - positive_mean) * (positive_mean - positive_minimum)
        if positive_variance > variance_upper_bound + max(1e-12, abs(variance_upper_bound) * 1e-12):
            raise ContractViolation("positive target variance contradicts its extrema and mean")
        if positive_count == 1 and (
            not math.isclose(positive_mean, maximum, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(positive_variance, 0.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ContractViolation("singleton positive target summary is inconsistent")
    if zero_count == 0 and minimum <= 0:
        raise ContractViolation("positive-only target support summary is inconsistent")
    if zero_count > 0 and minimum != 0:
        raise ContractViolation("zero-containing target support summary is inconsistent")


def validate_exact_count_likelihood_indices(values: Iterable[float]) -> None:
    """Require literal integer-valued observations before any discrete PMF indexing."""
    observed = list(values)
    if not observed:
        raise ContractViolation("count likelihood support mismatch: empty target")
    for raw in observed:
        if isinstance(raw, (bool, str, bytes)):
            raise ContractViolation("count likelihood support mismatch: nonnumeric target")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractViolation("count likelihood support mismatch: nonnumeric target") from exc
        if not math.isfinite(value) or value < 0:
            raise ContractViolation("count likelihood support mismatch: target must be finite and nonnegative")
        if value.is_integer():
            continue
        if abs(value - round(value)) <= 1e-6:
            raise ContractViolation("COUNT_LIKELIHOOD_INDEX_AMBIGUITY_HARD_STOP")
        raise ContractViolation("count likelihood support mismatch: target is not an exact integer")


def validate_split_contract(
    intervals: Mapping[str, Iterable[int]], *, origins: Iterable[int], horizon: int
) -> None:
    """Reject split overlap and origins that would use warmup targets as training data."""
    if horizon <= 0:
        raise ContractViolation("horizon must be positive")
    normalized: list[tuple[str, int, int]] = []
    for name, raw in intervals.items():
        bounds = list(raw)
        if len(bounds) != 2 or not all(isinstance(value, int) for value in bounds):
            raise ContractViolation(f"split {name} must be an integer half-open interval")
        start, end = bounds
        if start < 0 or end <= start:
            raise ContractViolation(f"split {name} is invalid")
        normalized.append((name, start, end))
    for index, (left_name, left_start, left_end) in enumerate(normalized):
        for right_name, right_start, right_end in normalized[index + 1 :]:
            if max(left_start, right_start) < min(left_end, right_end):
                raise ContractViolation(f"split overlap: {left_name} and {right_name}")
    warmup = next((end for name, _, end in normalized if name == "warmup"), None)
    if warmup is None:
        raise ContractViolation("warmup split is required")
    for origin in origins:
        if not isinstance(origin, int) or origin < warmup:
            raise ContractViolation("evaluation origin precedes completed warmup")


def validate_exact_split_contract(intervals: Mapping[str, Iterable[int]], *, origins: Iterable[int], horizon: int, lookback: int, total_length: int, spacing: int | None = None) -> None:
    """Enforce complete named chronological half-open geometry before target windows exist."""
    names = ("model_train", "validation", "warmup", "evaluation")
    integer_scalars = (horizon, lookback, total_length)
    if (
        set(intervals) != set(names)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in integer_scalars)
        or horizon <= 0
        or lookback <= 0
        or total_length <= 0
        or (spacing is not None and (isinstance(spacing, bool) or not isinstance(spacing, int) or spacing <= 0))
    ):
        raise ContractViolation("exact split schema is invalid")
    bounds = {name: tuple(intervals[name]) for name in names}
    if any(
        len(value) != 2
        or not all(isinstance(x, int) and not isinstance(x, bool) for x in value)
        or value[0] < 0
        or value[1] <= value[0]
        for value in bounds.values()
    ):
        raise ContractViolation("exact split bounds are invalid")
    if bounds["model_train"][0] != 0 or bounds["model_train"][1] < lookback or bounds["evaluation"][1] != total_length:
        raise ContractViolation("exact split dataset bounds are invalid")
    if any(bounds[a][1] != bounds[b][0] for a, b in zip(names, names[1:])):
        raise ContractViolation("exact split intervals must be adjacent")
    if bounds["validation"][1] - bounds["validation"][0] != horizon or bounds["warmup"][1] - bounds["warmup"][0] != horizon:
        raise ContractViolation("validation and warmup must each span one horizon")
    observed = list(origins)
    if (
        not observed
        or not all(isinstance(origin, int) and not isinstance(origin, bool) for origin in observed)
        or observed != sorted(set(observed))
        or observed[0] != bounds["evaluation"][0]
        or observed[-1] + horizon != bounds["evaluation"][1]
        or any(origin + horizon > bounds["evaluation"][1] for origin in observed)
    ):
        raise ContractViolation("evaluation origins are outside their declared range")
    required_spacing = horizon if spacing is None else spacing
    if observed[0] < lookback or any(b - a != required_spacing for a, b in zip(observed, observed[1:])):
        raise ContractViolation("evaluation origin lookback or spacing is invalid")


def build_source_manifest(
    repository_root: Path,
    expected_sha256_by_path: Mapping[str, str],
    *,
    repository_root_identity: str = "repository_root",
) -> dict[str, Any]:
    """Bind exact expected source bytes under one labeled repository root."""
    if not isinstance(repository_root_identity, str) or not repository_root_identity.strip():
        raise ContractViolation("source manifest repository root label must be nonempty")
    if not isinstance(expected_sha256_by_path, Mapping) or not expected_sha256_by_path:
        raise ContractViolation("source manifest requires a nonempty path-to-expected-SHA mapping")
    if not all(isinstance(path, str) and _valid_sha256(digest) for path, digest in expected_sha256_by_path.items()):
        raise ContractViolation("source manifest expected SHA mapping is invalid")
    root = Path(repository_root).resolve()
    records: list[dict[str, Any]] = []
    normalized_seen: set[str] = set()
    for raw, expected_sha256 in expected_sha256_by_path.items():
        relative = Path(raw)
        source = (root / relative).resolve()
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or not source.is_relative_to(root):
            raise ContractViolation("source manifest path is invalid")
        normalized = source.relative_to(root).as_posix()
        if normalized in normalized_seen:
            raise ContractViolation("source manifest paths must be unique after normalization")
        cursor = root
        for part in source.relative_to(root).parts:
            cursor = cursor / part
            if not os.path.lexists(cursor) or _is_link_or_reparse(cursor):
                raise ContractViolation("linked or missing source manifest paths are forbidden")
        if not source.is_file():
            raise ContractViolation("source manifest path is invalid")
        actual_sha256 = _sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise ContractViolation(f"source expected SHA mismatch: {normalized}")
        records.append({"path": normalized, "bytes": source.lstat().st_size, "sha256": actual_sha256})
        normalized_seen.add(normalized)
    records.sort(key=lambda item: item["path"])
    root_identity = {"label": repository_root_identity, "resolved_path": root.as_posix()}
    aggregate = hashlib.sha256(
        _canonical_json_bytes({"repository_root_identity": root_identity, "files": records})
    ).hexdigest()
    return {
        "schema_version": 2,
        "repository_root_identity": root_identity,
        "files": records,
        "aggregate_sha256": aggregate,
    }


def freeze_source_manifest(
    repository_root: Path,
    expected_sha256_by_path: Mapping[str, str],
    destination: Path,
    *,
    repository_root_identity: str = "repository_root",
) -> dict[str, Any]:
    """Build and independently bind a source manifest exactly once."""
    manifest = build_source_manifest(
        repository_root,
        expected_sha256_by_path,
        repository_root_identity=repository_root_identity,
    )
    _freeze_manifest(Path(destination), manifest)
    return manifest


def verify_source_manifest(repository_root: Path, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Compare every frozen source byte record before fits or resume."""
    _validate_source_manifest_structure(baseline)
    expected_root = baseline["repository_root_identity"]
    actual_root = Path(repository_root).resolve().as_posix()
    if expected_root["resolved_path"] != actual_root:
        raise ContractViolation("source manifest repository root mismatch")
    files = baseline["files"]
    try:
        current = build_source_manifest(
            repository_root,
            {item["path"]: item["sha256"] for item in files},
            repository_root_identity=expected_root["label"],
        )
    except ContractViolation as exc:
        raise ContractViolation("source bytes changed or became unavailable") from exc
    if current != baseline:
        raise ContractViolation("source bytes changed or became unavailable")
    return {"all_unchanged": True, "expected": baseline, "current": current}


def build_original_source_record(expected_sha256_by_absolute_path: Mapping[str, str]) -> dict[str, Any]:
    """Bind the six literal original input paths, including sources outside the worktree."""
    if not isinstance(expected_sha256_by_absolute_path, Mapping) or not expected_sha256_by_absolute_path:
        raise ContractViolation("original sources require a nonempty absolute-path-to-SHA mapping")
    records: list[dict[str, Any]] = []
    for raw, expected_sha256 in expected_sha256_by_absolute_path.items():
        if not isinstance(raw, str) or not _valid_sha256(expected_sha256):
            raise ContractViolation("original source mapping is invalid")
        source = Path(raw)
        if not source.is_absolute() or source.resolve().as_posix() != raw:
            raise ContractViolation("original source paths must be resolved absolute paths")
        if not os.path.lexists(source) or _is_link_or_reparse(source) or not source.is_file():
            raise ContractViolation("original source path is missing, linked, or not a regular file")
        actual_sha256 = _sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise ContractViolation(f"original source expected SHA mismatch: {raw}")
        records.append({"path": raw, "bytes": source.lstat().st_size, "sha256": actual_sha256})
    if len({item["path"] for item in records}) != len(records):
        raise ContractViolation("original source paths must be unique")
    records.sort(key=lambda item: item["path"])
    aggregate = hashlib.sha256(
        _canonical_json_bytes({"record_type": "ORIGINAL_SOURCE_PATHS", "files": records})
    ).hexdigest()
    return {
        "schema_version": 1,
        "record_type": "ORIGINAL_SOURCE_PATHS",
        "files": records,
        "aggregate_sha256": aggregate,
    }


def freeze_original_source_record(
    expected_sha256_by_absolute_path: Mapping[str, str], destination: Path
) -> dict[str, Any]:
    """Publish the exact original-source record and independent byte companion once."""
    record = build_original_source_record(expected_sha256_by_absolute_path)
    _freeze_manifest(Path(destination), record)
    return record


def verify_protected_manifest_path(repository_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Authenticate an on-disk protected baseline before comparing current bytes."""
    _verify_manifest_companion(Path(manifest_path))
    baseline = _read_json_object(Path(manifest_path), label="protected manifest")
    try:
        report = verify_protected_manifest(repository_root, baseline)
    except (ContractViolation, OSError, ValueError) as exc:
        raise ContractViolation("EXISTING_ARTIFACT_MUTATION_HARD_STOP") from exc
    if not report["all_unchanged"]:
        raise ContractViolation("EXISTING_ARTIFACT_MUTATION_HARD_STOP")
    return report


def verify_source_manifest_path(repository_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Authenticate an on-disk root-bound source manifest before rehashing sources."""
    _verify_manifest_companion(Path(manifest_path))
    baseline = _read_json_object(Path(manifest_path), label="source manifest")
    return verify_source_manifest(repository_root, baseline)


def verify_original_source_record_path(record_path: Path) -> dict[str, Any]:
    """Authenticate the frozen absolute-path record and rehash those exact originals."""
    source = Path(record_path)
    _verify_manifest_companion(source)
    record = _read_json_object(source, label="original source record")
    _validate_original_source_record_structure(record)
    current: list[dict[str, Any]] = []
    try:
        for frozen in record["files"]:
            path = Path(frozen["path"])
            if path.resolve().as_posix() != frozen["path"] or not os.path.lexists(path):
                raise ContractViolation("original path substitution or disappearance")
            actual = {"path": frozen["path"], "bytes": path.lstat().st_size, "sha256": _sha256_file(path)}
            if actual != frozen:
                raise ContractViolation("original source bytes changed")
            current.append(actual)
    except (OSError, ContractViolation) as exc:
        raise ContractViolation("DATA_SOURCE_MUTATION hard stop") from exc
    return {"all_unchanged": True, "expected": record["files"], "current": current}


@dataclass(frozen=True)
class GateStatus:
    gate: str
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.gate, str) or not self.gate.strip():
            raise ValueError("gate name must be a non-empty string")
        if self.status not in {"PASS", "SCIENTIFIC_FAILURE", "HARD_FAILURE"}:
            raise ValueError(f"unknown gate status: {self.status}")

    @property
    def halts_execution(self) -> bool:
        return self.status == "HARD_FAILURE"

    @classmethod
    def passed(cls, gate: str) -> "GateStatus":
        return cls(gate=gate, status="PASS")

    @classmethod
    def scientific_failure(cls, gate: str) -> "GateStatus":
        return cls(gate=gate, status="SCIENTIFIC_FAILURE")

    @classmethod
    def hard_failure(cls, gate: str) -> "GateStatus":
        return cls(gate=gate, status="HARD_FAILURE")


@dataclass(frozen=True)
class BranchEligibility:
    branch: str
    confirmatory_eligible: bool
    role: str
    upstream_required_gates: tuple[str, ...]
    upstream_gate_status: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.branch, str) or not self.branch:
            raise ContractViolation("branch eligibility requires a branch name")
        required = _REQUIRED_GATES.get(self.branch)
        if required is None:
            raise ContractViolation(f"unknown branch in gate lineage: {self.branch}")
        gates = [item[0] for item in self.upstream_gate_status]
        statuses = [item[1] for item in self.upstream_gate_status]
        if len(gates) != len(set(gates)) or tuple(gates) != self.upstream_required_gates:
            raise ContractViolation("branch eligibility gate lineage is inconsistent")
        if self.upstream_required_gates != required:
            raise ContractViolation("branch eligibility must contain the exact ordered registered dependencies")
        if any(status not in {"PASS", "SCIENTIFIC_FAILURE"} for status in statuses):
            raise ContractViolation("branch eligibility contains an unknown or hard status")
        failed = [gate for gate, status in self.upstream_gate_status if status == "SCIENTIFIC_FAILURE"]
        if self.confirmatory_eligible != (not failed):
            raise ContractViolation("branch eligibility contradicts upstream gate status")
        if self.confirmatory_eligible and self.role != "CONFIRMATORY":
            raise ContractViolation("confirmatory branch role is invalid")
        if failed and not (
            self.role.startswith("DIAGNOSTIC_CONTINUATION_AFTER_")
            and self.role.removeprefix("DIAGNOSTIC_CONTINUATION_AFTER_") in failed
        ):
            raise ContractViolation("diagnostic branch role is invalid")

    @classmethod
    def begin(cls, branch: str, upstream: Iterable[GateStatus], prior: "BranchEligibility | None" = None) -> "BranchEligibility":
        statuses = list(upstream)
        if not all(isinstance(status, GateStatus) for status in statuses):
            raise ContractViolation("upstream gate statuses must be GateStatus values")
        hard = next((status for status in statuses if status.halts_execution), None)
        if hard is not None:
            raise ContractViolation(f"hard failure in upstream gate {hard.gate} blocks {branch}")
        combined = list(prior.upstream_gate_status if prior is not None else ())
        by_gate = dict(combined)
        for status in statuses:
            if status.gate in by_gate and by_gate[status.gate] != status.status:
                raise ContractViolation(f"conflicting status for upstream gate {status.gate}")
            if status.gate not in by_gate:
                combined.append((status.gate, status.status))
                by_gate[status.gate] = status.status
        validate_gate_dependencies(branch, [GateStatus(gate, status) for gate, status in combined])
        failed = [gate for gate, status in combined if status == "SCIENTIFIC_FAILURE"]
        if prior is not None and not prior.confirmatory_eligible:
            return cls(branch, False, prior.role, tuple(gate for gate, _ in combined), tuple(combined))
        if failed:
            return cls(branch, False, f"DIAGNOSTIC_CONTINUATION_AFTER_{failed[0]}", tuple(gate for gate, _ in combined), tuple(combined))
        return cls(branch, True, "CONFIRMATORY", tuple(gate for gate, _ in combined), tuple(combined))

    def as_dict(self) -> dict[str, Any]:
        """Persist the eligibility fixed at branch start; later stages must not mutate it."""
        return {
            "branch": self.branch,
            "upstream_required_gates": list(self.upstream_required_gates),
            "upstream_gate_status": dict(self.upstream_gate_status),
            "confirmatory_eligible": self.confirmatory_eligible,
            "scientific_role": self.role,
        }


_REQUIRED_GATES = {
    "DGP_BALANCE": (),
    "S1": ("DGP_BALANCE",),
    "S2": ("DGP_BALANCE", "S1"),
    "S3": ("DGP_BALANCE", "S1", "S2"),
    "FINAL_HEAD": ("DGP_BALANCE", "S1", "S2", "S3"),
    "R1": (),
    "R2": ("R1",),
    "R3": ("R1", "R2"),
    "FINAL_REAL": ("R1", "R2", "R3", "TWEEDIE_VALID"),
    "A1": ("R2", "R3"),
    "A2": ("R2", "R3", "A1"),
    "A3": ("R2", "R3", "A1", "A2"),
    "A4": ("R2", "R3", "A1", "A2", "A3"),
    "CONTROL_A": ("R2", "R3", "A1", "A2", "A3", "A4"),
    "FINAL_A": ("R1", "R2", "R3", "A1", "A2", "A3", "A4", "CONTROL_A", "TWEEDIE_VALID"),
    "A_DISTRIBUTION_DISTILLATION": ("R1", "R2", "R3", "A1", "A2", "A3", "A4", "CONTROL_A", "TWEEDIE_VALID"),
    "B1": ("R2",),
    "B2": ("R2", "B1"),
    "CONTROL_B": ("R2", "B1", "B2"),
    "FINAL_B": ("R1", "R2", "B1", "B2", "CONTROL_B", "TWEEDIE_VALID"),
    "B_STRUCTURE_CONDITIONED_ROUTING": ("R1", "R2", "B1", "B2", "CONTROL_B", "TWEEDIE_VALID"),
    "C1": ("R1",),
    "C2": (),
    "C3": ("R1", "C1", "C2"),
    "CONTROL_C": ("R1", "C1", "C2", "C3"),
    "FINAL_C": ("R1", "C1", "C2", "C3", "CONTROL_C", "TWEEDIE_VALID"),
    "C_DISAGREEMENT_SENSOR": ("R1", "C1", "C2", "C3", "CONTROL_C", "TWEEDIE_VALID"),
    "TWEEDIE_VALID": (),
}


def validate_gate_dependencies(branch: str, upstream: Iterable[GateStatus]) -> None:
    """Require the complete named prerequisite ledger before a dependent branch begins."""
    if branch not in _REQUIRED_GATES:
        raise ContractViolation(f"unknown branch in gate lineage: {branch}")
    required = _REQUIRED_GATES[branch]
    observed_sequence = tuple(status.gate for status in upstream)
    missing = set(required).difference(observed_sequence)
    if missing:
        raise ContractViolation(f"missing required upstream gates for {branch}: {sorted(missing)}")
    if observed_sequence != required:
        raise ContractViolation(f"upstream gates for {branch} must match the exact ordered dependencies")

def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _canonical_json_bytes_list(payload: list[dict[str, Any]]) -> bytes:
    return _canonical_json_bytes(payload)


def _exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _freeze_manifest(destination: Path, manifest: Mapping[str, Any]) -> None:
    target = Path(destination)
    companion = target.with_suffix(target.suffix + ".sha256.json")
    if os.path.lexists(target) or os.path.lexists(companion):
        raise FileExistsError(target if os.path.lexists(target) else companion)
    _exclusive_write_json(target, manifest)
    record = {
        "schema_version": 1,
        "algorithm": "SHA-256",
        "relative_filename": target.name,
        "sha256": _sha256_file(target),
    }
    _exclusive_write_json(companion, record)


def _stage_root(root: Path, stage: str) -> Path:
    if (
        not isinstance(stage, str)
        or not stage
        or Path(stage).name != stage
        or stage in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stage)
    ):
        raise ContractViolation("stage must be a simple relative name")
    raw_root = Path(root)
    raw_root.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(raw_root) or not raw_root.is_dir():
        raise ContractViolation("attempt root must be an unlinked directory")
    resolved_root = raw_root.resolve()
    raw_target = raw_root / stage
    if os.path.lexists(raw_target):
        if _is_link_or_reparse(raw_target) or not raw_target.is_dir():
            raise ContractViolation("stage root must be an unlinked directory")
    else:
        try:
            raw_target.mkdir()
        except FileExistsError:
            if _is_link_or_reparse(raw_target) or not raw_target.is_dir():
                raise ContractViolation("stage root must be an unlinked directory")
    target = raw_target.resolve()
    if not target.is_relative_to(resolved_root):
        raise ContractViolation("stage root escapes the attempt root")
    return target


def _completion_path(attempt: Path) -> Path:
    return Path(attempt) / "completion.json"


def _validate_attempt_directory(attempt: Path) -> Path:
    raw = Path(attempt)
    if not _ATTEMPT_RE.fullmatch(raw.name) or int(raw.name[-4:]) < 1:
        raise ContractViolation("malformed attempt directory name")
    if not os.path.lexists(raw) or _is_link_or_reparse(raw) or not raw.is_dir():
        raise ContractViolation("attempt must be an unlinked directory")
    resolved = raw.resolve()
    if resolved.parent != raw.parent.resolve():
        raise ContractViolation("attempt directory escapes its stage")
    return resolved


def _attempt_entries(attempt: Path) -> list[Path]:
    root = _validate_attempt_directory(attempt)
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for entry in entries:
        if _is_link_or_reparse(entry):
            raise ContractViolation("linked attempt artifacts are forbidden")
        metadata = entry.lstat()
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ContractViolation("linked attempt artifacts are forbidden")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ContractViolation("attempts may contain only regular files and directories")
    return entries


def _artifact_record(root: Path, source: Path) -> dict[str, Any]:
    resolved = Path(source).resolve()
    if not resolved.is_relative_to(root) or resolved == _completion_path(root):
        raise ContractViolation("completion artifact path is invalid")
    digest = _sha256_file(resolved)
    metadata = resolved.lstat()
    return {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": metadata.st_size,
        "sha256": digest,
    }


def _validate_completion_artifact_record(attempt: Path, raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
        raise ContractViolation("completion artifact entry schema is invalid")
    relative_text = raw["path"]
    byte_count = raw["bytes"]
    if (
        not isinstance(relative_text, str)
        or not relative_text
        or Path(relative_text).is_absolute()
        or ".." in Path(relative_text).parts
        or Path(relative_text).as_posix() != relative_text
        or relative_text == "completion.json"
    ):
        raise ContractViolation("completion artifact path is invalid")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ContractViolation("completion artifact byte count is invalid")
    if not _valid_sha256(raw["sha256"]):
        raise ContractViolation("completion artifact SHA-256 is invalid")
    source = attempt / Path(relative_text)
    if not os.path.lexists(source):
        raise ContractViolation("completion artifact path is invalid")
    actual = _artifact_record(attempt, source)
    if raw != actual:
        raise ContractViolation("completion artifact hash mismatch")
    return raw


def _verify_completion(attempt: Path) -> dict[str, Any]:
    root = _validate_attempt_directory(attempt)
    _attempt_entries(root)
    completion_path = _completion_path(root)
    if not os.path.lexists(completion_path):
        raise ContractViolation("completion marker is missing")
    _sha256_file(completion_path)
    completion = _read_json_object(completion_path, label="completion marker")
    if set(completion) != {
        "schema_version", "status", "payload", "artifacts", "completion_payload_sha256"
    }:
        raise ContractViolation("completion marker schema is invalid")
    if completion["schema_version"] != 1 or isinstance(completion["schema_version"], bool):
        raise ContractViolation("completion marker schema version is invalid")
    if completion["status"] != "COMPLETE" or not isinstance(completion["payload"], dict) or not isinstance(completion["artifacts"], list):
        raise ContractViolation("completion marker is invalid")
    expected = completion["completion_payload_sha256"]
    if not _valid_sha256(expected):
        raise ContractViolation("completion payload hash is invalid")
    unhashed = dict(completion)
    unhashed.pop("completion_payload_sha256", None)
    try:
        actual_completion_hash = hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ContractViolation("completion payload is not finite JSON") from exc
    if expected != actual_completion_hash:
        raise ContractViolation("completion payload hash mismatch")
    artifacts = [_validate_completion_artifact_record(root, raw) for raw in completion["artifacts"]]
    paths = [record["path"] for record in artifacts]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContractViolation("completion artifact entries must be sorted and unique")
    actual_paths = [
        entry.relative_to(root).as_posix()
        for entry in _attempt_entries(root)
        if entry.is_file() and entry != completion_path
    ]
    if paths != actual_paths:
        raise ContractViolation("completion marker must bind all attempt files")
    return completion


def verify_completed_attempt(attempt: Path) -> dict[str, Any]:
    """Authenticate a completed append-only attempt from its on-disk marker and artifacts."""
    return _verify_completion(Path(attempt))


def _list_attempt_directories(stage_root: Path) -> list[Path]:
    attempts: list[Path] = []
    for entry in sorted(stage_root.iterdir(), key=lambda path: path.name):
        if not _ATTEMPT_RE.fullmatch(entry.name) or int(entry.name[-4:]) < 1 or _is_link_or_reparse(entry) or not entry.is_dir():
            raise ContractViolation(f"malformed stage entry: {entry.name}")
        _attempt_entries(entry)
        completion = _completion_path(entry)
        if os.path.lexists(completion) and (not completion.is_file() or _is_link_or_reparse(completion)):
            raise ContractViolation("completion marker is not a regular file")
        attempts.append(entry.resolve())
    return attempts


def reserve_or_resume_attempt(root: Path, stage: str) -> tuple[Path, bool]:
    """Reuse only a hash-verified successful attempt; otherwise reserve the next ID."""
    stage_root = _stage_root(root, stage)
    for _ in range(10_000):
        attempts = _list_attempt_directories(stage_root)
        completed: list[Path] = []
        for attempt in attempts:
            if os.path.lexists(_completion_path(attempt)):
                _verify_completion(attempt)
                completed.append(attempt)
        if completed:
            return completed[0], True
        next_number = max((int(path.name[-4:]) for path in attempts), default=0) + 1
        if next_number > 9999:
            raise ContractViolation("attempt identifier space exhausted")
        attempt = stage_root / f"attempt_{next_number:04d}"
        try:
            attempt.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return attempt.resolve(), False
    raise ContractViolation("could not reserve an exclusive attempt directory")


def publish_completion_marker(
    attempt: Path, payload: Mapping[str, Any], artifacts: Iterable[Path]
) -> dict[str, Any]:
    """Publish a hash-bound completion marker last, without replacing any attempt file."""
    root = _validate_attempt_directory(Path(attempt))
    if os.path.lexists(_completion_path(root)):
        raise FileExistsError(_completion_path(root))
    if not isinstance(payload, Mapping):
        raise ContractViolation("completion payload must be an object")
    records: list[dict[str, Any]] = []
    for artifact in artifacts:
        source = Path(artifact)
        if not os.path.lexists(source):
            raise ContractViolation("completion artifacts must be files inside their attempt")
        records.append(_artifact_record(root, source))
    records.sort(key=lambda record: record["path"])
    if len({record["path"] for record in records}) != len(records):
        raise ContractViolation("completion artifacts must be unique")
    actual_paths = [
        entry.relative_to(root).as_posix()
        for entry in _attempt_entries(root)
        if entry.is_file()
    ]
    if [record["path"] for record in records] != actual_paths:
        raise ContractViolation("completion marker must bind all attempt files")
    completion: dict[str, Any] = {"schema_version": 1, "status": "COMPLETE", "payload": dict(payload), "artifacts": records}
    try:
        completion["completion_payload_sha256"] = hashlib.sha256(_canonical_json_bytes(completion)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ContractViolation("completion payload is not finite JSON") from exc
    _exclusive_write_json(_completion_path(root), completion)
    return _verify_completion(root)
