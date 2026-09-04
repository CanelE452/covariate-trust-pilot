"""Read-only verification of the frozen result directories from Phase 0."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


BASELINE_EXTENSION_NAME = "forbidden_artifact_baseline_extension.json"
EXPERIMENT = "PH-ONLINE-MEMORY-GONO-v1"


_POWERSHELL_FINGERPRINT = r"""
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = [IO.Path]::GetFullPath($env:PH_GONO_REPO_ROOT)
$target = [IO.Path]::GetFullPath($env:PH_GONO_TARGET_ROOT)
$files = @(Get-ChildItem -LiteralPath $target -Recurse -File | Sort-Object FullName)
$records = @()
foreach ($file in $files) {
    $relative = [IO.Path]::GetRelativePath($repo, $file.FullName)
    $fileHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash
    $records += "$relative`t$fileHash"
}
$joined = $records -join "`n"
$bytes = [Text.Encoding]::UTF8.GetBytes($joined)
$hasher = [Security.Cryptography.SHA256]::Create()
try {
    $digest = [BitConverter]::ToString($hasher.ComputeHash($bytes)).Replace('-', '')
} finally {
    $hasher.Dispose()
}
@{file_count = $files.Count; sha256 = $digest} | ConvertTo-Json -Compress
"""


def _powershell_directory_fingerprint(
    repository_root: Path, target_root: Path
) -> dict[str, Any]:
    """Reproduce the exact Windows/PowerShell Phase-0 manifest ordering."""

    repository = Path(repository_root).resolve()
    target = Path(target_root).resolve()
    if not target.is_dir():
        raise FileNotFoundError(target)
    executable = (
        shutil.which("pwsh.exe")
        or shutil.which("pwsh")
        or shutil.which("powershell.exe")
        or shutil.which("powershell")
    )
    if executable is None:
        raise RuntimeError("PowerShell is required to reproduce the Phase 0 hash")
    environment = os.environ.copy()
    environment["PH_GONO_REPO_ROOT"] = str(repository)
    environment["PH_GONO_TARGET_ROOT"] = str(target)
    completed = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            _POWERSHELL_FINGERPRINT,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=repository,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"PowerShell fingerprint failed: {detail}")
    payload = json.loads(completed.stdout)
    return {
        "file_count": int(payload["file_count"]),
        "sha256": str(payload["sha256"]).upper(),
    }


def verify_forbidden_artifacts(
    repository_root: Path, audit_path: Path
) -> dict[str, Any]:
    """Compare all forbidden directories with the immutable Phase-0 baseline."""

    repository = Path(repository_root).resolve()
    with Path(audit_path).open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    baseline = audit.get("forbidden_artifact_directory_hashes_before")
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("audit has no forbidden artifact baseline")
    baseline = dict(baseline)

    extension_path = Path(audit_path).parent / BASELINE_EXTENSION_NAME
    extension: dict[str, Any] | None = None
    if extension_path.is_file():
        with extension_path.open("r", encoding="utf-8") as handle:
            extension = json.load(handle)
        prior = extension.get("prior_audit")
        if not isinstance(prior, dict):
            raise ValueError("forbidden baseline extension lacks prior audit binding")
        actual_audit_sha = hashlib.sha256(Path(audit_path).read_bytes()).hexdigest().upper()
        if str(prior.get("sha256", "")).upper() != actual_audit_sha:
            raise ValueError("forbidden baseline extension audit SHA256 mismatch")
        additional = extension.get(
            "additional_forbidden_artifact_directory_hashes_before"
        )
        if not isinstance(additional, dict) or not additional:
            raise ValueError("forbidden baseline extension has no additional hashes")
        overlap = sorted(set(baseline).intersection(additional))
        if overlap:
            raise ValueError(f"forbidden baseline extension duplicates {overlap}")
        baseline.update(additional)
        required = extension.get("required_exact_directories")
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)
            or len(required) != len(set(required))
            or set(required) != set(baseline)
        ):
            raise ValueError(
                "forbidden baseline extension does not exactly cover its required directories"
            )
    elif audit.get("experiment_name") == EXPERIMENT:
        raise FileNotFoundError(
            f"required forbidden baseline extension is missing: {extension_path}"
        )

    reports: dict[str, dict[str, Any]] = {}
    for relative_name in sorted(baseline):
        relative = Path(relative_name)
        target = (repository / relative).resolve()
        if (
            relative.is_absolute()
            or not target.is_relative_to(repository)
            or not relative.parts
            or relative.parts[0].lower() != "results"
        ):
            raise ValueError("forbidden target must be repository-relative under results")
        expected = baseline[relative_name]
        if not isinstance(expected, dict):
            raise ValueError("invalid forbidden artifact baseline entry")
        expected_normalized = {
            "file_count": int(expected["file_count"]),
            "sha256": str(expected["sha256"]).upper(),
        }
        current = _powershell_directory_fingerprint(repository, target)
        reports[str(relative_name)] = {
            "expected": expected_normalized,
            "current": current,
            "unchanged": current == expected_normalized,
        }
    all_unchanged = all(item["unchanged"] for item in reports.values())
    return {
        "status": "PASS" if all_unchanged else "FAIL",
        "all_unchanged": all_unchanged,
        "method": (
            "PowerShell Sort-Object FullName; repo-relative native path, TAB, "
            "uppercase file SHA256; LF join without trailing LF; UTF-8 SHA256"
        ),
        "baseline_sources": {
            "audit": str(Path(audit_path)),
            "extension": None if extension is None else str(extension_path),
        },
        "required_exact_directories": sorted(baseline),
        "directories": reports,
    }


__all__ = [
    "BASELINE_EXTENSION_NAME",
    "verify_forbidden_artifacts",
]
