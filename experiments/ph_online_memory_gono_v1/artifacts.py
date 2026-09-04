"""Append-only JSON and hashing helpers for the PH memory pilot."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd
import torch


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _exclusive_write_bytes(Path(path), encoded)


def exclusive_write_text(path: Path, text: str) -> int:
    """Publish UTF-8 text atomically without replacing an existing file."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    encoded = text.encode("utf-8")
    _exclusive_write_bytes(Path(path), encoded)
    return len(encoded)


def _exclusive_write_bytes(path: Path, payload: bytes) -> None:
    """Durably publish complete bytes without ever replacing an existing path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Publishing a hard link is atomic and fails if destination already
        # exists on both NTFS and POSIX filesystems.  The temporary name is
        # removed afterwards, leaving the same inode at the final path.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_write_parquet(path: Path, frame: pd.DataFrame) -> int:
    """Serialize a DataFrame and create ``path`` atomically with no overwrite."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    _exclusive_write_bytes(Path(path), payload)
    return len(payload)


def exclusive_torch_save(path: Path, payload: object) -> int:
    """Serialize a torch payload and create ``path`` with no overwrite."""

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    encoded = buffer.getvalue()
    _exclusive_write_bytes(Path(path), encoded)
    return len(encoded)


def verify_preregistration(path: Path) -> dict[str, Any]:
    """Load a frozen preregistration only when its self-excluding hash matches."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    if not isinstance(stored, dict):
        raise ValueError("preregistration must be a JSON object")
    expected = stored.get("preregistration_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("preregistration SHA-256 is missing or invalid")
    unhashed = dict(stored)
    unhashed.pop("preregistration_sha256", None)
    unhashed.pop("preregistration_hash_contract", None)
    actual = payload_sha256(unhashed)
    if actual != expected:
        raise ValueError(
            f"preregistration SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return stored
