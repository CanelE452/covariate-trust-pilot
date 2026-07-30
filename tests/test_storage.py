"""Storage checks (tests 32-34 plus supporting cases)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from covariate_trust.storage import (RUN_SUBDIRS, atomic_write_json, atomic_write_parquet,
                                     atomic_write_text, completed_task_hashes, create_run_dir,
                                     read_all_parts, write_task_part)


def test_32_atomic_write_leaves_no_temporary_file(tmp_path):
    """Test 32: writes are atomic - content lands in place, no .tmp remains."""
    p = atomic_write_text(tmp_path / "a.txt", "hello")
    assert p.read_text() == "hello"
    assert not (tmp_path / "a.txt.tmp").exists()

    df = pd.DataFrame({"a": [1, 2, 3]})
    q = atomic_write_parquet(tmp_path / "t.parquet", df)
    assert pd.read_parquet(q).equals(df)
    assert not (tmp_path / "t.parquet.tmp").exists()

    j = atomic_write_json(tmp_path / "m.json", {"k": 1})
    assert "\"k\": 1" in j.read_text()
    assert list(tmp_path.glob("*.tmp")) == []


def test_32b_failed_write_does_not_destroy_the_previous_file(tmp_path):
    path = tmp_path / "keep.parquet"
    atomic_write_parquet(path, pd.DataFrame({"a": [1]}))
    with pytest.raises(Exception):
        atomic_write_parquet(path, "not a dataframe")  # type: ignore[arg-type]
    assert pd.read_parquet(path)["a"].tolist() == [1]


def test_33_resume_skips_completed_tasks(tmp_path):
    """Test 33: completed task hashes are discovered from the parts directory."""
    run_dir = create_run_dir(tmp_path, "diagnostic")
    assert completed_task_hashes(run_dir) == set()
    write_task_part(run_dir, "abc123", pd.DataFrame({"h_index": [1], "q0.5": [0.4]}))
    write_task_part(run_dir, "def456", pd.DataFrame({"h_index": [1], "q0.5": [0.6]}))
    assert completed_task_hashes(run_dir) == {"abc123", "def456"}
    merged = read_all_parts(run_dir)
    assert len(merged) == 2
    assert sorted(merged["q0.5"].tolist()) == [0.4, 0.6]


def test_34_run_directories_are_never_overwritten(tmp_path):
    """Test 34: two runs in the same second still get distinct directories."""
    now = datetime(2026, 7, 30, 12, 0, 0)
    a = create_run_dir(tmp_path, "diagnostic", now)
    b = create_run_dir(tmp_path, "diagnostic", now)
    assert a != b
    assert a.exists() and b.exists()
    for sub in RUN_SUBDIRS:
        assert (a / sub).is_dir()
    assert (a / "predictions" / "parts").is_dir()


def test_34b_existing_artifacts_survive_a_second_run(tmp_path):
    now = datetime(2026, 7, 30, 12, 0, 0)
    a = create_run_dir(tmp_path, "diagnostic", now)
    atomic_write_text(a / "reports" / "report.md", "first run")
    create_run_dir(tmp_path, "diagnostic", now)
    assert (a / "reports" / "report.md").read_text() == "first run"
