from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from experiments.ph_online_memory_gono_v1.artifacts import (
    exclusive_torch_save,
    exclusive_write_json,
    exclusive_write_parquet,
    exclusive_write_text,
    verify_preregistration,
)
from experiments.ph_online_memory_gono_v1.prereg import (
    build_preregistered_spec,
    freeze_preregistration,
)


class AppendOnlyArtifactTests(unittest.TestCase):
    def test_text_collision_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "STATUS.md"
            written = exclusive_write_text(destination, "first\n")
            self.assertEqual(written, len("first\n".encode("utf-8")))
            with self.assertRaises(FileExistsError):
                exclusive_write_text(destination, "second\n")
            self.assertEqual(destination.read_text(encoding="utf-8"), "first\n")

    def test_json_collision_preserves_original_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "nested" / "report.json"
            exclusive_write_json(destination, {"version": 1})
            original = destination.read_bytes()

            with self.assertRaises(FileExistsError):
                exclusive_write_json(destination, {"version": 2})

            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])

    def test_parquet_and_checkpoint_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "values.parquet"
            checkpoint = root / "model.pt"
            frame = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})

            parquet_bytes = exclusive_write_parquet(parquet, frame)
            checkpoint_bytes = exclusive_torch_save(
                checkpoint, {"weight": torch.tensor([1.0, 2.0])}
            )
            self.assertEqual(parquet_bytes, parquet.stat().st_size)
            self.assertEqual(checkpoint_bytes, checkpoint.stat().st_size)
            pd.testing.assert_frame_equal(pd.read_parquet(parquet), frame)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(loaded["weight"], torch.tensor([1.0, 2.0])))

            with self.assertRaises(FileExistsError):
                exclusive_write_parquet(parquet, frame)
            with self.assertRaises(FileExistsError):
                exclusive_torch_save(checkpoint, {})

    def test_preregistration_verifier_rejects_payload_changes(self):
        spec = build_preregistered_spec(
            repository={"git_commit": "abc", "branch": "main", "dirty": True},
            environment={"python": "x"},
            implementation_sha256={"a.py": "123"},
            frozen_at_utc="2026-09-04T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preregistered_spec.json"
            freeze_preregistration(path, spec)
            verified = verify_preregistration(path)
            self.assertEqual(verified["experiment_name"], "PH-ONLINE-MEMORY-GONO-v1")

            raw = path.read_text(encoding="utf-8").replace(
                '"model_seed": 0', '"model_seed": 9'
            )
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "(?i)(hash|sha)"):
                verify_preregistration(path)


if __name__ == "__main__":
    unittest.main()
