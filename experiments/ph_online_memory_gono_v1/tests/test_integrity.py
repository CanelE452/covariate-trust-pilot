from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ph_online_memory_gono_v1.integrity import (
    _powershell_directory_fingerprint,
    verify_forbidden_artifacts,
)


class FrozenDirectoryIntegrityTests(unittest.TestCase):
    def test_matches_exact_audit_algorithm_and_detects_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "results" / "FrozenSet"
            frozen.mkdir(parents=True)
            (frozen / "a_file.txt").write_text("one", encoding="utf-8")
            (frozen / "B-file.txt").write_text("two", encoding="utf-8")
            relative = "results/FrozenSet"
            baseline = _powershell_directory_fingerprint(root, frozen)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {"forbidden_artifact_directory_hashes_before": {relative: baseline}}
                ),
                encoding="utf-8",
            )

            passing = verify_forbidden_artifacts(root, audit)
            self.assertEqual(passing["status"], "PASS")
            self.assertTrue(passing["all_unchanged"])

            (frozen / "a_file.txt").write_text("changed", encoding="utf-8")
            failing = verify_forbidden_artifacts(root, audit)
            self.assertEqual(failing["status"], "FAIL")
            self.assertFalse(failing["all_unchanged"])
            self.assertFalse(failing["directories"][relative]["unchanged"])

    def test_rejects_audit_target_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "forbidden_artifact_directory_hashes_before": {
                            "../escape": {"file_count": 0, "sha256": "0" * 64}
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "(?i)(repository|relative|target)"):
                verify_forbidden_artifacts(root, audit)

    def test_merges_audit_bound_extension_and_requires_exact_directory_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "results" / "first"
            second = root / "results" / "second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "a.txt").write_text("a", encoding="utf-8")
            (second / "b.txt").write_text("b", encoding="utf-8")
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "forbidden_artifact_directory_hashes_before": {
                            "results/first": _powershell_directory_fingerprint(
                                root, first
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            extension = root / "forbidden_artifact_baseline_extension.json"
            extension.write_text(
                json.dumps(
                    {
                        "prior_audit": {
                            "sha256": hashlib.sha256(audit.read_bytes())
                            .hexdigest()
                            .upper()
                        },
                        "required_exact_directories": [
                            "results/first",
                            "results/second",
                        ],
                        "additional_forbidden_artifact_directory_hashes_before": {
                            "results/second": _powershell_directory_fingerprint(
                                root, second
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )

            passing = verify_forbidden_artifacts(root, audit)
            self.assertEqual(
                set(passing["directories"]), {"results/first", "results/second"}
            )
            self.assertIsNotNone(passing["baseline_sources"]["extension"])

            payload = json.loads(extension.read_text(encoding="utf-8"))
            payload["required_exact_directories"].pop()
            extension.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                verify_forbidden_artifacts(root, audit)


if __name__ == "__main__":
    unittest.main()
