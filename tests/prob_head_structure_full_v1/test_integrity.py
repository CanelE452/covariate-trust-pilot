from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import experiments.prob_head_structure_full_v1.integrity as integrity_module

from experiments.prob_head_structure_full_v1.integrity import (
    BranchEligibility,
    ContractViolation,
    GateStatus,
    audit_target_support,
    build_protected_manifest,
    derive_seed,
    freeze_protected_manifest,
    freeze_source_manifest,
    freeze_original_source_record,
    publish_completion_marker,
    recover_manifest_companion,
    reserve_or_resume_attempt,
    validate_count_target_support,
    validate_exact_count_likelihood_indices,
    validate_gate_dependencies,
    build_source_manifest,
    verify_source_manifest,
    verify_source_manifest_path,
    verify_original_source_record_path,
    validate_exact_split_contract,
    validate_split_contract,
    verify_protected_manifest,
    verify_protected_manifest_path,
)


class ProtectedManifestTests(unittest.TestCase):
    def test_freeze_manifest_refuses_to_replace_the_before_fit_baseline(self):
        """Catches replacing the immutable audit baseline after a result-sensitive event."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results" / "frozen").mkdir(parents=True)
            destination = root / "audit" / "before.json"
            frozen = freeze_protected_manifest(root, ["results/frozen"], destination)
            self.assertEqual(frozen["directories"]["results/frozen"]["file_count"], 0)
            with self.assertRaises(FileExistsError):
                freeze_protected_manifest(root, ["results/frozen"], destination)

    def test_manifest_records_relative_paths_bytes_hashes_and_detects_mutation(self):
        """Catches a manifest that omits file content or fails to bind its directory tree."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "results" / "immutable"
            protected.mkdir(parents=True)
            (protected / "b.txt").write_text("second", encoding="utf-8")
            (protected / "nested").mkdir()
            (protected / "nested" / "a.txt").write_text("first", encoding="utf-8")

            manifest = build_protected_manifest(root, ["results/immutable"])

            files = manifest["directories"]["results/immutable"]["files"]
            self.assertEqual(
                [item["path"] for item in files],
                ["results/immutable/b.txt", "results/immutable/nested/a.txt"],
            )
            self.assertEqual([item["bytes"] for item in files], [6, 5])
            self.assertEqual(manifest["directories"]["results/immutable"]["file_count"], 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in files))
            self.assertEqual(verify_protected_manifest(root, manifest)["status"], "PASS")

            (protected / "nested" / "a.txt").write_text("changed", encoding="utf-8")
            report = verify_protected_manifest(root, manifest)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["directories"]["results/immutable"]["unchanged"])

    def test_protected_manifest_rescans_tree_paths_and_types_after_hashing(self):
        """Catches a protected file appearing after enumeration but before manifest publication."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "results" / "immutable"
            protected.mkdir(parents=True)
            source = protected / "source.txt"
            source.write_text("stable", encoding="utf-8")
            real_hash = integrity_module._sha256_file

            def hash_then_mutate(path: Path) -> str:
                digest = real_hash(path)
                (protected / "late.txt").write_text("late", encoding="utf-8")
                return digest

            with patch.object(integrity_module, "_sha256_file", side_effect=hash_then_mutate):
                with self.assertRaisesRegex(ContractViolation, "changed while hashing"):
                    build_protected_manifest(root, ["results/immutable"])

    def test_protected_manifest_rescan_binds_the_protected_directory_itself(self):
        """Catches swapping an empty protected directory between the two tree scans."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "results" / "immutable"
            protected.mkdir(parents=True)
            real_snapshot = integrity_module._protected_tree_snapshot
            calls = 0

            def swap_root_on_second_scan(repository: Path, target: Path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    replaced = target.with_name("immutable-old")
                    target.rename(replaced)
                    target.mkdir()
                return real_snapshot(repository, target)

            with patch.object(
                integrity_module,
                "_protected_tree_snapshot",
                side_effect=swap_root_on_second_scan,
            ):
                with self.assertRaisesRegex(ContractViolation, "changed while hashing"):
                    build_protected_manifest(root, ["results/immutable"])

    def test_manifest_rejects_paths_that_escape_the_repository(self):
        """Catches path traversal that could hash a non-protected outside directory."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "repository-relative"):
                build_protected_manifest(Path(directory), ["../outside"])

    def test_protected_manifest_is_bound_to_a_labeled_repository_root(self):
        """Catches verifying identical relative files in the wrong checkout or worktree."""
        with tempfile.TemporaryDirectory() as left_directory, tempfile.TemporaryDirectory() as right_directory:
            left = Path(left_directory)
            right = Path(right_directory)
            for root in (left, right):
                (root / "results" / "frozen").mkdir(parents=True)
                (root / "results" / "frozen" / "a.txt").write_text("same", encoding="utf-8")
            manifest = build_protected_manifest(
                left,
                ["results/frozen"],
                repository_root_identity="isolated_execution_worktree",
            )
            self.assertEqual(manifest["repository_root_identity"]["label"], "isolated_execution_worktree")
            self.assertEqual(manifest["repository_root_identity"]["resolved_path"], left.resolve().as_posix())
            self.assertEqual(verify_protected_manifest(left, manifest)["status"], "PASS")
            with self.assertRaisesRegex(ContractViolation, "repository root"):
                verify_protected_manifest(right, manifest)

    def test_protected_and_source_freezes_require_the_original_companion_trust_root(self):
        """Catches recreating a missing companion and blessing replacement manifest bytes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results" / "frozen").mkdir(parents=True)
            (root / "results" / "frozen" / "a.txt").write_text("a", encoding="utf-8")
            (root / "source.txt").write_text("source", encoding="utf-8")

            protected_path = root / "audit" / "protected.json"
            source_path = root / "audit" / "source.json"
            freeze_protected_manifest(root, ["results/frozen"], protected_path)
            expected = hashlib.sha256((root / "source.txt").read_bytes()).hexdigest()
            freeze_source_manifest(root, {"source.txt": expected}, source_path)

            for manifest_path in (protected_path, source_path):
                companion_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256.json")
                binding = json.loads(companion_path.read_text(encoding="utf-8"))
                self.assertEqual(binding["relative_filename"], manifest_path.name)
                self.assertEqual(len(binding["sha256"]), 64)
                companion_path.unlink()
                with self.assertRaisesRegex(ContractViolation, "missing companion"):
                    recover_manifest_companion(manifest_path)
                companion_path.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ContractViolation, "companion mismatch"):
                    recover_manifest_companion(manifest_path)

    def test_manifest_recovery_rejects_a_self_consistent_replacement_without_companion(self):
        """Catches a new internally valid manifest becoming a trust root after the binding is lost."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.txt"
            replacement = root / "replacement.txt"
            original.write_text("original", encoding="utf-8")
            replacement.write_text("replacement", encoding="utf-8")
            path = root / "manifest.json"
            freeze_source_manifest(
                root,
                {"original.txt": hashlib.sha256(original.read_bytes()).hexdigest()},
                path,
            )
            path.with_suffix(path.suffix + ".sha256.json").unlink()
            replacement_manifest = build_source_manifest(
                root,
                {"replacement.txt": hashlib.sha256(replacement.read_bytes()).hexdigest()},
            )
            path.write_text(json.dumps(replacement_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "missing companion"):
                recover_manifest_companion(path)

    def test_path_verifiers_authenticate_companions_roots_and_current_bytes(self):
        """Catches production verification from an unauthenticated in-memory manifest replacement."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results" / "frozen").mkdir(parents=True)
            (root / "results" / "frozen" / "a.txt").write_text("a", encoding="utf-8")
            source = root / "source.txt"
            source.write_text("source", encoding="utf-8")
            protected_path = root / "audit" / "protected.json"
            source_path = root / "audit" / "source.json"
            freeze_protected_manifest(root, ["results/frozen"], protected_path, repository_root_identity="test-root")
            freeze_source_manifest(
                root,
                {"source.txt": hashlib.sha256(source.read_bytes()).hexdigest()},
                source_path,
                repository_root_identity="test-root",
            )
            self.assertTrue(verify_protected_manifest_path(root, protected_path)["all_unchanged"])
            self.assertTrue(verify_source_manifest_path(root, source_path)["all_unchanged"])

            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "source bytes changed"):
                verify_source_manifest_path(root, source_path)
            companion = protected_path.with_suffix(protected_path.suffix + ".sha256.json")
            companion.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "companion"):
                verify_protected_manifest_path(root, protected_path)

    def test_path_protected_verifier_normalizes_deleted_tree_to_the_hard_stop_token(self):
        """Catches a missing protected directory escaping as FileNotFoundError instead of the frozen hard stop."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "results" / "frozen"
            protected.mkdir(parents=True)
            manifest_path = root / "audit" / "protected.json"
            freeze_protected_manifest(root, ["results/frozen"], manifest_path)
            protected.rmdir()
            with self.assertRaisesRegex(
                ContractViolation,
                "EXISTING_ARTIFACT_MUTATION_HARD_STOP",
            ):
                verify_protected_manifest_path(root, manifest_path)

    def test_source_manifest_is_root_bound_and_requires_exact_expected_hash_mapping(self):
        """Catches accepting caller path lists without literal expected hashes or verifying the wrong root."""
        with tempfile.TemporaryDirectory() as left_directory, tempfile.TemporaryDirectory() as right_directory:
            left = Path(left_directory); right = Path(right_directory)
            for root in (left, right):
                (root / "source.txt").write_text("same", encoding="utf-8")
            digest = hashlib.sha256((left / "source.txt").read_bytes()).hexdigest()
            manifest = build_source_manifest(left, {"source.txt": digest}, repository_root_identity="snapshot-root")
            self.assertEqual(manifest["repository_root_identity"], {"label": "snapshot-root", "resolved_path": left.resolve().as_posix()})
            self.assertTrue(verify_source_manifest(left, manifest)["all_unchanged"])
            with self.assertRaisesRegex(ContractViolation, "repository root"):
                verify_source_manifest(right, manifest)
            with self.assertRaisesRegex(ContractViolation, "expected SHA"):
                build_source_manifest(left, {"source.txt": "0" * 64})
            with self.assertRaises(ContractViolation):
                build_source_manifest(left, ["source.txt"])

    def test_original_source_record_binds_absolute_paths_and_rehashes_them(self):
        """Catches substituting a new original path after snapshots have been copied."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.bin"; second = root / "second.bin"
            first.write_bytes(b"first"); second.write_bytes(b"second")
            expected = {
                first.resolve().as_posix(): hashlib.sha256(first.read_bytes()).hexdigest(),
                second.resolve().as_posix(): hashlib.sha256(second.read_bytes()).hexdigest(),
            }
            destination = root / "audit" / "original_sources.json"
            freeze_original_source_record(expected, destination)
            self.assertTrue(verify_original_source_record_path(destination)["all_unchanged"])
            first.write_bytes(b"changed")
            with self.assertRaisesRegex(ContractViolation, "DATA_SOURCE_MUTATION"):
                verify_original_source_record_path(destination)

    def test_manifest_rejects_linked_sources_and_malformed_baselines(self):
        """Catches hashing aliases or accepting a hand-edited manifest structure."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results" / "frozen").mkdir(parents=True)
            source = root / "source.txt"
            source.write_text("source", encoding="utf-8")
            linked = root / "linked.txt"
            os.link(source, linked)
            with self.assertRaisesRegex(ContractViolation, "linked"):
                build_source_manifest(root, {"source.txt": hashlib.sha256(b"source").hexdigest()})

            malformed = {
                "schema_version": 1,
                "protected_directories": ["results/frozen"],
                "directories": {"results/frozen": {"file_count": 1, "files": "bad"}},
                "aggregate_sha256": "0" * 64,
            }
            with self.assertRaises(ContractViolation):
                verify_protected_manifest(root, malformed)


class ProtocolContractTests(unittest.TestCase):
    def test_malformed_or_linked_completion_is_a_contract_violation(self):
        """Catches malformed completion JSON escaping as parser errors or a linked attempt being reused."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); attempt = root / "stage" / "attempt_0001"; attempt.mkdir(parents=True)
            (attempt / "completion.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(ContractViolation):
                reserve_or_resume_attempt(root, "stage")

    def test_strict_json_rejects_duplicate_keys_in_completion_and_manifest(self):
        """Catches parser last-key-wins behavior changing an authenticated integrity object."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "stage" / "attempt_0001"
            attempt.mkdir(parents=True)
            (attempt / "completion.json").write_text(
                '{"schema_version":1,"status":"COMPLETE","status":"COMPLETE","payload":{},"artifacts":[],"completion_payload_sha256":"' + "0" * 64 + '"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractViolation, "duplicate JSON key"):
                reserve_or_resume_attempt(root, "stage")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("source", encoding="utf-8")
            manifest_path = root / "source-manifest.json"
            freeze_source_manifest(
                root,
                {"source.txt": hashlib.sha256(source.read_bytes()).hexdigest()},
                manifest_path,
            )
            companion = manifest_path.with_suffix(manifest_path.suffix + ".sha256.json")
            companion.write_text(
                '{"schema_version":1,"schema_version":1,"algorithm":"SHA-256",'
                '"relative_filename":"source-manifest.json","sha256":"' + "0" * 64 + '"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractViolation, "duplicate JSON key"):
                verify_source_manifest_path(root, manifest_path)

    def test_strict_integrity_json_rejects_recursive_numeric_overflow(self):
        """Catches JSON exponent overflow becoming an accepted nested infinity."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "stage" / "attempt_0001"
            attempt.mkdir(parents=True)
            (attempt / "completion.json").write_text(
                '{"schema_version":1,"status":"COMPLETE","payload":{"nested":[{"x":1e999}]},'
                '"artifacts":[],"completion_payload_sha256":"' + "0" * 64 + '"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractViolation, "nonfinite"):
                reserve_or_resume_attempt(root, "stage")

    def test_completion_schema_and_artifact_entries_are_strictly_validated(self):
        """Catches wrong marker types, extra fields, duplicate paths, or unbound attempt files."""
        malformed_markers = [
            "not-json",
            json.dumps([]),
            json.dumps({"schema_version": True, "status": "COMPLETE", "payload": {}, "artifacts": []}),
            json.dumps({"schema_version": 1, "status": "COMPLETE", "payload": [], "artifacts": []}),
            json.dumps({"schema_version": 1, "status": "COMPLETE", "payload": {}, "artifacts": [{"path": 3, "bytes": 0, "sha256": "0" * 64}]}),
        ]
        for marker in malformed_markers:
            with self.subTest(marker=marker[:30]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                attempt = root / "stage" / "attempt_0001"
                attempt.mkdir(parents=True)
                (attempt / "completion.json").write_text(marker, encoding="utf-8")
                with self.assertRaises(ContractViolation):
                    reserve_or_resume_attempt(root, "stage")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "stage" / "attempt_0001"
            attempt.mkdir(parents=True)
            malformed_entry = {
                "schema_version": 1,
                "status": "COMPLETE",
                "payload": {},
                "artifacts": [{"path": 3, "bytes": 0, "sha256": "0" * 64}],
            }
            encoded = json.dumps(malformed_entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            malformed_entry["completion_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
            (attempt / "completion.json").write_text(json.dumps(malformed_entry), encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "completion artifact"):
                reserve_or_resume_attempt(root, "stage")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "stage" / "attempt_0001"
            attempt.mkdir(parents=True)
            nonfinite = {
                "schema_version": 1,
                "status": "COMPLETE",
                "payload": {"metric": float("nan")},
                "artifacts": [],
                "completion_payload_sha256": "0" * 64,
            }
            (attempt / "completion.json").write_text(json.dumps(nonfinite), encoding="utf-8")
            with self.assertRaises(ContractViolation):
                reserve_or_resume_attempt(root, "stage")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt, _ = reserve_or_resume_attempt(root, "stage")
            listed = attempt / "listed.txt"
            unlisted = attempt / "unlisted.txt"
            listed.write_text("listed", encoding="utf-8")
            unlisted.write_text("unlisted", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "all attempt files"):
                publish_completion_marker(attempt, {"stage": "stage"}, [listed])

    def test_linked_attempt_artifacts_are_rejected_on_publish_and_resume(self):
        """Catches an artifact alias whose content can change outside the append-only attempt."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt, _ = reserve_or_resume_attempt(root, "stage")
            outside = root / "outside.txt"
            outside.write_text("shared", encoding="utf-8")
            linked = attempt / "linked.txt"
            os.link(outside, linked)
            with self.assertRaisesRegex(ContractViolation, "linked"):
                publish_completion_marker(attempt, {"stage": "stage"}, [linked])

    def test_malformed_attempt_entries_stop_resume_instead_of_being_ignored(self):
        """Catches silently skipping malformed attempt names, files, or linked directories."""
        malformed = ["attempt_1", "attempt_00001", "attempt_abcd", "notes.txt"]
        for name in malformed:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stage = root / "stage"
                stage.mkdir()
                (stage / name).write_text("bad", encoding="utf-8")
                with self.assertRaisesRegex(ContractViolation, "malformed stage entry"):
                    reserve_or_resume_attempt(root, "stage")

    def test_incomplete_attempt_advances_and_any_corrupt_completion_blocks_resume(self):
        """Catches overwriting an incomplete attempt or bypassing an older corrupt completed attempt."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, resumed = reserve_or_resume_attempt(root, "stage")
            self.assertFalse(resumed)
            (first / "partial.txt").write_text("partial", encoding="utf-8")
            second, resumed = reserve_or_resume_attempt(root, "stage")
            self.assertFalse(resumed)
            self.assertEqual(second.name, "attempt_0002")
            (first / "completion.json").write_text("{}", encoding="utf-8")
            report = second / "report.txt"
            report.write_text("done", encoding="utf-8")
            publish_completion_marker(second, {"stage": "stage"}, [report])
            with self.assertRaises(ContractViolation):
                reserve_or_resume_attempt(root, "stage")

    def test_attempt_reservation_is_exclusive_under_concurrent_callers(self):
        """Catches a check-then-create race that gives two workers the same attempt directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ThreadPoolExecutor(max_workers=2) as pool:
                reservations = list(pool.map(lambda _: reserve_or_resume_attempt(root, "stage"), range(2)))
            paths = [path for path, resumed in reservations if not resumed]
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(set(paths)), 2)
            self.assertEqual({path.name for path in paths}, {"attempt_0001", "attempt_0002"})

    def test_source_manifest_and_exact_split_reject_drift_and_bad_geometry(self):
        """Catches source drift or an origin that leaks outside a declared exact evaluation range."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "data.txt"; source.write_text("one", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = build_source_manifest(root, {"data.txt": digest})
            self.assertTrue(verify_source_manifest(root, manifest)["all_unchanged"])
            source.write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "source bytes changed"):
                verify_source_manifest(root, manifest)
        validate_exact_split_contract({"model_train": [0, 10], "validation": [10, 12], "warmup": [12, 14], "evaluation": [14, 20]}, origins=[14, 16, 18], horizon=2, lookback=4, total_length=20, spacing=2)
        with self.assertRaises(ContractViolation):
            validate_exact_split_contract({"model_train": [0, 10], "validation": [10, 12], "warmup": [12, 14], "evaluation": [14, 20]}, origins=[19], horizon=2, lookback=4, total_length=20, spacing=2)
        with self.assertRaisesRegex(ContractViolation, "schema"):
            validate_exact_split_contract({"train": [0, 10], "validation": [10, 12], "warmup": [12, 14], "evaluation": [14, 20]}, origins=[14, 16, 18], horizon=2, lookback=4, total_length=20, spacing=2)

    def test_exact_split_rejects_gaps_wrong_lengths_unsorted_origins_and_unused_tail(self):
        """Catches a nominally disjoint split that does not exactly cover the frozen forecast geometry."""
        good = {"model_train": [0, 10], "validation": [10, 12], "warmup": [12, 14], "evaluation": [14, 20]}
        invalid_cases = [
            ({"model_train": [0, 10], "validation": [10, 11], "warmup": [11, 14], "evaluation": [14, 20]}, [14, 16, 18], 20),
            (good, [16, 14, 18], 20),
            (good, [14, 16], 20),
            (good, [14, 16, 18], 21),
        ]
        for intervals, origins, total_length in invalid_cases:
            with self.subTest(intervals=intervals, origins=origins, total_length=total_length):
                with self.assertRaises(ContractViolation):
                    validate_exact_split_contract(intervals, origins=origins, horizon=2, lookback=4, total_length=total_length, spacing=2)

    def test_gate_status_is_closed_and_diagnostic_lineage_cannot_regain_confirmatory_eligibility(self):
        """Catches an unknown status or downstream reconstruction that erases an upstream scientific failure."""
        with self.assertRaises(ValueError):
            GateStatus("S1", "almost_pass")
        first = BranchEligibility.begin("S2", [GateStatus.passed("DGP_BALANCE"), GateStatus.scientific_failure("S1")])
        later = BranchEligibility.begin("S3", [GateStatus.passed("S2")], prior=first)
        self.assertFalse(later.confirmatory_eligible)
        self.assertEqual(later.role, "DIAGNOSTIC_CONTINUATION_AFTER_S1")
        with self.assertRaises(ContractViolation):
            validate_gate_dependencies("S3", [GateStatus.passed("S2")])

    def test_branch_begin_validates_complete_lineage_and_rejects_forged_prior_state(self):
        """Catches bypassing dependency validation or constructing a contradictory prior eligibility."""
        with self.assertRaisesRegex(ContractViolation, "missing required"):
            BranchEligibility.begin("S3", [GateStatus.passed("S2")])
        with self.assertRaises(ContractViolation):
            BranchEligibility(
                branch="S2",
                confirmatory_eligible=True,
                role="CONFIRMATORY",
                upstream_required_gates=("S1",),
                upstream_gate_status=(("S1", "SCIENTIFIC_FAILURE"),),
            )
        with self.assertRaisesRegex(ContractViolation, "unknown branch"):
            BranchEligibility.begin("S33_typo", [])
        with self.assertRaisesRegex(ContractViolation, "exact ordered"):
            BranchEligibility(
                branch="S3",
                confirmatory_eligible=True,
                role="CONFIRMATORY",
                upstream_required_gates=(),
                upstream_gate_status=(),
            )
        with self.assertRaisesRegex(ContractViolation, "missing required"):
            BranchEligibility.begin("FINAL_HEAD", [GateStatus.passed("S1"), GateStatus.passed("S2")])
        b_final = BranchEligibility.begin(
            "FINAL_B",
            [
                GateStatus.passed("R1"), GateStatus.passed("R2"), GateStatus.passed("B1"),
                GateStatus.passed("B2"), GateStatus.passed("CONTROL_B"),
                GateStatus.passed("TWEEDIE_VALID"),
            ],
        )
        self.assertTrue(b_final.confirmatory_eligible)

    def test_seed_derivation_is_stable_and_target_support_rejects_noninteger_values(self):
        """Catches nondeterministic seed routing or accidental continuous/count support mixing."""
        self.assertEqual(derive_seed(20260905, "model", 0), derive_seed(20260905, "model", 0))
        self.assertNotEqual(derive_seed(20260905, "model", 0), derive_seed(20260905, "model", 1))
        audit = audit_target_support([0.0, 1.0, 2.0, 0.0])
        self.assertEqual(audit["zero_ratio"], 0.5)
        self.assertEqual(audit["positive_count"], 2)
        invalid = audit_target_support([0.0, 1.25])
        self.assertEqual(invalid["noninteger_count"], 1)
        with self.assertRaisesRegex(ContractViolation, "noninteger"):
            validate_count_target_support(invalid)

    def test_count_support_validator_requires_a_complete_consistent_nonempty_audit(self):
        """Catches hand-built partial or contradictory support summaries being treated as valid count data."""
        valid = audit_target_support([0.0, 0.0, 2.0, 4.0])
        validate_count_target_support(valid)
        self.assertEqual(valid["total_count"], 4)
        malformed = [
            {key: value for key, value in valid.items() if key != "max"},
            {**valid, "total_count": 0},
            {**valid, "min": -1.0},
            {**valid, "zero_ratio": 0.25},
            {**valid, "positive_count": 5},
            {**valid, "positive_mean": -1.0},
            {**valid, "positive_variance": float("nan")},
            {**valid, "positive_mean": 5.0},
            {**valid, "positive_variance": 99.0},
        ]
        for audit in malformed:
            with self.subTest(audit=audit):
                with self.assertRaises(ContractViolation):
                    validate_count_target_support(audit)

        singleton = audit_target_support([0.0, 3.0])
        for contradiction in (
            {**singleton, "positive_mean": 2.0},
            {**singleton, "positive_variance": 1.0},
        ):
            with self.assertRaises(ContractViolation):
                validate_count_target_support(contradiction)

        positive_only = audit_target_support([2.0, 4.0])
        with self.assertRaises(ContractViolation):
            validate_count_target_support({**positive_only, "positive_mean": 1.0})

    def test_exact_count_likelihood_indices_never_round_tolerance_audit_values(self):
        """Catches converting a near-integer observation into a PMF array index after the tolerant audit."""
        validate_exact_count_likelihood_indices([0.0, 1.0, 7.0])
        with self.assertRaisesRegex(
            ContractViolation,
            "COUNT_LIKELIHOOD_INDEX_AMBIGUITY_HARD_STOP",
        ):
            validate_exact_count_likelihood_indices([0.0, 1.0000001])
        with self.assertRaisesRegex(ContractViolation, "support mismatch"):
            validate_exact_count_likelihood_indices([0.0, 1.25])
        for values in ([float("nan")], [-1.0], []):
            with self.subTest(values=values):
                with self.assertRaises(ContractViolation):
                    validate_exact_count_likelihood_indices(values)

    def test_split_contract_rejects_overlap_and_gate_failure_marks_new_branch_diagnostic(self):
        """Catches target leakage through overlapping splits or a falsely confirmatory downstream branch."""
        with self.assertRaisesRegex(ContractViolation, "overlap"):
            validate_split_contract(
                {"train": [0, 10], "validation": [9, 12], "warmup": [12, 15]},
                origins=[15],
                horizon=3,
            )
        scientific_failure = GateStatus.scientific_failure("S1")
        branch = BranchEligibility.begin("S2", [GateStatus.passed("DGP_BALANCE"), scientific_failure])
        self.assertFalse(branch.confirmatory_eligible)
        self.assertEqual(branch.role, "DIAGNOSTIC_CONTINUATION_AFTER_S1")
        self.assertEqual(
            branch.as_dict(),
            {
                "branch": "S2", "upstream_required_gates": ["DGP_BALANCE", "S1"],
                "upstream_gate_status": {"DGP_BALANCE": "PASS", "S1": "SCIENTIFIC_FAILURE"},
                "confirmatory_eligible": False,
                "scientific_role": "DIAGNOSTIC_CONTINUATION_AFTER_S1",
            },
        )
        self.assertTrue(GateStatus.hard_failure("support").halts_execution)

    def test_completion_marker_is_last_and_a_successful_attempt_is_reused(self):
        """Catches rerunning a completed stage or accepting an unbound completion marker."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt, resumed = reserve_or_resume_attempt(root, "stage")
            self.assertFalse(resumed)
            artifact = attempt / "report.txt"
            artifact.write_text("complete", encoding="utf-8")
            completion = publish_completion_marker(attempt, {"stage": "stage"}, [artifact])
            self.assertEqual(completion["status"], "COMPLETE")
            second, resumed = reserve_or_resume_attempt(root, "stage")
            self.assertTrue(resumed)
            self.assertEqual(second, attempt)
            artifact.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "hash"):
                reserve_or_resume_attempt(root, "stage")


if __name__ == "__main__":
    unittest.main()
