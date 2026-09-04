from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

from experiments.prob_head_structure_full_v1 import data
from experiments.prob_head_structure_full_v1 import integrity
from experiments.prob_head_structure_full_v1 import synthetic


class SyntheticDGPTests(unittest.TestCase):
    def setUp(self):
        self.runtime_decision = {
            "runtime_decision_sha256": "d" * 64,
            "runtime_tier": "MINIMAL-COMPLETE",
            "preregistration_payload_sha256": "p" * 64,
            "tier_contract": {
                "synthetic_data_seeds": [2026090501],
                "synthetic_series_per_cell": 24,
            },
        }
        self.preregistration_path = Path("verified-preregistered-spec-v4.json")
        self.runtime_verifier = mock.patch.object(
            synthetic,
            "verify_runtime_tier_decision",
            return_value=self.runtime_decision,
        )
        self.runtime_verifier.start()

    def tearDown(self):
        self.runtime_verifier.stop()

    def _audit(self, rows):
        return synthetic.audit_dgp_balance(
            rows,
            runtime_decision=self.runtime_decision,
            preregistration_path=self.preregistration_path,
        )

    def _grid_blocks(
        self,
        *,
        n_series: int = 24,
        seed: int = 2026090501,
        d_values: tuple[int, ...] = synthetic.SUPPORTED_D,
    ):
        blocks = []
        for d in d_values:
            base = synthetic.build_common_base(d=d, n_series=n_series, seed=seed)
            for rho_interval in synthetic.RHO_VALUES:
                for rho_magnitude in synthetic.RHO_VALUES:
                    block = synthetic.transform_common_base(
                        base,
                        rho_interval=rho_interval,
                        rho_magnitude=rho_magnitude,
                    )
                    blocks.append(block)
        return blocks

    def test_common_base_streams_are_rho_independent_but_markov_states_change(self):
        """Catches rho leaking into base innovations or non-Markov gap support."""
        base = synthetic.build_common_base(d=4, n_series=2, seed=20260905)

        persistent = synthetic.transform_common_base(
            base, rho_interval=0.8, rho_magnitude=-0.8
        )
        alternating = synthetic.transform_common_base(
            base, rho_interval=-0.8, rho_magnitude=-0.8
        )

        self.assertEqual(persistent["split"].train, (0, 380))
        self.assertEqual(persistent["split"].validation, (380, 408))
        self.assertEqual(persistent["split"].warmup, (408, 436))
        self.assertEqual(persistent["split"].origins, (436, 464, 492, 520, 548))
        self.assertEqual(persistent["split"].horizon, 28)
        self.assertEqual(persistent["split"].lookback, 96)
        self.assertEqual(persistent["data_seed"], 20260905)
        self.assertEqual(persistent["base_id"], "synthetic_d4_seed20260905")
        self.assertEqual(persistent["n_series"], 2)
        self.assertEqual(persistent["series_length"], synthetic.SERIES_LENGTH)
        self.assertEqual(persistent["base_series_index"].tolist(), [0, 1])
        self.assertEqual(
            persistent["base_innovation_id"].tolist(),
            [
                "synthetic_d4_seed20260905_series000000",
                "synthetic_d4_seed20260905_series000001",
            ],
        )
        np.testing.assert_array_equal(
            persistent["base_innovation_id"], alternating["base_innovation_id"]
        )
        self.assertEqual(persistent["rho_I"], 0.8)
        self.assertEqual(persistent["rho_M"], -0.8)
        self.assertEqual(
            persistent["generator_argument_adapter"],
            {"rho_interval": "rho_I", "rho_magnitude": "rho_M"},
        )
        self.assertIn("base_stream_fingerprint", persistent)
        self.assertIn("base_identity_sha256", persistent)
        self.assertEqual(
            persistent["base_identity_sha256"], alternating["base_identity_sha256"]
        )
        self.assertEqual(
            persistent["base_stream_fingerprint"],
            alternating["base_stream_fingerprint"],
        )
        np.testing.assert_array_equal(
            persistent["poisson_innovations"], alternating["poisson_innovations"]
        )
        self.assertEqual(
            set(np.unique(persistent["interval_states"]).tolist()), {-1, 1}
        )
        self.assertEqual(set(np.unique(persistent["gaps"]).tolist()), {3, 5})
        self.assertFalse(
            np.array_equal(persistent["interval_states"], alternating["interval_states"])
        )
        positive = persistent["positive_magnitude"][
            ~np.isnan(persistent["positive_magnitude"])
        ]
        self.assertTrue(bool(np.all(positive >= 1.0)))
        self.assertTrue(bool(np.allclose(positive, np.rint(positive))))

        corrupted = synthetic.build_common_base(d=4, n_series=2, seed=20260905)
        corrupted["interval_transition_uniforms"][0, 0] = (
            float(corrupted["interval_transition_uniforms"][0, 0]) + 0.25
        ) % 1.0
        with self.assertRaisesRegex(ValueError, "common-base fingerprint"):
            synthetic.transform_common_base(
                corrupted, rho_interval=0.0, rho_magnitude=0.0
            )

        duplicate_identity = synthetic.build_common_base(
            d=4, n_series=2, seed=20260905
        )
        duplicate_identity["base_innovation_id"][1] = duplicate_identity[
            "base_innovation_id"
        ][0]
        with self.assertRaisesRegex(ValueError, "base innovation identities"):
            synthetic.transform_common_base(
                duplicate_identity, rho_interval=0.0, rho_magnitude=0.0
            )

    def test_dgp_balance_audit_summarizes_blocks_and_keeps_pairwise_failures(self):
        """Catches theoretical-only balance checks or deleting failed diagnostic cells."""
        blocks = self._grid_blocks()
        summary = synthetic.summarize_generated_block(blocks[0])

        self.assertEqual(summary["n_observations"], 24 * synthetic.SERIES_LENGTH)
        self.assertGreater(summary["positive_count"], 0)
        self.assertIn("positive_histogram", summary)
        self.assertIn("ADI", summary)
        self.assertIn("CV2", summary)
        self.assertIn("realized_rho_interval", summary)
        self.assertIn("realized_rho_magnitude", summary)
        self.assertEqual(summary["n_series"], 24)
        self.assertEqual(summary["series_length"], synthetic.SERIES_LENGTH)
        self.assertIn("base_stream_fingerprint", summary)
        self.assertIn("base_identity_sha256", summary)
        self.assertEqual(summary["rho_I"], summary["rho_interval"])
        self.assertEqual(summary["rho_M"], summary["rho_magnitude"])

        report = self._audit(blocks)

        self.assertEqual(report["rows"][0]["status"], "DGP_BALANCE_FAIL")
        self.assertEqual(report["rows"][1]["status"], "DGP_BALANCE_FAIL")
        self.assertFalse(report["rows"][0]["confirmatory_eligible"])
        self.assertFalse(report["rows"][1]["confirmatory_eligible"])
        self.assertGreater(report["rows"][0]["positive_mean_pairwise_relative_range"], 0.05)
        self.assertEqual(len(report["rows"]), 18)
        self.assertEqual(report["tolerances"]["zero_rate_abs"], 0.02)
        self.assertEqual(report["tolerances"]["positive_mean_relative"], 0.05)
        self.assertEqual(report["tolerances"]["positive_variance_relative"], 0.10)
        with self.assertRaisesRegex(ValueError, "duplicate DGP"):
            self._audit(blocks + [blocks[0]])
        missing_one_cell = self._grid_blocks()
        missing_one_cell = [
            row
            for row in missing_one_cell
            if not (
                row["d"] == 4
                and row["rho_interval"] == synthetic.RHO_VALUES[-1]
                and row["rho_magnitude"] == synthetic.RHO_VALUES[-1]
            )
        ]
        with self.assertRaisesRegex(ValueError, "9-cell"):
            self._audit(missing_one_cell)

        with self.assertRaisesRegex(ValueError, "18-cell"):
            self._audit(self._grid_blocks(d_values=(4,)))
        with self.assertRaisesRegex(ValueError, "missing DGP data_seed"):
            self._audit(self._grid_blocks(seed=2026090502))
        with self.assertRaises(TypeError):
            synthetic.audit_dgp_balance(blocks)
        with self.assertRaisesRegex(ValueError, "expected_n_series"):
            self._audit(self._grid_blocks(n_series=1))

        inconsistent = self._grid_blocks()
        inconsistent[0] = dict(inconsistent[0], base_stream_fingerprint="other")
        with self.assertRaisesRegex(ValueError, "content SHA256 mismatch"):
            self._audit(inconsistent)

        identity_inconsistent = self._grid_blocks()
        identity_inconsistent[0] = dict(
            identity_inconsistent[0], base_identity_sha256="0" * 64
        )
        with self.assertRaisesRegex(ValueError, "content SHA256 mismatch"):
            self._audit(identity_inconsistent)

        with self.assertRaisesRegex(ValueError, "raw generated blocks"):
            self._audit(
                [synthetic.summarize_generated_block(row) for row in blocks],
            )

        tampered = deepcopy(blocks)
        tampered[0]["y"][0, 0] = tampered[0]["y"][0, 0] + 1.0
        with self.assertRaisesRegex(ValueError, "content SHA256 mismatch"):
            self._audit(tampered)

        self_consistent_forgery = deepcopy(blocks)
        self_consistent_forgery[0]["y"][0, 0] = (
            self_consistent_forgery[0]["y"][0, 0] + 1.0
        )
        self_consistent_forgery[0]["generated_block_sha256"] = (
            synthetic._generated_block_sha256(self_consistent_forgery[0])
        )
        with self.assertRaisesRegex(ValueError, "deterministic verified source"):
            self._audit(self_consistent_forgery)

    def test_dgp_audit_retains_no_positive_cells_with_json_safe_null_metrics(self):
        """Catches NaN/Inf audit rows or crashes on degenerate diagnostic cells."""
        blocks = self._grid_blocks()
        template = synthetic.summarize_generated_block(blocks[0])
        block = {
            "d": 4,
            "data_seed": template["data_seed"],
            "base_id": template["base_id"],
            "base_stream_fingerprint": template["base_stream_fingerprint"],
            "base_series_index": np.arange(24, dtype=np.int32),
            "base_innovation_id": np.asarray(
                [f"{template['base_id']}_series{index:06d}" for index in range(24)]
            ),
            "base_identity_sha256": template["base_identity_sha256"],
            "n_series": 24,
            "series_length": synthetic.SERIES_LENGTH,
            "rho_interval": template["rho_interval"],
            "rho_magnitude": template["rho_magnitude"],
            "y": np.zeros((24, synthetic.SERIES_LENGTH), dtype=np.float32),
            "positive_magnitude": np.full((24, synthetic.SERIES_LENGTH), np.nan, dtype=np.float32),
            "z": np.zeros((24, synthetic.SERIES_LENGTH), dtype=np.float32),
            "interval_states": np.array([], dtype=np.int8),
            "magnitude_states": np.array([], dtype=np.int8),
            "series_event_slices": tuple((0, 0) for _ in range(24)),
        }

        summary = synthetic.summarize_generated_block(block, cell_id=template["cell_id"])
        self.assertIsNone(summary["positive_mean"])
        self.assertIsNone(summary["positive_variance"])
        self.assertIsNone(summary["ADI"])
        self.assertIsNone(summary["CV2"])
        self.assertEqual(summary["nan_count"], 0)
        self.assertEqual(summary["inf_count"], 0)
        self.assertFalse(summary["positive_stats_valid"])
        json.dumps(summary, allow_nan=False)

    def test_known_change_sequences_use_calendar_cp_without_state_reinit(self):
        """Catches event-index changepoints and reinitializing latent states at cp."""
        self.assertEqual(
            synthetic.KNOWN_CHANGE_SHIFT_TYPES,
            (
                "rho_I_positive",
                "rho_I_negative",
                "rho_M_positive",
                "rho_M_negative",
                "rho_I_and_rho_M",
                "no_change",
            ),
        )

        block = synthetic.build_known_change_block(
            d=8,
            shift_type="rho_I_positive",
            n_series=3,
            seed=20260905,
        )

        self.assertEqual(block["changepoint"], 288)
        self.assertEqual(block["y"].shape, (3, synthetic.SERIES_LENGTH))
        self.assertEqual(block["z"].shape, (3, synthetic.SERIES_LENGTH))
        self.assertEqual(set(np.unique(block["gaps"]).tolist()), {7, 9})
        self.assertTrue(block["state_continuity"])
        self.assertTrue(all(block["state_continuity_checks"]))
        self.assertTrue(all(index < 200 for index in block["first_post_event_index_by_series"]))
        self.assertFalse(any(index == synthetic.KNOWN_CHANGE_POINT for index in block["first_post_event_index_by_series"]))
        for series, first_post in enumerate(block["first_post_event_index_by_series"]):
            retained_times = block["series_event_times"][series]
            self.assertGreaterEqual(retained_times[first_post], synthetic.KNOWN_CHANGE_POINT)
            if first_post:
                self.assertLess(retained_times[first_post - 1], synthetic.KNOWN_CHANGE_POINT)
        self.assertEqual(block["pre_rho_interval"], 0.0)
        self.assertEqual(block["post_rho_interval"], 0.8)
        self.assertEqual(block["pre_rho_magnitude"], block["post_rho_magnitude"])
        self.assertEqual(block["shift_token"], "rho_I_positive")
        self.assertEqual(block["transition_kernel_convention"], "previous_event_calendar_time")
        self.assertIn("first event at/after cp may use pre kernel", block["transition_kernel_note"])
        self.assertEqual(block["split"].origins, tuple(range(120, 541, 28)))
        self.assertEqual(block["split"].length, synthetic.SERIES_LENGTH)
        self.assertEqual(block["split"].changepoint, synthetic.KNOWN_CHANGE_POINT)
        self.assertEqual(block["split"].pre_change, (0, synthetic.KNOWN_CHANGE_POINT))
        self.assertEqual(
            block["split"].post_change,
            (synthetic.KNOWN_CHANGE_POINT, synthetic.SERIES_LENGTH),
        )
        self.assertFalse(hasattr(block["split"], "train"))
        self.assertFalse(hasattr(block["split"], "validation"))
        self.assertFalse(hasattr(block["split"], "warmup"))
        self.assertRegex(block["known_change_block_sha256"], r"^[0-9a-f]{64}$")

    def test_known_change_panel_audit_requires_exact_tier_bound_cells_and_raw_source(self):
        blocks = [
            synthetic.build_known_change_block(
                d=d,
                shift_type=shift_type,
                n_series=24,
                seed=2026090501,
            )
            for d in synthetic.SUPPORTED_D
            for shift_type in synthetic.KNOWN_CHANGE_SHIFT_TYPES
        ]
        report = synthetic.audit_known_change_panel(
            blocks,
            runtime_decision=self.runtime_decision,
            preregistration_path=self.preregistration_path,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["expected_cell_count"], 12)
        self.assertEqual(len(report["rows"]), 12)
        self.assertTrue(
            all(row["support_valid"] and row["geometry_valid"] for row in report["rows"])
        )
        self.assertEqual(
            len(
                {
                    (row["data_seed"], row["d"], row["base_stream_fingerprint"])
                    for row in report["rows"]
                }
            ),
            2,
        )

        with self.assertRaisesRegex(ValueError, "exact tier-bound panel"):
            synthetic.audit_known_change_panel(
                blocks[:-1],
                runtime_decision=self.runtime_decision,
                preregistration_path=self.preregistration_path,
            )

        undersized = [
            synthetic.build_known_change_block(
                d=d,
                shift_type=shift_type,
                n_series=1,
                seed=2026090501,
            )
            for d in synthetic.SUPPORTED_D
            for shift_type in synthetic.KNOWN_CHANGE_SHIFT_TYPES
        ]
        with self.assertRaisesRegex(ValueError, "sealed runtime-tier"):
            synthetic.audit_known_change_panel(
                undersized,
                runtime_decision=self.runtime_decision,
                preregistration_path=self.preregistration_path,
            )

        tampered = deepcopy(blocks)
        tampered[0]["y"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "content SHA256 mismatch"):
            synthetic.audit_known_change_panel(
                tampered,
                runtime_decision=self.runtime_decision,
                preregistration_path=self.preregistration_path,
            )

        self_consistent_forgery = deepcopy(blocks)
        self_consistent_forgery[0]["y"][0, 0] += 1.0
        self_consistent_forgery[0]["known_change_block_sha256"] = (
            synthetic._known_change_block_sha256(self_consistent_forgery[0])
        )
        with self.assertRaisesRegex(ValueError, "deterministic verified source"):
            synthetic.audit_known_change_panel(
                self_consistent_forgery,
                runtime_decision=self.runtime_decision,
                preregistration_path=self.preregistration_path,
            )


class DataAdapterTests(unittest.TestCase):
    @staticmethod
    def _toy_split(*, panel_length: int, horizon: int, lookback: int) -> data.RealSplit:
        first_evaluation_origin = int(panel_length) - 6 * int(horizon)
        train_end = first_evaluation_origin - 2 * int(horizon)
        return data.RealSplit(
            train=(0, train_end),
            validation=(train_end, train_end + horizon),
            warmup=(train_end + horizon, first_evaluation_origin),
            origins=tuple(
                first_evaluation_origin + index * horizon for index in range(6)
            ),
            horizon=horizon,
            lookback=lookback,
        )

    @staticmethod
    def _seal_toy_panel(
        panel: dict[str, object], split: data.RealSplit
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "toy.bin"
            source.write_bytes(b"toy-panel-source")
            digest = data.file_sha256(source)
            manifest = integrity.build_source_manifest(
                root,
                {source.name: digest},
                repository_root_identity="test_toy_source_root",
            )
            return data.seal_count_primary_dataset_audit(
                {
                    **panel,
                    "name": "toy",
                    "split": split,
                    "provenance": {
                        "dataset_id": "toy",
                        "sources": [
                            {
                                "path": source.resolve().as_posix(),
                                "size_bytes": source.stat().st_size,
                                "sha256": digest,
                                "expected_sha256": digest,
                                "status": "PASS",
                            }
                        ],
                    },
                },
                source_manifest=manifest,
            )

    @classmethod
    def _toy_window_request(
        cls,
        *,
        panel: dict[str, object],
        horizon: int,
        lookback: int,
        origins: tuple[int, ...],
        role: str = "model_train",
    ) -> tuple[data.WindowRequest, dict[str, object]]:
        panel_length = int(np.asarray(panel["y"]).shape[1])
        split = cls._toy_split(
            panel_length=panel_length, horizon=horizon, lookback=lookback
        )
        audit = cls._seal_toy_panel(panel, split)
        sample_manifest = data.seal_train_only_sample_manifest(
            panel,
            dataset_audit=audit,
            runtime_tier="MINIMAL-COMPLETE",
        )
        return data.build_window_request(
            dataset_id="toy",
            split=split,
            panel_length=panel_length,
            role=role,
            origins=origins,
            panel=panel,
            dataset_audit=audit,
            sample_manifest=sample_manifest,
        ), audit

    @staticmethod
    def _count_primary_panel(dataset_id: str) -> dict[str, object]:
        split = data.REAL_SPLITS[dataset_id]
        length = split.origins[-1] + split.horizon
        values = np.zeros((2, length), dtype=np.float64)
        values[:, ::7] = 2.0
        return {
            "name": dataset_id,
            "y": values,
            "split": split,
            "available_from": np.zeros(values.shape[0], dtype=np.int32),
            "series_id": np.asarray(["s0", "s1"]),
        }

    @classmethod
    def _sealed_count_primary_audit(cls, dataset_id: str) -> dict[str, object]:
        contract = data.CANONICAL_COUNT_PRIMARY_PANEL_CONTRACTS[dataset_id]
        split = data.REAL_SPLITS[dataset_id]
        record = {
            "audit_type": "COUNT_PRIMARY_DATASET_AUDIT",
            "dataset_id": dataset_id,
            "status": "PASS",
            "geometry_status": "PASS",
            "count_primary_eligible": True,
            "confirmatory_eligible": True,
            "support_audit": {
                "dataset_id": dataset_id,
                "status": "PASS",
                "target_scope": "model_train",
                "count_primary_eligible": True,
                "count_likelihood_index_exact": True,
            },
            "split_validation": data.validate_real_split(
                split, length=int(contract["panel_shape"][1])
            ),
            "source_records": [{"path": f"{dataset_id}/source", "sha256": "1" * 64}],
            "source_manifest_aggregate_sha256": "2" * 64,
            "panel_shape": list(contract["panel_shape"]),
            "panel_binding_sha256": contract["panel_binding_sha256"],
            "ordered_series_id_sha256": contract["ordered_series_id_sha256"],
            "canonical_source_attested": True,
        }
        record["audit_sha256"] = data._dataset_audit_hash(record)
        return record

    def test_unattested_count_panel_can_only_produce_diagnostic_windows(self):
        panel = {
            "name": "toy",
            "y": np.tile(np.arange(30, dtype=np.float64) % 3, (3, 1)),
            "series_id": np.asarray(["a", "b", "c"]),
            "available_from": np.zeros(3, dtype=np.int32),
        }
        split = self._toy_split(panel_length=30, horizon=3, lookback=4)
        audit = self._seal_toy_panel(panel, split)
        self.assertFalse(audit["canonical_source_attested"])
        self.assertFalse(audit["confirmatory_eligible"])
        sample = data.seal_train_only_sample_manifest(
            panel,
            dataset_audit=audit,
            runtime_tier="MINIMAL-COMPLETE",
        )
        request = data.build_window_request(
            dataset_id="toy",
            split=split,
            panel_length=30,
            role="evaluation",
            origins=split.origins,
            panel=panel,
            dataset_audit=audit,
            sample_manifest=sample,
        )
        self.assertFalse(request.confirmatory_eligible)
        batch = data.make_history_windows(
            panel, request=request, dataset_audit=audit
        )
        self.assertFalse(batch.confirmatory_eligible)

        forged_audit = dict(audit, confirmatory_eligible=True)
        forged_audit["audit_sha256"] = data._dataset_audit_hash(forged_audit)
        with self.assertRaisesRegex(ValueError, "not canonical|cannot be confirmatory"):
            data.seal_train_only_sample_manifest(
                panel,
                dataset_audit=forged_audit,
                runtime_tier="MINIMAL-COMPLETE",
            )

        forged_request = replace(request, confirmatory_eligible=True)
        forged_request = replace(
            forged_request,
            request_sha256=data._canonical_sha256(
                data._window_request_payload(
                    dataset_id=forged_request.dataset_id,
                    role=forged_request.role,
                    origins=forged_request.origins,
                    dataset_audit_sha256=forged_request.dataset_audit_sha256,
                    sample_manifest_sha256=forged_request.sample_manifest_sha256,
                    sampled_panel_binding_sha256=(
                        forged_request.sampled_panel_binding_sha256
                    ),
                    sampling_runtime_tier=forged_request.sampling_runtime_tier,
                    sampling_seed=forged_request.sampling_seed,
                    confirmatory_eligible=True,
                    split_contract_sha256=forged_request.split_contract_sha256,
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "canonical reconstruction"):
            data.make_history_windows(
                panel, request=forged_request, dataset_audit=audit
            )

    def test_support_audit_rejects_fractional_negative_and_nonfinite_count_targets(self):
        """Catches silent rounding, continuous-target leakage, or invalid support."""
        audit = data.audit_target_support(
            np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 1.0]], dtype=np.float32),
            dataset_id="toy",
        )

        self.assertEqual(audit["dataset_id"], "toy")
        self.assertEqual(audit["support"], "nonnegative_integer_count")
        self.assertEqual(audit["n_observations"], 6)
        self.assertEqual(audit["zero_count"], 3)
        self.assertEqual(audit["positive_count"], 3)
        self.assertEqual(audit["zero_ratio"], 0.5)
        self.assertEqual(audit["positive_mean"], 2.0)
        self.assertEqual(audit["positive_variance"], 1.0)
        self.assertEqual(audit["exact_integer_count"], 6)
        self.assertEqual(audit["nonexact_within_tolerance_count"], 0)
        self.assertTrue(audit["count_primary_eligible"])

        fractional = data.audit_target_support(np.array([0.0, 1.5]), dataset_id="bad")
        negative = data.audit_target_support(np.array([0.0, -1.0]), dataset_id="bad")
        nonfinite = data.audit_target_support(np.array([0.0, np.nan, np.inf]), dataset_id="bad")
        near_integer = data.audit_target_support(
            np.array([0.0, 1.0000004768371582], dtype=np.float64),
            dataset_id="ambiguous",
        )
        self.assertEqual(fractional["noninteger_count"], 1)
        self.assertEqual(negative["negative_count"], 1)
        self.assertEqual(nonfinite["nan_count"], 1)
        self.assertEqual(nonfinite["inf_count"], 1)
        self.assertEqual(nonfinite["nonfinite_count"], 2)
        self.assertEqual(near_integer["noninteger_count"], 0)
        self.assertEqual(near_integer["exact_integer_count"], 1)
        self.assertEqual(near_integer["nonexact_within_tolerance_count"], 1)
        self.assertEqual(
            near_integer["hard_stop_token"],
            "COUNT_LIKELIHOOD_INDEX_AMBIGUITY_HARD_STOP",
        )
        self.assertFalse(fractional["count_primary_eligible"])
        self.assertFalse(negative["count_primary_eligible"])
        self.assertFalse(nonfinite["count_primary_eligible"])
        self.assertFalse(near_integer["count_primary_eligible"])
        self.assertIn(
            "COUNT_LIKELIHOOD_INDEX_AMBIGUITY_HARD_STOP",
            near_integer["ineligible_reason"],
        )
        for invalid in (fractional, negative, nonfinite, near_integer):
            with self.assertRaises(ValueError):
                data.enforce_target_support(invalid)

        all_zero = data.audit_target_support(np.zeros((2, 3)), dataset_id="zero")
        self.assertIsNone(all_zero["positive_mean"])
        self.assertIsNone(all_zero["positive_variance"])
        json.dumps(all_zero, allow_nan=False)

    def test_model_train_support_audit_ignores_poisoned_future_targets(self):
        """Catches validation/test targets leaking into support audit or selection."""
        split = data.RealSplit(
            train=(0, 3),
            validation=(3, 4),
            warmup=(4, 5),
            origins=(5,),
            horizon=1,
            lookback=2,
        )
        y = np.array([[0.0, 1.0, 2.0, 1.5, -1.0, np.nan]], dtype=np.float32)
        clean = y.copy()
        clean[:, 3:] = 0.0

        audit = data.audit_model_train_target_support(y, split=split, dataset_id="toy")
        clean_audit = data.audit_model_train_target_support(clean, split=split, dataset_id="toy")

        self.assertEqual(audit, clean_audit)
        self.assertEqual(audit["target_scope"], "model_train")
        self.assertEqual(audit["time_start"], 0)
        self.assertEqual(audit["time_end"], 3)
        self.assertEqual(audit["status"], "PASS")
        data.enforce_target_support(audit)

        unavailable = data.audit_model_train_target_support(
            np.zeros((1, 6), dtype=np.float32),
            split=split,
            dataset_id="toy",
            available_from=np.array([3], dtype=np.int32),
            series_ids=np.array(["late"]),
        )
        self.assertEqual(unavailable["status"], "COUNT_SUPPORT_FAIL")
        self.assertEqual(unavailable["support"], "empty_count_support_scope")
        self.assertEqual(unavailable["availability_excluded_count"], 1)

        ambiguous = data.audit_model_train_target_support(
            np.array([[0.0, 1.0000004768371582, 2.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            split=split,
            dataset_id="toy",
            series_ids=np.array(["ambiguous"]),
        )
        self.assertEqual(ambiguous["noninteger_count"], 0)
        self.assertEqual(ambiguous["nonexact_within_tolerance_count"], 1)
        self.assertFalse(ambiguous["count_primary_eligible"])
        self.assertIn(
            "COUNT_LIKELIHOOD_INDEX_AMBIGUITY_HARD_STOP",
            ambiguous["ineligible_reason"],
        )

    def test_train_rms_scale_and_history_windows_never_use_future_values(self):
        """Catches scale fitting or history construction that reads validation/test targets."""
        y = np.arange(2 * 26, dtype=np.float32).reshape(2, 26)
        poisoned = y.copy()
        poisoned[:, 6:] = 1_000_000.0

        scale = data.train_rms_scale(y, train_end=6)
        poisoned_scale = data.train_rms_scale(poisoned, train_end=6)
        np.testing.assert_allclose(scale, poisoned_scale)
        np.testing.assert_allclose(
            scale,
            np.sqrt(np.mean(y[:, :6].astype(np.float64) ** 2, axis=1) + 1e-8),
        )

        panel = {"name": "toy", "y": y, "series_id": np.array(["a", "b"])}
        poisoned_panel = {
            "name": "toy",
            "y": poisoned,
            "series_id": np.array(["a", "b"]),
        }
        two_origin_request, audit = self._toy_window_request(
            panel=panel, horizon=2, lookback=3, origins=(6, 8)
        )
        poisoned_request, poisoned_audit = self._toy_window_request(
            panel=poisoned_panel, horizon=2, lookback=3, origins=(6, 8)
        )
        one_origin_request = data.build_window_request(
            dataset_id="toy",
            split=self._toy_split(panel_length=26, horizon=2, lookback=3),
            panel_length=26,
            role="model_train",
            origins=(6,),
            panel=panel,
            dataset_audit=audit,
            sample_manifest=data.seal_train_only_sample_manifest(
                panel,
                dataset_audit=audit,
                runtime_tier="MINIMAL-COMPLETE",
            ),
        )
        windows = data.make_history_windows(
            panel,
            request=two_origin_request,
            dataset_audit=audit,
        )
        poisoned_windows = data.make_history_windows(
            poisoned_panel,
            request=poisoned_request,
            dataset_audit=poisoned_audit,
        )
        np.testing.assert_array_equal(windows.history[:2], poisoned_windows.history[:2])
        np.testing.assert_array_equal(windows.target[0], np.array([6.0, 7.0]))
        self.assertEqual(windows.key_frame.columns.tolist(), ["dataset_id", "series_id", "origin", "step"])

        with self.assertRaisesRegex(ValueError, "duplicate series"):
            data.make_history_windows(
                {"name": "toy", "y": y, "series_id": np.array(["dup", "dup"])},
                request=one_origin_request,
                dataset_audit=audit,
            )
        nonfinite = y.copy()
        nonfinite[0, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "panel binding"):
            data.make_history_windows(
                {"name": "toy", "y": nonfinite, "series_id": np.array(["a", "b"])},
                request=one_origin_request,
                dataset_audit=audit,
            )
        np.testing.assert_allclose(
            windows.scale[:2],
            data.train_rms_scale(y, train_end=10),
        )
        with self.assertRaises(TypeError):
            data.make_history_windows(
                panel,
                request=one_origin_request,
                dataset_audit=audit,
                scale=data.train_rms_scale(y, train_end=y.shape[1]),
            )

    def test_history_window_count_audit_uses_raw_targets_before_float32_cast(self):
        """Catches float32 conversion hiding fractional count targets."""
        y = np.zeros((1, 12), dtype=np.float64)
        y[0, 4] = 16_777_216.5
        panel = {"name": "toy", "y": y, "series_id": np.array(["a"])}
        request, audit = self._toy_window_request(
            panel=panel, horizon=1, lookback=2, origins=(4,), role="validation"
        )

        with self.assertRaisesRegex(ValueError, "count target"):
            data.make_history_windows(
                panel,
                request=request,
                dataset_audit=audit,
            )

        y_integer = np.zeros((1, 12), dtype=np.float64)
        y_integer[0, 4] = 16_777_217.0
        integer_panel = {
            "name": "toy",
            "y": y_integer,
            "series_id": np.array(["a"]),
        }
        integer_request, integer_audit = self._toy_window_request(
            panel=integer_panel,
            horizon=1,
            lookback=2,
            origins=(4,),
            role="validation",
        )
        with self.assertRaisesRegex(ValueError, "float32 round-trip"):
            data.make_history_windows(
                integer_panel,
                request=integer_request,
                dataset_audit=integer_audit,
            )

        y_near_integer = np.zeros((1, 12), dtype=np.float64)
        y_near_integer[0, 4] = 1.0000004768371582
        near_panel = {
            "name": "toy",
            "y": y_near_integer,
            "series_id": np.array(["a"]),
        }
        near_request, near_audit = self._toy_window_request(
            panel=near_panel,
            horizon=1,
            lookback=2,
            origins=(4,),
            role="validation",
        )
        with self.assertRaisesRegex(
            ValueError, "COUNT_LIKELIHOOD_INDEX_AMBIGUITY_HARD_STOP"
        ):
            data.make_history_windows(
                near_panel,
                request=near_request,
                dataset_audit=near_audit,
            )

    def test_availability_masks_targets_without_changing_zero_padded_history(self):
        """Catches M5 pre-availability target leakage or history rewrites."""
        y = np.arange(2 * 31, dtype=np.float32).reshape(2, 31)
        panel = {
            "name": "toy",
            "y": y,
            "series_id": np.array(["late", "early"]),
            "available_from": np.array([5, 0], dtype=np.int32),
        }

        request, audit = self._toy_window_request(
            panel=panel, horizon=3, lookback=3, origins=(4,)
        )
        windows = data.make_history_windows(
            panel, request=request, dataset_audit=audit
        )

        late_index = int(np.flatnonzero(windows.series_id == "late")[0])
        early_index = int(np.flatnonzero(windows.series_id == "early")[0])
        np.testing.assert_array_equal(windows.history[late_index], y[0, 1:4])
        np.testing.assert_array_equal(windows.target[late_index], y[0, 4:7])
        np.testing.assert_array_equal(
            windows.target_mask[late_index], np.array([False, True, True])
        )
        np.testing.assert_array_equal(
            windows.target_mask[early_index], np.array([True, True, True])
        )
        bad_count_target = y.copy()
        bad_count_target[1, 7] = 1.5
        bad_panel = {**panel, "y": bad_count_target}
        bad_request, bad_audit = self._toy_window_request(
            panel=bad_panel,
            horizon=3,
            lookback=3,
            origins=(7,),
            role="validation",
        )
        with self.assertRaisesRegex(ValueError, "count target"):
            data.make_history_windows(
                bad_panel,
                request=bad_request,
                dataset_audit=bad_audit,
            )
        with self.assertRaisesRegex(ValueError, "available_from"):
            self._seal_toy_panel(
                {**panel, "available_from": np.array([32, 0])},
                self._toy_split(panel_length=31, horizon=3, lookback=3),
            )

    def test_window_build_rejects_any_post_audit_panel_identity_mutation(self):
        """Catches target, series-ID, or availability drift after source/support sealing."""
        y = np.zeros((2, 31), dtype=np.float64)
        y[:, :7] = np.arange(14, dtype=np.float64).reshape(2, 7)
        panel = {
            "name": "toy",
            "y": y,
            "series_id": np.array(["a", "b"]),
            "available_from": np.array([0, 2], dtype=np.int32),
        }
        request, audit = self._toy_window_request(
            panel=panel,
            horizon=3,
            lookback=3,
            origins=(7,),
            role="validation",
        )
        canonical = data.make_history_windows(
            panel, request=request, dataset_audit=audit
        )
        expected_scale = dict(
            zip(
                panel["series_id"],
                data.train_rms_scale(
                    y, train_end=7, available_from=panel["available_from"]
                ),
            )
        )
        np.testing.assert_allclose(
            canonical.scale,
            np.asarray([expected_scale[value] for value in canonical.series_id]),
        )

        mutated_target = y.copy()
        mutated_target[0, 7] = 1.0
        mutations = (
            {**panel, "y": mutated_target},
            {**panel, "series_id": np.array(["a", "changed"])},
            {**panel, "available_from": np.array([1, 2], dtype=np.int32)},
        )
        for mutated in mutations:
            with self.assertRaisesRegex(ValueError, "panel binding"):
                data.make_history_windows(
                    mutated, request=request, dataset_audit=audit
                )

    def test_split_and_prediction_key_helpers_reject_overlap_and_key_drift(self):
        """Catches split leakage and cross-head prediction-key mismatch."""
        data.validate_disjoint_splits({"train": (0, 3), "validation": (3, 5), "test": (5, 7)})
        with self.assertRaisesRegex(ValueError, "overlap"):
            data.validate_disjoint_splits({"train": (0, 4), "validation": (3, 5)})
        with self.assertRaisesRegex(ValueError, "origins"):
            data.validate_real_split(
                data.RealSplit(train=(0, 2), validation=(2, 3), warmup=(3, 4), origins=(4, 4), horizon=1, lookback=2),
                length=6,
            )
        with self.assertRaisesRegex(ValueError, "start at 0"):
            data.validate_real_split(
                data.RealSplit(train=(1, 2), validation=(2, 3), warmup=(3, 4), origins=(4, 5, 6, 7, 8, 9), horizon=1, lookback=2),
                length=10,
            )
        with self.assertRaisesRegex(ValueError, "horizon-spaced"):
            data.validate_real_split(
                data.RealSplit(train=(0, 2), validation=(2, 3), warmup=(3, 4), origins=(4, 6, 7, 8, 9, 10), horizon=1, lookback=2),
                length=11,
            )
        with self.assertRaisesRegex(ValueError, "validation.*horizon"):
            data.validate_real_split(
                data.RealSplit(train=(0, 2), validation=(2, 4), warmup=(4, 5), origins=(5, 6, 7, 8, 9, 10), horizon=1, lookback=2),
                length=11,
            )
        with self.assertRaisesRegex(ValueError, "warmup.*horizon"):
            data.validate_real_split(
                data.RealSplit(train=(0, 2), validation=(2, 3), warmup=(3, 5), origins=(5, 6, 7, 8, 9, 10), horizon=1, lookback=2),
                length=11,
            )
        with self.assertRaisesRegex(ValueError, "warmup.*first"):
            data.validate_real_split(
                data.RealSplit(train=(0, 2), validation=(2, 3), warmup=(3, 4), origins=(5, 6, 7, 8, 9, 10), horizon=1, lookback=2),
                length=11,
            )
        with self.assertRaisesRegex(ValueError, "final"):
            data.validate_real_split(data.REAL_SPLITS["online_retail"], length=375)
        with self.assertRaisesRegex(ValueError, "duplicate series"):
            data.origin_key_frame(
                dataset_id="toy",
                series_ids=np.array(["a", "a"]),
                origins=np.array([10], dtype=np.int32),
                horizon=2,
            )
        with self.assertRaisesRegex(ValueError, "duplicate origins"):
            data.origin_key_frame(
                dataset_id="toy",
                series_ids=np.array(["a", "b"]),
                origins=np.array([10, 10], dtype=np.int32),
                horizon=2,
            )

    def test_window_request_binds_role_to_the_frozen_split(self):
        """Catches outer/validation targets relabelled as train or crossing a split."""
        split = data.REAL_SPLITS["m5"]
        panel = {**self._count_primary_panel("m5"), "name": "toy"}
        audit = self._seal_toy_panel(panel, split)
        sample_manifest = data.seal_train_only_sample_manifest(
            panel,
            dataset_audit=audit,
            runtime_tier="MINIMAL-COMPLETE",
        )
        request = data.build_window_request(
            dataset_id="toy",
            split=split,
            panel_length=1941,
            role="evaluation",
            origins=split.origins,
            panel=panel,
            dataset_audit=audit,
            sample_manifest=sample_manifest,
        )
        self.assertEqual(request.origins, split.origins)
        self.assertEqual(request.role, "evaluation")

        with self.assertRaisesRegex(ValueError, "model_train target interval"):
            data.build_window_request(
                dataset_id="toy",
                split=split,
                panel_length=1941,
                role="model_train",
                origins=(1773,),
                panel=panel,
                dataset_audit=audit,
                sample_manifest=sample_manifest,
            )
        with self.assertRaisesRegex(ValueError, "inner_train target interval"):
            data.build_window_request(
                dataset_id="toy",
                split=split,
                panel_length=1941,
                role="inner_train",
                origins=(1717,),
                panel=panel,
                dataset_audit=audit,
                sample_manifest=sample_manifest,
            )
        with self.assertRaisesRegex(ValueError, "crosses.*target interval"):
            data.build_window_request(
                dataset_id="toy",
                split=split,
                panel_length=1941,
                role="teacher_train",
                origins=(1700,),
                panel=panel,
                dataset_audit=audit,
                sample_manifest=sample_manifest,
            )

        tampered = replace(request, role="model_train")
        with self.assertRaisesRegex(ValueError, "request hash"):
            data.make_history_windows(
                {
                    "name": "toy",
                    "y": np.zeros((1, 1941), dtype=np.float32),
                    "series_id": np.array(["one"]),
                },
                request=tampered,
                dataset_audit=audit,
            )

        keys = data.origin_key_frame(
            dataset_id="toy",
            series_ids=np.array(["b", "a"]),
            origins=np.array([10, 38], dtype=np.int32),
            horizon=2,
        )
        shuffled = keys.sample(frac=1.0, random_state=11).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "prediction keys"):
            data.assert_same_prediction_keys(keys, shuffled)
        data.assert_same_prediction_keys(keys, keys.copy())
        duplicate_keys = pd.concat([keys, keys.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate prediction keys"):
            data.assert_same_prediction_keys(duplicate_keys, duplicate_keys.copy())

    def test_descriptor_sampling_is_deterministic_stratified_and_train_only(self):
        """Catches validation/test leakage or nondeterministic descriptor sampling."""
        rows = []
        series = []
        for dataset_id in ("m5", "online_retail"):
            for idx in range(16):
                train = np.zeros(12, dtype=np.float32)
                train[idx % 12] = 1.0 + idx
                train[(idx * 3) % 12] += float(idx % 4)
                future = np.full(4, 10_000.0 + idx, dtype=np.float32)
                series.append(np.concatenate([train, future]))
                rows.append({
                    "dataset_id": dataset_id,
                    "series_id": f"{dataset_id}_{idx}",
                    "available_from": 12 if idx == 15 else idx % 3,
                })
        y = np.vstack(series)
        meta = pd.DataFrame(rows)

        descriptors = data.train_only_descriptors(y, meta, train_end=12)
        poisoned = y.copy()
        poisoned[:, 12:] = np.arange(poisoned.shape[0] * 4).reshape(poisoned.shape[0], 4)
        poisoned_descriptors = data.train_only_descriptors(poisoned, meta, train_end=12)
        pd.testing.assert_frame_equal(descriptors, poisoned_descriptors)
        self.assertIn("log_train_scale", descriptors.columns)
        self.assertNotIn("log_scale_train", descriptors.columns)
        self.assertFalse(
            bool(descriptors.loc[descriptors["series_id"] == "m5_15", "descriptor_eligible"].iloc[0])
        )

        padding_probe = np.asarray(
            [[0.0, 0.0, 3.0, 4.0], [91.0, 73.0, 3.0, 4.0]], dtype=np.float64
        )
        padding_meta = pd.DataFrame(
            {
                "dataset_id": ["m5", "m5"],
                "series_id": ["late_a", "late_b"],
                "available_from": [2, 2],
            }
        )
        padding_descriptors = data.train_only_descriptors(
            padding_probe, padding_meta, train_end=4
        )
        self.assertEqual(
            padding_descriptors.loc[0, "log_train_scale"],
            padding_descriptors.loc[1, "log_train_scale"],
        )
        np.testing.assert_allclose(
            data.train_rms_scale(
                padding_probe,
                train_end=4,
                available_from=np.asarray([2, 2], dtype=np.int32),
            ),
            np.sqrt(np.mean(np.asarray([3.0, 4.0]) ** 2) + 1e-8),
        )

        with self.assertRaisesRegex(ValueError, "n_per_dataset"):
            data.stratified_train_descriptor_sample(descriptors, n=12, seed=2026090521)
        first = data.stratified_train_descriptor_sample(descriptors, n_per_dataset=6, seed=2026090521)
        second = data.stratified_train_descriptor_sample(descriptors, n_per_dataset=6, seed=2026090521)
        permuted = descriptors.sample(frac=1.0, random_state=2026).reset_index(drop=True)
        from_permuted = data.stratified_train_descriptor_sample(permuted, n_per_dataset=6, seed=2026090521)
        self.assertEqual(first["series_id"].tolist(), second["series_id"].tolist())
        self.assertEqual(first["series_id"].tolist(), from_permuted["series_id"].tolist())
        self.assertEqual(len(first), 12)
        self.assertEqual(first["series_id"].nunique(), 12)
        self.assertNotIn("m5_15", first["series_id"].tolist())
        self.assertNotIn("online_retail_15", first["series_id"].tolist())
        self.assertEqual(set(first["dataset_id"]), {"m5", "online_retail"})
        single_dataset = descriptors[descriptors["dataset_id"] == "m5"]
        census = data.stratified_train_descriptor_sample(single_dataset, n=99, seed=2026090521)
        self.assertEqual(len(census), int(single_dataset["descriptor_eligible"].sum()))

    def test_sample_manifest_is_train_only_reconstructible_and_window_bound(self):
        """Catches hand-picked outer-aware subsets or sample lineage substitution."""
        n_series = 1_004
        panel_length = 31
        split = self._toy_split(panel_length=panel_length, horizon=3, lookback=3)
        y = np.zeros((n_series, panel_length), dtype=np.float64)
        for index in range(n_series):
            y[index, : split.train[1]] = np.asarray(
                [(index + step) % 5 for step in range(split.train[1])],
                dtype=np.float64,
            )
        series_ids = np.asarray([f"series_{index:04d}" for index in range(n_series)])
        panel = {
            "name": "toy",
            "y": y,
            "series_id": series_ids,
            "available_from": np.zeros(n_series, dtype=np.int32),
        }
        audit = self._seal_toy_panel(panel, split)
        manifest = data.seal_train_only_sample_manifest(
            panel,
            dataset_audit=audit,
            runtime_tier="MINIMAL-COMPLETE",
        )

        self.assertEqual(manifest["requested_n"], 1_000)
        self.assertEqual(manifest["actual_n"], 1_000)
        self.assertEqual(manifest["seed"], data.REAL_SAMPLING_SEED)
        self.assertEqual(manifest["descriptor_scope"], "model_train_only_availability_aware")
        self.assertEqual(len(manifest["selected_series_ids"]), 1_000)
        self.assertEqual(len(set(manifest["selected_series_ids"])), 1_000)
        data.verify_train_only_sample_manifest(
            panel, dataset_audit=audit, sample_manifest=manifest
        )

        poisoned_y = y.copy()
        poisoned_y[:, split.validation[0] :] = np.arange(
            n_series * (panel_length - split.validation[0]), dtype=np.float64
        ).reshape(n_series, -1)
        poisoned_panel = {**panel, "y": poisoned_y}
        poisoned_audit = self._seal_toy_panel(poisoned_panel, split)
        poisoned_manifest = data.seal_train_only_sample_manifest(
            poisoned_panel,
            dataset_audit=poisoned_audit,
            runtime_tier="MINIMAL-COMPLETE",
        )
        self.assertEqual(
            manifest["selected_series_ids"], poisoned_manifest["selected_series_ids"]
        )

        request = data.build_window_request(
            dataset_id="toy",
            split=split,
            panel_length=panel_length,
            role="validation",
            origins=(split.validation[0],),
            panel=panel,
            dataset_audit=audit,
            sample_manifest=manifest,
        )
        windows = data.make_history_windows(
            panel, request=request, dataset_audit=audit
        )
        self.assertEqual(windows.history.shape[0], 1_000)
        self.assertEqual(windows.sample_manifest_sha256, manifest["manifest_sha256"])
        self.assertEqual(
            windows.sampled_panel_binding_sha256,
            manifest["sampled_panel_binding_sha256"],
        )
        self.assertEqual(windows.series_id.tolist(), manifest["selected_series_ids"])

        forged = dict(manifest)
        forged["selected_series_ids"] = list(reversed(forged["selected_series_ids"]))
        forged["selected_positions"] = list(reversed(forged["selected_positions"]))
        forged["selected_strata"] = list(reversed(forged["selected_strata"]))
        forged["selected_series_id_sha256"] = data._ordered_string_sha256(
            forged["selected_series_ids"]
        )
        selected_positions = np.asarray(forged["selected_positions"], dtype=np.int64)
        forged["sampled_panel_binding_sha256"] = data._panel_binding_sha256(
            y[selected_positions],
            np.zeros(n_series, dtype=np.int32)[selected_positions],
            series_ids[selected_positions],
        )
        forged["manifest_sha256"] = data._sample_manifest_hash(forged)
        with self.assertRaisesRegex(ValueError, "canonical train-only reconstruction"):
            data.build_window_request(
                dataset_id="toy",
                split=split,
                panel_length=panel_length,
                role="validation",
                origins=(split.validation[0],),
                panel=panel,
                dataset_audit=audit,
                sample_manifest=forged,
            )

    def test_production_dataset_audit_rejects_a_preselected_outer_aware_pool(self):
        """Catches resealing a hand-picked M5 subset with genuine-looking provenance."""
        split = data.REAL_SPLITS["m5"]
        n_series = 1_004
        values = np.zeros((n_series, split.origins[-1] + split.horizon), dtype=np.float64)
        values[:, split.origins[0]] = np.arange(n_series, dtype=np.float64)
        panel = {
            "name": "m5",
            "y": values,
            "split": split,
            "series_id": np.asarray([f"outer_rank_{index:04d}" for index in range(n_series)]),
            "available_from": np.zeros(n_series, dtype=np.int32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sales_train_evaluation.csv"
            source.write_bytes(b"genuine-looking-but-not-canonical-m5-source")
            digest = data.file_sha256(source)
            source_manifest = integrity.build_source_manifest(
                root,
                {source.name: digest},
                repository_root_identity="test_outer_aware_source_root",
            )
            panel["provenance"] = {
                "dataset_id": "m5",
                "adapter_function": data.M5_CANONICAL_ADAPTER_FUNCTION,
                "sources": [
                    {
                        "path": source.resolve().as_posix(),
                        "size_bytes": source.stat().st_size,
                        "sha256": digest,
                        "expected_sha256": digest,
                        "status": "PASS",
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "frozen canonical full-pool"):
                data.seal_count_primary_dataset_audit(
                    panel, source_manifest=source_manifest
                )

    def test_source_hash_guard_and_online_retail_tweediegp_aggregation(self):
        """Catches mutable source files or non-TweedieGP UCI-352 aggregation."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text("frozen", encoding="utf-8")
            expected = data.file_sha256(path)
            self.assertEqual(data.verify_source_hash(path, expected)["status"], "PASS")
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash"):
                data.verify_source_hash(path, expected)
        with self.assertRaisesRegex(ValueError, "expected_sha256"):
            data.load_online_retail_tweediegp(Path("missing.xlsx"))
        with self.assertRaisesRegex(ValueError, "frozen"):
            data.load_online_retail_tweediegp(Path("missing.xlsx"), expected_sha256="0" * 64)
        with tempfile.TemporaryDirectory() as directory:
            arbitrary = Path(directory) / "online_retail_II.xlsx"
            arbitrary.write_text("not the snapshot", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_snapshots.*online_retail"):
                data.load_online_retail_tweediegp(
                    arbitrary, expected_sha256=data.ONLINE_RETAIL_EXPECTED_SHA256
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calendar_path = root / "calendar.csv"
            prices_path = root / "sell_prices.csv"
            calendar_path.write_text(
                "d,wm_yr_wk\n"
                "d_1,10\n"
                "d_2,10\n"
                "d_3,11\n"
                "d_4,11\n"
                "d_5,12\n",
                encoding="utf-8",
            )
            prices_path.write_text(
                "store_id,item_id,wm_yr_wk\n"
                "CA_1,FOO,11\n"
                "CA_1,BAR,10\n",
                encoding="utf-8",
            )
            starts = data._m5_availability_from_verified_paths(
                pd.DataFrame(
                    {"store_id": ["CA_1", "CA_1"], "item_id": ["FOO", "BAR"]}
                ),
                calendar_path=calendar_path,
                sell_prices_path=prices_path,
            )
            np.testing.assert_array_equal(starts, np.array([2, 0], dtype=np.int32))

        frame = pd.DataFrame(
            {
                "Description": ["Z item", "Z item", "BAD item", "LATE item", "add stock to allocate online orders"],
                "InvoiceDate": pd.to_datetime(
                    ["2010-12-01", "2010-12-03", "2010-12-02", "2011-08-01", "2010-12-01"]
                ),
                "Quantity": [2, 3, -1, 4, 5],
            }
        )

        grid = data.aggregate_online_retail_tweediegp_frame(frame)

        self.assertEqual(grid["series_id"].tolist(), ["Z item"])
        self.assertEqual(grid["y"].shape, (1, 244))
        np.testing.assert_array_equal(grid["y"][0, :3], np.array([2.0, 0.0, 3.0], dtype=np.float32))
        self.assertEqual(float(grid["y"][0, 3:].sum()), 0.0)
        self.assertEqual(grid["date_start"], "2010-12-01")
        self.assertEqual(grid["date_end"], "2011-08-01")

    def test_m5_adapter_split_contract_and_stage_a_exclusion(self):
        """Catches guessed M5 geometry or leaking prior Stage-A sampled IDs."""
        y = np.zeros((3, 1941), dtype=np.float32)
        y[:, [0, 100, 1800]] = 1.0
        poisoned = y.copy()
        poisoned[:, 1717:] = np.array([1.5, -1.0, np.nan] + [0.0] * (1941 - 1720), dtype=np.float32)
        fake_panel = {
            "y": poisoned,
            "series_id": np.array(["keep_b", "drop_a", "keep_a"]),
            "source": "fake existing pipeline",
        }

        with self.assertRaisesRegex(ValueError, "expected_source_hashes"):
            data.load_m5_via_existing_pipeline(Path("missing-data-dir"))
        with self.assertRaisesRegex(ValueError, "frozen"):
            data.load_m5_via_existing_pipeline(
                Path("missing-data-dir"),
                expected_source_hashes={
                    key: "0" * 64 for key in data.M5_EXPECTED_SOURCE_SHA256
                },
            )
        with tempfile.TemporaryDirectory() as directory:
            arbitrary = Path(directory) / "m5"
            arbitrary.mkdir()
            with self.assertRaisesRegex(ValueError, "source_snapshots.*m5"):
                data.load_m5_via_existing_pipeline(
                    arbitrary, expected_source_hashes=data.M5_EXPECTED_SOURCE_SHA256
                )
            with self.assertRaisesRegex(ValueError, "source_snapshots.*m5"):
                data.load_m5_stage_a_exclusion(
                    arbitrary, expected_sha256=data.M5_STAGE_A_EXPECTED_SHA256
                )
        with self.assertRaisesRegex(ValueError, "stage_a_exclusion"):
            data._load_m5_via_existing_pipeline_for_test(
                loader=lambda: fake_panel,
                expected_loader_identity="tests.fake_m5_loader",
                source_provenance=[
                    {
                        "path": "memory://m5",
                        "size_bytes": int(poisoned.nbytes),
                        "sha256": "0" * 64,
                        "source_identity": "unit-test fake M5 panel",
                        "adapter_function": "tests.fake_m5_loader",
                    }
                ],
            )
        too_large = y.astype(np.float64)
        too_large[0, 100] = 16_777_217.0
        with self.assertRaisesRegex(ValueError, "float32 round-trip"):
            data._load_m5_via_existing_pipeline_for_test(
                loader=lambda: {**fake_panel, "y": too_large},
                expected_loader_identity="tests.fake_m5_loader",
                source_provenance=[
                    {
                        "path": "memory://m5",
                        "size_bytes": int(too_large.nbytes),
                        "sha256": "0" * 64,
                        "source_identity": "unit-test fake M5 panel",
                        "adapter_function": "tests.fake_m5_loader",
                    }
                ],
                stage_a_exclusion=data._m5_stage_a_exclusion_for_test(["drop_a"]),
            )
        loaded = data._load_m5_via_existing_pipeline_for_test(
            loader=lambda: fake_panel,
            expected_loader_identity="tests.fake_m5_loader",
            source_provenance=[
                {
                    "path": "memory://m5",
                    "size_bytes": int(poisoned.nbytes),
                    "sha256": "0" * 64,
                    "source_identity": "unit-test fake M5 panel",
                    "adapter_function": "tests.fake_m5_loader",
                }
            ],
            stage_a_exclusion=data._m5_stage_a_exclusion_for_test(["drop_a"]),
        )

        self.assertEqual(loaded["name"], "m5")
        self.assertEqual(loaded["split"], data.REAL_SPLITS["m5"])
        self.assertEqual(loaded["series_id"].tolist(), ["keep_a", "keep_b"])
        self.assertEqual(loaded["y"].shape, (2, 1941))
        self.assertEqual(loaded["support_audit"]["support"], "nonnegative_integer_count")
        self.assertEqual(loaded["support_audit"]["n_observations"], 2 * data.REAL_SPLITS["m5"].train[1])
        self.assertEqual(loaded["provenance"]["sources"][0]["sha256"], "0" * 64)
        self.assertEqual(loaded["provenance"]["stage_a_exclusion"]["n_series"], 1)
        self.assertEqual(loaded["provenance"]["stage_a_overlap_count"], 0)
        self.assertEqual(loaded["provenance"]["support_audit"]["target_scope"], "model_train")
        self.assertIn("experiments.prob_head_structure_full_v1.data", loaded["provenance"]["local_adapter_function"])
        self.assertIn("confirmatory_h2.m5_full", loaded["provenance"]["upstream_reused_source"])
        self.assertEqual(
            data.validate_real_split(data.REAL_SPLITS["m5"], length=1941)["status"],
            "PASS",
        )
        with self.assertRaisesRegex(ValueError, "origin"):
            data.validate_real_split(data.REAL_SPLITS["m5"], length=1940)

    def test_real_dataset_selection_geometry_and_favorita_rejection(self):
        """Catches count-ineligible real datasets entering the confirmatory pool."""
        for dataset_id, length, horizon in (
            ("auto", 24, 6),
            ("carparts", 51, 6),
            ("raf", 84, 12),
        ):
            audit = data.audit_fixed_length_dataset(
                dataset_id=dataset_id, length=length, lookback=48, horizon=horizon
            )
            self.assertEqual(audit["status"], "INSUFFICIENT_LENGTH")
            self.assertFalse(audit["confirmatory_eligible"])
            self.assertFalse(audit["count_primary_eligible"])

        geometry_only_pass = data.audit_fixed_length_dataset(
            dataset_id="auto", length=1_000, lookback=48, horizon=6
        )
        self.assertEqual(geometry_only_pass["status"], "SUPPORT_SOURCE_AUDIT_REQUIRED")
        self.assertFalse(geometry_only_pass["confirmatory_eligible"])
        self.assertFalse(geometry_only_pass["count_primary_eligible"])

        favorita = data.audit_favorita_count_eligibility(
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            provenance={"transforms": ["np.rint"]},
        )
        self.assertEqual(favorita["status"], "COUNT_SUPPORT_FAIL")
        self.assertEqual(favorita["ineligible_reason"], "rounded_or_rint_provenance")
        benign = data.audit_favorita_count_eligibility(
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            provenance={"notes": "surrounding context contains no transform token"},
        )
        self.assertEqual(benign["status"], "PASS")

        audits = [
            self._sealed_count_primary_audit("m5"),
            data.audit_fixed_length_dataset(dataset_id="auto", length=24, lookback=48, horizon=6),
            data.audit_fixed_length_dataset(dataset_id="carparts", length=51, lookback=48, horizon=6),
            data.audit_fixed_length_dataset(dataset_id="raf", length=84, lookback=48, horizon=12),
            self._sealed_count_primary_audit("online_retail"),
        ]
        selection = data.select_real_datasets(audits)
        self.assertEqual(selection["priority"], ("m5", "auto", "carparts", "raf", "online_retail"))
        self.assertEqual(selection["selected_dataset_ids"], ["m5", "online_retail"])
        self.assertEqual(selection["status"], "PASS")
        self.assertEqual([row["dataset_id"] for row in selection["audits"]], list(data.REAL_DATASET_PRIORITY))
        self.assertRegex(selection["audit_manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(selection["selection_manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            data.verify_real_dataset_selection_manifest(selection), selection
        )
        forged_selection = deepcopy(selection)
        forged_selection["selected_dataset_ids"] = ["m5"]
        forged_selection["selected"] = forged_selection["selected"][:1]
        forged_selection["selection_manifest_sha256"] = (
            data._real_selection_manifest_hash(forged_selection)
        )
        with self.assertRaisesRegex(ValueError, "fixed-priority reconstruction"):
            data.verify_real_dataset_selection_manifest(forged_selection)
        fake_hash = dict(selection, selection_manifest_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "hash or schema"):
            data.verify_real_dataset_selection_manifest(fake_hash)
        with self.assertRaises(TypeError):
            data.select_real_datasets(audits, required_non_m5=1)

        limited_audits = [
            self._sealed_count_primary_audit("m5"),
            data.audit_fixed_length_dataset(dataset_id="auto", length=24, lookback=48, horizon=6),
            data.audit_fixed_length_dataset(dataset_id="carparts", length=51, lookback=48, horizon=6),
            data.audit_fixed_length_dataset(dataset_id="raf", length=84, lookback=48, horizon=12),
            data.audit_fixed_length_dataset(dataset_id="online_retail", length=1, lookback=96, horizon=28),
        ]
        limited = data.select_real_datasets(limited_audits)
        self.assertEqual(limited["status"], "REAL_CROSS_DATASET_EVIDENCE_LIMITED")
        with self.assertRaisesRegex(ValueError, "duplicate audit"):
            data.select_real_datasets(
                [
                    *audits,
                    self._sealed_count_primary_audit("m5"),
                ]
            )
        missing_m5_audits = [
            data.audit_fixed_length_dataset(dataset_id="m5", length=1, lookback=96, horizon=28),
            data.audit_fixed_length_dataset(dataset_id="auto", length=24, lookback=48, horizon=6),
            data.audit_fixed_length_dataset(dataset_id="carparts", length=51, lookback=48, horizon=6),
            data.audit_fixed_length_dataset(dataset_id="raf", length=84, lookback=48, horizon=12),
            self._sealed_count_primary_audit("online_retail"),
        ]
        missing_m5 = data.select_real_datasets(missing_m5_audits)
        self.assertEqual(missing_m5["selected_dataset_ids"], [])
        self.assertEqual(missing_m5["status"], "REAL_CROSS_DATASET_EVIDENCE_LIMITED")

        with self.assertRaisesRegex(ValueError, "complete canonical audit"):
            data.select_real_datasets([self._sealed_count_primary_audit("m5")])
        bare_rows = [
            {"dataset_id": dataset_id, "status": "PASS", "confirmatory_eligible": True}
            for dataset_id in data.REAL_DATASET_PRIORITY
        ]
        with self.assertRaisesRegex(ValueError, "sealed dataset audit"):
            data.select_real_datasets(bare_rows)
        tampered = [dict(row) for row in audits]
        tampered[0]["count_primary_eligible"] = False
        with self.assertRaisesRegex(ValueError, "hash"):
            data.select_real_datasets(tampered)

        forged = {
            "name": "m5",
            "support_audit": {
                "dataset_id": "m5",
                "status": "PASS",
                "target_scope": "model_train",
                "count_primary_eligible": True,
                "count_likelihood_index_exact": True,
            },
            "split_validation": {"status": "PASS"},
            "provenance": {
                "dataset_id": "m5",
                "sources": [
                    {
                        "path": "snapshot://missing",
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                        "expected_sha256": "0" * 64,
                        "status": "PASS",
                    }
                ],
            },
        }
        fake_manifest = {
            "schema_version": 2,
            "repository_root_identity": {
                "label": "forged",
                "resolved_path": Path(tempfile.gettempdir()).resolve().as_posix(),
            },
            "files": [],
            "aggregate_sha256": "0" * 64,
        }
        with self.assertRaisesRegex((ValueError, integrity.ContractViolation), "source|panel|manifest"):
            data.seal_count_primary_dataset_audit(forged, source_manifest=fake_manifest)


if __name__ == "__main__":
    unittest.main()
