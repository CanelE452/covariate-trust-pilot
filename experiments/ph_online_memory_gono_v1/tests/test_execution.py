"""Mock-only tests for append-only conditional full execution."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import pandas as pd
import torch

from experiments.ph_online_memory_gono_v1.artifacts import (
    file_sha256,
    payload_sha256,
)
from experiments.ph_online_memory_gono_v1.execution import (
    _default_heterogeneous_factory_builder,
    load_latest_completed_dataset,
    run_persisted_full_protocol,
    validate_runtime_authorization,
)


def _authorized_root(root: Path) -> dict[str, object]:
    prereg_payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_name": "PH-ONLINE-MEMORY-GONO-v1",
        "smoke": {"runtime_gate_gpu_hours": 6.0},
        "environment": {"python": "test"},
        "training": {"batch_size": 256},
        "models": {"point": {}, "hurdle": {}},
        "splits": {"m5": {}, "favorita": {}},
        "eligibility": {"primary_min_positive_train": 20},
    }
    prereg = dict(prereg_payload)
    prereg["preregistration_sha256"] = payload_sha256(prereg_payload)
    (root / "preregistered_spec.json").write_text(
        json.dumps(prereg), encoding="utf-8"
    )
    smoke_root = root / "smoke" / "attempt_0001"
    smoke_root.mkdir(parents=True)
    smoke_artifact = smoke_root / "predictions.parquet"
    smoke_artifact.write_bytes(b"immutable smoke evidence")
    runtime = {
        "experiment": "PH-ONLINE-MEMORY-GONO-v1",
        "stage": "M5_200_SERIES_SMOKE",
        "device": "cuda",
        "cuda_peak_memory_bytes": 1024,
        "preregistration_sha256": prereg["preregistration_sha256"],
        "attempt": {"id": "attempt_0001", "path": "smoke/attempt_0001"},
        "artifacts": {
            "predictions": {
                "path": "smoke/attempt_0001/predictions.parquet",
                "sha256": file_sha256(smoke_artifact),
            }
        },
        "runtime_gate": {
            "threshold_gpu_hours": 6.0,
            "projected_gpu_hours": 5.5,
            "exceeded": False,
            "action": "CONTINUE_FULL_SEED0",
        },
    }
    (root / "runtime_estimate.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )
    return prereg


def _arm_payload(dataset: str, seed: int, arm: str) -> dict[str, object]:
    model_id = {
        "point": "M0PM_point_mse_param_matched",
        "hurdle": "M1_factorized_mean",
    }[arm]
    return {
        "schema_version": 1,
        "dataset": dataset,
        "arm": arm,
        "model_id": model_id,
        "model_seed": seed,
        "state_dict": {"weight": torch.tensor([seed + (1 if arm == "point" else 2)])},
        "provenance": {
            "model_id": model_id,
            "model_seed": seed,
            "checkpoint_device": "cpu",
            "best_epoch": 2,
            "best_validation_mean_mse": 0.75,
            "train_seconds": 1.0,
            "n_parameters": 7056,
            "execution_device_type": "cuda",
            "end_to_end_wall_seconds": 1.25,
            "end_to_end_wall_definition": "device-synchronized test wall",
        },
        "predictions": {
            "mean_prediction": np.full((2, 3), seed + 0.25),
            **(
                {
                    "p_prediction": np.full((2, 3), 0.5),
                    "mu_prediction": np.full((2, 3), 2.0),
                }
                if arm == "hurdle"
                else {}
            ),
        },
    }


def _dataset_output(dataset: str, seed: int) -> dict[str, object]:
    base = pd.DataFrame(
        {
            "dataset_id": [dataset, dataset],
            "series_id": [f"{dataset}-a", f"{dataset}-b"],
            "origin": [100, 100],
        }
    )
    return {
        "dataset": dataset,
        "model_seed": seed,
        "schedule": {
            "warmup_origin": 72,
            "evaluation_origins": [100],
            "all_forecast_origins": [72, 100],
            "horizon": 28,
            "lookback": 96,
            "model_train_end": 44,
            "train_origin_stride": 7,
        },
        "population_manifest": {
            "dataset": dataset,
            "eligible_independent": 2,
        },
        "provenance": {
            arm: _arm_payload(dataset, seed, arm)["provenance"]
            for arm in ("point", "hurdle")
        },
        "state_dicts": {
            arm: _arm_payload(dataset, seed, arm)["state_dict"]
            for arm in ("point", "hurdle")
        },
        "predictions": base.assign(step=0, prediction=1.0),
        "step_predictions": base.assign(step=0, point=1.0, hurdle=2.0),
        "losses": base.assign(point_normalized_loss=1.0, hurdle_normalized_loss=2.0),
        "cases": base.assign(
            policy_scale_squared=1.0,
            history=[np.ones(3), np.ones(3)],
            point_forecast=[np.ones(3), np.ones(3)],
            hurdle_forecast=[np.ones(3), np.ones(3)],
            target=[np.zeros(3), np.zeros(3)],
            target_mask=[np.ones(3, dtype=bool), np.ones(3, dtype=bool)],
        ),
    }


def _terminal_seed0(_outputs, **_kwargs):
    return {
        "report": {
            "model_seed": 0,
            "next_action": "STOP",
            "terminal": True,
            "final_verdict": "FULL_NO_GO",
        },
        "tables": {"expert_evaluation_losses": pd.DataFrame({"loss": [1.0]})},
    }


class RuntimeAuthorizationTests(unittest.TestCase):
    def test_validates_prereg_gate_and_every_referenced_smoke_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prereg = _authorized_root(root)

            authorization = validate_runtime_authorization(root)

            self.assertEqual(
                authorization["preregistration"]["preregistration_sha256"],
                prereg["preregistration_sha256"],
            )
            self.assertEqual(authorization["runtime_gate"]["projected_gpu_hours"], 5.5)
            self.assertEqual(len(authorization["verified_smoke_artifacts"]), 1)

            artifact = root / "smoke" / "attempt_0001" / "predictions.parquet"
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "smoke artifact.*SHA-256"):
                validate_runtime_authorization(root)

    def test_rejects_each_non_authorizing_runtime_gate(self):
        mutations = {
            "wrong action": {"action": "STOP_FOR_APPROVAL"},
            "exceeded": {"exceeded": True},
            "above threshold": {"projected_gpu_hours": 6.0001},
            "threshold drift": {"threshold_gpu_hours": 7.0},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _authorized_root(root)
                path = root / "runtime_estimate.json"
                runtime = json.loads(path.read_text(encoding="utf-8"))
                runtime["runtime_gate"].update(mutation)
                path.write_text(json.dumps(runtime), encoding="utf-8")
                with self.assertRaises((RuntimeError, ValueError)):
                    validate_runtime_authorization(root)

    def test_runtime_authorization_requires_cuda_smoke_provenance(self):
        mutations = {
            "missing device": ("device", None),
            "cpu device": ("device", "cpu"),
            "zero peak": ("cuda_peak_memory_bytes", 0),
        }
        for label, (field, value) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _authorized_root(root)
                path = root / "runtime_estimate.json"
                runtime = json.loads(path.read_text(encoding="utf-8"))
                if value is None:
                    runtime.pop(field)
                else:
                    runtime[field] = value
                path.write_text(json.dumps(runtime), encoding="utf-8")
                with self.assertRaisesRegex((RuntimeError, ValueError), "CUDA|cuda"):
                    validate_runtime_authorization(root)

    def test_runtime_attempt_id_must_match_its_bound_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)
            path = root / "runtime_estimate.json"
            runtime = json.loads(path.read_text(encoding="utf-8"))
            runtime["attempt"]["id"] = "attempt_9999"
            path.write_text(json.dumps(runtime), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "attempt.*identity"):
                validate_runtime_authorization(root)

    def test_bad_runtime_gate_stops_before_upstream_check_loader_or_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)
            runtime_path = root / "runtime_estimate.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["runtime_gate"]["action"] = "STOP_FOR_APPROVAL"
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            touched: list[str] = []

            with mock.patch(
                "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts",
                side_effect=lambda *_: touched.append("upstream"),
            ):
                with self.assertRaisesRegex(RuntimeError, "CONTINUE_FULL_SEED0"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cpu"),
                        population_loader=lambda *_args, **_kwargs: touched.append("load"),
                        train_dataset=lambda *_args, **_kwargs: touched.append("train"),
                    )
            self.assertEqual(touched, [])

    def test_full_entry_requires_an_available_cuda_device_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)
            loaded: list[str] = []
            loader = lambda name, **_kwargs: loaded.append(name)
            with mock.patch(
                "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
            ):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cpu"),
                        population_loader=loader,
                    )
            self.assertEqual(loaded, [])

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=False,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "available CUDA"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=loader,
                    )
            self.assertEqual(loaded, [])

    def test_forbidden_artifact_drift_stops_before_population_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)
            loaded: list[str] = []
            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._verify_frozen_implementation"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._read_stage0_pass"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._read_pre_smoke_pass"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._verify_frozen_environment"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._verify_frozen_data_sources"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol.verify_forbidden_artifacts",
                    return_value={"status": "FAIL", "all_unchanged": False},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: loaded.append(name),
                    )
            self.assertEqual(loaded, [])

    def test_phase0_authorization_drift_stops_before_population_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)
            loaded: list[str] = []
            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution."
                    "torch.cuda.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol."
                    "_validate_phase0_authorization",
                    side_effect=RuntimeError("Phase 0 drift"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Phase 0 drift"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: loaded.append(name),
                    )
            self.assertEqual(loaded, [])

class DatasetCheckpointTests(unittest.TestCase):
    def test_arm_checkpoint_requires_canonical_training_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)

            def missing_train_seconds(
                population,
                _device,
                *,
                model_seed,
                persisted_arms,
                on_arm_complete,
            ):
                del persisted_arms
                payload = _arm_payload(population["name"], model_seed, "point")
                payload["provenance"].pop("train_seconds")
                on_arm_complete("point", payload)
                raise AssertionError("invalid arm callback should not return")

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                with self.assertRaisesRegex((TypeError, ValueError), "train_seconds"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: {"name": name},
                        train_dataset=missing_train_seconds,
                        seed0_analyzer=_terminal_seed0,
                    )

    def test_arm_checkpoint_rejects_cpu_provenance_in_full_cuda_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)

            def cpu_arm_train(
                population,
                _device,
                *,
                model_seed,
                persisted_arms,
                on_arm_complete,
            ):
                del persisted_arms
                payload = _arm_payload(population["name"], model_seed, "point")
                payload["provenance"]["execution_device_type"] = "cpu"
                on_arm_complete("point", payload)
                raise AssertionError("CPU arm callback should not return")

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "execution_device_type.*cuda"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: {"name": name},
                        train_dataset=cpu_arm_train,
                        seed0_analyzer=_terminal_seed0,
                    )

    def test_arm_checkpoint_requires_actual_device_wall_time_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)

            def missing_runtime_train(
                population,
                _device,
                *,
                model_seed,
                persisted_arms,
                on_arm_complete,
            ):
                del persisted_arms
                payload = _arm_payload(population["name"], model_seed, "point")
                payload["provenance"].pop("end_to_end_wall_seconds")
                on_arm_complete("point", payload)
                raise AssertionError("invalid arm callback should not return")

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "end_to_end_wall_seconds"
                ):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: {"name": name},
                        train_dataset=missing_runtime_train,
                        seed0_analyzer=_terminal_seed0,
                    )

    def test_dataset_completion_rejects_state_that_differs_from_arm_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)

            def inconsistent_train(
                population,
                _device,
                *,
                model_seed,
                persisted_arms,
                on_arm_complete,
            ):
                self.assertEqual(persisted_arms, {})
                for arm in ("point", "hurdle"):
                    on_arm_complete(
                        arm, _arm_payload(population["name"], model_seed, arm)
                    )
                output = _dataset_output(population["name"], model_seed)
                output["state_dicts"]["point"]["weight"] = torch.tensor([999])
                return output

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "state_dict.*arm checkpoint"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: {"name": name},
                        train_dataset=inconsistent_train,
                        seed0_analyzer=_terminal_seed0,
                    )
            attempt = root / "seed0" / "datasets" / "m5" / "attempt_0001"
            self.assertFalse((attempt / "completion.json").exists())
            self.assertTrue((attempt / "failure.json").is_file())

    def test_arm_failure_is_immutable_and_retry_reuses_completed_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prereg = _authorized_root(root)
            train_calls: list[tuple[str, int, tuple[str, ...]]] = []

            def first_train(population, _device, *, model_seed, persisted_arms, on_arm_complete):
                dataset = population["name"]
                train_calls.append((dataset, model_seed, tuple(sorted(persisted_arms))))
                on_arm_complete("point", _arm_payload(dataset, model_seed, "point"))
                raise RuntimeError("hurdle crashed")

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "hurdle crashed"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: {"name": name},
                        train_dataset=first_train,
                        seed0_analyzer=_terminal_seed0,
                    )

            failed = root / "seed0" / "datasets" / "m5" / "attempt_0001"
            self.assertTrue((failed / "point_completion.json").is_file())
            self.assertTrue((failed / "failure.json").is_file())
            self.assertFalse((failed / "completion.json").exists())
            preserved_point_sha = file_sha256(failed / "point_checkpoint.pt")

            def resumed_train(population, _device, *, model_seed, persisted_arms, on_arm_complete):
                dataset = population["name"]
                train_calls.append((dataset, model_seed, tuple(sorted(persisted_arms))))
                if dataset == "m5":
                    self.assertEqual(set(persisted_arms), {"point"})
                    on_arm_complete("hurdle", _arm_payload(dataset, model_seed, "hurdle"))
                else:
                    self.assertEqual(persisted_arms, {})
                    for arm in ("point", "hurdle"):
                        on_arm_complete(arm, _arm_payload(dataset, model_seed, arm))
                return _dataset_output(dataset, model_seed)

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                result = run_persisted_full_protocol(
                    result_root=root,
                    device=torch.device("cuda"),
                    population_loader=lambda name, **_kwargs: {"name": name},
                    train_dataset=resumed_train,
                    seed0_analyzer=_terminal_seed0,
                )

            self.assertEqual(result["report"]["final_verdict"], "FULL_NO_GO")
            m5_summary = result["seed_results"][0]["dataset_outputs"]["m5"]
            self.assertEqual(m5_summary["dataset"], "m5")
            self.assertIn("provenance", m5_summary)
            self.assertNotIn("cases", m5_summary)
            self.assertEqual(
                m5_summary["provenance"]["point"]["execution_device_type"],
                "cuda",
            )
            self.assertEqual(train_calls[1], ("m5", 0, ("point",)))
            self.assertEqual(
                file_sha256(failed / "point_checkpoint.pt"), preserved_point_sha
            )
            completed = root / "seed0" / "datasets" / "m5" / "attempt_0002"
            self.assertTrue((completed / "completion.json").is_file())
            original_failure = json.loads((failed / "failure.json").read_text("utf-8"))
            self.assertEqual(original_failure["exception_message"], "hurdle crashed")
            self.assertEqual(
                prereg["preregistration_sha256"],
                json.loads((completed / "completion.json").read_text("utf-8"))[
                    "preregistration_sha256"
                ],
            )

    def test_completed_dataset_round_trips_and_tampering_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prereg = _authorized_root(root)

            def train(population, _device, *, model_seed, persisted_arms, on_arm_complete):
                self.assertEqual(persisted_arms, {})
                for arm in ("point", "hurdle"):
                    on_arm_complete(arm, _arm_payload(population["name"], model_seed, arm))
                return _dataset_output(population["name"], model_seed)

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                run_persisted_full_protocol(
                    result_root=root,
                    device=torch.device("cuda"),
                    population_loader=lambda name, **_kwargs: {"name": name},
                    train_dataset=train,
                    seed0_analyzer=_terminal_seed0,
                )

            runtime_sha = file_sha256(root / "runtime_estimate.json")
            with warnings.catch_warnings(record=True) as observed_warnings:
                warnings.simplefilter("always")
                completed = load_latest_completed_dataset(
                    root,
                    dataset="m5",
                    model_seed=0,
                    preregistration_sha256=prereg["preregistration_sha256"],
                    runtime_estimate_sha256=runtime_sha,
                )
            self.assertFalse(
                [
                    warning
                    for warning in observed_warnings
                    if "TypedStorage is deprecated" in str(warning.message)
                ]
            )
            self.assertIsNotNone(completed)
            assert completed is not None
            self.assertEqual(completed.output["dataset"], "m5")
            self.assertEqual(set(completed.output["state_dicts"]), {"point", "hurdle"})
            pd.testing.assert_frame_equal(
                completed.output["losses"], _dataset_output("m5", 0)["losses"]
            )

            prediction_path = completed.attempt_root / "predictions.parquet"
            prediction_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "artifact.*SHA-256"):
                load_latest_completed_dataset(
                    root,
                    dataset="m5",
                    model_seed=0,
                    preregistration_sha256=prereg["preregistration_sha256"],
                    runtime_estimate_sha256=runtime_sha,
                )

    def test_final_restart_reconstructs_seed_summaries_without_loading_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)

            def train(population, _device, *, model_seed, persisted_arms, on_arm_complete):
                self.assertEqual(persisted_arms, {})
                for arm in ("point", "hurdle"):
                    on_arm_complete(
                        arm, _arm_payload(population["name"], model_seed, arm)
                    )
                return _dataset_output(population["name"], model_seed)

            common_patches = (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            )
            with common_patches[0], common_patches[1]:
                run_persisted_full_protocol(
                    result_root=root,
                    device=torch.device("cuda"),
                    population_loader=lambda name, **_kwargs: {"name": name},
                    train_dataset=train,
                    seed0_analyzer=_terminal_seed0,
                )

            touched: list[str] = []
            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                resumed = run_persisted_full_protocol(
                    result_root=root,
                    device=torch.device("cuda"),
                    population_loader=lambda name, **_kwargs: touched.append(name),
                    train_dataset=lambda *_args, **_kwargs: touched.append("train"),
                )
            self.assertEqual(touched, [])
            self.assertEqual(set(resumed["seed_results"]), {0})
            self.assertEqual(
                resumed["seed_results"][0]["dataset_outputs"]["favorita"][
                    "population_manifest"
                ]["dataset"],
                "favorita",
            )
            self.assertIn("expert_evaluation_losses", resumed["seed_results"][0]["tables"])


class ConditionalSeedTests(unittest.TestCase):
    def test_default_heterogeneous_fallback_never_holds_two_raw_populations(self):
        active: set[str] = set()

        class Population(dict):
            def __init__(self, dataset: str):
                super().__init__(dataset=dataset)
                self.dataset = dataset
                active.add(dataset)

            def __del__(self):
                active.discard(self.dataset)

        def loader(dataset, **_kwargs):
            self.assertEqual(active, set())
            return Population(dataset)

        def evaluate(population, _output, dataset):
            self.assertEqual(active, {dataset})
            self.assertEqual(population["dataset"], dataset)
            return {
                "dataset": dataset,
                "gain_percent": 1.0,
                "details": {},
                "hand_check": {"passed": True},
            }

        with (
            mock.patch(
                "experiments.ph_online_memory_gono_v1.heterogeneous.evaluate_heterogeneous_dataset",
                new=evaluate,
            ),
            mock.patch(
                "experiments.ph_online_memory_gono_v1.heterogeneous.assemble_heterogeneous_diagnostic",
                return_value={"passed": False},
            ) as assemble,
        ):
            factory = _default_heterogeneous_factory_builder(loader)
            result = factory({}, {"m5": {}, "favorita": {}})
        self.assertEqual(result, {"passed": False})
        self.assertEqual(active, set())
        assemble.assert_called_once()

    def test_forbidden_artifacts_are_rechecked_before_final_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)

            def train(population, _device, *, model_seed, persisted_arms, on_arm_complete):
                self.assertEqual(persisted_arms, {})
                for arm in ("point", "hurdle"):
                    on_arm_complete(
                        arm, _arm_payload(population["name"], model_seed, arm)
                    )
                return _dataset_output(population["name"], model_seed)

            integrity = mock.Mock(
                side_effect=[
                    {"status": "PASS", "all_unchanged": True},
                    {"status": "FAIL", "all_unchanged": False},
                ]
            )
            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._verify_frozen_implementation"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._read_stage0_pass"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._read_pre_smoke_pass"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._verify_frozen_environment"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol._verify_frozen_data_sources"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.protocol.verify_forbidden_artifacts",
                    integrity,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: {"name": name},
                        train_dataset=train,
                        seed0_analyzer=_terminal_seed0,
                    )
            self.assertEqual(integrity.call_count, 2)
            self.assertFalse((root / "final_gate_report.json").exists())

    def test_analysis_failure_gets_an_immutable_failed_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)

            def train(population, _device, *, model_seed, persisted_arms, on_arm_complete):
                self.assertEqual(persisted_arms, {})
                for arm in ("point", "hurdle"):
                    on_arm_complete(
                        arm, _arm_payload(population["name"], model_seed, arm)
                    )
                return _dataset_output(population["name"], model_seed)

            def failed_analysis(_outputs, **_kwargs):
                raise RuntimeError("analysis crashed")

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "analysis crashed"):
                    run_persisted_full_protocol(
                        result_root=root,
                        device=torch.device("cuda"),
                        population_loader=lambda name, **_kwargs: {"name": name},
                        train_dataset=train,
                        seed0_analyzer=failed_analysis,
                    )
            attempt = root / "seed0" / "analysis" / "attempt_0001"
            self.assertTrue((attempt / "failure.json").is_file())
            self.assertFalse((attempt / "completion.json").exists())

    def test_seed1_and_seed2_run_only_when_prior_gate_requests_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)
            trained_seeds: list[int] = []

            def train(population, _device, *, model_seed, persisted_arms, on_arm_complete):
                del persisted_arms
                trained_seeds.append(model_seed)
                for arm in ("point", "hurdle"):
                    on_arm_complete(arm, _arm_payload(population["name"], model_seed, arm))
                return _dataset_output(population["name"], model_seed)

            def seed0(_outputs, **_kwargs):
                return {
                    "report": {
                        "model_seed": 0,
                        "next_action": "RUN_SEED1",
                        "terminal": False,
                        "gate2": {
                            "passed": False,
                            "decision_state": "PENDING_GATE4",
                        },
                        "gate4": {"action": "RUN_SEED1", "passed": None},
                    },
                    "tables": {
                        "target_policy_losses": pd.DataFrame(
                            {
                                "dataset_id": ["m5", "favorita"],
                                "series_id": ["a", "b"],
                                "origin": [1, 1],
                                "m1_normalized_loss": [1.0, 1.0],
                                "b4_normalized_loss": [2.0, 2.0],
                                "b3_normalized_loss": [3.0, 3.0],
                            }
                        )
                    },
                }

            def additional(_outputs, *, model_seed, **_kwargs):
                return {
                    "report": {
                        "model_seed": model_seed,
                        "next_action": "SEED_POLICY_READY",
                        "terminal": False,
                    },
                    "tables": {
                        "target_policy_losses": pd.DataFrame(
                            {
                                "dataset_id": ["m5", "favorita"],
                                "series_id": ["a", "b"],
                                "origin": [1, 1],
                                "m1_normalized_loss": [1.0, 1.0],
                                "b4_normalized_loss": [2.0, 2.0],
                                "b3_normalized_loss": [3.0, 3.0],
                            }
                        )
                    },
                }

            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.evaluate_gate4_seed1",
                    return_value={"action": "RUN_SEED2", "passed": False},
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.evaluate_gate4_seed2",
                    return_value={"action": "ACCEPT_THREE_SEED", "passed": True},
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.evaluate_seed_average_gate2",
                    return_value={
                        "gate2": {"passed": True},
                        "seed_average_losses": pd.DataFrame(
                            {
                                "dataset_id": ["m5", "favorita"],
                                "series_id": ["a", "b"],
                                "origin": [1, 1],
                                "m1_normalized_loss": [1.0, 1.0],
                                "b4_normalized_loss": [2.0, 2.0],
                                "b3_normalized_loss": [3.0, 3.0],
                            }
                        ),
                        "m1_vs_b4": {},
                        "m1_vs_b3": {},
                        "b4_vs_b3": {},
                        "bootstrap": {},
                        "aggregation": "test seed average",
                    },
                ) as averaged_gate2,
            ):
                result = run_persisted_full_protocol(
                    result_root=root,
                    device=torch.device("cuda"),
                    population_loader=lambda name, **_kwargs: {"name": name},
                    train_dataset=train,
                    seed0_analyzer=seed0,
                    additional_seed_analyzer=additional,
                )

                resumed = run_persisted_full_protocol(
                    result_root=root,
                    device=torch.device("cuda"),
                    population_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("restart must not reload populations")
                    ),
                    train_dataset=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("restart must not retrain")
                    ),
                )

            self.assertEqual(trained_seeds, [0, 0, 1, 1, 2, 2])
            self.assertEqual(result["report"]["executed_model_seeds"], [0, 1, 2])
            self.assertEqual(result["report"]["final_verdict"], "RETRIEVAL_MEMORY_GO")
            self.assertTrue(result["report"]["averaged_gate2"]["passed"])
            self.assertIn("seed_average_policy_losses", result["tables"])
            self.assertEqual(result["tables"].keys(), resumed["tables"].keys())
            pd.testing.assert_frame_equal(
                result["tables"]["seed_average_policy_losses"],
                resumed["tables"]["seed_average_policy_losses"],
            )
            self.assertGreaterEqual(averaged_gate2.call_count, 2)
            self.assertTrue((root / "seed2" / "analysis" / "attempt_0001" / "completion.json").is_file())
            self.assertTrue((root / "final_gate_report.json").is_file())

    def test_persisted_gate4_pass_cannot_override_failed_averaged_gate2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _authorized_root(root)

            def train(population, _device, *, model_seed, persisted_arms, on_arm_complete):
                del persisted_arms
                for arm in ("point", "hurdle"):
                    on_arm_complete(
                        arm, _arm_payload(population["name"], model_seed, arm)
                    )
                return _dataset_output(population["name"], model_seed)

            def policy_frame():
                return pd.DataFrame(
                    {
                        "dataset_id": ["m5", "favorita"],
                        "series_id": ["a", "b"],
                        "origin": [1, 1],
                        "m1_normalized_loss": [1.0, 1.0],
                        "b4_normalized_loss": [2.0, 2.0],
                        "b3_normalized_loss": [3.0, 3.0],
                    }
                )

            def seed0(_outputs, **_kwargs):
                return {
                    "report": {
                        "model_seed": 0,
                        "next_action": "RUN_SEED1",
                        "terminal": False,
                        "gate2": {
                            "passed": False,
                            "decision_state": "PENDING_GATE4",
                        },
                        "gate4": {"action": "RUN_SEED1", "passed": None},
                    },
                    "tables": {"target_policy_losses": policy_frame()},
                }

            def additional(_outputs, *, model_seed, **_kwargs):
                return {
                    "report": {
                        "model_seed": model_seed,
                        "next_action": "SEED_POLICY_READY",
                        "terminal": False,
                    },
                    "tables": {"target_policy_losses": policy_frame()},
                }

            averaged = {
                "gate2": {"passed": False},
                "seed_average_losses": policy_frame(),
                "m1_vs_b4": {},
                "m1_vs_b3": {},
                "b4_vs_b3": {},
                "bootstrap": {},
                "aggregation": "test seed average",
            }
            with (
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution._verify_upstream_artifacts"
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.torch.cuda.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.evaluate_gate4_seed1",
                    return_value={"action": "ACCEPT_TWO_SEED", "passed": True},
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.evaluate_seed_average_gate2",
                    return_value=averaged,
                ),
                mock.patch(
                    "experiments.ph_online_memory_gono_v1.execution.evaluate_gate4_seed2"
                ) as gate4_seed2,
            ):
                result = run_persisted_full_protocol(
                    result_root=root,
                    device=torch.device("cuda"),
                    population_loader=lambda name, **_kwargs: {"name": name},
                    train_dataset=train,
                    seed0_analyzer=seed0,
                    additional_seed_analyzer=additional,
                )

            gate4_seed2.assert_not_called()
            self.assertEqual(
                result["report"]["final_verdict"],
                "SIMPLE_ONLINE_COMBINATION_GO_RETRIEVAL_NO_GO",
            )
            self.assertFalse(result["report"]["averaged_gate2"]["passed"])


if __name__ == "__main__":
    unittest.main()
