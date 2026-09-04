from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.util
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = ROOT / "results"
RAW_PATHS = {
    "m5": (
        REFERENCE_ROOT
        / "external_validity_screen"
        / "rule_replication"
        / "independent_raw_predictions.parquet"
    ),
    "favorita": (
        REFERENCE_ROOT
        / "external_validity_screen"
        / "favorita_independent"
        / "independent_raw_predictions.parquet"
    ),
}
REFERENCE_PATHS = {
    "condition_discovery": (
        REFERENCE_ROOT / "pointhurdle_condition_discovery" / "paired_panel.parquet"
    ),
    "recoverability": (
        REFERENCE_ROOT
        / "pointhurdle_recoverability"
        / "multi_origin_paired_panel.parquet"
    ),
}
EXPECTED_SHA256 = {
    "m5": "a2810787033baac622e6558a53da526f0cb9b2e80d09cf2c4dbe699d2f207f6f",
    "favorita": "0df546bb479f70b8667c515c421ffe2a4f91dada9f1d50c0ee0300a776c22822",
    "condition_discovery": (
        "0917e4cc69948c72bf9c33afcccf969a8dc8bb2c74271bb00777cba1d5da532f"
    ),
    "recoverability": (
        "4c21015897e140de30177d7b8c97093c8ccadbf3ed00545ef8070e6206485e41"
    ),
}
EXPECTED_RAW_ROWS = {"m5": 478_212, "favorita": 454_020}
EXPECTED_PANEL_ROWS = {"m5": 17_079, "favorita": 16_215}
EXPECTED_SERIES = {"m5": 5_693, "favorita": 5_405}
EXPECTED_ORIGINS = {
    "m5": [1857, 1885, 1913],
    "favorita": [1604, 1632, 1660],
}
EXPECTED_WINNER_COUNTS = {"point": 11_269, "neutral": 7_101, "hurdle": 14_924}
EXPECTED_WINNER_RATES = {
    "point": 33.846939388478404,
    "neutral": 21.32816723734006,
    "hurdle": 44.824893374181535,
}
EXPECTED_STATIC_CONVEX = {
    "favorita": {
        "loss_50_50": 2.4211301803588867,
        "best_static_hurdle_weight": 0.85,
        "best_static_loss": 2.407510995864868,
        "series_oracle_loss": 2.355938,
        "origin_oracle_loss": 2.3281784,
    },
    "m5": {
        "loss_50_50": 1.631527066230774,
        "best_static_hurdle_weight": 0.6,
        "best_static_loss": 1.6311854124069214,
        "series_oracle_loss": 1.6096307,
        "origin_oracle_loss": 1.5923551,
    },
}
EXPECTED_BOOTSTRAP_POINT_ESTIMATES = {
    "m5": 0.5612785090289383,
    "favorita": -0.47483706629762645,
}
EXPECTED_RAW_COLUMNS = (
    "series_id",
    "group",
    "origin",
    "step",
    "y_observed",
    "occurrence",
    "target_mask",
    "point_mean_prediction",
    "hurdle_mean_prediction",
    "hurdle_p_prediction",
    "hurdle_mu_prediction",
)
MODULE_NAME = "experiments.ph_online_memory_gono_v1.reproduction"


def _module_available() -> bool:
    try:
        return importlib.util.find_spec(MODULE_NAME) is not None
    except ModuleNotFoundError:
        return False


MODULE_AVAILABLE = _module_available()
REAL_READ_PARQUET = pd.read_parquet
REAL_PATH_OPEN = Path.open
REAL_BUILTIN_OPEN = builtins.open


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_hashes() -> dict[str, str]:
    return {
        "m5": _sha256(RAW_PATHS["m5"]),
        "favorita": _sha256(RAW_PATHS["favorita"]),
        "condition_discovery": _sha256(REFERENCE_PATHS["condition_discovery"]),
        "recoverability": _sha256(REFERENCE_PATHS["recoverability"]),
    }


def _reject_write(*_args, **_kwargs):
    raise AssertionError("reproduce_three_origin must not write")


def _guarded_path_open(path: Path, mode: str = "r", *args, **kwargs):
    if any(flag in mode for flag in "wax+"):
        _reject_write(path, mode)
    return REAL_PATH_OPEN(path, mode, *args, **kwargs)


def _guarded_builtin_open(file, mode: str = "r", *args, **kwargs):
    if any(flag in mode for flag in "wax+"):
        _reject_write(file, mode)
    return REAL_BUILTIN_OPEN(file, mode, *args, **kwargs)


@contextmanager
def _no_writes():
    dataframe_writers = (
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_json",
        "to_orc",
        "to_parquet",
        "to_pickle",
        "to_sql",
    )
    series_writers = ("to_csv", "to_excel", "to_hdf", "to_json", "to_pickle", "to_sql")
    path_writers = ("mkdir", "rename", "replace", "touch", "unlink", "write_bytes", "write_text")
    with ExitStack() as stack:
        stack.enter_context(patch.object(Path, "open", new=_guarded_path_open))
        stack.enter_context(patch.object(builtins, "open", new=_guarded_builtin_open))
        stack.enter_context(patch.object(os, "makedirs", side_effect=_reject_write))
        for method in dataframe_writers:
            if hasattr(pd.DataFrame, method):
                stack.enter_context(
                    patch.object(pd.DataFrame, method, side_effect=_reject_write)
                )
        for method in series_writers:
            if hasattr(pd.Series, method):
                stack.enter_context(patch.object(pd.Series, method, side_effect=_reject_write))
        for method in path_writers:
            stack.enter_context(patch.object(Path, method, side_effect=_reject_write))
        yield


def _load_api():
    module = importlib.import_module(MODULE_NAME)
    return module, module.reproduce_three_origin


@unittest.skipUnless(MODULE_AVAILABLE, "Task 3 reproduction module is not implemented yet")
class ReproductionPureDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with _no_writes():
            cls.module, _reproduce = _load_api()

    def _helper(self, name):
        helper = getattr(self.module, name, None)
        self.assertTrue(callable(helper), f"missing pure reproduction helper {name}")
        return helper

    def test_static_convex_grid_has_every_weight_and_correct_endpoints(self):
        steps = pd.DataFrame(
            {
                "dataset_id": ["toy", "toy"],
                "series_id": ["s1", "s1"],
                "origin": [10, 10],
                "y_observed": np.array([0.0, 2.0], dtype=np.float32),
                "point_mean_prediction": np.array([0.0, 0.0], dtype=np.float32),
                "hurdle_mean_prediction": np.array([0.0, 2.0], dtype=np.float32),
            }
        )

        losses = self._helper("_convex_loss_grid")(steps)

        self.assertEqual(list(losses.columns), [index / 20 for index in range(21)])
        self.assertAlmostEqual(float(losses.iloc[0][0.0]), np.sqrt(2.0))
        self.assertAlmostEqual(float(losses.iloc[0][0.5]), np.sqrt(0.5))
        self.assertEqual(float(losses.iloc[0][1.0]), 0.0)
        self.assertTrue(bool(losses.dtypes.eq(np.dtype(np.float32)).all()))
        self.assertLessEqual(
            float(losses.iloc[0].max()),
            max(float(losses.iloc[0][0.0]), float(losses.iloc[0][1.0])),
        )

    def test_exponential_hurdle_weight_has_correct_sign_at_extreme_scale(self):
        weights = self._helper("_stable_exponential_hurdle_weight")(
            point_loss=np.array([1_000.0, 2_000.0, 1_000_000.0]),
            hurdle_loss=np.array([2_000.0, 1_000.0, 1_000_000.0]),
            eta=1_000_000.0,
        )

        self.assertTrue(bool(np.isfinite(weights).all()))
        np.testing.assert_array_equal(weights, np.array([0.0, 1.0, 0.5]))

    def test_exponential_hurdle_weight_rejects_nonfinite_loss(self):
        with self.assertRaisesRegex(ValueError, "(?i)(finite|nan)"):
            self._helper("_stable_exponential_hurdle_weight")(
                point_loss=np.array([1.0, np.nan]),
                hurdle_loss=np.array([2.0, 3.0]),
                eta=8.0,
            )

    def test_exponential_hurdle_weight_rejects_bad_implementation_outputs(self):
        helper = self._helper("_stable_exponential_hurdle_weight")
        with patch.object(
            self.module,
            "exponential_hurdle_weight",
            return_value=np.array([0.9]),
        ), self.assertRaisesRegex(ValueError, "(?i)(sign|weight)"):
            helper(np.array([1.0]), np.array([2.0]), eta=8.0)
        with patch.object(
            self.module,
            "exponential_hurdle_weight",
            return_value=np.array([np.nan]),
        ), self.assertRaisesRegex(ValueError, "(?i)(finite|nan)"):
            helper(np.array([1.0]), np.array([2.0]), eta=8.0)
        with patch.object(
            self.module,
            "exponential_hurdle_weight",
            return_value=np.array([-1.0]),
        ), self.assertRaisesRegex(ValueError, "(?i)(inside|weight|range)"):
            helper(np.array([1.0]), np.array([2.0]), eta=8.0)
        with patch.object(
            self.module,
            "exponential_hurdle_weight",
            return_value=np.array([0.1]),
        ), self.assertRaisesRegex(ValueError, "(?i)(shape|weight)"):
            helper(
                np.array([1.0, 1.0]),
                np.array([2.0, 2.0]),
                eta=8.0,
            )

    def test_discounted_weight_gives_the_most_recent_loss_more_influence(self):
        weights = self._helper("_discounted_hurdle_weights")(
            point_history=np.array([[100.0, 1.0], [1.0, 100.0]]),
            hurdle_history=np.array([[1.0, 100.0], [100.0, 1.0]]),
            eta=1.0,
            half_life=1,
        )

        np.testing.assert_allclose(
            weights,
            np.array([1.7832472907828393e-11, 0.9999999999821675]),
            rtol=0.0,
            atol=1e-15,
        )

    def test_discounted_weight_rejects_nonfinite_history(self):
        with self.assertRaisesRegex(ValueError, "(?i)(finite|nan)"):
            self._helper("_discounted_hurdle_weights")(
                point_history=np.array([[1.0, np.nan]]),
                hurdle_history=np.array([[2.0, 3.0]]),
                eta=8.0,
                half_life=1,
            )

    def test_series_cluster_resample_keeps_all_three_origin_rows(self):
        indices = self._helper("_series_cluster_resample_indices")(
            np.repeat(np.array(["a", "b", "c"]), 3),
            np.array(["c", "a", "c"]),
        )

        np.testing.assert_array_equal(
            indices, np.array([6, 7, 8, 0, 1, 2, 6, 7, 8])
        )

    def test_series_cluster_bootstrap_is_seeded_deterministic_and_finite(self):
        series_ids = np.repeat(np.array(["a", "b", "c"]), 3)
        baseline = np.repeat(np.array([10.0, 20.0, 30.0]), 3)
        candidate = np.repeat(np.array([9.0, 22.0, 15.0]), 3)
        expected = np.array([50.0, 100.0 * 31.0 / 70.0, 35.0, 100.0 * 14.0 / 60.0])

        bootstrap = self._helper(
            "_series_cluster_bootstrap_relative_improvement"
        )
        first = bootstrap(
            series_ids,
            baseline,
            candidate,
            draws=4,
            seed=20260904,
        )
        second = bootstrap(
            series_ids,
            baseline,
            candidate,
            draws=4,
            seed=20260904,
        )

        np.testing.assert_allclose(first, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(second, first)
        self.assertTrue(bool(np.isfinite(first).all()))

    def test_series_cluster_bootstrap_rejects_nonfinite_loss(self):
        with self.assertRaisesRegex(ValueError, "(?i)(finite|nan)"):
            self._helper("_series_cluster_bootstrap_relative_improvement")(
                np.repeat(np.array(["a", "b"]), 3),
                np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
                np.array([0.5, 0.5, np.nan, 1.0, 1.0, 1.0]),
                draws=4,
                seed=20260904,
            )

    def test_constant_predictor_spearman_is_degenerate(self):
        diagnostic = self._helper("_constant_spearman_diagnostic")(
            np.array([4.0, 1.0, 3.0, 2.0])
        )

        self.assertEqual(
            diagnostic,
            {"status": "DEGENERATE", "rho": None, "pvalue": None, "n": 4},
        )


class ReproductionAvailabilityTests(unittest.TestCase):
    def test_read_only_reproduction_api_exists(self):
        if not MODULE_AVAILABLE:
            self.fail(
                "experiments.ph_online_memory_gono_v1.reproduction is absent "
                "(expected RED before Task 3 implementation)"
            )
        with _no_writes():
            _module, reproduce_three_origin = _load_api()
        self.assertTrue(callable(reproduce_three_origin))


@unittest.skipUnless(MODULE_AVAILABLE, "Task 3 reproduction module is not implemented yet")
class ReproductionBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with _no_writes():
            cls.module, reproduce_three_origin = _load_api()
            cls.reproduce_three_origin = staticmethod(reproduce_three_origin)
        cls.real_raw = {
            dataset: REAL_READ_PARQUET(path) for dataset, path in RAW_PATHS.items()
        }
        for dataset, frame in cls.real_raw.items():
            if tuple(frame.columns) != EXPECTED_RAW_COLUMNS:
                raise AssertionError(f"frozen {dataset} raw schema drifted")

    def _reproduce(self, raw_paths=None):
        return self.reproduce_three_origin(raw_paths or RAW_PATHS, REFERENCE_ROOT)

    def _reproduce_with_m5(self, corrupted: pd.DataFrame):
        m5_path = RAW_PATHS["m5"].resolve()

        def read_parquet(path, *args, **kwargs):
            if Path(path).resolve() == m5_path:
                return corrupted.copy(deep=True)
            return REAL_READ_PARQUET(path, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(_no_writes())
            stack.enter_context(
                patch.object(pd, "read_parquet", side_effect=read_parquet)
            )
            direct_readers = [
                name
                for name, value in vars(self.module).items()
                if value is REAL_READ_PARQUET
            ]
            for name in direct_readers:
                stack.enter_context(
                    patch.object(
                        self.module, name, side_effect=read_parquet
                    )
                )
            return self._reproduce()

    def test_real_frozen_inputs_reproduce_without_writing(self):
        before = _frozen_hashes()
        self.assertEqual(before, EXPECTED_SHA256)

        with _no_writes():
            report = self._reproduce()

        self.assertEqual(_frozen_hashes(), before)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["input_sha256"], {
            dataset: EXPECTED_SHA256[dataset] for dataset in RAW_PATHS
        })
        self.assertEqual(report["raw_rows"], EXPECTED_RAW_ROWS)
        self.assertEqual(report["panel_rows"], 33_294)
        self.assertEqual(report["dataset_rows"], EXPECTED_PANEL_ROWS)
        self.assertEqual(report["series"], EXPECTED_SERIES)
        self.assertEqual(report["origins"], EXPECTED_ORIGINS)

        for check in (
            "schema",
            "row_counts",
            "keys_unique",
            "pairing",
            "masks",
            "finiteness",
            "occurrence_identity",
            "hurdle_factorization",
            "condition_discovery_reference",
            "recoverability_reference",
            "aggregate_tolerances",
            "static_convex_grid",
            "exponential_weighting",
            "series_cluster_bootstrap",
            "constant_spearman_degenerate",
        ):
            with self.subTest(check=check):
                self.assertIs(report["checks"][check], True)

        self.assertLessEqual(report["panel_max_abs_diff"], 1e-7)
        self.assertLessEqual(
            report["recoverability_panel_max_abs_gain_diff"], 5e-5
        )
        self.assertEqual(report["winner_counts"], EXPECTED_WINNER_COUNTS)
        for winner, expected in EXPECTED_WINNER_RATES.items():
            with self.subTest(winner=winner):
                self.assertAlmostEqual(
                    report["winner_rates_percent"][winner],
                    expected,
                    delta=1e-12,
                )

        self.assertEqual(
            report["static_convex"]["hurdle_weight_grid"],
            [index / 20 for index in range(21)],
        )
        for dataset, expected in EXPECTED_STATIC_CONVEX.items():
            for quantity, value in expected.items():
                with self.subTest(dataset=dataset, static_quantity=quantity):
                    self.assertAlmostEqual(
                        report["static_convex"]["datasets"][dataset][quantity],
                        value,
                        delta=5e-6,
                    )
        self.assertIs(report["static_convex"]["convex_range_valid"], True)
        self.assertIs(
            report["static_convex"]["reported_quantity_ordering_valid"],
            True,
        )

        np.testing.assert_array_equal(
            report["exponential_weighting"]["extreme_hurdle_weights"],
            [0.0, 1.0, 0.5],
        )
        self.assertEqual(
            report["exponential_weighting"]["frozen_discounted_weight_count"],
            22_196,
        )
        self.assertIs(
            report["exponential_weighting"]["frozen_weights_finite"], True
        )
        self.assertIs(
            report["exponential_weighting"]["frozen_weight_sign_consistent"],
            True,
        )
        self.assertIs(
            report["exponential_weighting"]["extreme_underflow_guarded"],
            True,
        )
        self.assertGreaterEqual(
            report["exponential_weighting"]["frozen_hurdle_weight_min"], 0.0
        )
        self.assertLessEqual(
            report["exponential_weighting"]["frozen_hurdle_weight_max"], 1.0
        )

        bootstrap = report["series_cluster_bootstrap"]
        self.assertEqual(bootstrap["seed"], 20260904)
        self.assertEqual(bootstrap["draws"], 2000)
        self.assertIs(bootstrap["deterministic"], True)
        self.assertIs(bootstrap["finite"], True)
        for dataset, point_estimate in EXPECTED_BOOTSTRAP_POINT_ESTIMATES.items():
            self.assertAlmostEqual(
                bootstrap["datasets"][dataset]["point_estimate"],
                point_estimate,
                delta=5e-6,
            )
            for quantity in ("mean", "ci_low", "ci_high"):
                with self.subTest(dataset=dataset, bootstrap_quantity=quantity):
                    self.assertTrue(
                        np.isfinite(bootstrap["datasets"][dataset][quantity])
                    )
        self.assertEqual(
            report["constant_spearman"],
            {
                "status": "DEGENERATE",
                "rho": None,
                "pvalue": None,
                "n": 33_294,
            },
        )

    def test_raw_paths_require_exact_dataset_keys(self):
        cases = (
            {"m5": RAW_PATHS["m5"]},
            {**RAW_PATHS, "other": RAW_PATHS["m5"]},
        )
        for raw_paths in cases:
            with self.subTest(keys=sorted(raw_paths)):
                with _no_writes(), self.assertRaisesRegex(
                    (KeyError, ValueError), "(?i)(raw.paths|dataset|m5|favorita)"
                ):
                    self._reproduce(raw_paths)

    def test_raw_files_require_the_audited_sha256(self):
        swapped = {"m5": RAW_PATHS["favorita"], "favorita": RAW_PATHS["m5"]}
        with _no_writes(), self.assertRaisesRegex(ValueError, "(?i)(sha|hash)"):
            self._reproduce(swapped)

    def test_rejects_non_exact_raw_schema(self):
        cases = {
            "missing": self.real_raw["m5"].drop(columns=["hurdle_mu_prediction"]),
            "extra": self.real_raw["m5"].assign(unexpected_column=0.0),
        }
        for name, corrupted in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(
                ValueError, "(?i)schema"
            ):
                self._reproduce_with_m5(corrupted)

    def test_rejects_wrong_raw_row_count(self):
        corrupted = self.real_raw["m5"].iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "(?i)row"):
            self._reproduce_with_m5(corrupted)

    def test_rejects_incomplete_or_duplicate_step_pairing(self):
        corrupted = self.real_raw["m5"].copy(deep=True)
        corrupted.loc[corrupted.index[1], "step"] = corrupted.loc[
            corrupted.index[0], "step"
        ]
        with self.assertRaisesRegex(ValueError, "(?i)(pair|key|duplicate|step)"):
            self._reproduce_with_m5(corrupted)

    def test_rejects_invalid_target_mask(self):
        corrupted = self.real_raw["m5"].copy(deep=True)
        corrupted.loc[corrupted.index[0], "target_mask"] = False
        with self.assertRaisesRegex(ValueError, "(?i)mask"):
            self._reproduce_with_m5(corrupted)

    def test_rejects_non_finite_numeric_values(self):
        corrupted = self.real_raw["m5"].copy(deep=True)
        corrupted.loc[corrupted.index[0], "point_mean_prediction"] = np.nan
        with self.assertRaisesRegex(ValueError, "(?i)(finite|nan)"):
            self._reproduce_with_m5(corrupted)

    def test_rejects_occurrence_values_inconsistent_with_observations(self):
        corrupted = self.real_raw["m5"].copy(deep=True)
        row = corrupted.index[0]
        corrupted.loc[row, "occurrence"] = 1.0 - corrupted.loc[row, "occurrence"]
        with self.assertRaisesRegex(ValueError, "(?i)occurrence"):
            self._reproduce_with_m5(corrupted)

    def test_rejects_broken_hurdle_factorization(self):
        corrupted = self.real_raw["m5"].copy(deep=True)
        row = corrupted.index[0]
        corrupted.loc[row, "hurdle_mean_prediction"] += np.float32(1.0)
        with self.assertRaisesRegex(ValueError, "(?i)(hurdle|factor)"):
            self._reproduce_with_m5(corrupted)


if __name__ == "__main__":
    unittest.main()
