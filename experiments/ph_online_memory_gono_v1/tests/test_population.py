"""Independent natural-population construction contracts."""

from __future__ import annotations

import hashlib
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from experiments.ph_online_memory_gono_v1.data import (
    dataset_config,
    load_independent_population,
)


def _raw_dataset(name: str, length: int, rows: list[dict]) -> dict:
    y = np.zeros((len(rows), length), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for time_index, value in row["events"]:
            y[row_index, time_index] = value
    return {
        "name": name,
        "series_id": np.asarray([row["series_id"] for row in rows]),
        "y": y,
        "z": (y > 0).astype(np.float32),
        "available_from": np.asarray(
            [row.get("available_from", 0) for row in rows], dtype=np.int64
        ),
        "first_positive": np.asarray(
            [
                min((time for time, _ in row["events"]), default=length)
                for row in rows
            ],
            dtype=np.int64,
        ),
    }


class DatasetConfigTest(unittest.TestCase):
    def test_configs_use_the_new_frozen_cutoffs(self):
        """Catches reuse of the older external-screen train/validation bounds."""
        m5 = dataset_config("m5", n_series=7)
        favorita = dataset_config("favorita", n_series=11)

        self.assertEqual(
            (m5.n_series, m5.length, m5.train_end, m5.val_end),
            (7, 1941, 1717, 1745),
        )
        self.assertEqual(
            (favorita.n_series, favorita.length, favorita.train_end, favorita.val_end),
            (11, 1688, 1464, 1492),
        )
        self.assertEqual((m5.lookback, m5.horizon, m5.period), (96, 28, 28))
        self.assertEqual(
            (favorita.lookback, favorita.horizon, favorita.period), (96, 28, 28)
        )

    def test_unknown_dataset_is_rejected(self):
        """Catches silently applying one dataset's split to an unknown source."""
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            dataset_config("retail-x", n_series=2)


class IndependentPopulationTest(unittest.TestCase):
    def test_m5_recomputes_eligibility_excludes_stage_a_and_sorts_alignment(self):
        """Catches stale membership, raw-zero descriptors, or row/ID misalignment."""
        raw = _raw_dataset(
            "m5",
            1941,
            [
                {
                    "series_id": "b",
                    "available_from": 0,
                    "events": [(0, 1000.0), (1, 1000.0)],
                },
                {
                    "series_id": "stage",
                    "available_from": 2,
                    "events": [(2, 600.0), (3, 600.0)],
                },
                {
                    "series_id": "a",
                    "available_from": 5,
                    "events": [(5, 2000.0), (6, 2000.0)],
                },
                {
                    "series_id": "low",
                    "available_from": 0,
                    "events": [(10, 10.0)],
                },
            ],
        )
        stage_a = pd.DataFrame({"series_id": ["stage", "stage", "not-in-full"]})

        with (
            mock.patch(
                "experiments.external_validity_screen.confirmatory_h2.m5_full",
                return_value=raw,
            ),
            mock.patch("pandas.read_parquet", return_value=stage_a),
        ):
            population = load_independent_population("m5", min_positive=2)

        data = population["data"]
        descriptors = population["descriptors"]
        manifest = population["manifest"]
        self.assertEqual(data["series_id"].tolist(), ["a", "b"])
        self.assertEqual(descriptors["series_id"].tolist(), ["a", "b"])
        np.testing.assert_array_equal(data["y"][:, :7], raw["y"][[2, 0], :7])
        np.testing.assert_array_equal(data["available_from"], np.array([5, 0]))
        self.assertEqual(descriptors["segment_start"].tolist(), [5, 0])
        self.assertEqual(
            descriptors["descriptor_variant"].tolist(),
            ["availability_aware", "availability_aware"],
        )
        self.assertEqual(descriptors["n_positive_train"].tolist(), [2, 2])
        self.assertAlmostEqual(
            descriptors.loc[0, "zero_ratio_train"], (1712 - 2) / 1712
        )
        self.assertAlmostEqual(descriptors.loc[0, "train_scale"], 4000.0 / 1717)
        self.assertEqual(
            {
                key: manifest[key]
                for key in (
                    "full_total",
                    "eligible_before_stage_a",
                    "stage_a_excluded",
                    "eligible_independent",
                    "ineligible",
                )
            },
            {
                "full_total": 4,
                "eligible_before_stage_a": 3,
                "stage_a_excluded": 1,
                "eligible_independent": 2,
                "ineligible": 1,
            },
        )
        self.assertEqual(
            manifest["exclusion_reasons"],
            {"n_positive_train_below_2": 1, "stage_a_membership": 1},
        )
        self.assertEqual(
            manifest["independent_id_sha256"],
            "7e18f737311b2dc3b2f269dd78396b0351f14fb66efa879f768cb23181883c78",
        )
        self.assertEqual(
            manifest["distributions"]["n_positive_train"],
            {
                "count": 2,
                "min": 2.0,
                "p25": 2.0,
                "median": 2.0,
                "p75": 2.0,
                "max": 2.0,
            },
        )
        self.assertEqual(
            manifest["eligibility_sensitivity_no_training"],
            {
                "15": {
                    "min_positive_train": 15,
                    "eligible_before_stage_a": 0,
                    "stage_a_excluded": 0,
                    "eligible_independent": 0,
                    "independent_id_sha256": hashlib.sha256(b"").hexdigest(),
                },
                "20": {
                    "min_positive_train": 20,
                    "eligible_before_stage_a": 0,
                    "stage_a_excluded": 0,
                    "eligible_independent": 0,
                    "independent_id_sha256": hashlib.sha256(b"").hexdigest(),
                },
                "30": {
                    "min_positive_train": 30,
                    "eligible_before_stage_a": 0,
                    "stage_a_excluded": 0,
                    "eligible_independent": 0,
                    "independent_id_sha256": hashlib.sha256(b"").hexdigest(),
                },
            },
        )
        self.assertEqual(population["cfg"].n_series, 2)

    def test_favorita_uses_raw_train_start(self):
        """Catches accidental availability trimming of Favorita descriptors."""
        raw = _raw_dataset(
            "favorita",
            1688,
            [
                {"series_id": "z", "events": [(0, 2.0), (4, 3.0)]},
                {"series_id": "a", "events": [(1, 4.0), (5, 5.0)]},
            ],
        )

        with (
            mock.patch(
                "experiments.external_validity_screen.favorita_independent.load_pool",
                return_value=raw,
            ),
            mock.patch(
                "pandas.read_parquet",
                return_value=pd.DataFrame({"series_id": ["absent-stage-id"]}),
            ),
        ):
            population = load_independent_population("favorita", min_positive=2)

        self.assertEqual(population["data"]["series_id"].tolist(), ["a", "z"])
        self.assertEqual(population["descriptors"]["segment_start"].tolist(), [0, 0])
        self.assertEqual(
            population["descriptors"]["descriptor_variant"].tolist(),
            ["raw", "raw"],
        )
        self.assertEqual(population["manifest"]["stage_a_excluded"], 0)

    def test_malformed_source_arrays_fail_before_membership_is_built(self):
        """Catches duplicate IDs and wrong native length."""
        valid = _raw_dataset(
            "m5",
            1941,
            [
                {"series_id": "x", "events": [(0, 1.0)]},
                {"series_id": "y", "events": [(0, 1.0)]},
            ],
        )
        bad_cases = []
        duplicate = dict(valid)
        duplicate["series_id"] = np.array(["x", "x"])
        bad_cases.append((duplicate, "unique"))
        wrong_length = dict(valid)
        wrong_length["y"] = wrong_length["y"][:, :-1]
        wrong_length["z"] = wrong_length["z"][:, :-1]
        bad_cases.append((wrong_length, "length"))
        for bad, message in bad_cases:
            with self.subTest(message=message):
                with (
                    mock.patch(
                        "experiments.external_validity_screen.confirmatory_h2.m5_full",
                        return_value=bad,
                    ),
                    mock.patch(
                        "pandas.read_parquet",
                        return_value=pd.DataFrame({"series_id": []}),
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        load_independent_population("m5", min_positive=1)

    def test_late_ineligible_m5_rows_do_not_block_the_eligible_population(self):
        """Catches validating availability before new-cutoff eligibility."""
        raw = _raw_dataset(
            "m5",
            1941,
            [
                {"series_id": "keep", "events": [(10, 1.0)]},
                {
                    "series_id": "late",
                    "available_from": 1800,
                    "events": [(1800, 1.0)],
                },
            ],
        )
        with (
            mock.patch(
                "experiments.external_validity_screen.confirmatory_h2.m5_full",
                return_value=raw,
            ),
            mock.patch(
                "pandas.read_parquet",
                return_value=pd.DataFrame({"series_id": []}),
            ),
        ):
            population = load_independent_population("m5", min_positive=1)

        self.assertEqual(population["data"]["series_id"].tolist(), ["keep"])
        self.assertEqual(population["manifest"]["ineligible"], 1)

    def test_zero_threshold_still_rejects_late_selected_series(self):
        """Catches allowing a scored series to become available during evaluation."""
        raw = _raw_dataset(
            "m5",
            1941,
            [
                {
                    "series_id": "late",
                    "available_from": 1745,
                    "events": [(1745, 1.0)],
                }
            ],
        )
        with (
            mock.patch(
                "experiments.external_validity_screen.confirmatory_h2.m5_full",
                return_value=raw,
            ),
            mock.patch(
                "pandas.read_parquet",
                return_value=pd.DataFrame({"series_id": []}),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "available_from"):
                load_independent_population("m5", min_positive=0)


if __name__ == "__main__":
    unittest.main()
