"""Tests for creating download units by algorithm name."""

import unittest

from download_unit import (
    DownloadUnitFactory,
    ExponentialBackoffDownloadUnit,
    SingleRequestDownloadUnit,
)


class DownloadUnitFactoryTest(unittest.TestCase):
    def test_creates_exponential_backoff_download_unit(self) -> None:
        download_unit = DownloadUnitFactory.createDownloadUnit(
            "exponential_backoff",
            {
                "instrument_batch": 10,
                "dates_batch": 30,
                "time": 0.5,
            },
        )

        self.assertIsInstance(download_unit, ExponentialBackoffDownloadUnit)
        self.assertEqual(download_unit.instrument_batch, 10)
        self.assertEqual(download_unit.dates_batch, 30)
        self.assertEqual(download_unit.time, 0.5)

    def test_rejects_unknown_algorithm(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown download algorithm"):
            DownloadUnitFactory.createDownloadUnit("unknown", {})

    def test_creates_single_request_download_unit(self) -> None:
        download_unit = DownloadUnitFactory.createDownloadUnit(
            "single_request", {}
        )

        self.assertIsInstance(download_unit, SingleRequestDownloadUnit)


if __name__ == "__main__":
    unittest.main()
