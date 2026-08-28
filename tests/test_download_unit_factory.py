"""Tests for creating download units by algorithm name."""

import unittest

from download_unit import (
    DateChunker,
    DownloadUnitFactory,
    ExponentialBackoffDownloadUnit,
    InstrumentChunker,
    ProductChunker,
    SingleRequestDownloadUnit,
)


class DownloadUnitFactoryTest(unittest.TestCase):
    def test_creates_exponential_backoff_download_unit(self) -> None:
        chunker = ProductChunker(
            InstrumentChunker(10),
            DateChunker(30),
        )
        download_unit = DownloadUnitFactory.createDownloadUnit(
            "exponential_backoff",
            {
                "chunker": chunker,
                "time": 0.5,
            },
        )

        self.assertIsInstance(download_unit, ExponentialBackoffDownloadUnit)
        self.assertIs(download_unit.chunker, chunker)
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
