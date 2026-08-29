"""Tests for composable unified-command chunkers."""

import unittest
from datetime import date

from download_unit import (
    DateChunker,
    InstrumentChunker,
    Period,
    ProductChunker,
    DataRequest,
)


def data_request() -> DataRequest:
    return DataRequest(
        instruments=["AAPL", "MSFT", "NVDA"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5),
        period=Period.ONE_DAY,
        data_type="closePrices",
        provider_parameters={"location": "prices.csv"},
    )


class ChunkerTest(unittest.TestCase):
    def test_instrument_chunker_preserves_other_command_values(self) -> None:
        command = data_request()

        chunks = list(InstrumentChunker(2).createChunks(command))

        self.assertEqual(
            [chunk.instruments for chunk in chunks],
            [("AAPL", "MSFT"), ("NVDA",)],
        )
        self.assertTrue(
            all(
                chunk.provider_parameters["location"] == "prices.csv"
                for chunk in chunks
            )
        )
        self.assertTrue(all(chunk is not command for chunk in chunks))

    def test_date_chunker_creates_contiguous_ranges(self) -> None:
        chunks = list(DateChunker(2).createChunks(data_request()))

        self.assertEqual(
            [
                (chunk.start_date, chunk.end_date)
                for chunk in chunks
            ],
            [
                (date(2024, 1, 1), date(2024, 1, 3)),
                (date(2024, 1, 3), date(2024, 1, 5)),
            ],
        )

    def test_product_chunker_creates_cartesian_chunks(self) -> None:
        chunker = ProductChunker(InstrumentChunker(2), DateChunker(2))

        chunks = list(chunker.createChunks(data_request()))

        self.assertEqual(len(chunks), 4)
        self.assertEqual(
            [chunk.instruments for chunk in chunks],
            [
                ("AAPL", "MSFT"),
                ("AAPL", "MSFT"),
                ("NVDA",),
                ("NVDA",),
            ],
        )

    def test_batch_sizes_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            InstrumentChunker(0)
        with self.assertRaises(ValueError):
            DateChunker(0)


if __name__ == "__main__":
    unittest.main()
