"""Tests for cache-backed market-data storage."""

import tempfile
import unittest
from datetime import date

import pandas

from data_manager import DataManager, PartitionedPickleDataStore
from download_unit import DataRequest


def command(
    instruments: list[str],
    start: date = date(2024, 1, 1),
    end: date = date(2024, 1, 3),
) -> DataRequest:
    return DataRequest(
        instruments=instruments,
        start_date=start,
        end_date=end,
        data_type="closePrices",
    )


class DataManagerTest(unittest.TestCase):
    def test_restores_an_evicted_partition_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = DataManager(
                maxMemoryBytes=1,
                backingStore=PartitionedPickleDataStore(directory),
            )
            request = command(["AAPL"])
            manager.storeData(
                request,
                pandas.DataFrame(
                    {
                        "date": pandas.date_range("2024-01-01", periods=3),
                        "AAPL": [10, 11, 12],
                    }
                ),
            )

            result = manager.getData(request)

        self.assertIsNotNone(result)
        self.assertEqual(result["AAPL"].tolist(), [10, 11, 12])

    def test_reports_only_uncovered_request_fragments(self) -> None:
        manager = DataManager()
        manager.storeData(
            command(["AAPL"], date(2024, 1, 1), date(2024, 1, 3)),
            pandas.DataFrame(
                {
                    "date": pandas.date_range("2024-01-01", periods=3),
                    "AAPL": [10, 11, 12],
                }
            ),
        )

        missing = manager.getMissingRequests(
            command(
                ["AAPL", "MSFT"],
                date(2024, 1, 1),
                date(2024, 1, 5),
            )
        )

        self.assertEqual(
            {
                (request.instruments, request.start_date, request.end_date)
                for request in missing
            },
            {
                (("AAPL",), date(2024, 1, 4), date(2024, 1, 5)),
                (("MSFT",), date(2024, 1, 1), date(2024, 1, 5)),
            },
        )

    def test_byte_budget_evicts_oversized_frames(self) -> None:
        manager = DataManager(maxMemoryBytes=1)
        request = command(["AAPL"])

        stored = manager.storeData(
            request,
            pandas.DataFrame(
                {
                    "date": pandas.date_range("2024-01-01", periods=3),
                    "AAPL": [10, 11, 12],
                }
            ),
        )

        self.assertEqual(stored["AAPL"].tolist(), [10, 11, 12])
        self.assertIsNone(manager.getData(request))

    def test_returns_only_covered_dates_and_instruments(self) -> None:
        manager = DataManager(memorySize=2)
        stored_command = command(["AAPL", "MSFT"])
        manager.storeData(
            stored_command,
            pandas.DataFrame(
                {
                    "date": pandas.date_range("2024-01-01", periods=3),
                    "AAPL": [10, 11, 12],
                    "MSFT": [20, 21, 22],
                }
            ),
        )

        result = manager.getData(
            command(["MSFT"], date(2024, 1, 2), date(2024, 1, 3))
        )

        self.assertIsNotNone(result)
        self.assertEqual(list(result.columns), ["date", "MSFT"])
        self.assertEqual(result["MSFT"].tolist(), [21, 22])

    def test_returns_none_when_instrument_or_date_coverage_is_missing(self) -> None:
        manager = DataManager()
        manager.storeData(
            command(["AAPL"]),
            pandas.DataFrame(
                {
                    "date": pandas.date_range("2024-01-01", periods=3),
                    "AAPL": [10, 11, 12],
                }
            ),
        )

        self.assertIsNone(manager.getData(command(["MSFT"])))
        self.assertIsNone(
            manager.getData(
                command(["AAPL"], date(2023, 12, 31), date(2024, 1, 2))
            )
        )

    def test_merges_adjacent_downloads_for_reuse(self) -> None:
        manager = DataManager()
        manager.storeData(
            command(["AAPL"], date(2024, 1, 1), date(2024, 1, 2)),
            pandas.DataFrame(
                {"date": pandas.date_range("2024-01-01", periods=2), "AAPL": [10, 11]}
            ),
        )
        manager.storeData(
            command(["AAPL"], date(2024, 1, 3), date(2024, 1, 4)),
            pandas.DataFrame(
                {"date": pandas.date_range("2024-01-03", periods=2), "AAPL": [12, 13]}
            ),
        )

        result = manager.getData(
            command(["AAPL"], date(2024, 1, 1), date(2024, 1, 4))
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["AAPL"].tolist(), [10, 11, 12, 13])


if __name__ == "__main__":
    unittest.main()
