"""Tests for downloading a filtered local CSV dataset."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

try:
    import pandas
except ModuleNotFoundError:
    pandas = None  # type: ignore[assignment]

from download_unit import (
    LocalCSVDataProvider,
    Period,
    SingleRequestDownloadUnit,
    DataRequest,
)


@unittest.skipIf(pandas is None, "pandas is not installed")
class LocalCSVDownloadTest(unittest.TestCase):
    def test_reads_requested_instruments_and_inclusive_date_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "prices.csv"
            csv_path.write_text(
                "date,AAPL,MSFT,NVDA\n"
                "2024-01-01,10,20,30\n"
                "2024-01-02,11,21,31\n"
                "2024-01-03,12,22,32\n"
                "2024-01-04,13,23,33\n",
                encoding="utf-8",
            )
            command = DataRequest(
                instruments=["NVDA", "AAPL"],
                start_date=date(2024, 1, 2),
                end_date=date(2024, 1, 3),
                period=Period.ONE_DAY,
                data_type="closePrices",
                provider_parameters={"location": str(csv_path)},
            )

            result = SingleRequestDownloadUnit().getData(
                LocalCSVDataProvider(),
                command,
            )

        self.assertEqual(list(result.columns), ["date", "NVDA", "AAPL"])
        self.assertEqual(result["NVDA"].tolist(), [31, 32])
        self.assertEqual(result["AAPL"].tolist(), [11, 12])
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-02", "2024-01-03"],
        )

    def test_provider_conversion_methods_return_their_input(self) -> None:
        provider = LocalCSVDataProvider()
        command = DataRequest(
            instruments=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            period=Period.ONE_DAY,
            data_type="closePrices",
            provider_parameters={"location": "prices.csv"},
        )
        raw_data = object()

        self.assertIs(provider.convertCommand(command), command)
        self.assertIs(provider.convertRawData(raw_data), raw_data)


if __name__ == "__main__":
    unittest.main()
