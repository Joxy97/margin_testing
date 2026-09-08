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
    DateChunker,
    InstrumentChunker,
    ProductChunker,
    ExponentialBackoffDownloadUnit,
    DerivativeCSVDataProvider,
)


@unittest.skipIf(pandas is None, "pandas is not installed")
class LocalCSVDownloadTest(unittest.TestCase):
    def test_single_download_joins_columns_and_rejects_conflicting_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.csv", Path(directory) / "b.csv"
            first.write_text("date,A\n2024-01-01,10\n")
            second.write_text("date,B\n2024-01-01,20\n")
            request = DataRequest(instruments=["B", "A"], start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1), data_type="closePrices",
                provider_parameters={"locations": [str(first), str(second)]})
            result = SingleRequestDownloadUnit().getData(LocalCSVDataProvider(), request)
            self.assertEqual(result[["B", "A"]].values.tolist(), [[20, 10]])
            second.write_text("date,A,B\n2024-01-01,11,20\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                SingleRequestDownloadUnit().getData(LocalCSVDataProvider(), request)


    def test_chunked_derivative_quotes_keep_distinct_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.csv"
            path.write_text("date,symbol,instrument_type,expiration_date,price\n"
                            "2024-01-01,ES,future,2024-06-01,100\n"
                            "2024-01-01,ES,future,2024-09-01,110\n"
                            "2024-01-02,ES,future,2024-06-01,101\n")
            request = DataRequest(instruments=["ES"], start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 2), data_type="derivativeQuotes",
                provider_parameters={"location": str(path)})
            result = ExponentialBackoffDownloadUnit(DateChunker(1), time=0).getData(
                DerivativeCSVDataProvider(), request)
        self.assertEqual(result["price"].tolist(), [100, 110, 101])

    def test_conflicting_duplicate_observations_are_rejected(self) -> None:
        class RepeatedDates:
            def createChunks(self, command):
                yield command.withChanges(provider_parameters={"location": str(first)})
                yield command.withChanges(provider_parameters={"location": str(second)})

        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.csv", Path(directory) / "b.csv"
            first.write_text("date,A\n2024-01-01,10\n")
            second.write_text("date,A\n2024-01-01,11\n")
            request = DataRequest(instruments=["A"], start_date=date(2024, 1, 1),
                                  end_date=date(2024, 1, 1), data_type="closePrices")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                ExponentialBackoffDownloadUnit(RepeatedDates(), time=0).getData(
                    LocalCSVDataProvider(), request)

    def test_product_download_assembles_one_ordered_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text("date,A,B\n2024-01-01,10,20\n2024-01-02,11,21\n2024-01-03,12,22\n")
            request = DataRequest(
                instruments=["B", "A"], start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 3), data_type="closePrices",
                provider_parameters={"location": str(path)},
            )
            result = ExponentialBackoffDownloadUnit(
                ProductChunker(InstrumentChunker(1), DateChunker(2)), time=0,
            ).getData(LocalCSVDataProvider(), request)

        self.assertEqual(result.to_dict("list"), {
            "date": list(pandas.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])),
            "B": [20, 21, 22], "A": [10, 11, 12],
        })

    def test_accepts_capitalized_date_columns_across_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "history.csv"
            second = root / "backtest.csv"
            first.write_text("Date,AAPL\n2024-01-01,10\n", encoding="utf-8")
            second.write_text("Date,AAPL\n2024-01-02,11\n", encoding="utf-8")
            command = DataRequest(
                instruments=["AAPL"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 2),
                period=Period.ONE_DAY,
                data_type="closePrices",
                provider_parameters={"locations": (str(first), str(second))},
            )

            result = LocalCSVDataProvider().downloadData(command)

        self.assertEqual(result["AAPL"].tolist(), [10, 11])

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
