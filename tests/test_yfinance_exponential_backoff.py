"""Live integration test for chunked yfinance downloads."""

import os
import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas

from download_unit import ExponentialBackoffDownloadUnit, Period, UnifiedFormatCommand


@unittest.skipUnless(
    os.environ.get("RUN_YFINANCE_INTEGRATION") == "1",
    "set RUN_YFINANCE_INTEGRATION=1 to run the live yfinance test",
)
class YfinanceExponentialBackoffIntegrationTest(unittest.TestCase):
    def test_downloads_ten_random_tickers_over_three_date_batches(self) -> None:
        from download_unit.data_provider.yfinance import YfinanceDataProvider

        ticker_pool = [
            "AAPL",
            "AMZN",
            "AVGO",
            "BRK-B",
            "GOOG",
            "JPM",
            "LLY",
            "META",
            "MSFT",
            "NFLX",
            "NVDA",
            "ORCL",
            "TSLA",
            "UNH",
            "WMT",
        ]
        instruments = random.Random(42).sample(ticker_pool, 10)
        end_date = date.today()
        command = UnifiedFormatCommand(
            instruments=instruments,
            start_date=end_date - timedelta(days=300),
            end_date=end_date,
            period=Period.ONE_DAY,
        )
        unit = ExponentialBackoffDownloadUnit(
            instrument_batch=5,
            dates_batch=100,
            time=1,
        )
        provider = YfinanceDataProvider()

        data = unit.getData(provider, command)

        self.assertEqual(len(data), 6)
        self.assertTrue(any(not frame.empty for frame in data))
        dataframe = pandas.concat(data)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "yfinance_data.csv"
            unit.storeData(dataframe, csv_path)
            self.assertTrue(csv_path.is_file())
            self.assertGreater(csv_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
