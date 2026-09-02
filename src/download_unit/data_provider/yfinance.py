"""yfinance-backed data provider."""

from datetime import timedelta
from typing import Any

import yfinance

from ..command import DataRequest, YfinanceCommand
from .data_provider import DataProvider


class YfinanceDataProvider(DataProvider):
    """Download market data through the yfinance package."""

    def getDataTypes(self) -> set[str]:
        """Return the market data types supplied by yfinance."""
        return {"closePrices"}

    def convertRawData(self, raw_data: Any) -> Any:
        """Extract close prices into the application's tabular format."""
        import pandas

        if isinstance(raw_data, list):
            raw_data = pandas.concat(raw_data).sort_index()
        close_prices = raw_data["Close"]
        if isinstance(close_prices, pandas.Series):
            close_prices = close_prices.to_frame()
        close_prices = close_prices.copy()
        close_prices.index.name = "date"
        return close_prices.reset_index()

    def convertCommand(self, command: DataRequest) -> YfinanceCommand:
        """Convert a unified command to yfinance download parameters."""
        if not isinstance(command, DataRequest):
            raise TypeError("command must be a DataRequest")
        return YfinanceCommand(
            parameters=(
                command.instruments,
                command.start_date,
                command.end_date + timedelta(days=1),
                command.period.value,
            )
        )

    def downloadData(self, command: YfinanceCommand) -> Any:
        """Download data using the command's positional parameters."""
        if not isinstance(command, YfinanceCommand):
            raise TypeError("command must be a YfinanceCommand instance")
        instruments, start_date, end_date, _period = command.parameters
        return yfinance.download(
            tickers=instruments,
            start=start_date,
            end=end_date,
        )
