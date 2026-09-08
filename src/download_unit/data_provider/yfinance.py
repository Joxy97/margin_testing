"""yfinance-backed data provider."""

from datetime import timedelta
from typing import Any

import yfinance

from ..command import DataRequest, YfinanceCommand
from .data_provider import DataProvider


class YfinanceDataProvider(DataProvider):
    """Download market data through the yfinance package."""

    def isTransientError(self, error: Exception) -> bool:
        from yfinance.exceptions import YFRateLimitError
        from curl_cffi.requests.exceptions import ConnectionError, Timeout
        return super().isTransientError(error) or isinstance(error, (YFRateLimitError, ConnectionError, Timeout))

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
        import pandas
        from yfinance.exceptions import YFPricesMissingError

        # Bulk download suppresses ticker failures. The per-ticker boundary
        # propagates them without changing yfinance's process-global settings.
        frames = {}
        for instrument in instruments:
            ticker = yfinance.Ticker(instrument)
            try:
                frame = ticker.history(start=start_date, end=end_date,
                                       raise_errors=True, auto_adjust=True)
            except YFPricesMissingError:
                # The same error describes closed-market days and bad responses.
                # Confirm absence from a successfully parsed superset, never
                # by suppressing errors or assuming an exchange calendar.
                earlier = start_date - timedelta(days=min(7, start_date.toordinal() - 1))
                frame = ticker.history(start=earlier, end=end_date,
                                       raise_errors=True, auto_adjust=True)
                if frame.empty or "Close" not in frame or frame["Close"].dropna().empty:
                    raise
            frame = frame.copy()
            if frame.index.tz is not None:
                frame.index = frame.index.tz_localize(None)
            frames[instrument] = frame.loc[(frame.index >= pandas.Timestamp(start_date))
                                             & (frame.index < pandas.Timestamp(end_date))]
        result = pandas.concat(frames, axis=1).swaplevel(0, 1, axis=1)
        return result
