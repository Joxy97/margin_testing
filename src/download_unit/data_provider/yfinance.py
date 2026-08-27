"""yfinance-backed data provider."""

from typing import Any

import yfinance

from ..command import UnifiedFormatCommand, YfinanceCommand
from .data_provider import DataProvider


class YfinanceDataProvider(DataProvider):
    """Download market data through the yfinance package."""

    def __init__(self, format: str) -> None:
        super().__init__(format)

    def convertCommand(self, command: UnifiedFormatCommand) -> YfinanceCommand:
        """Convert a unified command to yfinance download parameters."""
        if not isinstance(command, UnifiedFormatCommand):
            raise TypeError("command must be a UnifiedFormatCommand instance")
        return YfinanceCommand(
            parameters=(
                command.instruments,
                command.start_date,
                command.end_date,
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
