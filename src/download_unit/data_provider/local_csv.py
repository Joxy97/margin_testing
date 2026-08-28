"""Data provider for CSV files addressable by pandas."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from ..command import Command, UnifiedFormatCommand
from .data_provider import DataProvider

if TYPE_CHECKING:
    import pandas


class LocalCSVDataProvider(DataProvider):
    """Load instrument data from a CSV file or URL."""

    def convertRawData(self, raw_data: Any) -> Any:
        """Return CSV data without modification."""
        return raw_data

    def convertCommand(
        self,
        command: Command,
    ) -> UnifiedFormatCommand:
        """Return the unified command without modification."""
        if not isinstance(command, UnifiedFormatCommand):
            raise TypeError("command must be a UnifiedFormatCommand instance")
        return command

    def downloadData(
        self,
        command: UnifiedFormatCommand,
    ) -> pandas.DataFrame:
        """Read and filter the CSV dataset described by ``command``."""
        if not isinstance(command, UnifiedFormatCommand):
            raise TypeError("command must be a UnifiedFormatCommand instance")

        import pandas

        data = pandas.read_csv(command.location)
        data = self._extractInstruments(data, command.instruments)
        return self._extractDates(data, command.start_date, command.end_date)

    def _extractInstruments(
        self,
        data: pandas.DataFrame,
        instruments: list[str],
    ) -> pandas.DataFrame:
        """Keep the date column and the requested instrument columns."""
        return data.loc[:, ["date", *instruments]].copy()

    def _extractDates(
        self,
        data: pandas.DataFrame,
        start_date: date,
        end_date: date,
    ) -> pandas.DataFrame:
        """Keep rows whose dates lie within the inclusive requested range."""
        import pandas

        data = data.copy()
        data["date"] = pandas.to_datetime(data["date"])
        dates = data["date"]
        in_range = (dates >= pandas.Timestamp(start_date)) & (
            dates <= pandas.Timestamp(end_date)
        )
        return data.loc[in_range]
