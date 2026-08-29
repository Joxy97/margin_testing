"""Data provider for CSV files addressable by pandas."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from ..command import DataRequest
from .data_provider import DataProvider

if TYPE_CHECKING:
    import pandas


class LocalCSVDataProvider(DataProvider):
    """Load instrument data from a CSV file or URL."""

    def getDataTypes(self) -> set[str]:
        """Return the data types available in a local price file."""
        return {"closePrices"}

    def convertRawData(self, raw_data: Any) -> Any:
        """Return CSV data without modification."""
        return raw_data

    def convertCommand(
        self,
        command: DataRequest,
    ) -> DataRequest:
        """Return the unified command without modification."""
        if not isinstance(command, DataRequest):
            raise TypeError("command must be a DataRequest")
        return command

    def downloadData(
        self,
        command: DataRequest,
    ) -> pandas.DataFrame:
        """Read and filter the CSV dataset described by ``command``."""
        if not isinstance(command, DataRequest):
            raise TypeError("command must be a DataRequest")

        import pandas

        data = pandas.read_csv(command.provider_parameters.get("location", ""))
        data = self._extractInstruments(data, list(command.instruments))
        return self._extractDates(
            data,
            command.start_date,
            command.end_date,
        )

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
