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

        locations = command.provider_parameters.get("locations")
        if locations is None:
            locations = (command.provider_parameters.get("location", ""),)
        elif isinstance(locations, (str, bytes)):
            raise TypeError("locations must be an iterable of CSV paths")
        frames = [
            self._normalizeDateColumn(pandas.read_csv(location))
            for location in locations
        ]
        if not frames:
            raise ValueError("at least one CSV location must be supplied")
        data = pandas.concat(frames, ignore_index=True)
        data["date"] = pandas.to_datetime(data["date"], errors="raise")
        data = data.drop_duplicates(subset="date", keep="last").sort_values("date")
        data = self._extractInstruments(data, list(command.instruments))
        return self._extractDates(
            data,
            command.start_date,
            command.end_date,
        )

    @staticmethod
    def _normalizeDateColumn(data: pandas.DataFrame) -> pandas.DataFrame:
        """Normalize common CSV date-column capitalization."""
        date_columns = [
            column for column in data.columns if str(column).casefold() == "date"
        ]
        if len(date_columns) != 1:
            raise ValueError("CSV data must contain exactly one date column")
        return data.rename(columns={date_columns[0]: "date"})

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
