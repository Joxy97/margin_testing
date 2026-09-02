"""Long-form CSV provider for futures and option-chain quotes."""

from __future__ import annotations

from typing import Any

from ..command import DataRequest
from .data_provider import DataProvider


class DerivativeCSVDataProvider(DataProvider):
    """Read normalized derivative quotes from one or more CSV files."""

    requiredColumns = {
        "date", "symbol", "instrument_type", "expiration_date", "price"
    }

    def getDataTypes(self) -> set[str]:
        return {"derivativeQuotes"}

    def convertRawData(self, raw_data: Any) -> Any:
        return raw_data

    def convertCommand(self, command: DataRequest) -> DataRequest:
        return command

    def downloadData(self, command: DataRequest):
        import pandas

        locations = command.provider_parameters.get("locations")
        if locations is None:
            locations = (command.provider_parameters.get("location", ""),)
        frames = [pandas.read_csv(location) for location in locations]
        data = pandas.concat(frames, ignore_index=True)
        data.columns = [str(column).strip().lower() for column in data.columns]
        missing = self.requiredColumns.difference(data.columns)
        if missing:
            raise ValueError(f"Derivative CSV is missing columns: {sorted(missing)}")
        data["date"] = pandas.to_datetime(data["date"], errors="raise")
        data["expiration_date"] = pandas.to_datetime(
            data["expiration_date"], errors="raise"
        )
        start = pandas.Timestamp(command.start_date)
        end = pandas.Timestamp(command.end_date)
        return data.loc[
            data["symbol"].astype(str).isin(command.instruments)
            & data["date"].between(start, end)
        ].copy()
