"""Cache long-form futures and option-chain quotes."""

from __future__ import annotations

from .interval_coverage import IntervalCoverage
from download_unit import DataRequest
from dataclasses import dataclass, field


@dataclass
class DerivativeQuotePartition:
    data: object = None
    coverage: IntervalCoverage = field(default_factory=IntervalCoverage)


class DerivativeQuoteDataManager:
    """Serve long-form quote rows through the standard manager methods."""

    identityColumns = (
        "date", "symbol", "instrument_type", "expiration_date",
        "strike", "option_type", "exercise_style",
    )

    def __init__(self) -> None:
        self.partitions = {}

    def _entry(self, command: DataRequest) -> DerivativeQuotePartition:
        key = command.datasetIdentity, command.data_type, command.period
        return self.partitions.setdefault(key, DerivativeQuotePartition())

    def getData(self, command: DataRequest):
        if self._entry(command).data is None or self.getMissingRequests(command):
            return None
        return self._select(command)

    def getMissingRequests(self, command: DataRequest) -> list[DataRequest]:
        return self._entry(command).coverage.missingRequests(command)

    def getAvailableData(self, command: DataRequest):
        """Copy any retained quotes independently of complete interval coverage."""
        return None if self._entry(command).data is None else self._select(command)

    def storeData(self, command: DataRequest, data):
        import pandas

        if not isinstance(data, pandas.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        normalized = data.copy()
        required = {"date", "symbol", "instrument_type", "expiration_date", "price"}
        missing = required.difference(normalized.columns)
        if missing:
            raise ValueError(f"Derivative quotes are missing columns: {sorted(missing)}")
        normalized["date"] = pandas.to_datetime(normalized["date"], errors="raise")
        normalized["expiration_date"] = pandas.to_datetime(
            normalized["expiration_date"], errors="raise"
        )
        for column, default in (
            ("strike", 0.0), ("option_type", ""), ("exercise_style", ""),
            ("multiplier", 1.0), ("dividend_yield", 0.0),
        ):
            if column not in normalized:
                normalized[column] = default
            normalized[column] = normalized[column].fillna(default)
        normalized["instrument_type"] = (
            normalized["instrument_type"].astype(str).str.lower()
        )
        normalized["option_type"] = normalized["option_type"].astype(str).str.upper()
        normalized["exercise_style"] = (
            normalized["exercise_style"].astype(str).str.upper()
        )
        for column in ("price", "strike", "multiplier", "dividend_yield"):
            normalized[column] = pandas.to_numeric(normalized[column], errors="raise")
        self._validate(normalized, command)
        combined = normalized if self._entry(command).data is None else pandas.concat(
            (self._entry(command).data, normalized), ignore_index=True
        )
        if (combined.groupby(list(self.identityColumns), dropna=False)
                .nunique(dropna=False) > 1).any().any():
            raise ValueError("Conflicting observations for the same derivative quote")
        self._entry(command).data = combined.drop_duplicates(
            list(self.identityColumns), keep="last"
        ).sort_values(["date", "symbol", "expiration_date", "strike"])
        self._entry(command).coverage.add(command)
        return self._select(command)

    @staticmethod
    def _validate(data, command: DataRequest) -> None:
        import numpy

        allowed = {"equity", "future", "equity_option", "futures_option"}
        unknown = set(data["instrument_type"]).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown derivative instrument types: {sorted(unknown)}")
        numeric = data[["price", "strike", "multiplier", "dividend_yield"]]
        if not numpy.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("derivative quote numbers must be finite")
        if (data["price"] < 0.0).any() or (data["multiplier"] <= 0.0).any():
            raise ValueError("prices must be nonnegative and multipliers positive")
        options = data["instrument_type"].isin({"equity_option", "futures_option"})
        if (data.loc[options, "strike"] <= 0.0).any():
            raise ValueError("option strikes must be positive")
        if not set(data.loc[options, "option_type"]).issubset({"C", "P"}):
            raise ValueError("option_type must be C or P")
        if not set(data.loc[options, "exercise_style"]).issubset({"E", "A"}):
            raise ValueError("exercise_style must be E or A")
        missing_symbols = set(command.instruments).difference(
            data["symbol"].astype(str)
        )
        if missing_symbols and not data.empty:
            raise ValueError(f"Derivative quotes are missing symbols: {sorted(missing_symbols)}")

    def _select(self, command: DataRequest):
        import pandas

        start = pandas.Timestamp(command.start_date)
        end = pandas.Timestamp(command.end_date)
        return self._entry(command).data.loc[
            self._entry(command).data["symbol"].astype(str).isin(command.instruments)
            & self._entry(command).data["date"].between(start, end)
        ].copy().reset_index(drop=True)
