"""Cache long-form futures and option-chain quotes."""

from __future__ import annotations

from datetime import timedelta

from download_unit import DataRequest


class DerivativeQuoteDataManager:
    """Serve long-form quote rows through the standard manager methods."""

    identityColumns = (
        "date", "symbol", "instrument_type", "expiration_date",
        "strike", "option_type", "exercise_style",
    )

    def __init__(self) -> None:
        self.data = None
        self.coverage = {}

    def getData(self, command: DataRequest):
        if self.data is None or self.getMissingRequests(command):
            return None
        return self._select(command)

    def getMissingRequests(self, command: DataRequest) -> list[DataRequest]:
        grouped = {}
        for symbol in command.instruments:
            for interval in self._missingIntervals(
                command.start_date, command.end_date, self.coverage.get(symbol, [])
            ):
                grouped.setdefault(interval, []).append(symbol)
        return [
            command.withChanges(
                instruments=tuple(symbols), start_date=start, end_date=end
            )
            for (start, end), symbols in grouped.items()
        ]

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
        combined = normalized if self.data is None else pandas.concat(
            (self.data, normalized), ignore_index=True
        )
        self.data = combined.drop_duplicates(
            list(self.identityColumns), keep="last"
        ).sort_values(["date", "symbol", "expiration_date", "strike"])
        for symbol in command.instruments:
            intervals = self.coverage.setdefault(symbol, [])
            self.coverage[symbol] = self._mergeIntervals(
                [*intervals, (command.start_date, command.end_date)]
            )
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
        if missing_symbols:
            raise ValueError(f"Derivative quotes are missing symbols: {sorted(missing_symbols)}")

    def _select(self, command: DataRequest):
        import pandas

        start = pandas.Timestamp(command.start_date)
        end = pandas.Timestamp(command.end_date)
        return self.data.loc[
            self.data["symbol"].astype(str).isin(command.instruments)
            & self.data["date"].between(start, end)
        ].copy().reset_index(drop=True)

    @staticmethod
    def _mergeIntervals(intervals):
        merged = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1] + timedelta(days=1):
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @classmethod
    def _missingIntervals(cls, start, end, covered):
        missing, cursor = [], start
        for covered_start, covered_end in cls._mergeIntervals(covered):
            if covered_end < cursor:
                continue
            if covered_start > end:
                break
            if covered_start > cursor:
                missing.append((cursor, min(end, covered_start - timedelta(days=1))))
            cursor = max(cursor, covered_end + timedelta(days=1))
        if cursor <= end:
            missing.append((cursor, end))
        return missing
