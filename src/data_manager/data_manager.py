"""Cache-backed storage and retrieval of tabular market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from cache import Cache, CacheFactory
from download_unit import DataRequest, Period

from .backing_store import DataBackingStore

if TYPE_CHECKING:
    import pandas


@dataclass
class MarketDataPartition:
    """Store indexed data and downloaded coverage for each instrument."""

    data: pandas.DataFrame
    coverage: dict[str, list[tuple[date, date]]] = field(default_factory=dict)


class DataManager:
    """Store normalized DataFrames and serve command-shaped slices."""

    def __init__(
        self,
        cache: Cache[tuple[str, Period], MarketDataPartition] | None = None,
        cacheType: str = "lru",
        memorySize: int = 16,
        maxMemoryBytes: int | None = None,
        backingStore: DataBackingStore[
            tuple[str, Period], MarketDataPartition
        ] | None = None,
    ) -> None:
        self.cache = cache or CacheFactory.createCache(cacheType, memorySize)
        if maxMemoryBytes is not None and maxMemoryBytes <= 0:
            raise ValueError("maxMemoryBytes must be positive or None")
        self.maxMemoryBytes = maxMemoryBytes
        self.backingStore = backingStore

    def getData(
        self,
        command: DataRequest,
    ) -> pandas.DataFrame | None:
        """Return data covering ``command``, or ``None`` on a cache miss."""
        import pandas

        entry = self._getEntry(self._cacheKey(command))
        if entry is None or not self._covers(entry, command):
            return None

        return self._selectData(entry.data, command)

    def getMissingRequests(self, command: DataRequest) -> list[DataRequest]:
        """Return only the uncovered instrument/date portions of a request."""
        entry = self._getEntry(self._cacheKey(command))
        if entry is None:
            return [command]
        grouped: dict[tuple[date, date], list[str]] = {}
        for instrument in command.instruments:
            for interval in self._missingIntervals(
                command.start_date,
                command.end_date,
                entry.coverage.get(instrument, []),
            ):
                grouped.setdefault(interval, []).append(instrument)
        return [
            command.withChanges(
                instruments=tuple(instruments),
                start_date=start,
                end_date=end,
            )
            for (start, end), instruments in grouped.items()
        ]

    def storeData(
        self,
        command: DataRequest,
        data: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Normalize and merge downloaded data associated with ``command``."""
        normalized = self._normalizeData(data)
        instruments = list(command.instruments)
        missing = set(instruments).difference(normalized.columns)
        if missing:
            raise ValueError(
                f"Downloaded data is missing instruments: {sorted(missing)}"
            )

        cache_key = self._cacheKey(command)
        entry = self._getEntry(cache_key)
        if entry is None:
            entry = MarketDataPartition(normalized)
        else:
            entry.data = normalized.combine_first(entry.data).sort_index()

        start = command.start_date
        end = command.end_date
        for instrument in instruments:
            intervals = entry.coverage.setdefault(instrument, [])
            entry.coverage[instrument] = self._mergeIntervals(
                [*intervals, (start, end)]
            )
        self.cache.insert(cache_key, entry)
        if self.backingStore is not None:
            self.backingStore.put(cache_key, entry)
        self._evictToMemoryBudget()
        return self._selectData(entry.data, command)

    def _getEntry(
        self,
        key: tuple[str, Period],
    ) -> MarketDataPartition | None:
        entry = self.cache.get(key)
        if entry is not None or self.backingStore is None:
            return entry
        entry = self.backingStore.get(key)
        if entry is not None:
            self.cache.insert(key, entry)
            self._evictToMemoryBudget()
        return entry

    @staticmethod
    def _cacheKey(command: DataRequest) -> tuple[str, Period]:
        return command.data_type, command.period

    @staticmethod
    def _selectData(
        data: pandas.DataFrame,
        command: DataRequest,
    ) -> pandas.DataFrame:
        import pandas

        selected = data.loc[
            pandas.Timestamp(command.start_date) : pandas.Timestamp(
                command.end_date
            ),
            list(command.instruments),
        ].copy()
        selected.index.name = "date"
        return selected.reset_index()

    def _evictToMemoryBudget(self) -> None:
        if self.maxMemoryBytes is None:
            return
        while sum(
            int(entry.data.memory_usage(index=True, deep=True).sum())
            for entry in self.cache.values()
        ) > self.maxMemoryBytes:
            if self.cache.popOldest() is None:
                break

    @staticmethod
    def _normalizeData(data: pandas.DataFrame) -> pandas.DataFrame:
        import pandas

        if not isinstance(data, pandas.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        normalized = data.copy()
        if "date" in normalized.columns:
            normalized["date"] = pandas.to_datetime(normalized["date"])
            normalized = normalized.set_index("date")
        elif not isinstance(normalized.index, pandas.DatetimeIndex):
            raise ValueError("data must have a date column or DatetimeIndex")
        normalized.index = pandas.to_datetime(normalized.index)
        normalized.index.name = "date"
        return normalized.sort_index()

    @staticmethod
    def _covers(
        entry: MarketDataPartition,
        command: DataRequest,
    ) -> bool:
        start = command.start_date
        end = command.end_date
        return all(
            any(
                covered_start <= start and covered_end >= end
                for covered_start, covered_end in entry.coverage.get(
                    instrument, []
                )
            )
            for instrument in command.instruments
        )

    @staticmethod
    def _mergeIntervals(
        intervals: list[tuple[date, date]],
    ) -> list[tuple[date, date]]:
        merged: list[tuple[date, date]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1] + timedelta(days=1):
                merged.append((start, end))
            else:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end))
        return merged

    @staticmethod
    def _missingIntervals(
        start: date,
        end: date,
        coveredIntervals: list[tuple[date, date]],
    ) -> list[tuple[date, date]]:
        missing = []
        cursor = start
        for covered_start, covered_end in DataManager._mergeIntervals(
            coveredIntervals
        ):
            if covered_end < cursor:
                continue
            if covered_start > end:
                break
            if covered_start > cursor:
                missing.append(
                    (cursor, min(end, covered_start - timedelta(days=1)))
                )
            cursor = max(cursor, covered_end + timedelta(days=1))
            if cursor > end:
                break
        if cursor <= end:
            missing.append((cursor, end))
        return missing
