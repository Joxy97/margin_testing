"""Cache-backed storage and retrieval of tabular market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from cache import Cache, CacheFactory
from download_unit import DataRequest, Period

from .backing_store import DataBackingStore
from .interval_coverage import IntervalCoverage

PartitionKey = tuple[str, Period] | tuple[str, str, Period]

if TYPE_CHECKING:
    import pandas


@dataclass
class MarketDataPartition:
    """Store indexed data and downloaded coverage for each instrument."""

    data: pandas.DataFrame
    coverage: IntervalCoverage = field(default_factory=IntervalCoverage)


class DataManager:
    """Store normalized DataFrames and serve command-shaped slices."""

    def __init__(
        self,
        cache: Cache[PartitionKey, MarketDataPartition] | None = None,
        cacheType: str = "lru",
        memorySize: int = 16,
        maxMemoryBytes: int | None = None,
        backingStore: DataBackingStore[
            PartitionKey, MarketDataPartition
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
        return IntervalCoverage(entry.coverage).missingRequests(command)

    def getAvailableData(self, command: DataRequest):
        """Copy retained observations before active acquisition can evict them."""
        entry = self._getEntry(self._cacheKey(command))
        if entry is None:
            return None
        return self._selectData(entry.data.reindex(columns=command.instruments), command)

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

        entry.coverage = IntervalCoverage(entry.coverage)
        entry.coverage.add(command)
        self.cache.insert(cache_key, entry)
        if self.backingStore is not None:
            self.backingStore.put(cache_key, entry)
        self._evictToMemoryBudget()
        return self._selectData(entry.data, command)

    def _getEntry(
        self,
        key: PartitionKey,
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
    def _cacheKey(command: DataRequest) -> PartitionKey:
        if command.datasetIdentity:
            return command.datasetIdentity, command.data_type, command.period
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
    def _covers(entry: MarketDataPartition, command: DataRequest) -> bool:
        return not IntervalCoverage(entry.coverage).missingRequests(command)
