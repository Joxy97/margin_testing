"""Typed configuration for market-data storage."""

from dataclasses import dataclass

from cache import Cache
from download_unit import Period

from .data_manager import DataManager, MarketDataPartition
from .derivative_data_manager import DerivativeQuoteDataManager
from .backing_store import DataBackingStore


@dataclass(frozen=True)
class DataManagerConfig:
    """Configuration used to create a cache-backed data manager."""

    cache: Cache[tuple[str, Period], MarketDataPartition] | None = None
    cacheType: str = "lru"
    memorySize: int = 16
    maxMemoryBytes: int | None = None
    backingStore: DataBackingStore[
        tuple[str, Period], MarketDataPartition
    ] | None = None

    def createDataManager(self) -> DataManager:
        return DataManager(
            cache=self.cache,
            cacheType=self.cacheType,
            memorySize=self.memorySize,
            maxMemoryBytes=self.maxMemoryBytes,
            backingStore=self.backingStore,
        )


@dataclass(frozen=True)
class DerivativeQuoteDataManagerConfig:
    """Configuration for long-form derivative quote storage."""

    def createDataManager(self) -> DerivativeQuoteDataManager:
        return DerivativeQuoteDataManager()
