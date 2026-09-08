"""PCA-grid provider interface."""

from __future__ import annotations

from typing import Any, ClassVar
from dataclasses import dataclass, field
from hashlib import sha256

from cache import Cache, CacheFactory

from .pca_grid import PCAGrid
from .pca_grid_factory import PCAGridFactory
from .pca_key import PCAKey
from .pca_backend import PCABackend, PCABackendConfig


@dataclass(frozen=True)
class _CachedGrid:
    provenance: str
    grid: PCAGrid


@dataclass(frozen=True)
class PCAGridProviderConfig:
    """Declarative PCA settings; every construction owns a fresh cache."""

    cacheType: str = "lru"
    memorySize: int = 128
    maxMemoryBytes: int | None = None
    backend: PCABackendConfig = field(default_factory=PCABackendConfig)

    def __post_init__(self) -> None:
        if self.cacheType != "lru":
            raise ValueError(f"Unknown cache type: {self.cacheType!r}")
        for name in ("memorySize", "maxMemoryBytes"):
            value = getattr(self, name)
            if value is None and name == "maxMemoryBytes":
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def createProvider(self) -> PCAGridProvider:
        return PCAGridProvider(cacheType=self.cacheType, memorySize=self.memorySize,
                               maxMemoryBytes=self.maxMemoryBytes,
                               backend=self.backend.createBackend())


class PCAGridProvider:
    """Maintain one generator's collection of PCA grids."""

    DEFAULT_CACHE_MEMORY_SIZE: ClassVar[int] = 128

    cache: Cache[PCAKey, PCAGrid]

    def __init__(
        self,
        cache: Cache[PCAKey, PCAGrid] | None = None,
        cacheType: str = "lru",
        memorySize: int = DEFAULT_CACHE_MEMORY_SIZE,
        backend: PCABackend | None = None,
        maxMemoryBytes: int | None = None,
    ) -> None:
        self.cache = cache or CacheFactory.createCache(cacheType, memorySize)
        self.backend = backend
        if maxMemoryBytes is not None and maxMemoryBytes <= 0:
            raise ValueError("maxMemoryBytes must be positive or None")
        self.maxMemoryBytes = maxMemoryBytes

    def getPCAGrid(self, key: PCAKey) -> PCAGrid | None:
        """Return the PCA grid stored under ``key``, if one exists."""
        entry = self.cache.get(key)
        return entry.grid if isinstance(entry, _CachedGrid) else entry

    def getOrCreate(self, key: PCAKey, data: Any) -> PCAGrid:
        """Own cache validity and fitting for the exact input provenance."""
        import pandas
        digest = sha256(pandas.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
        digest.update(pandas.util.hash_pandas_object(data.columns, index=True).to_numpy().tobytes())
        provenance = digest.hexdigest()
        cached = self.cache.get(key)
        if isinstance(cached, _CachedGrid) and cached.provenance == provenance:
            return cached.grid
        from .pca_grid import ReturnsPCAGrid
        from .pca_key import ReturnsPCAKey
        if not isinstance(key, ReturnsPCAKey):
            raise ValueError(f"Unsupported PCA key type: {type(key).__name__}")
        grid = ReturnsPCAGrid.construct(key, data, self.backend)
        self.cache.insert(key, _CachedGrid(provenance, grid))
        if self.maxMemoryBytes is not None:
            while sum(getattr(entry.grid if isinstance(entry, _CachedGrid) else entry,
                              "numericMemoryBytes", 0) for entry in self.cache.values()) > self.maxMemoryBytes:
                self.cache.popOldest()
        return grid

    def createPCAGrid(self, key: PCAKey, data: Any) -> PCAGrid:
        """Create, cache, and return a PCA grid for ``key`` and ``data``."""
        return self.getOrCreate(key, data)

    def setCache(self, cache: Cache[PCAKey, PCAGrid]) -> None:
        """Replace the cache used to store PCA grids."""
        self.cache = cache
