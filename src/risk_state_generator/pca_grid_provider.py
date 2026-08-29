"""PCA-grid provider interface."""

from __future__ import annotations

from typing import Any, ClassVar

from cache import Cache, CacheFactory

from .pca_grid import PCAGrid
from .pca_grid_factory import PCAGridFactory
from .pca_key import PCAKey


class PCAGridProvider:
    """Maintain one generator's collection of PCA grids."""

    DEFAULT_CACHE_MEMORY_SIZE: ClassVar[int] = 128

    cache: Cache[PCAKey, PCAGrid]

    def __init__(
        self,
        cache: Cache[PCAKey, PCAGrid] | None = None,
        cacheType: str = "lru",
        memorySize: int = DEFAULT_CACHE_MEMORY_SIZE,
    ) -> None:
        self.cache = cache or CacheFactory.createCache(cacheType, memorySize)

    def getPCAGrid(self, key: PCAKey) -> PCAGrid | None:
        """Return the PCA grid stored under ``key``, if one exists."""
        return self.cache.get(key)

    def createPCAGrid(self, key: PCAKey, data: Any) -> PCAGrid:
        """Create, cache, and return a PCA grid for ``key`` and ``data``."""
        pca_grid = PCAGridFactory.createPCAGrid(key, data)
        self.cache.insert(key, pca_grid)
        return pca_grid

    def setCache(self, cache: Cache[PCAKey, PCAGrid]) -> None:
        """Replace the cache used to store PCA grids."""
        self.cache = cache
