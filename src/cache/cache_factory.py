"""Factory for cache implementations."""

from .cache import Cache
from .lru_cache import LRUCache


class CacheFactory:
    """Create caches from their configured policy names."""

    @staticmethod
    def createCache(cache_type: str, memory_size: int = 128) -> Cache:
        """Create the cache associated with ``cache_type``."""
        if cache_type == "lru":
            return LRUCache(memory_size)
        raise ValueError(f"Unknown cache type: {cache_type!r}")

    @staticmethod
    def create(cache_type: str, memory_size: int = 128) -> Cache:
        """Create a cache; shorthand for :meth:`createCache`."""
        return CacheFactory.createCache(cache_type, memory_size)
