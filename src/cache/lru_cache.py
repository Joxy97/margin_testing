"""Least-recently-used cache implementation."""

from collections import OrderedDict

from .cache import Cache, Data, Key


class LRUCache(Cache[Key, Data]):
    """Evict the pair that has gone unused for the longest time."""

    def __init__(self, memory_size: int) -> None:
        super().__init__(memory_size)
        self.memory: OrderedDict[Key, Data] = OrderedDict()

    def get(self, key: Key) -> Data | None:
        """Return ``key``'s data and mark the pair as most recently used."""
        with self._memoryLock:
            if key not in self.memory:
                return None
            self.memory.move_to_end(key)
            return self.memory[key]

    def insert(self, key: Key, data: Data) -> None:
        """Insert a pair and evict the least recently used pair if full."""
        with self._memoryLock:
            self.memory[key] = data
            self.memory.move_to_end(key)
            while len(self.memory) > self.memory_size:
                self.memory.popitem(last=False)
