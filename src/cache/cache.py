"""Generic bounded in-memory cache."""

from collections.abc import Iterable
from threading import RLock
from typing import Generic, TypeVar


Key = TypeVar("Key")
Data = TypeVar("Data")


class Cache(Generic[Key, Data]):
    """Store key-data pairs in bounded memory."""

    def __init__(self, memory_size: int) -> None:
        if memory_size < 1:
            raise ValueError("memory_size must be positive")
        self.memory: dict[Key, Data] = {}
        self.memory_size = memory_size
        self._memoryLock = RLock()

    def get(self, key: Key) -> Data | None:
        """Return the data stored under ``key``, if present."""
        with self._memoryLock:
            return self.memory.get(key)

    def insert(self, key: Key, data: Data) -> None:
        """Insert a pair and evict the oldest pair when memory is full."""
        with self._memoryLock:
            self.memory.pop(key, None)
            self.memory[key] = data
            while len(self.memory) > self.memory_size:
                self.remove(next(iter(self.memory)))

    def insertPairs(self, pairs: Iterable[tuple[Key, Data]]) -> None:
        """Insert each supplied key-data pair."""
        with self._memoryLock:
            for key, data in pairs:
                self.insert(key, data)

    def remove(self, key: Key) -> None:
        """Remove ``key`` if it is present."""
        with self._memoryLock:
            self.memory.pop(key, None)

    def purge(self) -> None:
        """Remove all cached pairs."""
        with self._memoryLock:
            self.memory.clear()

    def popOldest(self) -> tuple[Key, Data] | None:
        """Remove and return the oldest pair, if one exists."""
        with self._memoryLock:
            if not self.memory:
                return None
            key = next(iter(self.memory))
            return key, self.memory.pop(key)

    def values(self) -> tuple[Data, ...]:
        """Return a stable snapshot of the cached values."""
        with self._memoryLock:
            return tuple(self.memory.values())
