"""Optional persistent storage for market-data partitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
import os
import pickle
from typing import Generic, TypeVar


Key = TypeVar("Key")
Data = TypeVar("Data")


class DataBackingStore(ABC, Generic[Key, Data]):
    """Persistent second-level storage used behind an in-memory cache."""

    @abstractmethod
    def get(self, key: Key) -> Data | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, key: Key, data: Data) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, key: Key) -> None:
        raise NotImplementedError


class PartitionedPickleDataStore(DataBackingStore[Key, Data]):
    """Store each cache partition in its own atomic local file.

    Pickle files must only be read from a trusted, application-owned directory.
    Partition filenames are stable hashes, so arbitrary keys cannot escape it.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: Key) -> Data | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("rb") as stream:
            stored_key, data = pickle.load(stream)
        if stored_key != key:
            raise RuntimeError(f"Backing-store key mismatch in {path}")
        return data

    def put(self, key: Key, data: Data) -> None:
        path = self._path(key)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                pickle.dump((key, data), stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def remove(self, key: Key) -> None:
        self._path(key).unlink(missing_ok=True)

    @staticmethod
    def _keyBytes(key: Key) -> bytes:
        return pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)

    def _path(self, key: Key) -> Path:
        digest = sha256(self._keyBytes(key)).hexdigest()
        return self.directory / f"{digest}.pkl"
