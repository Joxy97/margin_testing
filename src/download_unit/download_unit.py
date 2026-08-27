"""Base interface for download orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas

from .command import Command
from .data_provider import DataProvider


class DownloadUnit(ABC):
    """Coordinate data retrieval through a data provider."""

    def __init__(self) -> None:
        self._provider: DataProvider | None = None

    def getData(self, provider: DataProvider, command: Command) -> Any:
        """Download, convert, and return data for ``command``."""
        self._provider = provider
        raw_data = self.getRawData(provider, command)
        return self.convertRawData(raw_data)

    @abstractmethod
    def getRawData(self, provider: DataProvider, command: Command) -> Any:
        """Retrieve unconverted data from ``provider`` for ``command``."""
        raise NotImplementedError

    def convertRawData(self, raw_data: Any) -> Any:
        """Convert raw data through the provider used by :meth:`getData`."""
        if self._provider is None:
            raise RuntimeError("getData must be called before convertRawData")
        return self._provider.convertRawData(raw_data)

    @staticmethod
    def storeData(data: pandas.DataFrame, path: str | PathLike[str]) -> None:
        """Store ``data`` in the CSV file identified by ``path``."""
        import pandas

        if not isinstance(data, pandas.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        data.to_csv(path)
