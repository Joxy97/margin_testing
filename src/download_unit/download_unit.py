"""Base interface for download orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas

from .command import DataRequest
from .data_provider import DataProvider


class DownloadUnit(ABC):
    """Coordinate data retrieval through a data provider."""

    def getData(
        self,
        provider: DataProvider,
        command: DataRequest,
    ) -> Any:
        """Download, convert, and return data for ``command``."""
        raw_data = self.getRawData(provider, command)
        return provider.convertRawData(raw_data)

    @abstractmethod
    def getRawData(
        self,
        provider: DataProvider,
        command: DataRequest,
    ) -> Any:
        """Retrieve unconverted data from ``provider`` for ``command``."""
        raise NotImplementedError

    @staticmethod
    def storeData(data: pandas.DataFrame, path: str | PathLike[str]) -> None:
        """Store ``data`` in the CSV file identified by ``path``."""
        import pandas

        if not isinstance(data, pandas.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        data.to_csv(path)
