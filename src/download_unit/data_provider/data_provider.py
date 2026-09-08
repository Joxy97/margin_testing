"""Base interface for data providers."""

from abc import ABC, abstractmethod
from typing import Any
from hashlib import sha256
import json

from ..command import Command, DataRequest


class DataProvider(ABC):
    """Download data and translate commands for a particular provider."""

    def sourceRevision(self, command: DataRequest) -> Any:
        return command.provider_parameters.get("revision", "")

    def datasetIdentity(self, command: DataRequest) -> str:
        """Stable configured-source identity; adapters own source freshness."""
        document = {"provider": f"{type(self).__module__}.{type(self).__qualname__}",
                    "parameters": dict(command.provider_parameters),
                    "revision": self.sourceRevision(command)}
        return sha256(json.dumps(document, sort_keys=True, default=str).encode()).hexdigest()

    def isTransientError(self, error: Exception) -> bool:
        """Classify transport failures; adapters may add provider-specific errors."""
        return isinstance(error, (TimeoutError, ConnectionError))

    def retryAfter(self, error: Exception) -> float | None:
        """Optional provider delay hint in seconds, interpreted inside the adapter."""
        return None

    @abstractmethod
    def getDataTypes(self) -> set[str]:
        """Return the data types supplied by this provider."""
        raise NotImplementedError

    @abstractmethod
    def convertRawData(self, raw_data: Any) -> Any:
        """Convert provider-specific raw data to the unified representation."""
        raise NotImplementedError

    @abstractmethod
    def convertCommand(self, command: DataRequest) -> Command:
        """Translate an application command into a provider-specific command."""
        raise NotImplementedError

    @abstractmethod
    def downloadData(self, command: Command) -> Any:
        """Download raw data using a provider-specific command."""
        raise NotImplementedError
