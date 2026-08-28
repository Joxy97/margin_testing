"""Base interface for data providers."""

from abc import ABC, abstractmethod
from typing import Any

from ..command import Command


class DataProvider(ABC):
    """Download data and translate commands for a particular provider."""

    @abstractmethod
    def convertRawData(self, raw_data: Any) -> Any:
        """Convert provider-specific raw data to the unified representation."""
        raise NotImplementedError

    @abstractmethod
    def convertCommand(self, command: Command) -> Command:
        """Translate an application command into a provider-specific command."""
        raise NotImplementedError

    @abstractmethod
    def downloadData(self, command: Command) -> Any:
        """Download raw data using a provider-specific command."""
        raise NotImplementedError
