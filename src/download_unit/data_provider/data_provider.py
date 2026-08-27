"""Base interface for data providers."""

from abc import ABC, abstractmethod
from typing import Any

from ..command import Command, UnifiedFormatCommand
from ..format_adapter import FormatAdapterFactory


class DataProvider(ABC):
    """Download data and translate commands for a particular provider."""

    def __init__(self, format: str) -> None:
        self.formatAdapter = FormatAdapterFactory.create(format)

    def convertRawData(self, raw_data: Any) -> Any:
        """Convert raw data using this provider's format adapter."""
        return self.formatAdapter.convertRawData(raw_data)

    @abstractmethod
    def convertCommand(self, command: UnifiedFormatCommand) -> Command:
        """Translate an application command into a provider-specific command."""
        raise NotImplementedError

    @abstractmethod
    def downloadData(self, command: Command) -> Any:
        """Download raw data using a provider-specific command."""
        raise NotImplementedError
