"""Base interface for raw-data format adapters."""

from abc import ABC, abstractmethod
from typing import Any


class FormatAdapter(ABC):
    """Convert data in a provider-specific format to an application format."""

    @abstractmethod
    def convertRawData(self, raw_data: Any) -> Any:
        """Convert raw provider data into the desired application representation."""
        raise NotImplementedError
