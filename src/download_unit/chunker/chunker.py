"""Base interface for download-command chunking strategies."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..command import DataRequest


class Chunker(ABC):
    """Split a unified download command into smaller commands."""

    @abstractmethod
    def createChunks(
        self,
        command: DataRequest,
    ) -> Iterable[DataRequest]:
        """Create chunks from ``command``."""
        raise NotImplementedError
