"""Instrument-based download-command chunking."""

from collections.abc import Iterable

from ..command import UnifiedFormatCommand
from .chunker import Chunker


class InstrumentChunker(Chunker):
    """Split a command's instruments into fixed-size batches."""

    def __init__(self, batchSize: int) -> None:
        if batchSize <= 0:
            raise ValueError("batchSize must be greater than zero")
        self.batchSize = batchSize

    def createChunks(
        self,
        command: UnifiedFormatCommand,
    ) -> Iterable[UnifiedFormatCommand]:
        """Yield copies of ``command`` containing instrument batches."""
        instruments = command["instruments"]
        for start in range(0, len(instruments), self.batchSize):
            chunk = command.copy()
            chunk["instruments"] = instruments[start : start + self.batchSize]
            yield chunk
