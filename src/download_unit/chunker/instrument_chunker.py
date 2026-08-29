"""Instrument-based download-command chunking."""

from collections.abc import Iterable

from ..command import DataRequest
from .chunker import Chunker


class InstrumentChunker(Chunker):
    """Split a command's instruments into fixed-size batches."""

    def __init__(self, batchSize: int) -> None:
        if batchSize <= 0:
            raise ValueError("batchSize must be greater than zero")
        self.batchSize = batchSize

    def createChunks(
        self,
        command: DataRequest,
    ) -> Iterable[DataRequest]:
        """Yield copies of ``command`` containing instrument batches."""
        instruments = command.instruments
        for start in range(0, len(instruments), self.batchSize):
            yield command.withChanges(
                instruments=instruments[start : start + self.batchSize]
            )
