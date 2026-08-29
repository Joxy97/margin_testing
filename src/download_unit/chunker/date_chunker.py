"""Date-based download-command chunking."""

from collections.abc import Iterable
from datetime import timedelta

from ..command import DataRequest
from .chunker import Chunker


class DateChunker(Chunker):
    """Split a command's date interval into fixed-day ranges."""

    def __init__(self, batchSize: int) -> None:
        if batchSize <= 0:
            raise ValueError("batchSize must be greater than zero")
        self.batchSize = batchSize

    def createChunks(
        self,
        command: DataRequest,
    ) -> Iterable[DataRequest]:
        """Yield copies of ``command`` containing date-range batches."""
        chunk_start = command.start_date
        end_date = command.end_date

        while chunk_start < end_date:
            chunk_end = min(
                chunk_start + timedelta(days=self.batchSize),
                end_date,
            )
            yield command.withChanges(
                start_date=chunk_start,
                end_date=chunk_end,
            )
            chunk_start = chunk_end
