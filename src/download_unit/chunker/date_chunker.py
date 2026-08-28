"""Date-based download-command chunking."""

from collections.abc import Iterable
from datetime import timedelta

from ..command import UnifiedFormatCommand
from .chunker import Chunker


class DateChunker(Chunker):
    """Split a command's date interval into fixed-day ranges."""

    def __init__(self, batchSize: int) -> None:
        if batchSize <= 0:
            raise ValueError("batchSize must be greater than zero")
        self.batchSize = batchSize

    def createChunks(
        self,
        command: UnifiedFormatCommand,
    ) -> Iterable[UnifiedFormatCommand]:
        """Yield copies of ``command`` containing date-range batches."""
        chunk_start = command["start_date"]
        end_date = command["end_date"]
        if chunk_start > end_date:
            raise ValueError("start_date must not be later than end_date")

        while chunk_start < end_date:
            chunk_end = min(
                chunk_start + timedelta(days=self.batchSize),
                end_date,
            )
            chunk = command.copy()
            chunk["start_date"] = chunk_start
            chunk["end_date"] = chunk_end
            yield chunk
            chunk_start = chunk_end
