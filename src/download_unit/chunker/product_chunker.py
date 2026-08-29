"""Cartesian-product download-command chunking."""

from collections.abc import Iterable

from ..command import DataRequest
from .chunker import Chunker


class ProductChunker(Chunker):
    """Apply a second chunker to every chunk made by a first chunker."""

    def __init__(self, firstChunker: Chunker, secondChunker: Chunker) -> None:
        self.firstChunker = firstChunker
        self.secondChunker = secondChunker

    def createChunks(
        self,
        command: DataRequest,
    ) -> Iterable[DataRequest]:
        """Yield the product of both child chunkers' command chunks."""
        for first_chunk in self.firstChunker.createChunks(command):
            yield from self.secondChunker.createChunks(first_chunk)
