"""Downloads paced with exponential backoff."""

from time import sleep
from typing import Any

from .chunker import Chunker
from .command import UnifiedFormatCommand
from .data_provider import DataProvider
from .download_unit import DownloadUnit


class ExponentialBackoffDownloadUnit(DownloadUnit):
    """Download command chunks with an exponentially growing delay."""

    def __init__(
        self,
        chunker: Chunker,
        time: float,
    ) -> None:
        super().__init__()
        if time < 0:
            raise ValueError("time must not be negative")

        self.chunker = chunker
        self.time = time

    def getRawData(
        self,
        provider: DataProvider,
        command: UnifiedFormatCommand,
    ) -> list[Any]:
        """Download every instrument/date chunk and return its raw responses."""
        if not isinstance(command, dict):
            raise TypeError("command must be a dictionary")
        raw_data: list[Any] = []
        delay = self.time

        for chunk in self.chunker.createChunks(command):
            provider_command = provider.convertCommand(chunk)
            raw_data.append(provider.downloadData(provider_command))
            sleep(delay)
            delay *= 2

        return raw_data
