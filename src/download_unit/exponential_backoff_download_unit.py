"""Chunked downloads paced with exponential backoff."""

from datetime import timedelta
from itertools import product
from time import sleep
from typing import Any, Iterator

from .command import UnifiedFormatCommand
from .data_provider import DataProvider
from .download_unit import DownloadUnit


class ExponentialBackoffDownloadUnit(DownloadUnit):
    """Download instrument/date chunks with an exponentially growing delay."""

    def __init__(
        self,
        instrument_batch: int,
        dates_batch: int,
        time: float,
    ) -> None:
        super().__init__()
        if instrument_batch <= 0:
            raise ValueError("instrument_batch must be greater than zero")
        if dates_batch <= 0:
            raise ValueError("dates_batch must be greater than zero")
        if time < 0:
            raise ValueError("time must not be negative")

        self.instrument_batch = instrument_batch
        self.dates_batch = dates_batch
        self.time = time

    def getRawData(
        self,
        provider: DataProvider,
        command: UnifiedFormatCommand,
    ) -> list[Any]:
        """Download every instrument/date chunk and return its raw responses."""
        if not isinstance(command, UnifiedFormatCommand):
            raise TypeError("command must be a UnifiedFormatCommand instance")
        if command.start_date > command.end_date:
            raise ValueError("start_date must not be later than end_date")

        instrument_chunks = list(self._instrument_chunks(command))
        date_chunks = list(self._date_chunks(command))
        raw_data: list[Any] = []
        delay = self.time

        for instruments, (start_date, end_date) in product(
            instrument_chunks,
            date_chunks,
        ):
            chunk = UnifiedFormatCommand(
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
                period=command.period,
                location=command.location,
            )
            provider_command = provider.convertCommand(chunk)
            raw_data.append(provider.downloadData(provider_command))
            sleep(delay)
            delay *= 2

        return raw_data

    def _instrument_chunks(
        self,
        command: UnifiedFormatCommand,
    ) -> Iterator[list[Any]]:
        for start in range(0, len(command.instruments), self.instrument_batch):
            yield command.instruments[start : start + self.instrument_batch]

    def _date_chunks(
        self,
        command: UnifiedFormatCommand,
    ) -> Iterator[tuple[Any, Any]]:
        chunk_start = command.start_date
        while chunk_start < command.end_date:
            chunk_end = min(
                chunk_start + timedelta(days=self.dates_batch),
                command.end_date,
            )
            yield chunk_start, chunk_end
            chunk_start = chunk_end
