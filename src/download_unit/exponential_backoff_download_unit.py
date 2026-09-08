"""Chunked downloads with bounded retries for transient provider failures."""

from time import sleep
from random import uniform
from .retry_policy import DownloadRetryPolicy
from typing import Any

from .chunker import Chunker
from .command import DataRequest
from .data_provider import DataProvider
from .download_unit import DownloadUnit
from .data_assembly import assembleData


class ExponentialBackoffDownloadUnit(DownloadUnit):
    """Retry each failed chunk independently; successful chunks continue immediately."""

    def __init__(
        self,
        chunker: Chunker,
        time: float,
        maxAttempts: int = 3,
        maxDelay: float = 60.0,
        jitter: float = 0.0,
        requestInterval: float = 0.0,
    ) -> None:
        super().__init__()
        self.retryPolicy = DownloadRetryPolicy(time, maxAttempts, maxDelay, jitter, requestInterval)

        self.chunker = chunker
        self.time = time
        self.maxAttempts = maxAttempts
        self.maxDelay = maxDelay
        self.jitter = jitter
        self.requestInterval = requestInterval

    def getData(self, provider: DataProvider, command: DataRequest):
        """Normalize each response before assembling requested observations."""
        return assembleData(command, (
            provider.convertRawData(raw)
            for raw in self._rawResponses(provider, command)
        ))

    def getRawData(
        self,
        provider: DataProvider,
        command: DataRequest,
    ) -> list[Any]:
        """Download every instrument/date chunk and return its raw responses."""
        return list(self._rawResponses(provider, command))

    def _rawResponses(self, provider: DataProvider, command: DataRequest):
        if not isinstance(command, DataRequest):
            raise TypeError("command must be a DataRequest")
        for index, chunk in enumerate(self.chunker.createChunks(command)):
            if index and self.requestInterval:
                sleep(self.requestInterval)
            provider_command = provider.convertCommand(chunk)
            delay = min(self.time, self.maxDelay)
            for attempt in range(self.maxAttempts):
                try:
                    response = provider.downloadData(provider_command)
                except Exception as error:
                    if not provider.isTransientError(error) or attempt + 1 == self.maxAttempts:
                        raise
                    hint = provider.retryAfter(error)
                    wait = max(delay, hint or 0.0) + uniform(0.0, delay * self.jitter)
                    sleep(min(wait, self.maxDelay))
                    delay = min(delay * 2, self.maxDelay)
                else:
                    yield response
                    break
