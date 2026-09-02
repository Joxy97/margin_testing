"""Factory for download-unit implementations."""

from typing import Any, Mapping

from .download_unit import DownloadUnit
from .exponential_backoff_download_unit import ExponentialBackoffDownloadUnit
from .single_request_download_unit import SingleRequestDownloadUnit


class DownloadUnitFactory:
    """Create download units from their configured algorithm names."""

    @staticmethod
    def createDownloadUnit(
        downloadAlgorithm: str,
        parameters: Mapping[str, Any],
    ) -> DownloadUnit:
        """Create the download unit associated with ``downloadAlgorithm``."""
        if downloadAlgorithm == "exponential_backoff":
            return ExponentialBackoffDownloadUnit(**parameters)
        if downloadAlgorithm == "single_request":
            return SingleRequestDownloadUnit(**parameters)
        raise ValueError(f"Unknown download algorithm: {downloadAlgorithm!r}")

    @staticmethod
    def create(
        downloadAlgorithm: str,
        parameters: Mapping[str, Any],
    ) -> DownloadUnit:
        """Create a download unit; shorthand for :meth:`createDownloadUnit`."""
        return DownloadUnitFactory.createDownloadUnit(
            downloadAlgorithm,
            parameters,
        )
