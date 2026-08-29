"""Typed configuration for data-download services."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from download_unit.data_provider import DataProvider

from .download_manager import DownloadManager
from .provider_selection import LocalFirstProviderSelection, ProviderSelection


@dataclass(frozen=True)
class DownloadManagerConfig:
    """Configuration used to create an independent download manager."""

    providers: Mapping[str, DataProvider] = field(default_factory=dict)
    providerSelection: ProviderSelection = field(
        default_factory=LocalFirstProviderSelection
    )
    downloadAlgorithm: str = "single_request"
    downloadParameters: Mapping[str, Any] = field(default_factory=dict)
    requestParameters: Mapping[str, Any] = field(default_factory=dict)

    def createDownloadManager(self) -> DownloadManager:
        return DownloadManager(
            providers=self.providers,
            providerSelection=self.providerSelection,
            downloadAlgorithm=self.downloadAlgorithm,
            downloadParameters=self.downloadParameters,
        )
