"""Typed configuration for data-download services."""

from dataclasses import dataclass, field
from typing import Any, Mapping
from types import MappingProxyType
from download_unit.retry_policy import DownloadRetryPolicy

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

    def __post_init__(self) -> None:
        for name in ("providers", "downloadParameters", "requestParameters"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.downloadAlgorithm == "exponential_backoff":
            DownloadRetryPolicy(**{key: value for key, value in self.downloadParameters.items()
                                   if key != "chunker"})

    def createDownloadManager(self) -> DownloadManager:
        return DownloadManager(
            providers=self.providers,
            providerSelection=self.providerSelection,
            downloadAlgorithm=self.downloadAlgorithm,
            downloadParameters=self.downloadParameters,
        )
