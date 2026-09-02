"""Registry and selection service for data providers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from download_unit import DataRequest, DownloadUnitFactory
from download_unit.data_provider import DataProvider

from .provider_selection import LocalFirstProviderSelection, ProviderSelection


class DownloadManager:
    """Maintain one engine's named data-provider registry."""
    providers: dict[str, DataProvider]
    providerSelection: ProviderSelection
    downloadAlgorithm: str
    downloadParameters: dict[str, Any]

    def __init__(
        self,
        providers: Mapping[str, DataProvider] | None = None,
        providerSelection: ProviderSelection | None = None,
        downloadAlgorithm: str = "single_request",
        downloadParameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.providers = dict(providers or {})
        self.providerSelection = providerSelection or LocalFirstProviderSelection()
        self.downloadAlgorithm = downloadAlgorithm
        self.downloadParameters = dict(downloadParameters or {})

    def returnProviders(self, dataType: str) -> list[DataProvider]:
        """Return registered providers capable of supplying ``dataType``."""
        return [
            provider
            for provider in self.providers.values()
            if dataType in provider.getDataTypes()
        ]

    def addProvider(self, key: str, provider: DataProvider) -> None:
        """Add a provider, replacing the provider already stored under ``key``."""
        self.providers[key] = provider

    def addProviders(
        self, providers: Iterable[tuple[str, DataProvider]]
    ) -> None:
        """Add each key-provider pair from ``providers``."""
        for key, provider in providers:
            self.addProvider(key, provider)

    def removeProvider(self, key: str) -> None:
        """Remove ``key`` if it is present."""
        self.providers.pop(key, None)

    def removeProviders(self, keys: Iterable[str]) -> None:
        """Remove every key supplied by ``keys``."""
        for key in tuple(keys):
            self.removeProvider(key)

    def purgeProviders(self) -> None:
        """Remove all registered providers."""
        self.providers.clear()

    def downloadData(
        self,
        key: str,
        command: DataRequest,
        downloadAlgorithm: str,
        parameters: Mapping[str, Any],
    ) -> Any:
        """Download data with the named provider and download algorithm."""
        download_unit = DownloadUnitFactory.createDownloadUnit(
            downloadAlgorithm,
            parameters,
        )
        provider = self.providers[key]
        return download_unit.getData(provider, command)

    def downloadDataType(
        self,
        dataType: str,
        command: DataRequest,
    ) -> Any:
        """Select a provider and download the requested type of data."""
        providers = self.returnProviders(dataType)
        provider = self.providerSelection.selectProvider(providers)
        download_unit = DownloadUnitFactory.createDownloadUnit(
            self.downloadAlgorithm,
            self.downloadParameters,
        )
        return download_unit.getData(provider, command)
