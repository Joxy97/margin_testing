"""Singleton registry for data providers."""

from __future__ import annotations

from threading import Lock
from typing import Any, ClassVar, Iterable, Mapping

from download_unit import DownloadUnitFactory, UnifiedFormatCommand
from download_unit.data_provider import DataProvider


class DownloadManager:
    """Maintain the application's named data providers in one shared registry."""

    _instance: ClassVar[DownloadManager | None] = None
    _instance_lock: ClassVar[Lock] = Lock()
    providers: dict[str, DataProvider]

    def __new__(cls) -> DownloadManager:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.providers = {}
        return cls._instance

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
        command: UnifiedFormatCommand,
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
