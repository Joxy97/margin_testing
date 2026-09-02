"""Policies for choosing among providers of the same data type."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from download_unit import DataProvider, LocalCSVDataProvider, YfinanceDataProvider


class ProviderSelection(ABC):
    """Choose one provider from the available provider registry entries."""

    @abstractmethod
    def selectProvider(
        self,
        providers: Sequence[DataProvider],
    ) -> DataProvider:
        """Return the selected provider."""
        raise NotImplementedError


class LocalFirstProviderSelection(ProviderSelection):
    """Prefer local CSV data, then yfinance, then registration order."""

    def selectProvider(
        self,
        providers: Sequence[DataProvider],
    ) -> DataProvider:
        """Return the best provider according to the local-first policy."""
        if not providers:
            raise ValueError("No provider can supply the requested data type")
        for provider_type in (LocalCSVDataProvider, YfinanceDataProvider):
            for provider in providers:
                if isinstance(provider, provider_type):
                    return provider
        return providers[0]
