"""Download orchestration for providers that need a single request."""

from typing import Any

from .command import DataRequest
from .data_provider import DataProvider
from .download_unit import DownloadUnit


class SingleRequestDownloadUnit(DownloadUnit):
    """Retrieve a dataset with one provider request."""

    def getRawData(
        self,
        provider: DataProvider,
        command: DataRequest,
    ) -> Any:
        """Convert ``command`` and make one provider request."""
        provider_command = provider.convertCommand(command)
        return provider.downloadData(provider_command)
