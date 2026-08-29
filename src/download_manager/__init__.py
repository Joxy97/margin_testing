"""Download-provider registry management."""

from .download_manager import DownloadManager
from .config import DownloadManagerConfig
from .provider_selection import LocalFirstProviderSelection, ProviderSelection

__all__ = [
    "DownloadManager",
    "DownloadManagerConfig",
    "LocalFirstProviderSelection",
    "ProviderSelection",
]
