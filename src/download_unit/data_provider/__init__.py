"""Data-provider interfaces and implementations."""

from .data_provider import DataProvider
from .local_csv import LocalCSVDataProvider

__all__ = ["DataProvider", "LocalCSVDataProvider"]
