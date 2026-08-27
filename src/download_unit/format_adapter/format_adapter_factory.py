"""Factory for format adapters."""

from typing import Dict, Type

from .format_adapter import FormatAdapter
from .yfinance import YfinanceFormatAdapter


class FormatAdapterFactory:
    """Create format adapters registered under case-insensitive format names."""

    _adapters: Dict[str, Type[FormatAdapter]] = {
        "yfinance": YfinanceFormatAdapter,
    }

    @classmethod
    def register(cls, format: str, adapter: Type[FormatAdapter]) -> None:
        """Associate ``format`` with a concrete adapter class."""
        key = cls._normalize_format(format)
        if not isinstance(adapter, type) or not issubclass(adapter, FormatAdapter):
            raise TypeError("adapter must be a FormatAdapter subclass")
        if adapter is FormatAdapter or getattr(adapter, "__abstractmethods__", None):
            raise TypeError("adapter must be a concrete FormatAdapter subclass")
        cls._adapters[key] = adapter

    @classmethod
    def create(cls, format: str) -> FormatAdapter:
        """Create the adapter registered for ``format``."""
        key = cls._normalize_format(format)
        try:
            adapter = cls._adapters[key]
        except KeyError as error:
            supported = ", ".join(sorted(cls._adapters))
            raise ValueError(
                f"Unsupported format {format!r}. Registered formats: {supported}"
            ) from error
        return adapter()

    @classmethod
    def getFormatAdapter(cls, format: str) -> FormatAdapter:
        """Create an adapter; camel-case alias matching the domain API."""
        return cls.create(format)

    @staticmethod
    def _normalize_format(format: str) -> str:
        if not isinstance(format, str):
            raise TypeError("format must be a string")
        key = format.strip().lower()
        if not key:
            raise ValueError("format must not be empty")
        return key
