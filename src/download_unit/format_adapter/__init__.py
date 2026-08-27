"""Format-adapter interfaces, implementations, and factory."""

from .format_adapter import FormatAdapter
from .format_adapter_factory import FormatAdapterFactory
from .yfinance import YfinanceFormatAdapter

__all__ = [
    "FormatAdapter",
    "FormatAdapterFactory",
    "YfinanceFormatAdapter",
]
