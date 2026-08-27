"""Format adapter for data returned by yfinance."""

from typing import Any

from .format_adapter import FormatAdapter


class YfinanceFormatAdapter(FormatAdapter):
    """Pass yfinance data through without modification."""

    def convertRawData(self, raw_data: Any) -> Any:
        """Return yfinance data unchanged."""
        return raw_data
