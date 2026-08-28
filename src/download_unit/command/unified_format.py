"""Provider-independent download command format."""

from __future__ import annotations

from enum import Enum
from typing import Any


class Period(str, Enum):
    """Time periods accepted by :func:`yfinance.download`."""

    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    YEAR_TO_DATE = "ytd"
    MAX = "max"

Instrument = str

# The shared keys are ``instruments``, ``start_date``, ``end_date``, ``period``,
# and, when needed by a provider, ``location``.
UnifiedFormatCommand = dict[str, Any]
