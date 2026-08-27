"""Provider-independent download command."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from .command import Command


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

@dataclass(frozen=True)
class UnifiedFormatCommand(Command):
    """Describe a download request independently of a data provider."""

    instruments: list[Instrument]
    start_date: date
    end_date: date
    period: Period
