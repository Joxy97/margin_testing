"""Typed results produced by portfolio margin backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from portfolio import Portfolio


class BaselColor(str, Enum):
    """Basel traffic-light classification."""

    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"


@dataclass(frozen=True)
class DailyBacktestTimings:
    """Wall-clock time spent in the major stages of one backtest day."""

    dataAcquisitionSeconds: float = 0.0
    riskStateGenerationSeconds: float = 0.0
    marginCalculationSeconds: float = 0.0
    realizedDataAcquisitionSeconds: float = 0.0
    realizedPnLCalculationSeconds: float = 0.0
    totalSeconds: float = 0.0


@dataclass(frozen=True)
class DailyBacktestResult:
    """Store margin coverage information for one portfolio and trade date."""

    date: date
    margin: float
    realizedPnL: float
    grossExposure: float
    marginPercent: float
    breach: bool
    timings: DailyBacktestTimings = DailyBacktestTimings()

    @property
    def realizedLoss(self) -> float:
        return -self.realizedPnL

    @property
    def covered(self) -> bool:
        return not self.breach


@dataclass(frozen=True)
class BacktestResults:
    """Store daily observations and aggregate Basel classification."""

    portfolio: Portfolio
    dailyResults: tuple[DailyBacktestResult, ...]
    violations: int
    baselProbability: float
    baselColor: BaselColor
    confidenceLevel: float

    @property
    def days(self) -> int:
        return len(self.dailyResults)


@dataclass(frozen=True)
class PortfolioBacktestRequest:
    """Pair one portfolio with the dates on which it should be tested."""

    portfolio: Portfolio
    dates: tuple[date, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dates", tuple(self.dates))
        if not isinstance(self.portfolio, Portfolio):
            raise TypeError("portfolio must be a Portfolio")
        if any(not isinstance(item, date) for item in self.dates):
            raise TypeError("dates must contain date objects")


@dataclass(frozen=True)
class BacktestBatchResults:
    """Store named backtest results for several portfolios."""

    results: Mapping[str, BacktestResults]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
