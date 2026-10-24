"""Acquisition adapter shared by calculations and rolling replay."""

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from data_manager.acquisition import MarketDataAcquisition
from download_unit import DataRequest, Period
from .values import ExposurePortfolio


class MarketData(Protocol):
    def acquire(self, portfolio: ExposurePortfolio, marginDate: date) -> pd.DataFrame: ...
    def prepare(self, portfolio: ExposurePortfolio, dates: Sequence[date]) -> None: ...


class AcquiredMarketData:
    """Use existing coverage, source identity, cache and download policies."""

    def __init__(self, acquisition: MarketDataAcquisition, ew_window: int, requestParameters=None):
        self.acquisition = acquisition
        self.ew_window = ew_window
        self.requestParameters = dict(requestParameters or {})

    def _request(self, portfolio, day):
        # Same inclusive acquisition window as ReturnsVolaGridRiskStateGenerator.
        return DataRequest(portfolio.instruments, day-timedelta(days=2*self.ew_window),
                           day, "closePrices", Period.ONE_DAY, self.requestParameters)

    def acquire(self, portfolio, marginDate):
        return self.acquisition.acquire(self._request(portfolio, marginDate)).copy()

    def prepare(self, portfolio, dates):
        if not dates:
            raise ValueError("backtest dates must not be empty")
        first, last = self._request(portfolio, min(dates)), self._request(portfolio, max(dates))
        self.acquisition.acquire(first.withChanges(end_date=last.end_date))
