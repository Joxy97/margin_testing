"""Rolling portfolio margin coverage backtesting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from time import perf_counter
from typing import Protocol

import numpy
import pandas
from scipy.stats import binom

from margin_engine import MarginReport
from portfolio import Portfolio

from .backtest_results import (
    BacktestBatchResults,
    BacktestResults,
    BaselColor,
    DailyBacktestTimings,
    DailyBacktestResult,
    PortfolioBacktestRequest,
)


class BacktestMarginEngine(Protocol):
    """Public engine capabilities required by margin backtesting."""

    def generateReport(
        self,
        portfolio: Portfolio,
        marginDate: date,
    ) -> MarginReport: ...

    def getPortfolioMarketData(
        self,
        portfolio: Portfolio,
        asOfDate: date,
    ) -> pandas.DataFrame: ...

    def prepareBacktest(
        self,
        portfolio: Portfolio,
        dates: Sequence[date],
    ) -> None: ...


class MarginBacktester:
    """Evaluate daily margin coverage through the MarginEngine public API."""

    def backtest(
        self,
        marginEngine: BacktestMarginEngine,
        portfolio: Portfolio,
        dates: Sequence[date],
        confidenceLevel: float = 0.998,
    ) -> BacktestResults:
        """Backtest one portfolio over the supplied trade dates."""
        if not isinstance(portfolio, Portfolio):
            raise TypeError("portfolio must be a Portfolio")
        if not 0.0 < confidenceLevel < 1.0:
            raise ValueError("confidenceLevel must be between zero and one")
        supplied_dates = tuple(dates)
        if not supplied_dates:
            raise ValueError("dates must not be empty")
        if any(not isinstance(item, date) for item in supplied_dates):
            raise TypeError("dates must contain date objects")
        backtest_dates = tuple(sorted(supplied_dates))
        if len(backtest_dates) != len(set(backtest_dates)):
            raise ValueError("dates must not contain duplicates")

        preparation_started = perf_counter()
        marginEngine.prepareBacktest(portfolio, backtest_dates)
        preparation_seconds_per_day = (
            perf_counter() - preparation_started
        ) / len(backtest_dates)
        daily_results = tuple(
            self._backtestDay(
                marginEngine,
                portfolio,
                backtest_date,
                preparation_seconds_per_day,
            )
            for backtest_date in backtest_dates
        )
        violations = sum(result.breach for result in daily_results)
        basel_probability = float(
            binom.cdf(
                violations,
                len(daily_results),
                1.0 - confidenceLevel,
            )
        )
        return BacktestResults(
            portfolio=portfolio,
            dailyResults=daily_results,
            violations=violations,
            baselProbability=basel_probability,
            baselColor=self._baselColor(basel_probability),
            confidenceLevel=float(confidenceLevel),
        )

    def backtestMany(
        self,
        marginEngine: BacktestMarginEngine,
        requests: Mapping[str, PortfolioBacktestRequest],
        confidenceLevel: float = 0.998,
    ) -> BacktestBatchResults:
        """Backtest several named portfolios and their respective dates."""
        if not requests:
            raise ValueError("requests must not be empty")
        if any(
            not isinstance(request, PortfolioBacktestRequest)
            for request in requests.values()
        ):
            raise TypeError("requests must contain PortfolioBacktestRequest values")
        return BacktestBatchResults(
            {
                str(name): self.backtest(
                    marginEngine,
                    request.portfolio,
                    request.dates,
                    confidenceLevel,
                )
                for name, request in requests.items()
            }
        )

    @staticmethod
    def _backtestDay(
        marginEngine: BacktestMarginEngine,
        portfolio: Portfolio,
        backtestDate: date,
        preparationSeconds: float = 0.0,
    ) -> DailyBacktestResult:
        total_started = perf_counter()
        report = marginEngine.generateReport(portfolio, backtestDate)
        gross_exposure = float(
            sum(abs(float(position)) for position in portfolio.weights.values())
        )
        if not math.isfinite(gross_exposure) or gross_exposure <= 0.0:
            raise ValueError("portfolio gross exposure must be finite and positive")
        realized_data_started = perf_counter()
        market_data = marginEngine.getPortfolioMarketData(
            portfolio,
            backtestDate,
        )
        realized_data_seconds = perf_counter() - realized_data_started
        realized_pnl_started = perf_counter()
        realized_pnl = MarginBacktester._calculateRealizedPnL(
            portfolio,
            backtestDate,
            market_data,
        )
        realized_pnl_seconds = perf_counter() - realized_pnl_started
        if not math.isfinite(report.margin) or report.margin < 0.0:
            raise ValueError("margin must be finite and nonnegative")
        if not math.isfinite(realized_pnl):
            raise ValueError("realized PnL must be finite")
        realized_loss = -realized_pnl
        return DailyBacktestResult(
            date=backtestDate,
            margin=report.margin,
            realizedPnL=realized_pnl,
            grossExposure=gross_exposure,
            marginPercent=100.0 * report.margin / gross_exposure,
            breach=realized_loss > report.margin,
            timings=DailyBacktestTimings(
                dataAcquisitionSeconds=(
                    report.timings.dataAcquisitionSeconds
                    + preparationSeconds
                ),
                riskStateGenerationSeconds=(
                    report.timings.riskStateGenerationSeconds
                ),
                marginCalculationSeconds=(
                    report.timings.marginCalculationSeconds
                ),
                realizedDataAcquisitionSeconds=realized_data_seconds,
                realizedPnLCalculationSeconds=realized_pnl_seconds,
                totalSeconds=(
                    perf_counter() - total_started + preparationSeconds
                ),
            ),
        )

    @staticmethod
    def _calculateRealizedPnL(
        portfolio: Portfolio,
        backtestDate: date,
        marketData: pandas.DataFrame,
    ) -> float:
        """Calculate ex-post simple-return PnL from backtest market data."""
        if not isinstance(marketData, pandas.DataFrame):
            raise TypeError("marketData must be a pandas DataFrame")
        instruments = tuple(portfolio.weights)
        prices = marketData.copy()
        if "date" in prices.columns:
            prices["date"] = pandas.to_datetime(prices["date"], errors="raise")
            prices = prices.set_index("date")
        else:
            prices.index = pandas.to_datetime(prices.index, errors="raise")
        missing = set(instruments).difference(prices.columns)
        if missing:
            raise ValueError(f"marketData is missing instruments: {sorted(missing)}")
        prices = prices.sort_index().loc[:, list(instruments)].ffill().dropna()
        if not prices.index.is_unique:
            raise ValueError("market-data dates must be unique")
        current_date = pandas.Timestamp(backtestDate)
        if current_date not in prices.index:
            raise ValueError(f"No realized prices are available for {backtestDate}")
        previous_prices = prices.loc[prices.index < current_date]
        if previous_prices.empty:
            raise ValueError(f"No prior prices are available before {backtestDate}")
        current = prices.loc[current_date].to_numpy(dtype=float)
        previous = previous_prices.iloc[-1].to_numpy(dtype=float)
        if (
            not numpy.isfinite(current).all()
            or not numpy.isfinite(previous).all()
            or numpy.any(previous == 0.0)
        ):
            raise ValueError("prices used for realized PnL must be finite and nonzero")
        weights = numpy.asarray(
            [float(portfolio.weights[instrument]) for instrument in instruments],
            dtype=float,
        )
        return float(((current / previous) - 1.0) @ weights)

    @staticmethod
    def _baselColor(probability: float) -> BaselColor:
        if probability >= 0.9999:
            return BaselColor.RED
        if probability > 0.95:
            return BaselColor.YELLOW
        return BaselColor.GREEN
