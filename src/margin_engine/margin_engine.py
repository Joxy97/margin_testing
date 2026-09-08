"""Orchestrate data acquisition, risk generation, and margin calculation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from time import perf_counter
from typing import TYPE_CHECKING

from download_unit import DataRequest
from data_manager.acquisition import MarketDataAcquisition
from portfolio import Portfolio
from risk_state_generator import RiskStateGenerationContext

from .config import MarginEngineConfig
from .margin_report import MarginReport
from .calculation_measurements import CalculationMeasurements

if TYPE_CHECKING:
    import pandas


class MarginEngine:
    """Coordinate one independently configured margin-calculation pipeline."""

    def __init__(self, configs: MarginEngineConfig) -> None:
        if not isinstance(configs, MarginEngineConfig):
            raise TypeError("configs must be a MarginEngineConfig")
        self.configs = configs
        self.downloadManager = configs.downloadManager.createDownloadManager()
        self.dataManager = configs.dataManager.createDataManager()
        self.acquisition = MarketDataAcquisition(self.dataManager, self.downloadManager)
        self.riskStateGenerator = (
            configs.riskStateGenerator.createRiskStateGenerator()
        )
        self.marginCalculator = configs.marginCalculator.createMarginCalculator()
        self.numericalExecution = None if configs.numericalExecution is None else configs.numericalExecution.createExecution(
            self.riskStateGenerator, self.marginCalculator)

    def generateReport(
        self,
        portfolio: Portfolio,
        marginDate: date,
    ) -> MarginReport:
        """Acquire required data and calculate portfolio margin."""
        measurements = CalculationMeasurements(perf_counter)
        request, data = measurements.measure("dataAcquisitionSeconds", lambda: self._acquireMarketData(portfolio, marginDate))
        generation_context = RiskStateGenerationContext(
            marketData=data, dataRequest=request, marginDate=marginDate)
        if self.numericalExecution is not None:
            outcome = measurements.measure("marginCalculationSeconds", lambda:
                self.numericalExecution.calculate(generation_context, portfolio, measurements))
        else:
            risk_states = measurements.iterate(self.riskStateGenerator.getRiskStates(generation_context))
            try:
                outcome = measurements.measure("marginCalculationSeconds", lambda:
                    self.marginCalculator.calculateOutcome(risk_states, portfolio))
            finally:
                risk_states.close()
        return MarginReport(
            margin=float(outcome.margin),
            timings=measurements.timings(),
            comparisonMargins=outcome.comparisonMargins,
            numericalDiagnostics=outcome.numericalDiagnostics,
        )

    def prepareBacktest(
        self,
        portfolio: Portfolio,
        dates: Sequence[date],
    ) -> None:
        """Prefetch the union of market data needed by a rolling backtest."""
        backtest_dates = tuple(dates)
        if not backtest_dates:
            raise ValueError("dates must not be empty")
        if any(not isinstance(item, date) for item in backtest_dates):
            raise TypeError("dates must contain date objects")
        first_date = min(backtest_dates)
        last_date = max(backtest_dates)
        first_request = self.riskStateGenerator.createDataRequest(
            portfolio,
            first_date,
        )
        last_request = self.riskStateGenerator.createDataRequest(
            portfolio,
            last_date,
        )
        if (
            first_request.instruments != last_request.instruments
            or first_request.data_type != last_request.data_type
            or first_request.period != last_request.period
        ):
            raise ValueError(
                "risk-state generator produced incompatible backtest requests"
            )
        request = first_request.withChanges(
            start_date=min(first_request.start_date, last_request.start_date),
            end_date=max(first_request.end_date, last_request.end_date),
        ).withProviderParameters(
            self.configs.downloadManager.requestParameters
        )
        self.acquisition.acquire(request)

    def getPortfolioMarketData(
        self,
        portfolio: Portfolio,
        asOfDate: date,
    ) -> pandas.DataFrame:
        """Return acquired market data without exposing storage internals."""
        _, data = self._acquireMarketData(portfolio, asOfDate)
        return data.copy()

    def _acquireMarketData(
        self,
        portfolio: Portfolio,
        marginDate: date,
    ) -> tuple[DataRequest, pandas.DataFrame]:
        """Return request-shaped market data, using cache and downloads."""
        request = self.riskStateGenerator.createDataRequest(
            portfolio,
            marginDate,
        ).withProviderParameters(
            self.configs.downloadManager.requestParameters
        )
        return request, self.acquisition.acquire(request)
