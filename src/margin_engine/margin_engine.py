"""Orchestrate data acquisition, risk generation, and margin calculation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from time import perf_counter
from typing import TYPE_CHECKING

from download_unit import DataRequest
from portfolio import Portfolio
from risk_state_generator import RiskStateGenerationContext

from .config import MarginEngineConfig
from .margin_report import MarginEngineTimings, MarginReport

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
        self.riskStateGenerator = (
            configs.riskStateGenerator.createRiskStateGenerator()
        )
        self.marginCalculator = configs.marginCalculator.createMarginCalculator()

    def generateReport(
        self,
        portfolio: Portfolio,
        marginDate: date,
    ) -> MarginReport:
        """Acquire required data and calculate portfolio margin."""
        total_started = perf_counter()
        acquisition_started = perf_counter()
        request, data = self._acquireMarketData(portfolio, marginDate)
        acquisition_seconds = perf_counter() - acquisition_started
        generation_context = RiskStateGenerationContext(
            marketData=data,
            dataRequest=request,
            marginDate=marginDate,
        )
        risk_states = self.riskStateGenerator.getRiskStates(generation_context)
        generation_seconds = [0.0]

        def timedRiskStates():
            iterator = iter(risk_states)
            while True:
                generation_started = perf_counter()
                try:
                    risk_state = next(iterator)
                except StopIteration:
                    generation_seconds[0] += perf_counter() - generation_started
                    return
                generation_seconds[0] += perf_counter() - generation_started
                yield risk_state

        calculation_started = perf_counter()
        margin = self.marginCalculator.calculateMargin(
            timedRiskStates(),
            portfolio,
        )
        combined_seconds = perf_counter() - calculation_started
        total_seconds = perf_counter() - total_started
        return MarginReport(
            margin=float(margin),
            timings=MarginEngineTimings(
                dataAcquisitionSeconds=acquisition_seconds,
                riskStateGenerationSeconds=generation_seconds[0],
                marginCalculationSeconds=max(
                    0.0,
                    combined_seconds - generation_seconds[0],
                ),
                totalSeconds=total_seconds,
            ),
            comparisonMargins=getattr(
                self.marginCalculator,
                "lastComparisonMargins",
                {},
            ),
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
        self._acquireRequest(request)

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
        return request, self._acquireRequest(request)

    def _acquireRequest(self, request: DataRequest) -> pandas.DataFrame:
        """Acquire one exact request through cache and configured providers."""
        data = self.dataManager.getData(request)
        if data is None:
            missing_requests = self.dataManager.getMissingRequests(request)
            for missing_request in missing_requests:
                downloaded_data = self.downloadManager.downloadDataType(
                    missing_request.data_type,
                    missing_request,
                )
                stored_data = self.dataManager.storeData(
                    missing_request,
                    downloaded_data,
                )
                if len(missing_requests) == 1 and missing_request == request:
                    data = stored_data
            if data is None:
                data = self.dataManager.getData(request)
        if data is None:
            raise RuntimeError(
                f"Unable to store downloaded {request.data_type}"
            )
        return data
