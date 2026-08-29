"""Orchestrate data acquisition, risk generation, and margin calculation."""

from __future__ import annotations

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
        )

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
        return request, data
