"""Orchestrate data acquisition, risk generation, and margin calculation."""

from __future__ import annotations

from datetime import date

from portfolio import Portfolio
from risk_state_generator import RiskStateGenerationContext

from .config import MarginEngineConfig
from .margin_report import MarginReport


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

        generation_context = RiskStateGenerationContext(
            marketData=data,
            dataRequest=request,
            marginDate=marginDate,
        )
        risk_states = self.riskStateGenerator.getRiskStates(generation_context)
        margin = self.marginCalculator.calculateMargin(risk_states, portfolio)
        return MarginReport(float(margin))
