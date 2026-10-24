"""The calculation orchestrator; numerical behavior lives in injected modules."""

from time import perf_counter
from hashlib import sha256
import json

from data_manager.acquisition import MarketDataAcquisition
from .market_data import AcquiredMarketData, MarketData
from .scenarios import ExistingScenarioBuilder, ScenarioBuilder, equityGeometry, valuationDiagnostics
from .search import SingleAnchorSearch, BoundaryAwareSearch, StressSearch
from .diagnostics import DiagnosticEvaluator, ExistingDiagnostics
from .settings import EngineConfig
from .values import CalculationRequest, CalculationResult, primitive


class MarginApplication:
    """Run an independent pipeline without writing artifacts or altering inputs."""

    def __init__(self, marketData: MarketData, scenarios: ScenarioBuilder,
                 search: StressSearch, diagnostics: DiagnosticEvaluator,
                 configurationIdentity: str, radius: float = 3.):
        self.marketData, self.scenarios, self.search = marketData, scenarios, search
        self.diagnostics = diagnostics
        self.configurationIdentity = configurationIdentity
        self.radius = radius

    def calculate(self, request: CalculationRequest) -> CalculationResult:
        if not isinstance(request, CalculationRequest):
            raise TypeError("request must be CalculationRequest")
        timings = {}
        total = perf_counter()
        started = perf_counter()
        data = self.marketData.acquire(request.portfolio, request.marginDate)
        timings["dataAcquisitionSeconds"] = perf_counter()-started
        started = perf_counter()
        scenario = self.scenarios.build(request, data)
        timings["scenarioConstructionSeconds"] = perf_counter()-started
        started = perf_counter()
        selected = self.search.search(scenario.model)
        timings["searchSeconds"] = perf_counter()-started
        started = perf_counter()
        diagnostics = self.diagnostics.evaluate(scenario.model, selected)
        timings["diagnosticsSeconds"] = perf_counter()-started
        geometry = equityGeometry(scenario.model)
        log_returns = geometry.center+geometry.directions@selected.coordinates
        diagnostics.update(search=selected.diagnostics, calibrationStart=scenario.calibrationStart,
                           calibrationEnd=scenario.calibrationEnd, volatility=scenario.overlay,
                           valuation=valuationDiagnostics(scenario.model, self.radius))
        identity = sha256((self.configurationIdentity+json.dumps(primitive(request),
            sort_keys=True, allow_nan=False)).encode()).hexdigest()
        timings["totalSeconds"] = perf_counter()-total
        return CalculationResult(request.marginDate, max(0., -selected.pnl), selected.pnl,
            selected.coordinates, geometry.instruments, log_returns, selected.quadraticPnL,
            diagnostics, timings, identity)


def createApplication(config: EngineConfig, configurationIdentity: str) -> MarginApplication:
    """Composition root: construct each runtime dependency once per application."""
    if not isinstance(config, EngineConfig):
        raise TypeError("config must be EngineConfig")
    solver = config.solver.createBQMSolver()
    acquisition = MarketDataAcquisition(config.dataManager.createDataManager(),
                                        config.downloadManager.createDownloadManager())
    market = AcquiredMarketData(acquisition, config.estimation.ew_window,
                                config.downloadManager.requestParameters)
    scenarios = ExistingScenarioBuilder(config.estimation,
        config.estimation.pcaGridProvider.createProvider(), config.volatility)
    if config.search.type == "boundary_aware":
        search = BoundaryAwareSearch(solver, config.encoding, config.search.repair,
                                      config.solver.solverParameters, config.executionPolicy)
    else:
        search = SingleAnchorSearch(solver, config.encoding, config.search.repair,
            config.search.repairEnabled, config.solver.solverParameters, config.executionPolicy)
    return MarginApplication(market, scenarios, search,
                             ExistingDiagnostics(config.diagnostics, config.encoding), configurationIdentity,
                             config.encoding.radius)
