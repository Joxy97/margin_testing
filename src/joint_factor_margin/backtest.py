"""Adapt the application to existing backtest P&L, coverage and checkpoint logic."""

from portfolio import Portfolio
from margin_engine import MarginReport, MarginEngineTimings
from backtesting import MarginBacktester, BacktestBatchResults
from .values import ExposurePortfolio, CalculationRequest


class ApplicationBacktestEngine:
    """Implement BacktestMarginEngine without exposing numerical module internals."""

    def __init__(self, application):
        self.application = application

    def generateReport(self, portfolio: Portfolio, marginDate):
        result = self.application.calculate(CalculationRequest(ExposurePortfolio.fromPortfolio(portfolio), marginDate))
        timings = result.timings
        return MarginReport(result.margin, MarginEngineTimings(
            dataAcquisitionSeconds=timings["dataAcquisitionSeconds"],
            riskStateGenerationSeconds=timings["scenarioConstructionSeconds"],
            marginCalculationSeconds=timings["totalSeconds"]-timings["dataAcquisitionSeconds"],
            totalSeconds=timings["totalSeconds"],
        ), numericalDiagnostics={"configurationIdentity": result.configurationIdentity})

    def getPortfolioMarketData(self, portfolio, asOfDate):
        return self.application.marketData.acquire(ExposurePortfolio.fromPortfolio(portfolio), asOfDate)

    def prepareBacktest(self, portfolio, dates):
        self.application.marketData.prepare(ExposurePortfolio.fromPortfolio(portfolio), dates)


class BacktestRunner:
    """Replay stock exposures; optional persistence is supplied by the caller."""

    def __init__(self, backtester=None):
        self.backtester = backtester or MarginBacktester()

    def run(self, application, portfolio: ExposurePortfolio, config, *, checkpointStore=None, resume=False):
        if resume and checkpointStore is None:
            raise ValueError("resume requires a checkpoint store")
        if checkpointStore is not None and not resume:
            checkpointStore.startFresh(config.name)
        completed = checkpointStore.load(config.name) if resume else ()
        result = self.backtester.backtest(ApplicationBacktestEngine(application), portfolio.toPortfolio(),
            config.dates, config.confidenceLevel, completedResults=completed,
            onNewDay=None if checkpointStore is None else lambda day: checkpointStore.saveDay(config.name, day))
        return BacktestBatchResults({config.name: result})
