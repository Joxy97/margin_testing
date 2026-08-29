"""Tests for end-to-end margin orchestration."""

import tempfile
import unittest
from collections.abc import Iterable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

from data_manager import DataManager, DataManagerConfig
from download_manager import DownloadManager, DownloadManagerConfig
from download_unit import DataProvider, LocalCSVDataProvider
from margin_calculator import (
    BQMMarginCalculatorConfig,
    GreedyMarginCalculator,
    GreedyMarginCalculatorConfig,
    MarginCalculator,
)
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMSolverConfig,
)
from margin_engine import MarginEngine, MarginEngineConfig, MarginReport
from portfolio import Portfolio
from risk_state_generator import (
    ReturnsVolaGridRiskStateGenerator,
    ReturnsVolaGridRiskStateGeneratorConfig,
    RiskState,
)


class RecordingMarginCalculator(MarginCalculator):
    def __init__(self, margin: float) -> None:
        self.margin = margin
        self.calls: list[tuple[list[RiskState], Portfolio]] = []

    def calculateMargin(
        self,
        riskStates: Iterable[RiskState],
        portfolio: Portfolio,
    ) -> float:
        self.calls.append((list(riskStates), portfolio))
        return self.margin


class MarginEngineTest(unittest.TestCase):
    def test_generates_report_and_reuses_cached_close_prices(self) -> None:
        margin_date = date(2024, 1, 11)
        portfolio = Portfolio(weights={"AAPL": Decimal("1")})
        calculator = RecordingMarginCalculator(12.5)
        generator_config = ReturnsVolaGridRiskStateGeneratorConfig(
            ew_window=5,
            ew_lambda=0.94,
            components=1,
            scenariosPerComponents=(1,),
            nZBins=3,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            rows = ["date,AAPL"]
            for day in range(11):
                current_date = date(2024, 1, 1) + timedelta(days=day)
                rows.append(f"{current_date.isoformat()},{100 + day}")
            path.write_text("\n".join(rows), encoding="utf-8")
            provider = Mock(
                spec=DataProvider,
                wraps=LocalCSVDataProvider(),
            )
            engine = MarginEngine(
                MarginEngineConfig(
                    downloadManager=DownloadManagerConfig(
                        providers={"localCSV": provider},
                        requestParameters={"location": str(path)},
                    ),
                    dataManager=DataManagerConfig(memorySize=2),
                    riskStateGenerator=generator_config,
                    marginCalculator=BQMMarginCalculatorConfig(
                        solver=BQMSolverConfig(solverType="random")
                    ),
                )
            )
            engine.marginCalculator = calculator

            first_report = engine.generateReport(portfolio, margin_date)
            second_report = engine.generateReport(portfolio, margin_date)
            market_data = engine.getPortfolioMarketData(portfolio, margin_date)

        self.assertEqual(first_report.margin, 12.5)
        self.assertGreaterEqual(first_report.timings.totalSeconds, 0.0)
        self.assertEqual(second_report.margin, 12.5)
        self.assertEqual(market_data["AAPL"].iloc[-1], 110)
        self.assertEqual(provider.downloadData.call_count, 1)
        self.assertEqual(len(calculator.calls), 2)
        self.assertTrue(calculator.calls[0][0])
        self.assertIs(calculator.calls[0][1], portfolio)
        generator = engine.riskStateGenerator
        self.assertFalse(hasattr(generator, "pcaKey"))
        self.assertEqual(generator.ew_window, 5)
        self.assertFalse(hasattr(generator, "dataRequirements"))

    def test_constructs_components_from_configuration(self) -> None:
        engine = MarginEngine(
            MarginEngineConfig(
                dataManager=DataManagerConfig(
                    cacheType="lru",
                    memorySize=2,
                ),
                riskStateGenerator=ReturnsVolaGridRiskStateGeneratorConfig(
                    ew_window=5,
                    ew_lambda=0.94,
                    components=1,
                    scenariosPerComponents=(1,),
                ),
                marginCalculator=BQMMarginCalculatorConfig(
                    solver=BQMSolverConfig(solverType="random")
                ),
            )
        )

        self.assertIsInstance(engine.downloadManager, DownloadManager)
        self.assertIsInstance(engine.dataManager, DataManager)
        self.assertIsInstance(
            engine.riskStateGenerator,
            ReturnsVolaGridRiskStateGenerator,
        )

    def test_constructs_a_greedy_margin_calculator_from_configuration(self) -> None:
        engine = MarginEngine(
            MarginEngineConfig(
                marginCalculator=GreedyMarginCalculatorConfig(),
            )
        )

        self.assertIsInstance(engine.marginCalculator, GreedyMarginCalculator)


if __name__ == "__main__":
    unittest.main()
