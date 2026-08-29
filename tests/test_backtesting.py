"""Tests for rolling portfolio margin backtesting."""

import unittest
import csv
import tempfile
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import Mock
from pathlib import Path

import pandas
from scipy.stats import binom

from backtesting import (
    BacktestBatchResults,
    BacktestCSVReporter,
    BaselColor,
    MarginBacktester,
    PortfolioBacktestRequest,
)
from margin_engine import MarginEngine, MarginEngineConfig, MarginReport
from portfolio import Portfolio


class MarginBacktesterTest(unittest.TestCase):
    @staticmethod
    def _engine(
        reports: dict[date, MarginReport],
        realizedPnLs: dict[date, float],
    ) -> MarginEngine:
        engine = MarginEngine(MarginEngineConfig())
        engine.generateReport = Mock(
            side_effect=lambda _portfolio, backtest_date: reports[backtest_date]
        )
        def market_data(portfolio: Portfolio, backtest_date: date) -> pandas.DataFrame:
            previous_date = backtest_date - timedelta(days=1)
            if not portfolio.weights:
                return pandas.DataFrame(
                    {"date": pandas.to_datetime([previous_date, backtest_date])}
                )
            instrument = next(iter(portfolio.weights))
            position = float(portfolio.weights[instrument])
            realized_return = realizedPnLs[backtest_date] / position
            return pandas.DataFrame(
                {
                    "date": pandas.to_datetime([previous_date, backtest_date]),
                    instrument: [100.0, 100.0 * (1.0 + realized_return)],
                }
            )

        engine.getPortfolioMarketData = Mock(side_effect=market_data)
        engine.prepareBacktest = Mock()
        return engine

    def test_records_daily_margin_pnl_exposure_and_margin_percent(self) -> None:
        dates = [date(2024, 1, 3), date(2024, 1, 2)]
        reports = {
            date(2024, 1, 2): MarginReport(1.0),
            date(2024, 1, 3): MarginReport(2.0),
        }
        realized_pnls = {
            date(2024, 1, 2): -1.5,
            date(2024, 1, 3): 0.5,
        }
        portfolio = Portfolio(weights={"AAPL": Decimal("10")})

        result = MarginBacktester().backtest(
            self._engine(reports, realized_pnls),
            portfolio,
            dates,
        )

        self.assertEqual(
            tuple(item.date for item in result.dailyResults),
            (date(2024, 1, 2), date(2024, 1, 3)),
        )
        self.assertEqual(result.dailyResults[0].marginPercent, 10.0)
        self.assertAlmostEqual(result.dailyResults[0].realizedPnL, -1.5)
        self.assertAlmostEqual(result.dailyResults[0].realizedLoss, 1.5)
        self.assertTrue(result.dailyResults[0].breach)
        self.assertFalse(result.dailyResults[0].covered)
        self.assertFalse(result.dailyResults[1].breach)
        self.assertEqual(result.violations, 1)
        engine = self._engine(reports, realized_pnls)
        MarginBacktester().backtest(engine, portfolio, dates)
        engine.prepareBacktest.assert_called_once_with(
            portfolio,
            (date(2024, 1, 2), date(2024, 1, 3)),
        )

    def test_basel_color_matches_marginlab_binomial_rule(self) -> None:
        start = date(2024, 1, 1)
        dates = tuple(start + timedelta(days=offset) for offset in range(250))
        reports = {
            backtest_date: MarginReport(margin=1.0)
            for index, backtest_date in enumerate(dates)
        }
        realized_pnls = {
            backtest_date: -2.0 if index < 5 else 0.0
            for index, backtest_date in enumerate(dates)
        }

        result = MarginBacktester().backtest(
            self._engine(reports, realized_pnls),
            Portfolio(weights={"AAPL": Decimal("10")}),
            dates,
        )

        self.assertEqual(result.violations, 5)
        self.assertAlmostEqual(
            result.baselProbability,
            float(binom.cdf(5, 250, 0.002)),
        )
        self.assertEqual(result.baselColor, BaselColor.RED)

    def test_basel_color_uses_marginlab_threshold_boundaries(self) -> None:
        self.assertEqual(
            MarginBacktester._baselColor(0.95),
            BaselColor.GREEN,
        )
        self.assertEqual(
            MarginBacktester._baselColor(0.950001),
            BaselColor.YELLOW,
        )
        self.assertEqual(
            MarginBacktester._baselColor(0.9999),
            BaselColor.RED,
        )

    def test_backtests_multiple_named_portfolios_and_dates(self) -> None:
        first_date = date(2024, 1, 2)
        second_date = date(2024, 1, 3)
        engine = self._engine(
            {
                first_date: MarginReport(1.0),
                second_date: MarginReport(2.0),
            },
            {first_date: -0.5, second_date: -1.0},
        )

        results = MarginBacktester().backtestMany(
            engine,
            {
                "client-a": PortfolioBacktestRequest(
                    Portfolio(weights={"AAPL": Decimal("10")}),
                    (first_date,),
                ),
                "client-b": PortfolioBacktestRequest(
                    Portfolio(weights={"AAPL": Decimal("20")}),
                    (second_date,),
                ),
            },
        )

        self.assertIsInstance(results, BacktestBatchResults)
        self.assertEqual(set(results.results), {"client-a", "client-b"})
        self.assertEqual(results.results["client-a"].days, 1)
        self.assertEqual(results.results["client-b"].days, 1)

    def test_rejects_zero_gross_exposure(self) -> None:
        backtest_date = date(2024, 1, 2)

        with self.assertRaisesRegex(ValueError, "gross exposure"):
            MarginBacktester().backtest(
                self._engine(
                    {backtest_date: MarginReport(1.0)},
                    {backtest_date: 0.0},
                ),
                Portfolio(),
                [backtest_date],
            )

    def test_calculates_realized_pnl_from_backtest_prices(self) -> None:
        portfolio = Portfolio(
            weights={"AAPL": Decimal("10"), "MSFT": Decimal("-5")},
            cash=Decimal("100"),
        )
        prices = pandas.DataFrame(
            {
                "date": pandas.to_datetime(["2024-01-02", "2024-01-03"]),
                "AAPL": [100.0, 110.0],
                "MSFT": [50.0, 45.0],
            }
        )

        pnl = MarginBacktester._calculateRealizedPnL(
            portfolio,
            date(2024, 1, 3),
            prices,
        )

        self.assertAlmostEqual(pnl, 1.5)

    def test_writes_separate_breach_and_performance_csv_reports(self) -> None:
        backtest_date = date(2024, 1, 2)
        results = MarginBacktester().backtestMany(
            self._engine(
                {backtest_date: MarginReport(1.0)},
                {backtest_date: -2.0},
            ),
            {
                "client/a": PortfolioBacktestRequest(
                    Portfolio(weights={"AAPL": Decimal("10")}),
                    (backtest_date,),
                )
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            files = BacktestCSVReporter().write(results, directory)["client/a"]
            with files.breaches.open(newline="", encoding="utf-8") as stream:
                breaches = list(csv.DictReader(stream))
            with files.performanceMetrics.open(
                newline="", encoding="utf-8"
            ) as stream:
                performance = list(csv.DictReader(stream))

            self.assertEqual(files.breaches.parent, Path(directory) / "client_a")
            self.assertEqual(breaches[0]["breach"], "True")
            self.assertEqual(
                {row["stage"] for row in performance},
                {
                    "data_acquisition",
                    "risk_state_generation",
                    "margin_calculation",
                    "realized_data_acquisition",
                    "realized_pnl_calculation",
                    "total",
                },
            )


if __name__ == "__main__":
    unittest.main()
