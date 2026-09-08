"""Option pricing and scenario-margin tests."""

from __future__ import annotations

from datetime import date
from math import exp
from pathlib import Path
import tempfile
import unittest

import yaml

from margin_calculator import (
    StateAwareGreedyMarginCalculator,
    StateAwareGreedyMarginCalculatorConfig,
)
from margin_engine import MarginApplicationConfig
from option_pricing import (
    AmericanEquityBinomialPricingModel,
    Black76PricingModel,
    EquityBlackScholesPricingModel,
    impliedVolatility,
)
from portfolio import DerivativesPortfolio
from risk_state_generator import OptionScenarioRiskStateGeneratorConfig


class OptionPricingTest(unittest.TestCase):
    def test_contract_and_quote_share_canonical_identity(self) -> None:
        import pandas
        from decimal import Decimal
        from portfolio.derivatives import FuturesOptionContract, contractKey
        from option_pricing.conventions import FuturesOptionMarketConvention
        contract = FuturesOptionContract("ES", date(2025, 6, 20), Decimal("100.00"), "c")
        row = next(pandas.DataFrame({"symbol": ["ES"],
            "expiration_date": [pandas.Timestamp("2025-06-20")], "strike": [100.],
            "option_type": ["C"], "exercise_style": ["E"]}).itertuples(index=False))
        key = contractKey(contract)
        self.assertEqual(key, FuturesOptionMarketConvention().optionMarketPriceKey(row))
        self.assertEqual((key.symbol, key.strike), ("ES", "100"))

    def test_unmatched_shock_history_reports_fallback(self) -> None:
        import pandas
        from option_pricing.calibration import VolatilityShockEstimator

        quotes = pandas.DataFrame({
            "date": pandas.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
            "symbol": "A", "instrument_type": "equity", "price": [100, 102, 101],
        })
        result = VolatilityShockEstimator(minimumObservations=2, fallbackRho=-.6).estimate(
            quotes.iloc[::-1], "A", "equity_option",
            {date(2024, 1, day): value for day, value in ((2, .2), (3, .21), (5, .19))}, .2)
        self.assertEqual((result.rho, result.matchedIntervals, result.correlationFallback),
                         (-.6, 1, True))

    def test_shock_correlation_matches_both_interval_endpoints(self) -> None:
        import pandas
        from option_pricing.calibration import VolatilityShockEstimator

        quotes = pandas.DataFrame({
            "date": pandas.date_range("2024-01-01", periods=7),
            "symbol": "A", "instrument_type": "equity",
            "price": [100 * exp(x) for x in (0, .01, .03, .06, .10, .15, .21)],
        })
        history = {date(2024, 1, day): .2 * exp(value)
                   for day, value in ((1, 0), (2, .02), (4, .9), (5, .98), (6, 1.08), (7, 1.2))}
        result = VolatilityShockEstimator(minimumObservations=3).estimate(
            quotes, "A", "equity_option", history, .2)

        # Matched Jan 1–2, 4–5, 5–6, 6–7 changes are exactly twice price returns.
        self.assertAlmostEqual(result.rho, 1.0)

    def test_black76_put_call_parity(self) -> None:
        model = Black76PricingModel()
        call = model.price(100.0, 95.0, 0.5, 0.04, 0.20, "C")
        put = model.price(100.0, 95.0, 0.5, 0.04, 0.20, "P")
        self.assertAlmostEqual(call - put, exp(-0.04 * 0.5) * 5.0, places=10)

    def test_equity_implied_volatility_round_trip(self) -> None:
        class CountingModel(EquityBlackScholesPricingModel):
            def __init__(self) -> None:
                self.priceCalls = 0

            def price(self, *args, **kwargs) -> float:
                self.priceCalls += 1
                return super().price(*args, **kwargs)

        model = CountingModel()
        price = model.price(100.0, 105.0, 0.75, 0.03, 0.28, "P", 0.01)
        model.priceCalls = 0
        result = impliedVolatility(
            model, price, 100.0, 105.0, 0.75, 0.03, "P", 0.01
        )
        self.assertAlmostEqual(result, 0.28, places=8)
        self.assertLess(model.priceCalls, 15)

    def test_american_equity_put_is_not_cheaper_than_european(self) -> None:
        european = EquityBlackScholesPricingModel().price(
            100.0, 110.0, 1.0, 0.05, 0.20, "P"
        )
        american = AmericanEquityBinomialPricingModel(300).price(
            100.0, 110.0, 1.0, 0.05, 0.20, "P"
        )
        self.assertGreaterEqual(american, european)


class OptionMarginApplicationTest(unittest.TestCase):
    def test_lazy_scenarios_share_an_immutable_prepared_market(self) -> None:
        import pandas
        from download_unit import DataRequest
        from risk_state_generator import RiskStateGenerationContext
        from risk_state_generator.option_scenario_risk_state_generator import OptionScenarioRiskStateGenerator
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.csv"
            self._writeQuotes(path)
            quotes = pandas.read_csv(path)
        context = RiskStateGenerationContext(quotes, DataRequest(instruments=["ES"],
            start_date=date(2025, 1, 1), end_date=date(2025, 1, 10), data_type="derivativeQuotes"),
            date(2025, 1, 10))
        states = OptionScenarioRiskStateGenerator().getRiskStates(context)
        first = next(states)
        prices = dict(first.marketPrices)
        quotes.loc[:, "price"] = 0.
        second = next(states)
        self.assertIs(first.preparedMarket, second.preparedMarket)
        self.assertEqual(dict(second.marketPrices), prices)
        with self.assertRaises(TypeError):
            second.marketPrices[next(iter(prices))] = 0.
        states.close()

    def test_option_preparation_rejects_conflicting_duplicate_quotes(self) -> None:
        import pandas
        from download_unit import DataRequest
        from risk_state_generator import RiskStateGenerationContext
        from risk_state_generator.option_scenario_risk_state_generator import OptionScenarioRiskStateGenerator
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.csv"
            self._writeQuotes(path)
            quotes = pandas.read_csv(path)
        duplicate = quotes.iloc[[0]].copy()
        duplicate["price"] += 1.
        quotes = pandas.concat([quotes, duplicate], ignore_index=True)
        context = RiskStateGenerationContext(quotes, DataRequest(instruments=["ES"],
            start_date=date(2025, 1, 1), end_date=date(2025, 1, 10), data_type="derivativeQuotes"),
            date(2025, 1, 10))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            list(OptionScenarioRiskStateGenerator().getRiskStates(context))

    def test_option_preparation_rejects_future_quotes(self) -> None:
        import pandas
        from download_unit import DataRequest
        from risk_state_generator import RiskStateGenerationContext
        from risk_state_generator.option_scenario_risk_state_generator import OptionScenarioRiskStateGenerator
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.csv"
            self._writeQuotes(path)
            quotes = pandas.read_csv(path)
        context = RiskStateGenerationContext(quotes, DataRequest(instruments=["ES"],
            start_date=date(2025, 1, 1), end_date=date(2025, 1, 9), data_type="derivativeQuotes"),
            date(2025, 1, 9))
        with self.assertRaisesRegex(ValueError, "future"):
            list(OptionScenarioRiskStateGenerator().getRiskStates(context))

    def test_yaml_application_margins_a_futures_option_portfolio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quote_path = root / "quotes.csv"
            self._writeQuotes(quote_path)
            config_path = root / "margin.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2025-01-10",
                        "portfolio": {
                            "positions": [
                                {
                                    "instrumentType": "future",
                                    "symbol": "ES",
                                    "expirationDate": "2025-06-20",
                                    "quantity": 1,
                                    "multiplier": 50,
                                },
                                {
                                    "instrumentType": "futures_option",
                                    "symbol": "ES",
                                    "expirationDate": "2025-06-20",
                                    "strike": 100,
                                    "optionType": "C",
                                    "exerciseStyle": "E",
                                    "quantity": -1,
                                    "multiplier": 50,
                                },
                            ]
                        },
                        "engine": {
                            "downloadManager": {
                                "providers": {"quotes": "derivative_csv"},
                                "requestParameters": {"location": "quotes.csv"},
                            },
                            "dataManager": {"type": "derivative_quotes"},
                            "riskStateGenerator": {
                                "type": "option_scenarios",
                                "historyDays": 10,
                                "priceScenarioSteps": 5,
                                "volatilityShifts": [-0.02, 0, 0.02],
                            },
                            "marginCalculator": {
                                "type": "state_aware_greedy",
                                "pnlAnchor": "market",
                            },
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(config_path)
            report = application.generateReport()

        self.assertIsInstance(application.portfolio, DerivativesPortfolio)
        self.assertIsInstance(
            application.engine.riskStateGenerator,
            OptionScenarioRiskStateGeneratorConfig,
        )
        self.assertIsInstance(
            application.engine.marginCalculator,
            StateAwareGreedyMarginCalculatorConfig,
        )
        calculator = application.engine.marginCalculator.createMarginCalculator()
        self.assertIsInstance(calculator, StateAwareGreedyMarginCalculator)
        self.assertGreater(report.margin, 0.0)

    def test_yaml_application_margins_an_equity_option_portfolio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quote_path = root / "quotes.csv"
            model = EquityBlackScholesPricingModel()
            expiry = date(2025, 6, 20)
            time = (expiry - date(2025, 1, 10)).days / 365.0
            rows = [
                "date,symbol,instrument_type,expiration_date,strike,option_type,"
                "exercise_style,multiplier,price,dividend_yield",
                "2025-01-08,AAPL,equity,2025-01-08,0,,,1,98,0.01",
                "2025-01-09,AAPL,equity,2025-01-09,0,,,1,99,0.01",
                "2025-01-10,AAPL,equity,2025-01-10,0,,,1,100,0.01",
            ]
            for strike, volatility in ((90, 0.24), (100, 0.20), (110, 0.22)):
                price = model.price(
                    100.0, strike, time, 0.04, volatility, "C", 0.01
                )
                rows.append(
                    f"2025-01-10,AAPL,equity_option,2025-06-20,{strike},C,E,1,{price},0.01"
                )
            quote_path.write_text("\n".join(rows), encoding="utf-8")
            config_path = root / "margin.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2025-01-10",
                        "portfolio": {
                            "positions": [
                                {
                                    "instrumentType": "equity",
                                    "symbol": "AAPL",
                                    "quantity": 100,
                                },
                                {
                                    "instrumentType": "equity_option",
                                    "symbol": "AAPL",
                                    "expirationDate": "2025-06-20",
                                    "strike": 100,
                                    "optionType": "C",
                                    "quantity": -1,
                                    "multiplier": 100,
                                    "dividendYield": 0.01,
                                },
                            ]
                        },
                        "engine": {
                            "downloadManager": {
                                "providers": {"quotes": "derivative_csv"},
                                "requestParameters": {"location": "quotes.csv"},
                            },
                            "dataManager": {"type": "derivative_quotes"},
                            "riskStateGenerator": {
                                "type": "option_scenarios",
                                "historyDays": 10,
                                "priceScenarioSteps": 5,
                            },
                            "marginCalculator": {
                                "type": "state_aware_greedy"
                            },
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            report = MarginApplicationConfig.fromYaml(config_path).generateReport()

        self.assertGreater(report.margin, 0.0)

    @staticmethod
    def _writeQuotes(path: Path) -> None:
        model = Black76PricingModel()
        valuation = date(2025, 1, 10)
        expiry = date(2025, 6, 20)
        time = (expiry - valuation).days / 365.0
        rows = [
            "date,symbol,instrument_type,expiration_date,strike,option_type,"
            "exercise_style,multiplier,price"
        ]
        for day, price in (("2025-01-08", 98), ("2025-01-09", 99), ("2025-01-10", 100)):
            rows.append(f"{day},ES,future,2025-06-20,0,,,50,{price}")
        for strike, volatility in ((90, 0.24), (100, 0.20), (110, 0.22)):
            price = model.price(100.0, strike, time, 0.04, volatility, "C")
            rows.append(
                f"2025-01-10,ES,futures_option,2025-06-20,{strike},C,E,50,{price}"
            )
        path.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
