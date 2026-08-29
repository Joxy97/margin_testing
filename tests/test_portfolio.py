"""Tests for portfolio return-weight representation."""

import unittest
from decimal import Decimal

from portfolio import Portfolio


class PortfolioTest(unittest.TestCase):
    def test_stores_instrument_weights_and_cash(self) -> None:
        weights = {
            "AAPL": Decimal("0.6"),
            "MSFT": Decimal("-0.4"),
        }

        portfolio = Portfolio(weights=weights, cash=Decimal("1250.25"))

        self.assertIs(portfolio.weights, weights)
        self.assertEqual(portfolio.cash, Decimal("1250.25"))

    def test_defaults_to_independent_empty_weights_and_zero_cash(self) -> None:
        first = Portfolio()
        second = Portfolio()

        first.weights["AAPL"] = Decimal("1")

        self.assertEqual(first.cash, Decimal("0"))
        self.assertEqual(second.cash, Decimal("0"))
        self.assertEqual(second.weights, {})


if __name__ == "__main__":
    unittest.main()
