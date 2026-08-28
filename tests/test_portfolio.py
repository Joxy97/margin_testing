"""Tests for stock portfolio representation."""

import unittest
from decimal import Decimal

from portfolio import Portfolio


class PortfolioTest(unittest.TestCase):
    def test_stores_stock_positions_and_cash(self) -> None:
        positions = {
            "AAPL": Decimal("10.5"),
            "MSFT": Decimal("4"),
        }

        portfolio = Portfolio(positions=positions, cash=Decimal("1250.25"))

        self.assertIs(portfolio.positions, positions)
        self.assertEqual(portfolio.cash, Decimal("1250.25"))

    def test_defaults_to_independent_empty_positions_and_zero_cash(self) -> None:
        first = Portfolio()
        second = Portfolio()

        first.positions["AAPL"] = Decimal("1")

        self.assertEqual(first.cash, Decimal("0"))
        self.assertEqual(second.cash, Decimal("0"))
        self.assertEqual(second.positions, {})


if __name__ == "__main__":
    unittest.main()
