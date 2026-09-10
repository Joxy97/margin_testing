"""Greedy reference preserves signed weights, bounds and loss conventions."""

from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

with patch.object(sys, "path", [str(Path(__file__).parents[1]/"tools"), *sys.path]):
    from run_risky_greedy_pca import marginsFromBounds
    from report_risky_factor_backtests import loadGreedy
from margin_calculator.state_aware_greedy_margin_calculator import StateAwareGreedyMarginCalculator
from portfolio import Portfolio
from risk_state_generator import DenseReturnsVolaGrid, ReturnsVolaGridRiskState


class RiskyGreedyPCATest(unittest.TestCase):
    def test_signed_portfolios_match_public_calculator(self):
        bounds = np.array([[[-.1, .2], [-.3, .4]], [[-.5, .1], [-.2, .8]]])
        weights = np.array([[1, 0], [0, -1], [.5, -.5], [0, 0]])
        states = [ReturnsVolaGridRiskState(DenseReturnsVolaGrid.fromMapping(
            {name: np.column_stack([asset, np.zeros(2)]) for name, asset in zip(("A", "B"), scenario)}))
            for scenario in bounds]
        margins, pnl = marginsFromBounds(bounds, weights)
        np.testing.assert_allclose(margins, [.5, .8, .65, 0])
        for i, w in enumerate(weights):
            portfolio = Portfolio({name: Decimal(str(value)) for name, value in zip(("A", "B"), w)})
            self.assertAlmostEqual(margins[i], StateAwareGreedyMarginCalculator().calculateMargin(states, portfolio))
        np.testing.assert_allclose(pnl[:, 2], [-.25, -.65])

    def test_all_gains_are_zero_margin(self):
        margin, _ = marginsFromBounds([[[.1, .2]]], [[1]])
        self.assertEqual(margin[0], 0)

    def test_invalid_or_nonfinite_bounds_rejected(self):
        for bounds, weights in (([[[.2, -.1]]], [[1]]), ([[[0, float("nan")]]], [[1]]),
                                (np.zeros((0, 1, 2)), [[1]]), (np.zeros((2, 2, 2)), [[1]])):
            with self.assertRaises(ValueError):
                marginsFromBounds(bounds, weights)

    def test_plot_baseline_requires_matched_dates_and_losses(self):
        primary = pd.DataFrame(dict(date=["2025-07-08", "2025-07-09"], solver=["combined"]*2,
                                    realized_loss=[0., .2]))
        baseline = pd.DataFrame(dict(date=primary.date, margin=[.1, .1], realized_loss=[0., .2], breach=[False, True]))
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self.assertIsNone(loadGreedy(directory, primary))
            baseline.to_csv(directory/"greedy_pca.csv", index=False)
            self.assertEqual(len(loadGreedy(directory, primary)), 2)
            baseline.loc[1, "date"] = "2025-07-10"
            baseline.to_csv(directory/"greedy_pca.csv", index=False)
            with self.assertRaisesRegex(ValueError, "dates"):
                loadGreedy(directory, primary)


if __name__ == "__main__":
    unittest.main()
