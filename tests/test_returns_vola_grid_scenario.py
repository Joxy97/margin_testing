"""Tests for returns-volatility-grid scenarios."""

import unittest
from unittest.mock import Mock

import numpy

from scenario_generator import (
    BQMModelGenerator,
    CorrelatedReturnsVolaGridScenario,
    ReturnsVolaGridScenario,
    Scenario,
)


class ReturnsVolaGridScenarioTest(unittest.TestCase):
    def test_is_a_scenario_and_dispatches_itself_to_generator(self) -> None:
        grids = {
            "AAPL": numpy.array([[0.01, 0.20], [0.02, 0.25]]),
            "MSFT": numpy.array([[0.03, 0.15]]),
        }
        scenario = ReturnsVolaGridScenario(grids)
        generator = Mock()

        result = scenario.accept(generator)

        self.assertIsInstance(scenario, Scenario)
        self.assertIs(scenario.returnsVolaGrid, grids)
        generator.createReturnsVolaGridBQM.assert_called_once_with(scenario)
        self.assertIsNone(result)

    def test_correlated_scenario_dispatches_its_correlations(self) -> None:
        grids = {"AAPL": numpy.array([[0.01, 0.20]])}
        correlations = {("AAPL", "MSFT"): 0.75}
        scenario = CorrelatedReturnsVolaGridScenario(grids, correlations)
        generator = Mock()

        result = scenario.accept(generator)

        self.assertIsInstance(scenario, ReturnsVolaGridScenario)
        self.assertIs(scenario.returnsVolaGrid, grids)
        self.assertIs(scenario.correlations, correlations)
        generator.createCorrelatedReturnsVolaGridBQM.assert_called_once_with(
            scenario,
            correlations,
        )
        self.assertIsNone(result)

    def test_bqm_generator_methods_are_empty(self) -> None:
        generator = BQMModelGenerator()
        grids = {"AAPL": numpy.array([[0.01, 0.20]])}
        correlations = {("AAPL", "MSFT"): 0.75}
        scenario = ReturnsVolaGridScenario(grids)
        correlated_scenario = CorrelatedReturnsVolaGridScenario(
            grids,
            correlations,
        )

        self.assertIsNone(generator.createReturnsVolaGridBQM(scenario))
        self.assertIsNone(
            generator.createCorrelatedReturnsVolaGridBQM(
                correlated_scenario,
                correlations,
            )
        )

    def test_scenario_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Scenario()


if __name__ == "__main__":
    unittest.main()
