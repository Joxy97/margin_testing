"""Tests for returns-volatility-grid risk states."""

import unittest
from datetime import date
from unittest.mock import Mock

import numpy

from download_unit import Period, UnifiedFormatCommand
from portfolio import Portfolio
from scenario_generator import (
    BQMModelGenerator,
    CorrelatedReturnsVolaGridRiskState,
    RiskState,
    ReturnsVolaGridRiskState,
    ReturnsVolaGridScenarioGenerator,
    ScenarioGenerator,
)


class ReturnsVolaGridRiskStateTest(unittest.TestCase):
    def test_is_a_risk_state_and_dispatches_itself_to_generator(self) -> None:
        grids = {
            "AAPL": numpy.array([[0.01, 0.20], [0.02, 0.25]]),
            "MSFT": numpy.array([[0.03, 0.15]]),
        }
        risk_state = ReturnsVolaGridRiskState(grids)
        generator = Mock()
        portfolio = Portfolio()

        result = risk_state.accept(generator, portfolio)

        self.assertIsInstance(risk_state, RiskState)
        self.assertIs(risk_state.returnsVolaGrid, grids)
        generator.createReturnsVolaGridBQM.assert_called_once_with(
            risk_state,
            portfolio,
        )
        self.assertIsNone(result)

    def test_correlated_risk_state_dispatches_its_correlations(self) -> None:
        grids = {"AAPL": numpy.array([[0.01, 0.20]])}
        correlations = {("AAPL", "MSFT"): 0.75}
        risk_state = CorrelatedReturnsVolaGridRiskState(grids, correlations)
        generator = Mock()
        portfolio = Portfolio()

        result = risk_state.accept(generator, portfolio)

        self.assertIsInstance(risk_state, ReturnsVolaGridRiskState)
        self.assertIs(risk_state.returnsVolaGrid, grids)
        self.assertIs(risk_state.correlations, correlations)
        generator.createCorrelatedReturnsVolaGridBQM.assert_called_once_with(
            risk_state,
            portfolio,
        )
        self.assertIsNone(result)

    def test_bqm_generator_methods_are_empty(self) -> None:
        generator = BQMModelGenerator()
        grids = {"AAPL": numpy.array([[0.01, 0.20]])}
        correlations = {("AAPL", "MSFT"): 0.75}
        risk_state = ReturnsVolaGridRiskState(grids)
        correlated_risk_state = CorrelatedReturnsVolaGridRiskState(
            grids,
            correlations,
        )
        portfolio = Portfolio()

        self.assertIsNone(
            generator.createReturnsVolaGridBQM(risk_state, portfolio)
        )
        self.assertIsNone(
            generator.createCorrelatedReturnsVolaGridBQM(
                correlated_risk_state,
                portfolio,
            )
        )

    def test_risk_state_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            RiskState()

    def test_scenario_generator_returns_its_data_requirements(self) -> None:
        command = UnifiedFormatCommand(
            instruments=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            period=Period.ONE_DAY,
        )

        result = ReturnsVolaGridScenarioGenerator().dataRequirements(command)

        self.assertIs(result, command)

    def test_returns_generator_key_extraction_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ReturnsVolaGridScenarioGenerator().getRiskStates(object())

    def test_scenario_generator_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            ScenarioGenerator()


if __name__ == "__main__":
    unittest.main()
