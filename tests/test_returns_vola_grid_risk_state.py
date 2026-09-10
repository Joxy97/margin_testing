"""Tests for returns-volatility-grid risk states."""

import unittest
from itertools import product
from datetime import date
from decimal import Decimal

import numpy

from margin_calculator.optimization.optimization_result import BQMOptimizationResult
from margin_calculator.optimization.optimization_problem.qubo_problem import (
    QUBOProblem,
)
from portfolio import Portfolio
from risk_state_generator import (
    CorrelationFactors,
    CorrelatedReturnsVolaGridRiskState,
    DenseReturnsVolaGrid,
    PortfolioRiskStateBQMVisitor,
    RiskState,
    RiskStateGenerator,
    ReturnsVolaGridRiskState,
    ReturnsVolaGridRiskStateGenerator,
)


class ReturnsVolaGridRiskStateTest(unittest.TestCase):
    def test_returns_grid_is_a_portfolio_independent_risk_state(self) -> None:
        grids = {
            "AAPL": numpy.array([[0.01, 0.20], [0.02, 0.25]]),
            "MSFT": numpy.array([[0.03, 0.15]]),
        }
        risk_state = ReturnsVolaGridRiskState(grids)

        self.assertIsInstance(risk_state, RiskState)
        self.assertIsInstance(risk_state.returnsVolaGrid, DenseReturnsVolaGrid)
        self.assertEqual(tuple(risk_state.returnsVolaGrid), tuple(grids))
        for instrument, expected in grids.items():
            numpy.testing.assert_array_equal(
                risk_state.returnsVolaGrid[instrument],
                expected,
            )

    def test_returns_grid_pads_variable_state_counts_densely(self) -> None:
        risk_state = ReturnsVolaGridRiskState(
            {
                "AAPL": numpy.array([[0.01, 0.20], [0.02, 0.25]]),
                "MSFT": numpy.array([[0.03, 0.15]]),
            }
        )
        grid = risk_state.returnsVolaGrid

        self.assertEqual(grid.gridValues.shape, (2, 2, 2))
        numpy.testing.assert_array_equal(
            grid.validStateMask,
            numpy.array([[True, True], [True, False]]),
        )
        numpy.testing.assert_array_equal(grid.stateCounts, [2, 1])
        numpy.testing.assert_allclose(grid.validReturns, [0.01, 0.02, 0.03])
        numpy.testing.assert_array_equal(grid.fallbackAssetMask, [False, False])

    def test_correlated_risk_state_stores_its_correlations(self) -> None:
        grids = {"AAPL": numpy.array([[0.01, 0.20]])}
        correlations = CorrelationFactors(
            numpy.array([0]),
            numpy.array([0]),
            numpy.array([1]),
            numpy.array([0]),
            numpy.array([0.75]),
        )
        risk_state = CorrelatedReturnsVolaGridRiskState(grids, correlations)

        self.assertIsInstance(risk_state, ReturnsVolaGridRiskState)
        self.assertIsInstance(risk_state.returnsVolaGrid, DenseReturnsVolaGrid)
        numpy.testing.assert_array_equal(
            risk_state.returnsVolaGrid["AAPL"],
            grids["AAPL"],
        )
        self.assertIs(risk_state.correlations, correlations)

    def test_bqm_generator_builds_marginlab_terms(self) -> None:
        visitor = PortfolioRiskStateBQMVisitor()
        grids = {
            "AAPL": numpy.array([[0.01, 0.20], [0.02, 0.25]]),
            "MSFT": numpy.array([[-0.03, 0.15]]),
        }
        correlations = CorrelationFactors(
            numpy.array([0]),
            numpy.array([0]),
            numpy.array([1]),
            numpy.array([0]),
            numpy.array([0.75]),
        )
        risk_state = ReturnsVolaGridRiskState(grids)
        correlated_risk_state = CorrelatedReturnsVolaGridRiskState(
            grids,
            correlations,
        )
        portfolio = Portfolio(
            weights={
                "AAPL": Decimal("10"),
                "MSFT": Decimal("5"),
            }
        )

        uncorrelated_bqm = visitor.createBQM(
            risk_state,
            portfolio,
            {"lambdaOneHot": 2.0},
        )
        correlated_bqm = visitor.createBQM(
            correlated_risk_state,
            portfolio,
            {"lambdaOneHot": 2.0, "lambdaCompat": 0.5},
        )

        self.assertIsInstance(correlated_bqm, QUBOProblem)
        numpy.testing.assert_allclose(
            sorted(uncorrelated_bqm.linear),
            sorted((10.0 * 0.01 - 2.0, 10.0 * 0.02 - 2.0, 5.0 * -0.03 - 2.0)),
        )
        numpy.testing.assert_allclose(
            sorted(uncorrelated_bqm.quadraticBiases),
            [4.0],
        )
        self.assertAlmostEqual(uncorrelated_bqm.offset, 4.0)
        numpy.testing.assert_allclose(
            sorted(correlated_bqm.linear),
            sorted((10.0 * 0.01 - 2.0, 10.0 * 0.02 - 2.0, 5.0 * -0.03 - 2.0)),
        )
        numpy.testing.assert_allclose(
            sorted(correlated_bqm.quadraticBiases),
            [0.5, 4.0],
        )
        self.assertAlmostEqual(correlated_bqm.offset, 4.0)
        visitor.createBQM(
            risk_state,
            portfolio,
            {"lambdaOneHot": 2.0},
        )
        self.assertEqual(len(visitor.structuralCache.cache.memory), 1)

    def test_correlation_normalization_and_cutoff_preserve_other_energy_terms(self) -> None:
        visitor = PortfolioRiskStateBQMVisitor()
        grids = {
            "A": numpy.array([[-0.05, 0.2], [0.03, 0.2]]),
            "B": numpy.array([[-0.02, 0.2], [0.04, 0.2]]),
            "C": numpy.array([[0.01, 0.2]]),
        }
        portfolio = Portfolio(weights={"A": Decimal("10"), "B": Decimal("-5")})
        parameters = {"lambdaOneHot": 2.0, "lambdaCompat": 0.5}
        base = visitor.createBQM(ReturnsVolaGridRiskState(grids), portfolio, parameters)
        # The cutoff is relative to the global maximum, before lambdaCompat.
        for raw in ([100., 1., 0.999, 0.], [1., .01, .00999, 0.], [0., 0., 0., 0.]):
            with self.subTest(coefficients=raw):
                factors = CorrelationFactors(
                    numpy.array([0, 0, 0, 0]), numpy.array([0, 0, 1, 1]),
                    numpy.array([1, 2, 1, 1]), numpy.array([0, 0, 0, 1]),
                    numpy.array(raw))
                state = CorrelatedReturnsVolaGridRiskState(grids, factors)
                problem = visitor.createBQM(state, portfolio, parameters)
                numpy.testing.assert_array_equal(problem.linear, base.linear)
                self.assertEqual(problem.offset, base.offset)
                numpy.testing.assert_array_equal(problem.groupOffsets, base.groupOffsets)
                numpy.testing.assert_array_equal(factors.coefficients, raw)
                self.assertFalse(factors.coefficients.flags.writeable)
                self.assertEqual(len(problem.quadraticBiases), len(base.quadraticBiases) + (2 if raw[0] else 0))
                for bits in product((0, 1), repeat=5):
                    sample = numpy.array(bits)
                    expected = base.energy(sample)
                    if raw[0]:
                        expected += .5 * sample[0] * sample[2] + .005 * sample[0] * sample[4]
                    self.assertAlmostEqual(problem.energy(sample), expected)
                disabled = visitor.createBQM(state, portfolio, {**parameters, "lambdaCompat": 0.})
                numpy.testing.assert_array_equal(disabled.quadraticBiases, base.quadraticBiases)

    def test_correlation_cutoff_does_not_hide_invalid_indices(self) -> None:
        factors = CorrelationFactors(
            numpy.array([0, 0]), numpy.array([0, 99]),
            numpy.array([1, 1]), numpy.array([0, 0]), numpy.array([100., .001]))
        state = CorrelatedReturnsVolaGridRiskState(
            {"A": numpy.array([[.01, .2]]), "B": numpy.array([[.02, .2]])}, factors)
        with self.assertRaisesRegex(ValueError, "unknown state"):
            PortfolioRiskStateBQMVisitor().createBQM(state, Portfolio(), {})

    def test_bqm_manager_decodes_the_selected_portfolio_loss(self) -> None:
        risk_state = ReturnsVolaGridRiskState(
            {
                "AAPL": numpy.array([[-0.05, 0.20], [0.01, 0.25]]),
                "MSFT": numpy.array([[-0.02, 0.15]]),
            }
        )
        portfolio = Portfolio(
            weights={"AAPL": Decimal("10"), "MSFT": Decimal("5")}
        )
        result = BQMOptimizationResult(
            {"x_0_0": 1, "x_0_1": 0, "x_1_0": 1}
        )

        margin = PortfolioRiskStateBQMVisitor().decodeMargin(
            risk_state,
            portfolio,
            result,
        )

        self.assertAlmostEqual(margin, 0.6)

    def test_bqm_manager_uses_marginlab_fallback_for_invalid_sample(self) -> None:
        risk_state = ReturnsVolaGridRiskState(
            {"AAPL": numpy.array([[-0.05, 0.20], [0.01, 0.25]])}
        )
        portfolio = Portfolio(weights={"AAPL": Decimal("10")})
        result = BQMOptimizationResult({"x_0_0": 1, "x_0_1": 1})

        margin = PortfolioRiskStateBQMVisitor().decodeMargin(
            risk_state,
            portfolio,
            result,
        )

        self.assertAlmostEqual(margin, 0.5)

    def test_bqm_manager_rejects_an_unsupported_risk_state(self) -> None:
        visitor = PortfolioRiskStateBQMVisitor()
        portfolio = Portfolio()

        with self.assertRaisesRegex(
            TypeError, "Unsupported risk state: RiskState"
        ):
            visitor.createBQM(RiskState(), portfolio, {})
        with self.assertRaisesRegex(
            TypeError, "Unsupported risk state: RiskState"
        ):
            visitor.decodeMargin(
                RiskState(),
                portfolio,
                BQMOptimizationResult(()),
            )

    def test_generator_builds_its_own_typed_data_request(self) -> None:
        generator = ReturnsVolaGridRiskStateGenerator(ew_window=5)
        portfolio = Portfolio(
            weights={"AAPL": Decimal("1"), "MSFT": Decimal("2")}
        )

        request = generator.createDataRequest(portfolio, date(2024, 1, 11))

        self.assertEqual(request.instruments, ("AAPL", "MSFT"))
        self.assertEqual(request.start_date, date(2024, 1, 1))
        self.assertEqual(request.end_date, date(2024, 1, 11))
        self.assertEqual(request.data_type, "closePrices")

    def test_risk_state_generator_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            RiskStateGenerator()


if __name__ == "__main__":
    unittest.main()
