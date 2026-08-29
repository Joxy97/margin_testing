"""Tests for optimization-based portfolio margin calculation."""

import unittest
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import numpy

from margin_calculator import (
    BQMMarginCalculator,
    GreedyMarginCalculator,
    GreedyMarginCalculatorConfig,
    GreedyPortfolioRiskStateScenario,
    MarginCalculator,
    OptimizationMarginCalculator,
)
from margin_calculator.optimization.optimization_result import BQMOptimizationResult
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMSolver,
    BatchBQMExecutionPolicy,
    QUBOProblem,
)
from portfolio import Portfolio
from risk_state_generator import ReturnsVolaGridRiskState
from risk_state_generator import (
    CorrelatedReturnsVolaGridRiskState,
    CorrelationFactors,
    PortfolioRiskState,
)


class StubBQMSolver(BQMSolver):
    def __init__(self, results: list[BQMOptimizationResult]) -> None:
        self.results = iter(results)
        self.problems: list[QUBOProblem] = []
        self.solverParameters: list[dict[str, Any]] = []

    def solve(
        self,
        problem: QUBOProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> BQMOptimizationResult:
        self.problems.append(problem)
        self.solverParameters.append(dict(solverParameters or {}))
        return next(self.results)


class BatchTrackingSolver(BQMSolver):
    def __init__(self) -> None:
        self.batchSizes: list[int] = []

    def solve(
        self,
        problem: QUBOProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> BQMOptimizationResult:
        sample = tuple(
            int(any(variable == group[0] for group in problem.oneHotGroups))
            for variable in range(problem.variableCount)
        )
        return BQMOptimizationResult(sample, problem.energy(sample))

    def solveMany(
        self,
        problems: list[QUBOProblem] | tuple[QUBOProblem, ...],
        solverParameters: Mapping[str, Any] | None = None,
    ) -> list[BQMOptimizationResult]:
        self.batchSizes.append(len(problems))
        return [self.solve(problem, solverParameters) for problem in problems]


class MarginCalculatorTest(unittest.TestCase):
    def test_greedy_visitor_sums_position_times_lowest_bin_pnl(self) -> None:
        risk_state = ReturnsVolaGridRiskState(
            {
                "AAPL": numpy.array([[-0.05, 0.20], [0.02, 0.25]]),
                "MSFT": numpy.array([[-0.04, 0.15], [0.03, 0.18]]),
            }
        )
        portfolio = Portfolio(
            weights={"AAPL": Decimal("10"), "MSFT": Decimal("-5")}
        )
        visitor = GreedyPortfolioRiskStateScenario()

        pnl = PortfolioRiskState.fromRiskState(
            risk_state,
            portfolio,
        ).acceptGreedy(visitor)

        self.assertAlmostEqual(pnl, -0.3)

    def test_greedy_calculator_returns_the_worst_nonnegative_loss(self) -> None:
        risk_states = (
            ReturnsVolaGridRiskState(
                {"AAPL": numpy.array([[-0.02, 0.2], [0.01, 0.3]])}
            ),
            ReturnsVolaGridRiskState(
                {"AAPL": numpy.array([[-0.07, 0.2], [0.02, 0.3]])}
            ),
        )

        margin = GreedyMarginCalculator().calculateMargin(
            iter(risk_states),
            Portfolio(weights={"AAPL": Decimal("10")}),
        )

        self.assertAlmostEqual(margin, 0.7)

    def test_correlated_risk_state_has_its_own_greedy_dispatch(self) -> None:
        risk_state = CorrelatedReturnsVolaGridRiskState(
            {"AAPL": numpy.array([[-0.03, 0.2], [0.01, 0.3]])},
            CorrelationFactors.empty(),
        )

        margin = GreedyMarginCalculator().calculateMargin(
            [risk_state],
            Portfolio(weights={"AAPL": Decimal("10")}),
        )

        self.assertAlmostEqual(margin, 0.3)

    def test_greedy_margin_config_constructs_the_calculator(self) -> None:
        visitor = GreedyPortfolioRiskStateScenario()

        calculator = GreedyMarginCalculatorConfig(visitor).createMarginCalculator()

        self.assertIsInstance(calculator, GreedyMarginCalculator)
        self.assertIs(calculator.scenarioVisitor, visitor)

    def test_batch_policy_submits_bounded_native_batches(self) -> None:
        solver = BatchTrackingSolver()
        calculator = BQMMarginCalculator(
            solver,
            executionPolicy=BatchBQMExecutionPolicy(batchSize=2),
        )
        risk_states = (
            ReturnsVolaGridRiskState(
                {"AAPL": numpy.array([[-0.01 * index, 0.2], [0.01, 0.2]])}
            )
            for index in range(1, 5)
        )

        margin = calculator.calculateMargin(
            risk_states,
            Portfolio(weights={"AAPL": Decimal("10")}),
        )

        self.assertAlmostEqual(margin, 0.4)
        self.assertEqual(solver.batchSizes, [2, 2])

    def test_bqm_calculator_returns_the_largest_scenario_loss(self) -> None:
        risk_states = [
            ReturnsVolaGridRiskState(
                {"AAPL": numpy.array([[-0.05, 0.20], [0.01, 0.25]])}
            ),
            ReturnsVolaGridRiskState(
                {"AAPL": numpy.array([[-0.10, 0.20], [0.02, 0.25]])}
            ),
        ]
        solver = StubBQMSolver(
            [
                BQMOptimizationResult({"x_0_0": 1, "x_0_1": 0}),
                BQMOptimizationResult({"x_0_0": 1, "x_0_1": 0}),
            ]
        )
        model_parameters = {"lambdaOneHot": 2.0, "lambdaCompat": 0.1}
        solver_parameters = {"num_reads": 20}
        calculator = BQMMarginCalculator(
            solver,
            model_parameters,
            solver_parameters,
        )
        portfolio = Portfolio(weights={"AAPL": Decimal("10")})

        margin = calculator.calculateMargin(risk_states, portfolio)

        self.assertAlmostEqual(margin, 1.0)
        self.assertEqual(calculator.modelParameters, model_parameters)
        self.assertEqual(calculator.solverParameters, solver_parameters)
        self.assertEqual(len(solver.problems), 2)
        self.assertEqual(solver.solverParameters, [solver_parameters] * 2)
        self.assertTrue(
            all(isinstance(problem, QUBOProblem) for problem in solver.problems)
        )
        self.assertTrue(
            all(
                problem.oneHotGroups == ((0, 1),)
                for problem in solver.problems
            )
        )

    def test_margin_is_never_negative(self) -> None:
        risk_state = ReturnsVolaGridRiskState(
            {"AAPL": numpy.array([[0.05, 0.20]])}
        )
        solver = StubBQMSolver([BQMOptimizationResult({"x_0_0": 1})])
        calculator = BQMMarginCalculator(solver)
        portfolio = Portfolio(weights={"AAPL": Decimal("10")})

        margin = calculator.calculateMargin([risk_state], portfolio)

        self.assertEqual(margin, 0.0)

    def test_base_margin_calculators_are_abstract(self) -> None:
        with self.assertRaises(TypeError):
            MarginCalculator()
        with self.assertRaises(TypeError):
            OptimizationMarginCalculator()


if __name__ == "__main__":
    unittest.main()
