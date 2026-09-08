"""BQM-based portfolio margin calculation."""

from collections.abc import Iterable, Mapping
from typing import Any

from portfolio import Portfolio
from risk_state_generator.risk_state import RiskState
from .optimization.portfolio_risk_state_bqm_visitor import PortfolioRiskStateBQMVisitor

from .optimization.optimization_problem.qubo_problem import QUBOProblem
from .optimization.optimization_result import BQMOptimizationResult
from .optimization.optimization_solver.bqm_solver import BQMSolver
from .optimization.optimization_solver.bqm_solver.bqm_execution_policy import (
    BQMExecutionPolicy,
    SequentialBQMExecutionPolicy,
)
from .margin_calculator import MarginCalculator
from .calculation_outcome import CalculationOutcome
from .state_aware_greedy_risk_state_visitor import (
    StateAwareGreedyRiskStateVisitor,
)


class BQMMarginCalculator(MarginCalculator):
    """Build, solve, and decode one QUBO for every risk state."""

    def __init__(
        self,
        bqmSolver: BQMSolver,
        modelParameters: Mapping[str, Any] | None = None,
        solverParameters: Mapping[str, Any] | None = None,
        bqmVisitor: PortfolioRiskStateBQMVisitor | None = None,
        executionPolicy: BQMExecutionPolicy[RiskState] | None = None,
        comparisonPnlAnchor: str | None = None,
    ) -> None:
        self.solverParameters = dict(solverParameters or {})
        self.modelParameters: dict[str, Any] = dict(modelParameters or {})
        self.bqmSolver = bqmSolver
        self.bqmVisitor = bqmVisitor or PortfolioRiskStateBQMVisitor()
        self.executionPolicy = executionPolicy or SequentialBQMExecutionPolicy()
        self.comparisonVisitor = (
            None
            if comparisonPnlAnchor is None
            else StateAwareGreedyRiskStateVisitor(comparisonPnlAnchor)
        )

    def calculateMargin(
        self,
        riskStates: Iterable[RiskState],
        portfolio: Portfolio,
    ) -> float:
        return self.calculateOutcome(riskStates, portfolio).margin

    def calculateOutcome(self, riskStates: Iterable[RiskState], portfolio: Portfolio) -> CalculationOutcome:
        """Return the greatest decoded loss across all risk states."""
        maximum_margin = 0.0
        comparison_lowest_pnl = 0.0

        def encodedStates():
            nonlocal comparison_lowest_pnl
            for risk_state in riskStates:
                if self.comparisonVisitor is not None:
                    comparison_lowest_pnl = min(
                        comparison_lowest_pnl,
                        self.comparisonVisitor.portfolioPnl(
                            risk_state,
                            portfolio,
                        ),
                    )
                yield self._encodeRiskState(risk_state, portfolio)

        execution = self.executionPolicy.execute(
            self.bqmSolver, encodedStates(), self.solverParameters)
        try:
            for risk_state, result in execution:
                if not isinstance(result, BQMOptimizationResult):
                    raise TypeError("BQMSolver must return a BQMOptimizationResult")
                maximum_margin = max(
                    maximum_margin,
                    self.bqmVisitor.decodeMargin(risk_state, portfolio, result),
                )
        finally:
            close = getattr(execution, "close", None)
            if close is not None:
                close()
        comparison_margins = (
            {}
            if self.comparisonVisitor is None
            else {"greedy": -comparison_lowest_pnl}
        )
        return CalculationOutcome(maximum_margin, comparison_margins)

    def _encodeRiskState(
        self,
        riskState: RiskState,
        portfolio: Portfolio,
    ) -> tuple[RiskState, QUBOProblem]:
        return (
            riskState,
            self.bqmVisitor.createBQM(
                riskState,
                portfolio,
                self.modelParameters,
            ),
        )
