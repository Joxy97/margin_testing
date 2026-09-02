"""BQM-based portfolio margin calculation."""

from collections.abc import Iterable, Mapping
from typing import Any

from portfolio import Portfolio
from risk_state_generator import (
    PortfolioRiskStateBQMVisitor,
    RiskState,
)

from .optimization.optimization_problem.qubo_problem import QUBOProblem
from .optimization.optimization_result import BQMOptimizationResult
from .optimization.optimization_solver.bqm_solver import BQMSolver
from .optimization.optimization_solver.bqm_solver.bqm_execution_policy import (
    BQMExecutionPolicy,
    SequentialBQMExecutionPolicy,
)
from .optimization_margin_calculator import OptimizationMarginCalculator
from .state_aware_greedy_risk_state_visitor import (
    StateAwareGreedyRiskStateVisitor,
)


class BQMMarginCalculator(OptimizationMarginCalculator):
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
        super().__init__(solverParameters)
        self.modelParameters: dict[str, Any] = dict(modelParameters or {})
        self.bqmSolver = bqmSolver
        self.bqmVisitor = bqmVisitor or PortfolioRiskStateBQMVisitor()
        self.executionPolicy = executionPolicy or SequentialBQMExecutionPolicy()
        self.comparisonVisitor = (
            None
            if comparisonPnlAnchor is None
            else StateAwareGreedyRiskStateVisitor(comparisonPnlAnchor)
        )
        self.lastComparisonMargins: dict[str, float] = {}

    def calculateMargin(
        self,
        riskStates: Iterable[RiskState],
        portfolio: Portfolio,
    ) -> float:
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

        for risk_state, result in self.executionPolicy.execute(
            self.bqmSolver,
            encodedStates(),
            self.solverParameters,
        ):
            if not isinstance(result, BQMOptimizationResult):
                raise TypeError("BQMSolver must return a BQMOptimizationResult")
            maximum_margin = max(
                maximum_margin,
                self.bqmVisitor.decodeMargin(risk_state, portfolio, result),
            )
        self.lastComparisonMargins = (
            {}
            if self.comparisonVisitor is None
            else {"greedy": -comparison_lowest_pnl}
        )
        return maximum_margin

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
