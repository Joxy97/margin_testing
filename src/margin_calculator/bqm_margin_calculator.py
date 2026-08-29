"""BQM-based portfolio margin calculation."""

from collections.abc import Iterable, Mapping
from typing import Any

from portfolio import Portfolio
from risk_state_generator import (
    PortfolioRiskState,
    PortfolioRiskStateBQMManager,
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


class BQMMarginCalculator(OptimizationMarginCalculator):
    """Build, solve, and decode one QUBO for every risk state."""

    def __init__(
        self,
        bqmSolver: BQMSolver,
        modelParameters: Mapping[str, Any] | None = None,
        solverParameters: Mapping[str, Any] | None = None,
        bqmManager: PortfolioRiskStateBQMManager | None = None,
        executionPolicy: BQMExecutionPolicy[PortfolioRiskState] | None = None,
    ) -> None:
        super().__init__(solverParameters)
        self.modelParameters: dict[str, Any] = dict(modelParameters or {})
        self.bqmSolver = bqmSolver
        self.bqmManager = bqmManager or PortfolioRiskStateBQMManager()
        self.executionPolicy = executionPolicy or SequentialBQMExecutionPolicy()

    def calculateMargin(
        self,
        riskStates: Iterable[RiskState],
        portfolio: Portfolio,
    ) -> float:
        """Return the greatest decoded loss across all risk states."""
        maximum_margin = 0.0
        encoded_states = (
            self._encodeRiskState(risk_state, portfolio)
            for risk_state in riskStates
        )
        for portfolio_risk_state, result in self.executionPolicy.execute(
            self.bqmSolver,
            encoded_states,
            self.solverParameters,
        ):
            if not isinstance(result, BQMOptimizationResult):
                raise TypeError("BQMSolver must return a BQMOptimizationResult")
            maximum_margin = max(
                maximum_margin,
                portfolio_risk_state.acceptDecode(self.bqmManager, result),
            )
        return maximum_margin

    def _encodeRiskState(
        self,
        riskState: RiskState,
        portfolio: Portfolio,
    ) -> tuple[PortfolioRiskState, QUBOProblem]:
        portfolio_risk_state = PortfolioRiskState.fromRiskState(
            riskState,
            portfolio,
        )
        return (
            portfolio_risk_state,
            portfolio_risk_state.acceptBQM(
                self.bqmManager,
                self.modelParameters,
            ),
        )
