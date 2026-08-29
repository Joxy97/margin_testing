"""Portfolio-aware correlated returns-volatility-grid risk state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .correlated_returns_vola_grid_risk_state import (
    CorrelatedReturnsVolaGridRiskState,
)
from .portfolio_returns_vola_grid_risk_state import (
    PortfolioReturnsVolaGridRiskState,
)

if TYPE_CHECKING:
    from portfolio import Portfolio

    from margin_calculator.optimization.optimization_result import BQMOptimizationResult
    from margin_calculator.optimization.optimization_problem.qubo_problem import (
        QUBOProblem,
    )
    from margin_calculator.greedy_portfolio_risk_state_scenario import (
        GreedyPortfolioRiskStateScenario,
    )

    from ..portfolio_risk_state_bqm_manager import PortfolioRiskStateBQMManager


@dataclass
class PortfolioCorrelatedReturnsVolaGridRiskState(
    PortfolioReturnsVolaGridRiskState
):
    """Pair a correlated returns-volatility grid with a portfolio."""

    riskState: CorrelatedReturnsVolaGridRiskState
    portfolio: Portfolio

    def acceptGreedy(
        self,
        scenarioVisitor: GreedyPortfolioRiskStateScenario,
    ) -> float:
        return scenarioVisitor.getGreedyScenario(self)

    def acceptBQM(
        self,
        bqmManager: PortfolioRiskStateBQMManager,
        parameters: Mapping[str, Any],
    ) -> QUBOProblem:
        return bqmManager.createCorrelatedReturnsVolaGridBQM(
            self.riskState,
            self.portfolio,
            parameters,
        )

    def acceptDecode(
        self,
        bqmManager: PortfolioRiskStateBQMManager,
        bqmOptimizationResult: BQMOptimizationResult,
    ) -> float:
        return bqmManager.decodeCorrelatedReturnsVolaGridRiskState(
            self.riskState,
            self.portfolio,
            bqmOptimizationResult,
        )
