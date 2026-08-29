"""Base portfolio-aware risk-state dispatch object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .risk_state import RiskState

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
class PortfolioRiskState(ABC):
    """Pair a risk state with the portfolio evaluated under that state."""

    riskState: RiskState
    portfolio: Portfolio

    @staticmethod
    def fromRiskState(
        riskState: RiskState,
        portfolio: Portfolio,
    ) -> PortfolioRiskState:
        """Create the portfolio-aware wrapper matching ``riskState``."""
        from .correlated_returns_vola_grid_risk_state import (
            CorrelatedReturnsVolaGridRiskState,
        )
        from .portfolio_correlated_returns_vola_grid_risk_state import (
            PortfolioCorrelatedReturnsVolaGridRiskState,
        )
        from .portfolio_returns_vola_grid_risk_state import (
            PortfolioReturnsVolaGridRiskState,
        )
        from .returns_vola_grid_risk_state import ReturnsVolaGridRiskState

        if isinstance(riskState, CorrelatedReturnsVolaGridRiskState):
            return PortfolioCorrelatedReturnsVolaGridRiskState(
                riskState,
                portfolio,
            )
        if isinstance(riskState, ReturnsVolaGridRiskState):
            return PortfolioReturnsVolaGridRiskState(riskState, portfolio)
        raise TypeError(f"Unsupported risk state: {type(riskState).__name__}")

    @abstractmethod
    def acceptGreedy(
        self,
        scenarioVisitor: GreedyPortfolioRiskStateScenario,
    ) -> float:
        """Dispatch greedy scenario evaluation for this portfolio risk state."""
        raise NotImplementedError

    @abstractmethod
    def acceptBQM(
        self,
        bqmManager: PortfolioRiskStateBQMManager,
        parameters: Mapping[str, Any],
    ) -> QUBOProblem:
        """Dispatch BQM construction for this portfolio risk state."""
        raise NotImplementedError

    @abstractmethod
    def acceptDecode(
        self,
        bqmManager: PortfolioRiskStateBQMManager,
        bqmOptimizationResult: BQMOptimizationResult,
    ) -> float:
        """Dispatch result decoding for this portfolio risk state."""
        raise NotImplementedError
