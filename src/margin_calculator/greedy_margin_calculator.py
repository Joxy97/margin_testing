"""Greedy portfolio margin calculation."""

from collections.abc import Iterable

from portfolio import Portfolio
from risk_state_generator import PortfolioRiskState, RiskState

from .greedy_portfolio_risk_state_scenario import (
    GreedyPortfolioRiskStateScenario,
)
from .margin_calculator import MarginCalculator


class GreedyMarginCalculator(MarginCalculator):
    """Calculate margin from independently worst instrument bins."""

    def __init__(
        self,
        scenarioVisitor: GreedyPortfolioRiskStateScenario | None = None,
    ) -> None:
        self.scenarioVisitor = (
            scenarioVisitor or GreedyPortfolioRiskStateScenario()
        )

    def calculateMargin(
        self,
        riskStates: Iterable[RiskState],
        portfolio: Portfolio,
    ) -> float:
        """Return the non-negative loss of the lowest greedy scenario PnL."""
        lowest_pnl = 0.0
        for risk_state in riskStates:
            portfolio_risk_state = PortfolioRiskState.fromRiskState(
                risk_state,
                portfolio,
            )
            lowest_pnl = min(
                lowest_pnl,
                portfolio_risk_state.acceptGreedy(self.scenarioVisitor),
            )
        return -lowest_pnl
