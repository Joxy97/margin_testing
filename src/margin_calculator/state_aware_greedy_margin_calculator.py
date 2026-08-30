"""Worst-case aggregation with risk-state-specific scenario behavior."""

from __future__ import annotations

from collections.abc import Iterable

from risk_state_generator import RiskState

from .margin_calculator import MarginCalculator
from .state_aware_greedy_risk_state_visitor import (
    StateAwareGreedyRiskStateVisitor,
)


class StateAwareGreedyMarginCalculator(MarginCalculator):
    """Return the greatest loss produced by a risk-state visitor."""

    def __init__(self, pnlAnchor: str = "market") -> None:
        self._riskStateVisitor = StateAwareGreedyRiskStateVisitor(pnlAnchor)

    def calculateMargin(self, riskStates: Iterable[RiskState], portfolio) -> float:
        lowest_pnl = 0.0
        found = False
        for risk_state in riskStates:
            found = True
            lowest_pnl = min(
                lowest_pnl,
                self._riskStateVisitor.portfolioPnl(risk_state, portfolio),
            )
        if not found:
            raise ValueError("riskStates must not be empty")
        return -lowest_pnl
