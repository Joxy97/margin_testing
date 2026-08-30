"""Risk-state-specific portfolio scenario evaluation."""

from __future__ import annotations

from functools import singledispatchmethod

import numpy

from portfolio import Portfolio, contractKey
from risk_state_generator import (
    OptionScenarioRiskState,
    ReturnsVolaGridRiskState,
    RiskState,
)

from .option_scenario_valuator import OptionScenarioValuator


class StateAwareGreedyRiskStateVisitor:
    """Calculate portfolio P&L using behavior selected by risk-state type."""

    def __init__(
        self,
        pnlAnchor: str = "market",
        optionValuator: OptionScenarioValuator | None = None,
    ) -> None:
        if pnlAnchor not in {"market", "model"}:
            raise ValueError("pnlAnchor must be market or model")
        self.pnlAnchor = pnlAnchor
        self.optionValuator = optionValuator or OptionScenarioValuator()

    @singledispatchmethod
    def portfolioPnl(self, riskState: RiskState, portfolio) -> float:
        raise TypeError(f"Unsupported risk state: {type(riskState).__name__}")

    @portfolioPnl.register
    def _(self, riskState: ReturnsVolaGridRiskState, portfolio: Portfolio) -> float:
        grid = riskState.returnsVolaGrid
        if numpy.any(grid.stateCounts == 0):
            asset = int(numpy.flatnonzero(grid.stateCounts == 0)[0])
            raise ValueError(f"{grid.instruments[asset]} has no risk states")
        positions = numpy.fromiter(
            (float(portfolio.weights.get(item, 0)) for item in grid.instruments),
            dtype=float,
            count=len(grid),
        )
        if not numpy.isfinite(positions).all():
            raise ValueError("portfolio contains a non-finite position")
        worst_returns = numpy.where(
            positions >= 0.0,
            grid.returnBounds[:, 0],
            grid.returnBounds[:, 1],
        )
        return float(positions @ worst_returns)

    @portfolioPnl.register
    def _(self, riskState: OptionScenarioRiskState, portfolio) -> float:
        return sum(
            self._optionPositionPnl(position, riskState)
            for position in portfolio.positions
        )

    def _optionPositionPnl(self, position, state: OptionScenarioRiskState) -> float:
        contract = position.contract
        key = contractKey(contract)
        market_price = state.marketPrices.get(key)
        if market_price is None:
            raise ValueError(f"No exact market quote for {key}")
        base_price, scenario_price = self.optionValuator.prices(contract, state)
        anchor = market_price if self.pnlAnchor == "market" else base_price
        return (
            float(position.quantity)
            * float(contract.multiplier)
            * (scenario_price - anchor)
        )
