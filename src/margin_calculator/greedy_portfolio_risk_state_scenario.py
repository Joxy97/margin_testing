"""Greedy portfolio scenario selection for risk-state grids."""

from __future__ import annotations

from functools import singledispatchmethod

import numpy

from risk_state_generator import (
    PortfolioCorrelatedReturnsVolaGridRiskState,
    PortfolioReturnsVolaGridRiskState,
    PortfolioRiskState,
    ReturnsVolaGridRiskState,
)


class GreedyPortfolioRiskStateScenario:
    """Select each instrument's worst portfolio PnL independently."""

    @singledispatchmethod
    def getGreedyScenario(
        self,
        portfolioRiskState: PortfolioRiskState,
    ) -> float:
        """Return the greedy portfolio PnL for a supported risk state."""
        raise TypeError(
            "Unsupported portfolio risk state: "
            f"{type(portfolioRiskState).__name__}"
        )

    @getGreedyScenario.register
    def _(
        self,
        portfolioRiskState: PortfolioReturnsVolaGridRiskState,
    ) -> float:
        return self._getReturnsVolaGridScenario(portfolioRiskState)

    @getGreedyScenario.register
    def _(
        self,
        portfolioRiskState: PortfolioCorrelatedReturnsVolaGridRiskState,
    ) -> float:
        return self._getReturnsVolaGridScenario(portfolioRiskState)

    @staticmethod
    def _getReturnsVolaGridScenario(
        portfolioRiskState: PortfolioReturnsVolaGridRiskState,
    ) -> float:
        riskState: ReturnsVolaGridRiskState = portfolioRiskState.riskState
        portfolio = portfolioRiskState.portfolio
        portfolio_pnl = 0.0
        for instrument, grid in riskState.returnsVolaGrid.items():
            values = numpy.asarray(grid)
            if values.ndim != 2 or values.shape[1] < 1:
                raise ValueError(
                    f"{instrument} risk states must be a two-dimensional grid"
                )
            if len(values) == 0:
                raise ValueError(f"{instrument} has no risk states")
            position = float(portfolio.weights.get(instrument, 0))
            bin_pnls = values[:, 0]
            if not numpy.isfinite(bin_pnls).all() or not numpy.isfinite(position):
                raise ValueError(f"{instrument} contains a non-finite PnL")
            portfolio_pnl += position * float(numpy.min(bin_pnls))
        return portfolio_pnl
