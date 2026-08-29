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
        dense_grid = riskState.returnsVolaGrid
        if numpy.any(dense_grid.stateCounts == 0):
            empty_asset = int(numpy.flatnonzero(dense_grid.stateCounts == 0)[0])
            raise ValueError(
                f"{dense_grid.instruments[empty_asset]} has no risk states"
            )
        positions = numpy.fromiter(
            (
                float(portfolio.weights.get(instrument, 0))
                for instrument in dense_grid.instruments
            ),
            dtype=float,
            count=len(dense_grid),
        )
        if not numpy.isfinite(positions).all():
            raise ValueError("portfolio contains a non-finite position")
        worst_returns = numpy.where(
            positions >= 0.0,
            dense_grid.returnBounds[:, 0],
            dense_grid.returnBounds[:, 1],
        )
        return float(positions @ worst_returns)
