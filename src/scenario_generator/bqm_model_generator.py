"""BQM model generation interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portfolio import Portfolio

    from .risk_state import (
        CorrelatedReturnsVolaGridRiskState,
        ReturnsVolaGridRiskState,
    )


class BQMModelGenerator:
    """Build D-Wave binary quadratic models from risk states."""

    def createReturnsVolaGridBQM(
        self,
        riskState: ReturnsVolaGridRiskState,
        portfolio: Portfolio,
    ) -> None:
        """Create a BQM for a returns-volatility-grid risk state."""
        pass

    def createCorrelatedReturnsVolaGridBQM(
        self,
        riskState: CorrelatedReturnsVolaGridRiskState,
        portfolio: Portfolio,
    ) -> None:
        """Create a BQM for a correlated returns-volatility-grid risk state."""
        pass
