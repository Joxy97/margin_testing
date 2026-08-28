"""BQM model generation interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from download_unit import Instrument

if TYPE_CHECKING:
    from .scenario import (
        CorrelatedReturnsVolaGridScenario,
        ReturnsVolaGridScenario,
    )


class BQMModelGenerator:
    """Build D-Wave binary quadratic models from scenarios."""

    def createReturnsVolaGridBQM(
        self,
        scenario: ReturnsVolaGridScenario,
    ) -> None:
        """Create a BQM for a returns-volatility-grid scenario."""
        pass

    def createCorrelatedReturnsVolaGridBQM(
        self,
        scenario: CorrelatedReturnsVolaGridScenario,
        correlations: dict[tuple[Instrument, Instrument], float],
    ) -> None:
        """Create a BQM for a correlated returns-volatility-grid scenario."""
        pass
