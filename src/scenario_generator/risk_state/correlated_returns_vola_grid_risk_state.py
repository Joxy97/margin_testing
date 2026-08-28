"""Correlated returns-volatility-grid risk state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from download_unit import Instrument

from ..bqm_model_generator import BQMModelGenerator
from .returns_vola_grid_risk_state import ReturnsVolaGridRiskState

if TYPE_CHECKING:
    import numpy

    from portfolio import Portfolio


@dataclass
class CorrelatedReturnsVolaGridRiskState(ReturnsVolaGridRiskState):
    """Add pairwise instrument correlations to a returns-volatility grid."""

    returnsVolaGrid: dict[Instrument, numpy.ndarray]
    correlations: dict[tuple[Instrument, Instrument], float]

    def accept(
        self,
        bqmModelGenerator: BQMModelGenerator,
        portfolio: Portfolio,
    ) -> None:
        """Dispatch this correlated risk state to the generator."""
        bqmModelGenerator.createCorrelatedReturnsVolaGridBQM(
            self,
            portfolio,
        )
