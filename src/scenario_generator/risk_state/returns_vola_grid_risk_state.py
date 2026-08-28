"""Returns-volatility-grid risk state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from download_unit import Instrument

from ..bqm_model_generator import BQMModelGenerator
from .risk_state import RiskState

if TYPE_CHECKING:
    import numpy

    from portfolio import Portfolio


@dataclass
class ReturnsVolaGridRiskState(RiskState):
    """Associate each instrument with its two-dimensional risk-state grid."""

    returnsVolaGrid: dict[Instrument, numpy.ndarray]

    def accept(
        self,
        bqmModelGenerator: BQMModelGenerator,
        portfolio: Portfolio,
    ) -> None:
        """Dispatch this risk state to its BQM model generator method."""
        bqmModelGenerator.createReturnsVolaGridBQM(self, portfolio)
