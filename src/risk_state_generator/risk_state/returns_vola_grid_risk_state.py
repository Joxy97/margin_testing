"""Returns-volatility-grid risk state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy

from download_unit import Instrument

from .risk_state import RiskState
from .dense_returns_vola_grid import DenseReturnsVolaGrid

@dataclass
class ReturnsVolaGridRiskState(RiskState):
    """Associate each instrument with its two-dimensional risk-state grid."""

    returnsVolaGrid: DenseReturnsVolaGrid | Mapping[Instrument, numpy.ndarray]

    def __post_init__(self) -> None:
        self.returnsVolaGrid = DenseReturnsVolaGrid.fromMapping(
            self.returnsVolaGrid
        )

    @property
    def returnBounds(self) -> numpy.ndarray:
        return self.returnsVolaGrid.returnBounds
