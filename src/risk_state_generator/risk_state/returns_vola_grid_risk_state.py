"""Returns-volatility-grid risk state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from download_unit import Instrument

from .risk_state import RiskState

if TYPE_CHECKING:
    import numpy

@dataclass
class ReturnsVolaGridRiskState(RiskState):
    """Associate each instrument with its two-dimensional risk-state grid."""

    returnsVolaGrid: dict[Instrument, numpy.ndarray]
