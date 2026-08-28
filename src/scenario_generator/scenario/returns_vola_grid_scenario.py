"""Returns-volatility-grid scenario."""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from download_unit import Instrument

from ..bqm_model_generator import BQMModelGenerator
from .scenario import Scenario


@dataclass
class ReturnsVolaGridScenario(Scenario):
    """Associate each instrument with its two-dimensional scenario grid."""

    returnsVolaGrid: dict[Instrument, numpy.ndarray]

    def accept(self, bqmModelGenerator: BQMModelGenerator) -> None:
        """Dispatch this scenario to its BQM model generator method."""
        bqmModelGenerator.createReturnsVolaGridBQM(self)
