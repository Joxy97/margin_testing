"""Correlated returns-volatility-grid scenario."""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from download_unit import Instrument

from ..bqm_model_generator import BQMModelGenerator
from .returns_vola_grid_scenario import ReturnsVolaGridScenario


@dataclass
class CorrelatedReturnsVolaGridScenario(ReturnsVolaGridScenario):
    """Add pairwise instrument correlations to a returns-volatility grid."""

    returnsVolaGrid: dict[Instrument, numpy.ndarray]
    correlations: dict[tuple[Instrument, Instrument], float]

    def accept(self, bqmModelGenerator: BQMModelGenerator) -> None:
        """Dispatch this scenario and its correlations to the generator."""
        bqmModelGenerator.createCorrelatedReturnsVolaGridBQM(
            self,
            self.correlations,
        )
