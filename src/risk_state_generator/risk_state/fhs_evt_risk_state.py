"""Joint deterministic FHS-EVT portfolio-independent risk state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy

from ..fhs_evt.scenario import ScenarioType
from .returns_vola_grid_risk_state import ReturnsVolaGridRiskState


@dataclass
class FHSEVTRiskState(ReturnsVolaGridRiskState):
    """One indivisible row of the reduced deterministic scenario set."""

    scenarioId: str
    scenarioType: ScenarioType
    probability: float
    factorChanges: numpy.ndarray
    protected: bool = False
    protectionReason: str | None = None
    ownerRowId: str | None = None
    nodeId: str | None = None
    stressFamily: str | None = None
    sigmaLabel: int | None = None
    assignedParentMass: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        changes = numpy.ascontiguousarray(self.factorChanges, dtype=numpy.float64)
        if changes.shape != (len(self.returnsVolaGrid.instruments),):
            raise ValueError("factorChanges must contain one value per instrument")
        if not numpy.isfinite(changes).all():
            raise ValueError("factorChanges must be finite")
        if numpy.any(self.returnsVolaGrid.stateCounts != 1):
            raise ValueError(
                "an FHS-EVT state must contain one joint state per instrument"
            )
        if not self.scenarioId:
            raise ValueError("scenarioId must not be empty")
        if not math.isfinite(self.probability) or self.probability < 0.0:
            raise ValueError("scenario probability must be finite and nonnegative")
        if self.scenarioType is ScenarioType.STRESS:
            if self.probability != 0.0 or not self.protected:
                raise ValueError("FHS-EVT stresses must be protected and zero-weight")
        elif self.probability <= 0.0:
            raise ValueError("FHS-EVT probability scenarios must have positive weight")
        changes.setflags(write=False)
        self.factorChanges = changes
