"""Immutable scenario records shared by deterministic FHS-EVT stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy


class ScenarioType(str, Enum):
    """Separate probability-bearing scenarios from zero-weight stresses."""

    PROBABILITY = "PROBABILITY"
    STRESS = "STRESS"


def _optionalReadonly(values: numpy.ndarray | None) -> numpy.ndarray | None:
    if values is None:
        return None
    result = numpy.ascontiguousarray(values, dtype=numpy.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FHSScenario:
    """One indivisible joint factor scenario and its audit metadata."""

    scenarioId: str
    scenarioType: ScenarioType
    probability: float
    innovations: numpy.ndarray
    factorChanges: numpy.ndarray
    protected: bool = False
    protectionReason: str | None = None
    ownerRowId: str | None = None
    nodeId: str | None = None
    stressFamily: str | None = None
    sigmaLabel: int | None = None
    quantiles: numpy.ndarray | None = None
    assignedParentMass: float = 0.0

    def __post_init__(self) -> None:
        if not self.scenarioId:
            raise ValueError("scenarioId must not be empty")
        innovations = _optionalReadonly(self.innovations)
        changes = _optionalReadonly(self.factorChanges)
        quantiles = _optionalReadonly(self.quantiles)
        assert innovations is not None and changes is not None
        if innovations.ndim != 1 or changes.shape != innovations.shape:
            raise ValueError("scenario innovations and changes must be equal vectors")
        if not numpy.isfinite(innovations).all() or not numpy.isfinite(changes).all():
            raise ValueError("scenario vectors must be finite")
        if quantiles is not None and quantiles.shape != innovations.shape:
            raise ValueError("scenario quantiles must align with innovations")
        if quantiles is not None and (
            numpy.any(quantiles <= 0.0) or numpy.any(quantiles >= 1.0)
        ):
            raise ValueError("scenario quantiles must lie strictly inside (0, 1)")
        if not math.isfinite(self.probability) or self.probability < 0.0:
            raise ValueError("scenario probability must be finite and nonnegative")
        if self.scenarioType is ScenarioType.STRESS:
            if self.probability != 0.0 or not self.protected:
                raise ValueError(
                    "stress scenarios must be protected and have zero weight"
                )
        elif self.probability <= 0.0:
            raise ValueError("probability scenarios must have positive weight")
        if not math.isfinite(self.assignedParentMass) or self.assignedParentMass < 0.0:
            raise ValueError("assignedParentMass must be finite and nonnegative")
        object.__setattr__(self, "innovations", innovations)
        object.__setattr__(self, "factorChanges", changes)
        object.__setattr__(self, "quantiles", quantiles)
