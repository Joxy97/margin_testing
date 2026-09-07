"""Hard publication gates for reduced deterministic scenario sets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy

from .scenario import FHSScenario, ScenarioType


@dataclass(frozen=True)
class ScenarioValidationReport:
    """Structural validation result for a generated scenario family."""

    status: str
    scenarioCount: int
    factorCount: int
    probabilityCount: int
    stressCount: int
    probabilityTotal: float


def validateScenarioSet(
    scenarios: Sequence[FHSScenario],
    targetScenarios: int | None = 105,
) -> ScenarioValidationReport:
    """Require optional cardinality, finite arrays, and probability separation."""
    values = tuple(scenarios)
    if targetScenarios is not None and len(values) != targetScenarios:
        raise ValueError(
            f"scenario set must contain exactly {targetScenarios} scenarios"
        )
    identifiers = tuple(item.scenarioId for item in values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("scenario IDs must be distinct")
    shapes = {item.factorChanges.shape for item in values}
    if len(shapes) != 1:
        raise ValueError("scenario factor vectors must have one stable shape")
    shape = next(iter(shapes))
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError("scenario factor vectors must be nonempty")
    if any(
        not numpy.isfinite(item.factorChanges).all()
        or not numpy.isfinite(item.innovations).all()
        for item in values
    ):
        raise ValueError("scenario vectors must be finite")
    probability = tuple(
        item for item in values if item.scenarioType is ScenarioType.PROBABILITY
    )
    stresses = tuple(
        item for item in values if item.scenarioType is ScenarioType.STRESS
    )
    if not probability:
        raise ValueError("scenario set must contain probability scenarios")
    total = math.fsum(item.probability for item in probability)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError("probability-scenario weights must sum to one")
    if any(item.probability != 0.0 or not item.protected for item in stresses):
        raise ValueError("all stresses must be protected and zero-weight")
    return ScenarioValidationReport(
        status="PASS",
        scenarioCount=len(values),
        factorCount=shape[0],
        probabilityCount=len(probability),
        stressCount=len(stresses),
        probabilityTotal=total,
    )
