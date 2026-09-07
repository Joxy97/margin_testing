"""Protected deterministic stress scenarios for the FHS-EVT generator."""

from __future__ import annotations

import math
from typing import Sequence

import numpy

from .evt_marginal import SemiparametricMarginal
from .scenario import FHSScenario, ScenarioType


def buildJointTailStresses(
    marginals: Sequence[SemiparametricMarginal],
    meanForecast: numpy.ndarray,
    volatilityForecast: numpy.ndarray,
    crisisInnovations: numpy.ndarray,
    crisisOwnerRowIds: tuple[str, str],
    sigmaLevels: tuple[int, ...] = (3, 4, 5),
    modelVersion: str = "fhs-evt-v1",
) -> tuple[FHSScenario, ...]:
    """Build zero-weight stresses along observed lower/upper crisis directions.

    Each direction is owned by a synchronized historical dependence row. It is
    scaled radially until at least one coordinate reaches its fitted marginal
    sigma-equivalent magnitude; coordinates are never independently forced to
    the same tail because that would invent an unobserved dependence pattern.
    """
    mean = numpy.asarray(meanForecast, dtype=numpy.float64)
    volatility = numpy.asarray(volatilityForecast, dtype=numpy.float64)
    if mean.ndim != 1 or volatility.shape != mean.shape:
        raise ValueError("stress forecasts must be equal one-dimensional arrays")
    if len(marginals) != len(mean):
        raise ValueError("stress marginals must contain one value per factor")
    if not numpy.isfinite(mean).all() or not numpy.isfinite(volatility).all():
        raise ValueError("stress forecasts must be finite")
    if numpy.any(volatility <= 0.0):
        raise ValueError("stress volatility forecasts must be positive")
    directions = numpy.asarray(crisisInnovations, dtype=numpy.float64)
    if directions.shape != (2, len(mean)) or not numpy.isfinite(directions).all():
        raise ValueError(
            "crisisInnovations must contain two finite factor-aligned rows"
        )
    if len(crisisOwnerRowIds) != 2 or any(not item for item in crisisOwnerRowIds):
        raise ValueError("crisisOwnerRowIds must identify lower and upper rows")
    if numpy.any(numpy.max(numpy.abs(directions), axis=1) <= 1e-12):
        raise ValueError("crisis innovation directions must be nonzero")
    if not sigmaLevels or any(
        isinstance(level, bool) or not isinstance(level, int) or level <= 0
        for level in sigmaLevels
    ):
        raise ValueError("sigmaLevels must contain positive integers")
    if len(set(sigmaLevels)) != len(sigmaLevels):
        raise ValueError("sigmaLevels must be unique")

    result = []
    for level in sigmaLevels:
        alpha = 0.5 * math.erfc(level / math.sqrt(2.0))
        for direction_index, side in enumerate(("LOWER", "UPPER")):
            direction = directions[direction_index]
            anchor_probabilities = numpy.where(
                direction < 0.0,
                alpha,
                1.0 - alpha,
            )
            anchor_magnitudes = numpy.fromiter(
                (
                    abs(marginal.quantile(anchor_probabilities[factor]))
                    for factor, marginal in enumerate(marginals)
                ),
                dtype=numpy.float64,
                count=len(marginals),
            )
            material = numpy.abs(direction) > 1e-12
            radial_multiplier = max(
                1.0,
                float(
                    numpy.min(
                        anchor_magnitudes[material]
                        / numpy.abs(direction[material])
                    )
                ),
            )
            innovations = radial_multiplier * direction
            changes = mean + volatility * innovations
            owner_row_id = str(crisisOwnerRowIds[direction_index])
            result.append(
                FHSScenario(
                    scenarioId=(
                        f"STRESS:{modelVersion}:JOINT_{side}:{level}SIGMA"
                    ),
                    scenarioType=ScenarioType.STRESS,
                    probability=0.0,
                    innovations=innovations,
                    factorChanges=changes,
                    protected=True,
                    protectionReason=(
                        f"joint {side.lower()} {level}-sigma crisis direction "
                        f"owned by {owner_row_id}"
                    ),
                    ownerRowId=owner_row_id,
                    stressFamily=f"JOINT_{side}",
                    sigmaLabel=level,
                )
            )
    return tuple(result)
