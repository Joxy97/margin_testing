"""Deterministic enumeration of probability-bearing FHS-EVT scenarios."""

from __future__ import annotations

import math
from typing import Sequence

import numpy

from .dependence import DependenceCells
from .deterministic_rule import DeterministicRule
from .evt_marginal import SemiparametricMarginal
from .scenario import FHSScenario, ScenarioType


def generateProbabilityScenarios(
    cells: DependenceCells,
    marginals: Sequence[SemiparametricMarginal],
    rule: DeterministicRule,
    meanForecast: numpy.ndarray,
    volatilityForecast: numpy.ndarray,
    endpointProbability: float = 1e-10,
    modelVersion: str = "fhs-evt-v1",
) -> tuple[FHSScenario, ...]:
    """Map fixed within-cell nodes through fitted marginals and current scale."""
    factor_count = cells.lower.shape[1]
    mean = numpy.asarray(meanForecast, dtype=numpy.float64)
    volatility = numpy.asarray(volatilityForecast, dtype=numpy.float64)
    if len(marginals) != factor_count:
        raise ValueError("marginals must contain one quantile per factor")
    if rule.nodes.shape[1] != factor_count:
        raise ValueError("deterministic-rule dimension must match dependence cells")
    if mean.shape != (factor_count,) or volatility.shape != (factor_count,):
        raise ValueError("forecasts must contain one value per factor")
    if not numpy.isfinite(mean).all() or not numpy.isfinite(volatility).all():
        raise ValueError("forecasts must be finite")
    if numpy.any(volatility <= 0.0):
        raise ValueError("volatility forecasts must be positive")
    if (
        not math.isfinite(endpointProbability)
        or not 0.0 < endpointProbability < 0.5
    ):
        raise ValueError("endpointProbability must lie strictly between 0 and .5")

    row_count = len(cells.weights)
    node_count = len(rule.nodes)
    quantile_matrix = (
        cells.lower[:, None, :]
        + cells.weights[:, None, None] * rule.nodes[None, :, :]
    ).reshape(row_count * node_count, factor_count)
    numpy.clip(
        quantile_matrix,
        endpointProbability,
        1.0 - endpointProbability,
        out=quantile_matrix,
    )
    innovation_matrix = numpy.empty_like(quantile_matrix)
    for factor, marginal in enumerate(marginals):
        innovation_matrix[:, factor] = marginal.quantile(
            quantile_matrix[:, factor]
        )
    change_matrix = mean + volatility * innovation_matrix
    scenario_weights = (
        cells.weights[:, None] * rule.weights[None, :]
    ).reshape(-1)

    result = []
    for row, row_id in enumerate(cells.rowIds):
        for node in range(node_count):
            scenario_index = row * node_count + node
            probability = float(scenario_weights[scenario_index])
            result.append(
                FHSScenario(
                    scenarioId=f"PROB:{modelVersion}:{row_id}:{node:04d}",
                    scenarioType=ScenarioType.PROBABILITY,
                    probability=probability,
                    innovations=innovation_matrix[scenario_index],
                    factorChanges=change_matrix[scenario_index],
                    ownerRowId=row_id,
                    nodeId=f"{rule.version}:{node:04d}",
                    quantiles=quantile_matrix[scenario_index],
                    assignedParentMass=probability,
                )
            )
    total = math.fsum(item.probability for item in result)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise RuntimeError("generated probability-scenario weights do not sum to one")
    return tuple(result)
