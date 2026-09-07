"""Deterministic protected weighted-medoid scenario reduction."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

import numpy

from .scenario import FHSScenario, ScenarioType


class ScenarioReducer:
    """Reduce parent scenarios while preserving probability/stress separation."""

    def __init__(
        self,
        targetScenarios: int = 105,
        protectProbabilityExtremes: bool = True,
        localSwapPasses: int = 1,
        improvementTolerance: float = 1e-12,
    ) -> None:
        if (
            isinstance(targetScenarios, bool)
            or not isinstance(targetScenarios, int)
            or targetScenarios <= 0
        ):
            raise ValueError("targetScenarios must be a positive integer")
        if (
            isinstance(localSwapPasses, bool)
            or not isinstance(localSwapPasses, int)
            or localSwapPasses < 0
        ):
            raise ValueError("localSwapPasses must be a nonnegative integer")
        if not math.isfinite(improvementTolerance) or improvementTolerance < 0.0:
            raise ValueError("improvementTolerance must be finite and nonnegative")
        self.targetScenarios = targetScenarios
        self.protectProbabilityExtremes = bool(protectProbabilityExtremes)
        self.localSwapPasses = localSwapPasses
        self.improvementTolerance = float(improvementTolerance)
        self.lastAssignment: tuple[int, ...] = ()
        self.lastObjective: float | None = None

    def reduce(
        self,
        probabilityScenarios: Sequence[FHSScenario],
        stressScenarios: Sequence[FHSScenario],
    ) -> tuple[FHSScenario, ...]:
        """Select exact cardinality and aggregate mass to probability medoids."""
        probability = tuple(probabilityScenarios)
        stresses = tuple(stressScenarios)
        if any(
            item.scenarioType is not ScenarioType.PROBABILITY
            for item in probability
        ):
            raise ValueError("probabilityScenarios contains a stress scenario")
        if any(item.scenarioType is not ScenarioType.STRESS for item in stresses):
            raise ValueError("stressScenarios contains a probability scenario")
        if len({item.scenarioId for item in (*probability, *stresses)}) != (
            len(probability) + len(stresses)
        ):
            raise ValueError("parent scenario IDs must be unique")
        probability_budget = self.targetScenarios - len(stresses)
        if not 1 <= probability_budget <= len(probability):
            raise ValueError(
                "targetScenarios must leave room for at least one and no more "
                "than all probability scenarios"
            )
        dimensions = {item.factorChanges.shape for item in probability}
        if len(dimensions) != 1:
            raise ValueError("probability scenario vectors must have equal dimensions")

        weights = numpy.asarray(
            [item.probability for item in probability], dtype=numpy.float64
        )
        weights /= weights.sum()
        features = numpy.vstack([item.factorChanges for item in probability])
        feature_mean = weights @ features
        feature_scale = numpy.sqrt(weights @ (features - feature_mean) ** 2)
        feature_scale = numpy.where(feature_scale > 1e-12, feature_scale, 1.0)
        normalized = (features - feature_mean) / feature_scale
        distances = self._pairwiseDistances(normalized)
        protected = self._protectedIndices(normalized, probability)
        if len(protected) > probability_budget:
            raise ValueError(
                "protected probability scenarios exceed the reduction budget"
            )
        medoids = self._greedyMedoids(
            distances,
            weights,
            probability,
            protected,
            probability_budget,
        )
        medoids = self._improveMedoids(
            distances,
            weights,
            probability,
            medoids,
            protected,
        )
        ordered_medoids = tuple(
            sorted(medoids, key=lambda index: probability[index].scenarioId)
        )
        selected_distances = distances[:, ordered_medoids]
        assignment_positions = numpy.argmin(selected_distances, axis=1)
        assigned_indices = tuple(
            ordered_medoids[int(position)] for position in assignment_positions
        )
        reduced_weights = numpy.zeros(len(ordered_medoids), dtype=numpy.float64)
        numpy.add.at(reduced_weights, assignment_positions, weights)
        selected_probability = tuple(
            replace(
                probability[index],
                probability=float(reduced_weights[position]),
                protected=(
                    probability[index].protected or index in protected
                ),
                protectionReason=(
                    probability[index].protectionReason
                    if probability[index].protectionReason is not None
                    else "probability extreme"
                    if index in protected
                    else None
                ),
                assignedParentMass=float(reduced_weights[position]),
            )
            for position, index in enumerate(ordered_medoids)
        )
        self.lastAssignment = assigned_indices
        self.lastObjective = float(
            weights @ distances[numpy.arange(len(probability)), assigned_indices]
        )
        return selected_probability + tuple(
            sorted(stresses, key=lambda item: item.scenarioId)
        )

    @staticmethod
    def _pairwiseDistances(features: numpy.ndarray) -> numpy.ndarray:
        squared_norm = numpy.sum(features**2, axis=1)
        squared = (
            squared_norm[:, None]
            + squared_norm[None, :]
            - 2.0 * features @ features.T
        )
        numpy.maximum(squared, 0.0, out=squared)
        return numpy.sqrt(squared, out=squared)

    def _protectedIndices(
        self,
        features: numpy.ndarray,
        scenarios: tuple[FHSScenario, ...],
    ) -> frozenset[int]:
        explicit = {
            index for index, scenario in enumerate(scenarios) if scenario.protected
        }
        if not self.protectProbabilityExtremes:
            return frozenset(explicit)
        identifiers = tuple(item.scenarioId for item in scenarios)

        def stableExtreme(values: numpy.ndarray, largest: bool) -> int:
            target = numpy.max(values) if largest else numpy.min(values)
            candidates = numpy.flatnonzero(
                numpy.isclose(values, target, rtol=0.0, atol=1e-15)
            )
            return min(candidates, key=lambda index: identifiers[int(index)])

        aggregate = numpy.sum(features, axis=1)
        radial = numpy.linalg.norm(features, axis=1)
        explicit.update(
            {
                stableExtreme(aggregate, False),
                stableExtreme(aggregate, True),
                stableExtreme(radial, True),
            }
        )
        return frozenset(explicit)

    def _greedyMedoids(
        self,
        distances: numpy.ndarray,
        weights: numpy.ndarray,
        scenarios: tuple[FHSScenario, ...],
        protected: frozenset[int],
        budget: int,
    ) -> set[int]:
        medoids = set(protected)
        identifiers = tuple(item.scenarioId for item in scenarios)
        if medoids:
            nearest = numpy.min(distances[:, sorted(medoids)], axis=1)
        else:
            costs = weights @ distances
            first = min(
                range(len(scenarios)),
                key=lambda index: (float(costs[index]), identifiers[index]),
            )
            medoids.add(first)
            nearest = distances[:, first].copy()
        while len(medoids) < budget:
            best = None
            for candidate in range(len(scenarios)):
                if candidate in medoids:
                    continue
                updated = numpy.minimum(nearest, distances[:, candidate])
                cost = float(weights @ updated)
                key = (cost, identifiers[candidate], candidate)
                if best is None or key < best[0]:
                    best = (key, candidate, updated)
            if best is None:  # pragma: no cover - guarded by budget validation
                raise RuntimeError("unable to fill the probability medoid budget")
            medoids.add(best[1])
            nearest = best[2]
        return medoids

    def _improveMedoids(
        self,
        distances: numpy.ndarray,
        weights: numpy.ndarray,
        scenarios: tuple[FHSScenario, ...],
        medoids: set[int],
        protected: frozenset[int],
    ) -> set[int]:
        identifiers = tuple(item.scenarioId for item in scenarios)
        for _ in range(self.localSwapPasses):
            ordered = sorted(medoids)
            current_nearest = numpy.min(distances[:, ordered], axis=1)
            current_cost = float(weights @ current_nearest)
            best: tuple[tuple[float, str, str], int, int] | None = None
            nonmedoids = [
                index
                for index in range(len(scenarios))
                if index not in medoids
            ]
            for removed in ordered:
                if removed in protected:
                    continue
                remaining = [index for index in ordered if index != removed]
                base = (
                    numpy.min(distances[:, remaining], axis=1)
                    if remaining
                    else numpy.full(len(scenarios), numpy.inf)
                )
                for added in nonmedoids:
                    cost = float(weights @ numpy.minimum(base, distances[:, added]))
                    key = (cost, identifiers[removed], identifiers[added])
                    if cost < current_cost - self.improvementTolerance and (
                        best is None or key < best[0]
                    ):
                        best = (key, removed, added)
            if best is None:
                break
            medoids.remove(best[1])
            medoids.add(best[2])
        return medoids
