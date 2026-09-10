"""Deterministic repair and feasible local improvement of factor-stress samples."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isqrt
from time import perf_counter

import numpy as np

from risk_state_generator.factor_stress_model import FactorStressModel
from .factor_stress import FactorStressQUBO, _owned


@dataclass(frozen=True)
class FactorStressRepairConfig:
    maxSteps: int = 1024
    improvementTolerance: float = 1e-12

    def __post_init__(self) -> None:
        if isinstance(self.maxSteps, bool) or not isinstance(self.maxSteps, int):
            raise TypeError("maxSteps must be an integer")
        if self.maxSteps < 0:
            raise ValueError("maxSteps must be nonnegative")
        if isinstance(self.improvementTolerance, bool) or not isinstance(self.improvementTolerance, (int, float)):
            raise TypeError("improvementTolerance must be numeric")
        if not np.isfinite(self.improvementTolerance) or self.improvementTolerance < 0:
            raise ValueError("improvementTolerance must be finite and nonnegative")


@dataclass(frozen=True)
class FactorStressRepairResult:
    originalIntegers: np.ndarray
    projectedIntegers: np.ndarray
    integers: np.ndarray
    projectedSample: np.ndarray
    sample: np.ndarray
    rawPnL: float
    projectedPnL: float
    pnl: float
    steps: int
    neighborScores: int
    converged: bool
    decodeSeconds: float
    projectionSeconds: float
    initialRepricingSeconds: float
    auxiliaryRebuildSeconds: float
    improvementSeconds: float
    finalValidationSeconds: float
    totalSeconds: float

    def __post_init__(self) -> None:
        for name in ("originalIntegers", "projectedIntegers", "integers"):
            object.__setattr__(self, name, _owned(getattr(self, name), np.int64))
        for name in ("projectedSample", "sample"):
            object.__setattr__(self, name, _owned(getattr(self, name), np.uint8))


def projectIntegerBall(coordinates: np.ndarray, radius: int) -> np.ndarray:
    """Round radial projection toward zero using exact integer arithmetic.

    floor(|t_i| m / sqrt(t.T t)) = isqrt((t_i**2 * m**2) // (t.T t)).
    This avoids floating-point boundary errors and leaves feasible points intact.
    """
    t = np.asarray(coordinates)
    if t.ndim != 1 or not len(t) or not np.issubdtype(t.dtype, np.integer):
        raise ValueError("projection coordinates must be a nonempty integer vector")
    if isinstance(radius, bool) or not isinstance(radius, int):
        raise TypeError("projection radius must be an integer")
    if radius < 1:
        raise ValueError("projection radius must be positive")
    values = [int(v) for v in t]
    squared = sum(v*v for v in values)
    if squared <= radius*radius:
        return np.array(values, dtype=np.int64)
    return np.array([(1 if v >= 0 else -1)*isqrt((v*v*radius*radius)//squared)
                     for v in values], dtype=np.int64)


class FactorStressRepair:
    """Repair one sample, then search its feasible neighboring lattice points.

    All 3**d-1 unit-neighborhood moves are considered (26 for three coordinates),
    including simultaneous changes that allow motion along the sphere boundary.
    Moves minimize exact exponential P&L, not the penalized QUBO or its Taylor
    approximation. Precomputed exponential increments are shared across calls;
    no returned samples or solutions are cached.
    """

    def __init__(self, model: FactorStressModel, encoding: FactorStressQUBO,
                 config: FactorStressRepairConfig = FactorStressRepairConfig()) -> None:
        if model.dimension != encoding.objective.dimension or model.dimension > 3:
            raise ValueError("repair requires matching model/encoding dimensions, at most three")
        self.model, self.encoding, self.config = model, encoding, config
        self.scale = encoding.config.radius/encoding.latticeRadius
        moves = np.array([move for move in product((-1, 0, 1), repeat=model.dimension) if any(move)], dtype=np.int64)
        self.moves = _owned(moves, np.int64)
        with np.errstate(over="raise", invalid="raise"):
            self.increments = _owned(np.expm1((moves*self.scale) @ model.directions.T))

    def repair(self, sample: np.ndarray) -> FactorStressRepairResult:
        started = perf_counter()
        before = perf_counter()
        original = self.encoding.integerCoordinates(sample)
        decode_seconds = perf_counter()-before
        before = perf_counter()
        projected = projectIntegerBall(original, self.encoding.latticeRadius)
        projection_seconds = perf_counter()-before
        before = perf_counter()
        raw_pnl = float(self.model.pnl(original*self.scale))
        projected_pnl = raw_pnl if np.array_equal(original, projected) else float(self.model.pnl(projected*self.scale))
        initial_repricing_seconds = perf_counter()-before
        before = perf_counter()
        projected_sample = self.encoding.encodeIntegers(projected)
        auxiliary_seconds = perf_counter()-before
        before = perf_counter()
        current, current_pnl = projected.copy(), projected_pnl
        steps, scores, converged = 0, 0, False
        for _ in range(self.config.maxSteps):
            neighbors = current+self.moves
            feasible = np.sum(neighbors*neighbors, axis=1) <= self.encoding.latticeRadius**2
            if not np.any(feasible):
                converged = True
                break
            with np.errstate(over="raise", invalid="raise"):
                weighted = self.model.exposures*np.exp(self.model.center+self.model.directions@(current*self.scale))
                changes = self.increments @ weighted
            scores += len(changes)
            changes[~feasible] = np.inf
            best = int(np.argmin(changes))
            if changes[best] >= -self.config.improvementTolerance:
                converged = True
                break
            candidate = neighbors[best]
            # Recompute accepted moves directly to prevent accumulated field drift.
            candidate_pnl = float(self.model.pnl(candidate*self.scale))
            if candidate_pnl >= current_pnl-self.config.improvementTolerance:
                # Rare cancellation case: directly check every feasible neighbor
                # before declaring a local minimum.
                valid_indices = np.flatnonzero(feasible)
                values = self.model.pnl(neighbors[valid_indices]*self.scale)
                selected = int(np.argmin(values))
                candidate, candidate_pnl = neighbors[valid_indices[selected]], float(values[selected])
                if candidate_pnl >= current_pnl-self.config.improvementTolerance:
                    converged = True
                    break
            current, current_pnl = candidate.copy(), candidate_pnl
            steps += 1
        improvement_seconds = perf_counter()-before
        before = perf_counter()
        repaired = self.encoding.encodeIntegers(current)
        auxiliary_seconds += perf_counter()-before
        before = perf_counter()
        if not (self.encoding.diagnostics(projected_sample)["encoding_feasible"]
                and self.encoding.diagnostics(repaired)["encoding_feasible"]):
            raise AssertionError("repair produced an invalid encoding")
        if current_pnl > projected_pnl:
            raise AssertionError("local improvement worsened exact P&L")
        final_validation_seconds = perf_counter()-before
        return FactorStressRepairResult(original, projected, current, projected_sample, repaired,
            raw_pnl, projected_pnl, current_pnl, steps, scores, converged,
            decode_seconds, projection_seconds, initial_repricing_seconds, auxiliary_seconds,
            improvement_seconds, final_validation_seconds, perf_counter()-started)
