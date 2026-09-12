"""Deterministic repair and feasible local improvement of factor-stress samples."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod
from itertools import combinations, product
from math import isqrt
from time import perf_counter

import numpy as np

from risk_state_generator.factor_stress_model import FactorStressModel, FactorStressPnlModel
from .factor_stress import FactorStressQUBO, _owned


@dataclass(frozen=True)
class FactorStressRepairConfig:
    maxSteps: int = 1024
    improvementTolerance: float = 1e-12
    neighborhood: str = "full"

    def __post_init__(self) -> None:
        if self.neighborhood not in {"full", "pairwise"}:
            raise ValueError("neighborhood must be full or pairwise")
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

    The default considers all 3**d-1 moves for up to three coordinates. Explicit
    pairwise mode considers the 2*d*d single- and two-coordinate unit moves,
    including simultaneous changes that allow motion along the sphere boundary.
    Moves minimize exact portfolio P&L, not the penalized QUBO or its Taylor
    approximation. Precomputed exponential increments are shared across calls;
    no returned samples or solutions are cached.
    """

    def __init__(self, model: FactorStressPnlModel, encoding: FactorStressQUBO,
                 config: FactorStressRepairConfig = FactorStressRepairConfig()) -> None:
        if model.dimension != encoding.objective.dimension:
            raise ValueError("repair requires matching model/encoding dimensions")
        if config.neighborhood == "full" and model.dimension > 3:
            raise ValueError("full repair supports at most three dimensions; select pairwise explicitly")
        self.model, self.encoding, self.config = model, encoding, config
        self.scale = encoding.config.radius/encoding.latticeRadius
        if config.neighborhood == "full":
            moves = np.array([move for move in product((-1, 0, 1), repeat=model.dimension) if any(move)], dtype=np.int64)
        else:
            candidates = []
            for count in (1, 2):
                for axes in combinations(range(model.dimension), count):
                    for signs in product((-1, 1), repeat=count):
                        move = [0]*model.dimension
                        for axis, sign in zip(axes, signs):
                            move[axis] = sign
                        candidates.append(tuple(move))
            moves = np.array(sorted(candidates), dtype=np.int64)
        self.moves = _owned(moves, np.int64)
        self._prepareScoring(model)

    @singledispatchmethod
    def _prepareScoring(self, model):
        # General exact repricing also covers option books; stock exponential
        # increments cannot represent their normal-CDF terms.
        self.increments = None
        def changes(coordinates, feasible):
            # A pricing backend may only be valid inside the declared stress
            # domain. Infeasible neighbors must never be sent to that backend.
            result = np.full(len(self.moves), np.inf)
            result[feasible] = (model.pnl(coordinates + self.moves[feasible]*self.scale)
                                - model.pnl(coordinates))
            return result
        self._changes = changes

    @_prepareScoring.register
    def _(self, model: FactorStressModel):
        with np.errstate(over="raise", invalid="raise"):
            self.increments = _owned(np.expm1((self.moves*self.scale) @ model.directions.T))
        self._changes = lambda coordinates, feasible: self.increments @ (
            model.exposures*np.exp(model.center+model.directions@coordinates)
        )

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
                changes = self._changes(current*self.scale, feasible)
            scores += len(changes) if self.increments is not None else int(np.count_nonzero(feasible))
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
