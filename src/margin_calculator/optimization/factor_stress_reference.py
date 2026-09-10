"""Continuous references for the same joint factor stress envelope."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from .factor_stress import QuadraticStressObjective, _owned
from risk_state_generator.factor_stress_model import FactorStressModel


@dataclass(frozen=True)
class ContinuousStressResult:
    coordinates: np.ndarray
    objective: float
    lowerBound: float | None
    success: bool
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", _owned(self.coordinates))

    @property
    def optimalityGap(self) -> float | None:
        return None if self.lowerBound is None else max(0., self.objective-self.lowerBound)


def _radius(radius: float) -> None:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("stress radius must be finite and positive")


def solveLinearReference(objective: QuadraticStressObjective, radius: float) -> ContinuousStressResult:
    """Closed-form solution of the linearized P&L over the Euclidean ball."""
    _radius(radius)
    norm = np.linalg.norm(objective.gradient)
    point = -radius*objective.gradient/norm if norm else np.zeros(objective.dimension)
    value = float(objective.constant-radius*norm)
    return ContinuousStressResult(point, value, value, True, "analytic linear stress optimum")


def _solve(value: Callable, gradient: Callable, dimension: int, radius: float,
           convex: bool) -> ContinuousStressResult:
    from scipy.optimize import minimize

    _radius(radius)
    zero = np.zeros(dimension)
    g = gradient(zero)
    start = -radius*g/np.linalg.norm(g) if np.linalg.norm(g) else zero
    starts = [start, zero]
    if not convex:
        starts.extend(sign*radius*np.eye(dimension)[i]
                      for i in range(dimension) for sign in (-1, 1))
    candidates = []
    for initial in starts:
        result = minimize(value, initial, jac=gradient, method="SLSQP",
                          constraints={"type": "ineq", "fun": lambda z: radius**2-z@z,
                                       "jac": lambda z: -2*z},
                          options={"ftol": 1e-13, "maxiter": 1000})
        if not np.isfinite(result.x).all():
            continue
        point = result.x.copy()
        norm = np.linalg.norm(point)
        if norm > radius:
            point *= radius/norm
        evaluated = float(value(point))
        if np.isfinite(evaluated):
            candidates.append((evaluated, point, bool(result.success), str(result.message)))
    if not candidates:
        raise RuntimeError("continuous stress optimization returned no finite candidate")
    objective, point, success, message = min(candidates, key=lambda c: c[0])
    # Supporting hyperplane minimized over the whole ball: a global lower bound
    # for convex P&L. This certificate is more informative than SLSQP's status.
    g = gradient(point)
    lower = float(objective-g@point-radius*np.linalg.norm(g)) if convex else None
    return ContinuousStressResult(point, objective, lower, success, message)


def solveQuadraticReference(objective: QuadraticStressObjective, radius: float) -> ContinuousStressResult:
    convex = bool(np.linalg.eigvalsh(objective.hessian).min() >= 0)
    return _solve(objective.value, lambda z: objective.gradient+objective.hessian@z,
                  objective.dimension, radius, convex)


def solveRepricedReference(model: FactorStressModel, radius: float) -> ContinuousStressResult:
    """Exact expm1 repricing; a global convex certificate for long-only exposure.

    With signed exposure, the result is explicitly a multistart local reference.
    """
    return _solve(model.pnl, model.pnlGradient, model.dimension, radius,
                  bool(np.all(model.exposures >= 0)))
