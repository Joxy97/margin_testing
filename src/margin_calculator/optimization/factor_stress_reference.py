"""Continuous references for the same joint factor stress envelope."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from .factor_stress import QuadraticStressObjective, _owned
from risk_state_generator.factor_stress_model import FactorStressPnlModel


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
           convex: bool, *, boundCoordinates: bool = False) -> ContinuousStressResult:
    from scipy.optimize import minimize

    _radius(radius)
    if not isinstance(boundCoordinates, bool):
        raise TypeError("boundCoordinates must be a boolean")
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
                          **({"bounds": [(-radius, radius)]*dimension} if boundCoordinates else {}),
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


def solveRepricedReference(model: FactorStressPnlModel, radius: float, *,
                          boundCoordinates: bool = False) -> ContinuousStressResult:
    """Exact scenario repricing; a global certificate only for a convex model.

    With signed exposure, the result is explicitly a multistart local reference.
    Opt-in coordinate bounds contain the ball and constrain otherwise extreme
    infeasible line-search trials. Legacy defaults are unchanged.
    """
    return _solve(model.pnl, model.pnlGradient, model.dimension, radius,
                  model.isConvex, boundCoordinates=boundCoordinates)


@dataclass(frozen=True)
class QuadraticCertificate:
    """Numerical KKT/duality diagnostics, never an interval certificate."""

    coordinates: np.ndarray
    primal_value: float
    dual_lower_bound: float | None
    multiplier: float
    stationarity_norm: float
    min_shifted_eigenvalue: float
    radius_violation: float
    complementarity_residual: float
    gap: float | None
    certificate_kind: str
    status: str
    numerical_tolerance: float
    objective_hash: str
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", _owned(self.coordinates))


def referenceArray(values: np.ndarray, name: str) -> np.ndarray:
    """Reject reduced-precision arrays before conversion in certificate APIs."""
    array = np.asarray(values)
    if array.dtype.kind == "f" and array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must use float64 reference precision")
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def referenceRadius(radius: float) -> None:
    if isinstance(radius, (bool, np.float32)) or not isinstance(radius, (int, float, np.float64)):
        raise TypeError("reference radius must be a float64-compatible real scalar")
    _radius(radius)


def quadraticIdentity(hessian: np.ndarray, gradient: np.ndarray, constant: float) -> str:
    import hashlib

    digest = hashlib.sha256(b"factor-quadratic-v1")
    for value in (hessian, gradient, np.array([constant])):
        array = np.asarray(value, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def solveQuadraticTrustRegion(hessian: np.ndarray, gradient: np.ndarray,
                              constant: float, radius: float) -> QuadraticCertificate:
    """Globally minimize p0 + g.T z + z.T H z / 2 on a Euclidean ball.

    Scale to a unit ball and solve in the eigenspace, including singular PSD
    interiors and the indefinite hard case. Symmetry/rank tolerances are
    64*d*eps relative to the normalized problem. Report the unadjusted dual
    formula separately from its numerical checking tolerance.
    """
    from scipy.optimize import brentq

    referenceRadius(radius)
    h = referenceArray(hessian, "hessian")
    g = referenceArray(gradient, "gradient")
    if g.ndim != 1 or not len(g) or h.shape != (len(g), len(g)):
        raise ValueError("hessian and gradient dimensions must match and be nonempty")
    if isinstance(constant, (bool, np.float32)) or not np.isscalar(constant):
        raise TypeError("constant must be a float64-compatible scalar")
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    eps = 64 * len(g) * np.finfo(float).eps
    hscale = float(np.max(np.abs(h)))
    if np.max(np.abs(h-h.T)) > eps*hscale:
        raise ValueError("hessian must be symmetric within float64 rounding tolerance")
    h = h*.5 + h.T*.5
    scale = max(hscale, float(np.max(np.abs(g)))/radius) or 1.
    hn, gn = h/scale, (g/scale)/radius
    if not np.isfinite(gn).all():
        raise ValueError("problem scaling exceeds float64 range")
    eigenvalues, vectors = np.linalg.eigh(hn)
    projected = vectors.T @ gn
    endpoint = max(0., -float(eigenvalues[0]))
    shifted = eigenvalues + endpoint
    rank_tol = eps*max(1., float(np.max(np.abs(eigenvalues))))
    null = shifted <= rank_tol
    ranged = np.zeros(len(g))
    ranged[~null] = -projected[~null]/shifted[~null]
    in_range = np.linalg.norm(projected[null]) <= eps*max(1., np.linalg.norm(gn))
    norm = np.linalg.norm(ranged)
    if in_range and norm <= 1.:
        multiplier = endpoint
        if endpoint > 0:
            ranged[np.flatnonzero(null)[0]] = np.sqrt(max(0., 1.-norm**2))
            branch = "hard_case"
        else:
            branch = "interior"
        x = vectors @ ranged
    else:
        # Search distance from the lower endpoint to avoid subtracting nearby
        # eigenvalues repeatedly in near-hard cases.
        def secular(delta):
            denominators = shifted + delta
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                ratios = np.divide(projected, denominators,
                                   out=np.zeros_like(projected), where=denominators != 0)
            if np.any((denominators == 0) & (projected != 0)):
                return float("inf")
            return float(np.linalg.norm(ratios)-1.)

        upper = max(1., float(np.linalg.norm(gn)))
        while secular(upper) > 0:
            upper *= 2
            if not np.isfinite(upper):
                raise FloatingPointError("trust-region bracket overflow")
        delta = brentq(secular, 0., upper, xtol=np.nextafter(0., 1.),
                       rtol=4*np.finfo(float).eps, maxiter=2000)
        multiplier = endpoint + delta
        x = vectors @ (-projected/(shifted + delta))
        branch = "regular_boundary"
    z = x*radius
    norm = np.linalg.norm(z)
    while norm > radius:
        z *= np.nextafter(radius/norm, 0.)
        norm = np.linalg.norm(z)
    # Recompute every diagnostic after feasibility projection.
    lam = float(multiplier*scale)
    value = float(constant + g@z + .5*z@h@z)
    stationarity = float(np.linalg.norm(h@z + g + lam*z))
    violation = max(0., float(np.linalg.norm(z))-radius)
    complementarity = abs(lam*(float(z@z)-radius**2))
    shifted_original = eigenvalues*scale + lam
    minimum = float(shifted_original.min())
    tolerance = eps*max(abs(constant), abs(value), scale*radius**2,
                        np.linalg.norm(g)*radius, np.finfo(float).tiny)*16
    # Evaluate the spectral dual only when PSD and range checks pass.
    dual = None
    positive = shifted_original > rank_tol*scale
    pg = vectors.T @ g
    range_ok = np.linalg.norm(pg[~positive]) <= eps*max(np.linalg.norm(g), scale*radius)
    if minimum >= -rank_tol*scale and range_ok:
        dual = float(constant - .5*np.sum((pg[positive]/shifted_original[positive])*pg[positive])
                     - .5*lam*radius**2)
    gap = None if dual is None else value-dual
    valid = (dual is not None and np.isfinite([value, dual, stationarity,
             complementarity, tolerance]).all() and gap >= -tolerance
             and abs(gap) <= tolerance and stationarity*radius <= tolerance
             and complementarity <= tolerance and violation <= eps*radius)
    return QuadraticCertificate(z, value, dual if valid else None, lam, stationarity,
        minimum, violation, complementarity, gap if valid else None,
        "numerical" if valid else "none", branch if valid else "kkt_check_failed",
        tolerance, quadraticIdentity(h, g, constant), float(radius))


def solveGlobalQuadraticReference(objective: QuadraticStressObjective,
                                  radius: float) -> QuadraticCertificate:
    """Opt-in global reference; the historical SLSQP wrapper is unchanged."""
    return solveQuadraticTrustRegion(objective.hessian, objective.gradient,
                                    objective.constant, radius)
