"""Numerical Taylor and supporting-plane bounds for fixed stress envelopes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from risk_state_generator.factor_stress_model import FactorStressModel, _owned
from risk_state_generator.residual_operator import ResidualStressFit
from .factor_stress_reference import (
    QuadraticCertificate, referenceRadius, quadraticIdentity, referenceArray,
)


def stressSetHash(model: FactorStressModel, radius: float, *,
                  scope: str = 'legacy_reduced', fit: ResidualStressFit | None = None) -> str:
    """Canonical operator representation, radius and family; exposures are separate."""
    referenceRadius(radius)
    digest = hashlib.sha256(json.dumps({'version': 1, 'scope': scope,
        'family': 'log_return_ball', 'instruments': model.instruments,
        'radius': float(radius)}, sort_keys=True).encode())
    arrays = [model.center, model.directions]
    if scope == 'full_residual_ellipsoid':
        if fit is None:
            raise ValueError('full residual stress identity requires its operator')
        arrays = [model.center, fit.common_directions, fit.residual_operator.panel,
                  fit.residual_operator.weights]
    for value in arrays:
        array = np.asarray(value, dtype='<f8')
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def exponential_remainder_bound(a: float) -> float:
    """Positive-series evaluation of exp(a)-1-a-a²/2, a >= 0.

    Includes a geometric tail for small arguments. Ordinary float64 evaluation
    is numerical, not an outward-rounded proof. Underflow returns the least
    positive representable number; exponential overflow returns infinity.
    """
    if isinstance(a, (bool, np.float32)):
        raise TypeError('remainder argument must be float64-compatible')
    if not np.isfinite(a) or a < 0:
        raise ValueError('remainder argument must be finite and nonnegative')
    if a == 0:
        return 0.
    tiny = np.nextafter(0., 1.)
    if a > math.log(np.finfo(float).max):
        return float('inf')
    if a > 1:
        return math.expm1(a)-a-.5*a*a
    term = (a/6)*a*a
    if term == 0:
        return float(tiny)
    total, degree = term, 3
    while True:
        following = term*a/(degree+1)
        tail = following/(1-a/(degree+2))
        if tail <= np.finfo(float).eps*total:
            return float(np.nextafter(total+tail, np.inf))
        total += following
        term, degree = following, degree+1


@dataclass(frozen=True)
class BallErrorBound:
    per_asset: np.ndarray
    total: float
    relative_capital: float | None
    gross_exposure_normalized: float | None
    max_row_log_shock: float
    certificate_kind: str
    status: str

    def __post_init__(self):
        # Infinity is intentional for overflow bounds, unlike model arrays.
        values = np.asarray(self.per_asset, dtype=float)
        object.__setattr__(self, 'per_asset', np.frombuffer(values.tobytes(), dtype=float))


def _positive_exp(log_value):
    if log_value > math.log(np.finfo(float).max):
        return float('inf')
    return max(float(np.nextafter(0., 1.)), math.exp(log_value))


def _ball_bound(model, radius, kind, capital):
    referenceRadius(radius)
    if capital is not None and (not np.isfinite(capital) or capital <= 0):
        raise ValueError('capital must be finite and positive')
    for name in ('directions', 'center', 'exposures'):
        referenceArray(getattr(model, name), name)
    held = np.flatnonzero(model.exposures != 0)
    rows = np.zeros(len(model.exposures))
    rows[held] = np.hypot.reduce(model.directions[held], axis=1)
    with np.errstate(over='ignore'):
        shocks = radius*rows
    parts = np.zeros(len(rows))
    for i in np.flatnonzero((model.exposures != 0) & (rows != 0)):
        if not np.isfinite(shocks[i]):
            parts[i] = np.inf
            continue
        log_weight = math.log(abs(model.exposures[i])) + model.center[i]
        if kind == 'taylor':
            log_shock = math.log(radius)+math.log(rows[i])
            remainder = exponential_remainder_bound(float(shocks[i]))
            # For a tiny a, log-space leading term avoids multiplying an
            # underflow sentinel by a large exposure (which would be too loose).
            if shocks[i] < 1e-100:
                log_remainder = 3*log_shock-math.log(6.)
            elif math.isinf(remainder):
                parts[i] = np.inf
                continue
            else:
                log_remainder = math.log(remainder)
            parts[i] = _positive_exp(log_weight+log_remainder)
        else:
            parts[i] = _positive_exp(log_weight+shocks[i]+math.log(rows[i]))
    with np.errstate(over='ignore'):
        total = float(parts.sum())
        gross = float(np.abs(model.exposures).sum())
    status = 'ok' if np.isfinite(total) else 'bound_overflow'
    return BallErrorBound(parts, total, None if capital is None else total/capital,
        None if gross == 0 else total/gross, float(shocks.max(initial=0)),
        'numerical' if status == 'ok' else 'none', status)


def taylor_ball_bound(model: FactorStressModel, radius: float, *,
                      capital: float | None = None) -> BallErrorBound:
    return _ball_bound(model, radius, 'taylor', capital)


def pnl_lipschitz_bound(model: FactorStressModel, radius: float, *,
                        capital: float | None = None) -> BallErrorBound:
    return _ball_bound(model, radius, 'lipschitz', capital)


def lattice_discretization_bound(model: FactorStressModel, radius: float,
                                 bits_per_coordinate: int) -> BallErrorBound:
    if isinstance(bits_per_coordinate, bool) or not isinstance(bits_per_coordinate, int):
        raise TypeError('bits_per_coordinate must be an integer')
    if not 2 <= bits_per_coordinate <= 8:
        raise ValueError('bits_per_coordinate must lie between 2 and 8')
    bound = pnl_lipschitz_bound(model, radius)
    distance = math.sqrt(model.dimension)*radius/((1 << (bits_per_coordinate-1))-1)
    with np.errstate(over='ignore', invalid='ignore'):
        parts = bound.per_asset*distance
        total = float(parts.sum())
    return BallErrorBound(parts, total, None,
        None if bound.gross_exposure_normalized is None else bound.gross_exposure_normalized*distance,
        bound.max_row_log_shock, 'numerical' if np.isfinite(total) else 'none',
        'ok' if np.isfinite(total) else 'bound_overflow')


@dataclass(frozen=True)
class MarginBracket:
    attainable_margin: float
    upper_margin_bound: float | None
    exact_candidate_pnl: float
    quadratic_lower_bound: float | None
    taylor_error_bound: float
    stress_set_hash: str
    scope: str
    certificate_kind: str
    status: str
    numerical_tolerance: float


def _feasible(model, coordinates, radius):
    referenceRadius(radius)
    z = referenceArray(coordinates, 'candidate coordinates')
    if z.shape != (model.dimension,) or np.linalg.norm(z) > radius:
        raise ValueError('candidate must be feasible in the same stress ball')
    return z


def margin_bracket(model: FactorStressModel, radius: float, coordinates: np.ndarray,
                   quadratic: QuadraticCertificate, *, scope='legacy_reduced') -> MarginBracket:
    """Use a same-objective dual lower bound, never a local primal objective."""
    return _margin_bracket_with_bound(model, radius, coordinates, quadratic,
        taylor_ball_bound(model, radius), scope=scope)


def _margin_bracket_with_bound(model, radius, coordinates, quadratic, bound, *, scope):
    """Internal shared assembly; execution adapters own the matching bound."""
    z = _feasible(model, coordinates, radius)
    if not isinstance(quadratic, QuadraticCertificate):
        raise TypeError('margin bracket requires a quadratic dual certificate')
    p0, g, h = model.quadraticCoefficients()
    h = h*.5+h.T*.5
    if quadratic.radius != radius or quadratic.objective_hash != quadraticIdentity(h, g, p0):
        raise ValueError('quadratic certificate must match this objective and radius')
    pnl = float(model.pnl(z))
    if not np.isfinite(pnl):
        raise FloatingPointError('exact candidate P&L overflow')
    attained = max(0., -pnl)
    lower = quadratic.dual_lower_bound
    upper, status, kind = None, 'quadratic_certificate_unavailable', 'none'
    tolerance = quadratic.numerical_tolerance + 64*np.finfo(float).eps*max(attained, bound.total)
    if lower is not None and quadratic.certificate_kind == 'numerical':
        upper = max(0., -lower+bound.total)
        status = bound.status
        if not np.isfinite(upper):
            status, kind = 'bound_overflow', 'none'
        elif attained > upper+tolerance:
            upper, status = None, 'inconsistent_endpoints'
        else:
            kind = 'numerical'
    return MarginBracket(attained, upper, pnl, lower, bound.total,
        stressSetHash(model, radius, scope=scope), scope, kind, status, tolerance)


@dataclass(frozen=True)
class FullResidualCertificate:
    exact_candidate_pnl: float
    lower_bound: float | None
    upper_margin_bound: float | None
    gap: float | None
    full_gradient_norm: float
    unexplored_gradient_norm: float
    relative_unexplored_gradient: float
    certificate_kind: str
    status: str
    stress_set_hash: str
    scope: str = 'full_residual_ellipsoid'


def full_residual_convex_bound(fit: ResidualStressFit, radius: float,
                               coordinates: np.ndarray) -> FullResidualCertificate:
    z = _feasible(fit.model, coordinates, radius)
    y = fit.embed(z)
    op, common, basis = fit.residual_operator, fit.common_directions, fit.residual_basis
    k = common.shape[1]
    shocks = fit.model.center + common@y[:k] + op.matvec(y[k:])
    with np.errstate(over='raise', invalid='raise'):
        local = fit.model.exposures*np.exp(shocks)
        pnl = float(np.expm1(shocks)@fit.model.exposures)
    if not np.isfinite(pnl):
        raise FloatingPointError('full residual exact P&L overflow')
    residual_gradient = op.rmatvec(local)
    gradient = np.concatenate((common.T@local, residual_gradient))
    unexplored = residual_gradient-basis@(basis.T@residual_gradient)
    norm = float(np.linalg.norm(gradient))
    omitted = float(np.linalg.norm(unexplored))
    convex = bool(np.all(fit.model.exposures >= 0))
    lower = float(pnl-gradient@y-radius*norm) if convex else None
    gap = None if lower is None else pnl-lower
    kind, status = ('numerical', 'ok') if convex else ('none', 'signed_exposures')
    if lower is not None and (not np.isfinite(lower) or gap < -1e-12*max(1., abs(pnl))):
        lower, gap, kind, status = None, None, 'none', 'numerical_check_failed'
    return FullResidualCertificate(pnl, lower, None if lower is None else max(0., -lower),
        gap, norm, omitted, omitted/norm if norm else 0., kind, status,
        stressSetHash(fit.model, radius, scope='full_residual_ellipsoid', fit=fit))
