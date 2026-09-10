"""Hard spherical stress budgets reduced to unconstrained binary quadratics.

The integer lattice keeps the budget and slack exact. Source P&L coefficients
remain float64. These models have no one-hot groups: validate their product and
budget constraints explicitly with diagnostics(), not the one-hot repair path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .optimization_problem.qubo_problem import QUBOProblem


def _owned(values: np.ndarray, dtype=np.float64) -> np.ndarray:
    values = np.asarray(values, dtype=dtype)
    return np.frombuffer(values.tobytes(), dtype=dtype).reshape(values.shape)


@dataclass(frozen=True)
class QuadraticStressObjective:
    constant: float
    gradient: np.ndarray
    hessian: np.ndarray

    def __post_init__(self) -> None:
        for name in ("gradient", "hessian"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        if (self.gradient.ndim != 1 or not len(self.gradient)
                or self.hessian.shape != (len(self.gradient), len(self.gradient))):
            raise ValueError("quadratic stress coefficient dimensions do not match")
        if not (np.isfinite(self.constant) and np.isfinite(self.gradient).all()
                and np.isfinite(self.hessian).all()):
            raise ValueError("quadratic stress coefficients must be finite")
        if not np.array_equal(self.hessian, self.hessian.T):
            if not np.allclose(self.hessian, self.hessian.T, rtol=0, atol=1e-14):
                raise ValueError("quadratic stress hessian must be symmetric")
            object.__setattr__(self, "hessian", _owned((self.hessian + self.hessian.T) / 2))

    @property
    def dimension(self) -> int:
        return len(self.gradient)

    def value(self, coordinates: np.ndarray) -> np.ndarray:
        z = np.asarray(coordinates, dtype=float)
        if z.ndim < 1 or z.shape[-1] != self.dimension or not np.isfinite(z).all():
            raise ValueError("coordinates must be finite with the objective dimension last")
        return self.constant + z @ self.gradient + .5 * np.einsum("...i,ij,...j->...", z, self.hessian, z)


@dataclass(frozen=True)
class FactorStressQUBOConfig:
    """Resolution is independent of radius; penalty safety exceeds a range bound."""

    bitsPerCoordinate: int = 6
    radius: float = 3.0
    penaltySafety: float = 1.1
    penaltyMultiplier: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.bitsPerCoordinate, bool) or not isinstance(self.bitsPerCoordinate, int):
            raise TypeError("bitsPerCoordinate must be an integer")
        if not 2 <= self.bitsPerCoordinate <= 8:
            raise ValueError("bitsPerCoordinate must be between 2 and 8")
        for name in ("radius", "penaltySafety"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not np.isfinite(value) or value <= (1 if name == "penaltySafety" else 0):
                raise ValueError(f"{name} must be finite and greater than {'1' if name == 'penaltySafety' else '0'}")
        if isinstance(self.penaltyMultiplier, bool) or not isinstance(self.penaltyMultiplier, (int, float)):
            raise TypeError("penaltyMultiplier must be numeric")
        if not np.isfinite(self.penaltyMultiplier) or self.penaltyMultiplier < 0:
            raise ValueError("penaltyMultiplier must be finite and nonnegative")


@dataclass(frozen=True)
class FactorStressQUBO:
    objective: QuadraticStressObjective
    config: FactorStressQUBOConfig
    problem: QUBOProblem
    productPairs: tuple[tuple[int, int], ...]
    budgetCoefficients: np.ndarray
    budgetConstant: int
    slackWeights: np.ndarray
    penalty: float
    objectiveRangeBound: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "productPairs", tuple(tuple(p) for p in self.productPairs))
        for name in ("budgetCoefficients", "slackWeights"):
            object.__setattr__(self, name, _owned(getattr(self, name), np.int64))

    @property
    def scenarioBits(self) -> int:
        return self.objective.dimension * self.config.bitsPerCoordinate

    @property
    def latticeRadius(self) -> int:
        return (1 << (self.config.bitsPerCoordinate - 1)) - 1

    @classmethod
    def build(cls, objective: QuadraticStressObjective,
              config: FactorStressQUBOConfig = FactorStressQUBOConfig()) -> FactorStressQUBO:
        d, k = objective.dimension, config.bitsPerCoordinate
        m = (1 << (k - 1)) - 1
        powers = 1 << np.arange(k, dtype=np.int64)
        pairs = tuple((axis * k + i, axis * k + j)
                      for axis in range(d) for i, j in combinations(range(k), 2))
        slack = 1 << np.arange((m*m).bit_length(), dtype=np.int64)
        n = d*k + len(pairs) + len(slack)
        transform = np.zeros((d, n))
        for axis in range(d):
            transform[axis, axis*k:(axis+1)*k] = config.radius / m * powers
        lower = np.full(d, -config.radius)
        square = .5 * transform.T @ objective.hessian @ transform
        linear = transform.T @ (objective.gradient + objective.hessian @ lower) + np.diag(square)
        edges = 2 * np.triu(square, 1)
        offset = float(objective.value(lower))
        bound = float(np.abs(linear).sum() + np.abs(edges).sum())
        penalty = config.penaltyMultiplier * config.penaltySafety * max(bound, np.finfo(float).eps)
        # t = binaryInteger - m; sum(t*t) + slack - m*m = 0.
        weights = np.zeros(n, dtype=np.int64)
        weights[:d*k] = np.tile(powers*powers - 2*m*powers, d)
        for p, (i, j) in enumerate(pairs):
            weights[d*k+p] = 2 * powers[i % k] * powers[j % k]
        weights[d*k+len(pairs):] = slack
        constant = (d-1)*m*m
        linear += penalty * (weights*weights + 2*constant*weights)
        edges += 2*penalty*np.triu(np.outer(weights, weights), 1)
        offset += penalty*constant*constant
        for p, (i, j) in enumerate(pairs):
            y = d*k+p
            linear[y] += 3*penalty
            edges[i, j] += penalty
            edges[i, y] -= 2*penalty
            edges[j, y] -= 2*penalty
        heads, tails = np.nonzero(edges)
        problem = QUBOProblem(linear, heads, tails, edges[heads, tails], offset=offset)
        return cls(objective, config, problem, pairs, weights, constant, slack, penalty, bound)

    def _sample(self, sample: np.ndarray) -> np.ndarray:
        values = np.asarray(sample)
        if values.shape != (self.problem.variableCount,) or not np.all((values == 0) | (values == 1)):
            raise ValueError("sample must be binary and match the QUBO variable count")
        return values.astype(np.int64)

    def integerCoordinates(self, sample: np.ndarray) -> np.ndarray:
        bits = self._sample(sample)[:self.scenarioBits]
        k = self.config.bitsPerCoordinate
        return bits.reshape(self.objective.dimension, k) @ (1 << np.arange(k)) - self.latticeRadius

    def coordinates(self, sample: np.ndarray) -> np.ndarray:
        return self.integerCoordinates(sample) * (self.config.radius / self.latticeRadius)

    def encodeIntegers(self, coordinates: np.ndarray) -> np.ndarray:
        """Construct a fully consistent sample for a feasible lattice scenario."""
        t = np.asarray(coordinates)
        if t.shape != (self.objective.dimension,) or not np.issubdtype(t.dtype, np.integer):
            raise ValueError("integer coordinates must match objective dimension")
        m = self.latticeRadius
        if np.any(t < -m) or np.any(t > m) or int(t @ t) > m*m:
            raise ValueError("integer coordinates exceed the stress budget")
        k = self.config.bitsPerCoordinate
        bits = np.zeros(self.problem.variableCount, dtype=np.uint8)
        bits[:self.scenarioBits] = (((t+m)[:, None] >> np.arange(k)) & 1).ravel()
        for p, (i, j) in enumerate(self.productPairs):
            bits[self.scenarioBits+p] = bits[i]*bits[j]
        remaining = m*m-int(t@t)
        bits[self.scenarioBits+len(self.productPairs):] = (remaining >> np.arange(len(self.slackWeights))) & 1
        return bits

    def diagnostics(self, sample: np.ndarray) -> dict:
        """Check original constraints using integers, independent of QUBO energy."""
        bits = self._sample(sample)
        t = self.integerCoordinates(bits)
        violations = 0
        and_penalty = 0
        for p, (i, j) in enumerate(self.productPairs):
            y = int(bits[self.scenarioBits+p])
            a, b = int(bits[i]), int(bits[j])
            violations += y != a*b
            and_penalty += a*b - 2*a*y - 2*b*y + 3*y
        residual = int(self.budgetConstant + self.budgetCoefficients @ bits)
        budget_ok = int(t@t) <= self.latticeRadius**2
        original = float(self.objective.value(t * (self.config.radius / self.latticeRadius)))
        return dict(product_violations=int(violations), budget_equation_residual=residual,
                    integer_squared_radius=int(t@t), integer_budget=self.latticeRadius**2,
                    scenario_feasible=bool(budget_ok),
                    encoding_feasible=bool(budget_ok and violations == 0 and residual == 0),
                    objective=original,
                    decomposed_energy=original + self.penalty*(residual**2 + and_penalty))


def solveLattice(model: FactorStressQUBO) -> tuple[np.ndarray, float, int]:
    """Exact quadratic optimum over the encoded ball, for up to three dimensions.

    Enumerate the first d-1 integer coordinates. Along the final coordinate a
    quadratic needs only endpoints and the two neighboring integers at its
    stationary point. Auxiliary bits never need enumeration.
    """
    d, m = model.objective.dimension, model.latticeRadius
    if d > 3:
        raise ValueError("exact lattice reference supports at most three dimensions")
    scale = model.config.radius / m
    h, g = model.objective.hessian, model.objective.gradient
    best = np.zeros(d, dtype=np.int64)
    best_value = float(model.objective.value(best))
    evaluations = 1
    for prefix in product(range(-m, m+1), repeat=d-1):
        remaining = m*m - sum(v*v for v in prefix)
        if remaining < 0:
            continue
        from math import isqrt
        limit = isqrt(remaining)
        candidates = {-limit, limit}
        if h[-1, -1] > 0:
            slope = g[-1] + h[-1, :-1] @ (np.asarray(prefix) * scale)
            stationary = -slope / (h[-1, -1] * scale)
            clipped = np.clip(stationary, -limit, limit)
            candidates.update((int(np.floor(clipped)), int(np.ceil(clipped))))
        for last in sorted(candidates):
            point = np.array((*prefix, last), dtype=np.int64)
            value = float(model.objective.value(point*scale))
            evaluations += 1
            if value < best_value:
                best, best_value = point, value
    return best, best_value, evaluations
