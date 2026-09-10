"""Joint PCA and portfolio-residual log-return stresses for research models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portfolio import Portfolio
from .pca_grid import ReturnsPCAGrid


def _owned(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("factor stress arrays must be finite")
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


@dataclass(frozen=True)
class FactorStressModel:
    """Fixed portfolio repriced under logReturn = center + directions @ z.

    Directions contain whitened PCA loadings and one residual direction aligned
    with the portfolio's local P&L gradient. The residual reduction is exact for
    that gradient under the fitted covariance, not for nonlinear tail losses.
    """

    instruments: tuple[str, ...]
    exposures: np.ndarray
    center: np.ndarray
    directions: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(self.instruments))
        for name in ("exposures", "center", "directions"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        n = len(self.instruments)
        if not n or len(set(self.instruments)) != n:
            raise ValueError("factor stress instruments must be nonempty and unique")
        if self.exposures.shape != (n,) or self.center.shape != (n,):
            raise ValueError("factor stress exposures and center must match instruments")
        if self.directions.ndim != 2 or self.directions.shape[0] != n or not self.dimension:
            raise ValueError("factor stress directions must have shape (instruments, dimensions)")

    @property
    def dimension(self) -> int:
        return self.directions.shape[1] if self.directions.ndim == 2 else 0

    @classmethod
    def fromPCAGrid(cls, grid: ReturnsPCAGrid, portfolio: Portfolio) -> FactorStressModel:
        if tuple(grid.instruments) != portfolio.instruments:
            raise ValueError("PCA instruments must match canonical portfolio order")
        exposures = np.array([float(portfolio.weights[i]) for i in grid.instruments])
        center = grid.logReturnMean + grid.logReturnScale * grid.pcaMean
        factors = grid.logReturnScale[:, None] * grid.loadings.T * np.sqrt(grid.lambdas)
        residuals = grid.residuals * grid.logReturnScale
        weights = grid.ew_lambda ** np.arange(len(residuals) - 1, -1, -1, dtype=float)
        weights /= weights.sum()
        residuals = residuals - weights @ residuals
        local_exposures = exposures * np.exp(center)
        projected = residuals @ local_exposures
        variance = float(weights @ projected**2)
        # R @ local_exposures, without constructing an assets-by-assets matrix.
        residual_direction = (residuals.T @ (weights * projected) / np.sqrt(variance)
                              if variance > 0 else np.zeros(len(exposures)))
        return cls(tuple(grid.instruments), exposures, center,
                   np.column_stack((factors, residual_direction)))

    def pnl(self, coordinates: np.ndarray) -> np.ndarray:
        """Exact simple-return P&L of the reduced log-return scenario model."""
        z = np.asarray(coordinates, dtype=float)
        if z.ndim < 1 or z.shape[-1] != self.dimension or not np.isfinite(z).all():
            raise ValueError("coordinates must be finite with the model dimension last")
        with np.errstate(over="raise", invalid="raise"):
            return np.expm1(self.center + z @ self.directions.T) @ self.exposures

    def pnlGradient(self, coordinates: np.ndarray) -> np.ndarray:
        z = np.asarray(coordinates, dtype=float)
        if z.shape != (self.dimension,) or not np.isfinite(z).all():
            raise ValueError("coordinates must be a finite vector of model dimension")
        with np.errstate(over="raise", invalid="raise"):
            return self.directions.T @ (self.exposures * np.exp(self.center + self.directions @ z))

    def quadraticCoefficients(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Second-order Taylor coefficients at z=0, in original P&L units."""
        weighted = self.exposures * np.exp(self.center)
        return (float(np.expm1(self.center) @ self.exposures),
                self.directions.T @ weighted,
                self.directions.T @ (weighted[:, None] * self.directions))
