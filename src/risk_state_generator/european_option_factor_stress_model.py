"""Mixed stock/European-option P&L on the original reduced factor stress ball."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from option_pricing import EuropeanOptionBook
from portfolio import Portfolio
from .factor_stress_model import FactorStressModel, LocalQuadratic
from .pca_grid import ReturnsPCAGrid


@dataclass(frozen=True)
class EuropeanOptionFactorStressModel:
    """Share one underlying scenario across stocks and all European option legs."""

    equityModel: FactorStressModel
    options: EuropeanOptionBook

    def __post_init__(self) -> None:
        if not isinstance(self.equityModel, FactorStressModel):
            raise TypeError("equityModel must be a FactorStressModel")
        if not isinstance(self.options, EuropeanOptionBook):
            raise TypeError("options must be a EuropeanOptionBook")
        if self.equityModel.instruments != self.options.instruments:
            raise ValueError("option and equity instruments must have identical order")

    @property
    def dimension(self) -> int:
        return self.equityModel.dimension

    @property
    def isConvex(self) -> bool:
        # Even long puts need not be convex in log spot. The stock-only
        # supporting-hyperplane certificate does not apply to an option book.
        return not any(p.quantity for p in self.options.positions) and self.equityModel.isConvex

    @classmethod
    def fromPCAGrid(cls, grid: ReturnsPCAGrid, portfolio: Portfolio,
                    options: EuropeanOptionBook) -> EuropeanOptionFactorStressModel:
        """Keep PCA factors and align the one residual column to mixed local delta."""
        if tuple(grid.instruments) != options.instruments or portfolio.instruments != options.instruments:
            raise ValueError("PCA, portfolio and option instruments must have identical order")
        if grid.calibrationEndDate > options.valuationDate:
            raise ValueError("PCA calibration must not use prices after option valuationDate")
        center = grid.logReturnMean + grid.logReturnScale*grid.pcaMean
        option_delta, _ = options.logDerivatives(center)
        exposures = np.array([float(portfolio.weights[i]) for i in grid.instruments])
        equity = FactorStressModel.fromPCAGrid(
            grid, portfolio, localPnlGradient=exposures*np.exp(center)+option_delta,
        )
        return cls(equity, options)

    def _logReturns(self, coordinates: np.ndarray) -> np.ndarray:
        z = np.asarray(coordinates, dtype=np.float64)
        if z.ndim < 1 or z.shape[-1] != self.dimension or not np.isfinite(z).all():
            raise ValueError("coordinates must be finite with the model dimension last")
        return self.equityModel.center + z @ self.equityModel.directions.T

    def pnl(self, coordinates: np.ndarray) -> np.ndarray:
        return self.equityModel.pnl(coordinates) + self.options.pnl(self._logReturns(coordinates))

    def pnlGradient(self, coordinates: np.ndarray) -> np.ndarray:
        first, _ = self.options.logDerivatives(self._logReturns(coordinates))
        return self.equityModel.pnlGradient(coordinates) + self.equityModel.directions.T @ first

    def pnlHessian(self, coordinates: np.ndarray) -> np.ndarray:
        _, second = self.options.logDerivatives(self._logReturns(coordinates))
        directions = self.equityModel.directions
        hessian = self.equityModel.pnlHessian(coordinates) + directions.T @ (second[:, None]*directions)
        # Monetary exposures can amplify matrix-product roundoff beyond the
        # encoder's absolute symmetry tolerance. Preserve exact symmetry here.
        return .5*hessian + .5*hessian.T

    def quadraticCoefficients(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Taylor objective at z=0, including horizon theta and the observed marks."""
        zero = np.zeros(self.dimension)
        return float(self.pnl(zero)), self.pnlGradient(zero), self.pnlHessian(zero)

    def localQuadratic(self, coordinates: np.ndarray) -> LocalQuadratic:
        return LocalQuadratic(float(self.pnl(coordinates)), self.pnlGradient(coordinates),
                              self.pnlHessian(coordinates), coordinates)
