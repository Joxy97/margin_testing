"""Immutable observation-space residual covariance; never forms an asset covariance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portfolio import Portfolio
from .factor_stress_model import FactorStressModel, _owned
from .pca_grid import ReturnsPCAGrid


def _reference(values, name):
    values = np.asarray(values)
    if values.dtype.kind == 'f' and values.dtype != np.dtype('float64'):
        raise TypeError(f"{name} requires float64")
    if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite real values")
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class ResidualOperator:
    """Own one centered T-by-n panel and normalized observation weights.

    B = panel.T @ diag(sqrt(weights)). Construction copies the panel once;
    sharing this immutable object across portfolios shares that storage.
    """

    instruments: tuple[str, ...]
    panel: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        panel = _reference(self.panel, 'residual panel')
        weights = _reference(self.weights, 'residual weights')
        names = tuple(self.instruments)
        if (panel.ndim != 2 or not len(panel) or panel.shape[1] != len(names)
                or not names or len(set(names)) != len(names)):
            raise ValueError('residual panel must match nonempty unique instruments')
        if weights.shape != (len(panel),) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('residual weights must be nonnegative with positive sum')
        weights = weights/weights.sum()
        object.__setattr__(self, 'instruments', names)
        object.__setattr__(self, 'weights', _owned(weights))
        object.__setattr__(self, 'panel', _owned(panel - weights@panel))

    @classmethod
    def fromPCAGrid(cls, grid: ReturnsPCAGrid) -> ResidualOperator:
        _reference(grid.residuals, 'PCA residuals')
        _reference(grid.logReturnScale, 'PCA scales')
        weights = grid.ew_lambda**np.arange(len(grid.residuals)-1, -1, -1, dtype=float)
        return cls(grid.instruments, grid.residuals*grid.logReturnScale, weights)

    @property
    def n_assets(self):
        return self.panel.shape[1]

    @property
    def latent_dimension(self):
        return self.panel.shape[0]

    def matvec(self, coordinates):
        u = _reference(coordinates, 'residual coordinates')
        if u.shape != (self.latent_dimension,):
            raise ValueError('residual coordinates must match latent dimension')
        return self.panel.T @ (np.sqrt(self.weights)*u)

    def rmatvec(self, exposures):
        v = _reference(exposures, 'residual exposures')
        if v.shape != (self.n_assets,):
            raise ValueError('residual exposures must match assets')
        return np.sqrt(self.weights)*(self.panel@v)

    def cov_matvec(self, exposures):
        return self.matvec(self.rmatvec(exposures))

    def diag_cov(self):
        return self.weights @ (self.panel*self.panel)


@dataclass(frozen=True)
class ResidualStressFit:
    model: FactorStressModel
    common_directions: np.ndarray
    residual_operator: ResidualOperator
    residual_basis: np.ndarray
    active_dimension: int
    encoded_dimension: int
    rank_tolerance: float
    model_scope: str

    def __post_init__(self):
        for name in ('common_directions', 'residual_basis'):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        common, basis, op = self.common_directions, self.residual_basis, self.residual_operator
        if (common.ndim != 2 or op.instruments != self.model.instruments
                or common.shape[0] != op.n_assets):
            raise ValueError('residual fit instrument ordering must match')
        if basis.ndim != 2 or basis.shape[0] != op.latent_dimension:
            raise ValueError('residual basis must match operator latent dimension')
        if not np.allclose(basis.T@basis, np.eye(basis.shape[1]), rtol=0, atol=1e-12):
            raise ValueError('residual basis must be orthonormal')
        reconstructed = np.column_stack((common, *[op.matvec(q) for q in basis.T]))
        actual = self.model.directions
        if reconstructed.shape[1] < actual.shape[1]:
            reconstructed = np.column_stack((reconstructed, np.zeros(op.n_assets)))
        if reconstructed.shape != actual.shape or not np.allclose(reconstructed, actual, rtol=1e-10, atol=1e-14):
            raise ValueError('reduced directions must equal common and embedded residual directions')

    def embed(self, coordinates):
        z = _reference(coordinates, 'reduced coordinates')
        if z.shape != (self.model.dimension,):
            raise ValueError('coordinates must match reduced dimension')
        k, s = self.common_directions.shape[1], self.residual_basis.shape[1]
        return np.concatenate((z[:k], self.residual_basis@z[k:k+s]))


def buildResidualStressFit(grid: ReturnsPCAGrid, portfolio: Portfolio, *,
                           mode: str = 'legacy', rankTolerance: float = 1e-12,
                           residualOperator: ResidualOperator | None = None) -> ResidualStressFit:
    """Separate opt-in builder; legacy factory and its zero column are preserved."""
    if mode not in ('legacy', 'extended'):
        raise ValueError('mode must be legacy or extended')
    if not np.isfinite(rankTolerance) or not 0 < rankTolerance < 1:
        raise ValueError('rankTolerance must lie between zero and one')
    for name in ('residuals', 'logReturnMean', 'logReturnScale', 'loadings', 'lambdas', 'pcaMean'):
        _reference(getattr(grid, name), name)
    legacy = FactorStressModel.fromPCAGrid(grid, portfolio)
    op = residualOperator if residualOperator is not None else ResidualOperator.fromPCAGrid(grid)
    if op.instruments != legacy.instruments:
        raise ValueError('residual operator must match canonical instruments')
    # A supplied shared operator must describe this fit, not merely this universe.
    expected_weights = grid.ew_lambda**np.arange(len(grid.residuals)-1, -1, -1, dtype=float)
    expected_weights /= expected_weights.sum()
    panel = grid.residuals*grid.logReturnScale
    panel -= expected_weights@panel
    if (op.panel.shape != panel.shape or not np.allclose(op.weights, expected_weights, rtol=1e-13, atol=0)
            or not np.allclose(op.panel, panel, rtol=1e-12, atol=1e-15)):
        raise ValueError('shared residual operator does not describe this PCA fit')
    local = legacy.exposures*np.exp(legacy.center)
    numerator = op.rmatvec(local)
    norm = np.linalg.norm(numerator)
    threshold = rankTolerance*np.sqrt(op.diag_cov().sum())*np.linalg.norm(local)
    active = norm > threshold
    # Legacy keeps any nonzero direction, including numerically negligible ones.
    retained = norm > 0 if mode == 'legacy' else active
    basis = numerator[:, None]/norm if retained else np.empty((op.latent_dimension, 0))
    common = legacy.directions[:, :-1]
    model = legacy if mode == 'legacy' else FactorStressModel(legacy.instruments, legacy.exposures,
        legacy.center, np.column_stack((common, *[op.matvec(q) for q in basis.T])))
    active_dimension = int(np.linalg.matrix_rank(model.directions,
        tol=rankTolerance*np.linalg.norm(model.directions, ord=2)))
    return ResidualStressFit(model, common, op, basis, active_dimension, model.dimension,
                             rankTolerance, 'legacy_reduced' if mode == 'legacy' else 'enriched_reduced')
