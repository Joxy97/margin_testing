"""Returns-based PCA grid."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import numpy
import pandas

from download_unit import Instrument

from ..pca_key import ReturnsPCAKey
from .pca_grid import PCAGrid


@dataclass
class ReturnsPCAGrid(PCAGrid):
    """Store configuration and calculated values for a returns PCA grid."""

    instruments: Iterable[Instrument]
    ew_window: int
    current_date: date
    ew_lambda: float
    components: int
    lambdas: numpy.ndarray | None = field(init=False, default=None)
    explained: numpy.ndarray | None = field(init=False, default=None)
    loadings: numpy.ndarray | None = field(init=False, default=None)
    factors: numpy.ndarray | None = field(init=False, default=None)
    pcaMean: numpy.ndarray | None = field(init=False, default=None)
    residuals: numpy.ndarray | None = field(init=False, default=None)
    residualScale: numpy.ndarray | None = field(init=False, default=None)
    maxAbsoluteZ: numpy.ndarray | None = field(init=False, default=None)
    logReturnMean: numpy.ndarray | None = field(init=False, default=None)
    logReturnScale: numpy.ndarray | None = field(init=False, default=None)
    calibrationStartDate: date | None = field(init=False, default=None)
    calibrationEndDate: date | None = field(init=False, default=None)

    @classmethod
    def construct(
        cls,
        key: ReturnsPCAKey,
        data: pandas.DataFrame,
    ) -> "ReturnsPCAGrid":
        """Construct and fit a returns PCA grid from ``key`` and price data."""
        grid = cls(
            instruments=key.instruments,
            ew_window=key.ew_window,
            current_date=key.start_date,
            ew_lambda=key.ew_lambda,
            components=key.components,
        )
        prices = grid._extract_price_window(data)
        log_returns = grid._compute_log_returns(prices)
        weights = grid._getExponentialWeights(len(log_returns))
        standardized_returns = grid._standardize(log_returns, weights)
        grid._fit_pca(standardized_returns, weights)
        return grid

    def _extract_price_window(
        self,
        data: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Return the aligned price window strictly before ``current_date``."""
        if not isinstance(data, pandas.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if self.ew_window < 1:
            raise ValueError("ew_window must be positive")

        missing_instruments = set(self.instruments).difference(data.columns)
        if missing_instruments:
            missing = ", ".join(sorted(missing_instruments))
            raise ValueError(f"data is missing instrument columns: {missing}")

        prices = data.copy()
        if "date" in prices.columns:
            prices["date"] = pandas.to_datetime(prices["date"], errors="raise")
            prices = prices.set_index("date")
        else:
            prices.index = pandas.to_datetime(prices.index, errors="raise")
        if not prices.index.is_unique:
            raise ValueError("market-data dates must be unique")
        prices = (
            prices.sort_index()
            .loc[:, list(self.instruments)]
            .ffill()
            .dropna(axis=0, how="any")
        )
        prices = prices.loc[prices.index < pandas.Timestamp(self.current_date)]
        required_prices = self.ew_window + 1
        if len(prices) < required_prices:
            raise ValueError(
                "data does not contain ew_window + 1 rows before current_date"
            )
        selected = prices.iloc[-required_prices:]
        self.calibrationStartDate = selected.index[1].date()
        self.calibrationEndDate = selected.index[-1].date()
        return selected

    @staticmethod
    def _compute_log_returns(prices: pandas.DataFrame) -> pandas.DataFrame:
        """Compute log returns and discard the farthest price row."""
        price_values = prices.to_numpy(dtype=float)
        if not numpy.isfinite(price_values).all() or numpy.any(
            price_values <= 0.0
        ):
            raise ValueError("prices must be finite and strictly positive")
        log_returns = numpy.log(prices / prices.shift(1))
        return log_returns.dropna(axis=0, how="any")

    def _standardize(
        self,
        log_returns: pandas.DataFrame,
        weights: numpy.ndarray,
    ) -> numpy.ndarray:
        """Standardize returns with the same EW measure used by the PCA."""
        values = log_returns.to_numpy(dtype=numpy.float64)
        if weights.shape != (len(values),):
            raise ValueError("weights must contain one value per observation")
        self.logReturnMean = numpy.sum(weights[:, None] * values, axis=0)
        centered = values - self.logReturnMean
        variance = numpy.sum(weights[:, None] * centered**2, axis=0)
        if not numpy.isfinite(variance).all() or numpy.any(variance <= 0.0):
            raise ValueError("PCA requires positive finite return variance")
        self.logReturnScale = numpy.sqrt(variance)
        return centered / self.logReturnScale

    def _getExponentialWeights(
        self,
        observations: int,
    ) -> numpy.ndarray:
        """Return normalized exponentially decaying observation weights."""
        if not 0.0 < self.ew_lambda <= 1.0:
            raise ValueError("ew_lambda must be greater than 0 and at most 1")
        weights = self.ew_lambda ** numpy.arange(
            observations - 1,
            -1,
            -1,
            dtype=float,
        )
        return weights / weights.sum()

    def _fit_pca(
        self,
        standardizedReturns: numpy.ndarray,
        weights: numpy.ndarray,
    ) -> None:
        """Fit MarginLab's weighted covariance eigendecomposition."""
        if len(standardizedReturns) < 2:
            raise ValueError("PCA requires at least two return observations")
        maximum_components = min(standardizedReturns.shape)
        if not 1 <= self.components <= maximum_components:
            raise ValueError(
                "components must be between 1 and "
                f"{maximum_components}, inclusive"
            )

        self.pcaMean = numpy.sum(
            weights[:, None] * standardizedReturns,
            axis=0,
        )
        centered_returns = standardizedReturns - self.pcaMean
        observations, assets = centered_returns.shape
        if assets <= observations:
            covariance = centered_returns.T @ (
                weights[:, None] * centered_returns
            )
            eigenvalues, eigenvectors = numpy.linalg.eigh(covariance)
            order = numpy.argsort(eigenvalues)[::-1]
            eigenvalues = numpy.maximum(eigenvalues[order], 0.0)
            loadings = eigenvectors[:, order[: self.components]].T
        else:
            eigenvalues, loadings = self._fitObservationSpacePCA(
                centered_returns,
                weights,
            )

        total_variance = float(eigenvalues.sum())
        if total_variance <= 0.0:
            raise ValueError("PCA requires positive return variance")
        self.lambdas = eigenvalues[: self.components]
        self.explained = self.lambdas / total_variance
        self.loadings = self._canonicalizeLoadingSigns(loadings)
        self.factors = centered_returns @ self.loadings.T
        reconstructed_returns = self.pcaMean + self.factors @ self.loadings
        self.residuals = standardizedReturns - reconstructed_returns
        residual_mean = numpy.sum(weights[:, None] * self.residuals, axis=0)
        self.residualScale = numpy.sqrt(
            numpy.sum(
                weights[:, None] * (self.residuals - residual_mean) ** 2,
                axis=0,
            )
        )
        self.maxAbsoluteZ = numpy.nanmax(
            numpy.abs(standardizedReturns),
            axis=0,
        )

    @staticmethod
    def _canonicalizeLoadingSigns(loadings: numpy.ndarray) -> numpy.ndarray:
        """Choose deterministic eigenvector signs without changing the PCA."""
        canonical = numpy.asarray(loadings, dtype=numpy.float64).copy()
        pivots = numpy.argmax(numpy.abs(canonical), axis=1)
        signs = numpy.sign(canonical[numpy.arange(len(canonical)), pivots])
        signs[signs == 0.0] = 1.0
        canonical *= signs[:, None]
        return canonical

    def _fitObservationSpacePCA(
        self,
        centeredReturns: numpy.ndarray,
        weights: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Fit wide data through the equivalent observation-space problem."""
        weighted_centered = numpy.sqrt(weights[:, None]) * centeredReturns
        gram = weighted_centered @ weighted_centered.T
        eigenvalues, left_eigenvectors = numpy.linalg.eigh(gram)
        order = numpy.argsort(eigenvalues)[::-1]
        eigenvalues = numpy.maximum(eigenvalues[order], 0.0)
        selected_eigenvalues = eigenvalues[: self.components]
        if numpy.any(selected_eigenvalues <= numpy.finfo(float).eps):
            raise ValueError(
                "requested PCA components include a zero-variance mode"
            )

        selected_left_eigenvectors = left_eigenvectors[
            :,
            order[: self.components],
        ]
        right_eigenvectors = (
            weighted_centered.T @ selected_left_eigenvectors
        )
        right_eigenvectors /= numpy.sqrt(selected_eigenvalues)[None, :]
        return eigenvalues, right_eigenvectors.T
