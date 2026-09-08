"""Returns-based PCA grid."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import numpy
import pandas

from download_unit import Instrument

from ..pca_key import ReturnsPCAKey
from ..pca_backend import PCABackend, NumpyPCABackend
from .pca_grid import PCAGrid


@dataclass(frozen=True)
class ReturnsPCAInput:
    """Aligned host input shared by host and resident numerical execution."""

    key: ReturnsPCAKey
    values: numpy.ndarray
    weights: numpy.ndarray
    logReturnMean: numpy.ndarray
    logReturnScale: numpy.ndarray
    calibrationStartDate: date
    calibrationEndDate: date


@dataclass(frozen=True)
class ReturnsPCAGrid(PCAGrid):
    """A complete fitted PCA result with defensively owned immutable arrays."""

    instruments: tuple[Instrument, ...]
    ew_window: int
    current_date: date
    ew_lambda: float
    components: int
    lambdas: numpy.ndarray
    explained: numpy.ndarray
    loadings: numpy.ndarray
    factors: numpy.ndarray
    pcaMean: numpy.ndarray
    residuals: numpy.ndarray
    residualScale: numpy.ndarray
    maxAbsoluteZ: numpy.ndarray
    logReturnMean: numpy.ndarray
    logReturnScale: numpy.ndarray
    calibrationStartDate: date
    calibrationEndDate: date

    def __post_init__(self):
        object.__setattr__(self, "instruments", tuple(self.instruments))
        for name, value in vars(self).items():
            if isinstance(value, numpy.ndarray):
                owned = numpy.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
                object.__setattr__(self, name, owned)

    @property
    def numericMemoryBytes(self) -> int:
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, numpy.ndarray))

    @staticmethod
    def prepareInput(key: ReturnsPCAKey, data: pandas.DataFrame) -> ReturnsPCAInput:
        """Validate and standardize the same historical window for every backend."""
        grid = _ReturnsGridBuilder(key.instruments, key.ew_window, key.start_date,
                                   key.ew_lambda, key.components)
        prices = grid._extract_price_window(data)
        returns = grid._compute_log_returns(prices)
        weights = grid._getExponentialWeights(len(returns))
        values = grid._standardize(returns, weights)
        return ReturnsPCAInput(key, values, weights, grid.logReturnMean, grid.logReturnScale,
                               grid.calibrationStartDate, grid.calibrationEndDate)

    @classmethod
    def construct(cls, key: ReturnsPCAKey, data: pandas.DataFrame,
                  backend: PCABackend | None = None) -> ReturnsPCAGrid:
        return _ReturnsGridBuilder.construct(key, data, backend)


@dataclass
class _ReturnsGridBuilder:
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
        backend: PCABackend | None = None,
    ) -> "ReturnsPCAGrid":
        """Construct and fit a returns PCA grid from ``key`` and price data."""
        prepared = ReturnsPCAGrid.prepareInput(key, data)
        result = (backend or NumpyPCABackend()).fit(prepared.values, prepared.weights, key.components)
        return ReturnsPCAGrid(
            instruments=key.instruments, ew_window=key.ew_window, current_date=key.start_date,
            ew_lambda=key.ew_lambda, components=key.components,
            logReturnMean=prepared.logReturnMean, logReturnScale=prepared.logReturnScale,
            calibrationStartDate=prepared.calibrationStartDate,
            calibrationEndDate=prepared.calibrationEndDate, **vars(result))

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
