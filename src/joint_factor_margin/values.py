"""Owned requests and results at the application interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
import math

import numpy as np

from portfolio import Portfolio
from option_pricing import EuropeanOptionBook, OptionBook


def freeze(value):
    """Recursively detach public results from adapter-owned mutable state."""
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def primitive(value):
    """Convert results to strict JSON; retain nonfinite diagnostic values explicitly."""
    if is_dataclass(value):
        return {field.name: primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): primitive(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return primitive(value.tolist())
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    if isinstance(value, np.generic):
        return primitive(value.item())
    if isinstance(value, (date, Decimal, Path)):
        return str(value)
    if isinstance(value, Enum):
        return primitive(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonFinite": str(value)}
    return value


@dataclass(frozen=True)
class ExposurePortfolio:
    """Immutable legacy return/P&L exposures, with no share-unit conversion."""

    weights: Mapping[str, Decimal]
    cash: Decimal = Decimal("0")

    def __post_init__(self):
        copied = dict(self.weights)
        if not copied or any(not isinstance(key, str) or not key for key in copied):
            raise ValueError("portfolio weights require nonempty instrument names")
        for value in (*copied.values(), self.cash):
            if not isinstance(value, Decimal):
                raise TypeError("portfolio exposures and cash must be Decimal values")
            if not value.is_finite() or not math.isfinite(float(value)):
                raise ValueError("portfolio exposures and cash must be finite float64 values")
        object.__setattr__(self, "weights", MappingProxyType(dict(sorted(copied.items()))))

    @property
    def instruments(self) -> tuple[str, ...]:
        return tuple(self.weights)

    @classmethod
    def fromPortfolio(cls, portfolio: Portfolio) -> ExposurePortfolio:
        if not isinstance(portfolio, Portfolio):
            raise TypeError("portfolio must be a Portfolio")
        return cls(portfolio.weights, portfolio.cash)

    def toPortfolio(self) -> Portfolio:
        return Portfolio(dict(self.weights), self.cash)


@dataclass(frozen=True)
class CalculationRequest:
    portfolio: ExposurePortfolio
    marginDate: date
    options: EuropeanOptionBook | None = None
    localPnlGradient: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.portfolio, ExposurePortfolio):
            raise TypeError("portfolio must be an ExposurePortfolio")
        if type(self.marginDate) is not date:
            raise TypeError("marginDate must be a date")
        if self.options is not None:
            if not isinstance(self.options, EuropeanOptionBook):
                raise TypeError("options must be a prepared EuropeanOptionBook or OptionBook")
            if self.options.instruments != self.portfolio.instruments:
                raise ValueError("option instruments must match canonical portfolio order")
            if self.options.valuationDate >= self.marginDate:
                raise ValueError("option valuationDate must precede marginDate")
            if self.options.horizonDate != self.marginDate:
                raise ValueError("option horizonDate must equal marginDate")
        if self.localPnlGradient is not None:
            if not isinstance(self.options, OptionBook):
                raise ValueError("supplied localPnlGradient requires an OptionBook")
            gradient = np.asarray(self.localPnlGradient, dtype=np.float64)
            if gradient.shape != (len(self.portfolio.instruments),) or not np.isfinite(gradient).all():
                raise ValueError("localPnlGradient must be finite and match canonical instruments")
            object.__setattr__(self, "localPnlGradient", freeze(gradient))


@dataclass(frozen=True)
class SearchResult:
    pnl: float
    coordinates: np.ndarray
    quadraticPnL: float | None
    diagnostics: Mapping[str, Any]

    def __post_init__(self):
        if not math.isfinite(self.pnl):
            raise ValueError("selected P&L must be finite")
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        if coordinates.ndim != 1 or not len(coordinates) or not np.isfinite(coordinates).all():
            raise ValueError("selected coordinates must be a finite nonempty vector")
        object.__setattr__(self, "coordinates", freeze(coordinates))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))


@dataclass(frozen=True)
class CalculationResult:
    marginDate: date
    margin: float
    pnl: float
    coordinates: np.ndarray
    instruments: tuple[str, ...]
    logReturns: np.ndarray
    quadraticPnL: float | None
    diagnostics: Mapping[str, Any]
    timings: Mapping[str, float]
    configurationIdentity: str

    def __post_init__(self):
        if not math.isfinite(self.pnl) or self.margin != max(0., -self.pnl):
            raise ValueError("margin must equal the nonnegative selected loss")
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        logs = np.asarray(self.logReturns, dtype=np.float64)
        if coordinates.ndim != 1 or not len(coordinates) or not np.isfinite(coordinates).all():
            raise ValueError("result coordinates must be a finite nonempty vector")
        if logs.shape != (len(self.instruments),) or not np.isfinite(logs).all():
            raise ValueError("result logReturns must be finite and match instruments")
        if self.quadraticPnL is not None and not math.isfinite(self.quadraticPnL):
            raise ValueError("quadratic P&L must be finite when supplied")
        if any(not math.isfinite(value) or value < 0 for value in self.timings.values()):
            raise ValueError("result timings must be finite and nonnegative")
        object.__setattr__(self, "coordinates", freeze(coordinates))
        object.__setattr__(self, "logReturns", freeze(logs))
        for name in ("diagnostics", "timings"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        object.__setattr__(self, "instruments", tuple(self.instruments))

    def toMapping(self) -> dict:
        return primitive(self)
