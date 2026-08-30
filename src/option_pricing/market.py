"""Forward curves and volatility smiles used by scenario repricing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy


def yearFraction(start: date, end: date, basis: float = 365.0) -> float:
    return max((end - start).days / basis, 0.0)


@dataclass(frozen=True)
class FuturesForwardCurve:
    """Piecewise-linear futures curve with linear end extrapolation."""

    valuationDate: date
    expirationDates: tuple[date, ...]
    prices: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.prices or len(self.prices) != len(self.expirationDates):
            raise ValueError("forward curve dates and prices must be nonempty and aligned")
        if any(value <= 0.0 for value in self.prices):
            raise ValueError("forward prices must be positive")

    def price(self, expirationDate: date) -> float:
        times = numpy.asarray(
            [(item - self.valuationDate).days for item in self.expirationDates],
            dtype=float,
        )
        values = numpy.asarray(self.prices, dtype=float)
        target = float((expirationDate - self.valuationDate).days)
        if len(values) == 1:
            return float(values[0])
        right = int(numpy.searchsorted(times, target))
        right = min(max(right, 1), len(times) - 1)
        left = right - 1
        weight = (target - times[left]) / (times[right] - times[left])
        result = values[left] + weight * (values[right] - values[left])
        if result <= 0.0:
            raise ValueError("forward extrapolation produced a non-positive price")
        return float(result)


@dataclass(frozen=True)
class VolatilitySmile:
    """Bounded quadratic volatility in log-moneyness."""

    intercept: float
    skew: float
    curvature: float
    observedMinimum: float
    observedMaximum: float
    maximumExtrapolation: float = 0.15
    minimumVolatility: float = 0.01
    maximumVolatility: float = 3.0

    def volatility(self, logMoneyness: float) -> float:
        x = float(logMoneyness)
        boundary = min(max(x, self.observedMinimum), self.observedMaximum)
        value = self._quadratic(boundary)
        if x != boundary:
            slope = self.skew + 2.0 * self.curvature * boundary
            direction = -1.0 if x < boundary else 1.0
            if slope * direction >= 0.0:
                value += slope * direction * min(
                    abs(x - boundary), self.maximumExtrapolation
                )
        return float(numpy.clip(
            value, self.minimumVolatility, self.maximumVolatility
        ))

    def _quadratic(self, x: float) -> float:
        return self.intercept + self.skew * x + self.curvature * x * x
