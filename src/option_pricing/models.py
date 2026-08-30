"""European and American option pricing models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import erf, exp, isfinite, log, pi, sqrt

import numpy


def _normalCdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _normalPdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


class OptionPricingModel(ABC):
    """Price one option in quote units, without applying a multiplier."""

    @abstractmethod
    def price(
        self,
        underlyingPrice: float,
        strike: float,
        timeToExpiry: float,
        riskFreeRate: float,
        volatility: float,
        optionType: str,
        dividendYield: float = 0.0,
    ) -> float:
        raise NotImplementedError

    def vega(
        self,
        underlyingPrice: float,
        strike: float,
        timeToExpiry: float,
        riskFreeRate: float,
        volatility: float,
        optionType: str,
        dividendYield: float = 0.0,
    ) -> float:
        """Return dPrice/dVolatility using a stable numerical default."""
        step = max(1e-5, abs(volatility) * 1e-4)
        low = max(volatility - step, 1e-8)
        high = volatility + step
        return (
            self.price(
                underlyingPrice, strike, timeToExpiry, riskFreeRate, high,
                optionType, dividendYield,
            )
            - self.price(
                underlyingPrice, strike, timeToExpiry, riskFreeRate, low,
                optionType, dividendYield,
            )
        ) / (high - low)


class Black76PricingModel(OptionPricingModel):
    """Black-76 valuation for European futures options."""

    def price(self, underlyingPrice, strike, timeToExpiry, riskFreeRate,
              volatility, optionType, dividendYield=0.0) -> float:
        option = _validate(underlyingPrice, strike, optionType)
        intrinsic = max(
            underlyingPrice - strike if option == "C" else strike - underlyingPrice,
            0.0,
        )
        if timeToExpiry <= 0.0:
            return intrinsic
        discount = exp(-riskFreeRate * timeToExpiry)
        if volatility <= 0.0:
            return discount * intrinsic
        root_time = sqrt(timeToExpiry)
        d1 = (
            log(underlyingPrice / strike)
            + 0.5 * volatility * volatility * timeToExpiry
        ) / (volatility * root_time)
        d2 = d1 - volatility * root_time
        if option == "C":
            return discount * (
                underlyingPrice * _normalCdf(d1) - strike * _normalCdf(d2)
            )
        return discount * (
            strike * _normalCdf(-d2) - underlyingPrice * _normalCdf(-d1)
        )

    def vega(self, underlyingPrice, strike, timeToExpiry, riskFreeRate,
             volatility, optionType, dividendYield=0.0) -> float:
        _validate(underlyingPrice, strike, optionType)
        if timeToExpiry <= 0.0 or volatility <= 0.0:
            return 0.0
        root_time = sqrt(timeToExpiry)
        d1 = (
            log(underlyingPrice / strike)
            + 0.5 * volatility * volatility * timeToExpiry
        ) / (volatility * root_time)
        return (
            exp(-riskFreeRate * timeToExpiry)
            * underlyingPrice
            * _normalPdf(d1)
            * root_time
        )


class EquityBlackScholesPricingModel(OptionPricingModel):
    """Black-Scholes valuation with a continuous dividend yield."""

    def price(self, underlyingPrice, strike, timeToExpiry, riskFreeRate,
              volatility, optionType, dividendYield=0.0) -> float:
        option = _validate(underlyingPrice, strike, optionType)
        intrinsic = max(
            underlyingPrice - strike if option == "C" else strike - underlyingPrice,
            0.0,
        )
        if timeToExpiry <= 0.0:
            return intrinsic
        if volatility <= 0.0:
            forward = underlyingPrice * exp(
                (riskFreeRate - dividendYield) * timeToExpiry
            )
            payoff = max(forward - strike if option == "C" else strike - forward, 0.0)
            return exp(-riskFreeRate * timeToExpiry) * payoff
        root_time = sqrt(timeToExpiry)
        d1 = (
            log(underlyingPrice / strike)
            + (riskFreeRate - dividendYield + 0.5 * volatility**2) * timeToExpiry
        ) / (volatility * root_time)
        d2 = d1 - volatility * root_time
        spot_discount = exp(-dividendYield * timeToExpiry)
        strike_discount = exp(-riskFreeRate * timeToExpiry)
        if option == "C":
            return (
                underlyingPrice * spot_discount * _normalCdf(d1)
                - strike * strike_discount * _normalCdf(d2)
            )
        return (
            strike * strike_discount * _normalCdf(-d2)
            - underlyingPrice * spot_discount * _normalCdf(-d1)
        )

    def vega(self, underlyingPrice, strike, timeToExpiry, riskFreeRate,
             volatility, optionType, dividendYield=0.0) -> float:
        _validate(underlyingPrice, strike, optionType)
        if timeToExpiry <= 0.0 or volatility <= 0.0:
            return 0.0
        root_time = sqrt(timeToExpiry)
        d1 = (
            log(underlyingPrice / strike)
            + (riskFreeRate - dividendYield + 0.5 * volatility**2)
            * timeToExpiry
        ) / (volatility * root_time)
        return (
            underlyingPrice
            * exp(-dividendYield * timeToExpiry)
            * _normalPdf(d1)
            * root_time
        )


class _BinomialPricingModel(OptionPricingModel):
    def __init__(self, steps: int = 200) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        self.steps = steps

    @abstractmethod
    def _probability(self, up: float, down: float, dt: float,
                     riskFreeRate: float, dividendYield: float) -> float:
        raise NotImplementedError

    def price(self, underlyingPrice, strike, timeToExpiry, riskFreeRate,
              volatility, optionType, dividendYield=0.0) -> float:
        option = _validate(underlyingPrice, strike, optionType)
        intrinsic = max(
            underlyingPrice - strike if option == "C" else strike - underlyingPrice,
            0.0,
        )
        if timeToExpiry <= 0.0 or volatility <= 0.0:
            return intrinsic
        dt = timeToExpiry / self.steps
        up = exp(volatility * sqrt(dt))
        down = 1.0 / up
        probability = self._probability(
            up, down, dt, riskFreeRate, dividendYield
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("binomial probability is outside [0, 1]")
        nodes = numpy.arange(self.steps + 1, dtype=float)
        prices = underlyingPrice * up**nodes * down ** (self.steps - nodes)
        values = numpy.maximum(
            prices - strike if option == "C" else strike - prices, 0.0
        )
        discount = exp(-riskFreeRate * dt)
        for level in range(self.steps - 1, -1, -1):
            values = discount * (
                probability * values[1:] + (1.0 - probability) * values[:-1]
            )
            nodes = numpy.arange(level + 1, dtype=float)
            prices = underlyingPrice * up**nodes * down ** (level - nodes)
            exercise = numpy.maximum(
                prices - strike if option == "C" else strike - prices, 0.0
            )
            values = numpy.maximum(values, exercise)
        return float(values[0])


class AmericanFuturesBinomialPricingModel(_BinomialPricingModel):
    """American option tree for a martingale futures underlying."""

    def _probability(self, up, down, dt, riskFreeRate, dividendYield) -> float:
        return (1.0 - down) / (up - down)


class AmericanEquityBinomialPricingModel(_BinomialPricingModel):
    """CRR American equity-option tree with continuous dividends."""

    def _probability(self, up, down, dt, riskFreeRate, dividendYield) -> float:
        growth = exp((riskFreeRate - dividendYield) * dt)
        return (growth - down) / (up - down)


def _validate(underlyingPrice: float, strike: float, optionType: str) -> str:
    option = optionType.upper()
    if underlyingPrice <= 0.0 or strike <= 0.0:
        raise ValueError("underlyingPrice and strike must be positive")
    if option not in {"C", "P"}:
        raise ValueError("optionType must be C or P")
    return option


def impliedVolatility(
    model: OptionPricingModel,
    marketPrice: float,
    underlyingPrice: float,
    strike: float,
    timeToExpiry: float,
    riskFreeRate: float,
    optionType: str,
    dividendYield: float = 0.0,
    minimum: float = 1e-6,
    maximum: float = 5.0,
    initial: float = 0.20,
    maximumIterations: int = 20,
) -> float:
    """Invert an option model with bounded Newton-Raphson iterations."""

    if marketPrice < 0.0:
        raise ValueError("marketPrice must be nonnegative")
    if not 0.0 < minimum < maximum:
        raise ValueError("volatility bounds are invalid")
    if not isfinite(initial):
        raise ValueError("initial volatility must be finite")
    if maximumIterations < 1:
        raise ValueError("maximumIterations must be positive")
    tolerance = 1e-10 * max(1.0, marketPrice)
    low_price = model.price(
        underlyingPrice, strike, timeToExpiry, riskFreeRate, minimum,
        optionType, dividendYield,
    )
    high_price = model.price(
        underlyingPrice, strike, timeToExpiry, riskFreeRate, maximum,
        optionType, dividendYield,
    )
    if marketPrice < low_price - tolerance or marketPrice > high_price + tolerance:
        raise ValueError("marketPrice is outside model volatility bounds")
    if abs(marketPrice - low_price) <= tolerance:
        return minimum
    if abs(marketPrice - high_price) <= tolerance:
        return maximum

    volatility = min(max(initial, minimum), maximum)
    for _ in range(maximumIterations):
        price = model.price(
            underlyingPrice, strike, timeToExpiry, riskFreeRate, volatility,
            optionType, dividendYield,
        )
        error = price - marketPrice
        if abs(error) <= tolerance:
            return volatility
        vega = model.vega(
            underlyingPrice, strike, timeToExpiry, riskFreeRate, volatility,
            optionType, dividendYield,
        )
        if not isfinite(vega) or abs(vega) <= 1e-12:
            candidate = min(max(volatility * 2.0, 0.10), maximum)
            if candidate == volatility:
                raise ValueError("Newton-Raphson implied volatility has zero vega")
            volatility = candidate
            continue
        step = error / vega
        if not isfinite(step):
            raise ValueError("Newton-Raphson implied volatility diverged")
        candidate = volatility - step
        while not minimum <= candidate <= maximum:
            step *= 0.5
            candidate = volatility - step
        if not isfinite(candidate):
            raise ValueError("Newton-Raphson implied volatility diverged")
        volatility = candidate
    raise ValueError("Newton-Raphson implied volatility did not converge")
