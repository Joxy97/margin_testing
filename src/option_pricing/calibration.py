"""Volatility-smile calibration and price/volatility shock estimation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from math import log, sqrt

import numpy

from .market import FuturesForwardCurve, VolatilitySmile, yearFraction
from .conventions import (
    OptionMarketConvention,
    defaultOptionMarketConventions,
)
from .models import impliedVolatility


SmileKey = tuple[str, str, date]


class VolatilitySmileCalibrator:
    """Infer implied volatilities and fit expiry-level quadratic smiles."""

    def __init__(
        self,
        riskFreeRate: float = 0.04,
        dayCountBasis: float = 365.0,
        minimumVolatility: float = 0.01,
        maximumVolatility: float = 3.0,
        maximumExtrapolation: float = 0.15,
        americanOptionSteps: int = 200,
        marketConventions: Mapping[str, OptionMarketConvention] | None = None,
    ) -> None:
        self.riskFreeRate = riskFreeRate
        self.dayCountBasis = dayCountBasis
        self.minimumVolatility = minimumVolatility
        self.maximumVolatility = maximumVolatility
        self.maximumExtrapolation = maximumExtrapolation
        self.marketConventions = dict(
            marketConventions
            or defaultOptionMarketConventions(americanOptionSteps)
        )

    def calibrate(
        self,
        quotes,
        valuationDate: date,
        forwardCurves: dict[str, FuturesForwardCurve],
        spotPrices: dict[str, float],
    ) -> dict[SmileKey, VolatilitySmile]:
        observations = {}
        options = quotes.loc[
            quotes["instrument_type"].isin(self.marketConventions)
        ]
        for row in options.itertuples(index=False):
            kind, symbol = str(row.instrument_type), str(row.symbol)
            expiry = row.expiration_date.date()
            time = yearFraction(valuationDate, expiry, self.dayCountBasis)
            convention = self.marketConventions[kind]
            try:
                underlying, forward, dividend = convention.calibrationPrices(
                    row,
                    time,
                    self.riskFreeRate,
                    forwardCurves,
                    spotPrices,
                )
                volatility = impliedVolatility(
                    convention.pricingModel(str(row.exercise_style)),
                    float(row.price), underlying, float(row.strike), time,
                    self.riskFreeRate, str(row.option_type), dividend,
                    minimum=max(1e-6, self.minimumVolatility / 100.0),
                    maximum=max(5.0, self.maximumVolatility),
                )
            except (KeyError, ValueError):
                continue
            observations.setdefault((kind, symbol, expiry), []).append(
                (log(float(row.strike) / forward), volatility)
            )
        return {
            key: self._fit(points) for key, points in observations.items()
        }

    def _fit(self, points) -> VolatilitySmile:
        x, vol = map(numpy.asarray, zip(*points))
        unique_x, inverse = numpy.unique(x, return_inverse=True)
        unique_vol = numpy.asarray([
            vol[inverse == index].mean() for index in range(len(unique_x))
        ])
        degree = min(2, len(unique_x) - 1)
        coefficients = numpy.polyfit(unique_x, unique_vol, degree)
        full = numpy.zeros(3)
        full[3 - len(coefficients):] = coefficients
        return VolatilitySmile(
            intercept=float(full[2]),
            skew=float(full[1]),
            curvature=float(full[0]),
            observedMinimum=float(unique_x.min()),
            observedMaximum=float(unique_x.max()),
            maximumExtrapolation=self.maximumExtrapolation,
            minimumVolatility=self.minimumVolatility,
            maximumVolatility=self.maximumVolatility,
        )


@dataclass(frozen=True)
class VolatilityShockParameters:
    """Inputs to the price-dependent volatility-shift model."""

    spotVolatility: float
    volOfVolatility: float
    rho: float

    def predictedShift(self, priceShock: float) -> float:
        return self.rho * log(
            self.volOfVolatility / self.spotVolatility
        ) * numpy.log1p(priceShock)


class VolatilityShockEstimator:
    """Estimate EWMA spot volatility, ATM vol-of-volatility, and correlation."""

    def __init__(
        self,
        tradingDaysPerYear: int = 252,
        ewmaLambda: float = 0.94,
        minimumObservations: int = 5,
        fallbackRho: float = -0.75,
        fallbackVolOfVolatility: float = 0.90,
        marketConventions: Mapping[str, OptionMarketConvention] | None = None,
    ) -> None:
        self.tradingDaysPerYear = tradingDaysPerYear
        self.ewmaLambda = ewmaLambda
        self.minimumObservations = minimumObservations
        self.fallbackRho = fallbackRho
        self.fallbackVolOfVolatility = fallbackVolOfVolatility
        self.marketConventions = dict(
            marketConventions or defaultOptionMarketConventions()
        )

    def estimate(
        self,
        quotes,
        symbol: str,
        optionInstrumentType: str,
        atmVolatilityHistory: dict[date, float],
        fallbackSpotVolatility: float,
    ) -> VolatilityShockParameters:
        prices = self.marketConventions[optionInstrumentType].underlyingPrices(
            quotes, symbol
        )
        returns = numpy.diff(numpy.log(prices)) if len(prices) > 1 else numpy.array([])
        spot = self._ewma(returns)
        if not numpy.isfinite(spot) or spot <= 0.0:
            spot = fallbackSpotVolatility

        ordered = sorted(atmVolatilityHistory.items())
        atm = numpy.asarray([value for _, value in ordered], dtype=float)
        vol_changes = numpy.diff(numpy.log(atm)) if len(atm) > 1 else numpy.array([])
        vol_of_vol = (
            float(numpy.std(vol_changes, ddof=1) * sqrt(self.tradingDaysPerYear))
            if len(vol_changes) >= self.minimumObservations
            else self.fallbackVolOfVolatility
        )
        rho = self.fallbackRho
        if (
            len(returns) >= self.minimumObservations
            and len(vol_changes) >= self.minimumObservations
        ):
            count = min(len(returns), len(vol_changes))
            correlation = numpy.corrcoef(returns[-count:], vol_changes[-count:])[0, 1]
            if numpy.isfinite(correlation):
                rho = float(correlation)
        return VolatilityShockParameters(
            spotVolatility=max(float(spot), 1e-12),
            volOfVolatility=max(float(vol_of_vol), 1e-12),
            rho=rho,
        )

    def _ewma(self, values) -> float:
        variance = None
        for value in values:
            variance = value * value if variance is None else (
                self.ewmaLambda * variance
                + (1.0 - self.ewmaLambda) * value * value
            )
        return (
            float("nan")
            if variance is None
            else sqrt(float(variance) * self.tradingDaysPerYear)
        )
