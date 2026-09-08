"""Market-specific option calibration and underlying-price conventions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import date
from math import exp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import Series

import numpy
from portfolio.derivatives import DerivativeQuoteIdentity

from .models import (
    AmericanEquityBinomialPricingModel,
    AmericanFuturesBinomialPricingModel,
    Black76PricingModel,
    EquityBlackScholesPricingModel,
    OptionPricingModel,
)
from .market import FuturesForwardCurve


ContractKey = DerivativeQuoteIdentity


class OptionMarketConvention(ABC):
    """Supply behavior that varies between option-underlying markets."""

    optionInstrumentType: str
    underlyingInstrumentType: str

    def __init__(self, models: Mapping[str, OptionPricingModel]) -> None:
        self.models = dict(models)

    def pricingModel(self, exerciseStyle: str) -> OptionPricingModel:
        return self.models[exerciseStyle]

    def forwardPrice(self, underlying: float, time: float, riskFreeRate: float, dividendYield: float = 0.) -> float:
        """Carry an underlying quote to expiry under this market convention."""
        return underlying

    def forwardCurves(self, quotes, valuationDate: date) -> dict:
        return {}

    def spotPrices(self, quotes) -> dict[str, float]:
        return {}

    def optionMarketPriceKey(self, row) -> ContractKey:
        return DerivativeQuoteIdentity.create(
            self.optionInstrumentType,
            str(row.symbol),
            row.expiration_date.date(),
            row.strike,
            str(row.option_type),
            str(row.exercise_style),
        )

    @abstractmethod
    def underlyingMarketPriceKey(self, row) -> ContractKey:
        raise NotImplementedError

    @abstractmethod
    def calibrationPrices(
        self, row, time: float, riskFreeRate: float, forwardCurves, spotPrices
    ) -> tuple[float, float, float]:
        """Return underlying, forward, and dividend yield for one quote."""
        raise NotImplementedError

    def underlyingPrices(self, quotes, symbol: str) -> numpy.ndarray:
        """Compatibility array view; dated consumers use underlyingPriceHistory."""
        return numpy.asarray(list(self.underlyingPriceHistory(quotes, symbol).values()))

    def underlyingPriceHistory(self, quotes, symbol: str) -> dict[date, float]:
        """Return date-owned observations under this market's expiry convention."""
        rows = quotes.loc[
            (quotes["symbol"].astype(str) == symbol)
            & (quotes["instrument_type"] == self.underlyingInstrumentType)
        ].copy()
        if rows.empty:
            return {}
        seen = {}
        for row in rows.itertuples(index=False):
            key = (row.date, self.underlyingMarketPriceKey(row))
            if key in seen and seen[key] != float(row.price):
                raise ValueError(f"Conflicting observations for underlying quote {key}")
            seen[key] = float(row.price)
        return {day.date(): float(value) for day, value in self._orderedPrices(rows).items()}

    @abstractmethod
    def _orderedPrices(self, rows) -> Series:
        raise NotImplementedError


class FuturesOptionMarketConvention(OptionMarketConvention):
    optionInstrumentType = "futures_option"
    underlyingInstrumentType = "future"

    def __init__(self, americanOptionSteps: int = 200) -> None:
        super().__init__({
            "E": Black76PricingModel(),
            "A": AmericanFuturesBinomialPricingModel(americanOptionSteps),
        })

    def calibrationPrices(
        self, row, time, riskFreeRate, forwardCurves, spotPrices
    ) -> tuple[float, float, float]:
        forward = forwardCurves[str(row.symbol)].price(
            row.expiration_date.date()
        )
        return forward, forward, 0.0

    def forwardCurves(self, quotes, valuationDate: date) -> dict:
        curves = {}
        futures = quotes.loc[
            quotes["instrument_type"] == self.underlyingInstrumentType
        ]
        for symbol, rows in futures.groupby("symbol"):
            ordered = (
                rows.groupby("expiration_date", as_index=False)["price"]
                .median()
                .sort_values("expiration_date")
            )
            curves[str(symbol)] = FuturesForwardCurve(
                valuationDate=valuationDate,
                expirationDates=tuple(
                    item.date() for item in ordered["expiration_date"]
                ),
                prices=tuple(float(item) for item in ordered["price"]),
            )
        return curves

    def underlyingMarketPriceKey(self, row) -> ContractKey:
        return DerivativeQuoteIdentity.create(
            self.underlyingInstrumentType,
            str(row.symbol),
            row.expiration_date.date(),
            "0",
            "",
            "",
        )

    def _orderedPrices(self, rows) -> Series:
        first_expiry = (
            rows.sort_values("expiration_date").groupby("date").first()
        )
        return first_expiry.sort_index()["price"]


class EquityOptionMarketConvention(OptionMarketConvention):
    optionInstrumentType = "equity_option"
    underlyingInstrumentType = "equity"

    def __init__(self, americanOptionSteps: int = 200) -> None:
        super().__init__({
            "E": EquityBlackScholesPricingModel(),
            "A": AmericanEquityBinomialPricingModel(americanOptionSteps),
        })

    def forwardPrice(self, underlying: float, time: float, riskFreeRate: float, dividendYield: float = 0.) -> float:
        return underlying * exp((riskFreeRate - dividendYield) * time)

    def calibrationPrices(
        self, row, time, riskFreeRate, forwardCurves, spotPrices
    ) -> tuple[float, float, float]:
        spot = spotPrices[str(row.symbol)]
        dividend = float(row.dividend_yield)
        forward = self.forwardPrice(spot, time, riskFreeRate, dividend)
        return spot, forward, dividend

    def spotPrices(self, quotes) -> dict[str, float]:
        equities = quotes.loc[
            quotes["instrument_type"] == self.underlyingInstrumentType
        ]
        return {
            str(symbol): float(rows.iloc[-1]["price"])
            for symbol, rows in equities.groupby("symbol")
        }

    def underlyingMarketPriceKey(self, row) -> ContractKey:
        return DerivativeQuoteIdentity.create(
            self.underlyingInstrumentType,
            str(row.symbol),
            date.min,
            "0",
            "",
            "",
        )

    def _orderedPrices(self, rows) -> Series:
        return (
            rows.groupby("date")
            .last()
            .sort_index()["price"]
        )


def defaultOptionMarketConventions(
    americanOptionSteps: int = 200,
) -> dict[str, OptionMarketConvention]:
    conventions = (
        FuturesOptionMarketConvention(americanOptionSteps),
        EquityOptionMarketConvention(americanOptionSteps),
    )
    return {item.optionInstrumentType: item for item in conventions}
