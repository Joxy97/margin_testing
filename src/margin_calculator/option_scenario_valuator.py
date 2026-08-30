"""Polymorphic derivative valuation under one option scenario."""

from __future__ import annotations

from functools import singledispatchmethod
from math import exp, log

import numpy

from option_pricing import (
    AmericanEquityBinomialPricingModel,
    AmericanFuturesBinomialPricingModel,
    Black76PricingModel,
    EquityBlackScholesPricingModel,
    yearFraction,
)
from portfolio import (
    EquityContract,
    EquityOptionContract,
    FuturesContract,
    FuturesOptionContract,
)
from risk_state_generator import OptionScenarioRiskState


class OptionScenarioValuator:
    """Return base and stressed prices without coupling contracts to models."""

    @singledispatchmethod
    def prices(
        self,
        contract,
        state: OptionScenarioRiskState,
    ) -> tuple[float, float]:
        raise TypeError(f"Unsupported derivative: {type(contract).__name__}")

    @prices.register
    def _(self, contract: EquityContract, state: OptionScenarioRiskState):
        base = state.spotPrices[contract.symbol]
        return base, base * (1.0 + state.priceShock)

    @prices.register
    def _(self, contract: FuturesContract, state: OptionScenarioRiskState):
        base = state.forwardCurves[contract.symbol].price(contract.expirationDate)
        return base, base * (1.0 + state.priceShock)

    @prices.register
    def _(self, contract: FuturesOptionContract, state: OptionScenarioRiskState):
        underlying = state.forwardCurves[contract.symbol].price(
            contract.expirationDate
        )
        return self._optionPrices(
            contract,
            state,
            underlying,
            underlying,
            "futures_option",
            0.0,
        )

    @prices.register
    def _(self, contract: EquityOptionContract, state: OptionScenarioRiskState):
        underlying = state.spotPrices[contract.symbol]
        time = self._timeToExpiry(contract, state)
        forward = underlying * exp(
            (state.riskFreeRate - contract.dividendYield) * time
        )
        return self._optionPrices(
            contract,
            state,
            underlying,
            forward,
            "equity_option",
            contract.dividendYield,
        )

    def _optionPrices(
        self,
        contract,
        state: OptionScenarioRiskState,
        underlying: float,
        forward: float,
        instrumentType: str,
        dividendYield: float,
    ) -> tuple[float, float]:
        time = self._timeToExpiry(contract, state)
        projected_time = max(
            0.0,
            time - state.projectionHorizonDays / state.tradingDaysPerYear,
        )
        smile = state.smiles[
            (instrumentType, contract.symbol, contract.expirationDate)
        ]
        base_volatility = smile.volatility(log(float(contract.strike) / forward))
        stressed_underlying = underlying * (1.0 + state.priceShock)
        stressed_forward = (
            stressed_underlying
            if instrumentType == "futures_option"
            else stressed_underlying
            * exp((state.riskFreeRate - dividendYield) * projected_time)
        )
        stressed_volatility = float(numpy.clip(
            smile.volatility(log(float(contract.strike) / stressed_forward))
            + state.volatilityShift,
            state.minimumVolatility,
            state.maximumVolatility,
        ))
        model = self._model(instrumentType, contract.exerciseStyle, state)
        base_price = self._price(
            model,
            contract,
            state,
            underlying,
            time,
            base_volatility,
            dividendYield,
        )
        stressed_price = self._price(
            model,
            contract,
            state,
            stressed_underlying,
            projected_time,
            stressed_volatility,
            dividendYield,
        )
        return base_price, stressed_price

    @staticmethod
    def _price(
        model,
        contract,
        state,
        underlying: float,
        timeToExpiry: float,
        volatility: float,
        dividendYield: float,
    ) -> float:
        return model.price(
            underlying,
            float(contract.strike),
            timeToExpiry,
            state.riskFreeRate,
            volatility,
            contract.optionType,
            dividendYield,
        )

    @staticmethod
    def _model(instrumentType, exerciseStyle, state):
        factories = {
            ("futures_option", "E"): Black76PricingModel,
            ("futures_option", "A"): lambda: AmericanFuturesBinomialPricingModel(
                state.americanOptionSteps
            ),
            ("equity_option", "E"): EquityBlackScholesPricingModel,
            ("equity_option", "A"): lambda: AmericanEquityBinomialPricingModel(
                state.americanOptionSteps
            ),
        }
        return factories[(instrumentType, exerciseStyle)]()

    @staticmethod
    def _timeToExpiry(contract, state) -> float:
        return yearFraction(
            state.valuationDate,
            contract.expirationDate,
            state.dayCountBasis,
        )
