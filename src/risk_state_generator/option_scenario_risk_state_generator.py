"""Calibrate option markets and generate joint stress scenarios."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from math import sqrt
from statistics import NormalDist

import numpy

from download_unit import DataRequest, Period
from option_pricing import (
    VolatilityShockEstimator,
    VolatilitySmileCalibrator,
)
from portfolio import DerivativesPortfolio

from .risk_state import OptionScenarioRiskState
from .risk_state_generation_context import RiskStateGenerationContext
from .risk_state_generator import RiskStateGenerator


class OptionScenarioRiskStateGenerator(RiskStateGenerator):
    """Yield a deterministic price/volatility grid for one option market."""

    def __init__(
        self,
        historyDays: int = 365,
        riskFreeRate: float = 0.04,
        dayCountBasis: float = 365.0,
        tradingDaysPerYear: int = 252,
        marginPeriodDays: int = 5,
        projectionHorizonDays: int = 0,
        confidenceLevel: float = 0.99,
        stressMultiplier: float = 1.0,
        priceScenarioSteps: int = 9,
        minimumPriceShock: float = 0.03,
        maximumPriceShock: float = 0.15,
        volatilityShifts: tuple[float, ...] = (-0.03, 0.0, 0.03),
        minimumVolatility: float = 0.01,
        maximumVolatility: float = 3.0,
        maximumVolatilityShift: float = 0.10,
        ewmaLambda: float = 0.94,
        fallbackRho: float = -0.75,
        fallbackVolOfVolatility: float = 0.90,
        volShockMinimumObservations: int = 5,
        maximumSmileExtrapolation: float = 0.15,
        americanOptionSteps: int = 200,
    ) -> None:
        if historyDays < 1 or marginPeriodDays < 1 or tradingDaysPerYear < 1:
            raise ValueError("history and day-count settings must be positive")
        if projectionHorizonDays < 0:
            raise ValueError("projectionHorizonDays must be nonnegative")
        if priceScenarioSteps < 3 or priceScenarioSteps % 2 == 0:
            raise ValueError("priceScenarioSteps must be an odd integer of at least 3")
        if not 0.0 < confidenceLevel < 1.0:
            raise ValueError("confidenceLevel must be between zero and one")
        if not 0.0 < ewmaLambda < 1.0:
            raise ValueError("ewmaLambda must be between zero and one")
        if not 0.0 < minimumVolatility <= maximumVolatility:
            raise ValueError("volatility bounds are invalid")
        if not 0.0 <= minimumPriceShock <= maximumPriceShock < 1.0:
            raise ValueError("price-shock bounds are invalid")
        self.historyDays = historyDays
        self.riskFreeRate = riskFreeRate
        self.dayCountBasis = dayCountBasis
        self.tradingDaysPerYear = tradingDaysPerYear
        self.marginPeriodDays = marginPeriodDays
        self.projectionHorizonDays = projectionHorizonDays
        self.confidenceLevel = confidenceLevel
        self.stressMultiplier = stressMultiplier
        self.priceScenarioSteps = priceScenarioSteps
        self.minimumPriceShock = minimumPriceShock
        self.maximumPriceShock = maximumPriceShock
        self.volatilityShifts = tuple(sorted(float(x) for x in volatilityShifts))
        self.minimumVolatility = minimumVolatility
        self.maximumVolatility = maximumVolatility
        self.maximumVolatilityShift = maximumVolatilityShift
        self.ewmaLambda = ewmaLambda
        self.fallbackRho = fallbackRho
        self.fallbackVolOfVolatility = fallbackVolOfVolatility
        self.maximumSmileExtrapolation = maximumSmileExtrapolation
        self.americanOptionSteps = americanOptionSteps
        self.smileCalibrator = VolatilitySmileCalibrator(
            riskFreeRate=riskFreeRate,
            dayCountBasis=dayCountBasis,
            minimumVolatility=minimumVolatility,
            maximumVolatility=maximumVolatility,
            maximumExtrapolation=maximumSmileExtrapolation,
            americanOptionSteps=americanOptionSteps,
        )
        self.volatilityShockEstimator = VolatilityShockEstimator(
            tradingDaysPerYear=tradingDaysPerYear,
            ewmaLambda=ewmaLambda,
            minimumObservations=volShockMinimumObservations,
            fallbackRho=fallbackRho,
            fallbackVolOfVolatility=fallbackVolOfVolatility,
            marketConventions=self.smileCalibrator.marketConventions,
        )
        self.marketConventions = self.smileCalibrator.marketConventions

    def createDataRequest(
        self, portfolio: DerivativesPortfolio, marginDate: date
    ) -> DataRequest:
        if not isinstance(portfolio, DerivativesPortfolio):
            raise TypeError("option scenarios require a DerivativesPortfolio")
        if len(portfolio.symbols) != 1:
            raise ValueError("option scenario portfolios must contain one symbol")
        return DataRequest(
            instruments=portfolio.symbols,
            start_date=marginDate - timedelta(days=self.historyDays),
            end_date=marginDate,
            data_type="derivativeQuotes",
            period=Period.ONE_DAY,
        )

    def getRiskStates(
        self, context: RiskStateGenerationContext
    ) -> Iterator[OptionScenarioRiskState]:
        data = self._normalized(context.marketData)
        current = data.loc[data["date"].dt.date == context.marginDate].copy()
        if current.empty:
            raise ValueError(f"No derivative quotes for {context.marginDate}")
        curves, spots = self._marketInputs(current, context.marginDate)
        smiles = self.smileCalibrator.calibrate(
            current, context.marginDate, curves, spots
        )
        market_prices = self._marketPrices(current)
        symbol = context.dataRequest.instruments[0]
        market_smiles = [
            (kind, expiry, smile)
            for (kind, item_symbol, expiry), smile in smiles.items()
            if item_symbol == symbol
        ]
        if not market_smiles:
            raise ValueError(f"No option smile could be calibrated for {symbol}")
        market_kind, _, nearest_smile = min(
            market_smiles, key=lambda item: item[1]
        )
        atm_volatility = nearest_smile.volatility(0.0)
        parameters = self.volatilityShockEstimator.estimate(
            data,
            symbol,
            market_kind,
            self._atmVolatilityHistory(data, symbol),
            atm_volatility,
        )
        raw_scan = (
            NormalDist().inv_cdf(self.confidenceLevel)
            * atm_volatility
            * sqrt(self.marginPeriodDays / self.tradingDaysPerYear)
            * self.stressMultiplier
        )
        scan = float(numpy.clip(
            raw_scan, self.minimumPriceShock, self.maximumPriceShock
        ))
        price_shocks = numpy.linspace(-scan, scan, self.priceScenarioSteps)
        price_shocks[numpy.isclose(price_shocks, 0.0, atol=1e-14)] = 0.0
        for price_shock in price_shocks:
            predicted = parameters.predictedShift(float(price_shock))
            predicted = float(numpy.clip(
                predicted,
                -self.maximumVolatilityShift,
                self.maximumVolatilityShift,
            ))
            for band in self.volatilityShifts:
                yield OptionScenarioRiskState(
                    valuationDate=context.marginDate,
                    priceShock=float(price_shock),
                    volatilityShift=predicted + band,
                    forwardCurves=curves,
                    spotPrices=spots,
                    smiles=smiles,
                    marketPrices=market_prices,
                    riskFreeRate=self.riskFreeRate,
                    dayCountBasis=self.dayCountBasis,
                    tradingDaysPerYear=self.tradingDaysPerYear,
                    projectionHorizonDays=self.projectionHorizonDays,
                    minimumVolatility=self.minimumVolatility,
                    maximumVolatility=self.maximumVolatility,
                    americanOptionSteps=self.americanOptionSteps,
                )

    @staticmethod
    def _normalized(data):
        import pandas

        result = data.copy()
        result["date"] = pandas.to_datetime(result["date"], errors="raise")
        result["expiration_date"] = pandas.to_datetime(
            result["expiration_date"], errors="raise"
        )
        for column, default in (
            ("strike", 0.0), ("option_type", ""), ("exercise_style", "E"),
            ("dividend_yield", 0.0),
        ):
            if column not in result:
                result[column] = default
            result[column] = result[column].fillna(default)
        result["instrument_type"] = result["instrument_type"].astype(str).str.lower()
        result["option_type"] = result["option_type"].astype(str).str.upper()
        result["exercise_style"] = result["exercise_style"].astype(str).str.upper()
        return result

    def _marketInputs(self, quotes, valuationDate):
        curves = {}
        spots = {}
        for convention in self.marketConventions.values():
            curves.update(convention.forwardCurves(quotes, valuationDate))
            spots.update(convention.spotPrices(quotes))
        return curves, spots

    def _atmVolatilityHistory(self, data, symbol):
        history = {}
        for timestamp, rows in data.groupby("date"):
            valuation_date = timestamp.date()
            curves, spots = self._marketInputs(rows, valuation_date)
            smiles = self.smileCalibrator.calibrate(
                rows, valuation_date, curves, spots
            )
            candidates = [
                (expiry, smile)
                for (_, item_symbol, expiry), smile in smiles.items()
                if item_symbol == symbol
            ]
            if candidates:
                history[valuation_date] = min(
                    candidates, key=lambda item: item[0]
                )[1].volatility(0.0)
        return history

    def _marketPrices(self, current):
        key_builders = {}
        for convention in self.marketConventions.values():
            key_builders[convention.optionInstrumentType] = (
                convention.optionMarketPriceKey
            )
            key_builders[convention.underlyingInstrumentType] = (
                convention.underlyingMarketPriceKey
            )
        return {
            key_builders[str(row.instrument_type)](row): float(row.price)
            for row in current.itertuples(index=False)
        }
