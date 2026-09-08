"""Copy-once preparation of calibrated, immutable option markets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from collections.abc import Mapping
from types import MappingProxyType

from .market import FuturesForwardCurve, VolatilitySmile
from .calibration import VolatilitySmileCalibrator, VolatilityShockEstimator, VolatilityShockParameters
from portfolio.derivatives import DerivativeQuoteIdentity


@dataclass(frozen=True)
class PreparedOptionMarket:
    valuationDate: date
    forwardCurves: Mapping[str, FuturesForwardCurve]
    spotPrices: Mapping[str, float]
    smiles: Mapping[tuple[str, str, date], VolatilitySmile]
    marketPrices: Mapping[DerivativeQuoteIdentity, float]
    atmVolatility: float
    shockParameters: VolatilityShockParameters

    def __post_init__(self):
        for name in ("forwardCurves", "spotPrices", "smiles", "marketPrices"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class OptionMarketPreparer:
    def __init__(self, smileCalibrator: VolatilitySmileCalibrator,
                 volatilityShockEstimator: VolatilityShockEstimator):
        self.smileCalibrator = smileCalibrator
        self.volatilityShockEstimator = volatilityShockEstimator
        self.marketConventions = smileCalibrator.marketConventions

    def prepare(self, quotes, valuationDate: date, symbol: str) -> PreparedOptionMarket:
        data = self._normalized(quotes)
        builders = self._keyBuilders()
        seen = {}
        keep = []
        for index, row in enumerate(data.itertuples(index=False)):
            identity = (row.date, builders[str(row.instrument_type)](row))
            values = (float(row.price), float(row.dividend_yield),
                      float(getattr(row, "multiplier", 1.0)))
            if identity in seen:
                if seen[identity] != values:
                    raise ValueError(f"Conflicting observations for derivative quote {identity}")
            else:
                seen[identity] = values
                keep.append(index)
        data = data.iloc[keep].copy()
        if (data["date"].dt.date > valuationDate).any():
            raise ValueError("Option history must not contain future observations")
        current = data.loc[data["date"].dt.date == valuationDate].copy()
        if current.empty:
            raise ValueError(f"No derivative quotes for {valuationDate}")
        curves, spots = self._marketInputs(current, valuationDate)
        smiles = self.smileCalibrator.calibrate(
            current, valuationDate, curves, spots
        )
        market_prices = self._marketPrices(current)
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
        return PreparedOptionMarket(valuationDate, curves, spots, smiles, market_prices,
                                    atm_volatility, parameters)

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

    def _keyBuilders(self):
        key_builders = {}
        for convention in self.marketConventions.values():
            key_builders[convention.optionInstrumentType] = (
                convention.optionMarketPriceKey
            )
            key_builders[convention.underlyingInstrumentType] = (
                convention.underlyingMarketPriceKey
            )
        return key_builders

    def _marketPrices(self, current):
        key_builders = self._keyBuilders()
        return {
            key_builders[str(row.instrument_type)](row): float(row.price)
            for row in current.itertuples(index=False)
        }
