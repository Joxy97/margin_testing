"""Joint underlying-price and volatility option scenario."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Mapping

from option_pricing import FuturesForwardCurve, VolatilitySmile
from option_pricing.prepared_market import PreparedOptionMarket

from .risk_state import RiskState


SmileKey = tuple[str, str, date]
ContractKey = tuple[str, str, date, str, str, str]


@dataclass(frozen=True)
class OptionScenarioRiskState(RiskState):
    """Calibrated market plus one shared price/volatility stress."""

    valuationDate: date
    priceShock: float
    volatilityShift: float
    forwardCurves: Mapping[str, FuturesForwardCurve]
    spotPrices: Mapping[str, float]
    smiles: Mapping[SmileKey, VolatilitySmile]
    marketPrices: Mapping[ContractKey, float]
    riskFreeRate: float
    dayCountBasis: float
    tradingDaysPerYear: int
    projectionHorizonDays: int
    minimumVolatility: float
    maximumVolatility: float
    americanOptionSteps: int
    preparedMarket: PreparedOptionMarket | None = None

    def __post_init__(self) -> None:
        for name in ("forwardCurves", "spotPrices", "smiles", "marketPrices"):
            value = (MappingProxyType(dict(getattr(self, name))) if self.preparedMarket is None
                     else getattr(self.preparedMarket, name))
            object.__setattr__(self, name, value)
