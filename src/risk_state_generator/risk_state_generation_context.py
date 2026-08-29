"""Typed runtime input for risk-state generation."""

from dataclasses import dataclass
from datetime import date

import pandas

from download_unit import DataRequest
@dataclass(frozen=True)
class RiskStateGenerationContext:
    """Bind acquired market data to the request that produced it."""

    marketData: pandas.DataFrame
    dataRequest: DataRequest
    marginDate: date
