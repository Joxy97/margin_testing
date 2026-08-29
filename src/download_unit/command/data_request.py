"""Typed provider-independent market-data requests."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class Period(str, Enum):
    """Time periods accepted by market-data providers."""

    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    YEAR_TO_DATE = "ytd"
    MAX = "max"


Instrument = str


@dataclass(frozen=True)
class DataRequest:
    """Describe one provider-independent data retrieval operation."""

    instruments: tuple[Instrument, ...]
    start_date: date
    end_date: date
    data_type: str
    period: Period = Period.ONE_DAY
    provider_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(
            self,
            "provider_parameters",
            MappingProxyType(dict(self.provider_parameters)),
        )
        if not self.instruments:
            raise ValueError("DataRequest instruments must not be empty")
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be later than end_date")
        if not self.data_type:
            raise ValueError("data_type must not be empty")

    def withChanges(self, **changes: Any) -> DataRequest:
        """Return a request with the supplied typed fields replaced."""
        return replace(self, **changes)

    def withProviderParameters(
        self,
        parameters: Mapping[str, Any],
    ) -> DataRequest:
        """Return a request with merged provider-specific parameters."""
        return replace(
            self,
            provider_parameters={**self.provider_parameters, **parameters},
        )
