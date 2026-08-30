"""Typed derivative contracts and portfolios."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping


def _positive(name: str, value: Decimal) -> None:
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class EquityUnderlying:
    """An equity or spot-index underlying."""

    symbol: str
    currency: str = "USD"
    dividendYield: float = 0.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not 0.0 <= self.dividendYield < 1.0:
            raise ValueError("dividendYield must be in [0, 1)")


@dataclass(frozen=True)
class EquityContract:
    """An equity or spot-index position used alongside equity options."""

    symbol: str
    multiplier: Decimal = Decimal("1")
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        _positive("multiplier", self.multiplier)

    @property
    def instrumentType(self) -> str:
        return "equity"


@dataclass(frozen=True)
class FuturesContract:
    """A futures contract valued in quoted price points."""

    symbol: str
    expirationDate: date
    multiplier: Decimal = Decimal("1")
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        _positive("multiplier", self.multiplier)

    @property
    def instrumentType(self) -> str:
        return "future"


@dataclass(frozen=True)
class OptionContract:
    """Fields shared by equity and futures options."""

    symbol: str
    expirationDate: date
    strike: Decimal
    optionType: str
    exerciseStyle: str = "E"
    multiplier: Decimal = Decimal("1")
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        _positive("strike", self.strike)
        _positive("multiplier", self.multiplier)
        object.__setattr__(self, "optionType", self.optionType.upper())
        object.__setattr__(self, "exerciseStyle", self.exerciseStyle.upper())
        if self.optionType not in {"C", "P"}:
            raise ValueError("optionType must be C or P")
        if self.exerciseStyle not in {"E", "A"}:
            raise ValueError("exerciseStyle must be E or A")


@dataclass(frozen=True)
class FuturesOptionContract(OptionContract):
    """An option whose underlying is a futures curve."""

    @property
    def instrumentType(self) -> str:
        return "futures_option"


@dataclass(frozen=True)
class EquityOptionContract(OptionContract):
    """An option on an equity/spot index with continuous dividend yield."""

    dividendYield: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.dividendYield < 1.0:
            raise ValueError("dividendYield must be in [0, 1)")

    @property
    def instrumentType(self) -> str:
        return "equity_option"


DerivativeContract = (
    EquityContract | FuturesContract | FuturesOptionContract | EquityOptionContract
)


def contractKey(contract: DerivativeContract) -> tuple[str, str, date, str, str, str]:
    """Return the stable fields used to match a contract to a market quote."""

    return (
        contract.instrumentType,
        contract.symbol,
        getattr(contract, "expirationDate", date.min),
        f"{float(getattr(contract, 'strike', 0)):.12g}",
        getattr(contract, "optionType", ""),
        getattr(contract, "exerciseStyle", ""),
    )


@dataclass(frozen=True)
class DerivativePosition:
    """A signed quantity of one derivative contract."""

    contract: DerivativeContract
    quantity: Decimal

    def __post_init__(self) -> None:
        if not self.quantity.is_finite():
            raise ValueError("quantity must be finite")


@dataclass(frozen=True)
class DerivativesPortfolio:
    """A portfolio of futures and/or option positions."""

    positions: tuple[DerivativePosition, ...]
    cash: Decimal = Decimal("0")
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not self.positions:
            raise ValueError("positions must not be empty")
        if not self.cash.is_finite():
            raise ValueError("cash must be finite")

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.contract.symbol for item in self.positions))
