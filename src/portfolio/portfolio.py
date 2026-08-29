"""Portfolio representation."""

from dataclasses import dataclass, field
from decimal import Decimal

from download_unit import Instrument


@dataclass
class Portfolio:
    """Represent MarginLab-style instrument return weights and cash."""

    weights: dict[Instrument, Decimal] = field(default_factory=dict)
    cash: Decimal = Decimal("0")
