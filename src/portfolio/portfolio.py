"""Portfolio representation."""

from dataclasses import dataclass, field
from decimal import Decimal

from download_unit import Instrument


@dataclass
class Portfolio:
    """Represent stock quantities and uninvested cash."""

    positions: dict[Instrument, Decimal] = field(default_factory=dict)
    cash: Decimal = Decimal("0")
