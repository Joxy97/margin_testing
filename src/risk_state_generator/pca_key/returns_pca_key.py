"""Key type for returns-based PCA grids."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from download_unit import Instrument

from .pca_key import PCAKey


@dataclass(frozen=True)
class ReturnsPCAKey(PCAKey):
    """Identify a returns PCA grid by its input parameters."""

    instruments: tuple[Instrument, ...]
    ew_window: int
    start_date: date
    ew_lambda: float
    components: int

    def __init__(
        self,
        instruments: Iterable[Instrument],
        ew_window: int,
        start_date: date,
        ew_lambda: float,
        components: int,
    ) -> None:
        object.__setattr__(self, "instruments", tuple(instruments))
        object.__setattr__(self, "ew_window", ew_window)
        object.__setattr__(self, "start_date", start_date)
        object.__setattr__(self, "ew_lambda", ew_lambda)
        object.__setattr__(self, "components", components)

    def equals(self, other: Any) -> bool:
        """Return whether ``other`` is a returns PCA key with equal fields."""
        return self == other
