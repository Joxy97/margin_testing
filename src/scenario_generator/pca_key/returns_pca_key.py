"""Key type for returns-based PCA grids."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from download_unit import Instrument

from ..pca_grid import ReturnsPCAGrid
from .pca_key import PCAKey


@dataclass(frozen=True, eq=False)
class ReturnsPCAKey(PCAKey):
    """Identify a returns PCA grid by its input parameters."""

    instruments: tuple[Instrument, ...]
    dates: tuple[date, date]
    ew_lambda: float
    components: int

    def __init__(
        self,
        instruments: Iterable[Instrument],
        dates: tuple[date, date],
        ew_lambda: float,
        components: int,
    ) -> None:
        object.__setattr__(self, "instruments", tuple(instruments))
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "ew_lambda", ew_lambda)
        object.__setattr__(self, "components", components)

    def equals(self, pcaGrid: Any) -> bool:
        """Compare this key with another key or a returns PCA grid."""
        if isinstance(pcaGrid, ReturnsPCAKey):
            other_values = (
                pcaGrid.instruments,
                pcaGrid.dates,
                pcaGrid.ew_lambda,
                pcaGrid.components,
            )
        elif isinstance(pcaGrid, ReturnsPCAGrid):
            other_values = (
                tuple(pcaGrid.instruments),
                pcaGrid.dates,
                pcaGrid.ew_lambda,
                pcaGrid.components,
            )
        else:
            return False
        return self._values() == other_values

    def __eq__(self, other: object) -> bool:
        return self.equals(other)

    def __hash__(self) -> int:
        return hash(self._values())

    def _values(
        self,
    ) -> tuple[tuple[Instrument, ...], tuple[date, date], float, int]:
        return self.instruments, self.dates, self.ew_lambda, self.components
