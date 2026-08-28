"""Returns-based PCA grid."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from download_unit import Instrument

from .pca_grid import PCAGrid


@dataclass
class ReturnsPCAGrid(PCAGrid):
    """Store configuration and calculated values for a returns PCA grid."""

    instruments: Iterable[Instrument]
    dates: tuple[date, date]
    ew_lambda: float
    components: int
    lambdas: Any = field(init=False, default=None)
    loadings: Any = field(init=False, default=None)
    factors: Any = field(init=False, default=None)
