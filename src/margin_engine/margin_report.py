"""Margin calculation report."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class MarginEngineTimings:
    """Wall-clock time spent in the major margin-engine stages."""

    dataAcquisitionSeconds: float = 0.0
    riskStateGenerationSeconds: float = 0.0
    marginCalculationSeconds: float = 0.0
    totalSeconds: float = 0.0


@dataclass(frozen=True)
class MarginReport:
    """Contain the margin produced by one engine run."""

    margin: float
    timings: MarginEngineTimings = MarginEngineTimings()
    comparisonMargins: Mapping[str, float] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "comparisonMargins",
            MappingProxyType(
                {
                    str(name): float(value)
                    for name, value in self.comparisonMargins.items()
                }
            ),
        )
