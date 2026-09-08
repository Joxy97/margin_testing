"""Margin calculation report."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class MarginEngineTimings:
    """Version 2: calculation latency includes generation; generation is work time.

    These spans can overlap and must not be summed. No device synchronization
    is introduced; totalSeconds measures end-to-end host wall-clock latency.
    """

    dataAcquisitionSeconds: float = 0.0
    riskStateGenerationSeconds: float = 0.0
    marginCalculationSeconds: float = 0.0
    totalSeconds: float = 0.0
    measurementVersion: int = 2


@dataclass(frozen=True)
class MarginReport:
    """Contain the margin produced by one engine run."""

    margin: float
    timings: MarginEngineTimings = MarginEngineTimings()
    comparisonMargins: Mapping[str, float] = MappingProxyType({})

    numericalDiagnostics: Mapping[str, int | float | str] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "numericalDiagnostics", MappingProxyType(dict(self.numericalDiagnostics)))
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
