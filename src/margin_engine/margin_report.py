"""Margin calculation report."""

from dataclasses import dataclass


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
