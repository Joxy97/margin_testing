"""Coverage and paired-date statistics independent of report formatting."""

from dataclasses import dataclass
from collections.abc import Sequence
from scipy.stats import binom


@dataclass(frozen=True)
class CoverageEvaluation:
    days: int
    violations: int
    averageMargin: float
    averageMarginDifference: float
    baselProbability: float
    coveragePValue: float

    @property
    def breachRate(self) -> float:
        return self.violations / self.days

    @property
    def baselColor(self):
        from .backtest_results import BaselColor
        if self.baselProbability >= .9999:
            return BaselColor.RED
        if self.baselProbability > .95:
            return BaselColor.YELLOW
        return BaselColor.GREEN


def evaluateCoverage(dailyResults: Sequence, confidenceLevel: float, name: str | None = None) -> CoverageEvaluation:
    paired = [daily for daily in dailyResults if name is None or name in daily.comparisonMargins]
    if not paired:
        raise ValueError("Coverage evaluation requires at least one matched date")
    margins = [daily.margin if name is None else daily.comparisonMargins[name] for daily in paired]
    violations = sum(daily.realizedLoss > margin for daily, margin in zip(paired, margins))
    days = len(paired)
    return CoverageEvaluation(
        days, violations, sum(margins) / days,
        sum(daily.margin - margin for daily, margin in zip(paired, margins)) / days,
        float(binom.cdf(violations - 1, days, 1.0 - confidenceLevel)),
        float(binom.sf(violations - 1, days, 1.0 - confidenceLevel)),
    )


@dataclass(frozen=True)
class DailyComparison:
    margin: float
    primaryMargin: float
    realizedLoss: float

    @property
    def marginDifference(self) -> float:
        return self.primaryMargin - self.margin

    @property
    def marginError(self) -> float:
        return self.margin - self.realizedLoss

    @property
    def shortfall(self) -> float:
        return max(0.0, -self.marginError)

    @property
    def breach(self) -> bool:
        return self.realizedLoss > self.margin
