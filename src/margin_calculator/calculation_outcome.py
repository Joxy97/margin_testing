"""Immutable results owned by a single margin calculation."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType


@dataclass(frozen=True)
class CalculationOutcome:
    margin: float
    comparisonMargins: Mapping[str, float] = field(default_factory=dict)

    numericalDiagnostics: Mapping[str, int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparisonMargins", MappingProxyType(dict(self.comparisonMargins)))
        object.__setattr__(self, "numericalDiagnostics", MappingProxyType(dict(self.numericalDiagnostics)))
