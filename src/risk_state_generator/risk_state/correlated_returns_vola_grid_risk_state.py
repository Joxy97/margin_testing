"""Correlated returns-volatility-grid risk state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .returns_vola_grid_risk_state import ReturnsVolaGridRiskState


@dataclass(frozen=True)
class CorrelationFactors:
    """Compact numeric state-pair compatibility representation."""

    firstAssets: numpy.ndarray
    firstStates: numpy.ndarray
    secondAssets: numpy.ndarray
    secondStates: numpy.ndarray
    coefficients: numpy.ndarray

    def __post_init__(self) -> None:
        integer_fields = (
            "firstAssets",
            "firstStates",
            "secondAssets",
            "secondStates",
        )
        for name in integer_fields:
            values = numpy.ascontiguousarray(getattr(self, name), dtype=numpy.int32)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        coefficients = numpy.ascontiguousarray(
            self.coefficients,
            dtype=numpy.float64,
        )
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)

        lengths = {
            len(getattr(self, name))
            for name in (*integer_fields, "coefficients")
        }
        if len(lengths) != 1:
            raise ValueError("all correlation-factor arrays must have equal length")
        if not numpy.isfinite(coefficients).all() or numpy.any(coefficients < 0):
            raise ValueError("correlation coefficients must be finite and nonnegative")

    def __len__(self) -> int:
        return len(self.coefficients)

    @classmethod
    def empty(cls) -> CorrelationFactors:
        empty_indices = numpy.empty(0, dtype=numpy.int32)
        return cls(
            empty_indices,
            empty_indices,
            empty_indices,
            empty_indices,
            numpy.empty(0, dtype=numpy.float64),
        )


@dataclass
class CorrelatedReturnsVolaGridRiskState(ReturnsVolaGridRiskState):
    """Add compact pairwise compatibility factors to a returns grid."""

    correlations: CorrelationFactors
