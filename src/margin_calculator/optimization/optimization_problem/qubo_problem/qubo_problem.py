"""Compact numeric representation of a QUBO problem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy

from ..optimization_problem import OptimizationProblem


@dataclass(frozen=True)
class QUBOProblem(OptimizationProblem):
    """Store sparse QUBO coefficients without a solver-library dependency."""

    linear: numpy.ndarray
    quadraticHeads: numpy.ndarray
    quadraticTails: numpy.ndarray
    quadraticBiases: numpy.ndarray
    offset: float = 0.0
    oneHotGroups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        linear = self._immutableArray(self.linear, numpy.float64)
        heads = self._immutableArray(self.quadraticHeads, numpy.uint32)
        tails = self._immutableArray(self.quadraticTails, numpy.uint32)
        biases = self._immutableArray(self.quadraticBiases, numpy.float64)
        groups = tuple(
            tuple(int(variable) for variable in group)
            for group in self.oneHotGroups
        )

        if not len(linear):
            raise ValueError("QUBOProblem must contain at least one variable")
        if not (len(heads) == len(tails) == len(biases)):
            raise ValueError("quadratic coefficient arrays must have equal length")
        if not numpy.isfinite(linear).all() or not numpy.isfinite(biases).all():
            raise ValueError("QUBO coefficients must be finite")
        if not numpy.isfinite(self.offset):
            raise ValueError("QUBO offset must be finite")
        if len(heads) and (
            numpy.any(heads >= len(linear)) or numpy.any(tails >= len(linear))
        ):
            raise ValueError("quadratic coefficients contain an unknown variable")
        if any(not group for group in groups):
            raise ValueError("one-hot groups must not be empty")
        grouped_variables = tuple(
            variable for group in groups for variable in group
        )
        if len(grouped_variables) != len(set(grouped_variables)):
            raise ValueError("one-hot groups must be disjoint")
        if grouped_variables and (
            min(grouped_variables) < 0
            or max(grouped_variables) >= len(linear)
        ):
            raise ValueError("one-hot groups contain an unknown variable")

        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadraticHeads", heads)
        object.__setattr__(self, "quadraticTails", tails)
        object.__setattr__(self, "quadraticBiases", biases)
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "oneHotGroups", groups)

    @staticmethod
    def _immutableArray(values: Any, dtype: Any) -> numpy.ndarray:
        result = numpy.ascontiguousarray(values, dtype=dtype)
        result.setflags(write=False)
        return result

    @property
    def variableCount(self) -> int:
        return len(self.linear)

    @property
    def interactionCount(self) -> int:
        return len(self.quadraticBiases)

    def energy(self, sample: Mapping[int, int] | Sequence[int]) -> float:
        """Evaluate a binary sample against this QUBO."""
        if isinstance(sample, Mapping):
            values = numpy.fromiter(
                (sample.get(index, 0) for index in range(self.variableCount)),
                dtype=numpy.float64,
                count=self.variableCount,
            )
        else:
            values = numpy.asarray(sample, dtype=numpy.float64)
        if values.shape != (self.variableCount,):
            raise ValueError("sample length must match the QUBO variable count")
        if not numpy.all((values == 0.0) | (values == 1.0)):
            raise ValueError("QUBO samples must be binary")
        return float(
            self.offset
            + self.linear @ values
            + numpy.sum(
                self.quadraticBiases
                * values[self.quadraticHeads]
                * values[self.quadraticTails]
            )
        )
