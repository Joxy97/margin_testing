"""Weighted empirical-checkerboard dependence construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy


def _readonly(values: numpy.ndarray, dtype: object = numpy.float64) -> numpy.ndarray:
    result = numpy.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DependenceCells:
    """Factor-specific rank intervals owned by synchronized residual rows."""

    lower: numpy.ndarray
    upper: numpy.ndarray
    weights: numpy.ndarray
    rowIds: tuple[str, ...]

    def __post_init__(self) -> None:
        lower = _readonly(self.lower)
        upper = _readonly(self.upper)
        weights = _readonly(self.weights)
        if lower.ndim != 2 or upper.shape != lower.shape:
            raise ValueError(
                "dependence boundaries must be equal two-dimensional arrays"
            )
        if weights.shape != (lower.shape[0],):
            raise ValueError("dependence weights must contain one value per row")
        if len(self.rowIds) != lower.shape[0] or len(set(self.rowIds)) != len(
            self.rowIds
        ):
            raise ValueError("dependence row IDs must be unique and row-aligned")
        if (
            not numpy.isfinite(lower).all()
            or not numpy.isfinite(upper).all()
            or not numpy.isfinite(weights).all()
            or numpy.any(weights <= 0.0)
        ):
            raise ValueError("dependence cells and weights must be finite and positive")
        if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
            raise ValueError("dependence weights must sum to one")
        if not numpy.allclose(upper - lower, weights[:, None], atol=1e-12):
            raise ValueError("every dependence-cell width must equal its row weight")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "rowIds", tuple(str(item) for item in self.rowIds))


def constructDependenceCells(
    residuals: numpy.ndarray,
    weights: numpy.ndarray,
    rowDates: Sequence[object],
    rowIds: Sequence[str],
) -> DependenceCells:
    """Construct stable weighted rank intervals without breaking row ownership."""
    values = numpy.asarray(residuals, dtype=numpy.float64)
    probability = numpy.asarray(weights, dtype=numpy.float64)
    if values.ndim != 2 or not numpy.isfinite(values).all():
        raise ValueError("residuals must be a finite two-dimensional array")
    rows, factors = values.shape
    if probability.shape != (rows,) or not numpy.isfinite(probability).all():
        raise ValueError("weights must contain one finite value per residual row")
    if numpy.any(probability <= 0.0):
        raise ValueError("dependence weights must be positive")
    probability = probability / probability.sum()
    if len(rowDates) != rows or len(rowIds) != rows:
        raise ValueError("row dates and IDs must align with residual rows")
    stable_ids = tuple(str(item) for item in rowIds)
    if len(set(stable_ids)) != rows:
        raise ValueError("row IDs must be unique")

    lower = numpy.empty((rows, factors), dtype=numpy.float64)
    upper = numpy.empty_like(lower)
    for factor in range(factors):
        order = sorted(
            range(rows),
            key=lambda row: (
                float(values[row, factor]),
                str(rowDates[row]),
                stable_ids[row],
            ),
        )
        cumulative = 0.0
        for row in order:
            lower[row, factor] = cumulative
            cumulative += float(probability[row])
            upper[row, factor] = cumulative
        lower[order[0], factor] = 0.0
        upper[order[-1], factor] = 1.0
        ordered_lower = lower[order, factor]
        ordered_upper = upper[order, factor]
        if not numpy.allclose(ordered_lower[1:], ordered_upper[:-1], atol=1e-12):
            raise RuntimeError("dependence rank intervals contain a gap or overlap")
    return DependenceCells(lower, upper, probability, stable_ids)
