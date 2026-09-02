"""Compact numeric representation of a QUBO problem."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy

from ..optimization_problem import OptimizationProblem


@dataclass(frozen=True, init=False)
class QUBOProblem(OptimizationProblem):
    """Store sparse QUBO coefficients without a solver-library dependency."""

    linear: numpy.ndarray
    quadraticHeads: numpy.ndarray
    quadraticTails: numpy.ndarray
    quadraticBiases: numpy.ndarray
    offset: float = 0.0
    groupOffsets: numpy.ndarray
    seedOffset: int
    _legacyOneHotGroups: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        linear: numpy.ndarray,
        quadraticHeads: numpy.ndarray,
        quadraticTails: numpy.ndarray,
        quadraticBiases: numpy.ndarray,
        offset: float = 0.0,
        oneHotGroups: Sequence[Sequence[int]] = (),
        groupOffsets: Sequence[int] | numpy.ndarray | None = None,
        seedOffset: int | None = None,
    ) -> None:
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadraticHeads", quadraticHeads)
        object.__setattr__(self, "quadraticTails", quadraticTails)
        object.__setattr__(self, "quadraticBiases", quadraticBiases)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "_legacyOneHotGroups", tuple(oneHotGroups))
        object.__setattr__(
            self,
            "groupOffsets",
            numpy.empty(0, dtype=numpy.uint32)
            if groupOffsets is None
            else groupOffsets,
        )
        object.__setattr__(self, "seedOffset", seedOffset)
        self.__post_init__()

    def __post_init__(self) -> None:
        linear = self._immutableArray(self.linear, numpy.float64)
        heads = self._immutableArray(self.quadraticHeads, numpy.uint32)
        tails = self._immutableArray(self.quadraticTails, numpy.uint32)
        biases = self._immutableArray(self.quadraticBiases, numpy.float64)
        groups = tuple(
            tuple(int(variable) for variable in group)
            for group in self._legacyOneHotGroups
        )
        raw_group_offsets = numpy.asarray(self.groupOffsets)
        if raw_group_offsets.ndim != 1:
            raise ValueError("groupOffsets must be one-dimensional")
        if len(raw_group_offsets) and (
            not numpy.issubdtype(raw_group_offsets.dtype, numpy.integer)
            or numpy.any(raw_group_offsets < 0)
        ):
            raise ValueError("groupOffsets must contain nonnegative integers")
        group_offsets = self._immutableArray(raw_group_offsets, numpy.uint32)

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
        if len(group_offsets):
            if len(group_offsets) < 2 or group_offsets[0] != 0:
                raise ValueError("groupOffsets must start at zero")
            if numpy.any(numpy.diff(group_offsets.astype(numpy.int64)) <= 0):
                raise ValueError("groupOffsets must be strictly increasing")
            if group_offsets[-1] > len(linear):
                raise ValueError("groupOffsets contain an unknown variable")
            if groups:
                offset_groups = tuple(
                    tuple(range(int(start), int(stop)))
                    for start, stop in zip(group_offsets[:-1], group_offsets[1:])
                )
                if groups != offset_groups:
                    raise ValueError(
                        "oneHotGroups and groupOffsets describe different groups"
                    )

        stable_seed_offset = self.seedOffset
        if stable_seed_offset is None:
            digest = hashlib.blake2b(digest_size=8, person=b"QUBOseed")
            for array in (linear, heads, tails, biases, group_offsets):
                digest.update(numpy.asarray(array.shape, dtype=numpy.uint64))
                digest.update(memoryview(array).cast("B"))
            digest.update(numpy.float64(self.offset).tobytes())
            stable_seed_offset = int.from_bytes(digest.digest(), "little")
        if isinstance(stable_seed_offset, bool) or not isinstance(
            stable_seed_offset, (int, numpy.integer)
        ):
            raise TypeError("seedOffset must be an integer or None")
        stable_seed_offset = int(stable_seed_offset) % ((1 << 63) - 1)

        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadraticHeads", heads)
        object.__setattr__(self, "quadraticTails", tails)
        object.__setattr__(self, "quadraticBiases", biases)
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "groupOffsets", group_offsets)
        object.__setattr__(self, "seedOffset", stable_seed_offset)
        object.__setattr__(self, "_legacyOneHotGroups", groups)

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

    @property
    def oneHotGroups(self) -> tuple[tuple[int, ...], ...]:
        """Materialize legacy group tuples only at compatibility boundaries."""
        return tuple(tuple(group) for group in self.iterOneHotGroups())

    def iterOneHotGroups(self) -> Iterator[Sequence[int]]:
        """Iterate one-hot groups from compact offsets when available."""
        if len(self.groupOffsets):
            for start, stop in zip(self.groupOffsets[:-1], self.groupOffsets[1:]):
                yield range(int(start), int(stop))
            return
        yield from self._legacyOneHotGroups

    @property
    def numericMemoryBytes(self) -> int:
        """Return bytes occupied by the contiguous numeric coefficient arrays."""
        return sum(
            array.nbytes
            for array in (
                self.linear,
                self.quadraticHeads,
                self.quadraticTails,
                self.quadraticBiases,
                self.groupOffsets,
            )
        )

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
