"""Versioned deterministic within-cell integration rules."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy


def _readonly(values: numpy.ndarray) -> numpy.ndarray:
    result = numpy.ascontiguousarray(values, dtype=numpy.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DeterministicRule:
    """Fixed points and weights used inside every dependence cell."""

    nodes: numpy.ndarray
    weights: numpy.ndarray
    version: str = "midpoint-v1"

    def __post_init__(self) -> None:
        nodes = _readonly(self.nodes)
        weights = _readonly(self.weights)
        if nodes.ndim != 2 or not nodes.size:
            raise ValueError("deterministic nodes must be a nonempty matrix")
        if weights.shape != (len(nodes),):
            raise ValueError("deterministic weights must contain one value per node")
        if (
            not numpy.isfinite(nodes).all()
            or numpy.any(nodes <= 0.0)
            or numpy.any(nodes >= 1.0)
        ):
            raise ValueError("deterministic nodes must lie strictly inside (0, 1)")
        if not numpy.isfinite(weights).all() or numpy.any(weights < 0.0):
            raise ValueError("deterministic weights must be finite and nonnegative")
        if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
            raise ValueError("deterministic weights must sum to one")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def build(cls, dimensions: int, nodeCount: int = 1) -> "DeterministicRule":
        """Build the midpoint baseline or a centered antithetic Halton rule."""
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions <= 0
        ):
            raise ValueError("dimensions must be a positive integer")
        if (
            isinstance(nodeCount, bool)
            or not isinstance(nodeCount, int)
            or nodeCount <= 0
        ):
            raise ValueError("nodeCount must be a positive integer")
        if nodeCount == 1:
            return cls(
                numpy.full((1, dimensions), 0.5),
                numpy.ones(1),
                "midpoint-v1",
            )
        bases = cls._firstPrimes(dimensions)
        nodes = numpy.empty((nodeCount, dimensions), dtype=numpy.float64)
        pair_count = nodeCount // 2
        for pair in range(pair_count):
            base_node = numpy.fromiter(
                (
                    cls._radicalInverse(pair + 1, base)
                    for base in bases
                ),
                dtype=numpy.float64,
                count=dimensions,
            )
            nodes[2 * pair] = base_node
            nodes[2 * pair + 1] = 1.0 - base_node
        if nodeCount % 2:
            nodes[-1] = 0.5
        return cls(
            nodes,
            numpy.full(nodeCount, 1.0 / nodeCount),
            f"antithetic-halton-v1-{nodeCount}",
        )

    @staticmethod
    def _radicalInverse(index: int, base: int) -> float:
        inverse = 1.0 / base
        result = 0.0
        factor = inverse
        while index:
            index, digit = divmod(index, base)
            result += digit * factor
            factor *= inverse
        return result

    @staticmethod
    def _firstPrimes(count: int) -> tuple[int, ...]:
        if count == 1:
            return (2,)
        limit = max(16, int(count * (math.log(count) + math.log(math.log(count)))) + 8)
        while True:
            sieve = bytearray(b"\x01") * (limit + 1)
            sieve[0:2] = b"\x00\x00"
            for candidate in range(2, int(math.sqrt(limit)) + 1):
                if sieve[candidate]:
                    start = candidate * candidate
                    sieve[start : limit + 1 : candidate] = b"\x00" * (
                        (limit - start) // candidate + 1
                    )
            primes = tuple(index for index, flag in enumerate(sieve) if flag)
            if len(primes) >= count:
                return primes[:count]
            limit *= 2
