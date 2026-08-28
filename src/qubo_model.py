"""Compact numeric QUBO representation for the rolling backtest hot path."""

from __future__ import annotations

from dataclasses import dataclass

import dimod
import numpy as np


@dataclass(frozen=True)
class CompactQubo:
    """QUBO arrays without variable-label dictionaries or coefficient copies."""

    linear: np.ndarray
    heads: np.ndarray
    tails: np.ndarray
    quadratic: np.ndarray
    offset: float = 0.0

    def __post_init__(self) -> None:
        linear = np.ascontiguousarray(self.linear, dtype=np.float64)
        heads = np.ascontiguousarray(self.heads)
        tails = np.ascontiguousarray(self.tails)
        quadratic = np.ascontiguousarray(self.quadratic, dtype=np.float64)
        if heads.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            heads = heads.astype(np.int32)
        if tails.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            tails = tails.astype(np.int32)
        if len(heads) != len(tails) or len(heads) != len(quadratic):
            raise ValueError("quadratic heads, tails, and biases must have equal length")
        if len(linear) == 0:
            raise ValueError("a compact QUBO must contain variables")
        if len(heads) and (
            heads.min() < 0
            or tails.min() < 0
            or heads.max() >= len(linear)
            or tails.max() >= len(linear)
            or np.any(heads == tails)
        ):
            raise ValueError("compact QUBO interaction index is invalid")
        if not (
            np.isfinite(linear).all()
            and np.isfinite(quadratic).all()
            and np.isfinite(self.offset)
        ):
            raise ValueError("compact QUBO coefficients must be finite")
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "heads", heads)
        object.__setattr__(self, "tails", tails)
        object.__setattr__(self, "quadratic", quadratic)

    @property
    def num_variables(self) -> int:
        return len(self.linear)

    @property
    def num_interactions(self) -> int:
        return len(self.quadratic)

    def energies(self, samples: np.ndarray, edge_chunk: int = 1_000_000) -> np.ndarray:
        values = np.asarray(samples)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.num_variables:
            raise ValueError("sample matrix width does not match the compact QUBO")
        energies = np.asarray(values @ self.linear, dtype=np.float64)
        energies += self.offset
        for start in range(0, self.num_interactions, edge_chunk):
            stop = min(start + edge_chunk, self.num_interactions)
            products = values[:, self.heads[start:stop]] * values[:, self.tails[start:stop]]
            energies += products @ self.quadratic[start:stop]
        return energies

    def energy(self, sample: np.ndarray) -> float:
        return float(self.energies(np.asarray(sample).reshape(1, -1))[0])

    def to_dimod(self) -> dimod.BinaryQuadraticModel:
        return dimod.BinaryQuadraticModel.from_numpy_vectors(
            self.linear,
            (self.heads, self.tails, self.quadratic),
            self.offset,
            dimod.BINARY,
            variable_order=range(self.num_variables),
        )
