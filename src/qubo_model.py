"""Compact numeric QUBO representation for the rolling backtest hot path."""

from __future__ import annotations

from dataclasses import dataclass

import dimod
import numpy as np
from scipy import sparse


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


def repair_one_hot(
    bqm: dimod.BinaryQuadraticModel | CompactQubo,
    raw_sample: np.ndarray,
    group_offsets: np.ndarray,
    sweeps: int = 100,
) -> tuple[np.ndarray, int, float]:
    """Project to one-hot and run categorical descent to convergence.

    Every accepted categorical move strictly lowers QUBO energy, so descent
    terminates at a one-coordinate local optimum in the finite state space.
    A positive ``sweeps`` value is a safety cap: reaching it raises rather than
    silently returning a partially optimized sample. Zero means no cap.
    """

    if sweeps < 0:
        raise ValueError("sweeps cannot be negative")
    if isinstance(bqm, CompactQubo):
        labels = None
        linear = bqm.linear
        row = bqm.heads
        col = bqm.tails
        bias = bqm.quadratic
    else:
        labels = tuple(bqm.variables)
        vectors = bqm.to_numpy_vectors(
            variable_order=labels,
            sort_indices=True,
            sort_labels=False,
        )
        linear = np.asarray(vectors.linear_biases, dtype=np.float64)
        row = np.asarray(vectors.quadratic.row_indices, dtype=np.int64)
        col = np.asarray(vectors.quadratic.col_indices, dtype=np.int64)
        bias = np.asarray(vectors.quadratic.biases, dtype=np.float64)
    offsets = np.asarray(group_offsets, dtype=np.int64)
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(linear)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("group offsets must partition all QUBO variables")
    sample = np.asarray(raw_sample, dtype=np.uint8).copy()
    if sample.shape != (len(linear),):
        raise ValueError("raw sample width does not match the QUBO")

    adjacency = sparse.csr_matrix(
        (
            np.concatenate((bias, bias)),
            (np.concatenate((row, col)), np.concatenate((col, row))),
        ),
        shape=(len(linear), len(linear)),
    )
    counts = np.add.reduceat(sample, offsets[:-1])
    violations = int(np.count_nonzero(counts != 1))
    local_fields = linear + adjacency @ sample.astype(float)

    def flip(variable: int) -> None:
        change = -1.0 if sample[variable] else 1.0
        sample[variable] ^= np.uint8(1)
        start = adjacency.indptr[variable]
        stop = adjacency.indptr[variable + 1]
        local_fields[adjacency.indices[start:stop]] += (
            adjacency.data[start:stop] * change
        )

    for group in range(len(offsets) - 1):
        begin = int(offsets[group])
        end = int(offsets[group + 1])
        active = np.flatnonzero(sample[begin:end]) + begin
        if len(active) == 1:
            continue
        for variable in active:
            flip(int(variable))
        flip(begin + int(np.argmin(local_fields[begin:end])))

    completed_sweeps = 0
    while True:
        changed = False
        for group in range(len(offsets) - 1):
            begin = int(offsets[group])
            end = int(offsets[group + 1])
            current = begin + int(np.flatnonzero(sample[begin:end])[0])
            flip(current)
            candidate = begin + int(np.argmin(local_fields[begin:end]))
            tolerance = 32.0 * np.finfo(float).eps * max(
                1.0, abs(float(local_fields[current])), abs(float(local_fields[candidate]))
            )
            best = (
                candidate
                if float(local_fields[candidate]) < float(local_fields[current]) - tolerance
                else current
            )
            flip(best)
            changed |= best != current
        if not changed:
            break
        completed_sweeps += 1
        if sweeps and completed_sweeps >= sweeps:
            raise RuntimeError(
                "categorical repair did not converge within the sweep limit"
            )

    if isinstance(bqm, CompactQubo):
        energy = bqm.energy(sample)
    else:
        energy = float(bqm.energies((sample.reshape(1, -1), labels))[0])
    return sample, violations, energy
