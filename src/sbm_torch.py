"""In-memory batched simulated bifurcation using PyTorch sparse kernels.

The solver packs independent BQMs into a block-diagonal Ising matrix.  Every
matrix multiplication therefore advances all active scenarios and randomized
trajectories at once.  CUDA is used when requested and available; the same code
provides a deterministic CPU reference path for development and tests.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import dimod
import numpy as np
from scipy import sparse

from qubo_model import CompactQubo


_RUN_SEED_STRIDE = 0x9E3779B97F4A7C15
QuboLike = dimod.BinaryQuadraticModel | CompactQubo


@dataclass(frozen=True)
class SBMConfig:
    steps: int = 10_000
    runs: int = 16
    dt: float = 1.0
    a0: float = 1.0
    c0: float = 0.0
    gamma: float = 0.0
    initial_scale: float = 0.05
    dtype: str = "float32"
    run_batch_size: int | None = None

    def validate(self) -> None:
        if self.steps < 1 or self.runs < 1:
            raise ValueError("steps and runs must be positive")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(self.a0) or self.a0 <= 0.0:
            raise ValueError("a0 must be finite and positive")
        if not math.isfinite(self.c0) or self.c0 < 0.0:
            raise ValueError("c0 must be finite and non-negative")
        if not math.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        if not math.isfinite(self.initial_scale) or self.initial_scale <= 0.0:
            raise ValueError("initial_scale must be finite and positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.run_batch_size is not None and self.run_batch_size < 1:
            raise ValueError("run_batch_size must be positive")


@dataclass
class SBMSolveResult:
    sample: np.ndarray
    energy: float
    seed: int
    raw_run: int
    solve_seconds: float


@dataclass(frozen=True)
class _ProblemSlice:
    start: int
    stop: int
    labels: tuple[Any, ...] | None
    c0: float


def resolve_torch_device(requested: str) -> str:
    """Resolve ``auto``/``gpu`` and reject unavailable CUDA explicitly."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError("PyTorch is required for the batched SBM backend") from exc

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "gpu":
        requested = "cuda"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation has no CUDA device"
        )
    if device.type == "cuda" and device.index is not None:
        if not 0 <= device.index < torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"device_count={torch.cuda.device_count()}"
            )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA devices are supported")
    return str(device)


def _bqm_ising_vectors(
    bqm: QuboLike,
) -> tuple[tuple[Any, ...] | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, direct force field, upper K edges, and K weights.

    Dimod uses ``E = offset + h*s + J*s*s``.  dSB uses the force
    ``K*sign(x) - h``, so ``K=-J`` and the returned field is ``-h``.
    """

    if isinstance(bqm, CompactQubo):
        labels = None
        linear = bqm.linear
        row = bqm.heads
        col = bqm.tails
        quadratic = bqm.quadratic
        offset = bqm.offset
        vartype = dimod.BINARY
    elif isinstance(bqm, dimod.BinaryQuadraticModel):
        labels = tuple(bqm.variables)
        vectors = bqm.to_numpy_vectors(
            variable_order=labels,
            sort_indices=True,
            sort_labels=False,
        )
        linear = np.asarray(vectors.linear_biases, dtype=np.float64)
        row = np.asarray(vectors.quadratic.row_indices, dtype=np.int64)
        col = np.asarray(vectors.quadratic.col_indices, dtype=np.int64)
        quadratic = np.asarray(vectors.quadratic.biases, dtype=np.float64)
        offset = float(vectors.offset)
        vartype = bqm.vartype
    else:
        raise TypeError("every model must be a BinaryQuadraticModel or CompactQubo")
    if not (
        np.isfinite(linear).all()
        and np.isfinite(quadratic).all()
        and math.isfinite(offset)
    ):
        raise ValueError("BQM coefficients must be finite")

    if vartype is dimod.BINARY:
        ising_field = 0.5 * linear
        ising_quadratic = 0.25 * quadratic
        np.add.at(ising_field, row, ising_quadratic)
        np.add.at(ising_field, col, ising_quadratic)
    elif vartype is dimod.SPIN:
        ising_field = linear.copy()
        ising_quadratic = quadratic.copy()
    else:  # pragma: no cover - dimod currently supports these two vartypes
        raise ValueError(f"unsupported BQM vartype {vartype!r}")

    nonzero = ising_quadratic != 0.0
    return (
        labels,
        -ising_field,
        row[nonzero],
        col[nonzero],
        -ising_quadratic[nonzero],
    )


def _automatic_c0(force_field: np.ndarray, couplings: np.ndarray) -> float:
    """Scale coupling force using the same RMS principle as the CPU solver."""

    n = len(force_field)
    force_norm = math.sqrt(
        2.0 * float(np.dot(couplings, couplings))
        + 2.0 * float(np.dot(force_field, force_field))
    )
    return 0.5 * math.sqrt(max(n, 1)) / force_norm if force_norm > 0.0 else 1.0


def _pack_models(
    models: Sequence[QuboLike],
    dtype: np.dtype[Any],
    configured_c0: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[_ProblemSlice]]:
    total_variables = sum(
        model.num_variables if isinstance(model, CompactQubo) else len(model.variables)
        for model in models
    )
    if total_variables == 0:
        raise ValueError("BQM models must not be empty")

    force_fields: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    slices: list[_ProblemSlice] = []
    cursor = 0
    for model in models:
        labels, force_field, row, col, couplings = _bqm_ising_vectors(model)
        n = len(force_field)
        if n == 0:
            raise ValueError("BQM models must not be empty")
        c0 = configured_c0 or _automatic_c0(force_field, couplings)
        slices.append(_ProblemSlice(cursor, cursor + n, labels, c0))
        force_fields.append(force_field.astype(dtype, copy=False))
        if len(couplings):
            row_parts.extend((row + cursor, col + cursor))
            col_parts.extend((col + cursor, row + cursor))
            values = couplings.astype(dtype, copy=False)
            value_parts.extend((values, values))
        cursor += n

    if value_parts:
        rows = np.concatenate(row_parts)
        columns = np.concatenate(col_parts)
        values = np.concatenate(value_parts)
        matrix = sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(total_variables, total_variables),
            dtype=dtype,
        )
        matrix.sum_duplicates()
        matrix.sort_indices()
    else:
        matrix = sparse.csr_matrix(
            (total_variables, total_variables), dtype=dtype
        )
    field = np.concatenate(force_fields)
    c0_rows = np.empty(total_variables, dtype=dtype)
    for problem in slices:
        c0_rows[problem.start : problem.stop] = problem.c0
    return matrix, field, c0_rows, slices


class TorchBatchSBMSolver:
    """Solve independent BQMs together through one sparse block matrix."""

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_torch_device(device)

    def solve_batch(
        self,
        models: Sequence[QuboLike],
        config: SBMConfig,
        seeds: Sequence[int],
    ) -> list[SBMSolveResult]:
        config.validate()
        if not models:
            return []
        if len(models) != len(seeds):
            raise ValueError("one deterministic seed is required per BQM")
        if any(seed < 0 or seed >= 2**63 for seed in seeds):
            raise ValueError("seeds must be in [0, 2**63)")

        import torch

        torch_dtype = torch.float32 if config.dtype == "float32" else torch.float64
        numpy_dtype = np.dtype(np.float32 if config.dtype == "float32" else np.float64)
        pack_started = time.perf_counter()
        matrix_cpu, field_cpu, c0_cpu, slices = _pack_models(
            models, numpy_dtype, config.c0
        )
        device = torch.device(self.device)
        # SciPy uses int32 indices while the packed graph remains below its
        # limits.  Preserve that representation to halve GPU CSR index memory;
        # PyTorch accepts both int32 and int64 CSR indices.
        crow = torch.from_numpy(matrix_cpu.indptr).to(device)
        columns = torch.from_numpy(matrix_cpu.indices).to(device)
        values = torch.from_numpy(matrix_cpu.data).to(device=device, dtype=torch_dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            matrix = torch.sparse_csr_tensor(
                crow,
                columns,
                values,
                size=matrix_cpu.shape,
                dtype=torch_dtype,
                device=device,
                check_invariants=False,
            )
        field = torch.as_tensor(
            field_cpu, dtype=torch_dtype, device=device
        ).reshape(-1, 1)
        c0_rows = torch.as_tensor(
            c0_cpu, dtype=torch_dtype, device=device
        ).reshape(-1, 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pack_seconds = time.perf_counter() - pack_started

        best_energies = np.full(len(models), np.inf, dtype=np.float64)
        best_samples: list[np.ndarray | None] = [None] * len(models)
        best_runs = np.full(len(models), -1, dtype=np.int64)
        run_batch_size = min(config.run_batch_size or config.runs, config.runs)
        solve_started = time.perf_counter()

        with torch.inference_mode():
            for run_start in range(0, config.runs, run_batch_size):
                width = min(run_batch_size, config.runs - run_start)
                shape = (matrix_cpu.shape[0], width)
                x = torch.empty(shape, dtype=torch_dtype, device=device)
                y = torch.empty_like(x)
                for problem_index, problem in enumerate(slices):
                    for local_run in range(width):
                        run = run_start + local_run
                        generator = torch.Generator(device=device)
                        generator.manual_seed(
                            (int(seeds[problem_index]) + _RUN_SEED_STRIDE * run)
                            % (2**63)
                        )
                        x[problem.start : problem.stop, local_run].uniform_(
                            -config.initial_scale,
                            config.initial_scale,
                            generator=generator,
                        )
                        y[problem.start : problem.stop, local_run].uniform_(
                            -config.initial_scale,
                            config.initial_scale,
                            generator=generator,
                        )

                scratch = torch.empty_like(x)
                wall = torch.empty(shape, dtype=torch.bool, device=device)
                old_y = torch.empty_like(y) if config.gamma else None
                for step in range(config.steps):
                    torch.sign(x, out=scratch)
                    scratch.masked_fill_(scratch == 0, 1.0)
                    if old_y is not None:
                        old_y.copy_(y)
                    force = torch.sparse.mm(matrix, scratch)
                    force.add_(field).mul_(c0_rows)
                    pressure = config.a0 * step / config.steps
                    force.add_(x, alpha=pressure - config.a0)
                    y.add_(force, alpha=config.dt)
                    x.add_(y, alpha=config.a0 * config.dt)
                    torch.abs(x, out=scratch)
                    torch.gt(scratch, 1.0, out=wall)
                    x.clamp_(-1.0, 1.0)
                    y.masked_fill_(wall, 0.0)
                    if old_y is not None:
                        y.add_(old_y, alpha=config.gamma * config.dt)

                torch.sign(x, out=scratch)
                scratch.masked_fill_(scratch == 0, 1.0)
                binary = ((scratch + 1.0) * 0.5).to(torch.uint8).cpu().numpy()
                for problem_index, (model, problem) in enumerate(zip(models, slices)):
                    samples = binary[problem.start : problem.stop].T
                    if isinstance(model, CompactQubo):
                        energies = model.energies(samples)
                    else:
                        energies = np.asarray(
                            model.energies((samples, problem.labels)), dtype=np.float64
                        )
                    local = int(np.argmin(energies))
                    if energies[local] < best_energies[problem_index]:
                        best_energies[problem_index] = float(energies[local])
                        best_samples[problem_index] = samples[local].copy()
                        best_runs[problem_index] = run_start + local

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        solve_seconds = time.perf_counter() - solve_started
        per_model_seconds = (pack_seconds + solve_seconds) / len(models)
        results: list[SBMSolveResult] = []
        for index, sample in enumerate(best_samples):
            if sample is None:  # pragma: no cover - protected by validation
                raise RuntimeError("SBM produced no sample")
            results.append(
                SBMSolveResult(
                    sample=sample,
                    energy=float(best_energies[index]),
                    seed=int(seeds[index]),
                    raw_run=int(best_runs[index]),
                    solve_seconds=per_model_seconds,
                )
            )
        return results
