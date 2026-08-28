#!/usr/bin/env python3
"""GPU-capable simulated bifurcation solver for Ocean BQMs.

The public entry points are :func:`solve_bqm` and
:class:`SimulatedBifurcationSampler`.  Both return a ``dimod.SampleSet`` in the
input model's vartype.  The sample set contains raw variable assignments and
energies only; interpreting the ``x_<asset>_<state>`` labels produced by
``market_to_qubo.py`` is deliberately outside this module.

The dynamics implement ballistic or discrete simulated bifurcation (bSB/dSB)
with the symplectic-Euler update and perfectly inelastic walls described in
Goto, Tatsumura, and Dixon, Science Advances 7, eabe7953 (2021).  Linear Ising
biases enter as the direct field force used by generalized SB.  This avoids an
extra oscillator and is especially important for penalty-heavy QUBOs, whose
binary-to-spin conversion can produce large fields that nearly cancel edges.

PyTorch supplies parallel dense and sparse matrix kernels.  CUDA is selected
automatically when it is available; otherwise its multithreaded CPU backend is
used.  Independent reads are evaluated as matrix columns and automatically
batched to keep state memory bounded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import dimod
import numpy as np


Variant = Literal["discrete", "ballistic"]
MatrixMode = Literal["auto", "dense", "sparse"]

__all__ = [
    "SimulatedBifurcationSampler",
    "load_bqm",
    "simulated_bifurcation",
    "solve",
    "solve_bqm",
]


@dataclass(frozen=True)
class _IsingProblem:
    """Contiguous, label-preserving Ising representation of a BQM."""

    labels: tuple[Any, ...]
    linear: np.ndarray
    row: np.ndarray
    col: np.ndarray
    quadratic: np.ndarray
    offset: float
    vartype: dimod.Vartype

    @property
    def num_variables(self) -> int:
        return len(self.labels)

    @property
    def num_oscillators(self) -> int:
        return self.num_variables


def load_bqm(path: str | os.PathLike[str]) -> dimod.BinaryQuadraticModel:
    """Load an Ocean BQM file such as one exported by ``market_to_qubo.py``."""

    with Path(path).open("rb") as model_file:
        return dimod.BinaryQuadraticModel.from_file(model_file)


def _as_ising_problem(bqm: dimod.BinaryQuadraticModel) -> _IsingProblem:
    """Convert BINARY vectors to Ising vectors without allocating another BQM."""

    if not isinstance(bqm, dimod.BinaryQuadraticModel):
        raise TypeError("bqm must be a dimod.BinaryQuadraticModel")

    labels = tuple(bqm.variables)
    vectors = bqm.to_numpy_vectors(
        variable_order=labels,
        dtype=np.float64,
        index_dtype=np.int64,
        sort_indices=True,
        sort_labels=False,
    )
    linear = np.asarray(vectors.linear_biases, dtype=np.float64)
    row = np.asarray(vectors.quadratic.row_indices, dtype=np.int64)
    col = np.asarray(vectors.quadratic.col_indices, dtype=np.int64)
    quadratic = np.asarray(vectors.quadratic.biases, dtype=np.float64)
    offset = float(vectors.offset)

    if (
        not np.all(np.isfinite(linear))
        or not np.all(np.isfinite(quadratic))
        or not math.isfinite(offset)
    ):
        raise ValueError("all BQM biases and the offset must be finite")

    if bqm.vartype is dimod.BINARY:
        # x=(s+1)/2 gives q*x_i*x_j = q/4*(s_i*s_j+s_i+s_j+1).
        ising_linear = linear * 0.5
        ising_quadratic = quadratic * 0.25
        np.add.at(ising_linear, row, ising_quadratic)
        np.add.at(ising_linear, col, ising_quadratic)
        ising_offset = offset + 0.5 * float(linear.sum())
        ising_offset += 0.25 * float(quadratic.sum())
    elif bqm.vartype is dimod.SPIN:
        ising_linear = linear.copy()
        ising_quadratic = quadratic.copy()
        ising_offset = offset
    else:  # pragma: no cover - dimod BQMs currently allow only these vartypes.
        raise ValueError(f"unsupported BQM vartype: {bqm.vartype!r}")

    # Zero-valued interactions needlessly increase sparse work and density.
    nonzero = ising_quadratic != 0.0
    row = row[nonzero]
    col = col[nonzero]
    ising_quadratic = ising_quadratic[nonzero]
    return _IsingProblem(
        labels=labels,
        linear=ising_linear,
        row=row,
        col=col,
        quadratic=ising_quadratic,
        offset=ising_offset,
        vartype=bqm.vartype,
    )


def _normalization(problem: _IsingProblem, coupling_gain: float) -> float:
    """Return c0 generalized from c1/sqrt(N) to arbitrary edge weights.

    For a dense N-spin matrix with unit-magnitude weights this reduces to
    approximately ``coupling_gain / sqrt(N)``.  RMS normalization also behaves
    sensibly for sparse graphs and makes a uniform rescaling of a BQM irrelevant
    to the oscillator trajectory.
    """

    quadratic_norm = float(np.linalg.norm(problem.quadratic))
    linear_norm = float(np.linalg.norm(problem.linear))
    # Pair weights occur twice in the symmetric force matrix; direct fields
    # occur once.  Combining those squared norms preserves scale invariance.
    force_norm = math.hypot(math.sqrt(2.0) * quadratic_norm, linear_norm)
    if force_norm == 0.0:
        return 0.0
    return (
        coupling_gain
        * math.sqrt(max(problem.num_oscillators - 1, 1))
        / force_norm
    )


def _edge_vectors(
    problem: _IsingProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build upper-triangle couplings for E = offset - sum(K_ij*s_i*s_j)."""

    # The SB paper minimizes -sum(K_ij*s_i*s_j), whereas dimod's Ising
    # convention adds J_ij*s_i*s_j.  Hence K=-J.
    return problem.row, problem.col, -problem.quadratic


def _available_memory(torch: Any, device: Any) -> int:
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        # A conservative portable fallback.  It only affects automatic choices.
        return 2 * 1024**3


def _choose_matrix_mode(
    requested: MatrixMode,
    num_oscillators: int,
    directed_nonzero: int,
    element_size: int,
    available_memory: int,
) -> tuple[Literal["dense", "sparse"], float, int]:
    possible_entries = num_oscillators * num_oscillators
    density = directed_nonzero / possible_entries if possible_entries else 0.0
    dense_bytes = possible_entries * element_size

    if requested != "auto":
        return requested, density, dense_bytes

    dense_fits = dense_bytes <= int(available_memory * 0.35)
    # GEMM wins decisively for small matrices.  For larger matrices CSR avoids
    # both zero arithmetic and quadratic memory unless the graph is quite dense.
    use_dense = dense_fits and (
        num_oscillators <= 512
        or (num_oscillators <= 4096 and density >= 0.025)
        or density >= 0.15
    )
    return ("dense" if use_dense else "sparse"), density, dense_bytes


def _build_torch_matrix(
    problem: _IsingProblem,
    torch: Any,
    device: Any,
    dtype: Any,
    matrix_mode: Literal["dense", "sparse"],
) -> Any:
    rows, cols, weights = _edge_vectors(problem)
    n = problem.num_oscillators
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64

    if matrix_mode == "dense":
        matrix = torch.zeros((n, n), dtype=dtype, device=device)
        if len(weights):
            row_tensor = torch.as_tensor(rows, dtype=torch.int64, device=device)
            col_tensor = torch.as_tensor(cols, dtype=torch.int64, device=device)
            value_tensor = torch.as_tensor(
                weights.astype(numpy_dtype, copy=False),
                dtype=dtype,
                device=device,
            )
            matrix[row_tensor, col_tensor] = value_tensor
            matrix[col_tensor, row_tensor] = value_tensor
        return matrix

    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - market_to_qubo needs scipy.
        raise RuntimeError("sparse SB execution requires scipy") from exc

    if len(weights):
        upper = sparse.csr_matrix(
            (weights.astype(numpy_dtype, copy=False), (rows, cols)),
            shape=(n, n),
            dtype=numpy_dtype,
        )
        matrix_csr = upper + upper.T
    else:
        matrix_csr = sparse.csr_matrix((n, n), dtype=numpy_dtype)
    matrix_csr.sort_indices()

    crow = torch.from_numpy(
        matrix_csr.indptr.astype(np.int64, copy=False)
    ).to(device=device)
    column = torch.from_numpy(
        matrix_csr.indices.astype(np.int64, copy=False)
    ).to(device=device)
    values = torch.from_numpy(
        matrix_csr.data.astype(numpy_dtype, copy=False)
    ).to(device=device, dtype=dtype)
    # PyTorch still labels the CSR constructor beta even though CSR matrix
    # multiplication is the intended stable operation here.  Do not leak that
    # implementation-status warning from every solver process.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        return torch.sparse_csr_tensor(
            crow,
            column,
            values,
            size=(n, n),
            dtype=dtype,
            device=device,
            check_invariants=False,
        )


def _auto_batch_size(
    num_reads: int,
    num_oscillators: int,
    element_size: int,
    available_memory: int,
    device_type: str,
    heated: bool,
) -> int:
    # x, y, force, sign/scratch, wall mask, sparse-mm temporary, and possibly
    # old y must coexist.  The safety factor also covers allocator workspaces.
    state_arrays = 7 + int(heated)
    if device_type == "cuda":
        budget = max(int(available_memory * 0.60), 32 * 1024**2)
    else:
        budget = min(max(int(available_memory * 0.20), 128 * 1024**2), 1024**3)
    bytes_per_read = max(num_oscillators, 1) * element_size * state_arrays
    return max(1, min(num_reads, budget // max(bytes_per_read, 1)))


def _run_dynamics(
    *,
    problem: _IsingProblem,
    matrix: Any,
    matrix_mode: Literal["dense", "sparse"],
    torch: Any,
    device: Any,
    dtype: Any,
    num_reads: int,
    batch_size: int,
    num_steps: int,
    time_step: float,
    a0: float,
    c0: float,
    initial_scale: float,
    variant: Variant,
    heating: float,
    seed: int,
) -> np.ndarray:
    """Run all oscillator batches and return raw samples on the CPU."""

    n = problem.num_oscillators
    output = np.empty((num_reads, problem.num_variables), dtype=np.int8)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
    field = torch.as_tensor(
        (-problem.linear).astype(numpy_dtype, copy=False),
        dtype=dtype,
        device=device,
    ).reshape(n, 1)

    cursor = 0
    with torch.inference_mode():
        while cursor < num_reads:
            width = min(batch_size, num_reads - cursor)
            shape = (n, width)
            x = torch.empty(shape, dtype=dtype, device=device)
            y = torch.empty_like(x)
            x.uniform_(-initial_scale, initial_scale, generator=generator)
            y.uniform_(-initial_scale, initial_scale, generator=generator)

            # dSB needs a sign buffer; bSB reuses the same allocation as wall
            # scratch.  Dense mm writes into a persistent force buffer.
            scratch = torch.empty_like(x)
            force_buffer = torch.empty_like(x)
            wall = torch.empty(shape, dtype=torch.bool, device=device)
            old_y = torch.empty_like(y) if heating else None

            for step in range(num_steps):
                if variant == "discrete":
                    torch.sign(x, out=scratch)
                    source = scratch
                else:
                    source = x

                if old_y is not None:
                    old_y.copy_(y)

                if matrix_mode == "dense":
                    torch.mm(matrix, source, out=force_buffer)
                    force = force_buffer
                else:
                    force = torch.sparse.mm(matrix, source)

                # Symplectic Euler: update momentum first, then position using
                # the new momentum.  K*s-h is the generalized dSB/bSB force
                # for dimod energy +J*s*s+h*s.  a(t) rises from 0 to a0.
                pressure = a0 * step / (num_steps - 1)
                force.add_(field).mul_(c0).add_(
                    x, alpha=-(a0 - pressure)
                )
                y.add_(force, alpha=time_step)
                x.add_(y, alpha=a0 * time_step)

                # Perfectly inelastic walls: clamp x and erase the momentum of
                # every oscillator that crossed either wall during this step.
                torch.abs(x, out=scratch)
                torch.gt(scratch, 1.0, out=wall)
                x.clamp_(-1.0, 1.0)
                y.masked_fill_(wall, 0.0)

                # Heated SB applies gamma*y_old after wall handling (Kanao and
                # Goto, Communications Physics 5, 153, equations 16-22).
                if old_y is not None:
                    y.add_(old_y, alpha=heating * time_step)

            torch.sign(x, out=scratch)
            # Exact zeros have measure zero with random initialization, but a
            # deterministic tie rule keeps the returned domain strictly +/-1.
            scratch.masked_fill_(scratch == 0, 1.0)
            physical_spins = scratch

            if problem.vartype is dimod.BINARY:
                samples = ((physical_spins + 1.0) * 0.5).to(torch.int8)
            else:
                samples = physical_spins.to(torch.int8)
            output[cursor : cursor + width] = samples.T.cpu().numpy()
            cursor += width

    return output


def _greedy_polish(
    problem: _IsingProblem,
    samples: np.ndarray,
    sweeps: int,
    pair_rounds: int,
    pair_work_budget: int,
    tolerance: float,
) -> tuple[np.ndarray, int, int, int]:
    """Bounded one- and two-spin descent over a few best reads.

    A two-spin move only needs consideration when its variables share an edge:
    for an unconnected pair its delta is the sum of two non-improving one-spin
    deltas.  This observation makes pair descent practical without knowing any
    model-specific variable grouping.
    """

    if (
        not len(samples)
        or (sweeps == 0 and pair_rounds == 0)
        or problem.num_variables == 0
    ):
        return samples, 0, 0, 0

    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SB polishing requires scipy") from exc

    n = problem.num_variables
    upper = sparse.csr_matrix(
        (problem.quadratic, (problem.row, problem.col)),
        shape=(n, n),
        dtype=np.float64,
    )
    interactions = upper + upper.T
    interactions.sort_indices()

    polished = np.asarray(samples, dtype=np.int8).copy()
    local_fields = np.asarray(interactions @ polished.T).T
    local_fields += problem.linear[None, :]
    single_flips = 0

    def single_sweep() -> int:
        nonlocal single_flips
        sweep_flips = 0
        for variable in range(n):
            delta = -2.0 * polished[:, variable] * local_fields[:, variable]
            selected = np.flatnonzero(delta < -tolerance)
            if not len(selected):
                continue

            old_spins = polished[selected, variable].copy()
            polished[selected, variable] *= -1
            start = interactions.indptr[variable]
            stop = interactions.indptr[variable + 1]
            neighbors = interactions.indices[start:stop]
            weights = interactions.data[start:stop]
            if len(neighbors):
                local_fields[np.ix_(selected, neighbors)] += (
                    (-2.0 * old_spins)[:, None] * weights[None, :]
                )
            count = len(selected)
            single_flips += count
            sweep_flips += count
        return sweep_flips

    for _ in range(sweeps):
        if single_sweep() == 0:
            break

    edge_count = len(problem.quadratic)
    reads = len(polished)
    affordable_rounds = min(
        pair_rounds,
        pair_work_budget // max(edge_count * reads, 1),
    )
    pair_moves = 0
    pair_rounds_run = 0
    if edge_count and affordable_rounds:
        # Bound the largest temporary pair-delta block to about 32 MiB.
        chunk_edges = max(1, min(edge_count, (32 * 1024**2) // (8 * reads)))
        for _ in range(affordable_rounds):
            single_delta = -2.0 * polished * local_fields
            best_delta = np.full(reads, np.inf, dtype=np.float64)
            best_edge = np.full(reads, -1, dtype=np.int64)

            for start in range(0, edge_count, chunk_edges):
                stop = min(start + chunk_edges, edge_count)
                edge_slice = slice(start, stop)
                edge_row = problem.row[edge_slice]
                edge_col = problem.col[edge_slice]
                edge_weight = problem.quadratic[edge_slice]
                deltas = (
                    single_delta[:, edge_row]
                    + single_delta[:, edge_col]
                    + 4.0
                    * edge_weight[None, :]
                    * polished[:, edge_row]
                    * polished[:, edge_col]
                )
                local_choice = np.argmin(deltas, axis=1)
                local_best = deltas[np.arange(reads), local_choice]
                improve = local_best < best_delta
                best_delta[improve] = local_best[improve]
                best_edge[improve] = start + local_choice[improve]

            selected_reads = np.flatnonzero(best_delta < -tolerance)
            if not len(selected_reads):
                break
            pair_rounds_run += 1

            # Reads are independent.  Applying the two flips sequentially to
            # each read keeps its cached local fields exact.
            for read in selected_reads:
                edge = best_edge[read]
                variables = (problem.row[edge], problem.col[edge])
                for variable in variables:
                    old_spin = polished[read, variable]
                    polished[read, variable] = -old_spin
                    start = interactions.indptr[variable]
                    stop = interactions.indptr[variable + 1]
                    neighbors = interactions.indices[start:stop]
                    weights = interactions.data[start:stop]
                    local_fields[read, neighbors] += -2.0 * old_spin * weights
            pair_moves += len(selected_reads)

        # A pair move can expose a further one-spin decrease.
        if pair_moves:
            single_sweep()

    return polished, single_flips, pair_moves, pair_rounds_run


class SimulatedBifurcationSampler(dimod.Sampler):
    """A ``dimod`` sampler backed by batched simulated bifurcation dynamics."""

    # Concrete class attributes satisfy dimod.Sampler's abstract interface;
    # each instance receives independent dictionaries in __init__.
    properties: dict[str, Any] | None = None
    parameters: dict[str, list[Any]] | None = None

    def __init__(self) -> None:
        self.properties = {
            "algorithm": "simulated bifurcation",
            "supported_variants": ("discrete", "ballistic"),
            "gpu_backend": "PyTorch CUDA",
        }
        self.parameters = {
            "num_reads": [],
            "num_steps": [],
            "time_step": [],
            "variant": [],
            "heating": [],
            "device": [],
            "dtype": [],
            "matrix_mode": [],
            "batch_size": [],
            "seed": [],
            "initial_scale": [],
            "coupling_gain": [],
            "a0": [],
            "polish": [],
            "polish_reads": [],
            "polish_sweeps": [],
            "polish_pair_rounds": [],
            "polish_pair_work_budget": [],
            "threads": [],
        }

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        *,
        num_reads: int = 128,
        num_steps: int = 1000,
        time_step: float = 0.1,
        variant: Variant = "discrete",
        heating: float = 0.0,
        device: str = "auto",
        dtype: str = "float32",
        matrix_mode: MatrixMode = "auto",
        batch_size: int | None = None,
        seed: int | None = None,
        initial_scale: float = 0.01,
        coupling_gain: float = 0.5,
        a0: float = 1.0,
        polish: bool = True,
        polish_reads: int = 8,
        polish_sweeps: int = 2,
        polish_pair_rounds: int = 32,
        polish_pair_work_budget: int = 50_000_000,
        polish_tolerance: float = 1e-12,
        threads: int | None = None,
        **kwargs: Any,
    ) -> dimod.SampleSet:
        """Solve a BQM and return raw samples with detailed run metadata.

        Args:
            bqm: Binary or spin-valued ``dimod.BinaryQuadraticModel``.
            num_reads: Number of independent oscillator trajectories.
            num_steps: Symplectic-Euler steps in each trajectory.
            time_step: Integration time step ``Delta t``.
            variant: ``"discrete"`` (dSB, usually higher accuracy) or
                ``"ballistic"`` (bSB, often faster convergence).
            heating: Nonnegative heated-SB coefficient.  Zero uses canonical
                bSB/dSB; ``0.06`` is a useful experimental heated-SB value.
            device: Torch device (for example ``"cuda"`` or ``"cpu"``), or
                ``"auto"`` to prefer CUDA.
            dtype: Oscillator precision, ``"float32"`` or ``"float64"``.
            matrix_mode: Force dense GEMM, sparse CSR multiplication, or select
                automatically from graph density and available memory.
            batch_size: Parallel reads per dynamics batch.  ``None`` chooses a
                memory-aware value.
            seed: Random seed.  The generated seed is reported when omitted.
            initial_scale: Half-width of uniform initial x and y values.
            coupling_gain: Dimensionless coupling normalization (paper c1).
            a0: Oscillator detuning and final linear pressure.
            polish: Apply exact-objective single-spin descent to the best reads.
            polish_reads: Maximum number of best trajectories to polish.
            polish_sweeps: Maximum sequential descent sweeps.
            polish_pair_rounds: Maximum best improving two-spin moves per read.
            polish_pair_work_budget: Upper bound on edge/read pair evaluations.
            polish_tolerance: Required energy decrease for a polishing flip.
            threads: Optional Torch CPU thread count for this call.

        Returns:
            A ``dimod.SampleSet`` in ``bqm.vartype``, sorted by energy.  Runtime,
            backend, algorithm, model, and energy statistics are in ``.info``.
        """

        total_start = time.perf_counter()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown sampling argument(s): {unknown}")
        if num_reads < 1:
            raise ValueError("num_reads must be at least 1")
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be finite and positive")
        if variant not in ("discrete", "ballistic"):
            raise ValueError("variant must be 'discrete' or 'ballistic'")
        if not math.isfinite(heating) or heating < 0.0:
            raise ValueError("heating must be finite and nonnegative")
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if matrix_mode not in ("auto", "dense", "sparse"):
            raise ValueError("matrix_mode must be 'auto', 'dense', or 'sparse'")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not math.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ValueError("initial_scale must be finite and positive")
        if not math.isfinite(coupling_gain) or coupling_gain <= 0.0:
            raise ValueError("coupling_gain must be finite and positive")
        if not math.isfinite(a0) or a0 <= 0.0:
            raise ValueError("a0 must be finite and positive")
        if (
            polish_reads < 0
            or polish_sweeps < 0
            or polish_pair_rounds < 0
            or polish_pair_work_budget < 0
        ):
            raise ValueError("polishing counts and work budget cannot be negative")
        if not math.isfinite(polish_tolerance) or polish_tolerance < 0.0:
            raise ValueError("polish_tolerance must be finite and nonnegative")
        if threads is not None and threads < 1:
            raise ValueError("threads must be at least 1")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "simulated_bifurcation requires PyTorch (torch)"
            ) from exc

        if device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device_name = "cuda"
        else:
            device_name = device
        torch_device = torch.device(device_name)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if torch_device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA Torch devices are supported")

        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        element_size = torch.empty((), dtype=torch_dtype).element_size()
        if seed is None:
            seed = secrets.randbits(63)
        if not 0 <= seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")

        preprocess_start = total_start
        problem = _as_ising_problem(bqm)
        directed_nonzero = 2 * len(problem.quadratic)
        available = _available_memory(torch, torch_device)
        selected_mode, density, dense_bytes = _choose_matrix_mode(
            matrix_mode,
            problem.num_oscillators,
            directed_nonzero,
            element_size,
            available,
        )
        c0 = _normalization(problem, coupling_gain)

        previous_threads = torch.get_num_threads()
        if threads is not None and torch_device.type == "cpu":
            torch.set_num_threads(threads)

        try:
            if directed_nonzero:
                matrix = _build_torch_matrix(
                    problem,
                    torch,
                    torch_device,
                    torch_dtype,
                    selected_mode,
                )
                remaining_memory = _available_memory(torch, torch_device)
                actual_batch_size = batch_size or _auto_batch_size(
                    num_reads,
                    problem.num_oscillators,
                    element_size,
                    remaining_memory,
                    torch_device.type,
                    bool(heating),
                )
            else:
                matrix = None
                selected_mode = "sparse"
                actual_batch_size = min(batch_size or num_reads, num_reads)

            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            preprocess_time = time.perf_counter() - preprocess_start

            sample_start = time.perf_counter()
            if matrix is None:
                # A linear-only BQM is separable, so avoid launching dynamics
                # and return its exact minimizer (ties choose the lower bit).
                spins = np.where(problem.linear < 0.0, 1, -1).astype(np.int8)
                if bqm.vartype is dimod.BINARY:
                    one_sample = ((spins + 1) // 2).astype(np.int8)
                else:
                    one_sample = spins
                samples = np.broadcast_to(
                    one_sample, (num_reads, problem.num_variables)
                ).copy()
            else:
                samples = _run_dynamics(
                    problem=problem,
                    matrix=matrix,
                    matrix_mode=selected_mode,
                    torch=torch,
                    device=torch_device,
                    dtype=torch_dtype,
                    num_reads=num_reads,
                    batch_size=actual_batch_size,
                    num_steps=num_steps,
                    time_step=time_step,
                    a0=a0,
                    c0=c0,
                    initial_scale=initial_scale,
                    variant=variant,
                    heating=heating,
                    seed=seed,
                )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            sampling_time = time.perf_counter() - sample_start
        finally:
            if threads is not None and torch_device.type == "cpu":
                torch.set_num_threads(previous_threads)

        postprocess_start = time.perf_counter()
        energies = np.asarray(
            bqm.energies((samples, problem.labels)), dtype=np.float64
        )
        flips = 0
        pair_moves = 0
        pair_rounds_run = 0
        polished_count = 0
        if (
            polish
            and polish_reads
            and (polish_sweeps or polish_pair_rounds)
            and problem.num_variables
        ):
            polished_count = min(polish_reads, num_reads)
            candidates = np.argsort(energies, kind="stable")[:polished_count]
            if bqm.vartype is dimod.BINARY:
                candidate_spins = (2 * samples[candidates] - 1).astype(
                    np.int8, copy=False
                )
            else:
                candidate_spins = samples[candidates]
            polished_spins, flips, pair_moves, pair_rounds_run = _greedy_polish(
                problem,
                candidate_spins,
                polish_sweeps,
                polish_pair_rounds,
                polish_pair_work_budget,
                polish_tolerance,
            )
            if bqm.vartype is dimod.BINARY:
                samples[candidates] = ((polished_spins + 1) // 2).astype(np.int8)
            else:
                samples[candidates] = polished_spins
            energies = np.asarray(
                bqm.energies((samples, problem.labels)), dtype=np.float64
            )

        order = np.argsort(energies, kind="stable")
        samples = samples[order]
        energies = energies[order]
        postprocessing_time = time.perf_counter() - postprocess_start
        total_time = time.perf_counter() - total_start

        if torch_device.type == "cuda":
            device_description = torch.cuda.get_device_name(torch_device)
        else:
            device_description = "CPU"
        info: dict[str, Any] = {
            "solver": "simulated_bifurcation",
            "variant": variant,
            "heated": bool(heating),
            "heating_coefficient": float(heating),
            "device": str(torch_device),
            "device_description": device_description,
            "torch_version": torch.__version__,
            "cpu_threads": (
                (threads or previous_threads) if torch_device.type == "cpu" else None
            ),
            "dtype": dtype,
            "matrix_mode": selected_mode,
            "matrix_density": float(density),
            "dense_matrix_bytes_estimate": int(dense_bytes),
            "num_variables": problem.num_variables,
            "num_interactions": len(problem.quadratic),
            "num_oscillators": problem.num_oscillators,
            "linear_bias_mode": "direct_field_force",
            "num_reads": num_reads,
            "parallel_batch_size": actual_batch_size,
            "num_batches": math.ceil(num_reads / actual_batch_size),
            "num_steps": num_steps,
            "time_step": float(time_step),
            "a0": float(a0),
            "coupling_gain": float(coupling_gain),
            "effective_coupling_c0": float(c0),
            "initial_scale": float(initial_scale),
            "seed": int(seed),
            "polish_enabled": bool(polish),
            "polished_reads": polished_count,
            "polish_sweeps": polish_sweeps if polish else 0,
            "polish_single_flips": flips,
            "polish_pair_rounds": polish_pair_rounds if polish else 0,
            "polish_pair_rounds_run": pair_rounds_run,
            "polish_pair_moves": pair_moves,
            "polish_flips": flips + 2 * pair_moves,
            "polish_pair_work_budget": polish_pair_work_budget,
            "preprocessing_time_s": preprocess_time,
            "sampling_time_s": sampling_time,
            "postprocessing_time_s": postprocessing_time,
            "total_time_s": total_time,
            "reads_per_second": num_reads / max(sampling_time, 1e-12),
            "best_energy": float(energies[0]),
            "mean_energy": float(np.mean(energies)),
            "energy_std": float(np.std(energies)),
            "worst_energy": float(energies[-1]),
        }

        return dimod.SampleSet.from_samples(
            (samples, problem.labels),
            vartype=bqm.vartype,
            energy=energies,
            info=info,
            num_occurrences=np.ones(num_reads, dtype=np.int64),
            sort_labels=False,
        )


def solve_bqm(
    bqm: dimod.BinaryQuadraticModel | str | os.PathLike[str],
    **sample_kwargs: Any,
) -> dimod.SampleSet:
    """Solve an in-memory BQM or an Ocean ``.bqm`` file.

    This convenience function is the simplest API for the files produced by
    ``market_to_qubo.py``.  File loading time is added to ``SampleSet.info``.
    """

    load_start = time.perf_counter()
    if isinstance(bqm, (str, os.PathLike)):
        model = load_bqm(bqm)
        source = str(Path(bqm).resolve())
    else:
        model = bqm
        source = "in-memory"
    load_time = time.perf_counter() - load_start

    result = SimulatedBifurcationSampler().sample(model, **sample_kwargs)
    result.info["model_source"] = source
    result.info["model_load_time_s"] = load_time
    result.info["total_time_s"] += load_time
    return result


# A short alias is convenient in notebooks while retaining the explicit API.
solve = solve_bqm
simulated_bifurcation = solve_bqm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve an Ocean BQM with GPU-capable simulated bifurcation."
    )
    parser.add_argument("bqm", type=Path, help="Ocean .bqm file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the complete dimod SampleSet as JSON",
    )
    parser.add_argument("--num-reads", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument(
        "--variant", choices=("discrete", "ballistic"), default="discrete"
    )
    parser.add_argument("--heating", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--matrix-mode", choices=("auto", "dense", "sparse"), default="auto"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-scale", type=float, default=0.01)
    parser.add_argument("--coupling-gain", type=float, default=0.5)
    parser.add_argument("--a0", type=float, default=1.0)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--polish-reads", type=int, default=8)
    parser.add_argument("--polish-sweeps", type=int, default=2)
    parser.add_argument("--polish-pair-rounds", type=int, default=32)
    parser.add_argument("--polish-pair-work-budget", type=int, default=50_000_000)
    parser.add_argument("--threads", type=int)
    return parser


def _best_result_json(sampleset: dimod.SampleSet) -> dict[str, Any]:
    first = sampleset.first
    return {
        "sample": dict(first.sample),
        "energy": float(first.energy),
        "num_occurrences": int(first.num_occurrences),
        "metadata": sampleset.info,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = solve_bqm(
        args.bqm,
        num_reads=args.num_reads,
        num_steps=args.num_steps,
        time_step=args.time_step,
        variant=args.variant,
        heating=args.heating,
        device=args.device,
        dtype=args.dtype,
        matrix_mode=args.matrix_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        initial_scale=args.initial_scale,
        coupling_gain=args.coupling_gain,
        a0=args.a0,
        polish=not args.no_polish,
        polish_reads=args.polish_reads,
        polish_sweeps=args.polish_sweeps,
        polish_pair_rounds=args.polish_pair_rounds,
        polish_pair_work_budget=args.polish_pair_work_budget,
        threads=args.threads,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as destination:
            json.dump(result.to_serializable(), destination, separators=(",", ":"))
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "best_energy": result.info["best_energy"],
                    "total_time_s": result.info["total_time_s"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(_best_result_json(result), indent=2))


if __name__ == "__main__":
    main()
