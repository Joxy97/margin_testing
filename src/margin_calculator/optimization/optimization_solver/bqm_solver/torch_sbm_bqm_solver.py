"""Batched simulated bifurcation using PyTorch sparse matrix kernels."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy

from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)

from ...optimization_problem.qubo_problem import QUBOProblem
from .bqm_solver import BQMSolver
from .bqm_solver_factory import BQMSolverFactory


_RUN_SEED_STRIDE = 0x9E3779B97F4A7C15
_PROBLEM_SEED_STRIDE = 0x0D1B54A32D192ED03
_MAX_TORCH_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class _IsingProblem:
    forceField: numpy.ndarray
    heads: numpy.ndarray
    tails: numpy.ndarray
    couplings: numpy.ndarray
    c0: float


@dataclass(frozen=True)
class _PackedIsingBatch:
    forceField: numpy.ndarray
    c0Rows: numpy.ndarray
    rowOffsets: numpy.ndarray
    columns: numpy.ndarray
    couplings: numpy.ndarray
    variableOffsets: numpy.ndarray


class TorchSBMBQMSolver(BQMSolver):
    """Solve scenario and trajectory batches through Torch sparse SpMM.

    AMD ROCm builds of PyTorch intentionally expose GPUs through the
    ``torch.cuda`` API. Constructor aliases ``rocm``, ``amd``, and ``hip``
    therefore resolve to the same internal ``cuda`` device as NVIDIA GPUs.
    """

    _defaults = {
        "steps": 10_000,
        "runs": 16,
        "dt": 1.0,
        "a0": 1.0,
        "c0": 0.0,
        "gamma": 0.0,
        "initial_scale": 0.05,
        "seed": 1,
        "dtype": "float32",
        "run_batch_size": None,
        "energy_chunk_size": 1_000_000,
    }

    def __init__(self, device: str = "auto") -> None:
        self.requestedDevice = str(device)
        self._resolvedDevice: str | None = None
        self._solveLock = Lock()

    @property
    def device(self) -> str:
        """Return the resolved Torch device, importing Torch lazily."""
        if self._resolvedDevice is None:
            self._resolvedDevice = self._resolveDevice(self.requestedDevice)
        return self._resolvedDevice

    @property
    def acceleratorBackend(self) -> str:
        """Return ``cpu``, ``cuda``, or ``rocm`` for the resolved device."""
        if self.device == "cpu":
            return "cpu"
        return "rocm" if self._torch().version.hip is not None else "cuda"

    def solve(
        self,
        problem: QUBOProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> BQMOptimizationResult:
        return self.solveMany([problem], solverParameters)[0]

    def solveMany(
        self,
        problems: Sequence[QUBOProblem],
        solverParameters: Mapping[str, Any] | None = None,
    ) -> list[BQMOptimizationResult]:
        """Solve all problems in one block-diagonal sparse Torch batch."""
        if not problems:
            return []
        if any(not isinstance(problem, QUBOProblem) for problem in problems):
            raise TypeError("Torch SBM problems must be QUBOProblem objects")
        parameters = self._getParameters(solverParameters)
        with self._solveLock:
            return self._solveBatch(problems, parameters)

    def _solveBatch(
        self,
        problems: Sequence[QUBOProblem],
        parameters: Mapping[str, Any],
    ) -> list[BQMOptimizationResult]:
        torch = self._torch()
        torch_dtype = (
            torch.float32
            if parameters["dtype"] == "float32"
            else torch.float64
        )
        numpy_dtype = (
            numpy.float32
            if parameters["dtype"] == "float32"
            else numpy.float64
        )
        packed = self._packProblems(
            problems,
            numpy_dtype,
            parameters["c0"],
        )
        device = torch.device(self.device)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            matrix = torch.sparse_csr_tensor(
                torch.from_numpy(packed.rowOffsets).to(device),
                torch.from_numpy(packed.columns).to(device),
                torch.from_numpy(packed.couplings).to(
                    device=device,
                    dtype=torch_dtype,
                ),
                size=(len(packed.forceField), len(packed.forceField)),
                dtype=torch_dtype,
                device=device,
                check_invariants=False,
            )
        field = torch.as_tensor(
            packed.forceField,
            dtype=torch_dtype,
            device=device,
        ).reshape(-1, 1)
        c0_rows = torch.as_tensor(
            packed.c0Rows,
            dtype=torch_dtype,
            device=device,
        ).reshape(-1, 1)

        candidates: list[list[tuple[numpy.ndarray, float]]] = [
            [] for _ in problems
        ]
        run_batch_size = min(
            parameters["run_batch_size"] or parameters["runs"],
            parameters["runs"],
        )
        with torch.inference_mode():
            for run_start in range(0, parameters["runs"], run_batch_size):
                width = min(
                    run_batch_size,
                    parameters["runs"] - run_start,
                )
                samples = self._runTrajectories(
                    torch,
                    matrix,
                    field,
                    c0_rows,
                    packed.variableOffsets,
                    width,
                    run_start,
                    parameters,
                    torch_dtype,
                    device,
                )
                for problem_index, problem in enumerate(problems):
                    start = int(packed.variableOffsets[problem_index])
                    stop = int(packed.variableOffsets[problem_index + 1])
                    problem_samples = samples[start:stop].T
                    energies = self._energies(
                        problem,
                        problem_samples,
                        parameters["energy_chunk_size"],
                    )
                    candidates[problem_index].extend(
                        (sample, float(energy))
                        for sample, energy in zip(problem_samples, energies)
                    )

        return [
            BQMOptimizationResult(
                *self._selectBestCandidates(
                    problem_candidates,
                    problem.oneHotGroups,
                )
            )
            for problem, problem_candidates in zip(problems, candidates)
        ]

    @staticmethod
    def _runTrajectories(
        torch: Any,
        matrix: Any,
        field: Any,
        c0Rows: Any,
        variableOffsets: numpy.ndarray,
        width: int,
        runStart: int,
        parameters: Mapping[str, Any],
        torchDtype: Any,
        device: Any,
    ) -> numpy.ndarray:
        shape = (int(variableOffsets[-1]), width)
        positions = torch.empty(shape, dtype=torchDtype, device=device)
        momenta = torch.empty_like(positions)
        for problem_index in range(len(variableOffsets) - 1):
            start = int(variableOffsets[problem_index])
            stop = int(variableOffsets[problem_index + 1])
            for local_run in range(width):
                run = runStart + local_run
                seed = (
                    parameters["seed"]
                    + _PROBLEM_SEED_STRIDE * problem_index
                    + _RUN_SEED_STRIDE * run
                ) % _MAX_TORCH_SEED
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                positions[start:stop, local_run].uniform_(
                    -parameters["initial_scale"],
                    parameters["initial_scale"],
                    generator=generator,
                )
                momenta[start:stop, local_run].uniform_(
                    -parameters["initial_scale"],
                    parameters["initial_scale"],
                    generator=generator,
                )

        scratch = torch.empty_like(positions)
        wall = torch.empty(shape, dtype=torch.bool, device=device)
        old_momenta = (
            torch.empty_like(momenta) if parameters["gamma"] else None
        )
        for step in range(parameters["steps"]):
            torch.sign(positions, out=scratch)
            scratch.masked_fill_(scratch == 0, 1.0)
            if old_momenta is not None:
                old_momenta.copy_(momenta)
            force = torch.sparse.mm(matrix, scratch)
            force.add_(field).mul_(c0Rows)
            pressure = parameters["a0"] * step / parameters["steps"]
            force.add_(positions, alpha=pressure - parameters["a0"])
            momenta.add_(force, alpha=parameters["dt"])
            positions.add_(
                momenta,
                alpha=parameters["a0"] * parameters["dt"],
            )
            torch.abs(positions, out=scratch)
            torch.gt(scratch, 1.0, out=wall)
            positions.clamp_(-1.0, 1.0)
            momenta.masked_fill_(wall, 0.0)
            if old_momenta is not None:
                momenta.add_(
                    old_momenta,
                    alpha=parameters["gamma"] * parameters["dt"],
                )

        torch.sign(positions, out=scratch)
        scratch.masked_fill_(scratch == 0, 1.0)
        return ((scratch + 1.0) * 0.5).to(torch.uint8).cpu().numpy()

    @classmethod
    def _packProblems(
        cls,
        problems: Sequence[QUBOProblem],
        dtype: Any,
        configuredC0: float,
    ) -> _PackedIsingBatch:
        converted = [
            cls._toIsingProblem(problem, dtype, configuredC0)
            for problem in problems
        ]
        variable_offsets = numpy.empty(len(problems) + 1, dtype=numpy.int64)
        variable_offsets[0] = 0
        numpy.cumsum(
            [problem.variableCount for problem in problems],
            out=variable_offsets[1:],
        )
        force_field = numpy.concatenate(
            [problem.forceField for problem in converted]
        )
        c0_rows = numpy.concatenate(
            [
                numpy.full(len(problem.forceField), problem.c0, dtype=dtype)
                for problem in converted
            ]
        )

        rows = []
        columns = []
        values = []
        for index, problem in enumerate(converted):
            offset = variable_offsets[index]
            rows.extend((problem.heads + offset, problem.tails + offset))
            columns.extend((problem.tails + offset, problem.heads + offset))
            values.extend((problem.couplings, problem.couplings))
        if values:
            row = numpy.concatenate(rows)
            column = numpy.concatenate(columns)
            couplings = numpy.concatenate(values).astype(dtype, copy=False)
            order = numpy.lexsort((column, row))
            row = row[order]
            column = column[order]
            couplings = couplings[order]
        else:
            row = numpy.empty(0, dtype=numpy.int64)
            column = numpy.empty(0, dtype=numpy.int64)
            couplings = numpy.empty(0, dtype=dtype)

        index_dtype = (
            numpy.int32
            if int(variable_offsets[-1]) <= numpy.iinfo(numpy.int32).max
            else numpy.int64
        )
        row_offsets = numpy.empty(len(force_field) + 1, dtype=index_dtype)
        row_offsets[0] = 0
        numpy.cumsum(
            numpy.bincount(row, minlength=len(force_field)),
            out=row_offsets[1:],
        )
        return _PackedIsingBatch(
            forceField=numpy.ascontiguousarray(force_field, dtype=dtype),
            c0Rows=numpy.ascontiguousarray(c0_rows, dtype=dtype),
            rowOffsets=row_offsets,
            columns=numpy.ascontiguousarray(column, dtype=index_dtype),
            couplings=numpy.ascontiguousarray(couplings, dtype=dtype),
            variableOffsets=variable_offsets,
        )

    @staticmethod
    def _toIsingProblem(
        problem: QUBOProblem,
        dtype: Any,
        configuredC0: float,
    ) -> _IsingProblem:
        linear = problem.linear.copy()
        diagonal = problem.quadraticHeads == problem.quadraticTails
        if numpy.any(diagonal):
            numpy.add.at(
                linear,
                problem.quadraticHeads[diagonal],
                problem.quadraticBiases[diagonal],
            )
        heads = problem.quadraticHeads[~diagonal].astype(
            numpy.int64,
            copy=False,
        )
        tails = problem.quadraticTails[~diagonal].astype(
            numpy.int64,
            copy=False,
        )
        biases = problem.quadraticBiases[~diagonal]
        if len(biases):
            first = numpy.minimum(heads, tails)
            second = numpy.maximum(heads, tails)
            order = numpy.lexsort((second, first))
            first = first[order]
            second = second[order]
            biases = biases[order]
            starts = numpy.concatenate(
                (
                    numpy.array([0]),
                    numpy.flatnonzero(
                        (first[1:] != first[:-1])
                        | (second[1:] != second[:-1])
                    )
                    + 1,
                )
            )
            biases = numpy.add.reduceat(biases, starts)
            heads = first[starts]
            tails = second[starts]
            nonzero = biases != 0.0
            heads = heads[nonzero]
            tails = tails[nonzero]
            biases = biases[nonzero]

        force_field = -0.5 * linear
        couplings = -0.25 * biases
        if len(couplings):
            numpy.add.at(force_field, heads, couplings)
            numpy.add.at(force_field, tails, couplings)
        force_field = force_field.astype(dtype, copy=False)
        couplings = couplings.astype(dtype, copy=False)
        c0 = configuredC0 or TorchSBMBQMSolver._automaticC0(
            force_field,
            couplings,
        )
        return _IsingProblem(
            forceField=force_field,
            heads=heads,
            tails=tails,
            couplings=couplings,
            c0=c0,
        )

    @staticmethod
    def _automaticC0(
        forceField: numpy.ndarray,
        couplings: numpy.ndarray,
    ) -> float:
        force_norm = math.sqrt(
            2.0 * float(numpy.dot(couplings, couplings))
            + 2.0 * float(numpy.dot(forceField, forceField))
        )
        if force_norm <= 0.0:
            return 1.0
        return 0.5 * math.sqrt(max(len(forceField), 1)) / force_norm

    @staticmethod
    def _energies(
        problem: QUBOProblem,
        samples: numpy.ndarray,
        chunkSize: int,
    ) -> numpy.ndarray:
        values = numpy.asarray(samples, dtype=numpy.float64)
        energies = values @ problem.linear + problem.offset
        for start in range(0, problem.interactionCount, chunkSize):
            stop = min(start + chunkSize, problem.interactionCount)
            products = (
                values[:, problem.quadraticHeads[start:stop]]
                * values[:, problem.quadraticTails[start:stop]]
            )
            energies += products @ problem.quadraticBiases[start:stop]
        return numpy.asarray(energies, dtype=numpy.float64)

    @classmethod
    def _getParameters(
        cls,
        solverParameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        supplied = dict(solverParameters or {})
        unknown = supplied.keys() - cls._defaults.keys()
        if unknown:
            raise ValueError(
                f"Unknown Torch SBM parameters: {sorted(unknown)}"
            )
        values = cls._defaults | supplied
        normalized = {
            "steps": int(values["steps"]),
            "runs": int(values["runs"]),
            "dt": float(values["dt"]),
            "a0": float(values["a0"]),
            "c0": float(values["c0"]),
            "gamma": float(values["gamma"]),
            "initial_scale": float(values["initial_scale"]),
            "seed": int(values["seed"]),
            "dtype": str(values["dtype"]),
            "run_batch_size": (
                None
                if values["run_batch_size"] is None
                else int(values["run_batch_size"])
            ),
            "energy_chunk_size": int(values["energy_chunk_size"]),
        }
        if normalized["steps"] <= 0 or normalized["runs"] <= 0:
            raise ValueError("Torch SBM steps and runs must be positive")
        if normalized["dt"] <= 0.0 or normalized["a0"] <= 0.0:
            raise ValueError("Torch SBM dt and a0 must be positive")
        if normalized["c0"] < 0.0 or normalized["gamma"] < 0.0:
            raise ValueError("Torch SBM c0 and gamma must be nonnegative")
        if normalized["initial_scale"] < 0.0:
            raise ValueError("Torch SBM initial_scale must be nonnegative")
        if not 0 <= normalized["seed"] < _MAX_TORCH_SEED:
            raise ValueError("Torch SBM seed must be in [0, 2**63 - 1)")
        if normalized["dtype"] not in {"float32", "float64"}:
            raise ValueError("Torch SBM dtype must be float32 or float64")
        if (
            normalized["run_batch_size"] is not None
            and normalized["run_batch_size"] <= 0
        ):
            raise ValueError("Torch SBM run_batch_size must be positive")
        if normalized["energy_chunk_size"] <= 0:
            raise ValueError("Torch SBM energy_chunk_size must be positive")
        return normalized

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "TorchSBMBQMSolver requires PyTorch; install the appropriate "
                "CPU, CUDA, or ROCm wheel for this machine"
            ) from error
        return torch

    @classmethod
    def _resolveDevice(cls, requested: str) -> str:
        torch = cls._torch()
        normalized = requested.lower()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "gpu":
            normalized = "cuda"
        requested_rocm = normalized in {"rocm", "amd", "hip"}
        if requested_rocm:
            if torch.version.hip is None:
                raise RuntimeError(
                    "Torch SBM requested an AMD ROCm device, but the installed "
                    "PyTorch build has no HIP support; install a ROCm-enabled "
                    "PyTorch wheel compatible with this GPU"
                )
            normalized = "cuda"
        device = torch.device(normalized)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "Torch SBM supports CPU, CUDA, and ROCm devices"
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            backend = "ROCm" if torch.version.hip is not None else "CUDA"
            raise RuntimeError(
                f"Torch SBM requested {backend}, but no compatible GPU is "
                "available to PyTorch"
            )
        if (
            device.type == "cuda"
            and device.index is not None
            and not 0 <= device.index < torch.cuda.device_count()
        ):
            backend = "ROCm" if torch.version.hip is not None else "CUDA"
            raise ValueError(
                f"Torch SBM {backend} device is unavailable: {device}"
            )
        return str(device)


BQMSolverFactory.registerSolver("torch_sbm", TorchSBMBQMSolver)
