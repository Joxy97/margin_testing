"""Native batched adapter for the project's C++ SBM kernel."""

from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy

from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)

from ...optimization_problem.qubo_problem import QUBOProblem
from .bqm_solver import BQMSolver
from .bqm_solver_factory import BQMSolverFactory


@dataclass
class _NativeBatchBuffers:
    variableOffsets: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uintp)
    )
    quadraticOffsets: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uintp)
    )
    linear: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.float64)
    )
    quadraticHeads: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint32)
    )
    quadraticTails: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint32)
    )
    quadraticBiases: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.float64)
    )
    offsets: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.float64)
    )
    samples: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint8)
    )
    energies: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.float64)
    )


class SBMBQMSolver(BQMSolver):
    """Solve numeric QUBOs through one allocation-efficient C++ batch call."""

    _defaults = {
        "steps": 10_000,
        "runs": 16,
        "dt": 1.0,
        "a0": 1.0,
        "c0": 0.0,
        "gamma": 0.0,
        "initial_scale": 0.05,
        "seed": 1,
    }

    def __init__(self, libraryPath: str | Path | None = None) -> None:
        self.libraryPath = (
            Path(libraryPath)
            if libraryPath is not None
            else self._defaultLibraryPath()
        )
        self._library: ctypes.CDLL | None = None
        self._libraryLock = Lock()
        self._solveLock = Lock()
        self._buffers = _NativeBatchBuffers()

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
        """Solve all supplied QUBOs concurrently inside the C++ kernel."""
        if not problems:
            return []
        with self._solveLock:
            return self._solveMany(problems, solverParameters)

    def _solveMany(
        self,
        problems: Sequence[QUBOProblem],
        solverParameters: Mapping[str, Any] | None,
    ) -> list[BQMOptimizationResult]:
        parameters = self._getParameters(solverParameters)
        problem_count = len(problems)
        total_variables = sum(problem.variableCount for problem in problems)
        total_interactions = sum(
            problem.interactionCount for problem in problems
        )
        runs = parameters["runs"]
        self._ensureBufferCapacity(
            problem_count + 1,
            total_variables,
            total_interactions,
            total_variables * runs,
            problem_count * runs,
        )
        buffers = self._buffers
        variable_offsets = buffers.variableOffsets[: problem_count + 1]
        quadratic_offsets = buffers.quadraticOffsets[: problem_count + 1]
        variable_offsets[0] = 0
        quadratic_offsets[0] = 0
        for index, problem in enumerate(problems):
            variable_offsets[index + 1] = (
                variable_offsets[index] + problem.variableCount
            )
            quadratic_offsets[index + 1] = (
                quadratic_offsets[index] + problem.interactionCount
            )
            variable_slice = slice(
                int(variable_offsets[index]),
                int(variable_offsets[index + 1]),
            )
            quadratic_slice = slice(
                int(quadratic_offsets[index]),
                int(quadratic_offsets[index + 1]),
            )
            buffers.linear[variable_slice] = problem.linear
            buffers.quadraticHeads[quadratic_slice] = problem.quadraticHeads
            buffers.quadraticTails[quadratic_slice] = problem.quadraticTails
            buffers.quadraticBiases[quadratic_slice] = problem.quadraticBiases
            buffers.offsets[index] = problem.offset
        linear = buffers.linear[:total_variables]
        quadratic_heads = buffers.quadraticHeads[:total_interactions]
        quadratic_tails = buffers.quadraticTails[:total_interactions]
        quadratic_biases = buffers.quadraticBiases[:total_interactions]
        offsets = buffers.offsets[:problem_count]
        samples = buffers.samples[: total_variables * runs]
        energies = buffers.energies[: problem_count * runs]
        error = ctypes.create_string_buffer(1024)
        library = self._loadLibrary()

        status = library.sbm_solve_qubo_cpu_candidates_batch(
            problem_count,
            self._pointer(variable_offsets, ctypes.c_size_t),
            self._pointer(linear, ctypes.c_double),
            self._pointer(quadratic_offsets, ctypes.c_size_t),
            self._pointer(quadratic_heads, ctypes.c_uint32),
            self._pointer(quadratic_tails, ctypes.c_uint32),
            self._pointer(quadratic_biases, ctypes.c_double),
            self._pointer(offsets, ctypes.c_double),
            parameters["steps"],
            runs,
            parameters["dt"],
            parameters["a0"],
            parameters["c0"],
            parameters["gamma"],
            parameters["initial_scale"],
            parameters["seed"],
            self._pointer(samples, ctypes.c_uint8),
            self._pointer(energies, ctypes.c_double),
            error,
            len(error),
        )
        if status != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"SBM C++ solver failed: {message}")
        return self._decodeResults(problems, samples, energies, runs)

    def _ensureBufferCapacity(
        self,
        offsetCount: int,
        variableCount: int,
        interactionCount: int,
        sampleCount: int,
        energyCount: int,
    ) -> None:
        requirements = (
            ("variableOffsets", offsetCount, numpy.uintp),
            ("quadraticOffsets", offsetCount, numpy.uintp),
            ("linear", variableCount, numpy.float64),
            ("quadraticHeads", interactionCount, numpy.uint32),
            ("quadraticTails", interactionCount, numpy.uint32),
            ("quadraticBiases", interactionCount, numpy.float64),
            ("offsets", offsetCount - 1, numpy.float64),
            ("samples", sampleCount, numpy.uint8),
            ("energies", energyCount, numpy.float64),
        )
        for name, required, dtype in requirements:
            existing = getattr(self._buffers, name)
            if len(existing) < required:
                capacity = max(required, max(1, len(existing) * 2))
                setattr(self._buffers, name, numpy.empty(capacity, dtype=dtype))

    def _decodeResults(
        self,
        problems: Sequence[QUBOProblem],
        samples: numpy.ndarray,
        energies: numpy.ndarray,
        runs: int,
    ) -> list[BQMOptimizationResult]:
        results = []
        sample_cursor = 0
        for problem_index, problem in enumerate(problems):
            candidates = []
            for run in range(runs):
                next_cursor = sample_cursor + problem.variableCount
                candidates.append(
                    (
                        samples[sample_cursor:next_cursor],
                        energies[problem_index * runs + run],
                    )
                )
                sample_cursor = next_cursor
            sample, energy = self._selectBestCandidates(
                candidates,
                problem.oneHotGroups,
            )
            results.append(BQMOptimizationResult(sample, energy))
        return results

    @staticmethod
    def _pointer(array: numpy.ndarray, valueType: Any) -> Any:
        return array.ctypes.data_as(ctypes.POINTER(valueType))

    @classmethod
    def _getParameters(
        cls,
        solverParameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        supplied = dict(solverParameters or {})
        unknown = supplied.keys() - cls._defaults.keys()
        if unknown:
            raise ValueError(f"Unknown SBM solver parameters: {sorted(unknown)}")
        parameters = cls._defaults | supplied
        normalized = {
            "steps": int(parameters["steps"]),
            "runs": int(parameters["runs"]),
            "dt": float(parameters["dt"]),
            "a0": float(parameters["a0"]),
            "c0": float(parameters["c0"]),
            "gamma": float(parameters["gamma"]),
            "initial_scale": float(parameters["initial_scale"]),
            "seed": int(parameters["seed"]),
        }
        if normalized["steps"] <= 0 or normalized["runs"] <= 0:
            raise ValueError("SBM steps and runs must be positive")
        if normalized["dt"] <= 0.0 or normalized["a0"] <= 0.0:
            raise ValueError("SBM dt and a0 must be positive")
        if normalized["gamma"] < 0.0:
            raise ValueError("SBM gamma must be nonnegative")
        if normalized["initial_scale"] < 0.0:
            raise ValueError("SBM initial_scale must be nonnegative")
        if normalized["seed"] < 0:
            raise ValueError("SBM seed must be nonnegative")
        return normalized

    def _loadLibrary(self) -> ctypes.CDLL:
        if self._library is not None:
            return self._library
        with self._libraryLock:
            if self._library is None:
                self._library = self._createLibrary()
            return self._library

    def _createLibrary(self) -> ctypes.CDLL:
        if not self.libraryPath.is_file():
            raise FileNotFoundError(
                f"SBM C++ library not found at {self.libraryPath}; "
                "build the sbm_python CMake target first"
            )
        library = ctypes.CDLL(str(self.libraryPath))
        library.sbm_solve_qubo_cpu_candidates_batch.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.sbm_solve_qubo_cpu_candidates_batch.restype = ctypes.c_int
        return library

    @staticmethod
    def _defaultLibraryPath() -> Path:
        repository = Path(__file__).resolve().parents[5]
        return repository / "build" / "libsbm_python.so"


BQMSolverFactory.registerSolver("sbm", SBMBQMSolver)
