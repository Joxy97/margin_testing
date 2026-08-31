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
        default_factory=lambda: numpy.empty(0, dtype=numpy.float32)
    )
    quadraticHeads: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint32)
    )
    quadraticTails: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint32)
    )
    quadraticBiases: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.float32)
    )
    offsets: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.float32)
    )
    problemSeeds: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint64)
    )
    samples: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint8)
    )
    energies: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.float64)
    )
    warmSamples: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint8)
    )
    warmFlags: numpy.ndarray = field(
        default_factory=lambda: numpy.empty(0, dtype=numpy.uint8)
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
        "adaptive": False,
        "min_runs": 8,
        "runs_per_batch": 8,
        "stability_batches": 2,
        "energy_tolerance": 1e-9,
        "warm_start": False,
        "topology_cache_bytes": 0,
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
        self._seriesActive = False
        self._seriesPosition = 0
        self._warmStarts: dict[int, numpy.ndarray] = {}
        self._lastWarmStart: numpy.ndarray | None = None
        self.lastRunCounts: tuple[int, ...] = ()

    def beginSeries(self) -> None:
        self._seriesActive = True
        self._seriesPosition = 0

    def endSeries(self) -> None:
        self._seriesActive = False

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
            parameters = self._getParameters(solverParameters)
            positions = tuple(
                range(
                    self._seriesPosition,
                    self._seriesPosition + len(problems),
                )
            )
            initial_samples = self._getWarmStarts(
                problems,
                positions,
                parameters["warm_start"],
            )
            results = (
                self._solveAdaptive(problems, parameters, initial_samples)
                if parameters["adaptive"]
                else self._solveFixed(problems, parameters, initial_samples)
            )
            if parameters["warm_start"]:
                self._rememberWarmStarts(positions, results)
            if self._seriesActive:
                self._seriesPosition += len(problems)
            return results

    def _solveFixed(
        self,
        problems: Sequence[QUBOProblem],
        parameters: Mapping[str, Any],
        initialSamples: Sequence[numpy.ndarray | None],
    ) -> list[BQMOptimizationResult]:
        candidates = self._nativeCandidates(
            problems,
            parameters,
            parameters["runs"],
            initialSamples,
            parameters["seed"],
        )
        for problem, problem_candidates, warm in zip(
            problems,
            candidates,
            initialSamples,
        ):
            if warm is not None:
                problem_candidates.append((warm, problem.energy(warm)))
        self.lastRunCounts = (parameters["runs"],) * len(problems)
        return [
            BQMOptimizationResult(
                *self._selectBestCandidates(problem_candidates, problem)
            )
            for problem, problem_candidates in zip(problems, candidates)
        ]

    def _solveAdaptive(
        self,
        problems: Sequence[QUBOProblem],
        parameters: Mapping[str, Any],
        initialSamples: Sequence[numpy.ndarray | None],
    ) -> list[BQMOptimizationResult]:
        states = []
        for problem, warm in zip(problems, initialSamples):
            candidate = (
                (tuple(int(value) for value in warm), problem.energy(warm))
                if warm is not None
                else None
            )
            states.append(
                {
                    "runs": 0,
                    "stable": 0,
                    "bestOverall": candidate,
                    "bestValid": (
                        candidate
                        if candidate is not None
                        and self._isValidOneHotSample(
                            candidate[0], problem.iterOneHotGroups()
                        )
                        else None
                    ),
                    "allCandidates": [] if candidate is None else [candidate],
                    "warm": warm,
                }
            )
        active = list(range(len(problems)))
        round_index = 0
        while active:
            remaining = min(
                parameters["runs"] - states[index]["runs"]
                for index in active
            )
            run_count = min(parameters["runs_per_batch"], remaining)
            active_problems = [problems[index] for index in active]
            active_warm_starts = [states[index]["warm"] for index in active]
            batches = self._nativeCandidates(
                active_problems,
                parameters,
                run_count,
                active_warm_starts,
                (
                    parameters["seed"]
                    + round_index * 0x94D049BB133111EB
                )
                & ((1 << 64) - 1),
            )
            next_active = []
            for index, candidates in zip(active, batches):
                state = states[index]
                prior_valid = state["bestValid"]
                for sample, energy in candidates:
                    candidate = (tuple(int(value) for value in sample), float(energy))
                    state["allCandidates"].append(candidate)
                    if (
                        state["bestOverall"] is None
                        or candidate[1] < state["bestOverall"][1]
                    ):
                        state["bestOverall"] = candidate
                    if self._isValidOneHotSample(
                        candidate[0],
                        problems[index].iterOneHotGroups(),
                    ) and (
                        state["bestValid"] is None
                        or candidate[1] < state["bestValid"][1]
                    ):
                        state["bestValid"] = candidate
                state["runs"] += run_count
                best = state["bestValid"] or state["bestOverall"]
                state["warm"] = numpy.asarray(best[0], dtype=numpy.uint8)
                improved = (
                    state["bestValid"] is not None
                    and (
                        prior_valid is None
                        or state["bestValid"][1]
                        < prior_valid[1] - parameters["energy_tolerance"]
                    )
                )
                state["stable"] = 0 if improved else state["stable"] + 1
                converged = (
                    state["runs"] >= parameters["min_runs"]
                    and state["bestValid"] is not None
                    and state["stable"] >= parameters["stability_batches"]
                )
                if not converged and state["runs"] < parameters["runs"]:
                    next_active.append(index)
            active = next_active
            round_index += 1
        self.lastRunCounts = tuple(int(state["runs"]) for state in states)
        return [
            BQMOptimizationResult(
                *self._selectBestCandidates(state["allCandidates"], problem)
            )
            for state, problem in zip(states, problems)
        ]

    def _nativeCandidates(
        self,
        problems: Sequence[QUBOProblem],
        parameters: Mapping[str, Any],
        runs: int,
        initialSamples: Sequence[numpy.ndarray | None],
        seed: int,
    ) -> list[list[tuple[numpy.ndarray, float]]]:
        problem_count = len(problems)
        total_variables = sum(problem.variableCount for problem in problems)
        total_interactions = sum(
            problem.interactionCount for problem in problems
        )
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
            buffers.problemSeeds[index] = (
                int(seed) + problem.seedOffset
            ) & ((1 << 64) - 1)
            warm_sample = initialSamples[index]
            buffers.warmFlags[index] = warm_sample is not None
            if warm_sample is not None:
                if len(warm_sample) != problem.variableCount:
                    raise ValueError("warm-start sample size must match problem")
                buffers.warmSamples[variable_slice] = warm_sample
        linear = buffers.linear[:total_variables]
        quadratic_heads = buffers.quadraticHeads[:total_interactions]
        quadratic_tails = buffers.quadraticTails[:total_interactions]
        quadratic_biases = buffers.quadraticBiases[:total_interactions]
        offsets = buffers.offsets[:problem_count]
        samples = buffers.samples[: total_variables * runs]
        energies = buffers.energies[: problem_count * runs]
        error = ctypes.create_string_buffer(1024)
        library = self._loadLibrary()

        status = library.sbm_solve_qubo_cpu_candidates_seeded_batch(
            problem_count,
            self._pointer(variable_offsets, ctypes.c_size_t),
            self._pointer(linear, ctypes.c_float),
            self._pointer(quadratic_offsets, ctypes.c_size_t),
            self._pointer(quadratic_heads, ctypes.c_uint32),
            self._pointer(quadratic_tails, ctypes.c_uint32),
            self._pointer(quadratic_biases, ctypes.c_float),
            self._pointer(offsets, ctypes.c_float),
            parameters["steps"],
            runs,
            parameters["dt"],
            parameters["a0"],
            parameters["c0"],
            parameters["gamma"],
            parameters["initial_scale"],
            seed,
            self._pointer(
                buffers.problemSeeds[:problem_count],
                ctypes.c_uint64,
            ),
            self._pointer(buffers.warmSamples[:total_variables], ctypes.c_uint8),
            self._pointer(buffers.warmFlags[:problem_count], ctypes.c_uint8),
            parameters["topology_cache_bytes"],
            self._pointer(samples, ctypes.c_uint8),
            self._pointer(energies, ctypes.c_double),
            error,
            len(error),
        )
        if status != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"SBM C++ solver failed: {message}")
        return self._extractCandidates(problems, samples, energies, runs)

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
            ("linear", variableCount, numpy.float32),
            ("quadraticHeads", interactionCount, numpy.uint32),
            ("quadraticTails", interactionCount, numpy.uint32),
            ("quadraticBiases", interactionCount, numpy.float32),
            ("offsets", offsetCount - 1, numpy.float32),
            ("problemSeeds", offsetCount - 1, numpy.uint64),
            ("samples", sampleCount, numpy.uint8),
            ("energies", energyCount, numpy.float64),
            ("warmSamples", variableCount, numpy.uint8),
            ("warmFlags", offsetCount - 1, numpy.uint8),
        )
        for name, required, dtype in requirements:
            existing = getattr(self._buffers, name)
            if len(existing) < required:
                capacity = max(required, max(1, len(existing) * 2))
                setattr(self._buffers, name, numpy.empty(capacity, dtype=dtype))

    def _extractCandidates(
        self,
        problems: Sequence[QUBOProblem],
        samples: numpy.ndarray,
        _nativeEnergies: numpy.ndarray,
        runs: int,
    ) -> list[list[tuple[numpy.ndarray, float]]]:
        results: list[list[tuple[numpy.ndarray, float]]] = []
        sample_cursor = 0
        for problem in problems:
            next_problem_cursor = (
                sample_cursor + runs * problem.variableCount
            )
            problem_samples = samples[
                sample_cursor:next_problem_cursor
            ].reshape(runs, problem.variableCount)
            energies = self._energies(problem, problem_samples)
            candidates = []
            for run in range(runs):
                candidates.append((problem_samples[run], energies[run]))
            results.append(candidates)
            sample_cursor = next_problem_cursor
        return results

    @staticmethod
    def _energies(
        problem: QUBOProblem,
        samples: numpy.ndarray,
        chunkSize: int = 1_000_000,
    ) -> numpy.ndarray:
        """Evaluate candidate energies in float64 against the source QUBO."""
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

    def _getWarmStarts(
        self,
        problems: Sequence[QUBOProblem],
        positions: Sequence[int],
        enabled: bool,
    ) -> list[numpy.ndarray | None]:
        if not enabled:
            return [None] * len(problems)
        starts = []
        for problem, position in zip(problems, positions):
            warm = self._warmStarts.get(position)
            if warm is None and self._lastWarmStart is not None:
                warm = self._lastWarmStart
            if warm is None or len(warm) != problem.variableCount:
                warm = self._feasibleStart(problem)
            starts.append(warm)
        return starts

    @staticmethod
    def _feasibleStart(problem: QUBOProblem) -> numpy.ndarray:
        """Build a deterministic valid one-hot seed from linear biases."""
        sample = numpy.zeros(problem.variableCount, dtype=numpy.uint8)
        grouped = numpy.zeros(problem.variableCount, dtype=bool)
        for group in problem.iterOneHotGroups():
            variables = numpy.asarray(group, dtype=numpy.int64)
            selected = int(variables[numpy.argmin(problem.linear[variables])])
            sample[selected] = 1
            grouped[variables] = True
        sample[~grouped] = problem.linear[~grouped] < 0.0
        return sample

    def _rememberWarmStarts(
        self,
        positions: Sequence[int],
        results: Sequence[BQMOptimizationResult],
    ) -> None:
        for position, result in zip(positions, results):
            sample = numpy.asarray(result.sample, dtype=numpy.uint8)
            self._warmStarts[position] = sample
            self._lastWarmStart = sample

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
            "adaptive": bool(parameters["adaptive"]),
            "min_runs": int(parameters["min_runs"]),
            "runs_per_batch": int(parameters["runs_per_batch"]),
            "stability_batches": int(parameters["stability_batches"]),
            "energy_tolerance": float(parameters["energy_tolerance"]),
            "warm_start": bool(parameters["warm_start"]),
            "topology_cache_bytes": int(parameters["topology_cache_bytes"]),
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
        if normalized["min_runs"] <= 0 or (
            normalized["adaptive"]
            and normalized["min_runs"] > normalized["runs"]
        ):
            raise ValueError("SBM min_runs must be between one and runs")
        if normalized["runs_per_batch"] <= 0:
            raise ValueError("SBM runs_per_batch must be positive")
        if normalized["stability_batches"] <= 0:
            raise ValueError("SBM stability_batches must be positive")
        if normalized["energy_tolerance"] < 0.0:
            raise ValueError("SBM energy_tolerance must be nonnegative")
        if normalized["topology_cache_bytes"] < 0:
            raise ValueError("SBM topology_cache_bytes must be nonnegative")
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
        try:
            solve_batch = library.sbm_solve_qubo_cpu_candidates_seeded_batch
        except AttributeError as error:
            raise RuntimeError(
                "SBM native library uses an older ABI; rebuild sbm_python"
            ) from error
        solve_batch.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        solve_batch.restype = ctypes.c_int
        return library

    @staticmethod
    def _defaultLibraryPath() -> Path:
        repository = Path(__file__).resolve().parents[5]
        return repository / "build" / "libsbm_python.so"


BQMSolverFactory.registerSolver("sbm", SBMBQMSolver)
