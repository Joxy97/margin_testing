"""Shared Torch device lifetime, preparation, trajectory batching and collection."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from .torch_candidates import TorchCandidateAccumulator
from .resource_plan import BQMResourcePlan


_RUN_SEED_STRIDE = 0x9E3779B97F4A7C15
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


class TorchExecution(BQMSolver):
    """Solve scenario and trajectory batches through Torch sparse SpMM.

    AMD ROCm builds of PyTorch intentionally expose GPUs through the
    ``torch.cuda`` API. Constructor aliases ``rocm``, ``amd``, and ``hip``
    therefore resolve to the same internal ``cuda`` device as NVIDIA GPUs.
    """


    def __init__(
        self,
        device: str = "auto",
        devices: Sequence[str] | None = None,
    ) -> None:
        if devices is not None:
            if device != "auto":
                raise ValueError("Torch SBM cannot define both device and devices")
            if isinstance(devices, (str, bytes)) or not isinstance(
                devices, Sequence
            ):
                raise TypeError("Torch SBM devices must be a sequence")
            if not devices:
                raise ValueError("Torch SBM devices must not be empty")
            requested_devices = tuple(str(item) for item in devices)
        else:
            requested_devices = (str(device),)
        if any(not item.strip() for item in requested_devices):
            raise ValueError("Torch SBM devices must not contain empty names")
        self.requestedDevices = requested_devices
        self.requestedDevice = requested_devices[0]
        self._resolvedDevices: tuple[str, ...] | None = None
        self._workerSolvers: tuple[TorchExecution, ...] | None = None
        self._solveLock = Lock()

    @property
    def device(self) -> str:
        """Return the first resolved Torch device for compatibility."""
        return self.devices[0]

    @property
    def batchParallelism(self) -> int:
        """Return the number of configured device workers."""
        return len(self.requestedDevices)

    @property
    def devices(self) -> tuple[str, ...]:
        """Return all resolved Torch devices, importing Torch lazily."""
        if self._resolvedDevices is None:
            resolved = tuple(
                self._resolveDevice(requested)
                for requested in self.requestedDevices
            )
            if len(resolved) > 1:
                if any(
                    not value.startswith("cuda:")
                    or not value.removeprefix("cuda:").isdigit()
                    for value in resolved
                ):
                    raise ValueError(
                        "Torch SBM multi-device execution requires explicit "
                        "indexed CUDA/ROCm devices such as cuda:0"
                    )
                if len(set(resolved)) != len(resolved):
                    raise ValueError("Torch SBM devices must be unique")
            self._resolvedDevices = resolved
        return self._resolvedDevices

    @property
    def acceleratorBackend(self) -> str:
        """Return ``cpu``, ``cuda``, or ``rocm`` for the resolved device."""
        if self.device == "cpu":
            return "cpu"
        return "rocm" if self._torch().version.hip is not None else "cuda"

    def estimatedWorkingMemoryBytes(
        self,
        problem: QUBOProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> int:
        parameters = self._getParameters(solverParameters)
        width = min(parameters["run_batch_size"] or parameters["runs"], parameters["runs"])
        item_size = 4 if parameters["dtype"] == "float32" else 8
        # Include CSR packing, original coefficients, integration temporaries,
        # host candidates, and float64 chunked scoring/repair workspace.
        buffers = 16 + parameters.get("noise_chunk_size", 0)
        return int(
            8 * problem.numericMemoryBytes
            + problem.variableCount * width * item_size * buffers
            + problem.variableCount * parameters["runs"] * 16
            + min(problem.interactionCount, parameters["energy_chunk_size"]) * width * 32
        )

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
        """Solve problems in block-diagonal batches on one or more devices."""
        if any(not isinstance(problem, QUBOProblem) for problem in problems):
            raise TypeError("Torch problems must be QUBOProblem objects")
        plan = BQMResourcePlan.create(
            [self.estimatedWorkingMemoryBytes(problem, solverParameters) for problem in problems],
            self.batchParallelism,
            sum(problem.numericMemoryBytes for problem in problems),
        )
        return self.solvePlanned(problems, plan, solverParameters)

    def solvePlanned(self, problems: Sequence[QUBOProblem], plan: BQMResourcePlan,
                     solverParameters: Mapping[str, Any] | None = None) -> list[BQMOptimizationResult]:
        if not problems:
            return []
        if any(not isinstance(problem, QUBOProblem) for problem in problems):
            raise TypeError("Torch problems must be QUBOProblem objects")
        parameters = self._getParameters(solverParameters)
        with self._solveLock:
            if len(self.devices) > 1 and len(problems) > 1:
                return self._solveMultiDevice(problems, parameters, plan)
            return self._solveBatch(problems, parameters)

    def _solveMultiDevice(
        self,
        problems: Sequence[QUBOProblem],
        parameters: Mapping[str, Any],
        plan: BQMResourcePlan,
    ) -> list[BQMOptimizationResult]:
        """Split one ordered batch across independent device workers."""
        worker_count = len(plan.shards)
        workers = self._getWorkerSolvers()[:worker_count]

        def solve_shard(
            worker: TorchExecution,
            start: int,
            stop: int,
        ) -> tuple[int, list[BQMOptimizationResult]]:
            return start, worker.solveMany(
                problems[start:stop],
                parameters,
            )

        ordered: list[BQMOptimizationResult | None] = [None] * len(problems)
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="torch-sbm-device",
        ) as executor:
            futures = [
                executor.submit(solve_shard, worker, start, stop)
                for worker, (start, stop) in zip(workers, plan.shards)
            ]
            for future in futures:
                start, results = future.result()
                ordered[start : start + len(results)] = results
        if any(result is None for result in ordered):
            raise RuntimeError(
                "Torch SBM device worker returned an incomplete batch"
            )
        self._mergeWorkerState(workers)
        return [result for result in ordered if result is not None]

    def _getWorkerSolvers(self) -> tuple[TorchExecution, ...]:
        if self._workerSolvers is None:
            self._workerSolvers = tuple(
                type(self)(device=device) for device in self.devices
            )
        return self._workerSolvers

    def _mergeWorkerState(
        self,
        workers: Sequence[TorchExecution],
    ) -> None:
        """Merge diagnostic state after a multi-device solve, when needed."""

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
            parameters.get("c0", 1.0),
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

        candidates = [TorchCandidateAccumulator(torch, problem, device,
                      parameters["energy_chunk_size"], self._selectBestCandidates)
                      for problem in problems]
        return self._solvePrepared(problems, parameters, matrix, field, c0_rows,
                                   packed.variableOffsets, candidates)

    def _solvePrepared(self, problems, parameters, matrix, field, c0_rows, variableOffsets, candidates):
        torch = self._torch()
        torch_dtype, device = field.dtype, field.device
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
                    variableOffsets,
                    numpy.asarray(
                        [problem.seedOffset for problem in problems],
                        dtype=numpy.uint64,
                    ),
                    width,
                    run_start,
                    parameters,
                    torch_dtype,
                    device,
                )
                for problem_index, problem in enumerate(problems):
                    start = int(variableOffsets[problem_index])
                    stop = int(variableOffsets[problem_index + 1])
                    problem_samples = samples[start:stop].T
                    candidates[problem_index].add(problem_samples)

        return [
            BQMOptimizationResult(*accumulator.result()) for accumulator in candidates
        ]


    def solveResidentPlanned(self, problems, plan, solverParameters=None):
        """Consume Torch-owned coefficients without a host coefficient upload.

        Sources provide authoritative scoring and stable identity. Tensor lifetime
        belongs to the numerical execution module that constructed the batch.
        """
        if not problems:
            return []
        if len(self.devices) > 1:
            raise ValueError("Resident numerical execution currently requires one solver device")
        if plan.shards != ((0, len(problems)),):
            raise ValueError("Resident resource plan must cover the ordered batch")
        parameters = self._getParameters(solverParameters)
        torch = self._torch()
        device = torch.device(self.device)
        with self._solveLock, torch.inference_mode(), torch.profiler.record_function("margin.resident_solve"):
            resident = [problem.toDevice(device) for problem in problems]
            dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64
            fields, scales, all_heads, all_tails, all_couplings = [], [], [], [], []
            variable_offsets = numpy.cumsum([0] + [problem.source.variableCount for problem in resident])
            for index, problem in enumerate(resident):
                # Resident encoders emit canonical, distinct off-diagonal edges.
                coupling = -.25 * problem.biases
                field = -.5 * problem.linear
                field = field.scatter_add(0, problem.heads, coupling)
                field = field.scatter_add(0, problem.tails, coupling)
                field, coupling = field.to(dtype), coupling.to(dtype)
                configured_c0 = parameters.get("c0", 1.0)
                if configured_c0:
                    c0 = torch.full((), configured_c0, device=device, dtype=dtype)
                else:
                    norm = (2 * (coupling.square().sum().double() + field.square().sum().double())).sqrt()
                    c0 = torch.where(norm > 0, .5 * math.sqrt(len(field)) / norm, 1.).to(dtype)
                fields.append(field)
                scales.append(c0.expand(len(field)))
                all_heads.append(problem.heads + int(variable_offsets[index]))
                all_tails.append(problem.tails + int(variable_offsets[index]))
                all_couplings.append(coupling)
            heads, tails, couplings = torch.cat(all_heads), torch.cat(all_tails), torch.cat(all_couplings)
            count = int(variable_offsets[-1])
            indices = torch.stack((torch.cat((heads, tails)), torch.cat((tails, heads))))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
                matrix = torch.sparse_coo_tensor(indices, torch.cat((couplings, couplings)),
                    (count, count), device=device, dtype=dtype, check_invariants=False).coalesce().to_sparse_csr()
            candidates = [TorchCandidateAccumulator(torch, problem.source, device,
                parameters["energy_chunk_size"], self._selectBestCandidates, coefficients=problem)
                for problem in resident]
            return self._solvePrepared([problem.source for problem in resident], parameters,
                matrix, torch.cat(fields).reshape(-1, 1), torch.cat(scales).reshape(-1, 1),
                variable_offsets, candidates)

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
        c0 = configuredC0 or TorchExecution._automaticC0(
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


    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "TorchExecution requires PyTorch; install the appropriate "
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



