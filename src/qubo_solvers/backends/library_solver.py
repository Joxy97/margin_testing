"""Application adapters for the dense Torch solvers in :mod:`qubo_solvers`.

The library owns search algorithms. This boundary owns compact-QUBO conversion,
stable application seeds, source-float64 scoring and explicit one-hot repair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from numbers import Integral
from typing import Any

import numpy

from .result import BQMOptimizationResult

from .problem import QUBOProblem
from .resource_plan import BQMResourcePlan
from .torch_execution import TorchExecution


_MAX_SEED = (1 << 63) - 1
_RUN_DEFAULTS = {
    "runs": 32,
    "seed": 1,
    "dtype": "float32",
    "run_batch_size": None,
    "best_only": False,
    "memory_limit_bytes": None,
}


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


class LibraryBQMSolver(TorchExecution):
    """Run one named library solver on an explicitly selected CPU/CUDA device.

    ``steps`` aliases the algorithm's ``max_steps`` or ``sweeps``. ``runs`` and
    ``run_batch_size`` alias library ``restarts`` and ``batch_size``. Full result
    mode is the default so source scoring can rank every returned candidate;
    ``best_only`` is an explicit lower-memory option for unconstrained QUBOs.
    ``memory_limit_bytes`` bounds the conservative adapter host+tensor estimate
    before conversion, as well as the native library's own tensor estimate.
    Native histories/initial assignments remain available through the library's
    direct interface rather than this single-result application boundary.
    """

    solverId: str = ""

    supportsResident = True

    def __init__(self, device: str | None = None, devices: Sequence[str] | None = None) -> None:
        if device is not None and devices is not None:
            raise ValueError("cannot specify both device and devices")
        selected_device = "cpu" if device is None else device
        requested = devices if devices is not None else (selected_device,)
        if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
            raise TypeError("devices must be a sequence of explicit CPU/CUDA device strings")
        if not requested:
            raise ValueError("devices must not be empty")
        for value in requested:
            if not isinstance(value, str) or (value not in ("cpu", "cuda") and not (
                value.startswith("cuda:") and value[5:].isdigit()
            )):
                raise ValueError("device must be cpu, cuda or an indexed cuda device")
        if not self.solverId:
            raise TypeError("select a canonical library solver through create_bqm_solver")
        if devices is None:
            super().__init__(device=selected_device)
        else:
            super().__init__(devices=devices)

    def _getParameters(self, solverParameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        from qubo_solvers import (
            create_solver, default_algorithm_parameters, library_solver_class,
        )

        if solverParameters is not None and not isinstance(solverParameters, Mapping):
            raise TypeError("solverParameters must be a mapping")
        supplied = dict(solverParameters or {})
        for alias, canonical in (("restarts", "runs"), ("batch_size", "run_batch_size")):
            if alias in supplied:
                if canonical in supplied:
                    raise ValueError(f"cannot specify both {alias} and {canonical}")
                supplied[canonical] = supplied.pop(alias)
        algorithm_names = {item.name for item in fields(library_solver_class(self.solverId))}
        unknown = supplied.keys() - algorithm_names - _RUN_DEFAULTS.keys() - {"steps"}
        if unknown:
            raise ValueError(f"Unknown {self.solverId} parameters: {sorted(unknown)}")
        budget_name = "sweeps" if "sweeps" in algorithm_names else "max_steps"
        if "steps" in supplied and budget_name in supplied:
            raise ValueError(f"cannot specify both steps and {budget_name}")
        steps = _integer("steps", supplied.pop("steps", 1000), 0)
        algorithm = default_algorithm_parameters(self.solverId, steps=steps)
        algorithm.update({name: value for name, value in supplied.items() if name in algorithm_names})
        native = create_solver(self.solverId, **algorithm)
        parameters = {item.name: getattr(native, item.name) for item in fields(native)}
        parameters.update(_RUN_DEFAULTS)
        parameters.update({name: supplied[name] for name in _RUN_DEFAULTS if name in supplied})
        parameters["runs"] = _integer("runs", parameters["runs"], 1)
        parameters["seed"] = _integer("seed", parameters["seed"], 0)
        if parameters["seed"] >= _MAX_SEED:
            raise ValueError("seed must be in [0, 2**63 - 1)")
        if parameters["dtype"] not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(parameters["best_only"], bool):
            raise TypeError("best_only must be a boolean")
        for name in ("run_batch_size", "memory_limit_bytes"):
            if parameters[name] is not None:
                parameters[name] = _integer(name, parameters[name], 1)
        return parameters

    def estimatedWorkingMemoryBytes(
        self, problem: QUBOProblem, solverParameters: Mapping[str, Any] | None = None,
    ) -> int:
        """Conservative host+tensor estimate without allocating a dense problem.

        Includes dense conversion, native dynamics, result transfer and source
        repair. Excludes Python/allocator caches, CUDA contexts and BLAS storage.
        """
        parameters = self._getParameters(solverParameters)
        n, runs = problem.variableCount, parameters["runs"]
        batch = min(runs, parameters["run_batch_size"] or runs)
        width = 4 if parameters["dtype"] == "float32" else 8
        count = 1 if parameters["best_only"] else runs
        replicas = parameters.get("replicas", 0)
        return int(
            8 * problem.numericMemoryBytes
            + n * n * (8 + 6 * width)
            + width * ((108 + 12 * replicas) * batch * n + 288 * batch)
            + 16 * n + count * (4 * n + 2 * width + 24)
        )

    def _toLibraryProblem(self, problem: QUBOProblem, dtype: str):
        """Accumulate pair-once sparse terms before the one device transfer."""
        import torch
        from qubo_solvers import QUBO

        matrix = numpy.zeros((problem.variableCount, problem.variableCount), dtype=numpy.float64)
        numpy.fill_diagonal(matrix, problem.linear)
        heads, tails = problem.quadraticHeads, problem.quadraticTails
        # Both additions are intentional for diagonal entries. Every compact
        # term contributes once, including duplicate and reversed pairs.
        half_biases = problem.quadraticBiases * .5
        numpy.add.at(matrix, (heads, tails), half_biases)
        numpy.add.at(matrix, (tails, heads), half_biases)
        tensor = torch.from_numpy(matrix).to(
            device=self.device, dtype=torch.float32 if dtype == "float32" else torch.float64,
        )
        return QUBO(tensor, offset=problem.offset)

    def _checkMemory(self, problem, parameters):
        if parameters["best_only"] and any(True for _ in problem.iterOneHotGroups()):
            raise ValueError("best_only cannot preserve one-hot candidate selection; use False")
        limit = parameters["memory_limit_bytes"]
        if limit is not None:
            estimate = self.estimatedWorkingMemoryBytes(problem, parameters)
            if estimate > limit:
                raise MemoryError(
                    f"estimated adapter host+tensor storage {estimate} bytes exceeds "
                    f"memory_limit_bytes={limit}; reduce runs/run_batch_size or use best_only"
                )

    def _solveBatch(self, problems, parameters):
        results = []
        for problem in problems:
            self._checkMemory(problem, parameters)
            results.append(self._solveLibraryProblem(
                problem, self._toLibraryProblem(problem, parameters["dtype"]), parameters,
            ))
        return results

    def _toLibraryResident(self, resident, dtype):
        """Build the dense operator on the coefficient device, without upload."""
        import torch
        from qubo_solvers import QUBO

        n = resident.source.variableCount
        tensors = (resident.linear, resident.heads, resident.tails, resident.biases)
        device = torch.device(self.device)
        if any(value.device != device for value in tensors):
            raise ValueError("resident coefficients must already be on the solver device")
        if resident.linear.shape != (n,) or resident.biases.ndim != 1 or not (
            resident.heads.shape == resident.tails.shape == resident.biases.shape
        ):
            raise ValueError("resident coefficient arrays have invalid shapes")
        for indices in (resident.heads, resident.tails):
            if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
                raise TypeError("resident coefficient indices must use an integer dtype")
            if bool(((indices.long() < 0) | (indices.long() >= n)).any()):
                raise ValueError("resident coefficients contain an unknown variable")
        matrix = torch.zeros((n, n), dtype=torch.float64, device=device)
        matrix.diagonal().copy_(resident.linear)
        heads, tails = resident.heads.long(), resident.tails.long()
        biases = resident.biases.double() * .5
        matrix.index_put_((heads, tails), biases, accumulate=True)
        matrix.index_put_((tails, heads), biases, accumulate=True)
        return QUBO(matrix.to(dtype=getattr(torch, dtype)), offset=resident.source.offset)

    def solveResidentPlanned(self, problems, plan, solverParameters=None):
        if len(self.devices) != 1:
            raise ValueError("resident numerical execution requires one solver device")
        expected = ((0, len(problems)),) if problems else ()
        if plan.shards != expected:
            raise ValueError("resident resource plan must cover the ordered batch")
        parameters = self._getParameters(solverParameters)
        results = []
        with self._solveLock, self._torch().inference_mode():
            for resident in problems:
                self._checkMemory(resident.source, parameters)
                library_problem = self._toLibraryResident(resident, parameters["dtype"])
                results.append(self._solveLibraryProblem(resident.source, library_problem, parameters))
        return results

    def _solveLibraryProblem(self, problem, library_problem, parameters):
        from qubo_solvers import create_solver

        native = create_solver(self.solverId, **{
            name: value for name, value in parameters.items() if name not in _RUN_DEFAULTS
        })
        result = native.solve(
            library_problem,
            restarts=parameters["runs"],
            seed=(parameters["seed"] + problem.seedOffset) % _MAX_SEED,
            batch_size=parameters["run_batch_size"],
            best_only=parameters["best_only"],
            memory_limit_bytes=parameters["memory_limit_bytes"],
        )
        # Retain all returned candidates until authoritative float64 selection;
        # device-side energies are search diagnostics, not the reporting score.
        best = result.best_assignments.detach().cpu().numpy()
        final = (None if result.final_assignments is None
                 else result.final_assignments.detach().cpu().numpy())

        def candidates():
            for rows in (best, final):
                if rows is not None:
                    for row in rows:
                        yield row, 0.0

        sample, energy = self._selectBestCandidates(candidates(), problem)
        return BQMOptimizationResult(sample=sample, energy=energy)


# These configuration classes all delegate to the canonical native algorithms;
# no second implementation or independent application registry is created.
_LIBRARY_NAMES = (
    "SimulatedAnnealing", "GreedyLocalSearch", "SpinVectorLangevin",
    "AngularAnnealing", "TransverseRoute", "EasyAxisAnnealing",
    "SpinCoherentAnnealing", "VectorAmplitudeAnnealing", "MeanFieldAnnealing",
    "TAPAnnealing", "SphericalAnnealing", "ContactAnnealing", "ReplicaAnnealing",
    "HeatBathAnnealing", "TabuSearch", "RandomSearch", "ExchangeCascade",
)
_LIBRARY_IDS = (
    "lib_simulated_annealing", "lib_greedy_local_search", "lib_spin_vector_langevin",
    "lib_angular_annealing", "lib_transverse_route", "lib_easy_axis_annealing",
    "lib_spin_coherent_annealing", "lib_vector_amplitude_annealing", "lib_mean_field_annealing",
    "lib_tap_annealing", "lib_spherical_annealing", "lib_contact_annealing", "lib_replica_annealing",
    "lib_heat_bath_annealing", "lib_tabu_search", "lib_random_search", "lib_exchange_cascade",
)

for _name, _solver_id in zip(_LIBRARY_NAMES, _LIBRARY_IDS):
    _class_name = f"Library{_name}BQMSolver"
    _solver_class = type(_class_name, (LibraryBQMSolver,), {
        "solverId": _solver_id, "__module__": __name__,
        "__doc__": f"Application adapter for qubo_solvers.{_name}.",
    })
    globals()[_class_name] = _solver_class

del _name, _solver_id, _class_name, _solver_class


def create_library_bqm_solver(name, constructor_parameters=None):
    try:
        position = _LIBRARY_IDS.index(name)
    except ValueError as error:
        raise ValueError(f"Unknown native library solver: {name!r}") from error
    cls = globals()[f"Library{_LIBRARY_NAMES[position]}BQMSolver"]
    return cls(**dict(constructor_parameters or {}))
