"""Adaptive multi-agent simulated bifurcation implemented with PyTorch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy
from scipy.sparse import coo_matrix

from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)

from ...optimization_problem.qubo_problem import QUBOProblem
from .bqm_solver_factory import BQMSolverFactory
from .torch_sbm_bqm_solver import (
    _MAX_TORCH_SEED,
    _PROBLEM_SEED_STRIDE,
    _RUN_SEED_STRIDE,
    TorchSBMBQMSolver,
)


class AdaptiveTorchSBMBQMSolver(TorchSBMBQMSolver):
    """Torch SBM with variant selection, convergence checks, and polishing.

    The numerical core supports ballistic and discrete SB, optional heating,
    a pressure-slope schedule, periodic best-state retention, and energy-based
    early stopping. A final coordinate descent preserves declared one-hot
    groups while improving the exact source-QUBO energy.
    """

    _adaptiveDefaults = {
        "mode": "discrete",
        "heated": False,
        "heat_coefficient": 0.06,
        "pressure_slope": 0.01,
        "early_stopping": True,
        "sampling_period": 30,
        "convergence_threshold": 5,
        "convergence_tolerance": 1.0e-7,
        "track_best": True,
        "local_search_sweeps": 1,
        "local_search_tolerance": 1.0e-12,
    }
    _defaults = TorchSBMBQMSolver._defaults | _adaptiveDefaults

    def __init__(
        self,
        device: str = "auto",
        devices: Sequence[str] | None = None,
    ) -> None:
        super().__init__(device, devices)
        self.lastStepCount = 0
        self._stepCounts: list[int] = []

    def _mergeWorkerState(
        self,
        workers: Sequence[TorchSBMBQMSolver],
    ) -> None:
        adaptive_workers = [
            worker
            for worker in workers
            if isinstance(worker, AdaptiveTorchSBMBQMSolver)
        ]
        self._stepCounts = [
            count
            for worker in adaptive_workers
            for count in worker._stepCounts
        ]
        self.lastStepCount = max(
            (worker.lastStepCount for worker in adaptive_workers),
            default=0,
        )

    def _solveBatch(
        self,
        problems: Sequence[QUBOProblem],
        parameters: Mapping[str, Any],
    ) -> list[BQMOptimizationResult]:
        self._stepCounts = []
        results = super()._solveBatch(problems, parameters)
        self.lastStepCount = max(self._stepCounts, default=0)
        sweeps = parameters["local_search_sweeps"]
        if not sweeps:
            return results
        return [
            self._polish(
                problem,
                result,
                sweeps,
                parameters["local_search_tolerance"],
            )
            for problem, result in zip(problems, results)
        ]

    def _runTrajectories(
        self,
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

        problem_count = len(variableOffsets) - 1
        variable_counts = torch.as_tensor(
            numpy.diff(variableOffsets),
            dtype=torch.int64,
            device=device,
        )
        problem_indices = torch.repeat_interleave(
            torch.arange(problem_count, device=device),
            variable_counts,
        )
        activation = torch.empty_like(positions)
        coupling_force = torch.empty_like(positions)
        sampled_spins = torch.empty_like(positions)
        sampled_force = torch.empty_like(positions)
        absolute_positions = torch.empty_like(positions)
        wall = torch.empty(shape, dtype=torch.bool, device=device)
        previous_momenta = torch.empty_like(momenta)
        best_energies = torch.full(
            (problem_count, width),
            float("inf"),
            dtype=torchDtype,
            device=device,
        )
        best_activations = torch.empty_like(positions)
        previous_energies = None
        stability = torch.zeros(
            (problem_count, width),
            dtype=torch.int32,
            device=device,
        )
        heat_coefficient = (
            parameters["heat_coefficient"]
            if parameters["heated"]
            else parameters["gamma"]
        )

        completed_steps = 0
        for step in range(parameters["steps"]):
            self._activate(torch, positions, activation, parameters["mode"])
            coupling_force.copy_(torch.sparse.mm(matrix, activation))

            if step % parameters["sampling_period"] == 0:
                self._spinSample(torch, positions, sampled_spins)
                energy_force = coupling_force
                if parameters["mode"] == "ballistic":
                    sampled_force.copy_(
                        torch.sparse.mm(matrix, sampled_spins)
                    )
                    energy_force = sampled_force
                current_energies = self._isingEnergies(
                    torch,
                    sampled_spins,
                    energy_force,
                    field,
                    problem_indices,
                    problem_count,
                )
                if parameters["track_best"]:
                    improved = current_energies < best_energies
                    best_activations.copy_(
                        torch.where(
                            improved.index_select(0, problem_indices),
                            sampled_spins,
                            best_activations,
                        )
                    )
                    torch.minimum(
                        best_energies,
                        current_energies,
                        out=best_energies,
                    )
                if previous_energies is not None:
                    stable = torch.abs(
                        current_energies - previous_energies
                    ) <= parameters["convergence_tolerance"]
                    stability.copy_(
                        torch.where(stable, stability + 1, 0)
                    )
                    if (
                        parameters["early_stopping"]
                        and torch.all(
                            stability >= parameters["convergence_threshold"]
                        ).item()
                    ):
                        break
                previous_energies = current_energies

            if heat_coefficient:
                previous_momenta.copy_(momenta)
            pressure = min(
                parameters["dt"] * step * parameters["pressure_slope"],
                1.0,
            ) * parameters["a0"]
            coupling_force.add_(field).mul_(c0Rows)
            coupling_force.add_(
                positions,
                alpha=pressure - parameters["a0"],
            )
            momenta.add_(coupling_force, alpha=parameters["dt"])
            positions.add_(
                momenta,
                alpha=parameters["a0"] * parameters["dt"],
            )
            torch.abs(positions, out=absolute_positions)
            torch.gt(absolute_positions, 1.0, out=wall)
            positions.clamp_(-1.0, 1.0)
            momenta.masked_fill_(wall, 0.0)
            if heat_coefficient:
                momenta.add_(
                    previous_momenta,
                    alpha=parameters["dt"] * heat_coefficient,
                )
            completed_steps = step + 1

        self._activate(torch, positions, activation, parameters["mode"])
        if parameters["track_best"]:
            self._spinSample(torch, positions, sampled_spins)
            sampled_force.copy_(torch.sparse.mm(matrix, sampled_spins))
            current_energies = self._isingEnergies(
                torch,
                sampled_spins,
                sampled_force,
                field,
                problem_indices,
                problem_count,
            )
            improved = current_energies < best_energies
            best_activations.copy_(
                torch.where(
                    improved.index_select(0, problem_indices),
                    sampled_spins,
                    best_activations,
                )
            )
            activation = best_activations
        self._stepCounts.append(completed_steps)
        return ((torch.sign(activation) + 1.0) * 0.5).to(
            torch.uint8
        ).cpu().numpy()

    @staticmethod
    def _activate(
        torch: Any,
        positions: Any,
        output: Any,
        mode: str,
    ) -> None:
        if mode == "ballistic":
            output.copy_(positions)
        else:
            torch.sign(positions, out=output)
            output.masked_fill_(output == 0, 1.0)

    @staticmethod
    def _spinSample(torch: Any, positions: Any, output: Any) -> None:
        torch.sign(positions, out=output)
        output.masked_fill_(output == 0, 1.0)

    @staticmethod
    def _isingEnergies(
        torch: Any,
        activation: Any,
        couplingForce: Any,
        field: Any,
        problemIndices: Any,
        problemCount: int,
    ) -> Any:
        row_energies = -activation * field - 0.5 * activation * couplingForce
        energies = torch.zeros(
            (problemCount, activation.shape[1]),
            dtype=activation.dtype,
            device=activation.device,
        )
        energies.index_add_(0, problemIndices, row_energies)
        return energies

    @staticmethod
    def _polish(
        problem: QUBOProblem,
        result: BQMOptimizationResult,
        sweeps: int,
        tolerance: float,
    ) -> BQMOptimizationResult:
        sample = numpy.asarray(result.sample, dtype=numpy.uint8).copy()
        diagonal = problem.quadraticHeads == problem.quadraticTails
        linear = problem.linear.copy()
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
        adjacency = coo_matrix(
            (
                numpy.concatenate((biases, biases)),
                (
                    numpy.concatenate((heads, tails)),
                    numpy.concatenate((tails, heads)),
                ),
            ),
            shape=(problem.variableCount, problem.variableCount),
        ).tocsc()
        grouped = numpy.zeros(problem.variableCount, dtype=bool)
        original_valid = True
        for group in problem.oneHotGroups:
            variables = numpy.asarray(group, dtype=numpy.int64)
            grouped[variables] = True
            if int(sample[variables].sum()) != 1:
                original_valid = False
                best_sample = None
                best_energy = float("inf")
                for candidate in variables:
                    trial = sample.copy()
                    trial[variables] = 0
                    trial[int(candidate)] = 1
                    energy = problem.energy(trial)
                    if energy < best_energy:
                        best_sample = trial
                        best_energy = energy
                sample = best_sample
        local_fields = linear + adjacency @ sample

        for _ in range(sweeps):
            improved = False
            for group in problem.oneHotGroups:
                variables = numpy.asarray(group, dtype=numpy.int64)
                selected_values = variables[sample[variables] == 1]
                if len(selected_values) != 1:
                    continue
                selected = int(selected_values[0])
                best_variable = selected
                best_delta = 0.0
                for candidate in variables:
                    candidate = int(candidate)
                    if candidate == selected:
                        continue
                    delta = (
                        -local_fields[selected]
                        + local_fields[candidate]
                        - float(adjacency[selected, candidate])
                    )
                    if delta < best_delta - tolerance:
                        best_delta = float(delta)
                        best_variable = candidate
                if best_variable != selected:
                    sample[selected] = 0
                    sample[best_variable] = 1
                    local_fields -= adjacency.getcol(selected).toarray().ravel()
                    local_fields += adjacency.getcol(best_variable).toarray().ravel()
                    improved = True

            for variable in numpy.flatnonzero(~grouped):
                direction = 1 - 2 * int(sample[variable])
                delta = direction * local_fields[variable]
                if delta < -tolerance:
                    sample[variable] = 1 - sample[variable]
                    local_fields += (
                        direction
                        * adjacency.getcol(variable).toarray().ravel()
                    )
                    improved = True
            if not improved:
                break

        polished_energy = problem.energy(sample)
        if original_valid and polished_energy > result.energy + tolerance:
            return result
        return BQMOptimizationResult(
            tuple(int(value) for value in sample),
            polished_energy,
        )

    @classmethod
    def _getParameters(
        cls,
        solverParameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        supplied = dict(solverParameters or {})
        unknown = supplied.keys() - cls._defaults.keys()
        if unknown:
            raise ValueError(
                f"Unknown adaptive Torch SBM parameters: {sorted(unknown)}"
            )
        base_keys = TorchSBMBQMSolver._defaults.keys()
        parameters = TorchSBMBQMSolver._getParameters(
            {key: value for key, value in supplied.items() if key in base_keys}
        )
        values = cls._defaults | supplied
        parameters.update(
            {
                "mode": str(values["mode"]),
                "heated": bool(values["heated"]),
                "heat_coefficient": float(values["heat_coefficient"]),
                "pressure_slope": float(values["pressure_slope"]),
                "early_stopping": bool(values["early_stopping"]),
                "sampling_period": int(values["sampling_period"]),
                "convergence_threshold": int(
                    values["convergence_threshold"]
                ),
                "convergence_tolerance": float(values["convergence_tolerance"]),
                "track_best": bool(values["track_best"]),
                "local_search_sweeps": int(values["local_search_sweeps"]),
                "local_search_tolerance": float(
                    values["local_search_tolerance"]
                ),
            }
        )
        if parameters["mode"] not in {"ballistic", "discrete"}:
            raise ValueError(
                "adaptive Torch SBM mode must be ballistic or discrete"
            )
        nonnegative_parameters = (
            "heat_coefficient",
            "pressure_slope",
            "convergence_tolerance",
            "local_search_tolerance",
        )
        for name in nonnegative_parameters:
            if parameters[name] < 0.0:
                raise ValueError(
                    f"adaptive Torch SBM {name} must be nonnegative"
                )
        for name in ("sampling_period", "convergence_threshold"):
            if parameters[name] <= 0:
                raise ValueError(f"adaptive Torch SBM {name} must be positive")
        if parameters["local_search_sweeps"] < 0:
            raise ValueError(
                "adaptive Torch SBM local_search_sweeps must be nonnegative"
            )
        return parameters


BQMSolverFactory.registerSolver("adaptive_torch_sbm", AdaptiveTorchSBMBQMSolver)
