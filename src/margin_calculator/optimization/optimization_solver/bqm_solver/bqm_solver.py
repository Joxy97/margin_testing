"""Base interface for QUBO solvers."""

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy
from scipy.sparse import coo_matrix

from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)
from margin_calculator.optimization.optimization_solver import OptimizationSolver

from ...optimization_problem.qubo_problem import QUBOProblem


class BQMSolver(OptimizationSolver):
    """Solve an application-level QUBO problem."""

    @property
    def batchParallelism(self) -> int:
        """Return the number of independent workers available to a batch."""
        return 1

    @abstractmethod
    def solve(
        self,
        problem: QUBOProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> BQMOptimizationResult:
        """Solve ``problem`` and return the solver-specific result."""
        raise NotImplementedError

    def solveMany(
        self,
        problems: Sequence[QUBOProblem],
        solverParameters: Mapping[str, Any] | None = None,
    ) -> list[BQMOptimizationResult]:
        """Solve several problems; concrete solvers may provide a native batch."""
        return [self.solve(problem, solverParameters) for problem in problems]

    def beginSeries(self) -> None:
        """Begin an ordered problem series; stateful solvers may warm-start it."""

    def endSeries(self) -> None:
        """End the current ordered problem series."""

    @classmethod
    def _selectBestSample(
        cls,
        sampleSet: Any,
        problem: QUBOProblem,
    ) -> tuple[tuple[int, ...], float]:
        """Return the lowest-energy valid one-hot sample when available."""
        return cls._selectBestCandidates(
            (
                (
                    tuple(
                        int(row.sample[variable])
                        for variable in range(problem.variableCount)
                    ),
                    float(row.energy),
                )
                for row in sampleSet.data(
                    fields=["sample", "energy"],
                    sorted_by=None,
                )
            ),
            problem,
        )

    @classmethod
    def _selectBestCandidates(
        cls,
        candidates: Iterable[tuple[Sequence[int], float]],
        problem: QUBOProblem,
    ) -> tuple[tuple[int, ...], float]:
        """Choose the best feasible candidate or repair every infeasible one."""
        groups = tuple(tuple(group) for group in problem.iterOneHotGroups())
        evaluated: list[tuple[tuple[int, ...], float]] = []
        for sample, solver_energy in candidates:
            binary = tuple(int(value) for value in sample)
            problem.energy(binary)
            evaluated.append((binary, float(solver_energy)))
        if not evaluated:
            raise ValueError("BQM solver returned no samples")
        valid = [
            candidate
            for candidate in evaluated
            if cls._isValidOneHotSample(candidate[0], groups)
        ]
        if valid:
            return min(valid, key=lambda candidate: (candidate[1], candidate[0]))
        if not groups:
            return min(evaluated, key=lambda candidate: (candidate[1], candidate[0]))

        adjacency, linear = cls._repairModel(problem)
        repaired = [
            cls._repairCandidate(sample, problem, groups, adjacency, linear)
            for sample, _energy in evaluated
        ]
        return min(repaired, key=lambda candidate: (candidate[1], candidate[0]))

    @staticmethod
    def _repairModel(problem: QUBOProblem) -> tuple[Any, numpy.ndarray]:
        """Build the local-field representation used by categorical repair."""
        diagonal = problem.quadraticHeads == problem.quadraticTails
        linear = problem.linear.copy()
        if numpy.any(diagonal):
            numpy.add.at(
                linear,
                problem.quadraticHeads[diagonal],
                problem.quadraticBiases[diagonal],
            )
        heads = problem.quadraticHeads[~diagonal].astype(numpy.int64, copy=False)
        tails = problem.quadraticTails[~diagonal].astype(numpy.int64, copy=False)
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
        return adjacency, linear

    @staticmethod
    def _repairCandidate(
        sample: Sequence[int],
        problem: QUBOProblem,
        groups: tuple[tuple[int, ...], ...],
        adjacency: Any,
        linear: numpy.ndarray,
    ) -> tuple[tuple[int, ...], float]:
        """Project and improve one sample by deterministic categorical descent."""
        repaired = numpy.asarray(sample, dtype=numpy.uint8).copy()
        maximum_sweeps = min(100, max(3, 2 * len(groups) + 1))
        for _sweep in range(maximum_sweeps):
            changed = False
            local_fields = linear + adjacency @ repaired
            for group in groups:
                variables = numpy.asarray(group, dtype=numpy.int64)
                selected = variables[repaired[variables] == 1]
                previous = int(selected[0]) if len(selected) == 1 else None
                for variable in selected:
                    repaired[variable] = 0
                    local_fields -= adjacency.getcol(int(variable)).toarray().ravel()
                costs = local_fields[variables]
                best_position = int(numpy.argmin(costs))
                chosen = int(variables[best_position])
                if previous is not None:
                    previous_position = int(
                        numpy.flatnonzero(variables == previous)[0]
                    )
                    if costs[best_position] >= costs[previous_position] - 1e-12:
                        chosen = previous
                repaired[chosen] = 1
                local_fields += adjacency.getcol(chosen).toarray().ravel()
                changed |= previous != chosen
            if not changed:
                break
        result = tuple(int(value) for value in repaired)
        return result, problem.energy(result)

    @staticmethod
    def _isValidOneHotSample(
        sample: Sequence[int],
        oneHotGroups: Iterable[Iterable[int]] = (),
    ) -> bool:
        """Validate every explicitly declared one-hot group."""
        return all(
            sum(int(sample[variable]) for variable in group) == 1
            for group in oneHotGroups
        )
