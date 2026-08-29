"""Base interface for QUBO solvers."""

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)
from margin_calculator.optimization.optimization_solver import OptimizationSolver

from ...optimization_problem.qubo_problem import QUBOProblem


class BQMSolver(OptimizationSolver):
    """Solve an application-level QUBO problem."""

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
            problem.oneHotGroups,
        )

    @classmethod
    def _selectBestCandidates(
        cls,
        candidates: Iterable[tuple[Sequence[int], float]],
        oneHotGroups: Iterable[Iterable[int]] = (),
    ) -> tuple[tuple[int, ...], float]:
        """Choose the best valid candidate, falling back to the best overall."""
        best_overall: tuple[tuple[int, ...], float] | None = None
        best_valid: tuple[tuple[int, ...], float] | None = None
        groups = tuple(tuple(group) for group in oneHotGroups)
        for sample, energy in candidates:
            candidate = tuple(int(value) for value in sample), float(energy)
            if best_overall is None or candidate[1] < best_overall[1]:
                best_overall = candidate
            if cls._isValidOneHotSample(candidate[0], groups) and (
                best_valid is None or candidate[1] < best_valid[1]
            ):
                best_valid = candidate
        if best_overall is None:
            raise ValueError("BQM solver returned no samples")
        return best_valid or best_overall

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
