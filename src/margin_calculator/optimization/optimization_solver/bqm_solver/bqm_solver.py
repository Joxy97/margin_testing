"""Base interface for QUBO solvers."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)

from ...optimization_problem.qubo_problem import QUBOProblem
from .resource_plan import BQMResourcePlan


class BQMSolver(ABC):
    """Solve an application-level QUBO problem."""

    @property
    def batchParallelism(self) -> int:
        """Return the number of independent workers available to a batch."""
        return 1

    def estimatedWorkingMemoryBytes(self, problem: QUBOProblem,
                                    solverParameters: Mapping[str, Any] | None = None) -> int:
        """Estimate per-problem working storage without loading an accelerator."""
        return problem.numericMemoryBytes

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

    def solvePlanned(self, problems: Sequence[QUBOProblem], plan: BQMResourcePlan,
                     solverParameters: Mapping[str, Any] | None = None) -> list[BQMOptimizationResult]:
        """Execute admitted work; device adapters honor the supplied assignment."""
        return self.solveMany(problems, solverParameters)

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
        from .candidate_selection import CandidateSelection

        selection = CandidateSelection(problem)
        selection.add(candidates)
        return selection.result()

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
