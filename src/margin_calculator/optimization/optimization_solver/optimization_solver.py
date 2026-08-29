"""Base interface for optimization solvers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from ..optimization_problem import OptimizationProblem
from ..optimization_result import OptimizationSolverResult


class OptimizationSolver(ABC):
    """Solve optimization problems."""

    @abstractmethod
    def solve(
        self,
        problem: OptimizationProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> OptimizationSolverResult:
        """Solve ``problem`` and return its optimization result."""
        raise NotImplementedError
