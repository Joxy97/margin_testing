"""Base interface for QUBO solvers."""

from abc import abstractmethod

from decision_maker.optimization.optimization_result import (
    OptimizationSolverResult,
)
from decision_maker.optimization.optimization_solver import OptimizationSolver

from ...optimization_problem.qubo_problem import QUBOProblem


class BQMSolver(OptimizationSolver):
    """Solve an application-level QUBO problem."""

    @abstractmethod
    def solve(self, problem: QUBOProblem) -> OptimizationSolverResult:
        """Solve ``problem`` and return the solver-specific result."""
        raise NotImplementedError
