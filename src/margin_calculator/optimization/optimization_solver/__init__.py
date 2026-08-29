"""General optimization problem, solver, and result interfaces."""

from ..optimization_problem import OptimizationProblem
from ..optimization_result import OptimizationSolverResult
from .optimization_solver import OptimizationSolver

__all__ = [
    "OptimizationProblem",
    "OptimizationSolver",
    "OptimizationSolverResult",
]
