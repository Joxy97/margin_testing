"""Result returned by binary quadratic model optimization."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

from .optimization_solver_result import OptimizationSolverResult


@dataclass
class BQMOptimizationResult(OptimizationSolverResult):
    """Store a solver's binary sample and its corresponding energy."""

    sample: Mapping[Hashable, int] | Sequence[int] = field(default_factory=dict)
    energy: float = 0.0
