"""Base interface for binary quadratic model solvers."""

from abc import ABC, abstractmethod
from typing import Any

from dimod import BinaryQuadraticModel


class BQMSolver(ABC):
    """Solve a D-Wave binary quadratic model."""

    @abstractmethod
    def solve(self, bqm: BinaryQuadraticModel) -> Any:
        """Solve ``bqm`` and return the solver-specific result."""
        raise NotImplementedError
