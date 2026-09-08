"""Typed BQM-solver configuration."""

from dataclasses import dataclass, field
from typing import Any, Mapping
from types import MappingProxyType

from .bqm_solver import BQMSolver
from .bqm_solver_factory import BQMSolverFactory


@dataclass(frozen=True)
class BQMSolverConfig:
    """Select a BQM solver and separate construction from solve options."""

    solverType: str = "simulated_annealing"
    constructorParameters: Mapping[str, Any] = field(default_factory=dict)
    solverParameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("constructorParameters", "solverParameters"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def createBQMSolver(self) -> BQMSolver:
        return BQMSolverFactory.createBQMSolver(
            self.solverType,
            self.constructorParameters,
        )
