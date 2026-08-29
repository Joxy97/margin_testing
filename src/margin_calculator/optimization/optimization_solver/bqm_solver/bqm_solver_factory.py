"""Factory for binary quadratic model solvers."""

from typing import Any, ClassVar, Mapping

from .bqm_solver import BQMSolver


class BQMSolverFactory:
    """Create registered BQM solvers from string names."""

    _solvers: ClassVar[dict[str, type[BQMSolver]]] = {}

    @classmethod
    def registerSolver(
        cls,
        name: str,
        solverClass: type[BQMSolver],
    ) -> None:
        """Associate ``name`` with a concrete BQM solver class."""
        cls._solvers[name] = solverClass

    @classmethod
    def createBQMSolver(
        cls,
        name: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> BQMSolver:
        """Create the solver registered under ``name``."""
        try:
            solver_class = cls._solvers[name]
        except KeyError as error:
            raise ValueError(f"Unknown BQM solver: {name!r}") from error
        return solver_class(**dict(parameters or {}))

    @classmethod
    def create(
        cls,
        name: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> BQMSolver:
        """Create a solver; shorthand for :meth:`createBQMSolver`."""
        return cls.createBQMSolver(name, parameters)
