"""Application factory compatibility boundary for the central library registry."""
from typing import Any, ClassVar, Mapping
from qubo_solvers.registry import _CUSTOM_BQM_SOLVERS
from .bqm_solver import BQMSolver

class BQMSolverFactory:
    # Only caller extensions live here; built-ins have a single library owner.
    _solvers: ClassVar[dict[str,type[BQMSolver]]] = _CUSTOM_BQM_SOLVERS

    @classmethod
    def registerSolver(cls, name, solverClass):
        from qubo_solvers.registry import register_bqm_solver
        register_bqm_solver(name, solverClass)

    @classmethod
    def createBQMSolver(cls, name, parameters=None):
        from qubo_solvers import create_bqm_solver
        return create_bqm_solver(name, parameters)

    @classmethod
    def create(cls, name, parameters=None):
        return cls.createBQMSolver(name, parameters)
