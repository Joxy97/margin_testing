"""Binary quadratic model solver interfaces and factories."""

from margin_calculator.optimization.optimization_problem import OptimizationProblem
from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
    OptimizationSolverResult,
)

from .bqm_solver import BQMSolver
OptimizationSolver = BQMSolver
from .bqm_execution_policy import (
    BQMExecutionPolicy,
    BatchBQMExecutionPolicy,
    SequentialBQMExecutionPolicy,
)
from .bqm_solver_factory import BQMSolverFactory
from .bqm_solver_config import BQMSolverConfig
from .adaptive_torch_sbm_bqm_solver import AdaptiveTorchSBMBQMSolver
from .classical_dwave_bqm_solvers import (
    PlanarGraphBQMSolver,
    RandomBQMSolver,
    SimulatedAnnealingBQMSolver,
    SteepestDescentBQMSolver,
    TabuBQMSolver,
    TreeDecompositionBQMSolver,
    TreeDecompositionSamplerBQMSolver,
)
from .sbm_bqm_solver import SBMBQMSolver
from .torch_sbm_bqm_solver import TorchSBMBQMSolver
from .torch_svl_bqm_solver import TorchSVLBQMSolver
from ...optimization_problem.qubo_problem import QUBOProblem

from .resource_plan import BQMResourcePlan

from .execution_memory import BQMExecutionMemory

from .candidate_selection import CandidateSelection

__all__ = [
    "CandidateSelection",

    "BQMExecutionMemory",

    "BQMResourcePlan",

    "BQMSolver",
    "BQMExecutionPolicy",
    "BatchBQMExecutionPolicy",
    "BQMSolverFactory",
    "BQMSolverConfig",
    "BQMOptimizationResult",
    "AdaptiveTorchSBMBQMSolver",
    "OptimizationProblem",
    "OptimizationSolver",
    "OptimizationSolverResult",
    "PlanarGraphBQMSolver",
    "QUBOProblem",
    "RandomBQMSolver",
    "SBMBQMSolver",
    "SimulatedAnnealingBQMSolver",
    "SequentialBQMExecutionPolicy",
    "SteepestDescentBQMSolver",
    "TabuBQMSolver",
    "TorchSBMBQMSolver",
    "TorchSVLBQMSolver",
    "TreeDecompositionBQMSolver",
    "TreeDecompositionSamplerBQMSolver",
]
