"""Application interfaces and compatibility imports for qubo_solvers."""
from importlib import import_module
from margin_calculator.optimization.optimization_problem import OptimizationProblem
from margin_calculator.optimization.optimization_result import BQMOptimizationResult, OptimizationSolverResult
from .bqm_solver import BQMSolver
from .bqm_solver_factory import BQMSolverFactory
from .bqm_solver_config import BQMSolverConfig
from .bqm_execution_policy import BQMExecutionPolicy, BatchBQMExecutionPolicy, SequentialBQMExecutionPolicy
from .execution_memory import BQMExecutionMemory
from .resource_plan import BQMResourcePlan
from .candidate_selection import CandidateSelection
from ...optimization_problem.qubo_problem import QUBOProblem
OptimizationSolver = BQMSolver
_COMPAT = {
    'LibraryBQMSolver': 'library_bqm_solver',
    'AdaptiveTorchSBMBQMSolver': 'adaptive_torch_sbm_bqm_solver',
    'TorchSBMBQMSolver': 'torch_sbm_bqm_solver',
    'SBMBQMSolver': 'sbm_bqm_solver',
    'TorchSVLBQMSolver': 'torch_svl_bqm_solver',
    'TorchTransverseRouteBQMSolver': 'torch_transverse_route_bqm_solver',
    'TorchExchangeCascadeBQMSolver': 'torch_exchange_cascade_bqm_solver',
    'TorchAltermagnetBQMSolver': 'torch_altermagnet_bqm_solver',
    'TorchDynamicalGeometryBQMSolver': 'torch_dynamical_geometry_bqm_solver',
    'TorchGeometricBQMSolver': 'torch_geometric_bqm_solver',
    'TorchPhononExchangeBQMSolver': 'torch_phonon_exchange_bqm_solver',
    'TorchSupersymmetricBQMSolver': 'torch_supersymmetric_bqm_solver',
    'TorchCategoricalBQMSolver': 'torch_categorical_bqm_solver',
    'TorchCategoricalTRFBQMSolver': 'torch_categorical_trf_bqm_solver',
    **{name:'classical_dwave_bqm_solvers' for name in ('PlanarGraphBQMSolver','RandomBQMSolver','SimulatedAnnealingBQMSolver','SteepestDescentBQMSolver','TabuBQMSolver','TreeDecompositionBQMSolver','TreeDecompositionSamplerBQMSolver')},
}
def __getattr__(name):
    if name not in _COMPAT:
        raise AttributeError(name)
    return getattr(import_module(f'{__name__}.{_COMPAT[name]}'), name)
__all__ = ['BQMSolver','BQMSolverFactory','BQMSolverConfig','BQMExecutionPolicy','BatchBQMExecutionPolicy',
           'SequentialBQMExecutionPolicy','BQMExecutionMemory','BQMResourcePlan','CandidateSelection',
           'QUBOProblem','BQMOptimizationResult','OptimizationProblem','OptimizationSolver','OptimizationSolverResult', *_COMPAT]
