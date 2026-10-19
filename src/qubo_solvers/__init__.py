"""Dense unconstrained QUBO/Ising optimization on CPU and CUDA."""
from .problems import Ising, Problem, QUBO
from .results import OptimizationResult, TerminationReason
from .solvers import Solver, MemoryEstimate, estimate_memory
from .simulated_annealing import SimulatedAnnealing
from .greedy_local_search import GreedyLocalSearch
from .spin_vector_langevin import SpinVectorLangevin
from .angular_annealing import AngularAnnealing
from .transverse_route import TransverseRoute
from .easy_axis_annealing import EasyAxisAnnealing
from .spin_coherent_annealing import SpinCoherentAnnealing
from .vector_amplitude_annealing import VectorAmplitudeAnnealing
from .mean_field_annealing import MeanFieldAnnealing
from .tap_annealing import TAPAnnealing
from .spherical_annealing import SphericalAnnealing
from .contact_annealing import ContactAnnealing
from .replica_annealing import ReplicaAnnealing
from .heat_bath_annealing import HeatBathAnnealing
from .tabu_search import TabuSearch
from .random_search import RandomSearch
from .exchange_cascade import ExchangeCascade
from .su2_chiral_isospin import SU2ChiralIsospin
from .gravity_tunneling import GravityTunneling
from .hamiltonians import (
    HAMILTONIAN_SOLVERS, SSHDimerOscillator, SparseSYK, BoseHubbard, FermiHubbard,
    TJVacancy, Inozemtsev, Dicke, HaldaneShastry, QuantumClock, JaynesCummings,
    JaynesCummingsHubbard, MajumdarGhosh, J1J2, Kuramoto, PottsHeatBath, ZNChiral,
    NVector, QuantumRotor, XYCoordinate, QuantumDimerWorldline, KitaevChain,
    BogoliubovDeGennes, SpinSqueezing,
)
from .registry import (
    LIBRARY_SOLVERS, NATIVE_SOLVERS, BACKEND_SOLVERS, SOLVERS, BackendSolver,
    CPU_SOLVERS, GPU_SOLVERS, CATEGORICAL_SOLVERS, REMOVED_SOLVERS, solver_capabilities, register_bqm_solver,
    create_solver, create_bqm_solver, library_solver_class, default_algorithm_parameters,
)

__all__ = [
    'QUBO', 'Ising', 'Problem', 'Solver', 'OptimizationResult', 'TerminationReason',
    'MemoryEstimate', 'estimate_memory', 'SimulatedAnnealing', 'GreedyLocalSearch',
    'SpinVectorLangevin', 'AngularAnnealing', 'TransverseRoute', 'EasyAxisAnnealing',
    'SpinCoherentAnnealing', 'VectorAmplitudeAnnealing', 'MeanFieldAnnealing',
    'TAPAnnealing', 'SphericalAnnealing', 'ContactAnnealing', 'ReplicaAnnealing',
    'HeatBathAnnealing', 'TabuSearch', 'RandomSearch', 'ExchangeCascade',
    'SU2ChiralIsospin', 'GravityTunneling',
    'HAMILTONIAN_SOLVERS', 'SSHDimerOscillator', 'SparseSYK', 'BoseHubbard',
    'FermiHubbard', 'TJVacancy', 'Inozemtsev', 'Dicke', 'HaldaneShastry',
    'QuantumClock', 'JaynesCummings', 'JaynesCummingsHubbard', 'MajumdarGhosh',
    'J1J2', 'Kuramoto', 'PottsHeatBath', 'ZNChiral', 'NVector', 'QuantumRotor',
    'XYCoordinate', 'QuantumDimerWorldline', 'KitaevChain', 'BogoliubovDeGennes',
    'SpinSqueezing',
    'LIBRARY_SOLVERS', 'NATIVE_SOLVERS', 'BACKEND_SOLVERS', 'SOLVERS', 'BackendSolver',
    'CPU_SOLVERS', 'GPU_SOLVERS', 'CATEGORICAL_SOLVERS', 'REMOVED_SOLVERS', 'solver_capabilities', 'register_bqm_solver',
    'create_solver', 'create_bqm_solver', 'library_solver_class', 'default_algorithm_parameters',
]
