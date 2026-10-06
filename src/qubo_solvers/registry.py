"""Canonical solver selection: one library-owned implementation per retained method.

All 28 canonical entries are owned by this library. The 17 dense tensor solvers
provide full trajectories; the specialized execution backends provide selected
results through a best-only tensor adapter and their compact QUBO interface.
Historical names are migration hints, never separately registered algorithms.
"""

import math
from types import MappingProxyType
from typing import Any

import torch

from .angular_annealing import AngularAnnealing
from .contact_annealing import ContactAnnealing
from .easy_axis_annealing import EasyAxisAnnealing
from .exchange_cascade import ExchangeCascade
from .greedy_local_search import GreedyLocalSearch
from .heat_bath_annealing import HeatBathAnnealing
from .mean_field_annealing import MeanFieldAnnealing
from .problems import Ising, QUBO
from .random_search import RandomSearch
from .replica_annealing import ReplicaAnnealing
from .results import OptimizationResult
from .simulated_annealing import SimulatedAnnealing
from .solvers import _integer, _matrix
from .spherical_annealing import SphericalAnnealing
from .spin_coherent_annealing import SpinCoherentAnnealing
from .spin_vector_langevin import SpinVectorLangevin
from .tabu_search import TabuSearch
from .tap_annealing import TAPAnnealing
from .transverse_route import TransverseRoute
from .vector_amplitude_annealing import VectorAmplitudeAnnealing


_LIBRARY_CLASSES = {
    'lib_simulated_annealing': SimulatedAnnealing,
    'lib_greedy_local_search': GreedyLocalSearch,
    'lib_spin_vector_langevin': SpinVectorLangevin,
    'lib_angular_annealing': AngularAnnealing,
    'lib_transverse_route': TransverseRoute,
    'lib_easy_axis_annealing': EasyAxisAnnealing,
    'lib_spin_coherent_annealing': SpinCoherentAnnealing,
    'lib_vector_amplitude_annealing': VectorAmplitudeAnnealing,
    'lib_mean_field_annealing': MeanFieldAnnealing,
    'lib_tap_annealing': TAPAnnealing,
    'lib_spherical_annealing': SphericalAnnealing,
    'lib_contact_annealing': ContactAnnealing,
    'lib_replica_annealing': ReplicaAnnealing,
    'lib_heat_bath_annealing': HeatBathAnnealing,
    'lib_tabu_search': TabuSearch,
    'lib_random_search': RandomSearch,
    'lib_exchange_cascade': ExchangeCascade,
}
LIBRARY_SOLVERS = NATIVE_SOLVERS = tuple(_LIBRARY_CLASSES)
BACKEND_SOLVERS = (
    'lib_simulated_bifurcation', 'lib_altermagnet', 'lib_dynamical_geometry',
    'lib_geometric', 'lib_phonon_exchange', 'lib_supersymmetric',
    'lib_categorical', 'lib_categorical_trf', 'lib_planar_graph',
    'lib_tree_decomposition_solver', 'lib_tree_decomposition_sampler',
)
SOLVERS = NATIVE_SOLVERS + BACKEND_SOLVERS
CPU_SOLVERS = ('lib_planar_graph',)
TREE_SOLVERS = ('lib_tree_decomposition_solver', 'lib_tree_decomposition_sampler')
CATEGORICAL_SOLVERS = ('lib_categorical', 'lib_categorical_trf')
GPU_SOLVERS = tuple(name for name in SOLVERS if name not in CPU_SOLVERS)
REMOVED_SOLVERS = MappingProxyType({
    'random': 'lib_random_search', 'simulated_annealing': 'lib_simulated_annealing',
    'steepest_descent': 'lib_greedy_local_search', 'tabu': 'lib_tabu_search',
    'torch_svl': 'lib_spin_vector_langevin', 'torch_transverse_route': 'lib_transverse_route',
    'torch_exchange_cascade': 'lib_exchange_cascade',
    'sbm': 'lib_simulated_bifurcation', 'torch_sbm': 'lib_simulated_bifurcation',
    'adaptive_torch_sbm': 'lib_simulated_bifurcation',
    'torch_altermagnet': 'lib_altermagnet', 'torch_dynamical_geometry': 'lib_dynamical_geometry',
    'torch_geometric': 'lib_geometric', 'torch_phonon_exchange': 'lib_phonon_exchange',
    'torch_supersymmetric': 'lib_supersymmetric', 'torch_categorical': 'lib_categorical',
    'torch_categorical_trf': 'lib_categorical_trf', 'planar_graph': 'lib_planar_graph',
    'tree_decomposition_solver': 'lib_tree_decomposition_solver',
    'tree_decomposition_sampler': 'lib_tree_decomposition_sampler',
})
_CUSTOM_BQM_SOLVERS = {}


def _unknown(name):
    if name in REMOVED_SOLVERS:
        raise ValueError(f'Retired solver ID {name!r}; use {REMOVED_SOLVERS[name]!r} '
                         'and migrate its parameters. Old implementations were removed.')
    raise ValueError(f'Unknown solver: {name!r}')


def solver_capabilities(name):
    if name not in SOLVERS:
        _unknown(name)
    resident_mode = ('device' if name in NATIVE_SOLVERS or name == 'lib_simulated_bifurcation' else
                     None if name in CPU_SOLVERS + TREE_SOLVERS else 'source_snapshot')
    return dict(solver=name, gpu=name in GPU_SOLVERS,
                backend='dwave_samplers_cpu' if name in CPU_SOLVERS else 'torch',
                native_tensor_contract=name in NATIVE_SOLVERS,
                resident_preparation_mode=resident_mode,
                cpu_topology_preparation=name in TREE_SOLVERS,
                precisions=['float64'] if name in CPU_SOLVERS else ['float32', 'float64'],
                restriction=('Requires complete one-hot groups' if name in CATEGORICAL_SOLVERS else
                             'Requires planar zero-field Ising structure' if name=='lib_planar_graph' else
                             'Requires elimination width <= 25 and tables within the configured memory budget' if name in TREE_SOLVERS else None))


def library_solver_class(name: str):
    """Concrete class of one of the 17 native tensor algorithms."""
    try:
        return _LIBRARY_CLASSES[name]
    except KeyError as error:
        raise ValueError(f'Unknown native library solver: {name!r}') from error


def default_algorithm_parameters(name: str, steps: int = 1000) -> dict[str, Any]:
    library_solver_class(name)
    _integer('steps', steps, 0)
    if name in ('lib_simulated_annealing', 'lib_heat_bath_annealing'):
        return dict(sweeps=steps, start_temperature=2., end_temperature=.01)
    return dict(max_steps=steps)


def register_bqm_solver(name, solver_class):
    """Register an application extension without shadowing a canonical solver."""
    if name in SOLVERS or name in REMOVED_SOLVERS:
        raise ValueError(f'Cannot replace canonical or retired solver {name!r}')
    _CUSTOM_BQM_SOLVERS[name] = solver_class


def create_bqm_solver(name: str, constructor_parameters=None):
    """Instantiate a library-owned compact-QUBO execution path.

    No application factory or application numerical kernel is imported. Native
    algorithms and specialized backends share the same problem/result types.
    """
    if name in NATIVE_SOLVERS:
        from .backends.library_solver import create_library_bqm_solver
        return create_library_bqm_solver(name, constructor_parameters)
    if name in BACKEND_SOLVERS:
        from .backends import create_backend
        return create_backend(name, constructor_parameters)
    if name in _CUSTOM_BQM_SOLVERS:
        return _CUSTOM_BQM_SOLVERS[name](**dict(constructor_parameters or {}))
    _unknown(name)


def create_solver(name: str, **algorithm_parameters):
    """Create a canonical solver through the tensor problem interface.

    Specialized backends expose best-only results. For sparse compact QUBOs,
    grouped constraints, or resident application batches use create_bqm_solver.
    """
    if name in NATIVE_SOLVERS:
        parameters = default_algorithm_parameters(name)
        parameters.update(algorithm_parameters)
        return library_solver_class(name)(**parameters)
    if name in BACKEND_SOLVERS:
        return BackendSolver(name, **algorithm_parameters)
    _unknown(name)


class BackendSolver:
    """Best-only tensor view of a specialized library backend.

    Dense input conversion has a host boundary for original-coefficient scoring;
    search kernels run on the explicitly requested device. For a GPU resident
    application pipeline use create_bqm_solver().solveResidentPlanned().
    Unavailable iteration/restart diagnostics are -1, never invented histories.
    """
    def __init__(self, name, *, device=None, precision=None, **parameters):
        if name not in BACKEND_SOLVERS:
            _unknown(name)
        if device is not None and (not isinstance(device, str) or device == 'auto'):
            raise ValueError('device must be an explicit CPU/CUDA device string')
        if precision is not None and precision not in ('float32', 'float64'):
            raise ValueError('precision must be float32 or float64')
        reserved = set(parameters) & {'seed', 'runs', 'num_reads', 'dtype', 'devices', 'libraryPath'}
        if reserved:
            raise ValueError(f'Use solve(restarts=..., seed=...) and precision/device for {sorted(reserved)}')
        self.name, self.device, self.precision = name, device, precision
        self.parameters = MappingProxyType(dict(parameters))

    @torch.no_grad()
    def solve(self, problem, *, restarts=1, initial_assignments=None, seed=None,
              history_interval=None, batch_size=None, best_only=False, memory_limit_bytes=None):
        matrix = _matrix(problem)
        _integer('restarts', restarts, 1)
        if not best_only:
            raise ValueError('Specialized backends expose only a selected result; use best_only=True')
        requested = [name for name, value in dict(initial_assignments=initial_assignments,
                     history_interval=history_interval, batch_size=batch_size,
                     memory_limit_bytes=memory_limit_bytes).items() if value is not None]
        if requested:
            raise ValueError(f'Specialized backend cannot provide the common contract for {requested}')
        if seed is not None:
            _integer('seed', seed, 0)
        device = str(matrix.device)
        precision = str(matrix.dtype).removeprefix('torch.')
        if self.device is not None:
            requested_device = torch.device(self.device)
            if requested_device.type == 'cuda' and requested_device.index is None:
                requested_device = torch.device('cuda', torch.cuda.current_device())
            if requested_device != matrix.device:
                raise ValueError('Requested device must match the problem device; use problem.to()')
        if self.precision is not None and self.precision != precision:
            raise ValueError('Requested precision must match the problem dtype; use problem.to()')
        if self.name in CATEGORICAL_SOLVERS:
            raise ValueError('Categorical solvers require one-hot groups; use create_bqm_solver with grouped QUBOProblem')
        if self.name in CPU_SOLVERS and (matrix.device.type != 'cpu' or precision != 'float64'):
            raise ValueError(f'{self.name} requires CPU float64 coefficients; move the problem explicitly')
        import numpy as np
        from .backends.problem import QUBOProblem
        qubo = problem.to_qubo() if isinstance(problem, Ising) else problem
        coefficients = qubo.Q.detach().cpu().numpy()
        heads, tails = np.nonzero(np.triu(coefficients, 1))
        bqm = QUBOProblem(coefficients.diagonal().astype(np.float64), heads, tails,
                          2 * coefficients[heads, tails].astype(np.float64), float(qubo.offset))
        solver = create_bqm_solver(self.name, {'device': device} if self.name in GPU_SOLVERS else {})
        parameters = dict(self.parameters)
        if self.name in GPU_SOLVERS:
            parameters.update(runs=restarts, dtype=precision)
            if seed is not None:
                parameters['seed'] = seed
        else:
            sampler = solver._createSampler()
            try:
                if 'num_reads' in sampler.parameters:
                    parameters['num_reads'] = restarts
                elif restarts != 1:
                    raise ValueError(f'{self.name} does not support independent restarts')
                if seed is not None and 'seed' in sampler.parameters:
                    parameters['seed'] = seed
            finally:
                sampler.close()
        result = solver.solve(bqm, parameters)
        if not math.isfinite(result.energy):
            raise ValueError('Backend returned a nonfinite energy')
        sample = torch.as_tensor(result.sample, device=matrix.device)
        if sample.shape != (problem.n_variables,) or not ((sample == 0) | (sample == 1)).all():
            raise ValueError('Backend returned an invalid binary assignment')
        sample = sample.to(dtype=torch.int8)
        if isinstance(problem, Ising):
            sample = 2 * sample - 1
        energy = problem.energy(sample).reshape(1)
        unavailable = torch.full((1,), -1, device=matrix.device, dtype=torch.int64)
        return OptimizationResult(None, None, sample[None, :], energy,
                                  unavailable.clone(), unavailable.clone(), restart_indices=unavailable)
