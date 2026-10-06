"""Thin adapters to the repository's existing unconstrained binary solvers."""
import numpy as np
import inspect
import importlib.metadata
from pathlib import Path
from qubo_solvers import (
    LIBRARY_SOLVERS, SOLVERS, create_bqm_solver, solver_capabilities,
    GPU_SOLVERS as TORCH_SOLVERS, CPU_SOLVERS as CLASSICAL_SOLVERS,
    CATEGORICAL_SOLVERS,
)
from .catalog import ROOT
from .model import binaryVector
from .storage import sha256

TREE_SOLVERS = ('lib_tree_decomposition_solver', 'lib_tree_decomposition_sampler')
GPU_SOLVERS = tuple(name for name in TORCH_SOLVERS if name not in CATEGORICAL_SOLVERS + TREE_SOLVERS)
DETERMINISTIC_SOLVERS = ('lib_planar_graph', 'lib_tree_decomposition_solver')


class UnsupportedProblem(ValueError):
    """The existing algorithm cannot accept this problem unchanged."""


class BackendUnavailable(RuntimeError):
    """Required local dependency, native library, or requested device is absent."""


def solverCapabilities():
    return [solver_capabilities(name) for name in SOLVERS]


def structuralCheck(problem, solverId):
    if solverId in CATEGORICAL_SOLVERS:
        raise UnsupportedProblem('Categorical solvers require one-hot groups covering every variable; the 37 original QUBOs have none. Adding groups would change the benchmark.')
    if solverId == 'lib_planar_graph':
        diagonal=problem.rows == problem.cols
        twiceField=np.zeros(problem.n,dtype=np.float64)
        np.add.at(twiceField,problem.rows,problem.values)
        np.add.at(twiceField,problem.cols[~diagonal],problem.values[~diagonal])
        if np.any(twiceField != 0):
            raise UnsupportedProblem('PlanarGraphSolver requires zero Ising linear fields; this QUBO has nonzero fields.')
        if problem.n < 3:raise UnsupportedProblem('PlanarGraphSolver requires at least three variables')
        import networkx as nx
        graph=nx.Graph();graph.add_nodes_from(range(problem.n))
        graph.add_edges_from(zip(problem.rows[~diagonal],problem.cols[~diagonal]))
        if not nx.check_planarity(graph)[0]:raise UnsupportedProblem('Interaction graph is not planar')


def toSolverProblem(problem):
    from qubo_solvers.backends.problem import QUBOProblem
    diagonal=problem.rows == problem.cols
    linear=np.zeros(problem.n,dtype=np.float64)
    linear[problem.rows[diagonal]]=problem.values[diagonal]
    return QUBOProblem(linear,problem.rows[~diagonal],problem.cols[~diagonal],
                       2.*problem.values[~diagonal].astype(np.float64),float(problem.offset))


class Adapter:
    def __init__(self, problem, solverId, seed, device, precision, parameters, constructorParameters=None):
        if solverId not in SOLVERS:
            raise ValueError(f'Unknown solver {solverId!r}')
        if precision not in ('float32','float64') or device == 'auto':
            raise ValueError('Use explicit device and float32/float64 precision')
        if solverId in CLASSICAL_SOLVERS and precision != 'float64':
            raise ValueError('Classical sampler wrappers use float64 coefficients; request float64 explicitly')
        if set(parameters)&{'seed','dtype','device','devices'}:
            raise ValueError('Set seed/device/precision through runner fields, not solver parameters')
        structuralCheck(problem,solverId)
        constructor=dict(constructorParameters or {})
        if constructor:raise ValueError('Set device through the runner; canonical solvers need no native library path')
        self.solverId=solverId;self.backendDetails={}
        self.problem=toSolverProblem(problem)
        self.parameters=dict(parameters)
        self._treePlan=None
        if solverId in TREE_SOLVERS:
            from qubo_solvers.backends.tree_decomposition import elimination_preflight, TreewidthExceeded
            try:
                self._treePlan=elimination_preflight(self.problem, dict(parameters,dtype=precision),
                                                    sampling=solverId.endswith('sampler'))
            except TreewidthExceeded as exc:
                raise UnsupportedProblem(str(exc)) from exc
            self.parameters['elimination_order']=list(self._treePlan.order)
        if solverId in TORCH_SOLVERS:
            import torch
            if device.startswith('cuda') and not torch.cuda.is_available():raise BackendUnavailable(f'Requested {device}, but CUDA is unavailable')
            constructor['device']=device
        elif device != 'cpu':raise ValueError(f'{solverId} exposes a CPU interface only; request cpu explicitly')
        self.solver=create_bqm_solver(solverId,constructor)
        self.n=problem.n
        self.device=self.solver.device if solverId in TORCH_SOLVERS else 'cpu'
        if self.device != device and not (device=='cuda' and self.device=='cuda:0'):
            raise ValueError(f'Requested {device}, resolved {self.device}')
        if solverId in TORCH_SOLVERS:
            self.parameters.update(seed=seed, dtype=precision)
            self.resolvedParameters=self.solver._getParameters(self.parameters)
            if solverId in TREE_SOLVERS:
                self._prepareTree()
        else:
            self._prepareClassical(seed)

    def _prepareTree(self):
        from qubo_solvers.backends.tree_decomposition import TreewidthExceeded
        try:
            plan=self._treePlan or self.solver.preflight(self.problem,self.parameters)
        except TreewidthExceeded as exc:
            raise UnsupportedProblem(str(exc)) from exc
        self.parameters['elimination_order']=list(plan.order)
        self.resolvedParameters['elimination_order']=list(plan.order)
        self.backendDetails.update(elimination_width=plan.width,
                                   estimated_elimination_memory_bytes=plan.estimated_memory_bytes,
                                   seed_used=self.solverId=='lib_tree_decomposition_sampler')

    def _prepareClassical(self,seed):
        try:sampler=self.solver._createSampler()
        except (ImportError,OSError) as exc:raise BackendUnavailable(str(exc)) from exc
        try:
            unknown=set(self.parameters)-sampler.parameters.keys()
            if unknown:raise ValueError(f'Unknown {self.solverId} parameters: {sorted(unknown)}')
            if 'seed' in sampler.parameters:
                if not 0 <= seed < 2**31:raise ValueError('Classical adapter seeds must fit a nonnegative signed 32-bit integer')
                self.parameters['seed']=seed
            self.resolvedParameters={k:p.default for k,p in inspect.signature(sampler.sample).parameters.items()
                                     if k in sampler.parameters and p.default is not inspect.Parameter.empty}
            self.resolvedParameters.update(self.parameters)
            self.backendDetails.update(dwave_samplers_version=importlib.metadata.version('dwave-samplers'),
                                       dimod_version=importlib.metadata.version('dimod'),
                                       seed_used='seed' in sampler.parameters,
                                       sampler_source_sha256=sha256(Path(inspect.getfile(type(sampler))).read_bytes()))
        finally:sampler.close()

    def solve(self):
        result=self.solver.solve(self.problem,self.parameters)
        if not np.isfinite(result.energy):raise ValueError('Solver returned a nonfinite diagnostic energy')
        return binaryVector(result.sample,self.n),float(result.energy)

    def synchronize(self):
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)

    def describe(self):
        sourcePath=Path(inspect.getfile(type(self.solver)))
        details=dict(actual_device=self.device,
                    library='qubo_solvers',
                    capabilities=solver_capabilities(self.solverId),
                    preparation_mode='host_compact_input',
                    resident_preparation_mode=getattr(self.solver, 'residentMode', None),
                    adapter_transformation='Q diagonal -> linear; 2*Q upper offdiagonal -> pair-once biases; identity variable order; no adapter rounding/scaling',
                    internal_transformation='Existing solver owns Ising/spin conversion, normalization and binary rounding; original QUBO rescored independently.',
                    parameters=self.parameters,solver_class=type(self.solver).__name__,
                    resolved_parameters=self.resolvedParameters,
                    solver_source_sha256=sha256(sourcePath.read_bytes()),
                    transfer_seconds=None,transfer_note='Existing solve() includes preparation/transfers; cannot separately measure without changing solver interface.',
                    warmup_policy='Fresh process; device context initialized in setup; no solver warmup; first-call kernel/JIT cost included in solve.',
                    incumbent_history_available=False)
        if self.solverId in TORCH_SOLVERS:
            import torch
            details.update(torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),
                           gpu_name=torch.cuda.get_device_name(self.device) if self.device.startswith('cuda') else None,
                           cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        if self.solverId in LIBRARY_SOLVERS:
            from qubo_solvers import library_solver_class
            algorithm=library_solver_class(self.solverId)
            details.update(library='qubo_solvers',algorithm_class=algorithm.__name__,
                           algorithm_source_sha256=sha256(Path(inspect.getfile(algorithm)).read_bytes()),
                           history_note='Library supports iteration-indexed energies; no timed incumbent history is fabricated by the benchmark.')
        details.update(self.backendDetails)
        return details
