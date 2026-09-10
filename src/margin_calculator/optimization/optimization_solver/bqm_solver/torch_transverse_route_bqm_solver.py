"""Transverse-route angular flow with bounded candidates and reusable buffers.

Equations and defaults follow Joxy97/solver_testing, transverse_route.py,
revision f71e926c2d1fee5fb3ebe6fd872161cccde738d0. See
docs/benchmarks/transverse_route.md for provenance and numerical conventions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from threading import Lock
from typing import Any

import numpy

from ...optimization_result import BQMOptimizationResult
from .bqm_solver_factory import BQMSolverFactory
from .torch_execution import TorchExecution, _IsingProblem, _MAX_TORCH_SEED, _RUN_SEED_STRIDE


_CAPTURE_LOCK = Lock()


class _FlowWorkspace:
    """Own stable storage for one trajectory batch and its matrix products."""

    def __init__(self, torch, matrix, field, theta, parameters):
        self.torch, self.matrix, self.field = torch, matrix, field
        self.theta, self.parameters = theta, parameters
        self.width = theta.shape[1]
        self.channels = 1 + (parameters['route_strength'] != 0) + (parameters['gamma'] != 0)
        self.features = theta.new_empty((len(theta), self.width * self.channels))
        self.fields = torch.empty_like(self.features)
        self.x, self.y, self.route, self.temp, self.factor = (
            torch.empty_like(theta) for _ in range(5))
        self.first, self.second, self.predictor = (torch.empty_like(theta) for _ in range(3))

    def rhs(self, theta, kappa, out):
        torch, p, width = self.torch, self.parameters, self.width
        torch.cos(theta, out=self.x)
        torch.sin(theta, out=self.y)
        torch.mul(self.x, self.y, out=self.route)
        self.features[:, :width].copy_(self.x)
        channel = 1
        if p['route_strength']:
            self.features[:, width:2 * width].copy_(self.route)
            channel += 1
        if p['gamma']:
            torch.mul(self.y, self.y, out=self.temp)
            torch.mul(self.temp, self.y, out=self.features[:, channel * width:(channel + 1) * width])
        # CSR SpMM and dense GEMM both support a reusable output tensor.
        torch.mm(self.matrix, self.features, out=self.fields)
        torch.add(self.fields[:, :width], self.field, out=out)
        out.mul_(self.y)
        channel = 1
        if p['route_strength']:
            torch.mul(self.x, self.x, out=self.factor)
            torch.mul(self.y, self.y, out=self.temp)
            self.factor.sub_(self.temp)
            self.factor.mul_(self.fields[:, width:2 * width])
            out.add_(self.factor, alpha=-p['route_strength'])
            channel += 1
        if p['gamma']:
            torch.mul(self.y, self.y, out=self.factor)
            self.factor.mul_(self.x).mul_(self.fields[:, channel * width:(channel + 1) * width])
            out.add_(self.factor, alpha=-3.0 * p['gamma'])
        torch.mul(self.route, kappa, out=self.temp)
        out.sub_(self.temp).mul_(p['mobility'])
        return out

    @staticmethod
    def wrap(theta):
        return theta.add_(math.pi).remainder_(2.0 * math.pi).sub_(math.pi)

    def advance(self, kappa, nextKappa):
        p = self.parameters
        self.rhs(self.theta, kappa, self.first)
        if p['integrator'] == 'heun':
            self.predictor.copy_(self.theta).add_(self.first, alpha=p['time_step'])
            self.wrap(self.predictor)
            self.rhs(self.predictor, nextKappa, self.second)
            self.first.add_(self.second)
            self.theta.add_(self.first, alpha=0.5 * p['time_step'])
        else:
            self.theta.add_(self.first, alpha=p['time_step'])
        self.wrap(self.theta)

    def capture(self, kappas):
        """Capture a fixed step block without advancing the visible trajectory."""
        torch = self.torch
        with _CAPTURE_LOCK, torch.cuda.device(self.theta.device):
            saved = self.theta.clone()
            current = torch.cuda.current_stream(self.theta.device)
            stream = torch.cuda.Stream(device=self.theta.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.advance(kappas[0], kappas[1])
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                for step in range(len(kappas) - 1):
                    self.advance(kappas[step], kappas[step + 1])
            current.wait_stream(stream)
            self.theta.copy_(saved)
        return graph


class TorchTransverseRouteBQMSolver(TorchExecution):
    """Search general QUBOs with normalized transverse-route geometric flow."""

    _workspaceClass = _FlowWorkspace

    @staticmethod
    def _getParameters(solverParameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        parameters = dict(
            steps=1000, runs=64, time_step=0.05, mobility=1.0,
            route_strength=1.0, gamma=0.0, kappa_initial=-1.0,
            kappa_final=2.0, schedule_exponent=1.0, integrator='euler',
            candidate_interval=25, candidate_batch_size=128,
            matrix_format='sparse', sparse_threshold=0.15,
            max_dense_variables=10000, max_variables=100000,
            dtype='float32', seed=0, run_batch_size=16,
            energy_chunk_size=8192, cuda_graph=True, graph_steps=25,
            deduplicate_candidates=True,
        )
        supplied = dict(solverParameters or {})
        # Translate upstream work-budget names at this solver boundary only.
        for alias, canonical in (('agents', 'runs'), ('max_steps', 'steps')):
            if alias in supplied:
                if canonical in supplied:
                    raise ValueError(f'Transverse route cannot define both {alias} and {canonical}')
                supplied[canonical] = supplied.pop(alias)
        unknown = supplied.keys() - parameters.keys()
        if unknown:
            raise ValueError(f'Unknown transverse-route parameters: {sorted(unknown)}')
        parameters.update(supplied)
        for name in ('steps', 'runs', 'candidate_interval', 'candidate_batch_size',
                     'energy_chunk_size', 'graph_steps', 'max_dense_variables', 'max_variables',
                     'seed', 'run_batch_size'):
            value = parameters[name]
            if name == 'run_batch_size' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f'Transverse-route {name} must be an integer')
            if value < (0 if name == 'seed' else 1):
                raise ValueError(f'Transverse-route {name} is out of range')
            parameters[name] = int(value)
        for name in ('time_step', 'mobility', 'route_strength', 'gamma',
                     'kappa_initial', 'kappa_final', 'schedule_exponent', 'sparse_threshold'):
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'Transverse-route {name} must be a real number')
            if not math.isfinite(value):
                raise ValueError(f'Transverse-route {name} must be finite')
            parameters[name] = float(value)
        for name in ('time_step', 'mobility', 'schedule_exponent'):
            if parameters[name] <= 0:
                raise ValueError(f'Transverse-route {name} must be positive')
        if parameters['route_strength'] < 0 or parameters['gamma'] < 0:
            raise ValueError('Transverse-route route_strength and gamma must be nonnegative')
        if parameters['kappa_final'] < parameters['kappa_initial']:
            raise ValueError('Transverse-route kappa_final must be at least kappa_initial')
        if not 0 <= parameters['sparse_threshold'] <= 1:
            raise ValueError('Transverse-route sparse_threshold must be between zero and one')
        for name, choices in (('dtype', ('float32', 'float64')),
                              ('integrator', ('euler', 'heun')),
                              ('matrix_format', ('auto', 'dense', 'sparse'))):
            if parameters[name] not in choices:
                raise ValueError(f'Transverse-route {name} must be one of {choices}')
        for name in ('cuda_graph', 'deduplicate_candidates'):
            if not isinstance(parameters[name], bool):
                raise TypeError(f'Transverse-route {name} must be a boolean')
        return parameters

    def estimatedWorkingMemoryBytes(self, problem, solverParameters=None):
        p = self._getParameters(solverParameters)
        n = problem.variableCount
        if n > p['max_variables']:
            raise ValueError('Transverse-route problem exceeds max_variables')
        if p['matrix_format'] == 'dense' and n > p['max_dense_variables']:
            raise ValueError('Transverse-route dense matrix exceeds max_dense_variables')
        width = min(p['run_batch_size'] or p['runs'], p['runs'])
        candidates = max(width, p['candidate_batch_size'])
        size = 4 if p['dtype'] == 'float32' else 8
        dense = n * n * size if p['matrix_format'] != 'sparse' and n <= p['max_dense_variables'] else 0
        return int(12 * problem.numericMemoryBytes + dense + n * width * size * 24
                   + n * candidates * 32
                   + min(problem.interactionCount, p['energy_chunk_size']) * candidates * 32
                   + (p['steps'] + 1) * size)

    @staticmethod
    def _toIsingProblem(problem, dtype, configuredC0):
        # The shared converter uses force=-gradient; this flow needs positive J,h.
        model = TorchExecution._toIsingProblem(problem, numpy.float64, 1.0)
        bounds = numpy.abs(model.forceField)
        numpy.add.at(bounds, model.heads, numpy.abs(model.couplings))
        numpy.add.at(bounds, model.tails, numpy.abs(model.couplings))
        scale = float(bounds.max(initial=0.0)) or 1.0
        if not math.isfinite(scale):
            raise ValueError('Transverse-route Ising normalization overflowed')
        return _IsingProblem(
            forceField=(-model.forceField / scale).astype(dtype),
            heads=model.heads, tails=model.tails,
            couplings=(-model.couplings / scale).astype(dtype), c0=1.0)

    def _solveBatch(self, problems, parameters):
        # Avoid an O((sum N)^2) allocation for explicitly dense scenario batches.
        if parameters['matrix_format'] == 'dense' and len(problems) > 1:
            solve_batch = super()._solveBatch
            return [result for problem in problems
                    for result in solve_batch([problem], parameters)]
        return super()._solveBatch(problems, parameters)

    def solveResidentPlanned(self, problems, plan, solverParameters=None):
        """Normalize the authoritative host snapshot, including duplicate edges."""
        if len(self.devices) > 1:
            raise ValueError('Resident transverse route requires one solver device')
        if plan.shards != ((0, len(problems)),) and problems:
            raise ValueError('Resident resource plan must cover the ordered batch')
        return self.solvePlanned([problem.source for problem in problems], plan, solverParameters)

    def _solvePrepared(self, problems, parameters, matrix, field, c0_rows, variableOffsets, candidates):
        torch, p = self._torch(), parameters
        n = len(field)
        dense = p['matrix_format'] == 'dense'
        if p['matrix_format'] == 'auto' and len(problems) == 1:
            dense = n <= p['max_dense_variables'] and matrix.values().numel() / (n * n) > p['sparse_threshold']
        if dense:
            if n > p['max_dense_variables']:
                raise ValueError('Transverse-route dense matrix exceeds max_dense_variables')
            matrix = matrix.to_dense()
        width_limit = min(p['run_batch_size'] or p['runs'], p['runs'])
        schedule = p['kappa_initial'] + (p['kappa_final'] - p['kappa_initial']) * (
            numpy.arange(p['steps'] + 1, dtype=numpy.float64) / p['steps']) ** p['schedule_exponent']
        kappas = torch.tensor(schedule, dtype=field.dtype, device=field.device)
        block_size = min(p['graph_steps'], p['candidate_interval'], p['steps'])
        use_graph = p['cuda_graph'] and field.device.type == 'cuda'
        workspaces = {}
        with torch.inference_mode():
            for run_start in range(0, p['runs'], width_limit):
                width = min(width_limit, p['runs'] - run_start)
                if width not in workspaces:
                    theta = field.new_zeros((n, width))
                    workspace = self._workspaceClass(torch, matrix, field, theta, p)
                    graph_kappas = kappas[:block_size + 1].clone()
                    graph = workspace.capture(graph_kappas) if use_graph else None
                    workspaces[width] = workspace, graph_kappas, graph
                workspace, graph_kappas, graph = workspaces[width]
                theta = workspace.theta
                for index, problem in enumerate(problems):
                    start, stop = map(int, variableOffsets[index:index + 2])
                    for run in range(width):
                        generator = torch.Generator(device=field.device)
                        generator.manual_seed((p['seed'] + problem.seedOffset
                            + (run_start + run) * _RUN_SEED_STRIDE) % _MAX_TORCH_SEED)
                        theta[start:stop, run].copy_(torch.rand(stop - start, dtype=field.dtype,
                            device=field.device, generator=generator)).mul_(2 * math.pi).sub_(math.pi)
                capacity = max(1, p['candidate_batch_size'] // width)
                checkpoints = torch.empty((capacity, n, width), dtype=torch.bool, device=field.device)
                retained = 0

                def flush():
                    nonlocal retained
                    for index in range(len(problems)):
                        start, stop = map(int, variableOffsets[index:index + 2])
                        samples = checkpoints[:retained, start:stop].permute(0, 2, 1).reshape(-1, stop - start)
                        if p['deduplicate_candidates'] and candidates[index].groups:
                            # Exact duplicate samples have identical deterministic
                            # repairs. Collapse them before expensive host descent.
                            samples = torch.unique(samples, dim=0)
                        candidates[index].add(samples)
                    retained = 0

                def collect():
                    nonlocal retained
                    torch.cos(theta, out=workspace.x)
                    torch.ge(workspace.x, 0, out=checkpoints[retained])
                    retained += 1
                    if retained == capacity:
                        flush()

                collect()
                completed = 0
                while completed < p['steps']:
                    target = min(completed + p['candidate_interval'], p['steps'])
                    while completed < target:
                        length = min(block_size, target - completed)
                        if graph is not None and length == block_size:
                            graph_kappas.copy_(kappas[completed:completed + block_size + 1])
                            graph.replay()
                        else:
                            for step in range(completed, completed + length):
                                workspace.advance(kappas[step], kappas[step + 1])
                        completed += length
                    collect()
                if not bool(torch.isfinite(theta).all()):
                    raise ValueError('Transverse-route dynamics became non-finite; reduce time_step or mobility')
                if retained:
                    flush()
        return [BQMOptimizationResult(*accumulator.result()) for accumulator in candidates]


BQMSolverFactory.registerSolver('torch_transverse_route', TorchTransverseRouteBQMSolver)
