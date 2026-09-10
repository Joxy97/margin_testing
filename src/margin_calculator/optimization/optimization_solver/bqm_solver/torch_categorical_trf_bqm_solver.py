"""Categorical transverse-route-inspired mirror flow on a product of simplexes.

This is a constrained variant, NOT the original binary angular TRF equations.
See docs/benchmarks/categorical_trf.md for its potential and discretization.
"""

from __future__ import annotations

import math

import numpy
from scipy.sparse import triu

from ...optimization_result import BQMOptimizationResult
from .bqm_solver_factory import BQMSolverFactory
from .torch_categorical_bqm_solver import _CategoricalModel
from .torch_execution import _IsingProblem, _MAX_TORCH_SEED, _RUN_SEED_STRIDE
from .torch_transverse_route_bqm_solver import TorchTransverseRouteBQMSolver, _FlowWorkspace


def _groups(problem):
    groups = tuple(tuple(group) for group in problem.iterOneHotGroups())
    if not groups or sum(map(len, groups)) != problem.variableCount:
        raise ValueError("Categorical TRF requires one-hot groups covering every variable")
    return groups


class _CategoricalFlowWorkspace(_FlowWorkspace):
    """Reuse TRF graph capture, with categorical rather than angular dynamics."""

    def __init__(self, torch, matrix, field, theta, parameters, groups):
        self.torch, self.matrix, self.field = torch, matrix, field
        self.theta, self.parameters = theta, parameters
        self.n, self.width = theta.shape
        self.groupCount = len(groups)
        lengths = tuple(map(len, groups))
        table = numpy.full((len(groups), max(lengths)), self.n, dtype=numpy.int64)
        owner = numpy.empty(self.n, dtype=numpy.int64)
        slots = numpy.empty(self.n, dtype=numpy.int64)
        for index, group in enumerate(groups):
            table[index, :len(group)] = group
            owner[list(group)] = index
            slots[list(group)] = index * table.shape[1] + numpy.arange(len(group))
        self.table = torch.tensor(table, device=theta.device)
        self.flatTable = self.table.reshape(-1)
        self.owner = torch.tensor(owner, device=theta.device)
        self.slots = torch.tensor(slots, device=theta.device)
        self.padded = theta.new_empty((self.n + 1, self.width))
        self.groupLogits = theta.new_empty((*table.shape, self.width))
        self.groupProbabilities = torch.empty_like(self.groupLogits)
        self.groupValues = theta.new_empty((len(groups), self.width))
        self.groupArgmax = torch.empty((len(groups), self.width), dtype=torch.int64, device=theta.device)
        self.choices = torch.empty_like(self.groupArgmax)
        self.p, self.route, self.factor, self.temp, self.meanRows = (
            torch.empty_like(theta) for _ in range(5))
        self.first, self.second, self.predictor = (torch.empty_like(theta) for _ in range(3))
        channels = 2 if parameters['route_strength'] else 1
        self.features = theta.new_empty((self.n, self.width * channels))
        self.fields = torch.empty_like(self.features)

    def _gather(self, values, padding):
        self.padded[:self.n].copy_(values)
        self.padded[self.n].fill_(padding)
        self.torch.index_select(self.padded, 0, self.flatTable,
                               out=self.groupLogits.reshape(-1, self.width))

    def probabilities(self, theta):
        self._gather(theta, -math.inf)
        self.torch.softmax(self.groupLogits, dim=1, out=self.groupProbabilities)
        self.torch.index_select(self.groupProbabilities.reshape(-1, self.width), 0,
                               self.slots, out=self.p)
        return self.p

    def rhs(self, theta, kappa, out):
        torch, parameters, width = self.torch, self.parameters, self.width
        self.probabilities(theta)
        self.features[:, :width].copy_(self.p)
        if parameters['route_strength']:
            torch.sub(1., self.p, out=self.route)
            self.route.mul_(self.p)
            self.features[:, width:].copy_(self.route)
        torch.mm(self.matrix, self.features, out=self.fields)
        torch.add(self.fields[:, :width], self.field, out=out)
        if parameters['route_strength']:
            torch.mul(self.p, -2., out=self.factor)
            self.factor.add_(1.).mul_(self.fields[:, width:])
            out.add_(self.factor, alpha=-parameters['route_strength'])
        torch.mul(self.p, kappa, out=self.temp)
        out.sub_(self.temp)
        # Natural-gradient / mirror flow: center the gradient in each simplex.
        torch.mul(out, self.p, out=self.temp)
        self._gather(self.temp, 0.)
        torch.sum(self.groupLogits, dim=1, out=self.groupValues)
        torch.index_select(self.groupValues, 0, self.owner, out=self.meanRows)
        return out.sub_(self.meanRows).mul_(-parameters['mobility'])

    def center(self, theta, updateChoices=False):
        self._gather(theta, -math.inf)
        self.torch.max(self.groupLogits, dim=1, out=(self.groupValues, self.groupArgmax))
        self.torch.index_select(self.groupValues, 0, self.owner, out=self.meanRows)
        # Group shifts leave softmax unchanged. A finite log-probability floor
        # limits numerical saturation; it is not a repair of binary candidates.
        theta.sub_(self.meanRows).clamp_(min=-80. if theta.dtype == self.torch.float32 else -600.)
        if updateChoices:
            self.torch.gather(self.table, 1, self.groupArgmax, out=self.choices)

    def advance(self, kappa, nextKappa):
        p = self.parameters
        self.rhs(self.theta, kappa, self.first)
        if p['integrator'] == 'heun':
            self.predictor.copy_(self.theta).add_(self.first, alpha=p['time_step'])
            self.center(self.predictor)
            self.rhs(self.predictor, nextKappa, self.second)
            self.first.add_(self.second)
            self.theta.add_(self.first, alpha=.5 * p['time_step'])
        else:
            self.theta.add_(self.first, alpha=p['time_step'])
        self.center(self.theta, updateChoices=True)

    def sample(self, out):
        """Materialize the authoritative category indices, without binary repair."""
        out.zero_().scatter_(0, self.choices, True)


class TorchCategoricalTRFBQMSolver(TorchTransverseRouteBQMSolver):
    """Feasible checkpoint samples from categorical transverse-route mirror flow."""

    _workspaceClass = _CategoricalFlowWorkspace

    @staticmethod
    def _getParameters(solverParameters=None):
        parameters = TorchTransverseRouteBQMSolver._getParameters(solverParameters)
        if parameters['gamma'] != 0:
            raise ValueError("Categorical TRF does not implement the binary gamma channel; use gamma=0")
        if parameters['seed'] >= _MAX_TORCH_SEED:
            raise ValueError("Categorical TRF seed must be in [0, 2**63 - 1)")
        return parameters

    def estimatedWorkingMemoryBytes(self, problem, solverParameters=None):
        p = self._getParameters(solverParameters)
        groups = _groups(problem)
        width = min(p['run_batch_size'] or p['runs'], p['runs'])
        item_size = 4 if p['dtype'] == 'float32' else 8
        padded = len(groups) * max(map(len, groups))
        # Source scoring, sparse preparation and trajectories are counted by TRF;
        # add padded group reductions, index maps, choices and graph scratch.
        return (super().estimatedWorkingMemoryBytes(problem, p)
                + padded * (4 * width * item_size + 16)
                + problem.variableCount * (4 * width * item_size + 32))

    @staticmethod
    def _toIsingProblem(problem, dtype, configuredC0):
        """Reuse the packing container for categorical gradients, NOT Ising data."""
        _groups(problem)
        model = _CategoricalModel.fromProblem(problem)
        bounds = numpy.abs(model.linear) + numpy.asarray(abs(model.adjacency).sum(axis=1)).reshape(-1)
        scale = float(bounds.max(initial=0.)) or 1.
        if not math.isfinite(scale):
            raise ValueError("Categorical TRF coefficient normalization overflowed")
        upper = triu(model.adjacency, k=1, format='coo')
        return _IsingProblem(
            forceField=(model.linear / scale).astype(dtype),
            heads=upper.row.astype(numpy.int64), tails=upper.col.astype(numpy.int64),
            couplings=(upper.data / scale).astype(dtype), c0=1.)

    def _solvePrepared(self, problems, parameters, matrix, field, c0_rows, variableOffsets, candidates):
        torch, p = self._torch(), parameters
        n = len(field)
        groups = tuple(tuple(int(variableOffsets[index]) + v for v in group)
                       for index, problem in enumerate(problems) for group in _groups(problem))
        dense = p['matrix_format'] == 'dense'
        if p['matrix_format'] == 'auto' and len(problems) == 1:
            dense = n <= p['max_dense_variables'] and matrix.values().numel() / (n * n) > p['sparse_threshold']
        if dense:
            if n > p['max_dense_variables']:
                raise ValueError("Categorical TRF dense matrix exceeds max_dense_variables")
            matrix = matrix.to_dense()
        width_limit = min(p['run_batch_size'] or p['runs'], p['runs'])
        schedule = p['kappa_initial'] + (p['kappa_final'] - p['kappa_initial']) * (
            numpy.arange(p['steps'] + 1, dtype=numpy.float64) / p['steps']) ** p['schedule_exponent']
        kappas = torch.tensor(schedule, dtype=field.dtype, device=field.device)
        block = min(p['graph_steps'], p['candidate_interval'], p['steps'])
        use_graph = p['cuda_graph'] and field.device.type == 'cuda'
        workspaces = {}
        with torch.inference_mode():
            for run_start in range(0, p['runs'], width_limit):
                width = min(width_limit, p['runs'] - run_start)
                if width not in workspaces:
                    workspace = self._workspaceClass(torch, matrix, field,
                        field.new_zeros((n, width)), p, groups)
                    graph_kappas = kappas[:block + 1].clone()
                    graph = workspace.capture(graph_kappas) if use_graph else None
                    workspaces[width] = workspace, graph_kappas, graph
                workspace, graph_kappas, graph = workspaces[width]
                for index, problem in enumerate(problems):
                    start, stop = map(int, variableOffsets[index:index + 2])
                    for run in range(width):
                        generator = torch.Generator(device=field.device)
                        generator.manual_seed((p['seed'] + problem.seedOffset
                            + (run_start + run) * _RUN_SEED_STRIDE) % _MAX_TORCH_SEED)
                        workspace.theta[start:stop, run].uniform_(-.1, .1, generator=generator)
                workspace.center(workspace.theta, updateChoices=True)
                capacity = max(1, p['candidate_batch_size'] // width)
                checkpoints = torch.empty((capacity, n, width), dtype=torch.bool, device=field.device)
                retained = 0

                def flush():
                    nonlocal retained
                    for index in range(len(problems)):
                        start, stop = map(int, variableOffsets[index:index + 2])
                        samples = checkpoints[:retained, start:stop].permute(0, 2, 1).reshape(-1, stop - start)
                        if p['deduplicate_candidates']:
                            samples = torch.unique(samples, dim=0)
                        candidates[index].add(samples)
                        if not candidates[index].feasible:
                            raise RuntimeError("Categorical TRF emitted an infeasible candidate")
                    retained = 0

                def collect():
                    nonlocal retained
                    if not bool(torch.isfinite(workspace.theta).all()):
                        raise ValueError("Categorical TRF became non-finite; reduce time_step or mobility")
                    workspace.sample(checkpoints[retained])
                    retained += 1
                    if retained == capacity:
                        flush()

                collect()
                completed = 0
                while completed < p['steps']:
                    target = min(completed + p['candidate_interval'], p['steps'])
                    while completed < target:
                        length = min(block, target - completed)
                        if graph is not None and length == block:
                            graph_kappas.copy_(kappas[completed:completed + block + 1])
                            graph.replay()
                        else:
                            for step in range(completed, completed + length):
                                workspace.advance(kappas[step], kappas[step + 1])
                        completed += length
                    collect()
                if retained:
                    flush()
        return [BQMOptimizationResult(*accumulator.result()) for accumulator in candidates]


BQMSolverFactory.registerSolver('torch_categorical_trf', TorchCategoricalTRFBQMSolver)
