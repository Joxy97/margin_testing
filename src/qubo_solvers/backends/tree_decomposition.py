"""Shared Torch bucket elimination for exact minimization and Gibbs sampling.

Graph ordering and admission are CPU setup. Factor tables, reductions and
backtracking stay on the requested CPU/CUDA device. Width and a conservative
memory estimate are checked before any exponential table allocation. Numerical
exactness is subject to the explicitly selected floating-point precision.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from .base import BQMSolver
from .problem import QUBOProblem
from .result import BQMOptimizationResult


MAX_TREEWIDTH = 25
DEFAULT_MEMORY_LIMIT_BYTES = 512 * 1024**2
_MAX_SEED = 2**63 - 1


class TreewidthExceeded(ValueError):
    """The problem/order exceeds the requested factor-table width limit."""


@dataclass(frozen=True)
class EliminationPlan:
    order: tuple[int, ...]
    separators: tuple[tuple[int, ...], ...]
    width: int
    minimum_degree: int
    estimated_memory_bytes: int
    linear: tuple[float, ...]
    pairs: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class TreeDecompositionSamples:
    """Device-resident samples and source-coefficient energies.

    ``minimum_energy`` belongs to min-sum elimination; ``log_partition`` belongs
    to log-sum-exp elimination. Offsets are included in both. ``solve`` applies
    the application's candidate selection/one-hot repair; ``sample_all`` leaves
    the unconstrained Boltzmann samples unchanged.
    """

    assignments: object
    energies: object
    plan: EliminationPlan
    minimum_energy: object = None
    log_partition: object = None


def _positive_integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _parameters(parameters=None):
    supplied = dict(parameters or {})
    for alias in ('num_reads', 'restarts'):
        if alias in supplied:
            if 'runs' in supplied:
                raise ValueError(f'Cannot specify both {alias} and runs')
            supplied['runs'] = supplied.pop(alias)
    defaults = dict(runs=1, seed=1, dtype='float64', max_treewidth=MAX_TREEWIDTH,
                    elimination_order=None, beta=1., memory_limit_bytes=DEFAULT_MEMORY_LIMIT_BYTES)
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f'Unknown tree-decomposition parameters: {sorted(unknown)}')
    defaults.update(supplied)
    defaults['runs'] = _positive_integer('runs', defaults['runs'])
    defaults['seed'] = _positive_integer('seed', defaults['seed'], 0)
    if defaults['seed'] >= _MAX_SEED:
        raise ValueError('seed must be below 2**63 - 1')
    defaults['max_treewidth'] = _positive_integer('max_treewidth', defaults['max_treewidth'], 0)
    if defaults['max_treewidth'] > MAX_TREEWIDTH:
        raise ValueError(f'max_treewidth cannot exceed {MAX_TREEWIDTH}')
    defaults['memory_limit_bytes'] = _positive_integer('memory_limit_bytes', defaults['memory_limit_bytes'])
    if defaults['dtype'] not in ('float32', 'float64'):
        raise ValueError('dtype must be float32 or float64')
    beta = defaults['beta']
    if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    defaults['beta'] = float(beta)
    if defaults['elimination_order'] is not None:
        order = tuple(defaults['elimination_order'])
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in order):
            raise ValueError('elimination_order must contain integer variable indices')
        defaults['elimination_order'] = tuple(map(int, order))
    return defaults


def _canonical_terms(problem):
    linear = problem.linear.copy()
    pairs = {}
    for i, j, value in zip(problem.quadraticHeads, problem.quadraticTails, problem.quadraticBiases):
        i, j, value = int(i), int(j), float(value)
        if i == j:
            linear[i] += value
        else:
            key = min(i, j), max(i, j)
            pairs[key] = pairs.get(key, 0.) + value
    if not np.isfinite(linear).all() or any(not math.isfinite(v) for v in pairs.values()):
        raise ValueError('Accumulated tree coefficients are nonfinite')
    return tuple(map(float, linear)), tuple((i, j, v) for (i, j), v in sorted(pairs.items()) if v != 0)


def elimination_preflight(problem, parameters=None, *, sampling=False):
    """Plan CPU topology and reject width/memory before initializing any GPU.

    A deterministic min-fill heuristic chooses an order when none is supplied.
    Width rejection after heuristic fill describes that order; it does not prove
    the graph's minimum possible treewidth. Initial minimum degree is a valid
    lower bound and cheaply rejects the dense published benchmark problems.
    """
    if not isinstance(problem, QUBOProblem):
        raise TypeError('problem must be QUBOProblem')
    p = _parameters(parameters)
    linear, pairs = _canonical_terms(problem)
    n = problem.variableCount
    adjacent = {v: set() for v in range(n)}
    for i, j, _ in pairs:
        adjacent[i].add(j)
        adjacent[j].add(i)
    minimum_degree = min(map(len, adjacent.values()))
    if minimum_degree > p['max_treewidth']:
        raise TreewidthExceeded(f'Treewidth is at least minimum degree {minimum_degree}, '
                                f'exceeding max_treewidth={p["max_treewidth"]}')
    supplied = p['elimination_order']
    if supplied is not None and (len(supplied) != n or set(supplied) != set(range(n))):
        raise ValueError('elimination_order must be a permutation of every variable')
    order, raw_separators = [], []
    while adjacent:
        if supplied is not None:
            variable = supplied[len(order)]
        else:
            def priority(v):
                neighbors = adjacent[v]
                missing = sum(len(neighbors - adjacent[u] - {u}) for u in neighbors) // 2
                return missing, len(neighbors), v
            variable = min(adjacent, key=priority)
        neighbors = adjacent[variable]
        if len(neighbors) > p['max_treewidth']:
            raise TreewidthExceeded(f'Elimination order reaches width {len(neighbors)}, '
                                    f'exceeding max_treewidth={p["max_treewidth"]}')
        raw_separators.append(tuple(neighbors))
        order.append(variable)
        for neighbor in neighbors:
            adjacent[neighbor].update(neighbors - {neighbor})
            adjacent[neighbor].discard(variable)
        del adjacent[variable]
    rank = {v: i for i, v in enumerate(order)}
    separators = tuple(tuple(sorted(scope, key=rank.__getitem__)) for scope in raw_separators)
    width = max(map(len, separators), default=0)
    scalar_bytes = 4 if p['dtype'] == 'float32' else 8
    factor_entries = sum(1 << len(scope) for scope in separators)
    largest_joint = 1 << (width + 1)
    # Conservatively retain all messages as well as all backtracking tables.
    # Five joint workspaces cover logsumexp temporaries; scoring is tiled to
    # 32 samples by 1024 original interactions and always uses float64.
    draw_count = p['runs'] if sampling else 1
    estimate = (
        factor_entries * (2 * scalar_bytes if sampling else scalar_bytes + 1)
        + 5 * largest_joint * scalar_bytes
        + p['runs'] * (n + 8) + draw_count * (18 * width + 32)
        + 32 * n * 8 + 3 * 32 * min(problem.interactionCount, 1024) * 8
        + 8 * problem.numericMemoryBytes + 32 * (n + len(pairs) + sum(map(len, separators)))
    )
    if estimate > p['memory_limit_bytes']:
        raise MemoryError(f'Estimated tree tensor/setup storage {estimate} bytes exceeds '
                          f'memory_limit_bytes={p["memory_limit_bytes"]}; reduce runs or width')
    return EliminationPlan(tuple(order), separators, width, minimum_degree, int(estimate), linear, pairs)


def _source_energies(torch, problem, assignments):
    """Tiled original float64 evaluation without an R-by-all-edges allocation."""
    device = assignments.device
    linear = torch.tensor(problem.linear.copy(), dtype=torch.float64, device=device)
    heads = torch.tensor(problem.quadraticHeads.astype(np.int64), device=device)
    tails = torch.tensor(problem.quadraticTails.astype(np.int64), device=device)
    biases = torch.tensor(problem.quadraticBiases.copy(), dtype=torch.float64, device=device)
    result = torch.full((len(assignments),), problem.offset, dtype=torch.float64, device=device)
    for start in range(0, len(assignments), 32):
        sample = assignments[start:start + 32].to(torch.float64)
        energy = result[start:start + 32]
        energy.add_(sample @ linear)
        for edge in range(0, len(biases), 1024):
            left = sample.index_select(1, heads[edge:edge + 1024])
            right = sample.index_select(1, tails[edge:edge + 1024])
            left.mul_(right).mul_(biases[edge:edge + 1024])
            energy.add_(left.sum(1))
    return result


def _eliminate(torch, problem, plan, parameters, device, sampling):
    dtype = getattr(torch, parameters['dtype'])
    n = problem.variableCount
    coefficients = torch.tensor((*plan.linear, *(v for _, _, v in plan.pairs)), dtype=dtype, device=device)
    if sampling:
        coefficients.mul_(-parameters['beta'])
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError('Tree coefficients overflow selected precision/beta')
    rank = {v: i for i, v in enumerate(plan.order)}
    original = [[] for _ in range(n)]
    for index, (i, j, _) in enumerate(plan.pairs):
        first, second = (i, j) if rank[i] < rank[j] else (j, i)
        original[first].append((second, n + index))
    buckets = [[] for _ in range(n)]
    decisions = []
    root_value = torch.zeros((), dtype=dtype, device=device)
    for variable, separator in zip(plan.order, plan.separators):
        scope = (variable, *separator)
        table = torch.zeros((2,) * len(scope), dtype=dtype, device=device)
        table[1].add_(coefficients[variable])
        positions = {v: axis for axis, v in enumerate(separator)}
        for neighbor, coefficient_index in original[variable]:
            selection = [slice(None)] * len(separator)
            selection[positions[neighbor]] = 1
            table[1][tuple(selection)].add_(coefficients[coefficient_index])
        incoming = buckets[variable]
        for message_scope, message in incoming:
            shape = tuple(2 if v in message_scope else 1 for v in scope)
            table.add_(message.reshape(shape))
        buckets[variable] = []
        if sampling:
            reduced = torch.logsumexp(table, dim=0)
            choice = (table[1] - reduced).exp_().clamp_(0., 1.)
        else:
            reduced = torch.minimum(table[0], table[1])
            choice = (table[1] < table[0]).to(torch.int8)
        decisions.append(choice.reshape(-1))
        if separator:
            buckets[separator[0]].append((separator, reduced))
        else:
            root_value.add_(reduced)
        del table, incoming
    if not bool(torch.isfinite(root_value)):
        raise FloatingPointError('Tree elimination overflowed; use float64 or rescale the objective')
    draw_count = parameters['runs'] if sampling else 1
    samples = torch.zeros((draw_count, n), dtype=torch.int8, device=device)
    flat_scopes = tuple(v for scope in plan.separators for v in scope)
    scope_tensor = torch.tensor(flat_scopes, dtype=torch.int64, device=device)
    powers = torch.tensor([1 << k for k in reversed(range(plan.width))], dtype=torch.int64, device=device)
    offsets = np.cumsum([0, *(len(scope) for scope in plan.separators)])
    generator = torch.Generator(device=device)
    generator.manual_seed((parameters['seed'] + problem.seedOffset) % _MAX_SEED)
    for position in reversed(range(n)):
        variable, separator = plan.order[position], plan.separators[position]
        if separator:
            columns = scope_tensor[int(offsets[position]):int(offsets[position + 1])]
            selected = samples.index_select(1, columns).to(torch.int64)
            indices = (selected * powers[-len(separator):]).sum(1)
        else:
            indices = torch.zeros(draw_count, dtype=torch.int64, device=device)
        selected = decisions[position].index_select(0, indices)
        if sampling:
            uniforms = torch.rand(draw_count, dtype=dtype, device=device, generator=generator)
            selected = uniforms < selected
        samples[:, variable].copy_(selected)
    if not sampling and parameters['runs'] > 1:
        samples = samples.expand(parameters['runs'], -1).clone()
    energies = _source_energies(torch, problem, samples)
    if not bool(torch.isfinite(energies).all()):
        raise FloatingPointError('Original source objective overflowed float64 scoring')
    if sampling:
        return TreeDecompositionSamples(samples, energies, plan,
                                        log_partition=root_value - parameters['beta'] * problem.offset)
    return TreeDecompositionSamples(samples, energies, plan,
                                    minimum_energy=root_value + problem.offset)


class _TreeDecompositionBase(BQMSolver):
    sampling = False

    def __init__(self, device='cpu'):
        if not isinstance(device, str) or (device != 'cpu' and device != 'cuda' and not (
                device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('device must be explicit cpu or cuda[:index]')
        self.requestedDevice = device

    @property
    def device(self):
        if self.requestedDevice == 'cpu':
            return 'cpu'
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; no CPU fallback')
        index = torch.cuda.current_device() if self.requestedDevice == 'cuda' else int(self.requestedDevice[5:])
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index {index} is unavailable')
        return f'cuda:{index}'

    @classmethod
    def _getParameters(cls, solverParameters=None):
        return _parameters(solverParameters)

    @classmethod
    def preflight(cls, problem, solverParameters=None):
        return elimination_preflight(problem, solverParameters, sampling=cls.sampling)

    def estimatedWorkingMemoryBytes(self, problem, solverParameters=None):
        return self.preflight(problem, solverParameters).estimated_memory_bytes

    def sample_all(self, problem, solverParameters=None):
        """Return all unmodified assignments and diagnostics on the selected device."""
        parameters = self._getParameters(solverParameters)
        plan = self.preflight(problem, parameters)
        import torch
        with torch.no_grad():
            return _eliminate(torch, problem, plan, parameters, torch.device(self.device), self.sampling)

    def solve(self, problem, solverParameters=None):
        result = self.sample_all(problem, solverParameters)
        samples = result.assignments.cpu().numpy()
        sample, energy = self._selectBestCandidates(((row, 0.) for row in samples), problem)
        return BQMOptimizationResult(sample=sample, energy=energy)

    def solvePlanned(self, problems, plan, solverParameters=None):
        if len(plan.shards) > 1:
            raise ValueError('Tree decomposition uses one explicitly selected device per solver')
        return self.solveMany(problems, solverParameters)


class TreeDecompositionBQMSolver(_TreeDecompositionBase):
    """Exact min-sum elimination, with deterministic state-zero tie breaking."""


class TreeDecompositionSamplerBQMSolver(_TreeDecompositionBase):
    """Exact conditional Gibbs sampling proportional to exp(-beta * energy)."""

    sampling = True
