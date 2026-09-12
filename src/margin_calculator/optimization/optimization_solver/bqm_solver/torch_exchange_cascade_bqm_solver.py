"""Float32 Gaussian exchange-field cascade with compressed node memories.

See docs/benchmarks/exchange_field_cascade.md for equations, discretization,
terminal-frame policy, and the distinction between exactness and convergence.
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
from .torch_candidates import TorchCandidateAccumulator
from .torch_execution import TorchExecution, _MAX_TORCH_SEED, _RUN_SEED_STRIDE

_CAPTURE_LOCK = Lock()


def _fieldPropagators(torch, a, b, k, xi, duration):
    """Evaluate exp(-duration K) and K^-1(I-exp(-duration K)) on device.

    Diagonalize the two-column Gram matrix analytically. This avoids a CUDA/CPU
    synchronization for a batched 3x3 eigh at every integration step. Vanishing
    Gram modes contribute zero; expm1 avoids cancellation for small timesteps.
    """
    u, v = math.sqrt(xi) * a, math.sqrt(xi) * b
    aa, bb, ab = (u * u).sum(-1), (v * v).sum(-1), (u * v).sum(-1)
    angle = .5 * torch.atan2(2 * ab, aa - bb)
    cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
    modes = torch.stack((u * cosine + v * sine, -u * sine + v * cosine), dim=1)
    masses = (modes * modes).sum(-1)
    # A zero mode has a zero projector, including exactly zero/parallel seeds.
    projectors = modes[..., :, None] * modes[..., None, :]
    projectors = projectors / masses.clamp_min(torch.finfo(a.dtype).tiny)[..., None, None]
    eigenvalues = k + masses
    decay = torch.exp(-duration * eigenvalues)
    response = -torch.expm1(-duration * eigenvalues) / eigenvalues
    r0, q0 = math.exp(-duration * k), -math.expm1(-duration * k) / k
    identity = torch.eye(3, dtype=a.dtype, device=a.device)
    r = r0 * identity + ((decay - r0)[..., None, None] * projectors).sum(1)
    q = q0 * identity + ((response - q0)[..., None, None] * projectors).sum(1)
    return r, q, masses


class _ExchangeWorkspace:
    """One problem's shared interaction matrix and bounded trajectory state."""

    def __init__(self, torch, matrix, degree, spins, parameters):
        self.torch, self.matrix, self.degree, self.p = torch, matrix, degree, parameters
        self.x = spins  # (nodes, trajectories, Cartesian components)
        self.z = torch.zeros_like(spins)
        self.force = torch.empty_like(spins)
        width = spins.shape[1]
        self.a = spins.new_zeros((width, 3))
        self.b = spins.new_zeros((width, 3))
        self.a[:, 0] = parameters['order_seed']
        self.b[:, 1] = parameters['order_seed']
        self.orderScale = parameters['order_strength'] * max(float(degree.sum().item()) / 2, 1.)
        self.terminalPropagators = None
        self.preFreezeMasses = None
        self.frameFallback = None

    def feedback(self):
        torch = self.torch
        torch.mm(self.matrix, self.z.reshape(len(self.z), -1),
                 out=self.force.view(len(self.z), -1))
        self.force.neg_().add_(self.degree[:, None, None] * self.z).mul_(self.p['k'])
        return self.force

    def advance(self, rhoA, rhoB, terminal=False):
        torch, p = self.torch, self.p
        if terminal:
            r, q = self.terminalPropagators
        else:
            r, q, _ = _fieldPropagators(torch, self.a, self.b, p['k'], p['xi'],
                                      p['time_step'] / p['field_damping'])
        # Batched right multiplication in each trajectory, shared J via GEMM/SpMM.
        z, x = self.z.permute(1, 0, 2), self.x.permute(1, 0, 2)
        self.z.copy_((torch.bmm(z, r) + torch.bmm(x, q)).permute(1, 0, 2))
        force = self.feedback()
        tangent = force - self.x * (self.x * force).sum(-1, keepdim=True)
        delta = p['time_step'] * p['spin_mobility'] * tangent
        delta = delta / ((delta * delta).sum(-1, keepdim=True).sqrt()
                         / p['max_spin_step']).clamp_min(1.)
        self.x.add_(delta)
        self.x.div_(self.x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(self.x.dtype).tiny))
        if not terminal:
            covariance = torch.bmm(self.z.permute(1, 2, 0), force.permute(1, 0, 2))
            aa = (self.a * self.a).sum(-1, keepdim=True)
            bb = (self.b * self.b).sum(-1, keepdim=True)
            ab = (self.a * self.b).sum(-1, keepdim=True)
            ca = torch.bmm(covariance, self.a[:, :, None]).squeeze(-1)
            cb = torch.bmm(covariance, self.b[:, :, None]).squeeze(-1)
            strength = self.orderScale
            gradA = strength * ((aa - rhoA) * self.a + p['orthogonality'] * ab * self.b) + p['xi'] * ca
            gradB = strength * ((bb - rhoB) * self.b + p['orthogonality'] * ab * self.a) + p['xi'] * cb
            # Conservative local curvature scale for explicit order updates.
            bound = strength * (3 * (aa + bb) + rhoA.abs() + rhoB.abs()
                                 + p['orthogonality'] * (aa + bb))
            bound += p['xi'] * covariance.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).clamp_min(0)
            step = p['time_step'] * p['order_mobility'] / bound.clamp_min(1.)
            self.a.add_(-step * gradA)
            self.b.add_(-step * gradB)

    def freeze(self):
        """Select an independent terminal frame; enforce both masses > k."""
        torch, p = self.torch, self.p
        _, _, self.preFreezeMasses = _fieldPropagators(
            torch, self.a, self.b, p['k'], p['xi'], p['time_step'] / p['field_damping'])
        tiny = 1e-6
        aNorm = self.a.norm(dim=-1, keepdim=True)
        defaultA = torch.zeros_like(self.a)
        defaultA[:, 0] = 1
        axisA = torch.where(aNorm > tiny, self.a / aNorm.clamp_min(tiny), defaultA)
        perpendicular = self.b - (self.b * axisA).sum(-1, keepdim=True) * axisA
        bNorm = perpendicular.norm(dim=-1, keepdim=True)
        # Canonical least-aligned Cartesian axis, with lowest-index tie break.
        basis = torch.nn.functional.one_hot(axisA.abs().argmin(-1), 3).to(self.a.dtype)
        fallbackB = basis - (basis * axisA).sum(-1, keepdim=True) * axisA
        fallbackB /= fallbackB.norm(dim=-1, keepdim=True)
        axisB = torch.where(bNorm > tiny, perpendicular / bNorm.clamp_min(tiny), fallbackB)
        self.frameFallback = ((aNorm <= tiny) | (bNorm <= tiny)).squeeze(-1)
        radius = math.sqrt(p['terminal_mass'] / p['xi'])
        self.a.copy_(radius * axisA)
        self.b.copy_(radius * axisB)
        r, q, _ = _fieldPropagators(torch, self.a, self.b, p['k'], p['xi'],
                                  p['time_step'] / p['field_damping'])
        self.terminalPropagators = r, q

    def decoded(self):
        # Reference-spin gauge; dot == 0 selects binary 1. Isolated bits select 0.
        return (((self.x[1:] * self.x[:1]).sum(-1) >= 0)
                & (self.degree[1:, None] != 0)).T.contiguous()

    def capture(self, controls, terminal):
        """Capture a reusable step block, restoring all visible trajectory state."""
        torch = self.torch
        with _CAPTURE_LOCK, torch.cuda.device(self.x.device):
            saved = [value.clone() for value in (self.x, self.z, self.a, self.b)]
            current = torch.cuda.current_stream(self.x.device)
            stream = torch.cuda.Stream(device=self.x.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.advance(controls[0, 0], controls[0, 1], terminal)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                for row in range(len(controls)):
                    self.advance(controls[row, 0], controls[row, 1], terminal)
            current.wait_stream(stream)
            for value, original in zip((self.x, self.z, self.a, self.b), saved):
                value.copy_(original)
        return graph


class TorchExchangeCascadeBQMSolver(TorchExecution):
    """General QUBO exchange cascade; independent problem shards on multiple GPUs."""

    @staticmethod
    def _getParameters(solverParameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        p = dict(steps=1000, runs=64, run_batch_size=16, seed=0, dtype='float32',
                 time_step=.1, spin_mobility=1., order_mobility=1., max_spin_step=.2,
                 k=1., xi=4., field_damping=1., order_strength=4., orthogonality=1.,
                 order_seed=.05, first_stage=.2, second_stage=.45, terminal_stage=.75,
                 terminal_mass=2., candidate_interval=25, candidate_batch_size=128,
                 local_search_steps=16, matrix_format='auto', sparse_threshold=.15,
                 max_dense_variables=10001, max_variables=100000,
                 energy_chunk_size=8192, cuda_graph=True, graph_steps=10)
        supplied = dict(solverParameters or {})
        if supplied.keys() - p.keys():
            raise ValueError(f'Unknown exchange-cascade parameters: {sorted(supplied.keys() - p.keys())}')
        p.update(supplied)
        for name in ('steps', 'runs', 'run_batch_size', 'seed', 'candidate_interval',
                     'candidate_batch_size', 'local_search_steps', 'max_dense_variables',
                     'max_variables', 'energy_chunk_size', 'graph_steps'):
            value = p[name]
            if name == 'run_batch_size' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f'Exchange-cascade {name} must be an integer')
            if value < (0 if name in ('seed', 'local_search_steps') else 1):
                raise ValueError(f'Exchange-cascade {name} is out of range')
            p[name] = int(value)
        for name in ('time_step', 'spin_mobility', 'order_mobility', 'max_spin_step',
                     'k', 'xi', 'field_damping', 'order_strength', 'orthogonality',
                     'order_seed', 'first_stage', 'second_stage', 'terminal_stage',
                     'terminal_mass', 'sparse_threshold'):
            value = p[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'Exchange-cascade {name} must be a real number')
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Exchange-cascade {name} must be finite and nonnegative')
            if name not in ('orthogonality', 'first_stage', 'sparse_threshold') and value == 0:
                raise ValueError(f'Exchange-cascade {name} must be positive')
            p[name] = float(value)
        if not 0 <= p['first_stage'] < p['second_stage'] < p['terminal_stage'] < 1:
            raise ValueError('Exchange-cascade stages must satisfy 0 <= first < second < terminal < 1')
        if p['terminal_mass'] <= p['k'] * (1 + 16 * numpy.finfo(numpy.float32).eps):
            raise ValueError('Exchange-cascade terminal_mass must exceed k by a float32 margin')
        if p['time_step'] * p['order_mobility'] > .25 or p['max_spin_step'] > .5:
            raise ValueError('Exchange-cascade requires time_step * order_mobility <= .25 and max_spin_step <= .5')
        if not 0 <= p['sparse_threshold'] <= 1:
            raise ValueError('Exchange-cascade sparse_threshold must be between zero and one')
        if p['dtype'] != 'float32':
            raise ValueError('Exchange-cascade dynamics require dtype: float32')
        if p['matrix_format'] not in ('auto', 'dense', 'sparse'):
            raise ValueError('Exchange-cascade matrix_format must be auto, dense or sparse')
        if not isinstance(p['cuda_graph'], bool):
            raise TypeError('Exchange-cascade cuda_graph must be a boolean')
        return p

    def estimatedWorkingMemoryBytes(self, problem, solverParameters=None):
        p = self._getParameters(solverParameters)
        n = problem.variableCount + 1
        if problem.variableCount > p['max_variables']:
            raise ValueError('Exchange-cascade problem exceeds max_variables')
        if p['matrix_format'] == 'dense' and n > p['max_dense_variables']:
            raise ValueError('Exchange-cascade dense matrix exceeds max_dense_variables (including reference)')
        width = min(p['runs'], p['run_batch_size'] or p['runs'])
        candidates = max(width, p['candidate_batch_size'])
        dense = p['matrix_format'] != 'sparse' and n <= p['max_dense_variables']
        # Captured intermediates and per-stream library workspaces outlive a
        # single step. Reserve both phases, including a fixed small-shape floor
        # from the RTX 3060 measurements (see the benchmark notes).
        graphReserve = 0
        if p['cuda_graph']:
            graphReserve = (256 * 1024**2 + 4 * n * width * 96
                            * min(p['graph_steps'], p['candidate_interval'], p['steps']))
        # CSR construction + source float64 scoring + graph pools + two widths.
        return int(12 * problem.numericMemoryBytes + (4 * n * n if dense else 0)
                   + 4 * n * width * 160 + n * candidates * 32
                   + min(problem.interactionCount, p['energy_chunk_size']) * candidates * 32
                   + graphReserve)

    @staticmethod
    def _homogenize(problem):
        """Canonical positive-sign Ising edges, normalized including reference degree."""
        model = TorchExecution._toIsingProblem(problem, numpy.float64, 1.)
        fields = numpy.flatnonzero(model.forceField)
        heads = numpy.concatenate((model.heads + 1, numpy.zeros(len(fields), dtype=numpy.int64)))
        tails = numpy.concatenate((model.tails + 1, fields + 1))
        values = -numpy.concatenate((model.couplings, model.forceField[fields]))
        degree = numpy.zeros(problem.variableCount + 1)
        numpy.add.at(degree, heads, numpy.abs(values))
        numpy.add.at(degree, tails, numpy.abs(values))
        scale = float(degree.max(initial=0.)) or 1.
        if not math.isfinite(scale):
            raise ValueError('Exchange-cascade Ising normalization overflowed')
        return heads, tails, (values / scale).astype(numpy.float32), (degree / scale).astype(numpy.float32), scale

    def _solveBatch(self, problems, parameters):
        self.lastDiagnostics = []
        results = []
        torch = self._torch()
        with torch.inference_mode():
            for problem in problems:
                result, diagnostics = self._solveProblem(problem, parameters)
                results.append(result)
                self.lastDiagnostics.append(diagnostics)
        return results

    def _mergeWorkerState(self, workers):
        self.lastDiagnostics = [item for worker in workers for item in worker.lastDiagnostics]

    def solveResidentPlanned(self, problems, plan, solverParameters=None):
        """Use authoritative host QUBOs through the existing resident fallback seam."""
        if len(self.devices) > 1:
            raise ValueError('Resident exchange cascade requires one solver device')
        if problems and plan.shards != ((0, len(problems)),):
            raise ValueError('Resident resource plan must cover the ordered batch')
        return self.solvePlanned([problem.source for problem in problems], plan, solverParameters)

    def _solveProblem(self, problem, p):
        torch, device = self._torch(), self.device
        heads, tails, values, degreeHost, scale = self._homogenize(problem)
        n = problem.variableCount + 1
        if not len(values):
            # Constant objectives need no dynamics (and no empty CSR allocation).
            # Preserve declared one-hot constraints through the shared policy.
            sample, energy = self._selectBestCandidates([(numpy.zeros(n - 1, dtype=numpy.uint8), 0.)], problem)
            return BQMOptimizationResult(sample, energy), dict(device=device, dtype='float32',
                normalizationScale=scale, matrixFormat='none', constantObjective=True, trajectories=[])
        dense = p['matrix_format'] == 'dense' or (p['matrix_format'] == 'auto'
            and n <= p['max_dense_variables'] and 2 * len(values) / (n * n) > p['sparse_threshold'])
        if dense:
            # Canonicalization already guarantees unique off-diagonal edges.
            # Fill directly instead of staging and sorting a dense-sized COO.
            matrix = torch.zeros((n, n), dtype=torch.float32, device=device)
            h, t, v = (torch.tensor(array, device=device) for array in (heads, tails, values))
            matrix[h, t] = v
            matrix[t, h] = v
            del h, t, v
        else:
            rows, cols = numpy.concatenate((heads, tails)), numpy.concatenate((tails, heads))
            order = numpy.lexsort((cols, rows))
            offsets = numpy.r_[0, numpy.cumsum(numpy.bincount(rows, minlength=n))]
            matrix = torch.sparse_csr_tensor(torch.tensor(offsets, device=device),
                torch.tensor(cols[order], device=device),
                torch.tensor(numpy.concatenate((values, values))[order], device=device),
                size=(n, n), check_invariants=True)
        degree = torch.tensor(degreeHost, device=device)
        accumulator = TorchCandidateAccumulator(torch, problem, device, p['energy_chunk_size'], self._selectBestCandidates)
        widthLimit = min(p['runs'], p['run_batch_size'] or p['runs'])
        freezeStep = min(p['steps'] - 1, int(p['terminal_stage'] * p['steps']))
        progress = numpy.arange(p['steps']) / p['steps']
        # Piecewise-linear controls: negative exploration, then staggered ramps.
        rhoA = numpy.clip(2 * (progress - p['first_stage']) / (p['second_stage'] - p['first_stage']) - 1, -1, 1)
        rhoB = numpy.clip(2 * (progress - p['second_stage']) / (p['terminal_stage'] - p['second_stage']) - 1, -1, 1)
        controls = torch.tensor(numpy.stack((rhoA, rhoB), axis=1), dtype=torch.float32, device=device)
        diagnostics = dict(device=device, dtype='float32', normalizationScale=scale,
                           matrixFormat='dense' if dense else 'sparse', trajectories=[])
        for start in range(0, p['runs'], widthLimit):
            width = min(widthLimit, p['runs'] - start)
            # Host per-run seeds give identical initial states across CPU/GPU and shards.
            initial = numpy.empty((n, width, 3), dtype=numpy.float32)
            for run in range(width):
                seed = (p['seed'] + problem.seedOffset + (start + run) * _RUN_SEED_STRIDE) % _MAX_TORCH_SEED
                rng = numpy.random.default_rng(seed)
                initial[:, run] = rng.standard_normal((n, 3)).astype(numpy.float32)
            initial /= numpy.linalg.norm(initial, axis=-1, keepdims=True)
            workspace = _ExchangeWorkspace(torch, matrix, degree, torch.tensor(initial, device=device), p)
            capacity = max(1, p['candidate_batch_size'] // width)
            checkpoints = torch.empty((capacity, width, n - 1), dtype=torch.bool, device=device)
            retained = 0

            def collect():
                nonlocal retained
                if not bool(torch.isfinite(workspace.x).all() & torch.isfinite(workspace.z).all()
                            & torch.isfinite(workspace.a).all() & torch.isfinite(workspace.b).all()):
                    raise FloatingPointError('Exchange-cascade dynamics became nonfinite; reduce time_step or mobilities')
                checkpoints[retained].copy_(workspace.decoded())
                retained += 1
                if retained == capacity:
                    accumulator.add(checkpoints.reshape(-1, n - 1))
                    retained = 0

            collect()
            block = min(p['graph_steps'], p['candidate_interval'])
            graphControls = controls[:block].clone()
            graphs = {}
            completed = 0
            while completed < p['steps']:
                if completed == freezeStep:
                    workspace.freeze()
                terminal = completed >= freezeStep
                target = min(((completed // p['candidate_interval']) + 1) * p['candidate_interval'],
                             p['steps'], p['steps'] if terminal else freezeStep)
                while completed < target:
                    length = min(block, target - completed)
                    if p['cuda_graph'] and degree.device.type == 'cuda' and length == block:
                        graphControls.copy_(controls[completed:completed + block])
                        if terminal not in graphs:
                            graphs[terminal] = workspace.capture(graphControls, terminal)
                        graphs[terminal].replay()
                    else:
                        for step in range(completed, completed + length):
                            workspace.advance(controls[step, 0], controls[step, 1], terminal)
                    completed += length
                collect()
            if retained:
                accumulator.add(checkpoints[:retained].reshape(-1, n - 1))
            if p['local_search_steps']:
                # Score raw checkpoints first so float32 polishing cannot lose a winner.
                accumulator.add(self._polish(torch, matrix, workspace.decoded(), p['local_search_steps']))
            for masses, fallback in zip(workspace.preFreezeMasses.cpu().tolist(), workspace.frameFallback.cpu().tolist()):
                diagnostics['trajectories'].append(dict(preFreezeMassMin=min(masses),
                    terminalMass=p['terminal_mass'], frameFallback=fallback))
        if p['local_search_steps']:
            best = torch.tensor([accumulator.result()[0]], dtype=torch.bool, device=device)
            accumulator.add(self._polish(torch, matrix, best, p['local_search_steps']))
        sample, energy = accumulator.result()
        return BQMOptimizationResult(sample, energy), diagnostics

    @staticmethod
    def _polish(torch, matrix, samples, steps):
        """Fixed-budget best improving logical-bit flips; reference spin stays +1."""
        spins = torch.cat((torch.ones((1, len(samples)), dtype=torch.float32, device=samples.device),
                           samples.T.to(torch.float32) * 2 - 1), dim=0)
        columns = torch.arange(len(samples), device=samples.device)
        for _ in range(steps):
            delta = -2 * spins * torch.mm(matrix, spins)
            delta[0] = float('inf')
            best, indices = delta.min(dim=0)
            spins[indices, columns] *= torch.where(best < 0, -1., 1.)
        return (spins[1:].T > 0).contiguous()


BQMSolverFactory.registerSolver('torch_exchange_cascade', TorchExchangeCascadeBQMSolver)
