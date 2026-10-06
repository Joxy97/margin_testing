"""Shared execution for the five Hamiltonian QUBO reference solvers.

Dynamics use one globally normalized, zero-diagonal Ising operator. Source
QUBO scoring and one-hot repair remain owned by TorchCandidateAccumulator.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy
from .. import observation

from .result import BQMOptimizationResult
from .torch_execution import TorchExecution, _MAX_TORCH_SEED, _RUN_SEED_STRIDE
from .torch_candidates import TorchCandidateAccumulator


class PhysicsModel:
    """Positive-sign Ising coefficients, with optional isolated padding."""

    def __init__(self, solver, problem, p, size):
        self.torch = torch = solver._torch()
        self.n = problem.variableCount
        self.size = size
        model = solver._toIsingProblem(problem, numpy.float64, 1.)
        h, values = -model.forceField, -model.couplings
        degree = numpy.abs(h)
        numpy.add.at(degree, model.heads, numpy.abs(values))
        numpy.add.at(degree, model.tails, numpy.abs(values))
        self.scale = float(degree.max(initial=0.)) or 1.
        if not numpy.isfinite(self.scale):
            raise ValueError('Ising coefficient normalization overflowed')
        self.heads, self.tails = model.heads, model.tails
        self.values = values / self.scale
        self.hostField = h / self.scale
        dtype = getattr(torch, p['dtype'])
        self.h = torch.zeros((size, 1), device=solver.device, dtype=dtype)
        self.h[:self.n, 0] = torch.tensor(self.hostField, device=solver.device, dtype=dtype)
        self.degree = torch.zeros((size, 1), device=solver.device, dtype=dtype)
        self.degree[:self.n, 0] = torch.tensor(degree / self.scale, device=solver.device, dtype=dtype)
        dense = p['matrix_format'] == 'dense' or (p['matrix_format'] == 'auto'
            and size <= p['max_dense_variables']
            and 2 * len(values) / (size * size) > p['sparse_threshold'])
        self.format = 'dense' if dense else 'sparse'
        self.matrix = self.makeMatrix(self.values, dense)
        from ..observation import prepared
        prepared(self.matrix, original_variables=self.n, internal_coordinates=self.size,
                 matrix_format_policy=p['matrix_format'], sparse_threshold=p['sparse_threshold'])

    def makeMatrix(self, values, dense=None):
        torch = self.torch
        dense = self.format == 'dense' if dense is None else dense
        h, t = self.heads, self.tails
        if dense:
            matrix = torch.zeros((self.size, self.size), dtype=self.h.dtype, device=self.h.device)
            heads = torch.tensor(h, device=self.h.device)
            tails = torch.tensor(t, device=self.h.device)
            biases = torch.tensor(values, dtype=self.h.dtype, device=self.h.device)
            matrix[heads, tails] = biases
            matrix[tails, heads] = biases
            return matrix
        nonzero = values != 0
        rows = numpy.concatenate((h[nonzero], t[nonzero]))
        cols = numpy.concatenate((t[nonzero], h[nonzero]))
        order = numpy.lexsort((cols, rows))
        offsets = numpy.r_[0, numpy.cumsum(numpy.bincount(rows, minlength=self.size))]
        # Empty NumPy gathers can carry stride zero; CSR's invariant checker
        # requires unit-stride index/value storage even when there are no edges.
        columns = (torch.tensor(cols[order], device=self.h.device) if len(order)
                   else torch.empty(0, dtype=torch.int64, device=self.h.device))
        biases = (torch.tensor(numpy.concatenate((values[nonzero], values[nonzero]))[order],
                              dtype=self.h.dtype, device=self.h.device) if len(order)
                  else torch.empty(0, dtype=self.h.dtype, device=self.h.device))
        return torch.sparse_csr_tensor(
            torch.tensor(offsets, device=self.h.device),
            columns, biases,
            size=(self.size, self.size), check_invariants=True)

    def product(self, x):
        return self.torch.mm(self.matrix, x)

    def energy(self, x, product):
        # FP64 accumulation, including when the propagation operator is FP32.
        return (x.double() * (.5 * product.double() + self.h.double())).sum(0)


class TorchPhysicsSolver(TorchExecution):
    """Trajectory batching, stable starts, bounded candidates and CPU fallback."""

    defaults = {}
    residentMode = "source_snapshot"

    @classmethod
    def _getParameters(cls, solverParameters=None):
        p = dict(steps=500, runs=32, run_batch_size=16, seed=0, dtype='float32',
                 time_step=.1, candidate_interval=25, energy_chunk_size=8192,
                 local_search_steps=32, conditional_rounding=True,
                 matrix_format='auto', sparse_threshold=.15,
                 max_dense_variables=10000, max_variables=100000)
        p.update(cls.defaults)
        supplied = dict(solverParameters or {})
        unknown = supplied.keys() - p.keys()
        if unknown:
            raise ValueError(f'Unknown {cls.__name__} parameters: {sorted(unknown)}')
        baseline = p.copy()
        p.update(supplied)
        zeroAllowed = {'seed', 'local_search_steps', 'block_sweeps', 'proof_nodes',
                       'bound_iterations'}
        for name, default in baseline.items():
            value = p[name]
            if name == 'run_batch_size' and value is None:
                continue
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise TypeError(f'{name} must be a boolean')
            elif isinstance(default, int):
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise TypeError(f'{name} must be an integer')
                if value < (0 if name in zeroAllowed else 1):
                    raise ValueError(f'{name} is out of range')
                p[name] = int(value)
            elif isinstance(default, float):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(f'{name} must be a real number')
                if not math.isfinite(value):
                    raise ValueError(f'{name} must be finite')
                p[name] = float(value)
        if p['time_step'] <= 0:
            raise ValueError('time_step must be positive')
        if p['dtype'] not in ('float32', 'float64'):
            raise ValueError('dtype must be float32 or float64')
        if p['matrix_format'] not in ('auto', 'dense', 'sparse'):
            raise ValueError('matrix_format must be auto, dense or sparse')
        if not 0 <= p['sparse_threshold'] <= 1:
            raise ValueError('sparse_threshold must be between zero and one')
        cls._validateParameters(p)
        return p

    @classmethod
    def _validateParameters(cls, p):
        pass

    @staticmethod
    def _requirePositive(p, *names):
        for name in names:
            if p[name] <= 0:
                raise ValueError(f'{name} must be positive')

    @staticmethod
    def _requireNonnegative(p, *names):
        for name in names:
            if p[name] < 0:
                raise ValueError(f'{name} must be nonnegative')

    @staticmethod
    def _size(n):
        return n

    def estimatedWorkingMemoryBytes(self, problem, solverParameters=None):
        p = self._getParameters(solverParameters)
        n = self._size(problem.variableCount)
        if problem.variableCount > p['max_variables']:
            raise ValueError('Problem exceeds max_variables')
        if p['matrix_format'] == 'dense' and n > p['max_dense_variables']:
            raise ValueError('Padded matrix exceeds max_dense_variables')
        if p.get('proof_nodes', 0) and problem.variableCount > p['proof_max_variables']:
            raise ValueError('Proof search exceeds proof_max_variables')
        width = min(p['runs'], p['run_batch_size'] or p['runs'])
        item = 4 if p['dtype'] == 'float32' else 8
        dense = p['matrix_format'] != 'sparse' and n <= p['max_dense_variables']
        # Includes simultaneous accepted/trial states, sparse construction, source
        # scoring, and oscillator stencils. No per-step graph pools are retained.
        workspace = 128 + 32 * p.get('oscillator_points', 0)
        proof = 128 * n**3 if p.get('proof_nodes', 0) else 0
        return int(16 * problem.numericMemoryBytes + (2 * item * n*n if dense else 0)
                   + item*n*width*workspace
                   + min(problem.interactionCount, p['energy_chunk_size'])*width*32 + proof)

    def _solveBatch(self, problems, parameters):
        self.lastDiagnostics = []
        results = []
        with self._torch().inference_mode():
            for problem in problems:
                result, diagnostics = self._solveProblem(problem, parameters)
                results.append(result)
                self.lastDiagnostics.append(diagnostics)
        return results

    def _mergeWorkerState(self, workers):
        self.lastDiagnostics = [d for worker in workers for d in worker.lastDiagnostics]

    def solveResidentPlanned(self, problems, plan, solverParameters=None):
        if len(self.devices) != 1:
            raise ValueError('Resident physics solvers require one device')
        if problems and plan.shards != ((0, len(problems)),):
            raise ValueError('Resident resource plan must cover the ordered batch')
        return self.solvePlanned([problem.source for problem in problems], plan, solverParameters)

    def _initial(self, model, problem, p, start, width, channels=1):
        values = numpy.empty((model.size, width, channels))
        for run in range(width):
            seed = (p['seed'] + problem.seedOffset + (start + run)*_RUN_SEED_STRIDE) % _MAX_TORCH_SEED
            values[:, run] = numpy.random.default_rng(seed).uniform(-1., 1., (model.size, channels))
        return model.torch.tensor(values, dtype=model.h.dtype, device=model.h.device)

    def _solveProblem(self, problem, p):
        torch = self._torch()
        model = PhysicsModel(self, problem, p, self._size(problem.variableCount))
        accumulator = TorchCandidateAccumulator(torch, problem, self.device,
                                                p['energy_chunk_size'], self._selectBestCandidates)
        diagnostics = dict(device=self.device, dtype=p['dtype'], matrixFormat=model.format,
                           normalizationScale=model.scale, trajectories=[], status='HEURISTIC')
        widthLimit = min(p['runs'], p['run_batch_size'] or p['runs'])
        for start in range(0, p['runs'], widthLimit):
            width = min(widthLimit, p['runs'] - start)

            def collect(x, samples=None):
                if not bool(torch.isfinite(x).all()):
                    raise FloatingPointError(f'{type(self).__name__} dynamics became nonfinite; reduce time_step')
                accumulator.add((x[:model.n].T >= 0) if samples is None else samples[:, :model.n])

            x, detail = self._run(model, problem, p, start, width, collect)
            diagnostics['trajectories'].append(detail)
            if p['conditional_rounding']:
                accumulator.add(self._round(model, x))
            if p['local_search_steps']:
                accumulator.add(self._polish(model, x[:model.n].T >= 0, p['local_search_steps']))
        if p['local_search_steps']:
            best = torch.tensor([accumulator.result()[0]], device=self.device, dtype=torch.bool)
            accumulator.add(self._polish(model, best, p['local_search_steps']))
        sample, energy = accumulator.result()
        if p.get('block_sweeps', 0) or p.get('proof_nodes', 0):
            from .physics_exact import refineAndCertify
            sample, certificate = refineAndCertify(problem, sample, p)
            energy = problem.energy(sample)
            diagnostics.update(certificate)
        return BQMOptimizationResult(sample, energy), diagnostics

    @staticmethod
    def _round(model, x):
        """Sequential conditional rounding with sparse CPU column updates."""
        from scipy.sparse import coo_matrix
        values = x[:model.n].double().cpu().numpy().copy()
        matrix = coo_matrix((numpy.r_[model.values, model.values],
            (numpy.r_[model.heads, model.tails], numpy.r_[model.tails, model.heads])),
            shape=(model.n, model.n)).tocsc()
        field = matrix @ values + model.hostField[:, None]
        for i in range(model.n):
            chosen = numpy.where(field[i] > 0, -1., 1.)
            change = chosen - values[i]
            values[i] = chosen
            lo, hi = matrix.indptr[i:i+2]
            field[matrix.indices[lo:hi]] += matrix.data[lo:hi, None] * change
        return model.torch.tensor(values.T >= 0, device=model.h.device)

    @staticmethod
    def _polish(model, samples, steps):
        torch = model.torch
        spins = torch.zeros((model.size, len(samples)), device=model.h.device, dtype=model.h.dtype)
        spins[:model.n] = samples.T.to(model.h.dtype)*2-1
        columns = torch.arange(len(samples), device=model.h.device)
        for _ in range(steps):
            delta = -2*spins[:model.n]*(model.product(spins) + model.h)[:model.n]
            best, indices = delta.min(0)
            spins[indices, columns] *= torch.where(best < 0, -1., 1.)
        return (spins[:model.n].T > 0).contiguous()


def smoothSchedule(step, steps):
    fraction = observation.schedule_fraction(None)
    u = (min(1., step / max(1., .8 * (steps - 1)))
         if fraction is None else min(1., fraction/.8))
    return u*u*(3-2*u)


def guardedStep(torch, old, trial, oldEnergy, trialEnergy, step, maximum, decrease=0.):
    """Per-column finite energy guard; rejected attempts consume their budget."""
    # Energies are accumulated in FP64, including for FP32 propagation. This
    # is an acceptance tolerance, never a bound used by exact proof search.
    tolerance = 128 * torch.finfo(torch.float64).eps * (1 + oldEnergy.abs())
    accepted = torch.isfinite(trial).all(0) & torch.isfinite(trialEnergy)
    accepted &= trialEnergy <= oldEnergy - decrease + tolerance
    nextStep = torch.where(accepted, (step*1.02).clamp_max(maximum), (step*.5).clamp_min(1e-8))
    return accepted, nextStep
