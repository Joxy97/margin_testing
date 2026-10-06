"""One fresh seed, one state, nine objective scalars; no solution serialization.

Scoring uses original coefficients. An immutable GPU scalar becomes eligible only after
its producing stream has completed. At a checkpoint use the last certified
incumbent, never a candidate that completes after that checkpoint. Candidate
hooks synchronize work but copy no assignments. Proven-optimum checks also copy
one scalar after each completed capture. The old vector Observer is untouched.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np

from qubo_solvers.observation import SolveInterrupted, observing

CHECKPOINTS = (.05, .1, .2, .5, 1., 2., 5., 10., 20.)
PROTECTED = frozenset(('lib_simulated_annealing', 'lib_simulated_bifurcation',
                      'lib_greedy_local_search', 'lib_tabu_search', 'lib_tap_annealing'))


class ObjectiveValidationError(ValueError):
    """A completed independent score contradicts a proven reference."""


def _scoring_operator_required(worker, job):
    # SBM uses the compact solve interface but observes decoded device tensors
    # directly; the other compact backends submit already scored objectives.
    return worker.native or job['solver'] == 'lib_simulated_bifurcation'


class ObjectiveScorer:
    """Authoritative original coefficients, independent of reference targets."""

    def __init__(self, source, device, *, native=True, precision='float32'):
        import torch
        self.source, self.device, self.torch = source, device, torch
        bound = source._score_absolute_bound
        if source._score_integer and bound > 2**53:
            raise ValueError('Objective exceeds exact float64 integer range; scalar GPU scoring is unsafe')
        # Integer arithmetic remains exact in FP32 if every possible partial
        # absolute sum fits its integer mantissa. This is a proof, not a heuristic
        # tolerance or use of a reference. The normalized native Q is unchanged.
        self.exact_float32 = bool(source._score_integer and bound <= 2**24)
        self.matrix = None
        if native:
            score_dtype = torch.float32 if self.exact_float32 else torch.float64
            sparse = source.sparse()
            density = sparse.nnz / (source.n*source.n)
            if density < .2:
                self.matrix = torch.sparse_csr_tensor(
                    torch.as_tensor(sparse.indptr, dtype=torch.int64, device=device),
                    torch.as_tensor(sparse.indices, dtype=torch.int64, device=device),
                    torch.as_tensor(sparse.data, dtype=score_dtype, device=device),
                    size=(source.n, source.n), check_invariants=True)
            else:
                self.matrix = torch.as_tensor(source.dense(), dtype=score_dtype, device=device)

    def score(self, samples, raw=None):
        torch = self.torch
        if hasattr(samples, 'detach'):
            values = samples.detach()
            if values.ndim==1:
                values = values.unsqueeze(0)
            if values.ndim!=2 or values.shape[1]!=self.source.n:
                raise ValueError('Candidate dimension differs from original problem')
            valid = ((values==0)|(values==1)).all(1)
            if self.matrix is None:
                # Compact accumulators already score original float64 source
                # coefficients and must use capture_objectives on the device.
                raise ValueError('Device candidate has no authoritative scoring operator')
            x = values.to(self.matrix.dtype)
            product = (x @ self.matrix if self.matrix.layout==torch.strided
                       else torch.sparse.mm(self.matrix, x.T).T)
            energies = (x*product).sum(1)+self.source.offset
            return torch.where(valid, energies, torch.full_like(energies, float('nan'))).min()
        scores = [self.source.score(row) for row in samples]
        return min(scores) if scores else float('nan')


class ScalarObserver:
    scalar_only = True

    def __init__(self, scorer, budget=20., checkpoints=CHECKPOINTS, stop=None,
                 clock=time.perf_counter_ns, wall_schedule=False, reference=None,
                 reference_type='BKS', tolerance=0.0, stop_on_optimum=True,
                 cuda_graph_stream=None):
        self.scorer, self.budget, self.checkpoints = scorer, budget, tuple(checkpoints)
        if not self.checkpoints or self.checkpoints[-1]!=budget:
            raise ValueError('Last checkpoint must equal the continuous budget')
        if reference_type not in ('OPTIMUM', 'BKS'):
            raise ValueError('Reference type must be OPTIMUM or BKS')
        if reference is not None and not math.isfinite(reference):
            raise ValueError('Reference must be finite')
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Tolerance must be finite and nonnegative')
        self.stop, self.clock, self.origin = stop, clock, clock()
        self.wall_schedule = bool(wall_schedule)
        self.cuda_graph_stream = cuda_graph_stream
        self.reference, self.reference_type, self.tolerance = reference, reference_type, tolerance
        self.stop_on_optimum = bool(stop_on_optimum)
        self.time_to_optimum_s, self._optimum_energy = None, None
        self._optimum_stopped = False
        self.energies = np.full(len(self.checkpoints), np.nan, dtype=np.float64)
        self.cursor, self.certified, self.device_best = 0, None, None
        self.invalid = False
        self.preparation = {}  # No matrices, vectors, masses or diagnostics saved.

    def elapsed(self):
        return (self.clock()-self.origin)/1e9

    def schedule_fraction(self, legacy_value=0.):
        return min(1., max(0., self.elapsed()/self.budget)) if self.wall_schedule else legacy_value

    def prepared(self, matrix, **details):
        pass

    def _checkpoint(self, elapsed):
        if self.cursor>=len(self.checkpoints) or elapsed<self.checkpoints[self.cursor]:
            return
        value = self.certified
        if hasattr(value, 'detach'):
            # Exactly one scalar transfer, not a population or solution vector.
            value = float(value.detach().cpu())
        if value is None:
            value = float('nan')
        if not math.isfinite(value):
            self.invalid |= not math.isnan(value)
        while self.cursor<len(self.checkpoints) and elapsed>=self.checkpoints[self.cursor]:
            self.energies[self.cursor] = value
            self.cursor += 1

    def poll(self):
        if self.stop is not None and self.stop.is_set():
            raise SolveInterrupted('interrupted')
        if self._optimum_stopped:
            raise SolveInterrupted('optimum')
        elapsed = self.elapsed()
        self._checkpoint(elapsed)
        if elapsed>=self.budget:
            raise SolveInterrupted('deadline')

    def capture_objectives(self, energy):
        """Accept only original-source scores from a compact GPU accumulator."""
        self.poll()
        if hasattr(energy, 'detach'):
            torch = self.scorer.torch
            scalar = energy.detach().reshape(-1).min().double()
            self.device_best = (scalar if self.device_best is None
                                else torch.minimum(self.device_best, scalar))
            value = self.device_best.clone()
            if value.device.type=='cuda':
                # Bound in-flight GPU work without copying any candidate.
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(value.device))
                event.synchronize()
        else:
            value = float(energy)

        optimum = self.reference is not None and self.reference_type=='OPTIMUM'
        if optimum and hasattr(value, 'detach'):
            # GPU completion, scalar copy and validation all precede eligibility.
            value = float(value.detach().cpu())
        finite = not hasattr(value, 'detach') and math.isfinite(value)
        below_optimum = optimum and finite and value < self.reference-self.tolerance
        optimum_hit = optimum and finite and abs(value-self.reference)<=self.tolerance
        elapsed = self.elapsed()
        self._checkpoint(elapsed)  # Never back-credit late completed work.
        if elapsed>=self.budget:
            raise SolveInterrupted('deadline')
        if not hasattr(value, 'detach') and not finite:
            raise ValueError('Nonfinite candidate objective')
        if below_optimum:
            raise ObjectiveValidationError('Independently scored energy is below the proven optimum')
        self.certified = (value if self.certified is None or hasattr(value, 'detach')
                          else min(self.certified, value))
        if optimum_hit and self.time_to_optimum_s is None:
            self.time_to_optimum_s, self._optimum_energy = elapsed, value
        if optimum_hit and self.stop_on_optimum:
            self._optimum_stopped = True
            raise SolveInterrupted('optimum')

    def capture(self, samples, raw=None, *, phase='checkpoint', iteration=None):
        self.poll()
        self.capture_objectives(self.scorer.score(samples, raw))

    def finish(self):
        """A naturally stopped state is retained; never restart at checkpoints."""
        if self._optimum_stopped:
            self.energies[self.cursor:] = self._optimum_energy
            self.cursor = len(self.checkpoints)
            return self.energies.copy()
        while self.cursor<len(self.checkpoints):
            try:
                self.poll()
            except SolveInterrupted as exc:
                if str(exc)=='deadline':
                    break
                raise
            time.sleep(min(.01, max(0., self.checkpoints[self.cursor]-self.elapsed())))
        if hasattr(self.certified, 'detach'):
            self.invalid |= not math.isfinite(float(self.certified.detach().cpu()))
        return self.energies.copy()


def run_continuous_trial(worker, job, stop=None):
    """Uses existing WarmWorker immutable input/setup and one fresh solve call."""
    from dataclasses import fields
    from qubo_solvers import create_solver, create_bqm_solver, library_solver_class
    from .worker import synchronize
    torch, parameters, device = worker.torch, dict(job['parameters']), worker.device
    # Scoring operator is immutable input preparation, not useful optimization.
    needs_operator = _scoring_operator_required(worker, job)
    key = (worker.cache_key, parameters['dtype'], needs_operator)
    if getattr(worker, '_continuous_score_key', None)!=key:
        worker._continuous_scorer = ObjectiveScorer(worker.source, device,
            native=needs_operator, precision=parameters['dtype'])
        worker._continuous_score_key = key
    synchronize(torch, device)
    observer = ScalarObserver(worker._continuous_scorer, job.get('budget',20.),
        job.get('checkpoints', CHECKPOINTS), stop,
        wall_schedule=job.get('wall_schedule',False) and job['solver'] not in PROTECTED,
        reference=job.get('reference'), reference_type=job.get('reference_type','BKS'),
        tolerance=job.get('tolerance',0.0), stop_on_optimum=job.get('stop_on_optimum',True),
        cuda_graph_stream=getattr(worker,'_continuous_graph_stream',None))
    result = algorithm = solver = None
    status, reason, error = 'complete', 'natural_return', None
    try:
        with observing(observer):
            random.seed(job['seed']); np.random.seed(job['seed']%(2**32))
            torch.manual_seed(job['seed'])
            if device.startswith('cuda'):
                torch.cuda.manual_seed_all(job['seed'])
            observer.poll()
            if worker.native:
                names = {f.name for f in fields(library_solver_class(job['solver']))}
                algorithm = create_solver(job['solver'], **{k:v for k,v in parameters.items() if k in names})
                result = algorithm.solve(worker.prepared, restarts=parameters['runs'], seed=job['seed'],
                    batch_size=parameters['run_batch_size'], best_only=parameters['best_only'],
                    memory_limit_bytes=parameters['memory_limit_bytes'])
                observer.capture(result.best_assignments, result.best_energies, phase='final_returned')
            else:
                solver = create_bqm_solver(job['solver'], {'device':device})
                result = solver.solve(worker.compact, dict(parameters,seed=job['seed']))
                observer.capture([result.sample], [result.energy], phase='final_returned')
            observer.finish()
    except SolveInterrupted as exc:
        reason = str(exc)
        if reason=='interrupted':
            status = 'interrupted'
        else:
            observer.finish()
    except ObjectiveValidationError as exc:
        status = reason = 'validation_error'
        error = f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        status, reason, error = 'error', 'error', f'{type(exc).__name__}: {exc}'
        if isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
            status = reason = 'oom'
    synchronize(torch, device)
    if observer.invalid:
        status, reason = 'invalid_output', 'invalid_output'
    if status=='complete' and np.isnan(observer.energies).all():
        status = 'no_in_budget_candidate'
    worker.trials += 1
    return dict(energies=[float(v) if np.isfinite(v) else None for v in observer.energies],
        status=status, stop_reason=reason, error=error,
        time_to_optimum_s=observer.time_to_optimum_s if status=='complete' else None,
        actual_solve_wall_s=observer.elapsed())
