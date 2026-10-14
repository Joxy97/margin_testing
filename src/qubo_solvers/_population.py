"""Source-precision execution for the two new continuous population methods.

Only immutable operators cross trajectory batches. Dynamics may use FP32, but
every observed candidate is ranked with the original FP64 operator before
retention. FP64 is not a general exact-real arithmetic guarantee. Compact sparse
inputs never require a dense host/device matrix; no legacy solver uses this seam.
"""

from __future__ import annotations

import torch

from . import observation
from .problems import Ising, QUBO
from .results import OptimizationResult
from .solvers import MemoryEstimate, _integer, _matrix


def _dtype(value, fallback):
    if value is None:
        return fallback
    if value in (torch.float32, torch.float64):
        return value
    if value in ('float32', 'float64'):
        return getattr(torch, value)
    raise ValueError('precision must be float32 or float64')


def _mm(matrix, values, out=None):
    """Apply a symmetric operator to row-wise vectors without dense conversion."""
    if matrix.layout == torch.strided:
        if out is None:
            return values @ matrix
        torch.mm(values, matrix, out=out)
    else:
        product = torch.sparse.mm(matrix, values.T).T
        if out is None:
            return product
        out.copy_(product)
    return out


def _csr_from_scipy(matrix, device):
    """One-time authoritative coefficient transfer, including CSR indices."""
    return torch.sparse_csr_tensor(
        torch.tensor(matrix.indptr, dtype=torch.int64, device=device),
        torch.tensor(matrix.indices, dtype=torch.int64, device=device),
        torch.tensor(matrix.data, dtype=torch.float64, device=device),
        size=matrix.shape,
    )


class PopulationObjective:
    """Original source scorer plus a separately normalized Ising force operator."""

    def __init__(self, matrix, field, offset, spin, precision, normalize, threshold):
        self.source_terms = None
        self.source_matrix = matrix.to(dtype=torch.float64)
        self.source_field = None if field is None else field.to(dtype=torch.float64)
        self.offset = torch.as_tensor(offset, device=matrix.device, dtype=torch.float64)
        self.spin = spin
        self.n = matrix.shape[0]
        if self.source_matrix.layout == torch.strided:
            # The normalization uses Jphys (1/2 convention), not Jlib.
            force = self.source_matrix * (2. if spin else .5)
            force.diagonal().zero_()
            source_h = (self.source_field if spin else self.source_matrix.sum(1) * .5)
            row_bound = force.abs().sum(1)
            sparse = int(torch.count_nonzero(force)) < threshold * self.n * self.n
            if sparse:
                force = force.to_sparse_csr()
                # Sparse scoring matters too, even for a dense caller-owned input.
                if int(torch.count_nonzero(self.source_matrix)) < threshold * self.n * self.n:
                    self.source_matrix = self.source_matrix.to_sparse_csr()
        else:
            source = self.source_matrix
            rows = torch.repeat_interleave(
                torch.arange(self.n, device=source.device),
                source.crow_indices()[1:] - source.crow_indices()[:-1],
                output_size=source.values().numel(),
            )
            values = source.values() * (2. if spin else .5)
            values.masked_fill_(rows == source.col_indices(), 0.)
            source_h = (self.source_field if spin else
                        torch.zeros(self.n, dtype=torch.float64, device=source.device)
                        .index_add_(0, rows, source.values()) * .5)
            row_bound = torch.zeros_like(source_h).index_add_(0, rows, values.abs())
            force = torch.sparse_csr_tensor(source.crow_indices(), source.col_indices(),
                                            values, size=source.shape)
        self.scale = ((source_h.abs().max() + row_bound.max()).clamp_min(1.)
                      if normalize else self.offset.new_ones(()))
        if not bool(torch.isfinite(self.scale) & torch.isfinite(source_h).all()):
            raise FloatingPointError('source force/normalization overflows float64')
        if force.layout == torch.strided:
            force.div_(self.scale)
            self.matrix = force.to(dtype=precision)
        else:
            self.matrix = torch.sparse_csr_tensor(
                force.crow_indices(), force.col_indices(),
                (force.values() / self.scale).to(dtype=precision), size=force.shape)
        self.field = (source_h / self.scale).to(dtype=precision)
        coefficients = (self.matrix if self.matrix.layout == torch.strided
                        else self.matrix.values())
        if not bool(torch.isfinite(coefficients).all() & torch.isfinite(self.field).all()):
            raise FloatingPointError('force coefficients overflow dynamics precision; '
                                     'enable normalization or use float64 dynamics')
        self.device, self.dtype = matrix.device, precision

    @classmethod
    def from_problem(cls, problem, *, precision=None, normalize=True, threshold=.15):
        matrix = _matrix(problem)
        return cls(matrix, problem.h if isinstance(problem, Ising) else None,
                   problem.offset, isinstance(problem, Ising),
                   _dtype(precision, matrix.dtype), normalize, threshold)

    @classmethod
    def from_compact(cls, problem, *, device, precision, normalize, threshold):
        """Prepare on host once; density controls representation, not coefficients."""
        import numpy as np
        n, m = problem.variableCount, problem.interactionCount
        if 2 * m + n < threshold * n * n:
            from scipy.sparse import coo_matrix
            diagonal = np.arange(n, dtype=np.int64)
            matrix = coo_matrix((
                np.concatenate((problem.linear, problem.quadraticBiases * .5,
                                problem.quadraticBiases * .5)),
                (np.concatenate((diagonal, problem.quadraticHeads, problem.quadraticTails)),
                 np.concatenate((diagonal, problem.quadraticTails, problem.quadraticHeads))),
            ), shape=(n, n)).tocsr()
            matrix.eliminate_zeros()
            tensor = _csr_from_scipy(matrix, device)
        else:
            matrix = np.zeros((n, n), dtype=np.float64)
            np.fill_diagonal(matrix, problem.linear)
            biases = problem.quadraticBiases * .5
            np.add.at(matrix, (problem.quadraticHeads, problem.quadraticTails), biases)
            np.add.at(matrix, (problem.quadraticTails, problem.quadraticHeads), biases)
            tensor = torch.from_numpy(matrix).to(device=device, dtype=torch.float64)
        objective = cls(tensor, None, problem.offset, False, precision, normalize, threshold)
        objective.source_terms = (
            torch.tensor(problem.linear, dtype=torch.float64, device=device),
            torch.tensor(problem.quadraticHeads, dtype=torch.int64, device=device),
            torch.tensor(problem.quadraticTails, dtype=torch.int64, device=device),
            torch.tensor(problem.quadraticBiases, dtype=torch.float64, device=device),
        )
        # Force preparation may aggregate duplicate coefficients; authoritative
        # scoring must not reuse those rounded sums or split self-loop biases.
        objective.source_matrix = None
        return objective

    @classmethod
    def from_resident(cls, resident, *, device, precision, normalize, threshold):
        """Build from authoritative resident FP64 arrays without a host round trip."""
        n = resident.source.variableCount
        linear, heads, tails, biases = (
            resident.linear, resident.heads, resident.tails, resident.biases)
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if any(v.device != device for v in (linear, heads, tails, biases)):
            raise ValueError('resident coefficients must already be on the solver device')
        if linear.dtype != torch.float64 or biases.dtype != torch.float64:
            raise ValueError('resident source coefficients must use float64')
        if linear.shape != (n,) or heads.ndim != 1 or not (
                heads.shape == tails.shape == biases.shape):
            raise ValueError('resident coefficient arrays have invalid shapes')
        for indices in (heads, tails):
            if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
                raise TypeError('resident coefficient indices must use integer dtype')
            if bool(((indices < 0) | (indices >= n)).any()):
                raise ValueError('resident coefficient indices are out of range')
        if not bool(torch.isfinite(linear).all() & torch.isfinite(biases).all()):
            raise ValueError('resident source coefficients must be finite')
        heads, tails = heads.long(), tails.long()
        if 2 * biases.numel() + n < threshold * n * n:
            diagonal = torch.arange(n, device=device)
            indices = torch.stack((torch.cat((diagonal, heads, tails)),
                                   torch.cat((diagonal, tails, heads))))
            values = torch.cat((linear, biases * .5, biases * .5))
            matrix = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce().to_sparse_csr()
        else:
            matrix = torch.zeros((n, n), device=device, dtype=torch.float64)
            matrix.diagonal().copy_(linear)
            half = biases * .5
            matrix.index_put_((heads, tails), half, accumulate=True)
            matrix.index_put_((tails, heads), half, accumulate=True)
        objective = cls(matrix, None, resident.source.offset, False, precision, normalize, threshold)
        objective.source_terms = (linear, heads, tails, biases)
        objective.source_matrix = None
        return objective

    def gradient(self, values, out=None):
        out = _mm(self.matrix, values, out=out)
        out.add_(self.field)
        return out

    def score(self, assignments):
        values = assignments.to(dtype=torch.float64)
        if self.source_terms is not None:
            linear, heads, tails, biases = self.source_terms
            energy = values @ linear
            # Bound gathered edge products to one million cells, independent
            # of graph density. No B-by-m allocation and no scalar CUDA reads.
            chunk = max(1, 1_048_576 // max(1, values.shape[0]))
            for start in range(0, biases.numel(), chunk):
                stop = start + chunk
                terms = values[:, heads[start:stop]] * values[:, tails[start:stop]]
                terms.mul_(biases[start:stop])
                energy.add_(terms.sum(-1))
            return energy.add_(self.offset)
        energy = (_mm(self.source_matrix, values) * values).sum(-1)
        if self.spin:
            energy.add_((values * self.source_field).sum(-1))
        energy.add_(self.offset)
        return energy


# Private aliases kept explicit for focused source/scoring tests.
_PopulationObjective = PopulationObjective


class PopulationRun:
    """One swarm/batch, with per-row source-score archives and device RNG."""

    def __init__(self, objective, count, generator, initial=None, history_interval=None):
        self.objective, self.generator = objective, generator
        self.q, self.spin = objective.matrix, objective.spin
        self.history_interval = history_interval
        shape = (count, objective.n)
        if initial is None:
            self.x = torch.empty(shape, device=objective.device, dtype=objective.dtype)
            self.x.bernoulli_(.5, generator=generator)
            if self.spin:
                self.x.mul_(2).sub_(1)
        else:
            self.x = initial.to(device=objective.device, dtype=objective.dtype, copy=True)
        self.initialbits = self.x > 0
        self.best_x = self.x.clone()
        self.energies = objective.score(self.x)
        self.best = self.energies.clone()
        self.iterations = torch.zeros(count, dtype=torch.int64, device=objective.device)
        self.reasons = torch.zeros_like(self.iterations)

    def normal(self, shape):
        return torch.randn(shape, device=self.q.device, dtype=self.q.dtype,
                           generator=self.generator)

    def to_assignments(self, spins):
        values = (spins >= 0).to(dtype=self.q.dtype)
        return values.mul_(2).sub_(1) if self.spin else values

    def score(self, samples):
        return self.objective.score(samples)

    def retain(self, samples, energies=None):
        energies = self.score(samples) if energies is None else energies
        improved = energies < self.best
        torch.where(improved[:, None], samples, self.best_x, out=self.best_x)
        torch.minimum(self.best, energies, out=self.best)

    def publish(self, spins, step):
        observation.poll()
        self.x.copy_(self.to_assignments(spins))
        self.energies = self.score(self.x)
        self.retain(self.x, self.energies)
        self.iterations.fill_(step)

    def observe(self, samples, energies, *, phase='checkpoint', iteration=None):
        observation.capture(samples, energies, phase=phase, iteration=iteration)


_PopulationRun = PopulationRun


class PopulationSolver:
    """Shared two-loop lifecycle, FP64 results and preallocation admission."""

    def _options(self, restarts, batch_size, best_only, history_interval, seed=None):
        _integer('restarts', restarts, 1)
        if batch_size is not None:
            _integer('batch_size', batch_size, 1)
        if not isinstance(best_only, bool):
            raise ValueError('best_only must be boolean')
        if history_interval is not None:
            _integer('history_interval', history_interval, 1)
        if best_only and history_interval is not None:
            raise ValueError('best_only does not support history_interval')
        if seed is not None:
            _integer('seed', seed, 0)
        return min(restarts, batch_size or restarts)

    def _estimate(self, n, restarts, batch, best_only, interval, precision,
                  problem_bytes, operator_workspace):
        count = 1 if best_only else restarts
        width = torch.empty((), dtype=precision).element_size()
        samples = (self.max_steps // interval + 1 + bool(self.max_steps % interval)
                   if interval else 0)
        output = (count * (n + 32) if best_only else
                  restarts * (2 * n + 32) + samples * (8 * restarts + 8))
        # Current/best candidates, source scoring casts/products, masks and status.
        common = (8 * width + 32) * batch * n + 128 * batch + 64 * n
        extra = self._population_workspace(n, batch, width)
        return MemoryEstimate(int(problem_bytes), int(output),
                              int(operator_workspace + common + extra))

    def estimate_memory(self, problem, *, restarts=1, batch_size=None,
                        best_only=False, history_interval=None):
        batch = self._options(restarts, batch_size, best_only, history_interval)
        matrix = _matrix(problem)
        precision = _dtype(self.dynamics_dtype, matrix.dtype)
        width = torch.empty((), dtype=precision).element_size()
        n = problem.n_variables
        original = matrix.numel() * matrix.element_size() + matrix.element_size()
        if isinstance(problem, Ising):
            original += n * matrix.element_size()
        return self._estimate(n, restarts, batch, best_only, history_interval,
                              precision, original, (32 + 2 * width) * n * n)

    def estimate_compact_memory(self, problem, *, precision='float32', restarts=1,
                                batch_size=None, best_only=False, history_interval=None):
        batch = self._options(restarts, batch_size, best_only, history_interval)
        precision = _dtype(self.dynamics_dtype, _dtype(precision, torch.float32))
        width = torch.empty((), dtype=precision).element_size()
        n, m = problem.variableCount, problem.interactionCount
        if 2 * m + n < self.sparse_threshold * n * n:
            operators = (2 * m + n) * (96 + 4 * width) + (64 + 4 * width) * (n + 1)
        else:
            operators = (32 + 2 * width) * n * n
        # Original compact terms (including duplicates) remain authoritative;
        # edge scoring is chunked rather than materializing B-by-m products.
        operators += 8 * n + 24 * m + 32 * batch * min(m, max(1, 1_048_576 // batch))
        return self._estimate(n, restarts, batch, best_only, history_interval, precision,
                              problem.numericMemoryBytes, operators + 2 * problem.numericMemoryBytes)

    @staticmethod
    def _admit(estimate, limit):
        if limit is not None:
            _integer('memory_limit_bytes', limit, 1)
            if estimate.total_bytes > limit:
                raise MemoryError(f'estimated tensor storage {estimate.total_bytes} bytes exceeds '
                                  f'memory_limit_bytes={limit}; reduce batch_size or use best_only')

    @torch.no_grad()
    def solve(self, problem, *, restarts=1, initial_assignments=None, seed=None,
              history_interval=None, batch_size=None, best_only=False, memory_limit_bytes=None):
        batch = self._options(restarts, batch_size, best_only, history_interval, seed)
        estimate = self.estimate_memory(problem, restarts=restarts, batch_size=batch_size,
                                        best_only=best_only, history_interval=history_interval)
        self._admit(estimate, memory_limit_bytes)
        matrix = _matrix(problem)
        self._validate_initial(initial_assignments, restarts, problem.n_variables,
                               matrix.device, isinstance(problem, Ising), batch)
        try:
            objective = PopulationObjective.from_problem(
                problem, precision=self.dynamics_dtype, normalize=self.normalize,
                threshold=self.sparse_threshold)
            return self._execute(objective, restarts, batch, initial_assignments, seed,
                                 best_only, history_interval)
        except RuntimeError as error:
            self._allocation_error(error, estimate)

    @torch.no_grad()
    def solve_compact(self, problem, *, device='cpu', precision='float32', restarts=1,
                      initial_assignments=None, seed=None, history_interval=None,
                      batch_size=None, best_only=False, memory_limit_bytes=None):
        from .backends.problem import QUBOProblem
        if not isinstance(problem, QUBOProblem):
            raise TypeError('compact problem must be QUBOProblem')
        batch = self._options(restarts, batch_size, best_only, history_interval, seed)
        estimate = self.estimate_compact_memory(problem, precision=precision, restarts=restarts,
                    batch_size=batch_size, best_only=best_only, history_interval=history_interval)
        self._admit(estimate, memory_limit_bytes)
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        self._validate_initial(initial_assignments, restarts, problem.variableCount, device, False, batch)
        try:
            objective = PopulationObjective.from_compact(problem, device=device,
                precision=_dtype(self.dynamics_dtype, _dtype(precision, torch.float32)),
                normalize=self.normalize, threshold=self.sparse_threshold)
            return self._execute(objective, restarts, batch, initial_assignments, seed,
                                 best_only, history_interval)
        except RuntimeError as error:
            self._allocation_error(error, estimate)

    @torch.no_grad()
    def solve_compact_resident(self, resident, *, device='cpu', precision='float32',
                               restarts=1, initial_assignments=None, seed=None,
                               history_interval=None, batch_size=None, best_only=False,
                               memory_limit_bytes=None):
        batch = self._options(restarts, batch_size, best_only, history_interval, seed)
        estimate = self.estimate_compact_memory(resident.source, precision=precision,
                    restarts=restarts, batch_size=batch_size, best_only=best_only,
                    history_interval=history_interval)
        self._admit(estimate, memory_limit_bytes)
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        self._validate_initial(initial_assignments, restarts, resident.source.variableCount,
                               device, False, batch)
        try:
            objective = PopulationObjective.from_resident(resident, device=device,
                precision=_dtype(self.dynamics_dtype, _dtype(precision, torch.float32)),
                normalize=self.normalize, threshold=self.sparse_threshold)
            return self._execute(objective, restarts, batch, initial_assignments, seed,
                                 best_only, history_interval)
        except RuntimeError as error:
            self._allocation_error(error, estimate)

    @staticmethod
    def _validate_initial(initial, restarts, n, device, spin, batch):
        if initial is None:
            return
        if not isinstance(initial, torch.Tensor) or initial.shape != (restarts, n):
            raise ValueError('initial_assignments must have shape (restarts, n_variables)')
        if initial.layout != torch.strided or initial.is_complex() or initial.device != device:
            raise ValueError('initial assignments must be dense real tensors on the problem device')
        for start in range(0, restarts, batch):
            values = initial[start:start + batch]
            if not bool(((values == (-1 if spin else 0)) | (values == 1)).all()):
                raise ValueError('initial assignments contain values outside the problem domain')

    @staticmethod
    def _allocation_error(error, estimate):
        message = str(error).lower()
        if isinstance(error, torch.OutOfMemoryError) or any(
                word in message for word in ('out of memory', "can't allocate memory")):
            raise MemoryError(f'solve allocation failed (estimated tensor storage '
                              f'{estimate.total_bytes} bytes); no automatic retry was attempted') from error
        raise error

    def _execute(self, objective, restarts, batch, initial, seed, best_only, interval):
        generator = torch.Generator(device=objective.device)
        generator.seed() if seed is None else generator.manual_seed(seed)
        count, n, device = (1 if best_only else restarts), objective.n, objective.device
        best_x = torch.empty((count, n), device=device, dtype=torch.int8)
        best_e = torch.full((count,), float('inf'), device=device, dtype=torch.float64)
        final_x = None if best_only else torch.empty_like(best_x)
        final_e = None if best_only else torch.empty_like(best_e)
        iterations = torch.zeros(count, device=device, dtype=torch.int64)
        reasons = torch.zeros_like(iterations)
        winner = torch.zeros(1, device=device, dtype=torch.int64) if best_only else None
        steps = list(range(0, self.max_steps + 1, interval)) if interval else []
        if interval and steps[-1] != self.max_steps:
            steps.append(self.max_steps)
        step_rows = {step: row for row, step in enumerate(steps)}
        history = (torch.empty((len(steps), restarts), device=device, dtype=torch.float64)
                   if interval else None)
        observer = observation.current()
        if observer is not None and hasattr(observer, 'prepared'):
            observer.prepared(objective.matrix, original_variables=n, internal_coordinates=n,
                              dynamics_dtype=str(objective.dtype), scoring_dtype='torch.float64')
        for start in range(0, restarts, batch):
            width = min(batch, restarts - start)
            run = PopulationRun(objective, width, generator,
                                None if initial is None else initial[start:start + width], interval)

            def record(step):
                observation.poll()
                run.observe(run.x, run.energies,
                            phase='initial' if step == 0 else 'checkpoint', iteration=step)
                if history is not None and step in step_rows:
                    history[step_rows[step], start:start + width].copy_(run.energies)

            record(0)
            step = self._search(run, record)
            if not bool(torch.isfinite(run.best).all() & torch.isfinite(run.energies).all()):
                raise FloatingPointError('nonfinite source objective; rescale coefficients or use higher precision')
            if best_only:
                index = run.best.argmin().reshape(1)
                candidate_e = run.best.index_select(0, index)
                improve = candidate_e < best_e
                best_x.copy_(torch.where(improve[:, None],
                                        run.best_x.index_select(0, index).to(torch.int8), best_x))
                best_e.copy_(torch.where(improve, candidate_e, best_e))
                winner.copy_(torch.where(improve, index + start, winner))
                iterations.copy_(torch.where(improve, run.iterations.index_select(0, index), iterations))
                reasons.copy_(torch.where(improve, run.reasons.index_select(0, index), reasons))
            else:
                end = start + width
                best_x[start:end].copy_(run.best_x)
                best_e[start:end].copy_(run.best)
                final_x[start:end].copy_(run.x)
                final_e[start:end].copy_(run.energies)
                iterations[start:end].copy_(run.iterations)
                reasons[start:end].copy_(run.reasons)
                if history is not None:
                    history[-1, start:end].copy_(run.energies)
            del run
        return OptimizationResult(final_x, final_e, best_x, best_e, iterations, reasons,
                                  history, torch.tensor(steps, device=device, dtype=torch.int64)
                                  if history is not None else None, winner)
