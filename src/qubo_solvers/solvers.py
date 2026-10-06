"""Shared execution, memory estimates, and structural solver interface."""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .problems import Ising, Problem, QUBO, _assignments
from .results import OptimizationResult, TerminationReason


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _matrix(problem: Problem) -> Tensor:
    if not isinstance(problem, (QUBO, Ising)):
        raise TypeError("problem must be QUBO or Ising")
    return problem.J if isinstance(problem, Ising) else problem.Q


@dataclass(frozen=True)
class MemoryEstimate:
    """Tensor bytes only; workspace is conservative, not an allocator guarantee.

    Excludes caller initial assignments, Python, allocator caches, BLAS/CUDA
    workspaces and context. Problem storage is included in total_bytes.
    """

    problem_bytes: int
    output_bytes: int
    workspace_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.problem_bytes + self.output_bytes + self.workspace_bytes


def estimate_memory(problem: Problem, *, restarts: int = 1,
                    batch_size: int | None = None, best_only: bool = False,
                    history_samples: int = 0) -> MemoryEstimate:
    """Estimate before solving; histories require full-result mode."""
    q = _matrix(problem)
    _integer("restarts", restarts, 1)
    _integer("history_samples", history_samples, 0)
    if batch_size is not None:
        _integer("batch_size", batch_size, 1)
    if best_only and history_samples:
        raise ValueError("best_only does not support histories")
    b = min(restarts, batch_size or restarts)
    n, width = problem.n_variables, q.element_size()
    problem_bytes = q.numel() * width + width
    if isinstance(problem, Ising):
        problem_bytes += n * width
    output = (n + width + 24 if best_only else
              2 * restarts * n + restarts * (2 * width + 16)
              + history_samples * (restarts * width + 8))
    workspace = 12 * b * n * width + 32 * b * width + 16 * n
    return MemoryEstimate(problem_bytes, output, workspace)


class Solver(Protocol):
    def solve(
        self, problem: Problem, *, restarts: int = 1,
        initial_assignments: Tensor | None = None, seed: int | None = None,
        history_interval: int | None = None, batch_size: int | None = None,
        best_only: bool = False, memory_limit_bytes: int | None = None,
    ) -> OptimizationResult: ...


class _Run:
    """One batch; shares immutable problem storage, owns mutable search tensors."""

    def __init__(self, problem, restarts, initial, generator, order_generator,
                 shared_objectives=None):
        self.problem = problem
        self.q = _matrix(problem)
        self.spin = isinstance(problem, Ising)
        self.generator = generator
        self.order_generator = order_generator
        # Only immutable objective tensors cross restart batches. Search state
        # and generators remain local; this cache never survives a solve call.
        self.shared_objectives = {} if shared_objectives is None else shared_objectives
        shape = (restarts, problem.n_variables)
        if initial is None:
            self.x = torch.empty(shape, device=self.q.device, dtype=self.q.dtype)
            self.x.bernoulli_(0.5, generator=generator)
            if self.spin:
                self.x.mul_(2).sub_(1)
        else:
            self.x = initial.to(dtype=self.q.dtype, copy=True)
        self.field = torch.empty_like(self.x)
        self.best_x = self.x.clone()
        self.rows = torch.arange(restarts, device=self.q.device)
        self.iterations = torch.zeros(restarts, dtype=torch.int64, device=self.q.device)
        self.reasons = torch.zeros_like(self.iterations)
        self.refresh_field()
        self.best = self.energies.clone()

    def refresh_field(self):
        torch.mm(self.x, self.q, out=self.field)
        self.energies = (self.field * self.x).sum(dim=1)
        self.field.mul_(2)
        if self.spin:
            self.energies.add_((self.x * self.problem.h).sum(dim=1))
            self.field.add_(self.problem.h)

    def deltas(self):
        direction = -2 * self.x if self.spin else 1 - 2 * self.x
        delta = direction * self.field
        if not self.spin:
            delta.add_(self.q.diagonal())
        return delta

    def remember(self):
        from .observation import poll
        poll()
        improved = self.energies < self.best
        torch.where(improved[:, None], self.x, self.best_x, out=self.best_x)
        torch.minimum(self.best, self.energies, out=self.best)

    def flip_column(self, column, accepted, delta):
        """Annealing uses one shared row view, without a batched row gather."""
        x = self.x[:, column]
        direction = (-2 * x if self.spin else 1 - 2 * x) * accepted
        x.add_(direction)
        self.field.addcmul_(direction[:, None], self.q[column], value=2)
        self.energies.add_(delta * accepted)
        self.remember()

    def flip_rows(self, columns, accepted, delta):
        x = self.x[self.rows, columns]
        direction = (-2 * x if self.spin else 1 - 2 * x) * accepted
        self.x[self.rows, columns] += direction
        self.field.addcmul_(direction[:, None], self.q[columns], value=2)
        self.energies.add_(delta * accepted)
        self.remember()

    def refresh(self):
        self.refresh_field()
        # Direct comparison without the constant offset.
        self.best = ((self.best_x @ self.q) * self.best_x).sum(dim=1)
        if self.spin:
            self.best.add_((self.best_x * self.problem.h).sum(dim=1))
        self.remember()


class _Output:
    """Preallocated full results, or just one retained winner across batches."""

    def __init__(self, problem, restarts, best_only, interval, budget):
        self.problem = problem
        q = _matrix(problem)
        self.best_only = best_only
        count = 1 if best_only else restarts
        self.best_x = torch.empty((count, problem.n_variables), device=q.device, dtype=torch.int8)
        self.best_e = torch.full((count,), float('inf'), device=q.device, dtype=q.dtype)
        self.iterations = torch.empty(count, device=q.device, dtype=torch.int64)
        self.reasons = torch.empty_like(self.iterations)
        self.final_x = None if best_only else torch.empty_like(self.best_x)
        self.final_e = None if best_only else torch.empty_like(self.best_e)
        self.winner = torch.zeros(1, device=q.device, dtype=torch.int64) if best_only else None
        self.steps = list(range(0, budget + 1, interval)) if interval else []
        if interval and self.steps[-1] != budget:
            self.steps.append(budget)
        self.history = (torch.empty((len(self.steps), restarts), device=q.device, dtype=q.dtype)
                        if interval else None)
        self.max_step = 0
        self.has_winner = False

    def finish_batch(self, run, start, step):
        run.refresh()
        final_e = run.problem._energy(run.x)
        best_e = run.problem._energy(run.best_x)
        improved = final_e < best_e
        torch.where(improved[:, None], run.x, run.best_x, out=run.best_x)
        torch.minimum(final_e, best_e, out=best_e)
        if self.best_only:
            index = best_e.argmin().reshape(1)
            candidate_e = best_e.index_select(0, index)
            improve = (candidate_e < self.best_e) | (not self.has_winner)
            # copy_ prevents retaining views into completed batch tensors.
            # Tensor scalar indexing invokes aten::item, synchronizing CUDA.
            # Keep the winning index as a one-element device tensor throughout.
            self.best_x.copy_(torch.where(improve[:, None],
                                         run.best_x.index_select(0, index), self.best_x))
            self.best_e.copy_(torch.where(improve, candidate_e, self.best_e))
            self.iterations.copy_(torch.where(improve,
                                             run.iterations.index_select(0, index), self.iterations))
            self.reasons.copy_(torch.where(improve,
                                          run.reasons.index_select(0, index), self.reasons))
            self.winner.copy_(torch.where(improve, index + start, self.winner))
            self.has_winner = True
        else:
            end = start + run.x.shape[0]
            self.final_x[start:end].copy_(run.x)
            self.best_x[start:end].copy_(run.best_x)
            self.final_e[start:end].copy_(final_e)
            self.best_e[start:end].copy_(best_e)
            self.iterations[start:end].copy_(run.iterations)
            self.reasons[start:end].copy_(run.reasons)
        self.max_step = max(self.max_step, step)

    def result(self):
        history, indices = self.history, None
        if history is not None:
            length = sum(step <= self.max_step for step in self.steps)
            if self.steps[length - 1] != self.max_step:
                history[length].copy_(self.final_e)
                selected = self.steps[:length] + [self.max_step]
                length += 1
            else:
                selected = self.steps[:length]
                history[length - 1].copy_(self.final_e)
            history = history[:length]
            indices = torch.tensor(selected, device=history.device, dtype=torch.int64)
        return OptimizationResult(self.final_x, self.final_e, self.best_x, self.best_e,
                                  self.iterations, self.reasons, history, indices, self.winner)


def _solve(solver, problem, restarts, initial, seed, interval, batch_size, best_only, limit):
    q = _matrix(problem)
    if interval is not None:
        _integer("history_interval", interval, 1)
    if best_only and interval is not None:
        raise ValueError("best_only does not support history_interval")
    budget = solver.sweeps if hasattr(solver, "sweeps") else solver.max_steps
    samples = (budget // interval + 1 + bool(budget % interval)) if interval else 0
    estimate = estimate_memory(problem, restarts=restarts, batch_size=batch_size,
                               best_only=best_only, history_samples=samples)
    extra = getattr(solver, "_extra_workspace", lambda problem, batch: 0)(
        problem, min(restarts, batch_size or restarts))
    estimate = MemoryEstimate(estimate.problem_bytes, estimate.output_bytes,
                              estimate.workspace_bytes + extra)
    if limit is not None:
        _integer("memory_limit_bytes", limit, 1)
        if estimate.total_bytes > limit:
            raise MemoryError(f"estimated tensor storage {estimate.total_bytes} bytes exceeds "
                              f"memory_limit_bytes={limit}; reduce batch_size or use best_only")
    if initial is not None:
        if not isinstance(initial, Tensor) or initial.shape != (restarts, problem.n_variables):
            raise ValueError("initial_assignments must have shape (restarts, n_variables)")
    b = min(restarts, batch_size or restarts)
    # Validate by batch without casting a full restart array or allocating its mask.
    if initial is not None:
        for start in range(0, restarts, b):
            _assignments(initial[start:start + b], q, isinstance(problem, Ising), cast=False)
    generator = torch.Generator(device=q.device)
    generator.seed() if seed is None else generator.manual_seed(seed)
    order = torch.Generator().manual_seed(generator.initial_seed())
    try:
        output = _Output(problem, restarts, best_only, interval, budget)
        shared_objectives = {}
        for start in range(0, restarts, b):
            count = min(b, restarts - start)
            run = _Run(problem, count, None if initial is None else initial[start:start + count],
                       generator, order, shared_objectives)
            history_cursor = 0

            def record(step):
                nonlocal history_cursor
                from .observation import capture, poll
                poll()
                capture(run.x if step == 0 else run.best_x,
                        run.energies if step == 0 else run.best,
                        phase='initial' if step == 0 else 'checkpoint', iteration=step)
                if output.history is not None and history_cursor < len(output.steps):
                    if step == output.steps[history_cursor]:
                        output.history[history_cursor, start:start + count].copy_(problem._energy(run.x))
                        history_cursor += 1

            record(0)
            step = solver._search(run, record)
            output.finish_batch(run, start, step)
            if output.history is not None and history_cursor < len(output.steps):
                output.history[history_cursor:, start:start + count].copy_(output.final_e[start:start + count])
            del run
        return output.result()
    except RuntimeError as error:
        message = str(error).lower()
        if not isinstance(error, torch.OutOfMemoryError) and not any(
            phrase in message for phrase in ("out of memory", "can't allocate memory")
        ):
            raise
        raise MemoryError(f"solve allocation failed (estimated tensor storage "
                          f"{estimate.total_bytes} bytes); reduce batch_size, use best_only, "
                          "or reduce history storage; no automatic retry was attempted") from error


# Compatibility imports; implementations each live in their own module.
from .simulated_annealing import SimulatedAnnealing
from .greedy_local_search import GreedyLocalSearch
