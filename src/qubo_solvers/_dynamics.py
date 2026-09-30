"""Shared mechanics for continuous searches; all mutable state is per call."""

import math

import torch

from .solvers import MemoryEstimate, _integer, _matrix, _solve, estimate_memory


def positive(name, value, *, zero=False):
    if isinstance(value, bool) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


class IterativeSolver:
    """Private composition seam for the common run/result lifecycle."""

    @torch.no_grad()
    def solve(self, problem, *, restarts=1, initial_assignments=None, seed=None,
              history_interval=None, batch_size=None, best_only=False, memory_limit_bytes=None):
        return _solve(self, problem, restarts, initial_assignments, seed, history_interval,
                      batch_size, best_only, memory_limit_bytes)

    def estimate_memory(self, problem, *, restarts=1, batch_size=None,
                        best_only=False, history_interval=None):
        if history_interval is not None:
            _integer('history_interval', history_interval, 1)
        budget = self.sweeps if hasattr(self, 'sweeps') else self.max_steps
        samples = (budget // history_interval + 1 + bool(budget % history_interval)
                   if history_interval else 0)
        base = estimate_memory(problem, restarts=restarts, batch_size=batch_size,
                               best_only=best_only, history_samples=samples)
        return MemoryEstimate(base.problem_bytes, base.output_bytes, base.workspace_bytes
                              + self._extra_workspace(problem, min(restarts, batch_size or restarts)))

    def _extra_workspace(self, problem, batch):
        n = problem.n_variables
        return _matrix(problem).element_size() * (4 * n * n + 96 * batch * n + 256 * batch)


class SpinObjective:
    """E = s'Js + h's. Normalize forces only, score the original objective."""

    def __init__(self, run, normalize=True):
        key = ('spin_objective', normalize)
        cached = run.shared_objectives.get(key)
        if cached is not None:
            self.j, self.h, self.scale = cached
            return
        if run.spin:
            self.j, self.h = run.q, run.problem.h
        else:
            self.j = run.q * .25
            self.j.diagonal().zero_()
            self.h = run.q.sum(1) * .5
        self.scale = ((2 * self.j.abs().sum(1) + self.h.abs()).max().clamp_min(1)
                      if normalize else 1.)
        run.shared_objectives[key] = (self.j, self.h, self.scale)

    def gradient(self, x):
        return (2 * (x @ self.j) + self.h) / self.scale

    def energy(self, x):
        return ((x @ self.j) * x).sum(-1) + (x * self.h).sum(-1)


def spins(run):
    return run.x.clone() if run.spin else 2 * run.x - 1


def normal(run, shape):
    return torch.randn(shape, device=run.q.device, dtype=run.q.dtype, generator=run.generator)


def publish(run, values):
    """Decode with a positive tie, directly score and retain this checkpoint."""
    binary = values >= 0
    run.x.copy_(binary)
    if run.spin:
        run.x.mul_(2).sub_(1)
    run.refresh_field()
    run.remember()


def unit(x, radius=1.):
    return radius * x / x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)


def finite(*values):
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError('nonfinite dynamics; reduce time_step or rescale coefficients')
