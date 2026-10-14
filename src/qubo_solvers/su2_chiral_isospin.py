"""Low-rank chiral SU(2) annealing in real, unit-length Bloch coordinates.

The explicit geometric midpoint step evaluates the specified ambient force
twice and rotates the original state with the midpoint generator. Chirality
uses two replica-specific vectors; the skew matrix is never materialized.
Ordinary solves use a fixed step schedule. An explicitly enabled observer
wall schedule is a different control protocol, without the fixed-time
midpoint method's order guarantee. Original FP64 scores select candidates.
"""

from dataclasses import dataclass
import math

import torch

from . import observation
from ._dynamics import finite, positive
from ._population import PopulationSolver
from .solvers import _integer


def _rank_two_vectors(normal, shape):
    """Sample orthonormal u,v without device scalar reads or rejection loops.

    A fresh Gaussian second sample replaces a degenerate Gram--Schmidt result.
    The deterministic coordinate fallback also handles mocked zero/collinear
    RNG output. For one variable the chiral operator is identically zero.
    """
    u = normal(shape)
    if shape[-1] == 1:
        return torch.zeros_like(u), torch.zeros_like(u)
    tolerance = 32 * torch.finfo(u.dtype).eps
    length = u.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(u)
    fallback[..., 0] = 1
    u = torch.where(length > tolerance, u / length.clamp_min(tolerance), fallback)
    v, replacement = normal(shape), normal(shape)
    # A repeated projection removes residual parallel components in FP32.
    v.sub_(u * (u * v).sum(-1, keepdim=True))
    v.sub_(u * (u * v).sum(-1, keepdim=True))
    replacement.sub_(u * (u * replacement).sum(-1, keepdim=True))
    replacement.sub_(u * (u * replacement).sum(-1, keepdim=True))
    length = v.norm(dim=-1, keepdim=True)
    v = torch.where(length > tolerance, v, replacement)
    length = v.norm(dim=-1, keepdim=True)
    fallback.zero_().scatter_(-1, u.abs().argmin(-1, keepdim=True), 1)
    fallback.sub_(u * (u * fallback).sum(-1, keepdim=True))
    fallback.div_(fallback.norm(dim=-1, keepdim=True).clamp_min(tolerance))
    return u, torch.where(length > tolerance, v / length.clamp_min(tolerance), fallback)


def _rank_two_action(values, u, v, *, out=None):
    """Apply (u v^T - v u^T) to each replica row in O(batch*n)."""
    vr = (v * values).sum(-1, keepdim=True)
    ur = (u * values).sum(-1, keepdim=True)
    if out is None:
        out = torch.empty_like(values)
    torch.mul(u, vr, out=out)
    out.addcmul_(v, ur, value=-1)
    return out


def _controls(progress, transverse_field, penalty, chirality):
    """a, Gamma, kappa, chi, with exact zero endpoint drivers."""
    progress = max(0., min(1., progress))
    if progress == 1.:
        return 1., 0., penalty, 0.
    return (progress, transverse_field * (1 - progress)**2,
            penalty * progress**2, chirality * math.sin(math.pi * progress)**2)


def _ambient_gradient(vector, objective, u, v, controls, *, out=None):
    """Gradient of a E(z) - Gamma sum(x) + kappa/2 sum(1-z^2) + chi x^T K y."""
    if out is None:
        out = torch.empty_like(vector)
    a, gamma, kappa, chi = controls
    objective.gradient(vector[..., 2], out=out[..., 2])
    out[..., 2].mul_(a).add_(vector[..., 2], alpha=-kappa)
    if chi != 0. and u is not None:
        _rank_two_action(vector[..., 1], u, v, out=out[..., 0])
        out[..., 0].mul_(chi).sub_(gamma)
        _rank_two_action(vector[..., 0], u, v, out=out[..., 1])
        out[..., 1].mul_(-chi)
    else:
        out[..., 0].fill_(-gamma)
        out[..., 1].zero_()
    return out


def _generator(vector, objective, u, v, controls, damping, *, out=None, cross=None):
    """Omega = g - damping * (n cross g), using the ambient convention."""
    out = _ambient_gradient(vector, objective, u, v, controls, out=out)
    if damping != 0.:
        if cross is None:
            cross = torch.empty_like(vector)
        torch.linalg.cross(vector, out, dim=-1, out=cross)
        out.add_(cross, alpha=-damping)
    return out


class _RotationWorkspace:
    def __init__(self, like):
        self.first = torch.empty_like(like)
        self.second = torch.empty_like(like)
        self.angle = torch.empty_like(like[..., :1])
        self.sinc = torch.empty_like(self.angle)
        self.coefficient = torch.empty_like(self.angle)


def _rodrigues(vector, omega, time_step, *, out=None, workspace=None):
    """Real exact frozen-generator rotation, with unnormalized sinc.

    c(q) = .5*sinc(q/2)^2 avoids cancellation in (1-cos(q))/q^2.
    torch.sinc has a pi-normalized argument, hence the explicit division.
    The zero generator is exactly the identity, including in-place calls.
    """
    if workspace is None:
        workspace = _RotationWorkspace(vector)
    if out is None:
        out = torch.empty_like(vector)
    torch.linalg.vector_norm(omega, dim=-1, keepdim=True, out=workspace.angle)
    workspace.angle.mul_(time_step)
    torch.div(workspace.angle, math.pi, out=workspace.sinc)
    torch.sinc(workspace.sinc, out=workspace.sinc)
    torch.div(workspace.angle, 2 * math.pi, out=workspace.coefficient)
    torch.sinc(workspace.coefficient, out=workspace.coefficient)
    workspace.coefficient.square_().mul_(.5)
    torch.linalg.cross(omega, vector, dim=-1, out=workspace.first)
    torch.linalg.cross(omega, workspace.first, dim=-1, out=workspace.second)
    out.copy_(vector)
    out.addcmul_(workspace.sinc, workspace.first, value=time_step)
    out.addcmul_(workspace.coefficient, workspace.second, value=time_step**2)
    return out


class _GeometricMidpointWorkspace:
    """Fresh trajectory workspace; two force evaluations, no inner solves."""

    def __init__(self, vector):
        self.midpoint = torch.empty_like(vector)
        self.omega = torch.empty_like(vector)
        self.cross = torch.empty_like(vector)
        self.rotation = _RotationWorkspace(vector)

    def advance(self, vector, objective, u, v, start_controls, midpoint_controls,
                time_step, damping):
        _generator(vector, objective, u, v, start_controls, damping,
                   out=self.omega, cross=self.cross)
        _rodrigues(vector, self.omega, time_step / 2,
                   out=self.midpoint, workspace=self.rotation)
        _generator(self.midpoint, objective, u, v, midpoint_controls, damping,
                   out=self.omega, cross=self.cross)
        # The full rotation acts on n^k, not the half-step state.
        _rodrigues(vector, self.omega, time_step,
                   out=vector, workspace=self.rotation)


@dataclass(frozen=True)
class SU2ChiralIsospin(PopulationSolver):
    """Exploratory classical Bloch annealer; no quantum or optimum guarantee."""

    max_steps: int = 512
    time_step: float = .02
    damping: float = .3
    transverse_field: float = 1.
    penalty: float = 1.
    chirality: float = .2
    initial_noise: float = .1
    candidate_interval: int = 16
    normalize: bool = True
    dynamics_dtype: object = None
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('candidate_interval', self.candidate_interval, 1)
        positive('time_step', self.time_step)
        positive('damping', self.damping, zero=True)
        positive('transverse_field', self.transverse_field, zero=True)
        positive('penalty', self.penalty, zero=True)
        positive('initial_noise', self.initial_noise)
        if isinstance(self.chirality, bool) or not math.isfinite(self.chirality):
            raise ValueError('chirality must be finite')
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if self.dynamics_dtype not in (None, torch.float32, torch.float64,
                                        'float32', 'float64'):
            raise ValueError('dynamics_dtype must be float32, float64, or None')
        if (isinstance(self.sparse_threshold, bool)
                or not math.isfinite(self.sparse_threshold)
                or not 0 <= self.sparse_threshold <= 1):
            raise ValueError('sparse_threshold must be finite and within [0, 1]')

    def _population_workspace(self, n, batch, width):
        """Conservative trajectory buffers, beyond the common scorer/archive."""
        # Six 3-component tensors, rotation scalars, rank-two vectors, sampling
        # fallback and intermediate products. No dense K or extra target copy.
        return width * (48 * batch * n + 64 * batch)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        vector = run.normal((*run.x.shape, 3))
        vector[..., 0].fill_(1.)
        vector[..., 1].mul_(self.initial_noise)
        # Preserve the scored initial source-domain assignment while starting
        # near +x, away from stationary binary poles.
        signs = torch.where(run.initialbits, 1., -1.)
        vector[..., 2].abs_().clamp_min_(torch.finfo(vector.dtype).eps)
        vector[..., 2].mul_(self.initial_noise).mul_(signs)
        vector.div_(vector.norm(dim=-1, keepdim=True))
        u, v = (_rank_two_vectors(run.normal, run.x.shape)
                if self.chirality != 0. and run.x.shape[-1] > 1 else (None, None))
        work = _GeometricMidpointWorkspace(vector)
        for k in range(self.max_steps):
            observation.poll()
            start = observation.schedule_fraction(k / self.max_steps)
            midpoint = observation.schedule_fraction((k + .5) / self.max_steps)
            start_controls = _controls(start, self.transverse_field, self.penalty, self.chirality)
            midpoint_controls = _controls(midpoint, self.transverse_field, self.penalty, self.chirality)
            work.advance(vector, run.objective, u, v, start_controls, midpoint_controls,
                         self.time_step, self.damping)
            step = k + 1
            run.iterations.fill_(step)
            if (step % self.candidate_interval == 0 or step == self.max_steps
                    or (run.history_interval is not None
                        and step % run.history_interval == 0)):
                run.publish(vector[..., 2], step)
                record(step)
        # One terminal diagnostic; no per-step CUDA scalar synchronization.
        finite(vector)
        return self.max_steps
