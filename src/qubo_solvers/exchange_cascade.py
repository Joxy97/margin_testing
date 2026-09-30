"""Dense portable exchange-field cascade adapted from margin_testing.

Keeps the analytic field propagator, staggered order controls and terminal-frame
freeze. Float64 is additionally supported. No NumPy/SciPy or CUDA graphs needed.
"""

from dataclasses import dataclass
import math
import torch

from ._dynamics import IterativeSolver, SpinObjective, finite, normal, positive, publish, spins, unit
from .solvers import _integer

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
        # Match the original Python float arithmetic without extracting a CUDA
        # scalar. A zero-dimensional tensor retains scalar promotion semantics.
        self.orderScale = degree.sum().double().mul_(.5).clamp_min_(1.)
        self.orderScale.mul_(parameters['order_strength'])
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


@dataclass(frozen=True)
class ExchangeCascade(IterativeSolver):
    """margin_testing exchange cascade; iterations count integration steps."""

    max_steps: int = 500
    time_step: float = .1
    k: float = 1.
    xi: float = 4.
    terminal_mass: float = 2.

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        for name in ('time_step', 'k', 'xi', 'terminal_mass'):
            positive(name, getattr(self, name))
        if self.time_step > .25:
            raise ValueError('time_step must be <= .25')
        if self.terminal_mass <= self.k:
            raise ValueError('terminal_mass must exceed k')

    def _search(self, run, record):
        obj = SpinObjective(run, normalize=False)
        n, batch = run.x.shape[1]+1, run.x.shape[0]
        matrix = run.q.new_zeros((n, n))
        matrix[1:, 1:] = 2*obj.j
        matrix[0, 1:] = obj.h
        matrix[1:, 0] = obj.h
        degree = matrix.abs().sum(1)
        scale = degree.max().clamp_min(1)
        matrix /= scale
        degree /= scale
        vectors = unit(normal(run, (n, batch, 3)))
        # Reflect into the hemisphere encoding the required initial assignment.
        sign = torch.where((vectors[1:]*vectors[:1]).sum(-1) >= 0, 1., -1.)
        vectors[1:] *= (sign*spins(run).T)[..., None]
        parameters = dict(k=self.k, xi=self.xi, time_step=self.time_step, field_damping=1.,
                          spin_mobility=1., order_mobility=1., max_spin_step=.2,
                          order_strength=4., orthogonality=1., order_seed=.05,
                          terminal_mass=self.terminal_mass)
        work = _ExchangeWorkspace(torch, matrix, degree, vectors, parameters)
        freeze_step = min(self.max_steps-1, int(.75*self.max_steps))
        for step in range(self.max_steps):
            if step == freeze_step:
                work.freeze()
            progress = step/max(self.max_steps, 1)
            rho_a = run.q.new_tensor(max(-1., min(1., 2*(progress-.2)/.25-1)))
            rho_b = run.q.new_tensor(max(-1., min(1., 2*(progress-.45)/.3-1)))
            work.advance(rho_a, rho_b, terminal=step >= freeze_step)
            publish(run, 2*work.decoded().to(run.q.dtype)-1)
            run.iterations += 1
            record(step+1)
        finite(work.x, work.z, work.a, work.b)
        return self.max_steps
