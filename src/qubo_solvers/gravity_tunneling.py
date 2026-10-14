"""Softened configuration-space gravity and classical binary barrier jumps.

Implements the frozen-weight, reflected dissipative Verlet design in
``gravity_tunneling_solver.tex``. Trajectories can use reduced precision, while
every decoded population and proposal is ranked by the source-objective scorer.
The terminal controls describe the intended objective; they do not certify
convergence or establish an advantage over existing optimization methods.
"""

from dataclasses import dataclass
import math

import torch

from ._dynamics import finite, positive
from ._population import PopulationSolver
from .observation import poll
from .solvers import _integer


def _decode(positions):
    """Spin decoding with the specified positive tie at zero."""
    return (positions >= 0).to(positions.dtype).mul_(2).sub_(1)


def _softened_gravity(positions, weights, gravity, softening, *, out=None,
                      pair=None, squares=None):
    """Compute pair forces using B-by-B storage, never B-by-B-by-n storage."""
    batch, n = positions.shape
    if out is None:
        out = torch.empty_like(positions)
    if pair is None:
        pair = positions.new_empty((batch, batch))
    if squares is None:
        squares = torch.empty_like(positions)
    torch.mul(positions, positions, out=squares)
    norms = squares.sum(1)
    torch.mm(positions, positions.T, out=pair)
    pair.mul_(-2).add_(norms[:, None]).add_(norms[None, :]).div_(n)
    pair.clamp_min_(0).add_(softening * softening).pow_(-1.5)
    pair.mul_(weights[:, None]).mul_(weights[None, :]).mul_(gravity / n)
    pair.diagonal().zero_()
    torch.mm(pair, positions, out=out)
    out.addcmul_(pair.sum(1, keepdim=True), positions, value=-1)
    return out


def _reflect_drift(positions, momenta, duration):
    """In-place specular drift, including arbitrary crossings and wall ties."""
    folded = (positions + duration * momenta + 1).remainder_(4)
    positions.copy_(1 - (folded - 2).abs())
    # At -1 the outgoing momentum is positive; at +1 it is negative.
    # Choosing signs explicitly handles an exact endpoint after any crossings.
    reflected = torch.where(folded < 2, momenta, -momenta)
    reflected = torch.where(folded == 0, momenta.abs(), reflected)
    reflected = torch.where(folded == 2, -momenta.abs(), reflected)
    momenta.copy_(reflected)
    return positions, momenta


def _feedback_weights(energies, scale, mix, temperature, *, dtype=None):
    """Weights describe independently scored current positions, not archives."""
    weights = torch.softmax(-(energies - energies.min()) / (scale * temperature), dim=0)
    weights.mul_(mix).add_((1 - mix) / energies.shape[0])
    return weights.to(dtype=dtype) if dtype is not None else weights


def _jump_proposal(spins, generator, uniform_probability):
    """Symmetric log-uniform Hamming masks with a persistent uniform mixture."""
    batch, n = spins.shape
    options = dict(device=spins.device, dtype=spins.dtype, generator=generator)
    distance_draw = torch.rand((batch, 1), **options)
    probabilities = (.5 if n == 1 else
                     (distance_draw * math.log(n / 2) - math.log(n)).exp())
    flips = torch.rand(spins.shape, **options) < probabilities
    proposal = torch.where(flips, -spins, spins)
    uniform_branch = torch.rand((batch, 1), **options) < uniform_probability
    uniform_spins = (torch.rand(spins.shape, **options) < .5).to(spins.dtype).mul_(2).sub_(1)
    return torch.where(uniform_branch, uniform_spins, proposal)


def _metropolis_accept(current, proposed, scale, temperature, generator):
    """Relocation acceptance uses original objective differences and S*tau."""
    if temperature == 0:
        return proposed <= current
    uniform = torch.rand(current.shape, device=current.device, dtype=current.dtype,
                         generator=generator)
    delta = proposed - current
    return (delta <= 0) | (uniform.log() < -delta / (scale * temperature))


class _GravityWorkspace:
    """Fixed-inertia state and reusable force buffers for one interacting swarm."""

    def __init__(self, run, solver):
        self.run, self.solver = run, solver
        if solver.relocation_radius < torch.finfo(run.q.dtype).tiny:
            raise ValueError('relocation_radius is too small for dynamics precision')
        spins = (run.x if run.spin else 2 * run.x - 1).to(run.q.dtype)
        magnitudes = torch.rand(spins.shape, device=spins.device, dtype=spins.dtype,
                                generator=run.generator).mul_(.8).add_(.1)
        self.q = spins * magnitudes
        self.p = run.normal(spins.shape).mul_(.01)
        self.force_buffer = torch.empty_like(self.q)
        self.local_buffer = torch.empty_like(self.q)
        self.gravity_buffer = torch.empty_like(self.q)
        self.pair_buffer = self.q.new_empty((spins.shape[0], spins.shape[0]))
        self.weights = _feedback_weights(run.energies, run.objective.scale,
                                        solver.weight_mix, solver.weight_temperature,
                                        dtype=self.q.dtype)
        self.controls = solver._controls(0., *spins.shape)
        self.damping_half = math.exp(-solver.damping * solver.time_step / 2)

    def force(self):
        a, penalty, gravity, _ = self.controls
        force = self.run.objective.gradient(self.q, out=self.force_buffer)
        force.mul_(-a)
        torch.mul(self.q, self.q, out=self.local_buffer)
        self.local_buffer.sub_(1).mul_(self.q).mul_(-penalty)
        force.add_(self.local_buffer)
        if gravity:
            _softened_gravity(self.q, self.weights, gravity, self.solver.softening,
                              out=self.gravity_buffer, pair=self.pair_buffer,
                              squares=self.local_buffer)
            force.add_(self.gravity_buffer)
        return force

    def advance(self):
        duration = self.solver.time_step
        # The same controls and weights are used by both kicks.
        self.p.mul_(self.damping_half).add_(self.force(), alpha=duration / 2)
        _reflect_drift(self.q, self.p, duration)
        self.p.add_(self.force(), alpha=duration / 2).mul_(self.damping_half)

    def jump(self, step, temperature):
        run = self.run
        spins = _decode(self.q)
        proposed = _jump_proposal(spins, run.generator, self.solver.uniform_jump_probability)
        samples = run.to_assignments(proposed)
        energies = run.score(samples)
        # Every proposal enters the authoritative archive and observer, even
        # when relocation is rejected or leaves the decoded state unchanged.
        run.retain(samples, energies)
        run.observe(samples, energies, phase='proposal', iteration=step)
        accepted = _metropolis_accept(run.energies, energies, run.objective.scale,
                                      temperature, run.generator)
        changed = (proposed != spins).any(1)
        relocate = (accepted & changed)[:, None]
        self.q.copy_(torch.where(relocate, self.solver.relocation_radius * proposed, self.q))
        self.p.masked_fill_(relocate, 0)
        # Each resulting row is an already source-scored current/proposal row.
        # Preserve both mandatory captures without scoring those bits again.
        torch.where(relocate, samples, run.x, out=run.x)
        torch.where(relocate[:, 0], energies, run.energies, out=run.energies)
        run.iterations.fill_(step)


@dataclass(frozen=True)
class GravityTunneling(PopulationSolver):
    """Exploratory fixed-inertia population dynamics with archived binary jumps.

    Bodies within a batch interact. Changing ``batch_size`` changes the swarm
    and hence the algorithm; reproducibility requires matching batch settings.
    Controls follow the fixed integration-step schedule, including when an
    observer requests wall-time controls; this method has no wall-time variant.
    Original FP64 scoring ranks every candidate, but FP64 alone is not a proof
    of exact ranking for arbitrary floating coefficients.
    """

    max_steps: int = 512
    time_step: float = .02
    damping: float = .3
    penalty: float = 1.
    gravity: float = .03
    softening: float = .5
    weight_mix: float = .5
    weight_temperature: float = 1.
    start_temperature: float = .2
    uniform_jump_probability: float = .05
    relocation_radius: float = .95
    candidate_interval: int = 16
    normalize: bool = True
    dynamics_dtype: object = None
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('candidate_interval', self.candidate_interval, 1)
        for name in ('time_step', 'softening', 'weight_temperature', 'relocation_radius',
                     'uniform_jump_probability'):
            positive(name, getattr(self, name))
        for name in ('damping', 'penalty', 'gravity', 'start_temperature', 'weight_mix',
                     'sparse_threshold'):
            positive(name, getattr(self, name), zero=True)
        if self.weight_mix >= 1:
            raise ValueError('weight_mix must be < 1')
        for name in ('relocation_radius', 'uniform_jump_probability', 'sparse_threshold'):
            if getattr(self, name) > 1:
                raise ValueError(f'{name} must be <= 1')
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be a bool')
        if self.dynamics_dtype not in (None, torch.float32, torch.float64, 'float32', 'float64'):
            raise ValueError('dynamics_dtype must be float32, float64, or None')

    def _population_workspace(self, n, batch, width):
        # Extra live buffers and conservative temporary allowance, beyond the
        # common immutable matrices, FP64 scorer and per-body archive. There
        # is deliberately no B^2*n term or max_steps-sized allocation.
        return (width * (24 * batch * n + 4 * batch * batch + 32 * batch)
                + 8 * (4 * batch * n + 12 * batch)
                + 8 * batch * n + 8 * batch)

    def _controls(self, progress, batch, n):
        remaining = (1 - progress) ** 2
        return (progress, self.penalty * progress ** 2,
                batch * n * self.gravity * remaining,
                self.start_temperature * remaining)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        work = _GravityWorkspace(run, self)
        batch, n = work.q.shape
        for step in range(1, self.max_steps + 1):
            poll()
            work.advance()
            candidate = step % self.candidate_interval == 0 or step == self.max_steps
            history = run.history_interval is not None and step % run.history_interval == 0
            if candidate:
                run.publish(_decode(work.q), step)
                run.observe(run.x, run.energies, phase='checkpoint', iteration=step)
                progress = step / self.max_steps
                next_controls = self._controls(progress, batch, n)
                work.jump(step, next_controls[3])
                work.weights = _feedback_weights(run.energies, run.objective.scale,
                                                 self.weight_mix, self.weight_temperature,
                                                 dtype=work.q.dtype)
                # Final controls are exactly a=1, G=0, kappa=penalty. This
                # bookkeeping does not integrate or minimize the terminal H.
                work.controls = next_controls
                record(step)
            elif history:
                run.publish(_decode(work.q), step)
                record(step)
        finite(work.q, work.p)
        return self.max_steps
