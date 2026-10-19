"""Six product-state/phase Hamiltonian searches from the October 2026 specification.

The search coordinates are classical restricted states, not many-body quantum
simulations. Coefficients use PopulationObjective's *energy-gradient* convention
(opposite to the physical Ising field in the specification). The configurable
step-budget continuation includes a terminal relaxation fraction; backtracking
always freezes its controls. Auxiliary comparisons reduce in FP64, with a
roundoff allowance for dynamics-precision matrix products. Original-source FP64
scores alone select returned candidates. Numerical stalls retain the whole old
replica and use UNKNOWN termination status; they never commit a rejected trial.

All drivers are stencils or declared edge lists. Only the immutable logical
operator is shared across batches. Extra trajectories cost O(batch*n*q) for the
clock and O(batch*n) for the other default drivers; explicit J1/J2 graphs also
require O(batch*edges) temporary gathers, included in admission estimates.
"""

from dataclasses import dataclass
import math

import torch

from . import observation
from ._dynamics import finite, positive
from ._population import PopulationSolver
from .solvers import _integer


def _sum64(values, dims=-1):
    return values.sum(dim=dims, dtype=torch.float64)


def _target_energy(z, objective, gradient=None):
    """Normalized multilinear spin energy, omitting its constant."""
    gradient = objective.gradient(z) if gradient is None else gradient
    return .5 * _sum64(z.double() * (gradient.double() + objective.field.double()))


def _unit(values):
    return values / values.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(values.real.dtype).tiny)


def _controls(step, steps, relaxation_fraction):
    """Driver tau and pinning fraction, with exact terminal driver removal."""
    progress = 1. if steps == 1 else step / max(1, steps - 1)
    progress = observation.schedule_fraction(progress)
    progress = min(1., max(0., progress / (1 - relaxation_fraction)))
    tau = (1 - progress)**2 if progress < 1. else 0.
    return tau, 1 - tau


def _due(run, step, interval, maximum):
    return (step % interval == 0 or step == maximum
            or (run.history_interval is not None and step % run.history_interval == 0))


class _Stencil:
    """Symmetric open/periodic exchange; tiny rings have one unique edge."""

    def __init__(self, terms, periodic=False):
        self.terms, self.periodic = terms, periodic

    def action(self, values):
        out = torch.zeros_like(values)
        n = values.shape[1]
        for distance, coefficient in self.terms:
            if coefficient == 0. or distance >= n:
                continue
            if self.periodic:
                out.add_(torch.roll(values, distance, 1), alpha=coefficient)
                # For n=2 a nearest-neighbor ring has one edge, not two.
                if 2 * distance != n:
                    out.add_(torch.roll(values, -distance, 1), alpha=coefficient)
            else:
                out[:, distance:].add_(values[:, :-distance], alpha=coefficient)
                out[:, :-distance].add_(values[:, distance:], alpha=coefficient)
        return out


class _Edges:
    """Explicit two-graph driver, adding shared and repeated edge coefficients."""

    def __init__(self, n, device, edge_sets):
        self.groups = []
        for edges, coefficient in edge_sets:
            for edge in edges:
                if (len(edge) != 2 or any(isinstance(i, bool) or not isinstance(i, int)
                                          or i < 0 or i >= n for i in edge)
                        or edge[0] == edge[1]):
                    raise ValueError('driver edges must contain distinct in-range integer vertices')
            if edges and coefficient != 0.:
                indices = torch.tensor(edges, device=device, dtype=torch.int64).T
                self.groups.append((indices[0], indices[1], coefficient))

    def action(self, values):
        out = torch.zeros_like(values)
        for heads, tails, coefficient in self.groups:
            out.index_add_(1, heads, values.index_select(1, tails), alpha=coefficient)
            out.index_add_(1, tails, values.index_select(1, heads), alpha=coefficient)
        return out


def _driver_energy(values, driver):
    return .5 * _sum64((values.conj() * driver.action(values)).real,
                       tuple(range(1, values.ndim)))


def _backtrack(state, energy, propose, energy_fn, dt, trials, active=None):
    """Joint per-replica acceptance, retaining every coordinate on rejection."""
    count = state.shape[0]
    shape = (count,) + (1,) * (state.ndim - 1)
    pending = (torch.ones(count, dtype=torch.bool, device=state.device)
               if active is None else active.clone())
    accepted = torch.zeros_like(pending)
    result = state.clone()
    step = torch.as_tensor(dt, device=state.device, dtype=state.real.dtype)
    step = step.expand(count).clone()
    # The products defining energy use dynamics precision; FP64 reductions
    # cannot eliminate their FP32 error. Include that actual roundoff scale.
    tolerance = (1e-10 + 8 * torch.finfo(state.real.dtype).eps) * (1 + energy.abs())
    for _ in range(trials):
        observation.poll()
        trial = propose(step.reshape(shape))
        candidate = energy_fn(trial)
        good = (pending & torch.isfinite(candidate)
                & torch.isfinite(trial).reshape(count, -1).all(-1)
                & (candidate <= energy + tolerance))
        result.copy_(torch.where(good.reshape(shape), trial, result))
        accepted |= good
        pending &= ~good
        if not bool(pending.any()):
            break
        step.mul_(.5)
    return result, accepted


def _spin_gradient(vector, objective, driver, tau, pinning, *, with_energy=False):
    exchange = driver.action(vector)
    gradient = tau * exchange
    n = objective.field.numel()
    target = objective.gradient(vector[:, :n, 2])
    gradient[..., 2].add_(vector[..., 2], alpha=-2 * pinning)
    gradient[:, :n, 2].add_(target)
    if with_energy:
        energy = (_target_energy(vector[:, :n, 2], objective, target)
                  + pinning * _sum64(1 - vector[..., 2].double().square())
                  + .5 * tau * _sum64(vector.double() * exchange.double(), (-1, -2)))
        return gradient, energy
    return gradient


def _spin_energy(vector, objective, driver, tau, pinning):
    n = objective.field.numel()
    return (_target_energy(vector[:, :n, 2], objective)
            + pinning * _sum64(1 - vector[..., 2].double().square())
            + tau * _driver_energy(vector, driver))


def _llg_velocity(vector, gradient, damping, precession_sign):
    cross = torch.linalg.cross(vector, gradient)
    return precession_sign * cross + damping * torch.linalg.cross(vector, cross)


def _phase_gradient(theta, objective, driver, tau, pinning, *, with_energy=False):
    sine, cosine = theta.sin(), theta.cos()
    target = objective.gradient(cosine)
    sine_exchange, cosine_exchange = driver.action(sine), driver.action(cosine)
    gradient = (-sine * target + 2 * pinning * sine * cosine
                + tau * (sine * cosine_exchange - cosine * sine_exchange))
    if with_energy:
        energy = (_target_energy(cosine, objective, target)
                  + pinning * _sum64(sine.double().square())
                  - .5 * tau * _sum64(sine.double() * sine_exchange.double()
                                      + cosine.double() * cosine_exchange.double()))
        return gradient, energy
    return gradient


def _phase_energy(theta, objective, driver, tau, pinning):
    sine, cosine = theta.sin(), theta.cos()
    return (_target_energy(cosine, objective) + pinning * _sum64(sine.double().square())
            - tau * (_driver_energy(sine, driver) + _driver_energy(cosine, driver)))


def _rotor_step(theta, momentum, objective, driver, *, tau, pinning, mass,
                damping, temperature, time_step, normal):
    """BAOAB: both kicks use forces from the complete corresponding state."""
    momentum = momentum - .5 * time_step * _phase_gradient(
        theta, objective, driver, tau, pinning)
    theta = theta + .5 * time_step / mass * momentum
    decay = math.exp(-damping * time_step)
    momentum = decay * momentum
    if temperature > 0. and damping > 0.:
        momentum = momentum + math.sqrt(mass * temperature * (-math.expm1(
            -2 * damping * time_step))) * normal(momentum.shape)
    theta = (theta + .5 * time_step / mass * momentum + math.pi).remainder(2 * math.pi) - math.pi
    momentum = momentum - .5 * time_step * _phase_gradient(
        theta, objective, driver, tau, pinning)
    return theta, momentum


def _clock_moments(spinor, diagonal, clock):
    probability = spinor.abs().square()
    p = (probability * diagonal).sum(-1)
    z = (probability * clock).sum(-1)
    return p, z


def _clock_gradient(spinor, objective, driver, diagonal, clock, tau, pinning, field,
                    *, with_energy=False):
    p, z = _clock_moments(spinor, diagonal, clock)
    target = objective.gradient(2 * p - 1)
    logical = 2 * target + pinning * (1 - 2 * p)
    exchange = driver.action(z)
    local = logical[..., None] * diagonal
    local = local - tau * (exchange[..., None] * clock.conj()).real
    gradient = local * spinor
    gradient = gradient - .5 * tau * field * (
        torch.roll(spinor, 1, -1) + torch.roll(spinor, -1, -1))
    if with_energy:
        shift = (spinor.conj() * torch.roll(spinor, 1, -1)).sum(-1).real
        energy = (_target_energy(2 * p - 1, objective, target)
                  + pinning * _sum64(p.double() * (1 - p.double()))
                  - tau * (.5 * _sum64((z.conj() * exchange).real) + field * _sum64(shift)))
        return gradient, energy
    return gradient


def _clock_energy(spinor, objective, driver, diagonal, clock, tau, pinning, field):
    p, z = _clock_moments(spinor, diagonal, clock)
    shift = (spinor.conj() * torch.roll(spinor, 1, -1)).sum(-1).real
    return (_target_energy(2 * p - 1, objective)
            + pinning * _sum64(p.double() * (1 - p.double()))
            - tau * (_driver_energy(z, driver) + field * _sum64(shift)))


def _squeezing_gradient(vector, objective, amplitude, twisting, eta, field):
    gradient = torch.empty_like(vector)
    x, y, z = vector.unbind(-1)
    gradient[..., 0] = .5 * amplitude * twisting * (x.sum(-1, keepdim=True) - x) - amplitude * field
    gradient[..., 1] = -.5 * amplitude * twisting * eta * (y.sum(-1, keepdim=True) - y)
    gradient[..., 2] = objective.gradient(z)
    return gradient


def _squeezing_energy(vector, objective, amplitude, twisting, eta, field, *, target_gradient=None):
    x, y, z = vector.double().unbind(-1)
    n = vector.shape[1]
    twist = ((1 - eta) * n + x.sum(-1).square() - x.square().sum(-1)
             - eta * (y.sum(-1).square() - y.square().sum(-1)))
    return (_target_energy(vector[..., 2], objective, target_gradient) - amplitude * field * x.sum(-1)
            + .25 * amplitude * twisting * twist)


def _rotate(vector, omega, dt):
    """Stable Rodrigues formula, also valid at exactly zero angular velocity."""
    angle = dt * omega.norm(dim=-1, keepdim=True)
    first = torch.linalg.cross(omega, vector)
    second = torch.linalg.cross(omega, first)
    result = (vector + dt * torch.sinc(angle / math.pi) * first
              + .5 * dt.square() * torch.sinc(angle / (2 * math.pi)).square() * second)
    return _unit(result)


@dataclass(frozen=True)
class _HamiltonianBase(PopulationSolver):
    max_steps: int = 512
    time_step: float = .02
    candidate_interval: int = 10
    relaxation_fraction: float = 1 / 6
    backtracking_steps: int = 20
    polish_sweeps: int = 0
    normalize: bool = True
    dynamics_dtype: object = None
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('candidate_interval', self.candidate_interval, 1)
        _integer('backtracking_steps', self.backtracking_steps, 1)
        if self.backtracking_steps > 20:
            raise ValueError('backtracking_steps must not exceed 20')
        _integer('polish_sweeps', self.polish_sweeps, 0)
        positive('time_step', self.time_step)
        if (isinstance(self.relaxation_fraction, bool)
                or not math.isfinite(self.relaxation_fraction)
                or not 0 <= self.relaxation_fraction < 1):
            raise ValueError('relaxation_fraction must be within [0, 1)')
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if self.dynamics_dtype not in (None, torch.float32, torch.float64, 'float32', 'float64'):
            raise ValueError('dynamics_dtype must be float32, float64, or None')
        if (isinstance(self.sparse_threshold, bool) or not math.isfinite(self.sparse_threshold)
                or not 0 <= self.sparse_threshold <= 1):
            raise ValueError('sparse_threshold must be within [0, 1]')

    def _population_workspace(self, n, batch, width):
        # Old/trial/gradient, trigonometric/rotation and normalization scratch;
        # FP64 reductions and explicit polish candidates, beyond common buffers.
        return (80 * width + 64) * batch * max(n, 6) + 512 * batch

    def _checkpoint(self, run, record, values, step):
        run.iterations.fill_(step)
        if _due(run, step, self.candidate_interval, self.max_steps):
            run.publish(values, step)
            record(step)

    def _finish(self, run, record, *state):
        finite(*state)
        if self.polish_sweeps:
            # Independent replicas may commit in parallel; sites are sequential.
            # Exact source scores make this valid for duplicates and self-loops.
            samples, energies = run.best_x.clone(), run.best.clone()
            for _ in range(self.polish_sweeps):
                changed = torch.zeros_like(energies, dtype=torch.bool)
                for i in range(samples.shape[1]):
                    observation.poll()
                    trial = samples.clone()
                    trial[:, i] = -trial[:, i] if run.spin else 1 - trial[:, i]
                    candidate = run.score(trial)
                    accept = candidate < energies
                    samples.copy_(torch.where(accept[:, None], trial, samples))
                    energies.copy_(torch.where(accept, candidate, energies))
                    changed |= accept
                if not bool(changed.any()):
                    break
            run.x.copy_(samples)
            run.energies.copy_(energies)
            run.retain(samples, energies)
            record(self.max_steps)
        return self.max_steps


@dataclass(frozen=True)
class QuantumClock(_HamiltonianBase):
    """Even-q local complex spinors with cyclic coherence and clock exchange."""

    q: int = 4
    damping: float = 1.
    precession: float = 1.
    transverse_field: float = 1.
    exchange: float = .25
    pinning: float = 1.
    initial_noise: float = .05

    def __post_init__(self):
        super().__post_init__()
        _integer('q', self.q, 4)
        if self.q % 2:
            raise ValueError('q must be even')
        positive('damping', self.damping)
        for name in ('transverse_field', 'exchange', 'pinning', 'initial_noise'):
            positive(name, getattr(self, name), zero=True)
        if isinstance(self.precession, bool) or not math.isfinite(self.precession):
            raise ValueError('precession must be finite')

    def _population_workspace(self, n, batch, width):
        return super()._population_workspace(n, batch, width) + 40 * width * batch * n * self.q

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        shape = (*run.x.shape, self.q)
        spinor = torch.complex(run.normal(shape), run.normal(shape)) * self.initial_noise
        spinor.real.add_(1 / math.sqrt(self.q))
        spinor = _unit(spinor)
        angles = torch.arange(self.q, device=run.q.device, dtype=run.q.dtype) * (2 * math.pi / self.q)
        clock = torch.polar(torch.ones_like(angles), angles)
        diagonal = (1 + angles.cos()) * .5
        # The mathematical poles are exact, independent of trig roundoff.
        diagonal[0], diagonal[self.q // 2] = 1., 0.
        driver = _Stencil(((1, self.exchange),), periodic=True)
        active = torch.ones(run.x.shape[0], device=run.q.device, dtype=torch.bool)
        for k in range(self.max_steps):
            observation.poll()
            tau, fraction = _controls(k, self.max_steps, self.relaxation_fraction)
            pinning = self.pinning * fraction
            gradient, energy = _clock_gradient(spinor, run.objective, driver, diagonal, clock,
                tau, pinning, self.transverse_field, with_energy=True)
            tangent = gradient - (spinor.conj() * gradient).sum(-1, keepdim=True).real * spinor
            velocity = -(self.damping + 1j * self.precession) * tangent
            energy_fn = lambda value: _clock_energy(value, run.objective, driver, diagonal,
                                                    clock, tau, pinning, self.transverse_field)
            spinor, accepted = _backtrack(spinor, energy,
                lambda dt: _unit(spinor + dt * velocity), energy_fn, self.time_step,
                self.backtracking_steps, active)
            active &= accepted
            run.reasons.masked_fill_(~active, -1)
            p, _ = _clock_moments(spinor, diagonal, clock)
            self._checkpoint(run, record, p - .5, k + 1)
        return self._finish(run, record, spinor)


class _ExchangeSearch:
    def _exchange_search(self, run, record, *, padded, paired):
        if self.max_steps == 0:
            return 0
        n = run.x.shape[1]
        size = max(6, n + n % 2) if padded else n
        vector = run.normal((run.x.shape[0], size, 3))
        if paired:
            pair = run.normal((run.x.shape[0], size // 2, 2))
            pair = _unit(pair)
            vector[:, ::2, :2] = pair
            vector[:, 1::2, :2] = -pair
            vector[..., 2].zero_()
            translated = torch.arange(vector.shape[0], device=vector.device) % 2 == 1
            vector.copy_(torch.where(translated[:, None, None], torch.roll(vector, 1, 1), vector))
            vector.add_(run.normal(vector.shape), alpha=self.initial_noise)
        else:
            vector[..., 2].mul_(self.initial_noise)
        vector = _unit(vector)
        driver = self._driver(size, vector.device)
        active = torch.ones(vector.shape[0], device=vector.device, dtype=torch.bool)
        for k in range(self.max_steps):
            observation.poll()
            tau, fraction = _controls(k, self.max_steps, self.relaxation_fraction)
            pinning = self.pinning * fraction
            gradient, energy = _spin_gradient(vector, run.objective, driver, tau, pinning, with_energy=True)
            # MG/J1J2 use B=-gradient, u cross B - lambda u cross(u cross B).
            velocity = _llg_velocity(vector, gradient, self.damping, -1.)
            energy_fn = lambda value: _spin_energy(value, run.objective, driver, tau, pinning)
            vector, accepted = _backtrack(vector, energy,
                lambda dt: _unit(vector + dt * velocity), energy_fn,
                self.time_step, self.backtracking_steps, active)
            active &= accepted
            run.reasons.masked_fill_(~active, -1)
            self._checkpoint(run, record, vector[:, :n, 2], k + 1)
        return self._finish(run, record, vector)


@dataclass(frozen=True)
class MajumdarGhosh(_ExchangeSearch, _HamiltonianBase):
    """Padded frustrated A/4,A/8 ring with translated antiparallel pairs."""

    exchange: float = 1.
    damping: float = 1.
    pinning: float = .5
    initial_noise: float = .05

    def __post_init__(self):
        super().__post_init__()
        for name in ('exchange', 'pinning', 'initial_noise'):
            positive(name, getattr(self, name), zero=True)
        positive('damping', self.damping)

    def _driver(self, n, device):
        return _Stencil(((1, self.exchange / 4), (2, self.exchange / 8)), periodic=True)

    def _population_workspace(self, n, batch, width):
        return super()._population_workspace(max(6, n + n % 2), batch, width)

    def _search(self, run, record):
        return self._exchange_search(run, record, padded=True, paired=True)


@dataclass(frozen=True)
class J1J2(_ExchangeSearch, _HamiltonianBase):
    """Two explicitly configurable exchange graphs; default open 1/2 chain."""

    exchange1: float = 1.
    exchange2: float = .7
    edges1: object = None
    edges2: object = None
    damping: float = 1.
    pinning: float = .5
    initial_noise: float = .05

    def __post_init__(self):
        super().__post_init__()
        for name in ('exchange1', 'exchange2'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        for name in ('edges1', 'edges2'):
            edges = getattr(self, name)
            if edges is not None:
                try:
                    edges = tuple(tuple(edge) for edge in edges)
                except TypeError as error:
                    raise ValueError('driver edges must be pairs of vertices') from error
                if any(len(edge) != 2 for edge in edges):
                    raise ValueError('driver edges must be pairs of vertices')
                object.__setattr__(self, name, edges)
        positive('damping', self.damping)
        positive('pinning', self.pinning, zero=True)
        positive('initial_noise', self.initial_noise, zero=True)

    def _driver(self, n, device):
        if self.edges1 is None and self.edges2 is None:
            return _Stencil(((1, self.exchange1 / 4), (2, self.exchange2 / 4)))
        first = tuple((i, i + 1) for i in range(n - 1)) if self.edges1 is None else self.edges1
        second = tuple((i, i + 2) for i in range(n - 2)) if self.edges2 is None else self.edges2
        return _Edges(n, device, ((first, self.exchange1 / 4), (second, self.exchange2 / 4)))

    def _population_workspace(self, n, batch, width):
        edges = sum(max(0, n - distance) if graph is None else len(graph)
                    for distance, graph in ((1, self.edges1), (2, self.edges2)))
        return super()._population_workspace(n, batch, width) + edges * (32 + 18 * width * batch)

    def _search(self, run, record):
        return self._exchange_search(run, record, padded=False, paired=False)


@dataclass(frozen=True)
class Kuramoto(_HamiltonianBase):
    """Full logical cosine target, ring synchronization and annealed EM noise."""

    exchange: float = .25
    pinning: float = .5
    frequency_std: float = .1
    diffusion: float = .05

    def __post_init__(self):
        super().__post_init__()
        for name in ('exchange', 'pinning', 'frequency_std', 'diffusion'):
            positive(name, getattr(self, name), zero=True)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        theta = torch.empty_like(run.x).uniform_(0, 2 * math.pi, generator=run.generator)
        frequency = run.normal(run.x.shape) * self.frequency_std
        frequency.sub_(frequency.mean(-1, keepdim=True))
        driver = _Stencil(((1, self.exchange),), periodic=True)
        active = torch.ones(theta.shape[0], device=theta.device, dtype=torch.bool)
        for k in range(self.max_steps):
            observation.poll()
            tau, fraction = _controls(k, self.max_steps, self.relaxation_fraction)
            pinning = self.pinning * fraction
            gradient, energy = _phase_gradient(theta, run.objective, driver, tau, pinning, with_energy=True)
            drift = tau * frequency - gradient
            if tau > 0. and (self.diffusion > 0. or self.frequency_std > 0.):
                theta.add_(drift, alpha=self.time_step)
                if self.diffusion > 0.:
                    theta.add_(run.normal(theta.shape), alpha=math.sqrt(
                        2 * self.diffusion * tau * self.time_step))
                theta.remainder_(2 * math.pi)
            else:
                energy_fn = lambda value: _phase_energy(value, run.objective, driver, tau, pinning)
                theta, accepted = _backtrack(theta, energy,
                    lambda dt: (theta + dt * drift).remainder(2 * math.pi), energy_fn,
                    self.time_step, self.backtracking_steps, active)
                active &= accepted
                run.reasons.masked_fill_(~active, -1)
            self._checkpoint(run, record, theta.cos(), k + 1)
        return self._finish(run, record, theta)


@dataclass(frozen=True)
class QuantumRotor(_HamiltonianBase):
    """Classical inertial rotor with exact OU thermalization and two force kicks."""

    time_step: float = .05
    mass: float = 1.
    damping: float = 1.
    exchange: float = .25
    pinning: float = 2.
    temperature: float = .5
    final_temperature: float = .001

    def __post_init__(self):
        super().__post_init__()
        positive('mass', self.mass)
        for name in ('damping', 'exchange', 'pinning', 'temperature', 'final_temperature'):
            positive(name, getattr(self, name), zero=True)
        if self.final_temperature > self.temperature:
            raise ValueError('final_temperature must not exceed temperature')

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        theta = torch.empty_like(run.x).uniform_(-math.pi, math.pi, generator=run.generator)
        momentum = run.normal(run.x.shape) * math.sqrt(self.mass * self.temperature)
        # Unit-row-sum ring: n=2 has a single weight-one edge; n=1 no driver.
        weight = 1. if theta.shape[1] == 2 else .5
        driver = _Stencil(((1, weight),), periodic=True)
        matrix = run.objective.matrix
        if matrix.layout == torch.strided:
            row = matrix.abs().sum(-1)
        else:
            rows = torch.repeat_interleave(torch.arange(matrix.shape[0], device=matrix.device),
                matrix.crow_indices()[1:] - matrix.crow_indices()[:-1],
                output_size=matrix.values().numel())
            row = torch.zeros_like(run.objective.field).index_add_(0, rows, matrix.values().abs())
        bound = float((run.objective.field.abs() + 2 * row).max())
        bound += 2 * self.pinning + 2 * self.exchange
        dt = min(self.time_step, .25 * math.sqrt(self.mass / bound)) if bound > 0 else self.time_step
        for k in range(self.max_steps):
            observation.poll()
            progress = 1. if self.max_steps == 1 else k / (self.max_steps - 1)
            progress = min(1., observation.schedule_fraction(progress) / (1 - self.relaxation_fraction))
            tau = self.exchange * (1 - progress)**2
            pinning = self.pinning * progress**2
            temperature = (self.temperature * (self.final_temperature / self.temperature)**progress
                           if self.temperature > 0. and self.final_temperature > 0. else
                           self.temperature * (1 - progress))
            if progress >= 1.:
                tau, temperature = 0., 0.
            theta, momentum = _rotor_step(theta, momentum, run.objective, driver,
                tau=tau, pinning=pinning, mass=self.mass, damping=self.damping,
                temperature=temperature, time_step=dt, normal=run.normal)
            self._checkpoint(run, record, theta.cos(), k + 1)
        return self._finish(run, record, theta, momentum)


@dataclass(frozen=True)
class SpinSqueezing(_HamiltonianBase):
    """Exact collective product expectation, including Pauli self corrections."""

    time_step: float = .05
    damping: float = .2
    twisting: float = 1.
    eta: float = 1.
    transverse_field: float = 2.
    initial_noise: float = .1

    def __post_init__(self):
        super().__post_init__()
        for name in ('damping', 'transverse_field', 'initial_noise'):
            positive(name, getattr(self, name), zero=True)
        for name in ('twisting', 'eta'):
            if isinstance(getattr(self, name), bool) or not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        vector = run.normal((*run.x.shape, 3))
        near = torch.arange(vector.shape[0], device=vector.device) % 2 == 0
        start = vector * self.initial_noise
        start[..., 0] = 1.
        vector = _unit(torch.where(near[:, None, None], start, vector))
        twisting = self.twisting / vector.shape[1]
        active = torch.ones(vector.shape[0], device=vector.device, dtype=torch.bool)
        for k in range(self.max_steps):
            observation.poll()
            amplitude, _ = _controls(k, self.max_steps, self.relaxation_fraction)
            gradient = _squeezing_gradient(vector, run.objective, amplitude, twisting,
                                           self.eta, self.transverse_field)
            omega = -gradient - self.damping * torch.linalg.cross(vector, gradient)
            maximum = omega.norm(dim=-1).amax(-1).clamp_min(torch.finfo(vector.dtype).tiny)
            dt = torch.minimum(torch.full_like(maximum, self.time_step), .25 / maximum)
            energy_fn = lambda value: _squeezing_energy(value, run.objective, amplitude,
                                                        twisting, self.eta, self.transverse_field)
            if self.damping == 0.:
                vector = _rotate(vector, omega, dt[:, None, None])
            else:
                energy = _squeezing_energy(vector, run.objective, amplitude, twisting,
                    self.eta, self.transverse_field, target_gradient=gradient[..., 2])
                vector, accepted = _backtrack(vector, energy,
                    lambda step: _rotate(vector, omega, step), energy_fn,
                    dt, self.backtracking_steps, active)
                active &= accepted
                run.reasons.masked_fill_(~active, -1)
            self._checkpoint(run, record, vector[..., 2], k + 1)
        return self._finish(run, record, vector)


HAMILTONIAN_SPIN_SOLVERS = {
    'lib_quantum_clock': QuantumClock,
    'lib_majumdar_ghosh': MajumdarGhosh,
    'lib_j1_j2': J1J2,
    'lib_kuramoto': Kuramoto,
    'lib_quantum_rotor': QuantumRotor,
    'lib_spin_squeezing': SpinSqueezing,
}
