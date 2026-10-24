"""Compendium III, entries 59--86: bounded classical QUBO search engines.

The named mechanisms are engineered proposals, not many-body quantum
simulators. Every incumbent and binary acceptance uses original FP64 source
energies. Shared objective storage is never replicated per trajectory.
Directional operators use declared periodic variable-order stencils; they do
not assert a physical embedding of the supplied graph. Code-family entries
require explicit decoder metadata and never infer a code from a bare QUBO.
"""

from dataclasses import dataclass
import math
from typing import ClassVar

import torch

from . import observation
from ._dynamics import finite, positive
from ._population import PopulationSolver, _mm, _dtype
from .hamiltonian_discrete import _Target
from .open_system_spin import _align_field, _align_z, _precess, _unit
from .solvers import _integer


def _difference(v, stride=1):
    """Periodic centered stencil: D.T == -D, ||D|| <= 1."""
    return .5 * (torch.roll(v, -stride, -1) - torch.roll(v, stride, -1))


def _adjacency(v, stride=1):
    return .5 * (torch.roll(v, stride, -1) + torch.roll(v, -stride, -1))


def _laplacian(v):
    return 2 * v - torch.roll(v, 1, -1) - torch.roll(v, -1, -1)


def _diffusion(v, strength, passes):
    """Positive Neumann-polynomial mobility for I + strength*L."""
    omega = 1 / (1 + 4 * strength)
    out = torch.zeros_like(v)
    for _ in range(passes):
        out = out + omega * v
        v = v - omega * (v + strength * _laplacian(v))
    return out


def _cubic_difference(v):
    """Normalized sum D_x(D_y^2-D_z^2) + cyclic, matrix free."""
    n = v.shape[-1]
    strides = (1, max(2, round(n ** (1 / 3))), max(3, round(n ** (2 / 3))))
    out = torch.zeros_like(v)
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        term = (_difference(_difference(v, strides[b]), strides[b])
                - _difference(_difference(v, strides[c]), strides[c]))
        out = out + _difference(term, strides[a]) / 6
    return out


def _helical_mobility(n, like, m1, m2, phase):
    theta = torch.arange(n, device=like.device, dtype=like.dtype) * phase
    c, s = theta.cos(), theta.sin()
    scale = 2 / (m1 + m2)
    return ((m1 * c.square() + m2 * s.square()) * scale,
            (m1 * s.square() + m2 * c.square()) * scale,
            (m1 - m2) * c * s * scale)


def _random(run, shape):
    return torch.rand(shape, device=run.q.device, dtype=torch.float64,
                      generator=run.generator)


def _source_accept(run, current, proposal, temperature, scale):
    """Actual original-source scores decide all binary acceptance."""
    before, after = run.score(current), run.score(proposal)
    if temperature == 0:
        take = after < before
    else:
        difference = (after - before) / scale
        take = (difference <= 0) | (_random(run, after.shape).clamp_min(1e-300).log()
                                    < -difference / temperature)
    accepted = torch.where(take[:, None], proposal, current)
    scores = torch.where(take, after, before)
    run.retain(accepted, scores)
    return accepted


def _submatrix(target, indices):
    # Small k-by-n workspace, also valid for a source-preserving CSR target.
    basis = target.bias.new_zeros((indices.numel(), target.n))
    basis.scatter_(1, indices[:, None], 1.)
    return _mm(target.matrix, basis).index_select(1, indices)


def _conditional_energies(target, spins, indices):
    """All 2^k exact FP64 conditional gains; includes the current state."""
    size = indices.numel()
    numbers = torch.arange(1 << size, device=spins.device)
    toggles = ((numbers[:, None] >> torch.arange(size, device=spins.device)) & 1)
    old = spins.index_select(1, indices)
    delta = -2 * old[:, None, :] * toggles[None, :, :]
    field = target.field(spins).index_select(1, indices)
    matrix = _submatrix(target, indices)
    energy = (delta * field[:, None, :]).sum(-1)
    energy += .5 * ((delta @ matrix) * delta).sum(-1)
    return energy, toggles


def _conditional_move(run, target, sample, indices, temperature=0., bias=None):
    spins = sample.double() if run.spin else 2 * sample.double() - 1
    energies, toggles = _conditional_energies(target, spins, indices)
    selection = energies
    if bias is not None:
        trial = spins[:, None, indices] * (1 - 2 * toggles[None, :, :])
        selection = selection - (trial * bias).sum(-1)
    if temperature > 0:
        u = _random(run, selection.shape).clamp(1e-15, 1 - 1e-15)
        selection = selection + temperature * (-u.log()).log()
    chosen = selection.argmin(-1)
    mask = toggles[chosen].to(sample.dtype)
    proposal = sample.clone()
    old = sample.index_select(1, indices)
    proposal[:, indices] = old * (1 - 2 * mask) if run.spin else old + mask * (1 - 2 * old)
    # Conditional heat bath is already sampled with the desired probabilities;
    # at positive T accept its selected state, unless a biased score was used.
    if temperature > 0 and bias is None:
        run.retain(proposal)
        return proposal
    return _source_accept(run, sample, proposal, temperature, target.scale)


def _bdg_operator(diagonal, couplings, mixing, rashba_strength, pairing_strength):
    """Small Hermitian, particle-hole symmetric Nambu proposal operator."""
    k = diagonal.numel()
    rashba = couplings.new_zeros((k, k))
    row = torch.arange(max(0, k - 1), device=couplings.device)
    rashba[row, row + 1], rashba[row + 1, row] = 1., -1.
    normal = torch.diag(diagonal).to(torch.complex128) + mixing * couplings
    normal = normal + 1j * rashba_strength * rashba
    pairing = pairing_strength * rashba.to(torch.complex128)
    return torch.cat((torch.cat((normal, pairing), 1),
                      torch.cat((-pairing.conj(), -normal.conj()), 1)), 0)


@dataclass(frozen=True)
class _C3Base(PopulationSolver):
    max_steps: int = 512
    time_step: float = .08
    candidate_interval: int = 16
    pinning: float = .25
    initial_scale: float = .12
    terminal_fraction: float = .15
    polish_steps: int = 8
    normalize: bool = True
    dynamics_dtype: object = None
    sparse_threshold: float = .15

    def __post_init__(self):
        for name, minimum in (('max_steps', 0), ('candidate_interval', 1), ('polish_steps', 0)):
            _integer(name, getattr(self, name), minimum)
        positive('time_step', self.time_step)
        positive('pinning', self.pinning, zero=True)
        positive('initial_scale', self.initial_scale)
        if self.initial_scale >= 1:
            raise ValueError('initial_scale must be less than one')
        if (isinstance(self.terminal_fraction, bool) or not math.isfinite(self.terminal_fraction)
                or not 0 <= self.terminal_fraction < 1):
            raise ValueError('terminal_fraction must be in [0, 1)')
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if not math.isfinite(self.sparse_threshold) or not 0 <= self.sparse_threshold <= 1:
            raise ValueError('sparse_threshold must be in [0, 1]')
        _dtype(self.dynamics_dtype, torch.float32)

    def _population_workspace(self, n, batch, width):
        # Includes FP64 targets/fields, complex spinor/filter temporaries, old
        # and new integration states and maximum bounded block enumeration.
        return (144 * width + 192) * n * batch + 256 * n + 32768 * batch

    def _controls(self, step):
        fraction = observation.schedule_fraction(step / max(1, self.max_steps))
        fraction = min(1., max(0., fraction / (1 - self.terminal_fraction)))
        ramp = fraction * fraction * (3 - 2 * fraction)
        return 1 - ramp, self.pinning * ramp

    def _force(self, run, z, pinning):
        return -run.objective.gradient(z) + 2 * pinning * z

    def _initial(self, run):
        return (2 * run.initialbits.to(run.q.dtype) - 1) * self.initial_scale

    def _sphere(self, run):
        z = self._initial(run)
        angle = _random(run, z.shape).to(z.dtype) * (2 * math.pi)
        radius = (1 - z.square()).sqrt()
        return torch.stack((radius * angle.cos(), radius * angle.sin(), z), -1)

    def _checkpoint(self, run, record, z, step):
        if (step % self.candidate_interval == 0 or step == self.max_steps
                or run.history_interval and step % run.history_interval == 0):
            finite(z)
            run.publish(z, step)
            # Binary engines may already have retained a better state between
            # observation boundaries. Expose that verified incumbent, while
            # their independent local exploration arrays keep evolving.
            run.x.copy_(run.best_x)
            run.energies.copy_(run.best)
            record(step)

    def _finish(self, run, record, *states):
        finite(*states)
        sample = run.best_x.clone()
        rows = torch.arange(sample.shape[0], device=sample.device)
        for _ in range(self.polish_steps):
            observation.poll()
            spins = sample if run.spin else 2 * sample - 1
            gains = -2 * spins * run.objective.gradient(spins)
            indices = gains.argmin(-1)
            trial = sample.clone()
            trial[rows, indices] = -trial[rows, indices] if run.spin else 1 - trial[rows, indices]
            sample = _source_accept(run, sample, trial, 0., 1.)
        run.x.copy_(sample)
        run.energies = run.score(sample)
        run.retain(sample, run.energies)
        run.iterations.fill_(self.max_steps)
        record(self.max_steps)
        return self.max_steps


@dataclass(frozen=True)
class _Transformed(_C3Base):
    mixing: float = .5
    mechanism: ClassVar[str] = ''

    def __post_init__(self):
        super().__post_init__()
        positive('mixing', self.mixing, zero=True)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        q = torch.atanh(self._initial(run))
        y = torch.zeros_like(q)
        extra = [torch.zeros_like(q) for _ in range(3)] if self.mechanism == 'weyl' else []
        n = q.shape[-1]
        stride = max(2, round(math.sqrt(n)))
        handedness = (2 * (torch.arange(q.shape[0], device=q.device) % 2) - 1)[:, None]
        if self.mechanism == 'helix':
            mqq, myy, mqy = _helical_mobility(n, q, self.mobility_one,
                                            self.mobility_two, self.helix_phase)
        # For normalized forces, row norm <= 1 and max |F| <= 1+2*kappa.
        # Without normalization compute the same bound once from actual data.
        matrix = run.objective.matrix
        if matrix.layout == torch.strided:
            row = matrix.abs().sum(-1).max()
        else:
            rows = torch.repeat_interleave(torch.arange(n, device=q.device),
                matrix.crow_indices().diff(), output_size=matrix.values().numel())
            row = q.new_zeros(n).index_add_(0, rows, matrix.values().abs()).max()
        bound = float((3 * row + 2 * run.objective.field.abs().max()).cpu()) + 6 * self.pinning
        norm = 1 + 4 * self.mixing * self.mixing
        if self.mechanism == 'combined':
            norm = 1 + (2 * (abs(self.alpha) + abs(self.beta))) ** 2
        if self.mechanism == 'zeeman':
            norm = 1 + (self.mixing + abs(self.zeeman)) ** 2
        if self.mechanism == 'helix':
            norm = 2 * max(self.mobility_one, self.mobility_two) / (self.mobility_one + self.mobility_two)
        dt = min(self.time_step, 1 / (max(1., bound) * norm))
        for step in range(1, self.max_steps + 1):
            observation.poll()
            driver, pinning = self._controls(step)
            z = q.tanh()
            g = (1 - z.square()) * self._force(run, z, pinning)
            coupling = driver * self.mixing
            if self.mechanism == 'optical':
                q = q + dt * _diffusion(g, self.diffusion_strength, self.diffusion_passes)
            elif self.mechanism == 'weyl':
                strides = (1, stride, max(3, round(n ** (2 / 3))))
                transport = sum(_difference(v, s) for v, s in zip(extra, strides))
                next_q = q + dt * (g - coupling * transport)
                extra = [v - dt * (v - coupling * _difference(g, s))
                         for v, s in zip(extra, strides)]
                q = next_q
            elif self.mechanism == 'zeeman':
                bias = self.zeeman * handedness * driver
                next_q = q + dt * (g - coupling * _difference(y) - bias * y)
                y = y - dt * (y - coupling * _difference(g) + bias * g)
                q = next_q
            elif self.mechanism == 'helix':
                q, y = q + dt * (mqq * g - mqy * y), y + dt * (mqy * g - myy * y)
            elif self.mechanism == 'combined':
                def op(v):
                    return driver * ((self.beta - self.alpha) * _difference(v)
                                     + (self.alpha + self.beta) * _difference(v, stride))
                q, y = q + dt * (g - op(y)), y - dt * (y - op(g))
            elif self.mechanism == 'cubic':
                q, y = (q + dt * (g - coupling * _cubic_difference(y)),
                        y - dt * (y - coupling * _cubic_difference(g)))
            elif self.mechanism == 'dresselhaus':
                # B = (D_x-D_y)/2, B.T = -B, ||B|| <= 1.
                bg = .5 * (_difference(g) - _difference(g, stride))
                btbg = -.5 * (_difference(bg) - _difference(bg, stride))
                q = q + dt * (g + coupling * btbg)
            elif self.mechanism == 'rashba':
                q = q + dt * (g + coupling * handedness * _difference(g))
            else:
                raise AssertionError('unknown transformed mechanism')
            self._checkpoint(run, record, q, step)
        return self._finish(run, record, q, y, *extra)


@dataclass(frozen=True)
class C3OpticalLattice(_Transformed):
    """Entry60: positive polynomial graph-diffusion mobility."""
    diffusion_strength: float = .3
    diffusion_passes: int = 3
    mechanism: ClassVar[str] = 'optical'

    def __post_init__(self):
        super().__post_init__()
        positive('diffusion_strength', self.diffusion_strength, zero=True)
        _integer('diffusion_passes', self.diffusion_passes, 1)
        if self.diffusion_passes > 64:
            raise ValueError('diffusion_passes must be <= 64')


@dataclass(frozen=True)
class C3WeylSpinOrbit(_Transformed):
    """Entry80: three directional skew-coupled auxiliary channels."""
    mechanism: ClassVar[str] = 'weyl'


@dataclass(frozen=True)
class C3RashbaZeeman(_Transformed):
    """Entry81: spatial plus local skew mixing, no objective bias."""
    zeeman: float = .3
    mechanism: ClassVar[str] = 'zeeman'

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.zeeman):
            raise ValueError('zeeman must be finite')


@dataclass(frozen=True)
class C3PersistentSpinHelix(_Transformed):
    """Entry82: rotated positive local two-channel mobility."""
    mobility_one: float = 1.4
    mobility_two: float = .6
    helix_phase: float = .7
    mechanism: ClassVar[str] = 'helix'

    def __post_init__(self):
        super().__post_init__()
        positive('mobility_one', self.mobility_one)
        positive('mobility_two', self.mobility_two)
        if not math.isfinite(self.helix_phase):
            raise ValueError('helix_phase must be finite')


@dataclass(frozen=True)
class C3RashbaDresselhaus(_Transformed):
    """Entry83: combined, cancellable first-derivative channels."""
    alpha: float = .35
    beta: float = .2
    mechanism: ClassVar[str] = 'combined'

    def __post_init__(self):
        super().__post_init__()
        if not all(math.isfinite(v) for v in (self.alpha, self.beta)):
            raise ValueError('alpha and beta must be finite')


@dataclass(frozen=True)
class C3CubicDresselhaus(_Transformed):
    """Entry84: normalized cubic directional transport, matrix free."""
    mechanism: ClassVar[str] = 'cubic'


@dataclass(frozen=True)
class C3Dresselhaus(_Transformed):
    """Entry85: positive I + lambda B.T B mobility."""
    mechanism: ClassVar[str] = 'dresselhaus'


@dataclass(frozen=True)
class C3Rashba(_Transformed):
    """Entry86: direct skew mobility with paired circulation signs."""
    mechanism: ClassVar[str] = 'rashba'


@dataclass(frozen=True)
class _Vector(_C3Base):
    driver_strength: float = .4
    damping: float = 1.
    precession: float = .4
    mechanism: ClassVar[str] = ''

    def __post_init__(self):
        super().__post_init__()
        positive('driver_strength', self.driver_strength, zero=True)
        positive('damping', self.damping, zero=True)
        positive('precession', self.precession, zero=True)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        state = self._sphere(run)
        n = state.shape[1]
        stride = max(2, round(math.sqrt(n)))
        oscillator = self.initial_scale * run.normal(run.x.shape) if self.mechanism == 'rabi' else None
        momentum = torch.zeros_like(oscillator) if oscillator is not None else None
        stagger = (2 * (torch.arange(n, device=state.device) % 2) - 1).to(state.dtype)
        for step in range(1, self.max_steps + 1):
            observation.poll()
            driver, pinning = self._controls(step)
            strength = self.driver_strength * driver
            u, v, z = state.unbind(-1)
            force = self._force(run, z, pinning)
            if self.mechanism in ('xyz', 'xxz'):
                bx = strength * _adjacency(u)
                by = strength * _adjacency(v, stride if self.mechanism == 'xyz' else 1)
                if self.mechanism == 'xyz':
                    by = self.y_anisotropy * by
                field = torch.stack((bx, by, force), -1)
                state = _precess(state, field, self.time_step * self.precession)
                state = _align_field(state, field, self.time_step * self.damping)
            elif self.mechanism == 'compass':
                transverse = (step // self.switch_interval) % 2 == 0
                bx = strength * _adjacency(u) if transverse else torch.zeros_like(u)
                bz = force if transverse else force + strength * _adjacency(z, stride)
                field = torch.stack((bx, torch.zeros_like(v), bz), -1)
                state = _precess(state, field, self.time_step * self.precession)
                state = _align_field(state, field, self.time_step * self.damping)
            elif self.mechanism == 'raman':
                omega = torch.stack((torch.full_like(z, strength), torch.zeros_like(z),
                                     -self.detuning_feedback * force + strength * stagger), -1)
                # Existing helper evolves n x B; negate to obtain omega x n.
                state = _precess(state, -omega, self.time_step)
                state = _align_z(state, self._force(run, state[..., 2], pinning),
                                 self.time_step * self.damping)
            elif self.mechanism == 'rabi':
                equilibrium = -strength * u / self.oscillator_frequency ** 2
                angle = self.time_step * self.oscillator_frequency
                displacement = oscillator - equilibrium
                oscillator, momentum = (
                    equilibrium + displacement * math.cos(angle)
                    + momentum * math.sin(angle) / self.oscillator_frequency,
                    momentum * math.cos(angle)
                    - self.oscillator_frequency * displacement * math.sin(angle))
                momentum = momentum * math.exp(-self.damping * self.time_step)
                field = torch.stack((-strength * oscillator, torch.zeros_like(v), force), -1)
                state = _precess(state, field, self.time_step * self.precession)
                state = _align_field(state, field, self.time_step * self.damping)
            else:
                raise AssertionError('unknown vector mechanism')
            state = _unit(state)
            self._checkpoint(run, record, state[..., 2], step)
        return self._finish(run, record, state, *([] if oscillator is None else [oscillator, momentum]))


@dataclass(frozen=True)
class C3XYZSpin(_Vector):
    """Entry65: anisotropic transverse exchange on unit vectors."""
    y_anisotropy: float = .6
    mechanism: ClassVar[str] = 'xyz'

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.y_anisotropy):
            raise ValueError('y_anisotropy must be finite')


@dataclass(frozen=True)
class C3XXZSpin(_Vector):
    """Entry68: axially symmetric specialization of XYZ exchange."""
    mechanism: ClassVar[str] = 'xxz'


@dataclass(frozen=True)
class C3QuantumCompass(_Vector):
    """Entry70: alternating directional transverse/longitudinal drivers."""
    switch_interval: int = 16
    mechanism: ClassVar[str] = 'compass'

    def __post_init__(self):
        super().__post_init__()
        _integer('switch_interval', self.switch_interval, 1)


@dataclass(frozen=True)
class C3RamanSpinOrbit(_Vector):
    """Entry76: exact Raman-like local rotation followed by alignment."""
    detuning_feedback: float = 1.
    mechanism: ClassVar[str] = 'raman'

    def __post_init__(self):
        super().__post_init__()
        positive('detuning_feedback', self.detuning_feedback, zero=True)


@dataclass(frozen=True)
class C3QuantumRabi(_Vector):
    """Entry79: transient transverse spin/oscillator coupling."""
    oscillator_frequency: float = 1.
    mechanism: ClassVar[str] = 'rabi'

    def __post_init__(self):
        super().__post_init__()
        positive('oscillator_frequency', self.oscillator_frequency)


def _transport_spinors(spinors, angle, offset):
    """Exact two-site spin-dependent unitaries on one open-chain matching."""
    n = spinors.shape[1]
    if offset >= n - 1:
        return spinors
    left = torch.arange(offset, n - 1, 2, device=spinors.device)
    right = left + 1
    a, b = spinors[:, left].clone(), spinors[:, right].clone()
    phase = spinors.new_tensor([1., -1.])
    spinors = spinors.clone()
    spinors[:, left] = math.cos(angle) * a - 1j * math.sin(angle) * b * phase
    spinors[:, right] = math.cos(angle) * b - 1j * math.sin(angle) * a * phase
    return spinors


@dataclass(frozen=True)
class C3SpinOrbitBoseGas(_C3Base):
    """Entry75: local target alignment plus spin-dependent graph transport."""
    transport_strength: float = .6

    def __post_init__(self):
        super().__post_init__()
        positive('transport_strength', self.transport_strength, zero=True)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        state = self._sphere(run)
        for step in range(1, self.max_steps + 1):
            observation.poll()
            driver, pinning = self._controls(step)
            state = _align_z(state, self._force(run, state[..., 2], pinning), self.time_step)
            z = state[..., 2].clamp(-1, 1)
            a = ((1 + z) * .5).sqrt()
            denominator = (2 * (1 + z)).sqrt().clamp_min(torch.finfo(z.dtype).eps)
            b = torch.complex(state[..., 0], state[..., 1]) / denominator
            b = torch.where(z <= -1 + torch.finfo(z.dtype).eps,
                            torch.ones_like(b), b)
            spinors = torch.stack((a.to(b.dtype), b), -1)
            fallback = spinors.clone()
            for offset in (0, 1):
                spinors = _transport_spinors(spinors, self.time_step * driver * self.transport_strength, offset)
            norm = spinors.abs().square().sum(-1, keepdim=True).sqrt()
            # A vanished local amplitude keeps its seeded, aligned pre-pulse
            # spinor, rather than introducing a deterministic extra pole bias.
            spinors = torch.where(norm > torch.finfo(z.dtype).tiny,
                                  spinors / norm.clamp_min(torch.finfo(z.dtype).tiny), fallback)
            a, b = spinors.unbind(-1)
            ab = a.conj() * b
            state = torch.stack((2 * ab.real, 2 * ab.imag, a.abs().square() - b.abs().square()), -1)
            self._checkpoint(run, record, state[..., 2], step)
        return self._finish(run, record, state)


def _collective_rotation(momentum, mediator, pattern, angle):
    collective = (momentum * pattern).sum(-1)
    updated = math.cos(angle) * collective - math.sin(angle) * mediator
    mediator = math.sin(angle) * collective + math.cos(angle) * mediator
    return momentum + (updated - collective)[:, None] * pattern, mediator


@dataclass(frozen=True)
class C3TavisCummings(_C3Base):
    """Entry77: rank-one momentum/cavity exchange with exact rotation."""
    coupling: float = .8
    damping: float = .5

    def __post_init__(self):
        super().__post_init__()
        positive('coupling', self.coupling, zero=True)
        positive('damping', self.damping, zero=True)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        z = self._initial(run)
        momentum, mediator = torch.zeros_like(z), z.new_zeros(z.shape[0])
        pattern = z.new_full((z.shape[1],), 1 / math.sqrt(z.shape[1]))
        for step in range(1, self.max_steps + 1):
            observation.poll()
            driver, pinning = self._controls(step)
            momentum = momentum + self.time_step * self._force(run, z, pinning)
            momentum, mediator = _collective_rotation(momentum, mediator, pattern,
                                                       driver * self.coupling * self.time_step)
            decay = math.exp(-self.damping * self.time_step)
            momentum, mediator = momentum * decay, mediator * decay
            next_z = z + self.time_step * momentum
            momentum = momentum.masked_fill(next_z.abs() > 1, 0.)
            z = next_z.clamp(-1, 1)
            self._checkpoint(run, record, z, step)
        return self._finish(run, record, z, momentum, mediator)


@dataclass(frozen=True)
class _Binary(_C3Base):
    temperature: float = .15
    block_size: int = 4
    mechanism: ClassVar[str] = ''

    def __post_init__(self):
        super().__post_init__()
        positive('temperature', self.temperature, zero=True)
        _integer('block_size', self.block_size, 1)
        if self.block_size > 6:
            raise ValueError('block_size must be <= 6 for bounded enumeration')

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective, occupation=False, normalize=True)
        if target.constant:
            return 0
        sample = run.x.clone()
        n = sample.shape[1]
        order = torch.randperm(n, device=sample.device, generator=run.generator)
        edges = _cycle_weights(target) if self.mechanism == 'majorana' else None
        for step in range(1, self.max_steps + 1):
            observation.poll()
            driver, _ = self._controls(step)
            temperature = self.temperature * driver * driver
            spins = sample.double() if run.spin else sample.double() * 2 - 1
            gains = -2 * spins * target.field(spins)
            if self.mechanism == 'feshbach':
                anchor = order[(step - 1) % n].reshape(1)
                partner = order[step % n].reshape(1)
                indices = torch.cat((anchor, partner)) if n > 1 else anchor
                sample = _conditional_move(run, target, sample, indices, temperature)
            elif self.mechanism == 'active':
                if step % 3 == 0:
                    rank = gains.abs().mean(0)
                    indices = rank.topk(min(self.block_size, n), largest=False).indices
                else:
                    indices = order[(step - 1) % n].reshape(1)
                sample = _conditional_move(run, target, sample, indices, 0.)
            elif self.mechanism == 'xx':
                indices = order[torch.tensor([(step - 1) % n, step % n], device=sample.device)]
                if n == 1 or step % self.single_interval == 0:
                    sample = _conditional_move(run, target, sample, indices[:1], temperature)
                else:
                    a, b = indices.unbind()
                    proposal = sample.clone()
                    proposal[:, a], proposal[:, b] = sample[:, b], sample[:, a]
                    sample = _source_accept(run, sample, proposal, temperature, target.scale)
            elif self.mechanism == 'rydberg':
                bandwidth = self.resonance_width * (.1 + driver)
                weights = .05 + 1 / (1 + (gains / bandwidth).square())
                anchor = torch.multinomial(weights.mean(0), 1, generator=run.generator)
                if n > 1:
                    basis = target.bias.new_zeros((1, n))
                    basis[:, anchor] = 1.
                    strength = _mm(target.matrix, basis).abs()[0]
                    strength[anchor] = -1
                    partner = strength.argmax().reshape(1)
                    indices = torch.cat((anchor, partner))
                else:
                    indices = anchor
                sample = _conditional_move(run, target, sample, indices, temperature)
            elif self.mechanism == 'rice':
                start = (2 * (step - 1) + ((step - 1) // max(1, n // 2))) % n
                indices = order[torch.tensor([start, (start + 1) % n], device=sample.device)] if n > 1 else order
                bias = (self.staggered_bias * driver * math.sin(2 * math.pi * step / self.bias_period)
                        * (2 * (indices % 2) - 1))
                sample = _conditional_move(run, target, sample, indices, temperature, bias)
            elif self.mechanism == 'majorana':
                # C is the weighted cycle subset of the source objective.
                diagonal = gains.mean(0)
                v, w = run.normal((1, n)).double(), run.normal((1, n)).double()
                def t(v):
                    return diagonal * v + self.filter_coupling * _weighted_cycle(v, edges)
                bound = diagonal.abs().max() + 2 * self.filter_coupling * edges.abs().max()
                eta = .5 / bound.square().clamp_min(1e-12)
                for _ in range(self.filter_steps):
                    v, w = v - eta * t(t(v)), w - eta * t(t(w))
                    v = v / v.norm().clamp_min(1e-30)
                    w = w / w.norm().clamp_min(1e-30)
                rank = (v.square() + w.square())[0]
                indices = rank.topk(min(self.block_size, n)).indices
                sample = _conditional_move(run, target, sample, indices)
            elif self.mechanism == 'bdg':
                pool = gains.abs().mean(0).topk(min(self.active_size, n), largest=False).indices
                matrix = _submatrix(target, pool)
                diagonal = gains.mean(0)[pool]
                k = pool.numel()
                bdg = _bdg_operator(diagonal, matrix, self.filter_coupling,
                                    self.rashba_strength, self.pairing_strength)
                vector = torch.complex(run.normal((2 * k,)).double(), run.normal((2 * k,)).double())
                eta = .5 / bdg.abs().sum(-1).max().square().clamp_min(1e-12)
                for _ in range(self.filter_steps):
                    vector = vector - eta * (bdg @ (bdg @ vector))
                    vector = vector / vector.norm().clamp_min(1e-30)
                rank = vector[:k].abs().square() + vector[k:].abs().square()
                indices = pool[rank.topk(min(self.block_size, k)).indices]
                sample = _conditional_move(run, target, sample, indices)
            else:
                raise AssertionError('unknown binary mechanism')
            self._checkpoint(run, record, sample if run.spin else 2 * sample - 1, step)
        return self._finish(run, record, sample)


def _cycle_weights(target):
    """Read only adjacent source edges, O(n+nnz) setup, no dense conversion."""
    n = target.n
    matrix = target.matrix
    indices = torch.arange(n, device=matrix.device)
    if matrix.layout == torch.strided:
        return matrix[indices, (indices + 1) % n]
    rows = torch.repeat_interleave(indices, matrix.crow_indices().diff(),
                                  output_size=matrix.values().numel())
    keep = matrix.col_indices() == (rows + 1) % n
    return target.bias.new_zeros(n).index_add_(0, rows, matrix.values() * keep)


def _weighted_cycle(v, edges):
    return edges * torch.roll(v, -1, -1) + torch.roll(edges * v, 1, -1)


@dataclass(frozen=True)
class C3FeshbachTwoChannel(_Binary):
    """Entry59: four-state conditional pair conversion/heat bath."""
    mechanism: ClassVar[str] = 'feshbach'


@dataclass(frozen=True)
class C3DiracCoulombBreit(_Binary):
    """Entry61: exact bounded active blocks in a generic supplied QUBO."""
    mechanism: ClassVar[str] = 'active'


@dataclass(frozen=True)
class C3XXSpin(_Binary):
    """Entry67: exchanges plus an explicit unconstrained single-flip channel."""
    single_interval: int = 3
    mechanism: ClassVar[str] = 'xx'

    def __post_init__(self):
        super().__post_init__()
        _integer('single_interval', self.single_interval, 1)


@dataclass(frozen=True)
class C3RydbergAtom(_Binary):
    """Entry72: resonance-prioritized anchors with unrestricted pair search."""
    resonance_width: float = .2
    mechanism: ClassVar[str] = 'rydberg'

    def __post_init__(self):
        super().__post_init__()
        positive('resonance_width', self.resonance_width)


@dataclass(frozen=True)
class C3RiceMele(_Binary):
    """Entry73: shifting dimer coverings and a vanishing staggered bias."""
    staggered_bias: float = .1
    bias_period: int = 32
    mechanism: ClassVar[str] = 'rice'

    def __post_init__(self):
        super().__post_init__()
        positive('staggered_bias', self.staggered_bias, zero=True)
        _integer('bias_period', self.bias_period, 1)


@dataclass(frozen=True)
class C3KitaevHoneycomb(_Binary):
    """Entry69: matrix-free Majorana lift filtering, exact conditional blocks."""
    filter_steps: int = 3
    filter_coupling: float = .3
    mechanism: ClassVar[str] = 'majorana'

    def __post_init__(self):
        super().__post_init__()
        _integer('filter_steps', self.filter_steps, 1)
        positive('filter_coupling', self.filter_coupling, zero=True)


@dataclass(frozen=True)
class C3RashbaBdG(C3KitaevHoneycomb):
    """Entry74: bounded Hermitian BdG-square filter for block proposals."""
    active_size: int = 8
    rashba_strength: float = .2
    pairing_strength: float = .15
    mechanism: ClassVar[str] = 'bdg'

    def __post_init__(self):
        super().__post_init__()
        _integer('active_size', self.active_size, 1)
        if self.active_size > 32 or self.active_size < self.block_size:
            raise ValueError('active_size must be between block_size and 32')
        positive('rashba_strength', self.rashba_strength, zero=True)
        positive('pairing_strength', self.pairing_strength, zero=True)

    def _population_workspace(self, n, batch, width):
        return super()._population_workspace(n, batch, width) + 1024 * min(n, self.active_size) ** 2


@dataclass(frozen=True)
class C3LambdaThreeLevel(_C3Base):
    """Entry78: ternary routed transitions with nonzero conductance floors."""
    temperature: float = .25
    conductance_floor: float = .1
    phase_period: int = 32

    def __post_init__(self):
        super().__post_init__()
        positive('temperature', self.temperature, zero=True)
        positive('conductance_floor', self.conductance_floor)
        _integer('phase_period', self.phase_period, 1)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective, occupation=False, normalize=True)
        state = run.x.double() if run.spin else 2 * run.x.double() - 1
        field = target.field(state)
        n = state.shape[1]
        phases = torch.arange(state.shape[0], device=state.device, dtype=state.dtype) * .5
        for step in range(1, self.max_steps + 1):
            observation.poll()
            driver, pinning = self._controls(step)
            i = (step - 1) % n
            old = state[:, i].clone()
            phase = phases + 2 * math.pi * step / self.phase_period
            left = self.conductance_floor + phase.sin().square()
            right = self.conductance_floor + phase.cos().square()
            # Uniformize a symmetric three-state conductance chain: remaining
            # probability is an explicit self-loop, avoiding a Hastings bias.
            u = _random(run, old.shape) * (1 + 2 * self.conductance_floor)
            proposed = old.clone()
            proposed = torch.where((old == -1) & (u < left), 0., proposed)
            proposed = torch.where((old == 1) & (u < right), 0., proposed)
            proposed = torch.where((old == 0) & (u < left), -1., proposed)
            proposed = torch.where((old == 0) & (u >= left) & (u < left + right), 1., proposed)
            if step % 7 == 0:
                proposed = -old  # Symmetric direct channel remains accessible.
            change = proposed - old
            gain = change * field[:, i] - pinning * (proposed.square() - old.square())
            temp = self.temperature * driver * driver
            accept = gain < 0 if temp == 0 else (gain <= 0) | (
                _random(run, old.shape).clamp_min(1e-300).log() < -gain / temp)
            change = change * accept
            state[:, i] += change
            target.update(field, i, change)
            if step % 64 == 0:
                field = target.field(state)
            if step % self.candidate_interval == 0 or step == self.max_steps or (
                    run.history_interval and step % run.history_interval == 0):
                repaired, repair_field = state.clone(), target.field(state)
                for j in range(n):
                    observation.poll()
                    sign = torch.where(repair_field[:, j] <= 0, 1., -1.)
                    delta = torch.where(repaired[:, j] == 0, sign, 0.)
                    repaired[:, j] += delta
                    target.update(repair_field, j, delta)
                self._checkpoint(run, record, repaired, step)
        return self._finish(run, record, state)


def _binary_rref(matrix, syndrome):
    """Finite GF(2) elimination: feasible particular solution and null basis."""
    rows = [[int(bit) for bit in row] + [int(value)] for row, value in zip(matrix, syndrome)]
    width = len(matrix[0])
    pivots, at = [], 0
    for column in range(width):
        pivot = next((r for r in range(at, len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        rows[at], rows[pivot] = rows[pivot], rows[at]
        for r in range(len(rows)):
            if r != at and rows[r][column]:
                rows[r] = [a ^ b for a, b in zip(rows[r], rows[at])]
        pivots.append(column)
        at += 1
    if any(not any(row[:-1]) and row[-1] for row in rows):
        raise ValueError('decoder syndrome is inconsistent with dependent checks')
    particular = [0] * width
    for row, pivot in zip(rows, pivots):
        particular[pivot] = row[-1]
    free = [i for i in range(width) if i not in pivots]
    basis = []
    for column in free:
        support = [column]
        support.extend(pivot for row, pivot in zip(rows, pivots) if row[column])
        basis.append(tuple(support))
    return tuple(particular), tuple(basis)


def _validate_decoder_metadata(solver, n):
    checks = tuple(tuple(row) for row in solver.check_matrix)
    syndrome = tuple(solver.syndrome)
    slacks = tuple(tuple(row) for row in solver.slack_bits)
    if not checks or not checks[0] or not syndrome or not slacks:
        raise ValueError('structured decoder requires check_matrix, syndrome, and slack_bits metadata; '
                         'a bare QUBO does not define a code or detector model')
    width = len(checks[0])
    if width > n or len(syndrome) != len(checks) or len(slacks) != len(checks):
        raise ValueError('decoder metadata dimensions are inconsistent with the QUBO')
    if any(len(row) != width or any(v not in (0, 1) for v in row) for row in checks):
        raise ValueError('check_matrix must be a rectangular binary matrix')
    if any(v not in (0, 1) for v in syndrome):
        raise ValueError('syndrome must be binary')
    used = set(range(width))
    for check, slack in zip(checks, slacks):
        if (1 << len(slack)) - 1 < sum(check) // 2:
            raise ValueError('slack_bits cannot represent every permitted parity slack')
        for index in slack:
            if isinstance(index, bool) or not isinstance(index, int) or index < width or index >= n or index in used:
                raise ValueError('slack_bits must contain distinct in-range auxiliary indices')
            used.add(index)
    if used != set(range(n)):
        raise ValueError('logical bits and slack_bits must cover every supplied QUBO variable')
    for name in ('proposal_masks', 'logical_masks'):
        for support in getattr(solver, name):
            if not support or len(set(support)) != len(support) or any(
                    isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < width for i in support):
                raise ValueError(f'{name} must contain nonempty distinct logical-bit supports')
            if any(sum(row[i] for i in support) % 2 for row in checks):
                raise ValueError(f'{name} contains a mask that changes the supplied syndrome')
    if bool(solver.check_x) != bool(solver.check_z):
        raise ValueError('check_x and check_z must be supplied together')
    if solver.check_x:
        hx = tuple(tuple(row) for row in solver.check_x)
        hz = tuple(tuple(row) for row in solver.check_z)
        if len(hx) != len(hz) or len(hx) != len(checks) or not hx[0]:
            raise ValueError('symplectic check dimensions do not match')
        qubits = len(hx[0])
        if width != 2 * qubits or any(len(row) != qubits or any(v not in (0, 1) for v in row)
                                    for row in hx + hz):
            raise ValueError('symplectic checks must have two binary blocks of equal width')
        if any(check != zrow + xrow for check, xrow, zrow in zip(checks, hx, hz)):
            raise ValueError('check_matrix must equal [check_z | check_x] for Pauli errors')
        if any(sum(a * b + c * d for a, b, c, d in zip(x1, z2, z1, x2)) % 2
               for x1, z1 in zip(hx, hz) for x2, z2 in zip(hx, hz)):
            raise ValueError('supplied stabilizer checks do not commute symplectically')
    particular, basis = _binary_rref(checks, syndrome)
    return checks, syndrome, slacks, particular, basis


def _repair_slacks(sample, checks, syndrome, slacks):
    """Update squared integer parity penalties after a logical mask."""
    output = sample.clone()
    for row, bit, indices in zip(checks, syndrome, slacks):
        support = [i for i, value in enumerate(row) if value]
        total = sample[:, support].sum(-1).long() if support else sample.new_zeros(sample.shape[0], dtype=torch.long)
        value = (total - bit) // 2
        for power, index in enumerate(indices):
            output[:, index] = (value >> power) & 1
    return output


@dataclass(frozen=True)
class _StructuredDecoder(_C3Base):
    """Feasible configuration search on an explicitly expanded decoder QUBO.

    Logical/error bits occupy the first len(check_matrix[0]) coordinates.
    Each row's little-endian slack indices address the remaining coordinates.
    Caller-supplied code geometry is represented by validated mask supports;
    no implicit planar, toric, color, or gauge geometry is reconstructed.
    Returned bits map directly through that supplied metadata. This minimizes
    individual error configurations, never sums logical-coset probabilities.
    """
    check_matrix: tuple = ()
    syndrome: tuple = ()
    slack_bits: tuple = ()
    proposal_masks: tuple = ()
    logical_masks: tuple = ()
    check_x: tuple = ()
    check_z: tuple = ()
    temperature: float = .1
    polish_steps: int = 0
    requires_structure: ClassVar[bool] = True

    def __post_init__(self):
        super().__post_init__()
        positive('temperature', self.temperature, zero=True)
        if self.polish_steps:
            raise ValueError('decoder polishing must preserve constraints; use polish_steps=0')

    def _population_workspace(self, n, batch, width):
        # Dense check metadata is supplied, not inferred; account for it and
        # GF(2) basis/support storage explicitly. No per-replica n-by-n state.
        cells = sum(len(row) for row in self.check_matrix)
        return super()._population_workspace(n, batch, width) + 32 * cells + 16 * n * n

    def _execute(self, objective, restarts, batch, initial, seed, best_only, interval):
        if objective.spin:
            raise ValueError('decoder metadata addresses binary QUBO variables, not Ising inputs')
        checks, syndrome, slacks, particular, _ = _validate_decoder_metadata(self, objective.n)
        if initial is None:
            first = torch.zeros((1, objective.n), device=objective.device, dtype=objective.dtype)
            first[:, :len(particular)] = first.new_tensor(particular)
            first = _repair_slacks(first, checks, syndrome, slacks)
            initial = first.expand(restarts, -1)
        else:
            # Validate before PopulationSolver emits the initial observation.
            for start in range(0, restarts, batch):
                selected = initial[start:start + batch]
                for check, value in zip(checks, syndrome):
                    support = [i for i, bit in enumerate(check) if bit]
                    if bool((selected[:, support].sum(-1).long().remainder(2) != value).any()):
                        raise ValueError('initial_assignments violate the supplied decoder syndrome')
                if not bool((selected == _repair_slacks(selected, checks, syndrome, slacks)).all()):
                    raise ValueError('initial_assignments contain inconsistent parity slacks')
        return super()._execute(objective, restarts, batch, initial, seed, best_only, interval)

    def _search(self, run, record):
        checks, syndrome, slacks, _, basis = _validate_decoder_metadata(self, run.x.shape[1])
        supplied = tuple(tuple(mask) for mask in self.proposal_masks)
        logical = tuple(tuple(mask) for mask in self.logical_masks)
        # Null-space elementary corrections give complete feasible reachability;
        # named geometry masks provide the coordinated face/path/gauge channel.
        streams = (basis, supplied, logical)
        sample = run.x.clone()
        if self.max_steps == 0 or not any(streams):
            return 0
        scale = run.objective.scale
        for step in range(1, self.max_steps + 1):
            observation.poll()
            driver, _ = self._controls(step)
            stream = streams[(step - 1) % len(streams)] or basis or supplied or logical
            support = stream[((step - 1) // len(streams)) % len(stream)]
            proposal = sample.clone()
            # Independent self-loop decisions give replica diversity while the
            # shared support avoids host synchronization per trajectory.
            change = (_random(run, (sample.shape[0], 1)) < .8).to(sample.dtype)
            proposal[:, support] = sample[:, support] + change * (1 - 2 * sample[:, support])
            proposal = _repair_slacks(proposal, checks, syndrome, slacks)
            sample = _source_accept(run, sample, proposal, self.temperature * driver * driver, scale)
            self._checkpoint(run, record, 2 * sample - 1, step)
        return self._finish(run, record, sample)


@dataclass(frozen=True)
class C3ColorCode(_StructuredDecoder):
    """Entry62: supplied color-face/string masks with exact parity slacks."""


@dataclass(frozen=True)
class C3ToricCode(_StructuredDecoder):
    """Entry63: supplied contractible cycles and noncontractible loops."""


@dataclass(frozen=True)
class C3SurfaceCode(_StructuredDecoder):
    """Entry64: supplied boundary-aware detector paths and logical masks."""


@dataclass(frozen=True)
class C3XZZXSurfaceCode(_StructuredDecoder):
    """Entry66: supplied symplectic syndrome and bias-aligned strings."""


@dataclass(frozen=True)
class C3BaconShor(_StructuredDecoder):
    """Entry71: center checks, supplied gauge masks and separate logical masks."""


def compile_parity_decoder(check_matrix, syndrome, prior_linear, prior_pairs=(), *,
                           penalty=None, proposal_masks=(), logical_masks=(),
                           check_x=(), check_z=(), offset=0., device='cpu'):
    """Compile a bounded exact parity-slack configuration-decoding QUBO.

    ``prior_pairs`` is a sequence of (i, j, coefficient), counted once per
    supplied term, including duplicate indices and self-loops. Pair terms can
    encode the X/Z cross cost of a correlated Pauli prior. This is individual
    configuration decoding, not logical-coset maximum likelihood. Input rows
    are binary incidences (repeated physical effects must first combine in
    GF(2)). ``penalty`` must strictly exceed an upward-rounded bound on the
    full Boolean prior range; the default is chosen above that bound. A
    compatible syndrome guarantees a feasible zero-penalty configuration.

    Return ``(QUBO, metadata)``. The metadata can be passed to the five C
    decoder classes; code-specific geometry/check data remain the caller's
    responsibility. Dense compilation is limited to 4096 total variables;
    larger applications should expand the same squares into compact storage.
    """
    from .problems import QUBO

    checks = tuple(tuple(row) for row in check_matrix)
    measured = tuple(syndrome)
    linear = tuple(float(value) for value in prior_linear)
    if not checks or not linear or any(len(row) != len(linear) for row in checks):
        raise ValueError('check_matrix width must match nonempty prior_linear')
    if len(measured) != len(checks) or any(v not in (0, 1) for v in measured):
        raise ValueError('syndrome must have one binary value per check')
    if any(value not in (0, 1) for row in checks for value in row):
        raise ValueError('check_matrix must be binary')
    checks = tuple(tuple(int(value) for value in row) for row in checks)
    measured = tuple(int(value) for value in measured)
    _binary_rref(checks, measured)  # Feasibility is part of the penalty proof.
    pairs = []
    for term in prior_pairs:
        if len(term) != 3:
            raise ValueError('prior_pairs terms must contain i, j, coefficient')
        i, j, weight = term
        if any(isinstance(index, bool) or not isinstance(index, int) or
               not 0 <= index < len(linear) for index in (i, j)):
            raise ValueError('prior_pairs indices must address logical variables')
        pairs.append((i, j, float(weight)))
    coefficients = linear + tuple(weight for _, _, weight in pairs)
    if not math.isfinite(offset) or not all(math.isfinite(value) for value in coefficients):
        raise ValueError('prior coefficients and offset must be finite')
    try:
        prior_range = math.nextafter(math.fsum(abs(value) for value in coefficients), math.inf)
    except OverflowError as error:
        raise ValueError('prior-range bound overflows float64') from error
    selected_penalty = (max(1., prior_range * 1.01) if penalty is None else float(penalty))
    if not math.isfinite(selected_penalty) or selected_penalty <= prior_range:
        raise ValueError('penalty must exceed the full finite prior-range bound')
    cursor = len(linear)
    slack_bits = []
    for row in checks:
        count = (sum(row) // 2).bit_length()
        slack_bits.append(tuple(range(cursor, cursor + count)))
        cursor += count
    if cursor > 4096:
        raise ValueError('dense decoder compilation is limited to 4096 variables; use compact storage')
    metadata = dict(check_matrix=checks, syndrome=measured, slack_bits=tuple(slack_bits),
                    proposal_masks=tuple(tuple(row) for row in proposal_masks),
                    logical_masks=tuple(tuple(row) for row in logical_masks),
                    check_x=tuple(tuple(row) for row in check_x),
                    check_z=tuple(tuple(row) for row in check_z))
    _validate_decoder_metadata(C3ColorCode(**metadata), cursor)
    matrix = torch.zeros((cursor, cursor), device=device, dtype=torch.float64)
    matrix.diagonal()[:len(linear)] = matrix.new_tensor(linear)
    # Store each supplied QUBO pair once in an upper triangular matrix; QUBO
    # symmetrizes once. A self-loop contributes once to the diagonal.
    for i, j, weight in pairs:
        matrix[min(i, j), max(i, j)] += weight
    compiled_offset = float(offset)
    for row, value, slacks in zip(checks, measured, slack_bits):
        support = [i for i, bit in enumerate(row) if bit] + list(slacks)
        weights = [1.] * sum(row) + [-2. * (1 << bit) for bit in range(len(slacks))]
        if support:
            indices = torch.tensor(support, device=device, dtype=torch.long)
            factors = matrix.new_tensor(weights)
            square = selected_penalty * torch.outer(factors, factors)
            square.diagonal().add_(-2 * selected_penalty * value * factors)
            # A full symmetric square contributes every off-diagonal pair
            # twice under x.T Q x, precisely as the expanded square requires.
            matrix[indices[:, None], indices[None, :]] += square
        compiled_offset += selected_penalty * value * value
    if not bool(torch.isfinite(matrix).all()) or not math.isfinite(compiled_offset):
        raise ValueError('compiled decoder coefficients overflow float64; rescale the prior')
    return QUBO(matrix, offset=compiled_offset), metadata


COMPENDIUM_III_C_CLASSES = {
    'lib_c3_feshbach_two_channel': C3FeshbachTwoChannel,
    'lib_c3_optical_lattice': C3OpticalLattice,
    'lib_c3_dirac_coulomb_breit': C3DiracCoulombBreit,
    'lib_c3_color_code': C3ColorCode,
    'lib_c3_toric_code': C3ToricCode,
    'lib_c3_surface_code': C3SurfaceCode,
    'lib_c3_xyz_spin': C3XYZSpin,
    'lib_c3_xzzx_surface_code': C3XZZXSurfaceCode,
    'lib_c3_xx_spin': C3XXSpin,
    'lib_c3_xxz_spin': C3XXZSpin,
    'lib_c3_kitaev_honeycomb': C3KitaevHoneycomb,
    'lib_c3_quantum_compass': C3QuantumCompass,
    'lib_c3_bacon_shor': C3BaconShor,
    'lib_c3_rydberg_atom': C3RydbergAtom,
    'lib_c3_rice_mele': C3RiceMele,
    'lib_c3_rashba_bdg': C3RashbaBdG,
    'lib_c3_spin_orbit_bose_gas': C3SpinOrbitBoseGas,
    'lib_c3_raman_spin_orbit': C3RamanSpinOrbit,
    'lib_c3_tavis_cummings': C3TavisCummings,
    'lib_c3_lambda_three_level': C3LambdaThreeLevel,
    'lib_c3_quantum_rabi': C3QuantumRabi,
    'lib_c3_weyl_spin_orbit': C3WeylSpinOrbit,
    'lib_c3_rashba_zeeman': C3RashbaZeeman,
    'lib_c3_persistent_spin_helix': C3PersistentSpinHelix,
    'lib_c3_rashba_dresselhaus': C3RashbaDresselhaus,
    'lib_c3_cubic_dresselhaus': C3CubicDresselhaus,
    'lib_c3_dresselhaus': C3Dresselhaus,
    'lib_c3_rashba': C3Rashba,
}

__all__ = [cls.__name__ for cls in COMPENDIUM_III_C_CLASSES.values()] + ['compile_parity_decoder']
