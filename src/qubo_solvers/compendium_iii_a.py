"""Compendium III entries 01--29: bounded classical proposal engines.

These implement the document's *QUBO adaptations*, not quantum state-vector
simulation. All artificial graphs are explicitly periodic index stencils or
bounded strong-edge neighborhoods. The original objective is never truncated.
Continuous proposals use Torch GEMM/CSR; readout and discrete acceptance use
the authoritative FP64 source. RNG is the existing seeded device generator;
device and population batch size are part of reproducibility.
"""

from dataclasses import dataclass
import math

import torch

from . import observation
from ._population import PopulationSolver, _dtype
from .hamiltonian_continuous import _Target as _Readout
from .hamiltonian_discrete import _Target, _random
from .open_system_discrete import _submatrix, _enumerate_block, _neighborhoods
from .open_system_spin import _alignment_coefficients
from .solvers import _integer


def _positive(name, value, zero=False):
    if (isinstance(value, bool) or not math.isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def _clip(value, bound=1.):
    """Bounds do not conceal a divergent proposal as a plausible bitstring."""
    return value.clamp(-bound, bound).masked_fill(~torch.isfinite(value), float('nan'))


def _cycle(value, stride=1):
    """Normalized symmetric periodic stencil; distinct neighbors, no self links."""
    n = value.shape[1]
    stride %= n
    if stride == 0:
        return torch.zeros_like(value)
    return .5 * (value.roll(stride, 1) + value.roll(-stride, 1))


def _skew(value):
    return .5 * (value.roll(1, 1) - value.roll(-1, 1))


def _triangle_skew(z, g):
    """Disjoint oriented triples; exact skew action with O(B*n) scratch."""
    size = z.shape[1] // 3 * 3
    out = torch.zeros_like(g)
    if size:
        triples = z[:, :size].reshape(z.shape[0], -1, 3)
        gradients = g[:, :size].reshape(g.shape[0], -1, 3)
        out[:, :size] = torch.cross(gradients, triples, dim=-1).reshape(g.shape[0], size)
    return out


def _sphere_step(spins, field, dt, damping, precession=True):
    """Exact frozen-field alignment and rotation, including poles and B=0."""
    norm = field.norm(dim=-1, keepdim=True)
    axis = field / norm.clamp_min(torch.finfo(field.dtype).tiny)
    cosine = (spins * axis).sum(-1, keepdim=True).clamp(-1, 1)
    perpendicular = spins - cosine * axis
    aligned_cosine, transverse_scale = _alignment_coefficients(cosine, damping * dt * norm)
    # Do not reconstruct the tiny transverse radius from 1-cosine**2: FP32
    # can round cosine to a pole while a real transverse perturbation remains.
    perpendicular = perpendicular * transverse_scale
    if precession:
        angle = -dt * norm
        perpendicular = perpendicular * angle.cos() + torch.cross(
            axis, perpendicular, dim=-1) * angle.sin()
    result = aligned_cosine * axis + perpendicular
    result = torch.where(norm == 0, spins, result)
    length = result.norm(dim=-1, keepdim=True)
    result = result / length.clamp_min(torch.finfo(field.dtype).tiny)
    return result.masked_fill(~torch.isfinite(norm) | ~torch.isfinite(length), float('nan'))


def _ramsey(spins, angle, phase):
    """Rotation around the transverse axis (cos phase, sin phase, 0)."""
    axis = torch.stack((phase.cos(), phase.sin(), torch.zeros_like(phase)), -1)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (cosine * spins + sine * torch.cross(axis, spins, dim=-1)
            + (1 - cosine) * axis * (axis * spins).sum(-1, keepdim=True))


def _group_patterns(n, modes, reference):
    """Disjoint normalized group indicators: C C^T = I, no spectral setup."""
    modes = min(n, modes)
    labels = torch.arange(n, device=reference.device) % modes
    patterns = torch.nn.functional.one_hot(labels, modes).T.to(reference.dtype)
    return patterns / patterns.square().sum(1, keepdim=True).sqrt()


def _repulsion(z, epsilon):
    """Fixed sparse replica-cycle force; no R-by-R distance tensor."""
    if z.shape[0] < 2:
        return torch.zeros_like(z)
    difference = z - z.roll(1, 0)
    edge = difference / (epsilon + difference.square().mean(1, keepdim=True))
    factor = 1. if z.shape[0] == 2 else 2.
    return factor / z.shape[1] * (edge - edge.roll(-1, 0))


@dataclass(frozen=True)
class _Base(PopulationSolver):
    max_steps: int = 1024
    candidate_interval: int = 32
    polish_sweeps: int = 1
    time_step: float = .04
    pinning: float = .4
    temperature: float = .08
    terminal_fraction: float = .15
    dynamics_dtype: object = None
    normalize: bool = True
    sparse_threshold: float = .15

    def __post_init__(self):
        for name, minimum in (('max_steps', 0), ('candidate_interval', 1), ('polish_sweeps', 0)):
            _integer(name, getattr(self, name), minimum)
        _positive('time_step', self.time_step)
        for name in ('pinning', 'temperature'):
            _positive(name, getattr(self, name), zero=True)
        _dtype(self.dynamics_dtype, torch.float32)
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if (isinstance(self.terminal_fraction, bool) or not math.isfinite(self.terminal_fraction)
                or not 0 <= self.terminal_fraction < 1):
            raise ValueError('terminal_fraction must be in [0,1)')
        if (isinstance(self.sparse_threshold, bool) or not math.isfinite(self.sparse_threshold)
                or not 0 <= self.sparse_threshold <= 1):
            raise ValueError('sparse_threshold must be in [0,1]')

    def _population_workspace(self, n, batch, width):
        # Conservative bound includes all vector-spin exact-map temporaries,
        # FP64 readout, source targets, masks, and bounded enumeration scratch.
        modes = min(n, getattr(self, 'modes', 4))
        block = min(n, getattr(self, 'block_size', 8))
        return ((96 * width + 256) * n * batch + 64 * n * modes
                + 64 * batch * (256 + block) + 64 * block * block + 256 * n)

    def _schedule(self, step):
        fraction = observation.schedule_fraction(step / max(1, self.max_steps))
        u = min(1., max(0., fraction) / (1 - self.terminal_fraction))
        r = u * u * (3 - 2 * u)
        return self.pinning * r, self.temperature * (1 - r) ** 2, 1 - r

    def _checkpoint(self, run, readout, state, step, record, force=False):
        if (force or step % self.candidate_interval == 0 or step == self.max_steps
                or run.history_interval and step % run.history_interval == 0):
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError('nonfinite compendium III proposal')
            run.publish(readout.readout(state, self.polish_sweeps), step)
            run.x.copy_(run.best_x)
            run.energies.copy_(run.best)
            record(step)


@dataclass(frozen=True)
class _Continuous(_Base):
    initial_scale: float = .05
    initial_noise: float = .01

    def __post_init__(self):
        super().__post_init__()
        _positive('initial_scale', self.initial_scale)
        _positive('initial_noise', self.initial_noise, zero=True)
        if self.initial_scale >= 1:
            raise ValueError('initial_scale must be less than one')

    def _prepare(self, run, z):
        return z, {}

    def _coordinates(self, state):
        return state[0] if isinstance(state, tuple) else state

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        readout = _Readout(run.objective)
        if readout.constant:
            return 0
        z = self.initial_scale * (run.initialbits.to(run.x.dtype) * 2 - 1)
        z = z + self.initial_noise * run.normal(z.shape)
        # User-selected wide initial clouds must remain inside the spin box:
        # tanh-coordinate carriers use atanh and cannot start at its poles.
        margin = torch.finfo(z.dtype).eps
        z = z.clamp(-1 + margin, 1 - margin)
        state, cache = self._prepare(run, z)
        for step in range(1, self.max_steps + 1):
            observation.poll()
            pinning, temperature, driver = self._schedule(step)
            z = self._coordinates(state)
            force = -run.objective.gradient(z) + 2 * pinning * z
            state = self._advance(run, state, force, cache, step, temperature, driver)
            if (step % self.candidate_interval == 0 or step == self.max_steps
                    or run.history_interval and step % run.history_interval == 0):
                arrays = state if isinstance(state, tuple) else (state,)
                finite = torch.stack([torch.isfinite(value).all() for value in arrays]).all()
                if not bool(finite):
                    raise FloatingPointError('nonfinite compendium III auxiliary state')
            self._checkpoint(run, readout, self._coordinates(state), step, record)
        return self.max_steps


@dataclass(frozen=True)
class _Vector(_Continuous):
    driver_strength: float = 1.
    damping: float = 1.

    def __post_init__(self):
        super().__post_init__()
        _positive('driver_strength', self.driver_strength, zero=True)
        _positive('damping', self.damping, zero=True)

    def _prepare(self, run, z):
        phase = 2 * math.pi * torch.rand(z.shape, device=z.device,
                                        dtype=z.dtype, generator=run.generator)
        radius = (1 - z.square()).clamp_min(0).sqrt()
        return torch.stack((radius * phase.cos(), radius * phase.sin(), z), -1), {}

    def _coordinates(self, state):
        return state[..., 2]

    def _advance(self, run, spins, force, cache, step, temperature, driver):
        field = self._field(spins, force, self.driver_strength * driver)
        return _sphere_step(spins, field, self.time_step, self.damping)


@dataclass(frozen=True)
class C3DzyaloshinskiiMoriya(_Vector):
    """01. Unit spins with normalized oriented-cycle DM vectors along x."""
    def _field(self, spins, force, amplitude):
        oriented = _skew(spins)
        field = torch.stack((torch.zeros_like(force), -oriented[..., 2], oriented[..., 1]), -1)
        field.mul_(amplitude)
        field[..., 2].add_(force)
        return field


@dataclass(frozen=True)
class C3LipkinMeshkovGlick(_Vector):
    """03. Rank-one collective transverse driver, without an all-ones matrix."""
    anisotropy: float = .7

    def __post_init__(self):
        super().__post_init__()
        _positive('anisotropy', self.anisotropy, zero=True)

    def _field(self, spins, force, amplitude):
        u = amplitude * spins[..., 0].mean(1, keepdim=True).expand_as(force)
        v = amplitude * self.anisotropy * spins[..., 1].mean(1, keepdim=True).expand_as(force)
        return torch.stack((u, v, force), -1)


@dataclass(frozen=True)
class C3LongRangeTransverseIsing(_Vector):
    """07. Product-spin transverse-field annealing using the full input J."""
    def _field(self, spins, force, amplitude):
        return torch.stack((torch.full_like(force, amplitude), torch.zeros_like(force), force), -1)


@dataclass(frozen=True)
class C3KitaevHeisenbergGamma(_Vector):
    """10. Symmetric typed-bond exchange on an artificial periodic chain."""
    heisenberg: float = .25
    kitaev: float = .5
    gamma: float = .25

    def __post_init__(self):
        super().__post_init__()
        for name in ('heisenberg', 'kitaev', 'gamma'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')

    def _field(self, spins, force, amplitude):
        n = spins.shape[1]
        out = torch.zeros_like(spins)
        if n > 1:
            labels = torch.arange(n, device=spins.device) % 3
            right = spins.roll(-1, 1)
            edge_i, edge_j = torch.zeros_like(spins), torch.zeros_like(spins)
            for kind in range(3):
                a, b = (kind + 1) % 3, (kind + 2) % 3
                mask = (labels == kind)[None, :, None]
                for source, destination in ((right, edge_i), (spins, edge_j)):
                    applied = self.heisenberg * source
                    applied[..., kind] += self.kitaev * source[..., kind]
                    applied[..., a] += self.gamma * source[..., b]
                    applied[..., b] += self.gamma * source[..., a]
                    destination.add_(torch.where(mask, applied, 0.))
            bound = max(1., 2 * (abs(self.heisenberg) + abs(self.kitaev) + abs(self.gamma)))
            out = -amplitude * (edge_i + edge_j.roll(1, 1)) / bound
        out[..., 2].add_(force)
        return out


@dataclass(frozen=True)
class C3BosonicJosephson(C3LongRangeTransverseIsing):
    """26. Junction imbalance/tunneling; same specified equations as entry07.

    This declared mathematical overlap intentionally shares the vector kernel.
    """


@dataclass(frozen=True)
class C3SpinDependentOpticalLattice(_Vector):
    """27. Two nonnegative transverse Laplacians, index strides one and two."""
    def _field(self, spins, force, amplitude):
        u, v = spins[..., 0], spins[..., 1]
        return torch.stack((-amplitude * (u - _cycle(u)),
                            -.7 * amplitude * (v - _cycle(v, 2)), force), -1)


@dataclass(frozen=True)
class C3OpticalClock(_Vector):
    """28. Longitudinal alignment and streamed Ramsey escape/readout branches."""
    pulse_interval: int = 32
    pulse_angle: float = .6
    phase_strength: float = .7

    def __post_init__(self):
        super().__post_init__()
        _integer('pulse_interval', self.pulse_interval, 1)
        for name in ('pulse_angle', 'phase_strength'):
            _positive(name, getattr(self, name), zero=True)
        if self.pulse_angle > math.pi:
            raise ValueError('pulse_angle must be <= pi')

    def _advance(self, run, spins, force, cache, step, temperature, driver):
        field = torch.stack((torch.zeros_like(force), torch.zeros_like(force), force), -1)
        spins = _sphere_step(spins, field, self.time_step, self.damping, precession=False)
        if step % self.pulse_interval == 0 and driver:
            phase = self.phase_strength * force.tanh() + .05 * driver * run.normal(force.shape)
            best, best_energy = spins, run.score(run.to_assignments(spins[..., 2]))
            run.retain(run.to_assignments(spins[..., 2]), best_energy)
            # Stream zero,+theta,-theta branches; each receives one refreshed
            # alignment product. No hidden A-fold resident population.
            for angle in (self.pulse_angle * driver, -self.pulse_angle * driver):
                observation.poll()
                candidate = _ramsey(spins, angle, phase)
                z = candidate[..., 2]
                updated_force = -run.objective.gradient(z) + 2 * self._schedule(step)[0] * z
                branch_field = torch.stack((torch.zeros_like(z), torch.zeros_like(z), updated_force), -1)
                candidate = _sphere_step(candidate, branch_field, self.time_step, self.damping, False)
                energy = run.score(run.to_assignments(candidate[..., 2]))
                run.retain(run.to_assignments(candidate[..., 2]), energy)
                improved = energy < best_energy
                best = torch.where(improved[:, None, None], candidate, best)
                best_energy = torch.minimum(energy, best_energy)
            spins = best
        return spins


@dataclass(frozen=True)
class C3DipolarXXZ(_Vector):
    """29. Cutoff r^-3 transverse driver on an artificial periodic line."""
    cutoff: int = 3

    def __post_init__(self):
        super().__post_init__()
        _integer('cutoff', self.cutoff, 1)
        if self.cutoff > 32:
            raise ValueError('cutoff must be <= 32')

    def _field(self, spins, force, amplitude):
        transverse = torch.zeros_like(spins[..., :2])
        n = spins.shape[1]
        bound = 0.
        for distance in range(1, min(self.cutoff, n // 2) + 1):
            weight = distance ** -3
            transverse.add_(_cycle(spins[..., :2], distance), alpha=weight)
            bound += weight
        transverse.mul_(amplitude / max(bound, 1.))
        return torch.cat((transverse, force[..., None]), -1)


@dataclass(frozen=True)
class C3AubryAndre(_Continuous):
    """05. Positive quasiperiodic mobility with a normalized replica phase."""
    mobility_amplitude: float = .6
    phase_speed: float = .1

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.mobility_amplitude < 1 or isinstance(self.mobility_amplitude, bool):
            raise ValueError('mobility_amplitude must be in [0,1)')
        _positive('phase_speed', self.phase_speed, zero=True)

    def _prepare(self, run, z):
        index = torch.arange(z.shape[1], device=z.device, dtype=z.dtype)
        angle = math.pi * (math.sqrt(5) - 1) * index
        phase = 2 * math.pi * torch.rand((z.shape[0], 1), device=z.device,
                                        dtype=z.dtype, generator=run.generator)
        return z, {'cos': angle.cos(), 'sin': angle.sin(), 'phase': phase}

    def _advance(self, run, z, force, cache, step, temperature, driver):
        phase = cache['phase'] + self.phase_speed * self.time_step * step
        mobility = 1 + self.mobility_amplitude * (
            cache['cos'] * phase.cos() - cache['sin'] * phase.sin())
        mobility = mobility / mobility.mean(1, keepdim=True)
        update = z + self.time_step * mobility * force
        if temperature:
            update += (2 * temperature * self.time_step * mobility).sqrt() * run.normal(z.shape)
        return _clip(update)


@dataclass(frozen=True)
class _SkewContinuous(_Continuous):
    circulation: float = .7

    def __post_init__(self):
        super().__post_init__()
        _positive('circulation', self.circulation, zero=True)

    def _prepare(self, run, z):
        return (z, torch.atanh(z)), {}

    def _advance(self, run, state, force, cache, step, temperature, driver):
        z, q = state
        gradient = (1 - z.square()) * force
        direction = self._circulate(run, z, gradient, cache, step)
        q = q + self.time_step * (gradient + self.circulation * driver * direction)
        if temperature:
            q = q + math.sqrt(2 * temperature * self.time_step) * run.normal(q.shape)
        # Explicit numerical guard, not an unconstrained-gradient theorem.
        q = _clip(q, 8)
        return q.tanh(), q


@dataclass(frozen=True)
class C3Haldane(_SkewContinuous):
    """06. Skew cycle transport in tanh coordinates; opposite replica senses."""
    def _circulate(self, run, z, gradient, cache, step):
        orientation = 1 - 2 * (torch.arange(z.shape[0], device=z.device) % 2)
        return orientation[:, None] * _skew(gradient)


@dataclass(frozen=True)
class C3ScalarSpinChirality(_SkewContinuous):
    """21. State-dependent antisymmetric oriented-triangle circulation."""
    def _circulate(self, run, z, gradient, cache, step):
        # Shift disjoint triples to cover endpoints while keeping degree bounded.
        shift = (step // 32) % 3
        return _triangle_skew(z.roll(shift, 1), gradient.roll(shift, 1)).roll(-shift, 1)


@dataclass(frozen=True)
class C3CentralSpin(_SkewContinuous):
    """23. Rank-two center/normalized-neighbor skew flow, no dense K."""
    center_interval: int = 64

    def __post_init__(self):
        super().__post_init__()
        _integer('center_interval', self.center_interval, 1)

    def _prepare(self, run, z):
        return (z, torch.atanh(z)), {'target': _Target(run.objective, occupation=False, normalize=True)}

    def _circulate(self, run, z, gradient, cache, step):
        center = ((step - 1) // self.center_interval) % z.shape[1]
        if cache.get('center') != center:
            indices, values = cache['target'].column(center)
            vector = torch.zeros(z.shape[1], device=z.device, dtype=z.dtype)
            vector[indices] = values.to(z.dtype)
            vector[center] = 0
            cache['vector'] = vector / vector.norm().clamp_min(torch.finfo(z.dtype).tiny)
            cache['center'] = center
        vector = cache['vector']
        result = -gradient[:, center:center + 1] * vector
        result[:, center] += (gradient * vector).sum(1)
        return result


@dataclass(frozen=True)
class C3Holstein(_Continuous):
    """12. Compensated local deformation with exact overdamped relaxation."""
    stiffness: float = 1.
    coupling: float = .8
    relaxation: float = .7

    def __post_init__(self):
        super().__post_init__()
        for name in ('stiffness', 'relaxation'):
            _positive(name, getattr(self, name))
        _positive('coupling', self.coupling, zero=True)

    def _prepare(self, run, z):
        return (z, self.coupling * z.clone()), {}

    def _advance(self, run, state, force, cache, step, temperature, driver):
        z, q = state
        update = z + self.time_step * (force + self.stiffness * self.coupling * (q - self.coupling * z))
        if temperature:
            update += math.sqrt(2 * temperature * self.time_step) * run.normal(z.shape)
        z = _clip(update)
        center = self.coupling * z
        q = center + math.exp(-self.stiffness * self.relaxation * self.time_step) * (q - center)
        return z, q


@dataclass(frozen=True)
class C3RydbergDressedIsing(_Continuous):
    """13. Nonnegative soft-core defect repulsion on a cutoff cycle."""
    driver_strength: float = .6

    def __post_init__(self):
        super().__post_init__()
        _positive('driver_strength', self.driver_strength, zero=True)

    def _advance(self, run, z, force, cache, step, temperature, driver):
        defect = 1 - z.square()
        neighbors = (_cycle(defect) + _cycle(defect, 2) / 65.) / (1 + 1 / 65.)
        update = z + self.time_step * (force + 2 * self.driver_strength * driver * z * neighbors)
        if temperature:
            update += math.sqrt(2 * temperature * self.time_step) * run.normal(z.shape)
        return _clip(update)


@dataclass(frozen=True)
class C3BoseFermiMixture(_Continuous):
    """14. Alternating continuous/binary sectors with refreshed exact fields."""
    role_interval: int = 32

    def __post_init__(self):
        super().__post_init__()
        _integer('role_interval', self.role_interval, 1)

    def _prepare(self, run, z):
        mask = torch.arange(z.shape[1], device=z.device) % 2 == 0
        z[:, ~mask] = torch.where(z[:, ~mask] >= 0, 1., -1.).to(z.dtype)
        return z, {'continuous': mask, 'target': _Target(run.objective, occupation=False, normalize=False)}

    def _advance(self, run, z, force, cache, step, temperature, driver):
        mask = cache['continuous']
        update = z + self.time_step * force
        if temperature:
            update += math.sqrt(2 * temperature * self.time_step) * run.normal(z.shape)
        z = torch.where(mask, _clip(update), z)
        # Recompute after the continuous change, then one binary conditional move.
        site = (step - 1) % z.shape[1]
        field = cache['target'].field(z.double())
        new = torch.where(field[:, site] <= 0, 1., -1.).to(z.dtype)
        z[:, site] = torch.where(mask[site], z[:, site], new)
        if step % self.role_interval == 0:
            new_mask = ~mask
            # Promotion leaves coordinates unchanged. Demotion is an explicit
            # exploratory rounding operation, not a claimed energy descent.
            z = torch.where(new_mask, z, torch.where(z >= 0, 1., -1.))
            cache['continuous'] = new_mask
        return z


def _wall(z, p):
    invalid = ~torch.isfinite(z) | ~torch.isfinite(p)
    return (_clip(z).masked_fill(invalid, float('nan')),
            p.masked_fill(z.abs() > 1, 0.).masked_fill(invalid, float('nan')))


@dataclass(frozen=True)
class _Momentum(_Continuous):
    damping: float = .5
    pulse_interval: int = 32
    pulse_strength: float = .35

    def __post_init__(self):
        super().__post_init__()
        _positive('damping', self.damping, zero=True)
        _integer('pulse_interval', self.pulse_interval, 1)
        _positive('pulse_strength', self.pulse_strength, zero=True)
        if self.pulse_strength > 4:
            raise ValueError('pulse_strength must be <= 4')

    def _prepare(self, run, z):
        return (z, torch.zeros_like(z)), {}


@dataclass(frozen=True)
class C3DegenerateParametricAmplifier(_Momentum):
    """15. Damped force/drift with capped reciprocal quadrature pulses."""
    def _advance(self, run, state, force, cache, step, temperature, driver):
        z, p = state
        p = math.exp(-self.damping * self.time_step) * (p + self.time_step * force)
        if temperature:
            p += math.sqrt(2 * temperature * self.time_step) * run.normal(p.shape)
        z = z + self.time_step * p
        if step % self.pulse_interval == 0:
            amplitude = self.pulse_strength * driver
            z, p = math.exp(amplitude) * z, math.exp(-amplitude) * p
        return _wall(z, p)


@dataclass(frozen=True)
class C3MolmerSorensen(_Momentum):
    """17. Analytically closed geometric collective momentum shear."""
    modes: int = 2

    def __post_init__(self):
        super().__post_init__()
        _integer('modes', self.modes, 1)
        if self.modes > 32:
            raise ValueError('modes must be <= 32')

    def _prepare(self, run, z):
        return (z, torch.zeros_like(z)), {'patterns': _group_patterns(z.shape[1], self.modes, z)}

    def _advance(self, run, state, force, cache, step, temperature, driver):
        z, p = state
        p = p + self.time_step * force
        if step % self.pulse_interval == 0:
            c = cache['patterns']
            p = p - 2 * self.pulse_strength * driver * ((z @ c.T) @ c)
        p *= math.exp(-self.damping * self.time_step)
        if temperature:
            p += math.sqrt(2 * temperature * self.time_step) * run.normal(p.shape)
        return _wall(z + self.time_step * p, p)


@dataclass(frozen=True)
class C3TrappedIonSpinPhonon(_Continuous):
    """16. Complete-square low-rank oscillators, exact frozen-center rotation."""
    modes: int = 4
    frequency: float = 1.
    damping: float = .5
    coupling: float = .5

    def __post_init__(self):
        super().__post_init__()
        _integer('modes', self.modes, 1)
        if self.modes > 32:
            raise ValueError('modes must be <= 32')
        _positive('frequency', self.frequency)
        for name in ('damping', 'coupling'):
            _positive(name, getattr(self, name), zero=True)

    def _prepare(self, run, z):
        c = self.coupling * _group_patterns(z.shape[1], self.modes, z)
        center = z @ c.T
        return (z, center.clone(), torch.zeros_like(center)), {'patterns': c}

    def _advance(self, run, state, force, cache, step, temperature, driver):
        z, q, p = state
        c, frequency, dt = cache['patterns'], self.frequency, self.time_step
        update = z + dt * (force + frequency ** 2 * ((q - z @ c.T) @ c))
        if temperature:
            update += math.sqrt(2 * temperature * dt) * run.normal(z.shape)
        z = _clip(update)
        center = z @ c.T
        residual = q - center
        cosine, sine = math.cos(frequency * dt), math.sin(frequency * dt)
        q = center + cosine * residual + sine / frequency * p
        p = math.exp(-self.damping * dt) * (cosine * p - frequency * sine * residual)
        return z, q, p


@dataclass(frozen=True)
class C3FrohlichPolaron(_Continuous):
    """24. Passive velocity-mode memory with exact orthonormal-mode rotation."""
    modes: int = 4
    damping: float = .5
    coupling: float = .5
    memory_decay: float = .7

    def __post_init__(self):
        super().__post_init__()
        _integer('modes', self.modes, 1)
        if self.modes > 32:
            raise ValueError('modes must be <= 32')
        for name in ('damping', 'coupling'):
            _positive(name, getattr(self, name), zero=True)
        _positive('memory_decay', self.memory_decay)

    def _prepare(self, run, z):
        c = _group_patterns(z.shape[1], self.modes, z)
        return (z, torch.zeros_like(z), torch.zeros((z.shape[0], c.shape[0]),
            device=z.device, dtype=z.dtype)), {'patterns': c}

    def _advance(self, run, state, force, cache, step, temperature, driver):
        z, velocity, memory = state
        c, dt = cache['patterns'], self.time_step
        velocity = velocity + dt * force
        projected = velocity @ c.T
        cosine, sine = math.cos(self.coupling * dt), math.sin(self.coupling * dt)
        rotated = cosine * projected - sine * memory
        memory = (sine * projected + cosine * memory) * math.exp(-self.memory_decay * dt)
        velocity = (velocity + (rotated - projected) @ c) * math.exp(-self.damping * dt)
        if temperature:
            velocity += math.sqrt(2 * temperature * dt) * run.normal(velocity.shape)
        z, velocity = _wall(z + dt * velocity, velocity)
        return z, velocity, memory


@dataclass(frozen=True)
class C3LiebLiniger(_Continuous):
    """25. Logarithmic diversity repulsion on the device-local replica cycle."""
    repulsion: float = .3
    epsilon: float = .1

    def __post_init__(self):
        super().__post_init__()
        _positive('repulsion', self.repulsion, zero=True)
        _positive('epsilon', self.epsilon)

    def _advance(self, run, z, force, cache, step, temperature, driver):
        update = z + self.time_step * (force + self.repulsion * driver * _repulsion(z, self.epsilon))
        if temperature:
            update += math.sqrt(2 * temperature * self.time_step) * run.normal(z.shape)
        return _clip(update)


def _source_energy(run, state):
    return run.score(state if run.spin else (state + 1) * .5)


def _accept(run, old, proposal, temperature, scale):
    """Original source objective, including compact duplicates, decides moves."""
    proposed_energy, old_energy = _source_energy(run, proposal), _source_energy(run, old)
    run.retain(run.to_assignments(old), old_energy)
    run.retain(run.to_assignments(proposal), proposed_energy)
    change = proposed_energy - old_energy
    accepted = change < 0
    if temperature:
        accepted |= _random(change.shape, run).log() < -change / (temperature * scale)
    return torch.where(accepted[:, None], proposal, old)


@dataclass(frozen=True)
class _Discrete(_Base):
    block_size: int = 4

    def __post_init__(self):
        super().__post_init__()
        _integer('block_size', self.block_size, 1)
        if self.block_size > 8:
            raise ValueError('block_size must be <= 8')

    def _setup(self, run, target, state):
        return {}

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target = _Target(run.objective, occupation=False, normalize=False)
        if target.constant:
            return 0
        coefficients = target.matrix if target.bounds is None else target.matrix.values()
        if not bool(torch.isfinite(target.bias).all() & torch.isfinite(coefficients).all()):
            raise FloatingPointError('source coefficient overflow')
        readout = _Readout(run.objective)
        state = run.x.double().clone() if run.spin else run.x.double().mul(2).sub(1)
        cache = self._setup(run, target, state)
        # Normalized proposal temperatures; source-scale coefficients determine
        # all actual gain comparisons. This scale never discards small terms.
        cache['scale'] = _Target(run.objective, occupation=False, normalize=True).scale
        if not bool(torch.isfinite(cache['scale'])):
            raise FloatingPointError('source normalization overflows float64')
        for step in range(1, self.max_steps + 1):
            observation.poll()
            pinning, temperature, driver = self._schedule(step)
            field = target.field(state)
            state = self._move(run, target, state, field, cache, step, pinning, temperature, driver)
            self._checkpoint(run, readout, state, step, record)
        return self.max_steps


def _conditional_proposal(target, state, field, indices):
    matrix = _submatrix(target, indices)
    result = state.clone()
    result[:, indices] = _enumerate_block(state[:, indices], field[:, indices], matrix)
    return result


@dataclass(frozen=True)
class C3FrustratedJ1J2(_Discrete):
    """08. Exact four-site strong-neighborhood blocks with periodic random cover."""
    def __post_init__(self):
        super().__post_init__()
        if self.block_size != 4:
            raise ValueError('J1-J2 plaquettes require block_size=4')

    def _setup(self, run, target, state):
        return {'blocks': _neighborhoods(target, min(4, target.n))}

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        if step % 4 == 0:
            # One shared, bounded index transfer; never transfer replica states.
            ids = torch.randperm(target.n, generator=run.generator, device=state.device)[:min(4, target.n)].cpu().tolist()
        else:
            ids = cache['blocks'][(step - 1) % target.n]
        proposal = _conditional_proposal(target, state, field, ids)
        return _accept(run, state, proposal, 0., cache['scale'])


@dataclass(frozen=True)
class C3RingExchange(_Discrete):
    """22. Exact all-mask updates on translated artificial four-site loops."""
    def __post_init__(self):
        super().__post_init__()
        if self.block_size != 4:
            raise ValueError('ring enumeration requires block_size=4')

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        ids = [((step - 1) + k) % target.n for k in range(min(4, target.n))]
        return _accept(run, state, _conditional_proposal(target, state, field, ids), 0., cache['scale'])


@dataclass(frozen=True)
class C3Kanamori(_Discrete):
    """18. Shell channels: single, pair, four-site and full capped-shell solve."""
    block_size: int = 8

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        size = min((1, 2, 4, self.block_size)[(step - 1) % 4], target.n)
        shell = ((step - 1) // 4 * self.block_size) % target.n
        ids = [(shell + i) % target.n for i in range(size)]
        return _accept(run, state, _conditional_proposal(target, state, field, ids), 0., cache['scale'])


def _star_candidates(old, field, matrix):
    """Two center branches; exact boundary, with leaf-leaf terms excluded only
    from the proposal surrogate (full source scoring follows)."""
    boundary = field - old @ matrix
    candidates = []
    for center in (-1., 1.):
        proposed = torch.where(boundary + center * matrix[0] <= 0, 1., -1.).to(old.dtype)
        proposed[:, 0] = center
        candidates.append(proposed)
    return candidates


@dataclass(frozen=True)
class C3AndersonImpurity(_Discrete):
    """19. Two linear-cost star branches with exact leaf-coupling acceptance."""
    block_size: int = 8

    def _setup(self, run, target, state):
        return {'blocks': _neighborhoods(target, self.block_size)}

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        ids = cache['blocks'][(step - 1) % target.n]
        matrix = _submatrix(target, ids)
        for local in _star_candidates(state[:, ids], field[:, ids], matrix):
            proposal = state.clone()
            proposal[:, ids] = local
            state = _accept(run, state, proposal, 0., cache['scale'])
        return state


@dataclass(frozen=True)
class C3ResonantDipoleExchange(_Discrete):
    """02. Resonance-ranked bounded pairs mixed with unconstrained single flips."""
    single_probability: float = .25
    detuning: float = .1
    pair_candidates: int = 8

    def __post_init__(self):
        super().__post_init__()
        if (isinstance(self.single_probability, bool) or not math.isfinite(self.single_probability)
                or not 0 < self.single_probability <= 1):
            raise ValueError('single_probability must be in (0,1] for unconstrained QUBO')
        _positive('detuning', self.detuning)
        _integer('pair_candidates', self.pair_candidates, 1)
        if self.pair_candidates > 64:
            raise ValueError('pair_candidates must be <=64')

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        n, batch = target.n, state.shape[0]
        anchor = (step - 1) % n
        proposal = state.clone()
        proposal[:, anchor] *= -1
        if n > 1:
            ids = [(anchor + 1 + i) % n for i in range(min(self.pair_candidates, n - 1))]
            gains = -2 * state * field / cache['scale']
            coupling = torch.stack([target.entry(anchor, j) for j in ids]) / cache['scale']
            weights = (coupling.square() + 1e-6) / (
                self.detuning ** 2 + (gains[:, anchor:anchor + 1] - gains[:, ids]).square())
            valid = state[:, ids] != state[:, anchor:anchor + 1]
            weights = weights.clamp_max(1e6) * valid
            selected = weights.argmax(1)
            destinations = torch.tensor(ids, device=state.device)[selected]
            exchange = (_random((batch,), run) >= self.single_probability) & valid.any(1)
            rows = torch.arange(batch, device=state.device)
            proposal[rows, destinations] *= torch.where(exchange, -1., 1.)
        return _accept(run, state, proposal, temperature, cache['scale'])


@dataclass(frozen=True)
class C3PariserParrPople(_Discrete):
    """20. Bounded nonlocal charge-transfer pool plus unconstrained singles."""
    pair_candidates: int = 8

    def __post_init__(self):
        super().__post_init__()
        _integer('pair_candidates', self.pair_candidates, 1)
        if self.pair_candidates > 64:
            raise ValueError('pair_candidates must be <=64')

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        anchor = (step - 1) % target.n
        best = state.clone()
        best[:, anchor] *= -1
        if target.n > 1:
            # Fixed coprime-like stride plus a rotating origin ensures global
            # destinations without an n-by-n per-replica scan.
            ids = [(anchor + 1 + i * max(1, target.n // self.pair_candidates)) % target.n
                   for i in range(min(self.pair_candidates, target.n - 1))]
            ids = [i for i in dict.fromkeys(ids) if i != anchor]
            gains = -2 * state * field
            couplings = torch.stack([target.entry(anchor, j) for j in ids])
            pairs = (gains[:, anchor:anchor + 1] + gains[:, ids]
                     + 4 * couplings * state[:, anchor:anchor + 1] * state[:, ids])
            values, selected = pairs.min(1)
            destination = torch.tensor(ids, device=state.device)[selected]
            rows = torch.arange(state.shape[0], device=state.device)
            best[rows, destination] *= torch.where(values < gains[:, anchor], -1., 1.)
        return _accept(run, state, best, temperature, cache['scale'])


def _conflict_edges(target):
    matrix = target.matrix
    if target.bounds is None:
        return None
    rows = torch.repeat_interleave(torch.arange(target.n, device=matrix.device),
        matrix.crow_indices().diff(), output_size=matrix.values().numel())
    valid = (matrix.values() != 0) & (rows != matrix.col_indices())
    return rows[valid], matrix.col_indices()[valid]


def _independent_mask(target, priorities, edges=None):
    """Random-priority independent set in the FULL nonzero objective graph."""
    n, batch = target.n, priorities.shape[0]
    if target.bounds is None:
        # Complete-graph conservative fallback remains valid for every dense
        # operator; sparse inputs take actual graph parallelism below.
        winner = priorities.argmax(1, keepdim=True)
        mask = torch.zeros_like(priorities, dtype=torch.bool)
        return mask.scatter_(1, winner, priorities.gather(1, winner) >= 0)
    rows, cols = _conflict_edges(target) if edges is None else edges
    maximum = torch.full_like(priorities, -1.)
    chunk = max(1, 1_048_576 // batch)
    for start in range(0, cols.numel(), chunk):
        observation.poll()
        destination = rows[start:start + chunk].expand(batch, -1)
        maximum.scatter_reduce_(1, destination, priorities[:, cols[start:start + chunk]],
                                reduce='amax', include_self=True)
    return (priorities >= 0) & (priorities > maximum)


@dataclass(frozen=True)
class C3RydbergPXP(_Discrete):
    """04. Fair randomized conflict-free flip masks; exact whole-set scoring."""
    def _setup(self, run, target, state):
        return {'edges': _conflict_edges(target)}

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        gains = -2 * state * field / cache['scale']
        eligible = gains < 0 if temperature == 0 else (
            _random(state.shape, run) < torch.sigmoid(-gains / temperature))
        priorities = _random(state.shape, run).masked_fill(~eligible, -1.)
        mask = _independent_mask(target, priorities, cache['edges'])
        return _accept(run, state, torch.where(mask, -state, state), temperature, cache['scale'])


@dataclass(frozen=True)
class C3BilinearBiquadratic(_Discrete):
    """09. Ternary heat-bath with annealed sparse occupancy correlations."""
    occupancy_strength: float = .8

    def __post_init__(self):
        super().__post_init__()
        value = self.occupancy_strength
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('occupancy_strength must be finite')

    def _setup(self, run, target, state):
        state.masked_fill_(_random(state.shape, run) < .3, 0.)
        return {}

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        site = (step - 1) % target.n
        labels = state.new_tensor([-1., 0., 1.])
        occupancy = _cycle(state.square())[:, site]
        costs = (field[:, site:site + 1] / cache['scale'] * labels
                 + pinning * (1 - labels.square())
                 + self.occupancy_strength * driver * occupancy[:, None] * labels.square())
        if temperature:
            weights = torch.softmax(-costs / temperature, 1)
            selected = torch.multinomial(weights, 1, generator=run.generator).squeeze(1)
        else:
            selected = costs.argmin(1)
        state[:, site] = labels[selected]
        # Zero sites are conditionally rounded only in readout, without
        # destroying the ternary dynamics' vacancies.
        return state

    def _checkpoint(self, run, readout, state, step, record, force=False):
        if (force or step % self.candidate_interval == 0 or step == self.max_steps
                or run.history_interval and step % run.history_interval == 0):
            candidate = readout.readout(state, max(1, self.polish_sweeps), conditional=True)
            run.publish(candidate, step)
            run.x.copy_(run.best_x)
            run.energies.copy_(run.best)
            record(step)


def _hofstadter_filter(amplitude, diagonal, hopping, phase):
    """Hermitian cycle imaginary-time filter, shifted by Gershgorin bound."""
    n = amplitude.shape[1]
    if n < 2:
        return amplitude
    diagonal = diagonal.to(amplitude.real.dtype)
    factor = complex(math.cos(phase), math.sin(phase))
    product = diagonal * amplitude - .5 * hopping * (
        factor * amplitude.roll(-1, 1) + factor.conjugate() * amplitude.roll(1, 1))
    lower = diagonal.amin(1, keepdim=True) - hopping
    upper = diagonal.amax(1, keepdim=True) + hopping
    tau = .9 / (upper - lower).clamp_min(1e-12)
    result = amplitude - tau * (product - lower * amplitude)
    return result / result.norm(dim=1, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True)
class C3HarperHofstadter(_Discrete):
    """11. Complex flux filter selects a capped exact conditional block."""
    hopping: float = .5
    flux_phase: float = .7
    filter_steps: int = 3

    def __post_init__(self):
        super().__post_init__()
        _positive('hopping', self.hopping, zero=True)
        if isinstance(self.flux_phase, bool) or not math.isfinite(self.flux_phase):
            raise ValueError('flux_phase must be finite')
        _integer('filter_steps', self.filter_steps, 1)
        if self.filter_steps > 32:
            raise ValueError('filter_steps must be <=32')

    def _setup(self, run, target, state):
        phase = 2 * math.pi * _random(state.shape, run).to(run.q.dtype)
        return {'amplitude': torch.polar(torch.ones_like(phase), phase) / math.sqrt(target.n)}

    def _move(self, run, target, state, field, cache, step, pinning, temperature, driver):
        diagonal = (-2 * state * field / cache['scale']).clamp(-2, 2)
        amplitude = cache['amplitude']
        for _ in range(self.filter_steps):
            observation.poll()
            amplitude = _hofstadter_filter(amplitude, diagonal, self.hopping, self.flux_phase)
        cache['amplitude'] = amplitude
        # A shared block selected from mean intensity retains a single induced
        # matrix across replicas. A deterministic uniform cover prevents neglect.
        if step % 4 == 0:
            ids = [(step + i) % target.n for i in range(min(self.block_size, target.n))]
        else:
            ids = amplitude.abs().square().mean(0).topk(min(self.block_size, target.n)).indices.cpu().tolist()
        return _accept(run, state, _conditional_proposal(target, state, field, ids), 0., cache['scale'])


COMPENDIUM_III_A_CLASSES = {
    'lib_c3_dzyaloshinskii_moriya': C3DzyaloshinskiiMoriya,
    'lib_c3_resonant_dipole_exchange': C3ResonantDipoleExchange,
    'lib_c3_lipkin_meshkov_glick': C3LipkinMeshkovGlick,
    'lib_c3_rydberg_pxp': C3RydbergPXP,
    'lib_c3_aubry_andre': C3AubryAndre,
    'lib_c3_haldane': C3Haldane,
    'lib_c3_long_range_transverse_ising': C3LongRangeTransverseIsing,
    'lib_c3_frustrated_j1_j2': C3FrustratedJ1J2,
    'lib_c3_bilinear_biquadratic': C3BilinearBiquadratic,
    'lib_c3_kitaev_heisenberg_gamma': C3KitaevHeisenbergGamma,
    'lib_c3_harper_hofstadter': C3HarperHofstadter,
    'lib_c3_holstein': C3Holstein,
    'lib_c3_rydberg_dressed_ising': C3RydbergDressedIsing,
    'lib_c3_bose_fermi_mixture': C3BoseFermiMixture,
    'lib_c3_degenerate_parametric_amplifier': C3DegenerateParametricAmplifier,
    'lib_c3_trapped_ion_spin_phonon': C3TrappedIonSpinPhonon,
    'lib_c3_molmer_sorensen': C3MolmerSorensen,
    'lib_c3_kanamori': C3Kanamori,
    'lib_c3_anderson_impurity': C3AndersonImpurity,
    'lib_c3_pariser_parr_pople': C3PariserParrPople,
    'lib_c3_scalar_spin_chirality': C3ScalarSpinChirality,
    'lib_c3_ring_exchange': C3RingExchange,
    'lib_c3_central_spin': C3CentralSpin,
    'lib_c3_frohlich_polaron': C3FrohlichPolaron,
    'lib_c3_lieb_liniger': C3LiebLiniger,
    'lib_c3_bosonic_josephson': C3BosonicJosephson,
    'lib_c3_spin_dependent_optical_lattice': C3SpinDependentOpticalLattice,
    'lib_c3_optical_clock': C3OpticalClock,
    'lib_c3_dipolar_xxz': C3DipolarXXZ,
}
