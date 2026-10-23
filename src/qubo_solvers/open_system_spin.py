"""Six classical search engines specified by the open-system compendium.

These are engineered QUBO heuristics, not entangled quantum simulations. The
physical field F is minus PopulationObjective's energy gradient, with an
annealed endpoint pinning force. Only original-source FP64 energies retain
incumbents. State is O(batch*n); low-rank patterns never form dense projectors.
"""

from dataclasses import dataclass
import math

import torch

from . import observation
from ._dynamics import finite, positive
from ._population import PopulationSolver, _mm
from .solvers import _integer


def _unit(vector):
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(vector.dtype).tiny)


def _rotate_x(vector, angle):
    u, v, z = vector.unbind(-1)
    c, s = angle.cos(), angle.sin()
    return torch.stack((u, v * c - z * s, v * s + z * c), -1)


def _rotate_y(vector, angle):
    u, v, z = vector.unbind(-1)
    c, s = angle.cos(), angle.sin()
    return torch.stack((u * c + z * s, v, -u * s + z * c), -1)


def _rotate_z(vector, angle):
    u, v, z = vector.unbind(-1)
    c, s = angle.cos(), angle.sin()
    return torch.stack((u * c - v * s, u * s + v * c, z), -1)


def _precess(vector, field, dt):
    """Exact frozen-field n x B flow, including the zero-field limit."""
    angle = field.norm(dim=-1, keepdim=True) * dt
    first = torch.linalg.cross(-field, vector)
    second = torch.linalg.cross(-field, first)
    return (vector + dt * torch.sinc(angle / math.pi) * first
            + .5 * dt * dt * torch.sinc(angle / (2 * math.pi)).square() * second)


def _alignment_coefficients(c, strength):
    """Stable tanh addition and transverse scale for nonnegative strength.

    Exact antiparallel poles remain poles. The exponential expression avoids
    inverse hyperbolic evaluation and preserves nearly polar transverse state.
    """
    decay = (-strength).exp()
    denominator = (1 + c) + (1 - c) * decay.square()
    safe = denominator.clamp_min(torch.finfo(c.dtype).tiny)
    updated = ((1 + c) - (1 - c) * decay.square()) / safe
    scale = 2 * decay / safe
    pole = c <= -1
    # The transverse limiting scale at the antiparallel pole is exp(strength).
    # Using one would incorrectly freeze a nonzero transverse perturbation
    # when FP32 has rounded its longitudinal coordinate to exactly +/-1.
    limit = strength.clamp(max=math.log(torch.finfo(c.dtype).max) - 2).exp()
    return torch.where(pole, c, updated), torch.where(pole, limit, scale)


def _align_z(vector, force, duration):
    direction = torch.where(force < 0, -torch.ones_like(force), torch.ones_like(force))
    c = (vector[..., 2] * direction).clamp(-1, 1)
    z, scale = _alignment_coefficients(c, force.abs() * duration)
    return torch.cat((vector[..., :2] * scale[..., None], (direction * z)[..., None]), -1)


def _align_field(vector, field, duration):
    magnitude = field.norm(dim=-1, keepdim=True)
    axis = field / magnitude.clamp_min(torch.finfo(field.dtype).tiny)
    c = (vector * axis).sum(-1, keepdim=True).clamp(-1, 1)
    updated, scale = _alignment_coefficients(c, magnitude * duration)
    out = axis * updated + (vector - c * axis) * scale
    return torch.where(magnitude > 0, out, vector)


def _kicked_map(vector, rotation, twist):
    rotated = _rotate_y(vector, rotation)
    return _rotate_z(rotated, twist * rotated[..., 2])


def _tact_pulse(vector, strength):
    """Strang x/y/x rotation; each angle uses the updated axis component."""
    vector = _rotate_x(vector, strength * vector[..., 0])
    vector = _rotate_y(vector, -2 * strength * vector[..., 1])
    return _rotate_x(vector, strength * vector[..., 0])


def _patterns(n, rank, like):
    """Normalized disjoint group indicators; no spectral preprocessing."""
    rank = min(n, rank)
    result = like.new_zeros((n, rank))
    if rank:
        indices = torch.arange(n, device=like.device)
        result[indices, indices.remainder(rank)] = 1
        result /= result.square().sum(0, keepdim=True).sqrt()
    return result


def _low_rank(values, patterns):
    return (values @ patterns) @ patterns.T


def _oat_shear(z, momentum, patterns, strength):
    return momentum - strength * _low_rank(z, patterns)


def _cycle_skew(values):
    """Oriented cycle K, ||K||_2 <= 1; n=1,2 correctly give zero."""
    return .5 * (torch.roll(values, 1, -1) - torch.roll(values, -1, -1))


@dataclass(frozen=True)
class _OpenSpinBase(PopulationSolver):
    max_steps: int = 512
    time_step: float = .08
    candidate_interval: int = 16
    relaxation_fraction: float = .2
    pinning: float = .3
    initial_scale: float = .1
    polish_steps: int = 8
    normalize: bool = True
    dynamics_dtype: object = None
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('candidate_interval', self.candidate_interval, 1)
        _integer('polish_steps', self.polish_steps, 0)
        positive('time_step', self.time_step)
        positive('pinning', self.pinning, zero=True)
        positive('initial_scale', self.initial_scale)
        if self.initial_scale >= 1:
            raise ValueError('initial_scale must be less than one')
        for name in ('relaxation_fraction', 'sparse_threshold'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must lie in [0, 1]')
        if self.relaxation_fraction == 1:
            raise ValueError('relaxation_fraction must be less than one')
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if self.dynamics_dtype not in (None, 'float32', 'float64', torch.float32, torch.float64):
            raise ValueError('dynamics_dtype must be float32, float64, or None')

    def _population_workspace(self, n, batch, width):
        # Includes simultaneous old/new rotations, fields, FP64 candidate casts
        # and reduction intermediates; independent of objective representation.
        rank = min(n, getattr(self, 'rank', 0))
        return (100 * width + 64) * batch * n + 4 * width * rank * (n + batch)

    def _controls(self, step):
        fraction = 1. if self.max_steps <= 1 else step / (self.max_steps - 1)
        fraction = observation.schedule_fraction(fraction)
        fraction = min(1., max(0., fraction / (1 - self.relaxation_fraction)))
        fraction = fraction * fraction * (3 - 2 * fraction)
        return 1 - fraction, self.pinning * fraction

    def _force(self, run, z, pinning):
        return -run.objective.gradient(z) + 2 * pinning * z

    def _initial_sphere(self, run):
        azimuth = torch.rand(run.x.shape, device=run.q.device, dtype=run.q.dtype,
                             generator=run.generator) * (2 * math.pi)
        z = (2 * run.initialbits.to(run.q.dtype) - 1) * self.initial_scale
        radius = (1 - z.square()).sqrt()
        return torch.stack((radius * azimuth.cos(), radius * azimuth.sin(), z), -1)

    def _due(self, run, step):
        return (step % self.candidate_interval == 0 or step == self.max_steps
                or (run.history_interval is not None and step % run.history_interval == 0))

    def _checkpoint(self, run, record, z, step):
        run.iterations.fill_(step)
        if self._due(run, step):
            run.publish(z, step)
            record(step)

    def _before_pulse(self, run, record, z, step):
        # Publish a beneficial alignment result before exploratory rotations.
        if self._due(run, step):
            run.publish(z, step)
            record(step)

    def _finish(self, run, record, *state):
        finite(*state)
        # Bounded single-flip proposals. Source precision alone accepts moves;
        # dynamics-precision force merely chooses the proposed coordinate.
        sample, energy = run.best_x.clone(), run.best.clone()
        rows = torch.arange(sample.shape[0], device=sample.device)
        for _ in range(self.polish_steps):
            observation.poll()
            spins = sample if run.spin else 2 * sample - 1
            change = -2 * spins * run.objective.gradient(spins)
            index = change.argmin(-1)
            trial = sample.clone()
            trial[rows, index] = -trial[rows, index] if run.spin else 1 - trial[rows, index]
            score = run.score(trial)
            accept = score < energy
            sample = torch.where(accept[:, None], trial, sample)
            energy = torch.where(accept, score, energy)
        run.x.copy_(sample)
        run.energies.copy_(energy)
        run.retain(sample, energy)
        record(self.max_steps)
        return self.max_steps


@dataclass(frozen=True)
class NoisyLaserAtom(_OpenSpinBase):
    """Classical noisy Bloch vectors with engineered target-only alignment."""
    damping: float = 1.
    drive: float = 1.
    phase_diffusion: float = .15

    def __post_init__(self):
        super().__post_init__()
        for name in ('damping', 'drive', 'phase_diffusion'):
            positive(name, getattr(self, name), zero=True)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        vector = self._initial_sphere(run)
        for k in range(self.max_steps):
            observation.poll()
            amplitude, pinning = self._controls(k)
            force = self._force(run, vector[..., 2], pinning)
            field = torch.stack((torch.full_like(force, self.drive * amplitude),
                                 torch.zeros_like(force), force), -1)
            vector = _precess(vector, field, self.time_step)
            vector = _align_z(vector, force, self.damping * self.time_step)
            if amplitude and self.phase_diffusion:
                angle = run.normal(force.shape) * math.sqrt(2 * self.phase_diffusion * amplitude * self.time_step)
                vector = _rotate_z(vector, angle)
            if (k + 1) % 16 == 0:
                vector = _unit(vector)
            self._checkpoint(run, record, vector[..., 2], k + 1)
        return self._finish(run, record, vector)


@dataclass(frozen=True)
class KickedTop(_OpenSpinBase):
    """Longitudinal alignment followed by y rotation and z-dependent twist."""
    damping: float = 1.
    rotation: float = .3
    twisting: float = 1.
    pulse_interval: int = 4
    replica_spread: float = .2

    def __post_init__(self):
        super().__post_init__()
        positive('damping', self.damping, zero=True)
        for name in ('rotation', 'twisting'):
            if isinstance(getattr(self, name), bool) or not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        positive('replica_spread', self.replica_spread, zero=True)
        if self.replica_spread > 1:
            raise ValueError('replica_spread must not exceed one')
        _integer('pulse_interval', self.pulse_interval, 1)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        vector = self._initial_sphere(run)
        spread = 1 + self.replica_spread * (2 * torch.rand((run.x.shape[0], 1),
                 device=run.q.device, dtype=run.q.dtype, generator=run.generator) - 1)
        for k in range(self.max_steps):
            observation.poll()
            amplitude, pinning = self._controls(k)
            force = self._force(run, vector[..., 2], pinning)
            vector = _align_z(vector, force, self.damping * self.time_step)
            if amplitude and (k + 1) % self.pulse_interval == 0:
                self._before_pulse(run, record, vector[..., 2], k + 1)
                vector = _kicked_map(vector, self.rotation * amplitude * spread,
                                     self.twisting * amplitude * spread)
            if (k + 1) % 16 == 0:
                vector = _unit(vector)
            self._checkpoint(run, record, vector[..., 2], k + 1)
        return self._finish(run, record, vector)


@dataclass(frozen=True)
class KaneMele(_OpenSpinBase):
    """Paired-helicity skew-gradient flow on tanh coordinates, cycle K."""
    chirality: float = 1.

    def __post_init__(self):
        super().__post_init__()
        positive('chirality', self.chirality, zero=True)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        q = torch.atanh((2 * run.initialbits.to(run.q.dtype) - 1) * self.initial_scale)
        matrix = run.objective.matrix
        if matrix.layout == torch.strided:
            row_bound = matrix.abs().sum(-1).max()
        else:
            absolute = torch.sparse_csr_tensor(matrix.crow_indices(), matrix.col_indices(),
                                               matrix.values().abs(), size=matrix.shape)
            row_bound = _mm(absolute, torch.ones_like(q[:1])).max()
        # Global Hessian bound for U(tanh(q)) at any frozen pinning <= max.
        lipschitz = (3 * row_bound + 2 * run.objective.field.abs().max() + 6 * self.pinning).clamp_min(1e-12)
        helicity = (1 - 2 * torch.arange(q.shape[0], device=q.device).remainder(2)).to(q.dtype)[:, None]
        for k in range(self.max_steps):
            observation.poll()
            amplitude, pinning = self._controls(k)
            z = q.tanh()
            gradient = (1 - z.square()) * self._force(run, z, pinning)
            alpha = self.chirality * amplitude
            dt = torch.minimum(q.new_tensor(self.time_step), .9 / (lipschitz * (1 + alpha * alpha)))
            q = q + dt * (gradient + alpha * helicity * _cycle_skew(gradient))
            self._checkpoint(run, record, q.tanh(), k + 1)
        return self._finish(run, record, q)


@dataclass(frozen=True)
class BCS(_OpenSpinBase):
    """Low-rank transverse pairing with frozen-field damping/precession."""
    damping: float = 1.
    pairing: float = .6
    rank: int = 4
    precession: float = 1.

    def __post_init__(self):
        super().__post_init__()
        for name in ('damping', 'pairing', 'precession'):
            positive(name, getattr(self, name), zero=True)
        _integer('rank', self.rank, 0)
        if self.rank > 32:
            raise ValueError('rank must not exceed 32')

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        vector = self._initial_sphere(run)
        patterns = _patterns(run.x.shape[1], self.rank, run.x)
        for k in range(self.max_steps):
            observation.poll()
            amplitude, pinning = self._controls(k)
            longitudinal = self._force(run, vector[..., 2], pinning)
            strength = self.pairing * amplitude
            # At the endpoint both transverse fields vanish exactly.
            field = torch.stack((strength * _low_rank(vector[..., 0], patterns),
                                 strength * _low_rank(vector[..., 1], patterns), longitudinal), -1)
            vector = _align_field(vector, field, self.damping * self.time_step)
            vector = _precess(vector, field, self.precession * self.time_step)
            if (k + 1) % 16 == 0:
                vector = _unit(vector)
            self._checkpoint(run, record, vector[..., 2], k + 1)
        return self._finish(run, record, vector)


@dataclass(frozen=True)
class OneAxisTwisting(_OpenSpinBase):
    """Exact low-rank momentum shear between target kick and damping/drift."""
    damping: float = .7
    twisting: float = .3
    rank: int = 4
    pulse_interval: int = 4
    temperature: float = .02

    def __post_init__(self):
        super().__post_init__()
        for name in ('damping', 'temperature'):
            positive(name, getattr(self, name), zero=True)
        if isinstance(self.twisting, bool) or not math.isfinite(self.twisting) or abs(self.twisting) > 4:
            raise ValueError('twisting must be finite with absolute value <= 4')
        _integer('rank', self.rank, 0)
        if self.rank > 32:
            raise ValueError('rank must not exceed 32')
        _integer('pulse_interval', self.pulse_interval, 1)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        z = (2 * run.initialbits.to(run.q.dtype) - 1) * self.initial_scale
        momentum = torch.zeros_like(z)
        patterns = _patterns(z.shape[1], self.rank, z)
        decay = math.exp(-self.damping * self.time_step)
        noise = math.sqrt(self.temperature * -math.expm1(-2 * self.damping * self.time_step))
        for k in range(self.max_steps):
            observation.poll()
            amplitude, pinning = self._controls(k)
            momentum = momentum + self.time_step * self._force(run, z, pinning)
            if amplitude and (k + 1) % self.pulse_interval == 0:
                momentum = _oat_shear(z, momentum, patterns, self.twisting * amplitude)
            momentum = decay * momentum
            if amplitude and noise:
                momentum = momentum + noise * math.sqrt(amplitude) * run.normal(z.shape)
            trial = z + self.time_step * momentum
            invalid = ~torch.isfinite(trial) | ~torch.isfinite(momentum)
            momentum = torch.where(trial.abs() >= 1, torch.zeros_like(momentum), momentum)
            z = trial.clamp(-1, 1).masked_fill(invalid, float('nan'))
            if self._due(run, k + 1):
                healthy = (torch.isfinite(z) & torch.isfinite(momentum)).all(-1)
                fallback = run.best_x if run.spin else 2 * run.best_x - 1
                z = torch.where(healthy[:, None], z, self.initial_scale * fallback)
                momentum = torch.where(healthy[:, None], momentum, torch.zeros_like(momentum))
                run.reasons.masked_fill_(~healthy, -1)
            self._checkpoint(run, record, z, k + 1)
        return self._finish(run, record, z, momentum)


@dataclass(frozen=True)
class TwoAxisCounterTwisting(_OpenSpinBase):
    """Target alignment alternating with bounded nonlinear TACT pulses."""
    damping: float = 1.
    twisting: float = .3
    pulse_interval: int = 4

    def __post_init__(self):
        super().__post_init__()
        positive('damping', self.damping, zero=True)
        if isinstance(self.twisting, bool) or not math.isfinite(self.twisting) or abs(self.twisting) > 4:
            raise ValueError('twisting must be finite with absolute value <= 4')
        _integer('pulse_interval', self.pulse_interval, 1)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        vector = self._initial_sphere(run)
        for k in range(self.max_steps):
            observation.poll()
            amplitude, pinning = self._controls(k)
            force = self._force(run, vector[..., 2], pinning)
            vector = _align_z(vector, force, self.damping * self.time_step)
            if amplitude and (k + 1) % self.pulse_interval == 0:
                self._before_pulse(run, record, vector[..., 2], k + 1)
                vector = _tact_pulse(vector, self.twisting * amplitude)
            if (k + 1) % 16 == 0:
                vector = _unit(vector)
            self._checkpoint(run, record, vector[..., 2], k + 1)
        return self._finish(run, record, vector)


OPEN_SPIN_CLASSES = {
    'lib_noisy_laser_atom': NoisyLaserAtom,
    'lib_kicked_top': KickedTop,
    'lib_kane_mele': KaneMele,
    'lib_bcs': BCS,
    'lib_one_axis_twisting': OneAxisTwisting,
    'lib_two_axis_counter_twisting': TwoAxisCounterTwisting,
}
