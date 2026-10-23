"""Seven classical open-system QUBO proposals from the October 2026 compendium.

These are the explicitly engineered classical updates, not quantum simulations.
All use the same diagonal-free multilinear target, original FP64 binary scoring,
inelastic walls and bounded sequential FP64 readout. Dense GEMM and CSR SpMM
share one implementation; auxiliary state is O(batch*n), never O(batch*n*n).
Schedules use smoothstep, with a final zero-noise/zero-driver interval. Random
draws use the library's seeded device generator (not counter-keyed Philox API);
batch size and device are therefore part of reproducibility metadata.
"""

from dataclasses import dataclass
import math

import torch

from . import observation
from ._dynamics import positive
from ._population import PopulationSolver
from .hamiltonian_continuous import _Target
from .solvers import _integer


def _row_norms(objective):
    """Absolute normalized coupling row sums, without densifying CSR."""
    matrix = objective.matrix
    if matrix.layout == torch.strided:
        return matrix.abs().sum(1)
    rows = torch.repeat_interleave(torch.arange(objective.n, device=matrix.device),
        matrix.crow_indices()[1:] - matrix.crow_indices()[:-1],
        output_size=matrix.values().numel())
    return torch.zeros_like(objective.field).index_add_(0, rows, matrix.values().abs())


def _wall(z, p):
    """The specified inelastic clipping heuristic, not exact reflection."""
    hit = z.abs() > 1
    # Saturation must not turn an overflow into a plausible binary candidate.
    invalid = ~torch.isfinite(z) | ~torch.isfinite(p)
    return (z.clamp(-1, 1).masked_fill(invalid, float('nan')),
            p.masked_fill(hit, 0.).masked_fill(invalid, float('nan')))


def _clip(value, bound=1.):
    return value.clamp(-bound, bound).masked_fill(~torch.isfinite(value), float('nan'))


def _rotation(p, r, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return cosine*p - sine*r, sine*p + cosine*r


def _oscillator(r, p, frequency, dt):
    cosine, sine = math.cos(frequency*dt), math.sin(frequency*dt)
    return cosine*r + sine*p/frequency, cosine*p - frequency*sine*r


def _thermostat(p, decay, noise_scale, temperature, run):
    p = p*decay
    if temperature > 0:
        p = p + math.sqrt(temperature)*noise_scale*run.normal(p.shape)
    return p


def _ring_laplacian(value):
    # Periodic finite-difference convention: n=2 has two parallel links,
    # n=1 has zero Laplacian. Its exact spectrum is 4*sin(pi*k/n)**2.
    return 2*value - value.roll(1, 1) - value.roll(-1, 1)


def _positive_filter(value, length, terms):
    """Fixed positive Neumann polynomial; retains the uniform mode."""
    omega = 1/(1 + 4*length*length)
    power = value
    total = omega*power
    for _ in range(1, terms):
        power = power - omega*(power + length*length*_ring_laplacian(power))
        total = total + omega*power
    return total


def _sideband_rates(frequency, linewidth, coupling):
    """Population-linewidth convention, fixed red detuning Delta=-omega."""
    minus = torch.full_like(frequency, 4*coupling*coupling/linewidth)
    plus = coupling*coupling*linewidth/(linewidth*linewidth/4 + 4*frequency.square())
    return minus, plus


@dataclass(frozen=True)
class _OpenContinuous(PopulationSolver):
    max_steps: int = 1000
    time_step: float = .05
    candidate_interval: int = 32
    polish_sweeps: int = 1
    pinning: float = .5
    temperature: float = .1
    terminal_fraction: float = .15
    normalize: bool = True
    dynamics_dtype: object = None
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('candidate_interval', self.candidate_interval, 1)
        _integer('polish_sweeps', self.polish_sweeps, 0)
        for name in ('time_step',):
            positive(name, getattr(self, name))
        for name in ('pinning', 'temperature'):
            positive(name, getattr(self, name), zero=True)
        if (isinstance(self.terminal_fraction, bool) or not math.isfinite(self.terminal_fraction)
                or not 0 <= self.terminal_fraction < 1):
            raise ValueError('terminal_fraction must be within [0, 1)')
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if self.dynamics_dtype not in (None, torch.float32, torch.float64, 'float32', 'float64'):
            raise ValueError('dynamics_dtype must be float32, float64, or None')
        if (isinstance(self.sparse_threshold, bool) or not math.isfinite(self.sparse_threshold)
                or not 0 <= self.sparse_threshold <= 1):
            raise ValueError('sparse_threshold must be within [0, 1]')

    def _population_workspace(self, n, batch, width):
        # Includes FP64 readout adjacency/fields, temporary rotations, masks,
        # scalar/vector schedules and all polynomial-filter scratch arrays.
        return (width + 8)*(64*batch*n + 16*n + 128*batch)

    def _prepare(self, run, z):
        return (z, torch.zeros_like(z)), {'dt': self.time_step}

    def _publish(self, run, target, z, step, record):
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError('nonfinite open-system proposal')
        incumbent = run.best_x if run.spin else 2*run.best_x-1
        signs = torch.where(z == 0, incumbent, torch.where(z > 0, 1., -1.))
        run.publish(target.readout(signs, self.polish_sweeps), step)
        record(step)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target = _Target(run.objective)
        if target.constant:
            return 0
        z = (run.initialbits.to(run.x.dtype)*2-1)*.05
        state, cache = self._prepare(run, z)
        for step in range(1, self.max_steps + 1):
            observation.poll()
            u = min(1., max(0., observation.schedule_fraction(step/self.max_steps))
                    / (1-self.terminal_fraction))
            anneal = u*u*(3-2*u)
            temperature = self.temperature*(1-anneal)**2
            z = self._position(state)
            force = -target.gradient(z) + (2*self.pinning*anneal)*z
            state = self._advance(state, force, temperature, anneal, step, run, cache)
            if (step % self.candidate_interval == 0 or step == self.max_steps
                    or self._extra_checkpoint(step, anneal)
                    or (run.history_interval and step % run.history_interval == 0)):
                # No overflow-generated candidate can enter the FP64 archive.
                if any(not bool(torch.isfinite(value).all()) for value in state):
                    raise FloatingPointError('nonfinite open-system state; reduce time_step')
                self._publish(run, target, self._position(state), step, record)
        return self.max_steps

    def _extra_checkpoint(self, step, anneal):
        return False

    @staticmethod
    def _position(state):
        return state[0]


@dataclass(frozen=True)
class DopplerCooling(_OpenContinuous):
    """One-force kick, exact OU thermostat, drift and inelastic walls."""
    damping: float = 1.

    def __post_init__(self):
        super().__post_init__()
        positive('damping', self.damping, zero=True)

    def _prepare(self, run, z):
        state, cache = super()._prepare(run, z)
        cache.update(decay=math.exp(-self.damping*self.time_step),
                     noise=math.sqrt(-math.expm1(-2*self.damping*self.time_step)))
        return state, cache

    def _advance(self, state, force, temperature, anneal, step, run, cache):
        z, p = state
        p = _thermostat(p + cache['dt']*force, cache['decay'], cache['noise'], temperature, run)
        return _wall(z + cache['dt']*p, p)


@dataclass(frozen=True)
class CavityAtomCooling(DopplerCooling):
    """Passive momentum-memory rotation with independent matched baths."""
    coupling: float = 1.
    memory_damping: float = .5

    def __post_init__(self):
        super().__post_init__()
        positive('coupling', self.coupling, zero=True)
        positive('memory_damping', self.memory_damping, zero=True)

    def _prepare(self, run, z):
        (z, p), cache = super()._prepare(run, z)
        cache.update(memory_decay=math.exp(-self.memory_damping*self.time_step),
                     memory_noise=math.sqrt(-math.expm1(-2*self.memory_damping*self.time_step)))
        return (z, p, torch.zeros_like(z)), cache

    def _advance(self, state, force, temperature, anneal, step, run, cache):
        z, p, r = state
        p, r = _rotation(p + cache['dt']*force, r, self.coupling*cache['dt'])
        p = _thermostat(p, cache['decay'], cache['noise'], temperature, run)
        # With g=0 the disconnected bath is exactly removable; avoiding its RNG
        # draws makes this limit identical to Doppler under the same seed.
        r = _thermostat(r, cache['memory_decay'], cache['memory_noise'],
                        temperature if self.coupling else 0., run)
        z, p = _wall(z + cache['dt']*p, p)
        return z, p, r


@dataclass(frozen=True)
class CavityOptomechanics(_OpenContinuous):
    """Compensated oscillator: U(z)+(p^2+w^2*(q-c*z)^2)/2."""
    coupling: float = .5
    frequency: float = 1.
    damping: float = 1.
    mobility: float = 1.

    def __post_init__(self):
        super().__post_init__()
        for name in ('frequency', 'mobility'):
            positive(name, getattr(self, name))
        for name in ('coupling', 'damping'):
            positive(name, getattr(self, name), zero=True)

    def _prepare(self, run, z):
        bound = float(_row_norms(run.objective).max()) + 2*self.pinning
        bound += (self.frequency*self.coupling)**2
        dt = min(self.time_step, 1/(self.mobility*max(bound, 1e-12)))
        return (z, self.coupling*z, torch.zeros_like(z)), {
            'dt': dt, 'decay': math.exp(-self.damping*dt),
            'noise': math.sqrt(-math.expm1(-2*self.damping*dt))}

    def _advance(self, state, force, temperature, anneal, step, run, cache):
        z, q, p = state
        dt = cache['dt']
        gradient = force + self.frequency**2*self.coupling*(q-self.coupling*z)
        z = _clip(z + self.mobility*dt*gradient)
        r, p = _oscillator(q-self.coupling*z, p, self.frequency, dt)
        q = r + self.coupling*z
        p = _thermostat(p, cache['decay'], cache['noise'], temperature, run)
        return z, q, p


@dataclass(frozen=True)
class OptomechanicalCooling(DopplerCooling):
    """Row-norm frequency proxy and cached red-sideband damping/noise rates."""
    linewidth: float = 1.
    optical_coupling: float = .5
    heating_weight: float = 1.
    stiffness_epsilon: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        for name in ('linewidth', 'stiffness_epsilon'):
            positive(name, getattr(self, name))
        for name in ('optical_coupling', 'heating_weight'):
            positive(name, getattr(self, name), zero=True)

    def _prepare(self, run, z):
        state, cache = super()._prepare(run, z)
        if not self.optical_coupling:
            return state, cache  # Exact Doppler ablation, including arithmetic/RNG.
        frequency = (_row_norms(run.objective) + 2*self.pinning + self.stiffness_epsilon).sqrt()
        minus, plus = _sideband_rates(frequency, self.linewidth, self.optical_coupling)
        rate = self.damping + (minus-plus).clamp_min(0)
        temperature_factor = 1 + self.heating_weight*plus/(rate+self.stiffness_epsilon)
        cache.update(decay=torch.exp(-rate*self.time_step),
                     noise=(temperature_factor*(-torch.expm1(-2*rate*self.time_step))).sqrt())
        return state, cache


@dataclass(frozen=True)
class SteadyStateSqueezing(_OpenContinuous):
    """Frozen positive diagonal mobility with matched Langevin noise.

    Row-bound inverse mobilities are median-normalized then clipped. This
    keeps the final declared mobility limits and gives median one, since the
    limits must bracket one. It is not an entangled or unique QUBO steady state.
    """
    mobility_min: float = .25
    mobility_max: float = 4.
    mobility_epsilon: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        for name in ('mobility_min', 'mobility_max', 'mobility_epsilon'):
            positive(name, getattr(self, name))
        if not self.mobility_min <= 1 <= self.mobility_max:
            raise ValueError('mobility limits must bracket one')

    def _prepare(self, run, z):
        rows = _row_norms(run.objective)
        mobility = (rows + run.objective.field.abs() + self.mobility_epsilon).reciprocal()
        mobility = (mobility/mobility.median()).clamp(self.mobility_min, self.mobility_max)
        bound = float(rows.max()) + 2*self.pinning
        dt = min(self.time_step, 1/(self.mobility_max*max(bound, 1e-12)))
        return (z,), {'mobility': mobility, 'dt': dt, 'noise': (2*dt*mobility).sqrt()}

    def _advance(self, state, force, temperature, anneal, step, run, cache):
        z = state[0] + cache['dt']*cache['mobility']*force
        if temperature > 0:
            z = z + math.sqrt(temperature)*cache['noise']*run.normal(z.shape)
        return (_clip(z),)


@dataclass(frozen=True)
class TomonagaLuttinger(_OpenContinuous):
    """Positive periodic-chain polynomial mobility on q, with z=tanh(q).

    The fixed logical input ordering defines the ring. Noise is explicitly a
    heuristic perturbation in q, not an equilibrium sampler. The finite q cap
    is also a heuristic; unconstrained zero-noise flow is the descent identity.
    """
    filter_length: float = .5
    filter_terms: int = 2
    coordinate_cap: float = 4.
    temperature: float = .01

    def __post_init__(self):
        super().__post_init__()
        positive('filter_length', self.filter_length, zero=True)
        positive('coordinate_cap', self.coordinate_cap)
        if self.coordinate_cap > 8:
            raise ValueError('coordinate_cap must not exceed 8 (FP32 tanh saturation)')
        _integer('filter_terms', self.filter_terms, 1)

    def _prepare(self, run, z):
        rows = _row_norms(run.objective)
        fmax = float((rows+run.objective.field.abs()).max()) + 2*self.pinning
        bound = float(rows.max()) + 2*self.pinning + 2*fmax
        return (z.atanh(),), {'dt': min(self.time_step, 1/max(bound, 1e-12))}

    @staticmethod
    def _position(state):
        return state[0].tanh()

    def _advance(self, state, force, temperature, anneal, step, run, cache):
        q = state[0]
        gradient = (1-q.tanh().square())*force
        proposal = q + cache['dt']*_positive_filter(gradient, self.filter_length, self.filter_terms)
        if temperature > 0:
            proposal = proposal + math.sqrt(2*temperature*cache['dt'])*run.normal(q.shape)
        return (_clip(proposal, self.coordinate_cap),)


@dataclass(frozen=True)
class QuadratureSqueezing(DopplerCooling):
    """Bounded post-drift canonical stretches with a zero-pulse terminal phase.

    At the cumulative log-amplification limit, reset momentum and its pulse
    counter before the next pulse. Positions/incumbents are retained. Positive
    scaling cannot change signs, so a single readout also covers the immediately
    pre-pulse sign candidate, including the incumbent convention at zero.
    """
    temperature: float = 0.
    squeeze: float = .15
    pulse_interval: int = 8
    amplification_limit: float = 4.
    alternating_pulses: bool = True

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.squeeze, bool) or not math.isfinite(self.squeeze) or abs(self.squeeze) > 2:
            raise ValueError('squeeze must be finite and within [-2, 2]')
        _integer('pulse_interval', self.pulse_interval, 1)
        positive('amplification_limit', self.amplification_limit)
        if self.amplification_limit < abs(self.squeeze) or self.amplification_limit > 20:
            raise ValueError('amplification_limit must cover abs(squeeze) and not exceed 20')
        if not isinstance(self.alternating_pulses, bool):
            raise ValueError('alternating_pulses must be boolean')

    def _prepare(self, run, z):
        state, cache = super()._prepare(run, z)
        cache['amplification'] = 0.
        return state, cache

    def _extra_checkpoint(self, step, anneal):
        return step % self.pulse_interval == 0 and anneal < 1 and bool(self.squeeze)

    def _advance(self, state, force, temperature, anneal, step, run, cache):
        z, p = state
        p = _thermostat(p + cache['dt']*force, cache['decay'], cache['noise'], temperature, run)
        z = z + cache['dt']*p
        if step % self.pulse_interval == 0 and anneal < 1 and self.squeeze:
            pulse = self.squeeze*(1-anneal)
            if self.alternating_pulses and (step//self.pulse_interval) % 2 == 0:
                pulse = -pulse
            if cache['amplification'] + abs(pulse) > self.amplification_limit:
                p = torch.zeros_like(p)
                cache['amplification'] = 0.
            cache['amplification'] += abs(pulse)
            stretch = math.exp(pulse)
            z, p = z*stretch, p/stretch
        return _wall(z, p)


OPEN_CONTINUOUS_CLASSES = {
    'lib_doppler_cooling': DopplerCooling,
    'lib_cavity_atom_cooling': CavityAtomCooling,
    'lib_cavity_optomechanics': CavityOptomechanics,
    'lib_optomechanical_cooling': OptomechanicalCooling,
    'lib_steady_state_squeezing': SteadyStateSqueezing,
    'lib_tomonaga_luttinger': TomonagaLuttinger,
    'lib_quadrature_squeezing': QuadratureSqueezing,
}
