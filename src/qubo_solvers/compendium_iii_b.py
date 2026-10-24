"""Compendium III, entries 30--58: classical QUBO reductions and drivers.

Original FP64 coefficients score all incumbents. Auxiliary rings, ladders and
blocks select proposals only; they never truncate the input objective. Named
quantum mechanisms here are explicitly classical reductions, not Hilbert-space
simulations. Structured decoders require metadata and never impose a code on an
unrelated QUBO. RNG streams are PyTorch device-generator streams; batch layout
is consequently part of reproducibility, rather than a counter-keyed promise.
"""

from dataclasses import dataclass
import math

import torch

from . import observation
from ._population import PopulationSolver, _dtype, _mm
from .hamiltonian_continuous import _Target as _ContinuousTarget
from .hamiltonian_discrete import _Target, _random, _metropolis
from .open_system_discrete import _submatrix, _block_gain, _commit, _neighborhoods
from .solvers import _integer
from .compendium_iii_c import _StructuredDecoder


def _positive(name, value, zero=False):
    if isinstance(value, bool) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def _anneal(step, steps, terminal):
    t = min(1., max(0., observation.schedule_fraction(step / max(steps, 1))) / (1-terminal))
    return t*t*(3-2*t)


def _signs(run):
    return run.x.double().clone() if run.spin else run.x.double()*2-1


def _ring_neighbours(i, n):
    return [] if n == 1 else [(i+1) % n] if n == 2 else [(i-1) % n, (i+1) % n]


def _ring_sum(x):
    if x.shape[1] == 1:
        return torch.zeros_like(x)
    return x.flip((1,)) if x.shape[1] == 2 else x.roll(1, 1)+x.roll(-1, 1)


def _conditional(old, field, matrix, temperature, run):
    """Exact bounded categorical heat bath; old wins deterministic ties."""
    k = old.shape[1]
    labels = torch.arange(1 << k, device=old.device)
    candidates = ((labels[:, None] >> torch.arange(k, device=old.device)) & 1).double()*2-1
    boundary = field-old@matrix
    energy = boundary@candidates.T + .5*((candidates@matrix)*candidates).sum(-1)
    if temperature == 0:
        values, indices = energy.min(-1)
        current = (boundary*old+.5*(old@matrix)*old).sum(-1)
        return torch.where((values < current)[:, None], candidates[indices], old)
    weights = torch.softmax(-(energy-energy.min(-1, keepdim=True).values)/temperature, -1)
    index = torch.multinomial(weights, 1, generator=run.generator).squeeze(-1)
    return candidates[index]


def _block(target, state, field, indices, temperature, run):
    matrix = _submatrix(target, indices)
    proposed = _conditional(state[:, indices], field[:, indices], matrix, temperature, run)
    _commit(target, state, field, indices, proposed)


def _flip(target, state, field, indices, temperature, run):
    old = state[:, indices]
    proposed = -old
    gain = _block_gain(old, proposed, field[:, indices], _submatrix(target, indices))
    accepted = _metropolis(gain, temperature, run, strict=temperature == 0)
    _commit(target, state, field, indices, proposed, accepted)


def _single(target, state, field, i, temperature, run, activity=None):
    gain = -2*state[:, i]*field[:, i]
    accepted = _metropolis(gain, temperature, run, strict=temperature == 0)
    if activity is not None:
        accepted &= _random(gain.shape, run) < activity
    change = -2*state[:, i]*accepted
    state[:, i].add_(change)
    target.update(field, i, change)


def _box(z, p):
    invalid = ~torch.isfinite(z) | ~torch.isfinite(p)
    return (z.clamp(-1, 1).masked_fill(invalid, float('nan')),
            p.masked_fill(z.abs() > 1, 0.).masked_fill(invalid, float('nan')))


def _finite_clip(value):
    """Clipping is a boundary approximation; it must never mask overflow."""
    return value.clamp(-1, 1).masked_fill(~torch.isfinite(value), float('nan'))


def _unit(v):
    return v/v.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)


@dataclass(frozen=True)
class _BBase(PopulationSolver):
    max_steps: int = 512
    candidate_interval: int = 8
    polish_sweeps: int = 1
    temperature: float = .3
    terminal_fraction: float = .15
    dynamics_dtype: object = None
    normalize: bool = True
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('candidate_interval', self.candidate_interval, 1)
        _integer('polish_sweeps', self.polish_sweeps, 0)
        _positive('temperature', self.temperature, True)
        _dtype(self.dynamics_dtype, torch.float32)
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if not math.isfinite(self.terminal_fraction) or not 0 <= self.terminal_fraction < 1:
            raise ValueError('terminal_fraction must be in [0, 1)')
        if not math.isfinite(self.sparse_threshold) or not 0 <= self.sparse_threshold <= 1:
            raise ValueError('sparse_threshold must be in [0, 1]')

    def _population_workspace(self, n, batch, width):
        # FP64 target/readout, block candidates, all state and temporary arrays.
        # Source-operator preparation is separately charged by PopulationSolver.
        k = min(n, getattr(self, 'block_size', 8))
        replicas = max(2, getattr(self, 'levels', 1))
        modes = getattr(self, 'bath_modes', 1)
        rank = getattr(self, 'rank', 1)
        return (256*(replicas+modes+rank)*batch*n + 64*(batch+k)*(1 << k)
                + 64*k*k + 128*n*max(k, rank) + 256*batch)

    def _due(self, step, run):
        return (step % self.candidate_interval == 0 or step == self.max_steps or
                run.history_interval and step % run.history_interval == 0)

    def _publish(self, run, record, state, step):
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError('nonfinite Compendium III state')
        run.publish(state, step)
        # The evolving search state is owned by the caller. Checkpoint the
        # nonworsening verified archive, including already scored intermediate
        # group-label proposals, rather than a later worse exploration state.
        run.x.copy_(run.best_x)
        run.energies.copy_(run.best)
        record(step)

    def _finish(self, run, record, target, state, step):
        self._publish(run, record, state, step)
        if self.polish_sweeps:
            state = run.best_x.double().clone() if run.spin else run.best_x.double()*2-1
            field = target.field(state)
            for _ in range(self.polish_sweeps):
                for i in range(target.n):
                    observation.poll()
                    proposed = torch.where(field[:, i] < 0, 1., -1.)
                    delta = proposed-state[:, i]
                    state[:, i] = proposed
                    target.update(field, i, delta)
            self._publish(run, record, state, step)
        return step


@dataclass(frozen=True)
class _BDiscrete(_BBase):
    refresh_interval: int = 32

    def __post_init__(self):
        super().__post_init__()
        _integer('refresh_interval', self.refresh_interval, 1)

    def _prepare(self, target, state, run):
        return {}

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective, occupation=False, normalize=self.normalize)
        if target.constant:
            return 0
        state = _signs(run)
        field = target.field(state)
        cache = self._prepare(target, state, run)
        for step in range(1, self.max_steps+1):
            observation.poll()
            r = _anneal(step, self.max_steps, self.terminal_fraction)
            self._advance(target, state, field, cache, self.temperature*(1-r)**2, r, step, run)
            if step % self.refresh_interval == 0:
                field = target.field(state)
            if self._due(step, run):
                self._publish(run, record, state, step)
        return self._finish(run, record, target, state, self.max_steps)


@dataclass(frozen=True)
class C3KugelKhomskii(_BDiscrete):
    """30: four-state logical/orbital heat bath on a bounded auxiliary ring."""
    orbital_coupling: float = .5
    driver: float = .5

    def __post_init__(self):
        super().__post_init__()
        _positive('orbital_coupling', self.orbital_coupling, True)
        _positive('driver', self.driver, True)

    def _prepare(self, target, state, run):
        return {'orbital': torch.where(_random(state.shape, run) < .5, -1., 1.)}

    def _advance(self, target, state, field, cache, temperature, r, step, run):
        i = (step-1) % target.n
        orbital = cache['orbital']
        neighbours = _ring_neighbours(i, target.n)
        auxiliary = orbital[:, neighbours].sum(-1)/max(len(neighbours), 1)
        labels = state.new_tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        lam = self.driver*(1-r)
        energy = (field[:, i, None]*labels[:, 0] - lam*auxiliary[:, None]*labels[:, 1]
                  - lam*self.orbital_coupling*labels[:, 0]*labels[:, 1])
        if temperature == 0:
            chosen = energy.argmin(-1)
        else:
            chosen = torch.multinomial(torch.softmax(-(energy-energy.min(-1, keepdim=True).values)/temperature, -1),
                                       1, generator=run.generator).squeeze(-1)
        delta = labels[chosen, 0]-state[:, i]
        state[:, i], orbital[:, i] = labels[chosen, 0], labels[chosen, 1]
        target.update(field, i, delta)


@dataclass(frozen=True)
class C3KondoLattice(_BDiscrete):
    """31: overlapping, cyclically covering exact strong-neighbour patches."""
    block_size: int = 6

    def __post_init__(self):
        super().__post_init__()
        _integer('block_size', self.block_size, 1)
        if self.block_size > 10:
            raise ValueError('block_size must be <= 10 (bounded categorical enumeration)')

    def _prepare(self, target, state, run):
        return {'patches': _neighborhoods(target, self.block_size)}

    def _advance(self, target, state, field, cache, temperature, r, step, run):
        patch = cache['patches'][(step-1) % target.n]
        _block(target, state, field, patch, temperature, run)


@dataclass(frozen=True)
class _BContinuous(_BBase):
    time_step: float = .08
    pinning: float = .5
    damping: float = .5

    def __post_init__(self):
        super().__post_init__()
        _positive('time_step', self.time_step)
        _positive('pinning', self.pinning, True)
        _positive('damping', self.damping, True)

    def _prepare(self, z, run):
        return (z,), {}

    def _position(self, state):
        return state[0]

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target = _ContinuousTarget(run.objective)
        if target.constant:
            return 0
        z = (2*run.initialbits.to(run.x.dtype)-1)*.1
        state, cache = self._prepare(z, run)
        for step in range(1, self.max_steps+1):
            observation.poll()
            r = _anneal(step, self.max_steps, self.terminal_fraction)
            z = self._position(state)
            force = -target.gradient(z) + 2*self.pinning*r*z
            state = self._advance(state, force, self.temperature*(1-r)**2, r, step, run, cache)
            if self._due(step, run):
                if any(not bool(torch.isfinite(v).all()) for v in state):
                    raise FloatingPointError('nonfinite Compendium III dynamics; reduce time_step')
                # Cleanup is explicitly charged at candidate checkpoints.
                signs = target.readout(self._position(state), self.polish_sweeps)
                self._publish(run, record, signs, step)
        return self.max_steps


@dataclass(frozen=True)
class C3HubbardHolstein(_BContinuous):
    """32: positive-square local oscillator, exact frozen-equilibrium rotation."""
    frequency: float = 1.
    coupling: float = .7

    def __post_init__(self):
        super().__post_init__()
        _positive('frequency', self.frequency)
        _positive('coupling', self.coupling, True)

    def _prepare(self, z, run):
        return (z, self.coupling*z, torch.zeros_like(z)), {}

    def _advance(self, state, force, temperature, r, step, run, cache):
        z, q, p = state
        dt = min(self.time_step, .4/(1+2*self.pinning+self.frequency**2*self.coupling**2))
        z = _finite_clip(z+dt*(force+self.frequency**2*self.coupling*(q-self.coupling*z)))
        residual = q-self.coupling*z
        c, s = math.cos(self.frequency*dt), math.sin(self.frequency*dt)
        q, p = self.coupling*z+c*residual+s*p/self.frequency, c*p-self.frequency*s*residual
        decay = math.exp(-self.damping*dt)
        p = decay*p + math.sqrt(temperature*(1-decay**2))*run.normal(p.shape)
        return z, q, p


@dataclass(frozen=True)
class C3PairHoppingBoseHubbard(_BContinuous):
    """33: damped angular momentum with doubled-phase sparse-ring coupling."""
    phase_coupling: float = .5

    def __post_init__(self):
        super().__post_init__()
        _positive('phase_coupling', self.phase_coupling, True)

    def _prepare(self, z, run):
        return (z.acos(), torch.zeros_like(z)), {}

    def _position(self, state):
        return state[0].cos()

    def _advance(self, state, force, temperature, r, step, run, cache):
        theta, p = state
        sin2, cos2 = (2*theta).sin(), (2*theta).cos()
        coupling = sin2*_ring_sum(cos2)-cos2*_ring_sum(sin2)
        dt = min(self.time_step, .25/(1+self.phase_coupling+2*self.pinning))
        p = p*math.exp(-self.damping*dt)-dt*(force*theta.sin()+2*self.phase_coupling*(1-r)*coupling)
        p += math.sqrt(2*self.damping*temperature*dt)*run.normal(p.shape)
        theta = torch.remainder(theta+dt*p+math.pi, 2*math.pi)-math.pi
        return theta, p


@dataclass(frozen=True)
class C3QuantumEast(_BDiscrete):
    """34: predecessor-facilitated sequential Metropolis; singleton is ungated."""
    facilitation_floor: float = .1

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.facilitation_floor <= 1:
            raise ValueError('facilitation_floor must be in (0, 1]')

    def _advance(self, target, state, field, cache, temperature, r, step, run):
        i = (step-1) % target.n
        predecessor = (i-1) % target.n
        activity = (None if target.n == 1 else self.facilitation_floor+
                    (1-self.facilitation_floor)*(state[:, predecessor]+1)*.5)
        _single(target, state, field, i, temperature, run, activity)


@dataclass(frozen=True)
class C3QuantumPSpin(_BContinuous):
    """35: low-rank nonlinear collective driver, removed at terminal time."""
    power: int = 4
    driver: float = .5

    def __post_init__(self):
        super().__post_init__()
        _integer('power', self.power, 2)
        if self.power > 16:
            raise ValueError('power must be <= 16')
        _positive('driver', self.driver, True)

    def _prepare(self, z, run):
        return (z, torch.zeros_like(z)), {'pattern': torch.where(run.normal(z.shape) >= 0, 1., -1.)}

    def _advance(self, state, force, temperature, r, step, run, cache):
        z, p = state
        a = cache['pattern']
        m = (a*z).mean(-1, keepdim=True)
        dt = min(self.time_step, .3/(1+2*self.pinning+self.driver*self.power*(self.power-1)))
        p = math.exp(-self.damping*dt)*p + dt*(force+self.driver*(1-r)*self.power*a*m.pow(self.power-1))
        p += math.sqrt(2*self.damping*temperature*dt)*run.normal(p.shape)
        return _box(z+dt*p, p)


def _alignment(v, force, amount):
    # Half-angle amplitudes preserve the tiny transverse component when FP32
    # has already rounded z to +/-1. atan(h) or sqrt(1-z*z) would lose it.
    z = v[..., 2]
    transverse = torch.hypot(v[..., 0], v[..., 1])
    length = torch.hypot(transverse, z)
    major = ((length+z.abs())/(2*length)).sqrt()
    minor = transverse/(2*length*(length+z.abs())).sqrt()
    up = torch.where(z >= 0, major, minor)
    down = torch.where(z >= 0, minor, major)
    # tan(theta/2) evolves by exp(-amount*force). Log scaling handles strong
    # finite fields without overflow and retains the exact stationary poles.
    log_up, log_down = up.log()+.5*amount*force, down.log()-.5*amount*force
    scale = torch.maximum(log_up, log_down)
    up, down = (log_up-scale).exp(), (log_down-scale).exp()
    norm = torch.hypot(up, down)
    up, down = up/norm, down/norm
    xy = v[..., :2]/transverse[..., None].clamp_min(torch.finfo(v.dtype).tiny)
    return torch.cat((xy*(2*up*down)[..., None], (up.square()-down.square())[..., None]), -1)


@dataclass(frozen=True)
class C3KickedIsing(_BContinuous):
    """36: exact frozen-field alignment, longitudinal phase and transverse kick."""
    longitudinal_rotation: float = .4
    transverse_kick: float = .3

    def __post_init__(self):
        super().__post_init__()
        _positive('longitudinal_rotation', self.longitudinal_rotation, True)
        _positive('transverse_kick', self.transverse_kick, True)

    def _prepare(self, z, run):
        phi = 2*math.pi*run.normal(z.shape)
        v = torch.stack(((1-z*z).sqrt()*phi.cos(), (1-z*z).sqrt()*phi.sin(), z), -1)
        return (v,), {}

    def _position(self, state):
        return state[0][..., 2]

    def _advance(self, state, force, temperature, r, step, run, cache):
        v = _alignment(state[0], force, self.damping*self.time_step)
        phi = self.longitudinal_rotation*(1-r)*(force-2*self.pinning*r*state[0][..., 2])
        x, y, z = v.unbind(-1)
        x, y = phi.cos()*x-phi.sin()*y, phi.sin()*x+phi.cos()*y
        angle = self.transverse_kick*(1-r)
        y, z = math.cos(angle)*y-math.sin(angle)*z, math.sin(angle)*y+math.cos(angle)*z
        return (torch.stack((x, y, z), -1),)


def _rectangle(n, anchor, width=None):
    width = width or max(1, math.isqrt(n))
    return list(dict.fromkeys([anchor % n, (anchor+1) % n, (anchor+width) % n,
                               (anchor+width+1) % n]))


@dataclass(frozen=True)
class C3WenPlaquette(_BDiscrete):
    """37: generic-QUBO four-site plaquette masks, not syndrome decoding."""
    def _advance(self, target, state, field, cache, temperature, r, step, run):
        i = (step-1) % target.n
        indices = [i] if step % 4 == 0 else _rectangle(target.n, i)
        _flip(target, state, field, indices, temperature, run)


@dataclass(frozen=True)
class C3Schwinger(_BDiscrete):
    """38: interval transfers plus single flips; no unrequested charge sector."""
    def _advance(self, target, state, field, cache, temperature, r, step, run):
        i = (step-1) % target.n
        if step % 3 == 0 or target.n == 1:
            _single(target, state, field, i, temperature, run)
            return
        # Deterministic short/long interval schedule, independent of state.
        distance = 1 << min((step & -step).bit_length()-1, max(0, target.n.bit_length()-2))
        j = (i+distance) % target.n
        if j == i:
            return
        old = state[:, [i, j]]
        proposed = old.flip((1,))
        gain = _block_gain(old, proposed, field[:, [i, j]], _submatrix(target, [i, j]))
        _commit(target, state, field, [i, j], proposed, _metropolis(gain, temperature, run))


@dataclass(frozen=True)
class C3PlaquetteIsing(_BDiscrete):
    """40: exact rectangular conditional minimization, varied lattice offsets."""
    def _advance(self, target, state, field, cache, temperature, r, step, run):
        i = (step-1) % target.n
        width = max(1, math.isqrt(target.n)) + ((step//target.n) % 2)
        _block(target, state, field, _rectangle(target.n, i, width), temperature, run)


@dataclass(frozen=True)
class C3SpinTransferTorque(C3KickedIsing):
    """41: nonconservative transverse torque bursts and terminal damping."""
    torque: float = .8
    field_like_torque: float = .3
    stall_steps: int = 24
    burst_steps: int = 8

    def __post_init__(self):
        super().__post_init__()
        _positive('torque', self.torque, True)
        _positive('field_like_torque', self.field_like_torque, True)
        _integer('stall_steps', self.stall_steps, 1)
        _integer('burst_steps', self.burst_steps, 1)

    def _prepare(self, z, run):
        state, cache = super()._prepare(z, run)
        cache.update(best=run.best.clone(), age=torch.zeros(z.shape[0], device=z.device, dtype=torch.long))
        return state, cache

    def _advance(self, state, force, temperature, r, step, run, cache):
        v = state[0]
        improved = run.best < cache['best']
        cache['age'] = torch.where(improved, 0, cache['age']+1)
        cache['best'] = torch.minimum(cache['best'], run.best)
        active = ((cache['age'] % (self.stall_steps+self.burst_steps)) >= self.stall_steps).to(v.dtype)
        polar = torch.zeros_like(v)
        polar[..., 0], polar[..., 1] = math.cos(.13*step), math.sin(.13*step)
        h = torch.zeros_like(v)
        h[..., 2] = force
        cross = torch.linalg.cross(v, h)
        derivative = -cross-self.damping*torch.linalg.cross(v, cross)
        amp = (1-r)*active[:, None, None]
        derivative += amp*(self.torque*torch.linalg.cross(v, torch.linalg.cross(v, polar))
                           +self.field_like_torque*torch.linalg.cross(v, polar))
        dt = min(self.time_step, .1/(1+self.torque+self.field_like_torque+2*self.pinning))
        return (_unit(v+dt*derivative),)


@dataclass(frozen=True)
class C3LevinWen(_BDiscrete):
    """44: closed-loop proposal masks mixed with single-bit moves."""
    loop_size: int = 6

    def __post_init__(self):
        super().__post_init__()
        _integer('loop_size', self.loop_size, 3)
        if self.loop_size > 16:
            raise ValueError('loop_size must be <= 16')

    def _advance(self, target, state, field, cache, temperature, r, step, run):
        i = (step-1) % target.n
        indices = [i] if step % 4 == 0 else [(i+j) % target.n for j in range(min(self.loop_size, target.n))]
        _flip(target, state, field, indices, temperature, run)


@dataclass(frozen=True)
class C3DoubleSemion(C3LevinWen):
    """45: unbiased baseline alternating closed-loop and open-path masks."""
    def _advance(self, target, state, field, cache, temperature, r, step, run):
        i = (step-1) % target.n
        if step % 3 == 0:
            size = 1
        elif step % 3 == 1:
            size = min(self.loop_size, target.n)
        else:
            size = min(max(2, self.loop_size//2), target.n)
        _flip(target, state, field, [(i+j) % target.n for j in range(size)], temperature, run)


@dataclass(frozen=True)
class C3QuantumLink(_BDiscrete):
    """46: coupled matter/link-sized three-bit blocks on unconstrained QUBOs."""
    def _advance(self, target, state, field, cache, temperature, r, step, run):
        i = (step-1) % target.n
        _block(target, state, field, [(i+j) % target.n for j in range(min(3, target.n))], temperature, run)


@dataclass(frozen=True)
class C3MajumdarGhosh(_BDiscrete):
    """47: alternating dimer matchings with occasional merged four-site blocks."""
    def _advance(self, target, state, field, cache, temperature, r, step, run):
        pairs = max(1, target.n//2)
        epoch, position = divmod(step-1, pairs)
        i = (2*position+epoch % 2) % target.n
        size = min(4 if epoch % 4 == 3 else 2, target.n)
        _block(target, state, field, [(i+j) % target.n for j in range(size)], temperature, run)


def _ladder_skew(x):
    """Oriented open two-leg ladder, normalized max absolute row sum <= 1."""
    out = torch.zeros_like(x)
    for offset, sign in ((0, 1.), (1, -1.)):
        leg = x[:, offset::2]
        out[:, offset:-2:2] += sign*.25*leg[:, 1:]
        out[:, offset+2::2] -= sign*.25*leg[:, :-1]
    count = x.shape[1]//2
    out[:, :2*count:2] += .5*x[:, 1:2*count:2]
    out[:, 1:2*count:2] -= .5*x[:, :2*count:2]
    return out


@dataclass(frozen=True)
class C3CreutzLadder(_BContinuous):
    """48: paired opposite skew circulations in tanh coordinates."""
    circulation: float = .8

    def __post_init__(self):
        super().__post_init__()
        _positive('circulation', self.circulation, True)

    def _prepare(self, z, run):
        orientation = (torch.arange(z.shape[0], device=z.device) % 2)*2-1
        return (z.atanh(),), {'orientation': orientation[:, None]}

    def _position(self, state):
        return state[0].tanh()

    def _advance(self, state, force, temperature, r, step, run, cache):
        q = state[0]
        g = (1-q.tanh().square())*force
        dt = min(self.time_step, .3/((3+6*self.pinning)*(1+self.circulation**2)))
        dq = g+self.circulation*(1-r)*cache['orientation']*_ladder_skew(g)
        q = q+dt*dq+math.sqrt(2*temperature*dt)*run.normal(q.shape)
        return (q,)


@dataclass(frozen=True)
class C3PensonKolb(_BDiscrete):
    """49: strong-edge four-mask pair heat baths, including singles and no-op."""
    def _prepare(self, target, state, run):
        return {'pairs': _neighborhoods(target, 2)}

    def _advance(self, target, state, field, cache, temperature, r, step, run):
        _block(target, state, field, cache['pairs'][(step-1) % target.n], temperature, run)


@dataclass(frozen=True)
class C3RichardsonGaudin(_BContinuous):
    """50: exact positive-square low-rank auxiliary response, no spectral claim."""
    rank: int = 4
    stiffness: float = 1.
    response_rate: float = .5

    def __post_init__(self):
        super().__post_init__()
        _integer('rank', self.rank, 1)
        if self.rank > 32:
            raise ValueError('rank must be <= 32')
        _positive('stiffness', self.stiffness)
        _positive('response_rate', self.response_rate)

    def _prepare(self, z, run):
        rank = min(self.rank, z.shape[1])
        pattern = run.normal((z.shape[1], rank))
        pattern /= pattern.norm(dim=0, keepdim=True).clamp_min(1e-12)
        return (z, z@pattern), {'pattern': pattern}

    def _advance(self, state, force, temperature, r, step, run, cache):
        z, y = state
        a = cache['pattern']
        residual = y-z@a
        dt = min(self.time_step, .25/(1+2*self.pinning+self.stiffness*self.rank*(1+self.response_rate)))
        z = _finite_clip(z+dt*(force+self.stiffness*residual@a.T))
        equilibrium = z@a
        y = equilibrium+math.exp(-dt*self.response_rate*self.stiffness)*(y-equilibrium)
        return z, y


@dataclass(frozen=True)
class C3SpinorRydberg(_BContinuous):
    """51: normalized three-amplitude dissipative flow with annealed mixing."""
    mixing: float = .4

    def __post_init__(self):
        super().__post_init__()
        _positive('mixing', self.mixing, True)

    def _prepare(self, z, run):
        psi = torch.stack(((1-z)*.4, torch.full_like(z, .2), (1+z)*.4), -1).sqrt()
        return (psi,), {}

    def _position(self, state):
        psi = state[0]
        return psi[..., 2].square()-psi[..., 0].square()

    def _advance(self, state, force, temperature, r, step, run, cache):
        psi = state[0]
        m = self._position(state)
        ell = force-2*self.pinning*r*m
        costs = torch.stack((ell, torch.full_like(ell, self.pinning*r), -ell), -1)
        applied = costs*psi-self.mixing*(1-r)*(psi.sum(-1, keepdim=True)-psi)
        rayleigh = (psi*applied).sum(-1, keepdim=True)
        dt = min(self.time_step, .15/(1+self.pinning+2*self.mixing))
        return (_unit(psi-dt*(applied-rayleigh*psi)),)


@dataclass(frozen=True)
class C3QuantumHopfield(_BContinuous):
    """52: convex-combination mean-field response using all original bonds."""
    response: float = .25
    minimum_temperature: float = .01

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.response <= 1:
            raise ValueError('response must be in (0, 1]')
        _positive('minimum_temperature', self.minimum_temperature)

    def _advance(self, state, force, temperature, r, step, run, cache):
        z = state[0]
        ell = force-2*self.pinning*r*z
        target = torch.tanh(ell/max(temperature, self.minimum_temperature))
        return ((1-self.response)*z+self.response*target,)


@dataclass(frozen=True)
class C3QuantumPotts(C3KondoLattice):
    """53: bounded exact categorical blocks with shifted partitions."""
    block_size: int = 4

    def _prepare(self, target, state, run):
        return {}

    def _advance(self, target, state, field, cache, temperature, r, step, run):
        k = min(self.block_size, target.n)
        blocks = (target.n+k-1)//k
        epoch, slot = divmod(step-1, blocks)
        anchor = (slot*k+epoch) % target.n
        _block(target, state, field, [(anchor+j) % target.n for j in range(k)], temperature, run)


@dataclass(frozen=True)
class C3FrustratedSpinGlass(_BDiscrete):
    """54: each outer restart owns an independent finite temperature ladder."""
    levels: int = 4
    cold_temperature: float = .02
    hot_temperature: float = 1.

    def __post_init__(self):
        super().__post_init__()
        _integer('levels', self.levels, 2)
        if self.levels > 32:
            raise ValueError('levels must be <= 32')
        _positive('cold_temperature', self.cold_temperature)
        _positive('hot_temperature', self.hot_temperature)
        if self.hot_temperature < self.cold_temperature:
            raise ValueError('hot_temperature must be >= cold_temperature')

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target = _Target(run.objective, occupation=False, normalize=self.normalize)
        b, n = run.x.shape
        states = torch.where(_random((b, self.levels, n), run) < .5, -1., 1.).double()
        states[:, 0] = _signs(run)
        state = states.reshape(-1, n)
        field = target.field(state)
        ladder = torch.logspace(math.log10(self.cold_temperature), math.log10(self.hot_temperature),
                                self.levels, device=state.device, dtype=torch.float64)
        for step in range(1, self.max_steps+1):
            observation.poll()
            _single(target, state, field, (step-1) % n, ladder.repeat(b), run)
            if step % n == 0:
                energy = target.energy(state).reshape(b, self.levels)
                f = field.reshape(b, self.levels, n)
                for j in range((step//n) % 2, self.levels-1, 2):
                    log_accept = (1/ladder[j]-1/ladder[j+1])*(energy[:, j]-energy[:, j+1])
                    accept = _random((b,), run).log() < log_accept
                    old = states[:, j].clone()
                    states[:, j] = torch.where(accept[:, None], states[:, j+1], old)
                    states[:, j+1] = torch.where(accept[:, None], old, states[:, j+1])
                    old = f[:, j].clone()
                    f[:, j] = torch.where(accept[:, None], f[:, j+1], old)
                    f[:, j+1] = torch.where(accept[:, None], old, f[:, j+1])
            if step % self.refresh_interval == 0:
                field = target.field(state)
            if self._due(step, run):
                energies = run.objective.score(run.to_assignments(state)).reshape(b, self.levels)
                chosen = states[torch.arange(b, device=state.device), energies.argmin(-1)]
                self._publish(run, record, chosen, step)
        return self._finish(run, record, target, chosen, self.max_steps)


@dataclass(frozen=True)
class C3BlumeCapel(_BDiscrete):
    """55: ternary heat bath; zeros are filled by exact sequential readout."""
    pinning: float = .5

    def __post_init__(self):
        super().__post_init__()
        _positive('pinning', self.pinning)

    def _extra_occupancy(self, state, i, r):
        return torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target = _Target(run.objective, occupation=False, normalize=self.normalize)
        state = _signs(run)
        state.masked_fill_(_random(state.shape, run) < .35, 0.)
        field = target.field(state)
        labels = state.new_tensor([-1., 0., 1.])
        for step in range(1, self.max_steps+1):
            observation.poll()
            r = _anneal(step, self.max_steps, self.terminal_fraction)
            temperature = self.temperature*(1-r)**2
            i = (step-1) % target.n
            occupancy = self.pinning*r+self._extra_occupancy(state, i, r)
            energy = field[:, i, None]*labels-occupancy[:, None]*labels.square()
            if temperature == 0:
                choice = energy.argmin(-1)
            else:
                choice = torch.multinomial(torch.softmax(-(energy-energy.min(-1, keepdim=True).values)/temperature, -1),
                                           1, generator=run.generator).squeeze(-1)
            delta = labels[choice]-state[:, i]
            state[:, i] = labels[choice]
            target.update(field, i, delta)
            if step % self.refresh_interval == 0:
                field = target.field(state)
            if self._due(step, run):
                rounded = state.clone()
                rounded_field = target.field(rounded)
                for j in range(target.n):
                    observation.poll()
                    fill = torch.where(rounded_field[:, j] < 0, 1., -1.)
                    fill = torch.where(rounded[:, j] == 0, fill, rounded[:, j])
                    change = fill-rounded[:, j]
                    rounded[:, j] = fill
                    target.update(rounded_field, j, change)
                self._publish(run, record, rounded, step)
        return self._finish(run, record, target, rounded, self.max_steps)


def _disagreement_component(target, left, right):
    """Exact nonzero-edge connectivity, bounded O(batch*n) frontier memory."""
    disagreement = left != right
    component = torch.zeros_like(disagreement)
    first = disagreement.long().argmax(-1, keepdim=True)
    component.scatter_(1, first, disagreement.any(-1, keepdim=True))
    # A component can grow by at most n-1 graph distances. No host sync needed.
    # abs(matrix) remains one shared operator, never one copy per replica.
    if target.bounds is None:
        adjacency = target.matrix.abs()
    else:
        adjacency = torch.sparse_csr_tensor(target.matrix.crow_indices(), target.matrix.col_indices(),
                                            target.matrix.values().abs(), size=target.matrix.shape)
    for _ in range(max(0, target.n-1)):
        observation.poll()
        expanded = component | ((_mm(adjacency, component.double()) > 0) & disagreement)
        # This bounded graph traversal explicitly synchronizes at convergence.
        # On dense graphs two frontier products replace n otherwise wasted
        # products; ordinary local proposal steps remain device resident.
        if torch.equal(expanded, component):
            break
        component = expanded
    return component


@dataclass(frozen=True)
class C3AshkinTeller(_BDiscrete):
    """56: equal-temperature disagreement-cluster exchange between two replicas."""
    cluster_interval: int = 16

    def __post_init__(self):
        super().__post_init__()
        _integer('cluster_interval', self.cluster_interval, 1)

    def _population_workspace(self, n, batch, width):
        # Shared absolute adjacency needed by exact component propagation.
        return super()._population_workspace(n, batch, width)+8*n*n

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target = _Target(run.objective, occupation=False, normalize=self.normalize)
        left, right = _signs(run), torch.where(_random(run.x.shape, run) < .5, -1., 1.).double()
        lf, rf = target.field(left), target.field(right)
        for step in range(1, self.max_steps+1):
            observation.poll()
            r = _anneal(step, self.max_steps, self.terminal_fraction)
            temperature = self.temperature*(1-r)**2
            i = (step-1) % target.n
            _single(target, left, lf, i, temperature, run)
            _single(target, right, rf, i, temperature, run)
            if step % self.cluster_interval == 0:
                component = _disagreement_component(target, left, right)
                # Swapping the entire disagreement set only exchanges replicas
                # and creates no new configuration, as on a dense full graph.
                component &= (component != (left != right)).any(-1, keepdim=True)
                old = left.clone()
                left = torch.where(component, right, left)
                right = torch.where(component, old, right)
                lf, rf = target.field(left), target.field(right)
            elif step % self.refresh_interval == 0:
                lf, rf = target.field(left), target.field(right)
            if self._due(step, run):
                a, b = run.objective.score(run.to_assignments(left)), run.objective.score(run.to_assignments(right))
                chosen = torch.where((a <= b)[:, None], left, right)
                self._publish(run, record, chosen, step)
        return self._finish(run, record, target, chosen, self.max_steps)


@dataclass(frozen=True)
class C3BlumeEmeryGriffiths(C3BlumeCapel):
    """57: ternary occupancy channel on a sparse ring, removed terminally."""
    occupancy_coupling: float = .4

    def __post_init__(self):
        super().__post_init__()
        _positive('occupancy_coupling', self.occupancy_coupling, True)

    def _extra_occupancy(self, state, i, r):
        neighbours = _ring_neighbours(i, state.shape[1])
        return (self.occupancy_coupling*(1-r)*state[:, neighbours].square().sum(-1)
                / max(1, len(neighbours)))


@dataclass(frozen=True)
class C3FanoAnderson(_BContinuous):
    """58: passive Markovian memory bath, exact skew-coupling split rotations."""
    bath_modes: int = 3
    bath_coupling: float = .5
    slow_rate: float = .2
    fast_rate: float = 2.

    def __post_init__(self):
        super().__post_init__()
        _integer('bath_modes', self.bath_modes, 1)
        if self.bath_modes > 16:
            raise ValueError('bath_modes must be <= 16')
        _positive('bath_coupling', self.bath_coupling, True)
        _positive('slow_rate', self.slow_rate)
        _positive('fast_rate', self.fast_rate)
        if self.fast_rate < self.slow_rate:
            raise ValueError('fast_rate must be >= slow_rate')

    def _prepare(self, z, run):
        rates = [self.slow_rate*(self.fast_rate/self.slow_rate)**(i/max(1, self.bath_modes-1))
                 for i in range(self.bath_modes)]
        return (z, torch.zeros_like(z), torch.zeros((*z.shape, self.bath_modes), device=z.device, dtype=z.dtype)), {'rates': rates}

    def _advance(self, state, force, temperature, r, step, run, cache):
        z, p, bath = state
        dt = min(self.time_step, .25/(1+2*self.pinning))
        p = p+dt*force
        modes = []
        angle = dt*self.bath_coupling/math.sqrt(self.bath_modes)
        c, s = math.cos(angle), math.sin(angle)
        for j, rate in enumerate(cache['rates']):
            decay = math.exp(-rate*dt)
            bj = decay*bath[..., j]+math.sqrt(temperature*(1-decay**2))*run.normal(p.shape)
            p, bj = c*p-s*bj, s*p+c*bj
            modes.append(bj)
        decay = math.exp(-self.damping*dt)
        p = decay*p+math.sqrt(temperature*(1-decay**2))*run.normal(p.shape)
        z, p = _box(z+dt*p, p)
        return z, p, torch.stack(modes, -1)


@dataclass(frozen=True)
class C3XYZZHexagonalCode(_StructuredDecoder):
    """39: explicit commuting-check decoder; the ambiguous name fixes no lattice.

    check_x/check_z are mandatory. The caller supplies its actual geometry and
    masks; the engine validates symplectic commutation and syndrome feasibility.
    No assertion that arbitrary supplied checks form a hexagonal code is made.
    """
    def _execute(self, objective, restarts, batch, initial, seed, best_only, interval):
        if not self.check_x or not self.check_z:
            raise ValueError('XYZZ interpretation requires explicit check_x and check_z metadata')
        return super()._execute(objective, restarts, batch, initial, seed, best_only, interval)


def haah_metadata(size, syndrome=None):
    """Periodic f=1+x+y+z, g=1+xy+xz+yz CSS checks, with mod-2 cancellation.

    Returns solver metadata, not a decoder objective. Error bits are ordered
    (e_X,e_Z), each with the two qubit species interleaved by lattice vertex.
    The caller's QUBO must use these logical coordinates and returned slacks.
    """
    _integer('size', size, 1)
    if size > 8:
        raise ValueError('metadata helper supports size <= 8; explicit dense check storage grows cubically')
    count, qubits = size**3, 2*size**3
    f = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    g = ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
    def site(x, y, z):
        return ((x % size)*size+(y % size))*size+(z % size)
    xrows, zrows = [], []
    for x in range(size):
        for y in range(size):
            for z in range(size):
                xr, zr = [0]*qubits, [0]*qubits
                for kind, offsets in enumerate((f, g)):
                    for a, b, c in offsets:
                        xr[2*site(x+a, y+b, z+c)+kind] ^= 1
                for kind, offsets in enumerate((g, f)):
                    for a, b, c in offsets:
                        zr[2*site(x-a, y-b, z-c)+kind] ^= 1
                xrows.append(tuple(xr))
                zrows.append(tuple(zr))
    zero = (0,)*qubits
    hx, hz = tuple(xrows)+tuple(zero for _ in zrows), tuple(zero for _ in xrows)+tuple(zrows)
    checks = tuple(z+x for x, z in zip(hx, hz))
    if syndrome is None:
        syndrome = (0,)*(2*count)
    if len(syndrome) != 2*count:
        raise ValueError('Haah syndrome must contain 2*size**3 entries')
    slacks, next_bit = [], 2*qubits
    for row in checks:
        bits = (sum(row)//2).bit_length()
        slacks.append(tuple(range(next_bit, next_bit+bits)))
        next_bit += bits
    masks = [tuple(i for i, value in enumerate(row+zero) if value) for row in xrows]
    masks += [tuple(i for i, value in enumerate(zero+row) if value) for row in zrows]
    masks = [m for m in masks if m]
    # Products of translated stabilizers provide a small genuine multiscale
    # family. XOR cancels repeated finite-period supports exactly.
    translations = {(0, 0, 0)}
    for level in range(1, size.bit_length()):
        spacing = 1 << (level-1)
        next_translations = set()
        for ax, ay, az in translations:
            for dx, dy, dz in f:
                translated = ((ax+spacing*dx) % size, (ay+spacing*dy) % size,
                              (az+spacing*dz) % size)
                if translated in next_translations:
                    next_translations.remove(translated)
                else:
                    next_translations.add(translated)
        translations = next_translations
        support = set()
        for dx, dy, dz in sorted(translations):
            index = site(dx, dy, dz)
            support.symmetric_difference_update(i for i, bit in enumerate(xrows[index]) if bit)
        if support:
            masks.append(tuple(sorted(support)))
    return dict(lattice_size=size, check_matrix=checks, syndrome=tuple(syndrome),
                slack_bits=tuple(slacks), check_x=hx, check_z=hz,
                proposal_masks=tuple(masks))


@dataclass(frozen=True)
class C3HaahCubicCode(_StructuredDecoder):
    """42: explicitly validated periodic Haah checks and multiscale masks."""
    lattice_size: int = 0

    def _execute(self, objective, restarts, batch, initial, seed, best_only, interval):
        if self.lattice_size < 1:
            raise ValueError('Haah decoder requires lattice_size and generated check/slack metadata')
        expected = haah_metadata(self.lattice_size, self.syndrome or None)
        for key in ('check_matrix', 'check_x', 'check_z'):
            if tuple(tuple(row) for row in getattr(self, key)) != expected[key]:
                raise ValueError(f'{key} does not match the declared periodic Haah lattice')
        if not self.proposal_masks:
            raise ValueError('Haah decoder requires explicit stabilizer/multiscale proposal_masks')
        return super()._execute(objective, restarts, batch, initial, seed, best_only, interval)


def _group_tables(table):
    table = tuple(tuple(row) for row in table)
    q = len(table)
    if not 2 <= q <= 32 or any(len(row) != q for row in table):
        raise ValueError('group_table must be a square multiplication table of order 2..32')
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < q for row in table for v in row):
        raise ValueError('group_table entries must be in-range integer labels')
    identities = [e for e in range(q) if all(table[e][a] == a and table[a][e] == a for a in range(q))]
    if len(identities) != 1:
        raise ValueError('group_table must have exactly one identity')
    identity = identities[0]
    inverse = []
    for a in range(q):
        choices = [b for b in range(q) if table[a][b] == identity and table[b][a] == identity]
        if len(choices) != 1:
            raise ValueError('group_table must have two-sided inverses')
        inverse.append(choices[0])
    if any(table[table[a][b]][c] != table[a][table[b][c]]
           for a in range(q) for b in range(q) for c in range(q)):
        raise ValueError('group_table must be associative')
    return table, identity, tuple(inverse)


def _group_geometry(edges, faces):
    edges = tuple(tuple(edge) for edge in edges)
    faces = tuple(tuple(tuple(incidence) for incidence in face) for face in faces)
    if not edges or any(len(e) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in e)
                        for e in edges):
        raise ValueError('edge_vertices must contain nonnegative integer endpoint pairs')
    for face in faces:
        if not face:
            raise ValueError('face boundary cannot be empty')
        oriented = []
        for incidence in face:
            if len(incidence) != 2:
                raise ValueError('faces contain (edge_index, orientation) pairs')
            index, direction = incidence
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(edges) or direction not in (-1, 1):
                raise ValueError('invalid oriented face incidence')
            edge = edges[index]
            oriented.append(edge if direction == 1 else edge[::-1])
        if any(a[1] != b[0] for a, b in zip(oriented, oriented[1:]+oriented[:1])):
            raise ValueError('each oriented face must be an ordered closed path')
    return edges, faces


def _group_schema(solver, n, device):
    table, identity, inverse = _group_tables(solver.group_table)
    edges, faces = _group_geometry(solver.edge_vertices, solver.faces)
    q = len(table)
    eb = tuple(tuple(row) for row in solver.edge_bits)
    pb = tuple(tuple(tuple(row) for row in face) for face in solver.prefix_bits)
    tb = tuple(tuple(tuple(row) for row in face) for face in solver.tuple_bits)
    if len(eb) != len(edges) or any(len(row) != q for row in eb):
        raise ValueError('edge_bits must give q one-hot indices per edge')
    if len(pb) != len(faces) or len(tb) != len(faces):
        raise ValueError('prefix_bits and tuple_bits must match faces')
    rows = list(eb)
    for face, prefixes, tuples in zip(faces, pb, tb):
        if len(prefixes) != len(face) or len(tuples) != len(face):
            raise ValueError('each face incidence requires one prefix and multiplication tuple')
        if any(len(row) != q for row in prefixes) or any(len(row) != q*q for row in tuples):
            raise ValueError('prefix/tuple widths must be q and q*q')
        rows.extend(prefixes)
        rows.extend(tuples)
    flat = [index for row in rows for index in row]
    if (any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in flat)
            or len(set(flat)) != len(flat) or set(flat) != set(range(n))):
        raise ValueError('group metadata must cover every QUBO bit exactly once')
    return dict(table=torch.tensor(table, device=device), inverse=torch.tensor(inverse, device=device),
                identity=identity, q=q, edges=edges, faces=faces, edge_bits=eb,
                prefix_bits=pb, tuple_bits=tb, n=n,
                edge_index_tensors=tuple(torch.tensor(row, device=device) for row in eb),
                prefix_index_tensors=tuple(tuple(torch.tensor(row, device=device) for row in face) for face in pb),
                tuple_index_tensors=tuple(tuple(torch.tensor(row, device=device) for row in face) for face in tb))


def _group_encode(labels, schema, dtype=torch.float64):
    """Recompute all finite-group one-hot, prefix and multiplication auxiliaries."""
    b = labels.shape[0]
    rows = torch.arange(b, device=labels.device)
    state = torch.zeros((b, schema['n']), device=labels.device, dtype=dtype)
    for j, indices in enumerate(schema['edge_index_tensors']):
        state[rows, indices[labels[:, j]]] = 1
    for face, prefixes, tuples in zip(schema['faces'], schema['prefix_index_tensors'], schema['tuple_index_tensors']):
        prefix = torch.full((b,), schema['identity'], device=labels.device, dtype=torch.long)
        for (edge, direction), pb, tb in zip(face, prefixes, tuples):
            value = labels[:, edge]
            if direction == -1:
                value = schema['inverse'][value]
            pair = prefix*schema['q']+value
            state[rows, tb[pair]] = 1
            prefix = schema['table'][prefix, value]
            state[rows, pb[prefix]] = 1
    return state


@dataclass(frozen=True)
class C3QuantumDouble(_BBase):
    """43: legal finite-group labels, exact tuple repair and vertex gauge moves.

    Metadata addresses an already compiled classical finite-group QUBO. It does
    not represent a coherent quantum-double state or guarantee hard flatness.
    Single-edge changes permit face defects; gauge moves preserve flatness.
    """
    group_table: tuple = ()
    edge_vertices: tuple = ()
    faces: tuple = ()
    edge_bits: tuple = ()
    prefix_bits: tuple = ()
    tuple_bits: tuple = ()
    polish_sweeps: int = 0
    requires_structure = True

    def __post_init__(self):
        super().__post_init__()
        if self.polish_sweeps:
            raise ValueError('quantum-double polishing must preserve group auxiliaries; use polish_sweeps=0')

    def _population_workspace(self, n, batch, width):
        q = len(self.group_table)
        return super()._population_workspace(n, batch, width)+64*batch*n+32*q*q

    def _execute(self, objective, restarts, batch, initial, seed, best_only, interval):
        if objective.spin:
            raise ValueError('finite-group metadata addresses a binary QUBO')
        schema = _group_schema(self, objective.n, objective.device)
        if initial is None:
            labels = torch.full((1, len(self.edge_vertices)), schema['identity'], device=objective.device, dtype=torch.long)
            initial = _group_encode(labels, schema, objective.dtype).expand(restarts, -1)
        else:
            for start in range(0, restarts, batch):
                selected = initial[start:start+batch]
                labels = torch.stack([selected[:, ids].argmax(-1) for ids in schema['edge_bits']], -1)
                if not bool((selected == _group_encode(labels, schema, objective.dtype)).all()):
                    raise ValueError('initial_assignments violate group encoding constraints')
        return super()._execute(objective, restarts, batch, initial, seed, best_only, interval)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        schema = _group_schema(self, run.x.shape[1], run.x.device)
        labels = torch.stack([run.x[:, ids].argmax(-1) for ids in schema['edge_bits']], -1)
        vertices = sorted({v for edge in self.edge_vertices for v in edge})
        energy = run.objective.score(run.x)
        state = run.x.double().clone()
        for step in range(1, self.max_steps+1):
            observation.poll()
            proposed = labels.clone()
            element = torch.randint(schema['q'], (labels.shape[0],), device=labels.device, generator=run.generator)
            if step % 2:
                proposed[:, ((step-1)//2) % labels.shape[1]] = element
            else:
                vertex = vertices[((step-1)//2) % len(vertices)]
                for i, (a, b) in enumerate(schema['edges']):
                    if a == vertex:
                        proposed[:, i] = schema['table'][element, proposed[:, i]]
                    if b == vertex:
                        proposed[:, i] = schema['table'][proposed[:, i], schema['inverse'][element]]
            candidate = _group_encode(proposed, schema)
            candidate_energy = run.objective.score(candidate)
            r = _anneal(step, self.max_steps, self.terminal_fraction)
            accepted = _metropolis((candidate_energy-energy)/run.objective.scale,
                                   self.temperature*(1-r)**2, run)
            labels = torch.where(accepted[:, None], proposed, labels)
            energy = torch.where(accepted, candidate_energy, energy)
            state = torch.where(accepted[:, None], candidate, state)
            run.retain(state.to(run.x.dtype), energy)
            if self._due(step, run):
                self._publish(run, record, state*2-1, step)
        return self.max_steps


def compile_quantum_double(group_table, edge_vertices, faces, edge_costs, face_penalties,
                           *, penalty=None, device='cpu'):
    """Small exact multiplication-gate compiler, returning (QUBO, metadata).

    Dense output is bounded to 4096 variables; large applications should compile
    the same constraints into the existing compact sparse problem interface.
    The returned metadata also works with that sparse representation.
    """
    from .problems import QUBO
    table, identity, inverse = _group_tables(group_table)
    edges, faces = _group_geometry(edge_vertices, faces)
    q = len(table)
    if len(edge_costs) != len(edges) or any(len(row) != q for row in edge_costs) or len(face_penalties) != len(faces):
        raise ValueError('cost dimensions must match edge labels and faces')
    costs = [float(v) for row in edge_costs for v in row]+[float(v) for v in face_penalties]
    if any(not math.isfinite(v) for v in costs):
        raise ValueError('costs must be finite')
    try:
        bound = math.nextafter(math.fsum(abs(v) for v in costs), math.inf)
    except OverflowError as error:
        raise ValueError('nonpenalty coefficient range overflows float64; rescale costs') from error
    penalty = max(1., bound*1.01) if penalty is None else float(penalty)
    if not math.isfinite(penalty) or penalty <= bound:
        raise ValueError('penalty must exceed the full nonpenalty coefficient range')
    n = len(edges)*q+sum(len(face)*(q+q*q) for face in faces)
    if n > 4096:
        raise ValueError('dense compiler is bounded to 4096 variables; use compact sparse compilation')
    matrix = torch.zeros((n, n), dtype=torch.float64, device=device)
    offset = 0.
    cursor = 0
    def allocate(count):
        nonlocal cursor
        values = tuple(range(cursor, cursor+count))
        cursor += count
        return values
    eb = tuple(allocate(q) for _ in edges)
    pb, tb = [], []
    for face in faces:
        pb.append(tuple(allocate(q) for _ in face))
        tb.append(tuple(allocate(q*q) for _ in face))
    def square(terms, constant=0.):
        nonlocal offset
        # Combine repeated wire incidences before expanding the exact square.
        combined = {}
        for i, value in terms:
            combined[i] = combined.get(i, 0.)+value
        for i, a in combined.items():
            matrix[i, i] += penalty*(a*a+2*constant*a)
            for j, b in combined.items():
                if j > i:
                    matrix[i, j] += 2*penalty*a*b
        offset += penalty*constant*constant
    for ids, costs_row in zip(eb, edge_costs):
        square([(i, 1.) for i in ids], -1.)
        for i, value in zip(ids, costs_row):
            matrix[i, i] += float(value)
    for face, prefixes, tuples, face_cost in zip(faces, pb, tb, face_penalties):
        previous = None
        for (edge, orientation), prefix, pair in zip(face, prefixes, tuples):
            right = eb[edge] if orientation == 1 else tuple(eb[edge][inverse[a]] for a in range(q))
            square([(i, 1.) for i in prefix], -1.)
            for a in range(q):
                terms = [(pair[a*q+b], 1.) for b in range(q)]
                if previous is not None:
                    terms.append((previous[a], -1.))
                square(terms, -float(a == identity) if previous is None else 0.)
            for b in range(q):
                square([(pair[a*q+b], 1.) for a in range(q)]+[(right[b], -1.)])
            for c in range(q):
                square([(pair[a*q+b], 1.) for a in range(q) for b in range(q) if table[a][b] == c]
                       +[(prefix[c], -1.)])
            previous = prefix
        matrix[previous[identity], previous[identity]] -= float(face_cost)
        offset += float(face_cost)
    metadata = dict(group_table=table, edge_vertices=edges, faces=faces,
                    edge_bits=eb, prefix_bits=tuple(pb), tuple_bits=tuple(tb))
    return QUBO(matrix, offset=offset), metadata


COMPENDIUM_III_B_CLASSES = {
    'lib_c3_kugel_khomskii': C3KugelKhomskii,
    'lib_c3_kondo_lattice': C3KondoLattice,
    'lib_c3_hubbard_holstein': C3HubbardHolstein,
    'lib_c3_pair_hopping_bose_hubbard': C3PairHoppingBoseHubbard,
    'lib_c3_quantum_east': C3QuantumEast,
    'lib_c3_quantum_p_spin': C3QuantumPSpin,
    'lib_c3_kicked_ising': C3KickedIsing,
    'lib_c3_wen_plaquette': C3WenPlaquette,
    'lib_c3_schwinger': C3Schwinger,
    'lib_c3_xyzz_hexagonal_code': C3XYZZHexagonalCode,
    'lib_c3_plaquette_ising': C3PlaquetteIsing,
    'lib_c3_spin_transfer_torque': C3SpinTransferTorque,
    'lib_c3_haah_cubic_code': C3HaahCubicCode,
    'lib_c3_quantum_double': C3QuantumDouble,
    'lib_c3_levin_wen': C3LevinWen,
    'lib_c3_double_semion': C3DoubleSemion,
    'lib_c3_quantum_link': C3QuantumLink,
    'lib_c3_majumdar_ghosh': C3MajumdarGhosh,
    'lib_c3_creutz_ladder': C3CreutzLadder,
    'lib_c3_penson_kolb': C3PensonKolb,
    'lib_c3_richardson_gaudin': C3RichardsonGaudin,
    'lib_c3_spinor_rydberg': C3SpinorRydberg,
    'lib_c3_quantum_hopfield': C3QuantumHopfield,
    'lib_c3_quantum_potts': C3QuantumPotts,
    'lib_c3_frustrated_spin_glass': C3FrustratedSpinGlass,
    'lib_c3_blume_capel': C3BlumeCapel,
    'lib_c3_ashkin_teller': C3AshkinTeller,
    'lib_c3_blume_emery_griffiths': C3BlumeEmeryGriffiths,
    'lib_c3_fano_anderson': C3FanoAnderson,
}
