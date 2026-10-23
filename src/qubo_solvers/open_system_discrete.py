"""Seven classical search engines from the open-system compendium.

These are the explicitly declared optimization reductions, not simulations of
their entangled quantum namesakes. FP64 source coefficients drive discrete gains;
replicas run in parallel and interacting moves commit sequentially. Strong-edge
proposal graphs never replace the full objective. Enumeration is chunk bounded.
"""

from dataclasses import dataclass
import math

import torch

from ._population import PopulationSolver, _dtype, _mm
from .hamiltonian_discrete import _Target, _random, _metropolis
from .observation import poll
from .solvers import _integer


def _positive(name, value, zero=False):
    if isinstance(value, bool) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def _state(run):
    return run.x.double().clone() if run.spin else run.x.double().mul(2).sub(1)


def _target(run, normalize=False):
    target = _Target(run.objective, occupation=False, normalize=normalize)
    coefficients = target.matrix if target.bounds is None else target.matrix.values()
    if not bool(torch.isfinite(coefficients).all() & torch.isfinite(target.bias).all()):
        raise FloatingPointError('nonfinite source target coefficients')
    return target


def _scale(target):
    if target.bounds is None:
        bound = target.matrix.abs().sum(1)
    else:
        rows = torch.repeat_interleave(torch.arange(target.n, device=target.bias.device),
            target.matrix.crow_indices().diff(), output_size=target.matrix.values().numel())
        bound = torch.zeros_like(target.bias).index_add_(0, rows, target.matrix.values().abs())
    scale = (target.bias.abs() + bound).max()
    if not bool(torch.isfinite(scale)):
        raise FloatingPointError('source normalization overflows float64')
    return scale.clamp_min(torch.finfo(torch.float64).tiny)


def _submatrix(target, indices):
    """Only the bounded induced block is dense; CSR never expands globally."""
    ids = torch.tensor(indices, device=target.bias.device)
    if target.bounds is None:
        return target.matrix[ids[:, None], ids[None, :]]
    rows = []
    for i in indices:
        columns, values = target.column(i)
        rows.append(torch.where(columns[:, None] == ids[None, :], values[:, None], 0.).sum(0))
    return torch.stack(rows)


def _neighborhoods(target, size):
    """Fixed shared strong-neighbor graph, O(n*size), with disconnected fill."""
    size = min(size, target.n)
    if target.bounds is None:
        neighbours = target.matrix.abs().topk(size, dim=1).indices.cpu().tolist()
    else:
        cols = target.matrix.col_indices().cpu().tolist()
        vals = target.matrix.values().abs().cpu().tolist()
        neighbours = [sorted(range(target.bounds[i], target.bounds[i + 1]),
                             key=vals.__getitem__, reverse=True)[:size] for i in range(target.n)]
        neighbours = [[cols[j] for j in row] for row in neighbours]
    blocks = []
    for i, row in enumerate(neighbours):
        block = [i] + [j for j in row if j != i]
        for offset in range(1, size + 1):
            j = (i + offset) % target.n
            if j not in block:
                block.append(j)
        blocks.append(block[:size])
    return blocks


def _block_gain(old, new, field, matrix):
    delta = new - old
    return (delta * field).sum(-1) + .5 * ((delta @ matrix) * delta).sum(-1)


def _commit(target, state, field, indices, proposed, accepted=None):
    old = state[:, indices].clone()
    if accepted is not None:
        proposed = torch.where(accepted[:, None], proposed, old)
    delta = proposed - old
    state[:, indices] = proposed
    for j, i in enumerate(indices):
        target.update(field, i, delta[:, j])


def _enumerate_block(old, field, matrix, chunk_size=256):
    """Exact conditional minimum, streaming candidates; old wins exact ties."""
    k = old.shape[1]
    boundary = field - old @ matrix
    best = old.clone()
    best_energy = (boundary * old + .5 * (old @ matrix) * old).sum(-1)
    shifts = torch.arange(k, device=old.device)
    for start in range(0, 1 << k, chunk_size):
        poll()
        integers = torch.arange(start, min(start + chunk_size, 1 << k), device=old.device)
        candidates = ((integers[:, None] >> shifts) & 1).double().mul_(2).sub_(1)
        energy = boundary @ candidates.T + .5 * ((candidates @ matrix) * candidates).sum(-1)
        values, winners = energy.min(-1)
        better = values < best_energy
        best = torch.where(better[:, None], candidates[winners], best)
        best_energy = torch.minimum(best_energy, values)
    return best


def _range_two_minimum(boundary, first, second):
    """Batched exact open-chain DP for positive-energy convention h*s+.5*s*A*s."""
    batch, n = boundary.shape
    signs = boundary.new_tensor([-1., 1.])
    if n == 1:
        return torch.where(boundary <= 0, 1., -1.)
    scores = (boundary[:, 0, None, None] * signs[None, :, None]
              + boundary[:, 1, None, None] * signs[None, None, :]
              + first[0] * signs[:, None] * signs[None, :])
    parents = []
    for i in range(2, n):
        poll()
        candidates = (scores[..., None] + boundary[:, i, None, None, None] * signs
                      + first[i - 1] * signs[None, None, :, None] * signs
                      + second[i - 2] * signs[None, :, None, None] * signs)
        scores, predecessor = candidates.min(1)
        parents.append(predecessor)
    final = scores.reshape(batch, 4).argmin(-1)
    result = torch.empty((batch, n), device=boundary.device, dtype=torch.float64)
    a, b = final // 2, final % 2
    result[:, -2], result[:, -1] = signs[a], signs[b]
    rows = torch.arange(batch, device=boundary.device)
    for i in range(n - 1, 1, -1):
        previous = parents[i - 2][rows, a, b]
        result[:, i - 2] = signs[previous]
        a, b = previous, a
    return result


def _activity_step(activity, gains, heads, tails, hopping, *, dephasing,
                   reinjection, activity_temperature, rate_cap):
    """Symmetric edge diffusion with a positivity-bounded Euler time step."""
    eps = gains.clamp(-2., 2.)
    rates = (2 * hopping.square() * dephasing /
             (dephasing ** 2 + (eps[:, heads] - eps[:, tails]).square() + 1e-12)).clamp_max(rate_cap)
    row_rates = torch.zeros_like(activity)
    row_rates.index_add_(1, heads, rates)
    row_rates.index_add_(1, tails, rates)
    dt = .9 / (reinjection + row_rates.max(-1, keepdim=True).values).clamp_min(1e-12)
    flux = rates * (activity[:, tails] - activity[:, heads])
    derivative = torch.zeros_like(activity)
    derivative.index_add_(1, heads, flux)
    derivative.index_add_(1, tails, -flux)
    source = 1 / (1 + gains.abs() / activity_temperature)
    source = .95 * source / source.sum(-1, keepdim=True) + .05 / activity.shape[1]
    updated = activity + dt * (derivative + reinjection * (source - activity))
    return updated.clamp_min(0) / updated.sum(-1, keepdim=True)


@dataclass(frozen=True)
class _OpenBase(PopulationSolver):
    max_steps: int = 128
    check_interval: int = 1
    refresh_interval: int = 32
    polish_sweeps: int = 1
    dynamics_dtype: object = None
    normalize: bool = True
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('check_interval', self.check_interval, 1)
        _integer('refresh_interval', self.refresh_interval, 1)
        _integer('polish_sweeps', self.polish_sweeps, 0)
        _dtype(self.dynamics_dtype, torch.float32)
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if not math.isfinite(self.sparse_threshold) or not 0 <= self.sparse_threshold <= 1:
            raise ValueError('sparse_threshold must be in [0, 1]')

    def _population_workspace(self, n, batch, width):
        k = min(n, getattr(self, 'block_size', 8))
        return 256 * n * batch + 64 * k * k + 64 * batch * (256 + k) + 128 * n * k

    def _publish(self, run, record, state, step, force=False):
        if (force or step % self.check_interval == 0 or step == self.max_steps or
                run.history_interval and step % run.history_interval == 0):
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError('nonfinite open-system search state')
            run.publish(state, step)
            record(step)

    def _finish(self, run, record, target, state, step):
        self._publish(run, record, state, step, True)
        if self.polish_sweeps:
            state = run.best_x.double().clone() if run.spin else run.best_x.double().mul(2).sub(1)
            field = target.field(state)
            for _ in range(self.polish_sweeps):
                for i in range(target.n):
                    poll()
                    new = torch.where(field[:, i] < 0, 1., -1.)
                    delta = new - state[:, i]
                    state[:, i] = new
                    target.update(field, i, delta)
            self._publish(run, record, state, step, True)
        return step


@dataclass(frozen=True)
class Kondo(_OpenBase):
    """Exact strong-neighborhood conditional enumeration (zero temperature)."""
    block_size: int = 8

    def __post_init__(self):
        super().__post_init__()
        _integer('block_size', self.block_size, 1)
        if self.block_size > 16:
            raise ValueError('block_size must be <= 16 (exponential enumeration)')

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target, state = _target(run), _state(run)
        if target.constant:
            return 0
        field = target.field(state)
        blocks = _neighborhoods(target, self.block_size)
        # One setup transfer; no coordinate or replica device-to-host loop.
        anchors = torch.randperm(target.n, generator=run.generator, device=state.device).cpu().tolist()
        for step in range(1, self.max_steps + 1):
            poll()
            indices = blocks[anchors[(step - 1) % target.n]]
            matrix = _submatrix(target, indices)
            proposed = _enumerate_block(state[:, indices], field[:, indices], matrix)
            gain = _block_gain(state[:, indices], proposed, field[:, indices], matrix)
            _commit(target, state, field, indices, proposed, gain < 0)
            if step % self.refresh_interval == 0:
                field = target.field(state)
            self._publish(run, record, state, step)
        return self._finish(run, record, target, state, step)


@dataclass(frozen=True)
class MooreRead(_OpenBase):
    """Eight-mask triple search, strong wedges mixed with seeded random triples."""
    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target, state = _target(run), _state(run)
        if target.constant:
            return 0
        field = target.field(state)
        blocks = _neighborhoods(target, 3)
        order = torch.randperm(target.n, generator=run.generator, device=state.device).cpu().tolist()
        for step in range(1, self.max_steps + 1):
            poll()
            anchor = (step - 1) % target.n
            indices = (blocks[order[anchor]] if step % 5 else
                       [order[(anchor + j) % target.n] for j in range(min(3, target.n))])
            matrix = _submatrix(target, indices)
            proposed = _enumerate_block(state[:, indices], field[:, indices], matrix, 8)
            gain = _block_gain(state[:, indices], proposed, field[:, indices], matrix)
            _commit(target, state, field, indices, proposed, gain < 0)
            if step % self.refresh_interval == 0:
                field = target.field(state)
            self._publish(run, record, state, step)
        return self._finish(run, record, target, state, step)


@dataclass(frozen=True)
class ANNNI(_OpenBase):
    """Open range-two DP; general blocks require full-objective acceptance."""
    block_size: int = 24

    def __post_init__(self):
        super().__post_init__()
        _integer('block_size', self.block_size, 1)
        if self.block_size > 128:
            raise ValueError('block_size must be <= 128')

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target, state = _target(run), _state(run)
        if target.constant:
            return 0
        field = target.field(state)
        # Exact native open-chain path, without an n-by-n auxiliary allocation.
        if target.bounds is None:
            columns = torch.arange(target.n, device=state.device)
            violations = torch.zeros((), device=state.device, dtype=torch.bool)
            for start in range(0, target.n, 256):
                poll()
                rows = columns[start:start + 256]
                outside = (rows[:, None] - columns[None, :]).abs() > 2
                violations |= (outside & (target.matrix[start:start + 256] != 0)).any()
            chain = not bool(violations)
            first, second = target.matrix.diagonal(1), target.matrix.diagonal(2)
        else:
            coo = target.matrix.to_sparse_coo().coalesce()
            rows, cols = coo.indices()
            chain = bool(((rows - cols).abs()[coo.values() != 0] <= 2).all())
            first = state.new_zeros(max(0, target.n - 1))
            second = state.new_zeros(max(0, target.n - 2))
            first[rows[cols - rows == 1]] = coo.values()[cols - rows == 1]
            second[rows[cols - rows == 2]] = coo.values()[cols - rows == 2]
        if chain:
            proposed = _range_two_minimum(target.bias.expand_as(state), first, second)
            return self._finish(run, record, target, proposed, 1)
        blocks = _neighborhoods(target, self.block_size)
        order = torch.randperm(target.n, generator=run.generator, device=state.device).cpu().tolist()
        for step in range(1, self.max_steps + 1):
            poll()
            indices = blocks[order[(step - 1) % target.n]]
            matrix = _submatrix(target, indices)
            boundary = field[:, indices] - state[:, indices] @ matrix
            proposed = _range_two_minimum(boundary, matrix.diagonal(1), matrix.diagonal(2))
            gain = _block_gain(state[:, indices], proposed, field[:, indices], matrix)
            _commit(target, state, field, indices, proposed, gain < 0)
            if step % self.refresh_interval == 0:
                field = target.field(state)
            self._publish(run, record, state, step)
        return self._finish(run, record, target, state, step)


@dataclass(frozen=True)
class LiebMattis(_OpenBase):
    """Four exact collective masks on a seeded partition of a strong block."""
    block_size: int = 32

    def __post_init__(self):
        super().__post_init__()
        _integer('block_size', self.block_size, 1)
        if self.block_size > 128:
            raise ValueError('block_size must be <= 128')

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target, state = _target(run), _state(run)
        if target.constant:
            return 0
        field = target.field(state)
        blocks = _neighborhoods(target, self.block_size)
        order = torch.randperm(target.n, generator=run.generator, device=state.device).cpu().tolist()
        for step in range(1, self.max_steps + 1):
            poll()
            indices = blocks[order[(step - 1) % target.n]]
            matrix, old = _submatrix(target, indices), state[:, indices]
            partition = _random(old.shape, run) < .5
            proposals = [old, torch.where(partition, -old, old),
                         torch.where(partition, old, -old), -old]
            gains = torch.stack([_block_gain(old, new, field[:, indices], matrix)
                                 for new in proposals], -1)
            winners = gains.argmin(-1)
            proposed = torch.stack(proposals, 1)[torch.arange(state.shape[0], device=state.device), winners]
            _commit(target, state, field, indices, proposed, gains.min(-1).values < 0)
            if step % self.refresh_interval == 0:
                field = target.field(state)
            self._publish(run, record, state, step)
        return self._finish(run, record, target, state, step)


@dataclass(frozen=True)
class NoisyExcitationTransport(_OpenBase):
    """Positive activity diffusion on a sparse graph with exact single flips."""
    activity_degree: int = 4
    dephasing: float = .5
    reinjection: float = .2
    activity_temperature: float = .2
    rate_cap: float = 2.
    temperature: float = .25

    def __post_init__(self):
        super().__post_init__()
        _integer('activity_degree', self.activity_degree, 1)
        if self.activity_degree > 32:
            raise ValueError('activity_degree must be <= 32')
        for name in ('dephasing', 'reinjection', 'activity_temperature', 'rate_cap'):
            _positive(name, getattr(self, name))
        _positive('temperature', self.temperature, True)

    def _population_workspace(self, n, batch, width):
        return super()._population_workspace(n, batch, width) + 128 * n * self.activity_degree * batch

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target, state = _target(run), _state(run)
        if target.constant:
            return 0
        scale, field = _scale(target), target.field(state)
        blocks = _neighborhoods(target, self.activity_degree + 1)
        edges = sorted({tuple(sorted((i, j))) for i, row in enumerate(blocks) for j in row if i != j})
        heads = torch.tensor([a for a, b in edges], device=state.device, dtype=torch.long)
        tails = torch.tensor([b for a, b in edges], device=state.device, dtype=torch.long)
        # Graph coefficients are shared; sparse lookup allocates only row length.
        if target.bounds is None:
            hopping = target.matrix[heads, tails].abs() / scale
        else:
            coo = target.matrix.to_sparse_coo().coalesce()
            keys = coo.indices()[0] * target.n + coo.indices()[1]
            query = heads * target.n + tails
            if keys.numel():
                found = torch.searchsorted(keys, query).clamp_max(keys.numel() - 1)
                hopping = torch.where(keys[found] == query, coo.values()[found].abs(), 0.) / scale
            else:
                hopping = state.new_zeros(query.shape)
        activity = torch.full_like(state, 1 / target.n)
        rows = torch.arange(state.shape[0], device=state.device)
        for step in range(1, self.max_steps + 1):
            poll()
            gains = -2 * state * field
            activity = _activity_step(activity, gains / scale, heads, tails, hopping,
                dephasing=self.dephasing, reinjection=self.reinjection,
                activity_temperature=self.activity_temperature, rate_cap=self.rate_cap)
            chosen = torch.multinomial(activity, 1, generator=run.generator).squeeze(1)
            temperature = self.temperature * max(0., 1 - step / (.9 * self.max_steps)) ** 2
            accepted = _metropolis(gains[rows, chosen] / scale, temperature, run, strict=True)
            # A batched single-flip vector supports distinct sites per replica.
            # Full sparse field multiplication avoids dense CSR column gathering.
            changes = -2 * state[rows, chosen] * accepted
            state[rows, chosen] += changes
            if target.bounds is None:
                field.add_(changes[:, None] * target.matrix[chosen])
            else:
                delta = torch.zeros_like(state)
                delta[rows, chosen] = changes
                field.add_(_mm(target.matrix, delta))
            if step % self.refresh_interval == 0:
                field = target.field(state)
            self._publish(run, record, state, step)
        return self._finish(run, record, target, state, step)


@dataclass(frozen=True)
class AKLT(_OpenBase):
    """Ternary heat bath with positive endpoint pinning and sequential commits."""
    temperature: float = .5
    pinning: float = .5

    def __post_init__(self):
        super().__post_init__()
        _positive('temperature', self.temperature, True)
        _positive('pinning', self.pinning, True)

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target, state = _target(run), _state(run)
        if target.constant:
            return 0
        scale = _scale(target)
        state.masked_fill_(_random(state.shape, run) < .25, 0.)
        field = target.field(state)
        labels = state.new_tensor([-1., 0., 1.])
        order = torch.randperm(target.n, generator=run.generator, device=state.device).cpu().tolist()
        for step in range(1, self.max_steps + 1):
            poll()
            fraction = step / self.max_steps
            pin = self.pinning * (3 * fraction ** 2 - 2 * fraction ** 3)
            temperature = self.temperature * max(0., 1 - fraction / .9) ** 2
            for i in order:
                poll()
                energies = field[:, i, None] / scale * labels + pin * (1 - labels.square())
                if temperature:
                    # Subtract first: very small positive T must not form inf-inf.
                    weights = torch.exp(-(energies - energies.min(-1, keepdim=True).values) / temperature)
                    chosen = torch.multinomial(weights, 1, generator=run.generator).squeeze(1)
                else:
                    chosen = energies.argmin(-1)
                new = labels[chosen]
                delta = new - state[:, i]
                state[:, i] = new
                target.update(field, i, delta)
            if step % self.refresh_interval == 0:
                field = target.field(state)
            # Sequential zero filling on a copy leaves exploratory vacancies intact.
            decoded, decoded_field = state.clone(), field.clone()
            for i in range(target.n):
                poll()
                delta = torch.where(decoded[:, i] == 0,
                    torch.where(decoded_field[:, i] <= 0, 1., -1.), 0.)
                decoded[:, i].add_(delta)
                target.update(decoded_field, i, delta)
            self._publish(run, record, decoded, step)
        return self._finish(run, record, target, decoded, step)


def _emery_step(target, z, u, v, *, dt, pinning, stiffness_u, stiffness_v,
                coupling_u, coupling_v, mobility_u, mobility_v):
    r, t = u - coupling_u * z, v - coupling_v * z
    force = (-target.field(z) + 2 * pinning * z
             + stiffness_u * coupling_u * r + stiffness_v * coupling_v * t)
    proposal = z + dt * force
    # Projection must not turn overflow into a plausible finite corner.
    z = torch.where(torch.isfinite(proposal), proposal.clamp(-1, 1), float('nan'))
    u = coupling_u * z + math.exp(-stiffness_u * mobility_u * dt) * (u - coupling_u * z)
    v = coupling_v * z + math.exp(-stiffness_v * mobility_v * dt) * (v - coupling_v * z)
    return z, u, v


@dataclass(frozen=True)
class Emery(_OpenBase):
    """Three-layer square-compensated projected descent with two memory times."""
    dt: float = .15
    pinning: float = .5
    stiffness_u: float = .5
    stiffness_v: float = .5
    coupling_u: float = 1.
    coupling_v: float = .5
    mobility_u: float = .5
    mobility_v: float = 4.
    initial_residual: float = .1

    def __post_init__(self):
        super().__post_init__()
        for name in ('dt', 'stiffness_u', 'stiffness_v', 'mobility_u', 'mobility_v'):
            _positive(name, getattr(self, name))
        for name in ('pinning', 'initial_residual'):
            _positive(name, getattr(self, name), True)
        for name in ('coupling_u', 'coupling_v'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        try:
            bound = (2 * self.pinning + self.stiffness_u * self.coupling_u ** 2
                     + self.stiffness_v * self.coupling_v ** 2)
        except OverflowError as error:
            raise ValueError('auxiliary stiffness bound overflows float64') from error
        if not math.isfinite(bound):
            raise ValueError('auxiliary stiffness bound overflows float64')

    def _search(self, run, record):
        if not self.max_steps:
            return 0
        target = _target(run, normalize=self.normalize)
        if target.constant:
            return 0
        z = _state(run).mul(.5)
        u = self.coupling_u * z + self.initial_residual * (_random(z.shape, run) * 2 - 1)
        v = self.coupling_v * z + self.initial_residual * (_random(z.shape, run) * 2 - 1)
        bound = float(_scale(target)) + 2 * self.pinning + self.stiffness_u * self.coupling_u ** 2 + self.stiffness_v * self.coupling_v ** 2
        dt = min(self.dt, 1 / max(bound, 1e-12))
        for step in range(1, self.max_steps + 1):
            poll()
            fraction = step / self.max_steps
            pin = self.pinning * (3 * fraction ** 2 - 2 * fraction ** 3)
            z, u, v = _emery_step(target, z, u, v, dt=dt, pinning=pin,
                stiffness_u=self.stiffness_u, stiffness_v=self.stiffness_v,
                coupling_u=self.coupling_u, coupling_v=self.coupling_v,
                mobility_u=self.mobility_u, mobility_v=self.mobility_v)
            if not bool(torch.isfinite(u).all() & torch.isfinite(v).all()):
                raise FloatingPointError('nonfinite Emery auxiliary state')
            self._publish(run, record, z, step)
        return self._finish(run, record, _target(run), z, step)


OPEN_DISCRETE_CLASSES = {
    'lib_noisy_excitation_transport': NoisyExcitationTransport,
    'lib_kondo': Kondo,
    'lib_moore_read': MooreRead,
    'lib_emery': Emery,
    'lib_aklt': AKLT,
    'lib_annni': ANNNI,
    'lib_lieb_mattis': LiebMattis,
}
