"""Eight bounded Hamiltonian-inspired searches from the October 2026 specification.

The numerical variants are the specified projected semi-implicit SSH step,
normalized imaginary-time Bose step, normalized Heun spin/cavity step,
Armijo Haldane--Shastry step, normalized Euler JC/JCH step and stochastic
spherical n-vector retraction. Rejected trials freeze their controls and their
entire old state. Source-FP64 multilinear energies verify deterministic trials;
the source objective, including its offset, always ranks binary incumbents.

Periodic auxiliary graphs use distinct neighbours (one edge for two sites and
no self edge for one). FFT drivers never allocate a dense auxiliary matrix.
State, target fields, random draws, trial acceptance and readout stay on device.
Only immutable sparse row boundaries are copied to the host for coordinate
readout. A step budget includes terminal relaxation when ``terminal_steps`` is
positive; it never silently adds work beyond ``max_steps``.
"""

from dataclasses import dataclass
import math

import torch

from . import observation
from ._dynamics import finite, positive
from ._population import PopulationSolver, _mm
from .solvers import _integer


def _normalize(value):
    length = value.norm(dim=-1, keepdim=True)
    return value / length.clamp_min(torch.finfo(value.dtype).tiny)


def _project(vector, value):
    return value - vector * (vector * value).sum(-1, keepdim=True)


def _ring_sum(value):
    n = value.shape[1]
    if n == 1:
        return torch.zeros_like(value)
    if n == 2:
        return value.flip((1,))
    return value.roll(1, 1) + value.roll(-1, 1)


class _Ring:
    """A fixed independent logical-site permutation for each restart."""

    def __init__(self, run, permute=True):
        shape = run.x.shape
        self.order = (run.normal(shape).argsort(1) if permute else
                      torch.arange(shape[1], device=run.x.device).expand(shape[0], -1))
        self.inverse = self.order.argsort(1)

    @staticmethod
    def _gather(value, order):
        shape = (*order.shape, *value.shape[2:])
        while order.ndim < value.ndim:
            order = order.unsqueeze(-1)
        return value.gather(1, order.expand(shape))

    def arrange(self, value):
        return self._gather(value, self.order)

    def restore(self, value):
        return self._gather(value, self.inverse)

    def neighbours(self, value):
        return self.restore(_ring_sum(self.arrange(value)))

    def local_neighbours(self, value, site):
        """One site's distinct auxiliary neighbours, without a full ring pass."""
        n = value.shape[1]
        if n == 1:
            return torch.zeros_like(value[:, site])
        position = self.inverse[:, site:site+1]
        right = self.order.gather(1, (position+1) % n)
        result = self._gather(value, right).squeeze(1)
        if n > 2:
            left = self.order.gather(1, (position-1) % n)
            result = result+self._gather(value, left).squeeze(1)
        return result

    def convolve(self, value, spectrum):
        ordered = self.arrange(value)
        weight = spectrum.reshape(1, -1, *([1] * (value.ndim - 2)))
        product = torch.fft.irfft(torch.fft.rfft(ordered, dim=1) * weight,
                                  n=value.shape[1], dim=1)
        return self.restore(product)

    def dimer(self, value, intra, inter, transpose=False):
        ordered = self.arrange(value)
        out = intra * ordered
        if transpose:
            out[:, :-1].add_(ordered[:, 1:], alpha=inter)
        else:
            out[:, 1:].add_(ordered[:, :-1], alpha=inter)
        return self.restore(out)


class _Target:
    """FP64 source extension and exact coordinate fields, with no dense copy.

    QUBO diagonal entries and compact self loops are linear in occupations.
    Treating those entries as squared soft occupations is an incorrect target.
    The library gradient has the energy-gradient sign; physical fields are its
    negative. Its scalar normalization is retained throughout each trajectory.
    """

    def __init__(self, objective):
        observation.poll()
        self.objective = objective
        self.scale = objective.scale
        self.matrix = objective.source_matrix
        self.terms = objective.source_terms
        self.spin = objective.spin
        self.n = objective.n
        if self.terms is not None:
            linear, heads, tails, biases = self.terms
            loops = heads == tails
            self.linear = linear.clone().index_add_(0, heads[loops], biases[loops])
            heads, tails, biases = heads[~loops], tails[~loops], biases[~loops]
            self.edges = (heads, tails, biases)
            indices = torch.stack((torch.cat((heads, tails)), torch.cat((tails, heads))))
            values = torch.cat((biases, biases)) * .25
            self.canonical = torch.sparse_coo_tensor(indices, values, (self.n, self.n),
                device=linear.device, dtype=torch.float64).coalesce().to_sparse_csr()
            self.field = self.linear * .5
            self.field.index_add_(0, heads, biases * .25)
            self.field.index_add_(0, tails, biases * .25)
            self.multiplier = 1.
            self.diagonal = torch.zeros_like(linear)
        else:
            self.canonical = self.matrix
            if self.matrix.layout == torch.strided:
                self.diagonal = self.matrix.diagonal()
                row_sum = self.matrix.sum(1)
            else:
                rows = torch.repeat_interleave(torch.arange(self.n, device=self.matrix.device),
                    self.matrix.crow_indices()[1:] - self.matrix.crow_indices()[:-1])
                self.diagonal = torch.zeros(self.n, dtype=torch.float64, device=rows.device)
                mask = rows == self.matrix.col_indices()
                self.diagonal.index_add_(0, rows[mask], self.matrix.values()[mask])
                row_sum = torch.zeros_like(self.diagonal).index_add_(0, rows, self.matrix.values())
            self.field = (objective.source_field if self.spin else row_sum * .5)
            self.multiplier = 2. if self.spin else .5
        self.bounds = (None if self.canonical.layout == torch.strided else
                       self.canonical.crow_indices().cpu().tolist())
        coefficients = (self.canonical if self.canonical.layout == torch.strided
                        else self.canonical.values())
        # Source coefficients decide constancy; a small FP64 input can underflow
        # in FP32 dynamics without being an actually constant objective.
        self.constant = not bool((coefficients != 0).any() | (self.field != 0).any())
        observation.poll()

    def gradient(self, z):
        return self.objective.gradient(z)

    def exact_gradient(self, z):
        z = z.to(torch.float64)
        value = _mm(self.canonical, z) - z * self.diagonal
        return value * self.multiplier + self.field

    def energy(self, z, *, include_offset=False):
        z = z.to(torch.float64)
        if self.terms is not None:
            p = (z + 1) * .5
            energy = p @ self.linear
            heads, tails, biases = self.edges
            chunk = max(1, 1_048_576 // z.shape[0])
            for start in range(0, biases.numel(), chunk):
                observation.poll()
                stop = start + chunk
                energy.add_((p[:, heads[start:stop]] * p[:, tails[start:stop]]
                             * biases[start:stop]).sum(-1))
        elif self.spin:
            energy = (_mm(self.matrix, z) * z).sum(-1) + (z * self.field).sum(-1)
        else:
            p = (z + 1) * .5
            energy = (_mm(self.matrix, p) * p).sum(-1)
            energy.add_(((p - p.square()) * self.diagonal).sum(-1))
        if include_offset:
            energy = energy + self.objective.offset
        return energy / self.scale

    def update_field(self, field, site, delta):
        if self.bounds is None:
            row = self.canonical[site]
            field.add_(delta[:, None] * row[None], alpha=self.multiplier)
            field[:, site].sub_(delta * self.diagonal[site] * self.multiplier)
        else:
            start, stop = self.bounds[site:site + 2]
            columns = self.canonical.col_indices()[start:stop]
            values = self.canonical.values()[start:stop] * self.multiplier
            field.index_add_(1, columns, delta[:, None] * values[None])
            field[:, site].sub_(delta * self.diagonal[site] * self.multiplier)

    def readout(self, z, sweeps=1, conditional=False):
        """Sequential conditional rounding or strict original-scale bit flips."""
        values = (z.to(torch.float64).clone() if conditional else
                  torch.where(z >= 0, 1., -1.).to(torch.float64))
        field = self.exact_gradient(values)
        for sweep in range(max(sweeps, int(conditional))):
            for site in range(self.n):
                if site % 32 == 0:
                    observation.poll()
                old = values[:, site].clone()
                if conditional and sweep == 0:
                    new = torch.where(field[:, site] >= 0, -1., 1.)
                else:
                    improve = old * field[:, site] > 0
                    new = torch.where(improve, -old, old)
                values[:, site] = new
                self.update_field(field, site, new - old)
        return values


def _validate(solver):
    _integer('max_steps', solver.max_steps, 0)
    _integer('candidate_interval', solver.candidate_interval, 1)
    _integer('polish_sweeps', solver.polish_sweeps, 0)
    _integer('terminal_steps', solver.terminal_steps, 0)
    positive('time_step', solver.time_step)
    if not isinstance(solver.normalize, bool):
        raise ValueError('normalize must be boolean')
    if solver.dynamics_dtype not in (None, torch.float32, torch.float64, 'float32', 'float64'):
        raise ValueError('dynamics_dtype must be float32, float64, or None')
    if (isinstance(solver.sparse_threshold, bool) or not math.isfinite(solver.sparse_threshold)
            or not 0 <= solver.sparse_threshold <= 1):
        raise ValueError('sparse_threshold must be within [0, 1]')


def _progress(solver, step):
    anneal = max(1, solver.max_steps - min(solver.terminal_steps, solver.max_steps - 1))
    return max(0., min(1., observation.schedule_fraction(step / anneal)))


def _checkpoint(solver, run, target, record, z, step, *, conditional=False):
    if (step % solver.candidate_interval == 0 or step == solver.max_steps
            or (run.history_interval is not None and step % run.history_interval == 0)):
        run.publish(target.readout(z, solver.polish_sweeps, conditional), step)
        record(step)


def _constant_run(solver, run, target, record):
    """Preserve the input candidate and complete requested history on constants."""
    if not target.constant:
        return False
    if run.history_interval is not None:
        for step in range(run.history_interval, solver.max_steps + 1, run.history_interval):
            record(step)
    if solver.max_steps and (run.history_interval is None
                            or solver.max_steps % run.history_interval):
        record(solver.max_steps)
    return True


def _choose(mask, new, old):
    return torch.where(mask.reshape(mask.shape + (1,) * (new.ndim - 1)), new, old)


def _backtrack(state, trial, energy, time_step, *, tolerance=1e-12,
               decrease=None, floor=2.**-20):
    """Per-replica frozen-origin retries; capped at twenty halvings.

    A failed replica keeps every old coordinate. The caller marks a numerical
    stall, or performs its chapter's explicit restart. Ordinary acceptance needs
    one GPU reduction; no replica's accepted step is changed by another replica.
    """
    batch = state[0].shape[0]
    base = energy(state)
    dt = state[0].new_full((batch,), time_step)
    accepted = torch.zeros(batch, dtype=torch.bool, device=dt.device)
    output = tuple(value.clone() for value in state)
    for _ in range(21):
        observation.poll()
        proposed = trial(state, dt)
        value = energy(proposed)
        bound = base + tolerance * (1 + base.abs())
        if decrease is not None:
            bound = bound - 1e-4 * dt.to(torch.float64) * decrease
        valid = torch.isfinite(value)
        for component in proposed:
            valid &= torch.isfinite(component).reshape(batch, -1).all(-1)
        good = (~accepted) & valid & (value <= bound)
        output = tuple(_choose(good, new, old) for new, old in zip(proposed, output))
        accepted |= good
        if bool(accepted.all()):
            break
        dt = torch.where(accepted, dt, dt * .5)
        if bool(((dt < time_step * floor) | accepted).all()):
            break
    return output, ~accepted


def _step_size(dt, value):
    return dt.reshape(dt.shape + (1,) * (value.ndim - 1))


def _spin_velocity(vector, field, damping):
    cross = torch.linalg.cross(vector, field)
    return cross - damping * torch.linalg.cross(vector, cross)


def _hs_spectra(n, like):
    k = torch.arange(n // 2 + 1, device=like.device, dtype=torch.float64)
    laplacian = 2 * (math.pi / n)**2 * k * (n - k)
    degree = (math.pi / n)**2 * (n * n - 1) / 3
    return (degree - laplacian).to(like.dtype), laplacian.to(like.dtype)


def _inozemtsev_spectrum(n, a, like):
    """Uniform 1e-12 image-tail bound and overflow-safe sinh ratios in FP64."""
    if n < 3:
        return _hs_spectra(n, like)[0]
    images = 0
    while True:
        decay = math.exp(-2 * a * (images * n + 1))
        tail = 8 * math.sinh(a)**2 * decay / ((1 - decay)**2 * (-math.expm1(-2*a*n)))
        if tail < 1e-12:
            break
        images += 1
    r = torch.arange(n, device=like.device, dtype=torch.float64)
    r = torch.minimum(r, n - r)
    kernel = torch.zeros_like(r)
    for image in range(-images, images + 1):
        distance = (r + image * n).abs()
        exponent = a * distance
        # sinh(a)^2/sinh(a*d)^2, without overflowing the denominator.
        ratio = torch.exp(2 * (a - exponent)) * (
            (-math.expm1(-2*a)) / (-torch.expm1(-2*exponent)).clamp_min(1e-300))**2
        kernel.add_(torch.where(r > 0, ratio, 0.))
    return torch.fft.rfft(kernel).real.to(like.dtype)


@dataclass(frozen=True)
class _Continuous(PopulationSolver):
    max_steps: int = 2000
    time_step: float = .02
    candidate_interval: int = 20
    polish_sweeps: int = 1
    terminal_steps: int = 0
    normalize: bool = True
    dynamics_dtype: object = None
    sparse_threshold: float = .15

    def __post_init__(self):
        _validate(self)

    def _population_workspace(self, n, batch, width):
        # Source-FP64 trial verification, normalized states, rejected replicas,
        # FFT complex workspace, immutable permutations, and coordinate fields.
        # The base compact estimate already includes CSR construction and
        # source-term adjacency; no quadratic allocation is needed for it.
        return (width + 8) * (96 * batch * n + 128 * batch)


@dataclass(frozen=True)
class SSHDimerOscillator(_Continuous):
    """Bounded logical/partner oscillators with projected dissipative steps."""

    time_step: float = .05
    kappa: float = 1.
    damping: float = 1.
    intra_hopping: float = .5
    inter_hopping: float = 1.
    driver: float = 2.

    def __post_init__(self):
        super().__post_init__()
        for name in ('kappa', 'damping'):
            positive(name, getattr(self, name))
        for name in ('intra_hopping', 'inter_hopping', 'driver'):
            positive(name, getattr(self, name), zero=True)

    def _energy(self, state, target, ring, r, eta):
        a, d, v, w = (value.to(torch.float64) for value in state)
        return (target.energy(a) + .5 * (v.square() + w.square()).sum(-1)
                + self.kappa / 4 * ((a.square()-r).square() + (d.square()-r).square()).sum(-1)
                - eta * (a * ring.dimer(d, self.intra_hopping, self.inter_hopping)).sum(-1))

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target, ring = _Target(run.objective), _Ring(run)
        if _constant_run(self, run, target, record):
            return 0
        a, d = (torch.rand(run.x.shape, device=run.x.device, dtype=run.x.dtype,
                    generator=run.generator) * .1 - .05 for _ in range(2))
        state = (a, d, torch.zeros_like(a), torch.zeros_like(d))
        for step in range(1, self.max_steps + 1):
            progress = _progress(self, step)
            r, eta = -5 + 6*progress, self.driver*(1-progress)**2
            a, d, v, w = state
            fa = (-target.gradient(a) - self.kappa*a*(a.square()-r)
                  + eta*ring.dimer(d, self.intra_hopping, self.inter_hopping))
            fd = (-self.kappa*d*(d.square()-r)
                  + eta*ring.dimer(a, self.intra_hopping, self.inter_hopping, True))

            def trial(old, dt):
                aa, dd, vv, ww = old
                h = dt[:, None]
                an = (aa + h*(vv+h*fa)/(1+self.damping*h)).clamp(-1, 1)
                dn = (dd + h*(ww+h*fd)/(1+self.damping*h)).clamp(-1, 1)
                return an, dn, (an-aa)/h, (dn-dd)/h

            energy = lambda values: self._energy(values, target, ring, r, eta)
            state, failed = _backtrack(state, trial, energy, self.time_step)
            if bool(failed.any()):
                # The specified single zero-velocity retry precedes restart.
                a, d, v, w = state
                reset = (a, d, _choose(failed, torch.zeros_like(v), v),
                         _choose(failed, torch.zeros_like(w), w))
                retried, retry_failed = _backtrack(reset, trial, energy, self.time_step)
                state = tuple(_choose(failed, new, old) for new, old in zip(retried, state))
                retry_failed &= failed
                if bool(retry_failed.any()):
                    fresh = tuple(torch.rand(run.x.shape, device=run.x.device, dtype=run.x.dtype,
                        generator=run.generator)*.1-.05 for _ in range(2))
                    state = tuple(_choose(retry_failed, new, old) for new, old in
                                  zip((*fresh, torch.zeros_like(a), torch.zeros_like(d)), state))
            _checkpoint(self, run, target, record, state[0], step)
        finite(*state)
        return self.max_steps


@dataclass(frozen=True)
class BoseHubbard(_Continuous):
    """Two complex modes per cell with projected imaginary-time dynamics."""

    time_step: float = .1
    mixing: float = 1.
    hopping: float = .25
    kerr: float = 1.
    relaxation: float = 1.
    driver: float = 2.

    def __post_init__(self):
        super().__post_init__()
        for name in ('mixing', 'hopping', 'kerr', 'driver'):
            positive(name, getattr(self, name), zero=True)
        positive('relaxation', self.relaxation)

    def _energy(self, vector, target, ring, eta):
        z = vector.to(torch.float64)
        populations = z.square().sum(-1)
        p = populations[..., 1]
        driver = -2*self.mixing*(z[..., 0, :]*z[..., 1, :]).sum((-1, -2))
        driver -= self.hopping*(z*ring.neighbours(z)).sum((-1, -2, -3))
        driver += self.kerr/2*populations.square().sum((-1, -2))
        return target.energy(2*p-1) + eta*driver

    def _gradient(self, z, target, ring, eta):
        populations = z.square().sum(-1)
        g = eta*(-self.mixing*z.flip((-2,)) - self.hopping*ring.neighbours(z)
                 + self.kerr*populations[..., None]*z)
        g[..., 1, :].add_(2*target.gradient(2*populations[..., 1]-1)[..., None]*z[..., 1, :])
        # Wirtinger projection Re(z^dagger g); two real coordinates store each
        # complex amplitude, so no complex kernels or 2n-square matrix is needed.
        mu = (z*g).sum((-1, -2), keepdim=True)
        return g - mu*z

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target, ring = _Target(run.objective), _Ring(run)
        if _constant_run(self, run, target, record):
            return 0
        z = run.normal((*run.x.shape, 2, 2))
        z /= z.square().sum((-1, -2), keepdim=True).sqrt().clamp_min(torch.finfo(z.dtype).tiny)
        for step in range(1, self.max_steps + 1):
            eta = self.driver*(1-_progress(self, step))**2
            tangent = self._gradient(z, target, ring, eta)

            def trial(old, dt):
                value = old[0] - self.relaxation*_step_size(dt, z)*tangent
                length = value.square().sum((-1, -2), keepdim=True).sqrt()
                return (value/length.clamp_min(torch.finfo(z.dtype).tiny),)

            (z,), failed = _backtrack((z,), trial,
                lambda state: self._energy(state[0], target, ring, eta), self.time_step)
            if bool(failed.any()):
                fresh = run.normal(z.shape)
                fresh /= fresh.square().sum((-1, -2), keepdim=True).sqrt().clamp_min(torch.finfo(z.dtype).tiny)
                z = _choose(failed, fresh, z)
            p = z[..., 1, :].square().sum(-1)
            _checkpoint(self, run, target, record, 2*p-1, step, conditional=True)
        finite(z)
        return self.max_steps


@dataclass(frozen=True)
class Inozemtsev(_Continuous):
    """Damped unit-spin Heun search with a periodic image-summed exchange."""

    time_step: float = .003
    damping: float = .2
    exchange: float = 1.
    interaction_range: float = 1.
    transverse_field: float = 1.
    initial_pinning: float = .05
    final_pinning: float = .55

    def __post_init__(self):
        super().__post_init__()
        positive('interaction_range', self.interaction_range)
        if self.interaction_range > 100:
            raise ValueError('interaction_range must be at most 100 for stable image normalization')
        for name in ('damping', 'transverse_field', 'initial_pinning', 'final_pinning'):
            positive(name, getattr(self, name), zero=True)
        if isinstance(self.exchange, bool) or not math.isfinite(self.exchange):
            raise ValueError('exchange must be finite')

    def _field(self, vector, target, ring, spectrum, controls):
        tau, gamma, kappa = controls
        field = -tau*self.exchange/4*ring.convolve(vector, spectrum)
        field[..., 0].add_(gamma)
        field[..., 2].add_(-target.gradient(vector[..., 2])+2*kappa*vector[..., 2])
        return field

    def _energy(self, vector, target, ring, spectrum, controls):
        tau, gamma, kappa = controls
        value = vector.to(torch.float64)
        return (target.energy(value[..., 2]) + kappa*(1-value[..., 2].square()).sum(-1)
                - gamma*value[..., 0].sum(-1) + tau*self.exchange/8
                *(value*ring.convolve(value, spectrum.to(torch.float64))).sum((-1, -2)))

    def _velocity(self, vector, target, ring, spectrum, controls):
        return _spin_velocity(vector, self._field(vector, target, ring, spectrum, controls), self.damping)

    def _spectrum(self, n, like):
        return _inozemtsev_spectrum(n, self.interaction_range, like)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target, ring = _Target(run.objective), _Ring(run)
        if _constant_run(self, run, target, record):
            return 0
        vector = _normalize(run.normal((*run.x.shape, 3)))
        spectrum = self._spectrum(run.x.shape[1], vector)
        active = torch.ones(run.x.shape[0], dtype=torch.bool, device=vector.device)
        for step in range(1, self.max_steps + 1):
            r = _progress(self, step)
            controls = ((1-r)**2, self.transverse_field*(1-r),
                        self.initial_pinning+(self.final_pinning-self.initial_pinning)*r)
            first = self._velocity(vector, target, ring, spectrum, controls)

            def trial(old, dt):
                h = _step_size(dt, vector)
                predictor = _normalize(old[0]+h*first)
                second = self._velocity(predictor, target, ring, spectrum, controls)
                return (_normalize(old[0]+h/2*(first+second)),)

            (candidate,), failed = _backtrack((vector,), trial,
                lambda state: self._energy(state[0], target, ring, spectrum, controls), self.time_step)
            vector = _choose(active, candidate, vector)
            active &= ~failed
            run.reasons.masked_fill_(~active, -1)
            _checkpoint(self, run, target, record, vector[..., 2], step)
        finite(vector)
        return self.max_steps


@dataclass(frozen=True)
class HaldaneShastry(Inozemtsev):
    """Inverse-square exchange with positive, twice-projected FFT mobility."""

    time_step: float = .02
    damping: float = .3
    mobility: float = .5

    def __post_init__(self):
        super().__post_init__()
        positive('mobility', self.mobility, zero=True)

    def _spectrum(self, n, like):
        return _hs_spectra(n, like)[0]

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target, ring = _Target(run.objective), _Ring(run)
        if _constant_run(self, run, target, record):
            return 0
        vector = _normalize(run.normal((*run.x.shape, 3)))
        spectrum, laplacian = _hs_spectra(run.x.shape[1], vector)
        mobility = 1/(1+self.mobility*laplacian)
        active = torch.ones(run.x.shape[0], dtype=torch.bool, device=vector.device)
        for step in range(1, self.max_steps + 1):
            r = _progress(self, step)
            controls = ((1-r)**2, self.transverse_field*(1-r),
                        self.initial_pinning+(self.final_pinning-self.initial_pinning)*r)
            field = self._field(vector, target, ring, spectrum, controls)
            tangent = _project(vector, field)
            filtered = ring.convolve(tangent, mobility)
            velocity = torch.linalg.cross(vector, field)+self.damping*_project(vector, filtered)
            power = self.damping*(tangent.to(torch.float64)*filtered.to(torch.float64)).sum((-1, -2))
            trial = lambda old, dt: (_normalize(old[0]+_step_size(dt, vector)*velocity),)
            (candidate,), failed = _backtrack((vector,), trial,
                lambda state: self._energy(state[0], target, ring, spectrum, controls),
                self.time_step, decrease=power)
            vector = _choose(active, candidate, vector)
            active &= ~failed
            run.reasons.masked_fill_(~active, -1)
            _checkpoint(self, run, target, record, vector[..., 2], step)
        finite(vector)
        return self.max_steps


@dataclass(frozen=True)
class Dicke(_Continuous):
    """One dynamic collective cavity per replica, coupled to all unit spins."""

    time_step: float = .01
    cavity_frequency: float = 1.
    atomic_frequency: float = 1.
    coupling: float = .5
    cavity_damping: float = .1
    damping: float = .2
    transverse_field: float = 1.
    initial_pinning: float = .05
    final_pinning: float = .55

    def __post_init__(self):
        super().__post_init__()
        for name in ('cavity_frequency', 'atomic_frequency'):
            positive(name, getattr(self, name))
        for name in ('coupling', 'cavity_damping', 'damping', 'transverse_field',
                     'initial_pinning', 'final_pinning'):
            positive(name, getattr(self, name), zero=True)

    def _rhs(self, state, target, controls):
        spin, cavity = state
        tau, gamma, kappa = controls
        q, p = cavity.unbind(-1)
        cg = math.sqrt(2/spin.shape[1])*self.coupling
        field = torch.zeros_like(spin)
        field[..., 0] = gamma-tau*cg*q[:, None]
        field[..., 2] = -target.gradient(spin[..., 2])+2*kappa*spin[..., 2]-tau*self.atomic_frequency/2
        photon = torch.stack((self.cavity_frequency*p,
            -self.cavity_frequency*q-tau*cg*spin[..., 0].sum(-1)-self.cavity_damping*p), -1)
        return _spin_velocity(spin, field, self.damping), photon

    def _energy(self, state, target, controls):
        spin, cavity = (value.to(torch.float64) for value in state)
        tau, gamma, kappa = controls
        sx = spin[..., 0].sum(-1)
        cg = math.sqrt(2/spin.shape[1])*self.coupling
        return (target.energy(spin[..., 2])+kappa*(1-spin[..., 2].square()).sum(-1)-gamma*sx
                + self.cavity_frequency/2*cavity.square().sum(-1)
                + tau*(self.atomic_frequency/2*spin[..., 2].sum(-1)+cg*cavity[:, 0]*sx))

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective)
        if _constant_run(self, run, target, record):
            return 0
        state = (_normalize(run.normal((*run.x.shape, 3))), run.normal((run.x.shape[0], 2))*.01)
        active = torch.ones(run.x.shape[0], dtype=torch.bool, device=run.x.device)
        for step in range(1, self.max_steps + 1):
            r = _progress(self, step)
            controls = ((1-r)**2, self.transverse_field*(1-r),
                        self.initial_pinning+(self.final_pinning-self.initial_pinning)*r)
            first = self._rhs(state, target, controls)

            def trial(old, dt):
                predictor = tuple(value+_step_size(dt, value)*force for value, force in zip(old, first))
                predictor = (_normalize(predictor[0]), predictor[1])
                second = self._rhs(predictor, target, controls)
                result = tuple(value+_step_size(dt, value)/2*(one+two)
                               for value, one, two in zip(old, first, second))
                return _normalize(result[0]), result[1]

            candidate, failed = _backtrack(state, trial,
                lambda values: self._energy(values, target, controls), self.time_step)
            state = tuple(_choose(active, new, old) for new, old in zip(candidate, state))
            active &= ~failed
            run.reasons.masked_fill_(~active, -1)
            _checkpoint(self, run, target, record, state[0][..., 2], step)
        finite(*state)
        return self.max_steps


@dataclass(frozen=True)
class JaynesCummings(_Continuous):
    """Local real-quadrature photons and normalized Bloch-spin Euler trials."""

    time_step: float = .002857142857142857
    cavity_frequency: float = 2.
    atomic_frequency: float = 1.
    coupling: float = 1.
    photon_damping: float = 1.
    damping: float = 1.
    final_pinning: float = .5

    def __post_init__(self):
        super().__post_init__()
        positive('cavity_frequency', self.cavity_frequency)
        for name in ('atomic_frequency', 'coupling', 'photon_damping', 'damping', 'final_pinning'):
            positive(name, getattr(self, name), zero=True)

    def _photon_action(self, photon, ring):
        return self.cavity_frequency*photon

    def _rhs(self, state, target, ring, controls):
        spin, photon = state
        tau, kappa = controls
        gradient = self._photon_action(photon, ring)
        gradient[..., 0].add_(spin[..., 0], alpha=tau*self.coupling/2)
        gradient[..., 1].add_(spin[..., 1], alpha=-tau*self.coupling/2)
        real, imag = gradient.unbind(-1)
        photon_rhs = torch.stack((-self.photon_damping*real+imag,
                                  -self.photon_damping*imag-real), -1)
        field = torch.empty_like(spin)
        field[..., 0] = -tau*self.coupling*photon[..., 0]
        field[..., 1] = tau*self.coupling*photon[..., 1]
        field[..., 2] = -target.gradient(spin[..., 2])+2*kappa*spin[..., 2]-tau*self.atomic_frequency/2
        return _spin_velocity(spin, field, self.damping), photon_rhs

    def _energy(self, state, target, ring, controls):
        spin, photon = (value.to(torch.float64) for value in state)
        tau, kappa = controls
        coupling = photon[..., 0]*spin[..., 0]-photon[..., 1]*spin[..., 1]
        return (target.energy(spin[..., 2])+kappa*(1-spin[..., 2].square()).sum(-1)
                +(photon*self._photon_action(photon, ring)).sum((-1, -2))
                +tau*(self.atomic_frequency/2*spin[..., 2]+self.coupling*coupling).sum(-1))

    def _initial_photon(self, spin):
        return -self.coupling/(2*self.cavity_frequency)*torch.stack((spin[..., 0], -spin[..., 1]), -1)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target, ring = _Target(run.objective), _Ring(run)
        if _constant_run(self, run, target, record):
            return 0
        azimuth = torch.rand(run.x.shape, device=run.x.device, dtype=run.x.dtype,
                             generator=run.generator)*(2*math.pi)
        z = torch.rand(run.x.shape, device=run.x.device, dtype=run.x.dtype,
                        generator=run.generator)*.1-.05
        radius = (1-z.square()).sqrt()
        spin = torch.stack((radius*azimuth.cos(), radius*azimuth.sin(), z), -1)
        state = (spin, self._initial_photon(spin))
        active = torch.ones(run.x.shape[0], dtype=torch.bool, device=run.x.device)
        for step in range(1, self.max_steps + 1):
            tau = (1-_progress(self, step))**2
            controls = (tau, self.final_pinning*(1-tau))
            first = self._rhs(state, target, ring, controls)

            def trial(old, dt):
                result = tuple(value+_step_size(dt, value)*force for value, force in zip(old, first))
                return _normalize(result[0]), result[1]

            candidate, failed = _backtrack(state, trial,
                lambda values: self._energy(values, target, ring, controls),
                self.time_step, tolerance=1e-10,
                floor=max(2.**-20, 1e-12/self.time_step))
            state = tuple(_choose(active, new, old) for new, old in zip(candidate, state))
            active &= ~failed
            run.reasons.masked_fill_(~active, -1)
            _checkpoint(self, run, target, record, state[0][..., 2], step)
        finite(*state)
        return self.max_steps


@dataclass(frozen=True)
class JaynesCummingsHubbard(JaynesCummings):
    """Sparse periodic photon transport, with a strictly positive photon form."""

    cavity_frequency: float = 1.5
    hopping: float = .25

    def __post_init__(self):
        super().__post_init__()
        positive('hopping', self.hopping, zero=True)
        if self.cavity_frequency <= 2*self.hopping:
            raise ValueError('cavity_frequency must exceed 2*hopping for positive photon energy')

    def _photon_action(self, photon, ring):
        return self.cavity_frequency*photon-self.hopping*ring.neighbours(photon)

    def _initial_photon(self, spin):
        return torch.zeros_like(spin[..., :2])


@dataclass(frozen=True)
class NVector(_Continuous):
    """Projected spherical diffusion with an explicit temperature schedule."""

    dimension: int = 3
    driver: float = .25
    final_pinning: float = 2.
    initial_temperature: float = .25
    final_temperature: float = .001

    def __post_init__(self):
        super().__post_init__()
        _integer('dimension', self.dimension, 1)
        for name in ('driver', 'final_pinning', 'initial_temperature', 'final_temperature'):
            positive(name, getattr(self, name), zero=True)
        if self.initial_temperature == 0 and self.final_temperature != 0:
            raise ValueError('zero initial_temperature requires zero final_temperature')

    def _population_workspace(self, n, batch, width):
        return super()._population_workspace(n, batch, width)+(width+8)*12*self.dimension*n*batch

    def _field(self, vector, target, ring, tau, kappa):
        degree = min(2, max(0, vector.shape[1]-1))
        field = tau*ring.neighbours(vector)/max(1, degree)
        field[..., 0].add_(-target.gradient(vector[..., 0])+2*kappa*vector[..., 0])
        return field

    def _energy(self, vector, target, ring, tau, kappa):
        value = vector.to(torch.float64)
        degree = min(2, max(0, value.shape[1]-1))
        return (target.energy(value[..., 0])+kappa*(1-value[..., 0].square()).sum(-1)
                -tau/(2*max(1, degree))*(value*ring.neighbours(value)).sum((-1, -2)))

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target, ring = _Target(run.objective), _Ring(run)
        if _constant_run(self, run, target, record):
            return 0
        vector = _normalize(run.normal((*run.x.shape, self.dimension)))
        for step in range(1, self.max_steps + 1):
            r = _progress(self, step)
            tau, kappa = self.driver*(1-r)**2, self.final_pinning*r*r
            temperature = (self.initial_temperature*(self.final_temperature/self.initial_temperature)**r
                           if self.initial_temperature > 0 else 0.)
            if step > self.max_steps - min(self.terminal_steps, self.max_steps - 1):
                temperature = 0.
            if self.dimension == 1:
                # There is no tangent direction on S^0. Sequential exact-field
                # heat-bath updates are the specified discrete alternative.
                logical = vector[..., 0]
                cached = target.exact_gradient(logical)
                degree = max(1, min(2, vector.shape[1]-1))
                for site in range(run.x.shape[1]):
                    if site % 32 == 0:
                        observation.poll()
                    # Pinning is zero at either pole and has no heat-bath bias.
                    local = (-cached[:, site]/target.scale
                             +tau*ring.local_neighbours(logical, site)/degree)
                    probability = ((2*local/max(temperature, 1e-30)).sigmoid()
                                   if temperature > 0 else
                                   torch.where(local == 0, .5, (local > 0).to(local.dtype)))
                    draw = torch.rand(local.shape, device=local.device, dtype=local.dtype,
                                      generator=run.generator)
                    old = logical[:, site].clone()
                    logical[:, site] = torch.where(draw < probability, 1., -1.)
                    target.update_field(cached, site, logical[:, site]-old)
            else:
                field = self._field(vector, target, ring, tau, kappa)
                tangent = _project(vector, field)
                increment = self.time_step*tangent
                if temperature > 0:
                    increment += math.sqrt(2*temperature*self.time_step)*_project(vector, run.normal(vector.shape))
                    vector = _normalize(vector+increment)
                else:
                    trial = lambda old, dt: (_normalize(old[0]+_step_size(dt, vector)*tangent),)
                    (vector,), failed = _backtrack((vector,), trial,
                        lambda state: self._energy(state[0], target, ring, tau, kappa), self.time_step)
                    run.reasons.masked_fill_(failed, -1)
            logical = vector[..., 0]
            # Random zero ties are specified for n-vector readout.
            ties = logical == 0
            if bool(ties.any()):
                logical = torch.where(ties, run.normal(logical.shape), logical)
            _checkpoint(self, run, target, record, logical, step)
        finite(vector)
        return self.max_steps


HAMILTONIAN_CONTINUOUS_SOLVERS = {
    'lib_ssh': SSHDimerOscillator,
    'lib_bose_hubbard': BoseHubbard,
    'lib_inozemtsev': Inozemtsev,
    'lib_dicke': Dicke,
    'lib_haldane_shastry': HaldaneShastry,
    'lib_jaynes_cummings': JaynesCummings,
    'lib_jaynes_cummings_hubbard': JaynesCummingsHubbard,
    'lib_n_vector': NVector,
}
