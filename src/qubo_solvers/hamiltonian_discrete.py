"""Five Hamiltonian search candidates from the 23-model specification.

The ternary, categorical, cyclic, angular and worldline carriers are distinct
algorithms. Coefficients and move acceptance use FP64; replicas are parallel,
while interacting coordinates commit sequentially. Sparse inputs stay sparse.
``max_steps`` counts sweeps (worldlines count n*P slice visits per sweep).
Defaults are starting schedules, not performance or convergence claims.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from ._population import PopulationSolver, _mm, _dtype
from .observation import poll
from .solvers import _integer


class _Target:
    """Source-precision multilinear operator with bounded column updates."""

    def __init__(self, objective, *, occupation=True, normalize=True):
        n, device = objective.n, objective.device
        if objective.source_terms is not None:
            linear, heads, tails, biases = objective.source_terms
            diagonal = heads == tails
            h = linear.clone().mul_(.5)
            h.index_add_(0, heads, biases * torch.where(diagonal, .5, .25))
            h.index_add_(0, tails, biases * torch.where(diagonal, 0., .25))
            values = torch.where(diagonal, 0., biases * .25)
            matrix = torch.sparse_coo_tensor(
                torch.stack((torch.cat((heads, tails)), torch.cat((tails, heads)))),
                torch.cat((values, values)), (n, n)).coalesce().to_sparse_csr()
        else:
            source = objective.source_matrix
            h = (objective.source_field.clone() if objective.spin else
                 source.sum(1) * .5 if source.layout == torch.strided else
                 torch.sparse.sum(source.to_sparse_coo(), dim=1).to_dense() * .5)
            if source.layout == torch.strided:
                matrix = source.clone().mul_(2. if objective.spin else .5)
                matrix.diagonal().zero_()
            else:
                rows = torch.repeat_interleave(torch.arange(n, device=device),
                    source.crow_indices().diff(), output_size=source.values().numel())
                values = source.values().clone().mul_(2. if objective.spin else .5)
                values.masked_fill_(rows == source.col_indices(), 0.)
                matrix = torch.sparse_csr_tensor(source.crow_indices(), source.col_indices(),
                                                 values, size=source.shape)
        if occupation:
            ones = torch.ones((1, n), device=device, dtype=torch.float64)
            h = 2 * h - 2 * _mm(matrix, ones)[0]
            matrix = matrix * 4
        if matrix.layout == torch.strided:
            bound = matrix.abs().sum(1)
        else:
            rows = torch.repeat_interleave(torch.arange(n, device=device),
                matrix.crow_indices().diff(), output_size=matrix.values().numel())
            bound = torch.zeros_like(h).index_add_(0, rows, matrix.values().abs())
        scale = (h.abs() + bound).max()
        self.constant = bool(scale == 0)
        self.scale = scale.clamp_min(torch.finfo(torch.float64).tiny) if normalize else scale.new_ones(())
        self.bias = h / self.scale
        self.matrix = (matrix / self.scale if matrix.layout == torch.strided else
                       torch.sparse_csr_tensor(matrix.crow_indices(), matrix.col_indices(),
                           matrix.values() / self.scale, size=matrix.shape))
        self.n = n
        self.bounds = (None if matrix.layout == torch.strided else
                       self.matrix.crow_indices().cpu().tolist())

    def field(self, values):
        return _mm(self.matrix, values) + self.bias

    def energy(self, values):
        return (.5 * _mm(self.matrix, values) * values + self.bias * values).sum(-1)

    def column(self, i):
        if self.bounds is None:
            return slice(None), self.matrix[i]
        start, stop = self.bounds[i:i + 2]
        return self.matrix.col_indices()[start:stop], self.matrix.values()[start:stop]

    def update(self, field, i, delta):
        indices, values = self.column(i)
        field[:, indices] += delta[:, None] * values

    def entry(self, i, j):
        if self.bounds is None:
            return self.matrix[i, j]
        indices, values = self.column(i)
        return torch.where(indices == j, values, 0.).sum()


def _random(shape, run):
    return torch.rand(shape, device=run.q.device, dtype=torch.float64, generator=run.generator)


def _metropolis(delta, temperature, run, strict=False):
    if isinstance(temperature, (int, float)) and temperature == 0:
        return delta < 0 if strict else delta <= 0
    return (delta <= 0) | (_random(delta.shape, run).log() < -delta / temperature)


def _ring(n, run, permute=True):
    order = (torch.randperm(n, device=run.q.device, generator=run.generator).cpu().tolist()
             if permute else list(range(n)))
    neighbours = [[] for _ in order]
    if n == 2:
        neighbours[0].append((1, 1., 1.))
        neighbours[1].append((0, 1., -1.))
    elif n > 2:
        for a, b in zip(order, order[1:] + order[:1]):
            neighbours[a].append((b, .5, 1.))
            neighbours[b].append((a, .5, -1.))
    return order, neighbours


def _round(target, occupation):
    """Conditional-expectation readout, on a copy, with exact field updates."""
    values = occupation.clone()
    field = target.field(values)
    for i in range(target.n):
        poll()
        new = (field[:, i] < 0).to(values.dtype)
        delta = new - values[:, i]
        target.update(field, i, delta)
        values[:, i] = new
    return values


def _publish_occupation(run, occupation, step):
    run.publish(occupation.mul(2).sub(1), step)


def _decode(values, run):
    return torch.where(values == .5, _random(values.shape, run) < .5,
                       values > .5).to(torch.float64)


@dataclass(frozen=True)
class _DiscreteSolver(PopulationSolver):
    max_steps: int = 512
    check_interval: int = 1
    readout_interval: int = 20
    polish_sweeps: int = 0
    dynamics_dtype: object = None
    normalize: bool = True
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('check_interval', self.check_interval, 1)
        _integer('readout_interval', self.readout_interval, 1)
        _integer('polish_sweeps', self.polish_sweeps, 0)
        _dtype(self.dynamics_dtype, torch.float32)
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')
        if not math.isfinite(self.sparse_threshold) or not 0 <= self.sparse_threshold <= 1:
            raise ValueError('sparse_threshold must be in [0, 1]')

    def _population_workspace(self, n, batch, width):
        return 192 * n * batch + 4096 * batch + 128 * n

    def _record(self, run, record, values, target, step):
        _publish_occupation(run, _decode(values, run), step)
        if step % self.readout_interval == 0 or step == self.max_steps:
            rounded = _round(target, values)
            run.retain(run.to_assignments(rounded * 2 - 1))
        if (step % self.check_interval == 0 or step == self.max_steps or
                run.history_interval and step % run.history_interval == 0):
            record(step)

    def _polish(self, run, target):
        if self.polish_sweeps:
            values = ((run.best_x.double() + 1) * .5 if run.spin else run.best_x.double())
            for _ in range(self.polish_sweeps):
                values = _round(target, values)
            run.retain(run.to_assignments(values * 2 - 1))


@dataclass(frozen=True)
class TJVacancy(_DiscreteSolver):
    """Ternary Metropolis labels with symmetric vacancy exchanges."""

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective, occupation=False, normalize=self.normalize)
        if target.constant:
            return 0
        state = (run.x.double() if run.spin else run.x.double() * 2 - 1)
        state.masked_fill_(_random(state.shape, run) < .5, 0.)
        field = target.field(state)
        order, _ = _ring(target.n, run)
        edges = list(zip(order, order[1:] + order[:1])) if target.n > 2 else ([(0, 1)] if target.n == 2 else [])
        for step in range(1, self.max_steps + 1):
            poll()
            t = step / self.max_steps
            chemical, temperature = -1 + 2 * t, 2 * (.01 / 2) ** t
            sites = torch.randint(target.n, (target.n,), generator=run.generator,
                                  device=run.q.device).cpu().tolist()
            for i in sites:
                poll()
                old = state[:, i].clone()
                choices = (old.long() + 1 + torch.randint(1, 3, old.shape,
                    generator=run.generator, device=old.device)) % 3 - 1
                proposed = choices.double()
                delta = proposed - old
                cost = delta * field[:, i] + chemical * (old.square() - proposed.square())
                single = _random(old.shape, run) < (.5 if edges else 1.)
                accepted = _metropolis(cost, temperature, run) & single
                change = delta * accepted
                state[:, i].add_(change)
                target.update(field, i, change)
                if edges:
                    # A fixed ring edge distribution is symmetric; use one edge
                    # for all rows, with independent row acceptance decisions.
                    a, b = edges[i % len(edges)]
                    da = state[:, b] - state[:, a]
                    cost = da * (field[:, a] - field[:, b]) - target.entry(a, b) * da.square()
                    accepted = _metropolis(cost, temperature, run) & ~single
                    change = da * accepted
                    state[:, a].add_(change)
                    state[:, b].sub_(change)
                    target.update(field, a, change)
                    target.update(field, b, -change)
            if step % 100 == 0:
                field = target.field(state)
            # Fill holes by sequential conditional minimization, leaving the
            # ternary exploration state untouched.
            filled, fill_field = state.clone(), field.clone()
            for i in range(target.n):
                poll()
                value = torch.where(fill_field[:, i] <= 0, 1., -1.)
                delta = torch.where(filled[:, i] == 0, value, 0.)
                filled[:, i].add_(delta)
                target.update(fill_field, i, delta)
            run.publish(filled, step)
            record(step)
        self._polish(run, _Target(run.objective, normalize=self.normalize))
        return self.max_steps


@dataclass(frozen=True)
class PottsHeatBath(_DiscreteSolver):
    """Rank-one QUBO categorical heat bath with an implicit Potts ring."""
    labels: int = 4
    driver_strength: float = .25

    def __post_init__(self):
        super().__post_init__()
        _integer('labels', self.labels, 2)
        if not math.isfinite(self.driver_strength) or self.driver_strength < 0:
            raise ValueError('driver_strength must be finite and nonnegative')

    def _population_workspace(self, n, batch, width):
        return super()._population_workspace(n, batch, width) + 96 * batch * self.labels

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective, normalize=self.normalize)
        if target.constant:
            return 0
        batch, n = run.x.shape
        labels = torch.randint(self.labels, (batch, n), generator=run.generator, device=run.q.device)
        levels = torch.linspace(0, 1, self.labels, device=run.q.device, dtype=torch.float64)
        values = levels[labels]
        field = target.field(values)
        _, neighbours = _ring(n, run)
        for step in range(1, self.max_steps + 1):
            poll()
            t = step / self.max_steps
            tau, kappa, temperature = self.driver_strength * (1 - t) ** 2, t * t, .5 * (.01 / .5) ** t
            sites = torch.randperm(n, generator=run.generator, device=run.q.device).cpu().tolist()
            for i in sites:
                poll()
                costs = field[:, i, None] * levels + kappa * levels * (1 - levels)
                for j, weight, _ in neighbours[i]:
                    costs -= tau * weight * (torch.arange(self.labels, device=labels.device)[None] == labels[:, j, None])
                logits = -(costs - costs.min(-1, keepdim=True).values) / temperature
                draw = torch.multinomial(logits.softmax(-1), 1, generator=run.generator)[:, 0]
                delta = levels[draw] - values[:, i]
                labels[:, i] = draw
                values[:, i] += delta
                target.update(field, i, delta)
            if step % 100 == 0:
                field = target.field(values)
            self._record(run, record, values, target, step)
        self._polish(run, target)
        return self.max_steps


def _zn_conditional(label, neighbour, orientation, q, phase):
    return -torch.cos((label - neighbour) * (2 * math.pi / q) + orientation * phase)


@dataclass(frozen=True)
class ZNChiral(_DiscreteSolver):
    """Cyclic Metropolis labels with adjacent and antipodal proposals."""
    labels: int = 8
    chirality: float = math.pi / 6
    driver_strength: float = .25
    antipodal_probability: float = .2

    def __post_init__(self):
        super().__post_init__()
        _integer('labels', self.labels, 2)
        if self.labels % 2:
            raise ValueError('labels must be even for exact antipodal encoding')
        for name in ('chirality', 'driver_strength', 'antipodal_probability'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if self.driver_strength < 0 or not 0 <= self.antipodal_probability <= 1:
            raise ValueError('driver_strength must be nonnegative and antipodal_probability in [0,1]')

    def _population_workspace(self, n, batch, width):
        return super()._population_workspace(n, batch, width) + 32 * self.labels

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective, normalize=self.normalize)
        if target.constant:
            return 0
        batch, n = run.x.shape
        labels = torch.randint(self.labels, (batch, n), generator=run.generator, device=run.q.device)
        angles = torch.arange(self.labels, device=run.q.device, dtype=torch.float64) * (2 * math.pi / self.labels)
        levels, pin = (1 + angles.cos()) * .5, angles.sin().square()
        levels[0], levels[self.labels // 2] = 1., 0.
        # Make mathematical equator ties exact rather than trig-roundoff biased.
        if self.labels % 4 == 0:
            levels[self.labels // 4] = levels[3 * self.labels // 4] = .5
        values = levels[labels]
        field = target.field(values)
        _, neighbours = _ring(n, run, False)
        for step in range(1, self.max_steps + 1):
            poll()
            t = step / self.max_steps
            tau, kappa, temperature = self.driver_strength * (1 - t) ** 2, t * t, .5 * (.01 / .5) ** t
            sites = torch.randint(n, (n,), generator=run.generator, device=run.q.device).cpu().tolist()
            for i in sites:
                poll()
                old = labels[:, i].clone()
                adjacent = torch.where(_random(old.shape, run) < .5, -1, 1)
                shift = torch.where(_random(old.shape, run) < self.antipodal_probability,
                                    self.labels // 2, adjacent)
                new = (old + shift) % self.labels
                delta = levels[new] - levels[old]
                cost = delta * field[:, i] + kappa * (pin[new] - pin[old])
                for j, weight, orientation in neighbours[i]:
                    cost += 2 * weight * tau * (
                        _zn_conditional(new, labels[:, j], orientation, self.labels, self.chirality) -
                        _zn_conditional(old, labels[:, j], orientation, self.labels, self.chirality))
                accepted = _metropolis(cost, temperature, run)
                labels[:, i] = torch.where(accepted, new, old)
                change = delta * accepted
                values[:, i].add_(change)
                target.update(field, i, change)
            if step % 100 == 0:
                field = target.field(values)
            self._record(run, record, values, target, step)
        self._polish(run, target)
        return self.max_steps


def _polyval(coefficients, values):
    out = torch.zeros_like(values)
    for index in range(coefficients.shape[-1] - 1, -1, -1):
        out = out * values + coefficients[:, index, None]
    return out


def _bounded_roots(coefficients):
    """All real polynomial roots in [-1,1], by derivative partitions.

    Fixed degree <=4; FP64 bisection includes stationary boundaries for repeated
    roots. Invalid slots are NaN. Zero leading coefficients need no division.
    """
    degree = coefficients.shape[-1] - 1
    scale = coefficients.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    coefficients = coefficients / scale
    if degree == 1:
        root = -coefficients[:, :1] / torch.where(coefficients[:, 1:] != 0,
                                                 coefficients[:, 1:], 1.)
        valid = (coefficients[:, 1:] != 0) & (root.abs() <= 1)
        return torch.where(valid, root, float('nan'))
    derivative = coefficients[:, 1:] * torch.arange(1, degree + 1, device=coefficients.device)
    critical = _bounded_roots(derivative)
    critical = torch.nan_to_num(critical, nan=1.).sort(-1).values
    endpoints = torch.cat((-torch.ones_like(critical[:, :1]), critical,
                           torch.ones_like(critical[:, :1])), -1)
    low, high = endpoints[:, :-1].clone(), endpoints[:, 1:].clone()
    f_low, f_high = _polyval(coefficients, low), _polyval(coefficients, high)
    crosses = (f_low * f_high < 0) & (high > low)
    for _ in range(40):
        middle = (low + high) * .5
        f_middle = _polyval(coefficients, middle)
        same = (f_low * f_middle > 0)
        low = torch.where(same, middle, low)
        f_low = torch.where(same, f_middle, f_low)
        high = torch.where(same, high, middle)
    roots = (low + high) * .5
    candidates = torch.cat((roots, endpoints), -1)
    valid = torch.cat((crosses, _polyval(coefficients, endpoints).abs() < 1e-11), -1)
    valid &= _polyval(coefficients, candidates).abs() < 1e-9
    candidates = torch.where(valid, candidates, float('inf')).sort(-1).values
    unique = torch.ones_like(candidates, dtype=torch.bool)
    unique[:, 1:] = candidates[:, 1:] - candidates[:, :-1] > 1e-8
    candidates = torch.where(unique, candidates, float('inf')).sort(-1).values[:, :degree]
    return torch.where(torch.isfinite(candidates), candidates, float('nan'))


def _xy_minimum(a, b, kappa, old):
    """Exact conditional minimum via a convex scalar problem.

    At fixed u=|sin(theta)| the optimal signs are sign(A), sign(B), so
    f(u)=-|A|sqrt(1-u^2)-|B|u+kappa*u^2 is convex on [0,1]. Its monotone
    derivative has one root. This is equivalent to evaluating all quartic
    stationary points, with fewer GPU operations and no chart singularities.
    """
    if kappa == 0:
        return torch.where((a != 0) | (b != 0), torch.atan2(b, a), old)
    return _xy_positive_pinning_candidate(a, b, kappa, old)


def _xy_positive_pinning_candidate(a, b, kappa, old):
    """Tensor-only positive-pinning root and energy/tie safeguards."""
    aa, bb = a.abs(), b.abs()
    low, high = torch.zeros_like(a), (bb / (2 * kappa)).clamp_max(1)
    for _ in range(48):
        middle = (low + high) * .5
        derivative = aa * middle - (bb - 2 * kappa * middle) * (1 - middle.square()).clamp_min(0).sqrt()
        low = torch.where(derivative < 0, middle, low)
        high = torch.where(derivative < 0, high, middle)
    u = torch.where(aa == 0, (bb / (2 * kappa)).clamp_max(1), (low + high) * .5)
    sine = torch.where(b >= 0, u, -u)
    sign_cosine = torch.where(a == 0, torch.where(old.cos() >= 0, 1., -1.), torch.where(a > 0, 1., -1.))
    cosine = sign_cosine * (1 - u.square()).clamp_min(0).sqrt()
    selected = torch.atan2(sine, cosine)
    old_cost = -a * old.cos() - b * old.sin() + kappa * old.sin().square()
    new_cost = -a * cosine - b * sine + kappa * sine.square()
    result = torch.where(new_cost < old_cost - 1e-13 * (1 + old_cost.abs()), selected, old)
    # This shortcut is exact even for a degenerate quartic.
    pole = torch.where(a >= 0, 0., math.pi)
    pole = torch.where((a == 0) & (old.sin().abs() < 1e-14), old, pole)
    return torch.where(b == 0, pole, result)


def _xy_convex_candidate(a, b, kappa, old):
    """Compile-compatible equivalent, with scalar *tensor* continuation control.

    Keeping kappa on the device avoids specializing a new graph for each sweep.
    The safe denominator also covers the exact zero-pinning harmonic branch
    without a host scalar read or a data-dependent Python branch.
    """
    safe_pinning = torch.where(kappa > 0, kappa, torch.ones_like(kappa))
    pinned = _xy_positive_pinning_candidate(a, b, safe_pinning, old)
    harmonic = torch.where((a != 0) | (b != 0), torch.atan2(b, a), old)
    return torch.where(kappa == 0, harmonic, pinned)


@lru_cache(maxsize=1)
def _compiled_xy_coordinates():
    """Lazy shared code cache; never stores mutable trajectory tensors.

    This optional CUDA path uses PyTorch's Inductor/Triton stack. Compilation
    and its transient compiler storage belong to end-to-end setup cost; the
    ordinary CPU/eager path has no compiler or Triton requirement.
    """
    compile_function = getattr(torch, 'compile', None)
    if not callable(compile_function):
        raise RuntimeError('compile_coordinates=True requires torch.compile support')
    return compile_function(_xy_convex_candidate, fullgraph=True, dynamic=True)


@dataclass(frozen=True)
class XYCoordinate(_DiscreteSolver):
    """Exact angular minima, thermal perturbations and temperature exchanges.

    compile_coordinates optionally fuses the repeated scalar root on CUDA;
    its first-call compilation is charged to the solve, and CPU stays eager.
    """
    driver_strength: float = .25
    minimization_probability: float = .75
    exchange_interval: int = 10
    compile_coordinates: bool = False

    def __post_init__(self):
        super().__post_init__()
        _integer('exchange_interval', self.exchange_interval, 0)
        if not isinstance(self.compile_coordinates, bool):
            raise ValueError('compile_coordinates must be boolean')
        if not math.isfinite(self.driver_strength) or self.driver_strength < 0:
            raise ValueError('driver_strength must be finite and nonnegative')
        if not math.isfinite(self.minimization_probability) or not 0 <= self.minimization_probability <= 1:
            raise ValueError('minimization_probability must be in [0,1]')

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective, normalize=self.normalize)
        if target.constant:
            return 0
        batch, n = run.x.shape
        angles = _random((batch, n), run) * (2 * math.pi)
        cosine, sine = angles.cos(), angles.sin()
        values = (cosine + 1) * .5
        field = target.field(values)
        order, neighbours = _ring(n, run)
        ranks = torch.linspace(0, 1, batch, device=run.q.device, dtype=torch.float64)
        compiled_coordinates = (_compiled_xy_coordinates()
                                if self.compile_coordinates and run.q.device.type == 'cuda' else None)
        for step in range(1, self.max_steps + 1):
            poll()
            t = step / self.max_steps
            tau, kappa = self.driver_strength * (1 - t) ** 2, 2 * t * t
            kappa_control = values.new_tensor(kappa) if compiled_coordinates is not None else None
            tmin, tmax = .1 * (.001 / .1) ** t, .05 ** t
            temperatures = tmin * (tmax / tmin) ** ranks
            for i in torch.randperm(n, generator=run.generator, device=run.q.device).cpu().tolist():
                poll()
                a = -.5 * field[:, i]
                b = torch.zeros_like(a)
                for j, weight, _ in neighbours[i]:
                    a = a + tau * weight * cosine[:, j]
                    b = b + tau * weight * sine[:, j]
                old = angles[:, i]
                optimum = (_xy_minimum(a, b, kappa, old) if compiled_coordinates is None else
                           compiled_coordinates(a, b, kappa_control, old.contiguous()))
                proposal = old + (_random(old.shape, run) - .5) * math.pi
                delta = (-a * (proposal.cos() - cosine[:, i]) - b * (proposal.sin() - sine[:, i])
                         + kappa * (proposal.sin().square() - sine[:, i].square()))
                proposal = torch.where(_metropolis(delta, temperatures, run), proposal, old)
                new = torch.where(_random(old.shape, run) < self.minimization_probability, optimum, proposal)
                new_cosine, new_sine = new.cos(), new.sin()
                delta_p = .5 * (new_cosine - cosine[:, i])
                target.update(field, i, delta_p)
                values[:, i].add_(delta_p)
                angles[:, i], cosine[:, i], sine[:, i] = new, new_cosine, new_sine
            if self.exchange_interval and step % self.exchange_interval == 0 and batch > 1:
                energy = target.energy(values) + kappa * sine.square().sum(-1)
                if n == 2:
                    energy -= tau * (cosine[:, 0] * cosine[:, 1] + sine[:, 0] * sine[:, 1])
                elif n > 2:
                    c, s = cosine[:, order], sine[:, order]
                    energy -= .5 * tau * (c * c.roll(-1, 1) + s * s.roll(-1, 1)).sum(-1)
                parity = (step // self.exchange_interval) % 2
                for r in range(parity, batch - 1, 2):
                    log_accept = (1 / temperatures[r] - 1 / temperatures[r + 1]) * (energy[r] - energy[r + 1])
                    accept = _random((), run).log() < log_accept
                    # Device-side conditional row exchange, with no mutable aliases.
                    for array in (angles, cosine, sine, values, field):
                        left, right = array[r].clone(), array[r + 1].clone()
                        array[r] = torch.where(accept, right, left)
                        array[r + 1] = torch.where(accept, left, right)
            field = target.field(values)
            self._record(run, record, values, target, step)
        self._polish(run, target)
        return self.max_steps


def _segment_delta(state, field, bit, mask, beta, coupling):
    """Worldline action difference; boundary correction includes wrap bonds."""
    spins = state[:, :, bit] * 2 - 1
    target_delta = -(spins * field[:, :, bit] * mask).sum(-1) * beta / state.shape[1]
    boundary = mask != mask.roll(-1, -1)
    temporal = 2 * coupling * (spins * spins.roll(-1, -1) * boundary).sum(-1)
    return target_delta + temporal


@dataclass(frozen=True)
class QuantumDimerWorldline(_DiscreteSolver):
    """Periodic Trotter worldlines with single, partial and full-ring flips."""
    slices: int = 32
    inverse_temperature: float = 10.
    initial_transverse_field: float = 2.
    final_transverse_field: float = .01

    def __post_init__(self):
        super().__post_init__()
        _integer('slices', self.slices, 3)
        for name in ('inverse_temperature', 'initial_transverse_field', 'final_transverse_field'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')

    def _population_workspace(self, n, batch, width):
        return super()._population_workspace(n, batch, width) + 96 * batch * n * self.slices

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        target = _Target(run.objective, normalize=self.normalize)
        if target.constant:
            return 0
        batch, n = run.x.shape
        p = self.slices
        state = (_random((batch, p, n), run) < .5).double()
        state[:batch // 2] = state[:batch // 2, :1].expand(-1, p, -1)
        field = target.field(state.reshape(batch * p, n)).reshape(batch, p, n)
        rows = torch.arange(batch, device=run.q.device)
        time = torch.arange(p, device=run.q.device)
        for step in range(1, self.max_steps + 1):
            poll()
            t = step / self.max_steps
            gamma = self.initial_transverse_field * (self.final_transverse_field / self.initial_transverse_field) ** t
            coupling = -.5 * math.log(math.tanh(self.inverse_temperature * gamma / p))
            visits = 0
            draws = _random((n * p,), run)
            lengths = torch.randint(2, p, (n * p,), generator=run.generator, device=run.q.device)
            lengths = torch.where(draws < .5, 1, torch.where(draws < .75, p, lengths))
            bits = torch.randint(n, (n * p,), generator=run.generator, device=run.q.device)
            plan = torch.stack((lengths, bits), -1).cpu().tolist()
            proposal_index = 0
            # One common, state-independent segment length keeps visit accounting
            # identical across replicas; starts and Metropolis draws are independent.
            while visits < n * p:
                poll()
                length, bit = plan[proposal_index]
                proposal_index += 1
                starts = torch.randint(p, (batch, 1), generator=run.generator, device=run.q.device)
                mask = (time[None] - starts).remainder(p) < length
                delta = _segment_delta(state, field, bit, mask, self.inverse_temperature, coupling)
                accepted = _metropolis(delta, 1., run)
                change = (1 - 2 * state[:, :, bit]) * (mask & accepted[:, None])
                state[:, :, bit].add_(change)
                flat = field.reshape(batch * p, n)
                target.update(flat, bit, change.reshape(-1))
                visits += length
            # Each slice is a feasible original solution. Rank all slices with
            # the original source scorer, not their temporal action.
            source = run.objective.score(run.to_assignments(state.reshape(batch * p, n) * 2 - 1)).reshape(batch, p)
            best_slice = source.argmin(-1)
            chosen = state[rows, best_slice]
            _publish_occupation(run, chosen, step)
            majority = _decode(state.mean(1), run)
            run.retain(run.to_assignments(majority * 2 - 1))
            if step % self.readout_interval == 0 or step == self.max_steps:
                rounded = _round(target, majority)
                run.retain(run.to_assignments(rounded * 2 - 1))
            record(step)
            if step % 100 == 0:
                field = target.field(state.reshape(batch * p, n)).reshape(batch, p, n)
        self._polish(run, target)
        return self.max_steps


HAMILTONIAN_DISCRETE_SOLVERS = {
    'lib_tj_vacancy': TJVacancy,
    'lib_potts': PottsHeatBath,
    'lib_zn_chiral': ZNChiral,
    'lib_xy_coordinate': XYCoordinate,
    'lib_quantum_dimer': QuantumDimerWorldline,
}
