"""Gaussian-state and spectral-block QUBO research solvers.

The two covariance methods retain Wick correlations and exact orthogonal
coordinate rotations. The Hubbard method retains exchange in a real Slater
projector and retracts occupied orbitals with thin QR. Their quadratic state
and cubic conditional readout make them small/medium-instance comparators.
BdG modes only select bounded blocks; exact source scores accept every move.
"""

from dataclasses import dataclass
import math

import torch

from . import observation
from ._dynamics import positive
from ._population import PopulationSolver, _mm
from .solvers import MemoryEstimate, _integer


def _edges(matrix):
    """Upper-triangle coefficients without densifying a CSR operator."""
    if matrix.layout == torch.strided:
        indices = matrix.nonzero().T
        keep = indices[0] < indices[1]
        i, j = indices[:, keep]
        return i, j, matrix[i, j]
    rows = torch.repeat_interleave(torch.arange(matrix.shape[0], device=matrix.device),
                                  matrix.crow_indices()[1:] - matrix.crow_indices()[:-1],
                                  output_size=matrix.values().numel())
    keep = rows < matrix.col_indices()
    return rows[keep], matrix.col_indices()[keep], matrix.values()[keep]


def _publish_bits(run, bits, step):
    run.publish(bits.to(run.q.dtype).mul(2).sub(1), step)


def _polish(run, bits, proposals):
    """Sequential moves per replica, verified with the authoritative scorer."""
    x = bits.to(run.q.dtype)
    samples = x.mul(2).sub(1) if run.spin else x
    energy = run.score(samples)
    rows = torch.arange(x.shape[0], device=x.device)
    for _ in range(proposals):
        observation.poll()
        gain = (1 - 2 * x) * (2 * run.objective.gradient(2 * x - 1))
        cost, index = gain.min(-1)
        trial = x.clone()
        trial[rows, index] = 1 - trial[rows, index]
        candidate = trial.mul(2).sub(1) if run.spin else trial
        trial_energy = run.score(candidate)
        accept = (cost < 0) & (trial_energy < energy)
        x = torch.where(accept[:, None], trial, x)
        energy = torch.where(accept, trial_energy, energy)
        if not bool(accept.any()):
            break
    return x > .5


def _givens(covariance, u, v, angle):
    """O Gamma O^T in O(n) work, reading both old rows before writes."""
    first, second = covariance[:, u, :].clone(), covariance[:, v, :].clone()
    invariant = first[:, v].clone()
    c, s = angle.cos()[:, None], angle.sin()[:, None]
    first, second = c * first + s * second, -s * first + c * second
    first[:, u], first[:, v] = 0, invariant
    second[:, u], second[:, v] = -invariant, 0
    covariance[:, u, :], covariance[:, v, :] = first, second
    covariance[:, :, u], covariance[:, :, v] = -first, -second
    return covariance


def _restore_plane(covariance, u, v, first, second):
    covariance[:, u, :], covariance[:, v, :] = first, second
    covariance[:, :, u], covariance[:, :, v] = -first, -second


def _fock_covariance(bits, dtype):
    count, n = bits.shape
    covariance = torch.zeros((count, 2 * n, 2 * n), device=bits.device, dtype=dtype)
    a = 2 * torch.arange(n, device=bits.device)
    signs = bits.to(dtype).mul(2).sub(1)
    covariance[:, a, a + 1] = signs
    covariance[:, a + 1, a] = -signs
    return covariance


def _random_planes(run, count, size):
    # One small host transfer supplies control indices; state stays on device.
    planes = torch.randint(size, (count, 2), generator=run.generator, device=run.q.device)
    planes[:, 1] = (planes[:, 0] + 1 + planes[:, 1] % (size - 1)) % size
    return planes.cpu().tolist()


def _sparse_driver(run, n, arity, multiplier=4):
    size, count = 2 * n, min(multiplier * n, math.comb(2 * n, arity))
    if count == 0:
        return (torch.empty((0, arity), device=run.q.device, dtype=torch.long),
                torch.empty(0, device=run.q.device, dtype=run.q.dtype))
    if size <= 8:
        indices = torch.combinations(torch.arange(size, device=run.q.device), r=arity)
    else:
        indices = torch.randint(size, (4 * count, arity), generator=run.generator,
                                device=run.q.device).sort(-1).values
        indices = indices[(indices[:, 1:] != indices[:, :-1]).all(-1)].unique(dim=0)
    order = torch.randperm(indices.shape[0], generator=run.generator, device=run.q.device)
    indices = indices[order[:count]]
    weights = run.normal((indices.shape[0],))
    incident = torch.zeros(size, dtype=weights.dtype, device=weights.device)
    incident.index_add_(0, indices.flatten(), weights.abs().repeat_interleave(arity))
    weights.div_(incident.max().clamp_min(torch.finfo(weights.dtype).tiny))
    return indices, weights


class _WickFunctional:
    """Sparse bilinear/quartic polynomials with Majorana-to-term incidence."""

    def __init__(self, objective, driver_pairs, driver_pair_weights,
                 driver_quartets=None, driver_quartet_weights=None):
        n, device = objective.n, objective.device
        i, j, weight = _edges(objective.matrix)
        a = 2 * torch.arange(n, device=device)
        self.pairs = torch.cat((torch.stack((a, a + 1), -1), driver_pairs))
        self.pair_weights = torch.cat((objective.field, driver_pair_weights))
        self.pair_driver = torch.arange(self.pairs.shape[0], device=device) >= n
        self.quartets = torch.stack((2 * i, 2 * i + 1, 2 * j, 2 * j + 1), -1)
        self.quartet_weights = -weight
        target_count = weight.numel()
        if driver_quartets is not None:
            self.quartets = torch.cat((self.quartets, driver_quartets))
            self.quartet_weights = torch.cat((self.quartet_weights, driver_quartet_weights))
        self.quartet_driver = torch.arange(self.quartets.shape[0], device=device) >= target_count
        self.incidence = []
        for indices in (self.pairs, self.quartets):
            arity = indices.shape[1]
            order = indices.flatten().argsort()
            term = torch.arange(indices.shape[0], device=device).repeat_interleave(arity)[order]
            counts = torch.bincount(indices.flatten(), minlength=2 * n)
            pointers = torch.cat((counts.new_zeros(1), counts.cumsum(0))).cpu().tolist()
            self.incidence.append((term, pointers))

    def affected(self, u, v):
        return tuple(torch.cat((terms[pointer[u]:pointer[u + 1]],
                                terms[pointer[v]:pointer[v + 1]])).unique()
                     for terms, pointer in self.incidence)

    def energy(self, covariance, driver=0., affected=None):
        result = covariance.new_zeros(covariance.shape[0])
        for which, (indices, weights, driven) in enumerate((
                (self.pairs, self.pair_weights, self.pair_driver),
                (self.quartets, self.quartet_weights, self.quartet_driver))):
            if affected is not None:
                selected = affected[which]
                indices, weights, driven = indices[selected], weights[selected], driven[selected]
            chunk = max(1, 65_536 // covariance.shape[0])
            for start in range(0, indices.shape[0], chunk):
                index = indices[start:start + chunk]
                coefficient = weights[start:start + chunk].to(covariance.dtype)
                coefficient = coefficient * torch.where(driven[start:start + chunk], driver, 1.)
                if which == 0:
                    value = covariance[:, index[:, 0], index[:, 1]]
                else:
                    a, b, c, d = index.unbind(-1)
                    value = (-covariance[:, a, b] * covariance[:, c, d]
                             + covariance[:, a, c] * covariance[:, b, d]
                             - covariance[:, a, d] * covariance[:, b, c])
                result.add_((value * coefficient).sum(-1))
        return result

    def gradient(self, covariance, driver=0.):
        """Frobenius skew gradient: upper independent derivative divided by 2."""
        gradient = torch.zeros_like(covariance)
        batch = covariance.shape[0]
        linear = self.pair_weights.to(covariance.dtype) * torch.where(self.pair_driver, driver, 1.)
        flat = gradient.flatten(1)
        size = covariance.shape[-1]
        def add(u, v, values):
            index = (u * size + v).expand(batch, -1)
            flat.scatter_add_(1, index, values.expand(batch, -1) * .5)
        add(self.pairs[:, 0], self.pairs[:, 1], linear)
        chunk = max(1, 65_536 // batch)
        for start in range(0, self.quartets.shape[0], chunk):
            a, b, c, d = self.quartets[start:start + chunk].unbind(-1)
            w = self.quartet_weights[start:start + chunk].to(covariance.dtype)
            w = w * torch.where(self.quartet_driver[start:start + chunk], driver, 1.)
            for u, v, value in ((a, b, -covariance[:, c, d]), (c, d, -covariance[:, a, b]),
                                (a, c, covariance[:, b, d]), (b, d, covariance[:, a, c]),
                                (a, d, -covariance[:, b, c]), (b, c, -covariance[:, a, d])):
                add(u, v, value * w)
        return gradient - gradient.transpose(-1, -2)


def _condition_covariance(covariance, a, outcome):
    """Exact Gaussian occupation measurement, computed in float64 at readout."""
    d = a + 1
    probability_twice = 1 + outcome * covariance[:, a, d]
    denominator = torch.where(probability_twice > 0, probability_twice, 1.)
    update = (covariance[:, :, a, None] * covariance[:, None, d, :]
              - covariance[:, :, d, None] * covariance[:, None, a, :])
    conditioned = covariance + (outcome / denominator)[:, None, None] * update
    conditioned[:, a, :], conditioned[:, d, :] = 0, 0
    conditioned[:, :, a], conditioned[:, :, d] = 0, 0
    conditioned[:, a, d], conditioned[:, d, a] = outcome, -outcome
    return conditioned, probability_twice > 0


def _covariance_readout(covariance, functional):
    state = covariance.to(torch.float64).clone()
    bits = torch.empty((state.shape[0], state.shape[-1] // 2), device=state.device,
                       dtype=torch.bool)
    for i in range(bits.shape[1]):
        observation.poll()
        positive = state.new_ones(state.shape[0])
        plus, allowed_plus = _condition_covariance(state, 2 * i, positive)
        minus, allowed_minus = _condition_covariance(state, 2 * i, -positive)
        choose = allowed_plus & (~allowed_minus | (functional.energy(plus) <= functional.energy(minus)))
        state = torch.where(choose[:, None, None], plus, minus)
        bits[:, i] = choose
    return bits


class _CovarianceSolver(PopulationSolver):
    def _population_workspace(self, n, batch, width):
        # State, trial/readout branches in FP64, incidence and coefficient
        # construction, bounded Wick gathers, and an allocator/library margin.
        return (batch * (4 * width + 8 * 8) * (2 * n)**2
                + 192 * n * n + 256 * n + 16 * 65_536)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        n, count = run.objective.n, run.x.shape[0]
        functional = self._functional(run)
        covariance = _fock_covariance(run.initialbits, run.q.dtype)
        planes = _random_planes(run, n + self.max_steps, 2 * n)
        for u, v in planes[:n]:
            angle = (torch.rand(count, device=run.q.device, generator=run.generator,
                                dtype=run.q.dtype) - .5) * .2
            _givens(covariance, u, v, angle)
        interval = run.history_interval or self.check_interval
        # Reflection is an explicit restart into the opposite disconnected
        # parity component, including for a single replica or batch_size=1.
        reflection_step = max(1, self.max_steps // 2)
        for step, (u, v) in enumerate(planes[n:], 1):
            observation.poll()
            progress = step / max(1, self.max_steps)
            driver = self.driver_strength * (1 - progress)**2
            affected = functional.affected(u, v)
            old = functional.energy(covariance, driver, affected)
            first, second = covariance[:, u, :].clone(), covariance[:, v, :].clone()
            _givens(covariance, u, v, covariance.new_full((count,), math.pi))
            opposite = functional.energy(covariance, driver, affected)
            _restore_plane(covariance, u, v, first, second)
            _givens(covariance, u, v, covariance.new_full((count,), math.pi / 2))
            quarter = functional.energy(covariance, driver, affected)
            a, b = (old + opposite) / 2, (old - opposite) / 2
            c = quarter - a
            angle = self._angle(b, c)
            for _ in range(self.backtracking_steps):
                _restore_plane(covariance, u, v, first, second)
                _givens(covariance, u, v, angle)
                value = functional.energy(covariance, driver, affected)
                accept = torch.isfinite(value) & (value <= old)
                if bool(accept.all()):
                    break
                angle = torch.where(accept, angle, angle * .5)
            _restore_plane(covariance, u, v,
                           torch.where(accept[:, None], covariance[:, u, :], first),
                           torch.where(accept[:, None], covariance[:, v, :], second))
            if step % interval == 0 or step == self.max_steps or step == reflection_step:
                bits = self._readout(covariance, functional, run)
                bits = _polish(run, bits, self.polish_steps)
                _publish_bits(run, bits, step)
                record(step)
            if step == reflection_step:
                covariance[:, 0, :].neg_()
                covariance[:, :, 0].neg_()
                # The opposite component is also decoded when a one-step
                # budget leaves no subsequent coordinate update.
                bits = self._readout(covariance, functional, run)
                _publish_bits(run, _polish(run, bits, self.polish_steps), step)
                record(step)
        if self.max_steps == 0:
            _publish_bits(run, self._readout(covariance, functional, run), 0)
            record(0)
        return self.max_steps

    def _validate(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('check_interval', self.check_interval, 1)
        _integer('backtracking_steps', self.backtracking_steps, 1)
        if self.backtracking_steps > 20:
            raise ValueError('backtracking_steps must be <= 20')
        _integer('polish_steps', self.polish_steps, 0)
        positive('driver_strength', self.driver_strength, zero=True)
        _validate_common(self)


def _validate_common(solver):
    if not isinstance(solver.normalize, bool):
        raise ValueError('normalize must be boolean')
    if isinstance(solver.sparse_threshold, bool) or not 0 <= solver.sparse_threshold <= 1:
        raise ValueError('sparse_threshold must be in [0,1]')
    if solver.dynamics_dtype not in (None, 'float32', 'float64', torch.float32, torch.float64):
        raise ValueError('dynamics_dtype must be float32 or float64')


@dataclass(frozen=True)
class SparseSYK(_CovarianceSolver):
    """Sparse SYK Wick objective, exact Givens-plane minimization and measurement."""
    max_steps: int = 128
    driver_strength: float = 2.
    check_interval: int = 16
    backtracking_steps: int = 20
    polish_steps: int = 16
    dynamics_dtype: object = None
    normalize: bool = True
    sparse_threshold: float = .15

    def __post_init__(self):
        self._validate()

    def _functional(self, run):
        pairs, weights = _sparse_driver(run, run.objective.n, 2)
        quartets, quartic_weights = _sparse_driver(run, run.objective.n, 4)
        return _WickFunctional(run.objective, pairs, weights, quartets, quartic_weights)

    @staticmethod
    def _angle(b, c):
        return torch.where(b.square() + c.square() > 0, torch.atan2(-c, -b), torch.zeros_like(b))

    @staticmethod
    def _readout(covariance, functional, run):
        return _covariance_readout(covariance, functional)


@dataclass(frozen=True)
class KitaevChain(_CovarianceSolver):
    """Interacting Wick target with open-chain hopping/pairing and Givens descent."""
    max_steps: int = 128
    driver_strength: float = 1.
    hopping: float = 1.
    pairing: float = .7
    time_step: float = .1
    check_interval: int = 16
    backtracking_steps: int = 20
    polish_steps: int = 16
    dynamics_dtype: object = None
    normalize: bool = True
    sparse_threshold: float = .15

    def __post_init__(self):
        self._validate()
        positive('hopping', self.hopping, zero=True)
        positive('pairing', self.pairing, zero=True)
        positive('time_step', self.time_step)

    def _functional(self, run):
        n = run.objective.n
        order = torch.randperm(n, device=run.q.device, generator=run.generator)
        i, j = order[:-1], order[1:]
        pairs = torch.cat((torch.stack((2 * i, 2 * j + 1), -1),
                           torch.stack((2 * i + 1, 2 * j), -1)))
        weights = torch.cat((run.objective.field.new_full((n - 1,), -(self.hopping + self.pairing) / 2),
                             run.objective.field.new_full((n - 1,), (self.hopping - self.pairing) / 2)))
        weights = torch.where(pairs[:, 0] < pairs[:, 1], weights, -weights)
        return _WickFunctional(run.objective, pairs.sort(-1).values, weights)

    def _angle(self, b, c):
        # dF/dtheta(0)=c=-2 Omega_uv; theta=dt Omega_uv.
        return -self.time_step * c / 2

    @staticmethod
    def _readout(covariance, functional, run):
        m = covariance[:, ::2, 1::2].diagonal(dim1=-2, dim2=-1)
        deterministic = m >= 0
        sampled = torch.rand(m.shape, dtype=m.dtype, device=m.device,
                             generator=run.generator) < (1 + m) / 2
        first = deterministic.to(run.q.dtype).mul(2).sub(1)
        second = sampled.to(run.q.dtype).mul(2).sub(1)
        a, b = run.to_assignments(first), run.to_assignments(second)
        return torch.where((run.score(a) <= run.score(b))[:, None], deterministic, sampled)


def _hubbard_energy_gradient(projector, linear, heads, tails, weights, driver,
                              penalty, eta, *, gradient=True):
    """Real Slater expectation, including every target/cell exchange contraction."""
    density = projector.diagonal(dim1=-2, dim2=-1)
    rho0, rho1 = density[:, ::2], density[:, 1::2]
    n, batch = linear.numel(), projector.shape[0]
    cell = torch.arange(n, device=projector.device)
    a, b = 2 * cell, 2 * cell + 1
    coherence = projector[:, a, b]
    energy = ((rho1 * linear).sum(-1)
              + penalty * (1 - rho0 - rho1 + 2 * rho0 * rho1 - 2 * coherence.square()).sum(-1)
              + eta * (projector * driver).sum((-1, -2)))
    out = eta * driver.expand_as(projector).clone() if gradient else None
    if gradient:
        out[:, a, a] += penalty * (-1 + 2 * rho1)
        out[:, b, b] += linear + penalty * (-1 + 2 * rho0)
        out[:, a, b] -= 2 * penalty * coherence
        out[:, b, a] -= 2 * penalty * coherence
    chunk = max(1, 65_536 // batch)
    for start in range(0, weights.numel(), chunk):
        i, j, w = heads[start:start + chunk], tails[start:start + chunk], weights[start:start + chunk]
        off = projector[:, 2 * i + 1, 2 * j + 1]
        energy += ((rho1[:, i] * rho1[:, j] - off.square()) * w).sum(-1)
        if gradient:
            out.diagonal(dim1=-2, dim2=-1)[:, 1::2].scatter_add_(1, i.expand(batch, -1), rho1[:, j] * w)
            out.diagonal(dim1=-2, dim2=-1)[:, 1::2].scatter_add_(1, j.expand(batch, -1), rho1[:, i] * w)
            out[:, 2 * i + 1, 2 * j + 1] -= off * w
            out[:, 2 * j + 1, 2 * i + 1] -= off * w
    return energy, out


def _condition_projector(projector, index, outcome):
    probability = projector[:, index, index]
    denominator = torch.where(outcome, probability, 1 - probability)
    allowed = denominator > 0
    coefficient = torch.where(outcome, -torch.ones_like(probability), torch.ones_like(probability))
    coefficient /= torch.where(allowed, denominator, 1.)
    trial = projector + coefficient[:, None, None] * (projector[:, :, index, None]
                                                    * projector[:, None, index, :])
    trial[:, index, :], trial[:, :, index] = 0, 0
    trial[:, index, index] = outcome.to(trial.dtype)
    return trial, allowed


def _hubbard_readout(projector, linear, heads, tails, weights):
    state = projector.to(torch.float64).clone()
    empty_driver = state.new_zeros(state.shape[-2:])
    bits = torch.empty((state.shape[0], linear.numel()), dtype=torch.bool, device=state.device)
    for i in range(linear.numel()):
        observation.poll()
        ones = torch.ones(state.shape[0], device=state.device, dtype=torch.bool)
        plus, allowed_plus = _condition_projector(state, 2 * i + 1, ones)
        minus, allowed_minus = _condition_projector(state, 2 * i + 1, ~ones)
        eplus, _ = _hubbard_energy_gradient(plus, linear.double(), heads, tails, weights.double(),
                                           empty_driver, 0., 0., gradient=False)
        eminus, _ = _hubbard_energy_gradient(minus, linear.double(), heads, tails, weights.double(),
                                            empty_driver, 0., 0., gradient=False)
        choose = allowed_plus & (~allowed_minus | (eplus <= eminus))
        bits[:, i], state = choose, torch.where(choose[:, None, None], plus, minus)
    return bits


@dataclass(frozen=True)
class FermiHubbard(PopulationSolver):
    """Penalized real Slater projector, exchange-aware thin-QR gradient descent."""
    max_steps: int = 64
    time_step: float = .05
    penalty_start: float = .1
    penalty_end: float = 2.
    driver_strength: float = 2.
    hopping: float = .25
    intracell_hopping: float = 1.
    check_interval: int = 10
    backtracking_steps: int = 20
    polish_steps: int = 16
    dynamics_dtype: object = None
    normalize: bool = True
    sparse_threshold: float = .15

    def __post_init__(self):
        for name in ('max_steps', 'polish_steps'):
            _integer(name, getattr(self, name), 0)
        for name in ('check_interval', 'backtracking_steps'):
            _integer(name, getattr(self, name), 1)
        if self.backtracking_steps > 20:
            raise ValueError('backtracking_steps must be <= 20')
        for name in ('time_step', 'penalty_start', 'penalty_end'):
            positive(name, getattr(self, name))
        for name in ('driver_strength', 'hopping', 'intracell_hopping'):
            positive(name, getattr(self, name), zero=True)
        _validate_common(self)

    def _population_workspace(self, n, batch, width):
        # V/QR/projection work, P/G and backtracking, FP64 conditional branches,
        # dense driver, edge preparation and a generous QR/library allowance.
        return batch * (64 * width + 192) * n * n + 192 * n * n + 16 * 65_536

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        objective, n = run.objective, run.objective.n
        i, j, coefficient = _edges(objective.matrix)
        weights = 4 * coefficient
        linear = 2 * objective.field - 2 * _mm(objective.matrix, run.objective.field.new_ones((1, n))).squeeze(0)
        count = run.x.shape[0]
        orbitals = run.objective.field.new_zeros((count, 2 * n, n))
        cells = torch.arange(n, device=run.q.device)
        rows = 2 * cells[None, :] + run.initialbits.long()
        orbitals.scatter_(1, rows[:, None, :], 1.)
        # Sparse in the native model; a shared dense matrix is needed by GEMM.
        driver = run.objective.field.new_zeros((2 * n, 2 * n))
        driver[2 * cells, 2 * cells + 1] = -self.intracell_hopping
        driver[2 * cells + 1, 2 * cells] = -self.intracell_hopping
        order = torch.randperm(n, device=run.q.device, generator=run.generator)
        if n > 1:
            for alpha in (0, 1):
                a, b = 2 * order + alpha, 2 * order.roll(1) + alpha
                driver[a, b], driver[b, a] = -self.hopping, -self.hopping
        orbitals = torch.linalg.qr(orbitals + .02 * run.normal(orbitals.shape), mode='reduced').Q
        projector = orbitals @ orbitals.transpose(-1, -2)
        interval = run.history_interval or self.check_interval
        for step in range(1, self.max_steps + 1):
            observation.poll()
            progress = step / max(1, self.max_steps)
            penalty = self.penalty_start + (self.penalty_end - self.penalty_start) * progress
            eta = self.driver_strength * (1 - progress)**2
            old, gradient = _hubbard_energy_gradient(projector, linear, i, j, weights,
                                                     driver, penalty, eta)
            gv = gradient @ orbitals
            tangent = gv - orbitals @ (orbitals.transpose(-1, -2) @ gv)
            dt = run.objective.field.new_full((count, 1, 1), self.time_step)
            for _ in range(self.backtracking_steps):
                trial = torch.linalg.qr(orbitals - dt * tangent, mode='reduced').Q
                trial_projector = trial @ trial.transpose(-1, -2)
                value, _ = _hubbard_energy_gradient(trial_projector, linear, i, j, weights,
                                                     driver, penalty, eta, gradient=False)
                accept = torch.isfinite(value) & (value <= old)
                if bool(accept.all()):
                    break
                dt = torch.where(accept[:, None, None], dt, dt * .5)
            orbitals = torch.where(accept[:, None, None], trial, orbitals)
            projector = torch.where(accept[:, None, None], trial_projector, projector)
            if step % interval == 0 or step == self.max_steps:
                bits = _hubbard_readout(projector, linear, i, j, weights)
                _publish_bits(run, _polish(run, bits, self.polish_steps), step)
                record(step)
        if self.max_steps == 0:
            bits = _hubbard_readout(projector, linear, i, j, weights)
            _publish_bits(run, bits, 0)
            record(0)
        return self.max_steps


def _bdg_proposal(costs, interactions, hopping=.1, pairing=.5):
    a = -hopping * interactions.abs()
    a.diagonal(dim1=-2, dim2=-1).copy_(costs)
    b = pairing * (torch.triu(interactions, diagonal=1) - torch.triu(interactions, diagonal=1).transpose(-1, -2))
    return torch.cat((torch.cat((a, b), -1), torch.cat((-b, -a), -1)), -2)


def _source_density_coefficients(objective):
    """Original FP64 b,W (pair-once), separate from normalized FP32 forces."""
    if objective.source_terms is not None:
        linear, heads, tails, biases = objective.source_terms
        self_loop = heads == tails
        linear = linear.clone().index_add_(0, heads[self_loop], biases[self_loop])
        heads, tails, biases = heads[~self_loop], tails[~self_loop], biases[~self_loop]
        indices = torch.stack((torch.cat((heads, tails)), torch.cat((tails, heads))))
        matrix = torch.sparse_coo_tensor(indices, torch.cat((biases, biases)),
                                        (objective.n, objective.n)).coalesce().to_sparse_csr()
        return linear, matrix
    source = objective.source_matrix
    if source.layout == torch.strided:
        matrix = source * (8. if objective.spin else 2.)
        matrix.diagonal().zero_()
        linear = (2 * objective.source_field - .5 * matrix.sum(-1)
                  if objective.spin else source.diagonal().clone())
    else:
        rows = torch.repeat_interleave(torch.arange(objective.n, device=source.device),
                                      source.crow_indices()[1:] - source.crow_indices()[:-1],
                                      output_size=source.values().numel())
        diagonal = rows == source.col_indices()
        values = source.values() * (8. if objective.spin else 2.)
        values[diagonal] = 0
        matrix = torch.sparse_csr_tensor(source.crow_indices(), source.col_indices(), values,
                                         size=source.shape)
        if objective.spin:
            row_sum = values.new_zeros(objective.n).index_add_(0, rows, values)
            linear = 2 * objective.source_field - .5 * row_sum
        else:
            linear = values.new_zeros(objective.n).index_add_(0, rows[diagonal], source.values()[diagonal])
    return linear, matrix


def _exact_block_minimum(costs, interactions, tile_size=1024):
    """Tiled FP64 mask enumeration; includes zero and retains only row minima."""
    costs, interactions = costs.double(), interactions.double()
    batch, size = costs.shape
    powers = torch.arange(size, device=costs.device, dtype=torch.long)
    best = costs.new_zeros(batch)
    best_mask = costs.new_zeros((batch, size))
    for start in range(0, 1 << size, tile_size):
        observation.poll()
        masks = ((torch.arange(start, min(start + tile_size, 1 << size), device=costs.device)
                  [:, None] >> powers) & 1).to(torch.float64)
        energy = costs @ masks.T + .5 * torch.einsum('ti,bij,tj->bt', masks, interactions, masks)
        value, index = energy.min(-1)
        improve = value < best
        best_mask = torch.where(improve[:, None], masks[index], best_mask)
        best = torch.minimum(best, value)
    return best_mask, best


def _neighbour_strength(heads, tails, weights, seeds, n):
    """All seed neighbours in one bounded-memory pass through shared edges."""
    batch = seeds.shape[0]
    selected = torch.zeros((batch, n), device=weights.device, dtype=torch.bool)
    selected.scatter_(1, seeds, True)
    strength = weights.new_zeros((batch, n))
    chunk = max(1, 65_536 // batch)
    for start in range(0, weights.numel(), chunk):
        i, j, w = heads[start:start + chunk], tails[start:start + chunk], weights[start:start + chunk].abs()
        strength.scatter_add_(1, j.expand(batch, -1), selected[:, i] * w)
        strength.scatter_add_(1, i.expand(batch, -1), selected[:, j] * w)
    return strength


def _negative_pair(costs, sigma, heads, tails, weights):
    batch = costs.shape[0]
    best = costs.new_full((batch,), float('inf'))
    pair = torch.zeros((batch, 2), device=costs.device, dtype=torch.long)
    chunk = max(1, 65_536 // batch)
    for start in range(0, weights.numel(), chunk):
        i, j, w = heads[start:start + chunk], tails[start:start + chunk], weights[start:start + chunk]
        value = costs[:, i] + costs[:, j] + w * sigma[:, i] * sigma[:, j]
        gain, local = value.min(-1)
        improve = gain < best
        pair = torch.where(improve[:, None], torch.stack((i[local], j[local]), -1), pair)
        best = torch.minimum(best, gain)
    return best, pair


@dataclass(frozen=True)
class BogoliubovDeGennes(PopulationSolver):
    """Adaptive bounded BdG support proposals with exact source-verified blocks."""
    max_steps: int = 64
    pool_size: int = 64
    block_size: int = 12
    max_block_size: int = 16
    mode_count: int = 4
    hopping: float = .1
    pairing: float = .5
    rebuild_interval: int = 8
    enumeration_tile: int = 1024
    check_interval: int = 8
    polish_steps: int = 16
    dynamics_dtype: object = None
    normalize: bool = True
    sparse_threshold: float = .15

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('polish_steps', self.polish_steps, 0)
        for name in ('pool_size', 'block_size', 'max_block_size', 'mode_count',
                     'rebuild_interval', 'enumeration_tile', 'check_interval'):
            _integer(name, getattr(self, name), 1)
        if self.pool_size > 128 or self.max_block_size > 16 or self.block_size > self.max_block_size:
            raise ValueError('require pool_size <= 128 and block_size <= max_block_size <= 16')
        positive('hopping', self.hopping, zero=True)
        positive('pairing', self.pairing, zero=True)
        _validate_common(self)

    def _population_workspace(self, n, batch, width):
        pool, block = min(n, self.pool_size), min(n, self.max_block_size)
        return (batch * (128 * width * pool * pool + 128 * block * block
                         + 32 * self.enumeration_tile * block + 256 * n)
                + 16 * self.enumeration_tile * block + 64 * 65_536)

    def estimate_memory(self, problem, **options):
        estimate = super().estimate_memory(problem, **options)
        return MemoryEstimate(estimate.problem_bytes, estimate.output_bytes,
                              estimate.workspace_bytes + 64 * problem.n_variables**2)

    def estimate_compact_memory(self, problem, **options):
        estimate = super().estimate_compact_memory(problem, **options)
        n, m = problem.variableCount, problem.interactionCount
        extra = 128 * (m + n) if 2 * m + n < self.sparse_threshold * n * n else 64 * n * n
        return MemoryEstimate(estimate.problem_bytes, estimate.output_bytes,
                              estimate.workspace_bytes + extra)

    @staticmethod
    def _coefficient_block(matrix, indices):
        if matrix.layout == torch.strided:
            return matrix[indices[:, :, None], indices[:, None, :]]
        # Sparse row lookup in a bounded selected block, without n-by-pool scratch.
        values, crow, columns = matrix.values(), matrix.crow_indices(), matrix.col_indices()
        # CSR linear keys are sorted; searchsorted handles missing entries.
        rows = torch.repeat_interleave(torch.arange(matrix.shape[0], device=matrix.device),
                                      crow[1:] - crow[:-1], output_size=values.numel())
        keys = rows * matrix.shape[0] + columns
        wanted = indices[:, :, None] * matrix.shape[0] + indices[:, None, :]
        if keys.numel() == 0:
            return values.new_zeros(wanted.shape)
        position = torch.searchsorted(keys, wanted).clamp_max(keys.numel() - 1)
        return torch.where(keys[position] == wanted, values[position], 0.)

    def _search(self, run, record):
        if self.max_steps == 0:
            return 0
        n, count = run.objective.n, run.x.shape[0]
        bits = _polish(run, run.initialbits, self.polish_steps)
        # Exact block coefficients retain source precision even when the
        # spectral dynamics and other model states are configured as FP32.
        linear, matrix = _source_density_coefficients(run.objective)
        x = bits.to(torch.float64)
        i, j, weight = _edges(matrix)
        pool_width = min(n, self.pool_size)
        failures = torch.zeros(count, device=run.q.device, dtype=torch.long)
        support = None
        interval = run.history_interval or self.check_interval
        for step in range(1, self.max_steps + 1):
            observation.poll()
            sigma = 1 - 2 * x
            costs = sigma * (_mm(matrix, x) + linear)
            if support is None or (step - 1) % self.rebuild_interval == 0:
                # Half low gains, quarter their strong neighbours, quarter random.
                ranks = torch.rand((count, n), device=x.device, generator=run.generator)
                low = costs.topk(max(1, pool_width // 2), largest=False).indices
                ranks.scatter_(1, low, -3.)
                seed = low[:, :max(1, pool_width // 4)]
                if weight.numel():
                    strength = _neighbour_strength(i, j, weight, seed, n)
                    neighbours = strength.topk(max(1, pool_width // 4)).indices
                    # Preserve strongest-gain priority if categories overlap.
                    ranks.scatter_(1, neighbours, torch.minimum(ranks.gather(1, neighbours), ranks.new_full(neighbours.shape, -2.)))
                pool = ranks.topk(pool_width, largest=False).indices.sort(-1).values
                interactions = self._coefficient_block(matrix, pool)
                interactions *= sigma.gather(1, pool)[:, :, None] * sigma.gather(1, pool)[:, None, :]
                proposal = _bdg_proposal(costs.gather(1, pool), interactions, self.hopping, self.pairing)
                eigenvalues, eigenvectors = torch.linalg.eigh(proposal)
                selected = eigenvalues.abs().topk(min(self.mode_count, 2 * pool_width), largest=False).indices
                modes = eigenvectors.gather(-1, selected[:, None, :].expand(-1, 2 * pool_width, -1))
                # Subspace sum is stable under basis changes in +/- pairs.
                support = (modes[:, :pool_width].square() + modes[:, pool_width:].square()).sum(-1)
                support /= support.sum(-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
                residual = proposal @ modes - modes * eigenvalues.gather(1, selected)[:, None, :]
                reliable = residual.norm(dim=(-1, -2)) <= 1e-6 * proposal.norm(dim=(-1, -2)).clamp_min(1.)
                support = torch.where(reliable[:, None], support, torch.full_like(support, 1 / pool_width))
            enlarged = bool((failures >= 8).any())
            block_width = min(n, self.max_block_size if enlarged else self.block_size)
            if step % 8 in (0, 1, 2):
                seed = torch.randint(n, (count, 1), device=x.device, generator=run.generator)
            else:
                seed = pool.gather(1, torch.multinomial(support, 1, generator=run.generator))
            selection = torch.rand((count, n), dtype=x.dtype, device=x.device,
                                   generator=run.generator) * 1e-3
            selection.scatter_add_(1, pool, support)
            if weight.numel():
                # Grow from strong target neighbours, then spectral support.
                strength = _neighbour_strength(i, j, weight, seed, n)
                neighbour = strength.topk(min(block_width, max(1, block_width // 2))).indices
                selection.scatter_add_(1, neighbour, x.new_full(neighbour.shape, 2.))
            selection.scatter_(1, seed, float('inf'))
            if step % 8 == 0 and weight.numel() and block_width >= 2:
                pair_cost, pair = _negative_pair(costs, sigma, i, j, weight)
                # A preceding seed must not tie the two mandatory guard bits
                # at infinity when a block has only two available positions.
                selection = torch.where((pair_cost < 0)[:, None],
                                        selection.nan_to_num(posinf=4.), selection)
                priorities = torch.where((pair_cost < 0)[:, None], x.new_full(pair.shape, float('inf')),
                                         selection.gather(1, pair))
                selection.scatter_(1, pair, priorities)
            block = selection.topk(block_width).indices
            exact_interaction = self._coefficient_block(matrix, block).double()
            block_sigma = sigma.gather(1, block).double()
            exact_interaction *= block_sigma[:, :, None] * block_sigma[:, None, :]
            mask, _ = _exact_block_minimum(costs.gather(1, block), exact_interaction, self.enumeration_tile)
            trial = x.clone()
            trial.scatter_(1, block, x.gather(1, block) + block_sigma.to(x.dtype) * mask.to(x.dtype))
            old_samples = 2 * x - 1 if run.spin else x
            new_samples = 2 * trial - 1 if run.spin else trial
            old_energy, new_energy = run.score(old_samples), run.score(new_samples)
            accept = new_energy < old_energy
            # Archive every validated improvement before a possible perturbation,
            # independently of checkpoint/history frequency.
            run.retain(new_samples.to(run.best_x.dtype), new_energy)
            x = torch.where(accept[:, None], trial, x)
            failures = torch.where(accept, torch.zeros_like(failures), failures + 1)
            if step % interval == 0 or step == self.max_steps:
                _publish_bits(run, x > .5, step)
                record(step)
            # Explicit perturb/rebuild; best archive lives independently in run.
            restart = failures >= 32
            if bool(restart.any()):
                perturb = torch.rand(x.shape, device=x.device, generator=run.generator) < .05
                x = torch.where(restart[:, None] & perturb, 1 - x, x)
                failures.masked_fill_(restart, 0)
                support = None
        bits = _polish(run, x > .5, self.polish_steps)
        _publish_bits(run, bits, self.max_steps)
        record(self.max_steps)
        return self.max_steps


HAMILTONIAN_GAUSSIAN_SOLVERS = {
    'lib_syk': SparseSYK,
    'lib_fermi_hubbard': FermiHubbard,
    'lib_kitaev_chain': KitaevChain,
    'lib_bdg': BogoliubovDeGennes,
}
