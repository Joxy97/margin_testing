"""Equation-level validation for the four matrix-state/spectral models."""

import itertools

import pytest
import torch

from qubo_solvers import QUBO
from qubo_solvers._population import PopulationObjective, PopulationRun, _mm
from qubo_solvers.hamiltonian_gaussian import (
    HAMILTONIAN_GAUSSIAN_SOLVERS, BogoliubovDeGennes, FermiHubbard,
    KitaevChain, SparseSYK, _WickFunctional, _bdg_proposal,
    _condition_covariance, _condition_projector, _covariance_readout,
    _exact_block_minimum, _fock_covariance, _givens,
    _hubbard_energy_gradient, _hubbard_readout, _source_density_coefficients,
)


def _fixture(dtype=torch.float64, device='cpu'):
    return QUBO(torch.tensor([[-1., 1.5], [1.5, -2.]], dtype=dtype, device=device))


def _functional(problem):
    objective = PopulationObjective.from_problem(problem, precision=torch.float64,
                                                 normalize=False, threshold=0.)
    pairs = torch.empty((0, 2), device=problem.Q.device, dtype=torch.long)
    return _WickFunctional(objective, pairs, problem.Q.new_empty(0)), objective


def _random_covariance(n=3, batch=2):
    generator = torch.Generator().manual_seed(111)
    bits = torch.randint(2, (batch, n), generator=generator).bool()
    covariance = _fock_covariance(bits, torch.float64)
    for i in range(5 * n):
        u, v = i % (2 * n), (i + 1 + i // (2 * n)) % (2 * n)
        if u != v:
            angle = torch.randn(batch, generator=generator, dtype=torch.float64)
            _givens(covariance, u, v, angle)
    return covariance


def test_covariance_fock_endpoints_and_signs():
    problem = _fixture()
    functional, _ = _functional(problem)
    bits = torch.tensor(list(itertools.product((0, 1), repeat=2)), dtype=torch.bool)
    covariance = _fock_covariance(bits, torch.float64)
    # Constant in E(s)=C+f*s+sum K_ij*s_i*s_j.
    constant = problem.Q.diagonal().sum() / 2 + torch.triu(problem.Q, diagonal=1).sum() / 2
    torch.testing.assert_close(functional.energy(covariance) + constant,
                               torch.tensor([0., -2., -1., 0.], dtype=torch.float64))
    torch.testing.assert_close(functional.energy(covariance) + constant, problem.energy(bits))


def test_wick_skew_half_gradient_and_givens_direction():
    torch.manual_seed(23)
    problem = QUBO(torch.randn(3, 3, dtype=torch.float64))
    functional, objective = _functional(problem)
    # Include both kinds of SYK driver to test every Wick derivative.
    pairs = torch.tensor([[0, 3], [1, 4]])
    quartets = torch.tensor([[0, 1, 2, 5], [1, 2, 3, 4]])
    functional = _WickFunctional(objective, pairs, torch.tensor([.2, -.7], dtype=torch.float64),
                                 quartets, torch.tensor([.4, -.5], dtype=torch.float64))
    covariance = _random_covariance()
    gradient = functional.gradient(covariance, .6)
    perturbation = torch.randn_like(covariance)
    perturbation = perturbation - perturbation.transpose(-1, -2)
    eps = 1e-6
    numerical = (functional.energy(covariance + eps * perturbation, .6)
                 - functional.energy(covariance - eps * perturbation, .6)) / (2 * eps)
    torch.testing.assert_close(numerical, (gradient * perturbation).sum((-1, -2)), rtol=2e-8, atol=2e-8)
    u, v = 1, 4
    plus, minus = covariance.clone(), covariance.clone()
    _givens(plus, u, v, covariance.new_full((2,), eps))
    _givens(minus, u, v, covariance.new_full((2,), -eps))
    derivative = (functional.energy(plus, .6) - functional.energy(minus, .6)) / (2 * eps)
    omega = gradient @ covariance - covariance @ gradient
    torch.testing.assert_close(derivative, -2 * omega[:, u, v], rtol=2e-8, atol=2e-8)


def test_givens_preserves_purity_and_pfaffian_parity():
    covariance = _fock_covariance(torch.tensor([[False, True], [True, True]]), torch.float64)
    def pfaffian(matrix):
        return (matrix[:, 0, 1] * matrix[:, 2, 3] - matrix[:, 0, 2] * matrix[:, 1, 3]
                + matrix[:, 0, 3] * matrix[:, 1, 2])
    parity = pfaffian(covariance)
    for u, v in ((0, 2), (1, 3), (0, 3), (0, 1)):
        _givens(covariance, u, v, torch.tensor([.37, -.91], dtype=torch.float64))
    torch.testing.assert_close(covariance + covariance.transpose(-1, -2), torch.zeros_like(covariance))
    torch.testing.assert_close(covariance @ covariance, -torch.eye(4).double().expand(2, -1, -1))
    torch.testing.assert_close(pfaffian(covariance), parity)


def test_syk_exact_plane_harmonic_and_minimum():
    torch.manual_seed(12)
    problem = QUBO(torch.randn(3, 3, dtype=torch.float64))
    functional, _ = _functional(problem)
    covariance = _random_covariance()
    old = functional.energy(covariance)
    def at(theta):
        trial = covariance.clone()
        _givens(trial, 0, 3, covariance.new_full((2,), theta))
        return functional.energy(trial)
    opposite, quarter = at(torch.pi), at(torch.pi / 2)
    a, b = (old + opposite) / 2, (old - opposite) / 2
    c = quarter - a
    theta = .617
    torch.testing.assert_close(at(theta), a + b * torch.cos(torch.tensor(theta))
                               + c * torch.sin(torch.tensor(theta)), rtol=2e-7, atol=2e-7)
    trial = covariance.clone()
    _givens(trial, 0, 3, SparseSYK._angle(b, c))
    assert bool((functional.energy(trial) <= old + 1e-12).all())
    torch.testing.assert_close(functional.energy(trial), a - torch.sqrt(b.square() + c.square()))


def test_covariance_measurement_purity_and_conditional_expectation():
    torch.manual_seed(7)
    problem = QUBO(torch.randn(3, 3, dtype=torch.float64))
    functional, _ = _functional(problem)
    covariance = _random_covariance()
    plus, allowed_plus = _condition_covariance(covariance, 0, covariance.new_ones(2))
    minus, allowed_minus = _condition_covariance(covariance, 0, -covariance.new_ones(2))
    probability = (1 + covariance[:, 0, 1]) / 2
    assert bool(allowed_plus.all() & allowed_minus.all())
    torch.testing.assert_close(probability * functional.energy(plus)
                               + (1 - probability) * functional.energy(minus), functional.energy(covariance))
    for branch in (plus, minus):
        torch.testing.assert_close(branch @ branch, -torch.eye(6).double().expand(2, -1, -1))
    bits = _covariance_readout(covariance, functional)
    final = functional.energy(_fock_covariance(bits, torch.float64))
    assert bool((final <= functional.energy(covariance) + 1e-12).all())


def test_kitaev_driver_coefficients_and_chain_spectrum():
    # n=4, t=Delta=1, mu=0: three independent dimers of coefficient -1.
    n = 4
    problem = QUBO(torch.zeros(n, n, dtype=torch.float64))
    objective = PopulationObjective.from_problem(problem, precision=torch.float64, normalize=False)
    run = PopulationRun(objective, 1, torch.Generator().manual_seed(7))
    functional = KitaevChain(hopping=1., pairing=1.)._functional(run)
    driver_pairs = functional.pairs[n:]
    driver_weights = functional.pair_weights[n:]
    assert driver_pairs.shape[0] == 2 * (n - 1)
    assert int(torch.count_nonzero(driver_weights)) == n - 1
    assert float(driver_weights.abs().sum()) == pytest.approx(3.)


def test_hubbard_energy_gradient_exchange_and_projector():
    torch.manual_seed(42)
    n, count = 3, 2
    v = torch.linalg.qr(torch.randn(count, 2 * n, n, dtype=torch.float64), mode='reduced').Q
    projector = v @ v.transpose(-1, -2)
    linear = torch.tensor([-.7, .2, -1.], dtype=torch.float64)
    heads, tails = torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2])
    weights = torch.tensor([.3, -.8, .5], dtype=torch.float64)
    driver = torch.randn(2 * n, 2 * n, dtype=torch.float64)
    driver = (driver + driver.T) / 2
    def value(p, grad=False):
        return _hubbard_energy_gradient(p, linear, heads, tails, weights, driver, .6, .4, gradient=grad)
    energy, gradient = value(projector, True)
    delta = torch.randn_like(projector)
    delta = (delta + delta.transpose(-1, -2)) / 2
    eps = 1e-6
    numerical = (value(projector + eps * delta)[0] - value(projector - eps * delta)[0]) / (2 * eps)
    torch.testing.assert_close(numerical, (gradient * delta).sum((-1, -2)), rtol=2e-8, atol=2e-8)
    gv = gradient @ v
    tangent = gv - v @ (v.transpose(-1, -2) @ gv)
    trial = torch.linalg.qr(v - .001 * tangent, mode='reduced').Q
    new_projector = trial @ trial.transpose(-1, -2)
    assert bool((value(new_projector)[0] < energy).all())
    torch.testing.assert_close(new_projector @ new_projector, new_projector)
    # Exactly one fermion in a coherent cell has no doublon penalty.
    coherent = torch.tensor([[[.5, .5], [.5, .5]]], dtype=torch.float64)
    empty = torch.empty(0, dtype=torch.long)
    expectation, _ = _hubbard_energy_gradient(coherent, torch.zeros(1).double(), empty, empty,
        torch.empty(0).double(), torch.zeros(2, 2).double(), 100., 0.)
    torch.testing.assert_close(expectation, torch.zeros(1).double())


def test_hubbard_measurement_and_readout_bound():
    torch.manual_seed(3)
    problem = _fixture()
    objective = PopulationObjective.from_problem(problem, precision=torch.float64, normalize=False)
    linear, matrix = _source_density_coefficients(objective)
    heads, tails = torch.tensor([0]), torch.tensor([1])
    weights = matrix[heads, tails]
    v = torch.linalg.qr(torch.randn(2, 4, 2, dtype=torch.float64), mode='reduced').Q
    projector = v @ v.transpose(-1, -2)
    ones = torch.ones(2, dtype=torch.bool)
    plus, _ = _condition_projector(projector, 1, ones)
    minus, _ = _condition_projector(projector, 1, ~ones)
    for branch in (plus, minus):
        torch.testing.assert_close(branch @ branch, branch)
    target, _ = _hubbard_energy_gradient(projector, linear, heads, tails, weights,
                                         torch.zeros(4, 4).double(), 0., 0.)
    bits = _hubbard_readout(projector, linear, heads, tails, weights)
    assert bool((problem.energy(bits) <= target + 1e-12).all())


def test_bdg_flip_identity_paired_spectrum_and_exact_masks():
    torch.manual_seed(21)
    n = 5
    problem = QUBO(torch.randn(n, n, dtype=torch.float64), offset=3.2)
    objective = PopulationObjective.from_problem(problem, normalize=False)
    linear, matrix = _source_density_coefficients(objective)
    x = torch.randint(2, (2, n)).double()
    sigma = 1 - 2 * x
    costs = sigma * (_mm(matrix, x) + linear)
    interactions = matrix * sigma[:, :, None] * sigma[:, None, :]
    masks = torch.tensor(list(itertools.product((0, 1), repeat=n)), dtype=torch.float64)
    exact = costs @ masks.T + .5 * torch.einsum('ti,bij,tj->bt', masks, interactions, masks)
    expected = torch.stack([problem.energy(x + sigma * mask) - problem.energy(x) for mask in masks], -1)
    torch.testing.assert_close(exact, expected)
    chosen, value = _exact_block_minimum(costs, interactions, tile_size=7)
    torch.testing.assert_close(value, exact.min(-1).values)
    torch.testing.assert_close(value, problem.energy(x + sigma * chosen) - problem.energy(x))
    proposal = _bdg_proposal(costs, interactions)
    torch.testing.assert_close(proposal, proposal.transpose(-1, -2))
    spectrum = torch.linalg.eigvalsh(proposal)
    torch.testing.assert_close(spectrum, -spectrum.flip(-1))


def test_bdg_pair_trap_and_source_precision():
    problem = QUBO(torch.tensor([[1., -1.5], [-1.5, 1.]], dtype=torch.float64))
    initial = torch.zeros((1, 2), dtype=torch.float64)
    solver = BogoliubovDeGennes(max_steps=1, block_size=2, max_block_size=2, polish_steps=0)
    result = solver.solve(problem, initial_assignments=initial, seed=7)
    assert float(result.best_energies[0]) == -1.
    assert bool((result.best_assignments == 1).all())
    # Large cancellation must not narrow source coefficients to FP32.
    problem = QUBO(torch.tensor([[1e8 + 1., -.5e8], [-.5e8, -1.]], dtype=torch.float64))
    objective = PopulationObjective.from_problem(problem, precision=torch.float32)
    linear, matrix = _source_density_coefficients(objective)
    assert linear.dtype == matrix.dtype == torch.float64
    assert float(linear[0]) == 1e8 + 1.


@pytest.mark.parametrize('solver_class', HAMILTONIAN_GAUSSIAN_SOLVERS.values())
def test_gaussian_solver_scoring_reproducibility_and_memory_guard(solver_class):
    problem = _fixture()
    solver = solver_class(max_steps=3, check_interval=1)
    first = solver.solve(problem, restarts=2, seed=9, history_interval=1)
    second = solver.solve(problem, restarts=2, seed=9, history_interval=1)
    torch.testing.assert_close(first.best_energies, problem.energy(first.best_assignments))
    torch.testing.assert_close(first.best_assignments, second.best_assignments)
    estimate = solver.estimate_memory(problem, restarts=2)
    with pytest.raises(MemoryError, match='estimated tensor storage'):
        solver.solve(problem, restarts=2, memory_limit_bytes=estimate.total_bytes - 1)


@pytest.mark.parametrize('solver_class', HAMILTONIAN_GAUSSIAN_SOLVERS.values())
def test_single_variable_and_zero_step(solver_class):
    problem = QUBO(torch.tensor([[-1.]], dtype=torch.float64))
    result = solver_class(max_steps=0).solve(problem, restarts=1, seed=7)
    assert bool(torch.isfinite(result.best_energies).all())
    torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments))


@pytest.mark.parametrize('solver_class', HAMILTONIAN_GAUSSIAN_SOLVERS.values())
def test_zero_steps_preserve_supplied_starts_without_model_initialization(solver_class, monkeypatch):
    problem = _fixture()
    initial = torch.tensor([[1., 1.], [0., 0.]], dtype=torch.float64)
    def forbidden(*args, **kwargs):
        raise AssertionError('zero-step solve performed model initialization or polish')
    from qubo_solvers import hamiltonian_gaussian as module
    for name in ('_fock_covariance', '_hubbard_readout', '_polish', '_source_density_coefficients'):
        monkeypatch.setattr(module, name, forbidden)
    result = solver_class(max_steps=0).solve(problem, initial_assignments=initial,
        restarts=2, seed=7, history_interval=1)
    torch.testing.assert_close(result.best_assignments.double(), initial)
    torch.testing.assert_close(result.final_assignments.double(), initial)
    torch.testing.assert_close(result.best_energies, problem.energy(initial))
    assert bool((result.iterations == 0).all())


@pytest.mark.parametrize('solver_class', (SparseSYK, KitaevChain, FermiHubbard))
def test_backtracking_cap(solver_class):
    with pytest.raises(ValueError, match='backtracking_steps must be <= 20'):
        solver_class(backtracking_steps=21)


@pytest.mark.parametrize('solver_class', HAMILTONIAN_GAUSSIAN_SOLVERS.values())
def test_sparse_ising_compact_duplicates_self_loops_and_resident(solver_class):
    from types import SimpleNamespace
    np = pytest.importorskip('numpy')
    from qubo_solvers.backends.problem import QUBOProblem
    compact = QUBOProblem(np.array([-1., -2., 0., 0., 0., 0., 0., 0.]),
        np.array([0, 0, 1]), np.array([1, 1, 1]), np.array([1., 2., .5]), offset=2.)
    solver = solver_class(max_steps=2, check_interval=1, sparse_threshold=1.)
    resident = SimpleNamespace(source=compact, linear=torch.tensor(compact.linear),
        heads=torch.tensor(compact.quadraticHeads.astype('int64')),
        tails=torch.tensor(compact.quadraticTails.astype('int64')),
        biases=torch.tensor(compact.quadraticBiases))
    first = solver.solve_compact(compact, restarts=2, seed=7)
    second = solver.solve_compact_resident(resident, restarts=2, seed=7)
    torch.testing.assert_close(first.best_assignments, second.best_assignments)
    expected = torch.tensor([compact.energy(row.tolist()) for row in first.best_assignments], dtype=torch.float64)
    torch.testing.assert_close(first.best_energies, expected)
    diagonal = QUBO(torch.diag(torch.arange(-4., 4., dtype=torch.float64))).to_ising()
    result = solver.solve(diagonal, restarts=2, seed=7)
    torch.testing.assert_close(result.best_energies, diagonal.energy(result.best_assignments))
