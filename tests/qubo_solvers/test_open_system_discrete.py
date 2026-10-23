"""Small independent algebra oracles for seven compendium reductions."""
import itertools

import pytest
import torch

from qubo_solvers import QUBO, Ising
from qubo_solvers._population import PopulationObjective
from qubo_solvers.hamiltonian_discrete import _Target
from qubo_solvers.open_system_discrete import (
    OPEN_DISCRETE_CLASSES, Kondo, MooreRead, ANNNI, LiebMattis, Emery,
    _block_gain, _enumerate_block, _range_two_minimum, _activity_step,
    _emery_step, _submatrix, _commit,
)


DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


def _problem(device='cpu'):
    matrix = torch.tensor([[1., 2., -3., 0.], [2., -2., 1., -.5],
                           [-3., 1., .5, 2.], [0., -.5, 2., -1.]],
                          device=device, dtype=torch.float64)
    return QUBO(matrix, offset=3.)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('sparse', [False, True])
def test_block_gain_and_commit_with_untouched_environment(device, sparse):
    problem = _problem(device)
    target = _Target(PopulationObjective.from_problem(problem,
                     threshold=1. if sparse else 0.), occupation=False, normalize=False)
    states = torch.tensor(list(itertools.product([-1., 1.], repeat=4)),
                          device=device, dtype=torch.float64)
    field = target.field(states)
    indices = [0, 2, 3]
    matrix = _submatrix(target, indices)
    for labels in itertools.product([-1., 1.], repeat=3):
        new = torch.tensor(labels, device=device, dtype=torch.float64).expand(states.shape[0], -1)
        actual = states.clone()
        actual[:, indices] = new
        expected = problem.energy((actual + 1) / 2) - problem.energy((states + 1) / 2)
        torch.testing.assert_close(_block_gain(states[:, indices], new, field[:, indices], matrix), expected)
        updated, updated_field = states.clone(), field.clone()
        _commit(target, updated, updated_field, indices, new)
        torch.testing.assert_close(updated_field, target.field(actual))


def test_enumeration_crosses_pair_barrier_and_prefers_old_ties():
    # E=-3*s0*s1+s0+s1, current++: single cost+4, joint cost-4.
    matrix = torch.tensor([[0., -3.], [-3., 0.]], dtype=torch.float64)
    old = torch.ones((1, 2), dtype=torch.float64)
    field = old @ matrix + 1
    assert bool((-2 * old * field > 0).all())
    best = _enumerate_block(old, field, matrix, chunk_size=2)
    torch.testing.assert_close(best, -old)
    torch.testing.assert_close(_enumerate_block(old, torch.zeros_like(old), matrix * 0), old)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('n', [1, 2, 3, 7])
def test_range_two_dp_against_exhaustive_oracle(device, n):
    generator = torch.Generator(device=device).manual_seed(37)
    boundary = torch.randn((3, n), generator=generator, device=device, dtype=torch.float64)
    first = torch.randn(max(0, n - 1), generator=generator, device=device, dtype=torch.float64)
    second = torch.randn(max(0, n - 2), generator=generator, device=device, dtype=torch.float64)
    states = torch.tensor(list(itertools.product([-1., 1.], repeat=n)), device=device, dtype=torch.float64)
    def energy(x):
        return ((boundary[:, None, :] * x).sum(-1)
                + (first * x[:, :-1] * x[:, 1:]).sum(-1)
                + (second * x[:, :-2] * x[:, 2:]).sum(-1))
    expected = energy(states).min(-1).values
    result = _range_two_minimum(boundary, first, second)
    actual = (boundary * result).sum(-1) + (first * result[:, :-1] * result[:, 1:]).sum(-1) + (second * result[:, :-2] * result[:, 2:]).sum(-1)
    torch.testing.assert_close(actual, expected)


def test_annni_surrogate_improvement_can_be_rejected():
    # Endpoints coupled strongly outside path range; surrogate flips only first.
    old = torch.ones((1, 4), dtype=torch.float64)
    matrix = torch.zeros((4, 4), dtype=torch.float64)
    matrix[0, 3] = matrix[3, 0] = -10.
    boundary = torch.tensor([[1., -1., -1., -1.]], dtype=torch.float64)
    proposed = _range_two_minimum(boundary, matrix.diagonal(1), matrix.diagonal(2))
    assert float(((proposed - old) * boundary).sum()) < 0
    assert float(_block_gain(old, proposed, boundary + old @ matrix, matrix)) > 0


@pytest.mark.parametrize('device', DEVICES)
def test_activity_mass_positivity_and_two_site_diffusion(device):
    activity = torch.tensor([[.9, .1]], device=device, dtype=torch.float64)
    gains = torch.zeros_like(activity)
    heads, tails = torch.tensor([0], device=device), torch.tensor([1], device=device)
    hopping = torch.tensor([.3], device=device, dtype=torch.float64)
    updated = _activity_step(activity, gains, heads, tails, hopping,
        dephasing=.5, reinjection=.2, activity_temperature=.2, rate_cap=2.)
    k = 2 * .3 ** 2 * .5 / (.5 ** 2 + 1e-12)
    dt = .9 / (.2 + k)
    expected_difference = (.9 - .1) * (1 - dt * (2 * k + .2))
    torch.testing.assert_close(updated.sum(-1), torch.ones(1, device=device, dtype=torch.float64))
    assert bool((updated >= 0).all())
    assert abs(float(updated[0, 0] - updated[0, 1]) - expected_difference) < 1e-12


@pytest.mark.parametrize('device', DEVICES)
def test_emery_square_split_decreases_fixed_target(device):
    target = _Target(PopulationObjective.from_problem(_problem(device)), occupation=False)
    z = torch.tensor([[.3, -.4, .2, -.1]], device=device, dtype=torch.float64)
    u, v = z * .7 + .2, z * -.4 - .1
    args = dict(dt=.05, pinning=.1, stiffness_u=.5, stiffness_v=.8,
                coupling_u=.7, coupling_v=-.4, mobility_u=.5, mobility_v=3.)
    def energy(z, u, v):
        return (target.energy(z) + .1 * (1 - z.square()).sum(-1)
                + .25 * (u - .7 * z).square().sum(-1)
                + .4 * (v + .4 * z).square().sum(-1))
    before = energy(z, u, v)
    result = _emery_step(target, z, u, v, **args)
    assert bool((energy(*result) < before).all())
    torch.testing.assert_close(energy(z, .7 * z, -.4 * z), target.energy(z) + .1 * (1 - z.square()).sum(-1))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('solver_class', OPEN_DISCRETE_CLASSES.values())
@pytest.mark.parametrize('sparse', [False, True])
def test_small_feasible_source_scored_reproducible_solver(device, solver_class, sparse):
    problem = _problem(device)
    solver = solver_class(max_steps=3, sparse_threshold=1. if sparse else 0., polish_sweeps=1)
    first = solver.solve(problem, restarts=3, seed=27, history_interval=1)
    second = solver.solve(problem, restarts=3, seed=27)
    torch.testing.assert_close(first.best_energies, problem.energy(first.best_assignments.double()))
    torch.testing.assert_close(first.best_energies, second.best_energies)
    assert bool(((first.best_assignments == 0) | (first.best_assignments == 1)).all())
    assert bool(torch.isfinite(first.best_energies).all())


@pytest.mark.parametrize('solver_class', OPEN_DISCRETE_CLASSES.values())
def test_constant_and_empty_schedule(solver_class):
    problem = QUBO(torch.zeros((2, 2), dtype=torch.float64), offset=7.)
    result = solver_class(max_steps=3).solve(problem, restarts=2, seed=2, history_interval=1)
    torch.testing.assert_close(result.best_energies, torch.full((2,), 7., dtype=torch.float64))
    result = solver_class(max_steps=0).solve(_problem(), restarts=2, seed=2)
    torch.testing.assert_close(result.best_energies, _problem().energy(result.best_assignments.double()))


@pytest.mark.parametrize('solver_class', OPEN_DISCRETE_CLASSES.values())
def test_parameter_validation(solver_class):
    with pytest.raises(ValueError):
        solver_class(max_steps=-1)
    with pytest.raises(ValueError):
        solver_class(sparse_threshold=float('nan'))
    with pytest.raises(MemoryError):
        solver_class(max_steps=1).solve(_problem(), memory_limit_bytes=1)


def test_exact_open_chain_solver_and_small_full_block():
    problem = _problem()
    matrix = problem.Q.clone()
    matrix[0, 3] = matrix[3, 0] = 0.
    chain = QUBO(matrix)
    states = torch.tensor(list(itertools.product([0., 1.], repeat=4)), dtype=torch.float64)
    optimum = chain.energy(states).min()
    for solver in (ANNNI(max_steps=1, polish_sweeps=0), Kondo(max_steps=1, block_size=4, polish_sweeps=0)):
        result = solver.solve(chain, restarts=2, seed=2)
        torch.testing.assert_close(result.best_energies, optimum.expand(2))


@pytest.mark.parametrize('solver_class', OPEN_DISCRETE_CLASSES.values())
def test_ising_and_compact_duplicate_source_contract(solver_class):
    import numpy as np
    from qubo_solvers.backends.problem import QUBOProblem
    matrix = _problem().Q
    ising = Ising(matrix, torch.tensor([.2, -.3, .4, -.5], dtype=torch.float64), offset=7.)
    solver = solver_class(max_steps=2, sparse_threshold=1.)
    result = solver.solve(ising, restarts=2, seed=5, best_only=True)
    torch.testing.assert_close(result.best_energies, ising.energy(result.best_assignments.double()))
    compact = QUBOProblem(linear=np.array([-.25, .3, -.2, .7]),
        quadraticHeads=np.array([0, 1, 0, 2, 3]), quadraticTails=np.array([1, 0, 0, 3, 2]),
        quadraticBiases=np.array([2. ** 25, -(2. ** 25) + .5, -.6, 1.2, -.3]), offset=7.)
    result = solver.solve_compact(compact, restarts=3, batch_size=2, seed=5)
    values = result.best_assignments.double().numpy()
    expected = values @ compact.linear + (values[:, compact.quadraticHeads] * values[:, compact.quadraticTails] * compact.quadraticBiases).sum(-1) + compact.offset
    torch.testing.assert_close(result.best_energies, torch.from_numpy(expected), rtol=0, atol=1e-8)


def test_emery_overflow_cannot_be_hidden_by_box_projection():
    with pytest.raises(FloatingPointError, match='nonfinite'):
        Emery(max_steps=1, stiffness_u=1e308, initial_residual=1e308).solve(_problem(), seed=4)
    with pytest.raises(ValueError, match='overflows'):
        Emery(coupling_u=1e308)
