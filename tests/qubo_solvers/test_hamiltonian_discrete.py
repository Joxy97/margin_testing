"""Independent numerical checks for the five discrete-carrier methods."""
import itertools
import math

import pytest
import torch

from qubo_solvers import QUBO, Ising
from qubo_solvers._population import PopulationObjective
from qubo_solvers.hamiltonian_discrete import (
    _Target, _round, _bounded_roots, _xy_minimum, _segment_delta,
    _zn_conditional, HAMILTONIAN_DISCRETE_SOLVERS,
)


DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('spin', [False, True])
def test_multilinear_source_and_incremental_costs(device, sparse, spin):
    matrix = torch.tensor([[1., .3, -.2], [.3, -2., .7], [-.2, .7, .2]],
                          dtype=torch.float64, device=device)
    problem = (Ising(matrix, torch.tensor([.2, -.7, .8], device=device, dtype=torch.float64), offset=2.)
               if spin else QUBO(matrix, offset=2.))
    objective = PopulationObjective.from_problem(problem, precision='float32', threshold=1. if sparse else 0.)
    target = _Target(objective, normalize=False)
    values = torch.tensor([[.2, .3, .7], [.1, .5, .4]], device=device, dtype=torch.float64)
    field = target.field(values)
    for i in range(3):
        new = values.clone()
        new[:, i] = 1 - new[:, i]
        delta = new[:, i] - values[:, i]
        torch.testing.assert_close(target.energy(new) - target.energy(values), delta * field[:, i])
        cached = field.clone()
        target.update(cached, i, delta)
        torch.testing.assert_close(cached, target.field(new))
    bits = torch.tensor(list(itertools.product([0., 1.], repeat=3)), device=device, dtype=torch.float64)
    assignments = bits * 2 - 1 if spin else bits
    direct = problem.energy(assignments)
    derived = target.energy(bits)
    torch.testing.assert_close(direct - direct[0], derived - derived[0])
    rounded = _round(target, values)
    assert bool((target.energy(rounded) <= target.energy(values) + 1e-14).all())


@pytest.mark.parametrize('device', DEVICES)
def test_compact_duplicate_diagonal_source(device):
    import numpy as np
    from qubo_solvers.backends.problem import QUBOProblem
    problem = QUBOProblem(linear=np.array([-1., 2., -.4]),
        quadraticHeads=np.array([0, 1, 0, 2]), quadraticTails=np.array([1, 0, 0, 2]),
        quadraticBiases=np.array([3., -.5, .7, -.2]), offset=5.)
    objective = PopulationObjective.from_compact(problem, device=device, precision=torch.float32,
                                                normalize=True, threshold=1.)
    target = _Target(objective, normalize=False)
    bits = torch.tensor(list(itertools.product([0., 1.], repeat=3)), device=device, dtype=torch.float64)
    exact = objective.score(bits)
    derived = target.energy(bits)
    torch.testing.assert_close(exact - exact[0], derived - derived[0])


def test_all_ternary_single_and_swap_costs():
    q = QUBO(torch.tensor([[-1., 1.5], [1.5, -2.]], dtype=torch.float64))
    target = _Target(PopulationObjective.from_problem(q), occupation=False, normalize=False)
    chemical = .73
    states = torch.tensor(list(itertools.product([-1., 0., 1.], repeat=2)), dtype=torch.float64)
    energy = target.energy(states) + chemical * (1 - states.square()).sum(-1)
    field = target.field(states)
    for i in range(2):
        for label in [-1., 0., 1.]:
            new = states.clone()
            new[:, i] = label
            delta = label - states[:, i]
            expected = delta * field[:, i] + chemical * (states[:, i].square() - label ** 2)
            actual = target.energy(new) + chemical * (1 - new.square()).sum(-1) - energy
            torch.testing.assert_close(actual, expected)
    delta = states[:, 1] - states[:, 0]
    expected = delta * (field[:, 0] - field[:, 1]) - target.entry(0, 1) * delta.square()
    torch.testing.assert_close(target.energy(states.flip(-1)) - target.energy(states), expected)


def test_cyclic_oriented_bond_and_endpoint():
    q = 8
    a = torch.arange(q, dtype=torch.float64)[:, None]
    b = torch.arange(q, dtype=torch.float64)[None]
    forward = _zn_conditional(a, b, 1., q, math.pi / 6)
    reverse = _zn_conditional(b, a, -1., q, math.pi / 6)
    torch.testing.assert_close(forward, reverse)
    torch.testing.assert_close(forward, _zn_conditional(a + q, b, 1., q, math.pi / 6))


@pytest.mark.parametrize('device', DEVICES)
def test_bounded_polynomial_repeated_and_reduced_degree(device):
    # (x-.2)^2(x+.7)(x-.8), x^2-.25, and a constant polynomial.
    coefficients = torch.tensor([[-.0224, .172, -.56, -.5, 1.],
                                 [-.25, 0., 1., 0., 0.], [1., 0., 0., 0., 0.]],
                                dtype=torch.float64, device=device)
    # Construct the repeated-root polynomial independently to avoid fixture algebra.
    import numpy as np
    coefficients[0] = torch.tensor(np.poly([.2, .2, -.7, .8])[::-1].copy(), device=device)
    roots = _bounded_roots(coefficients)
    expected = torch.tensor([-.7, .2, .8], dtype=torch.float64, device=device)
    for value in expected:
        assert torch.nan_to_num((roots[0] - value).abs(), nan=10.).min() < 1e-6
    finite = roots[1][torch.isfinite(roots[1])]
    torch.testing.assert_close(finite, torch.tensor([-.5, .5], device=device, dtype=torch.float64))
    assert torch.isnan(roots[2]).all()


@pytest.mark.parametrize('device', DEVICES)
def test_xy_minimum_against_dense_independent_grid(device):
    generator = torch.Generator(device=device).manual_seed(87)
    a, b = torch.randn((2, 32), generator=generator, device=device, dtype=torch.float64)
    a[:4], b[:4] = 0., 0.
    old = torch.rand(32, generator=generator, device=device, dtype=torch.float64) * (2 * math.pi)
    grid = torch.linspace(-math.pi, math.pi, 20001, device=device, dtype=torch.float64)
    for kappa in (0., .05, 2.):
        angle = _xy_minimum(a, b, kappa, old)
        cost = -a * angle.cos() - b * angle.sin() + kappa * angle.sin().square()
        grid_cost = (-a[:, None] * grid.cos() - b[:, None] * grid.sin() + kappa * grid.sin().square()).min(-1).values
        assert bool((cost <= grid_cost + 1e-10).all())


@pytest.mark.parametrize('device', DEVICES)
def test_worldline_segment_cost_including_wrap_and_full_ring(device):
    generator = torch.Generator(device=device).manual_seed(41)
    problem = QUBO(torch.tensor([[-1., 1.5], [1.5, -1.]], dtype=torch.float64, device=device))
    target = _Target(PopulationObjective.from_problem(problem), normalize=False)
    state = torch.randint(2, (4, 5, 2), generator=generator, device=device).double()
    field = target.field(state.reshape(-1, 2)).reshape_as(state)
    beta, coupling = 3., .7
    def action(value):
        spin = value * 2 - 1
        return beta / 5 * target.energy(value.reshape(-1, 2)).reshape(4, 5).sum(-1) - coupling * (spin * spin.roll(-1, 1)).sum((1, 2))
    for mask_row in ([1, 0, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 1, 1, 1]):
        mask = torch.tensor([mask_row] * 4, dtype=torch.bool, device=device)
        for bit in (0, 1):
            new = state.clone()
            new[:, :, bit] = torch.where(mask, 1 - new[:, :, bit], new[:, :, bit])
            torch.testing.assert_close(action(new) - action(state), _segment_delta(state, field, bit, mask, beta, coupling))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', tuple(HAMILTONIAN_DISCRETE_SOLVERS))
def test_solver_source_score_restarts_history_and_memory(name, device):
    cls = HAMILTONIAN_DISCRETE_SOLVERS[name]
    solver = cls(max_steps=2, **({'slices': 4} if name == 'lib_quantum_dimer' else {}))
    problem = QUBO(torch.tensor([[-1., 1.5], [1.5, -2.]], dtype=torch.float64, device=device), offset=1.2)
    result = solver.solve(problem, restarts=3, batch_size=2, seed=12, history_interval=1)
    torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments))
    torch.testing.assert_close(result.final_energies, problem.energy(result.final_assignments))
    assert bool((result.best_energies <= result.final_energies + 1e-14).all())
    other = solver.solve(problem, restarts=3, batch_size=2, seed=12)
    torch.testing.assert_close(result.best_assignments, other.best_assignments)
    assert result.energy_history.shape == (3, 3)
    with pytest.raises(MemoryError):
        solver.solve(problem, restarts=3, memory_limit_bytes=1)


@pytest.mark.parametrize('name', tuple(HAMILTONIAN_DISCRETE_SOLVERS))
def test_zero_steps_preserve_explicit_starts(name):
    solver = HAMILTONIAN_DISCRETE_SOLVERS[name](max_steps=0)
    problem = QUBO(torch.eye(2, dtype=torch.float64))
    initial = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8)
    result = solver.solve(problem, restarts=2, initial_assignments=initial, history_interval=1)
    torch.testing.assert_close(result.final_assignments, initial)
