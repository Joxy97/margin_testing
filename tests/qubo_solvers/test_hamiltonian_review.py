"""Cross-family history regression and independent XY conditional checks."""

import math

import pytest
import torch

from qubo_solvers import QUBO
from qubo_solvers.hamiltonians import HAMILTONIAN_CLASSES
from qubo_solvers.hamiltonian_discrete import _xy_minimum


@pytest.mark.parametrize('name', tuple(HAMILTONIAN_CLASSES))
@pytest.mark.parametrize('compact', [False, True])
def test_all_23_constant_objectives_have_complete_source_scored_histories(name, compact, device):
    solver = HAMILTONIAN_CLASSES[name](max_steps=5)
    if compact:
        import numpy as np
        from qubo_solvers.backends.problem import QUBOProblem
        problem = QUBOProblem(np.zeros(3), np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32), np.empty(0), offset=2.5)
        result = solver.solve_compact(problem, device=device, restarts=3,
            batch_size=2, seed=4, history_interval=2)
    else:
        problem = QUBO(torch.zeros((3, 3), device=device, dtype=torch.float64), offset=2.5)
        result = solver.solve(problem, restarts=3, batch_size=2, seed=4, history_interval=2)
    assert result.history_iterations.tolist() == [0, 2, 4, 5]
    torch.testing.assert_close(result.energy_history, torch.full_like(result.energy_history, 2.5), rtol=0, atol=0)
    torch.testing.assert_close(result.best_energies, torch.full_like(result.best_energies, 2.5), rtol=0, atol=0)
    torch.testing.assert_close(result.final_energies, torch.full_like(result.final_energies, 2.5), rtol=0, atol=0)
    assert bool(((result.iterations >= 0) & (result.iterations <= 5)).all())


@pytest.mark.parametrize('kappa', [0., .01, 1., 100.])
def test_xy_global_minimum_against_independent_dense_angle_grid(kappa, device):
    generator = torch.Generator(device=device).manual_seed(1023)
    a, b = torch.randn((2, 128), device=device, dtype=torch.float64, generator=generator)
    scales = torch.logspace(-5, 5, 128, device=device, dtype=torch.float64)
    a, b = a * scales, b * scales
    # Include both sides of the half-angle chart point and exact degeneracies.
    a[:8] = a.new_tensor([0., 0., -1., -1., 1., 1., -100., 100.])
    b[:8] = b.new_tensor([0., 1., 1e-10, -1e-10, 1e-10, -1e-10, -3., 3.])
    old = torch.rand(128, device=device, dtype=torch.float64, generator=generator) * (2 * math.pi)
    angle = _xy_minimum(a, b, kappa, old)
    actual = -a * angle.cos() - b * angle.sin() + kappa * angle.sin().square()
    old_energy = -a * old.cos() - b * old.sin() + kappa * old.sin().square()
    allowance = 2e-12 * (1 + a.abs() + b.abs() + kappa)
    assert bool(torch.isfinite(angle).all())
    assert bool((actual <= old_energy + allowance).all())
    # A direct grid on the whole circle is independent of the scalar reduction.
    # Bound its temporary storage independently of case count.
    grid = torch.linspace(-math.pi, math.pi, 32769, device=device, dtype=torch.float64)
    cosine, sine = grid.cos(), grid.sin()
    for start in range(0, a.numel(), 16):
        end = start + 16
        grid_energy = (-a[start:end, None] * cosine - b[start:end, None] * sine
                       + kappa * sine.square()).min(-1).values
        assert bool((actual[start:end] <= grid_energy + allowance[start:end]).all())


@pytest.mark.parametrize('kappa', [0., .2, 2.])
def test_xy_degenerate_branches_have_exact_known_minima(kappa, device):
    a = torch.zeros(11, device=device, dtype=torch.float64)
    b = a.new_tensor([0., .01, -.01, .1, -.1, .4, -.4, 4., -4., 100., -100.])
    old = torch.linspace(-math.pi, math.pi, 11, device=device, dtype=torch.float64)
    angle = _xy_minimum(a, b, kappa, old)
    actual = -b * angle.sin() + kappa * angle.sin().square()
    u = (b.abs() / (2 * kappa)).clamp_max(1) if kappa > 0 else torch.ones_like(b)
    expected = -b.abs() * u + kappa * u.square()
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    if kappa == 0:
        assert angle[0] == old[0]


def test_xy_exact_minimizer_remains_on_old_angle_for_equal_energy_ties(device):
    old = torch.tensor([math.pi, -math.pi, 0., math.pi / 6, -math.pi / 6],
                       device=device, dtype=torch.float64)
    a = torch.zeros_like(old)
    b = torch.zeros_like(old)
    torch.testing.assert_close(_xy_minimum(a, b, 0., old), old, rtol=0, atol=0)
    # Both signs of cos(theta) attain the A=0, B=1, kappa=1 branches.
    stationary = old.new_tensor([math.pi / 6, 5 * math.pi / 6])
    torch.testing.assert_close(_xy_minimum(torch.zeros_like(stationary),
        torch.ones_like(stationary), 1., stationary), stationary, rtol=0, atol=0)
