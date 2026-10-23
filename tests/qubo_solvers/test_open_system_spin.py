"""Independent algebra/invariant checks for the six new spin-inspired engines."""

import math

import pytest
import torch

from qubo_solvers import QUBO
from qubo_solvers.open_system_spin import (
    OPEN_SPIN_CLASSES, BCS, KaneMele, KickedTop, NoisyLaserAtom,
    OneAxisTwisting, TwoAxisCounterTwisting, _align_field, _align_z,
    _cycle_skew, _kicked_map, _low_rank, _oat_shear, _patterns,
    _precess, _rotate_y, _rotate_z, _tact_pulse, _unit,
)


def _normal(shape, device, seed=421):
    return torch.randn(shape, device=device, dtype=torch.float64,
                       generator=torch.Generator(device=device).manual_seed(seed))


def test_frozen_alignment_exact_solution_and_poles(device):
    vector = _unit(_normal((3, 8, 3), device))
    force = _normal((3, 8), device, 17)
    updated = _align_z(vector, force, .13)
    expected = torch.tanh(torch.atanh(vector[..., 2]) + .13 * force)
    torch.testing.assert_close(updated[..., 2], expected, atol=2e-15, rtol=2e-14)
    torch.testing.assert_close(updated.norm(dim=-1), torch.ones_like(force), atol=2e-14, rtol=0)
    poles = vector.new_tensor([[[0., 0., 1.], [0., 0., -1.]]])
    opposing = vector.new_tensor([[-10000., 10000.]])
    torch.testing.assert_close(_align_z(poles, opposing, 1.), poles)
    # A small but nonzero transverse direction must not become a pole guard.
    near = vector.new_tensor([[[math.sqrt(1 - (.999999)**2), 0., .999999]]])
    away = _align_z(near, near.new_tensor([[-1.]]), .3)
    assert away[..., 0] > near[..., 0]
    rounded = vector.new_tensor([[[1e-5, 0., 1.]]]).float()
    away = _align_z(rounded, rounded.new_tensor([[-1.]]), .3)
    assert away[..., 0] > rounded[..., 0]


def test_precession_sign_zero_limit_and_norm(device):
    initial = _normal((4, 6, 3), device)
    field = _normal(initial.shape, device, 16)
    dt = 1e-7
    velocity = (_precess(initial, field, dt) - initial) / dt
    torch.testing.assert_close(velocity, torch.linalg.cross(initial, field), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(_precess(initial, field, .13).norm(dim=-1), initial.norm(dim=-1), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(_precess(initial, torch.zeros_like(field), 12.), initial)


def test_phase_diffusion_variance_and_ensemble_decay(device):
    # One fixed seed and a conservative sampling tolerance; no optimizer run.
    count, diffusion, dt = 20000, .7, .2
    vector = torch.zeros((count, 1, 3), device=device, dtype=torch.float64)
    vector[..., 0] = 1
    angles = _normal((count, 1), device) * math.sqrt(2 * diffusion * dt)
    rotated = _rotate_z(vector, angles)
    assert abs(float(rotated[..., 0].mean()) - math.exp(-diffusion * dt)) < .01
    assert abs(float(rotated[..., 1].mean())) < .01
    torch.testing.assert_close(rotated[..., 2], vector[..., 2])


def test_arbitrary_field_alignment_energy_and_zero_limit(device):
    vector = _unit(_normal((3, 8, 3), device))
    field = _normal(vector.shape, device, 16)
    aligned = _align_field(vector, field, .2)
    torch.testing.assert_close(aligned.norm(dim=-1), vector.norm(dim=-1), atol=2e-14, rtol=2e-14)
    assert torch.all((aligned * field).sum(-1) >= (vector * field).sum(-1) - 1e-14)
    torch.testing.assert_close(_align_field(vector, torch.zeros_like(field), 10.), vector)
    eps = 1e-7
    numerical = (_align_field(vector, field, eps) - vector) / eps
    expected = field - (vector * field).sum(-1, keepdim=True) * vector
    torch.testing.assert_close(numerical, expected, atol=1e-6, rtol=2e-6)


def test_kicked_top_inverse_requires_reverse_operator_order(device):
    vector = _unit(_normal((3, 8, 3), device))
    angle = vector.new_tensor(.31)
    mapped = _kicked_map(vector, angle, 1.7)
    restored = _rotate_y(_rotate_z(mapped, -1.7 * mapped[..., 2]), -angle)
    torch.testing.assert_close(restored, vector, atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(mapped.norm(dim=-1), vector.norm(dim=-1), atol=3e-15, rtol=3e-15)
    twist_only = _kicked_map(vector, angle * 0, 1.7)
    torch.testing.assert_close(twist_only[..., 2], vector[..., 2])


def test_tact_norm_inverse_and_classical_velocity(device):
    vector = _unit(_normal((3, 8, 3), device))
    mapped = _tact_pulse(vector, .3)
    torch.testing.assert_close(mapped.norm(dim=-1), vector.norm(dim=-1), atol=4e-15, rtol=4e-15)
    torch.testing.assert_close(_tact_pulse(mapped, -.3), vector, atol=4e-15, rtol=4e-15)
    dt = 1e-7
    u, v, z = vector.unbind(-1)
    expected = torch.stack((-2 * v * z, -2 * u * z, 4 * u * v), -1)
    torch.testing.assert_close((_tact_pulse(vector, dt) - vector) / dt, expected, atol=5e-7, rtol=2e-6)
    # u^2-v^2 has splitting error; do not demand exact conservation.
    initial = u.square() - v.square()
    def error(strength):
        result = _tact_pulse(vector, strength)
        return (result[..., 0].square() - result[..., 1].square() - initial).abs().max()
    assert error(.05) < error(.1) / 5


def test_oat_shear_symplectic_inverse_and_kinetic_change(device):
    z, p = _normal((3, 8), device), _normal((3, 8), device, 17)
    patterns = _patterns(8, 3, z)
    strength = .7
    projected = strength * (z @ (patterns @ patterns.T))
    updated = _oat_shear(z, p, patterns, strength)
    torch.testing.assert_close(updated, p - projected)
    torch.testing.assert_close(_oat_shear(z, updated, patterns, -strength), p)
    kinetic = .5 * (updated.square() - p.square()).sum(-1)
    expected = -(p * projected).sum(-1) + .5 * projected.square().sum(-1)
    torch.testing.assert_close(kinetic, expected)
    eye = torch.eye(8, dtype=z.dtype, device=device)
    zero = torch.zeros_like(eye)
    shear = torch.cat((torch.cat((eye, zero), 1), torch.cat((-strength * patterns @ patterns.T, eye), 1)), 0)
    omega = torch.cat((torch.cat((zero, eye), 1), torch.cat((-eye, zero), 1)), 0)
    torch.testing.assert_close(shear.T @ omega @ shear, omega)


@pytest.mark.parametrize('n', [1, 2, 3, 8])
def test_cycle_skew_and_chiral_descent_identities(n, device):
    identity = torch.eye(n, dtype=torch.float64, device=device)
    matrix = _cycle_skew(identity)
    torch.testing.assert_close(matrix + matrix.T, torch.zeros_like(matrix))
    g = _normal((3, n), device)
    kg = _cycle_skew(g)
    torch.testing.assert_close((g * kg).sum(-1), g.new_zeros(3), atol=2e-15, rtol=0)
    alpha = 1.3
    torch.testing.assert_close((g + alpha * kg).square().sum(-1), g.square().sum(-1) + alpha**2 * kg.square().sum(-1))
    assert torch.linalg.matrix_norm(matrix, 2) <= 1 + 1e-14


def test_bcs_field_is_negative_gradient_and_pairing_endpoint(device):
    vector = _normal((2, 7, 3), device).requires_grad_()
    patterns = _patterns(7, 3, vector)
    coupling = _normal((7, 7), device, 17)
    coupling = .5 * (coupling + coupling.T)
    linear = _normal((7,), device, 22)
    z, u, v = vector[..., 2], vector[..., 0], vector[..., 1]
    strength, pinning = .4, .3
    energy = (.5 * (z * (z @ coupling)).sum() + (z * linear).sum()
              - pinning * z.square().sum()
              - .5 * strength * ((u @ patterns).square().sum() + (v @ patterns).square().sum()))
    negative = -torch.autograd.grad(energy, vector)[0]
    expected = torch.stack((strength * _low_rank(u, patterns), strength * _low_rank(v, patterns),
                            -(z @ coupling + linear) + 2 * pinning * z), -1)
    torch.testing.assert_close(negative, expected)
    torch.testing.assert_close(_low_rank(u, patterns), u @ (patterns @ patterns.T))
    assert BCS(max_steps=10)._controls(9)[0] == 0


def test_chiral_step_bound_decreases_frozen_transformed_energy(device):
    q = _normal((4, 7), device)
    matrix = _normal((7, 7), device, 17)
    matrix = (matrix + matrix.T) / 2
    field = _normal((7,), device, 18)
    pinning, alpha = .3, 2.7
    def energy(coordinate):
        z = coordinate.tanh()
        return .5 * (z * (z @ matrix)).sum(-1) + (z * field).sum(-1) - pinning * z.square().sum(-1)
    differentiable = q.clone().requires_grad_()
    gradient = -torch.autograd.grad(energy(differentiable).sum(), differentiable)[0]
    z = q.tanh()
    torch.testing.assert_close(gradient, (1 - z.square()) * (-z @ matrix - field + 2 * pinning * z))
    bound = 3 * matrix.abs().sum(-1).max() + 2 * field.abs().max() + 6 * pinning
    dt = .9 / (bound * (1 + alpha**2))
    updated = q + dt * (gradient + alpha * _cycle_skew(gradient))
    assert torch.all(energy(updated) <= energy(q) + 1e-14)


@pytest.mark.parametrize('solver_class', list(OPEN_SPIN_CLASSES.values()))
def test_solver_seed_replay_source_scoring_and_initial_incumbent(solver_class, device):
    matrix = torch.tensor([[-3., 1., -.5], [1., 2., -.5], [-.5, -.5, -2.]], device=device, dtype=torch.float64)
    problem = QUBO(matrix, offset=.125)
    initial = torch.tensor([[0., 0., 0.], [1., 0., 1.], [1., 1., 1.]], device=device, dtype=torch.float64)
    solver = solver_class(max_steps=12, candidate_interval=4, polish_steps=3, dynamics_dtype='float32')
    kwargs = dict(restarts=3, seed=123, initial_assignments=initial, history_interval=3)
    result, repeated = solver.solve(problem, **kwargs), solver.solve(problem, **kwargs)
    torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments))
    torch.testing.assert_close(result.best_assignments, repeated.best_assignments)
    assert result.best_energies.dtype == torch.float64
    assert torch.all(result.best_energies <= problem.energy(initial))
    assert torch.all((result.best_assignments == 0) | (result.best_assignments == 1))


@pytest.mark.parametrize('solver_class', list(OPEN_SPIN_CLASSES.values()))
def test_invalid_controls_rejected(solver_class):
    for parameters in ({'time_step': 0}, {'pinning': -1}, {'relaxation_fraction': 1},
                       {'initial_scale': 1}, {'normalize': 1}, {'polish_steps': -1},
                       {'max_steps': True}, {'sparse_threshold': 2}):
        with pytest.raises((ValueError, TypeError)):
            solver_class(**parameters)


def test_model_specific_validation():
    for factory, parameters in ((BCS, {'rank': 33}), (OneAxisTwisting, {'twisting': 5}),
                                (TwoAxisCounterTwisting, {'twisting': float('nan')}),
                                (KaneMele, {'chirality': -1}), (KickedTop, {'replica_spread': 2}),
                                (NoisyLaserAtom, {'phase_diffusion': -1})):
        with pytest.raises(ValueError):
            factory(**parameters)
