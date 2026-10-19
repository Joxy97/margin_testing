"""Independent derivative, endpoint and integration checks for six new models."""

import itertools
import math

import pytest
import torch

from qubo_solvers import QUBO
from qubo_solvers._population import PopulationObjective
from qubo_solvers.hamiltonian_spin import (
    HAMILTONIAN_SPIN_SOLVERS, J1J2, MajumdarGhosh, QuantumClock, QuantumRotor,
    SpinSqueezing, _Edges, _Stencil, _backtrack, _clock_energy, _clock_gradient,
    _clock_moments, _controls, _llg_velocity, _phase_energy, _phase_gradient,
    _rotate, _rotor_step, _spin_energy, _spin_gradient, _squeezing_energy,
    _squeezing_gradient, _target_energy, _unit,
)


class _Objective:
    def __init__(self, device):
        self.matrix = torch.tensor([[0., .3, -.2], [.3, 0., .1], [-.2, .1, 0.]],
                                   device=device, dtype=torch.float64)
        self.field = self.matrix.new_tensor([.2, -.4, .1])

    def gradient(self, z):
        return z @ self.matrix + self.field


def _normal(shape, device, dtype=torch.float64, seed=170):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _directional(energy, value, direction):
    epsilon = 1e-6
    return (energy(value + epsilon * direction) - energy(value - epsilon * direction)) / (2 * epsilon)


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('n', [1, 2, 3, 6])
def test_stencil_is_symmetric_and_has_unique_tiny_ring_edges(periodic, n, device):
    stencil = _Stencil(((1, .4),), periodic=periodic)
    identity = torch.eye(n, device=device, dtype=torch.float64)
    matrix = stencil.action(identity)
    torch.testing.assert_close(matrix, matrix.T)
    assert torch.count_nonzero(matrix.diagonal()) == 0
    if n == 2:
        assert matrix[0, 1] == .4


@pytest.mark.parametrize('exchange2', [-.7, .7])
def test_exchange_force_and_dissipation_match_independent_energy(exchange2, device):
    objective = _Objective(device)
    vector = _unit(_normal((2, 3, 3), device))
    direction = _normal(vector.shape, device, seed=12)
    driver = _Edges(3, device, ((((0, 1), (1, 2)), .25), (((0, 2),), exchange2 / 4)))
    tau, pinning = .6, .35
    gradient = _spin_gradient(vector, objective, driver, tau, pinning)

    def energy(value):
        z = value[..., 2]
        target = .5 * (z * (z @ objective.matrix)).sum(-1) + (z * objective.field).sum(-1)
        first = (value[:, 0] * value[:, 1]).sum(-1) + (value[:, 1] * value[:, 2]).sum(-1)
        second = (value[:, 0] * value[:, 2]).sum(-1)
        return target + pinning * (1 - z.square()).sum(-1) + tau * (.25 * first + exchange2 / 4 * second)

    torch.testing.assert_close(_spin_energy(vector, objective, driver, tau, pinning), energy(vector))
    torch.testing.assert_close(_directional(energy, vector, direction),
                               (gradient * direction).sum((-1, -2)), rtol=1e-6, atol=1e-8)
    velocity = _llg_velocity(vector, gradient, .7, -1.)
    torque = torch.linalg.cross(vector, gradient)
    torch.testing.assert_close((gradient * velocity).sum((-1, -2)),
                               -.7 * torque.square().sum((-1, -2)), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close((vector * velocity).sum(-1), torch.zeros_like(vector[..., 0]), atol=1e-14, rtol=0)


def test_mg_padding_and_quarter_eighth_coefficients(device):
    solver = MajumdarGhosh(exchange=1.3)
    objective = _Objective(device)
    vector = _unit(_normal((1, 6, 3), device))
    direction = _normal(vector.shape, device, seed=7)
    driver = solver._driver(6, device)
    gradient = _spin_gradient(vector, objective, driver, .8, .2)
    finite_difference = _directional(lambda value: _spin_energy(value, objective, driver, .8, .2), vector, direction)
    torch.testing.assert_close(finite_difference, (gradient * direction).sum((-1, -2)), rtol=1e-6, atol=1e-8)
    # A dummy spin has no logical target gradient.
    g0 = _spin_gradient(vector, objective, driver, 0., 0.)
    torch.testing.assert_close(g0[:, 3:], torch.zeros_like(g0[:, 3:]))


def test_j1j2_shared_edges_add_and_edge_validation(device):
    driver = J1J2(edges1=((0, 1),), edges2=((0, 1),), exchange1=2., exchange2=-.5)._driver(3, device)
    values = torch.eye(3, device=device, dtype=torch.float64)
    expected = torch.zeros_like(values)
    expected[0, 1] = expected[1, 0] = .375
    torch.testing.assert_close(driver.action(values), expected)
    with pytest.raises(ValueError, match='in-range'):
        J1J2(edges1=((0, 3),))._driver(3, device)


def test_phase_force_retains_cosine_target_and_phase_difference_driver(device):
    objective = _Objective(device)
    theta = _normal((2, 3), device)
    direction = _normal(theta.shape, device, seed=15)
    driver = _Stencil(((1, .2),), periodic=True)
    derivative = _phase_gradient(theta, objective, driver, .7, .4)
    finite_difference = _directional(lambda value: _phase_energy(value, objective, driver, .7, .4), theta, direction)
    torch.testing.assert_close(finite_difference, (derivative * direction).sum(-1), rtol=1e-6, atol=1e-8)


def test_clock_wirtinger_force_and_local_phase_invariance(device):
    objective = _Objective(device)
    spinor = _unit(torch.complex(_normal((2, 3, 4), device), _normal((2, 3, 4), device, seed=44)))
    direction = torch.complex(_normal(spinor.shape, device, seed=20), _normal(spinor.shape, device, seed=21))
    angles = torch.arange(4, device=device, dtype=torch.float64) * (math.pi / 2)
    clock = torch.polar(torch.ones_like(angles), angles)
    diagonal = (1 + angles.cos()) / 2
    driver = _Stencil(((1, .25),), periodic=True)
    gradient = _clock_gradient(spinor, objective, driver, diagonal, clock, .7, .4, 1.2)
    energy = lambda value: _clock_energy(value, objective, driver, diagonal, clock, .7, .4, 1.2)
    torch.testing.assert_close(_directional(energy, spinor, direction),
        2 * (gradient.conj() * direction).real.sum((-1, -2)), rtol=1e-6, atol=1e-8)
    phase = torch.polar(torch.ones_like(spinor.real[..., :1]), _normal((2, 3, 1), device, seed=2))
    torch.testing.assert_close(energy(spinor * phase), energy(spinor), rtol=1e-12, atol=1e-12)
    tangent = gradient - (spinor.conj() * gradient).sum(-1, keepdim=True).real * spinor
    velocity = -(1. + 1j) * tangent
    torch.testing.assert_close(2 * (gradient.conj() * velocity).real.sum((-1, -2)),
                               -2 * tangent.abs().square().sum((-1, -2)), rtol=1e-12, atol=1e-12)


def test_squeezing_gradient_and_exact_pauli_self_correction(device):
    objective = _Objective(device)
    vector = _unit(_normal((2, 3, 3), device))
    direction = _normal(vector.shape, device, seed=4)
    gradient = _squeezing_gradient(vector, objective, .6, .4, .7, 1.2)
    energy = lambda value: _squeezing_energy(value, objective, .6, .4, .7, 1.2)
    torch.testing.assert_close(_directional(energy, vector, direction),
                               (gradient * direction).sum((-1, -2)), rtol=1e-6, atol=1e-8)
    single = type('Single', (), {'field': objective.field[:1], 'gradient': lambda self, z: torch.zeros_like(z)})()
    one = vector[:, :1]
    zero = _squeezing_gradient(one, single, .8, 2., .3, 0.)
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
    expected_constant = torch.full((2,), .8 * 2. * .7 / 4, device=device, dtype=torch.float64)
    single.field = torch.zeros_like(single.field)
    torch.testing.assert_close(_squeezing_energy(one, single, .8, 2., .3, 0.), expected_constant)


def test_squeezing_energy_matches_explicit_three_qubit_product_expectation(device):
    objective = _Objective(device)
    vector = _unit(_normal((1, 3, 3), device))[0]
    local = torch.stack((torch.sqrt((1 + vector[:, 2]) / 2).to(torch.complex128),
        (vector[:, 0] + 1j * vector[:, 1]) / torch.sqrt(2 * (1 + vector[:, 2]))), -1)
    state = torch.kron(torch.kron(local[0], local[1]), local[2])
    identity = torch.eye(2, device=device, dtype=torch.complex128)
    pauli = (identity.new_tensor([[0., 1.], [1., 0.]]),
             identity.new_tensor([[0., -1j], [1j, 0.]]),
             identity.new_tensor([[1., 0.], [0., -1.]]))
    operators = []
    for axis in pauli:
        operators.append([torch.kron(torch.kron(axis if i == 0 else identity,
            axis if i == 1 else identity), axis if i == 2 else identity) for i in range(3)])
    sx, sy = sum(operators[0]) / 2, sum(operators[1]) / 2
    amplitude, twisting, eta, field = .7, .4, .6, 1.2
    hamiltonian = amplitude * twisting * (sx @ sx - eta * sy @ sy) - amplitude * field * sum(operators[0])
    for i in range(3):
        hamiltonian += objective.field[i] * operators[2][i]
        for j in range(i + 1, 3):
            hamiltonian += objective.matrix[i, j] * (operators[2][i] @ operators[2][j])
    expected = (state.conj() * (hamiltonian @ state)).sum().real
    actual = _squeezing_energy(vector[None], objective, amplitude, twisting, eta, field)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_current_energy_reuses_force_products_without_changing_hamiltonian(device):
    objective = _Objective(device)
    vector = _unit(_normal((2, 3, 3), device))
    driver = _Stencil(((1, .2),), periodic=True)
    _, energy = _spin_gradient(vector, objective, driver, .6, .3, with_energy=True)
    torch.testing.assert_close(energy, _spin_energy(vector, objective, driver, .6, .3))
    theta = _normal((2, 3), device)
    _, energy = _phase_gradient(theta, objective, driver, .6, .3, with_energy=True)
    torch.testing.assert_close(energy, _phase_energy(theta, objective, driver, .6, .3))
    spinor = _unit(torch.complex(_normal((2, 3, 4), device), _normal((2, 3, 4), device, seed=4)))
    angles = torch.arange(4, device=device, dtype=torch.float64) * (math.pi / 2)
    clock, diagonal = torch.polar(torch.ones_like(angles), angles), (1 + angles.cos()) / 2
    _, energy = _clock_gradient(spinor, objective, driver, diagonal, clock, .6, .3, 1., with_energy=True)
    torch.testing.assert_close(energy, _clock_energy(spinor, objective, driver, diagonal, clock, .6, .3, 1.))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_controlled_frozen_steps_match_cpu_and_gpu():
    cpu, gpu = _Objective(torch.device('cpu')), _Objective(torch.device('cuda'))
    driver = _Stencil(((1, .2),), periodic=True)
    vector = _unit(_normal((2, 3, 3), 'cpu'))
    gradient = _spin_gradient(vector, cpu, driver, .7, .3)
    expected = _unit(vector + .01 * _llg_velocity(vector, gradient, .8, -1.))
    device_vector = vector.cuda()
    actual = _unit(device_vector + .01 * _llg_velocity(device_vector,
        _spin_gradient(device_vector, gpu, driver, .7, .3), .8, -1.))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-11, atol=1e-12)
    theta, momentum, noise = (_normal((2, 3), 'cpu', seed=i) for i in (1, 2, 3))
    common = dict(tau=.6, pinning=.3, mass=2., damping=.8, temperature=.4, time_step=.02)
    expected = _rotor_step(theta, momentum, cpu, driver, normal=lambda _: noise, **common)
    actual = _rotor_step(theta.cuda(), momentum.cuda(), gpu, driver, normal=lambda _: noise.cuda(), **common)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left.cpu(), right, rtol=1e-11, atol=1e-12)


def test_rodrigues_norm_and_zero_generator(device, dtype):
    vector = _unit(_normal((3, 5, 3), device, dtype))
    omega = _normal(vector.shape, device, dtype, seed=19)
    dt = torch.full((3, 1, 1), .03, device=device, dtype=dtype)
    result = _rotate(vector, omega, dt)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(result.norm(dim=-1), torch.ones_like(result[..., 0]), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(_rotate(vector, torch.zeros_like(omega), dt), vector, rtol=tolerance, atol=tolerance)


def test_joint_backtracking_retains_entire_rejected_replica(device):
    state = torch.tensor([[1., 2.], [2., 3.]], device=device, dtype=torch.float64)
    energy = lambda value: value.square().sum(-1)
    velocity = torch.tensor([[-1., -2.], [1., 1.]], device=device, dtype=torch.float64)
    result, accepted = _backtrack(state, energy(state), lambda dt: state + dt * velocity,
                                  energy, .1, 3)
    assert accepted.tolist() == [True, False]
    torch.testing.assert_close(result[1], state[1], rtol=0, atol=0)
    assert energy(result)[0] < energy(state)[0]


def test_rotor_ou_step_uses_mass_times_temperature_variance(device):
    class Free:
        def gradient(self, z):
            return torch.zeros_like(z)
    theta = torch.zeros((24_000, 1), device=device, dtype=torch.float64)
    momentum = torch.zeros_like(theta)
    generator = torch.Generator(device=device).manual_seed(45)
    normal = lambda shape: torch.randn(shape, device=device, dtype=theta.dtype, generator=generator)
    _, result = _rotor_step(theta, momentum, Free(), _Stencil(()), tau=0., pinning=0.,
        mass=3., damping=.8, temperature=.7, time_step=.4, normal=normal)
    expected = 3. * .7 * (1 - math.exp(-2 * .8 * .4))
    assert float(result.var()) == pytest.approx(expected, rel=.035)


def test_rotor_conservative_step_converges_when_halved(device):
    objective = _Objective(device)
    theta, momentum = _normal((2, 3), device), _normal((2, 3), device, seed=4)
    driver = _Stencil(((1, .2),), periodic=True)
    energy = lambda t, p: _phase_energy(t, objective, driver, .5, .3) + .5 * p.square().sum(-1)
    initial = energy(theta, momentum)
    errors = []
    for dt in (.04, .02):
        t, p = _rotor_step(theta, momentum, objective, driver, tau=.5, pinning=.3,
            mass=1., damping=0., temperature=0., time_step=dt, normal=None)
        errors.append((energy(t, p) - initial).abs().max())
    assert errors[1] < errors[0] / 5


def test_all_boolean_endpoints_equal_original_source_up_to_constant(device):
    problem = QUBO(torch.tensor([[-1., 1.5, .2], [1.5, -.7, -.3], [.2, -.3, .4]],
                                device=device, dtype=torch.float64), offset=2.3)
    objective = PopulationObjective.from_problem(problem, normalize=False, threshold=0.)
    bits = torch.tensor(list(itertools.product((0., 1.), repeat=3)), device=device, dtype=torch.float64)
    signs = 2 * bits - 1
    baseline = objective.score(torch.zeros_like(bits[:1])) - _target_energy(-torch.ones_like(bits[:1]), objective)
    source = objective.score(bits)
    vector = torch.zeros((8, 3, 3), device=device, dtype=torch.float64)
    vector[..., 2] = signs
    phase = torch.where(bits.bool(), torch.zeros_like(bits), torch.full_like(bits, math.pi))
    driver = _Stencil(((1, .25),), periodic=True)
    for energy in (_spin_energy(vector, objective, driver, 0., .7),
                   _phase_energy(phase, objective, driver, 0., .7),
                   _squeezing_energy(vector, objective, 0., .2, .7, 1.)):
        torch.testing.assert_close(energy + baseline, source, rtol=1e-13, atol=1e-13)
    spinor = torch.zeros((8, 3, 4), device=device, dtype=torch.complex128)
    spinor.scatter_(-1, torch.where(bits.bool(), 0, 2)[..., None], 1.)
    angles = torch.arange(4, device=device, dtype=torch.float64) * (math.pi / 2)
    diagonal = (1 + angles.cos()) / 2
    clock = torch.polar(torch.ones_like(angles), angles)
    energy = _clock_energy(spinor, objective, driver, diagonal, clock, 0., .7, 1.)
    torch.testing.assert_close(energy + baseline, source, rtol=1e-13, atol=1e-13)
    p, _ = _clock_moments(spinor, diagonal, clock)
    torch.testing.assert_close(p, bits, rtol=0, atol=0)


@pytest.mark.parametrize('solver_class', list(HAMILTONIAN_SPIN_SOLVERS.values()))
def test_short_search_returns_source_scores_and_exact_history_intervals(solver_class, device, dtype):
    matrix = torch.tensor([[-1., 1.5], [1.5, -1.]], device=device, dtype=dtype)
    problem = QUBO(matrix, offset=2.)
    solver = solver_class(max_steps=7, candidate_interval=4)
    result = solver.solve(problem, restarts=3, batch_size=2, seed=40, history_interval=2)
    objective = PopulationObjective.from_problem(problem)
    torch.testing.assert_close(result.best_energies, objective.score(result.best_assignments))
    torch.testing.assert_close(result.final_energies, objective.score(result.final_assignments))
    assert result.history_iterations.tolist() == [0, 2, 4, 6, 7]
    assert torch.isfinite(result.energy_history).all()
    assert (result.best_energies <= result.energy_history.min(0).values).all()


@pytest.mark.parametrize('solver_class', list(HAMILTONIAN_SPIN_SOLVERS.values()))
def test_zero_steps_preserve_initial_assignments(solver_class, device):
    problem = QUBO(torch.eye(2, device=device, dtype=torch.float64))
    initial = torch.tensor([[0., 1.], [1., 0.]], device=device, dtype=torch.float64)
    result = solver_class(max_steps=0, polish_sweeps=1).solve(problem, restarts=2, initial_assignments=initial)
    torch.testing.assert_close(result.final_assignments, initial.to(torch.int8), rtol=0, atol=0)


@pytest.mark.parametrize('solver_class', list(HAMILTONIAN_SPIN_SOLVERS.values()))
def test_optional_source_exact_polish_finds_no_improving_single_bit(solver_class, device):
    problem = QUBO(torch.tensor([[-1., 1.5], [1.5, -1.]], device=device, dtype=torch.float64))
    result = solver_class(max_steps=2, polish_sweeps=3).solve(problem, restarts=2, seed=3)
    objective = PopulationObjective.from_problem(problem)
    for i in range(2):
        trial = result.best_assignments.clone()
        trial[:, i] = 1 - trial[:, i]
        assert (objective.score(trial) >= result.best_energies).all()


@pytest.mark.parametrize('solver_class', list(HAMILTONIAN_SPIN_SOLVERS.values()))
def test_memory_admission_precedes_state_allocation(solver_class, device):
    problem = QUBO(torch.eye(3, device=device, dtype=torch.float64))
    with pytest.raises(MemoryError, match='estimated tensor storage'):
        solver_class(max_steps=1).solve(problem, restarts=2, memory_limit_bytes=1)


def test_controls_remove_driver_exactly_and_hold_relaxation():
    assert _controls(0, 13, 1 / 6) == (1., 0.)
    assert _controls(10, 13, 1 / 6) == (0., 1.)
    assert _controls(12, 13, 1 / 6) == (0., 1.)


@pytest.mark.parametrize('kwargs', [{'q': 3}, {'q': 5}, {'damping': 0.}, {'backtracking_steps': 21}])
def test_clock_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        QuantumClock(**kwargs)
