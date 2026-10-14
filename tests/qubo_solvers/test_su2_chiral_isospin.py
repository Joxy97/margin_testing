"""SU(2) forces, geometric integration, and original-score candidate retention."""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from qubo_solvers import Ising, QUBO
from qubo_solvers.observation import SolveInterrupted, observing
from qubo_solvers.su2_chiral_isospin import (
    SU2ChiralIsospin, _GeometricMidpointWorkspace, _ambient_gradient,
    _controls, _generator, _rank_two_action, _rank_two_vectors, _rodrigues,
)


class _Objective:
    """Document convention E=h.z + .5*z.Jphys.z, with explicit out support."""

    def __init__(self, matrix, field):
        self.matrix, self.field = matrix, field

    def gradient(self, values, out=None):
        if out is None:
            out = torch.empty_like(values)
        torch.mm(values, self.matrix, out=out)
        out.add_(self.field)
        return out

    def energy(self, values):
        return .5 * ((values @ self.matrix) * values).sum(-1) + (values * self.field).sum(-1)


def _normal(device, dtype, seed=37):
    generator = torch.Generator(device=device).manual_seed(seed)
    return lambda shape: torch.randn(shape, dtype=dtype, device=device, generator=generator)


def _source_problem(device, dtype):
    return QUBO(torch.tensor([[-1., .3, -.5], [.3, -.25, .4], [-.5, .4, .2]],
                             device=device, dtype=dtype), offset=7.25)


@pytest.mark.parametrize('n', [1, 2, 11])
def test_rank_two_operator_matches_explicit_skew_matrix(n, device, dtype):
    normal = _normal(device, dtype)
    u, v = _rank_two_vectors(normal, (4, n))
    values = normal((4, n))
    matrix = u[..., :, None] * v[..., None, :] - v[..., :, None] * u[..., None, :]
    actual = _rank_two_action(values, u, v)
    expected = torch.bmm(matrix, values[..., None]).squeeze(-1)
    tolerance = 5e-6 if dtype == torch.float32 else 5e-14
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(matrix, -matrix.transpose(-1, -2), rtol=0, atol=0)
    if n == 1:
        assert torch.count_nonzero(actual) == 0
    else:
        torch.testing.assert_close(u.norm(dim=-1), torch.ones(4, device=device, dtype=dtype),
                                   rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(v.norm(dim=-1), torch.ones(4, device=device, dtype=dtype),
                                   rtol=tolerance, atol=tolerance)
        torch.testing.assert_close((u * v).sum(-1), torch.zeros(4, device=device, dtype=dtype),
                                   rtol=0, atol=tolerance)
        torch.testing.assert_close(_rank_two_action(u, u, v), -v, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(_rank_two_action(v, u, v), u, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('zero', [False, True])
def test_degenerate_gram_schmidt_has_finite_orthonormal_fallback(zero, device, dtype):
    def degenerate(shape):
        return torch.full(shape, 0. if zero else 1., device=device, dtype=dtype)
    u, v = _rank_two_vectors(degenerate, (3, 2))
    tolerance = 5e-6 if dtype == torch.float32 else 5e-14
    assert torch.isfinite(u).all() and torch.isfinite(v).all()
    torch.testing.assert_close(u.norm(dim=-1), torch.ones_like(u[..., 0]),
                               rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(v.norm(dim=-1), torch.ones_like(v[..., 0]),
                               rtol=tolerance, atol=tolerance)
    torch.testing.assert_close((u * v).sum(-1), torch.zeros_like(u[..., 0]),
                               rtol=0, atol=tolerance)


def test_ambient_force_matches_finite_difference_chiral_hamiltonian(device):
    dtype = torch.float64
    normal = _normal(device, dtype)
    vector = normal((1, 3, 3)) * .3
    u, v = _rank_two_vectors(normal, (1, 3))
    matrix = torch.tensor([[0., .2, -.3], [.2, 0., .4], [-.3, .4, 0.]],
                          device=device, dtype=dtype)
    objective = _Objective(matrix, vector.new_tensor([.1, -.2, .3]))
    controls = (.43, .8, .7, -.25)
    gradient = _ambient_gradient(vector, objective, u, v, controls)

    def energy(value):
        a, gamma, kappa, chi = controls
        x, y, z = value.unbind(-1)
        return (a * objective.energy(z) - gamma * x.sum(-1)
                + .5 * kappa * (1 - z.square()).sum(-1)
                + chi * (x * _rank_two_action(y, u, v)).sum(-1)).sum()

    epsilon = 1e-6
    expected = torch.empty_like(gradient)
    for index in range(vector.numel()):
        plus, minus = vector.clone(), vector.clone()
        plus.view(-1)[index] += epsilon
        minus.view(-1)[index] -= epsilon
        expected.view(-1)[index] = (energy(plus) - energy(minus)) / (2 * epsilon)
    torch.testing.assert_close(gradient, expected, rtol=2e-8, atol=2e-9)


def test_generator_matches_tangent_descent_and_dissipation(device, dtype):
    normal = _normal(device, dtype)
    vector = normal((3, 4, 3))
    vector /= vector.norm(dim=-1, keepdim=True)
    u, v = _rank_two_vectors(normal, (3, 4))
    matrix = torch.zeros((4, 4), device=device, dtype=dtype)
    objective = _Objective(matrix, vector.new_tensor([.1, -.2, .3, .5]))
    controls, damping = (.6, .8, .4, .3), .37
    gradient = _ambient_gradient(vector, objective, u, v, controls)
    omega = _generator(vector, objective, u, v, controls, damping)
    flow = torch.linalg.cross(omega, vector)
    tangent = gradient - vector * (vector * gradient).sum(-1, keepdim=True)
    expected = torch.linalg.cross(gradient, vector) - damping * tangent
    tolerance = 5e-6 if dtype == torch.float32 else 5e-14
    torch.testing.assert_close(flow, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close((vector * flow).sum(-1), torch.zeros_like(vector[..., 0]),
                               rtol=0, atol=tolerance)
    descent = -damping * torch.linalg.cross(vector, gradient).square().sum(-1)
    torch.testing.assert_close((gradient * flow).sum(-1), descent,
                               rtol=tolerance, atol=tolerance)


def test_rodrigues_zero_tiny_and_full_angle(device, dtype):
    vector = torch.tensor([[[1., 0., 0.]]], device=device, dtype=dtype)
    torch.testing.assert_close(_rodrigues(vector, torch.zeros_like(vector), .2),
                               vector, rtol=0, atol=0)
    omega = vector.new_tensor([[[0., 0., 2.]]])
    expected = vector.new_tensor([[[0., 1., 0.]]])
    tolerance = 3e-6 if dtype == torch.float32 else 5e-14
    torch.testing.assert_close(_rodrigues(vector, omega, math.pi / 4), expected,
                               rtol=tolerance, atol=tolerance)
    tiny = _rodrigues(vector, omega, 1e-10)
    assert torch.isfinite(tiny).all()
    torch.testing.assert_close(tiny[..., 1], vector[..., 1] + 2e-10,
                               rtol=tolerance, atol=1e-18)


def test_geometric_midpoint_preserves_unit_norm(device, dtype):
    normal = _normal(device, dtype)
    vector = normal((4, 7, 3))
    vector /= vector.norm(dim=-1, keepdim=True)
    u, v = _rank_two_vectors(normal, (4, 7))
    raw = normal((7, 7)) * .1
    matrix = (raw + raw.T) / 2
    matrix.fill_diagonal_(0)
    objective = _Objective(matrix, normal((7,)) * .1)
    work = _GeometricMidpointWorkspace(vector)
    for k in range(300):
        work.advance(vector, objective, u, v, _controls(k / 300, 1., 1., .2),
                     _controls((k + .5) / 300, 1., 1., .2), .02, .3)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-13
    assert torch.isfinite(vector).all()
    torch.testing.assert_close(vector.norm(dim=-1), torch.ones_like(vector[..., 0]),
                               rtol=tolerance, atol=tolerance)


def test_midpoint_has_second_order_constant_field_convergence(device):
    dtype = torch.float64
    field, damping, duration, initial_z = .7, .3, 1.3, .2
    objective = _Objective(torch.zeros((1, 1), device=device, dtype=dtype),
                           torch.tensor([field], device=device, dtype=dtype))
    initial = torch.tensor([[[math.sqrt(1 - initial_z**2), 0., initial_z]]],
                           device=device, dtype=dtype)
    final_z = math.tanh(math.atanh(initial_z) - damping * field * duration)
    radius, phase = math.sqrt(1 - final_z**2), field * duration
    exact = initial.new_tensor([[[radius * math.cos(phase), radius * math.sin(phase), final_z]]])
    errors = []
    for steps in (40, 80, 160):
        vector = initial.clone()
        work = _GeometricMidpointWorkspace(vector)
        controls = (1., 0., 0., 0.)
        for _ in range(steps):
            work.advance(vector, objective, None, None, controls, controls,
                         duration / steps, damping)
        errors.append((vector - exact).norm())
    ratios = torch.stack([errors[0] / errors[1], errors[1] / errors[2]])
    torch.testing.assert_close(ratios, torch.full_like(ratios, 4.), rtol=.01, atol=.01)


def test_endpoint_drivers_are_exactly_zero():
    assert _controls(1., 1.7, .9, -.3) == (1., 0., .9, 0.)
    assert _controls(0., 1.7, .9, -.3) == (0., 1.7, 0., -0.)


@pytest.mark.parametrize('spin', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_document_target_gradient_and_normalization(monkeypatch, spin, normalize, device):
    source = _source_problem(device, torch.float64)
    problem = source.to_ising() if spin else source
    physical_matrix = source.Q / 2
    physical_matrix.fill_diagonal_(0)
    field = source.Q.sum(-1) / 2
    scale = (field.abs().max() + physical_matrix.abs().sum(-1).max()).clamp_min(1)
    if not normalize:
        scale = torch.ones_like(scale)
    values = source.Q.new_tensor([[.2, -.4, .7], [-.8, .3, .1]])
    seen = []

    def inspect(self, run, record):
        seen.append((run.objective.gradient(values), run.objective.scale))
        return 0

    monkeypatch.setattr(SU2ChiralIsospin, '_search', inspect)
    SU2ChiralIsospin(max_steps=0, normalize=normalize).solve(problem, restarts=2, seed=9)
    torch.testing.assert_close(seen[0][0], (values @ physical_matrix + field) / scale,
                               rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(torch.as_tensor(seen[0][1], device=device), scale,
                               rtol=0, atol=0)


@pytest.mark.parametrize('spin', [False, True])
def test_seeded_batched_results_keep_original_scores_and_inputs(spin, device, dtype):
    source = _source_problem(device, dtype)
    problem = source.to_ising() if spin else source
    saved = (problem.J if spin else problem.Q).clone()
    solver = SU2ChiralIsospin(max_steps=19, candidate_interval=7)
    options = dict(restarts=7, batch_size=3, seed=17, history_interval=4)
    full = solver.solve(problem, **options)
    again = solver.solve(problem, **options)
    best = solver.solve(problem, **dict(options, history_interval=None), best_only=True)
    for name in ('final_assignments', 'best_assignments', 'best_energies',
                 'iterations', 'energy_history', 'history_iterations'):
        torch.testing.assert_close(getattr(full, name), getattr(again, name), rtol=0, atol=0)
    reference = problem.to(dtype=torch.float64)
    torch.testing.assert_close(reference.energy(full.best_assignments), full.best_energies,
                               rtol=0, atol=0)
    torch.testing.assert_close(reference.energy(full.final_assignments), full.final_energies,
                               rtol=0, atol=0)
    # History checkpoints add eligible observations; they may improve a full
    # archive over a solve that observes only the configured candidate cadence.
    assert full.best_energy <= best.best_energy
    assert full.best_energies.dtype == torch.float64
    assert full.iterations.tolist() == [19] * 7
    assert full.history_iterations.tolist() == [0, 4, 8, 12, 16, 19]
    torch.testing.assert_close(full.energy_history[-1], full.final_energies, rtol=0, atol=0)
    torch.testing.assert_close(problem.J if spin else problem.Q, saved, rtol=0, atol=0)


def test_fp32_tie_does_not_discard_better_original_candidate(device):
    problem = QUBO(torch.diag(torch.tensor([-2**24, -1.], device=device, dtype=torch.float64)), 3.)
    initial = torch.tensor([[1, 0], [1, 1]], dtype=torch.int8, device=device)
    result = SU2ChiralIsospin(max_steps=0, dynamics_dtype='float32').solve(
        problem, restarts=2, initial_assignments=initial, best_only=True)
    assert result.best_restart_index == 1
    assert result.best_energy == -2**24 + 2
    torch.testing.assert_close(result.best_assignment, initial[1], rtol=0, atol=0)


@pytest.mark.parametrize('threshold', [0., 1.])
@pytest.mark.parametrize('resident', [False, True])
def test_compact_archive_scores_raw_integer_source_before_coefficient_aggregation(
        threshold, resident, device):
    from qubo_solvers.backends.problem import QUBOProblem

    # Absolute source sum is exactly 2**53, inside the documented FP64 integer
    # bound. Halving a diagonal -1 into two -.5 additions before cancellation
    # can lose part of that coefficient when converting the force operator.
    source = QUBOProblem([-2**52, 0., 0.], [0, 0], [0, 0], [-1., 2**52 - 1])
    starts = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.int8, device=device)
    solver = SU2ChiralIsospin(max_steps=0, sparse_threshold=threshold, dynamics_dtype='float32')
    options = dict(device=device, restarts=2, initial_assignments=starts, seed=7)
    if resident:
        prepared = SimpleNamespace(source=source,
            linear=torch.tensor(source.linear.copy(), device=device, dtype=torch.float64),
            heads=torch.tensor(source.quadraticHeads.astype('int64'), device=device),
            tails=torch.tensor(source.quadraticTails.astype('int64'), device=device),
            biases=torch.tensor(source.quadraticBiases.copy(), device=device, dtype=torch.float64))
        result = solver.solve_compact_resident(prepared, **options)
    else:
        result = solver.solve_compact(source, **options)
    expected = torch.tensor([source.energy([1, 0, 0]), source.energy([0, 0, 0])],
                            device=device, dtype=torch.float64)
    assert expected.tolist() == [-2., 0.]
    torch.testing.assert_close(result.best_energies, expected, rtol=0, atol=0)
    assert result.best_energy == -2.


@pytest.mark.parametrize('dynamics_dtype', ['float32', 'float64'])
def test_dynamics_dtype_is_separate_from_original_score(monkeypatch, dynamics_dtype, device, dtype):
    problem = _source_problem(device, dtype)
    original = _GeometricMidpointWorkspace.advance
    seen = []

    def inspect(work, vector, *args, **kwargs):
        seen.append(vector.dtype)
        return original(work, vector, *args, **kwargs)

    monkeypatch.setattr(_GeometricMidpointWorkspace, 'advance', inspect)
    result = SU2ChiralIsospin(max_steps=3, dynamics_dtype=dynamics_dtype).solve(
        problem, restarts=4, seed=7)
    assert seen == [getattr(torch, dynamics_dtype)] * 3
    assert result.best_energies.dtype == torch.float64
    torch.testing.assert_close(problem.to(dtype=torch.float64).energy(result.best_assignments),
                               result.best_energies, rtol=0, atol=0)


@pytest.mark.parametrize('spin', [False, True])
def test_initial_bloch_state_is_near_positive_x_and_preserves_start_bits(monkeypatch, spin, device, dtype):
    problem = _source_problem(device, dtype)
    bits = torch.tensor([[0, 0, 1], [1, 0, 1], [1, 1, 0]], device=device, dtype=torch.int8)
    starts = bits
    if spin:
        problem, starts = problem.to_ising(), 2 * bits - 1
    seen = []
    original = _GeometricMidpointWorkspace.advance

    def inspect(work, vector, *args, **kwargs):
        seen.append(vector.clone())
        return original(work, vector, *args, **kwargs)

    monkeypatch.setattr(_GeometricMidpointWorkspace, 'advance', inspect)
    SU2ChiralIsospin(max_steps=1).solve(problem, restarts=3, seed=7, initial_assignments=starts)
    initial = seen[0]
    assert (initial[..., 0] > 0).all()
    assert (initial[..., 1] != 0).all() and (initial[..., 2] != 0).all()
    torch.testing.assert_close((initial[..., 2] >= 0).to(torch.int8), bits, rtol=0, atol=0)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-14
    torch.testing.assert_close(initial.norm(dim=-1), torch.ones_like(initial[..., 0]),
                               rtol=tolerance, atol=tolerance)


def test_chirality_zero_skips_low_rank_products(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('zero chirality evaluated a low-rank product')

    monkeypatch.setattr('qubo_solvers.su2_chiral_isospin._rank_two_action', forbidden)
    monkeypatch.setattr('qubo_solvers.su2_chiral_isospin._rank_two_vectors', forbidden)
    result = SU2ChiralIsospin(max_steps=5, chirality=0.).solve(QUBO(-torch.eye(3)), seed=7)
    assert torch.isfinite(result.best_energy)


def test_midpoint_has_no_device_scalar_extraction(device):
    vector = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], device=device, dtype=torch.float64)
    objective = _Objective(torch.zeros((2, 2), device=device, dtype=vector.dtype),
                           vector.new_tensor([.2, -.3]))
    work = _GeometricMidpointWorkspace(vector)
    controls = (.5, .3, .2, 0.)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        work.advance(vector, objective, None, None, controls, controls, .02, .3)
    scalar_reads = sum(event.count for event in profile.key_averages()
                       if event.key == 'aten::_local_scalar_dense')
    assert scalar_reads == 0


@pytest.mark.parametrize('spin', [False, True])
def test_sparse_and_dense_target_paths_score_the_same_assignments(spin, device, dtype):
    problem = _source_problem(device, dtype)
    if spin:
        problem = problem.to_ising()
    options = dict(max_steps=9, time_step=.001, candidate_interval=3)
    solve = dict(restarts=4, batch_size=4, seed=13)
    dense = SU2ChiralIsospin(**options, sparse_threshold=0.).solve(problem, **solve)
    sparse = SU2ChiralIsospin(**options, sparse_threshold=1.).solve(problem, **solve)
    torch.testing.assert_close(sparse.final_assignments, dense.final_assignments, rtol=0, atol=0)
    torch.testing.assert_close(sparse.best_assignments, dense.best_assignments, rtol=0, atol=0)
    torch.testing.assert_close(sparse.best_energies, dense.best_energies, rtol=0, atol=0)


@pytest.mark.parametrize('spin', [False, True])
def test_initial_population_preserves_all_supplied_corners(spin, device):
    problem = _source_problem(device, torch.float64)
    if spin:
        problem = problem.to_ising()
    starts = torch.tensor(list(itertools.product((0, 1), repeat=3)),
                          dtype=torch.int8, device=device)
    if spin:
        starts = 2 * starts - 1
    result = SU2ChiralIsospin(max_steps=0).solve(
        problem, restarts=len(starts), batch_size=3, initial_assignments=starts, seed=7)
    torch.testing.assert_close(result.final_assignments, starts, rtol=0, atol=0)
    torch.testing.assert_close(result.best_energies, problem.energy(starts), rtol=0, atol=0)
    assert result.best_energy == problem.energy(starts).min()


@pytest.mark.parametrize('n', [1, 3])
def test_zero_target_has_valid_nonpole_initialization_and_outputs(n, device, dtype):
    problem = QUBO(torch.zeros((n, n), device=device, dtype=dtype), 8.25)
    result = SU2ChiralIsospin(max_steps=5, chirality=0.).solve(problem, restarts=4, seed=3)
    assert torch.isfinite(result.best_energies).all()
    torch.testing.assert_close(result.best_energies, torch.full_like(result.best_energies, 8.25),
                               rtol=0, atol=0)
    assert result.final_assignments.shape == (4, n)


def test_capture_cadence_includes_all_candidates_history_and_final():
    class Observer:
        def __init__(self):
            self.events = []

        def poll(self):
            pass

        def capture(self, samples, raw=None, *, phase='checkpoint', iteration=None):
            self.events.append((iteration, samples.clone(), raw.clone()))

    problem = _source_problem('cpu', torch.float64)
    observer = Observer()
    with observing(observer):
        SU2ChiralIsospin(max_steps=13, candidate_interval=5).solve(
            problem, restarts=3, seed=11, history_interval=4)
    assert [event[0] for event in observer.events] == [0, 4, 5, 8, 10, 12, 13]
    assert all(event[1].shape == (3, 3) for event in observer.events)
    for _, assignments, energies in observer.events:
        torch.testing.assert_close(problem.energy(assignments), energies, rtol=0, atol=0)


def test_deadline_poll_interrupts_before_another_step():
    class Observer:
        def __init__(self):
            self.initial = False

        def poll(self):
            if self.initial:
                raise SolveInterrupted('deadline')

        def capture(self, samples, raw=None, **kwargs):
            self.initial = True

    with observing(Observer()), pytest.raises(SolveInterrupted, match='deadline'):
        SU2ChiralIsospin(max_steps=5).solve(QUBO(-torch.eye(2)), seed=7)


@pytest.mark.parametrize('options', [dict(max_steps=-1), dict(candidate_interval=0),
    dict(time_step=0.), dict(damping=-.1), dict(transverse_field=-1.), dict(penalty=-1.),
    dict(chirality=float('nan')), dict(initial_noise=0.), dict(normalize=1),
    dict(dynamics_dtype='float16'), dict(sparse_threshold=-.1), dict(sparse_threshold=True)])
def test_invalid_parameters(options):
    with pytest.raises(ValueError):
        SU2ChiralIsospin(**options)


def test_workspace_scales_with_batch_and_refuses_too_small_memory_budget():
    problem = QUBO(torch.eye(30, dtype=torch.float64))
    solver = SU2ChiralIsospin(max_steps=2)
    full = solver.estimate_memory(problem, restarts=100, best_only=True)
    bounded = solver.estimate_memory(problem, restarts=100, batch_size=3, best_only=True)
    assert full.workspace_bytes > bounded.workspace_bytes
    assert bounded == solver.estimate_memory(problem, restarts=10000, batch_size=3, best_only=True)
    with pytest.raises(MemoryError, match='estimated tensor storage'):
        solver.solve(problem, memory_limit_bytes=1)


@pytest.mark.parametrize('options', [dict(restarts=0), dict(restarts=True), dict(batch_size=0),
    dict(best_only=1), dict(history_interval=0), dict(seed=-1), dict(seed=True),
    dict(memory_limit_bytes=0), dict(best_only=True, history_interval=1)])
def test_common_population_options_are_validated_before_search(options):
    with pytest.raises(ValueError):
        SU2ChiralIsospin(max_steps=0).solve(QUBO(torch.eye(2)), **options)


def test_invalid_late_batch_start_is_rejected_without_mutation(device):
    problem = _source_problem(device, torch.float64)
    initial = torch.zeros((7, 3), device=device)
    initial[-1, -1] = 2.
    saved = initial.clone()
    with pytest.raises(ValueError, match='outside the problem domain'):
        SU2ChiralIsospin(max_steps=2).solve(
            problem, restarts=7, batch_size=2, initial_assignments=initial)
    torch.testing.assert_close(initial, saved, rtol=0, atol=0)


def test_overflowing_selected_dynamics_precision_is_rejected_at_preparation(device):
    problem = Ising(torch.zeros((1, 1), device=device, dtype=torch.float64),
                    torch.tensor([1e40], device=device, dtype=torch.float64))
    with pytest.raises(FloatingPointError):
        SU2ChiralIsospin(max_steps=0, normalize=False, dynamics_dtype='float32').solve(problem)
