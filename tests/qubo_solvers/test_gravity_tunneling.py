"""Independent mathematical and source-score checks for gravity tunneling."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from qubo_solvers import QUBO
from qubo_solvers.gravity_tunneling import (
    GravityTunneling, _GravityWorkspace, _decode, _feedback_weights,
    _jump_proposal, _metropolis_accept, _reflect_drift, _softened_gravity,
)
from qubo_solvers.observation import observing


def _fixture_run(device):
    matrix = torch.tensor([[0., .8], [.8, 0.]], device=device, dtype=torch.float64)
    field = matrix.new_tensor([.3, -.6])
    generator = torch.Generator(device=device).manual_seed(91)

    def gradient(values, out=None):
        result = values @ matrix + field
        if out is not None:
            out.copy_(result)
            return out
        return result

    return SimpleNamespace(
        x=matrix.new_ones((3, 2)), q=matrix, spin=True, generator=generator,
        energies=matrix.new_tensor([.1, -.2, .7]),
        objective=SimpleNamespace(gradient=gradient, scale=matrix.new_tensor(1.)),
        normal=lambda shape: torch.randn(shape, generator=generator, device=device,
                                        dtype=matrix.dtype),
    ), matrix, field


def test_force_is_negative_finite_difference_gradient(device):
    run, matrix, field = _fixture_run(device)
    work = _GravityWorkspace(run, GravityTunneling(softening=.65))
    positions = matrix.new_tensor([[.2, -.4], [-.5, .3], [.7, .6]])
    weights = matrix.new_tensor([.15, .35, .5])
    a, penalty, gravity = .7, 1.1, 2.3
    work.q.copy_(positions)
    work.weights = weights
    work.controls = a, penalty, gravity, .2

    def potential(values):
        local = a * ((values @ matrix * values).sum() / 2 + (values * field).sum())
        local += penalty * (values.square() - 1).square().sum() / 4
        displacement = values[:, None] - values[None, :]
        distance = (work.solver.softening ** 2 + displacement.square().sum(-1) / 2).sqrt()
        pairs = torch.triu(weights[:, None] * weights[None, :] / distance, diagonal=1)
        return local - gravity * pairs.sum()

    expected = torch.empty_like(positions)
    epsilon = 1e-6
    for body in range(3):
        for coordinate in range(2):
            plus, minus = positions.clone(), positions.clone()
            plus[body, coordinate] += epsilon
            minus[body, coordinate] -= epsilon
            expected[body, coordinate] = -(potential(plus) - potential(minus)) / (2 * epsilon)
    torch.testing.assert_close(work.force(), expected, rtol=1e-7, atol=2e-9)


def test_matrix_gravity_matches_direct_pair_force(device, dtype):
    generator = torch.Generator().manual_seed(38)
    positions = torch.rand((5, 7), generator=generator, dtype=dtype).to(device) * 2 - 1
    weights = positions.new_tensor([.1, .15, .3, .2, .25])
    gravity, softening = 3.7, .6
    displacement = positions[None, :, :] - positions[:, None, :]
    distance = (softening ** 2 + displacement.square().sum(-1) / 7).sqrt()
    coefficient = gravity * weights[:, None] * weights[None, :] / (7 * distance.pow(3))
    coefficient.diagonal().zero_()
    expected = (coefficient[:, :, None] * displacement).sum(1)
    actual = _softened_gravity(positions, weights, gravity, softening)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.sum(0), torch.zeros(7, device=device, dtype=dtype),
                               rtol=0, atol=2e-6)


@pytest.mark.parametrize('batch', [1, 4])
def test_collision_and_gram_roundoff_are_finite(device, dtype, batch):
    positions = torch.ones((batch, 11), device=device, dtype=dtype)
    weights = positions.new_full((batch,), 1 / batch)
    force = _softened_gravity(positions, weights, 100., .01)
    assert torch.isfinite(force).all()
    torch.testing.assert_close(force, torch.zeros_like(force), rtol=0, atol=0)


def test_reflection_multiple_crossings_and_exact_wall_ties(device, dtype):
    positions = torch.tensor([[0., 0., .3, .2, 0., 0., 0., 0.]], device=device, dtype=dtype)
    momenta = positions.new_tensor([[10., -10., 21., -25., 1., -1., 5., -5.]])
    kinetic = momenta.square().sum()
    _reflect_drift(positions, momenta, 1.)
    expected_q = positions.new_tensor([[0., 0., .7, -.8, 1., -1., 1., -1.]])
    expected_p = positions.new_tensor([[-10., 10., -21., -25., -1., 1., -5., 5.]])
    torch.testing.assert_close(positions, expected_q, rtol=0, atol=2e-6)
    torch.testing.assert_close(momenta, expected_p, rtol=0, atol=0)
    torch.testing.assert_close(momenta.square().sum(), kinetic, rtol=0, atol=0)
    assert ((positions >= -1) & (positions <= 1)).all()


def test_decoding_has_positive_zero_tie(device, dtype):
    positions = torch.tensor([[-.1, 0., .1]], device=device, dtype=dtype)
    torch.testing.assert_close(_decode(positions), positions.new_tensor([[-1., 1., 1.]]))


def test_free_drift_has_exact_damping_halves_and_frozen_controls(device):
    run, _, _ = _fixture_run(device)
    solver = GravityTunneling(time_step=.2, damping=.7)
    work = _GravityWorkspace(run, solver)
    work.q.zero_()
    work.p.fill_(.1)
    work.controls = 0., 0., 0., .2
    old_p = work.p.clone()
    calls = []
    force = work.force

    def traced():
        calls.append((work.controls, work.weights.data_ptr()))
        return force()

    work.force = traced
    work.advance()
    torch.testing.assert_close(work.q, .2 * old_p * torch.exp(old_p.new_tensor(-.07)))
    torch.testing.assert_close(work.p, old_p * torch.exp(old_p.new_tensor(-.14)))
    assert len(calls) == 2 and calls[0] == calls[1]


def test_feedback_weights_scale_temperature_and_positive_floor(device, dtype):
    scores = torch.tensor([-9., -5., 3.], device=device, dtype=torch.float64)
    scale = scores.new_tensor(4.)
    actual = _feedback_weights(scores, scale, .5, 2., dtype=dtype)
    exponential = torch.exp(-(scores - scores.min()) / 8)
    expected = (1 / 6 + .5 * exponential / exponential.sum()).to(dtype)
    torch.testing.assert_close(actual, expected)
    assert (actual >= 1 / 6).all()
    torch.testing.assert_close(actual.sum(), actual.new_tensor(1.))
    equal = _feedback_weights(scores * 0, scale, .7, .3, dtype=dtype)
    torch.testing.assert_close(equal, equal.new_full((3,), 1 / 3))


@pytest.mark.parametrize('n', [1, 2, 9])
def test_jump_masks_follow_log_uniform_formula(device, dtype, n):
    spins = torch.ones((17, n), device=device, dtype=dtype)
    actual_generator = torch.Generator(device=device).manual_seed(22)
    reference_generator = torch.Generator(device=device).manual_seed(22)
    options = dict(generator=reference_generator, device=device, dtype=dtype)
    distance_draw = torch.rand((17, 1), **options)
    probability = .5 if n == 1 else (n / 2) ** distance_draw / n
    flips = torch.rand(spins.shape, **options) < probability
    mixture = torch.rand((17, 1), **options) < .2
    uniform = torch.where(torch.rand(spins.shape, **options) < .5, 1., -1.).to(dtype)
    expected = torch.where(mixture, uniform, torch.where(flips, -spins, spins))
    actual = _jump_proposal(spins, actual_generator, .2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_uniform_proposal_mixture_covers_small_binary_domain(device):
    spins = torch.ones((4096, 4), device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(29)
    proposal = _jump_proposal(spins, generator, 1.)
    assert torch.unique(proposal, dim=0).shape == (16, 4)
    assert ((proposal == -1) | (proposal == 1)).all()


def test_metropolis_uses_original_energy_difference_and_scale(device):
    current = torch.zeros(8, device=device, dtype=torch.float64)
    proposed = current.new_tensor([-5., 0., .1, .3, .7, 1., 3., 20.])
    scale = current.new_tensor(2.)
    generator = torch.Generator(device=device).manual_seed(17)
    reference = torch.Generator(device=device).manual_seed(17)
    uniform = torch.rand(8, generator=reference, device=device, dtype=torch.float64)
    expected = (proposed <= current) | (uniform < torch.exp(-(proposed - current) / 1.))
    actual = _metropolis_accept(current, proposed, scale, .5, generator)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    zero = _metropolis_accept(current, proposed, scale, 0., generator)
    torch.testing.assert_close(zero, proposed <= current, rtol=0, atol=0)


class _Observer:
    def __init__(self):
        self.candidates = []

    def poll(self):
        pass

    def capture(self, samples, raw=None, *, phase, iteration):
        self.candidates.append((samples.clone(), raw.clone() if raw is not None else None,
                                phase, iteration))


def test_every_proposal_is_source_scored_retained_and_captured_before_rejection(monkeypatch, device):
    import qubo_solvers.gravity_tunneling as module

    problem = QUBO(torch.diag(torch.tensor([-2. ** 24, -1.], device=device, dtype=torch.float64)))
    initial = torch.tensor([[1., 0.], [1., 0.]], device=device, dtype=torch.float64)
    proposal = initial.new_tensor([[1., -1.], [1., 1.]])
    monkeypatch.setattr(module, '_jump_proposal', lambda spins, *args: proposal.to(spins.dtype))
    # Force rejection to check archive/observation independently of relocation.
    # Default Metropolis correctly accepts the improving proposal, tested above.
    monkeypatch.setattr(module, '_metropolis_accept',
                        lambda current, *args: torch.zeros_like(current, dtype=torch.bool))
    observer = _Observer()
    with observing(observer):
        result = GravityTunneling(max_steps=1, gravity=0., dynamics_dtype='float32').solve(
            problem, restarts=2, initial_assignments=initial, seed=11)
    torch.testing.assert_close(result.final_assignments, initial.to(torch.int8), rtol=0, atol=0)
    torch.testing.assert_close(result.best_assignments,
                               torch.tensor([[1, 0], [1, 1]], device=device, dtype=torch.int8))
    assert result.best_energy == -2. ** 24 - 1
    captures = [(samples, scores) for samples, scores, phase, _ in observer.candidates
                if phase == 'proposal']
    assert len(captures) == 1 and captures[0][0].shape == (2, 2)
    assert captures[0][1].dtype == torch.float64
    torch.testing.assert_close(captures[0][1], problem.energy(captures[0][0]), rtol=0, atol=0)


def test_accepted_changed_jump_resets_position_and_momentum_only_for_changed_rows(monkeypatch, device):
    import qubo_solvers.gravity_tunneling as module

    run, _, _ = _fixture_run(device)
    work = _GravityWorkspace(run, GravityTunneling())
    work.q.copy_(run.q.new_tensor([[.4, -.3], [.2, .5], [-.6, -.7]]))
    work.p.fill_(.2)
    old_q, old_p = work.q.clone(), work.p.clone()
    proposed = run.q.new_tensor([[-1., 1.], [1., 1.], [1., -1.]])
    samples = (proposed + 1) / 2
    captured = []
    run.x.copy_((old_q >= 0).to(run.x.dtype))
    old_x, old_energies = run.x.clone(), run.energies.clone()
    run.iterations = torch.zeros(3, device=device, dtype=torch.int64)
    run.to_assignments = lambda spins: (spins + 1) / 2
    score_calls = []

    def score(candidates):
        score_calls.append(candidates.clone())
        return run.q.new_tensor([-1., 1., 2.])

    run.score = score
    run.retain = lambda candidates, energies: captured.append(('retain', candidates.clone()))
    run.observe = lambda candidates, energies, **kwargs: captured.append(('observe', candidates.clone()))
    monkeypatch.setattr(module, '_jump_proposal', lambda *args: proposed)
    monkeypatch.setattr(module, '_metropolis_accept',
                        lambda *args: torch.tensor([True, True, False], device=device))
    work.jump(16, .1)
    torch.testing.assert_close(work.q[0], .95 * proposed[0])
    torch.testing.assert_close(work.p[0], torch.zeros_like(work.p[0]))
    torch.testing.assert_close(work.q[1:], old_q[1:])
    torch.testing.assert_close(work.p[1:], old_p[1:])
    torch.testing.assert_close(run.x[0], samples[0])
    torch.testing.assert_close(run.x[1:], old_x[1:])
    expected_energies = old_energies.clone()
    expected_energies[0] = -1.
    torch.testing.assert_close(run.energies, expected_energies)
    assert (run.iterations == 16).all()
    assert [phase for phase, _ in captured] == ['retain', 'observe']
    assert len(score_calls) == 1
    torch.testing.assert_close(captured[0][1], samples)


@pytest.mark.parametrize('spin', [False, True])
def test_solver_seed_history_original_score_and_input_contract(device, dtype, spin):
    problem = QUBO(torch.tensor([[-2., .5, 1.], [.5, -1., -.25], [1., -.25, -.5]],
                                device=device, dtype=dtype), 7.)
    if spin:
        problem = problem.to_ising()
    solver = GravityTunneling(max_steps=21, candidate_interval=8)
    options = dict(restarts=7, batch_size=3, seed=41, history_interval=3)
    source = (problem.J if spin else problem.Q).clone()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    first = solver.solve(problem, **options)
    second = solver.solve(problem, **options)
    for name in ('final_assignments', 'final_energies', 'best_assignments', 'best_energies',
                 'energy_history', 'history_iterations', 'iterations'):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    torch.testing.assert_close(source, problem.J if spin else problem.Q, rtol=0, atol=0)
    torch.testing.assert_close(first.best_energies, problem.to(dtype=torch.float64).energy(first.best_assignments))
    torch.testing.assert_close(first.final_energies, problem.to(dtype=torch.float64).energy(first.final_assignments))
    torch.testing.assert_close(first.energy_history[-1], first.final_energies)
    assert (first.best_energies <= first.final_energies).all()
    assert (first.iterations == 21).all()
    assert first.best_assignments.device == source.device


def test_history_sampling_does_not_change_physical_schedule(device):
    problem = QUBO(torch.tensor([[-2., .5], [.5, -1.]], device=device))
    solver = GravityTunneling(max_steps=33, candidate_interval=8)
    options = dict(restarts=6, batch_size=3, seed=17)
    plain = solver.solve(problem, **options)
    history = solver.solve(problem, history_interval=3, **options)
    torch.testing.assert_close(plain.final_assignments, history.final_assignments, rtol=0, atol=0)
    # Extra observations can retain more corners, so final trajectories and
    # proposal RNG must match; best archives need not match.


def test_zero_steps_still_ranks_every_original_fp64_candidate(device):
    problem = QUBO(torch.diag(torch.tensor([-2. ** 24, -1.], device=device, dtype=torch.float64)))
    initial = torch.tensor([[1, 0], [1, 1]], device=device, dtype=torch.int8)
    result = GravityTunneling(max_steps=0, dynamics_dtype='float32').solve(
        problem, restarts=2, initial_assignments=initial, history_interval=1)
    assert result.best_restart_index == 1
    assert result.best_energy == -2. ** 24 - 1
    assert result.iterations.tolist() == [0, 0]
    assert result.history_iterations.tolist() == [0]
    torch.testing.assert_close(result.final_assignments, initial)


def test_terminal_controls_and_workspace_scale():
    solver = GravityTunneling()
    assert solver._controls(1., 8, 5) == (1., solver.penalty, 0., 0.)
    assert solver._controls(0., 8, 5)[2] == 8 * 5 * solver.gravity
    # For fixed swarm size, the solver-specific estimate is affine in n.
    small = solver._population_workspace(10, 8, 4)
    middle = solver._population_workspace(20, 8, 4)
    large = solver._population_workspace(30, 8, 4)
    assert large - middle == middle - small
    assert solver._population_workspace(20, 16, 4) > middle
    with pytest.raises(FrozenInstanceError):
        solver.gravity = .2


def test_relocation_radius_must_be_representable_in_dynamics_precision(device):
    problem = QUBO(torch.zeros((1, 1), device=device, dtype=torch.float64))
    with pytest.raises(ValueError, match='relocation_radius'):
        GravityTunneling(max_steps=1, relocation_radius=1e-100,
                        dynamics_dtype='float32').solve(problem, seed=0)


class _TensorShapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        values = result if isinstance(result, (tuple, list)) else (result,)
        self.shapes.extend(tuple(value.shape) for value in values if isinstance(value, torch.Tensor))
        return result


def test_gravity_does_not_allocate_cubic_population_displacements(device, dtype):
    positions = torch.zeros((7, 11), device=device, dtype=dtype)
    weights = positions.new_full((7,), 1 / 7)
    shapes = _TensorShapes()
    with shapes:
        _softened_gravity(positions, weights, 1., .5)
    assert (7, 7) in shapes.shapes
    assert not any(len(shape) == 3 for shape in shapes.shapes)


@pytest.mark.parametrize('parameters', [
    {'max_steps': -1}, {'max_steps': True}, {'candidate_interval': 0},
    {'time_step': 0}, {'time_step': float('inf')}, {'damping': -1},
    {'penalty': -1}, {'gravity': -1}, {'softening': 0}, {'weight_mix': 1},
    {'weight_mix': -1}, {'weight_temperature': 0}, {'start_temperature': -1},
    {'uniform_jump_probability': 0}, {'uniform_jump_probability': 1.1},
    {'relocation_radius': 0}, {'relocation_radius': 1.1}, {'normalize': 1},
    {'dynamics_dtype': torch.float16}, {'sparse_threshold': -1},
    {'sparse_threshold': 1.1}, {'penalty': float('nan')},
])
def test_invalid_controls(parameters):
    with pytest.raises(ValueError):
        GravityTunneling(**parameters)
